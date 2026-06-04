import copy
import heapq
import itertools
from collections import defaultdict
from collections.abc import Mapping, Callable
from typing import MutableMapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPast,
)

from data.components.transforms import VectorizedPerformanceStreamsBucketizer
from models.automata import tokens as tk
from models.automata.tokens import TranscriptionState
from models.automata.vectorization_torch import TokenTensor, ProblemType, clip
from models.components.embedding import MIDIEmbedding, StateEmbedding
from models.components.embedding import ScoreTokenEmbedding
from models.components.expert_head import ExpertHead
from models.components.roformer import Roformer
from utils.constants import PADDING_VALUE
from utils.debug import write_compare_to_file
from utils.generation import GenerationState, get_last_aligned_downbeats
from utils.musicxml import ARTICULATIONS_AND_ORNAMENTS


class TokensTranscriberModel(nn.Module):
    def __init__(
        self,
        stream_dimensions: dict[str, int],
        roformer: Roformer | None,
        expert_inner_dims: list[int] | dict[str, list[int]],
        router_inner_dims: list[int] = None,
        time_based_cross_attention: bool = False,
        latent_part_reserved_for_router: float | int | None = None,
        states_prediction: bool = False,
        states_prediction_mlp_dims: list[int] = None,
    ):
        super().__init__()

        self.roformer = roformer
        self.time_based_cross_attention = time_based_cross_attention

        self.embedding_dim = self.roformer.encoder.config.hidden_size
        self.layer_norm_eps = self.roformer.encoder.config.layer_norm_eps
        self.hidden_dropout_prob = self.roformer.encoder.config.hidden_dropout_prob

        if latent_part_reserved_for_router is not None:
            # In case of float, consider a proportion and convert to int (prop * hidden_size)
            if isinstance(latent_part_reserved_for_router, float):
                if not (0 < latent_part_reserved_for_router < 1):
                    raise ValueError(
                        f"latent_part_reserved_for_router must be between 0 and 1 (both "
                        f"excluded), got {latent_part_reserved_for_router}"
                    )
                latent_part_reserved_for_router = round(
                    latent_part_reserved_for_router * self.embedding_dim
                )
                if latent_part_reserved_for_router == 0:
                    logger.warning(
                        "Fraction of embedding dimension reserved for router is too small, "
                        "capping it to 1 dimension"
                    )
                    latent_part_reserved_for_router = 1
                elif latent_part_reserved_for_router == self.embedding_dim:
                    logger.warning(
                        "Fraction of embedding dimension reserved for router is too large, "
                        f"capping it to {self.embedding_dim - 1} dimensions"
                    )
                    latent_part_reserved_for_router = self.embedding_dim - 1
            # For an int, consider it directly the number of dimensions
            else:
                if not (0 < latent_part_reserved_for_router < self.embedding_dim):
                    raise ValueError(
                        f"latent_part_reserved_for_router must be between 0 and "
                        f"{self.embedding_dim} (both excluded), got {latent_part_reserved_for_router}"
                    )
        self.latent_part_reserved_for_router = latent_part_reserved_for_router
        if self.latent_part_reserved_for_router:
            logger.info(
                f"Embedding dimension is {self.embedding_dim}, "
                f"{self.latent_part_reserved_for_router} dimensions are reserved for the router, "
                f"and {self.embedding_dim - self.latent_part_reserved_for_router} for the experts. "
            )

        self.midi_embedding = MIDIEmbedding(
            stream_dimensions=stream_dimensions,
            embedding_dimension=self.embedding_dim,
            layer_norm_eps=self.layer_norm_eps,
            dropout_prob=self.hidden_dropout_prob,
        )

        self.score_embedding = ScoreTokenEmbedding(
            embedding_dimension=self.embedding_dim,
            layer_norm_eps=self.layer_norm_eps,
            dropout_prob=self.hidden_dropout_prob,
        )

        self.states_prediction = states_prediction
        if states_prediction_mlp_dims is not None and not states_prediction:
            logger.warning(
                f"states_prediction_head_dims is ignored when states_prediction is False"
            )
        if self.states_prediction:
            self.states_embedding = StateEmbedding(
                embedding_dimension=self.embedding_dim,
                layer_norm_eps=self.layer_norm_eps,
                dropout_prob=self.hidden_dropout_prob,
            )
            state_prediction_mlp_layers = []
            dim1 = self.embedding_dim
            for dim2 in states_prediction_mlp_dims[:-1]:
                state_prediction_mlp_layers.append(nn.Linear(dim1, dim2))
                state_prediction_mlp_layers.append(nn.ReLU())
                dim1 = dim2
            state_prediction_mlp_layers.append(
                nn.Linear(dim1, states_prediction_mlp_dims[-1])
            )  # No ReLU last
            self.states_prediction_mlp = nn.Sequential(*state_prediction_mlp_layers)

        modules = []
        last_dim = (
            self.embedding_dim
            if self.latent_part_reserved_for_router is None
            else self.latent_part_reserved_for_router
        )
        if router_inner_dims is not None:
            for dim in router_inner_dims:
                modules.append(nn.Linear(last_dim, dim))
                modules.append(nn.ReLU())
                last_dim = dim
        modules.append(nn.Linear(last_dim, len(TokenTensor.TOKEN_INDEXER)))
        self.token_type_router = nn.Sequential(*modules)

        if isinstance(expert_inner_dims, list):
            self.expert_inner_dims = {
                ttype.as_str(): expert_inner_dims.copy() for ttype in tk.TOKEN_TYPES
            }
        elif isinstance(expert_inner_dims, MutableMapping):
            default = expert_inner_dims.get("_default", None)
            # For backward compatibility due to config saving bug
            if len(expert_inner_dims) == 0 and default is None:
                logger.warning(f"Empty expert_inner_dims, using [256, 128] as default expert dims")
                default = [256, 128]
            provided_keys = set(k for k in expert_inner_dims.keys() if k != "_default")
            required_keys = {t.as_str() for t in tk.TOKEN_TYPES}
            # Make sure no unknown key passed
            if unknown_keys := provided_keys.difference(required_keys):
                raise ValueError(f"Unknown token types in expert_inner_dims: {unknown_keys}")
            # Make sure a default is passed if not all required keys are provided
            if default is None and (missing_keys := required_keys.difference(provided_keys)):
                raise ValueError(
                    f"expert_inner_dims is missing the following token types: {missing_keys}. "
                    f"Either provide them of provide a '_default' key."
                )
            # Build the complete dict
            self.expert_inner_dims = {k: expert_inner_dims.get(k, default) for k in required_keys}
        else:
            raise ValueError(
                f"expert_inner_dims must be a list[int] or a dict[str, list[int]], "
                f"got {type(expert_inner_dims)}"
            )
        self.expert_heads: Mapping[str, ExpertHead] = nn.ModuleDict(
            {
                token_type.as_str(): ExpertHead(
                    embedding_dim=(
                        self.embedding_dim
                        if self.latent_part_reserved_for_router is None
                        else self.embedding_dim - self.latent_part_reserved_for_router
                    ),
                    inner_dims=self.expert_inner_dims[token_type.as_str()],
                    output_dims={
                        dimension_name: TokenTensor.DIMS[dimension_name]
                        for dimension_name in TokenTensor.TOKEN_TYPE_TO_DIMS[token_type]
                        if dimension_name != "token_type"
                    },
                )
                for token_type in tk.TOKEN_TYPES  # TODO: Remove Tokens without attributes
                if token_type
                not in [
                    tk.StartOfScore,  # Not predicted
                    tk.EndOfScore,  # No attributes
                ]
            }
        )

    def forward_encoder(
        self,
        input_streams: dict[str, torch.Tensor],
        onset_times: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        also_return_pre_norm_encoder_outputs: bool = False,
        output_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, BaseModelOutputWithPast]:
        B, T = input_streams[next(iter(input_streams))].shape[:2]
        if not all(v.shape[:2] == (B, T) for v in input_streams.values()):
            raise ValueError(
                f"All input streams must have the same batch size and sequence length. "
                f"Got {[v.shape for v in input_streams.values()]}"
            )
        if onset_times is not None and onset_times.shape != (B, T):
            raise ValueError(
                f"onset_times must have shape (B, T). Got {onset_times.shape} instead of {(B, T)}"
            )

        # Get the embeddings for the input streams
        midi_embeddings = self.midi_embedding.forward(input_streams)

        # Forward through the encoder
        outputs_encoder = self.roformer.forward_enc(
            midi_embeddings=midi_embeddings,
            midi_onset_times=onset_times,
            attention_mask=attention_mask,
            also_return_pre_norm_encoder_output=also_return_pre_norm_encoder_outputs,
            output_attentions=output_attentions,
        )
        return outputs_encoder

    def forward_decoder(
        self,
        input_tokens: TokenTensor,
        input_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_positions: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        score_embeddings = self.score_embedding.forward(input_tokens)
        if self.states_prediction:
            if input_states is None:
                raise ValueError("Input states must be provided when states_prediction is True.")
            state_embeddings = self.states_embedding.forward(input_states)
            score_embeddings += state_embeddings  # Sum both embeddings
        outputs_decoder = self.roformer.forward_dec(
            score_embeddings=score_embeddings,
            attention_mask=attention_mask,
            positions=None,  # computed from attention_mask.cumsum
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_input_positions=encoder_positions,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        return outputs_decoder

    def forward_router(self, decoder_outputs: torch.Tensor) -> torch.Tensor:
        if self.latent_part_reserved_for_router is not None:
            decoder_outputs = decoder_outputs[..., : self.latent_part_reserved_for_router]
        return self.token_type_router(decoder_outputs)

    def forward_expert(
        self,
        token_type: tk.AnyToken,
        decoder_outputs: torch.Tensor,  # B * T * E
    ) -> dict[str, torch.Tensor]:  # i: B * T * dim[i]
        """
        Foward the tensor though one specific expert head. Note that all tokens will flow though
        the expert, you only need some indices to pass though this expert, make sure to filter them
        out of decoder_outputs before.
        """
        if self.latent_part_reserved_for_router is not None:
            decoder_outputs = decoder_outputs[..., self.latent_part_reserved_for_router :]
        expert_head: ExpertHead = self.expert_heads[token_type]
        expert_outputs = expert_head.forward(decoder_outputs)
        return expert_outputs

    def forward_experts(
        self,
        decoder_outputs: torch.Tensor,  # B * T * E
        token_type_masks: dict[str, torch.Tensor],  # B * T
        convert_to_tokentensor: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]] | TokenTensor:
        # Check that the shapes make sense
        if not all(mask.shape == decoder_outputs.shape[:2] for mask in token_type_masks.values()):
            raise ValueError(
                f"Dimensions mismatch: {decoder_outputs.shape[:2]=} != "
                f"{[mask.shape for mask in token_type_masks.values()]}"
            )

        # Split the decoder_outputs into experts
        # TODO: change to dict comprehension
        expert_outputs: dict[str, dict[str, torch.Tensor]] = {}
        for token_type in self.expert_heads:
            expert: ExpertHead = self.expert_heads[token_type]
            involved_decoder_outputs = decoder_outputs[token_type_masks[token_type]]
            if self.latent_part_reserved_for_router is not None:
                involved_decoder_outputs = involved_decoder_outputs[
                    ..., self.latent_part_reserved_for_router :
                ]
            expert_output = expert.forward(involved_decoder_outputs)
            expert_outputs[token_type] = expert_output

        if convert_to_tokentensor:
            return TokenTensor.build_from_dict(
                type_dim_values_dict=expert_outputs,
                token_type_masks=token_type_masks,
                device=decoder_outputs.device,
            )
        else:
            return expert_outputs

    def forward_state_predictor(self, decoder_outputs: torch.Tensor) -> torch.Tensor:
        if self.latent_part_reserved_for_router is not None:
            decoder_outputs = decoder_outputs[..., : self.latent_part_reserved_for_router]
        return self.states_prediction_mlp(decoder_outputs)

    def generate(
        self,
        performance_streams_raw: dict[str, torch.Tensor],  # pre-bucketization streams, each B*L
        performance_lengths: torch.Tensor,  # B
        measures_interchunk_overlap: int = 1,
        chunk_size: int = 500,
        max_chunks: int = 1000,
        max_tokens_per_chunk: int = 2000,
        constraint: bool = False,
        max_constraint_iterations: int = None,
        use_cache: bool = True,
        progress_bar: bool = True,
    ):
        streams_bucketizer = VectorizedPerformanceStreamsBucketizer()

        performances_pitches = performance_streams_raw["pitch"]  # B*L
        performances_durations = performance_streams_raw["duration"]  # B*L
        performances_onset_times = performance_streams_raw["onset"]  # B*L
        performances_offset_times = torch.where(
            performances_onset_times != PADDING_VALUE,
            performances_onset_times + performances_durations,
            PADDING_VALUE,
        )  # B*L
        batch_size, batch_length = performances_pitches.shape  # B, L

        # Timestamp where each transcription in the batch is currently at, all 0 at start
        start_times = torch.zeros((batch_size, 1), device=device)  # B*1

        # End of previous chunks used as context for next chunk, None at start
        # Note that it should LEFT-padded to avoid having pad tokens in the middle of the sequence
        prefixes = None  # B * L_prefix

        # Transcription of each sequence of the batch
        full_transcribed_sequences = [[] for _ in range(batch_size)]

        # Flag indicating for each performance the start of the transcription
        # For each performance:
        #   - idx 0: flag, all 0 at start (=UNFINISHED, do NOT change this enum value to something else)
        #   - idx 1: index of the chunk the current flag has happened, all 0 at start
        #   - idx 2: index of the flag position in the chunk, all 0 at start
        global_finished_flags = torch.zeros(
            (batch_size, 3), dtype=torch.long, device=self.device
        )  # B*3

        # Track performances for which all notes have been consumed
        performances_fully_consumed = torch.zeros(
            (batch_size,), dtype=torch.bool, device=self.device
        )  # B

        # Iterate until max chunks (will break earlier if needed)
        for chunk_idx in trange(max_chunks, desc="Generating chunks", disable=not progress_bar):
            # Get a mask of still unfinished performances
            unfinished_mask = global_finished_flags[:, 0] == GenerationState.UNFINISHED.value
            num_unfinished = unfinished_mask.sum()
            logger.debug(f"Number of unfinished performances: {num_unfinished}")

            # Break if no more unfinished performances
            if num_unfinished == 0:
                break

            assert prefixes is None or num_unfinished == prefixes.shape[0]
            logger.debug(
                f"Transcribing chunks for {num_unfinished} pieces from "
                f"{start_times[unfinished_mask].squeeze(-1)}"
            )

            # When a variable ends with *_uf, it means that is only contains information about the
            # unfinished performances, i.e. its first dim is B_uf = num_unfinished
            performances_onset_times_uf = performances_onset_times[unfinished_mask]
            performances_offset_times_uf = performances_offset_times[unfinished_mask]
            start_times_uf = start_times[unfinished_mask]
            performance_lengths_uf = performance_lengths[unfinished_mask]

            # Create a mapping from uf indices to global indices
            idx_uf_to_global = dict(enumerate(unfinished_mask.nonzero(as_tuple=True)[0].tolist()))

            # The next section basically extract the chunk notes in each performance, clipping the
            # ones starting before the required start
            # TODO: It would probably be useful to allow for a couple of extra notes if that ends
            #  up the performance rather than getting the next chunk almost empty

            # We first deal with the chunk start (the chunk end will be dealt with later)
            # We include in the chunk every note starting OR ending on/after the chunk start
            included_notes_mask_uf = torch.logical_or(
                performances_onset_times_uf >= start_times_uf,
                performances_offset_times_uf > start_times_uf,
            )  # B_uf * 1
            num_included_notes_uf = included_notes_mask_uf.sum(dim=-1)  # B_uf

            # Check how many notes remain in each performance
            num_remaining_notes_uf = included_notes_mask_uf.sum(dim=-1)  # B_uf
            # Get the actual chunk size (chunk_size or shorter if fewer notes remaining)
            actual_chunk_size = min(num_remaining_notes_uf.max(), chunk_size)  # used for padding

            # Argsort the mask, so that we get the index of all True, row-wise (performance-wise)
            # This is needed to cover overlapping cases that a naive cut at the first True would
            # not catch, see the following "piano roll":
            #                 | <- chunk start
            #  N0==  N2============  N5===  N7=
            #   N1==   N3==      N4====  N6====
            #                 |
            # Then included_notes_mask is:
            # [
            #   False,  # N0 is before the chunk
            #   False,  # N1 is before the chunk
            #   True,   # N2 ends in the chunk
            #   False,  # n3 is before the chunk
            #   True * 4,   # N4 to N7 start in the chunk
            # ]
            # Therefore, we argsort the list to get the indices of True, skipping eventual False
            # We have descending=True to have all True before (True > False)
            # We have stable=True to make sure the order is conserved within the True
            # And we get the indices [2, 4, 5, 6, 7], which are indeed the required indices
            # All of this happens batched for all performances (and we crop at max chunk size)
            #  -> Note that as lengths can differ, padded performance will end up including some
            #     False, in the end of the argsort, that would result in (with 6 as chunk size):
            #     [
            #         [2, 4, 5, 6, 7],  # Same example as above
            #         [6, 7, 0, 1, 2],  # 2nd sequence in the batch
            #     ]
            #     In the second sequence, there where only 2 True (index 6 and 7) in the mask, but
            #     as the chunk size is 6, the indices array contains some values corresponding to
            #     False (the last three: 0, 1 and 2). To ignore them, we have to make sure they are
            #     marked as masked in the padding mask (see just below).
            notes_indices_uf = included_notes_mask_uf.argsort(dim=-1, stable=True, descending=True)[
                :, :actual_chunk_size
            ]

            # Now we recompute the padding mask as it could have new elements, still keeping old
            # padding information.
            # For example, with 5 notes per chunk, same example as before for S1, new sequences S2 & S3:
            # (<> marks included notes in sequence below, PA a padding token)
            #           <>    <> <>
            # S1: N0 N1 N2 N3 N4 N5 N6 N7 PA   ==>   N2 N4 N5 PA PA
            #                       <> <>
            # S2: N0 N1 N2 N3 N4 N5 N6 N7 PA   ==>   N6 N7 PA PA PA
            #           <> <> <> <> <>
            # S3: N0 N1 N2 N3 N4 N5 N6 N7 PA   ==>   N3 N4 N5 N6 N7
            # The gather below will take the values in included_note_mask at the indices of
            # notes_indices_uf, therefore it will be True for note included, but will be False
            # for not included ones, i.e. for PAD tokens.
            performance_padding_mask_uf = ~included_notes_mask_uf.gather(1, notes_indices_uf)

            # Track the fully consumed performances to stop their transcription at next chunk
            performances_fully_consumed_uf = num_remaining_notes_uf <= chunk_size

            # Simple util to gather the required indices from a given tensor, handling padding
            def chunk_tensor(
                tensor: torch.Tensor,
                indices: torch.Tensor,
                padding_mask: torch.Tensor,
            ) -> torch.Tensor:
                chunk_tensor = tensor.gather(1, indices)
                chunk_tensor[padding_mask] = PADDING_VALUE
                return chunk_tensor

            # Initialize the chunk raw streams, keeping only the included notes
            chunk_streams_raw_uf = {
                k: chunk_tensor(
                    stream[unfinished_mask], notes_indices_uf, performance_padding_mask_uf
                )
                for k, stream in performance_streams_raw.items()
            }

            # Clip the notes starting before the chunk to the start time
            # First get notes to clip
            indices_to_clip_uf = torch.where(
                torch.logical_and(
                    chunk_streams_raw_uf["onset"] < start_times_uf,
                    chunk_streams_raw_uf["onset"] != PADDING_VALUE,
                )
            )
            # Then get the new start times for those notes
            corresponding_start_time_uf = start_times_uf[indices_to_clip_uf[0]].squeeze(1)
            # Remove the clipped duration to the note duration
            chunk_streams_raw_uf["duration"][indices_to_clip_uf] -= (
                corresponding_start_time_uf - chunk_streams_raw_uf["onset"][indices_to_clip_uf]
            )
            # Set the new onset
            chunk_streams_raw_uf["onset"][indices_to_clip_uf] = corresponding_start_time_uf

            # Offset the whole chunk so that it starts at 0
            chunk_streams_raw_uf["onset"] = torch.where(
                ~performance_padding_mask_uf,
                chunk_streams_raw_uf["onset"] - start_times_uf,
                PADDING_VALUE,
            )

            # for xi, x in enumerate(range(num_unfinished)):
            #     m = performance_streams_to_midi({k: v[xi] for k, v in chunk_streams_raw_uf.items()})
            #     m.write(f"/tmp/{xi}_chunk_{chunk_idx}.mid")

            # Bucketize the streams
            chunk_streams_uf = streams_bucketizer.process(chunk_streams_raw_uf)

            # Transcribe the chunk
            transcribed_tokens_tensor, finished_flags_and_positions = self.generate_single_chunk(
                performance_streams={k: v for k, v in chunk_streams_uf.items() if k != "pad"},
                performance_onset_times=chunk_streams_raw_uf["onset"],
                performance_attention_mask=performance_padding_mask_uf.logical_not(),
                prefix=prefixes,
                constraint=constraint,
                max_constraint_iterations=max_constraint_iterations,
                max_generated_length=max_tokens_per_chunk,
                use_cache=use_cache,
                progress_bar=progress_bar,
                progress_bar_leave=False,
                return_logits_and_mask=False,
            )

            # As prefixes are left-padded with SOS tokens, we remove them (still keeping one)
            transcribed_tokens_uf = [
                [tk.StartOfScore()]
                + [t for t in transcription if not isinstance(t, tk.StartOfScore)]
                for transcription in transcribed_tokens_tensor.to_tokens()
            ]

            # for xi, x in enumerate(transcribed_tokens_uf):
            #     from models.automata.io import token_sequence_to_partitura_permissive
            #     import partitura as pt
            #
            #     s = token_sequence_to_partitura_permissive(x)
            #     pt.save_musicxml(s, f"/tmp/{xi}_chunk_{chunk_idx}.musicxml")

            # Extract flag information
            finished_flags, flag_positions_uf = (
                finished_flags_and_positions[:, 0],
                finished_flags_and_positions[:, 1],
            )
            flags_uf = [GenerationState(f.item()) for f in finished_flags]
            # Check that all chunks have been correctly fully transcribed (i.e. FINISHED flag)
            for i, (flag, pos, stime) in enumerate(
                zip(flags_uf, flag_positions_uf.tolist(), start_times_uf.squeeze(-1).tolist())
            ):
                if flag != GenerationState.FINISHED:
                    logger.error(
                        f"Generation failed for performance {i} on chunk {chunk_idx} at "
                        f"t={stime:.2f} with flag {flag.name} at token {pos}."
                    )
                    global_finished_flags[idx_uf_to_global[i]] = torch.LongTensor(
                        [GenerationState.EXCEEDED_MAX_TOKENS.value, chunk_idx, pos]
                    )

            # Get the number of prefix tokens for each performance
            num_prefix_tokens_uf = (
                (prefixes[..., 0] != PADDING_VALUE).sum(-1)
                if prefixes is not None
                else torch.zeros(batch_size, dtype=torch.long, device=self.device)
            )

            # For performances either finished or failed, write the final chunk and flag
            performances_to_stop_transcription_mask_uf = torch.logical_or(
                performances_fully_consumed_uf,
                finished_flags != GenerationState.FINISHED.value,
            )
            for i in performances_to_stop_transcription_mask_uf.nonzero(as_tuple=True)[0].tolist():
                global_finished_flags[idx_uf_to_global[i]] = torch.LongTensor(
                    [flags_uf[i].value, chunk_idx, flag_positions_uf[i]]
                )
                full_transcribed_sequences[idx_uf_to_global[i]].extend(
                    transcribed_tokens_uf[i][num_prefix_tokens_uf[i] :]
                )

            # For performances that are still missing chunk (and that hadn't had errors), realign
            # the end of the transcription with the performance notes to build the prefix for the
            # next chunk (with the measure overlap specified in parameter). The idea is to drop
            # completly the last transcribed measure as potentially incomplete because of the hard
            # cut in the performance. Then, we want to provide the last x complete measures of the
            # transcription as context for the next chunk, but we also need to be able to know to
            # which notes in the performance those are related, to also provide them to the
            # transcription model at the next chunk.
            # First, find the candidate performance indices
            performances_to_continue_transcription_indices_uf = (
                performances_to_stop_transcription_mask_uf.logical_not()
                .nonzero(as_tuple=True)[0]
                .tolist()
            )
            # If there is at least one, run the realignment pipeline
            if len(performances_to_continue_transcription_indices_uf) > 0:
                # Get the last aligned downbeats in each performance
                # This is not parallelized as it would be quite challenging, so we loop over them
                new_prefixes = []
                for idx in performances_to_continue_transcription_indices_uf:
                    transcribed_tokens_unpadded = transcribed_tokens_uf[idx][
                        num_prefix_tokens_uf[idx] :
                    ]
                    # Get the last aligned downbeats between the transcription and the performance
                    aligned_downbeats = get_last_aligned_downbeats(
                        performance_pitches=chunk_streams_raw_uf["pitch"][idx].tolist(),
                        transcribed_tokens=transcribed_tokens_uf[idx],
                        min_onsets=3,
                        downbeats_to_fetch=(
                            [0]
                            if measures_interchunk_overlap == 0
                            else [0, measures_interchunk_overlap]
                        ),
                    )
                    # If no measure overlap, we simply drop everything after the last downbeat and
                    # call it a day
                    if measures_interchunk_overlap == 0:
                        # In case of no downbeat found, set the appropriate flag and skip
                        if len(aligned_downbeats) < 1:
                            logger.error(
                                f"Can't find location of last downbeat in the transcription: "
                                f"{aligned_downbeats}"
                            )
                            global_finished_flags[idx_uf_to_global[idx]] = torch.LongTensor(
                                [
                                    GenerationState.CHUNK_ALIGNMENT_ERROR.value,
                                    chunk_idx,
                                    flag_positions_uf[idx],
                                ]
                            )
                            full_transcribed_sequences[idx_uf_to_global[idx]].extend(
                                transcribed_tokens_uf[idx][num_prefix_tokens_uf[idx] :]
                            )
                            continue
                        assert len(aligned_downbeats) == 1, aligned_downbeats
                        # Extract both transcription and performance aligned downbeat index
                        last_transcribed_bar_idx, last_performance_db_idx = aligned_downbeats[0]
                        # Set index of start of next chunk transcription context (will be empty)
                        token_context_start = last_transcribed_bar_idx
                        # Set index of start of next chunk performance context (empty also)
                        performance_context_start = last_performance_db_idx
                    # If measure overlap, then we need to manage the context for next chunk
                    else:
                        # If less than 2 downbeats found (1 one is the place to drop the transcription,
                        # 2nd one is the place to start the prefix context), then flag as error
                        if len(aligned_downbeats) < 2:
                            logger.error(
                                f"Can't align performance with transcription on last {measures_interchunk_overlap} "
                                f"measures(s): {aligned_downbeats}"
                            )
                            global_finished_flags[idx_uf_to_global[idx]] = torch.LongTensor(
                                [
                                    GenerationState.CHUNK_ALIGNMENT_ERROR.value,
                                    chunk_idx,
                                    flag_positions_uf[idx],
                                ]
                            )
                            full_transcribed_sequences[idx_uf_to_global[idx]].extend(
                                transcribed_tokens_uf[idx][num_prefix_tokens_uf[idx] :]
                            )
                            continue
                        assert len(aligned_downbeats) == 2, aligned_downbeats
                        #  Else, extract in both transcription and performance the drop point and
                        # and the context start point
                        (
                            (token_context_start, performance_context_start),
                            (last_transcribed_bar_idx, last_performance_db_idx),
                        ) = aligned_downbeats

                    # Now, drop the last transcribed measure (as likely incomplete due hard clipping in the performance)
                    transcribed_tokens_uf[idx] = transcribed_tokens_uf[idx][
                        :last_transcribed_bar_idx
                    ]
                    # Extend the full transcription with the prefix tokens first
                    full_transcribed_sequences[idx_uf_to_global[idx]].extend(
                        transcribed_tokens_uf[idx][num_prefix_tokens_uf[idx] :]
                    )
                    # Prepare the next prefix with the context
                    prefix = transcribed_tokens_uf[idx][token_context_start:]

                    # Insert the music contextual elements at the context start (clefs, key sig.)
                    context_clefs = {}
                    context_key_signature = None
                    already_in_context = list(
                        itertools.takewhile(
                            lambda t: isinstance(t, (tk.Clef, tk.KeySignature)), prefix[1:]
                        )
                    )
                    for token in (
                        transcribed_tokens_uf[idx][:token_context_start] + already_in_context
                    ):
                        if isinstance(token, tk.Clef):
                            context_clefs[token.staff_idx] = token
                        elif isinstance(token, tk.KeySignature):
                            context_key_signature = token
                    new_context = [copy.copy(c) for s, c in sorted(context_clefs.items())] + [
                        copy.copy(context_key_signature)
                    ]
                    prefix[1 : len(already_in_context) + 1] = (
                        new_context  # +1 as we skip the bar in takewhile
                    )

                    # Insert SOS in prefix
                    prefix.insert(0, tk.StartOfScore())

                    # Convert prefix to tensor
                    prefix = TokenTensor.from_tokens(prefix)

                    # Set the next start time
                    next_start_time = (
                        chunk_streams_raw_uf["onset"][idx, performance_context_start]
                        + start_times_uf[idx]
                    )
                    # In case the new start time is the same, we error as this would end up
                    # retranscribing the same section again and again
                    if next_start_time == start_times_uf[idx]:
                        logger.error(
                            "The required measure overlap is too large relative to the chunk size, meaning "
                            "that the model is not able to generate more than the prefix passed. Either "
                            "reduce the the measure overlap/minimum of notes to register a downbeat, or "
                            "increase the chunk size/maximum tokens length. Stopping the transcription "
                            "of this performance."
                        )
                        global_finished_flags[idx_uf_to_global[idx]] = torch.LongTensor(
                            [
                                GenerationState.CHUNK_ALIGNMENT_ERROR.value,
                                chunk_idx,
                                flag_positions_uf[idx],
                            ]
                        )
                        continue

                    # Store the new start time for next chunk
                    start_times[idx_uf_to_global[idx]] = next_start_time

                    # Store the prefix of this transcription
                    new_prefixes.append(prefix)

                # Left-pad the prefixes to avoid a gap in the transcription process
                if new_prefixes:
                    prefixes = pad_sequence(
                        new_prefixes,
                        batch_first=True,
                        padding_value=PADDING_VALUE,
                        padding_side="left",
                    )
            else:
                break

        # Finally, return both the transcribed sequences and the flags
        return full_transcribed_sequences, global_finished_flags

    def generate_single_chunk(
        self,
        performance_streams: dict[str, torch.Tensor],
        performance_onset_times: torch.Tensor,
        performance_attention_mask: torch.BoolTensor = None,  # 1 for attended, 0 for masked
        prefix: list[tk.AnyToken] | None = None,
        constraint: bool = False,
        max_constraint_iterations: int = None,
        max_generated_length: int = 1024,
        use_cache: bool = True,
        progress_bar: bool = True,
        progress_bar_leave: bool = True,
        return_logits_and_mask: bool = False,
    ) -> TokenTensor | tuple[list[tk.Token], torch.Tensor, torch.Tensor]:
        device = performance_onset_times.device
        if not use_cache:
            raise NotImplementedError("Generating without cache is not implemented yet")
        if return_logits_and_mask:
            raise NotImplementedError("Returning logits and mask is disabled for now until fixed")

        if self.states_prediction:
            if not constraint:
                constraint = True
                logger.warning(f"Forcing constraint to True as states prediction is enabled.")

        # Check dimensions
        batch_size, seq_length = performance_onset_times.shape
        if any(s.shape[:-1] != (batch_size, seq_length) for s in performance_streams.values()) or (
            prefix is not None
            and len(prefix) != batch_size
            or (
                performance_attention_mask is not None
                and performance_attention_mask.shape != (batch_size, seq_length)
            )
        ):
            raise ValueError(
                f"Performance streams (B*L*Fi), onset times (B*L) stream, attention mask (B*L) and "
                f"prefix (list of size B) must have matching dimensions."
            )

        # Handle eventual logits and masks
        if return_logits_and_mask:
            logits_and_masks = []

        # Initialize eventual automata
        if constraint:
            from models.automata.automata import ConstrainedAutomata

            automata = [
                ConstrainedAutomata(
                    num_voices=8,
                    allowed_pitches=pitch_stream.sum(0)
                    .nonzero(as_tuple=True)[0]
                    .tolist(),  # as pitch stream is OHE
                )
                for pitch_stream in performance_streams["pitch"]
            ]
            if self.states_prediction:
                batched_states_vectors = [[] for _ in automata]
        elif max_constraint_iterations:
            logger.warning("max_constraint_iterations is not used for unconstrained generation")

        # Handle eventual prefix
        if prefix is None:
            sos_token = tk.StartOfScore()
            batched_tokens_tensor = TokenTensor.from_tokens([[sos_token]], device=device).expand(
                batch_size, -1, -1
            )
            if self.states_prediction:
                for a, states in zip(automata, batched_states_vectors):
                    a.transition(sos_token)
                    states.append(a.state.vectorize(device=device))
                batched_states_tensor = torch.stack(
                    [torch.stack(states_vectors) for states_vectors in batched_states_vectors]
                )
        else:
            # Left-pad prefixes and convert them to tensor
            prefix_tensor: TokenTensor = pad_sequence(
                sequences=[TokenTensor.from_tokens(p) for p in prefix],
                batch_first=True,
                padding_value=PADDING_VALUE,
                padding_side="left",
            ).as_subclass(TokenTensor)
            batched_tokens_tensor: TokenTensor = prefix_tensor
            # Pass automata on prefixes
            if constraint:
                for i, (sequence, automaton) in enumerate(zip(prefix, automata)):
                    for token in sequence:
                        automaton.transition(token)
                        if self.states_prediction:
                            batched_states_vectors[i].append(
                                automaton.state.vectorize(device=device)
                            )
                if self.states_prediction:
                    batched_states_tensor = pad_sequence(
                        sequences=batched_states_vectors,
                        batch_first=True,
                        padding_value=PADDING_VALUE,
                        padding_side="left",
                    )
            # last_predicted_tokens = batched_tokens_tensor[:, -1].to_tokens()

        # Prepare output flags
        batched_finished_flag = torch.tensor(
            [[GenerationState.UNFINISHED.value, 0]], device=device
        ).repeat(batch_size, 1)

        with torch.no_grad():
            # Pass performance streams through the encoder
            # TODO tomorrow: debug both those two files side by side to spot any difference
            batched_encoder_output = self.forward_encoder(
                input_streams=performance_streams,
                onset_times=performance_onset_times,
                attention_mask=performance_attention_mask,
            )

            # Generation loop
            init_pos = batched_tokens_tensor.shape[1]
            past_key_values = None
            for pos in (
                pbar := trange(
                    init_pos,
                    max_generated_length,
                    disable=not progress_bar,
                    leave=progress_bar_leave,
                )
            ):
                # Pass the last token if there is already cache, else the whole sequence
                if past_key_values is None:
                    batched_input_tokens = batched_tokens_tensor
                    batched_input_states = batched_states_tensor if self.states_prediction else None
                else:
                    batched_input_tokens = batched_tokens_tensor[:, -1:]
                    batched_input_states = (
                        batched_states_tensor[:, -1:] if self.states_prediction else None
                    )
                tokens_attention_mask = (  # Computed for whole sequence
                    batched_tokens_tensor[..., 0].as_tensor() != PADDING_VALUE
                )
                write_compare_to_file(batched_input_tokens[0], f"input_tokens_{pos}.txt")
                batched_decoder_outputs = self.forward_decoder(
                    input_tokens=batched_input_tokens,
                    input_states=batched_input_states,
                    attention_mask=tokens_attention_mask,
                    encoder_hidden_states=batched_encoder_output,
                    encoder_attention_mask=performance_attention_mask,
                    encoder_positions=(
                        performance_onset_times if self.time_based_cross_attention else None
                    ),
                    past_key_values=past_key_values if use_cache else None,
                    use_cache=use_cache,
                )
                last_hidden_state = batched_decoder_outputs.last_hidden_state[:, -1:]
                if use_cache:
                    past_key_values = batched_decoder_outputs.past_key_values

                # Pass decoder outputs through the router
                batched_token_type_logits = self.forward_router(last_hidden_state)
                batched_token_type_logits = F.softmax(batched_token_type_logits, dim=-1)
                batched_model_want_to_predict = torch.argmax(batched_token_type_logits, dim=-1)

                # Filter logits for the allowed token types
                if constraint:
                    batched_allowed_token_types = []
                    batched_token_type_masked = []
                    for sequence, automaton, flag in zip(
                        batched_token_type_logits, automata, batched_finished_flag
                    ):
                        if flag[0] == GenerationState.UNFINISHED.value:
                            allowed_token_types_idx = [
                                TokenTensor.TOKEN_INDEXER[t]
                                for t in automaton.allowed_tokens_types()
                                if t is not tk.StartOfScore
                            ]
                        else:  # Finished or errored
                            allowed_token_types_idx = [TokenTensor.TOKEN_INDEXER[tk.EndOfScore]]
                        if not allowed_token_types_idx:
                            raise RuntimeError("No token type allowed, this should not happen")
                        batched_allowed_token_types.append(allowed_token_types_idx)
                        batched_token_type_masked.append(
                            self._mask_unallowed(
                                sequence,
                                allowed_indices=allowed_token_types_idx,
                                mask_value=-1,
                                inplace=False,
                            )
                        )
                    batched_token_type_logits_masked = torch.stack(batched_token_type_masked)
                else:
                    batched_token_type_logits_masked = batched_token_type_logits

                # if return_logits_and_mask:
                #     # TODO: make this works with constrains
                #     mask = torch.zeros(
                #         batched_token_type_logits.shape[-1],
                #         dtype=torch.bool,
                #         device=batched_token_type_logits.device,
                #     )
                #     mask[allowed_token_types_idx] = True
                #     logits_and_masks.append([batched_token_type_logits, mask])

                # Get the predicted token types
                batched_predicted_token_types_tensor = batched_token_type_logits_masked.argmax(
                    dim=-1
                )  # B*1
                next_token_type = [
                    TokenTensor.TOKEN_INDEXER_INV[item_prediction.item()]
                    for item_prediction in batched_predicted_token_types_tensor.squeeze(1)
                ]  # B

                if constraint and not torch.equal(
                    batched_model_want_to_predict, batched_model_want_to_predict
                ):
                    logger.warning(f"At least one prediction was refused by the automaton")

                # Pass the decoder outputs through the correct expert head
                # token_types_mask = {
                #     next_token_type.as_str(): torch.ones(
                #         1, 1, dtype=torch.bool, device=last_hidden_state.device
                #     ),
                #     **{
                #         other_token.as_str(): torch.zeros(
                #             1, 1, dtype=torch.bool, device=last_hidden_state.device
                #         )
                #         for other_token in TokenTensor.TOKEN_INDEXER_INV.values()
                #         if other_token != next_token_type
                #     },
                # }
                batched_token_types_mask = {
                    token_type.as_str(): (batched_predicted_token_types_tensor == token_type_idx)
                    for token_type, token_type_idx in TokenTensor.TOKEN_INDEXER.items()
                }
                if constraint:
                    batched_expert_outputs = self.forward_experts(
                        decoder_outputs=last_hidden_state,  # B * E
                        token_type_masks=batched_token_types_mask,
                        convert_to_tokentensor=False,
                    )
                    batched_next_token = []
                    if self.states_prediction:
                        batched_next_state = []
                    current_index_per_token_type = defaultdict(itertools.count)
                    for i, (next_type, automaton, flag) in enumerate(
                        zip(next_token_type, automata, batched_finished_flag)
                    ):
                        if flag[0] != GenerationState.UNFINISHED.value:
                            batched_next_token.append(tk.EndOfScore())
                            if self.states_prediction:
                                batched_next_state.append(
                                    torch.zeros(TranscriptionState.compute_n_vec_dims(8), device=device)
                                )
                            continue
                        idx = next(current_index_per_token_type[next_type])
                        expert_outputs = {
                            key: values[idx]
                            for key, values in batched_expert_outputs.get(
                                next_type.as_str(), {}
                            ).items()
                        }  # Default None?
                        try:
                            next_token = self.decode_next_token_constrained(
                                token_type=next_type,
                                expert_outputs=expert_outputs,
                                state=automaton.state,
                                max_constraint_iterations=max_constraint_iterations,
                            )
                            found = True
                        except StopIteration as e:
                            if e.args[0] == "max_iterations reached":
                                batched_finished_flag[i, 0] = (
                                    GenerationState.EXCEED_CONSTRAINED_ITERATIONS.value
                                )
                            elif e.args[0] == "No combination found":
                                batched_finished_flag[i, 0] = (
                                    GenerationState.NO_ALLOWED_TOKENS.value
                                )
                            else:
                                raise ValueError(f"Unexpected error: {e}")
                            batched_finished_flag[i, 1] = pos
                            next_token = tk.EndOfScore()
                            found = False
                        if found:
                            automaton.transition(next_token, inplace=True)
                            if self.states_prediction:
                                next_state = automaton.state.vectorize(device=device)
                        else:
                            if self.states_prediction:
                                next_state = torch.zeros(TranscriptionState.compute_n_vec_dims(8), device=device)
                        batched_next_token.append(next_token)
                        if self.states_prediction:
                            batched_next_state.append(next_state)
                    batched_next_token_tensor = TokenTensor.from_tokens(
                        batched_next_token, device=device
                    ).unsqueeze(1)
                    if self.states_prediction:
                        batched_next_state_tensor = torch.stack(batched_next_state).unsqueeze(1)
                else:
                    batched_next_token_tensor = self.forward_experts(
                        decoder_outputs=last_hidden_state,  # 1 * E
                        token_type_masks=batched_token_types_mask,
                        convert_to_tokentensor=True,
                    ).quantize(inplace=True)

                # Mark EOS as finised
                batched_finished_flag[
                    torch.logical_and(
                        batched_next_token_tensor.squeeze(1)["token_type"][
                            ..., TokenTensor.TOKEN_INDEXER[tk.EndOfScore]
                        ]
                        == 1,
                        batched_finished_flag[:, 0] == GenerationState.UNFINISHED.value,
                    )
                ] = torch.tensor([[GenerationState.FINISHED.value, pos]], device=device)

                # Gather stats for progress bar
                if progress_bar:
                    flags, counts = batched_finished_flag[:, 0].unique(return_counts=True)
                    pbar.set_postfix(
                        {
                            GenerationState(flag.item()).name: count.item()
                            for flag, count in zip(flags, counts)
                        }
                    )

                # Make sure to not complete already complete sequences
                # TODO: avoid (and probably fix as it should not work) computations above when constrained
                batched_next_token_tensor[
                    batched_finished_flag[:, 0] != GenerationState.UNFINISHED.value
                ] = TokenTensor.from_tokens(tk.EndOfScore(), device=device)
                batched_tokens_tensor = torch.concat(
                    [batched_tokens_tensor, batched_next_token_tensor], dim=1
                )
                if self.states_prediction:
                    batched_states_tensor = torch.concat(
                        [batched_states_tensor, batched_next_state_tensor], dim=1
                    )

                # print(next_token)
                # print(automata.state)

                if all(batched_finished_flag[:, 0] != GenerationState.UNFINISHED.value):
                    break

        # Mark unfinished scores as max tokens exceeded
        batched_finished_flag[batched_finished_flag[:, 0] == GenerationState.UNFINISHED.value] = (
            torch.tensor([GenerationState.EXCEEDED_MAX_TOKENS.value, pos], device=device)
        )

        if return_logits_and_mask:
            logits, masks = zip(*logits_and_masks)
            return (
                batched_tokens_tensor,
                batched_finished_flag,
                torch.stack(logits, dim=0),
                torch.stack(masks, dim=0),
            )
        else:
            return batched_tokens_tensor, batched_finished_flag

    @staticmethod
    def _mask_unallowed(
        tensor: torch.Tensor,
        allowed_indices: list[int],
        mask_value: float = -torch.inf,
        inplace: bool = False,
    ) -> torch.Tensor:
        """
        Mask the tensor at the specified indices with the given mask value.
        """
        if not inplace:
            tensor = tensor.clone()
        mask = torch.zeros(tensor.shape[-1], dtype=torch.bool, device=tensor.device)
        mask[allowed_indices] = True
        tensor[..., ~mask] = mask_value
        return tensor

    @classmethod
    def decode_next_token_constrained(
        cls,
        token_type: type[tk.AnyToken],
        expert_outputs: dict[str, torch.Tensor],
        state: tk.TranscriptionState,
        max_constraint_iterations: int | None = None,
    ) -> tk.AnyToken:
        # # StartOfScore
        # if token_type == tk.StartOfScore:
        #     return tk.StartOfScore()

        # First, we deal with all tokens without interactions between attributes:
        # EOS, Bar, KS, Clef, ChordEnd, TupletEnd

        # EndOfScore
        if token_type == tk.EndOfScore:
            return tk.EndOfScore()

        # Bar
        if token_type == tk.Bar:
            # numerator = expert_outputs["bar_numerator"][0]
            # denominator_idx = expert_outputs["bar_denominator"].argmax().item()
            # denominator = TokenTensor.TS_DENOMINATOR_INDEXER_INV[denominator_idx]
            time_signature_idx = expert_outputs["bar_time_signature"].argmax().item()
            numerator, denominator = TokenTensor.TS_INDEXER_INV[time_signature_idx]
            is_nominal = bool(expert_outputs["bar_is_nominal"].argmax().item())
            return tk.Bar(
                # numerator=round(clip(numerator.item(), lower=1)),
                numerator=numerator,
                denominator=denominator,
                is_nominal=is_nominal,
            )

        # KeySignature
        if token_type == tk.KeySignature:
            fifths = expert_outputs["key_signature"].argmax().item() - 7
            return tk.KeySignature(accidentals=fifths)

        # Clef
        if token_type == tk.Clef:
            staff_idx = expert_outputs["staff_idx"].argmax().item()
            clef_idx = expert_outputs["clef_type"].argmax().item()
            clef_type = "G" if clef_idx == 0 else "F"
            return tk.Clef(clef=clef_type, staff_idx=staff_idx)

        # Get voices (all subsequent tokens should have a voice_idx)
        voice_outputs = expert_outputs["voice_idx"]

        # ChordEnd
        if token_type == tk.ChordEnd:
            # Find voices that are currently in a chord
            allowed_voices = [
                i for i, v in enumerate(state.voices) if v.in_chord and v.chord.can_resolve
            ]
            voice_idx = (
                cls._mask_unallowed(voice_outputs, allowed_indices=allowed_voices).argmax().item()
            )
            return tk.ChordEnd(voice_idx=voice_idx)

        # TupletEnd
        if token_type == tk.TupletEnd:
            # Find voices that are currently in a tuplet
            allowed_voices = [i for i, v in enumerate(state.voices) if v.in_tuplet]
            voice_idx = (
                cls._mask_unallowed(voice_outputs, allowed_indices=allowed_voices).argmax().item()
            )
            return tk.TupletEnd(voice_idx=voice_idx)

        # Now, we deal with tokens with interactions between attributes (for example, we can't
        # select the voice without it having an impact on the allowed durations).
        # We do that by computing the combination of attributes with the maximum probability,
        # while also being allowed by the automata

        # Note
        if token_type is tk.Note:
            classification_dimensions = [
                dim
                for dim in expert_outputs.keys()
                if TokenTensor.PROBLEM_TYPES[dim] == ProblemType.CLASSIFICATION
            ]
            classification_distributions = [
                F.softmax(value, dim=-1)
                for name, value in expert_outputs.items()
                if name in classification_dimensions
            ]

            def build_token(indices: tuple[int, ...]) -> tk.Note:
                dim_to_index = dict(zip(classification_dimensions, indices, strict=True))
                voice = dim_to_index["voice_idx"]
                staff = dim_to_index["staff_idx"]
                duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
                duration_dots = dim_to_index["duration_dots"]
                duration = tk.SymbolicDuration(type=duration_type, dots=duration_dots)
                step = TokenTensor.NOTE_INDEXER_INV[dim_to_index["note_class"]]
                octave = dim_to_index["note_octave"]
                accidental = dim_to_index["note_accidental"] - 2
                tied = dim_to_index["note_tied"] == 1
                stem_direction = TokenTensor.STEM_INDEXER_INV[dim_to_index["note_stem_direction"]]
                fermata = dim_to_index["fermata"] == 1
                arts_and_orns = {
                    art_or_orn: dim_to_index[art_or_orn] == 1
                    for art_or_orn in ARTICULATIONS_AND_ORNAMENTS
                }
                return tk.Note(
                    voice_idx=voice,
                    staff_idx=staff,
                    duration=duration,
                    note_class=step,
                    octave=octave,
                    accidental=accidental,
                    stem_direction=stem_direction,
                    tied=tied,
                    fermata=fermata,
                    **arts_and_orns,
                )

            best_combination, proba = cls._find_most_probable_combinations(
                distributions=[dist.tolist() for dist in classification_distributions],
                predicate=lambda x: build_token(x).is_token_allowed(state),
                max_iterations=max_constraint_iterations,
            )

            best_token = build_token(best_combination)
            return best_token

        # Rest
        if token_type == tk.Rest:
            classification_dimensions = [
                dim
                for dim in expert_outputs.keys()
                if TokenTensor.PROBLEM_TYPES[dim] != ProblemType.REGRESSION
            ]
            classification_distributions = [
                F.softmax(value, dim=-1)
                for name, value in expert_outputs.items()
                if name in classification_dimensions
            ]

            def build_token(indices: tuple[int, ...]) -> tk.Rest:
                dim_to_index = dict(zip(classification_dimensions, indices, strict=True))
                voice = dim_to_index["voice_idx"]
                staff = dim_to_index["staff_idx"]
                measure_rest = dim_to_index["measure_rest"] == 1
                if measure_rest:
                    duration_type = tk.SymbolicDurationType.WHOLE
                    duration_dots = 0
                else:
                    duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
                    duration_dots = dim_to_index["duration_dots"]
                duration = tk.SymbolicDuration(type=duration_type, dots=duration_dots)
                fermata = dim_to_index["fermata"] == 1
                return tk.Rest(
                    voice_idx=voice,
                    staff_idx=staff,
                    duration=duration,
                    measure_rest=measure_rest,
                    fermata=fermata,
                )

            best_combination, proba = cls._find_most_probable_combinations(
                distributions=[dist.tolist() for dist in classification_distributions],
                predicate=lambda x: build_token(x).is_token_allowed(state),
                max_iterations=max_constraint_iterations,
            )

            best_token = build_token(best_combination)
            return best_token

        # ChordStart
        if token_type == tk.ChordStart:
            classification_dimensions = [
                dim
                for dim in expert_outputs.keys()
                if TokenTensor.PROBLEM_TYPES[dim] != ProblemType.REGRESSION
            ]
            classification_distributions = [
                F.softmax(value, dim=-1)
                for name, value in expert_outputs.items()
                if name in classification_dimensions
            ]

            def build_token(indices: tuple[int, ...]) -> tk.ChordStart:
                dim_to_index = dict(zip(classification_dimensions, indices, strict=True))
                voice = dim_to_index["voice_idx"]
                duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
                duration_dots = dim_to_index["duration_dots"]
                return tk.ChordStart(
                    voice_idx=voice,
                    duration=tk.SymbolicDuration(type=duration_type, dots=duration_dots),
                )

            best_combination, proba = cls._find_most_probable_combinations(
                distributions=[dist.tolist() for dist in classification_distributions],
                predicate=lambda x: build_token(x).is_token_allowed(state),
                max_iterations=max_constraint_iterations,
            )

            best_token = build_token(best_combination)
            return best_token

        # TupletStart
        if token_type == tk.TupletStart:
            # raise NotImplementedError("TupletStart not implemented")
            classification_dimensions = [
                dim
                for dim in expert_outputs.keys()
                if TokenTensor.PROBLEM_TYPES[dim] != ProblemType.REGRESSION
            ]
            classification_distributions = [
                (
                    F.softmax(value, dim=-1)
                    if TokenTensor.PROBLEM_TYPES[name] == ProblemType.CLASSIFICATION
                    # TODO: remove binary classification as everything is now multiclass
                    else torch.concat([(x := F.sigmoid(value)), 1 - x], axis=-1)  # binary classif.
                )
                for name, value in expert_outputs.items()
                if name in classification_dimensions
            ]

            def build_token(indices: tuple[int, ...]) -> tk.TupletStart:
                dim_to_index = dict(zip(classification_dimensions, indices, strict=True))
                voice = dim_to_index["voice_idx"]
                duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
                duration_dots = dim_to_index["duration_dots"]
                n_actual = round(clip(expert_outputs["tuplet_actual_notes"].item(), 1))
                n_normal = round(clip(expert_outputs["tuplet_normal_notes"].item(), 1))
                return tk.TupletStart(
                    n_actual=n_actual,
                    n_normal=n_normal,
                    unit_duration=tk.SymbolicDuration(
                        type=tk.SymbolicDurationType(duration_type),
                        dots=duration_dots,
                    ),
                    voice_idx=voice,
                )

            best_combination, proba = cls._find_most_probable_combinations(
                distributions=[dist.tolist() for dist in classification_distributions],
                predicate=lambda x: build_token(x).is_token_allowed(state),
                max_iterations=max_constraint_iterations,
            )

            best_token = build_token(best_combination)
            return best_token

        elif token_type == tk.TupletEnd:
            # Find voices that are currently in a tuplet
            allowed_voices = [i for i, v in enumerate(state.voices) if v.in_tuplet and not v.in_chord and v.tuplet.remaining_duration == 0]
            voice_idx = (
                cls._mask_unallowed(voice_outputs, allowed_indices=allowed_voices).argmax().item()
            )
            return tk.TupletEnd(voice_idx=voice_idx)

        raise ValueError(f"Unknown token type {token_type}")

    @classmethod
    def _find_most_probable_combinations(
        cls,
        distributions: list[list[float]],
        predicate: Callable[[tuple[int, ...]], bool],
        max_iterations: int | None = None,
    ) -> tuple[tuple[int, ...], float]:
        # Validate distributions
        for distribution in distributions:
            if not isinstance(distribution, list):
                raise ValueError(
                    f"Expected distribution to be a list of floats, got {type(distribution)}"
                )
            if not all(0 <= p <= 1 for p in distribution):
                raise ValueError(f"All probabilities must lay between 0 and 1: {distribution}")

        # Add indices to each distribution and sort them by decreasing probabilities
        # (so that we first explore highly probable combinations)
        sorted_idxs = [
            sorted(range(len(dist)), key=lambda i: dist[i], reverse=True) for dist in distributions
        ]
        n = len(distributions)

        # Function to compute the probability of a combination
        def proba_combination(indices: tuple[int, ...]) -> float:
            p = 1.0
            for dim, pos in enumerate(indices):
                orig_idx = sorted_idxs[dim][pos]
                p *= distributions[dim][orig_idx]
            return p

        # Create the max-heap (min-heap with negative probabilities)
        heap: list[tuple[float, tuple[int, ...]]] = []

        # Keep track of already visited combinations
        visited = set()

        # We start the heap with the most probable item
        # (index 0 in all sorted distributions)
        init = tuple([0] * n)
        init_prob = proba_combination(init)
        heapq.heappush(heap, (-init_prob, init))
        visited.add(init)

        # While there are still items in the heap, yield them
        i = -1
        while heap:
            i += 1
            # Get most probable combination
            neg_p, sorted_tuple = heapq.heappop(heap)
            p = -neg_p
            # Map sorted‐space positions to original indices
            orig_tuple = tuple(sorted_idxs[dim][pos] for dim, pos in enumerate(sorted_tuple))
            # If the predicate is true, return the combination
            if predicate(orig_tuple):
                return orig_tuple, p
            # Generate neighbors by incrementing the position in each dimension
            for dim in range(n):
                if sorted_tuple[dim] + 1 < len(sorted_idxs[dim]):
                    neighbor = list(sorted_tuple)
                    neighbor[dim] += 1
                    neighbor = tuple(neighbor)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        heapq.heappush(heap, (-proba_combination(neighbor), neighbor))

            if max_iterations is not None and i >= max_iterations:
                raise StopIteration("max_iterations reached")

        raise StopIteration("No combination found")

    @property
    def device(self):
        return next(self.parameters()).device
