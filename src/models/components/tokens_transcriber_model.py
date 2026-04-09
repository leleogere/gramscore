import heapq
import itertools
from collections import defaultdict
from collections.abc import Mapping, Callable
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import trange
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPast,
)

from models.automata import tokens as tk
from models.automata.vectorization_torch import TokenTensor, ProblemType, clip
from models.components.embedding import MIDIEmbedding
from models.components.embedding import ScoreTokenEmbedding
from models.components.expert_head import ExpertHead
from models.components.roformer import Roformer
from utils.constants import PADDING_VALUE
from utils.debug import write_compare_to_file
from utils.musicxml import ARTICULATIONS_AND_ORNAMENTS


class TokensTranscriberModel(nn.Module):
    def __init__(
        self,
        stream_dimensions: dict[str, int],
        roformer: Roformer | None,
        expert_inner_dims: list[int],
        router_inner_dims: list[int] = None,
        time_based_cross_attention: bool = False,
    ):
        super().__init__()

        self.roformer = roformer
        self.time_based_cross_attention = time_based_cross_attention

        self.embedding_dim = self.roformer.encoder.config.hidden_size
        self.layer_norm_eps = self.roformer.encoder.config.layer_norm_eps
        self.hidden_dropout_prob = self.roformer.encoder.config.hidden_dropout_prob

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

        modules = []
        last_dim = self.embedding_dim
        if router_inner_dims is not None:
            for dim in router_inner_dims:
                modules.append(nn.Linear(last_dim, dim))
                modules.append(nn.ReLU())
                last_dim = dim
        modules.append(nn.Linear(last_dim, len(TokenTensor.TOKEN_INDEXER)))
        self.token_type_router = nn.Sequential(*modules)

        self.expert_inner_dims = expert_inner_dims
        self.expert_heads: Mapping[str, ExpertHead] = nn.ModuleDict(
            {
                token_type.as_str(): ExpertHead(
                    embedding_dim=self.embedding_dim,
                    inner_dims=self.expert_inner_dims,
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
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_positions: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        score_embeddings = self.score_embedding.forward(input_tokens)
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
        expert_head: ExpertHead = self.expert_heads[token_type]
        expert_outputs = expert_head.forward(decoder_outputs)
        return expert_outputs

    def forward_experts(
        self,
        decoder_outputs: torch.Tensor,  # B * T * E
        token_type_masks: dict[str, torch.Tensor],  # B * T
        convert_to_tokentensor: bool = False,
        device=None,
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
            expert_output = expert.forward(involved_decoder_outputs)
            expert_outputs[token_type] = expert_output

        if convert_to_tokentensor:
            return TokenTensor.build_from_dict(
                type_dim_values_dict=expert_outputs,
                token_type_masks=token_type_masks,
                device=device,
            )
        else:
            return expert_outputs

    def generate(
        self,
        performance_streams: dict[str, torch.Tensor],
        performance_onset_times: torch.Tensor,
        performance_attention_mask: torch.BoolTensor = None,  # 1 for attended, 0 for masked
        prefix: TokenTensor = None,
        constraint: bool = False,
        max_constraint_iterations: int = None,
        max_generated_length: int = 1024,
        use_cache: bool = True,
        progress_bar: bool = True,
        return_logits_and_mask: bool = False,
    ) -> TokenTensor | tuple[list[tk.Token], torch.Tensor, torch.Tensor]:
        device = performance_onset_times.device
        if not use_cache:
            raise NotImplementedError("Generating without cache is not implemented yet")
        # Check dimensions
        batch_size, seq_length = performance_onset_times.shape
        if any(s.shape[:-1] != (batch_size, seq_length) for s in performance_streams.values()) or (
            prefix is not None
            and prefix.shape[0] not in (batch_size, 1)
            or (
                performance_attention_mask is not None
                and performance_attention_mask.shape != (batch_size, seq_length)
            )
        ):
            raise ValueError(
                f"Performance streams (B*L*Fi), onset times (B*L) stream, attention mask (B*L) and "
                f"prefix (B*P*E or 1*P*E) must have matching dimensions."
            )

        # Handle eventual logits and masks
        if return_logits_and_mask:
            logits_and_masks = []

        # Initialize eventual automata
        if constraint:
            from models.automata.automata import ConstrainedAutomata

            automata = [
                ConstrainedAutomata(
                    num_voices=16,
                    allowed_pitches=pitch_stream.sum(0)
                    .nonzero(as_tuple=True)[0]
                    .tolist(),  # as pitch stream is OHE
                )
                for pitch_stream in performance_streams["pitch"]
            ]
        elif max_constraint_iterations:
            logger.warning("max_constraint_iterations is not used for unconstrained generation")

        # Handle eventual prefix
        if prefix is None:
            batched_tokens_tensor = TokenTensor.from_tokens(
                [[tk.StartOfScore()]], device=device
            ).expand(batch_size, -1, -1)
            # prefix_attention_mask = None
        else:
            batched_tokens_tensor: TokenTensor = prefix
            # prefix_attention_mask = batched_tokens_tensor[..., 0] != PADDING_VALUE
            # Adapt the prefix tokens to the shift caused by left padding
            # [0, 1, 2, 3...] -> [0, 0, 0, 1, 2, 3...] if padded 2 tokens
            # prefix_tokens_positions = torch.cumsum(~prefix_padding_mask, dim=1) - 1
            # if batched_tokens_tensor.ndim == 1:
            #     batched_tokens_tensor = batched_tokens_tensor.repeat(batch_size)
            if constraint:
                for sequence, automaton in zip(batched_tokens_tensor.to_tokens(), automata):
                    for token in sequence:
                        automaton.transition(token)
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

            # # Pass prefix
            # if batched_tokens_tensor.shape[1] > 1:
            #     batched_decoder_outputs = self.forward_decoder(
            #         input_tokens=batched_tokens_tensor,
            #         encoder_hidden_states=batched_encoder_output,
            #         encoder_attention_mask=performance_attention_mask,
            #         use_cache=use_cache,
            #         past_key_values=None,
            #     )  # TODO: make sure that it is equivalent to multiple forward passes
            #     past_key_values = batched_decoder_outputs.past_key_values
            # else:
            #     past_key_values = None

            # Generation loop
            init_pos = batched_tokens_tensor.shape[1]
            past_key_values = None
            for pos in (pbar := trange(init_pos, max_generated_length, disable=not progress_bar)):
                # Pass the last token if there is already cache, else the whole sequence
                if past_key_values is None:
                    batched_input_tokens = batched_tokens_tensor
                else:
                    batched_input_tokens = batched_tokens_tensor[:, -1:]
                tokens_attention_mask = (  # Computed for whole sequence
                    batched_tokens_tensor[..., 0].as_tensor() != PADDING_VALUE
                )
                write_compare_to_file(batched_input_tokens[0], f"input_tokens_{pos}.txt")
                batched_decoder_outputs = self.forward_decoder(
                    input_tokens=batched_input_tokens,
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

                if return_logits_and_mask:
                    # TODO: make this works with constrains
                    mask = torch.zeros(
                        batched_token_type_logits.shape[-1],
                        dtype=torch.bool,
                        device=batched_token_type_logits.device,
                    )
                    mask[allowed_token_types_idx] = True
                    logits_and_masks.append([batched_token_type_logits, mask])

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
                    current_index_per_token_type = defaultdict(itertools.count)
                    for i, (next_type, automaton, flag) in enumerate(
                        zip(next_token_type, automata, batched_finished_flag)
                    ):
                        if flag[0] != GenerationState.UNFINISHED.value:
                            batched_next_token.append(tk.EndOfScore())
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
                        batched_next_token.append(next_token)
                    batched_next_token_tensor = TokenTensor.from_tokens(
                        batched_next_token, device=device
                    ).unsqueeze(1)
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
                ] = TokenTensor.from_tokens(tk.EndOfScore())
                batched_tokens_tensor = torch.concat(
                    [batched_tokens_tensor, batched_next_token_tensor], dim=1
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
            allowed_voices = [i for i, v in enumerate(state.voices) if v.in_tuplet]
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


class GenerationState(Enum):
    UNFINISHED = 0
    FINISHED = 1
    EXCEEDED_MAX_TOKENS = 2
    EXCEED_CONSTRAINED_ITERATIONS = 3
    NO_ALLOWED_TOKENS = 4
    CHUNK_ALIGNMENT_ERROR = 5
