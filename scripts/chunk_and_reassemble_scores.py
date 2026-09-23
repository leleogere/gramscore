"""
Try to cut and reassemble scores to check the realigning algorithm
"""

import copy
import itertools
from pathlib import Path

import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange

import data.components.transforms as t
from data.base_tokens_datamodule import BaseTokensDataModule
from data.components.asap_dataset import AsapDataset
from models.automata.vectorization_torch import TokenTensor
from utils.constants import PADDING_VALUE
from utils.generation import GenerationState, get_last_aligned_downbeats
from models.automata import tokens as tk

asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()

asap = AsapDataset(
    directory=asap_path,
    split="all",
    include_midi_scores=False,
    valid_tsv_file=asap_path / "valid.tsv",
)

entry_reader = t.ToDict()
entry_processor = t.TransformChainer(
    [
        t.ScoreReader(),
        t.VoiceRestFiller(),
        t.VoiceReorderer(),
        t.ScoreVectorizer(),
        t.PerformanceReader(),
        t.PerformanceToPerformedNoteList(),
        t.PerformedNoteListVectorizer(),
    ]
)


batch = []
for piece in asap[:10]:
    piece_dict = entry_reader(piece)
    processed = entry_processor(piece_dict)
    batch.append(processed)

batch = BaseTokensDataModule.collate_fn(batch=batch, keep_fields="all")

performance_streams_raw = batch["performance_streams_raw"]
performance_lengths = batch["performance_lengths"]

max_chunks = 100
chunk_size = 500
measures_interchunk_overlap = 1

streams_bucketizer = t.VectorizedPerformanceStreamsBucketizer()

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
start_times = torch.zeros((batch_size, 1))  # B*1

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
global_finished_flags = torch.zeros((batch_size, 3), dtype=torch.long)  # B*3

# Track performances for which all notes have been consumed
performances_fully_consumed = torch.zeros((batch_size,), dtype=torch.bool)  # B

# Iterate until max chunks (will break earlier if needed)
for chunk_idx in trange(max_chunks, desc="Generating chunks"):
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
        k: chunk_tensor(stream[unfinished_mask], notes_indices_uf, performance_padding_mask_uf)
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

    # Pseudo-transcribe the chunk
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
        [tk.StartOfScore()] + [t for t in transcription if not isinstance(t, tk.StartOfScore)]
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
        else torch.zeros(batch_size, dtype=torch.long)
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
        performances_to_stop_transcription_mask_uf.logical_not().nonzero(as_tuple=True)[0].tolist()
    )
    # If there is at least one, run the realignment pipeline
    if len(performances_to_continue_transcription_indices_uf) > 0:
        # Get the last aligned downbeats in each performance
        # This is not parallelized as it would be quite challenging, so we loop over them
        new_prefixes = []
        for idx in performances_to_continue_transcription_indices_uf:
            transcribed_tokens_unpadded = transcribed_tokens_uf[idx][num_prefix_tokens_uf[idx] :]
            # Get the last aligned downbeats between the transcription and the performance
            aligned_downbeats = get_last_aligned_downbeats(
                performance_pitches=chunk_streams_raw_uf["pitch"][idx].tolist(),
                transcribed_tokens=transcribed_tokens_uf[idx],
                min_onsets=3,
                downbeats_to_fetch=(
                    [0] if measures_interchunk_overlap == 0 else [0, measures_interchunk_overlap]
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
            transcribed_tokens_uf[idx] = transcribed_tokens_uf[idx][:last_transcribed_bar_idx]
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
                itertools.takewhile(lambda t: isinstance(t, (tk.Clef, tk.KeySignature)), prefix[1:])
            )
            for token in transcribed_tokens_uf[idx][:token_context_start] + already_in_context:
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
                chunk_streams_raw_uf["onset"][idx, performance_context_start] + start_times_uf[idx]
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


pass
