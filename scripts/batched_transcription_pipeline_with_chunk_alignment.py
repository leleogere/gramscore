"""
Transcribe a performance with support for chunk-based transcription
"""

import torch.nn.functional as F

import itertools

import pretty_midi as pm

import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import partitura as pt
import torch
from loguru import logger
from pretty_midi import note_number_to_name
from torch.nn.utils.rnn import pad_sequence

import data.components.transforms as t
import models.automata.tokens as tk
from data.components.score_performance_dataset import ScorePerformanceEntry, ScorePerformanceDataset
from data.pdmx_tokens_datamodule import PdmxTokensDataModule
from models.automata.io import token_sequence_to_partitura, token_sequence_to_partitura_permissive
from models.automata.vectorization_torch import TokenTensor
from utils.generation import GenerationState
from models.tokens_transcriber_module import TokensTranscriberModule
from utils.constants import PADDING_VALUE
from utils.debug import as_pycharm_link, performance_streams_to_midi

# fmt: off
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-12-02_15-26-21_clean-salad-313/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-12-09_13-57-48_sage-serenity-318/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-01-15_18-44-16_glorious-wildflower-321/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-10_16-03-09.old/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-10_16-03-09_comic-cherry-323/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-10_16-03-09_comic-cherry-323/checkpoints/epoch_088.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-16_16-41-28_spring-sunset-330/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-27_19-03-52.736565_youthful-wave-390/checkpoints/last.ckpt").expanduser()
model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-05-19_18-02-40.669697_legendary-totem-1401_2/checkpoints/last.ckpt").expanduser()

# base_path = Path("~/Documents/datasets/asap-dataset").expanduser()
# performance_path = base_path / "Bach/Prelude/bwv_860/YoungS01M.mid"
# performance_path = base_path / "Bach/Prelude/bwv_860/midi_score.mid"
# performance_path = base_path / "Bach/Prelude/bwv_883/LuM01M.mid"
# performance_path = base_path / "Bach/Prelude/bwv_883/midi_score.mid"

base_path = Path("~/Documents/datasets/pdmx").expanduser()

performance_paths = [
    base_path / "mid/0/0/Qma1a1L2c1vSEuGfdPVSEEWfuYAEoBmjpVESX1ok2imu4b.mid",
    base_path / "mid/1/24/QmbHzrXuDGNG3HyzLpeBNeZ9wAj4bTDGWsZ5xz56abM8D8.mid",  # Test unfinished
    base_path / "mid/0/0/Qma1a2YHEJ8j9gxLCxgALAfjYaiYJQogh1bgyFSwNSRPCs.mid",  # Polyphonic, long
    base_path / "mid/1/11/Qmbbh58jD1ZLT1vevNsnyEZkexnx2USTFz6FSHX3KUiAzC.mid",  # Monophonic
    base_path / "mid/13/29/QmVKam2PcfGjsMF1QtwWEwSgG5bqFyiBRGiF5btFQX6vmV.mid",  # Test grace notes
    base_path / "mid/15/12/QmXB8B4F9cbHUrnrP4mRGyL3vyhTkyDjuVEbM3VdFu2EY9.mid",  # Polyphonic, chords
    base_path / "mid/1/32/QmbMQQPWGZpGiTV5gngSwmjTbqd7XosW8yyPwkSHnXbhY2.mid",  # Polyphonic, chords
    # Path("/home/gerel/Documents/Perso/Compositions/La Quête d'Ewilan/La Quête d'Ewilan.mid"),
    # Path("/home/gerel/Téléchargements/2026-02-25_21-01-14_729n_67s_STARRED.mid"),
]

reference_score_paths = [
    base_path / "mxl" / perf_path.relative_to(base_path / "mid").with_suffix(".mxl") for perf_path in performance_paths
]

print("Reference score paths:\n" + "\n  - ".join([p.as_posix() for p in reference_score_paths]))
print("Performance paths:\n" + "\n  - ".join([p.as_posix() for p in performance_paths]))
# fmt: on

NOTES_PER_CHUNK = 100
NOTES_PER_CHUNK_TOLERANCE = 10  # 10
TOKENS_LIMIT_PER_CHUNK = 1000
MAX_CHUNKS = 100


# Load the model
model: TokensTranscriberModule = TokensTranscriberModule.load_from_checkpoint(
    model_path,
    map_location="cpu",
    weights_only=False,
)
model.eval()


dataset = ScorePerformanceDataset(
    entries=[
        ScorePerformanceEntry(
            xml_score=score,
            midi_performance=performance,
            base_path=base_path,
        )
        for score, performance in zip(reference_score_paths, performance_paths)
    ],
    transforms=[
        t.ToDict(),
        # Also read score info for debugging and comparing
        t.ScoreReader(),
        t.ScoreVectorizer(),
        t.PerformanceReader(),
        t.PerformanceToPerformedNoteList(),
        t.PerformedNoteListVectorizer(),
        # Here we simply load the performance and vectorize the note list, no bucketing yet as it's
        # more direct to deal with explicit pitches for the chunking/realigning rather then reinfer
        # them from the buckets. The bucketing will be done manually after each chunking.
    ],
)

# Transcribe performance
performances = PdmxTokensDataModule.collate_fn(
    list(iter(dataset)),
    keep_fields="all",
    skip_none=False,
    skip_error_key=False,
)
print(f"{performances.keys() = }")
print(f"{performances['performance_onset_times'].shape = }")
# performances = list(iter(dataset))

# # pt.save_musicxml(performances["score"][0], Path("/tmp/reference.musicxml"))
# reconstructed_score = token_sequence_to_partitura_permissive(performances["score_tokens"][0])
# pt.save_musicxml(reconstructed_score, Path("/tmp/reconstructed.musicxml"))
# exit(0)


@dataclass
class PitchCounter:
    count = 0
    could_be_more = False

    def increment(self, count: int = 1):
        self.count += count

    def mark_minimum(self):
        self.could_be_more = True

    def unmark_minimum(self):
        self.could_be_more = False


class PitchTracker:
    def __init__(self):
        self.counters: dict[int | tuple[int, ...], PitchCounter] = defaultdict(PitchCounter)
        self.merged_pitches: dict[int, tuple[int, ...]] = {}

    def add(self, pitch: int | tuple[int, ...], count: int = 1):
        pitches = self.merged_pitches.get(pitch, pitch)
        self.counters[pitches].increment(count)

    def merge(self, first_pitch: int, *other_pitches: int):
        new_key = tuple(sorted([first_pitch, *other_pitches]))
        for pitch in new_key:
            # If already merged, merge with existing merged key
            if pitch in self.merged_pitches:
                previous_merge = self.merged_pitches[pitch]
                all_pitches = set(previous_merge).union(set(new_key))
                new_key = tuple(sorted(all_pitches))
                for p in previous_merge:
                    self.merged_pitches[p] = new_key
        self.counters[new_key].increment(sum(self.counters[pitch].count for pitch in new_key))
        for pitch in new_key:
            self.merged_pitches[pitch] = new_key
            if pitch in self.counters:
                del self.counters[pitch]

    def mark_minimum(self, pitch: int):
        pitches = self.merged_pitches.get(pitch, pitch)
        self.counters[pitches].mark_minimum()

    def __getitem__(self, pitch: int) -> PitchCounter:
        return self.counters[pitch]

    def __setitem__(self, pitch: int, count: int):
        self.counters[pitch].count = count

    def __str__(self):
        elements = []
        for p, c in sorted(
            self.counters.items(),
            key=lambda p_c: (p_c[0] if isinstance(p_c[0], int) else p_c[0][0]),
        ):
            if isinstance(p, tuple):
                element = f"<{'|'.join(note_number_to_name(p) for p in p)}>:{c.count}"
            else:
                element = f"{note_number_to_name(p)}:{c.count}"
            if c.could_be_more:
                element += "+"
            elements.append(element)
        return " / ".join(elements)

    def __delitem__(self, pitch: int):
        del self.counters[pitch]

    @staticmethod
    def subscript_number(s: str | int) -> str:
        return str(s).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))

    @classmethod
    def note_number_to_name(cls, pitch: int) -> str:
        pitch_str = cls.subscript_number(pitch)
        return f"{note_number_to_name(pitch)}{pitch_str}"

    def compute_score(self, other: Self) -> int:
        diff = 0
        assert self.merged_pitches == other.merged_pitches
        unique_keys = sorted(
            set(self.counters.keys()).union(set(other.counters.keys())),
            key=lambda x: x if isinstance(x, int) else x[0],
        )
        for key in unique_keys:
            counter = self.counters.get(key, PitchCounter())
            other_counter = other.counters.get(key, PitchCounter())
            if (
                (counter.could_be_more and other_counter.could_be_more)
                or (counter.could_be_more and counter.count <= other_counter.count)
                or (other_counter.could_be_more and other_counter.count <= counter.count)
            ):
                pass  # Diff not counted in that case
            else:
                diff += abs(counter.count - other_counter.count)
        return diff

    def copy(self, deep: bool = False):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    @classmethod
    def from_merged_pitches(cls, merged_pitches: dict[int, tuple[int, ...]]):
        tracker = cls()
        tracker.merged_pitches = merged_pitches
        return tracker


def get_last_aligned_downbeats(
    performance_pitches: list[int],  # L
    transcribed_tokens: list[tk.AnyToken],  # N
    downbeats_to_fetch: list[int] | None,
    min_onsets: int = 3,
) -> list[tuple[int, int, float]]:
    """
    Find the position of the last downbeats in the performance.

    :param performance_pitches: List of performance pitches.
    :param transcribed_tokens: List of transcribed tokens.
    :param downbeats_to_fetch: List of (reversed) indices of downbeats to fetch. Defaults to [0],
        which is the last *valid* downbeat. The indices start there, meaning that passing [0, 2]
        will fetch the first *valid* downbeats, and the 3rd valid one after that. Passing only [2]
        will only fetch the 3rd valid, but still from the first valid downbeat.
    :param min_onsets: Minimum number of onsets to mark a downbeat valid (allows avoiding bars with
        only 1 note, that could easily result in a bad alignment in case of noise).
    :return: A list of tuples containing, for each downbeat fetched:
        - the index of a Bar token in the transcribed token sequence
        - the index of the first note of that bar in the performed notes sequence
        Note that the list is sorted by ascending time, i.e. the last item corresponds to the last
        downbeat, aka the one marked as [0] in downbeats_to_fetch.
    """
    if downbeats_to_fetch is None:
        downbeats_to_fetch = [0]
    downbeats_to_fetch = sorted(set(downbeats_to_fetch))  # Make sure unique and sorted
    bar_token_indices = [
        i for i, token in enumerate(transcribed_tokens) if isinstance(token, tk.Bar)
    ]
    if max(downbeats_to_fetch) >= len(bar_token_indices):
        logger.warning(
            f"Only {len(bar_token_indices)} bars in transcription, ignoring additional ones"
        )
        downbeats_to_fetch = [i for i in downbeats_to_fetch if i < len(bar_token_indices)]
        if downbeats_to_fetch[-1] != len(bar_token_indices) - 1:
            downbeats_to_fetch.append(len(bar_token_indices) - 1)  # Complete context if needed

    # Go through all notes and set their previous tied note
    open_ties: dict[int, int] = {}  # From pitch to sequence index
    for note in transcribed_tokens:
        if isinstance(note, tk.Note):
            if note.pitch in open_ties:
                note._tied_prev = open_ties.pop(note.pitch)
                if note.tied:
                    open_ties[note.pitch] = note
            if note.tied:
                open_ties[note.pitch] = note
    # for note in transcribed_tokens:
    #     if isinstance(note, tk.Note) and hasattr(note, "_tied_prev"):
    #         print(f"* {note}")
    #         print(f"  {note._tied_prev}")

    # Prepare the reversed performance notes and tokens
    transcribed_tokens_reversed = transcribed_tokens[::-1]
    performances_pitches_reversed = performance_pitches[::-1]

    # Find the first adequate measure to run alignment on (no tied note)
    downbeat_counters = {}  # Store the counters at each downbeat when additional_downbeats is used
    transcription_counter = PitchTracker()
    found_onsets = 0
    valid_downbeats_counter = 0
    bar_counter = 0
    valid_downbeat_to_bar_idx = {}
    for token in transcribed_tokens_reversed:
        # print("###", token, transcription_counter, found_onsets, sep="\n")
        if isinstance(token, tk.Bar):
            if found_onsets >= min_onsets:
                # This is a valid downbeat, we store the counter if needed
                if valid_downbeats_counter in downbeats_to_fetch:
                    downbeat_counters[valid_downbeats_counter] = transcription_counter.copy(
                        deep=True
                    )
                valid_downbeat_to_bar_idx[valid_downbeats_counter] = bar_counter
                valid_downbeats_counter += 1
                found_onsets = 0  # Make sure that the next db has at least min_onsets onsets too
                if len(downbeat_counters) >= len(downbeats_to_fetch):
                    break
            bar_counter += 1
        elif isinstance(token, tk.Note):
            if hasattr(token, "_tied_prev"):
                print(f"Ignoring tied note {note_number_to_name(token.pitch)}")
                continue

            # Increment pitch
            transcription_counter.add(token.pitch)
            found_onsets += 1

            # Tweaks for ornament counting
            # For trills, at least one of main pitch, and one of either half or full next step
            if token.trill_mark:
                transcription_counter.mark_minimum(token.pitch)
                # Half/Full step above
                transcription_counter.merge(token.pitch + 1, token.pitch + 2)
                transcription_counter.add(token.pitch + 1)  # Also update the +2 as linked
                transcription_counter.mark_minimum(token.pitch + 1)  # Also update the +2 as linked
            # For mordents, one more occurrence of main note, and one more occurrence of half or full step above
            if token.mordent:
                transcription_counter.add(token.pitch)
                transcription_counter.merge(token.pitch + 1, token.pitch + 2)
                transcription_counter.add(token.pitch + 1)  # Also update the +2 as linked
            # For inverted mordents, one more occurrence of main note, and one more occurrence of half or full step below
            if token.inverted_mordent:
                transcription_counter.add(token.pitch)
                transcription_counter.merge(token.pitch - 1, token.pitch - 2)
                transcription_counter.add(token.pitch - 1)  # Also update the -2 as linked
            # For turns, one more occurrence of main note, and one more occurrence of half or full step above and below
            if token.turn:
                transcription_counter.add(token.pitch)
                transcription_counter.merge(token.pitch + 1, token.pitch + 2)
                transcription_counter.add(token.pitch + 1)  # Also update the +2 as linked
                transcription_counter.merge(token.pitch - 1, token.pitch - 2)
                transcription_counter.add(token.pitch - 1)  # Also update the -2 as linked

    print(f"Downbeat counters:")
    for db_idx, counter in downbeat_counters.items():
        print(f"  Downbeat {db_idx}: {counter}")

    # Now go over the (reversed) notes in the performance, and get the best candidate for each
    # transcription counters
    # QUESTION: is it better to keep the same performance_counter over the different downbeats?
    # or to reset it?
    #   * reset it:
    #      + avoid merging pitches not needed for that measure
    #      - impacted by an error in the previous downbeat detection
    #   * not reset it:  -> solution chosen for now
    #      + not impacted by a previous error
    #      - merges pitches eventually not needed
    performance_counter = PitchTracker.from_merged_pitches(transcription_counter.merged_pitches)
    score_histories = {db_idx: [] for db_idx in downbeat_counters.keys()}
    last_db_score = performance_counter.compute_score(transcription_counter)  # The one from last db
    max_score = last_db_score + 5
    max_notes = 100
    for i, pitch in enumerate(performances_pitches_reversed):
        if last_db_score > max_score or i > max_notes:
            print(f"Reached max score {max_score} or max notes {max_notes}")
            break
        performance_counter.add(pitch)
        for db_idx, counter in downbeat_counters.items():
            score = counter.compute_score(performance_counter)
            score_histories[db_idx].append(score)
        last_db_score = score
    num_notes_consumed = i

    # Print a table of the scores history for each downbeat (one column for each)
    import tabulate

    table = tabulate.tabulate(
        [
            (i, note_number_to_name(pitch), *(s[i] for s in score_histories.values()))
            for i, pitch in enumerate(performances_pitches_reversed[:num_notes_consumed])
        ],
        headers=["Note", "Pitch", *score_histories.keys()],
        tablefmt="github",
    )
    print(table)

    token_idx_to_note_idx = []
    best_score_idx_last_db = 0
    for db_idx, history in score_histories.items():
        # The best score is the minimum one
        # best_score_idx = len(history) - history[::-1].index(min(history)) - 1  # prefer later in case of tie
        best_score_idx = history.index(min(history))  # prefer earlier in case of tie
        if best_score_idx <= best_score_idx_last_db:
            logger.warning(
                f"Best score from db {db_idx} has a lower index than previous one "
                f"({best_score_idx_last_db} vs {best_score_idx}), forcing to look "
                f"for minimum from last best score."
            )
            clipped_history = history[best_score_idx_last_db + 1 :]
            best_score_idx = (
                clipped_history.index(min(clipped_history)) + best_score_idx_last_db + 1
            )
        print(best_score_idx)
        token_idx = bar_token_indices[
            len(bar_token_indices) - valid_downbeat_to_bar_idx[db_idx] - 1
        ]
        pitch_idx = len(performances_pitches_reversed) - best_score_idx - 1
        token_idx_to_note_idx.append((token_idx, pitch_idx))
        best_score_idx_last_db = best_score_idx
    # Check that the list is sorted (both values)
    token_idx_to_note_idx = token_idx_to_note_idx[::-1]
    if not all(
        t1 < t2 and n1 < p2
        for (t1, n1), (t2, p2) in zip(token_idx_to_note_idx, token_idx_to_note_idx[1:])
    ):
        raise ValueError(
            f"token_idx_to_start_time is not sorted, something must have gone wrong: "
            f"{token_idx_to_note_idx}"
        )
    return token_idx_to_note_idx


# performance_partial_vectorizer = t.TransformChainer(
#     [
#         t.PerformanceReader(),
#         t.PerformanceToPerformedNoteList(),
#         t.PerformedNoteListVectorizer(),
#     ]
# )  # Do not bucketize yet
performance_bucketizer = t.VectorizedPerformanceStreamsBucketizer()
# performance = performance_partial_vectorizer(
#     {
#         "entry": ScorePerformanceEntry(
#             xml_score=None,
#             midi_performance=p_path,
#             base_path=base_path,
#         )
#     }
# )

performances_streams_raw = performances["performance_streams_raw"]
performances_pitches = performances_streams_raw["pitch"]  # B*L
performances_durations = performances_streams_raw["duration"]  # B*L
performances_onset_times = performances["performance_onset_times"]  # B*L
performances_offset_times = torch.where(
    performances_onset_times != PADDING_VALUE,
    performances_onset_times + performances_durations,
    PADDING_VALUE,
)  # B*L
performances_num_notes = performances["performance_length"]
batch_size, batch_length = performances_pitches.shape

# TODO: collate is useless, we need to collate *after* each chunking as we need to pad again the
#  performance anyway to make all chunks the same length

use_cache = True
constrained = False
max_constraint_iterations = 10
measures_overlap = 1

device = performances_pitches.device

# batched_start_time contains the timestamp where each transcription in the batch is currently at
start_times = torch.zeros((batch_size, 1))  # B*1
# batched_start_time = torch.FloatTensor([[17.5], [10.0]], device=device)  # B
# Will contain the context prefixes at the start of each chunk (past the first one)
prefixes = None  # B * L_prefix
full_transcribed_sequences = [[] for _ in range(batch_size)]
# Contains flat enum value, chunk index, position of the flag in the chunk
global_finished_flags = torch.full((batch_size, 3), GenerationState.UNFINISHED.value, device=device)
performances_fully_consumed = torch.zeros((batch_size,), dtype=torch.bool, device=device)

# # TODO: remove this debugging
# logger.warning("Debugging, marking third score as finished, make sure to remove afterwards")
# global_finished_flags[1, 0] = GenerationState.FINISHED.value

for chunk_idx in range(MAX_CHUNKS):
    print(f"{chunk_idx = }")
    unfinished_mask = global_finished_flags[:, 0] == GenerationState.UNFINISHED.value
    num_unfinished = unfinished_mask.sum()

    if num_unfinished == 0:
        break

    assert prefixes is None or num_unfinished == prefixes.shape[0]
    print(
        f"Transcribing chunks for {num_unfinished} pieces from {start_times[unfinished_mask].squeeze(-1)}"
    )
    if prefixes is not None:
        print(f"{prefixes.to_tokens()}")

    # All *_uf variables correspond to the corresponding variable masked with the unfinished_mask
    # This allows to continue transcription only on the unfinished pieces, saving compute
    performances_onset_times_uf = performances_onset_times[unfinished_mask]
    performances_offset_times_uf = performances_offset_times[unfinished_mask]
    start_times_uf = start_times[unfinished_mask]
    performances_num_notes_uf = performances_num_notes[unfinished_mask]
    idx_uf_to_global = dict(enumerate(unfinished_mask.nonzero(as_tuple=True)[0].tolist()))

    print(f"{start_times_uf = }")
    print(f"{performances_num_notes_uf = }")
    print(f"{idx_uf_to_global = }")

    # Get all notes for the chunk, basically 500 notes, clipping the ones starting before the
    # required start, and allowing up to 10 extra notes if only a few remaining.

    # We first deal with the chunk start, and mark notes starting or ending after start_time
    included_notes_mask_uf = torch.logical_or(
        performances_onset_times_uf >= start_times_uf,
        performances_offset_times_uf > start_times_uf,
    )
    num_included_notes_uf = included_notes_mask_uf.sum(dim=-1)
    print(f"{num_included_notes_uf = }")
    # TODO: Implement some additional logic to allow for more notes in the chunk in case of very
    #  few notes remaining (to avoid the next chunk being almost empty)
    # # See if there are pieces with very few notes remaining (up to NOTES_PER_CHUNK_TOLERANCE)
    # num_additional_notes = (num_included_notes[num_included_notes <= NOTES_PER_CHUNK_TOLERANCE]).max()
    # if num_additional_notes > 0:
    #     # Pad all tracks up to the max
    #     included_notes_mask = torch.logical_or()
    #

    # Find indices of the first note
    # start_indices_uf = included_notes_mask_uf.long().argmax(dim=-1)
    # print(f"{start_indices_uf = }")
    # Prepare the index mask
    # For example, with start_indices at [5, 3], we construct:
    # [
    #   [5, 6, 7, 8, ..., 54, 55],
    #   [3, 4, 5, 6, ..., 52, 53]
    # ]
    # -> not working in some case (like __NNNN__ parallel to ___NN___) -> use only the mask instead
    num_remaining_notes_uf = included_notes_mask_uf.sum(dim=-1)
    performances_fully_consumed_uf = num_remaining_notes_uf <= NOTES_PER_CHUNK
    max_remaining_notes_uf = min(num_remaining_notes_uf.max(), NOTES_PER_CHUNK)
    notes_indices_uf = included_notes_mask_uf.argsort(dim=-1, stable=True, descending=True)[
        :, :max_remaining_notes_uf
    ]
    performance_padding_mask_uf = ~included_notes_mask_uf.gather(1, notes_indices_uf)
    print(f"{num_included_notes_uf = }")
    print(f"{num_remaining_notes_uf = }")
    print(f"{max_remaining_notes_uf = }")
    print(f"{notes_indices_uf = }")
    print(f"{performances_fully_consumed_uf = }")
    # Here we recompute the padding mask as it could have new elements.
    # For example, with 4 notes per chunk, and start_idx=[4,2]:
    #                 v S1 start index                 v this PAD is new to match other length
    # S1: n0 n1 n2 n3 n4 n5 PA       ==>      n5 n6 PA PA
    # S2: n0 n1 n2 n3 n4 n5 n6       ==>      n2 n3 n4 n5
    #           ^ S2 start index
    # Note: we could compute only the additional PAD and merge them with existing,
    # but it's not costly to recompute them all anyway
    print(f"{performance_padding_mask_uf = }")
    # notes_indices_uf = notes_indices_uf.clamp(max=performances_num_notes_uf.unsqueeze(-1) - 1)
    # print(f"{notes_indices_uf = }")

    def chunk_tensor(
        tensor: torch.Tensor,
        indices: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        chunk_tensor = tensor.gather(1, indices)
        chunk_tensor[padding_mask] = PADDING_VALUE
        return chunk_tensor

    chunk_streams_raw_uf = {
        k: chunk_tensor(stream[unfinished_mask], notes_indices_uf, performance_padding_mask_uf)
        for k, stream in performances_streams_raw.items()
    }

    # Clip notes starting before the chunk start time
    indices_to_clip_uf = torch.where(
        torch.logical_and(
            chunk_streams_raw_uf["onset"] < start_times_uf,
            chunk_streams_raw_uf["onset"] != PADDING_VALUE,
        )
    )
    print(indices_to_clip_uf, indices_to_clip_uf[0].shape, indices_to_clip_uf[1].shape)
    corresponding_start_time_uf = start_times_uf[indices_to_clip_uf[0]].squeeze(1)
    print(corresponding_start_time_uf.shape)
    chunk_streams_raw_uf["duration"][indices_to_clip_uf] -= (
        corresponding_start_time_uf - chunk_streams_raw_uf["onset"][indices_to_clip_uf]
    )
    chunk_streams_raw_uf["onset"][indices_to_clip_uf] = corresponding_start_time_uf

    # Offset by the chunk start time
    chunk_streams_raw_uf["onset"] = torch.where(
        ~performance_padding_mask_uf, chunk_streams_raw_uf["onset"] - start_times_uf, PADDING_VALUE
    )
    print(f"{chunk_streams_raw_uf['onset'] =}")

    # Debug by saving each chunk as MIDI
    # performance_streams_to_midi(chunk_streams_raw).write(f"/tmp/chunk_{start_time:.2f}.mid")
    for i in range(performance_padding_mask_uf.shape[0]):
        chunk_i_streams_raw = {
            k: v[i, ~performance_padding_mask_uf[i]] for k, v in chunk_streams_raw_uf.items()
        }
        performance_name = performances["entry"][idx_uf_to_global[i]].xml_score.stem
        print(f"Saving chunk {chunk_idx} for performance {performance_name}")
        performance_streams_to_midi(chunk_i_streams_raw).write(
            f"/tmp/{performance_name}_chunk_{chunk_idx}.mid"
        )

    # Bucketize the streams
    chunk_streams_uf = performance_bucketizer.process(chunk_streams_raw_uf)

    transcribed_tokens_tensor, finished_flags_and_positions = (
        model.transcriber_model.generate_single_chunk(
            performance_streams={k: v for k, v in chunk_streams_uf.items() if k != "pad"},
            performance_onset_times=chunk_streams_raw_uf["onset"],
            performance_attention_mask=performance_padding_mask_uf.logical_not(),
            prefix=prefixes,
            constraint=constrained,
            max_constraint_iterations=max_constraint_iterations,
            max_generated_length=TOKENS_LIMIT_PER_CHUNK,
            use_cache=use_cache,
            progress_bar=True,
            return_logits_and_mask=False,
        )
    )
    print(transcribed_tokens_tensor.shape, finished_flags_and_positions.shape)

    transcribed_tokens_uf = [
        [tk.StartOfScore()] + [t for t in transcription if not isinstance(t, tk.StartOfScore)]
        for transcription in transcribed_tokens_tensor.to_tokens()
    ]  # Remove eventual left-padded SOS tokens (but still keep one)
    for i, transcription in enumerate(transcribed_tokens_uf):
        performance_name = performances["entry"][idx_uf_to_global[i]].xml_score.stem
        next_start_time = start_times_uf[i].item()
        file_path = Path(f"/tmp/{performance_name}_chunk_{next_start_time:.2f}.tokens")
        with file_path.open("w") as f:
            f.write("\n".join(str(t) for t in transcription))
        pt.save_musicxml(
            token_sequence_to_partitura_permissive(transcription),
            file_path.with_suffix(".musicxml"),
        )

    finished_flags, flag_positions_uf = (
        finished_flags_and_positions[:, 0],
        finished_flags_and_positions[:, 1],
    )
    flags_uf = [GenerationState(f.item()) for f in finished_flags]
    for i, (flag, pos, stime) in enumerate(
        zip(flags_uf, flag_positions_uf.tolist(), start_times_uf.squeeze(-1).tolist())
    ):
        if flag != GenerationState.FINISHED:
            logger.error(
                f"Generation failed for "
                f"{performances['entry'][idx_uf_to_global[i]].midi_performance.as_posix()} "
                f"on chunk {chunk_idx} at t={stime:.2f} with flag {flag.name} at token {pos}."
            )
            global_finished_flags[idx_uf_to_global[i]] = torch.LongTensor(
                [GenerationState.EXCEEDED_MAX_TOKENS.value, chunk_idx, pos]
            )

    num_prefix_tokens_uf = (
        (prefixes[..., 0] != PADDING_VALUE).sum(-1)
        if prefixes is not None
        else torch.zeros(batch_size, dtype=torch.long, device=device)
    )

    # For performances fully consumed, simply finish the transcription
    performances_to_stop_transcription_mask_uf = torch.logical_or(
        performances_fully_consumed_uf,
        finished_flags != GenerationState.FINISHED.value,  # Stop transcription is chunk failed
    )
    for i in performances_to_stop_transcription_mask_uf.nonzero(as_tuple=True)[0].tolist():
        global_finished_flags[idx_uf_to_global[i]] = torch.LongTensor(
            [flags_uf[i].value, chunk_idx, flag_positions_uf[i]]
        )
        full_transcribed_sequences[idx_uf_to_global[i]].extend(
            transcribed_tokens_uf[i][num_prefix_tokens_uf[i] :]
        )
    # If there are still performances not fully consummed, realign their end
    performances_to_continue_transcription_indices_uf = (
        performances_to_stop_transcription_mask_uf.logical_not().nonzero(as_tuple=True)[0].tolist()
    )
    if len(performances_to_continue_transcription_indices_uf) > 0:
        # Get the last aligned downbeats in each performance
        # This is not parallelized as it would be quite challenging
        new_prefixes = []
        for idx in performances_to_continue_transcription_indices_uf:
            transcribed_tokens_unpadded = transcribed_tokens_uf[idx][num_prefix_tokens_uf[idx] :]
            aligned_downbeats = get_last_aligned_downbeats(
                performance_pitches=chunk_streams_raw_uf["pitch"][idx].tolist(),
                transcribed_tokens=transcribed_tokens_uf[idx],
                min_onsets=3,
                downbeats_to_fetch=[0] if measures_overlap == 0 else [0, measures_overlap],
            )
            if measures_overlap == 0:
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
                last_transcribed_bar_idx, last_performance_db_idx = aligned_downbeats[0]
                token_context_start = last_transcribed_bar_idx
                performance_context_start = last_performance_db_idx
            else:
                if len(aligned_downbeats) < 2:
                    logger.error(
                        f"Can't align performance with transcription on last {measures_overlap} "
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
                (
                    (token_context_start, performance_context_start),
                    (last_transcribed_bar_idx, last_performance_db_idx),
                ) = aligned_downbeats

            # Now, drop the last transcribed measure (as likely incomplete due hard clipping in the performance)
            transcribed_tokens_uf[idx] = transcribed_tokens_uf[idx][:last_transcribed_bar_idx]
            full_transcribed_sequences[idx_uf_to_global[idx]].extend(
                transcribed_tokens_uf[idx][num_prefix_tokens_uf[idx] :]
            )
            prefix = transcribed_tokens_uf[idx][token_context_start:]

            # Insert the context elements at the context start (clefs, then key sig.)
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

            # Insert SOS
            prefix.insert(0, tk.StartOfScore())

            # Convert to tensor
            prefix = TokenTensor.from_tokens(prefix)

            # Set the next start time
            next_start_time = (
                chunk_streams_raw_uf["onset"][idx, performance_context_start] + start_times_uf[idx]
            )
            if next_start_time == start_times_uf[idx]:
                logger.error(
                    "The required measure overlap is too large relative to the chunk size, meaning "
                    "that the model is not able to generate more than the prefix passed. Either "
                    "reduce the the measure overlap/minimum of notes to register a downbeat, or "
                    "increase the chunk size/maximum tokens length. Stopping the transcription "
                    "of this performance."
                )
                global_finished_flags[idx_uf_to_global[idx]] = torch.LongTensor(
                    [GenerationState.CHUNK_ALIGNMENT_ERROR.value, chunk_idx, flag_positions_uf[idx]]
                )
                continue

            # Store data for next chunk
            start_times[idx_uf_to_global[idx]] = next_start_time
            new_prefixes.append(prefix)

        if new_prefixes:
            prefixes = pad_sequence(
                new_prefixes, batch_first=True, padding_value=PADDING_VALUE, padding_side="left"
            )
        else:
            break
    else:
        break

for tokens, (flag, chunk, pos), performance_path, score_path, reference_tokens in zip(
    full_transcribed_sequences,
    global_finished_flags,
    performance_paths,
    reference_score_paths,
    performances["score_tokens"],
):
    print()
    print(f"      Performance MIDI: {as_pycharm_link(performance_path)}")
    print(f"       Reference score: {as_pycharm_link(score_path)}")
    transcribed_score = token_sequence_to_partitura_permissive(tokens)
    output_path = (
        Path("/tmp/transcriptions") / f"transcribed"
        f"_{performance_path.stem}"
        f"_bs={batch_size}"
        f"_constrained={constrained}"
        f"_cache={use_cache}"
        f"_flag={GenerationState(flag.item()).name}"
        f"_chunk={chunk.item()}"
        f"_pos={pos.item()}"
        f".musicxml"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pt.save_musicxml(transcribed_score, output_path)

exit(0)
for performance_tokens, flag, performance_path, score_path, reference_tokens in zip(
    full_transcribed_sequence,
    finished_flags,
    performance_paths,
    reference_score_paths,
    performances["score_tokens"],
):
    print()
    print(f"      Performance MIDI: {as_pycharm_link(performance_path)}")
    print(f"       Reference score: {as_pycharm_link(score_path)}")
    output_file_name_parts = [
        score_path.stem,
        f"constrained={constrained}",
        f"cache={use_cache}",
    ]
    # Transcribe permissive
    transcribed_score = token_sequence_to_partitura_permissive(performance_tokens)
    output_path = Path("/tmp") / ("_".join(output_file_name_parts + ["permissive"]) + ".musicxml")
    pt.save_musicxml(transcribed_score, output_path)
    print(f" Transcription (perm.): {as_pycharm_link(output_path)}")
    # Transcribe strict
    try:
        transcribed_score = token_sequence_to_partitura(performance_tokens)
        output_path = Path("/tmp") / ("_".join(output_file_name_parts + ["strict"]) + ".musicxml")
        pt.save_musicxml(transcribed_score, output_path)
        print(f"Transcription (strict): {as_pycharm_link(output_path)}")
    except Exception as e:
        print(f"Transcription (strict): {e}")
    print(f"   Transcription state: {GenerationState(flag[0].item())}@{flag[1].item()}")
    # Save tokens
    transcribed_tokens_path = Path("/tmp") / ("_".join(output_file_name_parts + [".tokens"]))
    reference_tokens_path = Path("/tmp") / ("_".join(output_file_name_parts + ["ref.tokens"]))
    with open(transcribed_tokens_path, "w") as f:
        for token in performance_tokens:
            f.write(str(token) + "\n")
    with open(reference_tokens_path, "w") as f:
        for token in reference_tokens:
            f.write(str(token) + "\n")
    print(f"  Transcription tokens: {as_pycharm_link(transcribed_tokens_path)}")
    print(f"Reference transcription tokens: {as_pycharm_link(reference_tokens_path)}")
    for token in performance_tokens:
        print("  *", token)
