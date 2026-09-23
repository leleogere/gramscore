"""
Transcribe a performance with support for chunk-based transcription
"""

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

import data.components.transforms as t
import models.automata.tokens as tk
from data.components.score_performance_dataset import ScorePerformanceEntry, ScorePerformanceDataset
from data.pdmx_tokens_datamodule import PdmxTokensDataModule
from models.automata.io import token_sequence_to_partitura, token_sequence_to_partitura_permissive
from models.automata.vectorization_torch import TokenTensor
from utils.generation import GenerationState
from models.tokens_transcriber_module import TokensTranscriberModule
from utils.debug import as_pycharm_link, performance_streams_to_midi

# fmt: off
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-12-02_15-26-21_clean-salad-313/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-12-09_13-57-48_sage-serenity-318/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-01-15_18-44-16_glorious-wildflower-321/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-10_16-03-09.old/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-10_16-03-09_comic-cherry-323/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-10_16-03-09_comic-cherry-323/checkpoints/epoch_088.ckpt").expanduser()
model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-16_16-41-28_spring-sunset-330/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-16_16-41-28_spring-sunset-330/checkpoints/epoch_000.ckpt").expanduser()

# base_path = Path("~/Documents/datasets/asap-dataset").expanduser()
# performance_path = base_path / "Bach/Prelude/bwv_860/YoungS01M.mid"
# performance_path = base_path / "Bach/Prelude/bwv_860/midi_score.mid"
# performance_path = base_path / "Bach/Prelude/bwv_883/LuM01M.mid"
# performance_path = base_path / "Bach/Prelude/bwv_883/midi_score.mid"

base_path = Path("~/Documents/datasets/pdmx").expanduser()

performance_paths = [
    # base_path / "mid/0/0/Qma1a1L2c1vSEuGfdPVSEEWfuYAEoBmjpVESX1ok2imu4b.mid",
    # base_path / "mid/1/24/QmbHzrXuDGNG3HyzLpeBNeZ9wAj4bTDGWsZ5xz56abM8D8.mid",  # Test unfinished
    # base_path / "mid/13/29/QmVKam2PcfGjsMF1QtwWEwSgG5bqFyiBRGiF5btFQX6vmV.mid",  # Test grace notes
    # base_path / "mid/0/0/Qma1a2YHEJ8j9gxLCxgALAfjYaiYJQogh1bgyFSwNSRPCs.mid",  # Polyphonic, long
    # base_path / "mid/1/11/Qmbbh58jD1ZLT1vevNsnyEZkexnx2USTFz6FSHX3KUiAzC.mid",  # Monophonic
    # base_path / "mid/15/12/QmXB8B4F9cbHUrnrP4mRGyL3vyhTkyDjuVEbM3VdFu2EY9.mid",  # Polyphonic, chords
    # base_path / "mid/1/32/QmbMQQPWGZpGiTV5gngSwmjTbqd7XosW8yyPwkSHnXbhY2.mid",  # Polyphonic, chords
    # Path("/home/gerel/Documents/Perso/Compositions/La Quête d'Ewilan/La Quête d'Ewilan.mid")
    Path("/home/gerel/Téléchargements/2026-02-25_21-01-14_729n_67s_STARRED.mid")
]

reference_score_paths = [
    # base_path / "mxl" / perf_path.relative_to(base_path / "mid").with_suffix(".mxl") for perf_path in performance_paths
]

print("Reference score paths:\n" + "\n  - ".join([p.as_posix() for p in reference_score_paths]))
print("Performance paths:\n" + "\n  - ".join([p.as_posix() for p in performance_paths]))
# fmt: on

if len(performance_paths) > 1:
    raise NotImplementedError("Only one performance path is supported for now")

NOTES_PER_CHUNK = 500
TOKENS_LIMIT_PER_CHUNK = 1000


# Load the model
model: TokensTranscriberModule = TokensTranscriberModule.load_from_checkpoint(
    model_path,
    map_location="cpu",
    weights_only=False,
)
model.eval()


# dataset = ScorePerformanceDataset(
#     entries=[
#         ScorePerformanceEntry(
#             xml_score=score,
#             midi_performance=performance,
#             base_path=base_path,
#         )
#         for score, performance in zip(reference_score_paths, performance_paths)
#     ],
# transforms=[
#     t.ToDict(),
#     # t.ScoreReader(),
#     # t.VoiceRestFiller(),
#     # t.VoiceReorderer(),
#     t.PerformanceReader(),
#     # t.QuantizedMusicChunker(num_performed_notes=MAX_PERFORMED_NOTES, mode=0),
#     t.PerformanceVectorizerUnbucketized(),
#     t.VectorizedPerformanceStreamsBucketizer(),
#     # t.ScoreVectorizer(),
# ],
# )

# Transcribe performance
# performances = PdmxTokensDataModule.collate_fn(
#     list(iter(dataset)), keep_fields="all", skip_none=False, skip_error_key=False
# )
# print(f"{performances.keys() = }")
# print(f"{performances['performance_onset_times'].shape = }")

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
    transcribed_tokens: list[tk.AnyToken],
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


performance_partial_vectorizer = t.TransformChainer(
    [
        t.PerformanceReader(),
        t.PerformanceToPerformedNoteList(),
        t.PerformedNoteListVectorizer(),
    ]
)  # Do not bucketize yet
performance_bucketizer = t.VectorizedPerformanceStreamsBucketizer()
performance = performance_partial_vectorizer(
    {
        "entry": ScorePerformanceEntry(
            xml_score=None,
            midi_performance=performance_paths[0],
            base_path=base_path,
        )
    }
)

performance_streams_raw = performance["performance_streams_raw"]
performance_pitches = performance_streams_raw["pitch"]
performance_durations = performance_streams_raw["duration"]
performance_onset_times = performance["performance_onset_times"]
performance_offset_times = performance_onset_times + performance_durations

use_cache = True
constrained = False
max_constraint_iterations = 10
measures_overlap = 1

total_notes = len(performance["performance_notes"])
start_time = 0.0
prefix = None
full_transcribed_sequence = []
last_chunk = False

while True:
    print(f"Transcribing chunk from {start_time}")

    # # Get the notes for the chunk
    # # We prefer manipulating the note list as the clipping at chunk start is tricky, even if it
    # # would be faster with tensors (as not re-vectorizing at each chunk)
    # chunk_note_list = performance["performance_notes"][start_note:end_note]

    # Get the notes for the chunk, i.e., the note starting or ending after start_time
    # (limiting their number to NOTES_PER_CHUNK)
    included_notes = torch.argwhere(
        torch.logical_or(
            performance_onset_times >= start_time, performance_offset_times > start_time
        )
    )[:, 0]
    remaining_notes = len(included_notes)
    # We allow transcribing a few more notes to avoid an almost empty chunk at the end
    if len(included_notes) <= NOTES_PER_CHUNK + 10:
        last_chunk = True
    else:
        included_notes = included_notes[:NOTES_PER_CHUNK]
    chunk_streams_raw = {k: stream[included_notes] for k, stream in performance_streams_raw.items()}

    # Clip notes starting before the chunk start time
    indices_to_clip = torch.argwhere(chunk_streams_raw["onset"] < start_time)
    chunk_streams_raw["duration"][indices_to_clip] -= (
        start_time - chunk_streams_raw["onset"][indices_to_clip]
    )
    chunk_streams_raw["onset"][indices_to_clip] = start_time

    # Offset by the chunk start time
    chunk_streams_raw["onset"] -= start_time

    performance_streams_to_midi(chunk_streams_raw).write(f"/tmp/chunk_{start_time:.2f}.mid")

    # Bucketize the streams
    chunk_streams = performance_bucketizer.process(chunk_streams_raw)

    transcribed_tokens_tensor, finished_flags = model.transcriber_model.generate(
        performance_streams={k: v.unsqueeze(0) for k, v in chunk_streams.items() if k != "pad"},
        performance_onset_times=chunk_streams_raw["onset"].unsqueeze(0),
        performance_attention_mask=None,  # TODO: fix padding mask for batched transcription
        prefix=prefix,
        constraint=constrained,
        max_constraint_iterations=max_constraint_iterations,
        max_generated_length=TOKENS_LIMIT_PER_CHUNK,
        use_cache=use_cache,
        progress_bar=True,
        return_logits_and_mask=False,
    )
    transcribed_tokens_tensor = transcribed_tokens_tensor.squeeze(0)
    finished_flags = finished_flags.squeeze(0)

    transcribed_tokens = transcribed_tokens_tensor.to_tokens()
    with open(f"/tmp/chunk_{start_time:.2f}.tokens", "w") as f:
        f.write("\n".join(str(t) for t in transcribed_tokens))
    pt.save_musicxml(
        token_sequence_to_partitura_permissive(transcribed_tokens),
        f"/tmp/chunk_{start_time:.2f}.musicxml",
    )

    flag = GenerationState(finished_flags[0].item())
    if flag != GenerationState.FINISHED:
        logger.error(f"Generation failed at chunk starting at time {start_time} with flag {flag}")
        exit(1)

    num_prefix_tokens = prefix.shape[1] if prefix is not None else 0

    # If last chunk, simply add the end to the full sequence
    if last_chunk:
        full_transcribed_sequence.extend(transcribed_tokens[num_prefix_tokens:])
        break
    # Else, realign the end of the chunk transcription and performance
    else:
        # Get the last aligned downbeats in the performance
        aligned_downbeats = get_last_aligned_downbeats(
            performance_pitches=chunk_streams_raw["pitch"].tolist(),
            transcribed_tokens=transcribed_tokens,
            min_onsets=3,
            downbeats_to_fetch=[0] if measures_overlap == 0 else [0, measures_overlap],
        )
        if measures_overlap == 0:
            assert len(aligned_downbeats) == 1
            last_transcribed_bar_idx, last_performance_db_idx = aligned_downbeats[0]
            token_context_start = last_transcribed_bar_idx
            performance_context_start = last_performance_db_idx
        else:
            assert len(aligned_downbeats) == 2
            (
                (token_context_start, performance_context_start),
                (last_transcribed_bar_idx, last_performance_db_idx),
            ) = aligned_downbeats

        # Now, drop the last transcribed measure (as likely incomplete due hard clipping in the performance)
        transcribed_tokens = transcribed_tokens[:last_transcribed_bar_idx]
        full_transcribed_sequence.extend(transcribed_tokens[num_prefix_tokens:])
        prefix = transcribed_tokens[token_context_start:]

        # Insert the context elements at the context start (clefs, then key sig.)
        context_clefs = {}
        context_key_signature = None
        already_in_context = list(
            itertools.takewhile(lambda t: isinstance(t, (tk.Clef, tk.KeySignature)), prefix[1:])
        )
        for token in transcribed_tokens[:token_context_start] + already_in_context:
            if isinstance(token, tk.Clef):
                context_clefs[token.staff_idx] = token
            elif isinstance(token, tk.KeySignature):
                context_key_signature = token
        new_context = [copy.copy(c) for s, c in sorted(context_clefs.items())] + [
            copy.copy(context_key_signature)
        ]
        prefix[1 : len(already_in_context) + 1] = new_context  # +1 as we skip the bar in takewhile

        # Insert SOS
        prefix.insert(0, tk.StartOfScore())

        # Convert to tensor
        prefix = TokenTensor.from_tokens(prefix).unsqueeze(0)

        # Set the next start time
        old_start_time = start_time
        start_time = chunk_streams_raw["onset"][performance_context_start] + start_time
        if start_time == old_start_time:
            raise RuntimeError(
                "The required measure overlap is too large relative to the chunk size, meaning "
                "that the model is not able to generate more than the prefix passed. Either "
                "reduce the the measure overlap/minimum of notes to register a downbeat, or "
                "increase the chunk size/maximum tokens length."
            )


transcribed_score = token_sequence_to_partitura_permissive(full_transcribed_sequence)
output_path = (
    Path("/tmp") / f"transcribed_score_chunk_constrained={constrained}_cache={use_cache}.musicxml"
)
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
