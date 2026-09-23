"""
Try to align a broken token sequence from the end, with cutting based on time in addition to
number of notes to avoid issues with very long tied notes across measures.
"""

import copy
from dataclasses import dataclass
from typing import Literal, Self

import data.components.transforms as t

import itertools
from collections import Counter, defaultdict

import pretty_midi as pm

from pathlib import Path

import partitura as pt
import tabulate
from pretty_midi import note_number_to_name as pm_note_number_to_name

import models.automata.tokens as tk
from models.automata.io import partitura_to_token_sequence, token_sequence_to_partitura_permissive
from models.automata.tokens import fill_rests
from utils.midi import read_pretty_midi


asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()

# test = 1
# test = 2
# test = 3
# test = 4
# test = 5
# test = 6
# test = 7
test = 8

if test == 1:  # Simple case
    performance_path = asap_path / "Bach/Fugue/bwv_846/Shi05M.mid"
    score_path = performance_path.parent / "xml_score.musicxml"
    num_performance_notes = 500
    last_bar = 19
    last_bar_num_notes = 10  # Number of *score* notes in the (last bars after last onset)
    should_find = 10
elif test == 2:  # With a trill
    performance_path = asap_path / "Bach/Fugue/bwv_846/Shi05M.mid"
    score_path = performance_path.parent / "xml_score.musicxml"
    num_performance_notes = 518  # Stop before G4 on right hand M19
    last_bar = 19
    last_bar_num_notes = 27  # Number of *score* notes in the (last bars after last onset)
    should_find = 28
elif test == 3:  # With a tied note
    performance_path = asap_path / "Bach/Fugue/bwv_846/Shi05M.mid"
    score_path = performance_path.parent / "xml_score.musicxml"
    num_performance_notes = 552  # Stop on last F#4 on right hand M20
    last_bar = 20
    last_bar_num_notes = 28  # Number of *score* notes in the (last bars after last onset)
    should_find = 26  # The bar line should fall that number of notes *before* the last one
    # Error of 1 on that one because of an extra C4 played on the same time as the F#3 and G4
elif test == 4:  # With a long tied note
    performance_path = Path("~/Documents/MuseScore4/Scores/tests.mid").expanduser()
    score_path = performance_path.parent / "tests.musicxml"
    num_performance_notes = 7  # Stop on A4 in M3
    last_bar = 3
    last_bar_num_notes = 3
    should_find = 2
elif test == 5:  # With a single measure trill
    performance_path = Path("~/Documents/MuseScore4/Scores/trill.mid").expanduser()
    score_path = performance_path.parent / "trill.musicxml"
    num_performance_notes = 23  # Stop on the A4
    last_bar = 2
    last_bar_num_notes = 4
    should_find = 19
elif test == 6:  # With a tie
    performance_path = Path("~/Documents/MuseScore4/Scores/ties_tests.mid").expanduser()
    score_path = performance_path.parent / "ties_tests.musicxml"
    num_performance_notes = 9  # Stop on A4
    last_bar = 2
    last_bar_num_notes = 3
    should_find = 2
elif test == 7:  # With a tie
    performance_path = Path("~/Documents/MuseScore4/Scores/trill_hard.mid").expanduser()
    score_path = performance_path.parent / "trill_hard.musicxml"
    num_performance_notes = 23  # Stop on last B4
    last_bar = 2
    last_bar_num_notes = 4
    should_find = 19
elif test == 8:  # With two ties
    performance_path = Path("~/Documents/MuseScore4/Scores/two_ties.mid").expanduser()
    score_path = performance_path.with_suffix(".musicxml")
    num_performance_notes = 13  # Stop on A5
    last_bar = 4
    last_bar_num_notes = 3
    should_find = 5
else:
    raise ValueError("Invalid test")


def subscript_number(s: str | int) -> str:
    return str(s).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def note_number_to_name(pitch: int) -> str:
    pitch_str = subscript_number(pitch)
    return f"{pm_note_number_to_name(pitch)}{pitch_str}"


def counter_to_string(c: Counter) -> str:
    return " / ".join([f"{note_number_to_name(p)}:{n}" for p, n in sorted(c.count())])


# Load score and simulate the tokenization
score = pt.load_musicxml(score_path)[0]
fill_rests(score)
# score = reorder_voices(score)
tokens = partitura_to_token_sequence(score)
performance = read_pretty_midi(performance_path, merge_into_single_track=True)
# for token in tokens:
#     print(token)
# exit(0)

# Get first x notes, and clip the overflowing ones
clipped_notes = performance.instruments[0].notes[:num_performance_notes]
last_offset = clipped_notes[-1]

##########################################
# SIMULATING A TRANSCRIBED TOKEN SEQUENCE
# Get the tokens corresponding (simulating the end of last measure padded, and tied notes allowed
# to extend over next bars)
bar = last_bar - 1  # fix 0-indexing of python
bar_indices = [i for i, token in enumerate(tokens) if isinstance(token, tk.Bar)] + [len(tokens)]
clipped_tokens = []
currently_tied: set[int] = set()
last_bar_counted_notes = 0
for i, t in enumerate(tokens):
    # If not last bar, simply add
    if (
        in_last_bar := (i >= bar_indices[bar] and last_bar_counted_notes < last_bar_num_notes)
    ) or i < bar_indices[bar]:
        clipped_tokens.append(t)
        # Keep track of tied notes for the end
        if isinstance(t, tk.Note):
            already_tied = t.pitch in currently_tied
            if in_last_bar:
                last_bar_counted_notes += 1
            if t.tied:
                currently_tied.add(t.pitch)
            elif already_tied:  # and not tied again as t.tied is False
                currently_tied.remove(t.pitch)
    # Else, try to end the sequence by only gathering already opened tied notes and replacing
    # other notes with rests
    else:
        # If no more tied notes and end of bar, end
        if len(currently_tied) == 0 and isinstance(t, tk.Bar):
            clipped_tokens.append(tk.EndOfScore())
            break
        if isinstance(t, tk.Note):
            if t.pitch in currently_tied:
                clipped_tokens.append(t)
                if not t.tied:
                    currently_tied.remove(t.pitch)
            else:
                clipped_tokens.append(
                    tk.Rest(duration=t.duration, staff_idx=t.staff_idx, voice_idx=t.voice_idx)
                )
        else:
            clipped_tokens.append(t)
# END SIMULATION
############################

# Go through all notes and set their previous tied note
open_ties: dict[int, tk.Note] = {}
for note in clipped_tokens:
    if isinstance(note, tk.Note):
        if note.pitch in open_ties:
            note._tied_prev = open_ties.pop(note.pitch)
            if note.tied:
                open_ties[note.pitch] = note
        if note.tied:
            open_ties[note.pitch] = note
for note in clipped_notes:
    if isinstance(note, tk.Note) and hasattr(note, "_tied_prev"):
        print(f"* {note}")
        print(f"  {note._tied_prev}")

# Print all tokens in last bar
# for token in clipped_tokens:
#     print(token)
# exit(0)

# Save for debugging purposes
clipped_score = token_sequence_to_partitura_permissive(clipped_tokens)
pt.save_musicxml(clipped_score, "/tmp/clipped_score.musicxml")

# Try to match the notes in the performance to the notes in the last measure in reverse order
clipped_notes_reversed: list[pm.Note] = clipped_notes[::-1]
clipped_tokens_reversed = clipped_tokens[::-1]
clipped_tokens_notes_and_bar_reversed: list[tk.Note | tk.Bar] = [
    t for t in clipped_tokens_reversed if isinstance(t, (tk.Note, tk.Bar))
]


def compute_score(c1: Counter, c2: Counter):
    diff = c2.copy()
    diff.subtract(c1)
    score = sum(map(abs, diff.values()))
    return score


@dataclass()
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

    # def _merge_as_this_one(self, other: Self) -> Self:
    #     """
    #     Merge the counters of another pitch tracker as the current one.
    #     """
    #     if other.merged_pitches:
    #         raise ValueError("Cannot merge other pitch tracker with merged pitches as this one")
    #     new = other.copy(deep=True)
    #     print(new)
    #     for pitch, group in self.merged_pitches.items():
    #         print(pitch, group)
    #         new[group] = 0
    #         for p in group:
    #             print(p)
    #             new[group].increment(self[p].count)
    #             print(f"deleting {p} in {new}")
    #             if p in new:
    #                 del new[p]
    #             print("deleted")
    #     return new


transcription_counter = PitchTracker()
min_notes = 3
num_notes = 0
for token in clipped_tokens_reversed:
    print("###", token, transcription_counter, num_notes, sep="\n")
    if isinstance(token, tk.Bar) and num_notes >= min_notes:
        break
    elif isinstance(token, tk.Note):
        if hasattr(token, "_tied_prev"):
            print(f"Ignoring tied note {note_number_to_name(token.pitch)}")
            continue

        # Increment pitch
        transcription_counter.add(token.pitch)
        num_notes += 1

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

print(transcription_counter)
# exit()

current_counter = PitchTracker.from_merged_pitches(transcription_counter.merged_pitches)
current_score = transcription_counter.compute_score(current_counter)
performance_counter_history = [current_counter]
score_history = [current_score]
tracker = [
    (
        -1,
        "",
        "",
        "",
        "",
        current_score,
        str(current_counter),
    )
]
max_notes = 100
max_score = current_score + 5
for i, note in enumerate(clipped_notes_reversed):
    if current_score > max_score or len(performance_counter_history) > max_notes:
        print(f"Reached max score ({max_score}) or max notes ({max_notes})")
        break
    current_counter.add(note.pitch)
    current_score = transcription_counter.compute_score(current_counter)
    tracker.append(
        (
            i,
            note.pitch,
            note_number_to_name(note.pitch),
            f"{note.start:.5f}",
            f"{note.duration:.5f}",
            current_score,
            str(current_counter),
        )
    )
    performance_counter_history.append(current_counter.copy(deep=True))
    score_history.append(current_score)

# Print tracker
print(
    tabulate.tabulate(
        tracker,
        headers=["Index", "Pitch", "Name", "Start", "Duration", "Score", "Counter"],
        tablefmt="github",
    )
)

print(score_history)
# best_score_idx_including_initial_position = tracker[score_history.index(min(score_history))][0]
# best_score_idx = best_score_idx_including_initial_position + 1
# Get best index, but LAST wins
best_score_idx = len(score_history) - score_history[::-1].index(min(score_history[::-1])) - 1
print(
    f"Best score {score_history[best_score_idx]} with {performance_counter_history[best_score_idx]} "
    f"after {best_score_idx}th note ({tracker[best_score_idx][2]}) from the end (should find {should_find}: "
    f"{'OK' if best_score_idx == should_find else 'KO'})"
)
print(
    "Performed notes in last measure: "
    + " ".join([note_number_to_name(note.pitch) for note in clipped_notes[-best_score_idx:]])
)
