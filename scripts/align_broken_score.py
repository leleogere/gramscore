"""
Try to align a broken token sequence from the end.
"""

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
test = 3
# test = 4

if test == 1:
    # Simple case
    performance_path = asap_path / "Bach/Fugue/bwv_846/Shi05M.mid"
    score_path = performance_path.parent / "xml_score.musicxml"
    num_performance_notes = 500
    last_bar = 19
    last_bar_num_notes = 10
    should_find = 10
elif test == 2:
    # With a trill
    performance_path = asap_path / "Bach/Fugue/bwv_846/Shi05M.mid"
    score_path = performance_path.parent / "xml_score.musicxml"
    num_performance_notes = 518
    last_bar = 19
    last_bar_num_notes = 27
    should_find = 28
elif test == 3:
    # With a tied note
    performance_path = asap_path / "Bach/Fugue/bwv_846/Shi05M.mid"
    score_path = performance_path.parent / "xml_score.musicxml"
    num_performance_notes = 552  # Stop on last F#4 on right hand M20
    last_bar = 20
    last_bar_num_notes = 28  # Number of *score* notes in the last bars after last onset
    should_find = 26  # The bar line should fall that number of notes *before* the last one
    # Error of 1 on that one because of an extra C4 played on the same time as the F#3 and G4
elif test == 4:
    # With a long tied note
    performance_path = Path("~/Documents/MuseScore4/Scores/tests.mid").expanduser()
    score_path = performance_path.parent / "tests.musicxml"
    num_performance_notes = 6  # Stop on G4 in M3
    last_bar = 3
    last_bar_num_notes = 2
    should_find = 1
else:
    raise ValueError("Invalid test")


def note_number_to_name(pitch: int) -> str:
    pitch_str = str(pitch).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
    return f"{pm_note_number_to_name(pitch)}{pitch_str}"


def counter_to_string(c: Counter) -> str:
    return " / ".join([f"{note_number_to_name(p)}:{n}" for p, n in sorted(c.items())])


# Load score and simulate the tokenization
score = pt.load_musicxml(score_path)[0]
fill_rests(score)
# score = reorder_voices(score)
tokens = partitura_to_token_sequence(score)
performance = read_pretty_midi(performance_path, merge_into_single_track=True)
# for token in tokens:
#     print(token)
# exit(0)

# Get first x notes
clipped_notes = performance.instruments[0].notes[:num_performance_notes]

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


transcription_counter = Counter()
ignored_pitches = set()
first_onset_reached = False
for token in clipped_tokens_reversed:
    print("###", token, counter_to_string(transcription_counter), sep="\n")
    if isinstance(token, tk.Bar) and first_onset_reached:
        break
    elif isinstance(token, tk.Note):
        if token.pitch in ignored_pitches:
            continue
        if token.trill_mark or token.inverted_mordent or token.mordent or token.turn:
            ignore = {token.pitch}
            if token.trill_mark or token.mordent or token.turn:
                ignore.update({token.pitch + 1, token.pitch + 2})  # In case of half or full step
            if token.inverted_mordent or token.turn:
                ignore.update({token.pitch - 1, token.pitch - 2})  # Same
            ignored_pitches.update(ignore)
            print(
                f"Ignoring potentially ornamented notes {[note_number_to_name(p) for p in ignore]}"
            )
            for p in ignore:
                del transcription_counter[p]
            continue
        if hasattr(token, "_tied_prev"):
            print(f"Ignoring tied note {note_number_to_name(token.pitch)}")
            continue
        first_onset_reached = True
        transcription_counter += Counter({token.pitch: 1})
        # print(reversed_cumulative_tokens_sequence[-1])
print(counter_to_string(transcription_counter))


max_notes = 100
performance_counter_history = [Counter()]
initial_score = compute_score(performance_counter_history[-1], transcription_counter)
score_history = [initial_score]
current_score = initial_score
tracker = []
for i, note in enumerate(clipped_notes_reversed):
    if current_score > initial_score + 5 or len(performance_counter_history) > max_notes:
        break
    if note.pitch not in ignored_pitches:
        new_pitch = {note.pitch: 1}
        performance_counter_history.append(
            performance_counter_history[-1] + Counter({note.pitch: 1})
        )
        current_score = compute_score(performance_counter_history[-1], transcription_counter)
        score_history.append(current_score)
        tracker.append(
            (
                i,
                note.pitch,
                note_number_to_name(note.pitch),
                note.start,
                note.duration,
                current_score,
                counter_to_string(performance_counter_history[-1]),
            )
        )

# Print tracker
print(
    tabulate.tabulate(
        tracker,
        headers=["Index", "Pitch", "Name", "onset", "Duration", "Score", "Counter"],
        tablefmt="github",
    )
)

best_score_idx = tracker[score_history.index(min(score_history))][0]
print(
    f"Best score: {performance_counter_history[best_score_idx]} "
    f"at position {best_score_idx} from the end (should find {should_find}: "
    f"{'OK' if best_score_idx == should_find else 'KO'})"
)
print(
    "Performed notes in last measure: "
    + " ".join([note_number_to_name(note.pitch) for note in clipped_notes[-best_score_idx:]])
)
