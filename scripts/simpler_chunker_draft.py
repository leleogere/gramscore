"""
Try to simplify Beyer's chunking algorithm to only use DB and add downbeats_score_map info
"""

import copy
import warnings
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import pandas as pd
import partitura as pt
import partitura.score
import symusic

asap_path = Path("~/Documents/datasets/asap-dataset-original").expanduser()
pieces = [
    # "Beethoven/Piano_Sonatas/29-4",  # Last measure tied
    # "Beethoven/Piano_Sonatas/2-1",  # Last measure empty
    # "Chopin/Sonata_2/2nd_no_repeat",  # 2 last measures tied
    # "Beethoven/Piano_Sonatas/24-1",  # No score map and diff (should error)
    "Haydn/Keyboard_Sonatas/39-2",  # No score map but valid
]
piece_paths = [asap_path / piece for piece in pieces]

metadata_path = asap_path / "metadata.csv"
metadata = pd.read_csv(metadata_path)
print(metadata)

annotations_path = asap_path / "asap_annotations.json"
annotations = pd.read_json(annotations_path, orient="index")
print(annotations)

# %% Select the first available performance
assert len(pieces) == 1, "Only one piece should be selected"
performances_for_this_piece = metadata[metadata["folder"] == pieces[0]]
for _, performance in performances_for_this_piece.iterrows():
    performance_annotations = annotations.loc[performance["midi_performance"]]
    if performance_annotations["downbeats_score_map"] is not None:
        print("Found a score map")
        downbeats_score_map = [
            [int(m) for m in str(group).split("-")]
            for group in performance_annotations["downbeats_score_map"]
        ]
        # Fixes the repetitions indices (as repetitions will be unfolded, we don't want things like
        # 0, 1, 2, 0, 1, 2, 3, 4, 5... but rather 0, 1, 2, 3, 4, 5, 6, 7, 8...)
        i = downbeats_score_map[0][0]  # Not start at 0 as anacrusis is dealt with later
        for group in downbeats_score_map:
            for j in range(len(group)):
                group[j] = i
                i += 1
        break
else:
    print(
        "No downbeats_score_map found for this performance, "
        "using default range 0 to len(midi_score_downbeats)"
    )
    downbeats_score_map = None
init_downbeats_score_map = copy.deepcopy(downbeats_score_map)

xml_score_path = asap_path / performance["xml_score"]
midi_score_path = asap_path / performance["midi_score"]
performance_path = asap_path / performance["midi_performance"]
print(f"{xml_score_path = }")
print(f"{midi_score_path = }")
print(f"{performance_path = }")

# Load score
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    xml_score = pt.load_musicxml(xml_score_path, force_note_ids=True)
    assert len(xml_score.parts) == 1
    xml_score = xml_score.parts[0]
    xml_score: pt.score.Part = pt.score.unfold_part_maximal(xml_score, ignore_leaps=False)

# Get downbeats
midi_score_dbs = performance_annotations["midi_score_downbeats"]
# If not a downbeat at start (anacrusis), add it (TODO: does not work for performances, does it?)
if midi_score_dbs[0] != 0.0:
    print("Adding anacrusis to downbeats")
    midi_score_dbs = [0.0] + midi_score_dbs
    if downbeats_score_map is not None:
        assert downbeats_score_map[0][0] != 0
        downbeats_score_map.insert(
            0, list(range(downbeats_score_map[0][0]))
        )  # Also add the info in the downbeats_score_map (from 0 to first measure referenced)

# Some debug prints
print(f"XML measures: {len(xml_score.measures)}")
print(f"DBs: {len(midi_score_dbs)}")
if downbeats_score_map is not None:
    print(f"SM:  {len(downbeats_score_map)}")
    print(f"USM: {sum(len(group) for group in downbeats_score_map)}")
else:
    print("SM:  None")

# Check coherence between number of downbeats, the score map and the measures
if downbeats_score_map is None:
    if len(midi_score_dbs) != len(xml_score.measures):
        raise ValueError(
            f"Number of downbeats ({len(midi_score_dbs)}) "
            f"does not match number of measures ({len(xml_score.measures)}) "
            f"and no score map is provided"
        )
    downbeats_score_map = [
        [i] for i in range(len(midi_score_dbs))
    ]  # Create a default score map (0 to len(midi_score_dbs))
else:
    if len(midi_score_dbs) != len(downbeats_score_map):
        raise ValueError(
            f"Number of downbeats ({len(midi_score_dbs)}) "
            f"does not match number of score map ({len(downbeats_score_map)}). "
            f"This should NEVER happen, please check your annotations."
        )
    num_measures_in_xml_score = len(xml_score.measures)
    num_measures_in_score_map = sum(len(group) for group in downbeats_score_map)
    if num_measures_in_xml_score != num_measures_in_score_map:
        difference = num_measures_in_xml_score - num_measures_in_score_map
        # If difference of 1 or 2, check if that's because last measure empty or all tied
        if difference == 1 or difference == 2:  # Max offset of 2 found in ASAP
            last_measure_notes: list[pt.score.Note] = list(
                xml_score.iter_all(
                    cls=pt.score.Note,
                    start=xml_score.measures[-difference].start,
                    include_subclasses=True,  # For grace notes
                )
            )
            if len(last_measure_notes) == 0 or all(
                note.tie_prev is not None for note in last_measure_notes
            ):
                if len(last_measure_notes) == 0:
                    print("Empty last measure, adding it to the score map")
                else:
                    print("Last measure all tied, adding it to the score map")
                if difference == 2:
                    downbeats_score_map[-1].append(num_measures_in_xml_score - 2)
                downbeats_score_map[-1].append(num_measures_in_xml_score - 1)
            else:
                raise ValueError(
                    f"Number of measures in XML score ({num_measures_in_xml_score}) "
                    f"does not match number of measures in score map ({num_measures_in_score_map})."
                )


# %% Read the MIDI score
midi_score = symusic.Score(midi_score_path, ttype="second")
if len(midi_score.tracks) > 1:
    for track in midi_score.tracks[1:]:
        midi_score.tracks[0].notes.extend(track.notes)
    midi_score.sort(inplace=True)


# Place the notes indices into measures
# def tick_to_second(
#     tick: int, tpq: int, tempos: symusic.core.TempoTickList, as_fraction: bool = False
# ) -> float | Fraction:
#     if not tempos:
#         raise ValueError("No tempo events provided")
#
#     ticks_times_mspq = Fraction(0) if as_fraction else 0.0
#     current_tempo = tempos[0]
#
#     # Process each tempo event segment
#     for next_tempo in tempos[1:]:
#         if tick < next_tempo.time:
#             delta = tick - current_tempo.time
#             ticks_times_mspq += delta * current_tempo.mspq
#             break
#         delta = next_tempo.time - current_tempo.time
#         ticks_times_mspq += delta * current_tempo.mspq
#         current_tempo = next_tempo
#     # Handle the last segment
#     else:
#         delta = tick - current_tempo.time
#         ticks_times_mspq += delta * current_tempo.mspq
#
#     return ticks_times_mspq / 1_000_000 / tpq


# assert len(xml_score._quarter_durations) == 1, "Only one division should be used"
# divisions = xml_score.quarter_duration_map(0)

THRESHOLD = 0.01  # seconds
indices_and_notes = enumerate(midi_score.tracks[0].notes)
i, note = next(indices_and_notes, (None, None))
measures = defaultdict(list)

# note_attribute = lambda n, i: n.pitch
# note_attribute = lambda n, i: (i, n.pitch)
note_attribute = lambda n, i: i
for next_db, xml_measures in zip(midi_score_dbs[1:], downbeats_score_map):
    current_measure = []
    while note is not None and note.start < next_db - THRESHOLD:
        print(f"{note}: {note.start}")
        # current_measure.append((i, note))
        # current_measure.append(i)
        current_measure.append(note_attribute(note, i))
        i, note = next(indices_and_notes, (None, None))
    print(f"DB: {next_db}, XML: {xml_measures}")
    print(f"  {current_measure}")
    measures["-".join(map(str, xml_measures))] = current_measure
# Add missing notes if needed
if note is not None:
    # last_measure = [(i, note)]
    # last_measure = [i]
    # print(f"{note}: {note.start}")
    last_measure = [note_attribute(note, i)]
    for i, note in indices_and_notes:
        # print(f"{note}: {note.start}")
        # last_measure.append((i, note))
        # last_measure.append(i)
        last_measure.append(note_attribute(note, i))
    measures["-".join(map(str, downbeats_score_map[-1]))] = last_measure

for measure, notes in measures.items():
    print(f"M{measure}\t{len(notes)} notes")


# Save the chunks
import json
import re


def json_indent_limit(json_str: str, indent: int, limit: int) -> str:
    indent = " " * indent
    pattern = re.compile(f'\n({indent}){{{limit}}}(({indent})+|(?=(}}|])))')
    return pattern.sub('', json_str)


with open(performance_path.with_name(f"midi_score_chunks.json"), "w") as f:
    indent = 2
    f.write(
        json_indent_limit(
            json.dumps(
                measures,
                indent=indent,
                sort_keys=False,
            ),
            indent=indent,
            limit=1,
        )
    )


# %%
# # Plot xml measure normalized times vs db normalized times
# import matplotlib.pyplot as plt
#
# # Plot xml
# plt.figure(figsize=(20, 1), dpi=200)
# xml_measure_times = [measure.start.t / xml_score.last_point.t for measure in xml_score.measures]
# # xml_measure_times = [measure.start.t for measure in xml_score.measures]
# plt.plot(xml_measure_times, [1] * len(xml_measure_times), "r+", label="XML measures")
# midi_score_db_times = [db / max(midi_score_dbs) for db in midi_score_dbs]
# # midi_score_db_times = [db for db in midi_score_dbs]
# plt.plot(midi_score_db_times, [0] * len(midi_score_db_times), "b+", label="MIDI DBs")
# plt.legend(loc="center left")
# plt.show()


# # %% Make sure that all quarter_durations are the same as ticks_per_quarter
# for row in metadata.iterrows():
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         try:
#             xml_score = pt.load_musicxml(
#                 asap_path / row[1]["xml_score"], force_note_ids=True
#             ).parts[0]
#         except Exception as e:
#             print(f"Error loading {row[1]['xml_score']}: {e}")
#         midi_score = symusic.Score(asap_path / row[1]["midi_score"])
#         assert xml_score._quarter_durations[0] == midi_score.ticks_per_quarter, (
#             f"{row[1]['xml_score']} and {row[1]['midi_score']} have different quarter_durations: "
#             f"{xml_score._quarter_durations[0]} and {midi_score.ticks_per_quarter}"
#         )
