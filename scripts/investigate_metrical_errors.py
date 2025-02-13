from collections import defaultdict
from pathlib import Path

import partitura as pt
import partitura.score

# %% Parameters
PATH = Path("/home/gerel/Documents/datasets/nasap-dataset")
assert PATH.exists(), f"{PATH} does not exist."

# %% Search for all scores
score_paths = list(PATH.rglob("xml_score.musicxml"))
print(f"Found {len(score_paths)} scores.")

# # %% Loop over them, stopping when finding a score with more/less beats than expected
# for i, score_path in enumerate(score_paths):
#     score = pt.load_musicxml(score_path)
#
#     if i % 100 == 0:
#         print(f"Processed {i} scores.")


# %% Investigate single score
score_path = PATH / "Beethoven/Piano_Sonatas/1-1/xml_score.musicxml"

score = pt.load_musicxml(score_path)
assert len(score.parts) == 1, f"Expected 1 part, found {len(score.parts)}."

part = score.parts[0]

# Iterate over measures
for measure in part.iter_all(cls=pt.score.Measure):
    measure: pt.score.Measure
    ts: pt.score.TimeSignature = next(measure.start.iter_prev(cls=pt.score.TimeSignature, eq=True))
    print(f"Measure {measure.number}: {ts.beats}/{ts.beat_type} (D={measure.duration}).")

    # Iterate over notes
    voices = defaultdict(list)
    for note in part.iter_all(
        cls=pt.score.GenericNote,
        start=measure.start,
        end=measure.end,
        include_subclasses=True,
    ):
        note: pt.score.GenericNote
        voices[note.voice].append(note)

    for voice, notes in voices.items():
        print(f"  V{voice}: {len(notes)} notes over {sum([n.duration for n in notes])}.")
