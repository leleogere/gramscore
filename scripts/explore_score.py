"""
Script to explore a score
"""

from pathlib import Path

import partitura as pt
import partitura.score


# fmt: off
# path = Path("~/Documents/MuseScore4/Scores/grace_notes.musicxml").expanduser()
path = Path("~/Documents/datasets/asap-dataset/Liszt/Concert_Etude_S145/2/xml_score.musicxml").expanduser()
# fmt: on

score = pt.load_musicxml(path)
part = score.parts[0]

measure = part.measures[8]
for element in part.iter_all(
    cls=pt.score.Note,
    include_subclasses=True,
    start=measure.start.t,
    end=measure.end.t,
):
    if isinstance(element, pt.score.GraceNote):
        print("   ", element)
    else:
        print(">>>", element)


# for point in part._points:
#     point: pt.score.TimePoint
#     print(point.t)
#     for cls in [pt.score.GraceNote, pt.score.Note]:
#         for element in point.starting_objects[cls]:
#             print(f"  * {element}")
