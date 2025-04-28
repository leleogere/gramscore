import copy
from pathlib import Path

import music21 as m21
import partitura as pt
import partitura.score
import partitura.utils.music as music


path = Path("~/Documents/datasets/asap-dataset/Bach/Fugue/bwv_846/xml_score.musicxml").expanduser()

# Transpose using music21
score_m21 = m21.converter.parse(str(path))
transposed_score_m21 = score_m21.transpose(m21.interval.DiatonicInterval("major", 3))
# Save
output_path = path.with_name("bwv_846_transposed_m21.musicxml")
transposed_score_m21.write("musicxml", fp=str(output_path))


# Transpose using partitura
score_pt = pt.load_musicxml(path)
assert len(score_pt.parts) == 1
part = score_pt.parts[0]
# transposed_score_pt = music.transpose()
# def transpose_note_inplace(note: pt.score.Note, interval: pt.score.Interval, key_signature, update_ties: bool = True):
#
#     return note
#
# def transpose_partitura(part: pt.score.Part, interval: pt.score.Interval, inplace: bool = False):
#     if not inplace:
#         part = copy.deepcopy(part)
#
#     for note in part.notes_tied:
