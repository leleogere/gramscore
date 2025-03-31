from pathlib import Path

import partitura as pt
import partitura.score

from models.automata.tokens import fill_rests
from src.models.automata.tokens import Bar
from src.models.automata.automata import ConstrainedAutomata
from src.models.automata.io import token_sequence_to_partitura, partitura_to_token_sequence

# fmt: off
# score_path = Path("~/Documents/datasets/asap-dataset/Chopin/Sonata_2/1st_no_repeat/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Debussy/Images_Book_1/1_Reflets_dans_lEau/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Beethoven/Piano_Sonatas/7-1/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Schubert/Impromptu_op142/3/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Beethoven/Piano_Sonatas/31-2/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Haydn/Keyboard_Sonatas/46-1/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Haydn/Keyboard_Sonatas/49-1/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Schubert/Piano_Sonatas/894-2/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Bach/Fugue/bwv_846/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Chopin/Ballades/2/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Haydn/Keyboard_Sonatas/39-1/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Beethoven/Piano_Sonatas/27-1/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Schumann/Kreisleriana/2/xml_score.musicxml").expanduser()
# RESTS
# score_path = Path("~/Documents/datasets/asap-dataset/Beethoven/Piano_Sonatas/18-1/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Chopin/Ballades/1/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Beethoven/Piano_Sonatas/17-2/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Beethoven/Piano_Sonatas/31-2/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Schubert/Piano_Sonatas/664-3/xml_score.musicxml").expanduser()
# TUPLET_END
# score_path = Path("~/Documents/datasets/asap-dataset/Beethoven/Piano_Sonatas/29-2/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Liszt/Transcendental_Etudes/10/xml_score.musicxml").expanduser()
# TUPLET_START
# score_path = Path("~/Documents/datasets/asap-dataset/Glinka/The_Lark/xml_score.musicxml").expanduser()
# BACH
# score_path = Path("~/Documents/datasets/asap-dataset/Bach/Fugue/bwv_846/xml_score.musicxml").expanduser()  # 2ae3fe3f85 - Merge branch 'fix_bach_fugue_bwv_846' into develop
# score_path = Path("~/Documents/datasets/asap-dataset/Bach/Fugue/bwv_892/xml_score.musicxml").expanduser()  # b74c0b3fc1 - Merge branch 'fix_bach_fugue_bwv_892' into develop
# score_path = Path("~/Documents/datasets/asap-dataset/Bach/Prelude/bwv_870/xml_score.musicxml").expanduser()  # 39795be6db - Merge branch 'fix_bach_fugue_bwv_870' into develop
# score_path = Path("~/Documents/datasets/asap-dataset/Bach/Prelude/bwv_876/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Bach/Prelude/bwv_874/xml_score.musicxml").expanduser()


# score_path = Path("~/Documents/datasets/asap-dataset/Chopin/Ballades/1/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Chopin/Etudes_op_25/5/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/bach-dataset/Bach/Fugue/bwv_868/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Beethoven/Piano_Sonatas/3-1/xml_score.musicxml").expanduser()
# score_path = Path("~/Documents/datasets/asap-dataset/Bach/Fugue/bwv_889/xml_score.musicxml").expanduser()
score_path = Path("~/Documents/datasets/asap-dataset/Liszt/Concert_Etude_S145/2/xml_score.musicxml").expanduser()

# score_path = Path("~/Documents/MuseScore4/Scores/test_remplissage_silences.musicxml").expanduser()
# fmt: on


score = pt.load_musicxml(score_path)
assert len(score.parts) == 1
score = score.parts[0]
print(score.pretty())
# pt.score.fill_rests(score)
fill_rests(score)

pt.save_musicxml(score, "/tmp/score_filled.musicxml")
# exit(0)

# measure81 = score.measures[80]
# time_point = None
# for element in score.iter_all(start=measure81.start.t, end=measure81.end.t):
#     if element.start.t != time_point:
#         time_point = element.start.t
#         print(f"{time_point = }")
#     print(f"  * {element}")

tokens = partitura_to_token_sequence(score)

automata = ConstrainedAutomata(8)
for token in tokens:
    # print(" > ", automata.state)
    print(" > ", token)
    if isinstance(token, Bar):
        print(f"Measure {token._partitura_elements[0].name}")
    # try:
    automata.transition(token)
    # except ValueError as e:
    # print(f"Error: {e}")
    # print(automata.state)
    # break

# Write tokens to disk
tokens_as_strings = [f"{token}\n" for token in tokens]
# # Insert a line break before each Bar token after the first one
with open("/tmp/tokens.tokens", "w") as f:
    f.writelines(tokens_as_strings)

# Reencode the score and save to disk
reencoded_score = token_sequence_to_partitura(tokens)
pt.save_musicxml(reencoded_score, "/tmp/reencoded_score.musicxml")
