"""
Test the new tokenization with the TransitionCondition logic
"""

import warnings
from pathlib import Path
import sys

import partitura as pt
import partitura.score

from scripts.score_encoding_verification.simpler_tokens import Bar
from scripts.score_encoding_verification.simpler_tokens import ConstrainedAutomata
from scripts.score_encoding_verification.simpler_tokens import TransitionError
from scripts.score_encoding_verification.simpler_tokens import partitura_to_token_sequence
from scripts.score_encoding_verification.simpler_tokens import fill_rests

sys.setrecursionlimit(2**31 - 1)


asap_path = Path("~/Documents/datasets/asap-dataset-original").expanduser()
# score_path = asap_path / "Beethoven/Piano_Sonatas/17-2" / "xml_score.musicxml"
# score_path = asap_path / "Schubert/Wanderer_fantasie/xml_score.musicxml"
# score_path = asap_path / "Schumann/Kreisleriana/7/xml_score.musicxml"
score_path = asap_path / "Chopin/Sonata_2/1st_no_repeat/xml_score.musicxml"


# Load score
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    score = pt.load_musicxml(score_path, force_note_ids=True)
    assert len(score.parts) == 1
    score = score.parts[0]
    score = pt.score.unfold_part_maximal(score, ignore_leaps=False)
    fill_rests(score)

print(score.pretty())

# Tokenize score
tokens = partitura_to_token_sequence(score)

# Setup automata
automata = ConstrainedAutomata(num_voices=8, allow_duplicate_notes_in_chord=True)

# Check the sequence
skip_until_next_bar = False
for token in tokens:
    print(token)
    if isinstance(token, Bar):
        skip_until_next_bar = False
    if skip_until_next_bar:
        continue
    try:
        automata.transition(token)
    except TransitionError as e:
        # print(e)
        continue
