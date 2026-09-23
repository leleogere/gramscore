import random
import sys

import partitura as pt
import partitura.score
import pretty_midi as pm
from pathlib import Path

from loguru import logger

from data.components.score_performance_dataset import PdmxPianoDataset
import data.components.transforms as t
from data.components.score_performance_dataset import ScorePerformanceEntry
from data.music_chunking import clip_quantized_midi
from data.music_chunking import clip_score
from utils.debug import open_in_midieditor
from utils.debug import open_in_musescore

# Log level
logger.remove()
logger.add(sys.stderr, level="DEBUG")

pdmx_path = Path("~/Documents/datasets/pdmx").expanduser()

dataset = PdmxPianoDataset(
    directory=pdmx_path,
    split="train",
    transforms=[
        t.ToDict(),
        t.ScoreReader(),
        t.VoiceRestFiller(),
        t.VoiceReorderer(),
        t.PreComputedAllowedTranspositionsReader(limit_semitones=4),
        t.PerformanceReader(),
        t.QuantizedMusicChunker(
            num_performed_notes=500,
            mode=0,
        ),
        t.PerformanceSpeedAugmenter(min_factor=0.8, max_factor=1.2),
        t.RandomTransposer(),
        t.ScoreVectorizer(),
    ],
)
# random.shuffle(dataset.entries)
dataset.entries = [
    e
    for e in dataset.entries
    if e.xml_score_relative.as_posix()
    # == "mxl/2/3/Qmc4PvPfVJdH6ZAmMHmBsm96soqi8u89FwL5jZrmvGr4mc.mxl"
    == "mxl/9/36/QmRpDUv71zXuX832ToUv4vFSk5kXPMuURqMWkD3HYBzgiJ.mxl"
] * 1

path_no_ext = Path("/tmp/cropped")
score_path = path_no_ext.with_suffix(".musicxml")
performance_path = path_no_ext.with_suffix(".mid")
start_measure = 0
while True:
    print(f"Starting measure {start_measure}")
    dataset.transforms[6].mode = start_measure
    print(next(iter(dataset)))
    start_measure += 1

    # pt.save_musicxml(score, score_path)
    # performance.write(performance_path)
    #
    # open_in_musescore(entry.xml_score)
    # open_in_musescore(score_path)
    # open_in_midieditor(performance_path)
    #
    # if input("Continue? [Y/n]").lower() == "n":
    #     break
    #
    # score_path.unlink()
    # performance_path.unlink()

# start = 3
#
# cropped_midi, (s, e) = clip_quantized_midi(
#     midi,
#     start_measure=3,
#     num_notes=30,
#     inplace=True,
#     check_num_measures=len(score.measures),
# )
# cropped_score = clip_score(score, start_measure=s, end_measure=e)
#
# cropped_midi.write("/tmp/cropped_midi.mid")
# pt.save_musicxml(cropped_score, "/tmp/cropped_score.musicxml")
