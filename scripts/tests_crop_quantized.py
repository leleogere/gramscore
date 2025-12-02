import random

import partitura as pt
import partitura.score
import pretty_midi as pm
from pathlib import Path

from data.components.score_performance_dataset import PdmxPianoDataset
import data.components.transforms as t
from data.components.score_performance_dataset import ScorePerformanceEntry
from data.music_chunking import clip_quantized_midi
from data.music_chunking import clip_score
from utils.debug import open_in_midieditor
from utils.debug import open_in_musescore

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
            mode="random",
        ),
        t.PerformanceSpeedAugmenter(min_factor=0.8, max_factor=1.2),
        t.RandomTransposer(),
    ],
)
random.shuffle(dataset.entries)

path_no_ext = Path("/tmp/cropped")
score_path = path_no_ext.with_suffix(".musicxml")
performance_path = path_no_ext.with_suffix(".mid")
for sample in dataset:
    entry: ScorePerformanceEntry = sample["entry"]
    score: pt.score.Part = sample["score"]
    performance: pm.PrettyMIDI = sample["performance"]

    print(entry.xml_score_relative.as_posix())

    pt.save_musicxml(score, score_path)
    performance.write(performance_path)

    open_in_musescore(entry.xml_score)
    open_in_musescore(score_path)
    open_in_midieditor(performance_path)

    if input("Continue? [Y/n]").lower() == "n":
        break

    score_path.unlink()
    performance_path.unlink()

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
