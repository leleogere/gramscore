"""
Test full transcription pipeline with batch and chunk support
"""

import torch.nn.functional as F

import itertools

import pretty_midi as pm

import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import partitura as pt
import torch
from loguru import logger
from pretty_midi import note_number_to_name
from torch.nn.utils.rnn import pad_sequence

import data.components.transforms as t
import models.automata.tokens as tk
from data.components.score_performance_dataset import ScorePerformanceEntry, ScorePerformanceDataset
from data.pdmx_tokens_datamodule import PdmxTokensDataModule
from models.automata.io import token_sequence_to_partitura, token_sequence_to_partitura_permissive
from models.automata.vectorization_torch import TokenTensor
from utils.generation import GenerationState
from models.tokens_transcriber_module import TokensTranscriberModule
from utils.constants import PADDING_VALUE
from utils.debug import as_pycharm_link, performance_streams_to_midi

# fmt: off
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-12-02_15-26-21_clean-salad-313/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-12-09_13-57-48_sage-serenity-318/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-01-15_18-44-16_glorious-wildflower-321/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-10_16-03-09.old/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-10_16-03-09_comic-cherry-323/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-10_16-03-09_comic-cherry-323/checkpoints/epoch_088.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-16_16-41-28_spring-sunset-330/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-27_19-03-52.736565_youthful-wave-390/checkpoints/last.ckpt").expanduser()
model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-05-19_18-02-40.669697_legendary-totem-1401_2/checkpoints/last.ckpt").expanduser()

base_path = Path("~/Documents/datasets/asap-dataset").expanduser()
performance_paths = [
    base_path / "Bach/Fugue/bwv_846/midi_score.mid",  # TODO: FIX THIS SCORE
    # base_path / "Bach/Prelude/bwv_860/YoungS01M.mid",
    # base_path / "Bach/Prelude/bwv_860/midi_score.mid",
    # base_path / "Bach/Prelude/bwv_883/LuM01M.mid",
    # base_path / "Bach/Prelude/bwv_883/midi_score.mid",
]
reference_score_paths = [p.with_name("xml_score.musicxml") for p in performance_paths]

# base_path = Path("~/Documents/datasets/pdmx").expanduser()
# performance_paths = [
#     # base_path / "mid/0/0/Qma1a1L2c1vSEuGfdPVSEEWfuYAEoBmjpVESX1ok2imu4b.mid",
#     # base_path / "mid/1/24/QmbHzrXuDGNG3HyzLpeBNeZ9wAj4bTDGWsZ5xz56abM8D8.mid",  # Test unfinished
#     base_path / "mid/0/0/Qma1a2YHEJ8j9gxLCxgALAfjYaiYJQogh1bgyFSwNSRPCs.mid",  # Polyphonic, long
#     base_path / "mid/1/11/Qmbbh58jD1ZLT1vevNsnyEZkexnx2USTFz6FSHX3KUiAzC.mid",  # Monophonic
#     base_path / "mid/13/29/QmVKam2PcfGjsMF1QtwWEwSgG5bqFyiBRGiF5btFQX6vmV.mid",  # Test grace notes
#     # base_path / "mid/15/12/QmXB8B4F9cbHUrnrP4mRGyL3vyhTkyDjuVEbM3VdFu2EY9.mid",  # Polyphonic, chords
#     # base_path / "mid/1/32/QmbMQQPWGZpGiTV5gngSwmjTbqd7XosW8yyPwkSHnXbhY2.mid",  # Polyphonic, chords
#     # Path("/home/gerel/Documents/Perso/Compositions/La Quête d'Ewilan/La Quête d'Ewilan.mid"),
#     # Path("/home/gerel/Téléchargements/2026-02-25_21-01-14_729n_67s_STARRED.mid"),
# ]
# reference_score_paths = [
#     base_path / "mxl" / perf_path.relative_to(base_path / "mid").with_suffix(".mxl")
#     if "pdmx" in perf_path.as_posix()
#     else Path("")
#     for perf_path in performance_paths
# ]

print("Reference score paths:\n" + "\n  - ".join([p.as_posix() for p in reference_score_paths]))
print("Performance paths:\n" + "\n  - ".join([p.as_posix() for p in performance_paths]))
# fmt: on

NOTES_PER_CHUNK = 500
TOKENS_LIMIT_PER_CHUNK = 1500
MAX_CHUNKS = 100


# Load the model
model: TokensTranscriberModule = TokensTranscriberModule.load_from_checkpoint(
    model_path,
    map_location="cpu",
    weights_only=False,
)
model.eval()


dataset = ScorePerformanceDataset(
    entries=[
        ScorePerformanceEntry(
            xml_score=score,
            midi_performance=performance,
            base_path=base_path,
        )
        for score, performance in zip(reference_score_paths, performance_paths)
    ],
    transforms=[
        t.ToDict(),
        # Also read score info for debugging and comparing
        t.ScoreReader(),
        t.ScoreVectorizer(),
        t.PerformanceReader(),
        t.PerformanceToPerformedNoteList(),
        t.PerformedNoteListVectorizer(),
        # Here we simply load the performance and vectorize the note list, no bucketing yet as it's
        # more direct to deal with explicit pitches for the chunking/realigning rather then reinfer
        # them from the buckets. The bucketing will be done manually after each chunking.
    ],
)

# Transcribe performance
performances = PdmxTokensDataModule.collate_fn(
    list(iter(dataset)),
    keep_fields="all",
    skip_none=False,
    skip_error_key=False,
)
print(f"{performances.keys() = }")
print(f"{performances['performance_onset_times'].shape = }")

contraint = False
use_cache = True
measure_interchunk_overlap = 1
transcriptions, flags = model.transcriber_model.generate(
    performance_streams_raw=performances["performance_streams_raw"],
    performance_lengths=performances["performance_length"],
    measures_interchunk_overlap=measure_interchunk_overlap,
    chunk_size=NOTES_PER_CHUNK,
    max_chunks=MAX_CHUNKS,
    max_tokens_per_chunk=TOKENS_LIMIT_PER_CHUNK,
    constraint=contraint,
    max_constraint_iterations=None,
    use_cache=use_cache,
    progress_bar=True,
)

for tokens, (flag, chunk, pos), performance_path, score_path, reference_tokens in zip(
    transcriptions,
    flags,
    performance_paths,
    reference_score_paths,
    performances["score_tokens"],
):
    print()
    print(f"      Performance MIDI: {as_pycharm_link(performance_path)}")
    print(f"       Reference score: {as_pycharm_link(score_path)}")
    transcribed_score = token_sequence_to_partitura_permissive(tokens)
    output_path = (
        Path("/tmp/transcriptions") / f"transcribed"
        f"_{performance_path.stem}"
        f"_constrained={contraint}"
        f"_cache={use_cache}"
        f"_flag={GenerationState(flag.item()).name}"
        f"_chunk={chunk.item()}"
        f"_pos={pos.item()}"
        f".musicxml"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pt.save_musicxml(transcribed_score, output_path)
