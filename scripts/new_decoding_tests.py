import data.components.transforms as t
from typing import Any

import pretty_midi as pm

import heapq
from collections.abc import Callable
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pretty_midi
import torch
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
import torch.nn.functional as F
import partitura as pt

from data.components.score_performance_dataset import ScorePerformanceEntry, ScorePerformanceDataset
from data.pdmx_tokens_datamodule import PdmxTokensDataModule
from models.automata.io import token_sequence_to_partitura, token_sequence_to_partitura_permissive
from utils.generation import GenerationState
from models.tokens_transcriber_module import TokensTranscriberModule
from utils.debug import as_pycharm_link

# fmt: off
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-05-27_15-12-11/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-06-11_22-43-52/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-06-27_00-10-57/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-09-09_23-50-39/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/2025-09-19_14-15-04_cosmic-totem-262/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/2025-09-22_09-51-15_jumping-shape-264/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/2025-09-23_15-09-32_grateful-frog-267_vocal-morning-268/checkpoints/last_vocal-morning-268.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/dry-planet-265.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/last-v1.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/rosy-salad-273_partial.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-11-03_12-53-43/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-11-03_12-53-43/checkpoints/epoch_060.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/2025-11-06_18-05-07/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/2025-11-07_09-30-46/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-11-12_11-37-22_leafy-serinity-289/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/2025-11-14_10-23-42_kind-universe-291/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/2025-11-14_17-13-25_noble-forest-293/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-11-28_22-23-03_clean-salad-312/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-11-28_20-21-02_scarlet-pyramid-311/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-12-02_15-26-21_clean-salad-313/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-12-09_13-57-48_sage-serenity-318/checkpoints/last.ckpt").expanduser()
model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-01-15_18-44-16_glorious-wildflower-321/checkpoints/last.ckpt").expanduser()

# base_path = Path("~/Documents/datasets/asap-dataset").expanduser()
# performance_path = base_path / "Bach/Prelude/bwv_860/YoungS01M.mid"
# performance_path = base_path / "Bach/Prelude/bwv_860/midi_score.mid"
# performance_path = base_path / "Bach/Prelude/bwv_883/LuM01M.mid"
# performance_path = base_path / "Bach/Prelude/bwv_883/midi_score.mid"

base_path = Path("~/Documents/datasets/pdmx").expanduser()

performance_paths = [
    # base_path / "mid/0/0/Qma1a1L2c1vSEuGfdPVSEEWfuYAEoBmjpVESX1ok2imu4b.mid",
    # base_path / "mid/1/24/QmbHzrXuDGNG3HyzLpeBNeZ9wAj4bTDGWsZ5xz56abM8D8.mid",  # Test unfinished
    base_path / "mid/13/29/QmVKam2PcfGjsMF1QtwWEwSgG5bqFyiBRGiF5btFQX6vmV.mid",  # Test grace notes
    # base_path / "mid/0/0/Qma1a2YHEJ8j9gxLCxgALAfjYaiYJQogh1bgyFSwNSRPCs.mid",  # Polyphonic, long
    # base_path / "mid/1/11/Qmbbh58jD1ZLT1vevNsnyEZkexnx2USTFz6FSHX3KUiAzC.mid",  # Monophonic
    # base_path / "mid/15/12/QmXB8B4F9cbHUrnrP4mRGyL3vyhTkyDjuVEbM3VdFu2EY9.mid",  # Polyphonic, chords
    # base_path / "mid/1/32/QmbMQQPWGZpGiTV5gngSwmjTbqd7XosW8yyPwkSHnXbhY2.mid",  # Polyphonic, chords
]

reference_score_paths = [
    base_path / "mxl" / perf_path.relative_to(base_path / "mid").with_suffix(".mxl") for perf_path in performance_paths
]

print("Reference score paths:\n" + "\n  - ".join([p.as_posix() for p in reference_score_paths]))
print("Performance paths:\n" + "\n  - ".join([p.as_posix() for p in performance_paths]))
# fmt: on


MAX_PERFORMED_NOTES = 500
MAX_GENERATED_TOKENS = 1000


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
        t.ScoreReader(),
        t.VoiceRestFiller(),
        t.VoiceReorderer(),
        t.PerformanceReader(),
        t.QuantizedMusicChunker(num_performed_notes=MAX_PERFORMED_NOTES, mode=0),
        t.PerformanceVectorizerUnbucketized(),
        t.VectorizedPerformanceStreamsBucketizer(),
        t.ScoreVectorizer(),
    ],
)


# Transcribe performance
performances = PdmxTokensDataModule.collate_fn(list(iter(dataset)), keep_fields="all")

# # pt.save_musicxml(performances["score"][0], Path("/tmp/reference.musicxml"))
# reconstructed_score = token_sequence_to_partitura_permissive(performances["score_tokens"][0])
# pt.save_musicxml(reconstructed_score, Path("/tmp/reconstructed.musicxml"))
# exit(0)

use_cache = True
constrained = False
max_constraint_iterations = 10
transcribed_tokens, finished_flags = model.transcriber_model.generate(
    performance_streams=performances["performance_streams"],
    performance_onset_times=performances["performance_onset_times"],
    performance_attention_mask=~performances["performance_padding_mask"],
    prefix=None,
    constraint=constrained,
    max_constraint_iterations=max_constraint_iterations,
    max_generated_tokens=MAX_GENERATED_TOKENS,
    use_cache=use_cache,
    progress_bar=True,
    return_logits_and_mask=False,
)
for performance_tokens, flag, performance_path, score_path, reference_tokens in zip(
    transcribed_tokens.to_tokens(),
    finished_flags,
    performance_paths,
    reference_score_paths,
    performances["score_tokens"],
):
    print()
    print(f"      Performance MIDI: {as_pycharm_link(performance_path)}")
    print(f"       Reference score: {as_pycharm_link(score_path)}")
    output_file_name_parts = [
        score_path.stem,
        f"constrained={constrained}",
        f"cache={use_cache}",
    ]
    # Transcribe permissive
    transcribed_score = token_sequence_to_partitura_permissive(performance_tokens)
    output_path = Path("/tmp") / ("_".join(output_file_name_parts + ["permissive"]) + ".musicxml")
    pt.save_musicxml(transcribed_score, output_path)
    print(f" Transcription (perm.): {as_pycharm_link(output_path)}")
    # Transcribe strict
    try:
        transcribed_score = token_sequence_to_partitura(performance_tokens)
        output_path = Path("/tmp") / ("_".join(output_file_name_parts + ["strict"]) + ".musicxml")
        pt.save_musicxml(transcribed_score, output_path)
        print(f"Transcription (strict): {as_pycharm_link(output_path)}")
    except Exception as e:
        print(f"Transcription (strict): {e}")
    print(f"   Transcription state: {GenerationState(flag[0].item())}@{flag[1].item()}")
    # Save tokens
    transcribed_tokens_path = Path("/tmp") / ("_".join(output_file_name_parts + [".tokens"]))
    reference_tokens_path = Path("/tmp") / ("_".join(output_file_name_parts + ["ref.tokens"]))
    with open(transcribed_tokens_path, "w") as f:
        for token in performance_tokens:
            f.write(str(token) + "\n")
    with open(reference_tokens_path, "w") as f:
        for token in reference_tokens:
            f.write(str(token) + "\n")
    print(f"  Transcription tokens: {as_pycharm_link(transcribed_tokens_path)}")
    print(f"Reference transcription tokens: {as_pycharm_link(reference_tokens_path)}")
    for token in performance_tokens:
        print("  *", token)
