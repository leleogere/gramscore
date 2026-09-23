import pretty_midi as pm
import data.components.transforms as t
from typing import Any
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
from data.components.transforms import PerformanceReader
from data.components.transforms import PerformanceVectorizer
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
model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-12-09_13-57-48_sage-serenity-318/checkpoints/last.ckpt").expanduser()
# fmt: on

asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
transcription_directory = (
    asap_path.parent / "transcriptions" / "ASAP" / model_path.parent.parent.name
)

MAX_PERFORMED_NOTES = 500
MAX_GENERATED_TOKENS = 1000


# Load the model
model: TokensTranscriberModule = TokensTranscriberModule.load_from_checkpoint(
    model_path,
    map_location="cpu",
    weights_only=False,
)
model.eval()


validation_dataset = ScorePerformanceDataset(
    entries=[
        ScorePerformanceEntry(
            xml_score=s_path,
            midi_performance=p_path,
            base_path=pdmx_path,
        )
        for s_path, p_path in zip(reference_paths, performance_paths)
    ],
    transforms=[
        t.ToDict(),
        t.ScoreReader(),
        t.VoiceRestFiller(),
        t.VoiceReorderer(),
        t.PerformanceReader(),
        t.QuantizedMusicChunker(num_performed_notes=500, mode=0),
        t.PerformanceVectorizer(),
    ],
)
batch_size = 16
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=batch_size,
    collate_fn=PdmxTokensDataModule.collate_fn,
    num_workers=2,
    persistent_workers=True,
)

# breakpoint()

# Transcribe performance
use_cache = True
constrained = False
max_constraint_iterations = 10
for i, batch in enumerate(tqdm(validation_dataloader)):
    transcribed_tokens, finished_flags = model.transcriber_model.generate(
        performance_streams=batch["performance_streams"],
        performance_onset_times=batch["performance_onset_times"],
        performance_attention_mask=~batch["performance_padding_mask"],
        prefix=None,
        constraint=constrained,
        max_constraint_iterations=max_constraint_iterations,
        max_generated_tokens=MAX_GENERATED_TOKENS,
        use_cache=use_cache,
        progress_bar=True,
        return_logits_and_mask=False,
    )
    for performance_tokens, flag, entry, clipped_perf, ref_clipped_score in zip(
        transcribed_tokens.to_tokens(),
        finished_flags,
        batch["entry"],
        batch["performance"],
        batch["score"],
        # range(i * batch_size, min((i + 1) * batch_size, len(reference_paths))),
    ):
        # Get variables
        # entry: ScorePerformanceEntry = batch["entry"]
        # clipped_perf: pm.PrettyMIDI = batch["performance"][idx]
        # ref_clipped_score: pt.score.Part = batch["score"][idx]

        # Prepare output directory
        output_dir = transcription_directory / entry.xml_score.relative_to(
            pdmx_path / "mxl"
        ).with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare output file name template
        file_name_parts = [
            entry.xml_score.stem,
            f"constrained={constrained}",
            f"cache={use_cache}",
            f"flag={GenerationState(flag[0].item()).name}@{flag[1].item()}",
        ]
        file_name_base = "_".join(file_name_parts)

        # Prepare output paths
        tokens_paths = (output_dir / file_name_base).with_suffix(".tokens")
        trans_permissive_path = (output_dir / (file_name_base + "_perm")).with_suffix(".musicxml")
        trans_strict_path = (output_dir / (file_name_base + "_strict")).with_suffix(".musicxml")
        symlink_ref_score = output_dir / entry.xml_score.name
        symlink_ref_performance = output_dir / entry.midi_performance.name
        ref_clipped_score_path = symlink_ref_score.with_name(
            f"{symlink_ref_score.stem}_clipped.musicxml"
        )
        ref_clipped_perf_path = symlink_ref_performance.with_name(
            f"{symlink_ref_performance.stem}_clipped.mid"
        )
        # print()
        # print(f"      Performance MIDI: {as_pycharm_link(performance_path)}")
        # print(f"       Reference score: {as_pycharm_link(reference_path)}")
        # print(f"   Transcription state: {GenerationState(flag.item())}")
        # print(f" Transcription (perm.): {as_pycharm_link(trans_permissive_path)}")
        # print(f"Transcription (strict): {as_pycharm_link(trans_strict_path)}")

        # Write reference score and performance
        pt.save_musicxml(ref_clipped_score, ref_clipped_score_path)
        clipped_perf.write(ref_clipped_perf_path)

        # Write tokens
        with tokens_paths.open("w") as f:
            f.write("\n".join(map(str, performance_tokens[: flag[1].item() + 1])))
        # for token in performance_tokens:
        #     print("  *", token)

        # Write permissive transcription
        transcribed_score = token_sequence_to_partitura_permissive(performance_tokens)
        pt.save_musicxml(transcribed_score, trans_permissive_path)

        # Write strict transcription
        try:
            transcribed_score = token_sequence_to_partitura(performance_tokens)
            pt.save_musicxml(transcribed_score, trans_strict_path)
        except Exception as e:
            pass
            # print(f"Transcription (strict): {e}")

        # Symlink reference files
        if not symlink_ref_score.exists():
            symlink_ref_score.symlink_to(entry.xml_score)
        if not symlink_ref_performance.exists():
            symlink_ref_performance.symlink_to(entry.midi_performance)
