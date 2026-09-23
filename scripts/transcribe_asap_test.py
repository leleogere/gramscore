"""
Try to load and run inference on a model
"""

from pathlib import Path

import partitura as pt
from loguru import logger
from tqdm import tqdm

from data.asap_tokens_datamodule import AsapTokensDataModule
from models.automata.io import token_sequence_to_partitura_permissive, token_sequence_to_partitura
from utils.generation import GenerationState
from models.tokens_transcriber_module import TokensTranscriberModule

# fmt: off
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-10_16-03-09_comic-cherry-323/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-16_16-41-28_spring-sunset-330/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-16_16-41-28_spring-sunset-330/checkpoints/epoch_000.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-19_12-05-10_fallen-wind-334/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-19_12-05-10_fallen-wind-334/checkpoints/epoch_004.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-19_12-05-10_fallen-wind-334/checkpoints/step=8500.ckpt").expanduser()
model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-24_19-23-04/checkpoints/last-v2_cerulean-pine-341.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-24_19-23-04/checkpoints/last-v2_cerulean-pine-341.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-24_19-23-04/checkpoints/last-v1_worldly-hill-340.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-24_19-23-04/checkpoints/last_electric-fire-339.ckpt").expanduser()

# base_path = Path("~/Documents/datasets/asap-dataset").expanduser()
# performance_path = base_path / "Bach/Prelude/bwv_860/YoungS01M.mid"
# performance_path = base_path / "Bach/Prelude/bwv_860/midi_score.mid"
# performance_path = base_path / "Bach/Prelude/bwv_883/LuM01M.mid"
# performance_path = base_path / "Bach/Prelude/bwv_883/midi_score.mid"

base_path = Path("~/Documents/datasets/asap-dataset").expanduser()
# fmt: on
print(model_path.as_posix())

asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
transcription_directory = (
    asap_path.parent
    / "transcriptions"
    / "ASAP"
    / (model_path.parent.parent.name + "_" + model_path.stem)
)

MAX_PERFORMED_NOTES = 500
MAX_GENERATED_TOKENS = 1500


# Load the model
model: TokensTranscriberModule = TokensTranscriberModule.load_from_checkpoint(
    model_path,
    map_location="cpu",
    weights_only=False,
)
model.global_metrics.train(mode=False)
model.eval()

# Load the dataloader
datamodule = AsapTokensDataModule(
    data_dir=base_path,
    include_midi_scores=False,
    valid_tsv_file=base_path / "valid.tsv",
    artificial_dataset_repetitions=1,
    test_validation_mode="first_chunk",
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    prefetch_factor=2,
)
datamodule.prepare_data()
datamodule.setup(stage="test")
dataloader = datamodule.test_dataloader()


constrained = False
max_constraint_iterations = 10
use_cache = True
for i, batch in enumerate(tqdm(dataloader)):
    transcribed_tokens, finished_flags = model.transcriber_model.generate(
        performance_streams=batch["performance_streams"],
        performance_onset_times=batch["performance_onset_times"],
        performance_attention_mask=~batch["performance_padding_mask"],
        prefix=None,
        constraint=constrained,
        max_constraint_iterations=max_constraint_iterations,
        max_generated_length=MAX_GENERATED_TOKENS,
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
        output_dir = transcription_directory / entry.midi_performance.relative_to(
            asap_path
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
        try:
            transcribed_score = token_sequence_to_partitura_permissive(performance_tokens)
            pt.save_musicxml(transcribed_score, trans_permissive_path)
        except Exception as e:
            logger.error(f"Error with transcription (perm.) of {output_dir.as_posix()}: {e}")

        # Write strict transcription
        try:
            transcribed_score = token_sequence_to_partitura(performance_tokens)
            pt.save_musicxml(transcribed_score, trans_strict_path)
        except Exception as e:
            logger.error(f"Error with transcription (strict) of {output_dir.as_posix()}: {e}")
            # print(f"Transcription (strict): {e}")

        # Symlink reference files
        if not symlink_ref_score.exists():
            symlink_ref_score.symlink_to(entry.xml_score)
        if not symlink_ref_performance.exists():
            symlink_ref_performance.symlink_to(entry.midi_performance)
