"""
Try to load and run inference on a model
"""

import warnings
from collections import defaultdict

import joblib
import torch

from pathlib import Path

import partitura as pt
from loguru import logger
from muster import muster
from tabulate import tabulate
from tqdm import tqdm

from data.asap_tokens_datamodule import AsapTokensDataModule
from metrics.score_similarity.score_similarity import score_similarity_normalized
from models.automata.io import token_sequence_to_partitura_permissive, token_sequence_to_partitura
from utils.generation import GenerationState
from models.tokens_transcriber_module import TokensTranscriberModule
from utils.debug import as_pycharm_link


def transcribe(
    dataloader: torch.utils.data.DataLoader,
    model: TokensTranscriberModule,
    transcription_directory: Path,
    max_generated_length: int = 1000,
    constrained: bool = False,
    max_constraint_iterations: int = 10,
    use_cache: bool = True,
    first_chunk_only: bool = True,
):
    if not first_chunk_only:
        raise NotImplementedError("Multi-chunk transcription is not implemented yet.")

    finished_flags_counter = defaultdict(int)
    for i, batch in enumerate(tqdm(dataloader)):
        transcribed_tokens, finished_flags = model.transcriber_model.generate_single_chunk(
            performance_streams=batch["performance_streams"],
            performance_onset_times=batch["performance_onset_times"],
            performance_attention_mask=~batch["performance_padding_mask"],
            prefix=None,
            constraint=constrained,
            max_constraint_iterations=max_constraint_iterations,
            max_generated_length=max_generated_length,
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
        ):
            # Get variables
            # entry: ScorePerformanceEntry = batch["entry"]
            # clipped_perf: pm.PrettyMIDI = batch["performance"][idx]
            # ref_clipped_score: pt.score.Part = batch["score"][idx]

            # Prepare output directory
            output_dir = transcription_directory / entry.midi_performance_relative.with_suffix("")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare output file name template
            generation_state = GenerationState(flag[0].item())
            file_name_parts = [
                entry.xml_score.stem,
                f"constrained={constrained}",
                f"cache={use_cache}",
                f"flag={generation_state.name}@{flag[1].item()}",
            ]
            file_name_base = "_".join(file_name_parts)

            # Prepare output paths
            tokens_paths = (output_dir / file_name_base).with_suffix(".tokens")
            trans_permissive_path = (output_dir / (file_name_base + "_perm")).with_suffix(
                ".musicxml"
            )
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

            # Count the flags
            finished_flags_counter[generation_state] += 1
    print("Generation states:")
    for state, count in finished_flags_counter.items():
        print(f"  - {state.name}: {count}")


def evaluate(transcription_directory: Path):
    transcribed_scores = [
        score
        for score in transcription_directory.rglob("*_perm.musicxml")
        if "flag=FINISHED" in score.stem
    ]
    print(transcription_directory.as_posix())
    print(f"Number of fully transcribed scores: {len(transcribed_scores)}")
    ground_truth_scores = [
        t_score.with_name(f"xml_score_clipped.musicxml") for t_score in transcribed_scores
    ]

    NUM_PRED = None
    if NUM_PRED:
        transcribed_scores = transcribed_scores[:NUM_PRED]
        ground_truth_scores = ground_truth_scores[:NUM_PRED]

    # for t_score, gt_score in zip(transcribed_scores, ground_truth_scores):
    #     print(as_pycharm_link(t_score))
    #     print(as_pycharm_link(gt_score))
    #     print()
    # exit(0)

    def eval(prediction: Path, ground_truth: Path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                score_sim = score_similarity_normalized(
                    est=prediction.as_posix(), gt=ground_truth.as_posix(), full=False
                )
            except Exception as e:
                print(
                    f"Error on score similarity on {prediction.relative_to(transcription_directory)}: {e}"
                )
                score_sim = None
            try:
                muster_sim = muster(score_est=prediction, score_gt=ground_truth, clean_scores=False)
                if all(v is None for v in muster_sim.values()):
                    raise Exception("All Muster metrics are None")
            except Exception as e:
                print(f"Error on MUSTER on {prediction.relative_to(transcription_directory)}: {e}")
                muster_sim = None
            return {"score_similarity": score_sim, "muster": muster_sim}

    sims_list = [
        s
        for s in tqdm(
            joblib.Parallel(n_jobs=12, return_as="generator")(
                joblib.delayed(eval)(est_score, gt_score)
                for est_score, gt_score in zip(transcribed_scores, ground_truth_scores)
            ),
            total=len(transcribed_scores),
        )
    ]

    print("-----------------")
    print("Number of errors:")
    for metric, num_err in {k: sum(d[k] is None for d in sims_list) for k in sims_list[0]}.items():
        print(f"  - {metric}: {num_err}")
    sims = {k: [d[k] for d in sims_list if d[k]] for k in sims_list[0]}

    # print("-----------------")
    # print(
    #     "Aggregate:",
    #     {k: len([s for s in sims[k] if s is not None]) for k in sims},
    #     len(sims['score_similarity']),
    # )
    sims_aggregate = {}
    for k, v in sims.items():
        # v is list of dicts
        aggregate = {k_: [d[k_] for d in v if d[k_] is not None] for k_ in v[0]}
        if any(key in k for key in ["TP", "FP", "FN", "TN"]):
            aggregate = {k_: sum(v_) for k_, v_ in aggregate.items()}
        else:
            aggregate = {k_: sum(v_) / (len(v_) + 1e-9) for k_, v_ in aggregate.items()}
        sims_aggregate[k] = aggregate
        # print(k, aggregate)
    sims["aggregate"] = sims_aggregate

    # print(f"Ours", end=" ")
    # for k in ["PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER", "MeanER"]:
    #     print(f"{round(sims_aggregate['muster'][k], 2):5.2f}", end=" ")
    # for k in [
    #     "NoteDeletion",
    #     "NoteInsertion",
    #     "NoteDuration",
    #     "StaffAssignment",
    #     "StemDirection",
    #     "NoteSpelling",
    # ]:
    #     print(f"{round(100*sims_aggregate['score_similarity'][k], 2):5.2f}", end=" ")

    # print("\nTable 4:")
    # print("SOTA  6.86 26.74  9.69   -     -     -")
    # print(f"Ours", end=" ")
    # for k in [
    #     "StaffAssignment",
    #     "StemDirection",
    #     "NoteSpelling",
    #     "GraceF1",
    #     "StaccatoF1",
    #     "TrillF1",
    # ]:
    #     print(f"{round(100*sims_aggregate['score_similarity'][k], 2):5.2f}", end=" ")
    # print()

    metric_names = {
        # "muster": ["PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER", "MeanER"],
        "muster": [
            "PitchER",
            "MissRate",
            "ExtraRate",
            "OnsetER",
            "OffsetER",
            "MeanER",
            "VoiceER",
            "NewMeanER",
        ],
        "score_similarity": [
            "NoteDeletion",
            "NoteInsertion",
            "NoteDuration",
            "StaffAssignment",
            "StemDirection",
            "NoteSpelling",
            "Voice",
        ],
    }
    # Keep only sub-metrics
    headers = ["MU" + k for k in metric_names["muster"]] + [
        "SS" + k for k in metric_names["score_similarity"]
    ]

    metric_round = lambda x: round(x, 2) if x is not None else None
    # t3_row_m2st = ["M2ST (paper)", 3.11, 7.56, 6.44, 15.55, 23.84, 11.30, 12.69, 9.06, 51.86, 6.62, 25.03, 8.69]
    # t3_row_m2st2 = ["M2ST (perso)", 2.26, 25.00, 36.04, 29.12, 13.08, 21.10, 28.33, 66.90, 41.66, 12.23, 24.76, 0.15]
    rows = [
        {
            "name": "M2ST (paper)",
            "muster": {
                "PitchER": 3.11,
                "MissRate": 7.56,
                "ExtraRate": 6.44,
                "OnsetER": 15.55,
                "OffsetER": 23.84,
                "MeanER": 11.30,
            },
            "score_similarity": {
                "NoteDeletion": 12.69,
                "NoteInsertion": 9.06,
                "NoteDuration": 51.86,
                "StaffAssignment": 6.62,
                "StemDirection": 25.03,
                "NoteSpelling": 8.69,
            },
        },
        {
            "name": "M2ST (perso)",
            "muster": {
                "PitchER": 2.26,
                "MissRate": 25.00,
                "ExtraRate": 36.04,
                "OnsetER": 29.12,
                "OffsetER": 13.08,
                "MeanER": 21.10,
            },
            "score_similarity": {
                "NoteDeletion": 28.33,
                "NoteInsertion": 66.90,
                "NoteDuration": 41.66,
                "StaffAssignment": 12.23,
                "StemDirection": 24.76,
                "NoteSpelling": 0.15,
            },
        },
        {
            "name": "Ours",
            "muster": sims_aggregate["muster"],
            "score_similarity": sims_aggregate["score_similarity"],
        },
    ]
    rows = [
        [row["name"]]
        + [
            round(row["muster"][k], 2) if row["muster"].get(k, None) is not None else None
            for k in metric_names["muster"]
        ]
        + [
            (
                round((100 if i >= 2 else 1) * row["score_similarity"][k], 2)
                if row["score_similarity"].get(k, None) is not None
                else None
            )
            for k in metric_names["score_similarity"]
        ]
        for i, row in enumerate(rows)
    ]
    # rows = (
    #     ["Ours"]
    #     + [
    #         round(sims_aggregate["muster"][k], 2)
    #         for k in ["PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER", "MeanER"]
    #     ]
    #     + [
    #         round(100 * sims_aggregate["score_similarity"][k], 2)
    #         for k in [
    #             "NoteDeletion",
    #             "NoteInsertion",
    #             "NoteDuration",
    #             "StaffAssignment",
    #             "StemDirection",
    #             "NoteSpelling",
    #         ]
    #     ]
    # )
    # print("Table 3")
    print("-----------------")
    print(tabulate(rows, headers=headers, floatfmt=".2f"))
    # Also print to CSV
    print("-----------------")
    print("TSV:")
    for l in rows:
        print("\t".join([str(x).replace(".", ",") for x in l[1:]]))

    # # ----- Table 4 -----
    # t4_headers = [
    #     "Model",
    #     "StaffAssignment",
    #     "StemDirection",
    #     "NoteSpelling",
    #     "GraceF1",
    #     "StaccatoF1",
    #     "TrillF1",
    # ]
    # t4_row_sota = ["SOTA", 6.86, 26.74, 9.69, "-", "-", "-"]
    # t4_row_m2st = ["M2ST (paper)", 6.62, 25.03, 8.69, 27.80, 18.19, 54.64]
    # t4_row_m2st2 = ["M2ST (perso)", 12.23, 24.76, 0.15, 34.67, 0.0, 100.0]
    # t4_row_ours = ["Ours"] + [
    #     round(100 * sims_aggregate["score_similarity"][k], 2)
    #     for k in [
    #         "StaffAssignment",
    #         "StemDirection",
    #         "NoteSpelling",
    #         "GraceF1",
    #         "StaccatoF1",
    #         "TrillF1",
    #     ]
    # ]
    #
    # print("\nTable 4:")
    # print(
    #     tabulate(
    #         [t4_row_sota, t4_row_m2st, t4_row_m2st2, t4_row_ours],
    #         headers=t4_headers,
    #         floatfmt=".2f",
    #     )
    # )

    top = 5
    print("\n" + "=" * 50)
    print(f"Top {top} Highest Scores per Metric (Table 3)")
    print("=" * 50)
    for category, metrics in metric_names.items():
        for metric in metrics:
            ranking = []
            for i, res in enumerate(sims_list):
                if res is not None and res.get(category) and metric in res[category]:
                    val = res[category][metric]
                    ranking.append((val, i))

            # Sort descending (highest numerical value)
            ranking.sort(key=lambda x: x[0] if x[0] is not None else -float('inf'), reverse=True)

            print(f"\nMetric: {metric}")
            for val, idx in ranking[:top]:
                score_path = transcribed_scores[idx]
                # Argument matches "parent transcription folder"
                # score_path is .../something_perm.musicxml, usage logic implies parent folder is the unit
                link = as_pycharm_link(score_path.parent)
                print(f"  {val:.4f}: {link}")


if __name__ == '__main__':
    FORCE_TRANSCRIPTION = False

    # fmt: off
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-10_16-03-09_comic-cherry-323/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-16_16-41-28_spring-sunset-330/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-16_16-41-28_spring-sunset-330/checkpoints/epoch_000.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-19_12-05-10_fallen-wind-334/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-19_12-05-10_fallen-wind-334/checkpoints/epoch_004.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-19_12-05-10_fallen-wind-334/checkpoints/step=8500.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-24_19-23-04/checkpoints/last-v2_cerulean-pine-341.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-24_19-23-04/checkpoints/last-v2_cerulean-pine-341.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-24_19-23-04/checkpoints/last-v1_worldly-hill-340.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-24_19-23-04/checkpoints/last_electric-fire-339.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-10_17-00-21.057562_raccoon-bowser-368/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-02-24_19-23-04/checkpoints/last-v2_cerulean-pine-341.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-10_17-00-21.057562_raccoon-bowser-368/checkpoints/e=001_s=001500.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-16_12-20-19.551783_zesty-resonance-371/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-16_12-21-05.750663_light-feather-372/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-16_15-49-56.653967_dulcet-dew-374/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-16_15-49-56.666726_helpful-wood-375/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-16_20-02-39.939096_sage-water-376/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-16_20-15-36.479036_cool-flower-377/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-18_11-31-57.255427_glad-pond-378/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-19_15-59-45.400818_lilac-eon-380/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-23_12-45-56.444839_fragrant-hill-383/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-23_12-45-57.852815_stoic-lion-384/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-24_10-28-18.964079_helpful-durian-386/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-24_10-28-18.963913_charmed-wave-388/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-24_10-28-18.964317_legendary-paper-387/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-24_10-29-18.822090_still-snowflake-389/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-25_16-08-01.644560_rural-mountain-389/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-04-01_00-32-12.791876_whole-oath-391/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-04-08_09-09-02.955002_solar-sound-393/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-04-14_15-25-48.752849_smart-puddle-403/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-04-14_15-25-48.746806_wandering-smoke-404/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-05-11_09-27-21.456443_morning-snowball-1170/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-05-11_09-31-53.414935_serene-jazz-1171/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-05-12_11-46-36.002881_morning-snowball-1170_2/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-05-13_12-18-46.946766_morning-snowball-1170_3/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-05-14_12-04-36.203562_genial-bush-1269/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-05-16_19-10-35.601209_legendary-totem-1401/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-05-19_18-02-40.669697_legendary-totem-1401_2/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-05-21_08-52-49.465600_legendary-totem-1401_3/checkpoints/last.ckpt").expanduser()
    # model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-05-27_09-17-04.791128_ethereal-plasma-1405/checkpoints/last.ckpt").expanduser()
    model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-05-27_09-17-04.791128_ethereal-plasma-1405_modified/checkpoints/last_fixed.ckpt").expanduser()

    # base_path = Path("~/Documents/datasets/asap-dataset").expanduser()
    # performance_path = base_path / "Bach/Prelude/bwv_860/YoungS01M.mid"
    # performance_path = base_path / "Bach/Prelude/bwv_860/midi_score.mid"
    # performance_path = base_path / "Bach/Prelude/bwv_883/LuM01M.mid"
    # performance_path = base_path / "Bach/Prelude/bwv_883/midi_score.mid"
    # fmt: on
    asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
    transcription_directory = (
        asap_path.parent
        / "transcriptions"
        / "ASAP"
        / (model_path.parent.parent.name + "_" + model_path.stem)
    )
    print(
        f"Transcribing ASAP using {model_path.as_posix()} into {transcription_directory.as_posix()}..."
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
        data_dir=asap_path,
        include_midi_scores=False,
        valid_tsv_file=asap_path / "valid.tsv",
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

    if transcription_directory.exists() and not FORCE_TRANSCRIPTION:
        print(
            f"Transcription directory {transcription_directory.as_posix()} already exists. Skipping transcription."
        )
    else:
        print("Starting transcription...")
        transcribe(
            dataloader=dataloader,
            model=model,
            transcription_directory=transcription_directory,
            max_generated_length=MAX_GENERATED_TOKENS,
            constrained=True,
            max_constraint_iterations=20,
            use_cache=True,
            first_chunk_only=True,
        )
        print("Transcription finished!")
    print("Starting evaluation...")
    evaluate(transcription_directory)
    print("Evaluation finished!")
