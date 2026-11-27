"""
Try to load and run inference on a model
"""

from functools import partial

import data.components.transforms as t

import csv

import pandas as pd

import warnings
from collections import defaultdict

import joblib
import torch

from pathlib import Path

import partitura as pt
from loguru import logger

from data.components.score_performance_dataset import ScorePerformanceEntry, ScorePerformanceDataset
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
    chunk_size: int = 500,
    max_chunks: int = 100,
    max_tokens_per_chunk: int = 1000,
    constrained: bool = False,
    max_constraint_iterations: int = 10,
    use_cache: bool = True,
):
    finished_flags_counter = defaultdict(int)
    for i, batch in enumerate(tqdm(dataloader)):
        transcribed_tokens, finished_flags = model.transcriber_model.generate(
            performance_streams_raw=batch["performance_streams_raw"],
            performance_lengths=batch["performance_length"],
            chunk_size=chunk_size,
            max_chunks=max_chunks,
            max_tokens_per_chunk=max_tokens_per_chunk,
            constraint=constrained,
            max_constraint_iterations=max_constraint_iterations,
            use_cache=use_cache,
            progress_bar=True,
        )
        for (
            performance_tokens,
            (flag, flag_chunk, flag_pos),
            entry,
            perf,
            # ref_score,
        ) in zip(
            transcribed_tokens,
            finished_flags,
            batch["entry"],
            batch["performance"],
            # batch["score"],
        ):
            # Get variables
            # entry: ScorePerformanceEntry = batch["entry"]
            # clipped_perf: pm.PrettyMIDI = batch["performance"][idx]
            # ref_clipped_score: pt.score.Part = batch["score"][idx]

            # Prepare output directory
            output_dir = transcription_directory / entry.midi_performance_relative.with_suffix("")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare output file name template
            generation_state = GenerationState(flag.item())
            file_name_parts = [
                entry.xml_score.stem,
                f"constrained={constrained}",
                f"cache={use_cache}",
                f"flag={generation_state.name}@{flag_chunk.item()}-{flag_pos.item()}",
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
            # pt.save_musicxml(ref_score, ref_clipped_score_path)
            # perf.write(ref_clipped_perf_path)

            # Write tokens
            with tokens_paths.open("w") as f:
                f.write("\n".join(map(str, performance_tokens)))
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
        t_score.with_name(f"xml_score.musicxml") for t_score in transcribed_scores
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
    model_path = Path("/home/leo/Documents/Thèse/gramscore/weights/2026-05-27_09-17-04.791128_ethereal-plasma-1405/checkpoints/last.ckpt").expanduser()
    # fmt: on

    CHUNK_SIZE = 500
    MAX_TOKENS_PER_CHUNK = 3000
    MAX_CHUNKS = 100
    CONSTRAINT = False
    CONSTRAINT_ITERATIONS = 10000
    EVALUATE=False

    asap_path = Path("~/Documents/Thèse/datasets/asap-dataset").expanduser()
    transcription_directory = (
        asap_path.parent
        / "transcriptions"
        / "ASAP_full"
        / f"{model_path.parent.parent.name}_{model_path.stem}_constraint={CONSTRAINT}"
    )
    print(
        f"Transcribing ASAP using {model_path.as_posix()} into {transcription_directory.as_posix()}..."
    )

    # Load the model
    model: TokensTranscriberModule = TokensTranscriberModule.load_from_checkpoint(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    model.global_metrics.train(mode=False)
    model.eval()

    # Load the dataloader
    with (asap_path / "test_pieces.txt").open("r") as f:
        test_pieces = [tuple(l.strip("\n").split("|")) for l in f.readlines()]
    with open(asap_path / "metadata.csv", "r") as f:
        reader = list(csv.DictReader(f))
    entries = [
        ScorePerformanceEntry(
            composer=entry["composer"],
            title=entry["title"],
            xml_score=asap_path / entry["xml_score"],
            midi_score=None,
            midi_performance=asap_path / entry["midi_performance"],
            midi_score_annotations=None,
            folder=asap_path / entry["folder"],
            base_path=asap_path,
        )
        for entry in reader
        if (entry["composer"], entry["title"]) in test_pieces
    ]
    dataset = ScorePerformanceDataset(
        entries=entries,
        transforms=[
            t.ToDict(),
            t.PerformanceReader(),
            t.PerformanceToPerformedNoteList(),
            t.PerformedNoteListVectorizer(),
            t.VectorizedPerformanceStreamsBucketizer(),
        ],
    )
    dataloader = torch.utils.data.dataloader.DataLoader(
        dataset=dataset,
        collate_fn=partial(AsapTokensDataModule.collate_fn, keep_fields="all"),
        batch_size=16,
        shuffle=None,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
    )
    # datamodule = AsapTokensDataModule(
    #     data_dir=asap_path,
    #     include_midi_scores=False,
    #     valid_tsv_file=None,  # asap_path / "valid.tsv",
    #     artificial_dataset_repetitions=1,
    #     test_validation_mode="full_piece",
    #     batch_size=16,
    #     num_workers=2,
    #     persistent_workers=True,
    #     prefetch_factor=2,
    # )
    # datamodule.prepare_data()
    # datamodule.setup(stage="test")
    # dataloader = datamodule.test_dataloader()
    print(f"Number of performances: {len(dataset.entries)}")
    print(f"Number of scores: {len(set(e.xml_score for e in dataset.entries))}")

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
            chunk_size=CHUNK_SIZE,
            max_tokens_per_chunk=MAX_TOKENS_PER_CHUNK,
            max_chunks=MAX_CHUNKS,
            constrained=CONSTRAINT,
            max_constraint_iterations=CONSTRAINT_ITERATIONS,
            use_cache=True,
        )
        print("Transcription finished!")
    if EVALUATE:
        print("Starting evaluation...")
        evaluate(transcription_directory)
        print("Evaluation finished!")
