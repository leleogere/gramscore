import warnings

import joblib
from tabulate import tabulate

import data.components.transforms as t
import json
from collections import defaultdict
from pathlib import Path
from pprint import pprint

from tqdm import tqdm

from metrics.score_similarity import score_similarity_normalized
from muster.muster import muster

from utils.debug import as_pycharm_link

base_path = Path("~/Documents/datasets/pdmx").expanduser()
transcriptions_path = Path(
    "~/Documents/datasets/transcriptions/2025-12-09_13-57-48_sage-serenity-318"
).expanduser()

transcribed_scores = [
    score for score in transcriptions_path.rglob("*_perm.musicxml") if "flag=FINISHED" in score.stem
]
print(f"Number of fully transcribed scores: {len(transcribed_scores)}")
ground_truth_scores = [
    t_score.with_name(f"{t_score.parent.stem}_clipped.musicxml") for t_score in transcribed_scores
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
                f"Error on score similarity on {prediction.relative_to(transcriptions_path)}: {e}"
            )
            score_sim = None
        try:
            muster_sim = muster(score_est=prediction, score_gt=ground_truth, clean_scores=False)
        except Exception as e:
            print(f"Error on M2ST on {prediction.relative_to(transcriptions_path)}: {e}")
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

# sims = defaultdict(lambda: defaultdict(list))
# for sim in tqdm(sims_generator, total=len(transcribed_scores)):
#     for metric, submetrics in sim.items():
#         for submetric, value in submetrics.items():
#             sims[metric][submetric].append(value)
sims = {k: [d[k] for d in sims_list if d[k]] for k in sims_list[0]}

print("-----------------")
print(
    "Aggregate:",
    {k: len([s for s in sims[k] if s is not None]) for k in sims},
    len(sims['score_similarity']),
)
sims_aggregate = {}
for k, v in sims.items():
    # v is list of dicts
    aggregate = {k_: [d[k_] for d in v if d[k_] is not None] for k_ in v[0]}
    if any(key in k for key in ["TP", "FP", "FN", "TN"]):
        aggregate = {k_: sum(v_) for k_, v_ in aggregate.items()}
    else:
        aggregate = {k_: sum(v_) / (len(v_) + 1e-9) for k_, v_ in aggregate.items()}
    sims_aggregate[k] = aggregate
    print(k, aggregate)
sims["aggregate"] = sims_aggregate

print(f"Ours", end=" ")
for k in ["PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER", "MeanER"]:
    print(f"{round(sims_aggregate['muster'][k], 2):5.2f}", end=" ")
for k in [
    "NoteDeletion",
    "NoteInsertion",
    "NoteDuration",
    "StaffAssignment",
    "StemDirection",
    "NoteSpelling",
]:
    print(f"{round(100*sims_aggregate['score_similarity'][k], 2):5.2f}", end=" ")

print("\nTable 4:")
print("SOTA  6.86 26.74  9.69   -     -     -")
print(f"Ours", end=" ")
for k in ["StaffAssignment", "StemDirection", "NoteSpelling", "GraceF1", "StaccatoF1", "TrillF1"]:
    print(f"{round(100*sims_aggregate['score_similarity'][k], 2):5.2f}", end=" ")
print()


t3_headers = (
    ["Model"]
    + ["PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER", "MeanER"]
    + [
        "NoteDeletion",
        "NoteInsertion",
        "NoteDuration",
        "StaffAssignment",
        "StemDirection",
        "NoteSpelling",
    ]
)

t3_row_m2st = ["M2ST", 3.11, 7.56, 6.44, 15.55, 23.84, 11.30, 12.69, 9.06, 51.86, 6.62, 25.03, 8.69]
t3_row_ours = (
    ["Ours"]
    + [
        round(sims_aggregate["muster"][k], 2)
        for k in ["PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER", "MeanER"]
    ]
    + [
        round(100 * sims_aggregate["score_similarity"][k], 2)
        for k in [
            "NoteDeletion",
            "NoteInsertion",
            "NoteDuration",
            "StaffAssignment",
            "StemDirection",
            "NoteSpelling",
        ]
    ]
)
print("Table 3")
print(tabulate([t3_row_m2st, t3_row_ours], headers=t3_headers, floatfmt=".2f"))

# ----- Table 4 -----
t4_headers = [
    "Model",
    "StaffAssignment",
    "StemDirection",
    "NoteSpelling",
    "GraceF1",
    "StaccatoF1",
    "TrillF1",
]

t4_row_sota = ["SOTA", 6.86, 26.74, 9.69, "-", "-", "-"]
t4_row_m2st = ["M2ST", 6.62, 25.03, 8.69, 27.80, 18.19, 54.64]
t4_row_ours = ["Ours"] + [
    round(100 * sims_aggregate["score_similarity"][k], 2)
    for k in [
        "StaffAssignment",
        "StemDirection",
        "NoteSpelling",
        "GraceF1",
        "StaccatoF1",
        "TrillF1",
    ]
]

print("\nTable 4:")
print(tabulate([t4_row_sota, t4_row_ours], headers=t4_headers, floatfmt=".2f"))

#     json_output_path = transcribed_score.with_suffix(".json")
#     sim = {
#         "score_similarity": score_similarity_normalized(
#             est=transcribed_score.as_posix(),
#             gt=ground_truth_path.as_posix(),
#             full=True,
#         ),
#         "muster": muster(
#             score_est=transcribed_score,
#             score_gt=ground_truth_path,
#             clean_scores=False,
#         ),
#     }
#     # with json_output_path.open("w") as f:
#     #     json.dump(sim, f)
#     for metric, submetrics in sim.items():
#         for submetric, value in submetrics.items():
#             scores[metric][submetric].append(value)


# sim = {
#     "mxl <-> gt_mxl": score_similarity_normalized(
#         est=transcribed_score_path.as_posix(),
#         gt=reference_score_path.as_posix(),
#         full=True,
#     ),
#     "muster": muster(
#         score_est=transcribed_score_path, score_gt=reference_score_path, clean_scores=False
#     ),
# }
# pprint(sim)

top = 5
print("\n" + "=" * 50)
print(f"Top {top} Highest Scores per Metric (Table 3)")
print("=" * 50)

metrics_source = {
    "muster": ["PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER", "MeanER"],
    "score_similarity": [
        "NoteDeletion",
        "NoteInsertion",
        "NoteDuration",
        "StaffAssignment",
        "StemDirection",
        "NoteSpelling",
    ],
}

for category, metrics in metrics_source.items():
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
