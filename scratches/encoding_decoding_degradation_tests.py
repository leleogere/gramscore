"""
Test how MUSTER scores degrade when tokenizing/detokenizing scores
"""

import csv
import shutil

import joblib
from muster import muster
from tabulate import tabulate

import data.components.transforms as t
import partitura as pt
import warnings
from pathlib import Path

from tqdm import tqdm

from metrics.score_similarity.score_similarity import score_similarity_normalized
from models.automata.io import partitura_to_token_sequence, token_sequence_to_partitura_permissive

asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
score_paths = list(asap_path.glob("**/xml_score.musicxml"))

output_path = Path("~/Documents/datasets/detokenization_tests2").expanduser()
success_file = output_path / "SUCCESS"

score_reader = t.ScoreReader()
rest_filler = t.VoiceRestFiller(tolerant=True)

if not success_file.exists():
    for score_path in tqdm(score_paths):
        detokenized_score_folder = output_path / score_path.relative_to(asap_path).parent
        symlink = detokenized_score_folder / "original.musicxml"
        if symlink.exists():
            continue
        detokenized_score_folder.mkdir(parents=True, exist_ok=True)

        # Read the score with Partitura, and save result
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                score = score_reader.process(score_path)
                pt.save_musicxml(score, detokenized_score_folder / "partitura.musicxml")
        except Exception as e:
            print(f"Score reading error on {score_path}: {e}")
            continue

        # Fill rests
        try:
            score = rest_filler.process(score)
            pt.save_musicxml(score, detokenized_score_folder / "partitura_filled.musicxml")
        except Exception as e:
            print(f"Rest filling error on {score_path}: {e}")
            continue

        # Tokenize the score
        try:
            tokens = partitura_to_token_sequence(score)
            with open(detokenized_score_folder / "tokens.tokens", "w") as f:
                f.write("\n".join(map(str, tokens)))
        except Exception as e:
            print(f"Tokenization error on {score_path}: {e}")
            continue

        # Detokenize the score
        try:
            detokenized_score = token_sequence_to_partitura_permissive(tokens)
            pt.save_musicxml(detokenized_score, detokenized_score_folder / "detokenized.musicxml")
        except Exception as e:
            print(f"Detokenization error on {score_path}: {e}")
            continue

        # Link the original score
        symlink.symlink_to(score_path)

    success_file.touch()
else:
    print("Skipping detokenization part, already done")

# Get all correctly encoded/decoded scores
detokenized_folders = [p.parent for p in output_path.glob("**/detokenized.musicxml")]
print(f"Number of correctly decoded scores: {len(detokenized_folders)}/{len(score_paths)}")


def eval(prediction: Path, ground_truth: Path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            score_sim = score_similarity_normalized(
                est=prediction.as_posix(), gt=ground_truth.as_posix(), full=False
            )
        except Exception as e:
            print(f"Error on score similarity on {prediction}: {e}")
            score_sim = None
        try:
            muster_sim = muster(score_est=prediction, score_gt=ground_truth, clean_scores=False)
            if all(v is None for v in muster_sim.values()):
                raise Exception("All Muster metrics are None")
        except Exception as e:
            print(f"Error on MUSTER on {prediction}: {e}")
            muster_sim = None
        return {"score_similarity": score_sim, "muster": muster_sim}


# Prepare the results table
results = {}
# Set up all test pairs
pairs = [
    ("original", "original"),  # no degradation
    ("original", "partitura"),  # partitura degradation
    ("partitura", "partitura_filled"),  # rest filling degradation
    ("original", "partitura_filled"),  # partitura + rest filling degradation
    ("partitura_filled", "detokenized"),  # enc/dec degradation
    ("partitura", "detokenized"),  # enc/dec + rest filling degradation
    ("original", "detokenized"),  # enc/dec + rest filling + partitura degradation
]
for i, (gt_filename, est_filename) in enumerate(pairs, start=1):
    print(f"{i}/{len(pairs)} Comparing {gt_filename} and {est_filename}")

    gt_files = [f / f"{gt_filename}.musicxml" for f in detokenized_folders]
    est_files = [f / f"{est_filename}.musicxml" for f in detokenized_folders]

    sims_list = [
        s
        for s in tqdm(
            joblib.Parallel(n_jobs=8, return_as="generator")(
                joblib.delayed(eval)(est_score, gt_score)
                for est_score, gt_score in zip(est_files, gt_files)
            ),
            total=len(detokenized_folders),
        )
    ]
    num_errors = {k: sum(d[k] is None for d in sims_list) for k in sims_list[0]}
    for metric, num_err in num_errors.items():
        print(f"  - {metric}: {num_err}")
    print("-----------------")
    print("Number of errors:")

    metric_keys = {
        metric: next(s[metric] for s in sims_list if s[metric] is not None).keys()
        for metric in ["score_similarity", "muster"]
    }

    # Save sims_list
    sims_list_to_save = [
        {
            "piece": folder.relative_to(output_path).as_posix(),
            **{
                f"SS_{k}": (
                    None
                    if s["score_similarity"] is None or s["score_similarity"][k] is None
                    else 100 * s["score_similarity"][k]
                )
                for k in metric_keys["score_similarity"]
            },
            **{
                f"MU_{k}": None if s["muster"] is None or s["muster"][k] is None else s["muster"][k]
                for k in metric_keys["muster"]
            },
        }
        for s, folder in zip(sims_list, detokenized_folders)
    ]
    with open(output_path / f"score_per_piece_gt={gt_filename}_est={est_filename}.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=sims_list_to_save[0].keys())
        writer.writeheader()
        writer.writerows(sims_list_to_save)

    sims = {k: [d[k] for d in sims_list if d[k]] for k in sims_list[0]}
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
    sims_aggregate["num_errors"] = num_errors
    results[f"gt={gt_filename}_est={est_filename}"] = sims_aggregate


headers = (
    [""]
    + ["MU_PitchER", "MU_MissRate", "MU_ExtraRate", "MU_OnsetER", "MU_OffsetER", "MU_MeanER"]
    + [
        "SS_NoteDeletion",
        "SS_NoteInsertion",
        "SS_NoteDuration",
        "SS_StaffAssignment",
        "SS_StemDirection",
        "SS_NoteSpelling",
    ]
    + ["MU_NumErrors", "SS_NumErrors"]
)
rows = [
    ["M2ST (paper)", 3.11, 7.56, 6.44, 15.55, 23.84, 11.30, 12.69, 9.06, 51.86, 6.62, 25.03, 8.69, "/", "/"]  # fmt: skip
] + [
    [name]
    + [
        round(agg["muster"][k], 2)
        for k in ["PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER", "MeanER"]
    ]
    + [
        round(100 * agg["score_similarity"][k], 2)
        for k in [
            "NoteDeletion",
            "NoteInsertion",
            "NoteDuration",
            "StaffAssignment",
            "StemDirection",
            "NoteSpelling",
        ]
    ]
    + [agg["num_errors"][k] for k in ["muster", "score_similarity"]]
    for name, agg in results.items()
]
print(rows)
# print("Table 3")
print("-----------------")
print(tabulate(rows, headers=headers, floatfmt=".2f"))
# Also print to CSV
print("-----------------")
print("TSV:")
for l in rows:
    print("\t".join([str(x).replace(".", ",") for x in l[1:]]))

# Save file
with (output_path / "final_results.csv").open("w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(headers)
    writer.writerows(rows)


# top = 5
# print("\n" + "=" * 50)
# print(f"Top {top} Highest Scores per Metric (Table 3)")
# print("=" * 50)
# metrics_source = {
#     "muster": ["PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER", "MeanER"],
#     "score_similarity": [
#         "NoteDeletion",
#         "NoteInsertion",
#         "NoteDuration",
#         "StaffAssignment",
#         "StemDirection",
#         "NoteSpelling",
#     ],
# }
# for category, metrics in metrics_source.items():
#     for metric in metrics:
#         ranking = []
#         for i, res in enumerate(sims_list):
#             if res is not None and res.get(category) and metric in res[category]:
#                 val = res[category][metric]
#                 ranking.append((val, i))
#
#         # Sort descending (highest numerical value)
#         ranking.sort(key=lambda x: x[0] if x[0] is not None else -float('inf'), reverse=True)
#
#         print(f"\nMetric: {metric}")
#         for val, idx in ranking[:top]:
#             score_path = detokenized_paths[idx]
#             # Argument matches "parent transcription folder"
#             # score_path is .../something_perm.musicxml, usage logic implies parent folder is the unit
#             link = as_pycharm_link(score_path.parent)
#             print(f"  {val:.4f}: {link}")
