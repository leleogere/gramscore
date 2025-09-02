import json
import warnings
from collections import defaultdict
from fractions import Fraction

import joblib
import tabulate
from lxml import etree
from pathlib import Path
import subprocess

from tqdm import tqdm

MUSESCORE_PATH = Path(
    "~/Applications/MuseScore-Studio-4.5.2.251141401-x86_64.AppImage"
).expanduser()


def check_path(path: Path) -> bool:
    process = subprocess.run(
        [
            MUSESCORE_PATH,
            path.as_posix(),
            "-o",
            f"/tmp/{path.relative_to(asap_path).as_posix().replace('/', '_')}",
        ]
    )
    return process.returncode == 0


def check_paths(asap_path: Path) -> list[Path]:
    score_paths = sorted(asap_path.rglob("xml_score.musicxml"))
    print(f"Scores to check: {len(score_paths)}")
    results = joblib.Parallel(n_jobs=-1, return_as="generator")(
        joblib.delayed(check_path)(path) for path in score_paths
    )
    return [
        path for path, valid in tqdm(zip(score_paths, results), total=len(score_paths)) if not valid
    ]


if __name__ == '__main__':
    # fmt: off
    # asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
    asap_path = Path("~/Documents/datasets/asap-dataset-original").expanduser()
    # fmt: on

    errors = check_paths(asap_path)

    print(f"Number of errors: {len(errors)}")
    for path in errors:
        print(f"  - {path.relative_to(asap_path).as_posix()}")
    breakpoint()
    #
    # # Compute some statistics about the results
    #
    # # Number of paths with errors
    # n_score_errors = len(all_exceptions)
    # print(f"Number of scores with errors: {n_score_errors}")
    #
    # # Total number of measures with errors
    # n_measure_errors = sum(len(measures) for measures in all_exceptions.values())
    # print(f"Number of measures with errors: {n_measure_errors}")
    #
    # # Average number of erroneous measure per score
    # avg_measure_error_per_score = n_measure_errors / n_score_errors
    # print(f"Average measure errors per score: {avg_measure_error_per_score}")
    #
    # # Median number of erroneous measure per score
    # median_measure_error_per_score = sorted(len(measures) for measures in all_exceptions.values())[
    #     n_score_errors // 2
    # ]
    # print(f"Median measure errors per score: {median_measure_error_per_score}")
    #
    # # For each path, proportion of measures affected
    # proportions = []
    # for path, path_exceptions in all_exceptions.items():
    #     with open(asap_path / path, "r") as f:
    #         num_m = f.read().count("<measure")
    #     proportions.append((path, len(path_exceptions), num_m, len(path_exceptions) / num_m))
    # proportions = sorted(proportions, key=lambda p: p[-1], reverse=True)
    # print(tabulate.tabulate(proportions))


"""
Number of errors: 40
  - Balakirev/Islamey/xml_score.musicxml
  - Beethoven/Piano_Sonatas/13-4/xml_score.musicxml
  - Beethoven/Piano_Sonatas/16-1/xml_score.musicxml
  - Beethoven/Piano_Sonatas/16-2/xml_score.musicxml
  - Beethoven/Piano_Sonatas/17-2/xml_score.musicxml
  - Beethoven/Piano_Sonatas/17-3/xml_score.musicxml
  - Beethoven/Piano_Sonatas/22-1/xml_score.musicxml
  - Beethoven/Piano_Sonatas/23-1/xml_score.musicxml
  - Beethoven/Piano_Sonatas/31-1/xml_score.musicxml
  - Beethoven/Piano_Sonatas/31-2/xml_score.musicxml
  - Beethoven/Piano_Sonatas/7-4/xml_score.musicxml
  - Beethoven/Piano_Sonatas/8-2/xml_score.musicxml
  - Chopin/Ballades/1/xml_score.musicxml
  - Chopin/Barcarolle/xml_score.musicxml
  - Chopin/Etudes_op_25/1/xml_score.musicxml
  - Chopin/Etudes_op_25/11/xml_score.musicxml
  - Chopin/Sonata_2/2nd/xml_score.musicxml
  - Chopin/Sonata_2/2nd_no_repeat/xml_score.musicxml
  - Chopin/Sonata_3/3rd/xml_score.musicxml
  - Debussy/Pour_le_Piano/1/xml_score.musicxml
  - Haydn/Keyboard_Sonatas/6-1/xml_score.musicxml
  - Liszt/Gran_Etudes_de_Paganini/2_La_campanella/xml_score.musicxml
  - Liszt/Gran_Etudes_de_Paganini/6_Theme_and_Variations_repeat/xml_score.musicxml
  - Liszt/Hungarian_Rhapsodies/6/xml_score.musicxml
  - Liszt/Mephisto_Waltz/xml_score.musicxml
  - Liszt/Sonata/xml_score.musicxml
  - Liszt/Transcendental_Etudes/1/xml_score.musicxml
  - Liszt/Transcendental_Etudes/4/xml_score.musicxml
  - Mozart/Piano_Sonatas/12-2/xml_score.musicxml
  - Rachmaninoff/Preludes_op_32/10/xml_score.musicxml
  - Ravel/Gaspard_de_la_Nuit/1_Ondine/xml_score.musicxml
  - Ravel/Miroirs/3_Une_Barque/xml_score.musicxml
  - Ravel/Miroirs/4_Alborada_del_gracioso/xml_score.musicxml
  - Ravel/Pavane/xml_score.musicxml
  - Schubert/Impromptu_op142/3/xml_score.musicxml
  - Schubert/Piano_Sonatas/664-1/xml_score.musicxml
  - Schumann/Kreisleriana/1/xml_score.musicxml
  - Schumann/Kreisleriana/1_no_first_repeat/xml_score.musicxml
  - Schumann/Kreisleriana/3/xml_score.musicxml
  - Scriabin/Sonatas/5/xml_score.musicxml
"""
