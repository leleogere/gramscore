"""
This script is used to numbers for the results table.
"""

import json
import warnings
from functools import partial

from lxml import etree
from pathlib import Path

from tqdm import tqdm

import partitura as pt
import partitura.score

import joblib


def get_all_scores(path: Path) -> list[Path]:
    return sorted(path.rglob("xml_score.musicxml"))


def count_measures(path: Path, unfold: bool = False) -> int:
    """
    Count the number of measures in a score.
    """
    if unfold:
        try:
            score = read_and_unfold_score(path)
        except Exception as e:
            print(f"{path.as_posix()}: {e}")
            return 0
        return count_measures_partitura(score)
    else:
        return count_measures_xml(path)


def count_measures_xml(path: Path) -> int:
    # Read XML and count <measure> tags
    tree = etree.parse(path)
    root = tree.getroot()
    measures = root.findall(".//measure")
    return len(measures)


def read_and_unfold_score(path: Path) -> pt.score.Part:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pt.load_musicxml(path)
        if len(score.parts) > 1:
            raise ValueError(f"Too much parts: {len(score.parts)}")
        score = score.parts[0]
        score = pt.score.unfold_part_maximal(score, ignore_leaps=False)
        return score


def count_measures_partitura(score: pt.score.Part) -> int:
    return len(score.measures)


if __name__ == '__main__':
    asap_path = Path("~/Documents/datasets/asap-dataset-original").expanduser()
    exclusion_file = asap_path / "asap-dataset-original_exceptions.json"
    with exclusion_file.open("r") as f:
        excluded = json.load(f)
        excluded = sorted([asap_path / p for p in excluded.keys()])

    for unfold in [True, False]:
        scores = get_all_scores(asap_path)
        if unfold:
            scores = [s for s in scores if s not in excluded]
        print(f"Counting measures for {unfold=}, with {len(scores)} scores:")
        print(
            sum(
                tqdm(
                    joblib.Parallel(njobs=-1, return_as="generator_unordered")(
                        joblib.delayed(partial(count_measures, unfold=unfold))(path=path)
                        for path in scores
                    ),
                    total=len(scores),
                    desc="Counting measures",
                ),
            )
        )
