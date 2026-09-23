"""
Count number of staves in each MusicXML file in PDMX
"""

import ast
from collections import defaultdict
from typing import Any, Callable

import joblib
import pandas as pd
from pathlib import Path

from lxml import etree
from tqdm import tqdm

from models.automata.vectorization_torch import TokenTensor
from utils.musicxml import read_musicxml

# Set pandas display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_colwidth", 200)
# Left-align columns
pd.set_option("display.colheader_justify", "left")


memory = joblib.Memory(location=Path("~/Documents/datasets/joblib_cache").expanduser())


def count_parts_measures_and_staves(path: Path) -> tuple[int, tuple[int], int]:
    xml = read_musicxml(path).encode("utf-8")
    root = etree.fromstring(xml)
    parts = root.findall(".//part")
    n_parts = len(parts)
    n_staves = []
    for part in parts:
        n_staves.append(len(set(int(s.text) for s in part.findall(".//staff"))) or 1)
    n_measures = len(parts[0].findall(".//measure"))  # parts[0] to avoid duplicate counting
    return n_parts, tuple(n_staves), n_measures


def error_handling_wrapper(fn: Callable[[Any], Any]) -> Any | None:
    def wrapped(*args: Any, **kwargs: Any) -> Any | None:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"Error reading {args[0]}: {e}")
            return None

    return wrapped


def load_pdmx_paths(dataset_path: Path) -> list[Path]:
    metadata = pd.read_csv(dataset_path / "PDMX_extended_metadata.csv")
    # metadata = metadata[metadata["usable"]]
    return [dataset_path / path for path in metadata["mxl"] if isinstance(path, str)]


def load_asap_paths(dataset_path: Path) -> list[Path]:
    metadata = pd.read_csv(dataset_path / "metadata.csv")
    return [dataset_path / path / "xml_score.musicxml" for path in metadata["folder"].unique()]


def create_dataframe(paths: list[Path]) -> pd.DataFrame:
    results = {
        p: v
        for p, v in tqdm(
            joblib.Parallel(n_jobs=12, return_as="generator_unordered")(
                joblib.delayed(
                    lambda p: (p, error_handling_wrapper(count_parts_measures_and_staves)(p))
                )(path)
                for path in paths
            ),
            total=len(paths),
        )
        if v is not None
    }

    df = pd.DataFrame.from_dict(
        results, orient="index", columns=["n_parts", "n_staves", "n_measures"]
    )
    return df


@memory.cache()
def create_dataframe_from_path(dataset_path: Path) -> pd.DataFrame:
    if dataset_path.name == "pdmx":
        paths = load_pdmx_paths(dataset_path)
    elif dataset_path.name == "asap-dataset":
        paths = load_asap_paths(dataset_path)
    else:
        raise ValueError(f"Unknown dataset path: {dataset_path}")
    print(f"Loaded {len(paths)} paths from {dataset_path}")
    return create_dataframe(paths)


if __name__ == '__main__':
    datasets_path = Path("~/Documents/datasets").expanduser()

    # dataset_path = datasets_path / "asap-dataset"
    # filter = False
    dataset_path = datasets_path / "pdmx"
    filter = True

    df = create_dataframe_from_path(dataset_path)
    df["paths"] = [p.relative_to(dataset_path) for p in df.index]

    if filter:
        if dataset_path.name == "pdmx":
            metadata = pd.read_csv(dataset_path / "PDMX_extended_metadata.csv")
            metadata = metadata[
                (metadata["instruments"] == "['Piano']")
                & (~metadata["similar_to_asap"])
                & (metadata["n_notes_midi"] >= 1)
                & (metadata["n_measures_score"] == metadata["n_measures_midi"])
                & (metadata["n_measures_score"] != -1)
                & (
                    metadata["time_signatures"]
                    .apply(
                        lambda set_str: (
                            ast.literal_eval(set_str)
                            if isinstance(set_str, str) and set_str != "{}"
                            else set()
                        )
                    )
                    .map(lambda tss: all(ts in TokenTensor.TS_INDEXER for ts in tss))
                )
            ]
            metadata["mxl"] = metadata["mxl"].apply(lambda x: dataset_path / x)
            metadata.set_index("mxl", inplace=True)
            df = df.join(metadata, how="inner")
        elif dataset_path.name == "asap-dataset":
            raise NotImplementedError("Filter not implemented for ASAP dataset")
        else:
            raise ValueError(f"Unknown dataset path: {dataset_path}")

    # convert n_staves from tuple to list
    df["n_staves"] = df["n_staves"].apply(lambda x: "|".join(map(str, x)))

    # Group by n_parts and n_staves, and display the dataframes agg with count
    print("Part statistics")
    total_paths = len(df)
    print(f"Total number of paths: {total_paths}")
    total_measures = df["n_measures"].sum()
    print(f"Total number of measures: {total_measures}")
    aggregated_df = (
        df.groupby(["n_parts", "n_staves"])
        .agg(
            count=("paths", "count"),
            prop=("paths", lambda x: x.size / total_paths),
            measures=("n_measures", "sum"),
            prop_measures=("n_measures", lambda x: x.sum() / total_measures),
            paths=("paths", list),
        )
        .sort_values(["n_parts", "n_staves"], ascending=True)
        .query("n_parts <= 4")
    )
    print(aggregated_df)
    # Simply print to see the amount of undisplayed pieces due to the <=4 parts query
    print(f"Sum prop: {aggregated_df['prop'].sum():.4%}")
    print(f"Sum prop measures: {aggregated_df['prop_measures'].sum():.4%}")

    # Print the sum of the rows with n_parts == 1 and n_staves <= 2
    filtered_df = aggregated_df[
        (aggregated_df.index.get_level_values('n_parts') == 1)
        & (
            aggregated_df.index.get_level_values('n_staves').map(
                lambda x: all(int(s) <= 2 for s in x.split('|'))
            )
        )
    ]
    print(f"\nFiltered (n_parts == 1 and all staves <= 2):")
    print(
        f"Pieces: {filtered_df['count'].sum()} / {total_paths} "
        f"({filtered_df['count'].sum() / total_paths:.2%})"
    )
    print(
        f"Measures: {filtered_df['measures'].sum()} / {total_measures} "
        f"({filtered_df['measures'].sum() / total_measures:.2%})"
    )
