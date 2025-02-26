import csv
import itertools
import json
import traceback
import warnings
from collections import Counter
from collections.abc import Callable
from collections.abc import Generator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import music21 as m21
import pandas as pd
import partitura as pt
import partitura.score
import symusic
import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from data.components.chunking_utils import ChunkingConfig
from data.components.chunking_utils import chunk_partitura_score_and_performances
from utils.abc_xml_conversion import conversion
from utils.constants import SKIP
from utils.constants import SUCCESS_FILENAME
from utils.constants import TEST_PIECE_IDS
from utils.constants import TO_IGNORE_INDICES


@dataclass
class ScorePerformanceEntry:
    xml_score: Path
    midi_performance: Path
    midi_score_annotations: Path = None
    midi_score: Path = None
    composer: str = None
    title: str = None
    split: str = None


class ScorePerformanceDataset(Dataset):
    """Dataset for a dataset of pairs of MusicXML scores and MIDI performances."""

    def __init__(
        self,
        entries: list[ScorePerformanceEntry],
        transform: Any = None,
    ) -> None:
        """Initialize a dataset of pairs of MusicXML scores and MIDI performances.

        :param entries: A list of ScorePerformanceEntry objects.
        """
        self.entries = entries
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        :return: The number of samples in the dataset.
        """
        return len(self.entries)

    def __getitem__(self, idx: int) -> ScorePerformanceEntry:
        """Return a sample from the dataset.

        :param idx: The index of the sample to return.
        :return: The sample at the given index.
        """
        entry = self.entries[idx]
        if self.transform:
            entry = self.transform(entry)
        return entry


class AsapDataset(ScorePerformanceDataset):
    """Dataset for the ASAP dataset.

    ASAP is a dataset of aligned musical scores (both MIDI and MusicXML) and performances (audio and MIDI), all with downbeat, beat, time
    signature, and key signature annotations.

    This dataset is used to load only the MIDI and MusicXML files from the dataset.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        filter_fn: Callable[[ScorePerformanceEntry], bool] = lambda x: True,
        transform: Any = None,
    ) -> None:
        """Initialize an `AsapDataset` (using the MIDI2ScoreTransformer method).

        :param data_dir: The directory where the ASAP dataset is stored.
        :param split: The split to use (either "all", "train", "validation", or "test").
        :param filter_fn: A function to filter the entries.
        :param transform: A transform to apply to the entries.
        """
        self.data_dir = data_dir
        self.split = split
        assert split in ["all", "train", "validation", "test"], f"Invalid split: {split}"
        entries = self.read_metadata_like_m2st(data_dir, split)
        entries = list(filter(filter_fn, entries))
        super().__init__(entries, transform)

    @staticmethod
    def read_metadata(data_dir: Path) -> list[ScorePerformanceEntry]:
        """Read the metadata of the dataset.

        CURRENTLY UNUSED

        :param data_dir: The directory where the dataset is stored.
        :return: The metadata of the dataset.
        """
        with open(data_dir / "metadata.csv", "r") as f:
            reader = csv.DictReader(f)
            return [
                ScorePerformanceEntry(
                    # fmt: off
                    composer=entry["composer"],
                    title=entry["title"],
                    xml_score=data_dir / Path(entry["xml_score"]),
                    midi_score=data_dir / Path(entry["midi_score"]) if "midi_score" in entry else None,
                    midi_performance=data_dir / Path(entry["midi_performance"]),
                    midi_score_annotations=data_dir / Path(entry["midi_score_annotations"]) if "midi_score_annotations" in entry else None,
                    # fmt: on
                )
                for entry in reader
            ]

    @staticmethod
    def read_metadata_like_m2st(
        data_dir: Path,
        split: str,
    ) -> list[ScorePerformanceEntry]:
        """Read the metadata of the dataset with the MIDI2ScoreTransformer method.

        :param data_dir: The directory where the ASAP dataset is stored.
        :return: The metadata of the dataset.
        """
        acpas_r_file = data_dir / "metadata_R.csv"
        acpas_s_file = data_dir / "metadata_S.csv"
        assert (
            acpas_r_file.exists() and acpas_s_file.exists()
        ), "ACPAS metadata files must be present in the ASAP directory"
        data_real = pd.read_csv(acpas_r_file)
        data_synthetic = pd.read_csv(acpas_s_file)
        asap_annotations = json.load(open(data_dir / "asap_annotations.json"))
        UNALIGNED = set(
            "{ASAP}/" + k
            for k, v in asap_annotations.items()
            if not v["score_and_performance_aligned"]
        )
        # Filter
        data = pd.concat([data_real, data_synthetic])
        # Initial filtering
        data = data[(data["source"] == "ASAP") & data["aligned"]]
        data = data[~data["performance_MIDI_external"].isin(SKIP)]
        data = data[~data["performance_MIDI_external"].isin(UNALIGNED)]
        data = data.drop_duplicates(subset=["performance_MIDI_external"])
        # Filter by annotations
        data.reset_index(inplace=True)
        data.drop(TO_IGNORE_INDICES, inplace=True)

        # Split
        test_idx = data["piece_id"].isin(TEST_PIECE_IDS)
        match split:
            case "all":
                dataset = data
            case "test":
                dataset = data[test_idx]
            case "validation":
                dataset = data[
                    (data["piece_id"] % 10 == 0) & (~data["piece_id"].isin(TEST_PIECE_IDS))
                ]
            case "train":
                dataset = data[
                    (data["piece_id"] % 10 != 0) & (~data["piece_id"].isin(TEST_PIECE_IDS))
                ]
            case _:
                raise ValueError(f"Invalid split: {split}")

        entries = []
        for entry in dataset.to_dict(orient="records"):
            performance_path = Path(
                entry["performance_MIDI_external"].replace("{ASAP}", str(data_dir))
            )
            folder_path = performance_path.parent
            score_path = folder_path / "xml_score.musicxml"
            entries.append(
                ScorePerformanceEntry(
                    composer=entry["composer"],
                    title=entry["title"],
                    xml_score=score_path,
                    midi_performance=performance_path,
                )
            )

        return entries


if __name__ == '__main__':
    ds = AsapDataset(
        Path("~/Documents/datasets/asap-dataset").expanduser(),
        split="all",
        filter_fn=lambda x: x.composer == "Bach",
    )
    print(f"Number of performances: {len(ds)}")
    print(f"Number of scores: {len(set(entry.xml_score for entry in ds.entries))}")
    print(ds[0].midi_performance)
    print(ds[0].xml_score)
