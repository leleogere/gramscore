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
import line_profiler
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


TEST_PIECES = [
    "Bach/Fugue/bwv_846",
    "Beethoven/Piano_Sonatas/10-1",
    "Brahms/Six_Pieces_op_118/2",
    "Chopin/Ballades/1",
    "Debussy/Images_Book_1/1_Reflets_dans_lEau",
    "Haydn/Keyboard_Sonatas/31-1",
    "Liszt/Annees_de_pelerinage_2/1_Gondoliera",
    "Mozart/Piano_Sonatas/12-1",  # Why this one rather than Mozart/Fantasie_475?
    "Prokofiev/Toccata",
    "Rachmaninoff/Preludes_op_23/4",
    "Ravel/Gaspard_de_la_Nuit/1_Ondine",
    "Schubert/Impromptu_op.90_D.899/1",
    "Schumann/Arabeske",
    "Scriabin/Etudes_op_8/11",
]


@dataclass
class ScorePerformanceEntry:
    xml_score: Path
    midi_performance: Path
    midi_score_annotations: Path | None = None
    midi_score: Path | None = None
    folder: Path | None = None
    base_path: Path | None = None
    composer: str | None = None
    title: str | None = None
    split: str | None = None

    def _get_relative_path(self, absolute_path: Path) -> Path:
        if self.base_path is None:
            raise ValueError("The attribute `base_path` must be set to get a relative path")
        return absolute_path.relative_to(self.base_path)

    @property
    def xml_score_relative(self) -> Path:
        return self._get_relative_path(self.xml_score)

    @property
    def midi_performance_relative(self) -> Path:
        return self._get_relative_path(self.midi_performance)

    @property
    def midi_score_annotations_relative(self) -> Path:
        return self._get_relative_path(self.midi_score_annotations)

    @property
    def midi_score_relative(self) -> Path:
        return self._get_relative_path(self.midi_score)

    @property
    def folder_relative(self) -> Path:
        return self._get_relative_path(self.folder)


class ScorePerformanceDataset(Dataset):
    """Dataset for a dataset of pairs of MusicXML scores and MIDI performances."""

    def __init__(
        self,
        entries: list[ScorePerformanceEntry],
        transforms: Callable | list[Callable] = None,
    ) -> None:
        """Initialize a dataset of pairs of MusicXML scores and MIDI performances.

        :param entries: A list of ScorePerformanceEntry objects.
        """
        self.entries = entries
        self.transforms = transforms
        if self.transforms is not None and not isinstance(transforms, list):
            self.transforms = [self.transforms]

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        :return: The number of samples in the dataset.
        """
        return len(self.entries)

    # @line_profiler.profile
    def __getitem__(self, idx: int) -> ScorePerformanceEntry | Any:
        """Return a sample from the dataset.

        :param idx: The index of the sample to return.
        :return: The sample at the given index.
        """
        entry = self.entries[idx]
        # worker = torch.utils.data.get_worker_info()
        # logger.trace(
        #     f"{worker.id+1}/{worker.num_workers}: Loading {entry.xml_score_relative} and {entry.midi_performance.name}"
        # )
        if self.transforms:
            entry = self.apply_transforms(entry, self.transforms)
        return entry

    @classmethod
    def apply_transforms(cls, sample: Any, transforms: list[Callable]) -> Any:
        for i, transform in enumerate(transforms):
            sample = transform(sample)
            if isinstance(sample, list):
                # If the transform returns a list, apply the transforms to each element
                sample = [cls.apply_transforms(s, transforms[i + 1 :]) for s in sample]
                break
        return sample


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
        include_midi_scores: bool = False,
        filter_fn: Callable[[ScorePerformanceEntry], bool] = lambda x: True,
        transforms: Callable | list[Callable] = None,
    ) -> None:
        """Initialize an `AsapDataset` (using the MIDI2ScoreTransformer method).

        :param data_dir: The directory where the ASAP dataset is stored.
        :param split: The split to use (either "all", "train", "validation", or "test").
        :param include_midi_scores: Whether to include the quantified MIDI score as performances in
            the dataset.
        :param filter_fn: A function to filter the entries.
        :param transforms: A transform to apply to the entries.
        """
        self.data_dir = data_dir
        self.split = split
        assert split in ["all", "train", "validation", "test"], f"Invalid split: {split}"
        # entries = self.read_metadata_like_m2st(data_dir, split)
        entries = self.read_metadata(data_dir, split, include_midi_scores=include_midi_scores)
        entries = list(filter(filter_fn, entries))
        super().__init__(entries, transforms)

    @staticmethod
    def read_metadata(
        data_dir: Path,
        split: str,
        include_midi_scores: bool = False,
    ) -> list[ScorePerformanceEntry]:
        """Read the metadata of the dataset.


        :param data_dir: The directory where the dataset is stored.
        :param split: The split to use (either "all", "train", "validation", or "test").
        :return: The metadata of the dataset.
        """
        # Read entries from metadata
        with open(data_dir / "metadata.csv", "r") as f:
            reader = list(csv.DictReader(f))
        entries = [
            ScorePerformanceEntry(
                # fmt: off
                composer=entry["composer"],
                title=entry["title"],
                xml_score=data_dir / entry["xml_score"],
                midi_score=data_dir / entry["midi_score"] if "midi_score" in entry else None,
                midi_performance=data_dir / entry["midi_performance"],
                midi_score_annotations=data_dir / entry["midi_score_annotations"] if "midi_score_annotations" in entry else None,
                folder=data_dir / entry["folder"],
                base_path=data_dir,
                # fmt: on
            )
            for entry in reader
        ]

        # all_folders = sorted(
        #     set(
        #         entry.folder_relative
        #         for entry in entries
        #         if entry.folder_relative not in TEST_PIECES
        #     )
        # )
        unique_scores = {entry.folder_relative: entry for entry in entries}
        if include_midi_scores:
            # Get a dummy performance for each score
            dummy_performances = unique_scores.values()
            for dummy_performance in dummy_performances:
                # Add the MIDI score as a performance
                entries.append(
                    ScorePerformanceEntry(
                        composer=dummy_performance.composer,
                        title=dummy_performance.title,
                        xml_score=dummy_performance.folder / "xml_score.musicxml",
                        midi_performance=dummy_performance.folder / "midi_score.mid",
                        midi_score_annotations=dummy_performance.folder
                        / "midi_score_annotations.mid",
                        midi_score=dummy_performance.folder / "midi_score.mid",
                        folder=dummy_performance.folder,
                        base_path=data_dir,
                    )
                )

        # Assign an index to each piece that will be used to split the dataset into train and
        # validation sets without piece overlap between them
        all_non_test_pieces = filter(lambda folder: folder not in TEST_PIECES, unique_scores.keys())
        all_non_test_pieces_indexer = {folder: i for i, folder in enumerate(all_non_test_pieces)}

        if split == "all":
            pass

        elif split == "train":
            entries = [
                entry
                for entry in entries
                if entry.folder_relative not in TEST_PIECES
                and all_non_test_pieces_indexer[entry.folder_relative] % 10 != 0
            ]

        elif split == "validation":
            entries = [
                entry
                for entry in entries
                if entry.folder_relative not in TEST_PIECES
                and all_non_test_pieces_indexer[entry.folder_relative] % 10 == 0
            ]

        elif split == "test":
            entries = [entry for entry in entries if entry.folder_relative in TEST_PIECES]

        else:
            raise ValueError(f"Invalid split: {split}")

        return entries

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
                    base_path=data_dir,
                )
            )

        return entries


if __name__ == '__main__':
    ds = AsapDataset(
        Path("~/Documents/datasets/asap-dataset").expanduser(),
        split="all",
        include_midi_scores=True,
        filter_fn=lambda x: x.composer == "Bach" and x.midi_performance.stem != "midi_score",
    )  # TODO: fix this to load the MIDI scores
    print(f"Number of performances: {len(ds)}")
    print(f"Number of scores: {len(set(entry.xml_score for entry in ds.entries))}")
    print(ds[0].midi_performance)
    print(ds[0].xml_score)

    # # Print all scores with the associated performances
    # from collections import defaultdict
    #
    # score_to_performances = defaultdict(list)
    # for entry in ds.entries:
    #     score_to_performances[entry.xml_score].append(entry.midi_performance)
    # score_to_performances = dict(sorted(score_to_performances.items(), key=lambda x: x[0].stem))
    # for score, performances in score_to_performances.items():
    #     print(f"{score}")
    #     for performance in performances:
    #         print(f"  * {performance}")
