import csv
import itertools
import json
import random
import traceback
import warnings
from collections import Counter
from collections.abc import Callable
from collections.abc import Generator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from pprint import pprint
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
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from data.components.chunking_utils import ChunkingConfig
import data.components.transforms as t
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


class ScorePerformanceDataset(IterableDataset):
    """
    Dataset for a dataset of pairs of MusicXML scores and MIDI performances.

    This is implemented as an IterableDataset, because torch was causing issues with prefetching
    items (stopping at the end of the epoch and struggling to dispatch the items across workers).
    """

    def __init__(
        self,
        entries: list[ScorePerformanceEntry],
        transforms: Callable | list[Callable] = None,
        shuffle: bool = False,
    ) -> None:
        """Initialize a dataset of pairs of MusicXML scores and MIDI performances.

        :param entries: A list of ScorePerformanceEntry objects.
        """
        self.entries = entries
        self.transforms = transforms
        if self.transforms is not None and not isinstance(transforms, list):
            self.transforms = [self.transforms]
        self.shuffle = shuffle

    # @line_profiler.profile
    def __iter__(self) -> Generator[dict[str, Any] | ScorePerformanceEntry, None, None]:
        """Return a sample from the dataset.

        :param idx: The index of the sample to return.
        :return: The sample at the given index.
        """
        worker = torch.utils.data.get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)

        if self.shuffle:
            # Shuffle the entries for each worker QUESTION: will this be the same for each worker?
            entries = self.entries[worker_id::num_workers]
            random.shuffle(entries)
        else:
            # Split the entries between workers
            entries = self.entries[worker_id::num_workers]

        for i, entry in enumerate(entries):
            if self.transforms:
                yield from self.apply_transforms(entry, self.transforms)
            else:
                yield entry

    @classmethod
    def apply_transforms(
        cls,
        entry: ScorePerformanceEntry,
        transforms: list[Callable],
        unfold_lists: bool = True,
    ):
        for i, transform in enumerate(transforms):
            entry = transform(entry)
            # In case a transform generates multiple entries, we need to yield them all
            if unfold_lists and isinstance(entry, list):
                for e in entry:
                    yield from cls.apply_transforms(e, transforms[i + 1 :])
                break
        else:
            yield entry


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
        shuffle: bool = False,
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
        super().__init__(entries, transforms, shuffle=shuffle)

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


class TrainAsapDataset(AsapDataset):
    """Train dataset for the ASAP dataset.

    This dataset is used to load only the MIDI and MusicXML files from the dataset.
    """

    def __init__(
        self,
        data_dir: Path,
        include_midi_scores: bool = False,
        filter_fn: Callable[[ScorePerformanceEntry], bool] = lambda x: True,
        batch_size=None,
        shuffle: bool = True,
    ) -> None:
        mp_manager = torch.multiprocessing.Manager()
        transforms = [
            t.ToDict(),
            t.Cache(
                transforms=[
                    t.ScoreReader(),
                    t.VoiceRestFiller(),
                    t.VoiceReorderer(),
                ],
                cache_key=lambda sample: sample["entry"].xml_score,
                # not shared across workers as some scores not picklable...
            ),
            t.Cache(
                transforms=[
                    t.AllowedTranspositionComputer(
                        intervals=[
                            pt.score.Interval(
                                number=int(inter[0]), quality=inter[1:], direction=direction
                            )
                            for inter in ["2M", "2m", "3M", "3m", "4P", "4A", "4d"]
                            for direction in ["up", "down"]
                        ]
                    ),
                ],
                cache_key=lambda sample: sample["entry"].xml_score,
                keys_to_cache=["allowed_transpositions"],
                cache=mp_manager.dict(),  # shared across workers
            ),
            t.PerformanceReader(),
            t.MusicChunker(
                num_performed_notes=500,
                annotations=data_dir / "asap_annotations.json",
                mode="random",
            ),
            t.RandomTransposer(),
            t.PerformanceSpeedAugmenter(min_factor=0.8, max_factor=1.2),
            t.ScoreVectorizer(),
            t.PerformanceVectorizer(),
        ]
        super().__init__(
            data_dir=data_dir,
            split="train",
            include_midi_scores=include_midi_scores,
            filter_fn=filter_fn,
            transforms=transforms,
            shuffle=shuffle,
        )
        self.batch_size = batch_size


class ValidationAsapDataset(AsapDataset):
    """Validation dataset for the ASAP dataset.

    This dataset is used to load only the MIDI and MusicXML files from the dataset.
    """

    def __init__(
        self,
        data_dir: Path,
        include_midi_scores: bool = False,
        filter_fn: Callable[[ScorePerformanceEntry], bool] = lambda x: True,
        batch_size=None,
    ) -> None:
        transforms = [
            t.ToDict(),
            t.Cache(
                transforms=[
                    t.ScoreReader(),
                    t.VoiceRestFiller(),
                    t.VoiceReorderer(),
                ],
                cache_key=lambda sample: sample["entry"].xml_score,
                # not shared across workers as some scores not picklable...
            ),
            t.PerformanceReader(),
            t.MusicChunker(
                num_performed_notes=500,
                annotations=data_dir / "asap_annotations.json",
                mode="all_wo_overlap",
            ),
            t.ScoreVectorizer(),
            t.PerformanceVectorizer(),
        ]
        super().__init__(
            data_dir=data_dir,
            split="validation",
            include_midi_scores=include_midi_scores,
            filter_fn=filter_fn,
            transforms=transforms,
            shuffle=False,
        )
        self.batch_size = batch_size


class TestAsapDataset(AsapDataset):
    """Test dataset for the ASAP dataset.

    TODO: Is test set working?

    This dataset is used to load only the MIDI and MusicXML files from the dataset.
    """

    def __init__(
        self,
        data_dir: Path,
        include_midi_scores: bool = False,
        filter_fn: Callable[[ScorePerformanceEntry], bool] = lambda x: True,
        batch_size=None,
    ) -> None:
        transforms = [
            t.ToDict(),
            t.ScoreReader(),
            t.VoiceRestFiller(),
            t.VoiceReorderer(),
            t.PerformanceReader(),
            t.MusicChunker(
                num_performed_notes=500,
                annotations=data_dir / "asap_annotations.json",
                mode="all_wo_overlap",
            ),
            t.ScoreVectorizer(),
            t.PerformanceVectorizer(),
        ]
        super().__init__(
            data_dir=data_dir,
            split="test",
            include_midi_scores=include_midi_scores,
            filter_fn=filter_fn,
            transforms=transforms,
            shuffle=False,
        )
        self.batch_size = batch_size


if __name__ == '__main__':
    import tabulate

    # for cls in [TrainAsapDataset, ValidationAsapDataset, TestAsapDataset]:
    for cls in [TrainAsapDataset]:
        ds = cls(
            Path("~/Documents/datasets/asap-dataset").expanduser(),
            include_midi_scores=True,
            filter_fn=lambda x: x.composer == "Bach" and x.midi_performance.stem == "midi_score",
            shuffle=True,
        )
        ds.entries = ds.entries[:5]
        # print(f"Number of performances: {len(ds.entries)}")
        # print(f"Number of scores: {len(set(entry.xml_score for entry in ds.entries))}")
        table = []
        for sample in ds:
            # print(f"{entry.xml_score} -> {entry.midi_performance.name}")
            table.append(
                [
                    sample["entry"].midi_performance_relative.as_posix(),
                    sample["chunk_start"],
                    sample["chunk_end"],
                ]
            )
        print(f"\nClass: {cls.__name__}")
        print(tabulate.tabulate(table, headers=["Performance", "Start", "End"]))

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
