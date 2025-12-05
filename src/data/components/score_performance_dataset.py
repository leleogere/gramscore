import ast
import csv
import json
import math
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from typing import Literal

import joblib
import pandas as pd
import partitura as pt
import partitura.score
import pretty_midi as pm
import symusic
import torch
from loguru import logger
from lxml import etree
from torch.utils.data import Dataset
from tqdm import tqdm

from data.components.check_tokenizable import check_full
from models.automata.vectorization_torch import TokenTensor
from utils.constants import SKIP
from utils.constants import TEST_PIECE_IDS
from utils.constants import TO_IGNORE_INDICES
from utils.musicxml import read_musicxml

tqdm.pandas()
sys.setrecursionlimit(2**31 - 1)


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
    allowed_transpositions: list[pt.score.Interval] | None = None

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

    @property
    def id(self) -> tuple[str, str]:
        """
        Returns a unique piece id, as a tuple (composer, title), used to split correctly into
        train/val/test sets without any piece overlap.
        """
        return (self.composer, self.title)


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
            try:
                sample = transform(sample)
            except Exception as e:
                logger.error(
                    f"Failed to apply transform {transform} on "
                    f"{sample['entry'].xml_score_relative}: {e}"
                )
                raise e
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

    TEST_PIECES = [
        ("Bach", "Fugue_bwv_846"),
        ("Beethoven", "Piano_Sonatas_10-1"),
        ("Brahms", "Six_Pieces_op_118_2"),
        ("Chopin", "Ballades_1"),
        ("Debussy", "Images_Book_1_1_Reflets_dans_lEau"),
        ("Haydn", "Keyboard_Sonatas_31-1"),
        ("Liszt", "Annees_de_pelerinage_2_1_Gondoliera"),
        ("Mozart", "Piano_Sonatas_12-1"),  # Why this one rather than Mozart/Fantasie_475?
        ("Prokofiev", "Toccata"),
        ("Rachmaninoff", "Preludes_op_23_4"),
        ("Ravel", "Gaspard_de_la_Nuit_1_Ondine"),
        ("Schubert", "Impromptu_op.90_D.899_1"),
        ("Schumann", "Arabeske"),
        ("Scriabin", "Etudes_op_8_11"),
    ]

    def __init__(
        self,
        data_dir: Path,
        split: str,
        include_midi_scores: bool = False,
        filter_fn: Callable[[ScorePerformanceEntry], bool] | None = None,
        valid_tsv_file: Path | None = None,
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
        if filter_fn is not None:
            entries = list(filter(filter_fn, entries))
        if valid_tsv_file is not None:
            with valid_tsv_file.open("r") as f:
                reader = csv.DictReader(f, delimiter="\t")
                valid_folders = {row["folder"] for row in reader if row["valid"] == "True"}
                entries = list(
                    filter(lambda e: e.folder_relative.as_posix() in valid_folders, entries)
                )
        super().__init__(entries, transforms)

    @classmethod
    def read_metadata(
        cls,
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
        unique_pieces = {entry.id: entry for entry in entries}
        if include_midi_scores:
            # Get a dummy performance for each score
            dummy_performances = unique_pieces.values()
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
        all_non_test_pieces = filter(lambda id_: id_ not in cls.TEST_PIECES, unique_pieces.keys())
        all_non_test_pieces_indexer = {id_: i for i, id_ in enumerate(all_non_test_pieces)}

        if split == "all":
            pass

        elif split == "train":
            entries = [
                entry
                for entry in entries
                if entry.id not in cls.TEST_PIECES
                and all_non_test_pieces_indexer[entry.id] % 10 != 0
            ]

        elif split == "validation":
            entries = [
                entry
                for entry in entries
                if entry.id not in cls.TEST_PIECES
                and all_non_test_pieces_indexer[entry.id] % 10 == 0
            ]

        elif split == "test":
            entries = [entry for entry in entries if entry.id in cls.TEST_PIECES]

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


class PdmxPianoDataset(ScorePerformanceDataset):
    """
    This dataset contains piano solo PDMX scores, with generated MIDI score performances associated.
    """

    def __init__(
        self,
        directory: Path,
        split: Literal["train", "validation", "all"],
        transforms: Callable | list[Callable] = None,
        force_metadata_extraction: bool = False,
        asap_path: Path | None = None,  # Used to filter PDMX for overlap
    ):
        self.directory = directory
        self.split = split
        assert split in ["all", "train", "validation"], f"Invalid split: {split}"

        # Extract additional metadata
        self.metadata_file = self.directory / "PDMX.csv"
        self.additional_metadata_file = self.directory / "PDMX_extended_metadata.csv"
        logger.info(f"Extracting additional metadata from {self.metadata_file}")
        pdmx = self.extract_additional_metadata(
            asap_path=asap_path,
            prev=None if force_metadata_extraction else pd.read_csv(self.additional_metadata_file),
        )
        pdmx.to_csv(self.additional_metadata_file, index=False)

        # Read entries
        entries = self.read_metadata()
        super().__init__(entries, transforms)

    def extract_additional_metadata(
        self, asap_path: Path | None = None, prev: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Extract additional metadata from the entries and return it as a DataFrame."""
        metadata_path: Path = self.directory / "PDMX.csv"
        if not metadata_path.exists():
            raise FileNotFoundError("PDMX.csv not found, make sure the dataset is correct.")
        if prev is not None:
            logger.info("Only adding missing information to existing file.")
            pdmx = prev
        else:
            pdmx = pd.read_csv(metadata_path)
            # pdmx = pdmx.head(100)
            pdmx["usable"] = True

        # Extract the instruments of each score
        if "instruments" not in pdmx.columns:
            logger.info(f"Extracting instruments information")
            pdmx["instruments"] = pdmx["metadata"].progress_apply(self._extract_instrument)
            piano_indices = pdmx["instruments"].apply(lambda i: i == ["Piano"])
            logger.info(f"Number of piano scores: {piano_indices.sum()}")
            pdmx.loc[~piano_indices, "usable"] = False

        # Get all time signatures in the piece
        if "time_signatures" not in pdmx.columns:
            logger.info(f"Extracting time signatures for each score")
            # pdmx.loc[pdmx["usable"], "time_signatures"] = pdmx["mxl"].progress_apply(
            #     self._extract_time_signatures
            # )
            pdmx.loc[pdmx["usable"], "time_signatures"] = [
                _
                for _ in tqdm(
                    joblib.Parallel(n_jobs=-1, return_as="generator")(
                        joblib.delayed(self._extract_time_signatures)(path)
                        for path in pdmx.loc[pdmx["usable"], "mxl"]
                    ),
                    total=pdmx["usable"].sum(),
                )
            ]

        # Filter for overlap with ASAP validation and test sets
        if asap_path is not None and "similar_to_asap" not in pdmx.columns:
            from sentence_transformers import SentenceTransformer
            import re
            import numpy as np

            # Load PDMX embeddings
            logger.info("Loading PDMX embeddings")
            pdmx_embeddings_path = self.directory / "embeddings.csv"
            pdmx_magnitudes_path = self.directory / "magnitudes.csv"
            if not pdmx_embeddings_path.exists() or not pdmx_magnitudes_path.exists():
                raise FileNotFoundError(
                    "PDMX embeddings not found, please generate them first."
                )  # TODO: automate the generation if not found
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pdmx_embeddings = torch.from_numpy(np.loadtxt(pdmx_embeddings_path, delimiter=",")).to(
                device
            )
            pdmx_magnitudes = torch.from_numpy(np.loadtxt(pdmx_magnitudes_path, delimiter=",")).to(
                device
            )

            # Function to create a descriptor for an ASAP entry
            def create_descriptor(title: str, composer: str):
                descriptor = f"{title}; by {composer}."
                descriptor = re.sub(pattern=r'[^ a-zA-Z0-9,.?!;:()&-]', repl=" ", string=descriptor)
                descriptor = " ".join(descriptor.split())  # remove wierd whitespace
                return descriptor

            # First load ASAP entries
            logger.info(f"Loading ASAP entries from {asap_path}")
            asap_val = AsapDataset(asap_path, split="validation")
            asap_test = AsapDataset(asap_path, split="test")
            asap_entries = asap_val.entries + asap_test.entries
            asap_entries = {(e.composer, e.title) for e in asap_entries}
            asap_descriptions = [
                create_descriptor(title=title, composer=composer)
                for composer, title in asap_entries
            ]
            logger.info(f"Number of ASAP validation and test pieces: {len(asap_descriptions)}")
            logger.info(f"Computing ASAP embeddings")
            model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2", device=str(device))
            asap_embeddings = model.encode(
                sentences=asap_descriptions,
                batch_size=64,
                show_progress_bar=True,
                output_value="sentence_embedding",
                device=device,
                convert_to_tensor=True,
            ).to(torch.float64)
            asap_magnitudes = torch.norm(asap_embeddings, dim=1)

            # Find pieces in PDMX that are too close to ASAP pieces
            similarity_threshold = 0.8
            logger.info(
                f"Finding PDMX pieces too similar to ASAP pieces (sim >= {similarity_threshold})"
            )
            for query_embedding, query_magnitude in zip(tqdm(asap_embeddings), asap_magnitudes):
                similarities = torch.matmul(input=pdmx_embeddings, other=query_embedding)
                similarities = similarities / (pdmx_magnitudes * query_magnitude)
                similarities = (similarities + 1) / 2
                pdmx["similar_to_asap"] = (similarities >= similarity_threshold).cpu()
            logger.info(
                f"Number of PDMX pieces similar to ASAP pieces: {pdmx['similar_to_asap'].sum()}"
            )
            pdmx.loc[pdmx["similar_to_asap"], "usable"] = False

        # Count the number of notes for each performance
        if "n_notes_midi" not in pdmx.columns:
            logger.info(f"Counting number of notes in each MIDI performance")
            pdmx["n_notes_midi"] = pdmx["mid"].progress_apply(self._count_midi_notes)

        # Count the number of measures in the unfolded score
        if "n_measures_score" not in pdmx.columns:
            logger.info(f"Counting number of measures in each MusicXML score")
            pdmx["n_measures_score"] = pdmx["mxl"].progress_apply(
                self._count_unfolded_score_measures
            )

        # Count the number of measures in the unfolded midi score
        if "n_measures_midi" not in pdmx.columns:
            logger.info(f"Counting number of measures in each MIDI score")
            pdmx["n_measures_midi"] = pdmx["mid"].progress_apply(self._count_unfolded_midi_measures)

        # Check if scores are tokenizable
        if "errors" not in pdmx.columns:
            logger.info(f"Checking tokenizability of each score")
            pdmx.loc[pdmx["usable"], "errors"] = pdmx.loc[pdmx["usable"], "mxl"].progress_apply(
                self._check_score_tokenizable
            )
            has_errors = pdmx["errors"] != ""
            pdmx.loc[has_errors, "usable"] = False
            logger.info(f"Number of scores with errors: {has_errors.sum()}")

        # Check the transposition range of each score
        if "allowed_transpositions" not in pdmx.columns:
            logger.info(f"Computing allowed transpositions for each score")
            tested_intervals = [
                pt.score.Interval(number=int(inter[0]), quality=inter[1:], direction=direction)
                for inter in ["2M", "2m", "3M", "3m", "4P", "4A", "4d"]
                for direction in ["up", "down"]
            ] + [pt.score.Interval(1, "P", "up")]
            pdmx.loc[pdmx["usable"], "allowed_transpositions"] = pdmx.loc[
                pdmx["usable"], "mxl"
            ].progress_apply(
                partial(self._compute_allowed_transpositions, intervals=tested_intervals)
            )

        return pdmx

    def read_metadata(self) -> list[ScorePerformanceEntry]:
        # Read PDMX metadata
        metadata_path = self.additional_metadata_file
        df = pd.read_csv(metadata_path)

        # Parse allowed transpositions from string to list
        df["allowed_transpositions"] = df["allowed_transpositions"].apply(
            lambda x: (
                [i for i in x.strip("[]").replace("'", "").split(", ") if i != '']
                if isinstance(x, str)
                else []
            )
        )  # TODO: avoid using the list repr for export/import

        # Parse time signatures from string to set
        df["time_signatures"] = df["time_signatures"].apply(
            lambda set_str: (
                ast.literal_eval(set_str) if isinstance(set_str, str) and set_str != "{}" else set()
            )
        )

        # Only keep usable scores
        df = df[df["usable"]]

        # Remove pieces too short
        # df = df[(df["n_notes_midi"] >= 10) & (df["n_notes_midi"] <= 500)]
        df = df[df["n_notes_midi"] >= 3]

        # Remove pieces with mismatching number of measures between score and midi
        df = df[(df["n_measures_score"] == df["n_measures_midi"]) & (df["n_measures_score"] != -1)]

        # Remove pieces with unallowed time signatures
        df = df[
            (len(df["time_signatures"]) > 0)  # At least one time signature (empty set = error)
            & (  # TS allowed to be vectorized
                df["time_signatures"].map(
                    lambda tss: all(ts in TokenTensor.TS_INDEXER for ts in tss)
                )
            )
        ]

        logger.info(f"Number of entries before split: {len(df)}")
        logger.info(f"Number of measures: {sum(df['n_measures_score'])}")

        # Create splits
        num_validation_pieces = 1000
        val_every_nth = int(len(df) / num_validation_pieces)
        if self.split == "all":
            indices = list(range(len(df)))
        elif self.split == "validation":
            indices = [i for i in range(len(df)) if i % val_every_nth == 0]
        elif self.split == "train":
            indices = [i for i in range(len(df)) if i % val_every_nth != 0]
        else:
            raise ValueError(f"Invalid split: {self.split}")

        # Create entries
        entries = [
            ScorePerformanceEntry(
                xml_score=self.directory / row["mxl"],
                midi_performance=self.directory / row["mid"],
                midi_score_annotations=None,
                midi_score=None,
                folder=None,
                base_path=self.directory,
                composer=self._get_composer(row),
                title=self._get_title(row),
                allowed_transpositions=[
                    self._interval_str_to_interval(interval)
                    for interval in row["allowed_transpositions"]
                ],
            )
            for _, row in df.iloc[indices].iterrows()
        ]

        return entries

    @staticmethod
    def _interval_str_to_interval(interval_str: str) -> pt.score.Interval:
        if interval_str.startswith("-"):
            direction = "down"
            interval_str = interval_str[1:]
        else:
            direction = "up"
        number = int(interval_str[0])
        quality = interval_str[1]
        return pt.score.Interval(number=number, quality=quality, direction=direction)

    @staticmethod
    def _get_composer(row) -> str:
        """Make a string from composer and artist"""
        composer = row["composer_name"]
        artist = row["artist_name"]
        # If missing, will be N/A, which is a float
        if not isinstance(composer, float) and not isinstance(artist, float):
            return f"artist ({composer})"
        elif not isinstance(composer, float):
            return f"{composer}"
        else:
            return f"{artist}"

    @staticmethod
    def _get_title(row) -> str:
        """Make a string from song name, title and subtitle"""
        song = row["song_name"]
        title = row["title"]
        subtitle = row["subtitle"]
        parts = []
        if not isinstance(song, float):
            parts.append(song)
        if not isinstance(title, float):
            parts.append(title)
        full_title = " - ".join(parts)
        if not isinstance(subtitle, float):
            full_title += f" [{subtitle}]"
        return full_title

    def _extract_instrument(self, path: Path | str) -> list[str] | None:
        file_path = self.directory / path
        try:
            with file_path.open('r') as file:
                data = json.load(file)
                # Extract the instrument information
                instruments = [i["name"] for i in data["data"]["score"]["instruments"]]
                return instruments
        except KeyError as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def _count_midi_notes(self, path: Path) -> int:
        if isinstance(path, float) and math.isnan(path):
            return -1  # Path not available
        path = self.directory / path
        try:
            score = symusic.Score(path)
            return sum(len(track.notes) for track in score.tracks)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return -1  # Indicate an error

    def _check_score_tokenizable(self, path: Path) -> str:
        if isinstance(path, float) and math.isnan(path):
            return "MissingScore"
        path = self.directory / path
        error = check_full(path)
        if error is None:
            return ""
        else:
            return type(error).__name__

    def _compute_allowed_transpositions(
        self,
        path: Path,
        intervals: list[pt.score.Interval],
    ) -> list[str]:
        if isinstance(path, float) and math.isnan(path):
            return []
        path = self.directory / path
        # No try as this assumes that any reading error has been caught before
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = pt.load_musicxml(path)
        allowed_intervals = []
        for interval in intervals:
            try:
                pt.utils.music.transpose(
                    score,
                    interval,
                    transpose_key_signatures=True,
                    inplace=False,
                )
                allowed_intervals.append(interval)
                logger.trace(
                    f"Score {path} can be transposed "
                    f"{interval.direction} by a {interval.number}{interval.quality} interval."
                )
            except (AssertionError, ValueError, Exception) as e:
                # Not ideal but the easiest way to capture the correct exception
                if isinstance(e, AssertionError) and e.args[0].endswith(
                    "is not in the range -2 to 2."
                ):
                    logger.trace(
                        f"Can't transpose {path} {interval.direction} by a "
                        f"{interval.number}{interval.quality} interval (alterations > 2)."
                    )
                elif isinstance(e, ValueError) and e.args[0].endswith(
                    "The maximum number of fifths is +/- 7."
                ):
                    logger.trace(
                        f"Can't transpose {path} {interval.direction} by a "
                        f"{interval.number}{interval.quality} interval (key fifths > 7)."
                    )
                elif isinstance(e, Exception) and e.args[0].startswith("Unknown mode "):
                    mode = e.args[0].split("Unknown mode ")[1]
                    logger.trace(
                        f"Can't transpose {path} {interval.direction} by a "
                        f"{interval.number}{interval.quality} interval (unknown mode: {mode})."
                    )
                else:
                    raise
        logger.trace(
            f"Allowed transpositions for {path}: "
            + ", ".join(
                f"{'-' if i.direction == 'down' else ''}{i.number}{i.quality}"
                for i in allowed_intervals
            )
        )
        return [
            f"{('-' if i.direction == 'down' else '')}{i.number}{i.quality}"
            for i in allowed_intervals
        ]

    def _count_unfolded_score_measures(self, mxl_path: Path) -> int:
        if isinstance(mxl_path, float) and math.isnan(mxl_path):
            return -1
        mxl_path = self.directory / mxl_path
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = pt.load_musicxml(mxl_path)
                part = score.parts[0]
                unfolded_part: pt.score.Part = pt.score.unfold_part_maximal(
                    part, ignore_leaps=False
                )
            n_measures = len(list(unfolded_part.measures))
            return n_measures
        except Exception as e:
            logger.error(f"Error reading {mxl_path}: {e}")
            return -1  # Indicate an error

    def _count_unfolded_midi_measures(self, midi_path: Path) -> int:
        if isinstance(midi_path, float) and math.isnan(midi_path):
            return -1
        path = self.directory / midi_path
        try:
            midi = pm.PrettyMIDI(path.as_posix())
            return len(midi.get_downbeats())
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return -1  # Indicate an error

    def _extract_time_signatures(self, mxl_path: Path) -> set[tuple[int, int]]:
        if isinstance(mxl_path, float) and math.isnan(mxl_path):
            return set()
        # Read XML
        xml = read_musicxml(self.directory / mxl_path)
        root = etree.fromstring(xml.encode("utf-8"))
        # Finding all <time> element
        time_elements = root.findall(".//time")
        time_signatures = set()
        for time in time_elements:
            try:
                time_signatures.add((int(time.findtext("beats")), int(time.findtext("beat-type"))))
            except ValueError as e:
                logger.error(f"Error reading TS in {mxl_path}: {e}")
                return set()
        return time_signatures


# if __name__ == '__main__':
#     ds = AsapDataset(
#         Path("~/Documents/datasets/asap-dataset").expanduser(),
#         split="all",
#         include_midi_scores=True,
#         filter_fn=lambda x: x.composer == "Bach" and x.midi_performance.stem != "midi_score",
#     )  # TODO: fix this to load the MIDI scores
#     print(f"Number of performances: {len(ds)}")
#     print(f"Number of scores: {len(set(entry.xml_score for entry in ds.entries))}")
#     print(ds[0].midi_performance)
#     print(ds[0].xml_score)

if __name__ == '__main__':
    ds = PdmxPianoDataset(
        Path("~/Documents/datasets/pdmx").expanduser(),
        split="train",
        # force_metadata_extraction=True,
    )
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
