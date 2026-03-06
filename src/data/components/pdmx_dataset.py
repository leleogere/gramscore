import ast
import itertools
import json
import math
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal
from typing import TYPE_CHECKING

import joblib
import lxml
import pandas as pd
import partitura as pt
import pretty_midi as pm
from loguru import logger
from lxml import etree
from tqdm import tqdm

from data.components.asap_dataset import AsapDataset
from data.components.check_tokenizable import check_full
from data.components.score_performance_dataset import ScorePerformanceDataset, ScorePerformanceEntry
from models.automata.vectorization_torch import TokenTensor
from utils.musicxml import read_musicxml

if TYPE_CHECKING:
    import torch


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
        validation_pieces_filename: str = "validation_pieces.txt",
    ):
        self.directory = directory
        self.split = split
        assert split in ["all", "train", "validation"], f"Invalid split: {split}"

        # Extract additional metadata
        self.metadata_file = self.directory / "PDMX.csv"
        self.additional_metadata_file = self.directory / "PDMX_extended_metadata.csv"
        logger.info(f"Extracting additional metadata from {self.metadata_file}")
        old_additional_metadata = (
            pd.read_csv(self.additional_metadata_file)
            if self.additional_metadata_file.exists()
            else None
        )
        pdmx = self.extract_additional_metadata(
            asap_path=asap_path,
            prev=None if force_metadata_extraction else old_additional_metadata,
        )
        if any(
            c1 != c2
            for c1, c2 in itertools.zip_longest(pdmx.columns, old_additional_metadata.columns)
        ):
            logger.info(f"Saving additional metadata to {self.additional_metadata_file}")
            pdmx.to_csv(self.additional_metadata_file, index=False)
        else:
            logger.info(
                f"Additional metadata already extracted and saved to {self.additional_metadata_file}"
            )

        # Read validations pieces
        validation_pieces_path = self.directory / validation_pieces_filename
        if not validation_pieces_path:
            raise ValueError(f"Validation pieces file not found: {validation_pieces_path}")
        with validation_pieces_path.open("r") as f:
            self.validation_pieces = set("./" + l for l in f.read().splitlines())

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
            logger.info(
                "Previous extraction already present, "
                "simply adding eventually missing information to it."
            )
            pdmx = prev
        else:
            pdmx = pd.read_csv(metadata_path)
            # pdmx = pdmx.head(100)
            pdmx["usable"] = True

        # Extract the staves in each piece
        if ("n_parts" not in pdmx.columns) or ("n_staves" not in pdmx.columns):
            logger.info(f"Extracting staves and parts information")
            staves_and_parts = pdmx["mxl"].progress_apply(self._count_staves_and_parts)
            pdmx["n_staves"] = staves_and_parts.apply(lambda x: x[0] if x is not None else -1)
            pdmx["n_parts"] = staves_and_parts.apply(lambda x: x[1] if x is not None else -1)

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
            similarity_threshold = 0.8
            pdmx["asap_similarity"] = self._compute_asap_similarity(asap_path=asap_path)
            pdmx["similar_to_asap"] = pdmx["asap_similarity"] >= similarity_threshold
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
        df = pd.read_csv(
            metadata_path,
            # TODO: avoid using the list repr for export/import (unsafe with literal_eval)
            converters={
                "allowed_transpositions": lambda transp_list: (
                    ast.literal_eval(transp_list)
                    if isinstance(transp_list, str) and transp_list != ""
                    else []
                ),
                "time_signatures": lambda set_str: (
                    ast.literal_eval(set_str)
                    if isinstance(set_str, str) and set_str != "{}" and set_str != ""
                    else set()
                ),
                "n_staves": lambda list_staves: (
                    tuple(ast.literal_eval(list_staves))
                    if isinstance(list_staves, str) and list_staves != ""
                    else ()
                ),
            },
        )

        # Only keep usable scores
        df = df[df["usable"]]

        # Remove pieces too short
        # df = df[(df["n_notes_midi"] >= 10) & (df["n_notes_midi"] <= 500)]
        df = df[df["n_notes_midi"] >= 3]

        # Keep only single-part, one/two-staves pieces
        df = df[df["n_staves"].isin([(1,), (2,)])]

        # Remove pieces with mismatching number of measures between score and midi
        df = df[(df["n_measures_score"] == df["n_measures_midi"]) & (df["n_measures_score"] != -1)]

        # Remove pieces with unallowed time signatures
        df = df[
            (df["time_signatures"].apply(len) > 0)  # At least one time signature (empty = error)
            & (  # TS allowed to be vectorized
                df["time_signatures"].map(
                    lambda tss: all(ts in TokenTensor.TS_INDEXER for ts in tss)
                )
            )
        ]

        logger.info(f"Number of entries before split: {len(df)}")
        logger.info(f"Number of measures: {sum(df['n_measures_score'])}")

        # Create splits
        if self.split == "all":
            pass
        elif self.split == "validation":
            df = df[df["mxl"].isin(self.validation_pieces)]
        elif self.split == "train":
            df = df[~df["mxl"].isin(self.validation_pieces)]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        logger.info(f"Number of entries after split '{self.split}': {len(df)}")
        logger.info(f"Number of measures after split '{self.split}': {sum(df['n_measures_score'])}")

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
            for _, row in df.iterrows()
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

    def _count_staves_and_parts(self, path: Path | str) -> tuple[list[int] | None, int | None]:
        if isinstance(path, float) and math.isnan(path):
            return None, None
        path = self.directory / path
        xml = read_musicxml(path)
        try:
            root = etree.fromstring(xml.encode("utf-8"))
        except lxml.etree.XMLSyntaxError as e:
            logger.error(f"Error reading {path}: {e}")
            return None, None
        n_staves = []
        for part in root.findall(".//part"):
            n_staves.append(len(set(int(s.text) for s in part.findall(".//staff"))) or 1)
        n_parts = len(n_staves)
        return n_staves, n_parts

    def _count_midi_notes(self, path: Path) -> int:
        if isinstance(path, float) and math.isnan(path):
            return -1  # Path not available
        path = self.directory / path
        try:
            score = pm.PrettyMIDI(path)
            return sum(len(inst.notes) for inst in score.instruments)
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

    def _compute_asap_similarity(self, asap_path: Path) -> "torch.Tensor":
        from sentence_transformers import SentenceTransformer
        import re
        import numpy as np
        import torch

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
            create_descriptor(title=title, composer=composer) for composer, title in asap_entries
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
        return similarities.cpu()
