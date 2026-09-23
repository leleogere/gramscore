import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd

from data.components.score_performance_dataset import ScorePerformanceDataset, ScorePerformanceEntry
from utils.constants import TEST_PIECE_IDS


class AsapDataset(ScorePerformanceDataset):
    """Dataset for the ASAP dataset.

    ASAP is a dataset of aligned musical scores (both MIDI and MusicXML) and performances (audio and MIDI), all with downbeat, beat, time
    signature, and key signature annotations.

    This dataset is used to load only the MIDI and MusicXML files from the dataset.
    """

    def __init__(
        self,
        directory: Path,
        split: Literal["train", "validation", "test", "all"],
        include_midi_scores: bool = False,
        filter_fn: Callable[[ScorePerformanceEntry], bool] | None = None,
        valid_tsv_file: Path | None = None,
        transforms: Callable | list[Callable] = None,
        test_pieces_filename: str | Path = "test_pieces.txt",
        validation_pieces_filename: str | Path = "validation_pieces.txt",
    ) -> None:
        """Initialize an `AsapDataset` (using the MIDI2ScoreTransformer method).

        :param directory: The directory where the ASAP dataset is stored.
        :param split: The split to use (either "all", "train", "validation", or "test").
        :param include_midi_scores: Whether to include the quantified MIDI score as performances in
            the dataset.
        :param filter_fn: A function to filter the entries.
        :param valid_tsv_file: A TSV file containing the list of valid folders.
        :param transforms: A transform to apply to the entries.
        :param test_pieces_filename: The path to a list of test pieces (relative to self.directory
            if a simple string), with a piece by line in the form `composer|title`. Each piece
            can contain several performances.
        :param validation_pieces_filename: Same as test_pieces_filename, but for validation pieces.
        """
        self.directory = directory
        self.split = split
        assert split in ["all", "train", "validation", "test"], f"Invalid split: {split}"

        # Read validation and test files
        validation_pieces_path = (
            validation_pieces_filename
            if isinstance(validation_pieces_filename, Path)
            else Path(self.directory, validation_pieces_filename)
        )
        test_pieces_path = (
            test_pieces_filename
            if isinstance(test_pieces_filename, Path)
            else Path(self.directory, test_pieces_filename)
        )
        self.validation_pieces = set(
            tuple(line.split("|")) for line in validation_pieces_path.read_text().splitlines()
        )
        self.test_pieces = set(
            tuple(line.split("|")) for line in test_pieces_path.read_text().splitlines()
        )

        # Read entries
        # entries = self.read_metadata_like_m2st(data_dir, split)
        entries = self.read_metadata(directory, split, include_midi_scores=include_midi_scores)

        # Filter entries
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

    def read_metadata(
        self,
        directory: Path,
        split: str,
        include_midi_scores: bool = False,
    ) -> list[ScorePerformanceEntry]:
        """Read the metadata of the dataset.


        :param data_dir: The directory where the dataset is stored.
        :param split: The split to use (either "all", "train", "validation", or "test").
        :return: The metadata of the dataset.
        """
        # Read entries from metadata
        with open(directory / "metadata.csv", "r") as f:
            reader = list(csv.DictReader(f))
        with open(directory / "asap_annotations.json", "r") as f:
            annotations = json.load(f)
        entries = [
            ScorePerformanceEntry(
                # fmt: off
                composer=entry["composer"],
                title=entry["title"],
                xml_score=directory / entry["xml_score"],
                midi_score=directory / entry["midi_score"] if "midi_score" in entry else None,
                midi_performance=directory / entry["midi_performance"],
                midi_score_annotations=directory / entry["midi_score_annotations"] if "midi_score_annotations" in entry else None,
                folder=directory / entry["folder"],
                base_path=directory,
                # fmt: on
            )
            for entry in reader
        ]

        def is_aligned(entry: ScorePerformanceEntry) -> bool:
            return annotations[entry.midi_performance_relative.as_posix()][
                "score_and_performance_aligned"
            ]

        entries = list(filter(is_aligned, entries))

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
                        base_path=directory,
                    )
                )

        if split == "all":
            pass

        elif split == "train":
            entries = [
                entry
                for entry in entries
                if entry.id not in self.test_pieces and entry.id not in self.validation_pieces
            ]

        elif split == "validation":
            entries = [
                entry
                for entry in entries
                if entry.id not in self.test_pieces and entry.id in self.validation_pieces
            ]

        elif split == "test":
            entries = [entry for entry in entries if entry.id in self.test_pieces]

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
        from utils.constants import SKIP
        from utils.constants import TO_IGNORE_INDICES

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
    for split in ["train", "validation", "test", "all"]:
        print(f"Split: {split}")
        ds = AsapDataset(
            directory=Path("~/Documents/datasets/asap-dataset").expanduser(),
            split=split,
            include_midi_scores=False,
            valid_tsv_file=Path("~/Documents/datasets/asap-dataset/valid.tsv").expanduser(),
        )
        print(f"  - Number of performances: {len(ds)}")
        print(f"  - Number of scores: {len(set(e.id for e in ds.entries))}")
