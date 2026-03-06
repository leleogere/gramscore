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
        directory: Path,
        split: Literal["train", "validation", "test", "all"],
        include_midi_scores: bool = False,
        filter_fn: Callable[[ScorePerformanceEntry], bool] | None = None,
        valid_tsv_file: Path | None = None,
        transforms: Callable | list[Callable] = None,
    ) -> None:
        """Initialize an `AsapDataset` (using the MIDI2ScoreTransformer method).

        :param directory: The directory where the ASAP dataset is stored.
        :param split: The split to use (either "all", "train", "validation", or "test").
        :param include_midi_scores: Whether to include the quantified MIDI score as performances in
            the dataset.
        :param filter_fn: A function to filter the entries.
        :param valid_tsv_file: A TSV file containing the list of valid folders.
        :param transforms: A transform to apply to the entries.
        """
        self.directory = directory
        self.split = split
        assert split in ["all", "train", "validation", "test"], f"Invalid split: {split}"
        # entries = self.read_metadata_like_m2st(data_dir, split)
        entries = self.read_metadata(directory, split, include_midi_scores=include_midi_scores)
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
