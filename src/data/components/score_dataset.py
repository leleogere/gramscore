from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

from torch.utils.data import Dataset


@dataclass
class AsapEntry:
    composer: str
    title: str
    folder: Path
    xml_score: Path
    midi_score: Path
    midi_performance: Path
    midi_score_annotations: Path


class AsapDataset(Dataset):
    """Dataset for the ASAP dataset.

    ASAP is a dataset of aligned musical scores (both MIDI and MusicXML) and performances (audio and MIDI), all with downbeat, beat, time
    signature, and key signature annotations.

    This dataset is used to load only the MIDI and MusicXML files from the dataset.
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize an `AsapDataset`.

        :param data_dir: The directory where the dataset is stored.
        """
        self.data_dir = data_dir

        # Read metadata.csv
        with open(data_dir / "metadata.csv", "r") as f:
            reader = csv.DictReader(f)
            self.metadata = list(reader)

        # Create entries
        self.entries = [
            AsapEntry(
                composer=entry["composer"],
                title=entry["title"],
                folder=self.data_dir / Path(entry["folder"]),
                xml_score=self.data_dir / Path(entry["xml_score"]),
                midi_score=self.data_dir / Path(entry["midi_score"]),
                midi_performance=self.data_dir / Path(entry["midi_performance"]),
                midi_score_annotations=self.data_dir / Path(entry["midi_score_annotations"]),
            )
            for entry in self.metadata
        ]

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        :return: The number of samples in the dataset.
        """
        return len(self.entries)

    def __getitem__(self, idx: int) -> AsapEntry:
        """Return a sample from the dataset.

        :param idx: The index of the sample to return.
        :return: The sample at the given index.
        """
        return self.entries[idx]
