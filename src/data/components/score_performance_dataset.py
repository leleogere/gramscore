import ast
import csv
import itertools
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
import lxml
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

import data.components.transforms as t
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

    @property
    def debug_path(self) -> Path:
        if self.base_path is not None:
            return (
                self.midi_performance_relative
                if self.midi_performance is not None
                else self.xml_score_relative
            )
        else:
            return self.midi_performance if self.midi_performance is not None else self.xml_score


class ScorePerformanceDataset(Dataset):
    """Dataset for a dataset of pairs of MusicXML scores and MIDI performances."""

    def __init__(
        self,
        entries: list[ScorePerformanceEntry],
        transforms: Callable | list[Callable] = None,
        exception_on_transform_error: bool = False,
    ) -> None:
        """Initialize a dataset of pairs of MusicXML scores and MIDI performances.

        :param entries: A list of ScorePerformanceEntry objects.
        :param transforms: A transform or list of transforms to apply to the entries.
        :param exception_on_transform_error: Whether to raise an exception if an error occurs
            during a transform. If false, will early return the sample with the t.ERROR_KEY key
            set to the error message.
        """
        self.entries = entries
        self.transforms = transforms
        if self.transforms is not None and not isinstance(transforms, list):
            self.transforms = [self.transforms]
        self.exception_on_transform_error = exception_on_transform_error

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

    def apply_transforms(self, sample: Any, transforms: list[Callable]) -> Any:
        for i, transform in enumerate(transforms):
            if isinstance(sample, dict) and t.ERROR_KEY in sample:
                # We should probably test i>0 for completude but it should not happen
                # as the ToDict transform must not cause any error
                last_transform_name = transforms[i - 1].__class__.__name__
                msg = (
                    f"{sample['entry'].debug_path}: error encountered after applying transform "
                    f"{last_transform_name}: {sample[t.ERROR_KEY]}"
                )
                if self.exception_on_transform_error:
                    raise ValueError(msg)
                else:
                    if not isinstance(transform, t.Cache):  # avoid relogging errors
                        logger.error(msg)
                    return sample
            try:
                sample = transform(sample)
            except Exception as e:
                sample[t.ERROR_KEY] = str(e)
            if isinstance(sample, list):
                # If the transform returns a list, apply the transforms to each element
                sample = [self.apply_transforms(s, transforms[i + 1 :]) for s in sample]
                break
        return sample


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
