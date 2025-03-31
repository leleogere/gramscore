import copy
import random
import urllib.request
import warnings
from collections.abc import Callable
from pathlib import Path
import typing
from typing import Any
from typing import Dict
from typing import Optional
import csv

import git
import pandas as pd
import symusic
import torch
from lightning import LightningDataModule
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import partitura as pt
import partitura.score
import partitura.utils
import line_profiler

from data.components.chunking_utils import ChunkingConfig
from data.components.patched_tokenizer import PatchedTokenizer
from data.components.score_performance_dataset import AsapDataset
from data.components.score_performance_dataset import ScorePerformanceDataset
from data.components.score_performance_dataset import ScorePerformanceEntry
from data.music_chunking import clip_midi
from data.music_chunking import clip_score
from models.automata.automata import ConstrainedAutomata
from models.automata.io import partitura_to_token_sequence
from models.automata.tokens import fill_rests
from models.automata.vectorization_torch import TokenTensor
from models.automata.voice_reordering import reorder_voices
from utils.constants import SUCCESS_FILENAME


torch.serialization.add_safe_globals([TokenTensor])

MAX_PITCH = 108
MIN_PITCH = 21


class MusicReader:
    @line_profiler.profile
    def __call__(self, entry: ScorePerformanceEntry) -> dict[str, Any]:
        # Read score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = pt.load_musicxml(entry.xml_score, force_note_ids=True)
            assert len(score.parts) == 1
            part = pt.score.unfold_part_maximal(score.parts[0])

        # Read performance
        performance = symusic.Score.from_file(entry.midi_performance)
        assert len(performance.tracks) == 1

        return {
            "entry": entry,
            "score": part,
            "performance": performance,
        }


class RandomTransposer:
    def __init__(self, intervals: list[pt.score.Interval]):
        self.intervals = intervals

    @line_profiler.profile
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        # Get highest and lowest notes
        highest_note, lowest_note = MIN_PITCH, MAX_PITCH
        for note in sample["performance"].tracks[0].notes:
            if note.pitch > highest_note:
                highest_note = note.pitch
            if note.pitch < lowest_note:
                lowest_note = note.pitch
        highest_note = max(note.pitch for note in sample["performance"].tracks[0].notes)
        lowest_note = min(note.pitch for note in sample["performance"].tracks[0].notes)

        # Sample interval (make sure that the transposed score fits in the pitch range)
        allowed_intervals = [
            inter
            for inter in self.intervals
            if MAX_PITCH - highest_note >= inter.semitones >= MIN_PITCH - lowest_note
        ]
        if not allowed_intervals:
            # No transposition fits in the pitch range, do nothing
            return sample
        interval: pt.score.Interval = random.choice(
            [
                inter
                for inter in self.intervals
                if MAX_PITCH - highest_note >= inter.semitones >= MIN_PITCH - lowest_note
            ]
        )
        semitones = interval.semitones * (1 if interval.direction == "up" else -1)

        # Transpose score
        print("hello")
        # sample["score"] = pt.utils.music.transpose(sample["score"], interval)
        # import sys

        # sys.setrecursionlimit(1_000_000_000)
        sample["score"] = copy.deepcopy(sample["score"])
        for note in sample["score"].notes_tied:
            print(f"BEFORE {note.alter = }")
            pt.utils.music._transpose_note_inplace(note, interval)
            print(f"AFTER  {note.alter = }")

        # Transpose performance
        for note in sample["performance"].tracks[0].notes:
            note.pitch += semitones

        return sample


class VoiceReorderer:
    @line_profiler.profile
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample["score"] = reorder_voices(sample["score"])
        return sample


class VoiceRestFiller:
    @line_profiler.profile
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        fill_rests(sample["score"])
        return sample


class MusicChunker:
    def __init__(
        self,
        num_performed_notes: int,
        annotations: pd.DataFrame | Path,
        start_measure: int | None = None,
    ):
        self.num_performed_notes = num_performed_notes
        if isinstance(annotations, pd.DataFrame):
            self.annotations = annotations
        else:
            self.annotations = pd.read_json(annotations, orient="index")
        self.start_measure = start_measure

    @line_profiler.profile
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:

        # If not enough notes, return the original score and performance
        total_notes = len(sample["score"].notes)
        if total_notes <= self.num_performed_notes:
            return sample

        num_measures = len(sample["score"].measures)
        if self.start_measure is None:
            # Get a random measure between the first and the third to last
            start_measure = random.randint(0, num_measures - 4)
        else:
            start_measure = self.start_measure

        print(f"{start_measure = }")

        # Get the first num_performed_notes notes in the MIDI file starting from this measure
        # (and add notes until the end of the measure if not filled)
        chunks_file = (
            sample["entry"].xml_score.parent
            / f"{sample['entry'].midi_performance.stem}_chunks.json"
        )
        # print("BEFORE clip_midi")
        performance, (start_measure, end_measure) = clip_midi(
            midi=sample["performance"],
            chunks=pd.read_json(chunks_file),
            performance_annotations=self.annotations.loc[
                str(sample["entry"].midi_performance_relative)
            ],
            measure_range=(start_measure, None),
            num_notes=self.num_performed_notes,
            inplace=False,
            return_measure_range=True,
            check_num_measures=num_measures,
        )
        # print("AFTER clip_midi")

        # Get the corresponding notes in the score
        score = clip_score(
            score=sample["score"],
            measure_range=(start_measure, end_measure),
        )
        # print("AFTER clip_score")

        return {
            "score": score,
            "performance": performance,
            **{k: v for k, v in sample.items() if k not in ["score", "performance"]},
        }


class MusicVectorizer:
    @line_profiler.profile
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        # Vectorize score
        sample["score_tokens"] = partitura_to_token_sequence(sample["score"])
        sample["score_tensor"] = TokenTensor.from_tokens(sample["score_tokens"])

        # Vectorize performance
        sample["performance_tensor"] = torch.FloatTensor(
            sample["performance"].pianoroll(modes=["frame"], pitch_range=(MIN_PITCH, MAX_PITCH))
        )

        return sample


class Cache:
    def __init__(
        self,
        transforms: list[Callable[[Any], Any]],
        key: Callable[[Any], Any] = lambda x: x,
    ):
        self.transforms = transforms
        self.key = key
        self.cache = {}

    # @line_profiler.profile
    def __call__(self, sample: Any) -> Any:
        key = self.key(sample)
        if key in self.cache:
            sample = self.cache[key]
        else:
            for transform in self.transforms:
                sample = transform(sample)
            self.cache[key] = sample
        return sample


class Debugger:
    def __init__(
        self,
        func: Callable[[Any], None],
    ):
        self.func = func

    @line_profiler.profile
    def __call__(self, sample: Any) -> Any:
        self.func(sample)
        return sample


class AsapTokensDataModule(LightningDataModule):
    """
    LightningDataModule for the (n)-ASAP dataset, formatted to be used with a LLM.
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 64,
        filter_fn: Callable[[ScorePerformanceEntry], bool] = lambda x: True,
        num_workers: int = 4,
    ) -> None:
        """Initialize a `AsapLMDataModule`

        :param data_dir: The data directory.
        :param batch_size: The batch size.
        :param filter_fn: A function to filter the entries.
        :param num_workers: The number of workers to use for data loading.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        self.splits = ["train", "validation", "test"]

        self._prepared = False

    @staticmethod
    def collate_fn(
        batch: list[dict[str, Any]],
    ) -> tuple[TokenTensor, torch.Tensor]:
        """
        Collate function for the `DataLoader`.

        :param batch: The batch to collate.
        :return: The collated batch.
        """
        scores = [x["score_tensor"] for x in batch]
        performances = [x["performance_tensor"].transpose(0, 1) for x in batch]
        scores_collated: TokenTensor = pad_sequence(scores, batch_first=True)
        performances_collated: torch.Tensor = pad_sequence(performances, batch_first=True)
        return scores_collated, performances_collated

    def prepare_data(self) -> None:
        """
        Download and split the dataset
        """
        # Download the dataset
        if self._prepared:
            return
        success_file = self.hparams.data_dir / SUCCESS_FILENAME
        if not success_file.exists():
            # NOT Clone the MIDI2ScoreTransformer modified ASAP dataset (commit 8cba199e15931975542010a7ea2ff94a6fc9cbee)
            # Clone the (n)-ASAP dataset
            logger.info("Cloning the ASAP dataset")
            if not self.hparams.data_dir.exists():
                git.Repo.clone_from(
                    url="https://github.com/CPJKU/asap-dataset.git",
                    to_path=self.hparams.data_dir,
                )
            # Download ACPAS metadata_{R,S}.csv files
            for key in ["R", "S"]:
                logger.info(f"Downloading ACPAS metadata_{key}.csv")
                urllib.request.urlretrieve(
                    f"https://cheriell.github.io/research/ACPAS_dataset/metadata_{key}.csv",
                    self.hparams.data_dir / f"metadata_{key}.csv",
                )
            # Create the success file
            success_file.touch()
        else:
            logger.info("ASAP dataset already downloaded")

        self._prepared = True

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})"
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Setup transforms
        transforms = {
            "train": [
                Cache(
                    transforms=[
                        MusicReader(),
                        VoiceRestFiller(),
                        VoiceReorderer(),
                    ],
                    key=lambda entry: entry.midi_performance,
                ),
                # # Debugger(lambda s: print("DEBUG", s["entry"])),
                MusicChunker(
                    num_performed_notes=500,
                    annotations=self.hparams.data_dir / "asap_annotations.json",
                ),
                # RandomTransposer(
                #     intervals=[
                #         pt.score.Interval(
                #             number=int(inter[0]), quality=inter[1:], direction=direction
                #         )
                #         for inter in ["2M", "2m", "3M", "3m", "4P", "4A", "4d", "5P", "5A", "5d"]
                #         for direction in ["up", "down"]
                #     ]
                # ),
                MusicVectorizer(),
            ],
            "validation": [
                Cache(
                    transforms=[
                        MusicReader(),
                        VoiceRestFiller(),
                        VoiceReorderer(),
                    ],
                    key=lambda entry: entry.midi_performance,
                ),
                # TODO: better way to evaluate than simply taking the start of the piece
                MusicChunker(
                    num_performed_notes=500,
                    annotations=self.hparams.data_dir / "asap_annotations.json",
                    start_measure=0,
                ),
                MusicVectorizer(),
            ],
            "test": [
                MusicReader(),
                VoiceReorderer(),
                VoiceRestFiller(),
                MusicChunker(
                    num_performed_notes=500,
                    annotations=self.hparams.data_dir / "asap_annotations.json",
                    start_measure=0,
                ),
                MusicVectorizer(),
            ],
        }

        # Load the splits
        if not self.data_train and not self.data_val and not self.data_test:
            sets = {
                split: AsapDataset(
                    data_dir=self.hparams.data_dir,
                    split=split,
                    transforms=transforms[split],
                    filter_fn=self.hparams.filter_fn,
                )
                for split in self.splits
            }
            self.data_train = sets["train"]
            self.data_val = sets["validation"]
            self.data_test = sets["test"]

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    import sys
    import time
    from tqdm import tqdm

    logger.remove()
    logger.add(sink=sys.stderr, level="INFO")

    data_module = AsapTokensDataModule(
        data_dir=Path("~/Documents/datasets/asap-dataset").expanduser(),
        filter_fn=lambda x: x.composer == "Bach",
        batch_size=1,
        num_workers=1,
    )
    data_module.prepare_data()
    data_module.setup()
    # for performances, scores in data_module.train_dataloader():
    #     print(f"{performances.shape = }\t{scores.shape = }")
    # for sample in data_module.data_train:
    #     # print(sample)
    #     print(sample["entry"])
    for _ in range(1):
        start = time.time()
        for i, sample in enumerate(data_module.data_train, start=1):
            print(f"{i}/{len(data_module.data_train)}: {sample['entry'].midi_performance}")
            pass
        end = time.time()
        print(f"{end - start:.2f} s")
