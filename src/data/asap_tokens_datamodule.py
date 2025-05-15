import copy
import itertools
import multiprocessing as mp
import random
import warnings
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
import sys

import git
import line_profiler
import pandas as pd
import partitura as pt
import partitura.score
import partitura.utils
import symusic
import torch
from lightning import LightningDataModule
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from data.components.better_dataloader import FastDataLoader
from data.components.score_performance_dataset import AsapDataset
from data.components.score_performance_dataset import ScorePerformanceEntry
from data.components.score_performance_dataset import TestAsapDataset
from data.components.score_performance_dataset import TrainAsapDataset
from data.components.score_performance_dataset import ValidationAsapDataset
from data.music_chunking import clip_midi
from data.music_chunking import clip_score
from data.music_chunking import read_chunk_file
from midi2scoretransformer.tokenizer import MultistreamTokenizer
from models.automata.io import partitura_to_token_sequence
from models.automata.tokens import fill_rests
from models.automata.vectorization_torch import TokenTensor
from models.automata.voice_reordering import reorder_voices
from utils.constants import PADDING_VALUE
from utils.constants import SUCCESS_FILENAME

torch.serialization.add_safe_globals([TokenTensor])

sys.setrecursionlimit(int(2**31 - 1))


class AsapTokensDataModule(LightningDataModule):
    """
    LightningDataModule for the ASAP dataset, formatted as tokens.
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 64,
        include_midi_scores: bool = False,
        filter_fn: Callable[[ScorePerformanceEntry], bool] = lambda x: True,
        num_workers: int = 4,
        prefetch_factor: int = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        artificial_dataset_repetitions: int = 1,
    ) -> None:
        """Initialize a `AsapTokensDataModule`

        :param data_dir: The data directory.
        :param batch_size: The batch size.
        :param include_midi_scores: Whether to include the MIDI scores as performances.
        :param filter_fn: A function to filter the entries.
        :param num_workers: The number of workers to use for data loading.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        # Hydra does not support lambda functions, so it is loaded as string
        # Make sure to only pass trusted code in here as it will be executed as is
        if isinstance(filter_fn, str):
            if not filter_fn.startswith("lambda"):
                raise ValueError("filter_fn must start with 'lambda'")
            self.filter_fn = eval(filter_fn)
            print(self.filter_fn)
        else:
            self.filter_fn = filter_fn

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        self.artificial_dataset_repetitions = artificial_dataset_repetitions

        self.splits = ["train", "validation", "test"]

        self._prepared = False

    @staticmethod
    @line_profiler.profile
    def collate_fn(
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Collate function for the `DataLoader`.

        :param batch: The batch to collate.
        :return: The collated batch.
        """
        # Pad and collate score tensors
        scores = [x["score_tensor"] for x in batch]  # each N_TOKENS * TOTAL_DIMENSIONS
        scores_collated: TokenTensor = pad_sequence(
            scores,
            batch_first=True,
            padding_value=PADDING_VALUE,
        )  # B * N_TOKENS_MAX * TOTAL_DIMENSIONS
        score_padding_mask = scores_collated[:, :, 0] == PADDING_VALUE  # B * N_TOKENS_MAX
        score_lengths = torch.LongTensor([s.size(0) for s in scores])

        # Pad and collate performance tensor streams
        performances = [x["performance_streams"] for x in batch]  # each N_NOTES * N_FEATURES_VALUE
        streams = ["onset", "duration", "pitch", "velocity"]
        # Make sure all stream have same batch and num_notes
        assert all(
            performance[streams[0]].shape[0] == performance[stream].shape[0]
            for performance in performances
            for stream in streams[1:]
        ), "All streams of a performance should have the same number of notes"
        performances_collated = {
            stream: pad_sequence(
                [x[stream] for x in performances],
                batch_first=True,
                padding_value=PADDING_VALUE,
            )
            for stream in streams
        }  # key: B * N_NOTES_MAX * N_FEATURES_VALUE
        performance_padding_mask = (
            performances_collated[streams[0]][:, :, 0] == PADDING_VALUE
        )  # B * N_NOTES_MAX
        performance_lengths = torch.LongTensor([p[streams[0]].size(0) for p in performances])
        # Security check to catch bug
        assert all(
            performances_collated[streams[0]].shape[0:2] == performances_collated[stream].shape[0:2]
            for stream in streams[1:]
        ), "All streams should have the same batch size and number of notes after padding"

        return {
            "entries": [x["entry"] for x in batch],
            "score_tensor": scores_collated,
            "score_padding_mask": score_padding_mask,
            "score_lengths": score_lengths,
            "performance_streams": performances_collated,
            "performance_padding_mask": performance_padding_mask,
            "performance_lengths": performance_lengths,
        }

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
                    url="https://github.com/leleogere/asap-dataset.git",
                    to_path=self.hparams.data_dir,
                    branch="develop_with_bwv874_bwv846_beeth17-2_merged",
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

        # Load the splits
        if not self.data_train and not self.data_val and not self.data_test:
            common_kwargs = dict(
                data_dir=self.hparams.data_dir,
                include_midi_scores=self.hparams.include_midi_scores,
                filter_fn=self.filter_fn,
            )
            self.data_train = TrainAsapDataset(
                **common_kwargs,
                batch_size=self.batch_size_per_device,
                shuffle=True,
            )
            self.data_train.entries = self.data_train.entries[:3]
            self.data_val = ValidationAsapDataset(
                **common_kwargs,
                batch_size=self.batch_size_per_device,
            )
            self.data_test = TestAsapDataset(
                **common_kwargs,
                batch_size=self.batch_size_per_device,
            )

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
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
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
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
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
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
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
    logger.add(sink=sys.stderr, level="DEBUG")

    data_module = AsapTokensDataModule(
        data_dir=Path("~/Documents/datasets/asap-dataset").expanduser(),
        include_midi_scores=True,
        filter_fn=lambda x: x.composer == "Bach" and x.midi_performance.stem == "midi_score",
        batch_size=4,
        num_workers=10,
        persistent_workers=True,
    )
    data_module.prepare_data()
    data_module.setup()
    size = len(data_module.train_dataloader())
    print(f"{size = }")

    for i, sample in enumerate(data_module.train_dataloader(), start=1):
        # for i, sample in enumerate(data_module.data_train, start=1):
        print(f"{i}/{size}")

    # print(f"{performances["onset"].shape = }\t{scores.shape = }")
    # print(
    #     f"score={scores.shape}\t"
    #     f"p_onset={performances['onset'].shape}\t"
    #     f"p_duration={performances['duration'].shape}\t"
    #     f"p_pitch={performances['pitch'].shape}\t"
    #     f"p_velocity={performances['velocity'].shape}"
    # )

    # for sample in data_module.data_train:
    #     # print(sample)
    #     print(sample["entry"])

    # diffs = []
    # for _ in range(10):
    #     print(_)
    #     start = time.time()
    #     # for i, sample in enumerate(data_module.train_dataloader(), start=1):
    #     for i, sample in enumerate(data_module.data_train, start=1):
    #         # AsapTokensDataModule.collate_fn([sample])
    #         # print(f"{i}/{size}")
    #         pass
    #     diff = time.time() - start
    #     diffs.append(diff)
    #     print(f"{diff:.3f}s  [{diff/len(data_module.data_train):.3f}s/sample]")
    #
    # # fmt: off
    # print(f"Avg:  {sum(diffs)/len(diffs):.3f}s  [{sum(diffs)/len(diffs)/len(data_module.data_train):.3f}s/sample]")
    # print(f"1st:  {diffs[0]:.3f}s  [{diffs[0]/len(data_module.data_train):.3f}s/sample]")
    # print(f"Avg1: {sum(diffs[1:])/len(diffs[1:]):.3f}s  [{sum(diffs[1:])/len(diffs[1:])/len(data_module.data_train):.3f}s/sample]")
    # # fmt: on

    # for _ in range(10):
    #     start = time.time()
    #     for i, sample in enumerate(data_module.data_train, start=1):
    #         # print(f"{i}/{len(data_module.data_train)}: {sample['entry'].midi_performance}")
    #         pass
    #     diff = time.time() - start
    #     print(f"{diff:.2f}s  [{diff/len(data_module.data_train):.2f}s/sample]")
