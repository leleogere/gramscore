import copy
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
import torch
from lightning import LightningDataModule
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import partitura as pt
import partitura.score
import partitura.utils

from data.components.chunking_utils import ChunkingConfig
from data.components.patched_tokenizer import PatchedTokenizer
from data.components.score_performance_dataset import AsapDataset
from data.components.score_performance_dataset import ScorePerformanceDataset
from data.components.score_performance_dataset import ScorePerformanceEntry
from models.automata.automata import ConstrainedAutomata
from models.automata.io import partitura_to_token_sequence
from models.automata.tokens import fill_rests
from models.automata.vectorization_torch import TokenTensor
from utils.constants import SUCCESS_FILENAME


torch.serialization.add_safe_globals([TokenTensor])


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

    def entry_to_midi_and_score_tokens(
        self,
        entry: ScorePerformanceEntry,
        fill_voices_with_rests: bool = True,
        unfold_repeats: bool = True,
        verify_sequence: bool = True,
        cache_tensor: bool = True,
    ) -> tuple[torch.FloatTensor, TokenTensor]:
        """
        Convert a `ScorePerformanceEntry` to a `TokenTensor`.

        :param entry: The entry to convert.
        :param fill_voices_with_rests: Whether to fill incomplete voices with rests.
        :param unfold_repeats: Whether to unfold repeats.
        :param verify_sequence: Whether to verify the sequence with an automata.
        :param cache_tensor: Whether to cache the tensor.
        :return: The converted entry.
        """
        if cache_tensor:
            cache_path = (
                self.hparams.data_dir
                / "tokenized_dataset"
                / entry.xml_score.relative_to(self.hparams.data_dir).parent
            )
            score_cache_path = cache_path / f"{entry.xml_score.stem}.pt"
            performance_cache_path = cache_path / f"{entry.midi_performance.stem}.pt"
            if score_cache_path.exists() and performance_cache_path.exists():
                logger.debug(f"Loading cached tensors from {cache_path}")
                score_tensor = torch.load(score_cache_path, weights_only=True)
                performance_tensor = torch.load(performance_cache_path, weights_only=True)
                return performance_tensor, score_tensor

        # Read score and convert to TokenTensor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = pt.load_musicxml(entry.xml_score)
            assert len(score.parts) == 1
            part = score.parts[0]
            # TODO: reorder voices
            if unfold_repeats:
                part = pt.score.unfold_part_maximal(part)
        if fill_voices_with_rests:
            fill_rests(part)
        tokens = partitura_to_token_sequence(part)
        if verify_sequence:
            automata = ConstrainedAutomata(8)
            for token in tokens:
                automata.transition(token)
        score_tensor = TokenTensor.from_tokens(tokens)

        # Read performance and render MIDI as a tensor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            performance = pt.load_performance_midi(entry.midi_performance)
            assert len(performance.performedparts) == 1
            part = performance.performedparts[0]
            performance_tensor = (
                torch.FloatTensor(
                    partitura.utils.compute_pianoroll(part, piano_range=True).todense()
                )
                / 127.0
            )

        if cache_tensor:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(score_tensor, score_cache_path)
            torch.save(performance_tensor, performance_cache_path)
        return performance_tensor, score_tensor

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.FloatTensor, TokenTensor]]
    ) -> tuple[torch.FloatTensor, TokenTensor]:
        """
        Collate function for the `DataLoader`.

        :param batch: The batch to collate.
        :return: The collated batch.
        """
        performances = pad_sequence([x[0].transpose(0, 1) for x in batch], batch_first=True)
        scores = pad_sequence([x[1] for x in batch], batch_first=True)
        return performances, scores

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

        # Load the splits
        if not self.data_train and not self.data_val and not self.data_test:
            sets = {
                split: AsapDataset(
                    data_dir=self.hparams.data_dir,
                    split=split,
                    transform=self.entry_to_midi_and_score_tokens,
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

    logger.remove()
    logger.add(sink=sys.stderr, level="INFO")

    data_module = AsapTokensDataModule(
        data_dir=Path("~/Documents/datasets/asap-dataset").expanduser(),
        filter_fn=lambda x: x.composer == "Bach",
        batch_size=16,
        num_workers=8,
    )
    data_module.prepare_data()
    data_module.setup()
    for performances, scores in data_module.train_dataloader():
        print(f"{performances.shape = }\t{scores.shape = }")
