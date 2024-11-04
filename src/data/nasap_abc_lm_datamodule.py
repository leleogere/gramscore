import copy
import urllib.request
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
import csv

import git
import torch
from lightning import LightningDataModule
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from data.components.chunking_utils import ChunkingConfig
from data.components.patched_tokenizer import PatchedTokenizer
from data.components.score_performance_dataset import AsapDataset
from data.components.score_performance_dataset import ScorePerformanceDataset
from data.components.score_performance_dataset import ScorePerformanceEntry
from data.components.score_performance_dataset import Modality
from utils.constants import SUCCESS_FILENAME


class AsapABCLanguageModelDataModule(LightningDataModule):
    """
    LightningDataModule for the (n)-ASAP dataset, formatted to be used with a LLM.
    """

    def __init__(
        self,
        data_dir: Path,
        tokenizer: str,
        chunking_config: ChunkingConfig | None = None,
        batch_size: int = 64,
    ) -> None:
        """Initialize a `AsapLMDataModule`

        :param data_dir: The data directory.
        :param tokenizer: Name or path of the tokenizer to use.
        :param chunking_config: The chunking configuration.
        :param batch_size: The batch size.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.tokenizer = PatchedTokenizer.from_pretrained(tokenizer)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        self.splits = ["train", "validation", "test"]

    def prepare_data(self) -> None:
        """
        Download and split the dataset
        """
        # Download the dataset
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

        # Chunk the dataset
        if self.hparams.chunking_config is not None:
            sets: dict[str, AsapDataset | ScorePerformanceDataset] = {
                split: AsapDataset(
                    self.hparams.data_dir,
                    split=split,
                    modality=Modality.TEXT,
                    text_tokenizer=self.tokenizer,
                )
                for split in self.splits
            }
            for split, split_set in sets.items():
                logger.info(f"Loaded {len(split_set)} {split} samples")
                logger.info(
                    f"Chunking {split} dataset into "
                    f"{self.hparams.chunking_config.num_measures_per_chunk}-measure chunks in "
                    f"{self.hparams.chunking_config.new_dataset_path}"
                )
                config: ChunkingConfig = copy.copy(self.hparams.chunking_config)
                config.new_dataset_path = config.new_dataset_path / split
                split_set.chunk_nasap_dataset(config)

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
            sets = {}
            if self.hparams.chunking_config is not None:
                for split in self.splits:
                    with open(
                        self.hparams.chunking_config.new_dataset_path / split / "metadata.csv", "r"
                    ) as f:
                        reader = csv.DictReader(f)
                        entries = [
                            ScorePerformanceEntry(**{k: v for k, v in row.items() if k != "path"})
                            for row in reader
                        ]
                    sets[split] = ScorePerformanceDataset(
                        entries,
                        modality=Modality.TEXT,
                        text_tokenizer=self.tokenizer,
                    )
            else:
                sets = {
                    split: AsapDataset(
                        self.hparams.data_dir,
                        split=split,
                        modality=Modality.TEXT,
                        text_tokenizer=self.tokenizer,
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
            collate_fn=self.data_train.collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=self.data_val.collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=self.data_test.collate_fn,
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
    data_module = AsapABCLanguageModelDataModule(
        data_dir=Path("./data/nasap"),
        tokenizer="mistralai/Codestral-22B-v0.1",
        chunking_config=ChunkingConfig(
            num_measures_per_chunk=4,
            new_dataset_path=Path("./data/nasap_chunked"),
            n_jobs=4,
            progress=True,
        ),
    )
    data_module.prepare_data()
    data_module.setup()
    for batch in data_module.train_dataloader():
        print(batch)
        break
