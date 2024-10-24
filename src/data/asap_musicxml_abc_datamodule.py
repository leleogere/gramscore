import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import git
import torch
import transformers
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from loguru import logger

from data.components.asap_dataset import AsapDataset
from data.components.patched_tokenizer import PatchedTokenizer
from data.components.score_performance_dataset import ScorePerformanceDataset
from data.components.score_performance_dataset import ScorePerformanceEntry
from utils.constants import SUCCESS_FILENAME


class AsapABCLanguageModelDataModule(LightningDataModule):
    """`LightningDataModule` for the ASAP dataset, formatted to be used with a LLM.
    The dataset returns a dictionary with the following keys:
        - `midi_performance`: A text sequence of events in the MIDI performance.
        - `abc_score`: The corresponding ABC sequence.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```
    """

    def __init__(
        self,
        data_dir: Path,
        tokenizer: str,
        chunking_path: Path | None = None,
        chunking_num_measures: int = 32,
        chunking_overlap: bool = True,
        chunking_cache: bool = True,
        batch_size: int = 64,
    ) -> None:
        """Initialize a `AsapLMDataModule`

        :param data_dir: The data directory.
        :param tokenizer: Name or path of the tokenizer to use.
        :param chunking_path: The path to the chunked dataset. If `None`, the dataset won't be chunked.
        :param chunking_num_measures: The chunk size, in measures.
        :param chunking_overlap: Whether to overlap chunks with a sliding window.
        :param chunking_cache: Whether to use a previously cached chunked dataset.
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

    def prepare_data(self) -> None:
        """
        Download the datasets
        """
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

        # Load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            splits = ["train", "validation", "test"]
            sets: dict[str, AsapDataset | ScorePerformanceDataset] = {
                split: AsapDataset(self.hparams.data_dir, split=split) for split in splits
            }
            for split, split_set in sets.items():
                logger.info(f"Loaded {len(split_set)} {split} samples")
                if self.hparams.chunking_path is not None:
                    logger.info(
                        f"Chunking {split} dataset into {self.hparams.chunking_num_measures} "
                        f"measures in {self.hparams.chunking_path}"
                    )
                    sets[split] = split_set.split_nasap_dataset(
                        new_dataset_path=self.hparams.chunking_path / split,
                        num_measures_per_chunk=self.hparams.chunking_num_measures,
                        overlap=self.hparams.chunking_overlap,
                        min_matched_notes_ratio=0.0,
                        skip_already_split=self.hparams.chunking_cache,
                        progress=False,
                    )
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
        )

    def collate_fn(self, batch: list[ScorePerformanceEntry]) -> Dict[str, torch.Tensor]:
        """
        Collate function to convert a batch of ScorePerformanceEntry objects into a dictionary of
        tokenized conversations.

        :param batch: A batch of ScorePerformanceEntry objects.
        :return: A dictionary of tokenized conversations with structure:
            {
                "input_ids": tensor of shape (batch_size, max_length)
                "attention_mask": tensor of shape (batch_size, max_length)
                "assistant_masks": tensor of shape (batch_size, max_length) with 1's where the assistant tokens are
            }
        """
        text_midis = torch.vmap(AsapABCLanguageModelDataModule.performance_to_text)(batch)
        text_scores = torch.vmap(AsapABCLanguageModelDataModule.score_to_text)(batch)
        conversations = [
            [
                dict(role="user", content=text_midi),
                dict(role="assistant", content=text_score),
            ]
            for text_midi, text_score in zip(text_midis, text_scores)
        ]
        tokenized_conversations: dict[str, torch.Tensor] = self.tokenizer.apply_chat_template(
            conversations,
            return_assistant_tokens_mask=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        return tokenized_conversations

    @staticmethod
    def performance_to_text(entry: ScorePerformanceEntry) -> str:
        return "\n".join(
            [
                f"Note(pitch={note.pitch}, start={round(note.start, 3)}, duration={round(note.duration, 3)})"
                for note in entry.get_midi_performance_as_symusic(ttype="second").tracks[0].notes
            ]
        )

    @staticmethod
    def score_to_text(entry: ScorePerformanceEntry) -> str:
        return entry.get_score_as_abc()

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

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
        chunking_path=Path("./data/nasap_chunked"),
    )
    data_module.prepare_data()
    data_module.setup()
