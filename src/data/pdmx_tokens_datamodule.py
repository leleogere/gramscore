import itertools
import multiprocessing as mp
from collections.abc import Callable
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import partitura as pt
import partitura.score
import torch
from lightning import LightningDataModule
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from data.components import transforms as t
from data.components.score_performance_dataset import AsapDataset
from data.components.score_performance_dataset import PdmxPianoDataset
from data.components.score_performance_dataset import ScorePerformanceEntry
from models.automata.vectorization_torch import TokenTensor
from utils.constants import PADDING_VALUE
from utils.constants import SUCCESS_FILENAME

torch.serialization.add_safe_globals([TokenTensor])


class PdmxTokensDataModule(LightningDataModule):
    """
    LightningDataModule for the PDMX dataset, formatted as tokens.
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 64,
        num_workers: int = 4,
        prefetch_factor: int = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `AsapTokensDataModule`

        :param data_dir: The data directory.
        :param batch_size: The batch size.
        :param include_midi_scores: Whether to include the MIDI scores as performances.
        :param filter_fn: A function to filter the entries.
        :param valid_tsv_file: An eventual TSV file with a column "folder" and a column "valid"
            being "True" for included folders (and eventually False for excluded but not needed)
        :param num_workers: The number of workers to use for data loading.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore="multiprocessing_manager")

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        self.splits = ["train", "validation"]

        self._prepared = False

    @staticmethod
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
        # """
        # Download and split the dataset
        # """
        # # Download the dataset
        # if self._prepared:
        #     return
        # success_file = self.hparams.data_dir / SUCCESS_FILENAME
        # if not success_file.exists():
        #     # NOT Clone the MIDI2ScoreTransformer modified ASAP dataset (commit 8cba199e15931975542010a7ea2ff94a6fc9cbee)
        #     # Clone the (n)-ASAP dataset
        #     logger.info("Cloning the ASAP dataset")
        #     if not self.hparams.data_dir.exists():
        #         git.Repo.clone_from(
        #             url="https://github.com/leleogere/asap-dataset.git",
        #             to_path=self.hparams.data_dir,
        #             branch="develop_with_bwv874_bwv846_beeth17-2_merged",
        #         )
        #     # Create the success file
        #     success_file.touch()
        # else:
        #     logger.info("ASAP dataset already downloaded")
        #
        # self._prepared = True
        pass

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
        # First setup the shared dicts (shared accross workers)
        transforms = {
            "train": [
                t.ToDict(),
                t.ScoreReader(),
                t.VoiceRestFiller(),
                t.VoiceReorderer(),
                # t.Cache(
                #     transforms=[
                #         t.ScoreReader(),
                #         t.VoiceRestFiller(),
                #         t.VoiceReorderer(),
                #         # t.AllowedTranspositionComputer(
                #         #     intervals=[
                #         #         pt.score.Interval(
                #         #             number=int(inter[0]), quality=inter[1:], direction=direction
                #         #         )
                #         #         for inter in ["2M", "2m", "3M", "3m", "4P", "4A", "4d"]
                #         #         for direction in ["up", "down"]
                #         #     ]
                #         #     + [pt.score.Interval(1, "P", "up")]
                #         # ),
                #     ],
                #     cache_key=lambda sample: sample["entry"].xml_score,
                #     cache=self._scores_cache,
                # ),
                # t.Cache(
                #     transforms=[
                #         t.AllowedTranspositionComputer(
                #             intervals=[
                #                 pt.score.Interval(
                #                     number=int(inter[0]), quality=inter[1:], direction=direction
                #                 )
                #                 for inter in ["2M", "2m", "3M", "3m", "4P", "4A", "4d"]
                #                 for direction in ["up", "down"]
                #             ]
                #             + [pt.score.Interval(1, "P", "up")]
                #         ),
                #     ],
                #     cache_key=lambda sample: sample["entry"].xml_score,
                #     keys_to_cache=["allowed_transpositions"],
                #     cache=allowed_transpositions_cache,  # shared across workers
                # ),
                t.PerformanceReader(),
                # # Debugger(lambda s: print("DEBUG", s["entry"])),
                # t.MusicChunker(
                #     num_performed_notes=500,
                #     annotations=self.hparams.data_dir / "asap_annotations.json",
                #     mode="random",
                # ),
                # t.RandomTransposer(),
                t.PerformanceSpeedAugmenter(min_factor=0.8, max_factor=1.2),
                t.ScoreVectorizer(),
                t.PerformanceVectorizer(),
            ],
            "validation": [
                t.ToDict(),
                t.ScoreReader(),
                t.VoiceRestFiller(),
                t.VoiceReorderer(),
                # t.Cache(
                #     transforms=[
                #         t.ScoreReader(),
                #         t.VoiceRestFiller(),
                #         t.VoiceReorderer(),
                #     ],
                #     cache_key=lambda sample: sample["entry"].xml_score,
                #     cache=self._scores_cache,
                # ),
                t.PerformanceReader(),
                # t.MusicChunker(
                #     num_performed_notes=500,
                #     annotations=self.hparams.data_dir / "asap_annotations.json",
                #     mode="all_wo_overlap",
                # ),
                t.ScoreVectorizer(),
                t.PerformanceVectorizer(),
            ],
        }

        # Load the splits
        if not self.data_train and not self.data_val:
            sets = {
                split: PdmxPianoDataset(
                    directory=self.hparams.data_dir,
                    split=split,
                    transforms=transforms[split],
                    only_valid=True,
                    excluded_indices_paths=self.hparams.data_dir / "overlapping_asap.csv",
                )
                for split in self.splits
            }
            self.data_train = sets["train"]
            self.data_val = sets["validation"]

            # Limit to problematic batch
            # xml_paths = """
            # # mxl/9/5/QmR6ZvdCfBXecGxiyNTAyvCXuLDsEoAR4QFzcysX66vE2M.mxl
            # mxl/13/2/QmV3Cpt73R6ZDy8xvgPYnzhYfPwXMyE3BpUze4YsJjbQxP.mxl
            # """
            # xml_paths = [
            #     self.data_train.directory / p.strip()
            #     for p in xml_paths.split("\n")
            #     if not p.strip().startswith("#")
            # ]
            # self.data_train.entries = [
            #     e for e in self.data_train.entries if e.xml_score in xml_paths
            # ]
            # print(len(self.data_train.entries))

            # self.data_train.entries = self.data_train.entries[:1] + self.data_train.entries[50:51]
            # self.data_val.entries = self.data_val.entries[:1]

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
            num_workers=min(len(self.data_val.entries), self.hparams.num_workers),
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

    logger.remove()
    logger.add(sink=sys.stderr, level="DEBUG")

    data_module = PdmxTokensDataModule(
        data_dir=Path("~/Documents/datasets/pdmx").expanduser(),
        batch_size=4,
        num_workers=10,
        persistent_workers=True,
    )
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.train_dataloader()
    # dataloader = data_module.val_dataloader()
    # dataloader.dataset.entries = dataloader.dataset.entries[:5]
    size = len(dataloader)
    print(f"{size = }")

    for i, batch in enumerate(dataloader, start=1):
        print(f"{i}/{size}")
        for entry in batch["entries"]:
            print(f" - {entry.midi_performance_relative}")
