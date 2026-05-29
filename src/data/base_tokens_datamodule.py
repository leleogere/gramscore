from functools import partial
from typing import Any, Literal
from typing import Dict
from typing import Optional

import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import data.components.transforms as t
from models.automata.vectorization_torch import TokenTensor
from utils.constants import PADDING_VALUE

torch.serialization.add_safe_globals([TokenTensor])


class BaseTokensDataModule(LightningDataModule):
    """
    Base class for tokens-based datamodules.
    """

    def __init__(
        self,
        batch_size: int = 64,
        test_validation_mode: Literal[
            "first_chunk", "all_wo_overlap", "all_w_overlap", "full_piece"
        ] = "first_chunk",
        num_workers: int = 4,
        prefetch_factor: int = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        _exclude_from_hparams_saving: list[str] | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=_exclude_from_hparams_saving)

        self.batch_size_per_device = batch_size
        self.test_validation_mode = test_validation_mode

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.splits = ("train", "validation", "test")

    @staticmethod
    def collate_fn(
        batch: list[dict[str, Any] | None],
        keep_fields: tuple[str, ...] | Literal["all"] = ("entry",),
        skip_none: bool = False,
        skip_error_key: bool = True,
    ) -> dict[str, Any]:
        """
        Collate function for the `DataLoader`.

        :param batch: The batch to collate.
        :param drop_fields: Fields to drop from the batch.
        :param skip_none: Whether to skip None values in the batch, or raise an error.
        :param skip_error_key: Whether to skip samples with the error key set, or raise an error.
        :return: The collated batch.
        """
        if not isinstance(batch, list) or not all(isinstance(d, dict) for d in batch):
            raise ValueError(f"Batch must be a list of dictionaries, but is {type(batch)}")

        if skip_none:
            batch = [x for x in batch if x is not None]
        elif None in batch:
            raise ValueError(
                "None value found in batch. Fix it or pass skip_none=True to the collate_fn."
            )

        if skip_error_key:
            batch = [x for x in batch if not t.ERROR_KEY in x]
        elif t.ERROR_KEY in batch:
            raise ValueError(
                "Error key found in batch. Fix it or pass skip_error_key=True to the collate_fn."
            )

        results = {}
        fields = [key for key in batch[0]]
        for field in fields:

            # Pad and collate score tensors
            if field == "score_tensor":
                scores = [x["score_tensor"] for x in batch]  # each N_TOKENS * TOTAL_DIMENSIONS
                scores_collated: TokenTensor = pad_sequence(
                    scores,
                    batch_first=True,
                    padding_value=PADDING_VALUE,
                )  # B * N_TOKENS_MAX * TOTAL_DIMENSIONS
                score_padding_mask = (
                    scores_collated[:, :, 0].as_tensor() == PADDING_VALUE
                )  # B * N_TOKENS_MAX
                score_lengths = torch.LongTensor([s.size(0) for s in scores])
                # Store results
                results["score_tensor"] = scores_collated
                results["score_padding_mask"] = score_padding_mask
                results["score_length"] = score_lengths

            # Pad and collate performance tensor streams (same behavior for raw and bucketized)
            elif field in ("performance_streams_raw", "performance_streams"):
                stream_names = ["onset", "duration", "pitch", "velocity"]
                batched_streams = [
                    x[field] for x in batch
                ]  # each N_NOTES (* N_FEATURES_VALUE for bucketized)
                # Make sure all streams have same batch size and number of notes
                assert all(
                    streams[stream_names[0]].shape[0] == streams[stream_name].shape[0]
                    for streams in batched_streams
                    for stream_name in stream_names[1:]
                ), "All streams of a performance should have the same number of notes"
                performances_collated = {
                    stream_name: pad_sequence(
                        [streams[stream_name] for streams in batched_streams],
                        batch_first=True,
                        padding_value=PADDING_VALUE,
                    )
                    for stream_name in stream_names
                }  # key: B * N_NOTES_MAX (* N_FEATURES_VALUE for bucketized)
                # Security check to catch bug
                assert all(
                    performances_collated[stream_names[0]].shape[0:2]
                    == performances_collated[stream_name].shape[0:2]
                    for stream_name in stream_names[1:]
                ), "All streams should have the same batch size and number of notes after padding"
                # Store results
                results[field] = performances_collated
                # Add missing additional fields (only added the first time)
                if not "performance_padding_mask" in results:
                    slicer = (
                        (slice(None), slice(None))
                        if field == "performance_streams_raw"
                        else (slice(None), slice(None), 0)  # 1st element of features if bucketized
                    )
                    results["performance_padding_mask"] = (
                        performances_collated[stream_names[0]][slicer] == PADDING_VALUE
                    )
                if not "performance_length" in results:
                    results["performance_length"] = torch.LongTensor(
                        [streams[stream_names[0]].size(0) for streams in batched_streams]
                    )

            elif field == "performance_onset_times":
                performance_onset_times = pad_sequence(
                    [x["performance_onset_times"] for x in batch],
                    batch_first=True,
                    padding_value=PADDING_VALUE,
                )
                results["performance_onset_times"] = performance_onset_times

            # Simply collate as list other fields
            elif keep_fields == "all" or field in keep_fields:
                results[field] = [item[field] for item in batch]

        return results

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
            batch_size=(
                self.batch_size_per_device
                if self.test_validation_mode in ("first_chunk", "full_piece")
                else 1
            ),
            shuffle=False,
            collate_fn=partial(
                (
                    partial(self.collate_fn, keep_fields=("entry", "score_tokens"))
                    if self.test_validation_mode in ("first_chunk", "full_piece")
                    else lambda b: self.collate_fn(b[0], keep_fields=("entry", "score_tokens"))
                ),
            ),
            num_workers=min(len(self.data_val.entries), self.hparams.num_workers),
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
            batch_size=(
                self.batch_size_per_device
                if self.test_validation_mode in ("first_chunk", "full_piece")
                else 1
            ),
            shuffle=False,
            collate_fn=(
                partial(self.collate_fn, keep_fields="all")
                if self.test_validation_mode in ("first_chunk", "full_piece")
                else lambda b: self.collate_fn(b[0], keep_fields="all")
            ),
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
