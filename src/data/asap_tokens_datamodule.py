import itertools
import multiprocessing as mp
from collections.abc import Callable
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Any, Literal
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

from data.base_tokens_datamodule import BaseTokensDataModule
from data.components import transforms as t
from data.components.asap_dataset import AsapDataset
from data.components.score_performance_dataset import ScorePerformanceEntry
from models.automata.vectorization_torch import TokenTensor
from utils.constants import PADDING_VALUE
from utils.constants import SUCCESS_FILENAME

torch.serialization.add_safe_globals([TokenTensor])


class AsapTokensDataModule(BaseTokensDataModule):
    """
    LightningDataModule for the ASAP dataset, formatted as tokens.
    """

    def __init__(
        self,
        data_dir: Path,
        include_midi_scores: bool = False,
        filter_fn: Callable[[ScorePerformanceEntry], bool] | None = None,
        valid_tsv_file: Path | None = None,
        validation_pieces_filename: str = "validation_pieces.txt",
        test_pieces_filename: str = "test_pieces.txt",
        artificial_dataset_repetitions: int = 1,
        chunk_size: int = 500,
        multiprocessing_manager: SyncManager | None = None,
        test_validation_mode: Literal[
            "first_chunk", "all_w_overlap", "all_wo_overlap"
        ] = "all_wo_overlap",
        **kwargs,
    ) -> None:
        """Initialize a `AsapTokensDataModule`

        :param data_dir: The data directory.
        :param batch_size: The batch size.
        :param include_midi_scores: Whether to include the MIDI scores as performances.
        :param filter_fn: A function to filter the entries.
        :param valid_tsv_file: An eventual TSV file with a column "folder" and a column "valid"
            being "True" for included folders (and eventually False for excluded but not needed)
        :param validation_pieces_filename: The filename of the validation pieces, one per line, in
            the form "composer|piece_name" from the metadata CSV file.
        :param test_pieces_filename: The filename of the test pieces, same format as above.
        :param num_workers: The number of workers to use for data loading.
        :param prefetch_factor: The number of samples to prefetch per worker.
        :param persistent_workers: Whether to use persistent workers.
        :param pin_memory: Whether to pin memory.
        :param artificial_dataset_repetitions: The number of times to repeat the dataset to
            simulate a larger dataset (useful to make prefetch work on small datasets).
        :param multiprocessing_manager: The multiprocessing manager to use (to store cached
            data between processes)
        """
        super().__init__(
            test_validation_mode=test_validation_mode,
            _exclude_from_hparams_saving=["multiprocessing_manager"],
            **kwargs,
        )

        # Hydra does not support lambda functions, so it is loaded as string
        # Make sure to only pass trusted code in here as it will be executed as is
        if isinstance(filter_fn, str):
            if not filter_fn.startswith("lambda"):
                raise ValueError("filter_fn must start with 'lambda'")
            self.filter_fn = eval(filter_fn)
            # print(self.filter_fn)
        else:
            self.filter_fn = filter_fn

        self.artificial_dataset_repetitions = artificial_dataset_repetitions

        self._prepared = False

        self.multiprocessing_manager = (
            multiprocessing_manager if multiprocessing_manager is not None else mp.Manager()
        )
        self._scores_cache = self.multiprocessing_manager.dict()

    def prepare_data(self) -> None:
        """
        Download and split the dataset
        """
        # Download the dataset
        if self._prepared:
            return
        success_file = self.hparams.data_dir / SUCCESS_FILENAME
        if not success_file.exists():
            import git

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

        # Setup transforms
        # First setup the shared dicts (shared accross workers)
        transforms = {
            "train": [
                t.ToDict(),
                t.Cache(
                    transforms=[
                        t.ScoreReader(),
                        t.VoiceRestFiller(tolerant=True),
                        t.VoiceReorderer(),
                        t.AllowedTranspositionComputer(
                            intervals=[
                                pt.score.Interval(
                                    number=int(inter[0]), quality=inter[1:], direction=direction
                                )
                                for inter in ["2M", "2m", "3M", "3m", "4P", "4A", "4d"]
                                for direction in ["up", "down"]
                            ]
                            + [pt.score.Interval(1, "P", "up")]
                        ),
                    ],
                    cache_key=lambda sample: sample["entry"].xml_score,
                    cache=self._scores_cache,
                ),
                t.PerformanceReader(),
                # # Debugger(lambda s: print("DEBUG", s["entry"])),
                t.AsapMusicChunker(
                    num_performed_notes=self.hparams.chunk_size,
                    annotations=self.hparams.data_dir / "asap_annotations.json",
                    mode="random",
                ),
                t.RandomTransposer(),
                t.PerformanceToPerformedNoteList(),
                t.PerformedNoteListVectorizer(),
                t.VectorizedPerformanceVariableSpeedAugmenter(
                    min_global_factor=0.8,
                    max_global_factor=1.2,
                    min_relative_local_factor=-0.1,
                    max_relative_local_factor=0.1,
                    min_time_factor_change=40.0,
                    max_time_factor_change=60.0,
                ),
                t.VectorizedPerformanceInterOnsetJitterer(std=0.05),
                t.VectorizedPerformanceDurationJitterer(min_factor=0.95, max_factor=1.05),
                t.VectorizedPerformanceStreamsBucketizer(),
                t.ScoreVectorizer(),
            ],
            "validation": [
                t.ToDict(),
                t.Cache(
                    transforms=[
                        t.ScoreReader(),
                        t.VoiceRestFiller(tolerant=True),
                        t.VoiceReorderer(),
                    ],
                    cache_key=lambda sample: sample["entry"].xml_score,
                    cache=self._scores_cache,
                ),
                t.PerformanceReader(),
                t.AsapMusicChunker(
                    num_performed_notes=self.hparams.chunk_size,
                    annotations=self.hparams.data_dir / "asap_annotations.json",
                    mode=(
                        0
                        if self.test_validation_mode == "first_chunk"
                        else self.test_validation_mode
                    ),
                ),
                t.ScoreVectorizer(),
                t.PerformanceToPerformedNoteList(),
                t.PerformedNoteListVectorizer(),
                t.VectorizedPerformanceStreamsBucketizer(),
            ],
            "test": [
                t.ToDict(),
                t.ScoreReader(),
                t.PerformanceReader(),
                t.VoiceRestFiller(tolerant=True),
                t.VoiceReorderer(),
                t.AsapMusicChunker(
                    num_performed_notes=self.hparams.chunk_size,
                    annotations=self.hparams.data_dir / "asap_annotations.json",
                    mode=(
                        0
                        if self.test_validation_mode == "first_chunk"
                        else self.test_validation_mode
                    ),
                ),
                t.ScoreVectorizer(),
                t.PerformanceToPerformedNoteList(),
                t.PerformedNoteListVectorizer(),
                t.VectorizedPerformanceStreamsBucketizer(),
            ],
        }

        # Load the splits
        if not self.data_train and not self.data_val and not self.data_test:
            sets = {
                split: AsapDataset(
                    directory=self.hparams.data_dir,
                    split=split,
                    include_midi_scores=self.hparams.include_midi_scores,
                    transforms=transforms[split],
                    filter_fn=self.filter_fn,
                    valid_tsv_file=self.hparams.valid_tsv_file,
                    test_pieces_filename=self.hparams.test_pieces_filename,
                    validation_pieces_filename=self.hparams.validation_pieces_filename,
                )
                for split in self.splits
            }
            self.data_train = sets["train"]
            self.data_val = sets["validation"]
            self.data_test = sets["test"]

            # self.data_train.entries = self.data_train.entries[:1] + self.data_train.entries[50:51]
            # self.data_val.entries = self.data_val.entries[:1]

            # Repeat the dataset to simulate a larger dataset (works better for workers)
            self.data_train.entries = list(
                itertools.chain.from_iterable(
                    itertools.repeat(self.data_train.entries, self.artificial_dataset_repetitions)
                )
            )

    # def train_dataloader(self) -> DataLoader[Any]:
    #     """Create and return the train dataloader.
    #
    #     :return: The train dataloader.
    #     """
    #     return DataLoader(
    #         dataset=self.data_train,
    #         batch_size=self.batch_size_per_device,
    #         shuffle=True,
    #         collate_fn=self.collate_fn,
    #         num_workers=self.hparams.num_workers,
    #         prefetch_factor=self.hparams.prefetch_factor,
    #         persistent_workers=self.hparams.persistent_workers,
    #         pin_memory=self.hparams.pin_memory,
    #     )
    #
    # def val_dataloader(self) -> DataLoader[Any]:
    #     """Create and return the validation dataloader.
    #
    #     :return: The validation dataloader.
    #     """
    #     return DataLoader(
    #         dataset=self.data_val,
    #         batch_size=1,  # 1 batch = all chunks of 1 piece (splitting dealt with by MusicChunker)
    #         shuffle=False,
    #         # the ZeroDivisionError below should not happen, it's here only to catch an eventual bug
    #         collate_fn=lambda b: self.collate_fn(b[0]) if len(b) == 1 else 1 / 0,
    #         num_workers=min(len(self.data_val.entries), self.hparams.num_workers),
    #         prefetch_factor=self.hparams.prefetch_factor,
    #         persistent_workers=self.hparams.persistent_workers,
    #         pin_memory=self.hparams.pin_memory,
    #     )
    #
    # def test_dataloader(self) -> DataLoader[Any]:
    #     """Create and return the test dataloader.
    #
    #     :return: The test dataloader.
    #     """
    #     return DataLoader(
    #         dataset=self.data_test,
    #         batch_size=1,
    #         shuffle=False,
    #         # ZeroDivisionError: same as validation above
    #         collate_fn=lambda b: self.collate_fn(b[0]) if len(b) == 1 else 1 / 0,
    #         num_workers=min(len(self.data_test.entries), self.hparams.num_workers),
    #         prefetch_factor=self.hparams.prefetch_factor,
    #         persistent_workers=self.hparams.persistent_workers,
    #         pin_memory=self.hparams.pin_memory,
    #     )


if __name__ == "__main__":
    import sys
    import time
    from tqdm import tqdm

    logger.remove()
    logger.add(sink=sys.stderr, level="DEBUG")

    data_module = AsapTokensDataModule(
        data_dir=Path("~/Documents/datasets/asap-dataset").expanduser(),
        include_midi_scores=True,
        filter_fn=lambda x: x.composer == "Bach",  # and x.midi_performance.stem == "midi_score",
        batch_size=4,
        num_workers=10,
        persistent_workers=True,
        artificial_dataset_repetitions=10,
    )
    data_module.prepare_data()
    data_module.setup()
    # dataloader = data_module.train_dataloader()
    dataloader = data_module.val_dataloader()
    # dataloader.dataset.entries = dataloader.dataset.entries[:5]
    size = len(dataloader)
    print(f"{size = }")

    for i, batch in enumerate(dataloader, start=1):
        print(f"{i}/{size}")
        for entry, chunk_start, chunk_end in zip(
            batch["entries"], batch["chunk_starts"], batch["chunk_ends"]
        ):
            print(f" - {entry.midi_performance_relative} [{chunk_start}-{chunk_end}]")

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
