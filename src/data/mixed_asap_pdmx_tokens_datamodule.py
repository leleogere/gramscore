import dataclasses
from collections.abc import Callable
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from torch.utils.data import DataLoader, WeightedRandomSampler
import multiprocessing as mp
from data.base_tokens_datamodule import BaseTokensDataModule
from data.components import transforms as t
from data.components.asap_dataset import AsapDataset
from data.components.pdmx_dataset import PdmxPianoDataset
from data.components.score_performance_dataset import ScorePerformanceEntry
from data.pdmx_tokens_datamodule import RatingOversamplerConfig
from models.automata.vectorization_torch import TokenTensor
import partitura as pt
import partitura.score

torch.serialization.add_safe_globals([TokenTensor])


class MixedAsapPdmxTokensDataModule(BaseTokensDataModule):
    """
    LightningDataModule mixing the ASAP and PDMX datasets, formatted as tokens.
    The validation is run only on ASAP dataset.
    """

    def __init__(
        self,
        asap_data_dir: Path,
        pdmx_data_dir: Path,
        proportion_asap: float = 0.5,
        # ASAP config
        asap_include_midi_scores: bool = False,
        asap_filter_fn: Callable[[ScorePerformanceEntry], bool] | None = None,
        asap_valid_tsv_file: Path | None = None,
        asap_validation_pieces_filename: str = "validation_pieces.txt",
        asap_test_pieces_filename: str = "test_pieces.txt",
        asap_test_validation_mode: str = "all_wo_overlap",
        asap_artificial_dataset_repetitions: int = 1,
        multiprocessing_manager: SyncManager | None = None,
        # PDMX config
        pdmx_rating_oversampler_config: RatingOversamplerConfig = None,
        pdmx_validation_pieces_filename: str = "validation_pieces.txt",  # Not used anyway
        pdmx_test_pieces_filename: str | None = None,  # Not used anyway
        pdmx_allowed_staffs: tuple[int, ...] = (1, 2),
        pdmx_test_validation_mode: str = "first_chunk",  # Not used anyway
        **kwargs,
    ) -> None:
        """Initialize a `PdmxTokensDataModule`

        :param data_dir: The data directory.
        :param asap_path: Optional path to the ASAP dataset (to exclude PDMX pieces that are also
             in ASAP validation and test sets).
        """
        super().__init__(**kwargs, _exclude_from_hparams_saving=["multiprocessing_manager"])
        self.splits = ["train", "validation"]

        self.multiprocessing_manager = (
            multiprocessing_manager if multiprocessing_manager is not None else mp.Manager()
        )
        self._scores_cache = self.multiprocessing_manager.dict()

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Can be called multiple times, avoid doing random things here

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if getattr(self, "trainer", None) is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})"
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Train set
        if stage in (None, "fit") and not self.data_train:
            self.data_train = torch.utils.data.ConcatDataset(
                datasets=[
                    PdmxPianoDataset(
                        directory=self.hparams.pdmx_data_dir,
                        split="train",
                        transforms=[
                            t.ToDict(),
                            t.ScoreReader(),
                            t.VoiceRestFiller(tolerant=True),
                            t.VoiceReorderer(),
                            t.PreComputedAllowedTranspositionsReader(limit_semitones=4),
                            t.PerformanceReader(),
                            t.QuantizedMusicChunker(num_performed_notes=500, mode="random"),
                            t.RandomTransposer(),
                            t.ScoreVectorizer(),
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
                            t.VectorizedPerformanceDurationJitterer(
                                min_factor=0.95, max_factor=1.05
                            ),
                            t.VectorizedPerformanceStreamsBucketizer(),
                        ],
                        asap_validation_pieces_path=self.hparams.asap_data_dir
                        / self.hparams.asap_validation_pieces_filename,
                        asap_test_pieces_path=self.hparams.asap_data_dir
                        / self.hparams.asap_test_pieces_filename,
                        validation_pieces_filename=self.hparams.pdmx_validation_pieces_filename,
                        test_pieces_filename=self.hparams.pdmx_test_pieces_filename,
                        allowed_staffs=self.hparams.pdmx_allowed_staffs,
                    ),
                    AsapDataset(
                        directory=self.hparams.asap_data_dir,
                        split="train",
                        include_midi_scores=self.hparams.asap_include_midi_scores,
                        transforms=[
                            t.ToDict(),
                            t.Cache(
                                transforms=[
                                    t.ScoreReader(),
                                    t.VoiceRestFiller(tolerant=True),
                                    t.VoiceReorderer(),
                                    t.AllowedTranspositionComputer(
                                        intervals=[
                                            pt.score.Interval(
                                                number=int(inter[0]),
                                                quality=inter[1:],
                                                direction=direction,
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
                                num_performed_notes=500,
                                annotations=self.hparams.asap_data_dir / "asap_annotations.json",
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
                            t.VectorizedPerformanceDurationJitterer(
                                min_factor=0.95, max_factor=1.05
                            ),
                            t.VectorizedPerformanceStreamsBucketizer(),
                            t.ScoreVectorizer(),
                        ],
                        filter_fn=self.hparams.asap_filter_fn,
                        valid_tsv_file=self.hparams.asap_valid_tsv_file,
                        test_pieces_filename=self.hparams.asap_test_pieces_filename,
                        validation_pieces_filename=self.hparams.asap_validation_pieces_filename,
                    ),
                ]
            )
        # Validation set
        if stage in (None, "fit", "validate") and not self.data_val:
            self.data_val = AsapDataset(
                directory=self.hparams.asap_data_dir,
                split="validation",
                include_midi_scores=self.hparams.asap_include_midi_scores,
                transforms=[
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
                        num_performed_notes=500,
                        annotations=self.hparams.asap_data_dir / "asap_annotations.json",
                        mode=(
                            0
                            if self.hparams.asap_test_validation_mode == "first_chunk"
                            else self.hparams.asap_test_validation_mode
                        ),
                    ),
                    t.ScoreVectorizer(),
                    t.PerformanceToPerformedNoteList(),
                    t.PerformedNoteListVectorizer(),
                    t.VectorizedPerformanceStreamsBucketizer(),
                ],
                filter_fn=self.hparams.asap_filter_fn,
                valid_tsv_file=self.hparams.asap_valid_tsv_file,
                test_pieces_filename=self.hparams.asap_test_pieces_filename,
                validation_pieces_filename=self.hparams.asap_validation_pieces_filename,
            )
            logger.info(
                f"List of validation pieces:\n"
                + "\n".join([e.xml_score_relative.as_posix() for e in self.data_val.entries])
            )

        # Test set
        if stage in (None, "test") and not self.data_test:
            self.data_test = AsapDataset(
                directory=self.hparams.asap_data_dir,
                split="test",
                include_midi_scores=self.hparams.asap_include_midi_scores,
                transforms=[
                    t.ToDict(),
                    t.ScoreReader(),
                    t.VoiceRestFiller(tolerant=True),
                    t.VoiceReorderer(),
                    t.PerformanceReader(),
                    t.AsapMusicChunker(
                        num_performed_notes=500,
                        annotations=self.hparams.asap_data_dir / "asap_annotations.json",
                        mode=(
                            0
                            if self.hparams.asap_test_validation_mode == "first_chunk"
                            else self.hparams.asap_test_validation_mode
                        ),
                    ),
                    t.ScoreVectorizer(),
                    t.PerformanceToPerformedNoteList(),
                    t.PerformedNoteListVectorizer(),
                    t.VectorizedPerformanceStreamsBucketizer(),
                ],
                filter_fn=self.hparams.asap_filter_fn,
                valid_tsv_file=self.hparams.asap_valid_tsv_file,
                test_pieces_filename=self.hparams.asap_test_pieces_filename,
                validation_pieces_filename=self.hparams.asap_validation_pieces_filename,
            )
            logger.info(
                f"List of test pieces:\n"
                + "\n".join([e.xml_score_relative.as_posix() for e in self.data_test.entries])
            )

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

    def train_dataloader(self) -> DataLoader:
        # Weight ASAP and PDMX (the concatenated dataset is PDMX then ASAP in this order)
        self.data_train: torch.utils.data.ConcatDataset
        pdmx, asap = self.data_train.datasets
        num_pdmx_pieces, num_asap_pieces = len(pdmx), len(asap)
        total_num_pieces = num_pdmx_pieces + num_asap_pieces
        logger.info(
            f"Train PDMX pieces: {num_pdmx_pieces}, train ASAP pieces: {num_asap_pieces}, "
            f"weighting them so that there is ~{self.hparams.proportion_asap:.2%} ASAP pieces"
        )
        weight_asap = self.hparams.proportion_asap / num_asap_pieces
        weight_pdmx = (1 - self.hparams.proportion_asap) / num_pdmx_pieces
        sample_weights = torch.ones(total_num_pieces)
        sample_weights[:num_pdmx_pieces] = weight_pdmx
        sample_weights[num_pdmx_pieces:] = weight_asap
        logger.info(
            f"ASAP/PDMX entries: {num_asap_pieces}/{num_pdmx_pieces} -> "
            f"weights: {weight_asap:.2E}/{weight_pdmx:.2E} -> "
            f"proba: {1/total_num_pieces * weight_asap:.2E}/{1/total_num_pieces * weight_pdmx:.2E}"
        )

        # Eventually oversample in PDMX
        if (ovs := self.hparams.pdmx_rating_oversampler_config) is not None:
            logger.info(f"Oversampling with high quality scores with {ovs}")
            ovs: RatingOversamplerConfig
            good_entries_mask = torch.tensor(
                [
                    e.extra["rating"] >= ovs.min_rating_to_consider_good
                    and e.extra["n_ratings"] >= ovs.min_n_ratings_to_consider_good
                    for e in self.data_train.datasets[0].entries
                ]
            )
            num_good_entries = good_entries_mask.count_nonzero()
            num_bad_entries = len(good_entries_mask) - num_good_entries
            weight_good = ovs.objective_percentage * self.hparams.proportion_asap / num_good_entries
            weight_bad = (
                (1 - ovs.objective_percentage) * self.hparams.proportion_asap / num_bad_entries
            )
            logger.info(
                f"Good/bad PDMX entries: {num_good_entries}/{num_bad_entries} -> "
                f"weights: {weight_good:.2E}/{weight_bad:.2E} -> "
                f"proba: {1/total_num_pieces * weight_good:.2E}/{1/total_num_pieces * weight_bad:.2E}"
            )
            sample_weights[:num_pdmx_pieces] = torch.where(
                good_entries_mask, weight_good, weight_bad
            )

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=False,  # Managed by sampler
            sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
        )


if __name__ == "__main__":
    import sys

    logger.remove()
    logger.add(sink=sys.stderr, level="DEBUG")

    data_module = MixedAsapPdmxTokensDataModule(
        pdmx_data_dir=Path("~/Documents/datasets/pdmx").expanduser(),
        asap_data_dir=Path("~/Documents/datasets/asap-dataset").expanduser(),
        proportion_asap=0.5,
        batch_size=4,
        num_workers=10,
        persistent_workers=True,
        pdmx_rating_oversampler_config=RatingOversamplerConfig(4.0, 3, 1 / 3),
    )
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.train_dataloader()
    # dataloader = data_module.val_dataloader()
    # dataloader.dataset.entries = dataloader.dataset.entries[:5]
    # size = len(dataloader)
    # print(f"{size = }")
    #
    # for i, batch in enumerate(dataloader, start=1):
    #     print(f"{i}/{size}")
    #     for entry in batch["entries"]:
    #         print(f" - {entry.midi_performance_relative}")
