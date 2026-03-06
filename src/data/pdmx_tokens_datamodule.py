import dataclasses
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.base_tokens_datamodule import BaseTokensDataModule
from data.components import transforms as t
from data.components.pdmx_dataset import PdmxPianoDataset
from models.automata.vectorization_torch import TokenTensor

torch.serialization.add_safe_globals([TokenTensor])


@dataclasses.dataclass
class RatingOversamplerConfig:
    min_rating_to_consider_good: float
    min_n_ratings_to_consider_good: int
    objective_percentage: float


class PdmxTokensDataModule(BaseTokensDataModule):
    """
    LightningDataModule for the PDMX dataset, formatted as tokens.
    """

    def __init__(
        self,
        data_dir: Path,
        asap_validation_pieces_path: Path | None = None,
        asap_test_pieces_path: Path | None = None,
        validation_pieces_filename: str = "validation_pieces.txt",
        rating_oversampler_config: RatingOversamplerConfig = None,
        **kwargs,
    ) -> None:
        """Initialize a `PdmxTokensDataModule`

        :param data_dir: The data directory.
        :param asap_path: Optional path to the ASAP dataset (to exclude PDMX pieces that are also
             in ASAP validation and test sets).
        """
        super().__init__(**kwargs)
        self.splits = ["train", "validation"]

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

        # Setup transforms
        # First setup the shared dicts (shared accross workers)
        transforms = {
            "train": [
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
                t.VectorizedPerformanceDurationJitterer(min_factor=0.95, max_factor=1.05),
                t.VectorizedPerformanceStreamsBucketizer(),
            ],
            "validation": [
                t.ToDict(),
                t.ScoreReader(),
                t.VoiceRestFiller(tolerant=True),
                t.VoiceReorderer(),
                t.PerformanceReader(),
                t.QuantizedMusicChunker(
                    num_performed_notes=500,
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

        if stage is None or stage == "fit":
            splits = self.splits
        elif stage == "validate":
            splits = ["validation"]
        else:
            raise ValueError(f"Unknown stage: {stage}")

        # Load the splits
        if not self.data_train or not self.data_val:  # TODO: fix this IF
            sets = {
                split: PdmxPianoDataset(
                    directory=self.hparams.data_dir,
                    split=split,
                    transforms=transforms[split],
                    asap_validation_pieces_path=self.hparams.asap_validation_pieces_path,
                    asap_test_pieces_path=self.hparams.asap_test_pieces_path,
                    validation_pieces_filename=self.hparams.validation_pieces_filename,
                )
                for split in splits
            }
            self.data_train = sets.get("train", None)
            self.data_val = sets.get("validation", None)
            self.data_test = sets.get("test", None)

            if stage == "fit" or stage == "validate":
                logger.info(
                    f"List of validation pieces:\n"
                    + "\n".join([e.xml_score_relative.as_posix() for e in self.data_val.entries])
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
        if (ovs := self.hparams.rating_oversampler_config) is not None:
            logger.info(f"Oversampling with high quality scores with {ovs}")
            ovs: RatingOversamplerConfig
            good_entries_mask = torch.tensor(
                [
                    e.extra["rating"] >= ovs.min_rating_to_consider_good
                    and e.extra["n_ratings"] >= ovs.min_n_ratings_to_consider_good
                    for e in self.data_train.entries
                ]
            )
            num_good_entries = good_entries_mask.count_nonzero()
            num_bad_entries = len(good_entries_mask) - num_good_entries
            weight_good = ovs.objective_percentage / num_good_entries
            weight_bad = (1 - ovs.objective_percentage) / num_bad_entries
            logger.info(
                f"Good/bad entries: {num_good_entries}/{num_bad_entries} "
                f"-> weights: {weight_good:.4f}/{weight_bad:.4f}"
            )
            sample_weights = torch.where(good_entries_mask, weight_good, weight_bad)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=shuffle,
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
