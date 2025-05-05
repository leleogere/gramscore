import time
from typing import Any

from lightning import Callback, Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger


class BatchDurationLogger(Callback):
    """
    Callback to log the duration of each epoch during training.
    """

    def __init__(self, log_every_n_steps: int = 1):
        self.log_every_n_steps = log_every_n_steps
        self.start_time = None

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.log_every_n_steps == 0:
            self.start_time = time.time()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.log_every_n_steps == self.log_every_n_steps - 1:
            if self.start_time is None:
                logger.warning("Epoch start time is not set. Cannot log duration.")
                return
            if trainer.global_rank == 0:
                duration = (time.time() - self.start_time) / self.log_every_n_steps
                logger.info(f"Average batch duration: {duration:.2f} seconds")
                pl_module.log(
                    "batch_duration",
                    duration,
                    rank_zero_only=True,
                    on_step=True,
                    on_epoch=False,
                )
