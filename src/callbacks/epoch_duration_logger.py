import time

from lightning import Callback, Trainer, LightningModule
from loguru import logger


class EpochDurationLogger(Callback):
    """
    Callback to log the duration of each epoch during training.
    """

    def __init__(self):
        self.start_time = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.start_time is None:
            logger.warning("Epoch start time is not set. Cannot log duration.")
            return
        if trainer.global_rank == 0:
            duration = time.time() - self.start_time
            logger.info(f"Epoch {trainer.current_epoch} duration: {duration:.2f} seconds")
            pl_module.log("epoch_duration", duration, rank_zero_only=True)
