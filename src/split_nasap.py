from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from data.components.asap_dataset import AsapDataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def split(cfg: DictConfig) -> None:
    """Split ASAP dataset in chunks.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    """
    log.info(f"Instantiating dataset <{cfg.data._target_}>")
    ds: AsapDataset = hydra.utils.instantiate(cfg.data)

    ds.split_asap_dataset(
        new_dataset_path=cfg.new_dataset_path,
        num_measures_per_chunk=cfg.num_measures_per_chunk,
        overlap=cfg.overlap,
        skip_already_split=cfg.skip_already_split,
        n_jobs=cfg.n_jobs,
        progress=cfg.progress,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="split_nasap.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for splitting asap.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    split(cfg)


if __name__ == "__main__":
    main()
