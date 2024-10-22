from pathlib import Path

import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from data.components.asap_dataset import AsapDataset


def split(cfg: DictConfig) -> None:
    """Split ASAP dataset in chunks.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    """
    logger.info(f"Instantiating dataset <{cfg.data._target_}>")
    ds: AsapDataset = hydra.utils.instantiate(cfg.data)

    ds.split_asap_dataset(
        new_dataset_path=Path(cfg.new_dataset_path),
        num_measures_per_chunk=cfg.num_measures_per_chunk,
        overlap=cfg.overlap,
        min_matched_notes_ratio=cfg.min_matched_notes_ratio,
        skip_already_split=cfg.skip_already_split,
        n_jobs=cfg.n_jobs,
        progress=cfg.progress,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="split_nasap.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for splitting asap.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    load_dotenv()
    split(cfg)


if __name__ == "__main__":
    main()
