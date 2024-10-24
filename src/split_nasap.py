from pathlib import Path

import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from data.components.asap_dataset import AsapDataset


def chunk(cfg: DictConfig) -> None:
    """
    Chunk (n)-ASAP dataset in chunks.

    :param cfg: A DictConfig configuration composed by Hydra.
    """
    logger.info(f"Instantiating dataset <{cfg.data._target_}>")
    ds: AsapDataset = hydra.utils.instantiate(cfg.data)

    ds.chunk_nasap_dataset(
        new_dataset_path=Path(cfg.new_dataset_path),
        num_measures_per_chunk=cfg.num_measures_per_chunk,
        overlap=cfg.overlap,
        min_matched_notes_ratio=cfg.min_matched_notes_ratio,
        skip_already_chunked=cfg.skip_already_chunked,
        n_jobs=cfg.n_jobs,
        progress=cfg.progress,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="chunk_nasap.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for chunking (n)-ASAP.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    load_dotenv()
    chunk(cfg)


if __name__ == "__main__":
    main()
