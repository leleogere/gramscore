import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf

from data.nasap_abc_lm_datamodule import AsapABCLanguageModelDataModule


def chunk(cfg: DictConfig) -> None:
    """
    Chunk (n)-ASAP dataset in chunks.

    :param cfg: A DictConfig configuration composed by Hydra.
    """
    # Print config
    print(OmegaConf.to_yaml(cfg))

    logger.info(f"Instantiating dataset <{cfg.data._target_}>")
    ds: AsapABCLanguageModelDataModule = hydra.utils.instantiate(cfg.data)

    ds.prepare_data()
    ds.setup()


@hydra.main(version_base="1.3", config_path="../configs", config_name="chunk_nasap.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for chunking (n)-ASAP.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    load_dotenv()
    chunk(cfg)


if __name__ == "__main__":
    main()
