from pathlib import Path
import os

import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf


def train(cfg: DictConfig) -> None:
    """
    Train a transcriber model.

    :param cfg: A DictConfig configuration composed by Hydra.
    """
    logger.info(f"Instantiating logger {cfg.logger._target_}")
    training_logger = hydra.utils.instantiate(cfg.logger)
    training_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    logger.info(f"Logger {training_logger} ready!")

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    logger.info(f"Instantiating datamodule {cfg.data._target_}")
    datamodule = hydra.utils.instantiate(cfg.data)
    logger.info(f"Preparing and setting up datamodule {datamodule}")
    datamodule.prepare_data()
    datamodule.setup()
    logger.info(f"Datamodule {datamodule} ready!")

    logger.info(f"Instantiating model {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"Model {model} loaded!")

    logger.info(f"Instantiating trainer {cfg.trainer._target_}")
    trainer = hydra.utils.instantiate(cfg.trainer, logger=training_logger)
    logger.info(f"Trainer {trainer} ready!")

    logger.info("Starting training")
    trainer.fit(model, datamodule)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for training a transcriber model.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["OC_CAUSE"] = "1"

    # Redirect loguru logs to hydra output directory
    hydra_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.add(hydra_path / "train.log")

    load_dotenv()
    train(cfg)


if __name__ == "__main__":
    main()
