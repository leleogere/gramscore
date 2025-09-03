import sys
import torch.multiprocessing
from pathlib import Path
import os

import hydra
import lightning
from dotenv import load_dotenv
from fsspec import Callback
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        logger.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def train(cfg: DictConfig) -> None:
    """Main entry point for training a transcriber model.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["OC_CAUSE"] = "1"

    # Redirect loguru logs to hydra output directory
    hydra_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("log_level", "INFO").upper())
    logger.add(hydra_path / "train.log")

    load_dotenv()

    lightning.seed_everything(10, workers=True)

    # Load WandB logger
    logger.info(f"Instantiating logger {cfg.logger._target_}")
    training_logger = hydra.utils.instantiate(cfg.logger)
    training_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    logger.info(f"Logger {training_logger} ready!")

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Load pretraining datamodule
    logger.info(f"Instantiating pretraining datamodule {cfg.pretraining_data._target_}")
    pretraining_datamodule = hydra.utils.instantiate(cfg.pretraining_data)
    # datamodule.prepare_data()
    # datamodule.setup()
    logger.info(f"Datamodule {pretraining_datamodule} ready!")

    # Load main datamodule
    logger.info(f"Instantiating main datamodule {cfg.data._target_}")
    datamodule_partial = hydra.utils.instantiate(cfg.data)
    datamodule = datamodule_partial(multiprocessing_manager=torch.multiprocessing.Manager())
    # logger.info(f"Preparing and setting up datamodule {datamodule}")
    # datamodule.prepare_data()
    # datamodule.setup()
    logger.info(f"Main datamodule {datamodule} ready!")

    logger.info(f"Instantiating model {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"Model {model} loaded!")

    logger.info(f"Instantiating callbacks")
    callbacks = instantiate_callbacks(cfg.callbacks)
    logger.info(f"Callbacks {callbacks} ready!")

    logger.info(f"Instantiating pretrainer {cfg.pretrainer._target_}")
    pretrainer = hydra.utils.instantiate(
        cfg.pretrainer, callbacks=callbacks, logger=training_logger
    )
    logger.info(f"Pretrainer {pretrainer} ready!")

    logger.info(f"Instantiating main trainer {cfg.trainer._target_}")
    trainer: lightning.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=training_logger
    )
    logger.info(f"Main trainer {trainer} ready!")

    logger.info("Starting pretraining")
    pretrainer.fit(model, pretraining_datamodule)
    logger.info(f"Pretraining finished at step {pretrainer.global_step}!")

    training_logger.experiment.log({"phase": "finetune"})

    logger.info(f"Starting main training at step {pretrainer.global_step}!")
    trainer.fit(model, datamodule)
    logger.info("Main training finished!")

    # logger.info("Testing model")
    # trainer.test(model, datamodule)


if __name__ == "__main__":
    train()
