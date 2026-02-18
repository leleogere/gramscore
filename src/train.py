import sys

import omegaconf
import torch.multiprocessing
from pathlib import Path
import os

import hydra
import lightning
from dotenv import load_dotenv
from fsspec import Callback
from lightning import LightningModule
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf

from models.tokens_transcriber_module import TokensTranscriberModule


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


@hydra.main(version_base="1.3", config_path="../configs")
def train(cfg: DictConfig) -> None:
    """Main entry point for training a transcriber model.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["OC_CAUSE"] = "1"

    # Redirect loguru logs to hydra output directory
    hydra_path = Path(cfg.paths.output_dir)
    log_level = cfg.get("log_level", "INFO").upper()
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add(hydra_path / "train.log", level=log_level)

    load_dotenv()

    lightning.seed_everything(10, workers=True)

    # Load WandB logger
    # If running in offline mode, use wandb-osh
    logger.info(f"Instantiating logger {cfg.logger._target_}")
    if os.environ.get("WANDB_MODE", "online") == "offline":
        communication_dir = Path(os.environ.get("WANDB_COMMUNICATION_DIR", "/tmp"))
        communication_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"WandB communication dir: {communication_dir}")
        training_logger = hydra.utils.instantiate(cfg.logger, offline=True)
        from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

        wandb_osh_callback = TriggerWandbSyncLightningCallback(communication_dir=communication_dir)
    else:
        training_logger = hydra.utils.instantiate(cfg.logger, offline=False)
        wandb_osh_callback = None
    training_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    logger.info(f"Logger {training_logger} ready!")

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    # exit()

    # Load main datamodule
    logger.info(f"Instantiating main datamodule {cfg.data._target_}")
    with omegaconf.open_dict(cfg.data):
        use_cache = cfg.data.pop("use_cache", False)
    datamodule_partial = hydra.utils.instantiate(cfg.data)
    if use_cache:
        logger.debug("Use a shared cache on the datamodule")
        datamodule = datamodule_partial(multiprocessing_manager=torch.multiprocessing.Manager())
    else:
        logger.debug("Do not use a shared cache on the datamodule")
        datamodule = datamodule_partial
    # logger.info(f"Preparing and setting up datamodule {datamodule}")
    # datamodule.prepare_data()
    # datamodule.setup()
    logger.info(f"Datamodule {datamodule} ready!")

    logger.info(f"Instantiating model {cfg.model._target_}")
    with omegaconf.open_dict(cfg.model):
        load_from_checkpoint = cfg.model.pop("load_from_checkpoint", "")
    if load_from_checkpoint != "":
        logger.info(f"Loading checkpoint from {load_from_checkpoint}")
        model_cls = hydra.utils.get_class(cfg.model._target_)
        # Also instanciate new optimizer and scheduler to avoid reusing the same as the checkpoint
        optimizer = hydra.utils.instantiate(cfg.model.optimizer)
        scheduler = hydra.utils.instantiate(cfg.model.scheduler) if cfg.model.scheduler else None
        model = model_cls.load_from_checkpoint(
            load_from_checkpoint, weights_only=False, optimizer=optimizer, scheduler=scheduler
        )
    else:
        model: TokensTranscriberModule = hydra.utils.instantiate(cfg.model)
    logger.info(f"Model {model} loaded!")

    logger.info(f"Instantiating callbacks")
    callbacks = instantiate_callbacks(cfg.callbacks)
    if wandb_osh_callback is not None:
        callbacks.append(wandb_osh_callback)
    logger.info(f"Callbacks {callbacks} ready!")

    logger.info(f"Instantiating trainer {cfg.trainer._target_}")
    trainer: lightning.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=training_logger
    )
    logger.info(f"Trainer {trainer} ready!")

    # Fit model
    trainer.fit(
        model,
        datamodule,
        ckpt_path=load_from_checkpoint if cfg.resume_training else None,
        weights_only=False,
    )

    # logger.info("Testing model")
    # trainer.test(model, datamodule)


if __name__ == "__main__":
    train()
