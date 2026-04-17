"""
Optimize loss weights using Optuna with Hydra integration.

This script expands grouped weight parameters from the Hydra config
into individual loss weights for all token attributes.

python scripts/optimize_loss_weights.py --config-name=common_config experiment=optimize_loss_weights --multirun
"""

import os
import sys
from functools import reduce
from pathlib import Path

import hydra
import lightning as L
import omegaconf
import torch
import wandb
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict

from models.tokens_transcriber_module import TokensTranscriberModule
from train import instantiate_callbacks


def expand_grouped_weights(cfg: DictConfig) -> DictConfig:
    """
    Expand grouped weight parameters into individual loss weights.

    Those parameters are defined in the config param_groups, composed of a list of mapping
    linking a main key (that should be defined in the params config) to subkeys that should
    share the same value.
    For example, in the following example, all pitch-related loss weights will be set to the same value:
    ```
    param_groups:
        - model.loss_weights.Note.note_class:
              - model.loss_weights.Note.note_octave
              - model.loss_weights.Note.note_accidental
    hydra:
        sweeper:
            params:
                model.loss_weights.Note.note_class: tag(log, interval(0.001, 10.0))
    ```
    """
    # Extract grouped weights from config
    groups = cfg.get("param_groups", [])

    if not groups:
        return cfg

    logger.info(f"Expanding shared weights: {groups}")

    # Expand grouped weights to individual parameters
    with open_dict(cfg):
        for group in groups:
            for main_key, linked_keys in group.items():
                # As keys are dot-separated strings, we need to split them to navigate the config
                main_subkeys = main_key.split(".")
                w = reduce(DictConfig.__getitem__, main_subkeys, cfg)
                for linked_key in linked_keys:
                    # To write to the linked keys, we need to get the parent dict to modify inplace
                    linked_subkeys = linked_key.split(".")
                    parent_dict = reduce(DictConfig.__getitem__, linked_subkeys[:-1], cfg)
                    parent_dict[linked_subkeys[-1]] = w

    logger.info("Config after expansion:")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    return cfg


def train_with_config(cfg: DictConfig) -> float:
    """
    Train a model with the given configuration and return a metric.

    Returns:
        float: metric to minimize
    """
    # Fix output dir to be in the sweep subdir
    with open_dict(cfg):
        cfg.paths.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        cfg.logger.save_dir = cfg.paths.output_dir

    # Setup logging
    hydra_path = Path(cfg.paths.output_dir)
    log_level = cfg.get("log_level", "INFO").upper()
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add(hydra_path / "train.log", level=log_level)

    load_dotenv()

    logger.info("=" * 80)
    logger.info("Starting Optuna trial")
    logger.info("=" * 80)

    # Expand grouped weights
    cfg = expand_grouped_weights(cfg)

    # Set seed
    L.seed_everything(10, workers=True)

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

    # Load model
    logger.info(f"Instantiating model {cfg.model._target_}")
    with omegaconf.open_dict(cfg.model):
        load_from_checkpoint = cfg.model.pop("load_from_checkpoint", "")
    if load_from_checkpoint != "":
        logger.info(f"Loading checkpoint from {load_from_checkpoint}")
        model_cls = hydra.utils.get_class(cfg.model._target_)
        # If resuming training, reload the whole state
        # Note: this does not allow overriding any parameters, the model will be exactly the same,
        # optimizer and scheduler states included
        if cfg.resume_training:
            model = model_cls.load_from_checkpoint(
                checkpoint_path=load_from_checkpoint, weights_only=False
            )
        # Else, reload the checkpoint, but override all parameters with the provided config
        # Note: this will allow to override any parameters, and also reset the optimizer and
        # scheduler states
        else:
            cfg.model._target_ += ".load_from_checkpoint"
            model: TokensTranscriberModule = hydra.utils.instantiate(
                cfg.model, checkpoint_path=load_from_checkpoint, weights_only=False
            )
    else:
        model: TokensTranscriberModule = hydra.utils.instantiate(cfg.model)
    logger.info(f"Model {model} loaded!")

    # Instantiate callbacks
    logger.info(f"Instantiating callbacks")
    callbacks = instantiate_callbacks(cfg.callbacks)
    if wandb_osh_callback is not None:
        callbacks.append(wandb_osh_callback)
    logger.info(f"Callbacks {callbacks} ready!")

    # Instantiate trainer
    logger.info(f"Instantiating trainer")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=training_logger,
    )
    logger.info(f"Trainer {trainer} ready!")

    # Train
    logger.info("Starting training")
    trainer.fit(model=model, datamodule=datamodule)

    # Validate
    model.force_full_evaluation_on_next_validation()
    trainer.validate(model=model, datamodule=datamodule)

    # Get validation metric
    # metric_name = "[V] Generation metrics/MUSTER/MeanER"
    if trainer.callback_metrics:
        optimized_metric = trainer.callback_metrics.get(cfg.optimized_metric, None)
        if optimized_metric is not None:
            optimized_metric = float(optimized_metric)
            logger.info(f"Trial completed with {cfg.optimized_metric}: {optimized_metric:.4f}")
            return optimized_metric

    logger.warning(f"Could not find {cfg.optimized_metric} in metrics, returning large value")
    return float('inf')


@hydra.main(version_base="1.3", config_path="../configs", config_name="common_config")
def main(cfg: DictConfig) -> float:
    """
    Main entry point for Optuna optimization with Hydra.

    This function is called by Hydra's Optuna sweeper for each trial.
    The sweeper manages the optimization loop and parameter suggestions.
    """
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["OC_CAUSE"] = "1"

    # Train and return metric
    try:
        optimized_metric = train_with_config(cfg)
    except Exception as e:
        wandb.finish(exit_code=1)
        raise e
    wandb.finish()

    return optimized_metric


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["OC_CAUSE"] = "1"
    main()
