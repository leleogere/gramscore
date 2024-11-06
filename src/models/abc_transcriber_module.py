from collections.abc import Callable
from typing import Any

import hydra.utils
import lightning as L
import litgpt.utils
import torch
import transformers
from loguru import logger
from torch import nn

from data.components.chunking_utils import ChunkingConfig
from utils.abc_xml_conversion import conversion
from utils.score_rendering import render_score


class ABCTranscriberModule(L.LightningModule):
    def __init__(
        self,
        language_model: nn.Module,
        tokenizer: transformers.PreTrainedTokenizerBase,
        optimizer: Callable[[Any], torch.optim.Optimizer],
    ):
        super().__init__()

        self.save_hyperparameters(
            ignore=[
                "tokenizer",  # causes issues with hydra as already saved by datamodule
                "language_model",  # already saved at checkpoints
            ]
        )
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer

    def training_step(self, batch: dict[str, torch.Tensor]):
        """
        Forward pass of the model.

        :param batch: A dictionary containing the input data in the structure:
            {
                "input_ids": torch.Tensor (batch_size, seq_len),
                "attention_mask": torch.Tensor (batch_size, seq_len),
                "assistant_masks": torch.Tensor (batch_size, seq_len)
                "assistant_response_index": torch.Tensor (batch_size)
            }
        """
        input_ids = batch["input_ids"]
        assistant_masks = batch["assistant_masks"]
        logits = self.language_model(input_ids).logits  # (batch_size, seq_len, vocab_size)
        input_ids[torch.logical_not(assistant_masks)] = self.tokenizer.unk_token_id
        loss = litgpt.utils.chunked_cross_entropy(
            logits=logits[:, :-1],
            targets=input_ids[:, 1:],
            ignore_index=self.tokenizer.unk_token_id,  # self.tokenizer.unk_token_id,
        )  # Maybe try something smarter to avoid passing the whole sequence through the model?
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        input_ids = batch["input_ids"]
        assistant_masks = batch["assistant_masks"]
        outputs = self.language_model(input_ids)
        logits = outputs.logits
        output_ids = logits.argmax(dim=-1)
        input_ids[torch.logical_not(assistant_masks)] = self.tokenizer.unk_token_id
        loss = litgpt.utils.chunked_cross_entropy(
            logits=logits[:, :-1],
            targets=input_ids[:, 1:],
            ignore_index=self.tokenizer.unk_token_id,
        )
        self.log("val/loss", loss)
        # Decode back a few images to ABC, then render the score and log the image
        for i in range(min(5, input_ids.shape[0])):
            for inout, ids in [("input", input_ids), ("output", output_ids)]:
                abc = self.tokenizer.decode(ids[i])
                xml = conversion.abc2xml(abc)
                image = render_score(as_text=xml, input_type="xml")
                self.logger.experiment.add_image(
                    f"val/{inout}_{i}",
                    image,
                    global_step=self.global_step,
                )
        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        raise NotImplementedError

    def configure_optimizers(self):
        return self.optimizer(self.language_model.parameters())


if __name__ == '__main__':
    import os
    from data.nasap_abc_lm_datamodule import AsapABCLanguageModelDataModule
    from pathlib import Path
    import hydra
    from loguru import logger
    from omegaconf import OmegaConf

    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["OC_CAUSE"] = "1"

    with hydra.initialize(config_path="../../configs"):
        config = hydra.compose(config_name="tests")

    print(OmegaConf.to_yaml(config))

    data_module = AsapABCLanguageModelDataModule(
        data_dir=Path("./data/nasap"),
        # tokenizer="mistralai/Codestral-22B-v0.1",
        # tokenizer="HuggingFaceTB/SmolLM-135M-Instruct",
        tokenizer=config.model.language_model.base_lm.pretrained_model_name_or_path,
        chunking_config=ChunkingConfig(
            new_dataset_path=Path("/tmp/chunked_nasap"),
            num_measures_per_chunk=32,
            n_jobs=4,
            progress=True,
        ),
        batch_size=2,
    )
    data_module.prepare_data()
    data_module.setup()

    model = hydra.utils.instantiate(config=config.model)
    logger.info("Model instantiated")

    for batch in data_module.train_dataloader():
        logger.debug(f"{batch['input_ids'].shape = }")
        logger.debug("Feeding batch to the model...")
        res = model.training_step(batch)
        logger.debug(res)
        break
