import hydra.utils
import lightning as L
import litgpt.utils
import torch
import transformers
from loguru import logger
from torch import nn


class ABCTranscriberModule(L.LightningModule):
    def __init__(self, language_model: nn.Module, tokenizer: transformers.PreTrainedTokenizerBase):
        super().__init__()
        self.language_model = language_model
        self.tokenizer = tokenizer

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
        input_ids[torch.logical_not(assistant_masks)] = 50256  # self.tokenizer.unk_token_id
        logger.debug(f"{logits.shape = }")
        loss = litgpt.utils.chunked_cross_entropy(
            logits=logits[:, :-1],
            targets=input_ids[:, 1:],
            ignore_index=50256,  # self.tokenizer.unk_token_id,
        )  # Maybe try something smarter to avoid passing the whole sequence through the model?
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        input_ids = batch["input_ids"]
        assistant_masks = batch["assistant_masks"]
        logits = self.language_model(input_ids).logits
        input_ids[torch.logical_not(assistant_masks)] = 50256
        loss = litgpt.utils.chunked_cross_entropy(
            logits=logits[:, :-1],
            targets=input_ids[:, 1:],
            ignore_index=50256,
        )
        self.log("val/loss", loss)
        # TODO: convert back to ABC, then MusicXML, then to image with verovio and log images
        return loss


if __name__ == '__main__':
    import os
    from data.asap_musicxml_abc_datamodule import AsapABCLanguageModelDataModule
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
        # chunking_path=Path("./data/nasap_chunked"),
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
