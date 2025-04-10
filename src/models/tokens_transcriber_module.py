from collections.abc import Callable
from typing import Any

import hydra.utils
import lightning as L
import litgpt.utils
import torch
import transformers
import wandb
from einops import rearrange
from loguru import logger
from torch import nn

from data.asap_tokens_datamodule import AsapTokensDataModule
from models.automata.transcriber_model import TokensTranscriberModel
from data.components.chunking_utils import ChunkingConfig
from models.automata.vectorization_torch import TokenTensor
from utils.abc_xml_conversion import conversion
from utils.score_rendering import render_score


class TokensTranscriberModule(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable[[Any], torch.optim.Optimizer],
    ):
        super().__init__()

        self.save_hyperparameters()
        self.transcriber_model = TokensTranscriberModel()
        self.optimizer = optimizer
        self.loss = nn.NLLLoss()

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        """
        Forward pass of the model.

        :param batch: A dictionary containing the input data in the structure:
            {
                "score_tensor": torch.Tensor,  # input tensor representing the score
                "performance_streams": dict[str, torch.Tensor],  # dict of performance streams
            }
        """
        performance_streams: dict[str, torch.Tensor] = batch["performance_streams"]
        score_tensor: TokenTensor = batch["score_tensor"]

        router_logits = self.transcriber_model.forward(
            src_streams=performance_streams,
            tgt_tokens=score_tensor,
        )  # (batch_size, num_tokens, num_token_types)

        true_token_types = score_tensor["token_type"]  # (batch_size, num_tokens, num_token_types)

        flattened_true_token_types = rearrange(true_token_types, "b t k -> (b t) k")
        flattened_router_logits = rearrange(router_logits, "b t k -> (b t) k")

        pad_mask = flattened_true_token_types[:, 0] != -100
        # Would be better if we could get the mask directly from the pad_sequence function

        loss = self.loss(
            input=flattened_router_logits[pad_mask],
            # Could be more efficient if NLLLoss would accept OHE vectors as targets...
            target=flattened_true_token_types[pad_mask].argmax(dim=-1),
        )

        self.log("train/loss", loss)
        return loss

        # input_ids = batch["input_ids"]
        # assistant_masks = batch["assistant_masks"]
        # logits = self.language_model(input_ids).logits  # (batch_size, seq_len, vocab_size)
        # input_ids[torch.logical_not(assistant_masks)] = self.tokenizer.unk_token_id
        # loss = litgpt.utils.chunked_cross_entropy(
        #     logits=logits[:, :-1],
        #     targets=input_ids[:, 1:],
        #     ignore_index=self.tokenizer.unk_token_id,  # self.tokenizer.unk_token_id,
        # )  # Maybe try something smarter to avoid passing the whole sequence through the model?
        # self.log("train/loss", loss)
        # return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        performance_streams: dict[str, torch.Tensor] = batch["performance_streams"]
        score_tensor: TokenTensor = batch["score_tensor"]

        router_logits = self.transcriber_model.forward(
            src_streams=performance_streams,
            tgt_tokens=score_tensor,
        )  # (batch_size, num_tokens, num_token_types)

        true_token_types = score_tensor["token_type"]  # (batch_size, num_tokens, num_token_types)

        flattened_true_token_types = rearrange(true_token_types, "b t k -> (b t) k")
        flattened_router_logits = rearrange(router_logits, "b t k -> (b t) k")

        pad_mask = flattened_true_token_types[:, 0] != -100
        # Would be better if we could get the mask directly from the pad_sequence function

        loss = self.loss(
            input=flattened_router_logits[pad_mask],
            # Could be more efficient if NLLLoss would accept OHE vectors as targets...
            target=flattened_true_token_types[pad_mask].argmax(dim=-1),
        )

        self.log("val/loss", loss)
        return loss

        # input_ids = batch["input_ids"]
        # assistant_masks = batch["assistant_masks"]
        # outputs = self.language_model(input_ids)
        # logits = outputs.logits
        # output_ids = logits.argmax(dim=-1)
        # input_ids[torch.logical_not(assistant_masks)] = self.tokenizer.unk_token_id
        # loss = litgpt.utils.chunked_cross_entropy(
        #     logits=logits[:, :-1],
        #     targets=input_ids[:, 1:],
        #     ignore_index=self.tokenizer.unk_token_id,
        # )
        # self.log("val/loss", loss)
        # # Decode back a few images to ABC, then render the score and log the image
        # for i in range(min(5, input_ids.shape[0])):
        #     for inout, ids in [("input", input_ids), ("output", output_ids)]:
        #         abc = self.tokenizer.decode(ids[i])
        #         xml = conversion.abc2xml(abc)
        #         image = render_score(as_text=xml, input_type="xml")
        #         self.logger.log_table(
        #             f"val/{inout}_{i}",
        #             columns=["abc", "score"],
        #             data=[[abc, wandb.Image(image)]],
        #             # global_step=self.global_step,
        #         )
        # return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        raise NotImplementedError

    def configure_optimizers(self):
        return self.optimizer(self.transcriber_model.parameters())


if __name__ == '__main__':
    from pathlib import Path
    from loguru import logger

    data_module = AsapTokensDataModule(
        data_dir=Path("~/Documents/datasets/asap-dataset").expanduser(),
        batch_size=2,
        include_midi_scores=True,
        filter_fn=lambda e: e.composer == "Bach" and e.midi_performance.stem == "midi_score",
    )
    data_module.prepare_data()
    data_module.setup()

    model = TokensTranscriberModule(
        optimizer=torch.optim.AdamW,
    )
    logger.info("Model instantiated")

    for score_tensor, performance_streams in data_module.train_dataloader():
        logger.debug(f"{score_tensor.shape = }")
        logger.debug(
            f"performance_streams = { {k: v.shape for k, v in performance_streams.items() } }"
        )
        logger.debug("Feeding batch to the model...")
        res = model.training_step(
            {
                "score_tensor": score_tensor,
                "performance_streams": performance_streams,
            }
        )
        logger.debug(res)
        break
