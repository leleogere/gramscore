from collections.abc import Callable
from typing import Any

import lightning as L
import torch
import wandb.plot
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, BinaryF1Score
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from loguru import logger

from data.asap_tokens_datamodule import AsapTokensDataModule
from models.automata.transcriber_model import TokensTranscriberModel
from models.automata.vectorization_torch import TokenTensor
import models.automata.tokens as tk
from models.components.roformer import Roformer


class TokensTranscriberModule(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable[[Any], torch.optim.Optimizer],
    ):
        super().__init__()

        self.save_hyperparameters()
        self.transcriber_model = Roformer(
            stream_dimensions={
                "pitch": 128,
                "onset": 200,
                "duration": 200,
                "velocity": 8,
            },
            embedding_dimension=512,
            layer_norm_eps=1.0e-12,
            dropout_prob=0.1,
        )
        self.optimizer = optimizer
        # self.loss = nn.NLLLoss()
        self.loss = nn.CrossEntropyLoss(
            # See scripts/estimate_tokens_distribution.py for the weights
            weight=torch.Tensor(
                [
                    48.63,  # SOS (not needed)
                    2.12,  # Bar
                    21.77,  # Clef
                    48.63,  # KeySignature
                    0.11,  # Note
                    1.24,  # Rest
                    10.53,  # ChordStart
                    10.53,  # ChordEnd
                    48.63,  # EOS
                    21.50,  # TupletStart
                    21.50,  # TupletEnd
                ]
            )
        )

        self.global_metrics = MetricCollection(
            dict(
                macro_accuracy=MulticlassAccuracy(
                    num_classes=len(TokenTensor.TOKEN_INDEXER),
                    average="macro",
                ),
                micro_accuracy=MulticlassAccuracy(
                    num_classes=len(TokenTensor.TOKEN_INDEXER),
                    average="micro",
                ),
                macro_f1=MulticlassF1Score(
                    num_classes=len(TokenTensor.TOKEN_INDEXER),
                    average="macro",
                ),
                micro_f1=MulticlassF1Score(
                    num_classes=len(TokenTensor.TOKEN_INDEXER),
                    average="micro",
                ),
            ),
            prefix="val/",
        )
        self.per_class_metrics = MetricCollection(
            {k.__name__: BinaryF1Score() for k in TokenTensor.TOKEN_INDEXER.keys()},
            prefix="val/class_accuracy/",
        )

    def common_step(self, batch: dict[str, Any]):
        """
        Common part between training_step and validation_step

        :param batch: A dictionary containing the input data in the structure:
            {
                "score_tensor": torch.Tensor,  # input tensor representing the score
                "performance_streams": dict[str, torch.Tensor],  # dict of performance streams
            }
        """
        performance_streams: dict[str, torch.Tensor] = batch["performance_streams"]
        score_tensor: TokenTensor = batch["score_tensor"]

        # Make sure the first token of all sequences is StartOfScore
        assert all(
            score_tensor[:, 0, "token_type"][..., TokenTensor.TOKEN_INDEXER[tk.StartOfScore]] == 1
        ), "First token should be StartOfScore"
        # Same for EOS, but last token NOT padded should be EndOfScore
        last_not_padded: torch.Tensor = (score_tensor[:, :, 0] == -100).int().argmax(-1) - 1
        last_not_padded[last_not_padded == -1] = score_tensor.shape[1] - 1
        assert all(
            score_tensor[
                torch.arange(score_tensor.shape[0]),  # needed or torch broadcasts
                last_not_padded,
                "token_type",
            ][..., TokenTensor.TOKEN_INDEXER[tk.EndOfScore]]
            == 1
        ), "Last token should be EndOfScore"

        # Shift score tensors to the right
        score_tensor_input: TokenTensor = score_tensor[:, :-1]
        score_tensor_expected: TokenTensor = score_tensor[:, 1:]

        # Pass the input through the model
        encoder_outputs = self.transcriber_model.forward_enc(
            input_streams=performance_streams,
            attention_mask=None,  # TODO: replace with padding mask
        )  # (batch_size, num_tokens, num_token_types)
        decoder_outputs = self.transcriber_model.forward_dec(
            input_tokens=score_tensor_input,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=None,  # TODO: replace with padding mask
        )
        router_logits = self.transcriber_model.forward_router(decoder_outputs.last_hidden_state)

        true_token_types: torch.Tensor = score_tensor_expected[
            "token_type"
        ]  # (batch, tokens, num_token_types)

        # Make sure the sequence length match
        assert (
            router_logits.shape[1] == true_token_types.shape[1]
        ), f"Router logits shape {router_logits.shape} does not match true token types shape {true_token_types.shape}"

        flattened_true_token_types = rearrange(true_token_types, "b t k -> (b t) k")
        flattened_router_logits = rearrange(router_logits, "b t k -> (b t) k")

        # Would be better if we could get the mask directly from the pad_sequence function
        not_padded = flattened_true_token_types[:, 0] != -100
        assert torch.all(flattened_true_token_types[not_padded] != -100), "Error in padding mask"

        # Compute the router loss
        loss = self.loss(
            input=flattened_router_logits[not_padded],
            target=flattened_true_token_types[not_padded],
        )
        return loss, {
            "flattened_router_logits": flattened_router_logits,
            "not_padded": not_padded,
            "flattened_true_token_types": flattened_true_token_types,
        }

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

    def training_step(self, batch: dict[str, Any], batch_idx: int = None) -> STEP_OUTPUT:
        loss, _ = self.common_step(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int = None):
        loss, results = self.common_step(batch)
        self.log("val/loss", loss)

        flattened_router_logits = results["flattened_router_logits"]
        flattened_true_token_types_ohe = results["flattened_true_token_types"]
        not_padded = results["not_padded"]

        flattened_pred_token_types = flattened_router_logits[not_padded].argmax(-1)
        flattened_true_token_types = flattened_true_token_types_ohe[not_padded].argmax(-1)

        # Count each token type
        bincount = torch.bincount(flattened_pred_token_types, minlength=11)
        self.logger.log_metrics(
            {
                f"val/tokens_distribution/{k.__name__}": v.item()
                for k, v in zip(
                    TokenTensor.TOKEN_INDEXER.keys(),
                    bincount,
                )
            },
            step=self.global_step,
        )

        # Compute entropy of predictions (to see if always predicting the same token type)
        prob = bincount / bincount.sum()
        prob = prob[prob > 0]
        entropy = -torch.sum(prob * torch.log(prob))
        self.log("val/tokens_entropy", entropy)
        logger.debug(f"Entropy: {entropy.item()}")
        logger.debug(f"Bincount: {bincount.tolist()}")
        logger.debug(f"Prob: {prob.tolist()}")

        # Log metrics
        for name, metric in self.global_metrics.items():
            metric(
                preds=flattened_pred_token_types,
                target=flattened_true_token_types,
            )
            self.log(name, metric)

        # Log confusion matrix
        cm = wandb.plot.confusion_matrix(
            y_true=flattened_true_token_types.cpu().numpy(),
            preds=flattened_pred_token_types.cpu().numpy(),
            class_names=[c.__name__ for c in TokenTensor.TOKEN_INDEXER.keys()],
        )
        self.logger.experiment.log({f"val/confusion_matrix": cm})

        # Log per-class metrics
        for name, metric in self.per_class_metrics.items():
            # Very hacky (needed bc MetricCollection reorders keys), I should probably index the
            # tokens types by string rather than class
            index = TokenTensor.TOKEN_INDEXER[getattr(tk, name.split("/")[-1])]
            metric(
                preds=(flattened_true_token_types == index),
                target=(flattened_pred_token_types == index),
            )
            self.log(name, metric)

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

    for batch in data_module.train_dataloader():
        performance_streams = batch["performance_streams"]
        score_tensor = batch["score_tensor"]

        logger.debug(f"{score_tensor.shape = }")
        logger.debug(
            f"performance_streams = { {k: v.shape for k, v in performance_streams.items() } }"
        )
        logger.debug("Feeding batch to the model...")
        res = model.training_step(batch)
        logger.debug(res)
        break
