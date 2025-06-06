import difflib
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Mapping
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

# from models.automata.transcriber_model import TokensTranscriberModel
from models.automata.vectorization_torch import TokenTensor
import models.automata.tokens as tk
from models.components.roformer import Roformer
from models.components.tokens_transcriber_model import TokensTranscriberModel
from utils.constants import PADDING_VALUE


class TokensTranscriberModule(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable[[Any], torch.optim.Optimizer],
        embedding_dim: int = 512,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 8,
        intermediate_size: int = 1536,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        experts_inner_dims: list[int] = None,
        expert_loss_weight: float = 1.0,
        log_logits_each_n_val_steps: int = 1,
    ):
        super().__init__()
        if experts_inner_dims is None:
            experts_inner_dims = [256, 128]

        self.save_hyperparameters()
        self.transcriber_model = TokensTranscriberModel(
            stream_dimensions={
                "pitch": 128,
                "onset": 200,
                "duration": 200,
                "velocity": 8,
            },
            roformer=Roformer(
                hidden_size=embedding_dim,
                layer_norm_eps=layer_norm_eps,
                hidden_dropout_prob=hidden_dropout_prob,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
            ),
            expert_inner_dims=experts_inner_dims,
        )
        self.optimizer = optimizer

        # See scripts/estimate_tokens_distribution.py for the weights
        self.token_weights = {
            tk.StartOfScore: 48.63,
            tk.EndOfScore: 48.63,
            tk.Bar: 2.12,
            tk.Clef: 21.77,
            tk.KeySignature: 48.63,
            tk.Note: 0.11,
            tk.Rest: 1.24,
            tk.ChordStart: 10.53,
            tk.ChordEnd: 10.53,
            tk.TupletStart: 21.50,
            tk.TupletEnd: 21.50,
        }

        self.router_loss = nn.CrossEntropyLoss(
            weight=torch.Tensor(
                [self.token_weights[token_type] for token_type in TokenTensor.TOKEN_INDEXER]
            )
        )
        self.expert_losses: Mapping[str, Mapping[str, nn.Module]] = nn.ModuleDict(
            {
                token_type: nn.ModuleDict(
                    self.transcriber_model.expert_heads[token_type].get_losses()
                )
                for token_type in tk.TOKEN_TYPES_STR
                if token_type
                in self.transcriber_model.expert_heads  # TODO: remove this when all heads implemented
            }
        )

        self.global_metrics = MetricCollection(
            {
                "Macro accuracy": MulticlassAccuracy(
                    num_classes=len(TokenTensor.TOKEN_INDEXER),
                    average="macro",
                ),
                "Micro accuracy": MulticlassAccuracy(
                    num_classes=len(TokenTensor.TOKEN_INDEXER),
                    average="micro",
                ),
                "Macro F1": MulticlassF1Score(
                    num_classes=len(TokenTensor.TOKEN_INDEXER),
                    average="macro",
                ),
                "Micro F1": MulticlassF1Score(
                    num_classes=len(TokenTensor.TOKEN_INDEXER),
                    average="micro",
                ),
            },
            prefix="[V] Router specific metrics/",
        )
        self.per_class_metrics = MetricCollection(
            {k.__name__: BinaryF1Score() for k in TokenTensor.TOKEN_INDEXER.keys()},
            prefix="[V] Router class accuracies/",
        )

        self.validation_counter = 0
        self.validation_logits = defaultdict(list)

    def common_step(self, batch: dict[str, Any]):
        """
        Common part between training_step and validation_step

        :param batch: A dictionary containing the input data in the structure:
            {
                "score_tensor": torch.Tensor,  # input tensor representing the score
                "performance_streams": dict[str, torch.Tensor],  # dict of performance streams
            }
        """
        performance_streams: dict[str, torch.Tensor] = batch["performance_streams"]  # B * N * F_str
        performance_padded_mask: torch.Tensor = batch["performance_padding_mask"]  # B * N
        performance_not_padded_mask = torch.logical_not(performance_padded_mask)
        performance_num_notes: torch.Tensor = batch["performance_lengths"]  # B
        score_tensor: TokenTensor = batch["score_tensor"]  # B * T * K
        score_padded_mask: torch.Tensor = batch["score_padding_mask"]  # B * T
        score_not_padded_mask = torch.logical_not(score_padded_mask)
        score_num_tokens: torch.Tensor = batch["score_lengths"]  # B

        # Make sure the first token of all sequences is StartOfScore
        assert all(
            score_tensor[:, 0, "token_type"][..., TokenTensor.TOKEN_INDEXER[tk.StartOfScore]] == 1
        ), "First token should be StartOfScore"
        # Same for EOS, but last token NOT padded should be EndOfScore
        assert all(
            score_tensor[
                torch.arange(score_tensor.shape[0]),  # needed or torch broadcasts
                score_num_tokens - 1,
                "token_type",
            ][..., TokenTensor.TOKEN_INDEXER[tk.EndOfScore]]
            == 1
        ), "Last token should be EndOfScore"

        # Shift score tokens to the right
        score_tensor_input: TokenTensor = score_tensor[:, :-1]
        score_tensor_expected: TokenTensor = score_tensor[:, 1:]

        # Pass the performance streams through the encoder
        encoder_outputs = self.transcriber_model.forward_encoder(
            input_streams=performance_streams,
            attention_mask=performance_not_padded_mask,
        )  # (B, N, E)

        # Pass the score input and the encoder outputs through the decoder
        decoder_outputs = self.transcriber_model.forward_decoder(
            input_tokens=score_tensor_input,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=performance_not_padded_mask,
        )  # (B, T, E)

        # Pass the decoder outputs through the router
        router_logits = self.transcriber_model.forward_router(
            decoder_outputs.last_hidden_state
        )  # (B, T, K)
        true_token_types: torch.Tensor = score_tensor_expected["token_type"]  # (B, T, K)
        true_token_types_argmax = true_token_types.argmax(-1)  # (B, T)
        true_token_types_argmax[score_padded_mask[..., 1:]] = PADDING_VALUE

        # Make sure the sequence length match
        assert (
            router_logits.shape[1] == true_token_types.shape[1]
        ), f"Router logits shape {router_logits.shape} does not match true token types shape {true_token_types.shape}"

        # Pass the decoder outputs through the expert heads
        token_type_masks = {
            t_str: true_token_types_argmax == TokenTensor.TOKEN_INDEXER[t_cls]
            for t_cls, t_str in zip(tk.TOKEN_TYPES, tk.TOKEN_TYPES_STR)
        }
        expert_outputs: dict[str, dict[str, torch.Tensor]] = self.transcriber_model.forward_experts(
            decoder_outputs=decoder_outputs.last_hidden_state,
            token_type_masks=token_type_masks,
            convert_to_tokentensor=False,
        )

        # Compute all losses
        losses = {}
        # losses:
        #   router_loss: <float>
        #   expert_losses:
        #     Note:
        #       _global: <float>
        #       note_class: <float>
        #       note_alteration: <float>
        #       note_octave: <float>
        #       duration: <float>
        #       ...
        #     Rest:
        #       _global: <float>
        #       duration: <float>
        #       ...
        #     ...

        # Compute the router loss
        flattened_true_token_types = rearrange(true_token_types, "b t k -> (b t) k")
        flattened_router_logits = rearrange(router_logits, "b t k -> (b t) k")
        flattened_score_not_padded_mask = rearrange(score_not_padded_mask[..., 1:], "b t -> (b t)")
        # Check padding OK
        assert torch.all(
            flattened_true_token_types[~flattened_score_not_padded_mask] == PADDING_VALUE
        ) and torch.all(
            flattened_true_token_types[flattened_score_not_padded_mask] != PADDING_VALUE
        ), "Error in padding mask"
        router_loss = self.router_loss(
            input=flattened_router_logits[flattened_score_not_padded_mask],
            target=flattened_true_token_types[flattened_score_not_padded_mask],
        )
        losses["router_loss"] = router_loss

        # Compute the expert losses
        losses["expert_losses"] = {}
        expected_output = score_tensor_expected.get_as_dict(
            include_token_type_dim=False,
            remove_padding=True,
            token_type_keys_as_str=True,
        )
        for token_type in expert_outputs:
            losses["expert_losses"][token_type] = {}
            for dim_name, loss_fn in self.expert_losses[token_type].items():
                inp = expert_outputs[token_type][dim_name]
                tgt = expected_output[token_type][dim_name]
                assert inp.shape == tgt.shape
                if tgt.shape[0] > 0:
                    expert_loss = loss_fn(input=inp, target=tgt)
                else:
                    expert_loss = None
                losses["expert_losses"][token_type][dim_name] = expert_loss
            losses["expert_losses"][token_type]["_global"] = sum(
                [l for l in losses["expert_losses"][token_type].values() if l is not None]
            )

        # Combine all losses
        loss = losses["router_loss"] + self.hparams.expert_loss_weight * sum(
            self.token_weights[tk.TOKEN_STR_TO_TYPE[token_type]]
            * losses["expert_losses"][token_type]["_global"]
            for token_type in losses["expert_losses"]
        )

        return loss, {
            "losses": losses,
            "router_logits": router_logits,
            "expert_outputs": expert_outputs,
            "true_token_types": true_token_types,
            "score_not_padded_mask": score_not_padded_mask,
            "flattened_router_logits": flattened_router_logits,
            "flattened_score_not_padded_mask": flattened_score_not_padded_mask,
            "flattened_true_token_types": flattened_true_token_types,
        }

    def _log_loss_components(self, losses: dict[str, Any], name: str):
        """
        Log all loss components to wandb
        :param loss: A dictionary containing the loss components
        :param prefix: A string to prefix the loss components
        """
        global_category_prefix = f"[{name}] Component global losses"
        for loss_name, loss_values in losses.items():
            # Log router loss
            if loss_name == "router_loss":
                self.log(f"{global_category_prefix}/Router loss", loss_values)
            # Log expert losses
            else:
                # First log the sum, so that it appears first in wandb
                for token_type, dimensions in loss_values.items():
                    self.log(f"{global_category_prefix}/{token_type} loss", dimensions["_global"])
                # Then log the individual components
                for token_type, dimensions in loss_values.items():
                    expert_components_prefix = f"[{name}] Expert {token_type} component losses"
                    for dim_name, value in dimensions.items():
                        if dim_name != "_global" and value is not None:
                            self.log(f"{expert_components_prefix}/{dim_name} loss", value)

    def training_step(self, batch: dict[str, Any], batch_idx: int = None) -> STEP_OUTPUT:
        loss, outputs = self.common_step(batch)
        self.log("[T|V] Global metrics/Train loss", loss)
        self._log_loss_components(outputs["losses"], name="T")
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int = None):
        loss, results = self.common_step(batch)
        self.log("[T|V] Global metrics/Validation loss", loss)
        self._log_loss_components(results["losses"], name="V")

        router_logits = results["router_logits"]
        true_token_types = results["true_token_types"]
        score_not_padded_mask = results["score_not_padded_mask"]
        flattened_router_logits = results["flattened_router_logits"]
        flattened_true_token_types_ohe = results["flattened_true_token_types"]
        flattened_score_not_padded_mask = results["flattened_score_not_padded_mask"]

        # Log the first 3 chunks of each batch
        if self.validation_counter % self.hparams.log_logits_each_n_val_steps == 0:
            n = min(3, batch["score_tensor"].shape[0])
            performance_path = batch["entries"][0].midi_performance_relative.as_posix()
            score_folder = batch["entries"][0].xml_score_relative.parent.as_posix()
            assert all(
                p.midi_performance_relative.as_posix() == performance_path for p in batch["entries"]
            )
            for i, (
                true_tokens_sequence,
                pred_tokens_sequence,
                entry,
                num_tokens,
                start,
                end,
                logits,
            ) in enumerate(
                zip(
                    true_token_types[:n].argmax(-1),
                    router_logits[:n].argmax(-1),
                    batch["entries"][:n],
                    batch["score_lengths"][:n],
                    batch["chunk_starts"][:n],
                    batch["chunk_ends"][:n],
                    router_logits[:n],
                )
            ):
                # Convert to string
                true_tokens_sequence = [
                    self._token_str_mapping[TokenTensor.TOKEN_INDEXER_INV[t.item()]]
                    for t in true_tokens_sequence[: num_tokens - 1]
                ]
                # pred_tokens_sequence = [
                #     self._token_str_mapping[TokenTensor.TOKEN_INDEXER_INV[t.item()]]
                #     for t in pred_tokens_sequence[: num_tokens - 1]
                # ]
                #
                # # Align the sequences
                # aligned_true, aligned_pred, diffs = self.align_character_sequences(
                #     true_tokens_sequence,
                #     pred_tokens_sequence,
                # )
                # # Log the sequences
                # self.logger.log_text(
                #     key=f"val_visu_alignments/{entry.xml_score_relative}",
                #     columns=["true", "pred", "diffs"],
                #     data=[[" ".join(aligned_true), " ".join(aligned_pred), " ".join(diffs)]],
                # )

                # Log the logits
                data = logits.softmax(-1).tolist()[: num_tokens - 1]
                for row, true_token in zip(data, true_tokens_sequence):
                    # add true token
                    row.append(true_token)
                    row.append(performance_path)
                    row.append(i)
                    row.append(f"{start}-{end}")
                self.validation_logits[score_folder].extend(data)

        flattened_pred_token_types = flattened_router_logits[
            flattened_score_not_padded_mask
        ].argmax(-1)
        flattened_true_token_types = flattened_true_token_types_ohe[
            flattened_score_not_padded_mask
        ].argmax(-1)

        # Count each token type
        bincount = torch.bincount(flattened_pred_token_types, minlength=11)
        self.logger.log_metrics(
            {
                f"[V] Router tokens distribution/{k.__name__}": v.item()
                for k, v in zip(
                    TokenTensor.TOKEN_INDEXER.keys(),
                    bincount,
                )
            },
        )

        # Compute entropy of predictions (to see if always predicting the same token type)
        prob = bincount / bincount.sum()
        prob = prob[prob > 0]
        entropy = -torch.sum(prob * torch.log(prob))
        self.log("[V] Router specific metrics/Tokens entropy", entropy)
        logger.trace(f"Entropy: {entropy.item()}")
        logger.trace(f"Bincount: {bincount.tolist()}")
        logger.trace(f"Prob: {prob.tolist()}")

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
        self.logger.experiment.log({f"[V] Confusion matrix/Confusion matrix": cm})

        # Log per-class metrics
        for (name, metric), token_type in zip(
            self.per_class_metrics.items(),
            self.per_class_metrics.keys(keep_base=True),
        ):
            index = TokenTensor.TOKEN_INDEXER[tk.TOKEN_STR_TO_TYPE[token_type]]
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

    _token_str_mapping = {
        tk.StartOfScore: "S",
        tk.EndOfScore: "E",
        tk.Bar: "B",
        tk.KeySignature: "K",
        tk.Clef: "C",
        tk.Rest: "R",
        tk.Note: "N",
        tk.ChordStart: "<",
        tk.ChordEnd: ">",
        tk.TupletStart: "(",
        tk.TupletEnd: ")",
    }

    @staticmethod
    def align_character_sequences(
        seq1: list[str],
        seq2: list[str],
        join_results_str: str | None = None,
    ) -> tuple[list[str], list[str], list[str]] | tuple[str, str, str]:
        """
        This function compares two sequences of characters and returns the aligned sequences.
        The input sequences must be lists of single-character strings.
        The output depends on the `join_results_str` parameter:
        - If `join_results_str` is None, the function returns three lists:
            - aligned1: the first aligned sequence
            - aligned2: the second aligned sequence
            - diffs: a list of differences between the two sequences (+ for insertions, - for deletions,
                ^ for replacements)
        - If `join_results_str` is a string, the function will return the three previous lists joined
            into strings using the provided string as a separator.

        :param seq1: The first sequence of characters to compare.
        :param seq2: The second sequence of characters to compare.
        :param join_results_str: The string to join the results. If None, the function returns lists.
        :return: A tuple containing the aligned sequences and the differences.
        """
        assert all(
            all(len(token) == 1 for token in seq) for seq in (seq1, seq2)
        ), "All tokens must be single characters"

        matcher = difflib.SequenceMatcher(None, seq1, seq2)
        aligned1, aligned2, diffs = [], [], []
        for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
            if opcode == 'equal':
                aligned1.extend(seq1[i1:i2])
                aligned2.extend(seq2[j1:j2])
                diffs.extend(" " * (i2 - i1))
            elif opcode == 'insert':
                aligned1.extend([' '] * (j2 - j1))
                aligned2.extend(seq2[j1:j2])
                diffs.extend(["+"] * (j2 - j1))
            elif opcode == 'delete':
                aligned1.extend(seq1[i1:i2])
                aligned2.extend([' '] * (i2 - i1))
                diffs.extend(["-"] * (i2 - i1))
            elif opcode == 'replace':
                aligned1.extend(seq1[i1:i2])
                aligned2.extend(seq2[j1:j2])
                diffs.extend(["^"] * (i2 - i1))
        if join_results_str is not None:
            return (
                join_results_str.join(aligned1),
                join_results_str.join(aligned2),
                join_results_str.join(diffs),
            )
        else:
            return aligned1, aligned2, diffs

    def on_validation_end(self) -> None:
        if self.validation_counter % self.hparams.log_logits_each_n_val_steps == 0:
            for score_folder, logits in self.validation_logits.items():
                self.logger.log_table(
                    # key=f"[V] Logits visualization/Logits",
                    key=f"validation_logits_visualization/{score_folder}",
                    columns=list(
                        self._token_str_mapping[t] for t in TokenTensor.TOKEN_INDEXER.keys()
                    )
                    + ["true"]
                    + ["performance"]
                    + ["chunk_idx"]
                    + ["chunk"],
                    data=logits,
                    step=self.global_step,
                )
            self.validation_logits = defaultdict(list)
        self.validation_counter += 1


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

    data_module.data_train.entries = data_module.data_train.entries[:5]

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
