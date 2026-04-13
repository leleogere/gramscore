import tempfile
import warnings
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L
import partitura as pt
import torch
import wandb.plot
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from muster import muster
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, BinaryF1Score

import data.components.transforms as t
import models.automata.tokens as tk
from data.asap_tokens_datamodule import AsapTokensDataModule
from metrics.score_similarity.score_similarity import score_similarity_normalized
from models.automata.io import token_sequence_to_partitura_permissive
from models.automata.vectorization_torch import TokenTensor
from models.components.positional_encoding import ContinuousPositionalEncoding1D
from models.components.roformer import Roformer
from models.components.tokens_transcriber_model import TokensTranscriberModel, GenerationState
from utils.constants import PADDING_VALUE


class TokensTranscriberModule(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable[[Any], torch.optim.Optimizer],
        scheduler: Callable[[Any], torch.optim.lr_scheduler.LRScheduler] | None = None,
        embedding_dim: int = 512,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 8,
        intermediate_size: int = 1536,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        max_position_embeddings=8192,
        positional_encoder: ContinuousPositionalEncoding1D = None,
        router_inner_dims: list[int] = None,
        experts_inner_dims: list[int] = None,
        expert_loss_weight: float = 0.01,
        latent_part_reserved_for_router: float | int | None = None,
        performance_dropout: float = 0.05,
        score_dropout: float = 0.75,
        log_logits_each_n_val_steps: int = 1,
        val_max_logits_logged: int = 5,
        scale_expert_losses_with_token_frequency: bool = False,
        tokens_positions_scaling_factor: float = 1.0,
        time_based_cross_attention: bool = False,
        chunk_size_scheduler: Callable[[int], int] | None = None,
        full_eval_every_n_val_steps: int = 100,
        full_eval_max_batches: int = None,
    ):
        super().__init__()
        if experts_inner_dims is None:
            experts_inner_dims = [256, 128]
        if router_inner_dims is None:
            router_inner_dims = []  # for backward compatibility (and coherence)

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
                max_position_embeddings=max_position_embeddings,
                positional_encoder=positional_encoder,
                token_positions_multiplier_in_cross_attention=tokens_positions_scaling_factor,
            ),
            expert_inner_dims=experts_inner_dims,
            router_inner_dims=router_inner_dims,
            time_based_cross_attention=time_based_cross_attention,
            latent_part_reserved_for_router=latent_part_reserved_for_router,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.chunk_size_scheduler = chunk_size_scheduler

        # See scripts/estimate_tokens_distribution.py for the weights
        self.token_weights = {
            tk.StartOfScore: 60.683,  # Not used
            tk.EndOfScore: 60.683,
            tk.Bar: 2.429,
            tk.Clef: 13.547,
            tk.KeySignature: 53.607,
            tk.Note: 0.131,
            tk.Rest: 1.370,
            tk.ChordStart: 1.149,
            tk.ChordEnd: 1.149,
            tk.TupletStart: 5.275,
            tk.TupletEnd: 5.275,
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
            prefix="[V] Router class F1/",
        )

        self.validation_counter = 0
        self.validation_logits = defaultdict(list)
        self.all_flattened_true_token_types = []
        self.all_flattened_pred_token_types = []

    def common_step(
        self,
        batch: dict[str, Any],
        train: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Common part between training_step and validation_step

        :param batch: A dictionary containing the input data in the structure:
            {
                "score_tensor": torch.Tensor,  # input tensor representing the score
                "performance_streams": dict[str, torch.Tensor],  # dict of performance streams
            }
        :param train: Whether the step is for training or validation (affects dropout)
        """
        performance_streams: dict[str, torch.Tensor] = batch["performance_streams"]  # B * N * F_str
        performance_padded_mask: torch.Tensor = batch["performance_padding_mask"]  # B * N
        performance_not_padded_mask = torch.logical_not(performance_padded_mask)
        performance_onset_times = batch["performance_onset_times"]
        performance_num_notes: torch.Tensor = batch["performance_length"]  # B
        score_tensor: TokenTensor = batch["score_tensor"]  # B * T * K
        score_padded_mask: torch.Tensor = batch["score_padding_mask"]  # B * T
        score_not_padded_mask = torch.logical_not(score_padded_mask)
        score_num_tokens: torch.Tensor = batch["score_length"]  # B

        if (
            score_tensor.size(1)
            > self.transcriber_model.roformer.encoder.config.max_position_embeddings
        ):
            for entry, mask in zip(batch["entry"], score_not_padded_mask):
                if torch.all(mask[-1] != -100):
                    raise RuntimeError(
                        f"The score {entry.xml_score_relative} is too long for the positional "
                        f"embedding of the roformer: {score_tensor.size(1)} > "
                        f"{self.transcriber_model.roformer.encoder.config.max_position_embeddings}"
                    )
            else:
                raise RuntimeError(
                    "Size error detected but no score triggered it, should not happen, just in case"
                )

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

        # Dropout some tokens when training
        if train:
            # Dropout performance tokens
            # QUESTION: in Beyer's, dropout is applied independently on all keys, is it OK?
            performance_dropout = self.hparams.performance_dropout
            performance_streams = {
                key: mask_tokens_randomly(
                    value,
                    p=performance_dropout,
                    padding_mask=performance_padded_mask,
                )
                for key, value in performance_streams.items()
            }
            # Dropout score tokens
            score_dropout = self.hparams.score_dropout
            for key in score_tensor_input.DIMS:
                score_tensor_input[key] = mask_tokens_randomly(
                    score_tensor_input[key],
                    p=score_dropout,
                    padding_mask=score_padded_mask[:, :-1],
                )

        # Pass the performance streams through the encoder
        encoder_outputs = self.transcriber_model.forward_encoder(
            input_streams=performance_streams,
            onset_times=performance_onset_times,
            attention_mask=performance_not_padded_mask,
        )  # (B, N, E)

        # Pass the score input and the encoder outputs through the decoder
        decoder_outputs = self.transcriber_model.forward_decoder(
            input_tokens=score_tensor_input,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=performance_not_padded_mask,
            encoder_positions=(
                performance_onset_times if self.hparams.time_based_cross_attention else None
            ),
        )  # (B, T, E)

        # Pass the decoder outputs through the router
        router_logits = self.transcriber_model.forward_router(
            decoder_outputs.last_hidden_state
        )  # (B, T, K)
        predicted_token_types = torch.argmax(router_logits, dim=-1)
        true_token_types: torch.Tensor = score_tensor_expected["token_type"]  # (B, T, K)
        true_token_types_argmax = true_token_types.argmax(-1)  # (B, T)
        true_token_types_argmax[score_padded_mask[..., 1:]] = PADDING_VALUE
        correct_predictions = predicted_token_types == true_token_types_argmax

        # Make sure the sequence length match
        assert (
            router_logits.shape[1] == true_token_types.shape[1]
        ), f"Router logits shape {router_logits.shape} does not match true token types shape {true_token_types.shape}"

        # Pass the decoder outputs through the expert heads
        true_token_type_masks = {
            t_str: true_token_types_argmax == TokenTensor.TOKEN_INDEXER[t_cls]
            for t_cls, t_str in zip(tk.TOKEN_TYPES, tk.TOKEN_TYPES_STR)
        }
        expert_outputs: dict[str, dict[str, torch.Tensor]] = self.transcriber_model.forward_experts(
            decoder_outputs=decoder_outputs.last_hidden_state,
            token_type_masks=true_token_type_masks,
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
            tokens_mask=true_token_type_masks,
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
        if self.hparams.scale_expert_losses_with_token_frequency:
            loss = losses["router_loss"] + self.hparams.expert_loss_weight * sum(
                self.token_weights[tk.TOKEN_STR_TO_TYPE[token_type]]
                * losses["expert_losses"][token_type]["_global"]
                for token_type in losses["expert_losses"]
            )
        else:
            loss = losses["router_loss"] + self.hparams.expert_loss_weight * sum(
                losses["expert_losses"][token_type]["_global"]
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
        loss, outputs = self.common_step(batch, train=True)
        self.log("[T|V] Global metrics/Train loss", loss)
        self._log_loss_components(outputs["losses"], name="T")
        self.log("Learning rate", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    def on_train_start(self) -> None:
        if (
            self.chunk_size_scheduler is not None
            and not self.trainer.reload_dataloaders_every_n_epochs
        ):
            raise ValueError(
                f"A chunk size scheduler is active, but the trainer is not reloading "
                f"dataloaders every epoch, it will likely cause issues with num_workers > 0. "
                "Please set reload_dataloaders_every_n_epochs=1 in the Trainer."
            )

    def on_train_epoch_end(self) -> None:
        if self.chunk_size_scheduler is not None:
            new_chunk_size = int(self.chunk_size_scheduler(self.global_step))
            for dl in (self.trainer.train_dataloader, self.trainer.val_dataloaders):
                quantizer: t.AsapMusicChunker = next(
                    tr
                    for tr in dl.dataset.transforms
                    if isinstance(tr, (t.QuantizedMusicChunker, t.AsapMusicChunker))
                )
                quantizer.num_performed_notes = new_chunk_size
            self.log("[T|V] Global metrics/Chunk size", new_chunk_size)

    def on_validation_start(self) -> None:
        logger.info(f"Validation starting")

    def validation_step(self, batch: dict[str, Any], batch_idx: int = None):
        loss, results = self.common_step(batch, train=False)
        self.log("[T|V] Global metrics/Validation loss", loss)
        self._log_loss_components(results["losses"], name="V")

        router_logits = results["router_logits"]
        true_token_types = results["true_token_types"]
        score_not_padded_mask = results["score_not_padded_mask"]
        flattened_router_logits = results["flattened_router_logits"]
        flattened_true_token_types_ohe = results["flattened_true_token_types"]
        flattened_score_not_padded_mask = results["flattened_score_not_padded_mask"]

        # Log the first chunks of each of the first val_max_logits_logged batches
        if (
            self.validation_counter % self.hparams.log_logits_each_n_val_steps == 0
            and batch_idx < self.hparams.val_max_logits_logged
        ):
            entry = batch["entry"][0]
            performance_path = entry.midi_performance_relative.as_posix()
            score_folder = entry.xml_score_relative.as_posix()
            true_tokens_sequence = true_token_types[0].argmax(-1)
            logits = router_logits[0]
            num_tokens = batch["score_length"][0]

            # Convert to string
            true_tokens_sequence = [
                self._token_str_mapping[TokenTensor.TOKEN_INDEXER_INV[t.item()]]
                for t in true_tokens_sequence[: num_tokens - 1]
            ]

            # Log the logits
            data = logits.softmax(-1).tolist()[: num_tokens - 1]
            for row, true_token in zip(data, true_tokens_sequence):
                # add true token
                row.append(true_token)
                row.append(performance_path)
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
        if self.validation_counter % self.hparams.log_logits_each_n_val_steps == 0:
            self.all_flattened_true_token_types.append(flattened_true_token_types)
            self.all_flattened_pred_token_types.append(flattened_pred_token_types)

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

        # Evaluate score with full metrics from time to time (not frequently as expensive)
        # and limit also the number of batches computed
        if (self.validation_counter % self.hparams.full_eval_every_n_val_steps == 0) and (
            (self.hparams.full_eval_max_batches is None)
            or (batch_idx < self.hparams.full_eval_max_batches)
        ):
            logger.info(f"Computing complete metrics on autoregressively transcribed scores")
            transcribed_token_tensor, flags_and_positions = self.transcriber_model.generate(
                performance_streams=batch["performance_streams"],
                performance_onset_times=batch["performance_onset_times"],
                performance_attention_mask=~batch["performance_padding_mask"],
                prefix=None,
                constraint=False,
                max_constraint_iterations=None,
                max_generated_length=1500,
                progress_bar=False,
                return_logits_and_mask=False,
            )
            flags = flags_and_positions[:, 0]
            positions = flags_and_positions[:, 1] + 1
            flag_counts = flags.unique(return_counts=True)
            for flag, count in zip(*flag_counts):
                self.log(
                    f"[V] Generation metrics/Flag {GenerationState(flag.item()).name}",
                    count.float(),
                    reduce_fx="sum",
                )
            # Only evaluate scores for which the transcription has finished (biases results toward
            # good transcription, but not sure how to deal with that differently, probably with a
            # large penalty for unfinished transcription but how large then?
            to_evaluate = flags == GenerationState.FINISHED.value
            if to_evaluate.any():
                end_positions = positions[to_evaluate].tolist()
                predicted_tokens = [
                    tokens[:end_position]
                    for tokens, end_position in zip(
                        transcribed_token_tensor.quantize(inplace=True).to_tokens(),
                        end_positions,
                    )
                ]
                full_eval_results = self.evaluate_transcription(
                    predictions=predicted_tokens,
                    ground_truth=[s for s, f in zip(batch["score_tokens"], to_evaluate) if f],
                )
                score_similarities_scores = full_eval_results["score_similarities_scores"]
                num_score_similarity_errors = full_eval_results["num_score_similarity_errors"]
                muster_scores = full_eval_results["muster_scores"]
                num_muster_errors = full_eval_results["num_muster_errors"]
                num_conversion_errors = full_eval_results["num_conversion_errors"]
                if score_similarities_scores:
                    keys = {k for d in score_similarities_scores for k in d}
                    for key in keys:
                        items = [
                            s[key]
                            for s in score_similarities_scores
                            if s.get(key, None) is not None
                        ]
                        if len(items) != 0:
                            self.log(
                                f"[V] Generation metrics/ScoreSimilarity/{key}",
                                torch.tensor(items, dtype=torch.float).mean(dim=0),
                            )
                self.log(
                    f"[V] Generation metrics/ScoreSimilarity/ERRORS",
                    num_score_similarity_errors,
                    reduce_fx="sum",
                )
                if muster_scores:
                    keys = {k for d in muster_scores for k in d}
                    for key in keys:
                        items = [s[key] for s in muster_scores if s.get(key, None) is not None]
                        if len(items) != 0:
                            self.log(
                                f"[V] Generation metrics/MUSTER/{key}",
                                torch.tensor(items, dtype=torch.float).mean(dim=0),
                            )
                self.log(
                    f"[V] Generation metrics/MUSTER/ERRORS", num_muster_errors, reduce_fx="sum"
                )
                self.log(
                    f"[V] Generation metrics/Detok. errors", num_conversion_errors, reduce_fx="sum"
                )

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
        optimizer = self.optimizer(self.transcriber_model.parameters())
        if self.scheduler is None:
            return optimizer
        else:
            scheduler = {"scheduler": self.scheduler(optimizer), "interval": "step", "frequency": 1}
            return [optimizer], [scheduler]

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

    def on_validation_end(self) -> None:
        # Log large metrics periodically
        if self.validation_counter % self.hparams.log_logits_each_n_val_steps == 0:
            # Log confusion matrix
            cm = wandb.plot.confusion_matrix(
                y_true=torch.concat(self.all_flattened_true_token_types).cpu().numpy(),
                preds=torch.concat(self.all_flattened_pred_token_types).cpu().numpy(),
                class_names=[c.__name__ for c in TokenTensor.TOKEN_INDEXER.keys()],
            )
            self.logger.experiment.log({f"[V] Confusion matrix/Confusion matrix": cm})
            self.all_flattened_true_token_types = []
            self.all_flattened_pred_token_types = []

            # Log logits
            for score_folder, logits in self.validation_logits.items():
                self.logger.log_table(
                    # key=f"[V] Logits visualization/Logits",
                    key=f"validation_logits_visualization/{score_folder}",
                    columns=list(
                        self._token_str_mapping[t] for t in TokenTensor.TOKEN_INDEXER.keys()
                    )
                    + ["true"]
                    + ["performance"],
                    # + ["chunk_idx"]
                    # + ["chunk"],
                    data=logits,
                    step=self.global_step,
                )
            self.validation_logits = defaultdict(list)
        self.validation_counter += 1

        logger.info("Validation finished!")

    def evaluate_transcription(
        self,
        predictions: list[list[tk.AnyToken]],
        ground_truth: list[list[tk.AnyToken]],
        sequence_names: list[str] | None = None,
    ):
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"The number of predictions ({len(predictions)}) and ground truth "
                f"({len(ground_truth)}) do not match."
            )
        if sequence_names is None:
            sequence_names = [str(i) for i in range(len(predictions))]
        else:
            if len(sequence_names) != len(predictions):
                raise ValueError(
                    f"The number of sequence names ({len(sequence_names)}) and predictions "
                    f"({len(predictions)}) do not match."
                )
        # Prepare metrics
        num_conversion_errors = 0
        score_similarities_scores = []
        num_score_similarity_errors = 0
        muster_scores = []
        num_muster_errors = 0
        for pred, gt, name in zip(predictions, ground_truth, sequence_names):
            with tempfile.TemporaryDirectory(prefix="transcription_") as tmpdir_name:
                # Prepare paths to the temporary files
                tmpdir = Path(tmpdir_name)
                pred_path = tmpdir / "pred.musicxml"
                gt_path = tmpdir / "gt.musicxml"
                # Convert both sequences to Partitura, then to MusicXML (start with
                # transcription to catch errors early
                try:
                    pred_pt = token_sequence_to_partitura_permissive(pred)
                    pt.save_musicxml(pred_pt, pred_path)
                    gt_pt = token_sequence_to_partitura_permissive(gt)
                    pt.save_musicxml(gt_pt, gt_path)
                except Exception as e:
                    logger.error(
                        f"Error converting token sequence '{name}' to partitura or MusicXML, "
                        f"skipping it: {e}"
                    )
                    num_conversion_errors += 1
                    continue
                # Evaluate them
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    try:
                        score_sim = score_similarity_normalized(
                            est=pred_path.as_posix(), gt=gt_path.as_posix(), full=False
                        )
                        score_similarities_scores.append(score_sim)
                    except Exception as e:
                        logger.error(f"Error on score similarity on {name}: {e}")
                        num_score_similarity_errors += 1
                    try:
                        muster_sim = muster(
                            score_est=pred_path, score_gt=gt_path, clean_scores=False
                        )
                        if all(v is None for v in muster_sim.values()):
                            raise Exception("All Muster metrics are None")
                        muster_scores.append(muster_sim)
                    except Exception as e:
                        logger.error(f"Error on MUSTER on {name}: {e}")
                        num_muster_errors += 1
        return {
            "num_conversion_errors": num_conversion_errors,
            "num_score_similarity_errors": num_score_similarity_errors,
            "num_muster_errors": num_muster_errors,
            "score_similarities_scores": score_similarities_scores,
            "muster_scores": muster_scores,
        }


def mask_tokens_randomly(
    input: torch.Tensor,
    p: float,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    This function randomly masks (set to 0) tokens in the input tensor with a probability p.

    :param input: The input tensor (B, T, F)
    :param padding_mask: A boolean mask tensor (B, T), where True indicates the padded positions,
        which are not affected by the masking
    :param p: The dropout probability

    For example, for:
        - input of shape (B, N, F)
        - not_dropped_out_mask of shape (B, N)
    each of the dropped out (B, N) position will be set to zero for all F features.

    :return: The output tensor with dropout applied
    """
    if p == 0.0:
        return input
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")
    drop_probabilities = torch.full_like(input, 1 - p)
    if padding_mask is not None:
        drop_probabilities[padding_mask] = 1.0  # Never drop padded tokens
    output = input * torch.bernoulli(drop_probabilities)
    return output


if __name__ == '__main__':
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
