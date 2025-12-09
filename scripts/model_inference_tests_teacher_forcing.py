"""
Try to load and run inference on a model
"""

from pprint import pprint
from pathlib import Path
from typing import Any

import symusic
import torch
import torch.nn.functional as F
from music21.common import defaultlist
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.components.score_performance_dataset import ScorePerformanceEntry
from data.components.transforms import PerformanceReader
from data.components.transforms import PerformanceVectorizer
from data.pdmx_tokens_datamodule import PdmxTokensDataModule
from models.automata.automata import ConstrainedAutomata
from models.automata.tokens import TranscriptionState
from models.automata.vectorization_torch import TokenTensor
from models.components.tokens_transcriber_model import TokensTranscriberModel
from models.tokens_transcriber_module import TokensTranscriberModule
from models.automata import tokens as tk
from utils.debug import open_in_meld
from utils.debug import show_matrix

# fmt: off
model_path = Path("~/Documents/transcription/gramscore/outputs/server/JZ/2025-11-07_09-30-46/checkpoints/last.ckpt").expanduser()


# base_path = Path("~/Documents/datasets/asap-dataset").expanduser()
# performance_path = base_path / "Bach/Prelude/bwv_860/YoungS01M.mid"
# performance_path = base_path / "Bach/Prelude/bwv_860/midi_score.mid"
# performance_path = base_path / "Bach/Prelude/bwv_883/LuM01M.mid"
# performance_path = base_path / "Bach/Prelude/bwv_883/midi_score.mid"

base_path = Path("~/Documents/datasets/pdmx").expanduser()
# performance_path = base_path / "mid/0/0/Qma1a1L2c1vSEuGfdPVSEEWfuYAEoBmjpVESX1ok2imu4b.mid"
performance_path = base_path / "mid/0/0/Qma1a2YHEJ8j9gxLCxgALAfjYaiYJQogh1bgyFSwNSRPCs.mid"
# fmt: on


def transcribe_performance(
    model: TokensTranscriberModel,
    performance_streams: dict[str, torch.Tensor],
    onset_times: torch.Tensor,
    sample_token_type: bool = False,  # TODO: implement sampling
    max_length: int = 1024,
    return_logits: bool = False,
    use_cache: bool = True,
) -> list[tk.AnyToken]:
    if sample_token_type:
        raise NotImplementedError("Sampling token types is not implemented yet.")  # TODO

    # model.roformer.encoder.embed_positions = RoFormerSinusoidalPositionalEmbedding(
    #     128,
    #     model.roformer.encoder.config.output_hidden_states
    #     // model.roformer.encoder.config.num_attention_heads,
    # )
    # model.roformer.decoder.embed_positions = RoFormerSinusoidalPositionalEmbedding(
    #     128,
    #     model.roformer.decoder.config.output_hidden_states
    #     // model.roformer.decoder.config.num_attention_heads,
    # )

    # Return logits if requested
    if return_logits:
        logits = []

    with torch.no_grad():
        # for key, value in performance_streams.items():
        #     performance_streams[key] = value.unsqueeze(0)

        # Pass performance streams through the encoder
        encoder_output = model.forward_encoder(
            input_streams=performance_streams,
            onset_times=onset_times,
        )

        # Prepare SOS token
        tokens_tensor: TokenTensor = TokenTensor.from_tokens([[tk.StartOfScore()]])  # (1, 1, 110)

        if use_cache:
            past_key_values = None  # Initialize past key values for caching
        for _ in tqdm(range(max_length)):
            input_tokens = tokens_tensor[:, -1:] if use_cache else tokens_tensor
            decoder_outputs = model.forward_decoder(
                input_tokens=input_tokens,
                encoder_hidden_states=encoder_output,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
            )
            last_hidden_state = decoder_outputs.last_hidden_state[:, -1:]
            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            # Pass decoder outputs through the router
            token_type_logits = model.forward_router(last_hidden_state)
            token_type_logits = F.softmax(token_type_logits, dim=-1)
            if return_logits:
                logits.append(token_type_logits.squeeze(0, 1))

            # Get the predicted token types
            predicted_token_types_tensor = torch.argmax(token_type_logits, dim=-1)
            next_token_type = TokenTensor.TOKEN_INDEXER_INV[
                predicted_token_types_tensor.squeeze(0, 1).item()
            ]

            # Pass the decoder outputs through the correct expert head
            new_token_tensor = model.forward_experts(
                decoder_outputs=last_hidden_state,  # 1 * E
                token_type_masks={
                    next_token_type.as_str(): torch.ones(
                        1, 1, dtype=torch.bool, device=last_hidden_state.device
                    ),
                    **{
                        other_token.as_str(): torch.zeros(
                            1, 1, dtype=torch.bool, device=last_hidden_state.device
                        )
                        for other_token in TokenTensor.TOKEN_INDEXER_INV.values()
                        if other_token != next_token_type
                    },
                },
                convert_to_tokentensor=True,
            )
            new_token_tensor.squeeze(0, 1)  # Remove batch and sequence dimensions

            new_token_tensor = TokenTensor.from_tokens(new_token_tensor.to_tokens())

            tokens_tensor = torch.concat([tokens_tensor, new_token_tensor], dim=1)

            if isinstance(next_token_type, tk.EndOfScore):
                break

    # Convert the token tensor to a list of tokens
    tokens = tokens_tensor.squeeze(0).to_tokens()

    if return_logits:
        return tokens, torch.stack(logits, dim=0)
    else:
        return tokens


# Load the model
model: TokensTranscriberModule = TokensTranscriberModule.load_from_checkpoint(
    model_path,
    map_location="cpu",
)
model.global_metrics.train(mode=False)
model.eval()

# Load the dataloader
datamodule = PdmxTokensDataModule(
    data_dir=base_path,
    batch_size=25,
    num_workers=4,
    prefetch_factor=2,
    asap_path=base_path.parent / "asap-dataset",
)
datamodule.prepare_data()
datamodule.setup(stage="validate")
dataloader = datamodule.val_dataloader()

# Test on a single item
# for batch in dataloader:
#     for item in batch:
#         if item["score_tensor"]["voice_idx"][..., 4].sum() == 0:
#             print(f"Skipping empty score {item['entries'].xml_score_relative.as_posix()}")
#             continue  # Try to find a score with at least two hands
#         print(item["entries"].xml_score_relative.as_posix())
#         loss, results = model.common_step(item, train=False)
#         show_matrix(results["router_logits"].softmax(-1)[:50], colorbar=True)
#         tokens = transcribe_performance(
#             model.transcriber_model,
#             performance_streams=item["performance_streams"],
#             onset_times=item["performance_onset_times"],
#             max_length=512,
#             use_cache=True,
#         )
#         open_in_meld(
#             "\n".join([str(tk) for tk in item["score_tensor"].to_tokens()]),
#             "\n".join([str(tk) for tk in tokens]),
#         )
#         break
#     break


def compute_metrics(model, results: dict[str, Any]) -> dict[str, float]:
    metrics = {}

    router_logits = results["router_logits"]
    true_token_types = results["true_token_types"]
    score_not_padded_mask = results["score_not_padded_mask"]
    flattened_router_logits = results["flattened_router_logits"]
    flattened_true_token_types_ohe = results["flattened_true_token_types"]
    flattened_score_not_padded_mask = results["flattened_score_not_padded_mask"]

    flattened_pred_token_types = flattened_router_logits[flattened_score_not_padded_mask].argmax(-1)
    flattened_true_token_types = flattened_true_token_types_ohe[
        flattened_score_not_padded_mask
    ].argmax(-1)

    for name, metric in model.global_metrics.items():
        metrics[name] = metric(
            preds=flattened_pred_token_types,
            target=flattened_true_token_types,
        )

    return {k.split("/")[-1]: v.item() for k, v in metrics.items()}


# Evaluate on whole validation set
metrics_list = defaultlist(dict)
with torch.no_grad():
    for batch in (pbar := tqdm(dataloader)):
        loss, results = model.common_step(batch, train=False)
        current_metrics = compute_metrics(model, results)
        current_metrics["loss"] = loss.item()
        pbar.set_postfix(current_metrics)
        metrics_list.append(current_metrics)

# Aggregate metrics
metrics = {
    name: sum(m[name] for m in metrics_list) / len(metrics_list) for name in metrics_list[0].keys()
}
pprint(metrics)

# Plot loss
plt.plot([m["loss"] for m in metrics_list])
plt.show()

breakpoint()
