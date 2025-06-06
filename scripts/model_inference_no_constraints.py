"""
Try to load and run inference on a model
"""

from pathlib import Path

import symusic
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.components.score_performance_dataset import ScorePerformanceEntry
from data.components.transforms import PerformanceReader
from data.components.transforms import PerformanceVectorizer
from models.automata.automata import ConstrainedAutomata
from models.automata.tokens import TranscriptionState
from models.automata.vectorization_torch import TokenTensor
from models.automata.vectorization_torch import clip_cast
from models.components.tokens_transcriber_model import TokensTranscriberModel
from models.tokens_transcriber_module import TokensTranscriberModule
from models.automata import tokens as tk

# fmt: off
model_path = Path("~/Documents/transcription/gramscore/outputs/server/2025-05-27/15-12-11/checkpoints/last.ckpt").expanduser()
performance_path = Path("~/Documents/datasets/asap-dataset/Bach/Prelude/bwv_860/YoungS01M.mid").expanduser()
# fmt: on

# Load the model
model: TokensTranscriberModule = TokensTranscriberModule.load_from_checkpoint(
    model_path,
    map_location="cpu",
)
model.eval()


# Load the performance
def load_performance(performance_path: Path, num_notes: int = -1) -> dict[str, torch.Tensor]:
    if not performance_path.exists():
        raise FileNotFoundError(f"Performance file {performance_path} does not exist.")
    performance_metadata = ScorePerformanceEntry(xml_score=None, midi_performance=performance_path)
    performance_reader = PerformanceReader()
    performance_vectorizer = PerformanceVectorizer()
    performance: symusic.core.ScoreSecond = performance_reader({"entry": performance_metadata})[
        "performance"
    ]
    if num_notes > 0:
        performance.clip(0, performance.tracks[0].notes[num_notes - 1].end, inplace=True)
        pianoroll = performance.pianoroll(modes=["frame"], pitch_range=(21, 108))
    plt.imshow(
        pianoroll.squeeze(), aspect='auto', cmap='gray_r', origin='lower', interpolation='none'
    )
    plt.show()
    vectorized_performance = performance_vectorizer({"performance": performance})[
        "performance_streams"
    ]
    return vectorized_performance


# Inference function
def transcribe_performance(
    model: TokensTranscriberModel,
    performance_streams: dict[str, torch.Tensor],
    sample_token_type: bool = False,  # TODO: implement sampling
    max_length: int = 1024,
    return_logits: bool = False,
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
        for key, value in performance_streams.items():
            performance_streams[key] = value.unsqueeze(0)

        # Pass performance streams through the encoder
        encoder_output = model.forward_encoder(input_streams=performance_streams)

        # Prepare SOS token
        tokens_tensor: TokenTensor = TokenTensor.from_tokens([[tk.StartOfScore()]])  # (1, 1, 110)

        past_key_values = None  # Initialize past key values for caching
        for _ in tqdm(range(max_length)):
            decoder_outputs = model.forward_decoder(
                input_tokens=tokens_tensor[:, -1:],
                encoder_hidden_states=encoder_output,
                past_key_values=past_key_values,  # TODO: Implement CV caching
                use_cache=True,  # TODO: Implement CV caching
            )
            last_hidden_state = decoder_outputs.last_hidden_state[:, -1:]
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

            tokens_tensor = torch.concat([tokens_tensor, new_token_tensor], dim=1)

            if isinstance(next_token_type, tk.EndOfScore):
                break

    # Convert the token tensor to a list of tokens
    tokens = tokens_tensor.squeeze(0).to_tokens()

    if return_logits:
        return tokens, torch.stack(logits, dim=0)
    else:
        return tokens


performance = load_performance(performance_path, num_notes=50)
transcribed_tokens, logits = transcribe_performance(
    model.transcriber_model,
    performance,
    max_length=20,
    return_logits=True,
)

import matplotlib.pyplot as plt

# Plot a heatmap of the logits
plt.imshow(logits.numpy().T, aspect='auto', cmap='viridis')
plt.yticks(
    range(len(TokenTensor.TOKEN_INDEXER)),
    [model._token_str_mapping[t] for t in TokenTensor.TOKEN_INDEXER.keys()],
)
plt.colorbar()
plt.ylabel('Token Types')
plt.xlabel('Time Steps')
plt.title('Logits Heatmap')
plt.show()


# print(model)
for t in transcribed_tokens:
    print(t)
