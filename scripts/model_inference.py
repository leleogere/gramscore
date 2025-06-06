"""
Try to load and run inference on a model
"""

import heapq
from collections.abc import Callable
from collections.abc import Generator
from functools import partial
from pathlib import Path

import torch
import symusic
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import partitura as pt
import partitura.score

from data.components.score_performance_dataset import ScorePerformanceEntry
from data.components.transforms import PerformanceReader
from data.components.transforms import PerformanceVectorizer
from models.automata.automata import ConstrainedAutomata
from models.automata.io import token_sequence_to_partitura
from models.automata.tokens import TranscriptionState
from models.automata.vectorization_torch import ProblemType
from models.automata.vectorization_torch import TokenTensor
from models.automata.vectorization_torch import clip_cast
from models.components.tokens_transcriber_model import TokensTranscriberModel
from models.tokens_transcriber_module import TokensTranscriberModule
from models.automata import tokens as tk
from utils.musicxml import ARTICULATIONS_AND_ORNAMENTS

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
    use_cache: bool = True,
) -> dict[str, torch.Tensor]:
    if sample_token_type:
        raise NotImplementedError("Sampling token types is not implemented yet.")  # TODO

    # Return logits if requested
    if return_logits:
        logits = []

    num_voices = 16
    automata = ConstrainedAutomata(num_voices=num_voices)
    tokens = [tk.StartOfScore()]  # Start with the SOS token
    automata.transition(tokens[0], inplace=True)
    with torch.no_grad():
        for key, value in performance_streams.items():
            performance_streams[key] = value.unsqueeze(0)

        # Pass performance streams through the encoder
        encoder_output = model.forward_encoder(input_streams=performance_streams)

        # Prepare SOS token
        tokens_tensor: TokenTensor = TokenTensor.from_tokens([[tk.StartOfScore()]])  # (1, 1, 110)

        # Prediction loop
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

            # Filter logits for the allowed token types
            allowed_token_types_idx = [
                TokenTensor.TOKEN_INDEXER[t]
                for t in automata.allowed_tokens_types()
                if t not in [tk.StartOfScore, tk.TupletStart, tk.TupletEnd]  # TODO: fix tuplets
            ]
            if not allowed_token_types_idx:
                raise RuntimeError("No token type allowed")
            token_type_logits = mask_unallowed(
                token_type_logits,
                allowed_indices=allowed_token_types_idx,
                mask_value=-1,
            )

            if return_logits:
                logits.append(token_type_logits.squeeze(0, 1))

            # Get the predicted token types
            predicted_token_types_tensor = torch.argmax(token_type_logits, dim=-1)
            next_token_type = TokenTensor.TOKEN_INDEXER_INV[
                predicted_token_types_tensor.squeeze(0, 1).item()
            ]

            # Pass the decoder outputs through the correct expert head
            expert_outputs = model.forward_experts(
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
                convert_to_tokentensor=False,
            )[next_token_type.as_str()]

            # Decode the expert output constrained by the automata
            next_token = decode_next_token_contrained(
                token_type=next_token_type,
                expert_outputs=expert_outputs,
                state=automata.state,
            )

            automata.transition(next_token, inplace=True)
            tokens.append(next_token)
            tokens_tensor = torch.concat(
                [tokens_tensor, TokenTensor.from_tokens([[next_token]])], dim=1
            )

            # print(next_token)
            # print(automata.state)

            if isinstance(next_token_type, tk.EndOfScore):
                break

    if return_logits:
        return tokens, torch.stack(logits, dim=0)
    else:
        return tokens


def mask_unallowed(
    tensor: torch.Tensor,
    allowed_indices: list[int],
    mask_value: float = -torch.inf,
) -> torch.Tensor:
    """
    Mask the tensor at the specified indices with the given mask value.
    """
    mask = torch.zeros(tensor.shape[-1], dtype=torch.bool, device=tensor.device)
    mask[allowed_indices] = True
    tensor[..., ~mask] = mask_value
    return tensor


def decode_next_token_contrained(
    token_type: type[tk.AnyToken],
    expert_outputs: dict[str, torch.Tensor],
    state: TranscriptionState,
) -> tk.AnyToken:
    # # StartOfScore
    # if token_type == tk.StartOfScore:
    #     return tk.StartOfScore()

    # First, we deal with all tokens without interactions between attributes:
    # EOS, Bar, KS, Clef, ChordEnd, TupletEnd

    # EndOfScore
    if token_type == tk.EndOfScore:
        return tk.EndOfScore()

    # Bar
    if token_type == tk.Bar:
        numerator = expert_outputs["bar_numerator"][0]
        denominator_idx = expert_outputs["bar_denominator"].argmax().item()
        denominator = TokenTensor.TS_DENOMINATOR_INDEXER_INV[denominator_idx]
        is_nominal = bool(expert_outputs["bar_is_nominal"][0] > 0.5)
        return tk.Bar(
            numerator=clip_cast(numerator, lower=1, type_=int),
            denominator=denominator,
            is_nominal=is_nominal,
        )

    # KeySignature
    if token_type == tk.KeySignature:
        fifths = expert_outputs["key_signature"].argmax().item() - 7
        return tk.KeySignature(accidentals=fifths)

    # Clef
    if token_type == tk.Clef:
        staff_idx = expert_outputs["staff_idx"].argmax().item()
        clef_idx = int(expert_outputs["clef_type"][0] > 0.5)
        clef_type = "G" if clef_idx == 0 else "F"
        return tk.Clef(clef=clef_type, staff_idx=staff_idx)

    # Get voices (all subsequent tokens should have a voice_idx)
    voice_outputs = expert_outputs["voice_idx"]

    # ChordEnd
    if token_type == tk.ChordEnd:
        # Find voices that are currently in a chord
        allowed_voices = [
            i for i, v in enumerate(state.voices) if v.in_chord and v.chord.can_resolve
        ]
        voice_idx = mask_unallowed(voice_outputs, allowed_indices=allowed_voices).argmax().item()
        return tk.ChordEnd(voice_idx=voice_idx)

    # TupletEnd
    if token_type == tk.TupletEnd:
        # Find voices that are currently in a tuplet
        allowed_voices = [i for i, v in enumerate(state.voices) if v.in_tuplet]
        voice_idx = mask_unallowed(voice_outputs, allowed_indices=allowed_voices).argmax().item()
        return tk.TupletEnd(voice_idx=voice_idx)

    # Now, we deal with tokens with interactions between attributes (for example, we can't
    # select the voice without it having an impact on the allowed durations).
    # We do that by computing the combination of attributes with the maximum probability,
    # while also being allowed by the automata

    # Note
    if token_type is tk.Note:
        classification_dimensions = [
            dim
            for dim in expert_outputs.keys()
            if TokenTensor.PROBLEM_TYPES[dim] != ProblemType.REGRESSION
        ]
        classification_distributions = [
            (
                F.softmax(value, dim=-1)
                if TokenTensor.PROBLEM_TYPES[name] == ProblemType.MULTI_CLASSIFICATION
                else torch.concat([(x := F.sigmoid(value)), 1 - x], axis=-1)  # binary classif.
            )
            for name, value in expert_outputs.items()
            if name in classification_dimensions
        ]

        def build_token(indices: tuple[int, ...]) -> tk.Note:
            dim_to_index = dict(zip(classification_dimensions, indices, strict=True))
            voice = dim_to_index["voice_idx"]
            staff = dim_to_index["staff_idx"]
            duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
            duration_dots = dim_to_index["duration_dots"]
            duration = tk.SymbolicDuration(type=duration_type, dots=duration_dots)
            step = TokenTensor.NOTE_INDEXER_INV[dim_to_index["note_class"]]
            octave = dim_to_index["note_octave"]
            accidental = dim_to_index["note_accidental"] - 2
            tied = dim_to_index["note_tied"] == 1
            stem_direction = TokenTensor.STEM_INDEXER_INV[dim_to_index["note_stem_direction"]]
            fermata = dim_to_index["fermata"] == 1
            arts_and_orns = {
                art_or_orn: dim_to_index[art_or_orn] for art_or_orn in ARTICULATIONS_AND_ORNAMENTS
            }
            return tk.Note(
                voice_idx=voice,
                staff_idx=staff,
                duration=duration,
                note_class=step,
                octave=octave,
                accidental=accidental,
                stem_direction=stem_direction,
                tied=tied,
                fermata=fermata,
                **arts_and_orns,
            )

        best_combination, proba = next(
            most_probable_combinations(
                distributions=[dist[0].tolist() for dist in classification_distributions],
                predicate=lambda x: build_token(x).is_token_allowed(state),
            )
        )

        best_token = build_token(best_combination)
        return best_token

    # Rest
    if token_type == tk.Rest:
        classification_dimensions = [
            dim
            for dim in expert_outputs.keys()
            if TokenTensor.PROBLEM_TYPES[dim] != ProblemType.REGRESSION
        ]
        classification_distributions = [
            (
                F.softmax(value, dim=-1)
                if TokenTensor.PROBLEM_TYPES[name] == ProblemType.MULTI_CLASSIFICATION
                else torch.concat([(x := F.sigmoid(value)), 1 - x], axis=-1)  # binary classif.
            )
            for name, value in expert_outputs.items()
            if name in classification_dimensions
        ]

        def build_token(indices: tuple[int, ...]) -> tk.Rest:
            dim_to_index = dict(zip(classification_dimensions, indices, strict=True))
            voice = dim_to_index["voice_idx"]
            staff = dim_to_index["staff_idx"]
            measure_rest = dim_to_index["measure_rest"] == 1
            if measure_rest:
                duration_type = tk.SymbolicDurationType.WHOLE
                duration_dots = 0
            else:
                duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
                duration_dots = dim_to_index["duration_dots"]
            duration = tk.SymbolicDuration(type=duration_type, dots=duration_dots)
            fermata = dim_to_index["fermata"] == 1
            return tk.Rest(
                voice_idx=voice,
                staff_idx=staff,
                duration=duration,
                measure_rest=measure_rest,
                fermata=fermata,
            )

        best_combination, proba = next(
            most_probable_combinations(
                distributions=[dist[0].tolist() for dist in classification_distributions],
                predicate=lambda x: build_token(x).is_token_allowed(state),
            )
        )

        best_token = build_token(best_combination)
        return best_token

    # ChordStart
    if token_type == tk.ChordStart:
        classification_dimensions = [
            dim
            for dim in expert_outputs.keys()
            if TokenTensor.PROBLEM_TYPES[dim] != ProblemType.REGRESSION
        ]
        classification_distributions = [
            (
                F.softmax(value, dim=-1)
                if TokenTensor.PROBLEM_TYPES[name] == ProblemType.MULTI_CLASSIFICATION
                else torch.concat([(x := F.sigmoid(value)), 1 - x], axis=-1)  # binary classif.
            )
            for name, value in expert_outputs.items()
            if name in classification_dimensions
        ]

        def build_token(indices: tuple[int, ...]) -> tk.ChordStart:
            dim_to_index = dict(zip(classification_dimensions, indices, strict=True))
            voice = dim_to_index["voice_idx"]
            duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
            duration_dots = dim_to_index["duration_dots"]
            return tk.ChordStart(
                voice_idx=voice,
                duration=tk.SymbolicDuration(type=duration_type, dots=duration_dots),
            )

        best_combination, proba = next(
            most_probable_combinations(
                distributions=[dist[0].tolist() for dist in classification_distributions],
                predicate=lambda x: build_token(x).is_token_allowed(state),
            )
        )

        best_token = build_token(best_combination)
        return best_token

    # TupletStart
    if token_type == tk.TupletStart:
        raise NotImplementedError("TupletStart not implemented")
        classification_dimensions = [
            dim
            for dim in expert_outputs.keys()
            if TokenTensor.PROBLEM_TYPES[dim] != ProblemType.REGRESSION
        ]
        classification_distributions = [
            (
                F.softmax(value, dim=-1)
                if TokenTensor.PROBLEM_TYPES[name] == ProblemType.MULTI_CLASSIFICATION
                else torch.concat([(x := F.sigmoid(value)), 1 - x], axis=-1)  # binary classif.
            )
            for name, value in expert_outputs.items()
            if name in classification_dimensions
        ]

        def build_token(indices: tuple[int, ...]) -> tk.Rest:
            dim_to_index = dict(zip(classification_dimensions, indices, strict=True))
            voice = dim_to_index["voice_idx"]
            staff = dim_to_index["staff_idx"]
            measure_rest = dim_to_index["measure_rest"] == 1
            if measure_rest:
                duration_type = tk.SymbolicDurationType.WHOLE
                duration_dots = 0
            else:
                duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
                duration_dots = dim_to_index["duration_dots"]
            duration = tk.SymbolicDuration(type=duration_type, dots=duration_dots)
            fermata = dim_to_index["fermata"] == 1
            return tk.Rest(
                voice_idx=voice,
                staff_idx=staff,
                duration=duration,
                measure_rest=measure_rest,
                fermata=fermata,
            )

        best_combination, proba = next(
            most_probable_combinations(
                distributions=[dist[0].tolist() for dist in classification_distributions],
                predicate=lambda x: build_token(x).is_token_allowed(state),
            )
        )

        best_token = build_token(best_combination)
        return best_token
        raise NotImplementedError("TupletStart not implemented")
        n_actual = clip_cast(self["tuplet_actual_notes"][0], 1, type_=int)
        n_normal = clip_cast(self["tuplet_normal_notes"][0], 1, type_=int)
        return tk.TupletStart(
            n_actual=n_actual,
            n_normal=n_normal,
            unit_duration=tk.SymbolicDuration(
                type=tk.SymbolicDurationType(duration_type),
                dots=duration_dots,
            ),
            voice_idx=voice_idx,
        )

    raise ValueError(f"Unknown token type {token_type}")


def convert_binary_distributions_into_multiclass_distribution(
    distributions: list[list[float], float],
) -> list[list[float]]:
    for i, distribution in enumerate(distributions):
        if isinstance(distribution, float):
            distributions[i] = [distribution, 1 - distribution]
    return distributions


def most_probable_combinations(
    distributions: list[list[float]], predicate: Callable[[float], bool] | None = None
) -> Generator[tuple[tuple[int, ...], float], None, None]:
    # Validate distributions
    for distribution in distributions:
        if not isinstance(distribution, list):
            raise ValueError(
                f"Expected distribution to be a list of floats, got {type(distribution)}"
            )
        if not all(0 <= p <= 1 for p in distribution):
            raise ValueError(f"All probabilities must lay between 0 and 1: {distribution}")

    # Add indices to each distribution and sort them by decreasing probabilities
    # (so that we first explore highly probable combinations)
    sorted_idxs = [
        sorted(range(len(dist)), key=lambda i: dist[i], reverse=True) for dist in distributions
    ]
    n = len(distributions)

    # Function to compute the probability of a combination
    def proba_combination(indices: tuple[int, ...]) -> float:
        p = 1.0
        for dim, pos in enumerate(indices):
            orig_idx = sorted_idxs[dim][pos]
            p *= distributions[dim][orig_idx]
        return p

    # Create the max-heap (min-heap with negative probabilities)
    heap: list[tuple[float, tuple[int, ...]]] = []

    # Keep track of already visited combinations
    visited = set()

    # We start the heap with the most probable item
    # (index 0 in all sorted distributions)
    init = tuple([0] * n)
    init_prob = proba_combination(init)
    heapq.heappush(heap, (-init_prob, init))
    visited.add(init)

    # While there are still items in the heap, yield them
    while heap:
        # Get most probable combination
        neg_p, sorted_tuple = heapq.heappop(heap)
        p = -neg_p
        # Map sorted‐space positions to original indices
        orig_tuple = tuple(sorted_idxs[dim][pos] for dim, pos in enumerate(sorted_tuple))
        # If the predicate is true, yield the combination
        if predicate(orig_tuple):
            yield orig_tuple, p
        # Generate neighbors by incrementing the position in each dimension
        for dim in range(n):
            if sorted_tuple[dim] + 1 < len(sorted_idxs[dim]):
                neighbor = list(sorted_tuple)
                neighbor[dim] += 1
                neighbor = tuple(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    heapq.heappush(heap, (-proba_combination(neighbor), neighbor))


performance = load_performance(performance_path, num_notes=500)
transcribed_tokens, logits = transcribe_performance(
    model.transcriber_model,
    performance,
    max_length=100,
    return_logits=True,
    use_cache=True,
)
logits = logits.numpy()
logits[logits == -1] = None

# Plot a heatmap of the logits
plt.figure(figsize=(len(logits) // 5, 8))
plt.imshow(logits.T, cmap='viridis')
plt.yticks(
    range(len(TokenTensor.TOKEN_INDEXER)),
    [model._token_str_mapping[t] for t in TokenTensor.TOKEN_INDEXER.keys()],
)
plt.colorbar()
plt.ylabel('Token Types')
plt.xlabel('Time Steps')
plt.title('Logits Heatmap')
plt.show()


for token in transcribed_tokens:
    print(token)

score = token_sequence_to_partitura(transcribed_tokens)

pt.save_musicxml(score, Path("/tmp/transcriber.musicxml"))
# print(model)
