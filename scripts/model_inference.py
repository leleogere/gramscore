"""
Try to load and run inference on a model
"""

import pretty_midi as pm

import heapq
from collections.abc import Callable
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pretty_midi
import torch
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
import torch.nn.functional as F
import partitura as pt

from data.components.score_performance_dataset import ScorePerformanceEntry
from data.components.transforms import PerformanceReader
from data.components.transforms import PerformanceVectorizer
from models.automata.automata import ConstrainedAutomata
from models.automata.io import token_sequence_to_partitura
from models.automata.io import token_sequence_to_partitura_permissive
from models.automata.tokens import TranscriptionState
from models.automata.vectorization_torch import ProblemType
from models.automata.vectorization_torch import TokenTensor
from models.automata.vectorization_torch import clip
from models.components.tokens_transcriber_model import TokensTranscriberModel
from models.tokens_transcriber_module import TokensTranscriberModule
from models.automata import tokens as tk
from utils.debug import concat_streams, write_compare_to_file, as_pycharm_link
from utils.debug import open_in_midieditor
from utils.debug import open_in_musescore
from utils.musicxml import ARTICULATIONS_AND_ORNAMENTS

# fmt: off
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-05-27_15-12-11/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-06-11_22-43-52/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-06-27_00-10-57/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-09-09_23-50-39/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/2025-09-19_14-15-04_cosmic-totem-262/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/2025-09-22_09-51-15_jumping-shape-264/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/2025-09-23_15-09-32_grateful-frog-267_vocal-morning-268/checkpoints/last_vocal-morning-268.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/dry-planet-265.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/last-v1.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/rosy-salad-273_partial.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-11-03_12-53-43/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-11-03_12-53-43/checkpoints/epoch_060.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/2025-11-06_18-05-07/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/2025-11-07_09-30-46/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-11-12_11-37-22_leafy-serinity-289/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/2025-11-14_10-23-42_kind-universe-291/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/JZ/2025-11-14_17-13-25_noble-forest-293/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-11-28_22-23-03_clean-salad-312/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-11-28_20-21-02_scarlet-pyramid-311/checkpoints/last.ckpt").expanduser()
# model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-12-02_15-26-21_clean-salad-313/checkpoints/last.ckpt").expanduser()
model_path = Path("~/Documents/transcription/gramscore_old/outputs/server/2025-12-09_13-57-48_sage-serenity-318/checkpoints/last.ckpt").expanduser()

# base_path = Path("~/Documents/datasets/asap-dataset").expanduser()
# performance_path = base_path / "Bach/Prelude/bwv_860/YoungS01M.mid"
# performance_path = base_path / "Bach/Prelude/bwv_860/midi_score.mid"
# performance_path = base_path / "Bach/Prelude/bwv_883/LuM01M.mid"
# performance_path = base_path / "Bach/Prelude/bwv_883/midi_score.mid"

base_path = Path("~/Documents/datasets/pdmx").expanduser()
# performance_path = base_path / "mid/0/0/Qma1a1L2c1vSEuGfdPVSEEWfuYAEoBmjpVESX1ok2imu4b.mid"
# performance_path = base_path / "mid/0/0/Qma1a2YHEJ8j9gxLCxgALAfjYaiYJQogh1bgyFSwNSRPCs.mid"  # Polyphonic, long
performance_path = base_path / "mid/0/39/QmaQ5PSCje4wFnvkjYZscvC7sQo46M7UcDZUzjPspUvQMZ.mid"  # test strict
# performance_path = base_path / "mid/1/11/Qmbbh58jD1ZLT1vevNsnyEZkexnx2USTFz6FSHX3KUiAzC.mid"  # Monophonic
# performance_path = base_path / "mid/15/12/QmXB8B4F9cbHUrnrP4mRGyL3vyhTkyDjuVEbM3VdFu2EY9.mid"  # Polyphonic, chords
# performance_path = base_path / "mid/1/32/QmbMQQPWGZpGiTV5gngSwmjTbqd7XosW8yyPwkSHnXbhY2.mid"  # Polyphonic, chords
reference_score_path = base_path / "mxl" / performance_path.relative_to(base_path / "mid").with_suffix(".mxl")
print(f"Reference score path: {reference_score_path.as_posix()}")
print(f"Performance path: {performance_path.as_posix()}")
# fmt: on

MAX_PERFORMED_NOTES = 500
MAX_GENERATED_TOKENS = 400

# Load the model
model: TokensTranscriberModule = TokensTranscriberModule.load_from_checkpoint(
    model_path,
    map_location="cpu",
    weights_only=False,
)
model.eval()


# Load the performance
def load_performance(performance_path: Path, num_notes: int = -1) -> dict[str, torch.Tensor]:
    if not performance_path.exists():
        raise FileNotFoundError(f"Performance file {performance_path} does not exist.")
    performance_metadata = ScorePerformanceEntry(xml_score=None, midi_performance=performance_path)
    performance_reader = PerformanceReader()
    performance_vectorizer = PerformanceVectorizer()
    performance: pretty_midi.PrettyMIDI = performance_reader({"entry": performance_metadata})[
        "performance"
    ]
    if num_notes > 0:
        print(len(performance.instruments[0].notes))
        num_notes = min(num_notes, len(performance.instruments[0].notes))
        # performance.clip(0, performance.instruments[0].notes[num_notes - 1].end, inplace=True)
        performance.instruments[0].notes = performance.instruments[0].notes[:num_notes]
    #     pianoroll = performance.pianoroll(modes=["frame"], pitch_range=(21, 108))
    # plt.imshow(
    #     pianoroll.squeeze(), aspect='auto', cmap='gray_r', origin='lower', interpolation='none'
    # )
    # plt.show()
    plt.matshow(performance.get_piano_roll())
    plt.show()
    vectorized_performance = performance_vectorizer({"performance": performance})
    return vectorized_performance


# Inference function
def transcribe_performance(
    model: TokensTranscriberModel,
    performance_streams: dict[str, torch.Tensor],
    onset_times: torch.Tensor,
    sample_token_type: bool = False,  # TODO: implement sampling
    max_length: int = 1024,
    constraint: bool = True,
    return_logits_and_mask: bool = False,
    use_cache: bool = True,
    allowed_pitches: set[int] = None,
    force_first_token: tk.AnyToken | None = None,
) -> TokenTensor | tuple[TokenTensor, torch.Tensor, torch.Tensor]:
    if sample_token_type:
        raise NotImplementedError("Sampling token types is not implemented yet.")  # TODO

    # Return logits if requested
    if return_logits_and_mask:
        logits_and_masks = []

    tokens = [tk.StartOfScore()]  # Start with the SOS token

    if constraint:
        num_voices = 16
        automata = ConstrainedAutomata(num_voices=num_voices, allowed_pitches=allowed_pitches)
        automata.transition(tokens[0], inplace=True)

    with torch.no_grad():
        for key, value in performance_streams.items():
            performance_streams[key] = value.unsqueeze(0)

        # Pass performance streams through the encoder
        encoder_output = model.forward_encoder(
            input_streams=performance_streams, onset_times=onset_times
        )

        # Prepare SOS token
        tokens_tensor: TokenTensor = TokenTensor.from_tokens([[tk.StartOfScore()]])  # (1, 1, 110)

        # Prediction loop
        if use_cache:
            past_key_values = None  # Initialize past key values for caching
        for i in (pbar := tqdm(range(max_length))):

            # Force the first token if specified (and pass thought model to update past_key_values)
            if force_first_token is not None and i == 0:
                next_token = force_first_token
                if constraint:
                    automata.transition(next_token, inplace=True)
                tokens_tensor = torch.concat(
                    [tokens_tensor, TokenTensor.from_tokens([[next_token]])], dim=1
                )
                tokens.append(next_token)
                if use_cache:
                    # QUESTION: Why doesn't it work in a single pass?
                    decoder_outputs = model.forward_decoder(
                        input_tokens=tokens_tensor[:, :0],
                        encoder_hidden_states=encoder_output,
                        past_key_values=None,
                        use_cache=True,
                    )
                    decoder_outputs = model.forward_decoder(
                        input_tokens=tokens_tensor[:, :-1],
                        encoder_hidden_states=encoder_output,
                        past_key_values=decoder_outputs.past_key_values,
                        use_cache=True,
                    )
                    past_key_values = decoder_outputs.past_key_values
                continue

            input_tokens = tokens_tensor[:, -1:] if use_cache else tokens_tensor
            write_compare_to_file(input_tokens, f"input_tokens_{i}.txt")
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
            model_want_to_predict = torch.argmax(token_type_logits, dim=-1)

            # Filter logits for the allowed token types
            if constraint:
                allowed_token_types_idx = [
                    TokenTensor.TOKEN_INDEXER[t]
                    for t in automata.allowed_tokens_types()
                    if t
                    not in [tk.StartOfScore]  # , tk.TupletStart, tk.TupletEnd]  # TODO: fix tuplets
                ]
                if not allowed_token_types_idx:
                    raise RuntimeError("No token type allowed")
                token_type_logits_masked = mask_unallowed(
                    token_type_logits.clone(),  # Clone only for debugging purposes to not change the initial one
                    allowed_indices=allowed_token_types_idx,
                    mask_value=-1,
                )
            else:
                token_type_logits_masked = token_type_logits
                allowed_token_types_idx = list(range(len(tk.TOKEN_TYPES)))

            if return_logits_and_mask:
                mask = torch.zeros(
                    token_type_logits.shape[-1], dtype=torch.bool, device=token_type_logits.device
                )
                mask[allowed_token_types_idx] = True
                logits_and_masks.append([token_type_logits.squeeze(0, 1), mask])

            # Get the predicted token types
            predicted_token_types_tensor = torch.argmax(token_type_logits_masked, dim=-1)
            next_token_type = TokenTensor.TOKEN_INDEXER_INV[
                predicted_token_types_tensor.squeeze(0, 1).item()
            ]
            if predicted_token_types_tensor != model_want_to_predict:
                logger.warning(
                    f"The models wants to predict a token "
                    f"{TokenTensor.TOKEN_INDEXER_INV[model_want_to_predict.squeeze(0, 1).item()].as_str()}, "
                    f"but is forced to produce a token {next_token_type.as_str()} because of the current "
                    f"state:\n" + str(automata.state)
                )

            # Pass the decoder outputs through the correct expert head
            token_types_mask = {
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
            }
            if constraint:
                expert_outputs = model.forward_experts(
                    decoder_outputs=last_hidden_state,  # 1 * E
                    token_type_masks=token_types_mask,
                    convert_to_tokentensor=False,
                ).get(next_token_type.as_str(), None)
                next_token = decode_next_token_contrained(
                    token_type=next_token_type,
                    expert_outputs=expert_outputs,
                    state=automata.state,
                )
                automata.transition(next_token, inplace=True)
            else:
                next_token_tensor = model.forward_experts(
                    decoder_outputs=last_hidden_state,  # 1 * E
                    token_type_masks=token_types_mask,
                    convert_to_tokentensor=True,
                )
                next_token = next_token_tensor.to_tokens()[0][0]

            # pbar.write(str(next_token))

            tokens.append(next_token)
            tokens_tensor = torch.concat(
                [tokens_tensor, TokenTensor.from_tokens([[next_token]])], dim=1
            )

            # print(next_token)
            # print(automata.state)

            if isinstance(next_token, tk.EndOfScore):
                break

    if return_logits_and_mask:
        logits, masks = zip(*logits_and_masks)
        return (tokens, torch.stack(logits, dim=0), torch.stack(masks, dim=0))
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
        # numerator = expert_outputs["bar_numerator"][0]
        # denominator_idx = expert_outputs["bar_denominator"].argmax().item()
        # denominator = TokenTensor.TS_DENOMINATOR_INDEXER_INV[denominator_idx]
        time_signature_idx = expert_outputs["bar_time_signature"].argmax().item()
        numerator, denominator = TokenTensor.TS_INDEXER_INV[time_signature_idx]
        is_nominal = bool(expert_outputs["bar_is_nominal"].argmax().item())
        return tk.Bar(
            # numerator=round(clip(numerator.item(), lower=1)),
            numerator=numerator,
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
        clef_idx = expert_outputs["clef_type"].argmax().item()
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
            if TokenTensor.PROBLEM_TYPES[dim] == ProblemType.CLASSIFICATION
        ]
        classification_distributions = [
            F.softmax(value, dim=-1)
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
                art_or_orn: dim_to_index[art_or_orn] == 1
                for art_or_orn in ARTICULATIONS_AND_ORNAMENTS
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
            F.softmax(value, dim=-1)
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
            F.softmax(value, dim=-1)
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
        # raise NotImplementedError("TupletStart not implemented")
        classification_dimensions = [
            dim
            for dim in expert_outputs.keys()
            if TokenTensor.PROBLEM_TYPES[dim] != ProblemType.REGRESSION
        ]
        classification_distributions = [
            (
                F.softmax(value, dim=-1)
                if TokenTensor.PROBLEM_TYPES[name] == ProblemType.CLASSIFICATION
                # TODO: remove binary classification as everything is now multiclass
                else torch.concat([(x := F.sigmoid(value)), 1 - x], axis=-1)  # binary classif.
            )
            for name, value in expert_outputs.items()
            if name in classification_dimensions
        ]

        def build_token(indices: tuple[int, ...]) -> tk.TupletStart:
            dim_to_index = dict(zip(classification_dimensions, indices, strict=True))
            voice = dim_to_index["voice_idx"]
            duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
            duration_dots = dim_to_index["duration_dots"]
            n_actual = round(clip(expert_outputs["tuplet_actual_notes"].item(), 1))
            n_normal = round(clip(expert_outputs["tuplet_normal_notes"].item(), 1))
            return tk.TupletStart(
                n_actual=n_actual,
                n_normal=n_normal,
                unit_duration=tk.SymbolicDuration(
                    type=tk.SymbolicDurationType(duration_type),
                    dots=duration_dots,
                ),
                voice_idx=voice,
            )

        best_combination, proba = next(
            most_probable_combinations(
                distributions=[dist[0].tolist() for dist in classification_distributions],
                predicate=lambda x: build_token(x).is_token_allowed(state),
            )
        )

        best_token = build_token(best_combination)
        return best_token

    elif token_type == tk.TupletEnd:
        # Find voices that are currently in a tuplet
        allowed_voices = [i for i, v in enumerate(state.voices) if v.in_tuplet]
        voice_idx = mask_unallowed(voice_outputs, allowed_indices=allowed_voices).argmax().item()
        return tk.TupletEnd(voice_idx=voice_idx)

    raise ValueError(f"Unknown token type {token_type}")


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


performance_vectorized = load_performance(performance_path, num_notes=MAX_PERFORMED_NOTES)
for k, v in performance_vectorized["performance_streams"].items():
    print(f"{k}: {v.shape}")

# Plot performance streams
# plt.figure(figsize=(12, 8))
plt.matshow(concat_streams(performance_vectorized["performance_streams"], separate_by_none=True))
plt.tight_layout()
plt.show()

# Transcribe performance
force_first_token = None
# force_first_token = tk.Bar(numerator=3, denominator=4, is_nominal=True)
use_cache = True
constraint = True
transcribed_tokens, logits, masks = transcribe_performance(
    model.transcriber_model,
    performance_vectorized["performance_streams"],
    onset_times=performance_vectorized["performance_onset_times"].unsqueeze(0),
    max_length=MAX_GENERATED_TOKENS,
    return_logits_and_mask=True,
    constraint=constraint,
    use_cache=use_cache,
    force_first_token=force_first_token,
    allowed_pitches=set(
        n.pitch for n in performance_vectorized["performance"].instruments[0].notes
    ),
)
logits = logits.numpy()
masks = masks.numpy()
# logits[logits == -1] = None

# Plot a heatmap of the logits
plt.figure(figsize=(len(logits) // 5, 4))
plt.imshow(logits.T, cmap='viridis')
plt.yticks(
    range(len(TokenTensor.TOKEN_INDEXER)),
    [model._token_str_mapping[t] for t in TokenTensor.TOKEN_INDEXER.keys()],
)
ys, xs = np.where(~masks.T)
plt.plot(xs, ys, "x", color='red')
# plt.colorbar()
plt.ylabel('Token Types')
plt.xlabel('Time Steps')
plt.title('Logits Heatmap')
plt.tight_layout()
plt.show()


for token in transcribed_tokens:
    print("  * " + str(token))
# Save tokens
with open(
    Path(
        f"/tmp/transcribed_tokens_forced={force_first_token is not None}_cache={use_cache}_constraint={constraint}.txt"
    ),
    "w",
) as f:
    for token in transcribed_tokens:
        f.write(str(token) + "\n")

if constraint:
    score = token_sequence_to_partitura(transcribed_tokens)
else:
    score = token_sequence_to_partitura_permissive(transcribed_tokens)

file_name = (performance_path.relative_to(base_path).with_suffix(".musicxml").as_posix()).replace(
    "/", "_"
)
file_path = Path("/tmp") / file_name
pt.save_musicxml(score, file_path)
# print(model)

# performance_vectorized["performance"].write("/tmp/cropped.mid")
# open_in_midieditor(Path("/tmp/cropped.mid"))
# open_in_musescore(file_path)
print(f" Performance MIDI: {as_pycharm_link(file_path)}")
print(f"  Reference score: {as_pycharm_link(reference_score_path)}")
print(f"Transcribed score: {as_pycharm_link(file_path)}")
# Only open the reference score if there is it is not already loaded by an existing MuseScore instance
# open_in_musescore(reference_score_path, only_if_not_open=True)
