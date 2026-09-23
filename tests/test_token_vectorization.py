"""
Test the tokens vectorization process
"""

import pytest
import torch

import models.automata.tokens as tk
from models.automata.vectorization_torch import TokenTensor


@pytest.fixture
def tokens_tensor():
    torch.random.manual_seed(42)
    tensor = TokenTensor(torch.randn(5, 10, TokenTensor.TOTAL_DIMENSIONS))
    return tensor


@pytest.fixture
def tokens():
    tokens = [
        tk.StartOfScore(),
        tk.Bar(numerator=6, denominator=8, is_nominal=True),
        tk.KeySignature(accidentals=-2),
        tk.Clef(clef="G", staff_idx=1),
        tk.TupletStart(
            n_actual=3,
            n_normal=2,
            unit_duration=tk.SymbolicDuration(type=tk.SymbolicDurationType.QUARTER, dots=0),
            voice_idx=1,
        ),
        tk.ChordStart(
            voice_idx=1, duration=tk.SymbolicDuration(type=tk.SymbolicDurationType.HALF, dots=2)
        ),
        tk.Note(
            note_class="C",
            octave=4,
            accidental=0,
            tied=False,
            stem_direction="up",
            staff_idx=1,
            voice_idx=5,
            duration=tk.SymbolicDuration(type=tk.SymbolicDurationType.EIGHTH, dots=0),
        ),
        tk.Note(
            note_class="D",
            octave=4,
            accidental=0,
            tied=False,
            stem_direction="up",
            staff_idx=1,
            voice_idx=5,
            duration=tk.SymbolicDuration(type=tk.SymbolicDurationType.EIGHTH, dots=0),
        ),
        tk.ChordEnd(voice_idx=3),
        tk.Rest(
            staff_idx=0,
            voice_idx=1,
            duration=tk.SymbolicDuration(type=tk.SymbolicDurationType.WHOLE, dots=1),
        ),
        tk.TupletEnd(voice_idx=1),
        tk.EndOfScore(),
    ]
    return tokens


def test_getattr(tokens_tensor: TokenTensor):
    """
    Test if the TokenTensor can be indexed by attribute names.
    """
    assert tokens_tensor["token_type"].shape == (5, 10, 11)
    assert tokens_tensor["bar_time_signature"].shape == (5, 10, 180)
    assert tokens_tensor["bar_is_nominal"].shape == (5, 10, 2)
    assert tokens_tensor["key_signature"].shape == (5, 10, 15)
    assert tokens_tensor["clef_type"].shape == (5, 10, 2)
    assert tokens_tensor["staff_idx"].shape == (5, 10, 2)
    assert tokens_tensor["voice_idx"].shape == (5, 10, 8)
    assert tokens_tensor["duration_type"].shape == (5, 10, 11)
    assert tokens_tensor["duration_dots"].shape == (5, 10, 4)
    assert tokens_tensor["note_class"].shape == (5, 10, 7)
    assert tokens_tensor["note_octave"].shape == (5, 10, 9)
    assert tokens_tensor["note_accidental"].shape == (5, 10, 5)
    assert tokens_tensor["note_tied"].shape == (5, 10, 2)
    assert tokens_tensor["note_stem_direction"].shape == (5, 10, 3)
    assert tokens_tensor["tuplet_actual_notes"].shape == (5, 10, 1)
    assert tokens_tensor["tuplet_normal_notes"].shape == (5, 10, 1)

    # Test slicing
    assert tokens_tensor[0, :5, "token_type"].shape == (5, 11)
    assert tokens_tensor[0, :5, "tuplet_actual_notes"].shape == (5, 1)
    assert tokens_tensor[0, 5, "tuplet_actual_notes"].shape == torch.Size([1])


def test_setattr(tokens_tensor: TokenTensor):
    """
    Test if the TokenTensor can be indexed by attribute names.
    """
    tokens_tensor["note_tied"] = 1
    assert torch.all(tokens_tensor["note_tied"] == 1)

    tokens_tensor[0, :5, "note_class"] = torch.randn((5, 7))
    with pytest.raises(RuntimeError):
        tokens_tensor[0, :5, "note_class"] = torch.randn((5, 8))


def test_encode_nested(tokens: list[list[tk.Token]]):
    """
    Test if encoding a nested list of tokens results in a TokenTensor of the correct shape.
    """
    nested_tokens = [tokens for _ in range(3)]
    nested_tokens = [nested_tokens for _ in range(2)]
    tensor = TokenTensor.from_tokens(nested_tokens)
    assert tensor.shape == (2, 3, len(tokens), TokenTensor.TOTAL_DIMENSIONS)


def test_decode_nested(tokens_tensor: TokenTensor):
    """
    Test if decoding a TokenTensor of arbitrary shape correctly returns a nested list of tokens.
    """
    tokens = tokens_tensor.to_tokens()
    assert len(tokens) == tokens_tensor.shape[0]
    assert all(len(token) == tokens_tensor.shape[1] for token in tokens)


def test_encode_decode_simple(tokens: list[tk.Token]):
    """
    Test if encoding and decoding a sequence of tokens results in the same sequence.
    """
    tensor = TokenTensor.from_tokens(tokens)
    decoded_tokens = tensor.to_tokens()
    for original, decoded in zip(tokens, decoded_tokens):
        assert original == decoded


def test_encode_decode_stress_test():
    """
    Test if encoding and decoding a sequence of tokens results in the same sequence.
    """
    torch.manual_seed(42)
    random_tensor = TokenTensor(torch.randn(3, 20, 100, TokenTensor.TOTAL_DIMENSIONS))
    tokens = random_tensor.to_tokens()

    tokens_tensor = TokenTensor.from_tokens(tokens)
    decoded_tokens = tokens_tensor.to_tokens()

    def recursive_test_encode_decode(t_original: list | tk.Token, t_decoded: list | tk.Token):
        if isinstance(t_original, list):
            assert isinstance(t_decoded, list)
            assert len(t_original) == len(t_decoded)
            for original, decoded in zip(t_original, t_decoded):
                recursive_test_encode_decode(original, decoded)
        else:
            assert t_original == t_decoded

    recursive_test_encode_decode(tokens, decoded_tokens)


def test_quantize():
    torch.manual_seed(42)
    tensor = TokenTensor(torch.randn(3, 20, 100, TokenTensor.TOTAL_DIMENSIONS))
    quantized = tensor.quantize()
    reconstructed = TokenTensor.from_tokens(tensor.to_tokens())
    assert torch.all(quantized == reconstructed)
