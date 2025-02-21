import itertools
from fractions import Fraction
from typing import Self

import torch

from src.models.automata import tokens as tk


def clip_cast(x, lower=None, upper=None, type_=None):
    assert lower is None or upper is None or lower <= upper
    if lower is not None:
        x = max(x, lower)
    if upper is not None:
        x = min(x, upper)
    if type_ is not None:
        x = type_(x)
    return x


class TokenTensor(torch.Tensor):
    """
    This class represent Token objects in their tensor form.
    The last dimension of this tensor should always be of dimensions `TokenTensor.TOTAL_DIMENSIONS`,
    the other dimensions are arbitrary batch dimensions.
    """

    TOKEN_INDEXER = {token: i for i, token in enumerate(tk.TOKEN_TYPES)}
    TOKEN_INDEXER_INV = {i: token for token, i in TOKEN_INDEXER.items()}
    TS_DENOMINATOR_INDEXER = {den: i for i, den in enumerate([1, 2, 4, 8, 16, 32])}
    TS_DENOMINATOR_INDEXER_INV = {i: den for den, i in TS_DENOMINATOR_INDEXER.items()}
    DURATION_INDEXER = {dur.value: i for i, dur in enumerate(tk.SymbolicDurationType)}
    DURATION_INDEXER_INV = {i: dur for dur, i in DURATION_INDEXER.items()}
    NOTE_INDEXER = {note: i for i, note in enumerate("CDEFGAB")}
    NOTE_INDEXER_INV = {i: note for note, i in NOTE_INDEXER.items()}
    DIMS = {
        "token_type": len(TOKEN_INDEXER),
        "bar_numerator": 1,
        "bar_denominator": len(TS_DENOMINATOR_INDEXER),
        "bar_actual_duration_numerator": 1,
        "bar_actual_duration_denominator": 1,
        "key_signature": 15,
        "clef_type": 1,
        "staff_idx": 4,
        "voice_idx": 16,
        "duration_type": len(DURATION_INDEXER),
        "duration_dots": 1,
        "note_class": 7,
        "note_octave": 9,
        "note_accidental": 5,
        "note_tied": 1,
        "note_stem_direction": 3,
        "tuplet_actual_notes": 1,
        "tuplet_normal_notes": 1,
    }
    CUM_DIMS = {k: v for k, v in zip(DIMS.keys(), itertools.accumulate(DIMS.values(), initial=0))}
    TOTAL_DIMENSIONS = sum(DIMS.values())

    _dim_getter = ""

    @classmethod
    def __new__(cls, *args, **kwargs):
        tensor = super().__new__(*args, **kwargs)
        if tensor.dim() < 1 or tensor.size(-1) != cls.TOTAL_DIMENSIONS:
            raise ValueError(f"Last dimension must be of size {cls.TOTAL_DIMENSIONS}")
        return tensor

    def __getitem__(self, item):
        # If query is a string, then it is a dimension name
        # -> return the corresponding slice in the last dimension
        # If query is a tuple and the last element is a string, then it is a dimension name
        # -> return the corresponding slice in the last dimension, but also slice along the other
        #    dimensions
        if (simple_query := (isinstance(item, str) and item.startswith(self._dim_getter))) or (
            (
                isinstance(item, tuple)
                and isinstance(item[-1], str)
                and item[-1].startswith(self._dim_getter)
            )
        ):
            # In case of single string
            if simple_query:
                dim_name = item[len(self._dim_getter) :]
                prev_slice = ...
            # In case of tuple
            else:
                dim_name = item[-1][len(self._dim_getter) :]
                prev_slice = item[:-1]
            if dim_name not in self.DIMS:
                raise KeyError(f"Unknown dimension name {dim_name}")
            res = self[
                prev_slice, self.CUM_DIMS[dim_name] : self.CUM_DIMS[dim_name] + self.DIMS[dim_name]
            ]
            # For one-dimensional attributes, we directly return the value (is this a good idea?)
            if self.DIMS[dim_name] == 1:
                res = res.squeeze(-1)
            return res
        else:
            return super().__getitem__(item)

    def __setitem__(self, key, value):
        # Similar to __getitem__
        if (simple_query := (isinstance(key, str) and key.startswith(self._dim_getter))) or (
            (
                isinstance(key, tuple)
                and isinstance(key[-1], str)
                and key[-1].startswith(self._dim_getter)
            )
        ):
            # In case of single string
            if simple_query:
                dim_name = key[len(self._dim_getter) :]
                prev_slice = ...
            # In case of tuple
            else:
                dim_name = key[-1][len(self._dim_getter) :]
                prev_slice = key[:-1]
            if dim_name not in self.DIMS:
                raise KeyError(f"Unknown dimension name {dim_name}")
            if self.DIMS[dim_name] == 1:
                self[prev_slice, self.CUM_DIMS[dim_name]] = value
            else:
                self[
                    prev_slice,
                    self.CUM_DIMS[dim_name] : self.CUM_DIMS[dim_name] + self.DIMS[dim_name],
                ] = value
        else:
            super().__setitem__(key, value)

    @classmethod
    def _create_onehot(cls, dim_name: str, value: int):
        if dim_name not in cls.DIMS:
            raise KeyError(f"Unknown dimension name {dim_name}")
        if not 0 <= value < cls.DIMS[dim_name]:
            raise ValueError(f"Value {value} is out of bounds for dimension {dim_name}")
        return torch.nn.functional.one_hot(
            torch.tensor(value),
            num_classes=cls.DIMS[dim_name],
        )

    def _onehot_decode(self, dim_name: str):
        if dim_name not in self.DIMS:
            raise KeyError(f"Unknown dimension name {dim_name}")
        return (
            self[
                ...,
                self.CUM_DIMS[dim_name] : self.CUM_DIMS[dim_name] + self.DIMS[dim_name],
            ]
            .argmax(-1)
            .item()
        )

    def _to_token_single(self) -> tk.Token:
        if self.dim() != 1:
            raise ValueError("Can only convert single token tensors")

        token_type = self.TOKEN_INDEXER_INV[self._onehot_decode("token_type")]

        # StartOfScore
        if token_type == tk.StartOfScore:
            return tk.StartOfScore()

        # EndOfScore
        if token_type == tk.EndOfScore:
            return tk.EndOfScore()

        # Bar
        if token_type == tk.Bar:
            numerator = self["bar_numerator"]
            denominator_idx = self._onehot_decode("bar_denominator")
            denominator = self.TS_DENOMINATOR_INDEXER_INV[denominator_idx]
            actual_duration_numerator = self["bar_actual_duration_numerator"]
            actual_duration_denominator = self["bar_actual_duration_denominator"]
            return tk.Bar(
                numerator=clip_cast(numerator, 1, type_=int),
                denominator=clip_cast(denominator, 1, type_=int),
                actual_duration=Fraction(
                    clip_cast(actual_duration_numerator, 1, type_=int),
                    clip_cast(actual_duration_denominator, 1, type_=int),
                ),
            )

        # KeySignature
        if token_type == tk.KeySignature:
            fifths = self._onehot_decode("key_signature") - 7
            return tk.KeySignature(accidentals=fifths)

        # Get eventual staff
        staff_idx = self._onehot_decode("staff_idx")

        # Clef
        if token_type == tk.Clef:
            clef_idx = int(self["clef_type"] > 0.5)
            clef_type = "G" if clef_idx == 0 else "F"
            return tk.Clef(clef=clef_type, staff_idx=staff_idx)

        # Get eventual voice
        voice_idx = self._onehot_decode("voice_idx")

        # Get eventual duration
        duration_type_idx = self._onehot_decode("duration_type")
        duration_type = self.DURATION_INDEXER_INV[duration_type_idx]
        duration_dots = clip_cast(self["duration_dots"], 0, 3, type_=int)

        # Note
        if token_type == tk.Note:
            note_class = self.NOTE_INDEXER_INV[self._onehot_decode("note_class")]
            octave = self._onehot_decode("note_octave")
            accidental = self._onehot_decode("note_accidental") - 2
            tied = bool(self["note_tied"] > 0.5)
            stem_direction_idx = self._onehot_decode("note_stem_direction")
            stem_direction = ["up", "down", None][stem_direction_idx]
            return tk.Note(
                note_class=note_class,
                octave=octave,
                accidental=accidental,
                tied=tied,
                stem_direction=stem_direction,
                staff_idx=staff_idx,
                voice_idx=voice_idx,
                duration=tk.SymbolicDuration(
                    type=tk.SymbolicDurationType(duration_type), dots=duration_dots
                ),
            )

        # Rest
        if token_type == tk.Rest:
            return tk.Rest(
                staff_idx=staff_idx,
                voice_idx=voice_idx,
                duration=tk.SymbolicDuration(
                    type=tk.SymbolicDurationType(duration_type), dots=duration_dots
                ),
            )

        # ChordStart
        if token_type == tk.ChordStart:
            return tk.ChordStart(
                voice_idx=voice_idx,
                duration=tk.SymbolicDuration(
                    type=tk.SymbolicDurationType(duration_type), dots=duration_dots
                ),
            )

        # ChordEnd
        if token_type == tk.ChordEnd:
            return tk.ChordEnd(voice_idx=voice_idx)

        # TupletStart
        if token_type == tk.TupletStart:
            n_actual = clip_cast(self["tuplet_actual_notes"], 1, type_=int)
            n_normal = clip_cast(self["tuplet_normal_notes"], 1, type_=int)
            return tk.TupletStart(
                n_actual=n_actual,
                n_normal=n_normal,
                unit_duration=tk.SymbolicDuration(
                    type=tk.SymbolicDurationType(duration_type),
                    dots=duration_dots,
                ),
                voice_idx=voice_idx,
            )

        # TupletEnd
        if token_type == tk.TupletEnd:
            return tk.TupletEnd(voice_idx=voice_idx)

        raise ValueError(f"Unknown token type {token_type}")

    def to_token(self) -> tk.Token | list[tk.Token] | list:
        if self.dim() == 1:
            return self._to_token_single()
        else:
            # Recursively build a list of tokens along each dimension
            return [subtensor.to_token() for subtensor in self]

    @classmethod
    def _from_token_single(cls, token: tk.Token) -> torch.Tensor:
        assert isinstance(token, tk.Token), f"Expected Token, got {type(token)}"

        vector = cls(torch.zeros(cls.TOTAL_DIMENSIONS, dtype=torch.float))

        # Vectorize token type
        vector["token_type"] = cls._create_onehot("token_type", cls.TOKEN_INDEXER[type(token)])

        # Bar
        if isinstance(token, tk.Bar):
            vector["bar_numerator"] = token.numerator
            vector["bar_denominator"] = cls._create_onehot(
                "bar_denominator",
                cls.TS_DENOMINATOR_INDEXER[token.denominator],
            )
            vector["bar_actual_duration_numerator"] = token.actual_duration.numerator
            vector["bar_actual_duration_denominator"] = token.actual_duration.denominator

        # KeySignature
        if isinstance(token, tk.KeySignature):
            fifths = token.accidentals + 7
            vector["key_signature"] = vector._create_onehot("key_signature", fifths)

        # Clef
        if isinstance(token, tk.Clef):
            clef_type = 0 if token.clef == "G" else 1
            vector["clef_type"] = clef_type

        # HasStaffMixin
        if isinstance(token, tk.HasStaffMixin):
            vector["staff_idx"] = vector._create_onehot("staff_idx", token.staff_idx)

        # HasVoiceMixin
        if isinstance(token, tk.HasVoiceMixin):
            vector["voice_idx"] = vector._create_onehot("voice_idx", token.voice_idx)

        # HasDurationMixin
        if isinstance(token, tk.HasDurationMixin):
            vector["duration_type"] = vector._create_onehot(
                "duration_type",
                cls.DURATION_INDEXER[token.duration.type.value],
            )
            vector["duration_dots"] = token.duration.dots

        # Note
        if isinstance(token, tk.Note):
            # vector[CUM_DIMS["note_class"] + NOTE_INDEXER[token.note_class]] = 1
            # vector[CUM_DIMS["note_octave"] + token.octave] = 1
            # vector[CUM_DIMS["note_accidental"] + token.accidental + 2] = 1
            # vector[CUM_DIMS["note_tied"]] = 1 if token.tied else 0
            # stem = 0 if token.stem_direction == "up" else 1 if token.stem_direction == "down" else 2
            # vector[CUM_DIMS["note_stem_direction"] + stem] = 1
            vector["note_class"] = vector._create_onehot(
                "note_class", cls.NOTE_INDEXER[token.note_class]
            )
            vector["note_octave"] = vector._create_onehot("note_octave", token.octave)
            vector["note_accidental"] = vector._create_onehot(
                "note_accidental", token.accidental + 2
            )
            vector["note_tied"] = 1 if token.tied else 0
            stem_direction = {"up": 0, "down": 1, None: 2}[token.stem_direction]
            vector["note_stem_direction"] = vector._create_onehot(
                "note_stem_direction", stem_direction
            )

        # TupletStart
        if isinstance(token, tk.TupletStart):
            # vector[CUM_DIMS["tuplet_actual_notes"]] = token.n_actual
            # vector[CUM_DIMS["tuplet_normal_notes"]] = token.n_normal
            # vector[CUM_DIMS["duration_type"] + DURATION_INDEXER[token.unit_duration.type.value]] = 1
            # vector[CUM_DIMS["duration_dots"]] = token.unit_duration.dots
            vector["tuplet_actual_notes"] = token.n_actual
            vector["tuplet_normal_notes"] = token.n_normal
            vector["duration_type"] = vector._create_onehot(
                "duration_type",
                cls.DURATION_INDEXER[token.unit_duration.type.value],
            )
            vector["duration_dots"] = token.unit_duration.dots

        return vector

    @classmethod
    def from_tokens(cls, token: tk.Token | list[tk.Token] | list) -> Self:
        if isinstance(token, tk.Token):
            return cls._from_token_single(token)
        else:
            return torch.stack([cls.from_tokens(t) for t in token])


if __name__ == '__main__':
    # token_tensor = TokenTensor(torch.arange(2 * 95, dtype=torch.float).reshape(2, 95))
    token_tensor = TokenTensor(torch.randn(2, 95))
    # token_tensor = TokenTensor(torch.zeros(2, 95))
    print(token_tensor)
    print(f"{token_tensor[..., 5] = }")
    print(f"{token_tensor[0, 'token_type'] = }")
    print(f"{token_tensor['duration_dots'] = }")

    # Test allocation
    token_tensor["bar_denominator"] = torch.ones(1, 6) * 1000
    # token_tensor[0, "bar_denominator"] = torch.ones(6, dtype=torch.float) * 1000
    print(token_tensor)

    # Test decoding
    tokens = token_tensor.to_token()
    for token in tokens:
        print(token)

    # Test encoding
    token = [
        tk.Rest(
            staff_idx=2,
            voice_idx=4,
            duration=tk.SymbolicDuration(type=tk.SymbolicDurationType("quarter"), dots=2),
        ),
        tk.Note(
            note_class="C",
            octave=4,
            accidental=0,
            tied=False,
            stem_direction="up",
            staff_idx=2,
            voice_idx=4,
            duration=tk.SymbolicDuration(type=tk.SymbolicDurationType("quarter"), dots=2),
        ),
    ]
    token_tensor = TokenTensor.from_tokens(token)
    print(token_tensor)

    # Test encoding/decoding
    random_token_tensor = TokenTensor(torch.randn(3, 100, 2, 95))
    tokens = random_token_tensor.to_token()
    revectorized = TokenTensor.from_tokens(tokens)
    retokenized = revectorized.to_token()
    print(all(t == rt for t, rt in zip(tokens, retokenized)))
