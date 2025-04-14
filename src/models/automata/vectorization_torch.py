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

    The interface is the same as a torch.Tensor object, except that it allows to manipulate the
    last dimension directly using the dimension names associated with different token attributes.
    The tensor is slicable using dimension names:
        - TokenTensor(10x15xNUM_DIMS)["note_class"] -> 10x15x7
        - TokenTensor(10x15xNUM_DIMS)[5, 3:7, "note_class"] -> 4x7
    """

    TOKEN_INDEXER = {token: i for i, token in enumerate(tk.TOKEN_TYPES)}
    TOKEN_INDEXER_INV = {i: token for token, i in TOKEN_INDEXER.items()}
    TS_DENOMINATOR_INDEXER = {den: i for i, den in enumerate([1, 2, 4, 8, 16, 32])}
    TS_DENOMINATOR_INDEXER_INV = {i: den for den, i in TS_DENOMINATOR_INDEXER.items()}
    DURATION_INDEXER = {dur.value: i for i, dur in enumerate(tk.SymbolicDurationType)}
    DURATION_INDEXER_INV = {i: dur for dur, i in DURATION_INDEXER.items()}
    NOTE_INDEXER = {note: i for i, note in enumerate("CDEFGAB")}
    NOTE_INDEXER_INV = {i: note for note, i in NOTE_INDEXER.items()}
    STEM_INDEXER = {i: stem for i, stem in enumerate(["up", "down", None])}
    STEM_INDEXER_INV = {stem: i for i, stem in STEM_INDEXER.items()}
    DIMS = {
        "token_type": len(TOKEN_INDEXER),
        "bar_numerator": 1,
        "bar_denominator": len(TS_DENOMINATOR_INDEXER),
        "bar_is_nominal": 1,  # Boolean
        "key_signature": 15,  # -7 to 7
        "clef_type": 1,  # G or F
        "staff_idx": 4,  # Max 4 staves
        "voice_idx": 16,  # Max 4 voices per staff
        "duration_type": len(DURATION_INDEXER),
        "duration_dots": 4,  # Max 3 dots
        "note_class": 7,
        "note_octave": 9,
        "note_accidental": 5,
        "note_tied": 1,
        "note_stem_direction": len(STEM_INDEXER),
        "tuplet_actual_notes": 1,
        "tuplet_normal_notes": 1,
    }
    CUM_DIMS = {k: v for k, v in zip(DIMS.keys(), itertools.accumulate(DIMS.values(), initial=0))}
    TOTAL_DIMENSIONS = sum(DIMS.values())

    @classmethod
    def __new__(cls, *args, **kwargs):
        tensor = super().__new__(*args, **kwargs)
        if tensor.dim() < 1 or tensor.size(-1) != cls.TOTAL_DIMENSIONS:
            raise ValueError(f"Last dimension must be of size {cls.TOTAL_DIMENSIONS}")
        return tensor

    def new_empty(
        self, size, *, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False
    ):
        # Create a new empty tensor with the same class as this one
        tensor = super().new_empty(
            size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
        )
        tensor = tensor.as_subclass(self.__class__)
        return tensor

    def __getitem__(self, item: int | str | tuple):
        # If query is a string, then it is a dimension name
        # -> return the corresponding slice in the last dimension
        # If query is a tuple and the last element is a string, then it is a dimension name
        # -> return the corresponding slice in the last dimension, but also slice along the other
        #    dimensions
        if (is_simple_str := isinstance(item, str)) or (
            isinstance(item, tuple) and isinstance(item[-1], str)
        ):
            dim_name, prev_slice = (item, [...]) if is_simple_str else (item[-1], item[:-1])
            if dim_name not in self.DIMS:
                raise KeyError(f"Unknown dimension name {dim_name}")
            res = self[
                *prev_slice, self.CUM_DIMS[dim_name] : self.CUM_DIMS[dim_name] + self.DIMS[dim_name]
            ].as_subclass(torch.Tensor)
            # For one-dimensional attributes, we directly return the value (is this a good idea?)
            if self.DIMS[dim_name] == 1:
                res = res.squeeze(-1)
            return res
        else:
            return super().__getitem__(item)

    def __setitem__(self, key, value):
        # Similar to __getitem__
        if (is_simple_str := isinstance(key, str)) or (
            isinstance(key, tuple) and isinstance(key[-1], str)
        ):
            dim_name, prev_slice = (key, [...]) if is_simple_str else (key[-1], key[:-1])
            if self.DIMS[dim_name] == 1:
                self[*prev_slice, self.CUM_DIMS[dim_name]] = value
            else:
                self[
                    *prev_slice,
                    self.CUM_DIMS[dim_name] : self.CUM_DIMS[dim_name] + self.DIMS[dim_name],
                ] = value
        else:
            super().__setitem__(key, value)

    @classmethod
    def _create_onehot(cls, dim_name: str, value: int):
        return torch.nn.functional.one_hot(
            torch.tensor(value),
            num_classes=cls.DIMS[dim_name],
        )

    def _onehot_decode(self, dim_name: str) -> int:
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
            is_nominal = bool(self["bar_is_nominal"] > 0.5)
            return tk.Bar(
                numerator=clip_cast(numerator, 1, type_=int),
                denominator=denominator,
                is_nominal=is_nominal,
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
        duration_dots = self._onehot_decode("duration_dots")

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
                    type=tk.SymbolicDurationType(duration_type),
                    dots=duration_dots,
                ),
            )

        # Rest
        if token_type == tk.Rest:
            return tk.Rest(
                staff_idx=staff_idx,
                voice_idx=voice_idx,
                duration=tk.SymbolicDuration(
                    type=tk.SymbolicDurationType(duration_type),
                    dots=duration_dots,
                ),
            )

        # ChordStart
        if token_type == tk.ChordStart:
            return tk.ChordStart(
                voice_idx=voice_idx,
                duration=tk.SymbolicDuration(
                    type=tk.SymbolicDurationType(duration_type),
                    dots=duration_dots,
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

    def to_tokens(self) -> tk.Token | list[tk.Token] | list:
        if self.dim() == 1:
            return self._to_token_single()
        else:
            # Recursively build a list of tokens along each dimension
            return [subtensor.to_tokens() for subtensor in self]

    def __embed_token_type(self, token: tk.Token) -> Self:
        self["token_type"] = self._create_onehot("token_type", self.TOKEN_INDEXER[type(token)])

    def __embed_start_of_score_information(self, sos: tk.StartOfScore) -> Self:
        pass  # Nothing to do

    def __embed_end_of_score_information(self, eos: tk.EndOfScore) -> Self:
        pass  # Nothing to do

    def __embed_bar_information(self, bar: tk.Bar) -> Self:
        self["bar_numerator"] = bar.numerator
        self["bar_denominator"] = self._create_onehot(
            "bar_denominator",
            self.TS_DENOMINATOR_INDEXER[bar.denominator],
        )
        self["bar_is_nominal"] = int(bar.is_nominal)

    def __embed_key_signature_information(self, key_signature: tk.KeySignature) -> Self:
        fifths = key_signature.accidentals + 7
        self["key_signature"] = self._create_onehot("key_signature", fifths)

    def __embed_clef_information(self, clef: tk.Clef) -> Self:
        clef_type = 0 if clef.clef == "G" else 1
        self["clef_type"] = clef_type
        self.__embed_staff_information(clef)

    def __embed_duration_information(self, has_duration: tk.HasDurationMixin) -> Self:
        self["duration_type"] = self._create_onehot(
            "duration_type",
            self.DURATION_INDEXER[has_duration.duration.type.value],
        )
        self["duration_dots"] = self._create_onehot("duration_dots", has_duration.duration.dots)

    def __embed_staff_information(self, has_staff: tk.HasStaffMixin) -> Self:
        self["staff_idx"] = self._create_onehot("staff_idx", has_staff.staff_idx)

    def __embed_voice_information(self, has_voice: tk.HasVoiceMixin) -> Self:
        self["voice_idx"] = self._create_onehot("voice_idx", has_voice.voice_idx)

    def __embed_note_information(self, note: tk.Note) -> Self:
        self["note_class"] = self._create_onehot("note_class", self.NOTE_INDEXER[note.note_class])
        self["note_octave"] = self._create_onehot("note_octave", note.octave)
        self["note_accidental"] = self._create_onehot("note_accidental", note.accidental + 2)
        self["note_tied"] = 1 if note.tied else 0
        stem_direction = (
            0 if note.stem_direction == "up" else 1 if note.stem_direction == "down" else 2
        )
        self["note_stem_direction"] = self._create_onehot("note_stem_direction", stem_direction)
        self.__embed_staff_information(note)
        self.__embed_voice_information(note)
        self.__embed_duration_information(note)

    def __embed_rest_information(self, rest: tk.Rest) -> Self:
        self.__embed_staff_information(rest)
        self.__embed_voice_information(rest)
        self.__embed_duration_information(rest)

    def __embed_tuplet_start_information(self, tuplet_start: tk.TupletStart) -> Self:
        self["tuplet_actual_notes"] = tuplet_start.n_actual
        self["tuplet_normal_notes"] = tuplet_start.n_normal
        self["duration_type"] = self._create_onehot(
            "duration_type",
            self.DURATION_INDEXER[tuplet_start.unit_duration.type.value],
        )
        self["duration_dots"] = self._create_onehot(
            "duration_dots",
            tuplet_start.unit_duration.dots,
        )
        self.__embed_voice_information(tuplet_start)

    def __embed_tuplet_end_information(self, tuplet_end: tk.TupletEnd) -> Self:
        self.__embed_voice_information(tuplet_end)

    def __embed_chord_start_information(self, chord_start: tk.ChordStart) -> Self:
        self.__embed_voice_information(chord_start)
        self.__embed_duration_information(chord_start)

    def __embed_chord_end_information(self, chord_end: tk.ChordEnd) -> Self:
        self.__embed_voice_information(chord_end)

    @classmethod
    def _from_token_single(cls, token: tk.Token) -> torch.Tensor:
        assert isinstance(token, tk.Token), f"Expected Token, got {type(token)}"

        vector = cls(torch.zeros(cls.TOTAL_DIMENSIONS, dtype=torch.float))

        # Vectorize token type
        vector.__embed_token_type(token)

        # Add the information specific to the token type
        if isinstance(token, tk.StartOfScore):
            vector.__embed_start_of_score_information(token)
        elif isinstance(token, tk.EndOfScore):
            vector.__embed_end_of_score_information(token)
        elif isinstance(token, tk.Bar):
            vector.__embed_bar_information(token)
        elif isinstance(token, tk.KeySignature):
            vector.__embed_key_signature_information(token)
        elif isinstance(token, tk.Clef):
            vector.__embed_clef_information(token)
        elif isinstance(token, tk.Note):
            vector.__embed_note_information(token)
        elif isinstance(token, tk.Rest):
            vector.__embed_rest_information(token)
        elif isinstance(token, tk.ChordStart):
            vector.__embed_chord_start_information(token)
        elif isinstance(token, tk.ChordEnd):
            vector.__embed_chord_end_information(token)
        elif isinstance(token, tk.TupletStart):
            vector.__embed_tuplet_start_information(token)
        elif isinstance(token, tk.TupletEnd):
            vector.__embed_tuplet_end_information(token)
        else:
            raise ValueError(f"Unexpected token: {token}")

        return vector

    @classmethod
    def from_tokens(cls, token: tk.Token | list[tk.Token] | list) -> Self:
        if isinstance(token, tk.Token):
            return cls._from_token_single(token)
        else:
            return torch.stack([cls.from_tokens(t) for t in token])


if __name__ == '__main__':
    # token_tensor = TokenTensor(torch.arange(2 * 97, dtype=torch.float).reshape(2, 97))
    token_tensor = TokenTensor(torch.randn(2, 97))
    # token_tensor = TokenTensor(torch.zeros(2, 97))
    print(token_tensor)
    print(f"{token_tensor[..., 5] = }")
    print(f"{token_tensor[0, 'token_type'] = }")
    print(f"{token_tensor['duration_dots'] = }")

    # Test allocation
    token_tensor["bar_denominator"] = torch.ones(1, 6) * 1000
    # token_tensor[0, "bar_denominator"] = torch.ones(6, dtype=torch.float) * 1000
    print(token_tensor)

    # Test decoding
    tokens = token_tensor.to_tokens()
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
    random_token_tensor = TokenTensor(torch.randn(3, 100, 2, 97))
    tokens = random_token_tensor.to_tokens()
    revectorized = TokenTensor.from_tokens(tokens)
    retokenized = revectorized.to_tokens()
    print(all(t == rt for t, rt in zip(tokens, retokenized)))
