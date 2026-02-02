import itertools
from enum import Enum
from fractions import Fraction
from typing import Self

import torch

from models.automata import tokens as tk
from utils.constants import PADDING_VALUE
from utils.musicxml import ARTICULATIONS_AND_ORNAMENTS


TS_MAX_NUMERATOR = 30
TS_DENOMINATORS = [1, 2, 4, 8, 16, 32]


class ProblemType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


def clip(x, lower=None, upper=None):
    assert lower is None or upper is None or lower <= upper
    if lower is not None:
        x = max(x, lower)
    if upper is not None:
        x = min(x, upper)
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
    TS_INDEXER = {
        ts: i
        for i, ts in enumerate(
            [(n, d) for d in TS_DENOMINATORS for n in range(1, TS_MAX_NUMERATOR + 1)]
        )
    }
    TS_INDEXER_INV = {i: ts for ts, i in TS_INDEXER.items()}
    DURATION_INDEXER = {dur.value: i for i, dur in enumerate(tk.SymbolicDurationType)}
    DURATION_INDEXER_INV = {i: dur for dur, i in DURATION_INDEXER.items()}
    NOTE_INDEXER = {note: i for i, note in enumerate("CDEFGAB")}
    NOTE_INDEXER_INV = {i: note for note, i in NOTE_INDEXER.items()}
    STEM_INDEXER = {stem: i for i, stem in enumerate(["up", "down", None])}
    STEM_INDEXER_INV = {i: stem for stem, i in STEM_INDEXER.items()}
    _DIMS_AND_PROBLEM_TYPE = {
        "token_type": (len(TOKEN_INDEXER), ProblemType.CLASSIFICATION),
        "bar_time_signature": (len(TS_INDEXER), ProblemType.CLASSIFICATION),
        "bar_is_nominal": (2, ProblemType.CLASSIFICATION),  # Boolean
        "key_signature": (15, ProblemType.CLASSIFICATION),  # -7 to 7
        "clef_type": (2, ProblemType.CLASSIFICATION),  # G or F
        "staff_idx": (2, ProblemType.CLASSIFICATION),  # Max 2 staves
        "voice_idx": (8, ProblemType.CLASSIFICATION),  # Max 4 voices per staff
        "duration_type": (len(DURATION_INDEXER), ProblemType.CLASSIFICATION),
        "duration_dots": (4, ProblemType.CLASSIFICATION),  # Max 3 dots
        "measure_rest": (2, ProblemType.CLASSIFICATION),
        "note_class": (7, ProblemType.CLASSIFICATION),
        "note_octave": (9, ProblemType.CLASSIFICATION),
        "note_accidental": (5, ProblemType.CLASSIFICATION),
        "note_grace": (2, ProblemType.CLASSIFICATION),
        "note_tied": (2, ProblemType.CLASSIFICATION),
        "note_stem_direction": (len(STEM_INDEXER), ProblemType.CLASSIFICATION),
        "fermata": (2, ProblemType.CLASSIFICATION),
        **{art: (2, ProblemType.CLASSIFICATION) for art in ARTICULATIONS_AND_ORNAMENTS},
        "tuplet_actual_notes": (1, ProblemType.REGRESSION),
        "tuplet_normal_notes": (1, ProblemType.REGRESSION),
    }
    DIMS = {k: v[0] for k, v in _DIMS_AND_PROBLEM_TYPE.items()}
    PROBLEM_TYPES = {k: v[1] for k, v in _DIMS_AND_PROBLEM_TYPE.items()}
    CUM_DIMS = {k: v for k, v in zip(DIMS.keys(), itertools.accumulate(DIMS.values(), initial=0))}
    TOTAL_DIMENSIONS = sum(DIMS.values())
    TOKEN_TYPE_TO_DIMS = {
        tk.StartOfScore: ["token_type"],
        tk.EndOfScore: ["token_type"],
        tk.Bar: ["token_type", "bar_time_signature", "bar_is_nominal"],
        tk.KeySignature: ["token_type", "key_signature"],
        tk.Clef: ["token_type", "clef_type", "staff_idx"],
        tk.Rest: [
            "token_type",
            "staff_idx",
            "voice_idx",
            "duration_type",
            "duration_dots",
            "measure_rest",
            "fermata",
        ],
        tk.Note: [
            "token_type",
            "staff_idx",
            "voice_idx",
            "duration_type",
            "duration_dots",
            "note_class",
            "note_octave",
            "note_accidental",
            "note_grace",
            "note_tied",
            "note_stem_direction",
            "fermata",
            *ARTICULATIONS_AND_ORNAMENTS,
        ],
        tk.ChordStart: ["token_type", "voice_idx", "duration_type", "duration_dots"],
        tk.ChordEnd: ["token_type", "voice_idx"],
        tk.TupletStart: [
            "token_type",
            "voice_idx",
            "duration_type",
            "duration_dots",
            "tuplet_actual_notes",
            "tuplet_normal_notes",
        ],
        tk.TupletEnd: ["token_type", "voice_idx"],
    }

    @classmethod
    def __new__(cls, *args, **kwargs):
        tensor = super().__new__(*args, **kwargs)
        if tensor.dim() < 1 or tensor.size(-1) != cls.TOTAL_DIMENSIONS:
            raise ValueError(
                f"Last dimension must be of size {cls.TOTAL_DIMENSIONS}, but is {tensor.size(-1)}"
            )
        return tensor

    def new_empty(
        self, size, *, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False
    ):
        # Create a new empty tensor with the same class as this one
        tensor = super().new_empty(
            size,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        tensor = tensor.as_subclass(self.__class__)
        return tensor

    def __getitem__(self, item: int | str | tuple) -> Self | torch.Tensor:
        # If query is a string, then it is a dimension name
        # -> return the corresponding slice in the last dimension
        # If query is a tuple and the last element is a string, then it is a dimension name
        # -> return the corresponding slice in the last dimension, but also slice along the other
        #    dimensions
        if (is_simple_str := isinstance(item, str)) or (
            isinstance(item, tuple) and len(item) > 0 and isinstance(item[-1], str)
        ):
            dim_name, prev_slice = (item, [...]) if is_simple_str else (item[-1], item[:-1])
            if dim_name not in self.DIMS:
                raise KeyError(f"Unknown dimension name {dim_name}")
            res = self[
                *prev_slice, self.CUM_DIMS[dim_name] : self.CUM_DIMS[dim_name] + self.DIMS[dim_name]
            ].as_subclass(torch.Tensor)
            # For one-dimensional attributes, we directly return the value (is this a good idea?)
            # if self.DIMS[dim_name] == 1:
            #     res = res.squeeze(-1)
            return res
        else:
            return super().__getitem__(item)

    def __setitem__(self, key, value):
        # Similar to __getitem__
        if (is_simple_str := isinstance(key, str)) or (
            isinstance(key, tuple) and len(key) > 0 and isinstance(key[-1], str)
        ):
            dim_name, prev_slice = (key, [...]) if is_simple_str else (key[-1], key[:-1])
            # if self.DIMS[dim_name] == 1:
            #     self[*prev_slice, self.CUM_DIMS[dim_name]] = value
            # else:
            self[
                *prev_slice,
                self.CUM_DIMS[dim_name] : self.CUM_DIMS[dim_name] + self.DIMS[dim_name],
            ] = value
        else:
            super().__setitem__(key, value)

    @classmethod
    def _create_onehot(cls, dim_name: str, value: int | torch.Tensor):
        if cls.PROBLEM_TYPES[dim_name] != ProblemType.CLASSIFICATION:
            raise ValueError(f"Can't decode dimension {dim_name} as {cls.PROBLEM_TYPES[dim_name]}")
        return torch.nn.functional.one_hot(
            value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long),
            num_classes=cls.DIMS[dim_name],
        )

    def _onehot_decode(self, dim_name: str) -> torch.Tensor:
        if dim_name not in self.DIMS:
            raise KeyError(f"Unknown dimension name {dim_name}")
        if self.PROBLEM_TYPES[dim_name] != ProblemType.CLASSIFICATION:
            raise ValueError(f"Can't decode dimension {dim_name} as {self.PROBLEM_TYPES[dim_name]}")
        return self[
            ...,
            self.CUM_DIMS[dim_name] : self.CUM_DIMS[dim_name] + self.DIMS[dim_name],
        ].argmax(-1)

    def _quantize_dim(self, dim_name: str):
        return self._create_onehot(dim_name, self._onehot_decode(dim_name))

    def _to_token_single(self) -> tk.Token:
        if self.dim() != 1:
            raise ValueError("Can only convert single token tensors")

        token_type = self.TOKEN_INDEXER_INV[self._onehot_decode("token_type").item()]

        # StartOfScore
        if token_type == tk.StartOfScore:
            return tk.StartOfScore()

        # EndOfScore
        if token_type == tk.EndOfScore:
            return tk.EndOfScore()

        # Bar
        if token_type == tk.Bar:
            time_signature_idx = self._onehot_decode("bar_time_signature").item()
            numerator, denominator = self.TS_INDEXER_INV[time_signature_idx]
            is_nominal = self._onehot_decode("bar_is_nominal").item() == 1
            return tk.Bar(
                numerator=numerator,
                denominator=denominator,
                is_nominal=is_nominal,
            )

        # KeySignature
        if token_type == tk.KeySignature:
            fifths = self._onehot_decode("key_signature").item() - 7
            return tk.KeySignature(accidentals=fifths)

        # Get eventual staff
        staff_idx = self._onehot_decode("staff_idx").item()

        # Clef
        if token_type == tk.Clef:
            clef_idx = self._onehot_decode("clef_type").item()
            clef_type = "G" if clef_idx == 0 else "F"
            return tk.Clef(clef=clef_type, staff_idx=staff_idx)

        # Get eventual voice
        voice_idx = self._onehot_decode("voice_idx").item()

        # Get eventual duration
        duration_type_idx = self._onehot_decode("duration_type").item()
        duration_type = self.DURATION_INDEXER_INV[duration_type_idx]
        duration_dots = self._onehot_decode("duration_dots").item()

        # Note
        if token_type == tk.Note:
            note_class = self.NOTE_INDEXER_INV[self._onehot_decode("note_class").item()]
            octave = self._onehot_decode("note_octave").item()
            accidental = self._onehot_decode("note_accidental").item() - 2
            grace = self._onehot_decode("note_grace").item() == 1
            tied = self._onehot_decode("note_tied").item() == 1
            stem_direction_idx = self._onehot_decode("note_stem_direction").item()
            stem_direction = self.STEM_INDEXER_INV[stem_direction_idx]
            # Articulations and ornaments
            articulations_and_ornaments = {
                art: self._onehot_decode(art).item() == 1
                for art in [
                    "accent",
                    "strong_accent",
                    "staccato",
                    "staccatissimo",
                    "tenuto",
                    "detached_legato",
                    "trill_mark",
                    "turn",
                    "mordent",
                    "inverted_mordent",
                    "tremolo",
                    "fermata",
                ]
            }
            return tk.Note(
                note_class=note_class,
                octave=octave,
                accidental=accidental,
                grace=grace,
                tied=tied,
                stem_direction=stem_direction,
                staff_idx=staff_idx,
                voice_idx=voice_idx,
                duration=tk.SymbolicDuration(
                    type=tk.SymbolicDurationType(duration_type),
                    dots=duration_dots,
                ),
                **articulations_and_ornaments,
            )

        # Rest
        if token_type == tk.Rest:
            measure_rest = self._onehot_decode("measure_rest").item() == 1
            fermata = self._onehot_decode("fermata").item() == 1
            # if measure_rest:
            #     duration = tk.SymbolicDuration(
            #         type=tk.SymbolicDurationType.WHOLE,
            #         dots=0,
            #     )
            # else:
            #     duration = tk.SymbolicDuration(
            #         type=tk.SymbolicDurationType(duration_type),
            #         dots=duration_dots,
            #     )
            duration = tk.SymbolicDuration(
                type=tk.SymbolicDurationType(duration_type),
                dots=duration_dots,
            )
            return tk.Rest(
                staff_idx=staff_idx,
                voice_idx=voice_idx,
                duration=duration,
                measure_rest=measure_rest,
                fermata=fermata,
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
            n_actual = round(clip(self["tuplet_actual_notes"][0].item(), 1))
            n_normal = round(clip(self["tuplet_normal_notes"][0].item(), 1))
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
        self["bar_time_signature"] = self._create_onehot(
            "bar_time_signature",
            self.TS_INDEXER[(bar.numerator, bar.denominator)],
        )
        self["bar_is_nominal"] = self._create_onehot("bar_is_nominal", int(bar.is_nominal))

    def __embed_key_signature_information(self, key_signature: tk.KeySignature) -> Self:
        fifths = key_signature.accidentals + 7
        self["key_signature"] = self._create_onehot("key_signature", fifths)

    def __embed_clef_information(self, clef: tk.Clef) -> Self:
        clef_type = 0 if clef.clef == "G" else 1
        self["clef_type"] = self._create_onehot("clef_type", clef_type)
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
        self["note_grace"] = self._create_onehot("note_grace", int(note.grace))
        self["note_tied"] = self._create_onehot("note_tied", int(note.tied))
        stem_direction = self.STEM_INDEXER[note.stem_direction]
        self["note_stem_direction"] = self._create_onehot("note_stem_direction", stem_direction)
        # Articulations and ornaments
        for art in ARTICULATIONS_AND_ORNAMENTS + ["fermata"]:
            self[art] = self._create_onehot(art, 1 if getattr(note, art) else 0)
        # Inherited attributes
        self.__embed_staff_information(note)
        self.__embed_voice_information(note)
        self.__embed_duration_information(note)

    def __embed_rest_information(self, rest: tk.Rest) -> Self:
        self["measure_rest"] = self._create_onehot("measure_rest", int(rest.measure_rest))
        self["fermata"] = self._create_onehot("fermata", int(rest.fermata))
        self.__embed_staff_information(rest)
        self.__embed_voice_information(rest)
        self.__embed_duration_information(rest)

    def __embed_tuplet_start_information(self, tuplet_start: tk.TupletStart) -> Self:
        self["tuplet_actual_notes"][0] = tuplet_start.n_actual
        self["tuplet_normal_notes"][0] = tuplet_start.n_normal
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

    def get_as_dict(
        self,
        include_token_type_dim: bool = True,
        remove_padding: bool = True,
        token_type_keys_as_str: bool = False,
        tokens_mask: dict[tk.AnyToken | str, torch.Tensor] | None = None,
    ) -> dict[tk.AnyToken | str, dict[str, torch.Tensor]]:
        """
        Get the tensor as a dictionary of token type -> dimension -> value

        :param include_token_type_dim: Whether to include or not the dimension "token_type"
        :param remove_padding: Whether to filter out padding tokens
        :param token_type_keys_as_str: Whether to use str rather than classes for the returned dict
            keys
        :param tokens_mask: Mask dict for each token type (only keep the ones with True)
        """
        if tokens_mask is None:
            tokens_mask = {}
        elif any(mask.shape != self.shape[:-1] for mask in tokens_mask.values()):
            raise ValueError(
                f"Expected tokens_mask tensors of shape {self.shape}, "
                f"got {[m.shape for m in tokens_mask]}"
            )
        if tokens_mask and isinstance(next(iter(tokens_mask.keys())), str):
            tokens_mask = {tk.TOKEN_STR_TO_TYPE[t_str]: v for t_str, v in tokens_mask.items()}
        if remove_padding:
            # TODO: replace this by extending the tokens_mask rather than cloning
            tensor = self.clone()
            indices = tensor[..., 0] != PADDING_VALUE
            tensor = tensor[indices]
            tokens_mask = {t: v[indices] for t, v in tokens_mask.items()}
        else:
            tensor = self
        d = {}
        for token_type in self.TOKEN_INDEXER.keys():
            d_token = {}
            for dim_name in self.TOKEN_TYPE_TO_DIMS[token_type]:
                if not include_token_type_dim and dim_name == "token_type":
                    continue
                indices = tensor["token_type"].argmax(-1) == self.TOKEN_INDEXER[token_type]
                if token_type in tokens_mask:
                    indices = torch.logical_and(indices, tokens_mask[token_type])
                d_token[dim_name] = tensor[indices][dim_name]
            d[token_type.__name__ if token_type_keys_as_str else token_type] = d_token
        return d

    @classmethod
    def build_from_dict(
        cls,
        type_dim_values_dict: dict[str, dict[str, torch.Tensor]],
        token_type_masks: dict[str, torch.Tensor],  # B * T
        padding_mask: torch.Tensor | None = None,  # B * T
    ) -> Self:
        """
        Convert the expert outputs to a TokenTensor.

        :param type_dim_values_dict: Dict of token type -> dimension -> values
        :param token_type_masks: Dict of token type -> mask (true for token i)
        :param padding_mask: Optional padding mask (true if token i padded)
        :return: The TokenTensor.
        """
        initial_shape = list(token_type_masks.values())[0].shape
        tensor = TokenTensor(torch.zeros((*initial_shape, TokenTensor.TOTAL_DIMENSIONS)))
        for token_type in TokenTensor.TOKEN_INDEXER:
            mask = token_type_masks[token_type.as_str()]
            mask_indices = mask.nonzero(as_tuple=True)
            tensor["token_type"][*mask_indices, TokenTensor.TOKEN_INDEXER[token_type]] = 1
            for dim, values in type_dim_values_dict.get(token_type.as_str(), {}).items():
                tensor[*mask_indices, dim] = values
        if padding_mask is not None:
            tensor[padding_mask] = PADDING_VALUE
        return tensor

    def quantize(self, inplace: bool = False) -> Self:
        quantized: TokenTensor = self.clone() if not inplace else self
        quantized["token_type"] = self._quantize_dim("token_type")
        for token_type, token_type_idx in self.TOKEN_INDEXER.items():
            related_idx = torch.where(quantized["token_type"][..., token_type_idx] == 1)
            if related_idx[0].numel() > 0:
                if self.ndim == 1:
                    related_idx = ()  # Fix for torch.where behaviour with 0-D tensors
                for token_sub_dimension in self.DIMS:
                    # if token_sub_dimension == "duration_dots":
                    #     breakpoint()
                    # Token type dimension already quantized
                    if token_sub_dimension == "token_type":
                        continue
                    # Dimension used for the token type
                    elif token_sub_dimension in self.TOKEN_TYPE_TO_DIMS[token_type]:
                        if self.PROBLEM_TYPES[token_sub_dimension] == ProblemType.CLASSIFICATION:
                            quantized[(*related_idx, token_sub_dimension)] = (
                                self[*related_idx]
                                ._quantize_dim(token_sub_dimension)
                                .to(torch.float)
                            )
                        elif self.PROBLEM_TYPES[token_sub_dimension] == ProblemType.REGRESSION:
                            quantized[(*related_idx, token_sub_dimension)] = torch.round(
                                torch.clamp(
                                    self[*related_idx, token_sub_dimension],
                                    min=(
                                        1
                                        if token_sub_dimension
                                        in ("tuplet_actual_notes", "tuplet_normal_notes")
                                        else None
                                    ),
                                )
                            )
                        else:  # Should not happen
                            raise ValueError(
                                f"Unknown problem type: {self.PROBLEM_TYPES[token_sub_dimension]}"
                            )
                    # Dimension unused for the token type
                    else:
                        quantized[(*related_idx, token_sub_dimension)] = 0.0

        return quantized

    # def get_token_type_specific_part(
    #     self,
    #     token_type: type[tk.Token],
    #     include_token_type_dims: bool = False,
    # ) -> torch.Tensor:
    #     """
    #     Get the part of the tensor containing all information related to the token type.
    #     """
    #     output = torch.concat([self[dim_name] for dim_name in self.TOKEN_TYPE_TO_DIMS[token_type]])


if __name__ == '__main__':
    # token_tensor = TokenTensor(torch.arange(2 * 110, dtype=torch.float).reshape(2, 97))
    token_tensor = TokenTensor(torch.randn(2, 299))
    # token_tensor = TokenTensor(torch.zeros(2, 110))
    print(token_tensor)
    print(f"{token_tensor[..., 5] = }")
    print(f"{token_tensor[0, 'token_type'] = }")
    print(f"{token_tensor['duration_dots'] = }")
    print(f"{token_tensor['bar_time_signature'] = }")

    # Test allocation
    token_tensor["bar_time_signature"] = torch.ones(1, 180) * 1000
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
    random_token_tensor = TokenTensor(torch.randn(3, 100, 2, 299))
    tokens = random_token_tensor.to_tokens()
    revectorized = TokenTensor.from_tokens(tokens)
    retokenized = revectorized.to_tokens()
    print(all(t == rt for t, rt in zip(tokens, retokenized)))
