from dataclasses import dataclass, fields, field
from abc import abstractmethod
from collections.abc import Callable
from fractions import Fraction
from typing import Any
import inspect
import param

ALLOWED_DURATIONS = [Fraction(1, 2**i) for i in range(0, 6)]  # whole note to 32nd note


@dataclass
class TranscriptionState:
    S: list[Fraction] = field(default_factory=lambda: [Fraction(0)])
    n: int = 0
    L: dict[tuple[str, int, int], int] = field(default_factory=dict)  # ((cls, oct, acc), n)
    C: int = 0
    T: list[Fraction] = field(default_factory=list)
    started: bool = False

    def copy(self):
        return TranscriptionState(
            S=self.S.copy(),
            n=self.n,
            L=self.L.copy(),
            C=self.C,
            T=self.T.copy(),
        )

    def L_prev(self, n: int) -> dict[tuple[str, int, int], int]:
        return {note: n_ for note, n_ in self.L.items() if n_ < n}

    @property
    def in_chord(self) -> bool:
        return self.C > 0


@dataclass
class Token:
    @classmethod
    def _init_args(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    @abstractmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        """Returns whether the token type is allowed in the current state, i.e. if
        at least one combination of instantiation kwargs will result in the token
        being allowed.
        To check if a specific token instance is allowed, use the method `is_token_allowed`.

        :param state: The current state of the automaton.
        :return: Whether the token type is allowed in the current state.
        """
        ...

    @abstractmethod
    def is_token_allowed(self, state: TranscriptionState) -> bool:
        """Returns whether this token instance is allowed in the current state.
        To check if the token type is allowed, rather than this specific instance,
        use the class method `is_token_type_allowed`.

        :param state: The current state of the automaton.
        :return: Whether the token is allowed in the current state.
        """
        ...

    @classmethod
    def _get_allowed_kwargs(self, state: TranscriptionState) -> dict[str, list[Any] | None]:
        """Returns instantiation kwargs that will result in the token being allowed.
        This method
        Make sure that all __init__ args are present in the returned dictionary.

        :param state: The current state of the automaton.
        :return: The instantiation kwargs that will result in the token being allowed:
            {
                "init_arg1": [value1, value2, ...],
                "init_arg2": None,  # None means that any value is allowed
                "init_arg3": [],  # Empty list means that no value is allowed
                ...
            }
        """
        return {}

    @classmethod
    def allowed_kwargs(cls, state: TranscriptionState) -> dict[str, list[Any] | None]:
        """Returns instantiation kwargs that will result in the token being allowed.

        :param state: The current state of the automaton.
        :return: The instantiation kwargs that will result in the token being allowed:
            {
                "init_arg1": [value1, value2, ...],
                "init_arg2": None,  # None means that any value is allowed
                "init_arg3": [],  # Empty list means that no value is allowed
                ...
            }
        """
        kwargs = cls._get_allowed_kwargs(state)
        if not all(varname in kwargs for varname in cls._init_args()):
            raise ValueError(
                f"Token {cls.__name__} is missing some attributes of the __init__ method. "
                "Make sure that all attributes are present in the _allowed_kwargs method."
            )
        return kwargs

    def transition_fn(self, state: TranscriptionState, in_place: bool = True) -> TranscriptionState:
        """Returns the new state after applying the token."""
        assert self.is_token_allowed(state), f"Token {self} is not allowed in state {state}"
        new_state = state if in_place else state.copy()
        self._transition_fn(new_state)
        return new_state

    @abstractmethod
    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        """Modify a state IN PLACE after applying the token."""
        ...


@dataclass
class StartOfScore(Token):
    def is_token_allowed(self, state: TranscriptionState) -> bool:
        return not state.started

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.started = True
        return state


@dataclass
class EndOfScore(Token):
    def is_token_allowed(self, state: TranscriptionState) -> bool:
        return state.S[-1] == 0 and len(state.L) == 0 and state.started

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        return state


@dataclass
class Bar(Token):
    numerator: int
    denominator: int
    real_numerator: int | None = None

    def __post_init__(self):
        if self.denominator.bit_count() != 1:
            raise ValueError("The denominator of the TS must be a power of 2.")
        if (
            self.numerator < 1
            or self.denominator < 1
            or (self.real_numerator is not None and self.real_numerator < 1)
        ):
            raise ValueError("All time signature attributes values must be positive integers.")
        if self.real_numerator is None:
            self.real_numerator = self.numerator

    def is_token_allowed(self, state: TranscriptionState) -> bool:
        return state.S[-1] == 0

    @classmethod
    def _get_allowed_kwargs(cls, state: TranscriptionState) -> dict[str, list[Any] | None]:
        if state.S[-1] == 0:
            return {name: None for name in cls._init_args()}  # No restrictions
        else:
            return {name: [] for name in cls._init_args()}  # No values allowed

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.S[0] = Fraction(self.real_numerator)
        state.n = 0
        state.L = {}
        state.C = 0
        state.T = []
        return state

    @property
    def duration(self) -> Fraction:
        denominator_duration = Fraction(2 ** (3 - self.denominator.bit_length()))
        return self.numerator * denominator_duration


@dataclass
class KeySignature(Token):
    accidentals: int

    def __post_init__(self):
        if abs(self.accidentals) > 7:
            raise ValueError("The number of accidentals must be between -7 and 7.")

    def is_token_allowed(self, state: TranscriptionState) -> bool:
        return len(state.S) == 1 and state.C == 0

    @classmethod
    def _get_allowed_kwargs(cls, state: TranscriptionState) -> dict[str, list[Any] | None]:
        if len(state.S) == 1 and state.C == 0:
            return {name: None for name in cls._init_args()}
        else:
            return {name: [] for name in cls._init_args()}

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        return state


@dataclass
class Rest(Token):
    duration: Fraction

    def __post_init__(self):
        if not self.duration in ALLOWED_DURATIONS:
            raise ValueError("The duration of a rest must be a power of 2.")

    def is_token_allowed(self, state: TranscriptionState) -> bool:
        return self.duration <= state.S[-1] and len(state.L) == 0

    @classmethod
    def _get_allowed_kwargs(cls, state: TranscriptionState) -> dict[str, list[Any] | None]:
        if len(state.L) != 0:
            return {"duration": []}  # No rest when there are tied notes
        return {"duration": [duration for duration in ALLOWED_DURATIONS if duration <= state.S[-1]]}

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.S[0] -= self.duration
        state.n += 1
        return state


@dataclass
class Note(Token):
    duration: Fraction
    note_class: str
    octave: int
    accidental: int
    tied: bool

    _note_classes = "ABCDEFG"
    _note_alterations = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}

    def __post_init__(self):
        if not (
            self.duration in ALLOWED_DURATIONS
            or self.duration == Fraction(0)  # 0 allowed for chords
        ):
            raise ValueError("The duration of a note must be a power of 2 or zero.")
        if self.note_class not in self._note_classes:
            raise ValueError(f"The note class must be one of {', '.join(self._note_classes)}.")
        if self.octave < 0 or self.octave > 9:
            raise ValueError("The octave must be between 0 and 9.")
        if self.accidental not in self._note_alterations.keys():
            raise ValueError(
                f"The accidental must be one of {', '.join(map(str, self._note_alterations.keys()))}."
            )

    @property
    def note_tuple(self) -> tuple[str, int, int]:
        return self.note_class, self.octave, self.accidental

    def is_token_allowed(self, state: TranscriptionState) -> bool:
        # If not in chord
        if not state.in_chord:
            duration_ok = 0 < self.duration <= state.S[0]
            if len(state.L) != 0:  # No need to specify L_prev as not in chord
                return duration_ok and any(note == self.note_tuple for note in state.L.keys())
            return duration_ok
        # If in chord
        return self.duration == 0

    @classmethod
    def _get_allowed_kwargs(cls, state: TranscriptionState) -> dict[str, list[Any] | None]:
        # If not in chord
        if not state.in_chord:
            durations_ok = [duration for duration in ALLOWED_DURATIONS if duration <= state.S[0]]
            # If there are tied notes, only allow those notes to be produced
            if len(state.L) != 0:
                notes = {note for note in state.L.keys()}
                return {
                    "duration": durations_ok,
                    "note_class": [n[0] for n in notes],
                    "octave": [n[1] for n in notes],
                    "accidental": [n[2] for n in notes],
                    "tied": [True, False],
                }
            # If there are no tied notes, allow any note to be produced
            return {
                "duration": durations_ok,
                "note_class": [note_class for note_class in cls._note_classes],
                "octave": list(range(10)),
                "accidental": [alt for alt in cls._note_alterations.keys()],
                "tied": [True, False],
            }
        # If in chord, only allow duration 0 (duration dealt in the chord tokens)
        return {
            "duration": [Fraction(0)],
            "note_class": [note_class for note_class in cls._note_classes],
            "octave": list(range(10)),
            "accidental": [alt for alt in cls._note_alterations.keys()],
            "tied": [True, False],
        }

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.S[0] -= self.duration
        if self.tied:
            state.L[self.note_tuple] = state.n
        if self.note_tuple in state.L_prev(state.n):
            state.L.pop(self.note_tuple)
        if not state.in_chord:
            state.n += 1
        return state


TOKEN_TYPES = [StartOfScore, EndOfScore, Bar, KeySignature, Rest, Note]
