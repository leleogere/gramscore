import copy
import dataclasses
import random
from dataclasses import dataclass, fields, field
from abc import abstractmethod
from collections.abc import Callable
from fractions import Fraction
from typing import Any
import inspect
from typing import Self
import partitura as pt
import partitura.score

import param

ALLOWED_DURATIONS = [Fraction(2**i) for i in range(-3, 3)]  # 32nd to whole (in quarter notes)


@dataclass
class ChordState:
    duration: Fraction = 0
    notes: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def num_notes(self) -> int:
        """Returns the number of notes in the chord."""
        return len(self.notes)

    @property
    def active(self) -> bool:
        """Returns whether the chord is currently being constructed."""
        return self.duration > 0

    @property
    def can_resolve(self) -> bool:
        """Returns whether the chord can be resolved (produce the ChordEnd token)."""
        assert self.active, "Chord is not active."
        return self.num_notes >= 2

    def reset(self):
        self.duration = Fraction(0)
        self.notes = []


# @dataclass
# class TupletState:
#     n_actual: int = 0
#     n_real: int = 0
#     unit_duration: Fraction = 0
#
#     @property
#     def active(self) -> bool:
#         """Returns whether the tuplet is currently being constructed."""
#         return self.unit_duration > 0
#
#     @property
#     def real_duration(self) -> Fraction:
#         """Returns the real duration of the tuplet."""
#         return self.n_real * self.unit_duration
#
#     @property
#     def actual_duration(self) -> Fraction:
#         """Returns the actual duration of the tuplet."""
#         return self.n_actual * self.unit_duration
#
#     def reset(self):
#         self.n_actual = 0
#         self.n_real = 0
#         self.unit_duration = Fraction(0)


@dataclass
class TranscriptionState:
    S: list[Fraction] = field(default_factory=lambda: [Fraction(0)])
    n: int = 0
    L: dict[tuple[str, int, int], int] = field(default_factory=dict)  # ((cls, oct, acc), n)
    chord: ChordState = field(default_factory=ChordState)
    T: list[Fraction] = field(default_factory=list)
    started: bool = False

    def copy(self, deep: bool = True) -> Self:
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    @property
    def L_prev(self) -> dict[tuple[str, int, int], int]:
        return {note: n_ for note, n_ in self.L.items() if n_ < self.n}


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
        raise NotImplementedError("A token must implement the method `is_token_type_allowed`.")

    def is_token_allowed(self, state: TranscriptionState) -> bool:
        """Returns whether this token instance is allowed in the current state.
        To check if the token type is allowed, rather than this specific instance,
        use the class method `is_token_type_allowed`.

        :param state: The current state of the automaton.
        :return: Whether the token is allowed in the current state.
        """
        return self.is_token_type_allowed(state) and self._is_token_instance_allowed(state)

    def _is_token_instance_allowed(self, state: TranscriptionState) -> bool:
        """Returns whether this token instance is allowed in the current state.
        This does NOT check for the token type, only the token instance, the type
        is checked in the wrapper function `is_token_allowed`.

        :param state: The current state of the automaton.
        :return: Whether the token is allowed in the current state.
        """
        return True  # By default, all instances are allowed if the type is allowed

    @classmethod
    @abstractmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        """Sample a valid token given the current state. Note that this token type
        must be allowed or this will raise an error.

        :param state: The current state of the automaton.
        :return: A valid token
        """
        raise NotImplementedError("A token must implement the method `_sample_valid_token`.")

    @classmethod
    def sample_valid_token(cls, state: TranscriptionState) -> Self:
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
        if not cls.is_token_type_allowed(state):
            raise ValueError(f"Token {cls} is not allowed in state {state}")
        token = cls._sample_valid_token(state)
        assert token.is_token_allowed(state), f"Token {token} is not allowed in state {state}"
        return token

    def transition_fn(self, state: TranscriptionState, in_place: bool = True) -> TranscriptionState:
        """Returns the new state after applying the token.

        :param state: The current state of the automaton.
        :param in_place: Whether to modify the state in place or return a new state.
        :return: The new state after applying the token.
        """
        assert self.is_token_allowed(state), f"Token {self} is not allowed in state {state}"
        new_state = state if in_place else state.copy()
        self._transition_fn(new_state)
        return new_state

    @abstractmethod
    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        """Modify a state IN PLACE after applying the token.

        :param state: The current state of the automaton.
        :return: The modified state.
        """
        raise NotImplementedError("A token must implement the method `_transition_fn`.")


@dataclass
class StartOfScore(Token):
    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return not state.started

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        return cls()

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.started = True
        return state


@dataclass
class EndOfScore(Token):
    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return state.started and state.S[-1] == 0 and len(state.L) == 0

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        return cls()

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

    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return state.started and state.S == [0]

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        numerator = random.randint(1, 12)
        denominator = 2 ** random.randint(1, 4)
        return cls(numerator=numerator, denominator=denominator)

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.S[0] = self.duration
        return state

    @property
    def duration(self) -> Fraction:
        """Compute the duration of the measure in quarter notes.

        The unit duration (in quarter notes) is computed using some bit manipulation
        to find the closest power of 2 (x = int.bit_length() - 1), then the duration
        is computed using 2**(2 - x)
            - For denominator = 2 (half note): 2**(2 - (2 - 1)) = 2
            - For denominator = 4 (quarter note): 2**(2 - (3 - 1)) = 1
            - For denominator = 8 (eighth note): 2**(2 - (4 - 1)) = 1/2
        """
        denominator_duration = Fraction(2 ** (3 - self.denominator.bit_length()))
        return self.real_numerator * denominator_duration


@dataclass
class KeySignature(Token):
    accidentals: int

    def __post_init__(self):
        if abs(self.accidentals) > 7:
            raise ValueError("The number of accidentals must be between -7 and 7.")

    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return (
            state.started  # SOS have been produced
            and len(state.T) == 0  # Not in tuplet
            and not state.chord.active  # Not in chord
            and not state.L  # No tied as there might be ambiguity between ties and legatos
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        return cls(accidentals=random.randint(-7, 7))

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        return state


@dataclass
class Rest(Token):
    duration: Fraction

    def __post_init__(self):
        if not self.duration in ALLOWED_DURATIONS:
            raise ValueError("The duration of a rest must be a power of 2.")

    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return (
            state.S[0] > 0  # Space left in the measure (or tuplet)
            and len(state.L) == 0  # No previous notes tied
            and not state.chord.active  # Not in chord
        )

    def _is_token_instance_allowed(self, state: TranscriptionState) -> bool:
        return self.duration <= state.S[0]

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        return cls(duration=random.choice([d for d in ALLOWED_DURATIONS if d <= state.S[0]]))

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.S[0] -= self.duration
        state.n += 1
        return state


@dataclass
class Note(Token):
    duration: Fraction
    # dots: int
    note_class: str
    octave: int
    accidental: int
    tied: bool

    _note_classes = "ABCDEFG"
    _note_alterations = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}
    _octave_range = (4, 5)

    def __post_init__(self):
        if not (
            self.duration in ALLOWED_DURATIONS
            or self.duration == Fraction(0)  # 0 allowed for chords
        ):
            raise ValueError("The duration of a note must be a power of 2 or zero.")
        if self.note_class not in self._note_classes:
            raise ValueError(f"The note class must be one of {', '.join(self._note_classes)}.")
        if self.octave < self._octave_range[0] or self.octave > self._octave_range[1]:
            raise ValueError(
                f"The octave must be between {self._octave_range[0]} and {self._octave_range[1]}."
            )
        if self.accidental not in self._note_alterations.keys():
            raise ValueError(
                f"The accidental must be one of {', '.join(map(str, self._note_alterations.keys()))}."
            )

    @property
    def note_tuple(self) -> tuple[str, int, int]:
        return self.note_class, self.octave, self.accidental

    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return state.S[0] > 0  # Space left in the measure (or tuplet)

    def _is_token_instance_allowed(self, state: TranscriptionState) -> bool:
        # If not in chord
        if not state.chord.active:
            duration_ok = 0 < self.duration  # Max duration already checked in is_token_type_allowed
            if len(state.L) != 0:  # No need to specify L_prev as not in chord
                return (
                    duration_ok  # Duration is allowed
                    and self.note_tuple in state.L.keys()  # Note completes a tied note
                )
            return duration_ok
        # If in chord
        return (
            self.duration == 0  # Only allow duration 0 in chord
            and not self.note_tuple in state.chord.notes  # Note not already in chord
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        # If not in chord
        if not state.chord.active:
            # If no notes tied
            if len(state.L) == 0:
                return cls(
                    duration=random.choice([d for d in ALLOWED_DURATIONS if d <= state.S[0]]),
                    note_class=random.choice(cls._note_classes),
                    octave=random.randint(*cls._octave_range),
                    accidental=random.choice(list(cls._note_alterations.keys())),
                    tied=random.choice([True] + [False] * 3),
                )
            # If notes tied
            note_tuple = random.choice(list(state.L.keys()))
            return cls(
                duration=random.choice([d for d in ALLOWED_DURATIONS if d <= state.S[0]]),
                note_class=note_tuple[0],
                octave=note_tuple[1],
                accidental=note_tuple[2],
                tied=random.choice([True] + [False] * 3),
            )
        # If in chord, force tie resolution first if there are tied notes (or it will cause issue
        # with an eventual note that want to be tied to a future note but hasn't been resolved yet)
        if len(state.L_prev) > 0:
            print("Tie resolution")
            note_tuple = random.choice(list(state.L_prev.keys()))
            print(note_tuple)
            return cls(
                duration=Fraction(0),
                note_class=note_tuple[0],
                octave=note_tuple[1],
                accidental=note_tuple[2],
                tied=random.choice([True] + [False] * 3),
            )
        # Make sure the note is not already in the chord (randomly sample until ok, maybe better approach?)
        while True:
            note_class = random.choice(cls._note_classes)
            octave = random.randint(*cls._octave_range)
            accidental = random.choice(list(cls._note_alterations.keys()))
            if (note_class, octave, accidental) not in state.chord.notes:
                break
        return cls(
            duration=Fraction(0),
            note_class=note_class,
            octave=octave,
            accidental=accidental,
            tied=random.choice([True] + [False] * 3),
        )

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.S[0] -= self.duration
        if self.note_tuple in state.L_prev:
            state.L.pop(self.note_tuple)
        if self.tied:
            state.L[self.note_tuple] = state.n
        if state.chord.active:
            state.chord.notes.append(self.note_tuple)
        else:
            state.n += 1
        return state


@dataclass
class ChordStart(Token):
    duration: Fraction

    def __post_init__(self):
        if not self.duration in ALLOWED_DURATIONS:
            raise ValueError("The duration of a chord must be a power of 2.")

    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return (
            state.S[0] > 0  # Space left in the measure (or tuplet)
            and not state.chord.active  # Not already in chord
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        return cls(duration=random.choice([d for d in ALLOWED_DURATIONS if d <= state.S[0]]))

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.chord.duration = self.duration
        return state


@dataclass
class ChordEnd(Token):
    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return (
            state.chord.active  # In chord
            and state.chord.can_resolve  # Chord can be resolved (at least 2 notes)
            and len(state.L_prev) == 0  # Previous ties are resolved
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        return cls()

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.S[0] -= state.chord.duration
        state.n += 1
        state.chord.reset()
        return state


@dataclass
class TupletStart(Token):
    n_actual: int
    n_real: int
    unit_duration: Fraction

    _max_nested_tuplets = 1

    def __post_init__(self):
        if not self.unit_duration in ALLOWED_DURATIONS:
            raise ValueError("The unit duration of a tuplet must be a power of 2.")
        if self.n_actual < 2 or self.n_real < 2:
            raise ValueError("A tuplet must have at n_actual and n_real > 1.")
        if self.n_actual == self.n_real:
            raise ValueError("A tuplet must have n_actual != n_real.")

    @property
    def actual_duration(self) -> Fraction:
        return self.n_actual * self.unit_duration

    @property
    def real_duration(self) -> Fraction:
        return self.n_real * self.unit_duration

    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return (
            state.S[0] > 0  # Space left in the measure
            and not state.chord.active  # Not in chord
            and len(state.T) < cls._max_nested_tuplets  # Not in another tuplet
            and min(ALLOWED_DURATIONS) * 2 <= state.S[0]  # At least 2 32nd notes left
        )

    def _is_token_instance_allowed(self, state: TranscriptionState) -> bool:
        return self.real_duration <= state.S[0]

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        # unit_duration = random.choice([d for d in ALLOWED_DURATIONS if d * 2 <= state.S[0]])
        # allowed_n_real = [n for n in range(2, 8) if n * unit_duration <= state.S[0]]
        # n_real = random.choice(allowed_n_real)
        # allowed_n_actual = [n for n in range(2, 8) if n != n_real]
        # n_actual = random.choice(allowed_n_actual)
        # return cls(
        #     n_actual=n_actual,
        #     n_real=n_real,
        #     unit_duration=unit_duration,
        # )
        return cls(
            n_actual=3,
            n_real=2,
            unit_duration=Fraction(1, 4),
        )

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.S.insert(0, self.actual_duration)
        state.T.insert(0, self.real_duration)
        return state


@dataclass
class TupletEnd(Token):
    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return (
            state.S[0] == 0  # Tuplet duration is over
            and len(state.T) > 0  # In a tuplet
            and not state.chord.active  # Not in chord
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        return cls()

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.S.pop(0)
        state.S[0] -= state.T.pop(0)  # Remove tuplet duration from measure duration and pop it
        return state


TOKEN_TYPES = [
    StartOfScore,
    EndOfScore,
    Bar,
    # KeySignature,
    Rest,
    Note,
    # ChordStart,
    # ChordEnd,
    TupletStart,
    TupletEnd,
]
