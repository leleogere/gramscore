import copy
import dataclasses
import functools
import random
from dataclasses import dataclass, fields, field
from abc import abstractmethod
from collections.abc import Callable
from enum import Enum
from enum import StrEnum
from fractions import Fraction
from typing import Any
import inspect
from typing import Self
from collections import namedtuple
import partitura as pt
import partitura.score

import param


class DurationType(StrEnum):
    WHOLE = "whole", Fraction(4)
    HALF = "half", Fraction(2)
    QUARTER = "quarter", Fraction(1)
    EIGHTH = "eighth", Fraction(1, 2)
    SIXTEENTH = "16th", Fraction(1, 4)
    THIRTY_SECOND = "32nd", Fraction(1, 8)

    def __new__(cls, value: str, quarter_length: Fraction):
        obj = str.__new__(cls)
        obj._value_ = value
        obj.quarter_length = quarter_length
        return obj

    @classmethod
    def min(cls):
        return cls.THIRTY_SECOND

    @classmethod
    def max(cls):
        return cls.WHOLE


@dataclass(frozen=True)
class Duration:
    type: DurationType
    dots: int = 0

    def __post_init__(self):
        if self.dots > 2:
            raise ValueError("A note can't have more than 2 dots.")
        # Avoid issues with too short time remaining, like a 64th or 128th-long time at the end of a
        # measure that would be too short to be filled
        # if self.dots >= 2 and self.type == DurationType.SIXTEENTH:
        #     raise ValueError("A 16th note can't have more than 1 dot.")
        # if self.dots >= 1 and self.type == DurationType.THIRTY_SECOND:
        #     raise ValueError("A 32nd note can't have any dots.")

    @functools.cached_property
    def quarter_length(self) -> Fraction:
        return sum(
            (self.type.quarter_length * Fraction(1, 2) ** i for i in range(self.dots + 1)),
            Fraction(0),
        )

    #
    # @classmethod
    # def _get_duration_or_fraction(cls, object):
    #     if isinstance(object, cls):
    #         return object.quarter_length
    #     elif isinstance(object, Fraction):
    #         return object
    #     else:
    #         raise ValueError(f"Object {object} is not a Duration or a Fraction.")
    #
    # def __add__(self, other):
    #     return self.quarter_length + self._get_duration_or_fraction(other)
    #
    # def __radd__(self, other):
    #     return self._get_duration_or_fraction(other) + self.quarter_length
    #
    # def __sub__(self, other):
    #     return self.quarter_length - self._get_duration_or_fraction(other)
    #
    # def __rsub__(self, other):
    #     return self._get_duration_or_fraction(other) - self.quarter_length


NoteTuple = namedtuple("NoteTuple", ["name", "octave", "accidental"])


@dataclass
class ChordState:
    duration: Duration | None = None
    notes: list[NoteTuple] = field(default_factory=list)

    @property
    def num_notes(self) -> int:
        """Returns the number of notes in the chord."""
        return len(self.notes)

    @property
    def active(self) -> bool:
        """Returns whether the chord is currently being constructed."""
        return self.duration is not None

    @property
    def can_resolve(self) -> bool:
        """Returns whether the chord can be resolved (produce the ChordEnd token)."""
        assert self.active, "Chord is not active."
        return self.num_notes >= 2

    def reset(self):
        self.duration = None
        self.notes = []


@dataclass
class TupletState:
    n_actual: int = 0
    n_real: int = 0
    unit_duration_type: DurationType | None = None
    remaining_time: Fraction = Fraction(0)

    @property
    def active(self) -> bool:
        """Returns whether the tuplet is currently being constructed."""
        return self.unit_duration_type is not None

    @property
    def real_duration(self) -> Fraction:
        """Returns the real duration of the tuplet."""
        return self.n_real * self.unit_duration_type.quarter_length

    @property
    def actual_duration(self) -> Fraction:
        """Returns the actual duration of the tuplet."""
        return self.n_actual * self.unit_duration_type.quarter_length

    @property
    def duration_multiplier(self) -> Fraction:
        return Fraction(self.n_real, self.n_actual)

    def reset(self):
        self.n_actual = 0
        self.n_real = 0
        self.unit_duration_type = None
        self.remaining_time = Fraction(0)


@dataclass
class TranscriptionState:
    measure_remaining_time: Fraction = Fraction(0)
    n: int = 0
    tied_notes: dict[NoteTuple, int] = field(default_factory=dict)
    chord: ChordState = field(default_factory=ChordState)
    tuplet: TupletState = field(default_factory=TupletState)
    started: bool = False

    def copy(self, deep: bool = True) -> Self:
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    @property
    def prev_tied_notes(self) -> dict[NoteTuple, int]:
        return {note: n_ for note, n_ in self.tied_notes.items() if n_ < self.n}

    @property
    def section_remaining_time(self) -> Fraction:
        if self.tuplet.active:
            return self.tuplet.remaining_time
        else:
            return self.measure_remaining_time

    def decrement_section_remaining_time(self, duration: Fraction) -> None:
        if self.tuplet.active:
            self.tuplet.remaining_time -= duration
            self.measure_remaining_time -= duration * self.tuplet.duration_multiplier
        else:
            self.measure_remaining_time -= duration


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
        return True

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        return cls()

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        return state


@dataclass
class EndOfScore(Token):
    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return state.measure_remaining_time == 0 and len(state.tied_notes) == 0

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
        return (
            state.measure_remaining_time == 0 and not state.tuplet.active and not state.chord.active
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        # numerator = random.randint(1, 12)
        # denominator = 2 ** random.randint(1, 4)
        numerator = 1
        denominator = 4
        return cls(numerator=numerator, denominator=denominator)

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.measure_remaining_time = self.quarter_length
        return state

    @property
    def quarter_length(self) -> Fraction:
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
            not state.tuplet.active  # Not in tuplet
            and not state.chord.active  # Not in chord
            and not state.tied_notes  # No tied as there might be ambiguity between ties and legatos
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        return cls(accidentals=random.randint(-7, 7))

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        return state


@dataclass
class NoteLike(Token):
    """Common behaviour for Note, Rest and ChordStart tokens."""

    # TODO: Add ChordStart

    duration: Duration

    def is_duration_accepted(self, state: TranscriptionState) -> bool:
        return self.duration.quarter_length <= state.section_remaining_time

    @classmethod
    def get_allowed_duration(cls, state: TranscriptionState) -> list[Duration]:
        from pprint import pprint

        durations = [Duration(type_, dots) for type_ in DurationType for dots in range(3)]
        # pprint(durations)
        return [
            d
            for type_ in DurationType
            for dots in range(3)
            if (d := Duration(type_, dots)).quarter_length <= state.section_remaining_time
            and not (d.type is DurationType.SIXTEENTH and d.dots > 1)
            and not (d.type is DurationType.THIRTY_SECOND and d.dots > 0)
        ]


@dataclass
class Rest(NoteLike):
    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return (
            state.section_remaining_time > 0  # Space left in the measure (or tuplet)
            and len(state.tied_notes) == 0  # No previous notes tied
            and not state.chord.active  # Not in chord
        )

    def _is_token_instance_allowed(self, state: TranscriptionState) -> bool:
        return self.is_duration_accepted(state)

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        # from pprint import pprint
        #
        # print(state.measure_remaining_time)
        durations = cls.get_allowed_duration(state)
        # quarter_lengths = [d.quarter_length for d in durations]
        # pprint(list(zip(durations, quarter_lengths)))
        return cls(random.choice(durations))

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.decrement_section_remaining_time(self.duration.quarter_length)
        state.n += 1
        return state


@dataclass
class Note(NoteLike):
    duration: Duration
    note_class: str
    octave: int
    accidental: int
    tied: bool

    _note_classes = "ABCDEFG"
    _note_alterations = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}
    _octave_range = (4, 5)

    def __post_init__(self):
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
        return state.section_remaining_time > 0  # Space left in the measure (or tuplet)

    def _is_token_instance_allowed(self, state: TranscriptionState) -> bool:
        # If not in chord
        if not state.chord.active:
            duration_ok = self.is_duration_accepted(state)
            # If notes tied, resolve them
            if len(state.tied_notes) != 0:  # No need to specify L_prev as not in chord
                return (
                    duration_ok  # Duration is allowed
                    and self.note_tuple in state.tied_notes.keys()  # Note completes a tied note
                )
            # If no notes tied, any note is allowed
            return duration_ok
        # If in chord
        return (
            self.duration == state.chord.duration  # Duration matches chord duration
            and not self.note_tuple in state.chord.notes  # Note not already in chord
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        durations = cls.get_allowed_duration(state)
        # from pprint import pprint
        #
        # print(state.measure_remaining_time)
        # quarter_lengths = [d.quarter_length for d in durations]
        # pprint(list(zip(durations, quarter_lengths)))
        duration = random.choice(durations)
        # If not in chord
        if not state.chord.active:
            # If no notes tied
            if len(state.tied_notes) == 0:
                return cls(
                    duration=duration,
                    note_class=random.choice(cls._note_classes),
                    octave=random.randint(*cls._octave_range),
                    accidental=random.choice(list(cls._note_alterations.keys())),
                    tied=random.choice([True] + [False] * 3),
                )
            # If notes tied
            note_tuple = random.choice(list(state.tied_notes.keys()))
            return cls(
                duration=duration,
                note_class=note_tuple[0],
                octave=note_tuple[1],
                accidental=note_tuple[2],
                tied=random.choice([True] + [False] * 3),
            )
        # If in chord, force tie resolution first if there are tied notes (or it will cause issue
        # with an eventual note that want to be tied to a future note but hasn't been resolved yet)
        if len(state.prev_tied_notes) > 0:
            print("Tie resolution")
            note_tuple = random.choice(list(state.prev_tied_notes.keys()))
            print(note_tuple)
            return cls(
                duration=state.chord.duration,
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
            duration=state.chord.duration,
            note_class=note_class,
            octave=octave,
            accidental=accidental,
            tied=random.choice([True] + [False] * 3),
        )

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        if self.note_tuple in state.prev_tied_notes:
            state.tied_notes.pop(self.note_tuple)
        if self.tied:
            state.tied_notes[self.note_tuple] = state.n
        if state.chord.active:
            state.chord.notes.append(self.note_tuple)
        else:
            state.decrement_section_remaining_time(self.duration.quarter_length)
            state.n += 1
        return state


@dataclass
class ChordStart(NoteLike):
    duration: Duration

    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return (
            state.section_remaining_time > 0  # Space left in the measure (or tuplet)
            and not state.chord.active  # Not already in chord
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        duration = random.choice(cls.get_allowed_duration(state))
        return cls(duration)

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
            and len(state.prev_tied_notes) == 0  # Previous ties are resolved
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        return cls()

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.decrement_section_remaining_time(state.chord.duration.quarter_length)
        state.n += 1
        state.chord.reset()
        return state


@dataclass
class TupletStart(Token):
    n_actual: int
    n_real: int
    unit_duration_type: DurationType

    _allowed_divisions = [3, 5]

    def __post_init__(self):
        if self.n_actual < 2 or self.n_real < 2:
            raise ValueError("A tuplet must have at n_actual and n_real > 1.")
        if self.n_actual == self.n_real:
            raise ValueError("A tuplet must have n_actual != n_real.")

    @property
    def actual_duration(self) -> Fraction:
        return self.n_actual * self.unit_duration_type.quarter_length

    @property
    def real_duration(self) -> Fraction:
        return self.n_real * self.unit_duration_type.quarter_length

    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return (
            state.section_remaining_time > 0  # Space left in the measure
            and not state.chord.active  # Not in chord
            and not state.tuplet.active  # Not in tuplet
            and DurationType.min().quarter_length * min(cls._allowed_divisions)
            <= state.section_remaining_time  # At least 2 32nd notes left
        )

    def _is_token_instance_allowed(self, state: TranscriptionState) -> bool:
        return self.real_duration <= state.section_remaining_time

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        unit_duration_type: DurationType = random.choice(
            [
                d
                for d in DurationType
                if d.quarter_length * min(cls._allowed_divisions) <= state.section_remaining_time
            ]
        )
        allowed_n_real = [
            n
            for n in cls._allowed_divisions
            if n * unit_duration_type.quarter_length <= state.section_remaining_time
        ]
        n_real = random.choice(allowed_n_real)
        allowed_n_actual = [n for n in cls._allowed_divisions if n != n_real]
        n_actual = random.choice(allowed_n_actual)
        return cls(
            n_actual=n_actual,
            n_real=n_real,
            unit_duration_type=unit_duration_type,
        )
        # return cls(
        #     n_actual=3,
        #     n_real=2,
        #     unit_duration_type=DurationType.THIRTY_SECOND,
        # )

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.tuplet.n_actual = self.n_actual
        state.tuplet.n_real = self.n_real
        state.tuplet.unit_duration_type = self.unit_duration_type
        state.tuplet.remaining_time = self.actual_duration
        return state


@dataclass
class TupletEnd(Token):
    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return (
            state.tuplet.active  # In a tuplet
            and not state.chord.active  # Not in chord
            and state.tuplet.remaining_time == 0  # Tuplet duration is over
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        return cls()

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.tuplet.reset()
        return state


TOKEN_TYPES = [
    EndOfScore,
    Bar,
    KeySignature,
    Rest,
    Note,
    ChordStart,
    ChordEnd,
    TupletStart,
    TupletEnd,
]
