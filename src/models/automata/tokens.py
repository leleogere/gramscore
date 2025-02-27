import copy
import dataclasses
import functools
import operator
import random
from collections import defaultdict
from dataclasses import dataclass, fields, field
from abc import abstractmethod
from collections.abc import Callable
from enum import Enum
from enum import StrEnum
from fractions import Fraction
from pprint import pprint
from typing import Any
import inspect
from typing import Literal
from typing import Self
from collections import namedtuple
import partitura as pt
import partitura.score
from loguru import logger
from partitura.utils.globals import INT_TO_ALT


class SymbolicDurationType(StrEnum):
    LONG = "long", Fraction(16)
    BREVE = "breve", Fraction(8)
    WHOLE = "whole", Fraction(4)
    HALF = "half", Fraction(2)
    QUARTER = "quarter", Fraction(1)
    EIGHTH = "eighth", Fraction(1, 2)
    N16TH = "16th", Fraction(1, 4)
    N32ND = "32nd", Fraction(1, 8)
    N64TH = "64th", Fraction(1, 16)
    N128TH = "128th", Fraction(1, 32)
    N256TH = "256th", Fraction(1, 64)

    def __new__(cls, value: str, quarter_length: Fraction):
        obj = str.__new__(cls)
        obj._value_ = value
        obj.quarter_length = quarter_length
        return obj

    @classmethod
    def min(cls):
        return min(cls, key=operator.attrgetter("quarter_length"))

    @classmethod
    def max(cls):
        return max(cls, key=operator.attrgetter("quarter_length"))

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class SymbolicDuration:
    type: SymbolicDurationType
    dots: int = 0

    def __post_init__(self):
        if isinstance(self.type, str):
            object.__setattr__(self, "type", SymbolicDurationType(self.type))
        if self.dots > 3:
            raise ValueError("A note can't have more than 3 dots.")
        # Avoid issues with too short time remaining, like a 64th or 128th-long time at the end of a
        # measure that would be too short to be filled
        # if self.dots >= 2 and self.type == SymbolicDurationType.N16TH:
        #     raise ValueError("A 16th note can't have more than 1 dot.")
        # if self.dots >= 1 and self.type == SymbolicDurationType.N32ND:
        #     raise ValueError("A 32nd note can't have any dots.")

    @functools.cached_property
    def quarter_length(self) -> Fraction:
        return sum(
            (self.type.quarter_length * Fraction(1, 2) ** i for i in range(self.dots + 1)),
            Fraction(0),
        )

    def __str__(self):
        return f"{self.type.value}{'.' * self.dots}"


NoteTuple = namedtuple("NoteTuple", ["name", "octave", "accidental"])
TieProperties = namedtuple("TieProperties", ["note_tuple", "min_resolution_pos"])


class ChordState:
    def __init__(self, symbolic_duration: SymbolicDuration):
        self._symbolic_duration: SymbolicDuration = symbolic_duration
        self.notes: list[NoteTuple] = []

    @property
    def duration(self) -> SymbolicDuration:
        return self._symbolic_duration

    @property
    def num_notes(self) -> int:
        """Returns the number of notes in the chord."""
        return len(self.notes)

    @property
    def can_resolve(self):
        return self.num_notes >= 2

    def __str__(self):
        return f"Chord({self.duration}, {self.num_notes} notes)"


class SegmentCompletionState:
    """This class represent the completion state of an abstract segment (tuplet or measure)."""

    def __init__(self, duration: Fraction):
        self._duration = duration
        self._current_position = Fraction(0)

    @property
    def duration(self) -> Fraction:
        return self._duration

    @property
    def current_position(self) -> Fraction:
        return self._current_position

    @property
    def remaining_duration(self) -> Fraction:
        return self.duration - self.current_position

    @property
    def has_space_left(self) -> bool:
        return self.remaining_duration > 0

    def advance_position(self, increment: Fraction) -> None:
        """Increment the current position by a given positive duration.
        The increment must not exceed the total duration.
        """
        assert increment >= 0, "Increment must be positive."
        # assert (
        #     self._current_position + increment <= self._duration
        # ), "Increment must not exceed the total duration."
        if self._current_position + increment > self._duration:
            raise ValueError("Increment must not exceed the total duration.")
        self._current_position += increment


class TupletCompletionState(SegmentCompletionState):
    def __init__(self, n_actual: int, n_normal: int, unit_duration: SymbolicDuration):
        self.n_actual = n_actual
        self.n_normal = n_normal
        self.unit_duration = unit_duration
        super().__init__(self.actual_duration)

    @property
    def normal_duration(self) -> Fraction:
        """Returns the normal duration of the tuplet."""
        return self.n_normal * self.unit_duration.quarter_length

    @property
    def actual_duration(self) -> Fraction:
        """Returns the actual duration of the tuplet."""
        return self.n_actual * self.unit_duration.quarter_length

    @property
    def duration_multiplier(self) -> Fraction:
        return Fraction(self.n_normal, self.n_actual)

    def __str__(self):
        return f"Tuplet(pos={self.current_position}, rem={self.remaining_duration}, type={self.n_actual}:{self.n_normal} {self.unit_duration})"


class VoiceCompletionState(SegmentCompletionState):
    def __init__(self, total_time: Fraction):
        super().__init__(total_time)
        self.chord: ChordState | None = None
        self.tuplet: TupletCompletionState | None = None

    @property
    def in_chord(self) -> bool:
        return self.chord is not None

    @property
    def in_tuplet(self) -> bool:
        return self.tuplet is not None

    @property
    def completed(self) -> bool:
        return not self.has_space_left and not self.in_tuplet and not self.in_chord

    @property
    def active(self) -> bool:
        """Returns whether the voice is active, i.e. started but not completed."""
        return self.started and not self.completed

    @property
    def inactive(self) -> bool:
        return not self.active

    @property
    def started(self) -> bool:
        return self.current_position > 0 or self.in_chord or self.in_tuplet  # is in_chord needed?

    # @property
    # def prev_tied_notes(self) -> dict[NoteTuple, Fraction]:
    #     """Returns all unresolved tied notes from previous positions."""
    #     return dict(filter(lambda x: x[1] < self.current_position, self.tied_notes.items()))

    @property
    def segment_remaining_duration(self) -> Fraction:
        """Returns the remaining time in the current segment (measure or tuplet)."""
        if self.in_tuplet:
            return self.tuplet.remaining_duration
        else:
            return self.remaining_duration

    def advance_position(self, duration: Fraction) -> None:
        """Advance the current position by a given positive duration, taking into
        account an eventual tuplet (in that case will advance of the duration in
        the tuplet time referential)."""
        if self.in_tuplet:
            self.tuplet.advance_position(duration)
            super().advance_position(duration * self.tuplet.duration_multiplier)
        else:
            super().advance_position(duration)

    def __str__(self):
        return f"VoiceCompletionState(pos={self.current_position}, tot={self.duration}, rem={self.remaining_duration}, chord={self.chord}, tuplet={self.tuplet})"


class TranscriptionState:
    def __init__(self, num_voices: int = 1, initial_measure_number: int = 0):
        self._voices: list[VoiceCompletionState] = [
            VoiceCompletionState(Fraction(-1)) for _ in range(num_voices)
        ]
        self.tied_notes: dict[int, TieProperties] = {}  # pitch -> [note_tuple, min_resolution_pos]
        self._voices_initialized = False
        self.measure_number = initial_measure_number

    @property
    def num_voices(self) -> int:
        return len(self.voices)

    @property
    def voices(self) -> list[VoiceCompletionState]:
        if not self._voices:
            raise ValueError("Voices must be initialized before being accessed.")
        return self._voices

    def resolvable_tied_notes(self, position: Fraction) -> dict[int, TieProperties]:
        return {
            pitch: props
            for pitch, props in self.tied_notes.items()
            if props.min_resolution_pos <= position
        }

    def copy(self, deep: bool = True) -> Self:
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def check_for_all_voices(
        self,
        condition: Callable[[VoiceCompletionState], bool],
        filter_fn: Callable[[VoiceCompletionState], bool] | None = lambda _: True,
    ) -> bool:
        return all(condition(voice) for voice in self.voices if filter_fn(voice))

    def check_for_any_voices(
        self,
        condition: Callable[[VoiceCompletionState], bool],
        filter_fn: Callable[[VoiceCompletionState], bool] | None = lambda _: True,
    ) -> bool:
        return any(condition(voice) for voice in self.voices if filter_fn(voice))

    def are_started_voices_synced(self) -> bool:
        all_positions = [v.current_position for v in self.voices]
        return self.check_for_all_voices(
            lambda v: v.current_position == max(all_positions),
            filter_fn=operator.attrgetter("started"),
        )

    def are_started_voices_complete(self) -> bool:
        return self.are_started_voices_synced() and self.check_for_any_voices(
            operator.attrgetter("completed")
        )

    def reset_voices(self, total_time: Fraction) -> None:
        self._voices = [VoiceCompletionState(total_time) for _ in range(self.num_voices)]
        self._voices_initialized = True

    @property
    def voices_initialized(self) -> bool:
        return self._voices_initialized

    def __str__(self):
        tied_notes_str = [
            f'{nt.name}{nt.octave}{INT_TO_ALT[nt.accidental]} ({p}): {pos}'
            for p, (nt, pos) in self.tied_notes.items()
        ]
        components = [
            f"measure={self.measure_number}",
            f"tied_notes={{{', '.join(tied_notes_str)}}}",
            *[f"voice{i}={v}" for i, v in enumerate(self.voices)],
        ]
        sep = ",\n  "
        return f"TranscriptionState(\n  {sep.join(components)}\n)"


def sample_random_voice(allowed: list[int]):
    assert all(0 <= v < 8 for v in allowed), "Voice index must be between 0 and 7."
    weigths = {
        0: 1,
        1: 0.2,
        2: 0.05,
        3: 0.01,
    }
    weigths.update({i: w for i, w in zip(range(4, 8), weigths.values())})
    voice = random.choices(allowed, weights=[weigths[v] for v in allowed])[0]
    return voice


@dataclass(kw_only=True)
class Token:
    position_in_bar: Fraction | None = None

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

    def __str__(self):
        elements = [
            f"{k}={str(v)}"
            for k, v in self.__dict__.items()
            if not k.startswith("_") and v is not None
        ]
        return f"{self.__class__.__name__}({', '.join(elements)})"


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
        return (
            state.are_started_voices_complete()  # All started voices are complete
            and not state.tied_notes  # No notes tied
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        return cls()

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        return state


@dataclass
class Bar(Token):
    numerator: int
    denominator: int
    actual_duration: Fraction | None = None
    name: str | None = None

    def __post_init__(self):
        if self.denominator.bit_count() != 1:
            raise ValueError("The denominator of the TS must be a power of 2.")
        if self.numerator < 1 or self.denominator < 1:
            raise ValueError("All time signature attributes values must be positive integers.")
        if self.actual_duration is None:
            self.actual_duration = self.normal_duration
        elif self.actual_duration <= 0:
            raise ValueError("The actual duration must be positive.")
        # elif self.actual_duration > self.normal_duration:
        #     raise ValueError(
        #         "The actual duration must be less than the normal duration."
        #     )  # Must it?

    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return not state.voices_initialized or (  # Voices not initialized (first bar token)
            state.are_started_voices_complete()  # All started voices are complete
            and not state.check_for_any_voices(operator.attrgetter("in_tuplet"))  # Not in tuplet
            and not state.check_for_any_voices(operator.attrgetter("in_chord"))  # Not in chord
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        # numerator = random.randint(1, 12)
        # denominator = 2 ** random.randint(1, 4)
        numerator = 1
        denominator = 4
        return cls(numerator=numerator, denominator=denominator)

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        # Reset all voices to zero
        state.reset_voices(self.actual_duration)
        # Mark all ties as resolvable (after a bar line, all ties should be resolvable)
        for pitch, props in state.tied_notes.items():
            if props.min_resolution_pos != 0:  # Avoid resetting already resolvable ties
                state.tied_notes[pitch] = TieProperties(props.note_tuple, Fraction(0))
        # Update the measure number
        if hasattr(self, "_partitura_elements"):
            state.measure_number = self._partitura_elements[0].name
        else:
            state.measure_number += 1
        return state

    @property
    def normal_duration(self) -> Fraction:
        """Compute the normal duration of the measure in quarter notes, according to the time
        signature. Note that this is not necessarily the actual true duration of the measure,
        in case of anacrusis or incomplete measures for example.

        The unit duration (in quarter notes) is computed using some bit manipulation
        to find the closest power of 2 (x = int.bit_length() - 1), then the duration
        is computed using 2**(2 - x)
            - For denominator = 2 (half note): 2**(2 - (2 - 1)) = 2
            - For denominator = 4 (quarter note): 2**(2 - (3 - 1)) = 1
            - For denominator = 8 (eighth note): 2**(2 - (4 - 1)) = 1/2
        """
        denominator_duration = Fraction(2 ** (3 - self.denominator.bit_length()))
        return self.numerator * denominator_duration


@dataclass
class KeySignature(Token):
    accidentals: int

    def __post_init__(self):
        if abs(self.accidentals) > 7:
            raise ValueError("The number of accidentals must be between -7 and 7.")

    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        all_positions = [v.current_position for v in state.voices]
        # print(f"{all_positions = }")
        # print(state.check_for_any_voices(operator.attrgetter("in_chord")))
        return (
            all(
                pos in (Fraction(0), max(all_positions)) for pos in all_positions
            )  # All voices at the same position (or not started)
            and not state.check_for_any_voices(operator.attrgetter("in_tuplet"))  # Not in tuplet
            and not state.check_for_any_voices(operator.attrgetter("in_chord"))  # Not in chord
            # Allow ties across key signature changes?
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        return cls(accidentals=random.randint(-7, 7))

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        return state


@dataclass
class HasStaffMixin:
    """Used for elements that are tied to a staff."""

    staff_idx: int

    @property
    def staff_idx_1b(self):
        """Returns the staff index but one-based."""
        return self.staff_idx + 1


@dataclass
class HasVoiceMixin:
    """Used for elements that are tied to a voice."""

    voice_idx: int

    @property
    def voice_idx_1b(self):
        """Returns the voice index but one-based."""
        return self.voice_idx + 1


@dataclass
class Clef(Token, HasStaffMixin):
    clef: Literal["G", "F"]
    # TODO: support for octaved clefs and other clefs?

    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return state.check_for_all_voices(lambda v: not v.in_chord)

    def _is_token_instance_allowed(self, state: TranscriptionState) -> bool:
        allowed_voices = [i for i, voice in enumerate(state.voices) if not voice.in_chord]
        return self.staff_idx in allowed_voices

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        staff = random.randint(0, 1)
        if staff == 0:
            clef = "G" if random.random() < 0.9 else "F"
        else:
            clef = "F" if random.random() < 0.9 else "G"
        return cls(clef=clef, staff_idx=staff)

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        return state

    @property
    def line(self):
        return 2 if self.clef == "G" else 4

    @property
    def octave_change(self):
        return 0  # TODO: deal with this


@dataclass
class HasDurationMixin:
    """Used for elements that have a duration."""

    duration: SymbolicDuration

    @classmethod
    def get_allowed_voices_and_duration(
        cls,
        state: TranscriptionState,
        keep_empty: bool = False,
    ) -> dict[int, list[SymbolicDuration]]:
        voices_and_durations = {
            i: [
                d
                for type_ in SymbolicDurationType
                for dots in range(4)
                if (d := SymbolicDuration(type_, dots)).quarter_length
                <= v.segment_remaining_duration
                and not (d.type is SymbolicDurationType.N64TH and d.dots > 1)
                and not (d.type is SymbolicDurationType.N128TH and d.dots > 1)
                and not (d.type is SymbolicDurationType.N256TH and d.dots > 0)
                # TODO: are the 3 conditions above replaceable by d.quarter_length.denominator > min_dur.denominator?
            ]
            for i, v in enumerate(state.voices)
        }
        if not keep_empty:
            voices_and_durations = {k: v for k, v in voices_and_durations.items() if v}
        return voices_and_durations


@dataclass
class Rest(Token, HasVoiceMixin, HasStaffMixin, HasDurationMixin):
    measure_rest: bool = False
    fermata: bool = False

    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return state.check_for_any_voices(
            lambda v: (
                v.segment_remaining_duration > 0  # Space left in the measure (or tuplet)
                and not v.in_chord  # Not in chord
            )
        )

    def _is_token_instance_allowed(self, state: TranscriptionState) -> bool:
        # print(self.voice_idx)
        voice = state.voices[self.voice_idx]
        if self.measure_rest:
            duration_ok = voice.current_position == 0
        else:
            duration_ok = self.duration.quarter_length <= voice.segment_remaining_duration
        return (
            duration_ok  # Space left in the measure (or tuplet)
            and not voice.in_chord  # Not in chord
        )

    # TODO: TOFIX after global ties management
    # @classmethod
    # def _sample_valid_token(cls, state: TranscriptionState) -> Self:
    #     # TODO: fix after adding measure rests
    #     voices_and_durations = cls.get_allowed_voices_and_duration(state)
    #     # Filter out voices in chord or with tied notes
    #     voices_and_durations = {
    #         v: duration
    #         for v, duration in voices_and_durations.items()
    #         if not state.voices[v].in_chord and len(state.voices[v].tied_notes) == 0
    #     }
    #     voice_idx = sample_random_voice(list(voices_and_durations.keys()))
    #     return cls(
    #         duration=random.choice(voices_and_durations[voice_idx]),
    #         voice_idx=voice_idx,
    #         staff_idx=default_voice_to_staff(voice_idx),
    #     )

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        voice = state.voices[self.voice_idx]
        if self.measure_rest:
            voice.advance_position(voice.duration)
        else:
            voice.advance_position(self.duration.quarter_length)
        return state


@dataclass
class Note(Token, HasVoiceMixin, HasStaffMixin, HasDurationMixin):
    note_class: str
    octave: int
    accidental: int
    tied: bool
    stem_direction: Literal["up", "down"] | None = None
    grace: bool = False
    fermata: bool = False
    # Articulations
    accent: bool = False
    strong_accent: bool = False
    staccato: bool = False
    staccatissimo: bool = False
    tenuto: bool = False
    detached_legato: bool = False
    # Ornaments
    trill_mark: bool = False
    turn: bool = False
    mordent: bool = False
    inverted_mordent: bool = False
    tremolo: bool = False

    _note_classes = "ABCDEFG"
    _note_alterations = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}
    _octave_range = (0, 8)

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
        if self.stem_direction not in ["up", "down", None]:
            raise ValueError("The stem direction must be either 'up' or 'down'.")

    @property
    def articulations(self):
        articulations = [
            attr.replace("_", "-")
            for attr in [
                "accent",
                "strong_accent",
                "staccato",
                "staccatissimo",
                "tenuto",
                "detached_legato",
            ]
            if getattr(self, attr)
        ]
        return articulations

    @property
    def ornaments(self):
        ornaments = [
            attr.replace("_", "-")
            for attr in ["trill_mark", "turn", "mordent", "inverted_mordent", "tremolo"]
            if getattr(self, attr)
        ]
        return ornaments

    @property
    def note_tuple(self) -> NoteTuple:
        return NoteTuple(name=self.note_class, octave=self.octave, accidental=self.accidental)

    @property
    def pitch(self) -> int:
        return pt.score.pitch_spelling_to_midi_pitch(
            step=self.note_class,
            alter=self.accidental,
            octave=self.octave,
        )

    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return state.check_for_any_voices(lambda v: (v.segment_remaining_duration > 0))

    def _is_token_instance_allowed(self, state: TranscriptionState) -> bool:
        voice = state.voices[self.voice_idx]
        # If not in chord
        if not voice.in_chord:
            return self.duration.quarter_length <= voice.segment_remaining_duration
        # If in chord
        else:
            return (
                self.duration == voice.chord.duration  # SymbolicDuration matches chord duration
                and not self.note_tuple in voice.chord.notes  # Note not already in chord
            )

    # TODO: TOFIX after global ties management
    # @classmethod
    # def _sample_valid_token(cls, state: TranscriptionState) -> Self:
    #     voices_and_durations = cls.get_allowed_voices_and_duration(state)
    #     voice_idx = sample_random_voice(list(voices_and_durations.keys()))
    #     voice = state.voices[voice_idx]
    #     duration = random.choice(voices_and_durations[voice_idx])
    #     # TODO: randomize staff selection, and maybe assign staff to rests too?
    #     staff = default_voice_to_staff(voice_idx)
    #     # If not in chord
    #     if not voice.in_chord:
    #         # If no notes tied
    #         if len(voice.tied_notes) == 0:
    #             return cls(
    #                 duration=duration,
    #                 voice_idx=voice_idx,
    #                 note_class=random.choice(cls._note_classes),
    #                 octave=random.randint(*cls._octave_range),
    #                 accidental=random.choice(list(cls._note_alterations.keys())),
    #                 tied=random.choice([True] + [False] * 3),
    #                 staff_idx=staff,
    #             )
    #         # If notes tied
    #         note_tuple = random.choice(list(voice.tied_notes.keys()))
    #         return cls(
    #             duration=duration,
    #             voice_idx=voice_idx,
    #             note_class=note_tuple[0],
    #             octave=note_tuple[1],
    #             accidental=note_tuple[2],
    #             tied=random.choice([True] + [False] * 3),
    #             staff_idx=staff,
    #         )
    #     # If in chord, force tie resolution first if there are tied notes (or it will cause issue
    #     # with an eventual note that want to be tied to a future note but hasn't been resolved yet)
    #     if len(voice.prev_tied_notes) > 0:
    #         print("Tie resolution")
    #         note_tuple = random.choice(list(voice.prev_tied_notes.keys()))
    #         print(note_tuple)
    #         return cls(
    #             duration=voice.chord.duration,
    #             voice_idx=voice_idx,
    #             note_class=note_tuple[0],
    #             octave=note_tuple[1],
    #             accidental=note_tuple[2],
    #             tied=random.choice([True] + [False] * 3),
    #             staff_idx=staff,
    #         )
    #     # Make sure the note is not already in the chord (randomly sample until ok, maybe better approach?)
    #     while True:
    #         note_class = random.choice(cls._note_classes)
    #         octave = random.randint(*cls._octave_range)
    #         accidental = random.choice(list(cls._note_alterations.keys()))
    #         if (note_class, octave, accidental) not in voice.chord.notes:
    #             break
    #     return cls(
    #         duration=voice.chord.duration,
    #         voice_idx=voice_idx,
    #         note_class=note_class,
    #         octave=octave,
    #         accidental=accidental,
    #         tied=random.choice([True] + [False] * 3),
    #         staff_idx=staff,
    #     )

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        voice = state.voices[self.voice_idx]
        # If resolving a previous tie, remove the note from the tied notes
        if self.pitch in state.resolvable_tied_notes(voice.current_position):
            state.tied_notes.pop(self.pitch)
        # If tied, add the note to tied notes
        if self.tied:
            state.tied_notes[self.pitch] = TieProperties(
                note_tuple=self.note_tuple,
                min_resolution_pos=voice.current_position + self.duration.quarter_length,
            )
        # Only advance position if in chord
        if voice.in_chord:
            voice.chord.notes.append(self.note_tuple)
        else:
            voice.advance_position(self.duration.quarter_length)
        return state


@dataclass
class ChordStart(Token, HasVoiceMixin, HasDurationMixin):
    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return state.check_for_any_voices(
            lambda v: (
                v.segment_remaining_duration > 0  # Space left in the measure (or tuplet)
                and not v.in_chord  # Not in chord
            )
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        voices_and_durations = cls.get_allowed_voices_and_duration(state)
        # Discard voices that are already in a chord
        valid_voice_idxs = [
            v
            for v, d in voices_and_durations.items()
            if not state.voices[v].in_chord and len(d) > 0
        ]
        voice_idx = sample_random_voice(valid_voice_idxs)
        return cls(
            duration=random.choice(voices_and_durations[voice_idx]),
            voice_idx=voice_idx,
        )

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.voices[self.voice_idx].chord = ChordState(self.duration)
        return state


@dataclass
class ChordEnd(Token, HasVoiceMixin):
    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return state.check_for_any_voices(
            lambda v: (
                v.in_chord  # In chord
                and v.chord.can_resolve  # Chord can be resolved (at least 2 notes)
            )
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        # Get voices with active chord
        valid_voice_idxs = [
            i for i, v in enumerate(state.voices) if v.in_chord and v.chord.can_resolve
        ]
        voice_idx = sample_random_voice(valid_voice_idxs)
        return cls(voice_idx=voice_idx)

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        voice = state.voices[self.voice_idx]
        voice.advance_position(voice.chord.duration.quarter_length)
        voice.chord = None
        return state


@dataclass
class TupletStart(Token, HasVoiceMixin):
    n_actual: int
    n_normal: int
    unit_duration: SymbolicDuration

    # _allowed_divisions = [3, 5]
    _allowed_divisions = [2, 3, 5]

    def __post_init__(self):
        pass
        # if self.n_actual < 2 or self.n_normal < 2:
        #     raise ValueError("A tuplet must have at n_actual and n_normal > 1.")
        # if self.n_actual == self.n_normal:
        #     raise ValueError("A tuplet must have n_actual != n_normal.")

    @property
    def actual_duration(self) -> Fraction:
        return self.n_actual * self.unit_duration.quarter_length

    @property
    def normal_duration(self) -> Fraction:
        return self.n_normal * self.unit_duration.quarter_length

    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        return state.check_for_any_voices(
            lambda v: (
                v.segment_remaining_duration > 0  # Space left in the measure
                and not v.in_chord  # Not in chord
                and not v.in_tuplet  # Not already in tuplet
                and SymbolicDurationType.min().quarter_length * min(cls._allowed_divisions)
                <= v.segment_remaining_duration  # At least 2 32nd notes left
            )
        )

    def _is_token_instance_allowed(self, state: TranscriptionState) -> bool:
        return self.normal_duration <= state.voices[self.voice_idx].segment_remaining_duration

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        voices_and_unit_types = {
            i: [
                SymbolicDuration(d)
                for d in SymbolicDurationType
                if d.quarter_length * min(cls._allowed_divisions) <= v.segment_remaining_duration
            ]
            for i, v in enumerate(state.voices)
            if (not v.in_chord and not v.in_tuplet)  # Not in chord  # Not already in tuplet
        }
        voices_and_unit_types = {k: v for k, v in voices_and_unit_types.items() if v}
        voice_idx = sample_random_voice(list(voices_and_unit_types.keys()))
        unit_duration = random.choice(voices_and_unit_types[voice_idx])
        allowed_n_normal = [
            n
            for n in cls._allowed_divisions
            if n * unit_duration.quarter_length
            <= state.voices[voice_idx].segment_remaining_duration
        ]
        n_normal = random.choice(allowed_n_normal)
        allowed_n_actual = [n for n in cls._allowed_divisions if n != n_normal]
        n_actual = random.choice(allowed_n_actual)
        return cls(
            voice_idx=voice_idx,
            n_actual=n_actual,
            n_normal=n_normal,
            unit_duration=unit_duration,
        )
        # return cls(
        #     n_actual=3,
        #     n_normal=2,
        #     unit_duration=SymbolicDurationType.N32ND,
        # )

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.voices[self.voice_idx].tuplet = TupletCompletionState(
            n_actual=self.n_actual,
            n_normal=self.n_normal,
            unit_duration=self.unit_duration,
        )
        return state


@dataclass
class TupletEnd(Token, HasVoiceMixin):
    @classmethod
    def is_token_type_allowed(cls, state: TranscriptionState) -> bool:
        # return (
        #     state.tuplet.active  # In a tuplet
        #     and not state.chord.active  # Not in chord
        #     and state.tuplet.remaining_duration == 0  # Tuplet duration is over
        # )
        return state.check_for_any_voices(
            lambda v: (
                v.in_tuplet  # In a tuplet
                and not v.in_chord  # Not in chord
                and not v.tuplet.has_space_left  # Tuplet duration is over
            )
        )

    @classmethod
    def _sample_valid_token(cls, state: TranscriptionState) -> Self:
        # Get voices with active tuplet
        valid_voice_idxs = [
            i
            for i, v in enumerate(state.voices)
            if v.in_tuplet and not v.tuplet.has_space_left and not v.in_chord
        ]
        voice_idx = sample_random_voice(valid_voice_idxs)
        return cls(voice_idx=voice_idx)

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.voices[self.voice_idx].tuplet = None
        return state


TOKEN_TYPES = [
    StartOfScore,
    EndOfScore,
    Bar,
    KeySignature,
    Clef,
    Rest,
    Note,
    ChordStart,
    ChordEnd,
    TupletStart,
    TupletEnd,
]


def fill_rests(part: pt.score.Part) -> None:
    """
    Fill rests in incomplete voices of a part.
    TODO: refactor and move it somewhere more logical

    :param part: Part to fill rests in
    """
    divisions = int(part.quarter_duration_map(0))
    logger.trace(f"{divisions = }")

    def create_rests(
        duration: Fraction,
        largest_last: bool = False,
    ) -> list[tuple[pt.score.Rest, SymbolicDuration]]:
        # print(f"Creating rests for duration {duration}")
        position = Fraction(0)
        allowed_durations = iter(SymbolicDurationType)
        tested_duration = next(allowed_durations)
        rests: list[pt.score.Rest, SymbolicDuration] = []
        while position < duration:
            # print(f"{position=} \t{duration=} \t{tested_duration=}")
            if position + tested_duration.quarter_length <= duration:
                # print(f"Adding rest of duration {tested_duration.value} at position {position}")
                rests.append(
                    (
                        pt.score.Rest(symbolic_duration={"type": tested_duration.value}),
                        tested_duration,
                    )
                )
                position += tested_duration.quarter_length
            else:
                try:
                    tested_duration = next(allowed_durations)
                except StopIteration:
                    raise ValueError(
                        f"Cannot fill rest of duration {duration} at position {position}"
                    )
        if largest_last:
            rests = list(reversed(rests))
        return rests

    for measure in part.measures:
        logger.trace(f"Measure {measure.number}")
        start_time = measure.start.t
        end_time = measure.end.t
        notes_and_tuplets = list(
            filter(
                lambda n: not isinstance(n, pt.score.GraceNote),
                part.iter_all(
                    # [pt.score.GenericNote, pt.score.Tuplet],
                    pt.score.GenericNote,
                    start_time,
                    end_time,
                    include_subclasses=True,
                ),
            )
        )
        voices = defaultdict(list)
        for note_or_tuplet in notes_and_tuplets:
            idx = (
                note_or_tuplet.voice
                if isinstance(note_or_tuplet, pt.score.GenericNote)
                else note_or_tuplet.start_note.voice
            )
            voices[idx].append(note_or_tuplet)
        for voice_idx, voice_notes_and_tups in voices.items():
            time = start_time
            for note_or_tup in voice_notes_and_tups:
                if isinstance(note_or_tup, pt.score.Tuplet):
                    pass
                else:
                    note = note_or_tup
                    if note.start.t > time:
                        # print(
                        #     f"Voice {voice_idx} has a space from {time} to {note.start.t} "
                        #     f"(measure {measure.number}, from {Fraction(time - start_time, divisions)} "
                        #     f"to {Fraction(note.start.t - start_time, divisions)})"
                        # )
                        rests = create_rests(Fraction(note.start.t - time, divisions))
                        # print(rests)
                        for rest, dur in rests:
                            start = time
                            end = start + int(dur.quarter_length * divisions)
                            rest.voice = voice_idx
                            rest.staff = note.staff
                            part.add(rest, start=start, end=end)
                            time = end
                        # print(end, note.start.t)
                        assert end == note.start.t
                time = note_or_tup.end.t
            if time < end_time:
                # print(
                #     f"Voice {voice_idx} has a space from {time} to {end_time} "
                #     f"(measure {measure.number}, from {Fraction(time - start_time, divisions)} "
                #     f"to {Fraction(end_time - start_time, divisions)})"
                # )
                rests = create_rests(Fraction(end_time - time, divisions), largest_last=True)
                for rest, dur in rests:
                    start = time
                    end = time + int(dur.quarter_length * divisions)
                    rest.voice = voice_idx
                    rest.staff = note.staff
                    part.add(rest, start=start, end=end)
                    time = end
