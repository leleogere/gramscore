import functools
import itertools
import operator
import warnings
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pprint import pprint
import math
from typing import NamedTuple

import partitura as pt
import partitura.score
from loguru import logger

from models.automata.voice_reordering import reorder_voices
from models.automata.tokens import HasVoiceMixin
from models.automata.tokens import SymbolicDurationType
from models.automata.tokens import SymbolicDuration
from models.automata.tokens import Token
from models.automata.tokens import TranscriptionState
from models.automata.tokens import (
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
)
import models.automata.tokens as tk
from utils.ansi_colors import Colors
from utils.musicxml import voice_absolute_into_voice_relative_to_staff


def token_sequence_to_partitura(
    tokens: list[Token],
    states: list[TranscriptionState] | None = None,
) -> pt.score.Part:
    """Convert a sequence of tokens to a Partitura part.
    If the states are not provided, they will be computed from the tokens sequence.

    :param tokens: The sequence of tokens to convert.
    :param states: The sequence of states corresponding to the tokens.
    :return: The Partitura part.
    """
    if states is None:
        num_voices = max(
            token.voice_idx + 1 for token in tokens if isinstance(token, HasVoiceMixin)
        )
        from models.automata.automata import ConstrainedAutomata

        automata = ConstrainedAutomata(num_voices=num_voices)
        states = []
        for token in tokens:
            states.append(automata.state.copy())
            automata.transition(token)

    if len(tokens) != len(states):
        raise ValueError("The number of tokens and states must be the same.")

    # Get all possible divisions per measure (32 is default to allow for double-dotted 32nd notes)
    all_divisions = {32}
    for token, state in zip(tokens, states):
        if isinstance(token, TupletStart):
            all_divisions.add(token.n_actual)
    global_division = math.lcm(*all_divisions)
    logger.trace(f"{global_division = }")

    def to_partitura_duration(d: Fraction | SymbolicDuration) -> int:
        """Converts a duration in quarter notes into a duration."""
        if isinstance(d, SymbolicDuration):
            d = d.quarter_length
        duration_in_div = d * global_division
        assert (
            duration_in_div.denominator == 1
        ), f"Partitura duration is supposed to be an int, got {duration_in_div}."
        return duration_in_div.numerator

    score = pt.score.Part("P1", quarter_duration=global_division)
    measure_long_objets: set[pt.score.Rest | pt.score.Measure] = set()
    measure = None
    # We store pending clefs, so that they are added when the next note of the staff is produced
    pending_clefs = {}  # staff -> clef

    voices_info = [
        {
            "time": 0,
            "tuplet": None,  # {"pt_obj": pt.score.Tuplet(), "notes": []}
            "grace_notes": [],
        }
        for _ in range(states[0].num_voices)
    ]
    tied_notes = {}
    for i, (token, state) in enumerate(zip(tokens, states)):
        # print(state.n, time, token, state)
        # print(
        #     # f"{Colors.RED}{state.}{Colors.END} "
        #     f"{Colors.GREEN}{[float(v['time'] / global_division) for v in voices_info]}{Colors.END} "
        #     f"{Colors.BLUE}{token}{Colors.END} "
        #     f"{Colors.RED}{state}{Colors.END}"
        # )

        # SOS
        if isinstance(token, StartOfScore):
            continue

        # EOS
        if isinstance(token, EndOfScore):
            time = max(v["time"] for v in voices_info)
            for obj in measure_long_objets:
                # Remove and re-add objects with fixed end time
                init_start = obj.start.t
                score.remove(obj)
                score.add(obj, start=init_start, end=time)
            measure_long_objets.clear()  # Should not be needed as we are at the end
            # Add the final barline
            score.add(pt.score.Barline("light-heavy"))

        # Bar
        elif isinstance(token, Bar):
            # Resync all voices in case of empty voices in previous bars
            time = max(v["time"] for v in voices_info)
            if (
                measure is not None and time == measure.start.t
            ):  # In case of empty measure with only full rests
                assert token.is_nominal, "Non-nominal measures should not be empty"
                time += to_partitura_duration(token.nominal_duration)
            for v in voices_info:
                v["time"] = time

            # print(f"Measure length: {token.actual_duration} ({duration})")

            # Set last measure and measure rests end time
            for obj in measure_long_objets:
                # Remove and re-add objects with fixed end time
                init_start = obj.start.t
                score.remove(obj)
                score.add(obj, start=init_start, end=time)
            measure_long_objets.clear()
            # Start new measure
            measure = pt.score.Measure(number=state.measure_number, name=state.measure_number)
            measure_long_objets.add(measure)
            ts = pt.score.TimeSignature(token.numerator, token.denominator)
            # print(time, duration)
            score.add(measure, start=time, end=None)  # End will be set at next bar
            prev_ts = next(score.last_point.iter_prev(cls=pt.score.TimeSignature), None)
            # Only add TS if different one than previous one
            if prev_ts is None or prev_ts.beats != ts.beats or prev_ts.beat_type != ts.beat_type:
                score.add(ts, start=time)

        # Key signature
        elif isinstance(token, KeySignature):
            time = max(v["time"] for v in voices_info)
            ks = pt.score.KeySignature(token.accidentals, mode=None)
            score.add(ks, start=time)

        # Clef
        elif isinstance(token, Clef):
            time = max(
                (
                    note.end.t
                    for note in score.iter_all(cls=pt.score.GenericNote, include_subclasses=True)
                    if voice_absolute_into_voice_relative_to_staff(note.voice, one_based=True)[1]
                    == token.staff_idx_1b
                ),
                default=0,
            )
            clef = pt.score.Clef(
                staff=token.staff_idx_1b,
                sign=token.clef,
                line=token.line,
                octave_change=token.octave_change,
            )
            if clef.staff in pending_clefs:
                logger.warning("Another clef is already pending for this staff, replacing it")
            # In case of start of measure, put the clef right away
            if time == measure.start.t:
                score.add(clef, start=time)
            # Else, delay it until the next note in the staff or next measure
            else:
                pending_clefs[token.staff_idx] = clef

        # Note and rest
        elif isinstance(token, (Note, Rest)):
            voice_info = voices_info[token.voice_idx]
            voice_state = state.voices[token.voice_idx]
            duration = token.duration.quarter_length  # Measure rests' duration will be fixed later
            time = voice_info["time"]
            # print(duration)

            # Add clef if there is one pending
            # We take the default staff of the voice, not the actual staff, because clef changes
            # like measure 55 in Liszt/Concert_Etude_S145/1, that appear before the note in the
            # lower staff
            default_staff = voice_absolute_into_voice_relative_to_staff(token.voice_idx)[1]
            if default_staff in pending_clefs:
                score.add(pending_clefs.pop(default_staff), start=time)

            # Rest
            if isinstance(token, Rest):
                note = pt.score.Rest(voice=token.voice_idx_1b, staff=token.staff_idx_1b)
            # Note
            else:
                cls = (
                    pt.score.Note
                    if not token.grace
                    else functools.partial(pt.score.GraceNote, grace_type="acciaccatura")
                )
                note = cls(
                    step=token.note_class,
                    octave=token.octave,
                    alter=token.accidental,
                    voice=token.voice_idx_1b,
                    staff=token.staff_idx_1b,
                    stem_direction=token.stem_direction,
                    articulations=token.articulations,
                    ornaments=token.ornaments,
                )

            if token.fermata:
                note.fermata = pt.score.Fermata()
            if isinstance(token, Rest) and token.measure_rest:
                note.symbolic_duration = {}
                measure_long_objets.add(note)  # Fix duration at next bar
            elif isinstance(token, Rest) or (isinstance(token, Note) and not token.grace):
                note.symbolic_duration = {
                    "type": token.duration.type.value,
                    "dots": token.duration.dots,
                }
                # Update duration within tuplets
                if state.voices[token.voice_idx].in_tuplet:
                    voice_info["tuplet"]["notes"].append(note)
                    duration *= voice_state.tuplet.duration_multiplier
                    note.symbolic_duration = {
                        "actual_notes": voice_state.tuplet.n_actual,
                        "normal_notes": voice_state.tuplet.n_normal,
                        **note.symbolic_duration,
                    }
                    # print(f"  > {duration = }")
            if isinstance(token, Note) and token.grace:
                note.symbolic_duration = {"type": "eighth"}
                duration = 0
                if voice_info["grace_notes"]:
                    last_grace = voice_info["grace_notes"][-1]
                    last_grace.grace_next = note
                    note.grace_prev = last_grace
                voice_info["grace_notes"].append(note)
            # When first non-grace note, set the main note and clear the list
            elif voice_info["grace_notes"]:
                voice_info["grace_notes"][-1].grace_next = note
                voice_info["grace_notes"] = []

            duration = to_partitura_duration(duration)

            if isinstance(token, Note):
                # Resolve previous ties
                if token.note_tuple in tied_notes.keys():
                    prev_note = tied_notes[token.note_tuple]
                    note.tie_prev = prev_note
                    prev_note.tie_next = note
                    tied_notes.pop(token.note_tuple)
                # Deal with future ties
                if token.tied:
                    tied_notes[token.note_tuple] = note

            score.add(note, start=time, end=time + duration)
            if not voice_state.in_chord and not (isinstance(token, Rest) and token.measure_rest):
                voice_info["time"] += duration

        # Chord start
        elif isinstance(token, ChordStart):
            pass

        # Chord end
        elif isinstance(token, ChordEnd):
            duration = states[i - 1].voices[token.voice_idx].chord.duration.quarter_length
            voice = state.voices[token.voice_idx]
            if voice.in_tuplet:
                duration *= voice.tuplet.duration_multiplier
            duration = to_partitura_duration(duration)
            voices_info[token.voice_idx]["time"] += duration

        # Tuplet start
        elif isinstance(token, TupletStart):
            voices_info[token.voice_idx]["tuplet"] = {
                "pt_obj": pt.score.Tuplet(
                    actual_notes=token.n_actual,
                    normal_notes=token.n_normal,
                    actual_type=token.unit_duration.type.value,
                    actual_dots=token.unit_duration.dots,
                    normal_type=token.unit_duration.type.value,
                    normal_dots=token.unit_duration.dots,
                ),
                "notes": [],
            }

        # Tuplet end
        elif isinstance(token, TupletEnd):
            voice_info = voices_info[token.voice_idx]
            tuplet = voice_info["tuplet"]
            # Get the highest note starting at the beginning of the tuplet
            # Only needed when a chord is the first event of the tuplet, so that the <tuplet>
            # tag is associated to the correct note at export time
            first_note = max(
                filter(lambda n: n.start == tuplet["notes"][0].start, tuplet["notes"]),
                key=lambda x: (x.midi_pitch, x.step) if isinstance(x, pt.score.Note) else -1,
            )
            # Same is the last event is a chord
            last_note = max(
                filter(lambda n: n.end == tuplet["notes"][-1].end, tuplet["notes"]),
                key=lambda x: (x.midi_pitch, x.step) if isinstance(x, pt.score.Note) else -1,
            )
            tuplet["pt_obj"].start_note = first_note
            tuplet["pt_obj"].end_note = last_note
            # print(tuplet["pt_obj"].start_note, tuplet["pt_obj"].end_note)
            # print(f"{tuplet["pt_obj"].start_note == tuplet["pt_obj"].end_note = }")
            # print(f"{tuplet["pt_obj"].start_note is tuplet["pt_obj"].end_note = }")
            score.add(tuplet["pt_obj"], start=first_note.start.t, end=last_note.end.t)
            voice_info["tuplet"] = None

        # Error
        else:
            raise ValueError(f"Unknown token {token}.")

    # Eventually set the end of non-ending objects to the final time
    for obj in measure_long_objets:
        init_start = obj.start.t
        score.remove(obj)
        score.add(obj, start=init_start, end=score.last_point.t)

    return score


@dataclass
class TupletInfo:
    pt_obj: pt.score.Tuplet
    notes: list[pt.score.GenericNote]


@dataclass
class ChordInfo:
    duration: Fraction
    notes: list[pt.score.Note]


@dataclass
class VoiceInfo:
    time: int = 0  # divisions
    tuplet: TupletInfo | None = None
    chord: ChordInfo | None = None
    grace_notes: list[pt.score.GraceNote] | None = None

    def __post_init__(self):
        if self.grace_notes is None:
            self.grace_notes = []


def token_sequence_to_partitura_permissive(tokens: list[Token]) -> pt.score.Part:
    """
    Same as above, but tries to output a score even in case of malformed token sequence.
    """
    # Get all possible divisions per measure (32 is default to allow for double-dotted 32nd notes)
    global_division = math.lcm(
        *[tuplet.n_actual for tuplet in tokens if isinstance(tuplet, tk.TupletStart)] + [32]
    )
    logger.trace(f"{global_division = }")

    def to_partitura_duration(d: Fraction | SymbolicDuration) -> int:
        """Converts a duration in quarter notes into a duration in divisions."""
        if isinstance(d, SymbolicDuration):
            d = d.quarter_length
        duration_in_div = d * global_division
        if duration_in_div.denominator != 1:
            breakpoint()
        assert (
            duration_in_div.denominator == 1
        ), f"Partitura duration is supposed to be an int, got {duration_in_div}."
        return duration_in_div.numerator

    score = pt.score.Part("P1", quarter_duration=global_division)
    measure_long_objets: set[pt.score.Rest | pt.score.Measure] = set()
    measure = None
    measure_number = 0
    # We store pending clefs, so that they are added when the next note of the staff is produced
    pending_clefs = {}  # staff -> clef

    time = 0
    voices_info: defaultdict[int, VoiceInfo] = defaultdict(lambda: VoiceInfo(time=time))
    tied_notes = {}

    for i, token in enumerate(tokens):
        # SOS
        if isinstance(token, StartOfScore):
            if i != 0:
                logger.warning(
                    "StartOfScore token produce in the middle of the sequence, ignoring it"
                )
            continue

        # EOS
        if isinstance(token, EndOfScore):
            time = max((v.time for v in voices_info.values()), default=0)
            for obj in measure_long_objets:
                # Remove and re-add objects with fixed end time
                init_start = obj.start.t
                score.remove(obj)
                score.add(obj, start=init_start, end=time)
            # Add the final barline
            score.add(pt.score.Barline("light-heavy"))
            # if i < len(tokens) - 1:
            #     logger.warning(
            #         "An EndOfScore token was produced before the last element of the sequence, "
            #         "the remaining tokens will be ignored."
            #     )
            break

        # Bar
        elif isinstance(token, Bar):
            measure_number += 1
            # Resync all voices in case of empty voices in previous bars
            time = max((v.time for v in voices_info.values()), default=0)
            if measure is not None and time == measure.start.t:  # In case of empty measure
                time += to_partitura_duration(token.nominal_duration)
            for v in voices_info:
                voices_info[v].time = time
            # print(f"Measure length: {token.actual_duration} ({duration})")

            # Set last measure and measure rests end time
            for obj in measure_long_objets:
                # Remove and re-add objects with fixed end time
                init_start = obj.start.t
                score.remove(obj)
                score.add(obj, start=init_start, end=time)
            measure_long_objets.clear()
            # Start new measure
            measure = pt.score.Measure(number=measure_number, name=measure_number)
            measure_long_objets.add(measure)
            ts = pt.score.TimeSignature(token.numerator, token.denominator)
            # print(time, duration)
            score.add(measure, start=time, end=None)  # End will be set at next bar
            prev_ts = next(score.last_point.iter_prev(cls=pt.score.TimeSignature), None)
            # Only add TS if different one than previous one
            if prev_ts is None or prev_ts.beats != ts.beats or prev_ts.beat_type != ts.beat_type:
                score.add(ts, start=time)

        # Key signature
        elif isinstance(token, KeySignature):
            time = max((v.time for v in voices_info.values()), default=0)
            ks = pt.score.KeySignature(token.accidentals, mode=None)
            score.add(ks, start=time)

        # Clef
        elif isinstance(token, Clef):
            time = max(
                (
                    note.end.t
                    for note in score.iter_all(cls=pt.score.GenericNote, include_subclasses=True)
                    if voice_absolute_into_voice_relative_to_staff(note.voice, one_based=True)[1]
                    == token.staff_idx_1b
                ),
                default=0,
            )
            clef = pt.score.Clef(
                staff=token.staff_idx_1b,
                sign=token.clef,
                line=token.line,
                octave_change=token.octave_change,
            )
            if clef.staff in pending_clefs:
                logger.warning("Another clef is already pending for this staff, replacing it")
            # In case of start of measure, put the clef right away
            if time == measure.start.t:
                score.add(clef, start=time)
            # Else, delay it until the next note in the staff or next measure
            else:
                pending_clefs[token.staff_idx] = clef

        # Note and rest
        elif isinstance(token, (Note, Rest)):
            voice_info = voices_info[token.voice_idx]
            duration = token.duration.quarter_length  # Measure rests' duration will be fixed later
            time = voice_info.time
            # print(duration)

            # Add clef if there is one pending
            # We take the default staff of the voice, not the actual staff, because clef changes
            # like measure 55 in Liszt/Concert_Etude_S145/1, that appear before the note in the
            # lower staff
            default_staff = voice_absolute_into_voice_relative_to_staff(token.voice_idx)[1]
            if default_staff in pending_clefs:
                score.add(pending_clefs.pop(default_staff), start=time)

            # Rest
            if isinstance(token, Rest):
                note = pt.score.Rest(voice=token.voice_idx_1b, staff=token.staff_idx_1b)
            # Note
            else:
                cls = (
                    pt.score.Note
                    if not token.grace
                    else functools.partial(pt.score.GraceNote, grace_type="acciaccatura")
                )
                note = cls(
                    step=token.note_class,
                    octave=token.octave,
                    alter=token.accidental,
                    voice=token.voice_idx_1b,
                    staff=token.staff_idx_1b,
                    stem_direction=token.stem_direction,
                    articulations=token.articulations,
                    ornaments=token.ornaments,
                )

            if token.fermata:
                note.fermata = pt.score.Fermata()
            if isinstance(token, Rest) and token.measure_rest:
                note.symbolic_duration = {}
                measure_long_objets.add(note)  # Fix duration at next bar
            elif isinstance(token, Rest) or (isinstance(token, Note) and not token.grace):
                note.symbolic_duration = {
                    "type": token.duration.type.value,
                    "dots": token.duration.dots,
                }
                # Update duration within tuplets
                if (tuplet := voice_info.tuplet) is not None:
                    tuplet.notes.append(note)
                    duration *= tuplet.pt_obj.duration_multiplier
                    note.symbolic_duration = {
                        "actual_notes": tuplet.pt_obj.actual_notes,
                        "normal_notes": tuplet.pt_obj.normal_notes,
                        **note.symbolic_duration,
                    }
                    # print(f"  > {duration = }")

            if isinstance(token, Note) and token.grace:
                note.symbolic_duration = {"type": "eighth"}
                duration = 0
                if voice_info.grace_notes:
                    last_grace = voice_info.grace_notes[-1]
                    last_grace.grace_next = note
                    note.grace_prev = last_grace
                voice_info.grace_notes.append(note)
            # When first non-grace note, set the main note and clear the list
            elif voice_info.grace_notes:
                voice_info.grace_notes[-1].grace_next = note
                voice_info.grace_notes = []

            duration = to_partitura_duration(duration)

            if isinstance(token, Note):
                # Resolve previous ties
                if token.note_tuple in tied_notes.keys():
                    prev_note = tied_notes[token.note_tuple]
                    note.tie_prev = prev_note
                    prev_note.tie_next = note
                    tied_notes.pop(token.note_tuple)
                # Deal with future ties
                if token.tied:
                    tied_notes[token.note_tuple] = note

            score.add(note, start=time, end=time + duration)
            if voice_info.chord is None and not (isinstance(token, Rest) and token.measure_rest):
                voice_info.time += duration

        # Chord start
        elif isinstance(token, ChordStart):
            voice_info = voices_info[token.voice_idx]
            voice_info.chord = ChordInfo(duration=token.duration.quarter_length, notes=[])

        # Chord end
        elif isinstance(token, ChordEnd):
            voice_info = voices_info[token.voice_idx]
            if voice_info.chord is None:
                logger.warning(
                    f"ChordEnd produced on voice {token.voice_idx} without ChordStart, skipping"
                )
                continue
            duration = voice_info.chord.duration
            if voice_info.tuplet is not None:
                duration *= voice_info.tuplet.pt_obj.duration_multiplier
            duration = to_partitura_duration(duration)
            voices_info[token.voice_idx].time += duration
            voice_info.chord = None

        # Tuplet start
        elif isinstance(token, TupletStart):
            voices_info[token.voice_idx].tuplet = TupletInfo(
                pt_obj=pt.score.Tuplet(
                    actual_notes=token.n_actual,
                    normal_notes=token.n_normal,
                    actual_type=token.unit_duration.type.value,
                    actual_dots=token.unit_duration.dots,
                    normal_type=token.unit_duration.type.value,
                    normal_dots=token.unit_duration.dots,
                ),
                notes=[],
            )

        # Tuplet end
        elif isinstance(token, TupletEnd):
            voice_info = voices_info[token.voice_idx]
            if voice_info.tuplet is None:
                logger.warning(
                    f"TupletEnd produced on voice {token.voice_idx} without TupletStart, skipping"
                )
                continue
            tuplet = voice_info.tuplet
            if len(tuplet.notes) == 0:
                logger.warning(
                    f"TupletEnd produced on voice {token.voice_idx} with no note, skipping"
                )
                voice_info.tuplet = None
                continue
            # Get the highest note starting at the beginning of the tuplet
            # Only needed when a chord is the first event of the tuplet, so that the <tuplet>
            # tag is associated to the correct note at export time
            first_note = max(
                filter(lambda n: n.start == tuplet.notes[0].start, tuplet.notes),
                key=lambda x: (x.midi_pitch, x.step) if isinstance(x, pt.score.Note) else -1,
            )
            # Same if the last event is a chord
            last_note = max(
                filter(lambda n: n.end == tuplet.notes[-1].end, tuplet.notes),
                key=lambda x: (x.midi_pitch, x.step) if isinstance(x, pt.score.Note) else -1,
            )
            tuplet.pt_obj.start_note = first_note
            tuplet.pt_obj.end_note = last_note
            # print(tuplet.pt_obj.start_note, tuplet.pt_obj.end_note)
            # print(f"{tuplet.pt_obj.start_note == tuplet.pt_obj.end_note = }")
            # print(f"{tuplet.pt_obj.start_note is tuplet.pt_obj.end_note = }")
            score.add(tuplet.pt_obj, start=first_note.start.t, end=last_note.end.t)
            voice_info.tuplet = None

        # Error
        else:
            raise ValueError(f"Unknown token {token}.")

    # Eventually set the end of non-ending objects to the final time
    for obj in measure_long_objets:
        init_start = obj.start.t
        score.remove(obj)
        score.add(obj, start=init_start, end=score.last_point.t)

    return score


def partitura_to_token_sequence(
    score: pt.score.Part,
) -> list[Token]:
    """Convert a Partitura part to a sequence of tokens.

    :param score: The Partitura part to convert.
    :return: A list of tokens (not necessarily valid, will need to be checked by an automata).
    """
    tokens = [StartOfScore()]
    time_signature = None
    pending_tuplets: list[pt.score.Tuplet] = []

    for point in score._points:
        point: pt.score.TimePoint
        # print(f"{point.t = }")

        # Start by looking for eventual TS change
        if point.starting_objects[pt.score.TimeSignature]:
            time_signature = list(point.starting_objects[pt.score.TimeSignature].keys())[0]

        # Finish pending tuplets if needed
        for tuplet in reversed(pending_tuplets):
            if tuplet.end_note.end.t <= point.t:
                token = TupletEnd(
                    voice_idx=tuplet.end_note.voice - 1,  # 1-indexed
                )
                token._partitura_elements = [tuplet]
                tokens.append(token)
                pending_tuplets.remove(tuplet)

        # Look over different objet types, ORDER MATTERS

        # Look for TimeSignature
        time_signatures = list(point.starting_objects[pt.score.TimeSignature].keys())
        assert (
            len(time_signatures) <= 1
        ), "Multiple time signatures at the same time are not supported."
        if len(time_signatures) == 1:
            time_signature = time_signatures[0]

        # Look for Measure
        measures = list(point.starting_objects[pt.score.Measure].keys())
        if len(measures) > 1:
            raise ValueError(f"Simultaneous bars are not supported, got {len(measures)} measures.")
        elif len(measures) == 1:
            measure = measures[0]
            logger.trace(measure)
            measure_token = Bar(
                numerator=time_signature.beats,
                denominator=time_signature.beat_type,
            )
            real_duration_in_divs = measure.end.t - measure.start.t
            real_duration_in_quarters = Fraction(real_duration_in_divs, point.quarter)
            if (
                measure_token.nominal_duration != real_duration_in_quarters
                and real_duration_in_quarters != 0  # Empty measures duration is 0 bc of full rests
            ):
                measure_token.is_nominal = False
            measure_token._partitura_elements = [measure, time_signature]
            tokens.append(measure_token)

        # Look for Clef
        clefs = sorted(list(point.starting_objects[pt.score.Clef].keys()), key=lambda c: c.staff)
        for clef in clefs:
            tokens.append(
                Clef(
                    staff_idx=clef.staff - 1,  # 1-indexed
                    clef=clef.sign,
                )
            )

        # Look for KeySignature
        key_signatures = list(point.starting_objects[pt.score.KeySignature].keys())
        assert (
            len(key_signatures) <= 1
        ), "Multiple key signatures at the same time are not supported."
        if len(key_signatures) == 1:
            ks = key_signatures[0]
            token = KeySignature(accidentals=ks.fifths)
            token._partitura_elements = [ks]
            tokens.append(token)

        # Look for Tuplet
        tuplets = sorted(
            point.starting_objects[pt.score.Tuplet].keys(),
            key=lambda t: t.start_note.voice,
        )
        for tuplet in tuplets:
            # print(tuplet)
            # Weird case where we encounter things like 6:6 tuplets (which shouldn't be a tuplet)
            if (
                tuplet.actual_type == tuplet.normal_type
                and tuplet.actual_dots == tuplet.normal_dots
                and tuplet.actual_notes == tuplet.normal_notes
            ):
                continue
            # Deal with the rare case of different tuplet types
            if tuplet.actual_type != tuplet.normal_type or tuplet.actual_dots != tuplet.normal_dots:
                durations_ratio = tuplet.duration_multiplier
                normal_notes = tuplet.actual_notes * durations_ratio
                assert (
                    normal_notes.denominator == 1
                ), f"Tuplet type conversion not possible: {tuplet.actual_notes}-{tuplet.actual_type} against {tuplet.normal_notes}-{tuplet.normal_type}."
                normal_notes = int(normal_notes)
            else:
                normal_notes = tuplet.normal_notes
            token = TupletStart(
                voice_idx=tuplet.start_note.voice - 1,  # 1-indexed
                n_actual=tuplet.actual_notes,
                n_normal=normal_notes,
                unit_duration=SymbolicDuration(
                    type=SymbolicDurationType(tuplet.actual_type),
                    dots=tuplet.actual_dots,
                ),
            )
            token._partitura_elements = [tuplet]
            tokens.append(token)
            pending_tuplets.append(tuplet)

        # Look for notes and grace note
        # We sort grace notes first, without further sorting as it would break
        # the correct order (partitura does not store attributes about grace
        # notes order), then we sort normal notes, with a deterministic order
        # voice id > ascending pitch > step (to distinguish enharmonic spellings)
        notes = sorted(
            itertools.chain(
                point.starting_objects[pt.score.GraceNote].keys(),
                point.starting_objects[pt.score.Note].keys(),
            ),
            key=lambda n: (
                (0,) if isinstance(n, pt.score.GraceNote) else (1, n.voice, n.midi_pitch, n.step)
            ),
        )
        # Get notes per voice
        notes_per_voice = defaultdict(lambda: {"grace": [], "note": []})
        for note in notes:
            type_ = "grace" if isinstance(note, pt.score.GraceNote) else "note"
            notes_per_voice[note.voice][type_].append(note)
        for voice, vnotes in notes_per_voice.items():
            gnotes, nnotes = vnotes.values()
            # Add grace notes
            for gnote in gnotes:
                token = Note(
                    note_class=gnote.step,
                    octave=gnote.octave,
                    accidental=gnote.alter or 0,
                    duration=SymbolicDuration(type=SymbolicDurationType.EIGHTH),
                    tied=False,
                    voice_idx=gnote.voice - 1,  # 1-indexed
                    staff_idx=gnote.staff - 1,  # 1-indexed
                    stem_direction=gnote.stem_direction,
                    fermata=False,
                    grace=True,
                )
                token._partitura_elements = [gnote]
                tokens.append(token)
            if len(nnotes) > 1:
                token = ChordStart(
                    voice_idx=voice - 1,
                    duration=SymbolicDuration(
                        type=SymbolicDurationType(nnotes[0].symbolic_duration["type"]),
                        dots=(
                            nnotes[0].symbolic_duration["dots"]
                            if "dots" in nnotes[0].symbolic_duration
                            else 0
                        ),
                    ),
                )
                token._partitura_elements = nnotes
                tokens.append(token)
            for note in nnotes:
                token = Note(
                    note_class=note.step,
                    octave=note.octave,
                    accidental=note.alter or 0,
                    duration=SymbolicDuration(
                        type=SymbolicDurationType(note.symbolic_duration["type"]),
                        dots=(
                            note.symbolic_duration["dots"]
                            if "dots" in note.symbolic_duration
                            else 0
                        ),
                    ),
                    tied=note.tie_next is not None,
                    # tied_prev=note.tie_prev is not None,  # TODO: remove at some point?
                    voice_idx=note.voice - 1,  # 1-indexed
                    staff_idx=note.staff - 1,  # 1-indexed
                    stem_direction=note.stem_direction,
                    fermata=note.fermata is not None,
                )
                # print(f"{note.articulations = }")
                # print(f"{note.ornaments = }")
                for attr in note.articulations + note.ornaments:
                    if hasattr(token, attr_ := attr.replace("-", "_")):
                        setattr(token, attr_, True)
                token._partitura_elements = [note]
                tokens.append(token)
            if len(nnotes) > 1:
                token = ChordEnd(voice_idx=voice - 1)  # 1-indexed
                token._partitura_elements = nnotes
                tokens.append(token)
            # TODO: tied notes

        # Look for Rest
        rests = sorted(
            point.starting_objects[pt.score.Rest].keys(),
            key=operator.attrgetter("voice"),
        )
        for rest in rests:
            is_measure = "type" not in rest.symbolic_duration
            rest_type = rest.symbolic_duration.get("type", "whole")  # default whole for bar-rests
            token = Rest(
                measure_rest=is_measure,
                duration=SymbolicDuration(
                    type=SymbolicDurationType(rest_type),
                    dots=(
                        rest.symbolic_duration["dots"] if "dots" in rest.symbolic_duration else 0
                    ),
                ),
                voice_idx=rest.voice - 1,  # 1-indexed
                staff_idx=rest.staff - 1,  # 1-indexed
                fermata=rest.fermata is not None,
            )
            token._partitura_elements = [rest]
            tokens.append(token)

    tokens.append(EndOfScore())

    return tokens


if __name__ == '__main__':
    from pathlib import Path
    from models.automata.tokens import fill_rests
    import warnings

    path = Path(
        "~/Documents/datasets/asap-dataset/Bach/Fugue/bwv_860/xml_score.musicxml"
        # "~/Documents/datasets/asap-dataset/Bach/Prelude/bwv_883/xml_score.musicxml"
    ).expanduser()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pt.load_musicxml(path, force_note_ids=True).parts[0]
        score = pt.score.unfold_part_maximal(score)
    fill_rests(score)
    score = reorder_voices(score)

    tokens = partitura_to_token_sequence(score)

    detokenized_score = token_sequence_to_partitura(tokens)
    # print(detokenized_score.pretty())
    # pt.save_musicxml(detokenized_score, "/tmp/regenerated_score.musicxml")

    retokenized_tokens = partitura_to_token_sequence(detokenized_score)

    # assert all(
    #     token == retokenized_token for token, retokenized_token in zip(tokens, retokenized_tokens)
    # ), "Tokens do not match after detokenization."

    for token, retokenized_token in zip(tokens, retokenized_tokens):
        if token != retokenized_token:
            print(f"* {token}")
            print(f"  {retokenized_token}")
            # print(token._partitura_elements)
            # print(retokenized_token._partitura_elements)
