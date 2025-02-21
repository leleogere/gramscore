import functools
import itertools
from collections import defaultdict
from fractions import Fraction
from pprint import pprint
import math

import partitura as pt
import partitura.score

from src.models.automata.tokens import HasVoiceMixin
from src.models.automata.tokens import SymbolicDurationType
from src.models.automata.tokens import SymbolicDuration
from src.models.automata.tokens import Token
from src.models.automata.tokens import TranscriptionState
from src.models.automata.tokens import (
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
from src.utils.ansi_colors import Colors


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

    print(len(tokens), len(states))
    if len(tokens) != len(states):
        raise ValueError("The number of tokens and states must be the same.")

    # Get all possible divisions per measure (32 is default to allow for double-dotted 32nd notes)
    all_divisions = {32}
    for token, state in zip(tokens, states):
        if isinstance(token, TupletStart):
            all_divisions.add(token.n_actual)
    global_division = math.lcm(*all_divisions)
    print(global_division)

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

    voices_info = [
        {
            "time": 0,
            "tied_notes": {},
            "tuplet": None,  # {"pt_obj": pt.score.Tuplet(), "notes": []}
        }
        for _ in range(states[0].num_voices)
    ]
    for i, (token, state) in enumerate(zip(tokens, states)):
        # print(state.n, time, token, state)
        print(
            # f"{Colors.RED}{state.}{Colors.END} "
            f"{Colors.GREEN}{[float(v['time'] / global_division) for v in voices_info]}{Colors.END} "
            f"{Colors.BLUE}{token}{Colors.END} "
            f"{Colors.RED}{state}{Colors.END}"
        )

        # SOS and EOS
        if isinstance(token, (StartOfScore, EndOfScore)):
            pass

        # Bar
        elif isinstance(token, Bar):
            # Resync all voices in case of empty voices in previous bars
            time = max(v["time"] for v in voices_info)
            for v in voices_info:
                v["time"] = time

            duration = to_partitura_duration(token.actual_duration)
            print(f"Measure length: {token.actual_duration} ({duration})")
            measure = pt.score.Measure(number=state.measure_number, name=state.measure_number)
            ts = pt.score.TimeSignature(token.numerator, token.denominator)
            print(time, duration)
            score.add(measure, start=time, end=time + duration)
            prev_ts = next(score.last_point.iter_prev(cls=pt.score.TimeSignature), None)
            # Do not repeat TS if already same one
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
                    if note.staff == token.staff_idx_1b
                ),
                default=0,
            )
            print(f"clef={token.clef}, staff={token.staff_idx}, {time=}")
            clef = pt.score.Clef(
                staff=token.staff_idx_1b,
                sign=token.clef,
                line=token.line,
                octave_change=token.octave_change,
            )
            score.add(clef, start=time)

        # Note and rest
        elif isinstance(token, (Note, Rest)):
            voice_info = voices_info[token.voice_idx]
            if isinstance(token, Rest) and token.measure_rest:
                duration = measure.duration // global_division
            else:
                duration = token.duration.quarter_length
            time = voice_info["time"]
            print(duration)

            note = (
                pt.score.Note(
                    step=token.note_class,
                    octave=token.octave,
                    alter=token.accidental,
                    voice=token.voice_idx_1b,
                    staff=token.staff_idx_1b,
                    stem_direction=token.stem_direction,
                )
                if isinstance(token, Note)
                else pt.score.Rest(voice=token.voice_idx + 1, staff=token.staff_idx_1b)
            )
            if isinstance(token, Rest) and token.measure_rest:
                note.symbolic_duration = {}
            else:
                note.symbolic_duration = {
                    "type": token.duration.type.value,
                    "dots": token.duration.dots,
                }
                # Update duration within tuplets
                voice_state = state.voices[token.voice_idx]
                if state.voices[token.voice_idx].in_tuplet:
                    voice_info["tuplet"]["notes"].append(note)
                    duration *= voice_state.tuplet.duration_multiplier
                    note.symbolic_duration = {
                        "actual_notes": voice_state.tuplet.n_actual,
                        "normal_notes": voice_state.tuplet.n_normal,
                        **note.symbolic_duration,
                    }
                    print(f"  > {duration = }")

            duration = to_partitura_duration(duration)

            if isinstance(token, Note):
                # Resolve previous ties
                if token.note_tuple in voice_info["tied_notes"].keys():
                    prev_note = voice_info["tied_notes"][token.note_tuple]
                    note.tie_prev = prev_note
                    prev_note.tie_next = note
                    voice_info["tied_notes"].pop(token.note_tuple)
                # Deal with future ties
                if token.tied:
                    voice_info["tied_notes"][token.note_tuple] = note

            score.add(note, start=time, end=time + duration)
            if not voice_state.in_chord:
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
            voices_info[token.voice_idx]["tuplet"] = {"pt_obj": pt.score.Tuplet(), "notes": []}

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
            print(tuplet["pt_obj"].start_note, tuplet["pt_obj"].end_note)
            print(f"{tuplet["pt_obj"].start_note == tuplet["pt_obj"].end_note = }")
            print(f"{tuplet["pt_obj"].start_note is tuplet["pt_obj"].end_note = }")
            score.add(tuplet["pt_obj"], start=first_note.start.t, end=last_note.end.t)
            voice_info["tuplet"] = None

        # Error
        else:
            raise ValueError(f"Unknown token {token}.")

    # # Add measure-long silences if a staff is empty on a measure
    # for measure in score.iter_all(cls=pt.score.Measure):
    #     for staff in (1, 2):
    #         for note in score.iter_all(
    #             cls=pt.score.GenericNote,
    #             start=measure.start.t,
    #             end=measure.end.t,
    #             include_subclasses=True,
    #         ):
    #             if note.staff == staff:
    #                 break
    #         else:  # If the loop did not break, the staff is empty
    #             rest = pt.score.Rest(voice=(staff - 1) * 4 + 1, staff=staff)
    #             rest.symbolic_duration = {}  # Manually remove the duration
    #             score.add(rest, start=measure.start.t, end=measure.end.t)

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
        assert len(measures) <= 1, "Multiple bar at the same time are not supported."
        if len(measures) == 1:
            measure = measures[0]
            print(measure)
            measure_token = Bar(
                numerator=time_signature.beats,
                denominator=time_signature.beat_type,
                name=measure.name,
            )
            real_duration_in_divs = measure.end.t - measure.start.t
            real_duration_in_quarters = Fraction(real_duration_in_divs, point.quarter)
            if measure_token.normal_duration != real_duration_in_quarters:
                # print(
                #     f"Measure ({measure_token.normal_duration}) does not match actual duration "
                #     f"({real_duration_in_quarters}), marking as incomplete measure."
                # )
                measure_token.actual_duration = real_duration_in_quarters
            measure_token._partitura_elements = [measure, time_signature]
            tokens.append(measure_token)

        # Look for Clef
        clefs = list(point.starting_objects[pt.score.Clef].keys())
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
        tuplets = list(point.starting_objects[pt.score.Tuplet].keys())
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
            print(tuplet)
            if tuplet.actual_type != tuplet.normal_type or tuplet.actual_dots != tuplet.normal_dots:
                durations_ratio = tuplet.duration_multiplier
                normal_notes = tuplet.actual_notes * durations_ratio
                assert (
                    normal_notes.is_integer()
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
                    # dots=tuplet.nodots,
                ),
            )
            token._partitura_elements = [tuplet]
            tokens.append(token)
            pending_tuplets.append(tuplet)

        # Look for Note
        notes = list(point.starting_objects[pt.score.Note].keys())
        # Get notes per voice
        notes_per_voice = defaultdict(list)
        for note in notes:
            notes_per_voice[note.voice].append(note)
        for voice, vnotes in notes_per_voice.items():
            # All notes in the voice should have the same duration
            assert len(set(n.duration for n in vnotes)) == 1
            sym_dur = SymbolicDuration(
                type=SymbolicDurationType(vnotes[0].symbolic_duration["type"]),
                dots=(
                    vnotes[0].symbolic_duration["dots"]
                    if "dots" in vnotes[0].symbolic_duration
                    else 0
                ),
            )
            if len(vnotes) > 1:
                token = ChordStart(voice_idx=voice - 1, duration=sym_dur)  # 1-indexed
                token._partitura_elements = vnotes
                tokens.append(token)
            for note in vnotes:
                token = Note(
                    note_class=note.step,
                    octave=note.octave,
                    accidental=note.alter or 0,
                    duration=sym_dur,
                    tied=note.tie_next is not None,
                    voice_idx=note.voice - 1,  # 1-indexed
                    staff_idx=note.staff - 1,  # 1-indexed
                    stem_direction=note.stem_direction,
                )
                token._partitura_elements = [note]
                tokens.append(token)
            if len(vnotes) > 1:
                token = ChordEnd(voice_idx=voice - 1)  # 1-indexed
                token._partitura_elements = vnotes
                tokens.append(token)
            # TODO: tied notes

        # Look for Rest
        rests = list(point.starting_objects[pt.score.Rest].keys())
        for rest in rests:
            # Measure rest
            if "type" not in rest.symbolic_duration:
                token = Rest(
                    measure_rest=True,
                    duration=SymbolicDuration(type=SymbolicDurationType("whole")),
                    voice_idx=rest.voice - 1,  # 1-indexed
                    staff_idx=rest.staff - 1,  # 1-indexed
                )
            # Regular rest
            else:
                token = Rest(
                    duration=SymbolicDuration(
                        type=SymbolicDurationType(rest.symbolic_duration["type"]),
                        dots=(
                            rest.symbolic_duration["dots"]
                            if "dots" in rest.symbolic_duration
                            else 0
                        ),
                    ),
                    voice_idx=rest.voice - 1,  # 1-indexed
                    staff_idx=rest.staff - 1,  # 1-indexed
                )
            token._partitura_elements = [rest]
            tokens.append(token)

    return tokens
