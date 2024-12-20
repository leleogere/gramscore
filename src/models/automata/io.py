import functools
import itertools
from fractions import Fraction
from pprint import pprint
import math

import partitura as pt
import partitura.score

from src.models.automata.tokens import SymbolicDuration
from src.models.automata.tokens import Token
from src.models.automata.tokens import TranscriptionState
from src.models.automata.tokens import (
    StartOfScore,
    EndOfScore,
    Bar,
    KeySignature,
    Rest,
    Note,
    ChordStart,
    ChordEnd,
    TupletStart,
    TupletEnd,
)
from utils.ansi_colors import Colors


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
        num_voices = max(token.voice_idx + 1 for token in tokens if hasattr(token, "voice_idx"))
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
        duration = 0

        # SOS and EOS
        if isinstance(token, (StartOfScore, EndOfScore)):
            pass

        # Bar
        elif isinstance(token, Bar):
            # Resync all voices in case of empty voices in previous bars
            time = max(v["time"] for v in voices_info)
            for v in voices_info:
                v["time"] = time

            duration = to_partitura_duration(token.duration)
            print(f"Measure length: {token.duration} ({duration})")
            measure = pt.score.Measure()
            ts = pt.score.TimeSignature(token.real_numerator, token.denominator)
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

        # Note and rest
        elif isinstance(token, (Note, Rest)):
            voice_info = voices_info[token.voice_idx]
            duration = token.duration.quarter_length
            time = voice_info["time"]
            print(duration)

            note = (
                pt.score.Note(
                    step=token.note_class,
                    octave=token.octave,
                    alter=token.accidental,
                    voice=token.voice_idx + 1,
                    staff=token.staff,
                )
                if isinstance(token, Note)
                else pt.score.Rest(voice=token.voice_idx + 1)
            )
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

    # print(score.pretty())
    # print("#" * 80)

    return score
