import functools
import itertools
from fractions import Fraction
from pprint import pprint
import math

import partitura as pt
import partitura.score

from src.models.automata.tokens import Duration
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
        from models.automata.automata import ConstrainedAutomata

        automata = ConstrainedAutomata()
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

    def to_partitura_duration(d: Fraction | Duration) -> int:
        """Converts a duration in quarter notes into a duration."""
        if isinstance(d, Duration):
            d = d.quarter_length
        duration_in_div = d * global_division
        assert (
            duration_in_div.denominator == 1
        ), f"Partitura duration is supposed to be an int, got {duration_in_div}."
        return duration_in_div.numerator

    score = pt.score.Part("P1", quarter_duration=global_division)
    time = 0
    tied_notes = {}
    tuplets = []
    # tuplet = [
    #    {
    #       n_actual: x,
    #       n_real: y,
    #       notes: [note1, note2, ...]
    #    },...
    # ]
    for i, (token, state) in enumerate(zip(tokens, states)):
        # print(state.n, time, token, state)
        print(
            f"{Colors.RED}{state.n}{Colors.END} "
            f"{Colors.GREEN}{time}{Colors.END} "
            f"{Colors.BLUE}{token}{Colors.END} "
            f"{Colors.BLACK}{state}{Colors.END}"
        )
        pprint(tuplets)
        duration = 0

        # SOS and EOS
        if isinstance(token, (StartOfScore, EndOfScore)):
            pass

        # Bar
        elif isinstance(token, Bar):
            duration = to_partitura_duration(token.quarter_length)
            print(f"Measure length: {token.quarter_length} ({duration})")
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
            ks = pt.score.KeySignature(token.accidentals, mode=None)
            score.add(ks, start=time)

        # Note and rest
        elif isinstance(token, (Note, Rest)):
            duration = token.duration.quarter_length
            print(duration)

            note = (
                pt.score.Note(step=token.note_class, octave=token.octave, alter=token.accidental)
                if isinstance(token, Note)
                else pt.score.Rest()
            )
            note.symbolic_duration = {
                "type": token.duration.type.value,
                "dots": token.duration.dots,
            }
            # Update duration within tuplets
            if len(tuplets) > 0:
                ratio = 1
                for tuplet in tuplets:
                    ratio *= Fraction(tuplet["token"].n_real, tuplet["token"].n_actual)
                    print(f"  {ratio = }")
                    tuplet["notes"].append(note)
                duration *= ratio
                # TODO: change this to more modular approach
                note.symbolic_duration = {
                    "actual_notes": state.tuplet.n_actual,
                    "normal_notes": state.tuplet.n_real,
                    **note.symbolic_duration,
                }
                print(f"  > {duration = }")
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

        # Chord start
        elif isinstance(token, ChordStart):
            pass

        # Chord end
        elif isinstance(token, ChordEnd):
            duration = states[i - 1].chord.duration.quarter_length
            if state.tuplet.active:
                duration *= state.tuplet.duration_multiplier
            duration = to_partitura_duration(duration)

        # Tuplet start
        elif isinstance(token, TupletStart):
            tuplets.insert(
                0,
                {
                    "token": token,
                    "notes": [],
                    "pt_obj": pt.score.Tuplet(),
                },
            )

        # Tuplet end
        elif isinstance(token, TupletEnd):
            tuplet = tuplets.pop(0)
            # Get the highest note starting at the begining of the tuplet
            # Only needed when a chord is the first event of the tuplet, so that the <tuplet>
            # tag is associated to the correct note at export time
            first_note = max(
                filter(lambda n: n.start == tuplet["notes"][0].start, tuplet["notes"]),
                key=lambda x: x.midi_pitch if isinstance(x, pt.score.Note) else -1,
            )
            # Same is the last event is a chord
            last_note = max(
                filter(lambda n: n.end == tuplet["notes"][-1].end, tuplet["notes"]),
                key=lambda x: x.midi_pitch if isinstance(x, pt.score.Note) else -1,
            )
            tuplet["pt_obj"].start_note = first_note
            tuplet["pt_obj"].end_note = last_note
            print(tuplet["pt_obj"].start_note, tuplet["pt_obj"].end_note)
            score.add(tuplet["pt_obj"], start=first_note.start.t, end=last_note.end.t)

        # Error
        else:
            raise ValueError(f"Unknown token {token}.")

        # If change in n, mean that new rhythmic event has happened, so update time
        # TODO: how to deal with multivoice in this?
        if i < len(tokens) - 1 and state.n < states[i + 1].n:
            print(f"  + {duration}")
            time += duration

    # print(score.pretty())
    # print("#" * 80)

    return score
