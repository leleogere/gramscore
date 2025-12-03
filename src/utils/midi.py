import copy
import math
from typing import Literal

import numpy as np
import pretty_midi as pm
import symusic
from partitura.utils.music import midi_pitch_to_pitch_spelling

ScoreType = symusic.core.ScoreTick | symusic.core.ScoreSecond | symusic.core.ScoreQuarter


def merge_into_single_track(midi: ScoreType, inplace: bool = True) -> ScoreType:
    if not inplace:
        midi = midi.copy(deep=True)
    track: symusic.core.TrackTick = midi.tracks[0]
    if len(midi.tracks) > 1:
        for other in midi.tracks[1:]:
            for attribute in ["notes", "controls", "lyrics", "pedals", "pitch_bends"]:
                getattr(track, attribute).extend(getattr(other, attribute))
        midi.tracks = midi.tracks[:1]
        track.sort(inplace=True)
    return midi


def merge_pretty_midi_into_single_track(midi: pm.PrettyMIDI, inplace: bool = True) -> pm.PrettyMIDI:
    if not inplace:
        midi = copy.deepcopy(midi)
    if len(midi.instruments) > 1:
        track: pm.Instrument = midi.instruments[0]  # Only reference so OK modifying it
        merged_attributes = {
            "notes": (track.notes, lambda n: n.start),
            "control_changes": (track.control_changes, lambda cc: cc.time),
            "pitch_bends": (track.pitch_bends, lambda pb: pb.time),
        }
        for attr_name, (attr_values, sort_fn) in merged_attributes.items():
            for other_track in midi.instruments[1:]:
                attr_values.extend(getattr(other_track, attr_name))
            # Sort values
            setattr(track, attr_name, sorted(attr_values, key=sort_fn))
        midi.instruments = midi.instruments[:1]
    return midi


def midi_pitch_to_note_name(pitch: int) -> str:
    """
    Convert MIDI pitch to note name.
    """
    step, alteration, octave = midi_pitch_to_pitch_spelling(pitch)
    pitch_name = step + alteration * "♯" + -alteration * "♭" + str(octave)
    return pitch_name


def crop_pretty_midi(
    midi: pm.PrettyMIDI,
    start: float = None,
    end: float = None,
    start_mode: Literal["drop", "clip"] = "drop",
    end_mode: Literal["keep", "drop", "clip"] = "clip",
    inplace: bool = False,
) -> pm.PrettyMIDI:
    """
    Crop a PrettyMIDI object between start and end time (in seconds).
    If inplace is True, modify the original object.
    """
    if not inplace:
        midi = copy.deepcopy(midi)

    if start is None:
        start = 0
    if end is None:
        end = math.inf

    # Track the last event for sanity checks
    midi_end = 0.0

    # Loop over instruments
    new_instruments = []
    for instrument in midi.instruments:

        # Loop over notes, only keeping those within the time range (and cropping them if necessary)
        new_notes = []
        for note in instrument.notes:
            if note.end <= start or note.start >= end:
                continue

            # Check note at the beginning of the range
            if start_mode == "drop":
                if note.start < start:
                    continue
                else:
                    new_start = note.start - start
            elif start_mode == "clip":
                new_start = max(note.start, start) - start
            else:
                raise ValueError(f"Unknown mode '{start_mode}' for start_mode")

            # Check notes at the end of the range
            if end_mode == "drop":
                if note.end > end:
                    continue
                else:
                    new_end = note.end - start
            elif end_mode == "keep":
                new_end = note.end - start
            elif end_mode == "clip":
                new_end = min(note.end, end) - start
            else:
                raise ValueError(f"Unknown mode '{end_mode}' for end_mode")

            # Create the new note
            new_note = pm.Note(
                velocity=note.velocity, pitch=note.pitch, start=new_start, end=new_end
            )
            new_notes.append(new_note)
            if new_end > midi_end:
                midi_end = new_end
        instrument.notes = new_notes

        # Loop over other instrument events, only keeping those within the time range
        for attribute in ["control_changes", "pitch_bends"]:
            new_events = []
            for event in getattr(instrument, attribute):
                if event.time < start or event.time > end:
                    continue
                new_event = copy.deepcopy(event)
                new_event.time = event.time - start if start is not None else event.time
                new_events.append(new_event)
                if new_event.time > midi_end:
                    midi_end = new_event.time
            setattr(instrument, attribute, new_events)

        new_instruments.append(instrument)
    midi.instruments = new_instruments

    # Loop over other events
    for attribute, move_to_start in [
        ("key_signature_changes", True),
        ("time_signature_changes", True),
        ("lyrics", False),
        ("text_events", False),
    ]:
        new_events = []
        for event in getattr(midi, attribute):
            if event.time < start or event.time > end:
                continue
            new_event = copy.deepcopy(event)
            if move_to_start:
                new_event.time = event.time - start if start is not None else event.time
            new_events.append(new_event)
            if new_event.time > midi_end:
                midi_end = new_event.time
        setattr(midi, attribute, new_events)

    # print(midi._tick_scales)

    # Loop over tempo changes
    # Not working currently, it should not be very important, as long as we do not need metrical
    # information after cropping

    # Sanity check
    computed_midi_end = compute_midi_end_without_tempo_changes(midi)
    assert midi_end == computed_midi_end, (
        f"Sanity check failed: cropped MIDI end time does not match: "
        f"{midi_end} != {computed_midi_end}"
    )
    return midi


def compute_midi_end_without_tempo_changes(midi: pm.PrettyMIDI) -> float:
    return max(
        [i.get_end_time() for i in midi.instruments]
        + [
            e.time
            for m in [
                midi.time_signature_changes,
                midi.key_signature_changes,
                midi.lyrics,
                midi.text_events,
            ]
            for e in m
        ],
        default=0.0,
    )
