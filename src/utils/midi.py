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


def midi_pitch_to_note_name(pitch: int) -> str:
    """
    Convert MIDI pitch to note name.
    """
    step, alteration, octave = midi_pitch_to_pitch_spelling(pitch)
    pitch_name = step + alteration * "♯" + -alteration * "♭" + str(octave)
    return pitch_name
