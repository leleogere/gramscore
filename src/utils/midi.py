import symusic


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
