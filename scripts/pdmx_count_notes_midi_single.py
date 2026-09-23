"""
Count number of notes in a MIDI file.
"""

from pathlib import Path

import symusic


path = Path(
    # "/home/gerel/Documents/datasets/pdmx/mid/9/5/QmR6ZvdCfBXecGxiyNTAyvCXuLDsEoAR4QFzcysX66vE2M.mid"
    "/home/gerel/Documents/datasets/pdmx/mid/0/0/Qma1a2YHEJ8j9gxLCxgALAfjYaiYJQogh1bgyFSwNSRPCs.mid"
)


def count_notes(path: Path) -> int:
    return sum(len(track.notes) for track in symusic.Score(path).tracks)


print(f"Number of notes: {count_notes(path)}")
