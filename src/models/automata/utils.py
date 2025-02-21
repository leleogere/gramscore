import itertools
from collections import defaultdict

import partitura as pt
import partitura.score


def voice_absolute_into_voice_relative_to_staff(
    voice_idx: int,
    one_based: bool = False,
) -> tuple[int, int]:
    """
    Converts an absolute voice_index into a voice index relative to a staff and its associated
    staff. Following MusicXML/MuseScore standards, the four first voices are mapped to the first
    staff, the four next voices to the second staff, and so on.

    :param voice_idx: Voice index (0-based or 1-based)
    :param one_based: Whether the voice and staff indices are 1-based (True) or 0-based (False)

    :return: Relative voice index and associated staff index (0 or 1-based depending on the
        parameter `one_based`)
    """
    if one_based:
        voice_idx -= 1
    staff_idx = voice_idx // 4
    voice_in_staff_idx = voice_idx % 4
    if one_based:
        staff_idx += 1
        voice_in_staff_idx += 1
    return voice_in_staff_idx, staff_idx


def voice_relative_to_staff_into_voice_absolute(
    voice_in_staff_idx: int,
    staff_idx: int,
    one_based: bool = False,
) -> int:
    """
    Converts a voice index relative into a staff to an absolute voice index. Following MusicXML/
    MuseScore standards, the four first voices are mapped to the first staff, the four next voices
    to the second staff, and so on.

    :param voice_in_staff_idx: Voice index in the staff (0-based or 1-based)
    :param staff_idx: Staff index (0 or 1-based depending on the parameter `one_based`)
    :param one_based: Whether the voice and staff indices are 1-based (True) or 0-based (False)

    :return: Voice index (0 or 1-based depending on the parameter `one_based`)
    """
    if one_based:
        staff_idx -= 1
        voice_in_staff_idx -= 1
    voice_idx = voice_in_staff_idx + staff_idx * 4
    return voice_idx + 1 if one_based else voice_idx


# def reorder_voices(part: pt.score.Part) -> pt.score.Part:
#     """
#     Reorder voices so that in each staff, the "first" (% 4) voice is always the first one starting.
#     In details, the order is the following:
#         - First elements come first
#         - When simultaneous events:
#             - Notes come before rests
#             - Notes are sorted by ascending MIDI pitch
#             - Enharmonic notes are sorted by ascending step
#
#     :param part: Partitura Part object
#     :return: Partitura Part object with reordered voices (in place)
#     """
#
#     def sorting_key(note: pt.score.GenericNote) -> tuple:
#         note.duration
#         return (
#             note.start.t,
#             *((0, note.midi_pitch, note.step) if isinstance(note, pt.score.Note) else (1, 0, "")),
#         )
#
#     def reorder_measure(notes: list[pt.score.GenericNote]) -> None:
#         notes.sort(key=sorting_key)
#         voices_mapping: dict[int, dict[int, int]] = defaultdict(
#             dict
#         )  # staff -> old_voice -> new_voice
#         for note in notes:
#             staff_of_voice = note.voice // 4  # staff in which this voice belong (by default)
#             voice_in_staff = note.voice % 4  # voice index in the staff
#             if voice_in_staff not in voices_mapping[staff_of_voice]:
#                 voices_mapping[staff_of_voice][voice_in_staff] = len(voices_mapping[staff_of_voice])
#             note.voice = voices_mapping[staff_of_voice][voice_in_staff] + staff_of_voice * 4
#
#     for measure in part.measures:
#         notes = list(
#             part.iter_all(
#                 cls=pt.score.GenericNote,
#                 start=measure.start.t,
#                 end=measure.end.t,
#                 include_subclasses=True,
#             )
#         )
#         reorder_measure(notes)


def reorder_voices(part: pt.score.Part) -> pt.score.Part:
    """
    Reorder voices so that in each staff, the "first" (% 4) voice is always the first note starting.
    In details, the order is the following:
        - First elements come first
        - When simultaneous events:
            - Notes come before rests
            - Notes are sorted by descending MIDI pitch
            - Enharmonic notes are sorted by ascending step

    :param part: Partitura Part object
    :return: Partitura Part object with reordered voices (in place)
    """

    def get_voices_mapping(notes_and_rests: list[pt.score.GenericNote]):
        notes = [n for n in notes_and_rests if isinstance(n, pt.score.Note)]
        rests = [r for r in notes_and_rests if isinstance(r, pt.score.Rest)]
        notes.sort(
            key=lambda n: (
                n.start.t,
                -n.midi_pitch,
                n.step,
                n.duration,
            )
        )
        voices_mapping: dict[int, int] = {}
        voices_idx = defaultdict(lambda: itertools.count(1))
        for note in notes:
            if note.voice not in voices_mapping:
                _, staff_of_voice = voice_absolute_into_voice_relative_to_staff(
                    note.voice,
                    one_based=True,
                )
                new_voice_in_staff = next(voices_idx[staff_of_voice])
                assert (
                    new_voice_in_staff < 4
                ), f"Voice {note.voice} is not in the first 4 voices of its staff"
                # voices_mapping[note.voice] = next(voices_idx[staff_of_voice]) + staff_of_voice * 4
                voices_mapping[note.voice] = voice_relative_to_staff_into_voice_absolute(
                    new_voice_in_staff,
                    staff_of_voice,
                    one_based=True,
                )

        # Add rest voices if needed
        remaining_rests = sorted(
            filter(lambda r: r.voice not in voices_mapping, rests),
            key=lambda r: (r.start.t, r.duration, r.voice),
        )
        for rest in remaining_rests:
            if rest.voice not in voices_mapping:
                _, staff_of_voice = voice_absolute_into_voice_relative_to_staff(
                    rest.voice,
                    one_based=True,
                )
                new_voice_in_staff = next(voices_idx[staff_of_voice])
                assert (
                    new_voice_in_staff < 4
                ), f"Voice {rest.voice} is not in the first 4 voices of its staff {staff_of_voice}"
                voices_mapping[rest.voice] = voice_relative_to_staff_into_voice_absolute(
                    new_voice_in_staff,
                    staff_of_voice,
                    one_based=True,
                )
        return voices_mapping

    def reorder_measure(notes_and_rests: list[pt.score.GenericNote]) -> None:
        voice_mapping = get_voices_mapping(notes_and_rests)
        for note in notes:
            note.voice = voice_mapping[note.voice]

    for measure in part.measures:
        notes = list(
            part.iter_all(
                cls=pt.score.GenericNote,
                start=measure.start.t,
                end=measure.end.t,
                include_subclasses=True,
            )
        )
        reorder_measure(notes)
    return part


if __name__ == '__main__':
    from pathlib import Path

    path = Path("/home/gerel/Documents/datasets/bach-dataset/Bach/Fugue/bwv_868/xml_score.musicxml")
    score = pt.load_musicxml(path)
    assert len(score.parts) == 1
    score = score.parts[0]
    reorder_voices(score)
    pt.save_musicxml(score, "/tmp/reordered_xml_score.musicxml")
