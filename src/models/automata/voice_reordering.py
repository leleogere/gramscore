import operator
from collections import defaultdict
import itertools
from pprint import pprint

import partitura as pt
import partitura.score

from utils.musicxml import voice_absolute_into_voice_relative_to_staff
from utils.musicxml import voice_relative_to_staff_into_voice_absolute


def sorting_criterion(note_or_rest: pt.score.GenericNote):
    if isinstance(note_or_rest, pt.score.Note):
        return (
            0,  # Notes before rests
            note_or_rest.start.t,  # First notes come first
            not isinstance(note_or_rest, pt.score.GraceNote),  # Grace notes come
            -note_or_rest.midi_pitch,  # Higher pitches come first
            note_or_rest.step,  # Enharmonic notes are sorted by ascending step
            note_or_rest.duration,  # Shorter notes come first
        )
    else:
        note_or_rest: pt.score.Rest  # Only for tying purposes
        return (
            1,  # Rests after notes
            note_or_rest.start.t,  # First rests come first
            False,  # No grace notes for rests
            0,  # No pitch for rests
            "",  # No step for rests
            note_or_rest.duration,  # Shorter rests come first
        )


def get_voice_mapping(notes_and_rests: list[pt.score.GenericNote]) -> dict[int, int]:
    """
    Build a mapping to reorder voices a sequence of notes, so that each voice has an increasing
    index, corresponding to the first time it appears on the sequence, or, in case of ties, to
    the first time one of those events will break the tie.
    """
    # Sort notes and rests by the sorting criterion (store it a used multiple times)
    notes_and_criterion = [(n, sorting_criterion(n)) for n in notes_and_rests]
    notes_and_criterion.sort(key=operator.itemgetter(1))

    # Note that in case of shared note-head between voices, we have an ambiguity as we do not
    # have a way to order them for now. We will then postpone the resolution of this ambiguity
    # to a later note: the first one that won't have this ambiguity.
    notes_grouped_by_ambiguity = [
        [n for n, _ in ambiguous_notes]  # do not need the criteria anymore
        for _, ambiguous_notes in itertools.groupby(notes_and_criterion, key=operator.itemgetter(1))
    ]

    # Create tracking variables
    mapping: dict[int, int | None] = {}  # Mapping from old voices to new voices
    groups_to_resolve: list[tuple[set[int], list[int]]] = []  # Ambiguities to resolve
    current_index_per_staff = defaultdict(lambda: 1)  # Index of the next allowed voice per staff

    for ambiguous_notes in notes_grouped_by_ambiguity:
        # Get all present voices in this group
        voices = set(n.voice for n in ambiguous_notes)

        # Deal with voices that have not been seen so far
        unseen_voices = voices - set(mapping.keys())
        if unseen_voices:
            # Split the unseen voices per staff
            unseen_voice_per_staff = defaultdict(set)
            for voice in unseen_voices:
                _, staff = voice_absolute_into_voice_relative_to_staff(voice, one_based=True)
                unseen_voice_per_staff[staff].add(voice)

            # Deal with each staff separately
            for staff, staff_unseen_voices in unseen_voice_per_staff.items():
                # We store voices in groups containing old voices and new voices
                # When the voice is alone, the mapping is easy (will be delt with latter)
                # When there are multiple voices, there is an ambiguity that we will need to
                # resolve later
                new_voices_on_staff = range(
                    current_index_per_staff[staff],
                    current_index_per_staff[staff] + len(staff_unseen_voices),
                )
                new_voices_absolute = [
                    voice_relative_to_staff_into_voice_absolute(
                        new_voice_on_staff,
                        staff,
                        one_based=True,
                    )
                    for new_voice_on_staff in new_voices_on_staff
                ]
                current_index_per_staff[staff] += len(staff_unseen_voices)
                groups_to_resolve.append((staff_unseen_voices, new_voices_absolute))
                for e in staff_unseen_voices:
                    mapping[e] = None

        # See if some voices can be resolved
        events = set(voices)
        idx = 0  # No for loop as we are modifying the list as iterating
        while idx < len(groups_to_resolve):
            old_voices, new_voices = groups_to_resolve[idx]

            # If there is only one item in the group, we can resolve it and set the mapping
            if len(old_voices) == 1:
                e = old_voices.pop()
                i = new_voices[0]
                mapping[e] = i
                groups_to_resolve.pop(idx)

            # Else, we see if the current events can resolve the group
            elif 0 < len((intersection := old_voices.intersection(events))) < len(old_voices):
                difference = old_voices - intersection
                inter_indices = new_voices[: len(intersection)]
                diff_indices = new_voices[len(intersection) :]
                groups_to_resolve.pop(idx)
                groups_to_resolve.extend(
                    [
                        (intersection, inter_indices),
                        (difference, diff_indices),
                    ]
                )

            else:
                # Only increment when we do not remove a group (otherwise we would miss a group)
                idx += 1

    # Remaining ambiguities are due to voices being exactly identical
    # In those cases, we arbitrarily choose voice with the lowest index and drop the others
    staffs_to_fix_order = set()  # Keep track of staffs to be fixed after discarding voices
    if groups_to_resolve:
        # Go through the groups to resolve, choose the first voice each time, and drop the others
        for old_voices, new_voices in groups_to_resolve:
            # Arbitrary take the lowest voice in the original score
            chosen = sorted(old_voices)[0]
            staffs_to_fix_order.add(
                voice_absolute_into_voice_relative_to_staff(chosen, one_based=True)[1]
            )
            mapping[chosen] = new_voices[0]
            # Drop other voices in the group (they are identical)
            for e in old_voices:
                if e != chosen:
                    mapping.pop(e)

    # Remove voices with a single measure-rest (except it's the only voice of the staff)
    voices_with_full_rests_to_remove = set(
        (r.voice, r.staff)
        for r in notes_and_rests
        if isinstance(r, pt.score.Rest)  # Get rests
        and not r.symbolic_duration  # Only measure-rests (they do not have symbolic duration)
        and r.voice in mapping  # Discard if the voice have been deleted before
        and voice_absolute_into_voice_relative_to_staff(mapping[r.voice], one_based=True)[0]
        != 1  # They are not the first (and only because of sorting) voice of the staff
    )
    for voice, staff in voices_with_full_rests_to_remove:
        mapping.pop(voice)
        staffs_to_fix_order.add(staff)

    # Shift mapping to fill holes introduced by removing some voices (staff-wise)
    for staff in staffs_to_fix_order:
        first_voice_of_staff = voice_relative_to_staff_into_voice_absolute(
            1,
            staff,
            one_based=True,
        )
        # The following statement does:
        #   - Take all pairs (old_voice, new_voice) that are in the staff (e.g. 5-8 for staff 2)
        #   - Sort them by new_voice (e.g. 5, 7, 8, if voice 6 is missing)
        #   - Enumerate this list, starting with the first voice of the staff (e.g. 5)
        #   - Modify the mapping associating the old voice with this new enumerated value,
        #     therefore without any holes
        staff_mapping = {
            old: new_without_holes
            for new_without_holes, (old, new) in enumerate(
                sorted(
                    filter(
                        lambda item: first_voice_of_staff <= item[0] < first_voice_of_staff + 4,
                        mapping.items(),
                    ),
                    key=operator.itemgetter(1),
                ),
                start=first_voice_of_staff,
            )
        }
        mapping.update(staff_mapping)

    # Sort mapping (not needed, only for debugging)
    # mapping = {e: i for e, i in sorted(mapping.items(), key=operator.itemgetter(0))}

    assert all(v is not None for v in mapping.values()), "Some voices have not been resolved"
    return mapping


def reorder_voices(part: pt.score.Part) -> pt.score.Part:
    """
    Reorder voices so that in each staff, the "first" (% 4) voice is always the first note starting.
    In details, the order is the following is detailed in the function `sorting_criterion`.

    In case of multiple exactly identical voices, only one of them will be kept.

    :param part: Partitura Part object
    :return: Partitura Part object with reordered voices (in place)
    """
    for measure in part.measures:
        notes = list(
            part.iter_all(
                cls=pt.score.GenericNote,
                start=measure.start.t,
                end=measure.end.t,
                include_subclasses=True,
            )
        )
        # notes = [n for n in notes if n.staff == 2]
        voice_mapping = get_voice_mapping(notes)
        for note in notes:
            if note.voice in voice_mapping:
                # print(f"Note H{note.staff} V{note.voice} -> V{voice_mapping[note.voice]}")
                note.voice = voice_mapping[note.voice]
            else:
                # print(f"Note H{note.staff} V{note.voice} has no voice mapping")
                part.remove(note)
    return part


if __name__ == '__main__':
    from pathlib import Path

    # fmt: off
    path = Path("~/Documents/datasets/bach-dataset/Bach/Fugue/bwv_868/xml_score.musicxml").expanduser()
    # path = Path("~/Documents/MuseScore4/Scores/voices_to_reorder.musicxml").expanduser()
    # fmt: on

    score = pt.load_musicxml(path)
    assert len(score.parts) == 1
    score = score.parts[0]
    reorder_voices(score)
    pt.save_musicxml(score, "/tmp/reordered_xml_score.musicxml")
