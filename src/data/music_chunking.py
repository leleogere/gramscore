import pretty_midi as pm
import copy
import itertools
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import assert_never

import pandas as pd
import partitura as pt
import partitura.score
import symusic
from loguru import logger

import models.automata.tokens as tk
from models.automata.io import partitura_to_token_sequence
from models.automata.io import token_sequence_to_partitura
from models.automata.tokens import HasStaffMixin
from utils.midi import crop_pretty_midi, compute_midi_end_without_tempo_changes
from utils.musicxml import voice_absolute_into_voice_relative_to_staff


def read_chunk_file(path: Path) -> pd.DataFrame:
    """
    This function reads a JSON chunk files, in the following format:
    {
        "0": [0, 1, 2, 3],
        "1-2": [4, 5, 6, 7, 8, 9, 10],
        "3": [11, 12, 13, 14, 15],
        "4-5-6": [16, 17, 18, 19, 20, 21, 22],
        ...
    }
    The keys are the index of the measures in the score, and the values are the indices of the
    notes in the performance belonging to that measure (or group of measures).
    There is one entry per downbeat in the annotations.

    It returns a DataFrame with one row per downbeat/group, with the following columns:
        - "measures" <list[int]>: The index of the measures in the score
        - "first_measure_of_group" <int>: The index of the first measure in the group
        - "last_measure_of_group" <int>: The index of the last measure in the group
        - "notes" <list[int]>: The indices of the notes in the performance belonging to that
            downbeat
        - "first_note_of_group" <int>: The index of the first note in the group (if group empty,
            the first note of the next group)
        - "last_note_of_group" <int>: The index of the last note in the group (if group empty,
            the last note of the previous group)  # TODO remove that info
        - "is_empty" <bool>: Whether the group is empty or not (in that case we have
            last_note_of_group > first_note_of_group)

    :param path: The path to the chunk file.
    :return: A DataFrame with the chunk information.
    """
    with path.open("r") as f:
        raw_chunks = json.load(f)

    # Convert to DataFrame
    chunks = pd.DataFrame(raw_chunks.items(), columns=["measures", "notes"])

    # Convert the string measure column to a list of integers
    chunks["measures"] = chunks["measures"].map(lambda x: [int(m) for m in str(x).split("-")])

    # Get the first measure of each group
    chunks["first_measure_of_group"] = chunks["measures"].map(lambda x: x[0])

    # Get the last measure of each group
    chunks["last_measure_of_group"] = chunks["measures"].map(lambda x: x[-1])

    # Get the first note of each group
    chunks["first_note_of_group"] = (
        chunks["notes"]
        .map(
            lambda x: x[0] if len(x) > 0 else None,
        )
        .astype(pd.Int64Dtype())
    )

    # Get the last note of each group
    chunks["last_note_of_group"] = (
        chunks["notes"]
        .map(
            lambda x: x[-1] if len(x) > 0 else None,
        )
        .astype(pd.Int64Dtype())
    )

    # Add empty column
    chunks["is_empty"] = chunks["notes"].map(lambda x: len(x) == 0)

    return chunks


def clip_midi(
    midi: pm.PrettyMIDI,
    chunks: pd.DataFrame,
    start_measure: int,
    num_notes: int | None = None,
    inplace: bool = True,
    check_num_measures: int | None = None,
) -> tuple[pm.PrettyMIDI, tuple[int, int]]:
    """
    Clip a MIDI performance to an approximate number of notes, starting from a given measure.
    This function can return more than the required number of notes as it will accumulate notes
    to the end of the last measure so that it returns a complete last measure.
    This function can also return less than the required number of notes if there are less than
    the required number of notes in the performance starting from the given measure.

    :param midi: The MIDI performance to clip.
    :param chunks: A dict mapping the measure number (or numbers) to the note indices in the
        performance belonging to that measure (or group of measures). Each entry correspond to a
        downbeat in the annotations. See ASAP `downbeats_score_map`.
    :param start_measure: Index of the measure to start the chunk. If it belongs to a group of
        measures, the first measure of the group will be used instead.
    :param num_notes: Required number of notes to gather, starting from the start measure.
        Note that there might be more than the required number of notes (if there are still notes
        in the last downbeat group), or less (if the remaining notes from the start measure is
        less than the required number of notes).
    :param inplace: Whether to modify the input MIDI performance or return a new one.
    :param check_num_measures: Only used to check that the total number of measures in the MIDI
        performance matches some externally computed number of measures.

    :return: The clipped MIDI performance and the range of measures (inclusive) used to clip it.

    # TODO: update the docstring
    """
    if len(midi.instruments) > 1:
        raise ValueError("MIDI performance must have only one track.")

    if not inplace:
        midi = copy.deepcopy(midi)

    # Make sure all numbers are consecutive in "midi" when they are flattened
    # TODO: remove when everything is working
    flatten = sum(chunks["measures"], [])
    assert all(flatten[i] == flatten[i - 1] + 1 for i in range(1, len(flatten)))

    # Check number of measures if needed
    num_measures = sum(len(measures) for measures in chunks["measures"])
    if check_num_measures is not None and check_num_measures != num_measures:
        raise ValueError(
            f"Number of measures in the chunks file ({num_measures}) does not match the "
            f"required number of measures ({check_num_measures})"
        )

    # If retrieve the index of the downbeat group corresponding to the required measure
    if start_measure > chunks["last_measure_of_group"].iloc[-1]:
        raise ValueError(
            f"Start measure {start_measure} is greater than the last measure "
            f"{chunks['last_measure_of_group'].iloc[-1]}"
        )

    # Find the downbeat group contining the required start measure
    start_downbeat = next(
        (
            i
            for i, row in chunks.iterrows()
            if row["first_measure_of_group"] <= start_measure <= row["last_measure_of_group"]
        ),
    )
    if chunks.iloc[start_downbeat]["first_measure_of_group"] != start_measure:
        logger.info(
            f"Start measure {start_measure} is not the first measure of a group, "
            f"using first measure of {chunks.iloc[start_downbeat]['measures']} instead."
        )

    # If the start_downbeat group is empty, find the first non-empty group
    if chunks.iloc[start_downbeat]["is_empty"]:
        # Find the first non-empty group
        new_start_downbeat = next(
            (i for i, row in chunks.iloc[start_downbeat:].iterrows() if not row["is_empty"])
        )
        # Will error in case of all empty groups after this one, but as we make sure we take a
        # couple of measures of margin when sampling start_downbeat, this should not happen. I do
        # not catch the error as if it happens it will likely indicate a bug.
        logger.info(
            f"Start downbeat group {start_downbeat} is empty, using {new_start_downbeat} from "
            f"{chunks.iloc[new_start_downbeat]['measures']} instead."
        )
        start_downbeat = new_start_downbeat

    total_notes = len(midi.instruments[0].notes)
    assert (
        chunks["last_note_of_group"].max() == total_notes - 1
    ), f"{chunks['last_note_of_group'].max()} != {total_notes - 1}"

    # If num_notes, find the end measure based on the required number of notes
    end_downbeat = (
        next(
            (
                i
                for i, row in chunks.iterrows()
                if not row["is_empty"]
                and row["first_note_of_group"]
                <= chunks.iloc[start_downbeat]["first_note_of_group"] + num_notes
                <= row["last_note_of_group"]
            ),
            # In case of not enough notes, take the first not empty measure from the end
            next(i for i, row in chunks.iloc[::-1].iterrows() if not row["is_empty"]),
        )
        if num_notes is not None
        else chunks.index[-1]
    )
    end_measure = chunks.iloc[end_downbeat]["last_measure_of_group"]

    # Keep only the performed notes in the queried measures
    last_note = chunks.iloc[end_downbeat]["last_note_of_group"]
    clipped_notes = midi.instruments[0].notes[
        chunks.iloc[start_downbeat]["first_note_of_group"] : (
            last_note + 1 if not pd.isna(last_note) else total_notes
        )
    ]
    midi.instruments[0].notes = clipped_notes

    # Clip the performance to the timing of the first and last notes
    start_time = clipped_notes[0].start
    end_time = max(note.end for note in clipped_notes) if not pd.isna(last_note) else compute_midi_end_without_tempo_changes(midi)
    crop_pretty_midi(
        midi, start=start_time, end=end_time, start_mode="drop", end_mode="clip", inplace=True
    )

    return midi, (start_measure, end_measure)


def clip_score(
    score: pt.score.Part,
    start_measure: int,
    end_measure: int,
    check_num_measures: int | None = None,
) -> pt.score.Part:
    """
    Clip a score to a range of measures or an approximate number of notes.

    :param score: The score to clip.
    :param start_measure: The index of the measure to start the clip.
    :param end_measure: The index of the measure to end the clip (inclusive).
    :param check_num_measures: Only used to check that the total number of measure in the score
        matches some externally computed number of measures.

    :return: The clipped score.
    """
    # Check and process measure boundaries parameters
    if end_measure < start_measure:
        raise ValueError("End measure must be greater than or equal to start measure")
    start_point = score.measures[start_measure].start

    # Check number of measures if needed
    if check_num_measures is not None and check_num_measures != len(score.measures):
        raise ValueError(
            f"Number of measures in the score ({check_num_measures}) does not match the "
            f"required number of measures ({len(score.measures)})"
        )

    # Create a new ScoreVariant
    sv = pt.score.ScoreVariant(score)
    sv.add_segment(start_point, score.measures[end_measure].end)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clipped_score: pt.score.Part = sv.create_variant_part()
    clipped_score.set_quarter_duration(0, int(score.quarter_duration_map(start_point.t)))

    # When creating a score variant, notes that where tied before the start of the variant have
    # been kept, but the tie has been removed. Ideally, we want those notes to be completly removed
    # (i.e. replaced by rests) because they are not included in the corresponding chunked MIDI.
    # Therefore, we find those in the original score, and replace them with rests in the variant.

    # Mapping from note id (same in both original and clipped score) to note in the clipped score
    id_to_clipped_score_note: dict[str, pt.score.Note] = {n.id: n for n in clipped_score.notes}
    if any(x is None for x in id_to_clipped_score_note.keys()):
        raise ValueError(
            "Some notes do not have an ID, make sure to load them with `force_note_ids=True`"
        )
    # Track the number of notes in each voice at each start time (to replace by rest only if needed)
    num_notes_per_voice_and_start = defaultdict(int)
    # Store IDs of notes that will need to be removed (and original note to access the timings)
    notes_to_remove = {}
    # Iterate over notes in the original start measure (and the notes tied to them)
    for note in start_point.starting_objects[pt.score.Note]:
        # Store the voice and start time
        num_notes_per_voice_and_start[(note.voice, note.start.t)] += 1
        # If tied to a previous note, mark as to remove
        if note.tie_prev:
            notes_to_remove[note.id] = note
            # If this note is also tied to next notes, mark them as to remove
            for next_tie in note.tie_next_notes:
                notes_to_remove[next_tie.id] = next_tie
                # Also track this timepoint to add a rest there too if needed (only needed once
                # as subsequent iteration on this timepoint will be the same)
                # The first None check is required for some weird case to investigate, e.g. in
                # bar 28 of PDMX score mxl/2/3/Qmc4PvPfVJdH6ZAmMHmBsm96soqi8u89FwL5jZrmvGr4mc.mxl
                if (
                    next_tie.start is not None
                    and not (next_tie.voice, next_tie.start.t) in num_notes_per_voice_and_start
                ):
                    num_notes_per_voice_and_start[(next_tie.voice, next_tie.start.t)] = len(
                        list(next_tie.iter_chord())
                    )
    # Go over notes to remove, remove them and if needed replace by rests
    # Also make sure that tuplets are correctly updated in case of rest replacement
    tuplets_to_fix = []
    for note_id, original_note in notes_to_remove.items():
        # Remove the note from the clipped score
        if note_id not in id_to_clipped_score_note:
            logger.warning(f"{note_id} absent so can't be removed from score")
            continue
        note_to_remove = id_to_clipped_score_note[note_id]
        # Check if a tuplet was linked to that note
        for tuplet in clipped_score.first_point.starting_objects[pt.score.Tuplet]:
            if note_to_remove is tuplet.start_note:
                # Store the tuplet with its updated start and end note times and voice
                tuplets_to_fix.append(
                    (
                        tuplet,
                        tuplet.start_note.start.t,
                        tuplet.end_note.start.t,
                        tuplet.start_note.voice,
                    )
                )
                assert (
                    tuplet.start_note.voice == tuplet.end_note.voice
                ), "Mismatching voices in tuplet"
        # Remove the note
        clipped_score.remove(note_to_remove)
        # Make as removed
        num_notes_per_voice_and_start[(note_to_remove.voice, original_note.start.t)] -= 1
        # If no more notes in the voice at that start, replace by a rest
        if not num_notes_per_voice_and_start[(note_to_remove.voice, original_note.start.t)]:
            rest = pt.score.Rest(
                voice=note_to_remove.voice,
                staff=note_to_remove.staff,
                symbolic_duration=note_to_remove.symbolic_duration,
            )
            clipped_score.add(
                rest,
                start=original_note.start.t - start_point.t,
                end=original_note.end.t - start_point.t,
            )

    # Fix the eventual tuplets
    for tuplet, start_note_time, end_note_time, voice in tuplets_to_fix:
        # Try to find the new start/end notes of the tuplet
        # Simple matching with ID is not feasible, as in the previous loop, the note might have
        # been removed (in which case try to find a note in the same voice and at the same time)
        # or replaced by a rest (same but with a rest)
        start_pt: pt.score.TimePoint = clipped_score.get_point(start_note_time)
        start_element = next(
            filter(
                lambda e: e.voice == voice,
                itertools.chain(
                    start_pt.starting_objects[pt.score.Note],
                    start_point.starting_objects[pt.score.Rest],
                ),
            )
        )
        tuplet.start_note = start_element
        end_pt: pt.score.TimePoint = clipped_score.get_point(end_note_time)
        end_element = next(
            filter(
                lambda e: e.voice == voice,
                itertools.chain(
                    end_pt.starting_objects[pt.score.Note],
                    end_pt.starting_objects[pt.score.Rest],
                ),
            )
        )
        tuplet.end_note = end_element

    # Finally, add back global context elements (key/time signature and clefs)
    context_elements_classes_still_to_be_found = [
        pt.score.TimeSignature,
        pt.score.KeySignature,
        pt.score.Clef,
    ]
    context_elements_found = {
        pt.score.TimeSignature: False,
        pt.score.KeySignature: False,
        pt.score.Clef: [False] * clipped_score.number_of_staves,
    }
    # Look for elements already present at the start
    for context_cls in context_elements_classes_still_to_be_found[:]:
        if elements := start_point.starting_objects.get(context_cls, None):
            if context_cls == pt.score.Clef:
                for clef in elements:
                    context_elements_found[context_cls][clef.staff - 1] = True
                if all(context_elements_found[context_cls]):
                    context_elements_classes_still_to_be_found.remove(context_cls)
            else:
                context_elements_found[context_cls] = True
                context_elements_classes_still_to_be_found.remove(context_cls)
    # Look for elements in previous measures for the ones not found yet
    prev_context_elements = start_point.iter_prev(
        cls=list(context_elements_classes_still_to_be_found)
    )
    while context_elements_classes_still_to_be_found:
        # Find the previous element
        try:
            prev_context_element = next(prev_context_elements)
        except StopIteration:
            logger.warning(
                f"Could not find all context elements: {context_elements_classes_still_to_be_found}"
            )
            break
        element_cls = type(prev_context_element)
        new_element = None

        # Deal with Clef elements
        if (
            isinstance(prev_context_element, pt.score.Clef)
            and not context_elements_found[pt.score.Clef][prev_context_element.staff - 1]
        ):
            context_elements_found[pt.score.Clef][prev_context_element.staff - 1] = True
            new_element = pt.score.Clef(
                staff=prev_context_element.staff,
                sign=prev_context_element.sign,
                line=prev_context_element.line,
                octave_change=prev_context_element.octave_change,
            )
            # If all clefs found, do not search for it anymore
            if all(context_elements_found[pt.score.Clef]):
                context_elements_classes_still_to_be_found.remove(pt.score.Clef)

        # Deal with TimeSignature and KeySignature elements
        elif not context_elements_found[element_cls]:
            context_elements_found[type(prev_context_element)] = True
            context_elements_classes_still_to_be_found.remove(type(prev_context_element))
            if isinstance(prev_context_element, pt.score.TimeSignature):
                new_element = pt.score.TimeSignature(
                    beats=prev_context_element.beats,
                    beat_type=prev_context_element.beat_type,
                )
            elif isinstance(prev_context_element, pt.score.KeySignature):
                new_element = pt.score.KeySignature(
                    fifths=prev_context_element.fifths,
                    mode=prev_context_element.mode,
                )
            else:
                assert_never(prev_context_element)

        # Add new element if found
        if new_element is not None:
            clipped_score.add(new_element, start=0)

    return clipped_score


def clip_quantized_midi(
    midi: pm.PrettyMIDI,
    start_measure: int,
    num_notes: int,
    inplace: bool = True,
    check_num_measures: int | None = None,
) -> tuple[pm.PrettyMIDI, tuple[int, int]]:
    """
    Clip a quantized MIDI performance (with correct metricl information) to an approximate number
    of notes, starting from a given measure.
    This function can return more than the required number of notes as it will accumulate notes
    to the end of the last measure so that it returns a complete last measure.
    This function can also return less than the required number of notes if there are less than
    the required number of notes in the performance starting from the given measure.

    :param midi: The quantized MIDI performance to clip.
    :param start_measure: Index of the measure to start the chunk.
    :param num_notes: Required number of notes to gather, starting from the start measure.
        Note that there might be more than the required number of notes (if there are still notes
        in the last downbeat group), or less (if the remaining notes from the start measure is
        less than the required number of notes).
    :param inplace: Whether to modify the input MIDI performance or return a new one.
    :param check_num_measures: Only used to check that the total number of measures in the MIDI
        performance matches some externally computed number of measures.

    :return: The clipped MIDI performance and the range of measures (inclusive) used to clip it.
    """
    if len(midi.instruments) > 1:
        raise ValueError("MIDI performance must have only one instrument.")

    if not inplace:
        midi = copy.deepcopy(midi)

    # Get downbeat times
    downbeat_times = midi.get_downbeats()
    total_measures = len(downbeat_times)

    # Check number of measures if needed
    if check_num_measures is not None and check_num_measures != total_measures:
        raise ValueError(
            f"Number of measures in the MIDI ({total_measures}) does not match the "
            f"required number of measures ({check_num_measures})"
        )
    if start_measure >= total_measures:
        raise ValueError(
            f"Start measure {start_measure} is greater than or equal to the total number of "
            f"measures ({total_measures})"
        )

    # Find start and end times
    start_time = downbeat_times[start_measure]
    accumulated_notes = 0
    end_measure = start_measure
    for measure_idx in range(start_measure, total_measures):
        measure_start_time = downbeat_times[measure_idx]
        measure_end_time = (
            downbeat_times[measure_idx + 1]
            if measure_idx + 1 < total_measures
            else midi.instruments[0].get_end_time()
        )
        # Count notes in this measure
        notes_in_measure = [
            note
            for note in midi.instruments[0].notes
            if note.start >= measure_start_time and note.start < measure_end_time
        ]
        accumulated_notes += len(notes_in_measure)
        end_measure = measure_idx
        if accumulated_notes >= num_notes:
            break
    end_time = (
        downbeat_times[end_measure + 1]
        if end_measure + 1 < total_measures
        else midi.instruments[0].get_end_time()
    )

    # Crop the MIDI
    midi = crop_pretty_midi(
        midi,
        start=start_time,
        end=end_time,
        inplace=True,
        start_mode="drop",
        end_mode="clip",
    )

    return midi, (start_measure, end_measure)


def clip_token_sequence(
    tokens: list[tk.Token],
    measure_range: tuple[int, int | None],
    num_notes: int | None = None,
    return_measure_range: bool = False,
    check_num_measures: int | None = None,
) -> list[tk.Token] | tuple[list[tk.Token], tuple[int, int]]:
    """
    Clip a token sequence to a range of measures or an approximate number of notes.

    :param tokens: The token sequence to clip.
    :param measure_range: The range of measures (start, end) to clip the token sequence to. If the
        end measure is None, it will be computed based on num_notes.
    :param num_notes: Only used when the end measure is None, it will accumulate the required
        number of notes starting from the measure start, continuing until the end of the measure.
        Note that there might be more than the required number of notes (if there are still notes
        in the last measure), or less (if the remaining notes from the start measure is less than
        the required number of notes).
    :param return_measure_range: Whether to return the actual measure range used to clip the
        token sequence (useful when num_notes is used).
    :param check_num_measures: Only used to check that the total number of measure in the token
        sequence matches some externally computed number of measures.
    """
    # Strip SOS and EOS as will be added back later
    if not isinstance(tokens[0], tk.StartOfScore) or not isinstance(tokens[-1], tk.EndOfScore):
        raise ValueError("Token sequence must start with StartOfScore and end with EndOfScore")
    tokens = tokens[1:-1]

    # Check and process measure boundaries parameters
    start_measure, end_measure = measure_range
    if (end_measure is None) == (num_notes is None):
        raise ValueError("Exactly one of end_measure and num_notes must be specified")

    measure_to_index = dict(
        enumerate(i for i, token in enumerate(tokens) if isinstance(token, tk.Bar))
    )
    measure_to_index[len(measure_to_index)] = len(tokens)  # Add for last measure
    start_measure_idx = measure_to_index[start_measure]
    end_measure_idx = measure_to_index[end_measure] if end_measure is not None else None

    # Check number of measures if needed
    if check_num_measures is not None and check_num_measures != len(measure_to_index) - 1:
        raise ValueError(
            f"Number of measures in the token sequence ({check_num_measures}) does not match the "
            f"required number of measures ({len(measure_to_index) - 1})"
        )

    # Compute the end measure from accumulated number of notes
    if num_notes is not None:
        accumulated_notes = 0
        for i, token in enumerate(tokens[start_measure_idx:], start=start_measure_idx):
            if isinstance(token, tk.Note):
                accumulated_notes += 1
            if accumulated_notes >= num_notes:
                end_measure, end_measure_idx = next(
                    ((meas, idx) for meas, idx in measure_to_index.items() if idx > i),
                    None,
                )
                if end_measure_idx is None:
                    end_measure, end_measure_idx = len(measure_to_index), len(tokens)
                break
    # print(f"{start_measure = }, {end_measure = }")
    # print(f"{start_measure_idx = }, {end_measure_idx = }")

    # Clip the token sequence
    clipped_tokens = tokens[start_measure_idx:end_measure_idx]

    # Add context elements
    # TODO: search for all context elements in a single pass (more efficient)
    context_cls = (tk.Bar, tk.KeySignature, tk.Clef)
    current_context = list(
        itertools.takewhile(lambda t: isinstance(t, context_cls), clipped_tokens)
    )
    # Find previous key signature
    if not any(isinstance(t, tk.KeySignature) for t in current_context):
        for token in reversed(tokens[:start_measure_idx]):
            if isinstance(token, tk.KeySignature):
                clipped_tokens.insert(1, token)  # Insert in 1 as 0 is Bar token
                break
    # Find previous clefs
    num_staves = len(set(t.staff_idx for t in tokens if isinstance(t, HasStaffMixin)))
    for staff_idx in reversed(
        range(num_staves)
    ):  # Reversed to compensate for ".insert" reversing order
        if not any(isinstance(t, tk.Clef) and t.staff_idx == staff_idx for t in current_context):
            for token in reversed(tokens[:start_measure_idx]):
                if isinstance(token, tk.Clef) and token.staff_idx == staff_idx:
                    clipped_tokens.insert(1, token)  # Insert in 1, before eventual KS
                    break

    # Fix unresolved tied notes
    notes_tied_before_first_measure = {}
    for i, token in enumerate(tokens[:start_measure_idx]):
        if isinstance(token, tk.Note):
            if token.tied:
                notes_tied_before_first_measure[token.pitch] = token
            elif token.pitch in notes_tied_before_first_measure:
                notes_tied_before_first_measure.pop(token.pitch)

    notes_tied = {}
    i = 0
    voice_with_chords_started = set()
    while i < len(clipped_tokens):  # not for-loop because of modifying list
        token = clipped_tokens[i]

        # Deal with opening and closing chords
        if isinstance(token, tk.ChordStart):
            voice_with_chords_started.add(token.voice_idx)
        elif isinstance(token, tk.ChordEnd):
            voice_with_chords_started.remove(token.voice_idx)

        # Deal with tied notes
        elif isinstance(token, tk.Note):
            # If tied to a note before first measure, either remove it or replace it with a rest
            # (depending on weather we are in a chord, note that a chord might end empty, this will
            # be dealt with later on
            if token.pitch in notes_tied_before_first_measure:
                # If in chord, simply remove the token
                if token.voice_idx in voice_with_chords_started:
                    clipped_tokens.pop(i)
                    i -= 1  # Decrement as we removed an item
                # If not in chord, replace it with a rest
                else:
                    clipped_tokens[i] = tk.Rest(
                        fermata=token.fermata,
                        duration=token.duration,
                        staff_idx=token.staff_idx,
                        voice_idx=token.voice_idx,
                        position_in_bar=token.position_in_bar,
                    )
                # Only pop if not still tied as in that case there will be other notes to replace
                if not token.tied:
                    notes_tied_before_first_measure.pop(token.pitch)
            # If not tied to a too-early note, keep track of start/end ties
            else:
                if token.tied:
                    notes_tied[token.pitch] = token
                elif token.pitch in notes_tied:
                    notes_tied.pop(token.pitch)
        i += 1

    # Cut the tie of the last note of each group (the one exiting the last measure)
    for unfinished_tied_note in notes_tied.values():
        unfinished_tied_note.tied = False

    # Clean eventual rests in chords
    i = 0
    voice_with_chords_started = defaultdict(list)  # {voice -> [(i, ChordStart), (j, Note)...]}
    while i < len(clipped_tokens):
        token = clipped_tokens[i]
        print(i, token)

        if isinstance(token, tk.ChordStart):
            voice_with_chords_started[token.voice_idx].append((i, token))

        elif isinstance(token, tk.ChordEnd):
            chord_elements = voice_with_chords_started.pop(token.voice_idx)
            # If len <= 3 then this is either a single note (2) or an empty chord (1)
            if len(chord_elements) <= 2:
                assert len(chord_elements) > 0, "Should not happen"
                chords_start_idx, chord_start = chord_elements[0]
                clipped_tokens.pop(i)  # Pop ChordEnd (before ChordStart to avoid shift)
                clipped_tokens.pop(chords_start_idx)  # Pop ChordStart
                i -= 2
                # If len == 2, replace by a rest
                if len(chord_elements) == 1:
                    i += 1
                    staff = voice_absolute_into_voice_relative_to_staff(chord_start.voice_idx)[1]
                    rest = tk.Rest(
                        duration=chord_start.duration,
                        staff_idx=staff,
                        voice_idx=chord_start.voice_idx,
                        position_in_bar=chord_start.position_in_bar,
                        fermata=False,
                    )
                    clipped_tokens.insert(i, rest)

        elif isinstance(token, tk.Note) and token.voice_idx in voice_with_chords_started:
            voice_with_chords_started[token.voice_idx].append((i, token))
        print(voice_with_chords_started)

        i += 1

    # Add SOS and EOS back
    clipped_tokens.insert(0, tk.StartOfScore())
    clipped_tokens.append(tk.EndOfScore())

    if return_measure_range:
        return clipped_tokens, (start_measure, end_measure)
    else:
        return clipped_tokens


if __name__ == '__main__':
    import warnings

    asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
    # asap_path = Path(
    #     "~/Documents/transcription/MIDI2ScoreTransformer/midi2scoretransformer/data/asap-dataset"
    # ).expanduser()

    # piece_path = asap_path / "Bach/Fugue/bwv_846"
    # performance_path = piece_path / "Shi05M.mid"
    # piece_path = asap_path / "Beethoven/Piano_Sonatas/13-4"
    # performance_path = piece_path / "Taverna01.mid"
    # piece_path = asap_path / "Chopin/Scherzos/31"
    piece_path = asap_path / "Beethoven/Piano_Sonatas/29-4"
    performance_path = piece_path / "midi_score.mid"
    score_path = piece_path / "xml_score.musicxml"

    chunks_filename = f"{performance_path.stem}_chunks.json"
    annotations_path = asap_path / "asap_annotations.json"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pt.load_musicxml(score_path, force_note_ids=True)
        assert len(score.parts) == 1
        score = pt.score.unfold_part_maximal(score.parts[0], ignore_leaps=True, update_ids=True)
        print(f"{len(score.measures)} measures")
        # tk.fill_rests(score)

    midi = symusic.Score.from_file(performance_path)
    if len(midi.tracks) > 1:
        for track in midi.tracks[1:]:
            midi.tracks[0].notes.extend(track.notes)
        # Remove other tracks
        midi.tracks = midi.tracks[:1]
        # Sort by time
        midi.tracks[0].sort()

    chunks = read_chunk_file(piece_path / chunks_filename)
    annotations = pd.read_json(annotations_path, orient="index")
    if performance_path.stem != "midi_score":
        performance_annotations = annotations.loc[str(performance_path.relative_to(asap_path))]
    else:
        some_performance = next(
            filter(lambda f: f.stem != "midi_score", performance_path.parent.glob("*.mid"))
        ).relative_to(asap_path)
        # annotations.loc[str(performance_path.relative_to(asap_path))] = {
        performance_annotations = {
            "score_and_performance_aligned": True,
            "performance_beats": annotations.loc[str(some_performance)]["midi_score_beats"],
            "performance_beats_type": annotations.loc[str(some_performance)][
                "midi_score_beats_type"
            ],
        }

    tokens = partitura_to_token_sequence(score)

    m_start = 13
    n_notes = 500
    output_path = Path("/tmp")

    clipped_midi, (real_m_start, real_m_end) = clip_midi(
        midi=midi,
        chunks=chunks,
        start_measure=m_start,
        num_notes=n_notes,
        check_num_measures=len(score.measures),
        inplace=True,
    )
    print(f"Clipped from {real_m_start} to {real_m_end}")

    clipped_score = clip_score(
        score=score,
        start_measure=real_m_start,
        end_measure=real_m_end,
    )

    # clipped_tokens = clip_token_sequence(
    #     tokens=tokens, measure_range=(m_start, m_end), check_num_measures=len(score.measures)
    # )
    # automata = ConstrainedAutomata(num_voices=8)
    # automata.transition_sequence(clipped_tokens)

    pt.save_musicxml(clipped_score, output_path / "clipped_score.musicxml")
    clipped_midi.dump_midi(output_path / "clipped_performance.mid")
    # pt.save_musicxml(
    #     token_sequence_to_partitura(clipped_tokens),
    #     "/tmp/clipped_score_tokens.musicxml",
    # )
