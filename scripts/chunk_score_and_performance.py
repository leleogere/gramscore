import copy
import itertools
import warnings
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import pandas as pd
import partitura as pt
import partitura.score
import symusic
from loguru import logger

import models.automata.tokens as tk
from models.automata.automata import ConstrainedAutomata
from models.automata.io import token_sequence_to_partitura
from models.automata.tokens import fill_rests
from utils.musicxml import voice_absolute_into_voice_relative_to_staff
from models.automata.io import partitura_to_token_sequence
from models.automata.tokens import HasStaffMixin


def clip_midi(
    midi: symusic.Score,
    chunks: pd.DataFrame,
    performance_annotations: pd.DataFrame,
    measure_range: tuple[int, int | None],
    num_notes: int | None = None,
    inplace: bool = True,
    return_measure_range: bool = False,
    check_num_measures: int | None = None,
) -> symusic.factory.ScoreFactory | tuple[symusic.factory.ScoreFactory, tuple[int, int]]:
    """
    Clip a MIDI performance to a range of measures or an approximate number of notes.

    :param midi: The MIDI performance to clip.
    :param chunks: A dataframe containing the notes contained in every beat.
    :param performance_annotations: The annotation of every beat (b or db) in the performance.
    :param measure_range: The range of measures (start, end_excluded) to clip the performance to.
        If the end measure is None, it will be computed based on num_notes.
    :param num_notes: Only used when the end measure is None, it will accumulate the required
        number of notes starting from the measure start, continuing until the end of the measure.
        Note that there might be more than the required number of notes (if there are still notes
        in the last measure), or less (if the remaining notes from the start measure is less than
        the required number of notes).
    :param inplace: Whether to modify the input MIDI performance or return a new one.
    :param return_measure_range: Only used to check that the total number of measure in the MIDI
        matches some externally computed number of measures.
    """
    if not inplace:
        midi = midi.copy(deep=True)

    # Check measure boundaries parameters
    start_measure, end_measure = measure_range
    if end_measure is not None:
        # In this script, we will treat end_measure as inclusive, but we keep it exclusive in
        # the function signature to be consistent with the other clip_* functions
        end_measure_included = end_measure - 1
    if (end_measure is None) == (num_notes is None):
        raise ValueError("Exactly one of end_measure and num_notes must be specified")

    # Get beat types
    beat_types = pd.Series(performance_annotations["performance_beats_type"].values())

    # Merge with beat_type sequentially, with NaN at beginning if not same length
    # This happens when the first note does not fall on a beat
    num_nan = len(chunks) - len(beat_types)
    beat_type = pd.concat(
        [pd.Series([""] * num_nan), beat_types],
        axis=0,
        ignore_index=True,
    )
    assert len(chunks) == len(beat_type)  # TODO: remove this, unneeded check
    chunks["beat_type"] = beat_type

    # Assign each beat to its measure index
    chunks["measure"] = chunks["beat_type"].str.startswith("db").cumsum()
    chunks = chunks[["midi", "measure"]].groupby("measure").agg(lambda x: sum(x, []))

    # Make sure all numbers are consecutive in "midi" when they are flattened
    # TODO: remove when everything is working
    flatten = sum(chunks["midi"], [])
    assert all(flatten[i] == flatten[i - 1] + 1 for i in range(1, len(flatten)))

    # Keep only note index of each measure
    chunks["start"] = (
        chunks["midi"].apply(lambda x: int(x[0]) if x else pd.NA).astype(pd.Int64Dtype()).ffill()
    )
    chunks["end"] = chunks["start"].shift(-1)
    total_notes = len(midi.tracks[0].notes)
    chunks["end"].iloc[-1] = total_notes
    if check_num_measures is not None and check_num_measures != len(chunks):
        raise ValueError(
            f"Number of measures in the MIDI ({check_num_measures}) does not match the "
            f"required number of measures ({len(chunks)})"
        )

    # Add last entry to not skip the last measure

    # If num_notes, find the end measure based on the required number of notes
    if num_notes is not None:
        if num_notes >= total_notes - chunks.iloc[start_measure]["start"]:
            end_measure_included = len(chunks) - 1
        else:
            end_measure_included = chunks[
                chunks["end"] - chunks.loc[start_measure]["start"] >= num_notes
            ].index[0]

    # Keep only the performed notes in the queried measures
    # print()
    # print(start_measure, end_measure)
    clipped_notes = midi.tracks[0].notes[
        chunks.loc[start_measure]["start"] : chunks.loc[end_measure_included]["end"]
    ]
    midi.tracks[0].notes = clipped_notes

    # Clip the performance to the timing of the first and last notes
    start_time = clipped_notes[0].time
    end_time = max(note.time + note.duration for note in clipped_notes)
    midi.clip(start_time, end_time, inplace=True)

    # Adjust the timings so the performances starts at 0
    midi.shift_time(-start_time, inplace=True)

    if return_measure_range:
        return midi, (start_measure, end_measure_included + 1)
    else:
        return midi


def clip_score(
    score: pt.score.Part,
    measure_range: tuple[int, int | None],
    num_notes: int | None = None,
    return_measure_range: bool = False,
    check_num_measures: int | None = None,
) -> pt.score.Part | tuple[pt.score.Part, tuple[int, int]]:
    """
    Clip a score to a range of measures or an approximate number of notes.

    :param score: The score to clip.
    :param measure_range: The range of measures (start, end) to clip the score to. If the end
        measure is None, it will be computed based on num_notes.
    :param num_notes: Only used when the end measure is None, it will accumulate the required
        number of notes starting from the measure start, continuing until the end of the measure.
        Note that there might be more than the required number of notes (if there are still notes
        in the last measure), or less (if the remaining notes from the start measure is less than
        the required number of notes).
    :param return_measure_range: Whether to return the actual measure range used to clip the
        score (useful when num_notes is used).
    :param check_num_measures: Only used to check that the total number of measure in the score
        matches some externally computed number of measures.
    """
    # Check and process measure boundaries parameters
    start_measure_idx, end_measure_idx = measure_range
    if (end_measure_idx is None) == (num_notes is None):
        raise ValueError("Exactly one of end_measure and num_notes must be specified")
    current_measure = score.measures[start_measure_idx]
    start_point = current_measure.start

    # Compute end measure index if needed
    if num_notes is not None:
        end_measure_idx = start_measure_idx
        accumulated_notes = 0
        for element in score.iter_all(
            cls=(pt.score.Measure, pt.score.Note),
            include_subclasses=True,  # Include grace notes
            start=start_point.t,
        ):
            # print(element)
            if accumulated_notes >= num_notes:
                break
            elif isinstance(element, pt.score.Measure):
                end_measure_idx += 1
            elif element.tie_prev is None:  # skip tied notes
                accumulated_notes += 1
            # print(f"{end_measure_idx=}, {accumulated_notes=}")

    # Check number of measures if needed
    if check_num_measures is not None and check_num_measures != len(score.measures):
        raise ValueError(
            f"Number of measures in the score ({check_num_measures}) does not match the "
            f"required number of measures ({len(score.measures)})"
        )

    # Create a new ScoreVariant
    sv = pt.score.ScoreVariant(score)
    sv.add_segment(current_measure.start, score.measures[end_measure_idx - 1].end)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clipped_score = sv.create_variant_part()
    clipped_score.set_quarter_duration(0, score.quarter_duration_map(current_measure.start.t))

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
                # Also track them to add a rest there too if needed
                num_notes_per_voice_and_start[(next_tie.voice, next_tie.start.t)] += 1
    # Go over notes to remove, remove them and if needed replace by rests
    for note_id, original_note in notes_to_remove.items():
        # Remove the note from the clipped score
        if note_id not in id_to_clipped_score_note:
            logger.warning(f"{note_id} absent so can't be removed from score")
            continue
        note_to_remove = id_to_clipped_score_note[note_id]
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

    # Finally, add back global context elements (key/time signature and clefs)
    for context_cls in [pt.score.KeySignature, pt.score.TimeSignature]:
        if not start_point.starting_objects.get(context_cls, None):
            try:
                prev_context_element = next(start_point.iter_prev(cls=context_cls, eq=False))
            except StopIteration:
                continue
            # print(f"Adding {prev_context_element}")
            clipped_score.add(copy.copy(prev_context_element), start=0)

    # Specific behaviour for clefs as there are tied to staves
    clefs_found = [False] * clipped_score.number_of_staves
    if start_point.starting_objects.get(pt.score.Clef, None) is not None:
        for clef in start_point.starting_objects[pt.score.Clef]:
            clefs_found[clef.staff - 1] = True
            clipped_score.add(copy.copy(clef), start=start_point.t)
    clef_iterator = start_point.iter_prev(cls=pt.score.Clef, eq=False)
    while not all(clefs_found):
        try:
            clef = next(clef_iterator)
        except StopIteration:
            break
        if not clefs_found[clef.staff - 1]:
            clefs_found[clef.staff - 1] = True
            clipped_score.add(copy.copy(clef), start=0)

    if return_measure_range:
        if end_measure_idx is None:
            end_measure_idx = len(score.measures)
        return clipped_score, (start_measure_idx, end_measure_idx)
    else:
        return clipped_score


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
    # asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
    asap_path = Path(
        "~/Documents/transcription/MIDI2ScoreTransformer/midi2scoretransformer/data/asap-dataset"
    ).expanduser()

    # piece_path = asap_path / "Bach/Fugue/bwv_846"
    # performance_path = piece_path / "Shi05M.mid"
    piece_path = asap_path / "Beethoven/Piano_Sonatas/13-4"
    performance_path = piece_path / "Taverna01.mid"
    score_path = piece_path / "xml_score.musicxml"

    chunks_filename = f"{performance_path.stem}_chunks.json"
    annotations_path = asap_path / "asap_annotations.json"

    score = pt.load_musicxml(score_path, force_note_ids=True)
    assert len(score.parts) == 1
    score = pt.score.unfold_part_maximal(score.parts[0])
    # fill_rests(score)

    midi = symusic.Score.from_file(performance_path)

    chunks = pd.read_json(piece_path / chunks_filename)
    annotations = pd.read_json(annotations_path, orient="index")
    performance_annotations = annotations.loc[str(performance_path.relative_to(asap_path))]

    tokens = partitura_to_token_sequence(score)

    m_start = 19
    m_end = None
    n_notes = 20
    output_path = Path("/tmp")

    clipped_score, (m_start, m_end) = clip_score(
        score=score,
        measure_range=(m_start, m_end),
        num_notes=n_notes,
        return_measure_range=True,
    )
    print(f"Clipped from {m_start} to {m_end}")

    clipped_midi = clip_midi(
        midi=midi,
        chunks=chunks,
        performance_annotations=performance_annotations,
        measure_range=(m_start, m_end),
        check_num_measures=len(score.measures),
        inplace=True,
    )

    clipped_tokens = clip_token_sequence(
        tokens=tokens, measure_range=(m_start, m_end), check_num_measures=len(score.measures)
    )
    # automata = ConstrainedAutomata(num_voices=8)
    # automata.transition_sequence(clipped_tokens)

    pt.save_musicxml(clipped_score, output_path / "clipped_score.musicxml")
    clipped_midi.dump_midi(output_path / "clipped_performance.mid")
    pt.save_musicxml(
        token_sequence_to_partitura(clipped_tokens),
        "/tmp/clipped_score_tokens.musicxml",
    )
