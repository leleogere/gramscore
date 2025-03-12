# def make_measures(midi, midi_score, mxl, annots, swap=True):
#     measures = {"midi": [], "mxl": []}
#     # TODO: Maybe do beats instead of downbeats?
#     swap_tol = 0.05
#     measures = {"midi": [], "mxl": []}
#     i, j = (0, 0)
#     for sb_seconds in annots["midi_score_beats"]:
#         # Convert the score beats from seconds to musical time using
#         # the provided MIDI scores
#         sb = midi_score.time_to_tick(sb_seconds) / midi_score.resolution
#         measures["mxl"].append([])
#         while j < len(mxl) and mxl[j].offset < sb:
#             measures["mxl"][-1].append((j, mxl[j].pitch.midi))
#             j += 1
#
#     # Use the performance beats to split the MIDI into pseudo-measures of beat length
#     for pb in annots["performance_beats"]:
#         measures["midi"].append([])
#         while i < len(midi) and midi[i].start < pb - swap_tol:
#             measures["midi"][-1].append((i, midi[i].pitch))
#             i += 1
#
#     measures["midi"].append([])
#     measures["mxl"].append([])
#     while i < len(midi):
#         measures["midi"][-1].append((i, midi[i].pitch))
#         i += 1
#     while j < len(mxl):
#         measures["mxl"][-1].append((j, mxl[j].pitch.midi))
#         j += 1
#
#     if not swap:
#         for i in range(len(measures["midi"])):
#             measures["midi"][i] = [i_ for i_, p in measures["midi"][i]]
#             measures["mxl"][i] = [i_ for i_, p in measures["mxl"][i]]
#         measures["swapped"] = False
#         return measures
#     from collections import Counter
#
#     n_swaps = 0
#     swap_tol = 0.5
#     for i in range(len(measures["midi"]) - 1):
#         swapped = True
#         while swapped:
#             swapped = False
#             # Figure out which MIDI pitches can be moved forward/backward by one beat
#             # because that would yield better alignment. Only considers notes within
#             # 0.5s of the beat boundary.
#             c_midi = Counter([p for _, p in measures["midi"][i]])
#             c_mxl = Counter([p for _, p in measures["mxl"][i]])
#
#             c_midi_next = Counter([p for _, p in measures["midi"][i + 1]])
#             c_mxl_next = Counter([p for _, p in measures["mxl"][i + 1]])
#             too_much = c_midi - c_mxl
#             lacking = c_mxl - c_midi
#             too_much_next = c_midi_next - c_mxl_next
#             lacking_next = c_mxl_next - c_midi_next
#             can_be_moved_forward = too_much & lacking_next
#             can_be_moved_backward = too_much_next & lacking
#             for pitch in can_be_moved_forward:
#                 last_j, last_p = measures["midi"][i][-1]
#                 # Only swap notes if they are within 0.5s of the beat boundary
#                 if (
#                     last_p == pitch
#                     and annots["performance_beats"][i] - midi[last_j].start < swap_tol
#                 ):
#                     measures["midi"][i + 1].insert(0, measures["midi"][i].pop())
#                     n_swaps += 1
#                     swapped = True
#                     break
#
#             for pitch in can_be_moved_backward.keys():
#                 first_j, first_p = measures["midi"][i + 1][0]
#                 # ---only swap notes if they are within 0.5s of the beat boundary
#                 if (
#                     first_p == pitch
#                     and midi[first_j].start - annots["performance_beats"][i] < swap_tol
#                 ):
#                     measures["midi"][i].append(measures["midi"][i + 1].pop(0))
#                     n_swaps += 1
#                     swapped = True
#                     break
#
#     # print("Swaps", n_swaps, n_swaps/len(midi))
#     # If we had to swap a lot of notes, we probably messed up and should just return
#     # the raw alignment...
#     if n_swaps / len(midi) > 0.1:
#         return make_measures(midi, midi_score, mxl, annots, swap=False)
#     for i in range(len(measures["midi"])):
#         measures["midi"][i] = sorted([j for j, p in measures["midi"][i]])
#         measures["mxl"][i] = sorted([j for j, p in measures["mxl"][i]])
#     assert max([max(m + [0]) for m in measures["midi"]]) == len(midi) - 1
#     assert sum(len(m) for m in measures["midi"]) == len(midi)
#     assert max([max(m + [0]) for m in measures["mxl"]]) == len(mxl) - 1
#     assert sum(len(m) for m in measures["mxl"]) == len(mxl)
#     measures["swapped"] = True
#     return measures
import copy
import warnings
from pathlib import Path
from pprint import pprint

import pandas as pd
import partitura as pt
import partitura.score
import symusic


# The above code is only used as reference, the code below is not directly adapted from it
# but might take some inspiration from it


def clip_midi(
    midi: symusic.Score,
    chunks: pd.DataFrame,
    performance_annotations: pd.DataFrame,
    measure_range: tuple[int, int],
    inplace: bool = True,
    return_measure_range: bool = False,
    _check_num_measures: bool = None,
) -> symusic.Score:
    if not inplace:
        midi = midi.copy(deep=True)

    start_measure, end_measure = measure_range

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
    chunks["midi"] = (
        chunks["midi"].apply(lambda x: int(x[0]) if x else pd.NA).astype(pd.Int64Dtype()).ffill()
    )
    if _check_num_measures is not None:
        assert (
            len(chunks) == _check_num_measures
        ), f"Number of measures does not match ({len(chunks)} != {_check_num_measures})"

    # Add last entry to not skip the last measure
    chunks.loc[len(chunks)] = [len(midi.tracks[0].notes)]

    # Keep only the performed notes in the queried measures
    # print()
    # print(start_measure, end_measure)
    clipped_notes = midi.tracks[0].notes[
        chunks.iloc[start_measure]["midi"] : chunks.iloc[end_measure]["midi"]
    ]
    midi.tracks[0].notes = clipped_notes

    # Clip the performance to the timing of the first and last notes
    start_time = clipped_notes[0].time
    end_time = max(note.time + note.duration for note in clipped_notes)
    midi.clip(start_time, end_time, inplace=True)

    # Adjust the timings so the performances starts at 0
    midi.adjust_time(
        original_times=[start_time, end_time],
        new_times=[0, end_time - start_time],
        inplace=True,
    )

    return midi


def clip_score(
    score: pt.score.Part,
    measure_range: tuple[int | None, int | None],
    num_notes: int | None = None,
) -> tuple[pt.score.Part, tuple[int, int]]:
    start_measure_idx, end_measure_idx = measure_range
    if start_measure_idx is None:
        start_measure_idx = 0
    if end_measure_idx is None and num_notes is None:
        end_measure_idx = len(score.measures)
    elif end_measure_idx is not None and num_notes is not None:
        raise ValueError("Cannot specify both end_measure and num_notes")

    current_measure = score.measures[start_measure_idx]
    start_point = current_measure.start
    end_point = score.measures[end_measure_idx - 1].end if end_measure_idx is not None else None
    clipped_score = pt.score.Part(id=score.id, quarter_duration=score.quarter_duration_map(0))

    print(f"{start_point.t = }")
    print(f"{end_point = }")

    # Add all elements from the start measure to the end measure
    accumulated_notes = 0
    tied_notes = {}
    for element in score.iter_all(
        start=start_point.t,
        end=end_point.t if end_point is not None else None,
    ):
        print(accumulated_notes, element)
        if isinstance(element, pt.score.Measure):
            current_measure = element

        # Accumulate notes if needed
        if num_notes is not None:
            if isinstance(element, pt.score.Note):
                accumulated_notes += 1
            # Break at the end of measure if enough notes have been accumulated
            if accumulated_notes >= num_notes and element.start.t >= current_measure.end.t:
                end_measure_idx = current_measure.number
                # Remove unresolved ties
                pprint(tied_notes)
                for note in tied_notes.values():
                    note.tie_next = None
                break

        # Copy element as start/end will be modified
        new_element = copy.copy(element)

        # Deal with ties crossing the chunk boundaries
        if isinstance(element, pt.score.Note):
            # Notes tied to a previous note
            if element.tie_prev:
                # Replace notes tied to previous measure by rests
                if element.tie_prev_notes[0].start.t < start_point.t:
                    new_element = pt.score.Rest(
                        voice=element.voice,
                        staff=element.staff,
                        symbolic_duration=element.symbolic_duration,
                    )
                # Discard it from the tied notes list (only used with num_notes)
                tied_notes.pop(element.tie_prev)

            # Notes tied to a future note
            if element.tie_next:
                # If end measure specified, easy check: remove if next note is after end
                if end_point is not None and element.tie_next.start.t > end_point.t:
                    new_element.tie_next = None
                # Else, we need to aggregate the tied notes to check later if they are in the chunk
                # (only used with num_notes)
                else:
                    tied_notes[element] = new_element

        # Add the element to the score
        clipped_score.add(
            new_element,
            start=element.start.t - start_point.t if element.start is not None else None,
            end=element.end.t - start_point.t if element.end is not None else None,
        )
        print(tied_notes)

    # Add all context elements
    for context_cls in [pt.score.KeySignature, pt.score.TimeSignature]:
        if not start_point.starting_objects.get(context_cls, None):
            try:
                prev_context_element = next(start_point.iter_prev(cls=context_cls, eq=False))
            except StopIteration:
                continue
            print(f"Adding {prev_context_element}")
            clipped_score.add(prev_context_element, start=0)

    # Specific behaviour for clefs as there are tied to staves
    clefs_found = [False] * clipped_score.number_of_staves
    if start_point.starting_objects.get(pt.score.Clef, None) is not None:
        for clef in start_point.starting_objects[pt.score.Clef]:
            clefs_found[clef.staff - 1] = True
            clipped_score.add(clef, start=start_point.t)
    clef_iterator = start_point.iter_prev(cls=pt.score.Clef, eq=False)
    while not all(clefs_found):
        try:
            clef = next(clef_iterator)
        except StopIteration:
            break
        if not clefs_found[clef.staff - 1]:
            clefs_found[clef.staff - 1] = True
            clipped_score.add(clef, start=0)

    return clipped_score, (start_measure_idx, end_measure_idx)


def clip_score_and_performance(
    xml_score: Path | pt.score.Part,
    midi_performance: Path | symusic.factory.ScoreFactory,
    chunks: Path | pd.DataFrame,
    annotations: Path | pd.DataFrame,
    measure_range: tuple[int | None, int | None],
    num_notes: int | None = None,
) -> tuple[pt.score.Part, symusic.Score]:
    """
    This function clips an aligned score and performance to a range of measures.

    :param xml_score: The score to clip, either a path to a MusicXML file or a Part object.
    :param midi_performance: The performance to clip, either a path to a MIDI file or a Score
        object.
    :param chunks: The beat-level annotations used to align the score and performance.
    :param annotations: The performance annotations used to align the score and performance.
    :param measure_range: The range (start_included, end_excluded) of measures to clip the score
        and performance to. If start is None, starts at the beginning of the score. If end is None,
        ends at the end of the score.
    :param num_notes: In combination with end=None in measure_range, will accumulate the required
        number of notes, starting from the measure start, and automatically get the measure end
        based on that number of notes. Note that there might be more than the required number of
        notes (if there are still notes in the last measure), or less (if the remaining notes from
        the start measure is less than the required number of notes).
    """
    # Load paths if needed
    if isinstance(xml_score, Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xml_score = pt.load_musicxml(xml_score)
            assert len(xml_score.parts) == 1
            xml_score = pt.score.unfold_part_maximal(xml_score.parts[0])
    if isinstance(midi_performance, Path):
        midi_performance = symusic.Score(midi_performance)
        assert len(midi_performance.tracks) == 1
    if isinstance(chunks, Path):
        chunks = pd.read_json(chunks)
    if isinstance(annotations, Path):
        annotations = pd.read_json(annotations, orient="index")

    # Get some variables
    num_score_measures = len(xml_score.measures)

    # Process the score
    clipped_score, (start_measure, end_measure) = clip_score(
        score=xml_score,
        measure_range=measure_range,
        num_notes=num_notes,
    )
    print(f"{start_measure = }, {end_measure = }")

    # Process the performance
    clipped_midi = clip_midi(
        midi=midi_performance,
        chunks=chunks,
        performance_annotations=annotations.loc[
            str(piece_path.relative_to(asap_path) / performance_filename)
        ],  # TODO: avoid referencing globals
        start_measure=start_measure,
        end_measure=end_measure,
        inplace=True,
        _check_num_measures=num_score_measures,
    )

    return clipped_score, clipped_midi


if __name__ == '__main__':
    # asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
    asap_path = Path(
        "~/Documents/transcription/MIDI2ScoreTransformer/midi2scoretransformer/data/asap-dataset"
    ).expanduser()
    score_filename = "xml_score.musicxml"

    # piece_path = asap_path / "Bach/Fugue/bwv_846"
    # performance_filename = "Shi05M.mid"
    piece_path = asap_path / "Beethoven/Piano_Sonatas/13-4"
    performance_filename = "Taverna01.mid"

    chunks_filename = f"{performance_filename.split('.')[0]}_chunks.json"
    annotations_path = asap_path / "asap_annotations.json"

    clipped_score, clipped_perf = clip_score_and_performance(
        xml_score=piece_path / score_filename,
        midi_performance=piece_path / performance_filename,
        chunks=piece_path / chunks_filename,
        annotations=annotations_path,
        measure_range=(17, None),
        num_notes=30,
    )

    output_path = Path("/tmp")
    pt.save_musicxml(clipped_score, output_path / "clipped_score.musicxml")
    clipped_perf.dump_midi(output_path / "clipped_performance.mid")
