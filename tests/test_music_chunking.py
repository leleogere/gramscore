"""
Test the chunking process
"""

# TODO: to fix after chunking refactoring

import copy
import itertools
from importlib.resources import files
from pathlib import Path

import pandas as pd
import partitura as pt
import partitura.score
import pytest
import symusic

import tests.resources
from data.music_chunking import clip_midi
from data.music_chunking import clip_score


def get_resource(file_name: str) -> Path:
    return Path(files(tests.resources).joinpath(file_name))


@pytest.fixture
def score():
    path = get_resource("test_chunking.musicxml")
    score = pt.load_musicxml(path, force_note_ids=True)
    return score


@pytest.fixture
def midi_and_chunks_and_annotations():
    midi_path = get_resource("Bach-Fugue-bwv_846-Shi05M.mid")
    # The file below is obtained with the chunker.py script of MIDI2ScoreTransformer
    chunks_path = get_resource("Bach-Fugue-bwv_846-Shi05M-chunks.json")
    # The file below is obtained from the ASAP annotations JSON file with the command:
    # jq '{sample: ."Bach/Fugue/bwv_846/Shi05M.mid"}' asap_annotations.json
    annotations_path = get_resource("Bach-Fugue-bwv_846-Shi05M-annotations.json")
    midi = symusic.Score.from_file(midi_path)
    chunks = pd.read_json(chunks_path)
    annotations = pd.read_json(annotations_path, orient="index").loc["sample"]
    return midi, chunks, annotations


def assert_notes_equal(
    n1: pt.score.GenericNote,
    n2: pt.score.GenericNote,
    check_tie_prev: bool = True,
    check_tie_next: bool = True,
    n2_time_shift: int = 0,
):
    # fmt: off
    assert type(n1) == type(n2)
    assert n1.start.t == n2.start.t - n2_time_shift, f"Different start times: {n1.start.t} != {n2.start.t - n2_time_shift}"
    assert n1.end.t == n2.end.t - n2_time_shift, f"Different end times: {n1.end.t} != {n2.end.t - n2_time_shift}"
    assert n1.duration == n2.duration, f"Different durations: {n1.duration} != {n2.duration}"
    assert n1.staff == n2.staff, f"Different staffs: {n1.staff} != {n2.staff}"
    assert n1.voice == n2.voice, f"Different voices: {n1.voice} != {n2.voice}"
    if isinstance(n1, pt.score.Note):
        assert n1.step == n2.step, f"Different steps: {n1.step} != {n2.step}"
        assert n1.alter == n2.alter, f"Different alters: {n1.alter} != {n2.alter}"
        assert n1.octave == n2.octave, f"Different octaves: {n1.octave} != {n2.octave}"
        assert n1.midi_pitch == n2.midi_pitch, f"Different MIDI pitches: {n1.midi_pitch} != {n2.midi_pitch}"
        assert type(n1.tie_prev) == type(n2.tie_prev), f"Different tie_prev types: {type(n1.tie_prev)} != {type(n2.tie_prev)}"
        assert type(n1.tie_next) == type(n2.tie_next), f"Different tie_next types: {type(n1.tie_next)} != {type(n2.tie_next)}"
        if check_tie_prev and n1.tie_prev is not None:
            assert_notes_equal(
                n1.tie_prev,
                n2.tie_prev,
                check_tie_next=False,  # Do not recheck the tie_next of the previous note
                n2_time_shift=n2_time_shift,
            )
        if check_tie_next and n1.tie_next is not None:
            assert_notes_equal(
                n1.tie_next,
                n2.tie_next,
                check_tie_prev=False,  # Do not recheck the tie_prev of the next note
                n2_time_shift=n2_time_shift,
            )
    # fmt: on


def test_clip_score_per_measure(score):
    """
    In this score, the first part is the actual score to chunk (into 2-measure chunks), and the two
    other parts are the expected results:
        M1-2 -> P2, M2-3 -> P3, M3-4 -> P2, M4-5 -> P3, M5-6 -> P2, M6-7 -> P3...
    Open the file in MuseScore to better understand, the two bottom parts are the interleaved
    expected results.
    """
    part_to_chunk = score.parts[0]  # Part containing the score to chunk
    expected_odd = score.parts[1]  # Part containing the expected results for odd measures (1-based)
    expected_even = score.parts[2]  # Part containing the expected results for even measures

    for measure_start_idx in range(len(part_to_chunk.measures) - 1):
        # print(f"Chunking measures {measure_start_idx + 1}-{measure_start_idx + 2}")
        measure_end_idx = measure_start_idx + 2

        part = copy.deepcopy(part_to_chunk)
        expected = expected_odd if measure_start_idx % 2 == 0 else expected_even  # inverted 0-based

        measure_start = part.measures[measure_start_idx]
        measure_end = part.measures[measure_end_idx - 1]  # -1 because exclusive

        # Chunk the score
        clipped_test = clip_score(
            part,
            measure_range=(measure_start_idx, measure_end_idx),
        )

        # Check the elements
        # Do not check KS bc difficulties to input them independently in the parts in MuseScore
        cls_to_check = [
            pt.score.GenericNote,
            pt.score.TimeSignature,
            pt.score.Clef,
            # pt.score.KeySignature,
        ]
        test_elements = list(
            clipped_test.iter_all(
                cls=cls_to_check,
                include_subclasses=True,
                start=0,
                end=measure_end.end.t - measure_start.start.t,
            )
        )
        expected_elements = list(
            expected.iter_all(
                cls=cls_to_check,
                include_subclasses=True,
                start=measure_start.start.t,
                end=measure_end.end.t,
            )
        )
        for el1, el2 in itertools.zip_longest(test_elements, expected_elements):
            # print("*", el1)
            # print(" ", el2)
            # Check same type
            assert type(el1) is type(el2), f"Different types: {type(el1)} != {type(el2)}"

            # Check timings
            assert (
                el1.start.t == el2.start.t - measure_start.start.t
            ), f"Different start times: {el1.start.t} != {el2.start.t - measure_start.start.t}"
            if el1.end is not None:
                assert (
                    el1.end.t == el2.end.t - measure_start.start.t
                ), f"Different end times: {el1.end.t} != {el2.end.t - measure_start.start.t}"
            else:
                assert el2.end is None, "Expected end to be None"

            # Check notes
            if isinstance(el1, pt.score.GenericNote):
                assert_notes_equal(el1, el2, n2_time_shift=measure_start.start.t)
            # Check non-notes elements
            else:
                if isinstance(el1, pt.score.TimeSignature):
                    attributes_to_check = ["beats", "beat_type"]
                elif isinstance(el1, pt.score.Clef):
                    attributes_to_check = ["sign", "line", "octave_change", "staff"]
                elif isinstance(el1, pt.score.KeySignature):
                    attributes_to_check = ["fifths"]
                else:
                    raise ValueError(f"Unexpected class: {type(el1)}")
                for attr in attributes_to_check:
                    assert getattr(el1, attr) == getattr(
                        el2, attr
                    ), f"Different {attr}: {getattr(el1, attr)} != {getattr(el2, attr)}"


def test_clip_score_per_num_notes(score):
    """
    Test if the chunking by number of notes works correctly.
    """
    part_to_chunk = score.parts[0]

    for start_measure, num_notes, expected_num_measures in [
        [0, 6, 2],
        [0, 7, 3],
        [2, 1, 1],
        [2, 2, 2],
        [2, 3, 2],
        [8, 6, 1],
        [8, 7, 2],
        [8, 11, 2],
        [8, 12, 3],
        [8, 100, 5],
        [0, 100, 13],
    ]:
        part = copy.deepcopy(part_to_chunk)
        clipped_part = clip_score(
            part,
            measure_range=(start_measure, None),
            num_notes=num_notes,
        )
        assert (
            len(clipped_part.measures) == expected_num_measures
        ), f"Expected {expected_num_measures} measures, got {len(clipped_part.measures)}"


def test_clip_midi(midi_and_chunks_and_annotations):
    midi, chunks, annotations = midi_and_chunks_and_annotations

    test_cases = [
        # Measures 14 to 16 (included)
        {
            "measure_range": (13, 16),
            "expected_num_notes": 72,
            "first_pitches": {69, 57, 45},  # Use sets as order in chords not guaranteed
            "last_pitches": {77},  # Same here
            "queried_num_notes": None,
        },
        # Measures 25 to 26 (included)
        {
            "measure_range": (24, 26),
            "expected_num_notes": 61,  # 60 in the score but extra C3 in the performance
            "first_pitches": {50, 70, 74},
            "last_pitches": {67},
            "queried_num_notes": None,
        },
        # Measure 25 with 33 notes (only 1 measure)
        {
            "measure_range": (24, None),
            "expected_num_notes": 33,
            "first_pitches": {50, 70, 74},
            "last_pitches": {55, 70, 74},
            "queried_num_notes": 33,
        },
        # Measure 25 with 34 notes: first note of second measure, so 2 measures (same as 2nd test)
        {
            "measure_range": (24, None),
            "expected_num_notes": 61,
            "first_pitches": {50, 70, 74},
            "last_pitches": {67},
            "queried_num_notes": 34,
        },
        # Measure 25 to the end
        {
            "measure_range": (24, None),
            "expected_num_notes": 75,
            "first_pitches": {50, 70, 74},
            "last_pitches": {76, 84},  # should be {76, 79, 84} but G5 missing in the performance
            "queried_num_notes": 1000,
        },
    ]

    for test_case in test_cases:
        extracted_midi = clip_midi(
            midi=midi,
            chunks=chunks.copy(deep=True),
            performance_annotations=annotations,
            measure_range=test_case["measure_range"],
            num_notes=test_case["queried_num_notes"],
            inplace=False,
        )
        # extracted_midi.dump_midi(
        #     f"/tmp/extracted_midi_{test_case['measure_range'][0]}-{test_case['measure_range'][1]}.mid"
        # )

        # Check if the number of notes is correct in a few chunks
        num_notes = len(extracted_midi.tracks[0].notes)
        assert (
            num_notes == test_case["expected_num_notes"]
        ), f"Expected {test_case['expected_num_notes']} notes, got {num_notes}"

        # Check first notes
        num_first_notes = len(test_case["first_pitches"])
        first_pitches = set(note.pitch for note in extracted_midi.tracks[0].notes[:num_first_notes])
        assert (
            first_pitches == test_case["first_pitches"]
        ), f"Expected {test_case['first_pitches']} as first pitches, got {first_pitches}"

        # Check last notes
        num_last_notes = len(test_case["last_pitches"])
        last_pitches = set(note.pitch for note in extracted_midi.tracks[0].notes[-num_last_notes:])
        assert (
            last_pitches == test_case["last_pitches"]
        ), f"Expected {test_case['last_pitches']} as last pitches, got {last_pitches}"
