"""
Test the chunking process
"""

import copy
from importlib.resources import files

import pytest

import partitura as pt
import partitura.score
import itertools
import tests.resources

from pprint import pprint

from scripts.chunk_score_and_performance import clip_token_sequence, clip_score


@pytest.fixture
def score():
    path = files(tests.resources).joinpath("test_chunking.musicxml")
    score = pt.load_musicxml(path, force_note_ids=True)
    return score


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

        # Get the notes and rests of the chunked score
        test_notes = list(
            clipped_test.iter_all(
                cls=pt.score.GenericNote,
                include_subclasses=True,
                start=0,
                end=measure_end.end.t - measure_start.start.t,
            )
        )
        # Get the expected notes and rests
        expected_notes = list(
            expected.iter_all(
                cls=pt.score.GenericNote,
                include_subclasses=True,
                start=measure_start.start.t,
                end=measure_end.end.t,
            )
        )

        assert len(test_notes) == len(expected_notes)
        for n1, n2 in itertools.zip_longest(test_notes, expected_notes):
            assert_notes_equal(n1, n2, n2_time_shift=measure_start.start.t)


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
        [8, 100, 4],
        [0, 100, 12],
    ]:
        part = copy.deepcopy(part_to_chunk)
        clipped_part = clip_score(
            part,
            measure_range=(start_measure, None),
            num_notes=num_notes,
        )
        # print(clipped_part.pretty())
        assert (
            len(clipped_part.measures) == expected_num_measures
        ), f"Expected {expected_num_measures} measures, got {len(clipped_part.measures)}"
