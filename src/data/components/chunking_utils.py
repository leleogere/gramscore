from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import partitura as pt
import partitura.performance
import partitura.score
import partitura.utils.music
from loguru import logger


@dataclass
class PerformanceAligner:
    score: pt.score.Part
    performance: pt.performance.PerformedPart
    alignment: list[dict[str, str]]

    def __post_init__(self):
        logger.trace(f"Creating PerformanceAligner for {self.performance.id}")
        self._sid_to_pid = {}
        self._pid_to_sid = {}
        self._sid_unmatched = []
        self._pid_unmatched = []
        for a in self.alignment:
            match a["label"]:
                case "match":
                    pid = a["performance_id"]
                    sid = a["score_id"]
                    self._sid_to_pid[sid] = pid
                    self._pid_to_sid[pid] = sid
                case "deletion":
                    self._sid_unmatched.append(a["score_id"])
                case "insertion":
                    self._pid_unmatched.append(a["performance_id"])
                case _:
                    raise ValueError(f"Unknown label {a["label"]}")
        self._pid_to_pnote = {pnote["id"]: pnote for pnote in self.performance.notes}
        self._ptime_to_stime, self._stime_to_ptime = (
            pt.musicanalysis.performance_codec.get_time_maps_from_alignment(
                self.performance,
                self.score,
                self.alignment,
            )
        )
        self._inv_beat_map = self.score.inv_beat_map
        self._measure_number_map = self.score.measure_number_map

    def sid_to_pid(self, score_note_id: str, default=None) -> str | None:
        return self._sid_to_pid.get(score_note_id, default)

    def pid_to_sid(self, performance_note_id: str, default=None) -> str | None:
        return self._pid_to_sid.get(performance_note_id, default)

    def sid_unmatched(self) -> list[str]:
        return self._sid_unmatched

    def pid_unmatched(self) -> list[str]:
        return self._pid_unmatched

    def pid_to_pnote(
        self,
        performance_note_id: str,
        default=None,
    ) -> pt.performance.PerformedNote | None:
        return self._pid_to_pnote.get(performance_note_id, default)

    def sid_to_pnote(self, score_note_id: str, default=None) -> pt.performance.PerformedNote | None:
        pid = self.sid_to_pid(score_note_id)
        if pid is not None:
            return self.pid_to_pnote(pid, default=default)
        return default

    def ptime_to_stime(
        self,
        performance_time: float | np.ndarray[float],
        clip_at_zero: bool = True,
    ) -> float | np.ndarray[float]:
        stime = self._ptime_to_stime(performance_time)
        if clip_at_zero:
            stime = np.maximum(0, stime)
        return stime

    def stime_to_ptime(
        self,
        score_time: float | np.ndarray[float],
        clip_at_zero: bool = True,
    ) -> float | np.ndarray[float]:
        ptime = self._stime_to_ptime(score_time)
        if clip_at_zero:
            ptime = np.maximum(0, ptime)
        return ptime

    def ptime_to_measure_number(
        self,
        performance_time: float | np.ndarray[float],
    ) -> int | np.ndarray[int]:
        return self._measure_number_map(
            self._inv_beat_map(self.ptime_to_stime(performance_time, clip_at_zero=True))
        )

    @property
    def num_matched_notes(self) -> int:
        return len(self._sid_to_pid)

    @property
    def num_unmatched_score_notes(self) -> int:
        return len(self._sid_unmatched)

    @property
    def num_unmatched_performance_notes(self) -> int:
        return len(self._pid_unmatched)

    @property
    def ratio_matched_notes(self) -> float:
        return self.num_matched_notes / len(self.alignment)


def chunk_partitura_score_and_performances(
    score: pt.score.Part,
    aligned_performances: list[tuple[pt.performance.PerformedPart, list[dict[str, str]]]],
    num_measures_per_chunk: int,
    overlap: bool = True,
    min_matched_notes_ratio: float = 0.8,
    score_name_for_logging: str = None,
) -> list[dict[str, Any]]:
    """
    Chunk a partitura score and performances into chunks.

    :param score: The partitura score to chunk (as Partitura Part).
    :param aligned_performances: A list of tuples containing the aligned performances and their alignments (as Partitura PerformedPart and partitura alignment format).
    :param num_measures_per_chunk: The number of measures per chunk.
    :param overlap: If True, the chunks will overlap using a sliding window.
    :param min_matched_notes_ratio: The minimum ratio of matched notes to keep a performance.
    :param score_name_for_logging: The name of the score to use for logging.

    :return: A list of dictionaries containing the new chunks with the score and performances, with the following schema:
    ```
        [
            {
                "score": pt.score.Part,  # The current chunk's score
                "performances": [        # The performances for the current chunk
                    {
                        "performance": pt.performance.PerformedPart,  # The performance chunk
                        "performance_start": float,                   # The start time of the performance chunk (used for offsets)
                        "alignment": list[dict[str, str]],            # The alignment of the performance (as partitura alignment format)
                    }
                    for each performance
                ]
                "score_start": float,    # The start time of the score chunk (used for offsets)
                "first_measure_number": int,  # First measure in the chunk
                "last_measure_number": int,   # Last measure in the chunk (included)
            }
            for each chunk
        ]
    ```
    """
    if score_name_for_logging is None:
        score_name_for_logging = ""
    else:
        score_name_for_logging = f"{score_name_for_logging} - "

    num_measures = len(score.measures)

    if num_measures_per_chunk > num_measures:
        logger.warning(
            score_name_for_logging
            + "The number of measures per chunk must is greater than the total number of measures."
            "The resulting score will be shorter."
        )

    # All performances must have different IDs
    ids = [p.id for p, _ in aligned_performances]
    assert len(set(ids)) == len(ids), (
        score_name_for_logging + "All performances must have a distinct ID"
    )

    # Get the aligner tool for each performance
    aligners = [
        PerformanceAligner(score, performance, alignment)
        for performance, alignment in aligned_performances
    ]

    # Filter for badly aligned performances
    drop_indices = [
        i
        for i, aligner in enumerate(aligners)
        if aligner.ratio_matched_notes < min_matched_notes_ratio
    ]
    if drop_indices:
        logger.warning(
            score_name_for_logging
            + f"Dropping {len(drop_indices)} performances with a note matched ratio below "
            f"{min_matched_notes_ratio}."
        )
        aligned_performances = [
            p for i, p in enumerate(aligned_performances) if i not in drop_indices
        ]
        aligners = [a for i, a in enumerate(aligners) if i not in drop_indices]
        if not aligned_performances:
            logger.warning(
                score_name_for_logging + "All performances were dropped. Aborting chunking."
            )
            return []

    chunks = []

    if overlap:
        step = 1
    else:
        step = num_measures_per_chunk
        if num_measures % num_measures_per_chunk != 0:
            logger.warning(
                score_name_for_logging
                + f"Last chunk will only have {num_measures % num_measures_per_chunk} measures."
            )

    # Create all chunks
    for start_measure_idx in range(0, num_measures - num_measures_per_chunk + 1, step):
        # Get the start and end measures of the chunk
        end_measure_idx = min(start_measure_idx + num_measures_per_chunk, num_measures)
        start_measure = score.measures[start_measure_idx]
        end_measure = score.measures[end_measure_idx - 1]
        # print(f"Chunk {start_measure_idx}-{end_measure_idx}")

        # Create a new ScoreVariant
        sv = pt.score.ScoreVariant(score)
        sv.add_segment(start_measure.start, end_measure.end)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_score_chunk = sv.create_variant_part()

        # Fix the quarter lengths of the new chunk (see PR #385 of partitura)
        new_score_chunk.set_quarter_duration(0, score.quarter_duration_map(start_measure.start.t))

        # Get the matched notes of all performances
        perf_chunks = []
        for (performance, _), aligner in zip(aligned_performances, aligners):
            # Get the notes
            pnotes: list[pt.performance.PerformedNote] = []
            chunk_sids = [snote.id for snote in new_score_chunk.notes]
            for sid in chunk_sids:
                pnote = aligner.sid_to_pnote(sid)
                if pnote is not None:  # Skip unmatched notes, will be added back later
                    pnotes.append(pnote.copy())  # Shallow copy should be ok?

            if not pnotes:
                logger.warning(
                    score_name_for_logging
                    + f"No notes matched for performance {performance.id} on measures "
                    f"{start_measure_idx}-{end_measure_idx}."
                )
                continue

            # Fix the note timings to start at 0
            p_start = min(pnote["note_on"] for pnote in pnotes)
            p_end = max(pnote["note_on"] for pnote in pnotes)  # note_off? (used for controls)
            for pnote in pnotes:
                pnote["note_on"] -= p_start
                pnote["note_off"] -= p_start

            # Get the controls
            controls = [
                control.copy()
                for control in performance.controls
                if p_start <= control["time"] <= p_end
            ]  # Shallow copy ok
            for control in controls:
                control["time"] -= p_start

            perf_chunk = {
                "performance": pt.performance.PerformedPart(
                    notes=pnotes,
                    controls=controls,
                    id=performance.id,
                    ppq=performance.ppq,
                    mpq=performance.mpq,
                ),
                "performance_start": p_start,
                "alignment": [
                    a for a in aligner.alignment if "score_id" in a and a["score_id"] in chunk_sids
                ],
            }
            if len(perf_chunk["alignment"]) == 0:
                logger.warning(
                    score_name_for_logging
                    + f"Empty alignment for performance {performance.id} on measures "
                    f"{start_measure_idx}-{end_measure_idx}."
                )
            perf_chunks.append(perf_chunk)

        # Store the new chunk
        chunks.append(
            {
                "score": new_score_chunk,
                "performances": perf_chunks,
                "score_start": start_measure.start.t,
                "first_measure_number": start_measure.number,
                "last_measure_number": end_measure.number,
            }
        )

    # Add eventual missing score contextual information on all chunks (clefs, key & time signatures)
    for chunk in chunks:
        # For each contextual information, get it from the context, check against the first measure
        # of the chunk and add it if missing, and then add it to the chunk if needed.
        for contextual_getter, contextual_adder in [
            (
                get_time_signature_map_object_wrapper(score.time_signature_map),
                add_time_signature_if_missing,
            ),
            (
                get_key_signature_map_object_wrapper(score.key_signature_map),
                add_key_signature_if_missing,
            ),
            (get_clef_map_object_wrapper(score.clef_map), add_clefs_if_missing),
        ]:
            contextual_adder(chunk["score"], contextual_getter(chunk["score_start"]))

    # Add non-matched notes back to the performance chunks
    for p_idx, ((performance, _), aligner) in enumerate(zip(aligned_performances, aligners)):
        for pnote_id in aligner.pid_unmatched():
            # Get the note and the measure it should belong to
            pnote = aligner.pid_to_pnote(pnote_id)
            pnote_measure = aligner.ptime_to_measure_number(pnote["note_on"])
            # Get all chunks related to this measure
            related_chunk_idxs = list(
                range(
                    max(0, pnote_measure - num_measures_per_chunk),
                    min(pnote_measure, num_measures - num_measures_per_chunk + 1),
                )
                if overlap
                else (
                    [pnote_measure // num_measures_per_chunk]
                    if 1 <= pnote_measure <= num_measures
                    else []
                )
            )
            # Add the note to all related chunks
            for i, chunk in enumerate([chunks[chunk_idx] for chunk_idx in related_chunk_idxs]):
                current_perf_chunk = chunk["performances"][p_idx]
                new_pnote = pnote.copy()

                # Check note end for note entirely before the chunk
                new_note_off = new_pnote["note_off"] - current_perf_chunk["performance_start"]
                if new_pnote["note_off"] < current_perf_chunk["performance_start"]:
                    logger.warning(
                        score_name_for_logging
                        + f"Note {new_pnote["id"]} seems to end before chunk {i} "
                        f"({new_pnote["note_off"]} < {current_perf_chunk["performance_start"]}). "
                        f"Skipping note."
                    )
                    continue

                # Check note start for note starting before the chunk
                new_note_on = new_pnote["note_on"] - current_perf_chunk["performance_start"]
                if new_note_on < 0:
                    logger.warning(
                        score_name_for_logging
                        + f"Note {new_pnote["id"]} seems to start before chunk {i} "
                        f"({new_pnote["note_on"]} < {current_perf_chunk["performance_start"]}). "
                        f"Clipping start to 0."
                    )
                    new_pnote["note_on"] = 0
                else:
                    new_pnote["note_on"] = new_note_on

                new_pnote["note_off"] = new_note_off

                # Add the note
                current_perf_chunk["performance"].notes.append(new_pnote)
                current_perf_chunk["alignment"].append(
                    {"label": "insertion", "performance_id": new_pnote["id"]}
                )

    return chunks


def add_object_at_start_if_missing(
    part: pt.score.Part,
    obj: pt.score.TimedObject,
    equality_fn: Callable[[pt.score.TimedObject, pt.score.TimedObject], bool] = lambda x, y: x == y,
    filter_fn: Callable[[pt.score.TimedObject], bool] = lambda x: True,
):
    """
    Add an object at the beginning of a part if it is missing.

    :param part: The part to add the time signature to.
    :param obj: The object to add.
    :param equality_fn: Function that returns True if two objects are equal.
    :param filter_fn: Usefull to further filter objects to compare (not only by type).
    """
    obj_type = type(obj)
    existing_objs: list[pt.score.TimedObject] = [
        o for o in part.iter_all(obj_type, start=0, end=1) if filter_fn(o)
    ]
    assert (
        len(existing_objs) <= 1
    ), f"There should be at most one existing object after filtering, found {existing_objs}. Update the filter_fn to further restrict the objects to compare."

    # if obj_type == pt.score.Clef:
    #     print(f"{[m.number for m in part.measures] = }")
    #     print(f"{str(obj) = }")
    #     for existing_obj in existing_objs:
    #         print(f"{str(existing_obj) = }")
    #         print(f"{equality_fn(existing_obj, obj) = }")

    if len(existing_objs) == 0:
        # Add the object as it is missing
        part.add(obj, start=0)
    else:
        existing_obj = existing_objs[0]
        if equality_fn(existing_obj, obj):
            # Do nothing as the (same) object is already there
            pass
        else:
            raise ValueError(
                f"An object of type {obj_type} already exists and is different: {str(existing_obj)}."
            )


def add_time_signature_if_missing(part: pt.score.Part, ts: pt.score.TimeSignature):
    """
    Add a time signature at the beginning of a part if it is missing.

    :param part: The part to add the time signature to.
    :param ts: The time signature to add.
    """
    add_object_at_start_if_missing(
        part=part,
        obj=ts,
        equality_fn=lambda ts1, ts2: ts1.beats == ts2.beats and ts1.beat_type == ts2.beat_type,
    )


def add_key_signature_if_missing(part: pt.score.Part, ks: pt.score.KeySignature):
    """
    Add a key signature at the beginning of a part if it is missing.

    :param part: The part to add the key signature to.
    :param ks: The key signature to add.
    """
    add_object_at_start_if_missing(
        part=part,
        obj=ks,
        equality_fn=lambda ks1, ks2: ks1.fifths == ks2.fifths,
    )


def add_clefs_if_missing(part: pt.score.Part, clefs: list[pt.score.Clef]):
    """
    Add clefs at the beginning of a part if they are missing.

    :param part: The part to add the clefs to.
    :param clefs: The clefs to add.
    """
    for clef in clefs:
        add_object_at_start_if_missing(
            part=part,
            obj=clef,
            equality_fn=lambda c1, c2: (
                (c1.sign == c2.sign)
                and (c1.line == c2.line)
                and ((c1.octave_change or 0) == (c2.octave_change or 0))  # As None ~ 0
            ),
            filter_fn=lambda c: c.staff == clef.staff,
        )


def get_time_signature_map_object_wrapper(
    time_signature_map: Callable[[int], np.ndarray],
) -> Callable[[int], pt.score.TimeSignature]:
    """
    This function wrap the partitura `time_signature_map` to that it returns a
    `pt.score.TimeSignature` object instead of a numpy array.
    """

    def wrapper(t: int) -> pt.score.TimeSignature:
        ts_np = time_signature_map(t)
        return pt.score.TimeSignature(beats=int(ts_np[0]), beat_type=int(ts_np[1]))

    return wrapper


def get_key_signature_map_object_wrapper(
    key_signature_map: Callable[[int], np.ndarray],
) -> Callable[[int], pt.score.KeySignature]:
    """
    This function wrap the partitura `key_signature_map` to that it returns a
    `pt.score.KeySignature` object instead of a numpy array.
    """

    def wrapper(t: int) -> pt.score.KeySignature:
        ks_np = key_signature_map(t)
        # noinspection PyTypeChecker
        return pt.score.KeySignature(
            fifths=int(ks_np[0]),
            mode=partitura.utils.music.key_int_to_mode(int(ks_np[1])),  # bad partitura typing
        )

    return wrapper


def get_clef_map_object_wrapper(
    clef_map: Callable[[int], np.ndarray],
) -> Callable[[int], list[pt.score.Clef]]:
    """
    This function wrap the partitura `clef_map` to that it returns a
    list of `pt.score.Clef` objects instead of a numpy array.
    Note: this function need the PR #384 of partitura to work (implement clef_map).
    """

    def wrapper(t: int) -> list[pt.score.Clef]:
        clefs_np = clef_map(t)
        return [
            pt.score.Clef(
                staff=int(staff),
                sign=partitura.utils.music.clef_int_to_sign(int(sign_int)),
                line=int(line),
                octave_change=int(octave_change),
            )
            for staff, sign_int, line, octave_change in clefs_np
        ]

    return wrapper


if __name__ == '__main__':
    n = 8

    path = Path(
        # "/home/gerel/Documents/datasets/nasap-dataset/Beethoven/Piano_Sonatas/29-4/xml_score.musicxml"
        # "/home/gerel/Documents/datasets/nasap-dataset/Beethoven/Piano_Sonatas/7-3/xml_score.musicxml"
        # "/home/gerel/Documents/datasets/nasap-dataset/Bach/Fugue/bwv_888/xml_score.musicxml"
        "/home/gerel/Documents/datasets/nasap-dataset/Schubert/Wanderer_fantasie/xml_score.musicxml"
        # "/home/gerel/Documents/datasets/nasap-dataset/Chopin/Etudes_op_10/12/xml_score.musicxml"
    )
    # print("READING SCORE")
    score = pt.score.unfold_part_maximal(pt.load_musicxml(path).parts[0], ignore_leaps=False)
    # print("SCORE READ")

    performances_and_alignments = []
    for file in path.parent.glob("*.match"):
        performance, alignment = pt.load_match(file)
        performance = performance.performedparts[0]
        performance.id = file.stem
        performances_and_alignments.append((performance, alignment))

    chunks = chunk_partitura_score_and_performances(score, performances_and_alignments, n)

    tmp_dir = (
        Path("/tmp/unary_tests")
        / path.relative_to("/home/gerel/Documents/datasets/nasap-dataset").parent
    )
    for i, chunk in enumerate(chunks):
        m_start = i + 1
        m_end = i + n
        chunk_dir = tmp_dir / f"chunk_{m_start}-{m_end}"
        chunk_dir.mkdir(exist_ok=True, parents=True)
        print(f"Chunk {i}")
        pt.save_musicxml(chunk["score"], chunk_dir / path.name)
        for performance in chunk["performances"]:
            pt.save_performance_midi(
                performance["performance"],
                chunk_dir / f"{performance["performance"].id}.mid",
            )


@dataclass
class ChunkingConfig:
    """
    Config for chunking a dataset.

    :param new_dataset_path: The path to save the new dataset to.
    :param num_measures_per_chunk: The number of measures per chunk.
    :param overlap: Whether to overlap the chunks (use a sliding window).
    :param min_matched_notes_ratio: The minimum ratio of matched notes between the score and the performance to consider the alignment valid.
    :param skip_already_chunked: Whether to skip scores that have already been chunked.
    :param progress: Whether to show a progress bar.
    :param n_jobs: The number of jobs to run in parallel.
    """

    new_dataset_path: Path
    num_measures_per_chunk: int
    overlap: bool = True
    min_matched_notes_ratio: float = 0.8
    skip_already_chunked: bool = False
    n_jobs: int = 1
    progress: bool = True
