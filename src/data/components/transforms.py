import random
import warnings
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any
from typing import Literal

import cachetools
import pandas as pd
import partitura as pt
import partitura.score
import partitura.utils
import symusic
import torch
import torch.utils.data
from loguru import logger
from torchdata import nodes

from data.components.score_performance_dataset import ScorePerformanceEntry
from data.music_chunking import clip_midi
from data.music_chunking import clip_score
from data.music_chunking import read_chunk_file
from midi2scoretransformer.tokenizer import MultistreamTokenizer
from models.automata.io import partitura_to_token_sequence
from models.automata.tokens import fill_rests
from models.automata.vectorization_torch import TokenTensor
from models.automata.voice_reordering import reorder_voices

MAX_PITCH = 108
MIN_PITCH = 21


def add_keys_to_result(
    fn: Callable[[dict[str, Any]], dict[str, Any]]
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Add the new keys to the input.
    """

    def wrapper(sample: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        # Get the new keys
        new_keys = fn(sample, *args, **kwargs)
        # Add the new keys to the input
        return {k: v for k, v in sample.items() if k not in new_keys} | new_keys

    return wrapper


def to_dict(entry: Any) -> dict[str, Any]:
    """Transforme une entrée en un dictionnaire avec une clé unique 'entry'."""
    return {"entry": entry}


# class AbstractProcessor:
#     def _process(self, sample: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
#         """
#         This method takes a dict sample, and returns only the newly added/modified keys.
#
#         This avoids modifying the sample inplace, which causes issues with the Cache object.
#         """
#         raise NotImplementedError("Abstract method")
#
#     def __call__(self, sample: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
#         processed = self._process(sample)
#         if isinstance(processed, list):
#             return [
#                 {k: v for k, v in sample.items() if k not in new_keys} | new_keys
#                 for new_keys in processed
#             ]
#         return {k: v for k, v in sample.items() if k not in processed} | processed


@add_keys_to_result
def read_score(sample: dict[str, Any]) -> dict[str, Any]:
    """Lit un fichier MusicXML et retourne le score."""
    worker = torch.utils.data.get_worker_info()
    if worker is not None:
        logger.debug(
            f"Worker {worker.id}/{worker.num_workers} reading score {sample['entry'].xml_score.as_posix()}"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pt.load_musicxml(sample["entry"].xml_score, force_note_ids=True)
        assert len(score.parts) == 1
        score = pt.score.unfold_part_maximal(score.parts[0])
    return {"score": score}


@add_keys_to_result
def read_performance(sample: dict[str, Any]) -> dict[str, Any]:
    """Lit un fichier MIDI et retourne la performance."""
    performance = symusic.Score.from_file(sample["entry"].midi_performance)
    if len(performance.tracks) > 1:
        track = performance.tracks[0]
        for other in performance.tracks[1:]:
            for attr in ["notes", "controls", "pitch_bends", "pedals", "lyrics"]:
                getattr(track, attr).extend(getattr(other, attr))
            performance.tracks.remove(other)
    assert len(performance.tracks) == 1
    return {"performance": performance}


@add_keys_to_result
def compute_allowed_transpositions(
    sample: dict[str, Any], intervals: list[pt.score.Interval]
) -> dict[str, Any]:
    # Try all intervals
    allowed_intervals = []
    for interval in intervals:
        try:
            # Try transposing the score
            pt.utils.music.transpose(
                sample["score"],
                interval,
                transpose_key_signatures=True,
                inplace=False,
            )
            allowed_intervals.append(interval)
            logger.trace(
                f"Score {sample['entry'].xml_score.as_posix()} can be transposed "
                f"{interval.direction} by a {interval.number}{interval.quality} interval."
            )
        except (AssertionError, ValueError) as e:
            # Not ideal but the easiest way to capture the correct exception
            if isinstance(e, AssertionError) and e.args[0].endswith("is not in the range -2 to 2."):
                logger.trace(
                    f"Can't transpose {sample['entry'].xml_score.as_posix()} "
                    f"{interval.direction} by a {interval.number}{interval.quality} interval "
                    f"(alterations > 2)."
                )
            elif isinstance(e, ValueError) and e.args[0].endswith(
                "The maximum number of fifths is +/- 7."
            ):
                logger.trace(
                    f"Can't transpose {sample['entry'].xml_score.as_posix()} "
                    f"{interval.direction} by a {interval.number}{interval.quality} interval "
                    f"(key fifths > 7)."
                )
            else:
                raise

    logger.debug(
        f"Allowed transpositions for {sample['entry'].xml_score.as_posix()}: "
        + ", ".join(
            f"{'-' if i.direction == 'down' else ''}{i.number}{i.quality}"
            for i in allowed_intervals
        )
    )

    return {"allowed_transpositions": allowed_intervals}


@add_keys_to_result
def random_transpose(
    sample: dict[str, Any], intervals: list[pt.score.Interval] | None = None
) -> dict[str, Any]:
    # See if some allowed intervals have been precomputed
    if "allowed_transpositions" in sample:
        intervals = sample["allowed_transpositions"]
    else:
        intervals = intervals

    # Get the score and performance
    entry = sample["entry"]
    score = sample["score"]
    performance = sample["performance"]

    # Get highest and lowest notes in the performance
    highest_note, lowest_note = MIN_PITCH, MAX_PITCH
    for note in performance.tracks[0].notes:
        if note.pitch > highest_note:
            highest_note = note.pitch
        if note.pitch < lowest_note:
            lowest_note = note.pitch
    # highest_note = max(note.pitch for note in sample["performance"].tracks[0].notes)
    # lowest_note = min(note.pitch for note in sample["performance"].tracks[0].notes)

    # Sample interval (make sure that the transposed score fits in the pitch range)
    non_overflowing_intervals = [
        inter
        for inter in intervals
        if MAX_PITCH - highest_note
        >= inter.semitones * (1 if inter.direction == "up" else -1)
        >= MIN_PITCH - lowest_note
    ]

    if not non_overflowing_intervals:
        logger.warning(f"No allowed intervals for {entry.xml_score.as_posix()}, returning as is")
        return {"score": score, "performance": performance}

    # Randomly sample an allowed interval
    interval = random.choice(non_overflowing_intervals)
    semitones = interval.semitones * (1 if interval.direction == "up" else -1)
    logger.debug(
        f"Transposing {entry.xml_score.as_posix()} {interval.direction} by "
        f"{interval.number}{interval.quality} ({semitones} semitones)"
    )
    # Transpose score
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transposed_score = pt.utils.music.transpose(
                score,
                interval,
                transpose_key_signatures=True,
                inplace=True,
            )
    except Exception as e:
        logger.error(
            f"Error when transposing score {entry.xml_score.as_posix()} up by "
            f"{interval.number}{interval.quality}: {e}"
        )
        raise

    # Transpose performance
    for note in performance.tracks[0].notes:
        note.pitch += semitones

    return {"score": transposed_score, "performance": performance}


@add_keys_to_result
def voice_reorder(sample: dict[str, Any]) -> dict[str, Any]:
    reordered_score = reorder_voices(sample["score"])
    return {"score": reordered_score}


@add_keys_to_result
def rest_fill(sample: dict[str, Any]) -> dict[str, Any]:
    fill_rests(sample["score"])
    return {"score": sample["score"]}


@add_keys_to_result
def chunk_music(
    sample: dict[str, Any],
    num_performed_notes: int,
    annotations: pd.DataFrame | Path,
    mode: Literal["random", "all_w_overlap", "all_wo_overlap"] | int = "random",
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    This class chunks a score and a performance into smaller segments.

    :param num_performed_notes: Approx number of performed notes to keep in the chunk
    :param annotations: Path to the annotations file or a DataFrame containing the annotations
    :param mode: Mode of chunking:
        - "random": Randomly sample a chunk of num_performed_notes notes
        - "all_w_overlap": Sample all chunks of num_performed_notes notes, with a sliding
            window of one measure
        - "all_wo_overlap": Sample all chunks of num_performed_notes notes without overlap
        - int: Measure number to start the chunking from
    """
    num_performed_notes = num_performed_notes
    if isinstance(annotations, pd.DataFrame):
        annotations = annotations
    else:
        annotations = pd.read_json(annotations, orient="index")

    # Get the score and performance
    entry: ScorePerformanceEntry = sample["entry"]
    score: pt.score.Part = sample["score"]
    performance: symusic.core.ScoreSecond = sample["performance"]

    # If not enough notes, return the original score and performance
    # We copy them as chunking is supposed to be a copy
    total_notes = len(score.notes)
    if total_notes <= num_performed_notes:
        # Copy score
        sv = pt.score.ScoreVariant(score)
        sv.add_segment(score.first_point, score.last_point)
        copy_score = sv.create_variant_part()
        # Copy performance
        copy_performance = performance.copy()
        return {"score": copy_score, "performance": copy_performance}

    num_measures = len(score.measures)
    # if self.start_measure is None:
    #     # Get a random measure between the first and a bit before the end
    #     start_measure = random.randint(0, num_measures - 10) if num_measures > 15 else 0
    # else:
    #     start_measure = self.start_measure

    # Get the first num_performed_notes notes in the MIDI file starting from this measure
    # (and add notes until the end of the measure if not filled)
    chunks_file = entry.xml_score.parent / f"{entry.midi_performance.stem}_chunks.json"
    if not chunks_file.exists():
        logger.error(f"{chunks_file} does not exist, skipping the sample")
        raise FileNotFoundError(f"{chunks_file} does not exist, skipping the sample")
    chunks = read_chunk_file(chunks_file)

    def clip_performance_and_score(
        start_measure: int,
        num_notes: int,
    ) -> tuple[symusic.core.ScoreSecond, pt.score.Part, int, int]:
        clipped_performance, (start, end) = clip_midi(
            midi=performance,
            chunks=chunks,
            start_measure=start_measure,
            num_notes=num_notes,
            inplace=False,
            check_num_measures=num_measures,
        )
        clipped_score = clip_score(
            score=score,
            start_measure=start,
            end_measure=end,
        )
        return clipped_performance, clipped_score, start, end

    if mode == "random" or isinstance(mode, int):
        start_measure = (
            random.randint(0, num_measures - 1 - 5)  # Random sample between 0 and 5th to last
            if mode == "random"
            else mode
        )
        clipped_perf, clipped_sco, start, end = clip_performance_and_score(
            start_measure, num_performed_notes
        )
        results = {
            "score": clipped_sco,
            "performance": clipped_perf,
            "chunk_start": start,
            "chunk_end": end,
        }

    else:
        results = []
        # In case of overlap, clip until the end of the score
        if mode == "all_w_overlap":
            for start_meas in chunks["first_measure_of_group"]:
                clipped_perf, clipped_sco, start, end = clip_performance_and_score(
                    start_meas, num_performed_notes
                )
                assert start_meas == start, "Start measure should be the same, probable bug"
                results.append(
                    {
                        "score": clipped_sco,
                        "performance": clipped_perf,
                        "chunk_start": start,
                        "chunk_end": end,
                    }
                )
                if end == num_measures - 1:
                    break
        elif mode == "all_wo_overlap":
            start_meas = chunks["first_measure_of_group"][0]
            end = 0
            while end < num_measures - 1:
                # Get the next chunk
                clipped_perf, clipped_sco, start, end = clip_performance_and_score(
                    start_meas, num_performed_notes
                )
                assert start_meas == start, "Start measure should be the same, probable bug"
                new_result = {
                    "score": clipped_sco,
                    "performance": clipped_perf,
                    "chunk_start": start,
                    "chunk_end": end,
                }
                # If less than 5 measures left, recompute the last segment with the remaining
                if 0 < end - num_measures + 1 < 10:
                    logger.debug(
                        f"Last segment with {num_measures - end} measures, adding it to the last chunk"
                    )
                    new_result["chunk_end"] = num_measures - 1
                    clipped_perf, clipped_sco, start, end = clip_performance_and_score(
                        start_meas,
                        num_performed_notes + 100000,  # Large number to get all remaining
                    )
                    new_result = {
                        "score": clipped_sco,
                        "performance": clipped_perf,
                        "chunk_start": start,
                        "chunk_end": end,
                    }
                results.append(new_result)
                start_meas = end + 1
        else:
            raise ValueError(f"Unknown mode {mode}")

    return results


@add_keys_to_result
def scale_performance_tempo(
    sample: dict[str, Any], min_factor: float, max_factor: float
) -> dict[str, Any]:
    # Randomly speed up or slow down the performance
    factor = random.uniform(min_factor, max_factor)
    performance: symusic.core.ScoreTick = sample["performance"]
    performance.adjust_time(
        original_times=[0, performance.end()],
        new_times=[0, int(performance.end() * factor)],
        inplace=True,
    )
    return {"performance": performance}


@add_keys_to_result
def vectorize_score(sample: dict[str, Any]) -> dict[str, Any]:
    # Vectorize score
    score_tokens = partitura_to_token_sequence(sample["score"])
    score_tensor = TokenTensor.from_tokens(score_tokens)
    return {"score_tokens": score_tokens, "score_tensor": score_tensor}


@add_keys_to_result
def vectorize_performance(sample: dict[str, Any]) -> dict[str, Any]:
    # Vectorize performance
    # sample["performance_tensor"] = torch.FloatTensor(
    #     sample["performance"].pianoroll(modes=["frame"], pitch_range=(MIN_PITCH, MAX_PITCH))
    # )
    assert len(sample["performance"].tracks) == 1, "Only one track supported"
    midi_list: list[symusic.core.NoteSecond] = sample["performance"].tracks[0].notes
    # This is directly from Beyer's MIDI2ScoreTransformer code
    midi_streams = {
        "onset": torch.FloatTensor([n.start for n in midi_list]),
        "duration": torch.FloatTensor([n.duration for n in midi_list]),
        "pitch": torch.LongTensor([n.pitch for n in midi_list]),
        "velocity": torch.LongTensor([n.velocity for n in midi_list]),
    }
    performance_streams = MultistreamTokenizer.bucket_midi(midi_streams)
    return {"performance_streams": performance_streams}


def delete_keys(sample: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """
    Delete the keys from the sample.
    """
    for key in keys:
        if key in sample:
            del sample[key]
    return sample


def dict_cache(
    keys_to_cache: list[str] | None = None,
    **kwargs,
):
    def decorator(
        fn: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        @add_keys_to_result
        @cachetools.cached(**kwargs)
        def cached_keys(sample: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
            # Get the new keys
            new_keys = fn(sample, *args, **kwargs)
            # Add the new keys to the input
            return {
                k: v for k, v in new_keys.items() if keys_to_cache is None or k in keys_to_cache
            }

        return cached_keys

    return decorator


def debug(fn: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Debug function to print the sample."""

    def wrapper(sample: dict[str, Any]) -> dict[str, Any]:
        fn(sample)
        return sample

    return wrapper


# class TransformNode(nodes.ParallelMapper):
#     def __init__(
#         self,
#         source: nodes.BaseNode,
#         transform: AbstractProcessor,
#         num_workers: int = 0,
#         in_order: bool = True,
#         method: Literal["thread", "process"] = "thread",
#         multiprocessing_context: str | None = None,
#         max_concurrent: int | None = None,
#         snapshot_frequency: int = 1,
#         prebatch: int | None = None,
#     ) -> None:
#         self.transform = transform
#         super().__init__(
#             source,
#             map_fn=self.transform,
#             num_workers=num_workers,
#             in_order=in_order,
#             method=method,
#             multiprocessing_context=multiprocessing_context,
#             max_concurrent=max_concurrent,
#             snapshot_frequency=snapshot_frequency,
#             prebatch=prebatch,
#         )
