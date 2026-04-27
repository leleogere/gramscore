import random
import sys
import warnings
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING
from typing import Literal
from typing import assert_never

import pandas as pd
import partitura as pt
import partitura.score
import partitura.utils
import pretty_midi as pm
import torch
import torch.utils.data
from loguru import logger

from data.music_chunking import clip_midi
from data.music_chunking import clip_quantized_midi
from data.music_chunking import clip_score
from data.music_chunking import read_chunk_file
from midi2scoretransformer.tokenizer import MultistreamTokenizer
from models.automata.io import partitura_to_token_sequence
from models.automata.tokens import fill_rests, AnyToken
from models.automata.vectorization_torch import TokenTensor
from models.automata.voice_reordering import reorder_voices
from utils.midi import compute_midi_end_without_tempo_changes, read_pretty_midi

if TYPE_CHECKING:
    from data.components.score_performance_dataset import ScorePerformanceEntry


MAX_PITCH = 108
MIN_PITCH = 21

ERROR_KEY = "__error__"

sys.setrecursionlimit(2**31 - 1)  # For transpose copies

pm.pretty_midi.MAX_TICK = 1e10  # Some bugs with high PPQ MIDI files


class ToDict:
    """Simply transforms an entry into a dict with a single key "entry" containing it."""

    def __call__(self, entry: "ScorePerformanceEntry") -> dict[str, Any]:
        return {"entry": entry}


class AbstractProcessor:
    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        """
        This method takes a dict sample, and returns only the newly added/modified keys.

        This avoids modifying the sample inplace, which causes issues with the Cache object.
        """
        raise NotImplementedError("Abstract method")

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        processed = self._dict_process(sample)
        if isinstance(processed, list):
            return [
                {k: v for k, v in sample.items() if k not in new_keys} | new_keys
                for new_keys in processed
            ]
        return {k: v for k, v in sample.items() if k not in processed} | processed


class ScoreReader(AbstractProcessor):
    def process(self, path: Path) -> pt.score.Part:
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            logger.trace(
                f"Worker {worker.id + 1}/{worker.num_workers}: " f"Reading score {path.as_posix()}"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = pt.load_musicxml(path, force_note_ids=True)
            if len(score) > 1:
                raise ValueError(f"More than one part in {path.as_posix()}")
            score = pt.score.unfold_part_maximal(score.parts[0], ignore_leaps=False)
        return score

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        score = self.process(sample["entry"].xml_score)
        return {"score": score}


class PerformanceReader(AbstractProcessor):
    def process(self, path: Path) -> pm.PrettyMIDI:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            performance = read_pretty_midi(path, merge_into_single_track=True)
        assert len(performance.instruments) == 1
        return performance

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {"performance": self.process(sample["entry"].midi_performance)}


class PreComputedAllowedTranspositionsReader(AbstractProcessor):
    """
    This class reads a pre-computed list of allowed transpositions from a ScorePerformanceEntry.
    This list is added to the sample as "allowed_transpositions".
    """

    def __init__(self, limit_semitones: int = None):
        self.limit_semitones = limit_semitones

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        allowed_intervals = sample["entry"].allowed_transpositions
        if self.limit_semitones is not None:
            allowed_intervals = [
                i for i in allowed_intervals if i.semitones <= self.limit_semitones
            ]
        logger.trace(
            f"Pre-computed allowed transpositions for {sample['entry'].xml_score.as_posix()}: "
            + ", ".join(
                f"{'-' if i.direction == 'down' else ''}{i.number}{i.quality}"
                for i in allowed_intervals
            )
        )
        return {"allowed_transpositions": allowed_intervals}


class AllowedTranspositionComputer(AbstractProcessor):
    """
    This class simply computes a list of allowed transpositions for a given score:
        - it does not lead to triple alterations
        - it does not lead to key signatures with more than 7 sharps/flats

    This list is added to the sample as "allowed_transpositions".

    This computation is quite heavy as each score needs to be deep-copied before transposing,
    so it should better be wrapped in the Cache class to only compute it once per score.
    """

    def __init__(self, intervals: list[pt.score.Interval]):
        self.intervals = intervals

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        # Try all intervals
        allowed_intervals = []
        for interval in self.intervals:
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
            except (AssertionError, ValueError, Exception) as e:
                # Not ideal but the easiest way to capture the correct exception
                if isinstance(e, AssertionError) and e.args[0].endswith(
                    "is not in the range -2 to 2."
                ):
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
                elif isinstance(e, Exception) and e.args[0].startswith("Unknown mode "):
                    mode = e.args[0].split("Unknown mode ")[1]
                    logger.trace(
                        f"Can't transpose {sample['entry'].xml_score.as_posix()} "
                        f"{interval.direction} by a {interval.number}{interval.quality} interval "
                        f"(unknown mode: {mode})."
                    )
                else:
                    raise

        logger.trace(
            f"Allowed transpositions for {sample['entry'].xml_score.as_posix()}: "
            + ", ".join(
                f"{'-' if i.direction == 'down' else ''}{i.number}{i.quality}"
                for i in allowed_intervals
            )
        )

        return {"allowed_transpositions": allowed_intervals}


class RandomTransposer(AbstractProcessor):
    """
    This class transposes a score and a performance by a random allowed interval.

    It takes as input a list of allowed intervals. If the sample has a key "allowed_transpositions"
    (see class AllowedTranspositionComputer), it will use this list to sample a random interval
    rather than the one passed in the constructor.
    """

    def __init__(self, intervals: list[pt.score.Interval] = None):
        if intervals is None:
            # By default, minor and major seconds and thirds, up and down
            intervals = [
                pt.score.Interval(n, q, d)
                for n in [2, 3]
                for q in ["m", "M"]
                for d in ["up", "down"]
            ] + [pt.score.Interval(1, "P", "up")]
        self.intervals = intervals

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        # See if some allowed intervals have been precomputed
        if "allowed_transpositions" in sample:
            intervals = sample["allowed_transpositions"]
        else:
            intervals = self.intervals

        # Get the score and performance
        entry = sample["entry"]
        score = sample["score"]
        performance = sample["performance"]

        # Get highest and lowest notes in the performance
        highest_note, lowest_note = MIN_PITCH, MAX_PITCH
        for note in performance.instruments[0].notes:
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
            logger.warning(
                f"No allowed intervals for {entry.xml_score.as_posix()}, returning as is"
            )
            return {"score": score, "performance": performance}

        # Randomly sample an allowed interval
        interval = random.choice(non_overflowing_intervals)
        semitones = interval.semitones * (1 if interval.direction == "up" else -1)
        logger.trace(
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
        for note in performance.instruments[0].notes:
            note.pitch += semitones

        return {"score": transposed_score, "performance": performance}


class VoiceReorderer(AbstractProcessor):
    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        reordered_score = reorder_voices(sample["score"])
        return {"score": reordered_score}


class VoiceRestFiller(AbstractProcessor):
    def __init__(self, tolerant: bool = False):
        self.tolerant = tolerant

    def process(self, score: pt.score.Part) -> pt.score.Part:
        fill_rests(score, tolerant=self.tolerant)  # WARNING: relace in place
        return score

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        self.process(sample["score"])
        return {"score": sample["score"]}


class QuantizedMusicChunker(AbstractProcessor):
    def __init__(
        self,
        num_performed_notes: int,
        mode: Literal["random", "all_w_overlap", "all_wo_overlap"] | int = "random",
    ):
        """
        This class chunks a score and a performance into smaller segments.

        :param num_performed_notes: Approx number of performed notes to keep in the chunk
        :param mode: Mode of chunking:
            - "random": Randomly sample a chunk of num_performed_notes notes
            - "all_w_overlap": Sample all chunks of num_performed_notes notes, with a sliding
                window of one measure
            - "all_wo_overlap": Sample all chunks of num_performed_notes notes without overlap
            - int: Measure number to start the chunking from
        """
        self.num_performed_notes = num_performed_notes
        self.mode = mode

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        # Get the score and performance
        score: pt.score.Part = sample.get("score", None)
        performance: pm.PrettyMIDI = sample["performance"]

        num_score_measures = len(score.measures) if score is not None else None

        if num_score_measures == 1:
            output = {
                "chunk_start": 0,
                "chunk_end": 0,
            }
            if self.mode == "all_w_overlap" or self.mode == "all_wo_overlap":
                output = [output]
            return output

        def clip_performance_and_score(
            start_measure: int,
            num_notes: int,
        ) -> tuple[pm.PrettyMIDI, pt.score.Part, int, int]:
            clipped_performance, (start, end) = clip_quantized_midi(
                midi=performance,
                start_measure=start_measure,
                num_notes=num_notes,
                inplace=False,
                check_num_measures=num_score_measures,
            )
            clipped_score = (
                clip_score(
                    score=score,
                    start_measure=start,
                    end_measure=end,
                )
                if score is not None
                else None
            )
            return clipped_performance, clipped_score, start, end

        if self.mode == "random" or isinstance(self.mode, int):
            if self.mode == "random":
                if num_score_measures >= 2:
                    start_measure = random.randint(0, num_score_measures - 2)
                else:
                    start_measure = 0
            elif isinstance(self.mode, int):
                start_measure = self.mode
            else:
                assert_never(self.mode)
            clipped_perf, clipped_sco, start, end = clip_performance_and_score(
                start_measure, self.num_performed_notes
            )
            if len(clipped_perf.instruments[0].notes) == 0 or (
                clipped_sco is not None and len(clipped_sco.notes) == 0
            ):
                logger.warning(
                    f"Empty performance or score {sample['entry'].midi_performance_relative} clipped at {start, end}, returning the first chunk."
                )
                clipped_perf, clipped_sco, start, end = clip_performance_and_score(
                    0, self.num_performed_notes
                )
            logger.debug(f"Clipping {sample['entry'].debug_path} from {start, end}")
            results = {
                "score": clipped_sco,
                "performance": clipped_perf,
                "chunk_start": start,
                "chunk_end": end,
            }

        else:
            results = []
            # In case of overlap, clip until the end of the score
            if self.mode == "all_w_overlap":
                for start_meas in range(num_score_measures):
                    clipped_perf, clipped_sco, start, end = clip_performance_and_score(
                        start_meas, self.num_performed_notes
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
                    if end == num_score_measures - 1:
                        break
            elif self.mode == "all_wo_overlap":
                start_meas = 0
                end = 0
                while end < num_score_measures - 1:
                    # Get the next chunk
                    clipped_perf, clipped_sco, start, end = clip_performance_and_score(
                        start_meas, self.num_performed_notes
                    )
                    assert start_meas == start, "Start measure should be the same, probable bug"
                    new_result = {
                        "score": clipped_sco,
                        "performance": clipped_perf,
                        "chunk_start": start,
                        "chunk_end": end,
                    }
                    # If less than 5 measures left, recompute the last segment with the remaining
                    if 0 < end - num_score_measures + 1 < 10:
                        logger.trace(
                            f"Last segment with {num_score_measures - end} measures, "
                            f"adding it to the last chunk"
                        )
                        new_result["chunk_end"] = num_score_measures - 1
                        clipped_perf, clipped_sco, start, end = clip_performance_and_score(
                            start_meas,
                            self.num_performed_notes
                            + 100_000_000,  # Large number to get all remaining
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
                raise ValueError(f"Unknown mode {self.mode}")

        return results


class AsapMusicChunker(AbstractProcessor):
    def __init__(
        self,
        num_performed_notes: int,
        annotations: pd.DataFrame | Path,
        mode: Literal["random", "all_w_overlap", "all_wo_overlap"] | int = "random",
    ):
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
        self.num_performed_notes = num_performed_notes
        if isinstance(annotations, pd.DataFrame):
            self.annotations = annotations
        else:
            self.annotations = pd.read_json(annotations, orient="index")
        self.mode = mode

    def process(
        self,
        score: pt.score.Part,
        performance: pm.PrettyMIDI,
        chunks_file: Path,
    ) -> (
        tuple[pm.PrettyMIDI, pt.score.Part, int, int]
        | list[tuple[pm.PrettyMIDI, pt.score.Part, int, int]]
    ):
        """
        Chunk an ASAP pair (performance, score) using pre-computed chunk data.

        :param score: Score to chunk
        :param performance: Performance to chunk
        :param chunks_file: Path to the JSON chunks file (pre-computed)
        """
        num_measures = len(score.measures)

        # Get the first num_performed_notes notes in the MIDI file starting from this measure
        # (and add notes until the end of the measure if not filled)
        if not chunks_file.exists():
            logger.error(f"{chunks_file} does not exist, skipping the sample")
            raise FileNotFoundError(f"{chunks_file} does not exist, skipping the sample")
        chunks = read_chunk_file(chunks_file)

        def clip_performance_and_score(
            start_measure: int,
            num_notes: int | None,  # all notes if None
        ) -> tuple[pm.PrettyMIDI, pt.score.Part, int, int]:
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
            return clipped_performance, clipped_score, int(start), int(end)

        if self.mode == "random" or isinstance(self.mode, int):
            start_measure = (
                random.randint(0, num_measures - 1 - 5)  # Random sample between 0 and 5th to last
                if self.mode == "random"
                else self.mode
            )
            outputs = clip_performance_and_score(start_measure, self.num_performed_notes)

        else:
            outputs = []
            # In case of overlap, clip until the end of the score
            if self.mode == "all_w_overlap":
                for start_meas in chunks["first_measure_of_group"]:
                    clipped_perf, clipped_sco, start, end = clip_performance_and_score(
                        start_meas, self.num_performed_notes
                    )
                    assert start_meas == start, "Start measure should be the same, probable bug"
                    outputs.append((clipped_perf, clipped_sco, start, end))
                    if end == num_measures - 1:
                        break
            elif self.mode == "all_wo_overlap":
                start_meas = chunks["first_measure_of_group"][0]
                end = 0
                while end < num_measures - 1:
                    # Get the next chunk
                    clipped_perf, clipped_sco, start, end = clip_performance_and_score(
                        start_meas, self.num_performed_notes
                    )
                    assert start_meas == start, "Start measure should be the same, probable bug"
                    # If less than 3 measures left, recompute the last segment with the remaining
                    if 0 < num_measures - end - 1 <= 3:
                        logger.trace(
                            f"Last segment with {num_measures - end} measures, adding it to the last chunk"
                        )
                        clipped_perf, clipped_sco, start, end = clip_performance_and_score(
                            start_meas, None  # get all remaining notes
                        )
                    outputs.append((clipped_perf, clipped_sco, start, end))
                    start_meas = end + 1
            else:
                raise ValueError(f"Unknown mode {self.mode}")

        return outputs

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]] | None:
        entry = sample["entry"]
        chunk_file = entry.xml_score.parent / f"{entry.midi_performance.stem}_chunks.json"
        if not chunk_file.exists():
            return {ERROR_KEY: f"{chunk_file} does not exist, skipping the sample"}
        outputs = self.process(
            score=sample["score"],
            performance=sample["performance"],
            chunks_file=chunk_file,
        )
        if isinstance(outputs, list):
            return [
                {
                    "performance": clipped_perf,
                    "score": clipped_sco,
                    "chunk_start": start,
                    "chunk_end": end,
                }
                for clipped_perf, clipped_sco, start, end in outputs
            ]
        else:
            clipped_perf, clipped_sco, start, end = outputs
            return {
                "performance": clipped_perf,
                "score": clipped_sco,
                "chunk_start": start,
                "chunk_end": end,
            }


class PerformanceSpeedAugmenter(AbstractProcessor):
    def __init__(self, min_factor: float, max_factor: float, end_variation: float | None = None):
        """
        Randomly speed up or slow down the performance.

        :param min_factor: Minimum factor to speed up or slow down the performance
        :param max_factor: Maximum factor to speed up or slow down the performance
        :param end_variation: If None, simply scale the whole performance of a constant factor. If set, will set the
            final tempo randomly within (1 +/- end_variation)*initial_scaled_tempo, and linearly interpolate. This will
            result in the piece slowly speeding up or slowing down for the duration of the piece.
        :example: For min=0.8, max=1.2, the initial tempo will be randomly scaled between 80% and 120% of the original
            tempo (let's say 93%). If end_variation is None, the tempo will stay at 93% for the whole piece. If
            end_variation is set to 0.05, the final tempo will be set randomly between (uniformely) 88% and 98% (let's
            say 96%). Then the tempo will be linearly interpolated between 93% and 96% for the duration of the piece.
        """
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.end_variation = end_variation

    # TODO: refactor all processors to use this template: one process function with only the
    #   needed arguments, and the _dict_process that will take care of the dictionary
    def process(self, performance: pm.PrettyMIDI) -> pm.PrettyMIDI:
        # Randomly speed up or slow down the performance
        initial_speed_factor = random.uniform(self.min_factor, self.max_factor)  # as pos=1/speed
        initial_position_factor = 1 / initial_speed_factor
        print(f"{initial_speed_factor=}")
        print(f"{initial_position_factor=}")
        if self.end_variation is None:
            for instrument in performance.instruments:
                for note in instrument.notes:
                    note.start *= initial_position_factor
                    note.end *= initial_position_factor
        else:
            # In this case, the initial *factor* (not position) is y0, and the final
            # factor is yf. Then for a time t, we have factor(t) = y0 + t * (yf - y0) / tf.
            # Hence, the new time is t * factor(t) = t * (y0 + t * (yf - y0) / tf))
            end = compute_midi_end_without_tempo_changes(performance)
            final_speed_factor = max(
                0.1,  # Do not allow to get slower than 10% of initial speed
                random.uniform(
                    initial_speed_factor - self.end_variation,
                    initial_speed_factor + self.end_variation,
                ),
            )
            final_position_factor = 1 / final_speed_factor
            print(f"{final_speed_factor=}")
            print(f"{final_position_factor=}")
            coef = (final_position_factor - initial_position_factor) / end
            print(f"{coef=}")
            for instrument in performance.instruments:
                for note in instrument.notes:
                    note.start = note.start * (initial_position_factor + coef * note.start)
                    note.end = note.end * (initial_position_factor + coef * note.end)
        return performance

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        augmented_performance = self.process(sample["performance"])
        return {"performance": augmented_performance}


class VectorizedPerformanceVariableSpeedAugmenter(AbstractProcessor):
    def __init__(
        self,
        min_global_factor: float = 0.8,
        max_global_factor: float = 1.2,
        min_relative_local_factor: float = -0.1,
        max_relative_local_factor: float = 0.1,
        min_time_factor_change: float = 40.0,
        max_time_factor_change: float = 60.0,
    ):
        """
        Augment a vectorized performance with a variable speed change. This combines a global tempo
        scaling (uniform in min/max_global_factor) with an eventual local tempo scaling (if
        min/max_relative_local_factor are not zero). The local scaling basically chooses in speed
        between sampled_global_tempo +/- sampled_local_tempo every random amount of time in
        min/max_time_factor_change seconds. In between each tempo change, the factor is linearly
        interpolated.

        :param min_global_factor: Minimum global factor to speed up or slow down the performance
        :param max_global_factor: Maximum global factor to speed up or slow down the performance
        :param min_relative_local_factor: Minimum local factor to speed up or slow down the
            performance, relative to the global tempo
        :param max_relative_local_factor: Maximum local factor to speed up or slow down the
            performance, relative to the global tempo
        :param min_time_factor_change: Minimum seconds between two consecutive tempo changes
        :param max_time_factor_change: Maximum seconds between two consecutive tempo changes
        """
        if not (0 < min_global_factor <= max_global_factor):
            raise ValueError(
                f"{min_global_factor=} must be positive and less than {max_global_factor=}"
            )
        if min_relative_local_factor > max_relative_local_factor:
            raise ValueError(
                f"{min_relative_local_factor=} must be less than {max_relative_local_factor=}"
            )
        if not (0 < min_time_factor_change <= max_time_factor_change):
            raise ValueError(
                f"{min_time_factor_change=} must be positive and less than {max_time_factor_change=}"
            )

        self.min_global_factor = min_global_factor
        self.max_global_factor = max_global_factor
        self.min_relative_local_factor = min_relative_local_factor
        self.max_relative_local_factor = max_relative_local_factor
        self.min_time_factor_change = min_time_factor_change
        self.max_time_factor_change = max_time_factor_change

    def process(self, streams: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        onsets = streams["onset"]
        offsets = onsets + streams["duration"]

        # Scale with the global factor
        global_factor = random.uniform(self.min_global_factor, self.max_global_factor)
        onsets *= global_factor
        offsets *= global_factor

        # Scale with the local factor if needed
        if self.min_relative_local_factor != 0 or self.max_relative_local_factor != 0:
            # Concatenate all times to scale
            times = torch.cat([onsets, offsets], dim=0)

            # Compute upper bound for number of segments
            t_end = times.max()
            max_segments = int(torch.ceil(t_end / self.min_time_factor_change).item()) + 2

            # Sample the delta times
            dt = torch.empty(max_segments, device=onsets.device).uniform_(
                self.min_time_factor_change, self.max_time_factor_change
            )

            # Get the min and max absolute scaling values
            # QUESTION: Should I also multiply those by global_factor?
            min_scaling = global_factor + self.min_relative_local_factor
            max_scaling = global_factor + self.max_relative_local_factor

            # Sample the scaling at each change (1 more knot than segments)
            knots_s = torch.empty(max_segments + 1, device=onsets.device).uniform_(
                min_scaling, max_scaling
            )
            knots_t = torch.zeros(max_segments + 1, device=onsets.device)
            knots_t[1:] = torch.cumsum(dt, dim=0)

            # Only keep the knots that are within the bounds
            valid = knots_t <= t_end
            last = valid.sum().item()
            knots_t = knots_t[: last + 1]
            knots_s = knots_s[: last + 1]

            # Compute linear ramp parameters
            seg_dt = knots_t[1:] - knots_t[:-1]  # segment durations
            seg_ds = knots_s[1:] - knots_s[:-1]  # scaling differences
            slopes = seg_ds / seg_dt  # slope of the scaling ramp for each segment

            # Prefix integral at know times
            # ∫(si + m*x)dx = si * dt + 0.5 * m * dt**2
            F = torch.zeros_like(knots_t)
            F[1:] = torch.cumsum(knots_s[:-1] * seg_dt + 0.5 * slopes * seg_dt**2, dim=0)

            # Find the segment each time belongs to
            idx = torch.searchsorted(knots_t, times, right=True) - 1
            idx = idx.clamp(0, len(knots_t) - 2)

            # Local offset in the segment
            x = times - knots_t[idx]

            # Compute time scaling
            warped_times = F[idx] + knots_s[idx] * x + 0.5 * slopes[idx] * x**2

            # Retreive onsets and offsets
            onsets, offsets = warped_times[: len(onsets)], warped_times[len(onsets) :]

        streams["onset"] = onsets
        streams["duration"] = offsets - onsets
        return streams

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        augmented_streams = self.process(sample["performance_streams_raw"])
        return {
            "performance_streams_raw": augmented_streams,
            "performance_onset_times": augmented_streams["onset"],
        }


class VectorizedPerformanceInterOnsetJitterer(AbstractProcessor):
    """
    Jitter the inter-onset interval of notes, following a centered normal distribution.
    QUESTION: as in Beyer, but this could cause issues and cause overlapping notes right?
    """

    def __init__(self, std: float = 0.05):
        self.std = std

    def process(self, streams: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert streams["onset"].ndim == 1, "Onsets should be a 1D tensor"
        factor = 1 + torch.randn(streams["onset"].shape) * self.std
        inter_onset_intervals = torch.clamp(
            torch.diff(streams["onset"], prepend=torch.tensor([0.0]), dim=0),
            min=0,
        )
        streams["onset"] = torch.cumsum(factor * inter_onset_intervals, dim=0)
        return streams

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        augmented_streams = self.process(sample["performance_streams_raw"])
        return {
            "performance_streams_raw": augmented_streams,
            "performance_onset_times": augmented_streams["onset"].clone(),
        }


class VectorizedPerformanceDurationJitterer(AbstractProcessor):
    """
    Jitter the inter-onset interval of notes, following a centered normal distribution.
    QUESTION: as in Beyer, but this could cause issues and cause overlapping notes right?
    """

    def __init__(self, min_factor: float = 0.95, max_factor: float = 1.05):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def process(self, streams: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert streams["duration"].ndim == 1, "Onsets should be a 1D tensor"
        factor = self.min_factor + torch.rand(streams["duration"].shape) * (
            self.max_factor - self.min_factor
        )
        streams["duration"] *= factor
        return streams

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        augmented_streams = self.process(sample["performance_streams_raw"])
        return {"performance_streams_raw": augmented_streams}


class ScoreVectorizer(AbstractProcessor):
    def process(self, part: pt.score.Part) -> tuple[list[AnyToken], torch.Tensor]:
        score_tokens = partitura_to_token_sequence(part)
        score_tensor = TokenTensor.from_tokens(score_tokens)
        return score_tokens, score_tensor

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        score_tokens, score_tensor = self.process(sample["score"])
        return {"score_tokens": score_tokens, "score_tensor": score_tensor}


class PerformanceToPerformedNoteList(AbstractProcessor):
    def process(self, performance: pm.PrettyMIDI) -> list[pm.Note]:
        assert len(performance.instruments) == 1, "Only works with single-instrument performances"
        return list(performance.instruments[0].notes)

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {"performance_notes": self.process(sample["performance"])}


class PerformedNoteListVectorizer(AbstractProcessor):
    def process(self, notes: list[pm.Note]) -> dict[str, torch.Tensor]:
        notes = sorted(notes, key=lambda n: (n.start, n.pitch, n.duration))
        # This is directly from Beyer's MIDI2ScoreTransformer code
        streams = {
            "onset": torch.FloatTensor([n.start for n in notes]),
            "duration": torch.FloatTensor([n.duration for n in notes]),
            "pitch": torch.LongTensor([n.pitch for n in notes]),
            "velocity": torch.LongTensor([n.velocity for n in notes]),
        }
        return streams

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        streams = self.process(sample["performance_notes"])
        return {
            "performance_streams_raw": streams,
            "performance_onset_times": streams["onset"].clone(),
        }


class VectorizedPerformanceStreamsBucketizer(AbstractProcessor):
    def process(self, streams: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return MultistreamTokenizer.bucket_midi(streams)

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {"performance_streams": self.process(sample["performance_streams_raw"])}


class Cache(AbstractProcessor):
    def __init__(
        self,
        transforms: list[Callable[[Any], Any]],
        cache_key: Callable[[Any], Any] = lambda x: x,
        keys_to_cache: list[str] = None,
        cache: Mapping = None,  # Can use a multiprocessing dict to share between workers
    ):
        self.transforms = transforms
        self.cache_key = cache_key
        self.keys_to_cache = keys_to_cache
        if cache is not None:
            self.cache = cache
        else:
            self.cache = {}

    def _dict_process(self, sample: Any) -> Any:
        key = self.cache_key(sample)
        worker = torch.utils.data.get_worker_info()
        if key not in self.cache:
            if worker is not None:
                logger.trace(
                    f"Worker {worker.id + 1}/{worker.num_workers}: "
                    f"Cache miss for {sample['entry'].xml_score.as_posix()}!"
                )
            for transform in self.transforms:
                try:
                    sample = transform(sample)
                except Exception as e:
                    sample[ERROR_KEY] = transform.__class__.__name__ + ": " + str(e)
                    break
            self.cache[key] = {
                k: v
                for k, v in sample.items()
                if self.keys_to_cache is None or k in self.keys_to_cache
            }
        else:
            if worker is not None:
                logger.trace(
                    f"Worker {worker.id + 1}/{worker.num_workers}: "
                    f"Cache hit for {sample['entry'].xml_score.as_posix()}!"
                )
        return self.cache[key]


class TransformChainer(AbstractProcessor):
    def __init__(self, transforms: list[Callable[[Any], Any]]):
        self.transforms = transforms

    def _dict_process(self, sample: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class Debugger(AbstractProcessor):
    def __init__(
        self,
        func: Callable[[Any], None],
    ):
        self.func = func

    def _dict_process(self, sample: Any) -> Any:
        self.func(sample)
        return {}
