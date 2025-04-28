import random
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
import sys

import git
import line_profiler
import pandas as pd
import partitura as pt
import partitura.score
import partitura.utils
import symusic
import torch
from lightning import LightningDataModule
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from data.components.score_performance_dataset import AsapDataset
from data.components.score_performance_dataset import ScorePerformanceEntry
from data.music_chunking import clip_midi
from data.music_chunking import clip_score
from data.music_chunking import read_chunk_file
from midi2scoretransformer.tokenizer import MultistreamTokenizer
from models.automata.io import partitura_to_token_sequence
from models.automata.tokens import fill_rests
from models.automata.vectorization_torch import TokenTensor
from models.automata.voice_reordering import reorder_voices
from utils.constants import PADDING_VALUE
from utils.constants import SUCCESS_FILENAME

torch.serialization.add_safe_globals([TokenTensor])

MAX_PITCH = 108
MIN_PITCH = 21


class MusicReader:
    @line_profiler.profile
    def __call__(self, entry: ScorePerformanceEntry) -> dict[str, Any]:
        # Read score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = pt.load_musicxml(entry.xml_score, force_note_ids=True)
            assert len(score.parts) == 1
            part = pt.score.unfold_part_maximal(score.parts[0])

        # Read performance
        performance = symusic.Score.from_file(entry.midi_performance)
        # Merge all tracks into first one if multiple ones
        if len(performance.tracks) > 1:
            track = performance.tracks[0]  # Only reference so OK modifying it
            for other in performance.tracks[1:]:
                for attr in ["notes", "controls", "pitch_bends", "pedals", "lyrics"]:
                    getattr(track, attr).extend(getattr(other, attr))
                performance.tracks.remove(other)
        assert len(performance.tracks) == 1

        return {
            "entry": entry,
            "score": part,
            "performance": performance,
        }


class RandomTransposer:
    def __init__(self, intervals: list[pt.score.Interval]):
        self.intervals = intervals

    @line_profiler.profile
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        # Fix recursion limit to avoid RecursionError when deepcopying scores
        recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10**6)

        # Get highest and lowest notes
        highest_note, lowest_note = MIN_PITCH, MAX_PITCH
        for note in sample["performance"].tracks[0].notes:
            if note.pitch > highest_note:
                highest_note = note.pitch
            if note.pitch < lowest_note:
                lowest_note = note.pitch
        # highest_note = max(note.pitch for note in sample["performance"].tracks[0].notes)
        # lowest_note = min(note.pitch for note in sample["performance"].tracks[0].notes)

        # Sample interval (make sure that the transposed score fits in the pitch range)
        non_overflowing_intervals = [
            inter
            for inter in self.intervals
            if MAX_PITCH - highest_note
            >= inter.semitones * (1 if inter.direction == "up" else -1)
            >= MIN_PITCH - lowest_note
        ]

        # Randomy sample intervals until we get one without triple alterations
        random.shuffle(non_overflowing_intervals)
        for interval in non_overflowing_intervals:
            semitones = interval.semitones * (1 if interval.direction == "up" else -1)
            logger.trace(
                f"Transposing {sample['entry'].xml_score.as_posix()} by {interval} ({semitones} semitones)"
            )
            try:
                # Transpose score
                transposed_score = pt.utils.music.transpose(
                    sample["score"],
                    interval,
                    transpose_key_signatures=True,
                    inplace=False,
                )
                break
            except (AssertionError, ValueError) as e:
                # Not ideal but easiest way to capture the correct exception
                if isinstance(e, AssertionError) and e.args[0].endswith(
                    "is not in the range -2 to 2."
                ):
                    logger.trace(
                        f"Can't transpose {sample['entry'].xml_score.as_posix()} by {interval} (alterations > 2)"
                    )
                elif isinstance(e, ValueError) and e.args[0].endswith(
                    "The maximum number of fifths is +/- 7."
                ):
                    logger.trace(
                        f"Can't transpose {sample['entry'].xml_score.as_posix()} by {interval} (key fifths > 7)"
                    )
                else:
                    raise
        else:
            # No transposition possible, return the sample as is
            logger.trace(
                f"Can't transpose {sample['entry'].xml_score.as_posix()} by any of the intervals"
            )
            return sample

        # Save transposed score
        sample["score"] = transposed_score

        # Transpose performance
        for note in sample["performance"].tracks[0].notes:
            note.pitch += semitones

        # Set recursion limit back to normal
        sys.setrecursionlimit(recursion_limit)

        return sample


class VoiceReorderer:
    @line_profiler.profile
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample["score"] = reorder_voices(sample["score"])
        return sample


class VoiceRestFiller:
    @line_profiler.profile
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        fill_rests(sample["score"])
        return sample


class MusicChunker:
    def __init__(
        self,
        num_performed_notes: int,
        annotations: pd.DataFrame | Path,
        start_measure: int | None = None,
    ):
        self.num_performed_notes = num_performed_notes
        if isinstance(annotations, pd.DataFrame):
            self.annotations = annotations
        else:
            self.annotations = pd.read_json(annotations, orient="index")
        self.start_measure = start_measure

    @line_profiler.profile
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:

        # If not enough notes, return the original score and performance
        total_notes = len(sample["score"].notes)
        if total_notes <= self.num_performed_notes:
            return sample

        num_measures = len(sample["score"].measures)
        if self.start_measure is None:
            # Get a random measure between the first and a bit before the end
            start_measure = random.randint(0, num_measures - 10) if num_measures > 15 else 0
        else:
            start_measure = self.start_measure

        # print(f"{start_measure = }")

        # Get the first num_performed_notes notes in the MIDI file starting from this measure
        # (and add notes until the end of the measure if not filled)
        chunks_file = (
            sample["entry"].xml_score.parent
            / f"{sample['entry'].midi_performance.stem}_chunks.json"
        )
        if not chunks_file.exists():
            logger.error(f"{chunks_file} does not exist, skipping the sample")
            raise FileNotFoundError(f"{chunks_file} does not exist, skipping the sample")
        chunks = read_chunk_file(chunks_file)
        # print("BEFORE clip_midi")

        performance, (start_measure, end_measure) = clip_midi(
            midi=sample["performance"],
            chunks=chunks,
            start_measure=start_measure,
            num_notes=self.num_performed_notes,
            inplace=False,
            check_num_measures=num_measures,
        )
        # print("AFTER clip_midi")

        # Get the corresponding notes in the score
        score = clip_score(
            score=sample["score"],
            start_measure=start_measure,
            end_measure=end_measure,
        )
        # print("AFTER clip_score")

        return {
            "score": score,
            "performance": performance,
            **{k: v for k, v in sample.items() if k not in ["score", "performance"]},
        }


class PerformanceSpeedAugmenter:
    def __init__(self, min_factor: float, max_factor: float):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        # Randomly speed up or slow down the performance
        factor = random.uniform(self.min_factor, self.max_factor)
        performance: symusic.core.ScoreTick = sample["performance"]
        performance.adjust_time(
            original_times=[0, performance.end()],
            new_times=[0, int(performance.end() * factor)],
            inplace=True,
        )
        return sample


class ScoreVectorizer:
    @line_profiler.profile
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        # Vectorize score
        sample["score_tokens"] = partitura_to_token_sequence(sample["score"])
        sample["score_tensor"] = TokenTensor.from_tokens(sample["score_tokens"])
        return sample


class PerformanceVectorizer:
    @line_profiler.profile
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
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
        sample["performance_streams"] = MultistreamTokenizer.bucket_midi(midi_streams)
        return sample


class Cache:
    def __init__(
        self,
        transforms: list[Callable[[Any], Any]],
        key: Callable[[Any], Any] = lambda x: x,
    ):
        self.transforms = transforms
        self.key = key
        self.cache = {}

    @line_profiler.profile
    def __call__(self, sample: Any) -> Any:
        key = self.key(sample)
        if key in self.cache:
            sample = self.cache[key]
        else:
            for transform in self.transforms:
                sample = transform(sample)
            self.cache[key] = sample
        return sample


class Debugger:
    def __init__(
        self,
        func: Callable[[Any], None],
    ):
        self.func = func

    def __call__(self, sample: Any) -> Any:
        self.func(sample)
        return sample


class TmpTrainTransform:
    """
    This class regroup all transforms that are only used for training for easier profiling.
    """

    def __init__(
        self,
        num_performed_notes: int,
        annotations: pd.DataFrame | Path,
        start_measure: int | None = None,
    ):
        self.num_performed_notes = num_performed_notes
        if isinstance(annotations, pd.DataFrame):
            self.annotations = annotations
        else:
            self.annotations = pd.read_json(annotations, orient="index")
        self.start_measure = start_measure
        CACHE = True
        if CACHE:
            self.cache = {}
        else:
            self.cache = None

    @line_profiler.profile
    def __call__(self, entry: ScorePerformanceEntry) -> dict[str, Any]:
        if self.cache is not None and entry.midi_performance in self.cache:
            sample = self.cache[entry.midi_performance]
        else:
            # ===========
            # MusicReader
            # ===========
            # Read score
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = pt.load_musicxml(entry.xml_score, force_note_ids=True)
                assert len(score.parts) == 1
                part = pt.score.unfold_part_maximal(score.parts[0])

            # Read performance
            performance = symusic.Score.from_file(entry.midi_performance)
            # Merge all tracks into first one if multiple ones
            if len(performance.tracks) > 1:
                track = performance.tracks[0]  # Only reference so OK modifying it
                for other in performance.tracks[1:]:
                    for attr in ["notes", "controls", "pitch_bends", "pedals", "lyrics"]:
                        getattr(track, attr).extend(getattr(other, attr))
                    performance.tracks.remove(other)
            assert len(performance.tracks) == 1

            sample = {
                "entry": entry,
                "score": part,
                "performance": performance,
            }

            # ===============
            # VoiceRestFiller
            # ===============
            fill_rests(sample["score"])

            # ==============
            # VoiceReorderer
            # ==============
            sample["score"] = reorder_voices(sample["score"])

            if self.cache is not None:
                self.cache[entry.midi_performance] = sample

        # ============
        # MusicChunker
        # ============
        # If not enough notes, return the original score and performance
        total_notes = len(sample["score"].notes)
        if total_notes <= self.num_performed_notes:
            return sample

        num_measures = len(sample["score"].measures)
        if self.start_measure is None:
            # Get a random measure between the first and a bit before the end
            start_measure = random.randint(0, num_measures - 10) if num_measures > 15 else 0
        else:
            start_measure = self.start_measure

        # print(f"{start_measure = }")

        # Get the first num_performed_notes notes in the MIDI file starting from this measure
        # (and add notes until the end of the measure if not filled)
        chunks_file = (
            sample["entry"].xml_score.parent
            / f"{sample['entry'].midi_performance.stem}_chunks.json"
        )
        if not chunks_file.exists():
            logger.error(f"{chunks_file} does not exist, skipping the sample")
            raise FileNotFoundError(f"{chunks_file} does not exist, skipping the sample")
        chunks = read_chunk_file(chunks_file)
        # print("BEFORE clip_midi")

        performance, (start_measure, end_measure) = clip_midi(
            midi=sample["performance"],
            chunks=chunks,
            start_measure=start_measure,
            num_notes=self.num_performed_notes,
            inplace=False,
            check_num_measures=num_measures,
        )
        # print("AFTER clip_midi")

        # Get the corresponding notes in the score
        score = clip_score(
            score=sample["score"],
            start_measure=start_measure,
            end_measure=end_measure,
        )

        sample = {
            "score": score,
            "performance": performance,
            **{k: v for k, v in sample.items() if k not in ["score", "performance"]},
        }

        # ===============
        # ScoreVectorizer
        # ===============
        sample["score_tokens"] = partitura_to_token_sequence(sample["score"])
        sample["score_tensor"] = TokenTensor.from_tokens(sample["score_tokens"])

        # =====================
        # PerformanceVectorizer
        # =====================
        assert len(sample["performance"].tracks) == 1, "Only one track supported"
        midi_list: list[symusic.core.NoteSecond] = sample["performance"].tracks[0].notes
        # This is directly from Beyer's MIDI2ScoreTransformer code
        midi_streams = {
            "onset": torch.FloatTensor([n.start for n in midi_list]),
            "duration": torch.FloatTensor([n.duration for n in midi_list]),
            "pitch": torch.LongTensor([n.pitch for n in midi_list]),
            "velocity": torch.LongTensor([n.velocity for n in midi_list]),
        }
        sample["performance_streams"] = MultistreamTokenizer.bucket_midi(midi_streams)

        return sample


class AsapTokensDataModule(LightningDataModule):
    """
    LightningDataModule for the ASAP dataset, formatted as tokens.
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 64,
        include_midi_scores: bool = False,
        filter_fn: Callable[[ScorePerformanceEntry], bool] = lambda x: True,
        num_workers: int = 4,
        prefetch_factor: int = None,
        persistent_workers: bool = False,
    ) -> None:
        """Initialize a `AsapTokensDataModule`

        :param data_dir: The data directory.
        :param batch_size: The batch size.
        :param include_midi_scores: Whether to include the MIDI scores as performances.
        :param filter_fn: A function to filter the entries.
        :param num_workers: The number of workers to use for data loading.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        # Hydra does not support lambda functions, so it is loaded as string
        # Make sure to only pass trusted code in here as it will be executed as is
        if isinstance(filter_fn, str):
            if not filter_fn.startswith("lambda"):
                raise ValueError("filter_fn must start with 'lambda'")
            self.filter_fn = eval(filter_fn)
            print(self.filter_fn)
        else:
            self.filter_fn = filter_fn

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        self.splits = ["train", "validation", "test"]

        self._prepared = False

    @staticmethod
    @line_profiler.profile
    def collate_fn(
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Collate function for the `DataLoader`.

        :param batch: The batch to collate.
        :return: The collated batch.
        """
        # Pad and collate score tensors
        scores = [x["score_tensor"] for x in batch]  # each N_TOKENS * TOTAL_DIMENSIONS
        scores_collated: TokenTensor = pad_sequence(
            scores,
            batch_first=True,
            padding_value=PADDING_VALUE,
        )  # B * N_TOKENS_MAX * TOTAL_DIMENSIONS
        score_padding_mask = scores_collated[:, :, 0] == PADDING_VALUE  # B * N_TOKENS_MAX
        score_lengths = torch.LongTensor([s.size(0) for s in scores])

        # Pad and collate performance tensor streams
        performances = [x["performance_streams"] for x in batch]  # each N_NOTES * N_FEATURES_VALUE
        streams = ["onset", "duration", "pitch", "velocity"]
        # Make sure all stream have same batch and num_notes
        assert all(
            performance[streams[0]].shape[0] == performance[stream].shape[0]
            for performance in performances
            for stream in streams[1:]
        ), "All streams of a performance should have the same number of notes"
        performances_collated = {
            stream: pad_sequence(
                [x[stream] for x in performances],
                batch_first=True,
                padding_value=PADDING_VALUE,
            )
            for stream in streams
        }  # key: B * N_NOTES_MAX * N_FEATURES_VALUE
        performance_padding_mask = (
            performances_collated[streams[0]][:, :, 0] == PADDING_VALUE
        )  # B * N_NOTES_MAX
        performance_lengths = torch.LongTensor([p[streams[0]].size(0) for p in performances])
        # Security check to catch bug
        assert all(
            performances_collated[streams[0]].shape[0:2] == performances_collated[stream].shape[0:2]
            for stream in streams[1:]
        ), "All streams should have the same batch size and number of notes after padding"

        return {
            "score_tensor": scores_collated,
            "score_padding_mask": score_padding_mask,
            "score_lengths": score_lengths,
            "performance_streams": performances_collated,
            "performance_padding_mask": performance_padding_mask,
            "performance_lengths": performance_lengths,
        }

    def prepare_data(self) -> None:
        """
        Download and split the dataset
        """
        # Download the dataset
        if self._prepared:
            return
        success_file = self.hparams.data_dir / SUCCESS_FILENAME
        if not success_file.exists():
            # NOT Clone the MIDI2ScoreTransformer modified ASAP dataset (commit 8cba199e15931975542010a7ea2ff94a6fc9cbee)
            # Clone the (n)-ASAP dataset
            logger.info("Cloning the ASAP dataset")
            if not self.hparams.data_dir.exists():
                git.Repo.clone_from(
                    url="https://github.com/leleogere/asap-dataset.git",
                    to_path=self.hparams.data_dir,
                    branch="develop_with_bwv874_bwv846_beeth17-2_merged",
                )
            # Create the success file
            success_file.touch()
        else:
            logger.info("ASAP dataset already downloaded")

        self._prepared = True

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})"
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Setup transforms
        transforms = {
            "train": [
                # TmpTrainTransform(
                #     num_performed_notes=500,
                #     annotations=self.hparams.data_dir / "asap_annotations.json",
                # )
                Cache(
                    transforms=[
                        MusicReader(),
                        VoiceRestFiller(),
                        VoiceReorderer(),
                    ],
                    key=lambda entry: entry.midi_performance,
                ),
                # # Debugger(lambda s: print("DEBUG", s["entry"])),
                MusicChunker(
                    num_performed_notes=500,
                    annotations=self.hparams.data_dir / "asap_annotations.json",
                ),
                RandomTransposer(
                    intervals=[
                        pt.score.Interval(
                            number=int(inter[0]), quality=inter[1:], direction=direction
                        )
                        for inter in ["2M", "2m", "3M", "3m", "4P", "4A", "4d"]
                        for direction in ["up", "down"]
                    ]
                ),
                PerformanceSpeedAugmenter(min_factor=0.8, max_factor=1.2),
                ScoreVectorizer(),
                PerformanceVectorizer(),
            ],
            "validation": [
                Cache(
                    transforms=[
                        MusicReader(),
                        VoiceRestFiller(),
                        VoiceReorderer(),
                    ],
                    key=lambda entry: entry.midi_performance,
                ),
                # TODO: better way to evaluate than simply taking the start of the piece
                MusicChunker(
                    num_performed_notes=500,
                    annotations=self.hparams.data_dir / "asap_annotations.json",
                    start_measure=0,
                ),
                ScoreVectorizer(),
                PerformanceVectorizer(),
            ],
            "test": [
                MusicReader(),
                VoiceReorderer(),
                VoiceRestFiller(),
                MusicChunker(
                    num_performed_notes=500,
                    annotations=self.hparams.data_dir / "asap_annotations.json",
                    start_measure=0,
                ),
                ScoreVectorizer(),
                PerformanceVectorizer(),
            ],
        }

        # Load the splits
        if not self.data_train and not self.data_val and not self.data_test:
            sets = {
                split: AsapDataset(
                    data_dir=self.hparams.data_dir,
                    split=split,
                    include_midi_scores=self.hparams.include_midi_scores,
                    transforms=transforms[split],
                    filter_fn=self.filter_fn,
                )
                for split in self.splits
            }
            self.data_train = sets["train"]
            self.data_val = sets["validation"]
            self.data_test = sets["test"]

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
        )

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    import sys
    import time
    from tqdm import tqdm

    logger.remove()
    logger.add(sink=sys.stderr, level="INFO")

    data_module = AsapTokensDataModule(
        data_dir=Path("~/Documents/datasets/asap-dataset").expanduser(),
        include_midi_scores=True,
        filter_fn=lambda x: x.composer == "Bach" and x.midi_performance.stem == "midi_score",
        batch_size=4,
        num_workers=10,
    )
    data_module.prepare_data()
    data_module.setup()
    size = len(data_module.train_dataloader())
    print(f"{size = }")

    for i, (scores, performances) in enumerate(data_module.train_dataloader(), start=1):
        # for i, sample in enumerate(data_module.data_train, start=1):
        print(f"{i}/{size}")

    # print(f"{performances["onset"].shape = }\t{scores.shape = }")
    # print(
    #     f"score={scores.shape}\t"
    #     f"p_onset={performances['onset'].shape}\t"
    #     f"p_duration={performances['duration'].shape}\t"
    #     f"p_pitch={performances['pitch'].shape}\t"
    #     f"p_velocity={performances['velocity'].shape}"
    # )

    # for sample in data_module.data_train:
    #     # print(sample)
    #     print(sample["entry"])

    # diffs = []
    # for _ in range(10):
    #     print(_)
    #     start = time.time()
    #     # for i, sample in enumerate(data_module.train_dataloader(), start=1):
    #     for i, sample in enumerate(data_module.data_train, start=1):
    #         # AsapTokensDataModule.collate_fn([sample])
    #         # print(f"{i}/{size}")
    #         pass
    #     diff = time.time() - start
    #     diffs.append(diff)
    #     print(f"{diff:.3f}s  [{diff/len(data_module.data_train):.3f}s/sample]")
    #
    # # fmt: off
    # print(f"Avg:  {sum(diffs)/len(diffs):.3f}s  [{sum(diffs)/len(diffs)/len(data_module.data_train):.3f}s/sample]")
    # print(f"1st:  {diffs[0]:.3f}s  [{diffs[0]/len(data_module.data_train):.3f}s/sample]")
    # print(f"Avg1: {sum(diffs[1:])/len(diffs[1:]):.3f}s  [{sum(diffs[1:])/len(diffs[1:])/len(data_module.data_train):.3f}s/sample]")
    # # fmt: on

    # for _ in range(10):
    #     start = time.time()
    #     for i, sample in enumerate(data_module.data_train, start=1):
    #         # print(f"{i}/{len(data_module.data_train)}: {sample['entry'].midi_performance}")
    #         pass
    #     diff = time.time() - start
    #     print(f"{diff:.2f}s  [{diff/len(data_module.data_train):.2f}s/sample]")
