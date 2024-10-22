from __future__ import annotations

import csv
import itertools
import traceback
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import partitura as pt
from loguru import logger
from tqdm import tqdm

from data.components.score_performance_dataset import ScorePerformanceAsTextDataset
from data.components.score_performance_dataset import ScorePerformanceDataset
from data.components.score_performance_dataset import ScorePerformanceEntry
from data.split_utils import split_partitura_score_and_performances
from utils.constants import SUCCESS_FILENAME

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    import partitura.score


class AsapDataset(ScorePerformanceDataset):
    """Dataset for the ASAP dataset.

    ASAP is a dataset of aligned musical scores (both MIDI and MusicXML) and performances (audio and MIDI), all with downbeat, beat, time
    signature, and key signature annotations.

    This dataset is used to load only the MIDI and MusicXML files from the dataset.
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize an `AsapDataset`.

        :param data_dir: The directory where the dataset is stored.
        """
        self.data_dir = data_dir
        super().__init__(self.read_metadata(self.data_dir))

    @staticmethod
    def read_metadata(data_dir: Path) -> list[ScorePerformanceEntry]:
        """Read the metadata of the dataset.

        :param data_dir: The directory where the dataset is stored.
        :return: The metadata of the dataset.
        """
        with open(data_dir / "metadata.csv", "r") as f:
            reader = csv.DictReader(f)
            return [
                ScorePerformanceEntry(
                    composer=entry["composer"],
                    title=entry["title"],
                    xml_score=data_dir / Path(entry["xml_score"]),
                    midi_score=data_dir / Path(entry["midi_score"]),
                    midi_performance=data_dir / Path(entry["midi_performance"]),
                    midi_score_annotations=data_dir / Path(entry["midi_score_annotations"]),
                )
                for entry in reader
            ]

    def split_asap_dataset(
        self,
        new_dataset_path: Path,
        num_measures_per_chunk: int,
        overlap: bool = True,
        min_matched_notes_ratio: float = 0.8,
        skip_already_split: bool = False,
        n_jobs: int = 1,
        progress: bool = True,
    ) -> AsapDataset:
        """
        Split an ASAP dataset into chunks of a given number of measures and save the new dataset to a new directory.

        :param new_dataset_path: The path to save the new dataset to.
        :param num_measures_per_chunk: The number of measures per chunk.
        :param overlap: Whether to overlap the chunks (use a sliding window).
        :param min_matched_notes_ratio: The minimum ratio of matched notes between the score and the performance to consider the alignment valid.
        :param skip_already_split: Whether to skip scores that have already been split.
        :param progress: Whether to show a progress bar.
        :param n_jobs: The number of jobs to run in parallel.
        :return: The new dataset.
        """
        success_file = new_dataset_path / SUCCESS_FILENAME
        if skip_already_split and success_file.exists():
            logger.info(f"Dataset already split at {new_dataset_path}")
            return AsapDataset(new_dataset_path)
        else:
            success_file.unlink(missing_ok=True)

        def _split_single_score_and_performances(
            entries: list[ScorePerformanceEntry],
        ) -> tuple[bool, Path]:
            """
            Function used to parallelize the splitting of the dataset. The entries must all have
            the same score (but different performances). Return a boolean indicating whether the
            splitting was successful, as well as the output directory of the split.
            """
            score_path = entries[0].xml_score
            assert all(entry.xml_score == score_path for entry in entries)

            out_dir = new_dataset_path / score_path.relative_to(self.data_dir).parent

            # Do not recompute if the score has already been split (and the flag is set)
            success_file = out_dir / SUCCESS_FILENAME
            if skip_already_split and success_file.exists():
                logger.debug(f"Skipping {score_path} (already split)")
                return True, out_dir
            else:
                success_file.unlink(missing_ok=True)

            # Load the score
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    score = pt.load_musicxml(score_path)
                    assert len(score.parts) == 1
                    score = pt.score.unfold_part_maximal(score.parts[0], ignore_leaps=False)
            except Exception as e:
                logger.error(f"Error loading {score_path}: {e}")
                return False, out_dir

            # Load the aligned performances
            performances_and_alignments = []
            for entry in entries:
                alignment_path = entry.midi_performance.with_suffix(".match")
                try:
                    performance, alignment = pt.load_match(alignment_path)
                    assert len(performance.performedparts) == 1
                    performance = performance.performedparts[0]
                    performance.id = entry.midi_performance.stem
                    performances_and_alignments.append((performance, alignment))
                except Exception as e:
                    logger.error(f"Error loading {alignment_path}: {e}")

            # Split the score
            try:
                chunks = split_partitura_score_and_performances(
                    score=score,
                    aligned_performances=performances_and_alignments,
                    num_measures_per_chunk=num_measures_per_chunk,
                    overlap=overlap,
                    min_matched_notes_ratio=min_matched_notes_ratio,
                    score_name_for_logging=str(score_path.relative_to(self.data_dir)),
                )
            except Exception as e:
                logger.error(f"Error splitting {score_path}: {e}\n{traceback.format_exc()}")
                return False, out_dir

            # Save the split to disk
            for i, chunk in enumerate(chunks, start=1):
                m_start, m_end = (
                    (i, i + num_measures_per_chunk - 1)
                    if overlap
                    else (i * num_measures_per_chunk, (i + 1) * num_measures_per_chunk - 1)
                )
                chunk_dir = out_dir / f"M{m_start}-{m_end}"
                chunk_dir.mkdir(exist_ok=True, parents=True)
                name = f"{score_path.stem}.musicxml"
                out_path = chunk_dir / name
                pt.save_musicxml(chunk["score"], out_path)
                for performance in chunk["performances"]:
                    pt.save_performance_midi(
                        performance["performance"],
                        chunk_dir / f"{performance["performance"].id}.mid",
                    )
                    pt.save_match(
                        alignment=performance["alignment"],
                        performance_data=performance["performance"],
                        score_data=chunk["score"],
                        out=chunk_dir / f"{performance["performance"].id}.match",
                        mpq=performance["performance"].mpq,
                        ppq=performance["performance"].ppq,
                        assume_unfolded=True,
                    )

            # Create a success file (only if there are chunks)
            if len(chunks) > 0:
                success_file.touch()
            else:
                return False, out_dir

            return True, out_dir

        entries_by_score = [
            list(elements)
            for _, elements in itertools.groupby(
                sorted(self.entries, key=lambda entry: entry.xml_score),
                key=lambda entry: entry.xml_score,
            )
        ]
        results: Generator[tuple[tuple[bool, Path], ScorePerformanceEntry]] = joblib.Parallel(
            n_jobs=n_jobs, return_as="generator"
        )(
            joblib.delayed(lambda e: (_split_single_score_and_performances(e), e))(ent)
            for ent in entries_by_score
        )

        metadata: list[dict] = []
        with tqdm(total=len(entries_by_score), disable=not progress, unit="score") as pbar:
            for i, ((success, out_dir), ent) in enumerate(results):
                if success:
                    for chunk_dir in sorted(out_dir.iterdir()):
                        score_path = chunk_dir / ent.xml_score.stem
                        performance_paths = list(chunk_dir.glob("*.mid"))
                        metadata.extend(
                            [
                                {
                                    "composer": ent.composer,
                                    "title": ent.title,
                                    "folder": score_path.relative_to(new_dataset_path).parent,
                                    "xml_score": score_path.relative_to(new_dataset_path),
                                    "midi_performance": performance_path.relative_to(
                                        new_dataset_path
                                    ),
                                }
                                for performance_path in performance_paths
                            ]
                        )
                if i % 10 == 0:
                    logger.info(f"Processed {i}/{len(entries_by_score)} scores")
                pbar.update(1)
            # Last update
            if i % 10 != 0:
                logger.info(f"Processed {i}/{len(entries_by_score)} scores")

        # Save the metadata
        metadata_path = new_dataset_path / "metadata.csv"
        with open(metadata_path, "w") as f:
            writer = csv.DictWriter(
                f,
                metadata[0].keys(),
                delimiter=",",
            )
            writer.writeheader()
            writer.writerows(metadata)
        
        # Save the success file
        success_file.touch(exist_ok=True)
        
        return AsapDataset(new_dataset_path)


class AsapAsTextDataset(ScorePerformanceAsTextDataset):
    """Dataset for the ASAP dataset, formatted to be used with a language model.

    The dataset returns a dictionary with the following keys:
        - `midi_performance`: A text sequence of events in the MIDI performance.
        - `abc_score`: The corresponding ABC sequence.
    """

    def __init__(self, data_dir: Path, tokenizer: PreTrainedTokenizerBase) -> None:
        """Initialize an `AsapTextABCDataset`.

        :param data_dir: The directory where the dataset is stored.
        :param tokenizer: The tokenizer to use.
        """
        # Probably not the cleanest implementation, maybe using multiple inheritance or mixins to
        # inherit from both AsapDataset and ScorePerformanceAsTextDataset would be better.
        self.data_dir = data_dir
        super().__init__(AsapDataset.read_metadata(self.data_dir), tokenizer)


if __name__ == '__main__':
    ds = AsapDataset(Path("/tmp/asap-dataset"))
    ds.split_asap_dataset(
        new_dataset_path=Path("/tmp/split_asap"),
        num_measures_per_chunk=8,
        overlap=True,
        skip_already_split=True,
        n_jobs=-1,
        progress=True,
    )
