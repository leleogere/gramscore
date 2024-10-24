from __future__ import annotations

import csv
import itertools
import json
import traceback
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import pandas as pd
import partitura as pt
from loguru import logger
from tqdm import tqdm

from data.components.score_performance_dataset import ScorePerformanceAsTextDataset
from data.components.score_performance_dataset import ScorePerformanceDataset
from data.components.score_performance_dataset import ScorePerformanceEntry
from data.components.chunking_utils import chunk_partitura_score_and_performances
from utils.constants import SKIP
from utils.constants import SUCCESS_FILENAME
from utils.constants import TEST_PIECE_IDS
from utils.constants import TO_IGNORE_INDICES

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    import partitura.score


class AsapDataset(ScorePerformanceDataset):
    """Dataset for the ASAP dataset.

    ASAP is a dataset of aligned musical scores (both MIDI and MusicXML) and performances (audio and MIDI), all with downbeat, beat, time
    signature, and key signature annotations.

    This dataset is used to load only the MIDI and MusicXML files from the dataset.
    """

    def __init__(self, data_dir: Path, split: str) -> None:
        """Initialize an `AsapDataset` (using the MIDI2ScoreTransformer method).

        :param data_dir: The directory where the ASAP dataset is stored.
        :param acpas_dir: The directory where the ACPAS metadata is stored.
        :param split: The split to use (either "all", "train", "validation", or "test").
        """
        self.data_dir = data_dir
        self.split = split
        assert split in ["all", "train", "validation", "test"], f"Invalid split: {split}"
        super().__init__(self.read_metadata_like_m2st(self.data_dir, self.split))

    @staticmethod
    def read_metadata(data_dir: Path) -> list[ScorePerformanceEntry]:
        """Read the metadata of the dataset.

        CURRENTLY UNUSED

        :param data_dir: The directory where the dataset is stored.
        :return: The metadata of the dataset.
        """
        with open(data_dir / "metadata.csv", "r") as f:
            reader = csv.DictReader(f)
            return [
                ScorePerformanceEntry(
                    # fmt: off
                    composer=entry["composer"],
                    title=entry["title"],
                    xml_score=data_dir / Path(entry["xml_score"]),
                    midi_score=data_dir / Path(entry["midi_score"]) if "midi_score" in entry else None,
                    midi_performance=data_dir / Path(entry["midi_performance"]),
                    midi_score_annotations=data_dir / Path(entry["midi_score_annotations"]) if "midi_score_annotations" in entry else None,
                    # fmt: on
                )
                for entry in reader
            ]

    @staticmethod
    def read_metadata_like_m2st(
        data_dir: Path,
        split: str,
    ) -> list[ScorePerformanceEntry]:
        """Read the metadata of the dataset with the MIDI2ScoreTransformer method.

        :param data_dir: The directory where the ASAP dataset is stored.
        :return: The metadata of the dataset.
        """
        acpas_r_file = data_dir / "metadata_R.csv"
        acpas_s_file = data_dir / "metadata_S.csv"
        assert (
            acpas_r_file.exists() and acpas_s_file.exists()
        ), "ACPAS metadata files must be present in the ASAP directory"
        data_real = pd.read_csv(acpas_r_file)
        data_synthetic = pd.read_csv(acpas_s_file)
        asap_annotations = json.load(open(data_dir / "asap_annotations.json"))
        UNALIGNED = set(
            "{ASAP}/" + k
            for k, v in asap_annotations.items()
            if not v["score_and_performance_aligned"]
        )
        # Filter
        data = pd.concat([data_real, data_synthetic])
        # Initial filtering
        data = data[(data["source"] == "ASAP") & data["aligned"]]
        data = data[~data["performance_MIDI_external"].isin(SKIP)]
        data = data[~data["performance_MIDI_external"].isin(UNALIGNED)]
        data = data.drop_duplicates(subset=["performance_MIDI_external"])
        # Filter by annotations
        data.reset_index(inplace=True)
        data.drop(TO_IGNORE_INDICES, inplace=True)

        # Split
        test_idx = data["piece_id"].isin(TEST_PIECE_IDS)
        match split:
            case "all":
                dataset = data
            case "test":
                dataset = data[test_idx]
            case "validation":
                dataset = data[
                    (data["piece_id"] % 10 == 0) & (~data["piece_id"].isin(TEST_PIECE_IDS))
                ]
            case "train":
                dataset = data[
                    (data["piece_id"] % 10 != 0) & (~data["piece_id"].isin(TEST_PIECE_IDS))
                ]
            case _:
                raise ValueError(f"Invalid split: {split}")

        entries = []
        for entry in dataset.to_dict(orient="records"):
            performance_path = Path(
                entry["performance_MIDI_external"].replace("{ASAP}", str(data_dir))
            )
            folder_path = performance_path.parent
            score_path = folder_path / "xml_score.musicxml"
            entries.append(
                ScorePerformanceEntry(
                    composer=entry["composer"],
                    title=entry["title"],
                    xml_score=score_path,
                    midi_performance=performance_path,
                )
            )

        return entries

    def chunk_nasap_dataset(
        self,
        new_dataset_path: Path,
        num_measures_per_chunk: int,
        overlap: bool = True,
        min_matched_notes_ratio: float = 0.8,
        skip_already_chunked: bool = False,
        n_jobs: int = 1,
        progress: bool = True,
    ) -> ScorePerformanceDataset:
        """
        Chunk an ASAP dataset into chunks of a given number of measures and save the new dataset to a new directory.

        :param new_dataset_path: The path to save the new dataset to.
        :param num_measures_per_chunk: The number of measures per chunk.
        :param overlap: Whether to overlap the chunks (use a sliding window).
        :param min_matched_notes_ratio: The minimum ratio of matched notes between the score and the performance to consider the alignment valid.
        :param skip_already_chunked: Whether to skip scores that have already been chunked.
        :param progress: Whether to show a progress bar.
        :param n_jobs: The number of jobs to run in parallel.
        :return: The new dataset.
        """
        # If already successfully chunked and skip is allowed, return the dataset
        success_file = new_dataset_path / SUCCESS_FILENAME
        if skip_already_chunked and success_file.exists():
            logger.info(f"Dataset already chunked at {new_dataset_path}")
            with open(new_dataset_path / "metadata.csv", "r") as f:
                reader = csv.DictReader(f)
                metadata = [
                    ScorePerformanceEntry(
                        composer=entry["composer"],
                        title=entry["title"],
                        xml_score=new_dataset_path / Path(entry["xml_score"]),
                        midi_performance=new_dataset_path / Path(entry["midi_performance"]),
                        split=self.split,
                    )
                    for entry in reader
                ]
            return ScorePerformanceDataset(metadata)
        # Otherwise remove an eventual success file (like if we want to recompute) and continue
        else:
            success_file.unlink(missing_ok=True)

        def _chunk_single_score_and_performances(
            entries: list[ScorePerformanceEntry],
        ) -> tuple[bool, Path]:
            """
            Function used to parallelize the chunking of the dataset. The entries must all have
            the same score (but different performances). Return a boolean indicating whether the
            chunking was successful, as well as the output directory of the chunk.
            """
            score_path = entries[0].xml_score
            assert all(entry.xml_score == score_path for entry in entries)

            out_dir = new_dataset_path / score_path.relative_to(self.data_dir).parent

            # Do not recompute if the score has already been chunked (and the flag is set)
            success_file = out_dir / SUCCESS_FILENAME
            if skip_already_chunked and success_file.exists():
                logger.debug(f"Skipping {score_path} (already chunked)")
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

            # Chunk the score
            try:
                chunks = chunk_partitura_score_and_performances(
                    score=score,
                    aligned_performances=performances_and_alignments,
                    num_measures_per_chunk=num_measures_per_chunk,
                    overlap=overlap,
                    min_matched_notes_ratio=min_matched_notes_ratio,
                    score_name_for_logging=str(score_path.relative_to(self.data_dir)),
                )
            except Exception as e:
                logger.error(f"Error chunking {score_path}: {e}\n{traceback.format_exc()}")
                return False, out_dir

            # Save the chunks to disk
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

        # Group entries by score as we need to have all performance for a given score together to
        # chunk them
        entries_by_score = [
            list(elements)
            for _, elements in itertools.groupby(
                sorted(self.entries, key=lambda entry: entry.xml_score),
                key=lambda entry: entry.xml_score,
            )
        ]

        # Perform the chunking in parallel
        results: Generator[tuple[tuple[bool, Path], list[ScorePerformanceEntry]]] = joblib.Parallel(
            n_jobs=n_jobs, return_as="generator"
        )(
            joblib.delayed(lambda e: (_chunk_single_score_and_performances(e), e))(ent)
            for ent in entries_by_score
        )

        # Collect all metadata and track progress
        metadata: list[ScorePerformanceEntry] = []
        with tqdm(total=len(entries_by_score), disable=not progress, unit="score") as pbar:
            for i, ((success, out_dir), ent) in enumerate(results):
                ent0 = ent[0]
                if success:
                    for chunk_dir in sorted(out_dir.iterdir()):
                        score_path = chunk_dir / ent0.xml_score.stem
                        performance_paths = list(chunk_dir.glob("*.mid"))
                        metadata.extend(
                            [
                                ScorePerformanceEntry(
                                    composer=ent0.composer,
                                    title=f"ent0.title ({chunk_dir.stem})",
                                    xml_score=score_path,
                                    midi_performance=performance_path,
                                    split=self.split,
                                )
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
            fields = metadata[0].__dict__.keys()
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for entry in metadata:
                writer.writerow(entry.__dict__)

        # Save the success file
        success_file.touch(exist_ok=True)

        return ScorePerformanceDataset(metadata)


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
    ds = AsapDataset(Path("/tmp/asap-dataset"), "test")
    ds.chunk_nasap_dataset(
        new_dataset_path=Path("/tmp/chunk_test_asap"),
        num_measures_per_chunk=8,
        overlap=True,
        skip_already_chunked=True,
        n_jobs=-1,
        progress=True,
    )
