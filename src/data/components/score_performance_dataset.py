import csv
import itertools
import json
import traceback
import warnings
from collections.abc import Generator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import music21 as m21
import pandas as pd
import partitura as pt
import partitura.score
import symusic
import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from data.components.chunking_utils import ChunkingConfig
from data.components.chunking_utils import chunk_partitura_score_and_performances
from utils.abc_xml_conversion import conversion
from utils.constants import SKIP
from utils.constants import SUCCESS_FILENAME
from utils.constants import TEST_PIECE_IDS
from utils.constants import TO_IGNORE_INDICES


class Modality(StrEnum):
    TEXT = "text"


@dataclass
class ScorePerformanceEntry:
    xml_score: Path
    midi_performance: Path
    midi_score_annotations: Path = None
    midi_score: Path = None
    composer: str = None
    title: str = None
    split: str = None

    def get_score_as_musicxml(self) -> str:
        """
        Return the MusicXML file as a string.
        """
        with open(self.xml_score, "r") as f:
            return f.read()

    def get_score_as_abc(self):
        """
        Return the MusicXML file converted into ABC format.
        """
        abc = conversion.xmlfile2abc(self.xml_score)
        return abc

    def get_score_as_music21(self):
        """
        Return the MusicXML file as a music21 score.
        """
        score = m21.converter.parse(self.xml_score)
        return score

    def get_midi_performance_as_symusic(self, ttype: str = "second"):
        """
        Return the MIDI performance as a symusic object.

        :param ttype: The time type to use. Can be "tick", "quarter" or "second".
        """
        midi = symusic.Score(self.midi_performance, ttype=ttype)
        return midi


class ScorePerformanceDataset(Dataset):
    """Dataset for a dataset of pairs of MusicXML scores and MIDI performances."""

    def __init__(
        self,
        entries: list[ScorePerformanceEntry],
        modality: Modality,
        text_tokenizer: PreTrainedTokenizerBase = None,
    ) -> None:
        """Initialize a dataset of pairs of MusicXML scores and MIDI performances.

        :param entries: A list of ScorePerformanceEntry objects.
        :param modality: The modality of the dataset.
        :param text_tokenizer: Tokenizer for the text modality (not used for other modalities).
        """
        self.entries = entries
        self.modality = modality
        if not self.modality in Modality:
            raise ValueError(f"Invalid modality: {self.modality}")
        self.text_tokenizer = text_tokenizer

        if self.modality == Modality.TEXT and self.text_tokenizer is None:
            raise ValueError("A text tokenizer must be provided for text modality.")
        elif self.modality != Modality.TEXT and self.text_tokenizer is not None:
            logger.warning(
                f"A text tokenizer was provided, but it is not used for modality {self.modality}."
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        :return: The number of samples in the dataset.
        """
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a sample from the dataset.

        :param idx: The index of the sample to return.
        :return: The sample at the given index.
        """
        entry = self.entries[idx]
        match self.modality:

            # Text representation of music, suitable for language models
            case Modality.TEXT:
                # Load and convert MIDI performance to text (sequence of notes)
                midi = symusic.Score(entry.midi_performance, ttype="second")
                midi_as_text = "\n".join(
                    [
                        f"Note(pitch={note.pitch}, start={round(note.start, 3)}, duration={round(note.duration, 3)})"  # Velocity unused for now
                        for note in midi.tracks[0].notes
                    ]
                )

                # Load and convert MusicXML score to text (ABC+)
                score_as_text = conversion.xmlfile2abc(entry.xml_score)

                return {
                    "performance": midi_as_text,
                    "score": score_as_text,
                }

            case _:
                raise ValueError(f"Invalid modality: {self.modality}")

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate function to process a batch of samples.

        :param batch: A list of samples.
        :return: A dictionary containing the processed batch:
            - for text modality:
                - "input_ids": token ids of the input text (batch_size, seq_len)
                - "attention_mask": attention mask of the input text (batch_size, seq_len)
                - "assistant_masks": mask indicating assistant tokens (batch_size, seq_len)
                - "first_assistant_token": index of the first assistant token in each sequence (batch_size)
        """
        match self.modality:

            case Modality.TEXT:
                conversation = self.text_tokenizer.apply_chat_template(
                    [
                        [
                            dict(role="user", content=entry["performance"]),
                            dict(role="assistant", content=entry["score"]),
                        ]
                        for entry in batch
                    ],
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                conversation["assistant_masks"] = torch.Tensor(conversation["assistant_masks"])
                # Get the index of the first assistant token in each sequence
                conversation["first_assistant_token"] = torch.argmax(
                    conversation["assistant_masks"],
                    dim=1,
                )
                # Set to -1 for sequences without assistant tokens
                conversation["first_assistant_token"][
                    conversation["first_assistant_token"] == 0  # don't happen for non-empty MIDI
                ] = -1
                return conversation

            case _:
                raise ValueError(f"Invalid modality: {self.modality}")


class AsapDataset(ScorePerformanceDataset):
    """Dataset for the ASAP dataset.

    ASAP is a dataset of aligned musical scores (both MIDI and MusicXML) and performances (audio and MIDI), all with downbeat, beat, time
    signature, and key signature annotations.

    This dataset is used to load only the MIDI and MusicXML files from the dataset.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        modality: Modality,
        text_tokenizer: PreTrainedTokenizerBase = None,
    ) -> None:
        """Initialize an `AsapDataset` (using the MIDI2ScoreTransformer method).

        :param data_dir: The directory where the ASAP dataset is stored.
        :param split: The split to use (either "all", "train", "validation", or "test").
        :param modality: The modality of the dataset.
        :param text_tokenizer: Tokenizer for the text modality (not used for other modalities).
        """
        self.data_dir = data_dir
        self.split = split
        assert split in ["all", "train", "validation", "test"], f"Invalid split: {split}"
        super().__init__(
            self.read_metadata_like_m2st(self.data_dir, self.split),
            modality=modality,
            text_tokenizer=text_tokenizer,
        )

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

    def chunk_nasap_dataset(self, chunking_config: ChunkingConfig) -> ScorePerformanceDataset:
        """
        Chunk an ASAP dataset into chunks of a given number of measures and save the new dataset to a new directory.

        :param chunking_config: The configuration for chunking the dataset.
        :return: The new dataset.
        """
        new_dataset_path = chunking_config.new_dataset_path
        num_measures_per_chunk = chunking_config.num_measures_per_chunk
        overlap = chunking_config.overlap
        min_matched_notes_ratio = chunking_config.min_matched_notes_ratio
        skip_already_chunked = chunking_config.skip_already_chunked
        n_jobs = chunking_config.n_jobs
        progress = chunking_config.progress

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
            return ScorePerformanceDataset(
                metadata,
                modality=self.modality,
                text_tokenizer=self.text_tokenizer,
            )
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
            success_file_local = out_dir / SUCCESS_FILENAME
            if skip_already_chunked and success_file_local.exists():
                logger.debug(f"Skipping {score_path} (already chunked)")
                return True, out_dir
            else:
                logger.debug(f"Chunking {score_path}")
                success_file_local.unlink(missing_ok=True)

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
                        chunk_dir / f"{performance['performance'].id}.mid",
                    )
                    try:
                        pt.save_match(
                            alignment=performance["alignment"],
                            performance_data=performance["performance"],
                            score_data=chunk["score"],
                            out=chunk_dir / f"{performance['performance'].id}.match",
                            mpq=performance["performance"].mpq,
                            ppq=performance["performance"].ppq,
                            assume_unfolded=True,
                        )
                    except ValueError as e:
                        logger.error(
                            f"Error saving match of score {score_path} and performance {performance['performance'].id}: {e}"
                        )
                        raise e

            # Create a success file (only if there are chunks)
            if len(chunks) > 0:
                success_file_local.touch()
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
            n_jobs=n_jobs,
            return_as="generator",
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
                                    title=f"{ent0.title} ({chunk_dir.stem})",
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

        return ScorePerformanceDataset(
            entries=metadata,
            modality=self.modality,
            text_tokenizer=self.text_tokenizer,
        )


if __name__ == '__main__':
    from src.data.components.patched_tokenizer import PatchedTokenizer

    tokenizer = PatchedTokenizer.from_pretrained("mistralai/Codestral-22B-v0.1")
    ds = AsapDataset(Path("data/nasap"), "test", modality=Modality.TEXT, text_tokenizer=tokenizer)
    print(ds[0]["performance"])
    print(ds[0]["score"])

    res = ds.collate_fn([ds[0], ds[1], ds[2]])
    # breakpoint()
    ds.chunk_nasap_dataset(
        ChunkingConfig(
            new_dataset_path=Path("/tmp/chunk_test_asap"),
            num_measures_per_chunk=8,
            overlap=True,
            skip_already_chunked=True,
            n_jobs=-1,
            progress=True,
        )
    )
