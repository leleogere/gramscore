"""
Transcribe a single performance with a trained model.
"""

from functools import partial
from pathlib import Path

import partitura as pt
import torch
from loguru import logger

import data.components.transforms as t

from data.asap_tokens_datamodule import AsapTokensDataModule
from data.components.score_performance_dataset import ScorePerformanceEntry, ScorePerformanceDataset
from models.automata.io import token_sequence_to_partitura, token_sequence_to_partitura_permissive
from models.tokens_transcriber_module import TokensTranscriberModule
from utils.generation import GenerationState

source_path = Path(
    "~/Documents/datasets/pdmx/mid/1/11/Qmbbh58jD1ZLT1vevNsnyEZkexnx2USTFz6FSHX3KUiAzC.mid"
).expanduser()
model_path = Path(
    "~/Documents/transcription/gramscore2/outputs/server/2026-09-08_17-19-56.888858_DA_tough-snowflake-1423/checkpoints/last.ckpt"
).expanduser()

# Generation parameters
CHUNK_SIZE = 500
MAX_TOKENS_PER_CHUNK = 3000
MAX_CHUNKS = 100
CONSTRAINT = True
CONSTRAINT_ITERATIONS = 10000
USE_CACHE = True

# Output directory for the transcriptions
transcription_directory = (
    Path("/tmp/transcriptions") / model_path.parent.parent.name / source_path.stem
)


if __name__ == '__main__':
    print(f"Transcribing {source_path.as_posix()}")
    print(f"       with model {model_path.as_posix()}")
    print(f"      to {transcription_directory.as_posix()}")

    # Load the model
    model: TokensTranscriberModule = TokensTranscriberModule.load_from_checkpoint(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    model.global_metrics.train(mode=False)
    model.eval()

    # Build the dataset on the single performance
    dataset = ScorePerformanceDataset(
        entries=[
            ScorePerformanceEntry(
                composer=None,
                title=source_path.stem,
                xml_score=None,
                midi_score=None,
                midi_performance=source_path,
                midi_score_annotations=None,
                base_path=source_path.parents[3],  # PDMX dataset root
            )
        ],
        transforms=[
            t.ToDict(),
            t.PerformanceReader(),
            t.PerformanceToPerformedNoteList(),
            t.PerformedNoteListVectorizer(),
            t.VectorizedPerformanceStreamsBucketizer(),
        ],
    )
    dataloader = torch.utils.data.dataloader.DataLoader(
        dataset=dataset,
        collate_fn=partial(AsapTokensDataModule.collate_fn, keep_fields="all"),
        batch_size=1,
        shuffle=None,
        num_workers=0,
    )

    # Transcribe the performance
    batch = next(iter(dataloader))
    transcribed_tokens, finished_flags = model.transcriber_model.generate(
        performance_streams_raw=batch["performance_streams_raw"],
        performance_lengths=batch["performance_length"],
        chunk_size=CHUNK_SIZE,
        max_chunks=MAX_CHUNKS,
        max_tokens_per_chunk=MAX_TOKENS_PER_CHUNK,
        constraint=CONSTRAINT,
        max_constraint_iterations=CONSTRAINT_ITERATIONS,
        use_cache=USE_CACHE,
        progress_bar=True,
    )

    # Prepare output directory and file names
    transcription_directory.mkdir(parents=True, exist_ok=True)

    performance_tokens = transcribed_tokens[0]
    flag, flag_chunk, flag_pos = finished_flags[0]
    generation_state = GenerationState(flag.item())
    file_name_base = (
        "transcribed"
        f"_constrained={CONSTRAINT}"
        f"_cache={USE_CACHE}"
        f"_flag={generation_state.name}@{flag_chunk.item()}-{flag_pos.item()}"
    )
    tokens_path = (transcription_directory / file_name_base).with_suffix(".tokens")
    trans_permissive_path = (transcription_directory / (file_name_base + "_perm")).with_suffix(
        ".musicxml"
    )
    trans_strict_path = (transcription_directory / (file_name_base + "_strict")).with_suffix(
        ".musicxml"
    )
    print(f"Generation state: {generation_state.name}")

    # Write tokens
    with tokens_path.open("w") as f:
        f.write("\n".join(map(str, performance_tokens)))

    # Write permissive transcription
    try:
        transcribed_score = token_sequence_to_partitura_permissive(performance_tokens)
        pt.save_musicxml(transcribed_score, trans_permissive_path)
    except Exception as e:
        logger.error(
            f"Error with transcription (perm.) of {transcription_directory.as_posix()}: {e}"
        )

    # Write strict transcription
    try:
        transcribed_score = token_sequence_to_partitura(performance_tokens)
        pt.save_musicxml(transcribed_score, trans_strict_path)
    except Exception as e:
        logger.error(
            f"Error with transcription (strict) of {transcription_directory.as_posix()}: {e}"
        )

    print("Transcription finished!")
