"""
Transcribe a single MIDI performance into a MusicXML score.

Example:
    uv run python scripts/transcribe.py --model weights/.../last.ckpt --output transcription.musicxml performance.mid
"""

import sys
from pathlib import Path

# Make the project importable when running this script directly without requiring PYTHONPATH to be set.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import click
import partitura as pt
import torch
from loguru import logger

import data.components.transforms as t
from data.asap_tokens_datamodule import AsapTokensDataModule
from data.components.score_performance_dataset import (
    ScorePerformanceDataset,
    ScorePerformanceEntry,
)
from models.automata.io import (
    token_sequence_to_partitura,
    token_sequence_to_partitura_permissive,
)
from models.tokens_transcriber_module import TokensTranscriberModule
from utils.generation import GenerationState


# fmt: off
@click.command()
@click.argument("performance", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-m", "--model", "model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Path to the checkpoint (.ckpt) of a trained TokensTranscriberModule.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Path of the MusicXML file to write (default: <performance_stem>_transcription.musicxml).")
@click.option("--chunk-size", type=int, default=500, show_default=True, help="Number of performed notes transcribed per chunk.")
@click.option("--max-chunks", type=int, default=100, show_default=True, help="Maximum number of chunks generated for the performance.")
@click.option("--max-tokens-per-chunk", type=int, default=3000, show_default=True, help="Maximum number of tokens generated per chunk.")
@click.option("--overlap", type=int, default=1, show_default=True, help="Number of measures overlapped between consecutive chunks, used for realignment.")
@click.option("--constrained", is_flag=True, default=False, show_default=True, help="Use constrained (automaton-guided) decoding.")
@click.option("--max-constraint-iterations", type=int, default=10000, show_default=True, help="Maximum number of search iterations used by the constrained decoder (ignored otherwise).")
@click.option("--strict", is_flag=True, default=False, show_default=True, help="Use the strict token-to-MusicXML converter (fails on malformed sequences, instead of the default permissive one).")
@click.option("--tokens-output", type=click.Path(dir_okay=False, path_type=Path), default=None, help="If set, also save the generated token sequence to this path.")
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]), default="cpu", show_default=True, help="Device on which to run the model.")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default="INFO", show_default=True, help="Logging level.")
# fmt: on
def transcribe(
    performance: Path,
    model_path: Path,
    output: Path | None,
    chunk_size: int,
    max_chunks: int,
    max_tokens_per_chunk: int,
    overlap: int,
    constrained: bool,
    max_constraint_iterations: int,
    strict: bool,
    tokens_output: Path | None,
    device: str,
    log_level: str,
) -> None:
    """Transcribe a MIDI performance into a MusicXML score."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    if output is None:
        output = performance.with_name(f"{performance.stem}_transcription.musicxml")

    click.echo(f"Transcribing {performance.as_posix()} with {model_path.as_posix()} ...")

    # Load the model
    model: TokensTranscriberModule = TokensTranscriberModule.load_from_checkpoint(
        model_path,
        map_location=device,
        weights_only=False,
    )
    model.global_metrics.train(mode=False)
    model.eval()

    # Load and vectorize the performance
    entry = ScorePerformanceEntry(xml_score=None, midi_performance=performance)
    dataset = ScorePerformanceDataset(
        entries=[entry],
        transforms=[
            t.ToDict(),
            t.PerformanceReader(),
            t.PerformanceToPerformedNoteList(),
            t.PerformedNoteListVectorizer(),
            t.VectorizedPerformanceStreamsBucketizer(),
        ],
        exception_on_transform_error=True,
    )
    batch = AsapTokensDataModule.collate_fn([dataset[0]], keep_fields="all")
    num_notes = int(batch["performance_length"].item())
    click.echo(f"Performance contains {num_notes} notes.")

    # Generate the transcription
    with torch.inference_mode():
        transcribed_tokens, finished_flags = model.transcriber_model.generate(
            performance_streams_raw=batch["performance_streams_raw"],
            performance_lengths=batch["performance_length"],
            measures_interchunk_overlap=overlap,
            chunk_size=chunk_size,
            max_chunks=max_chunks,
            max_tokens_per_chunk=max_tokens_per_chunk,
            constraint=constrained,
            max_constraint_iterations=max_constraint_iterations if constrained else None,
            use_cache=True,
            progress_bar=True,
        )

    tokens = transcribed_tokens[0]
    flag, flag_chunk, flag_pos = finished_flags[0].tolist()
    state = GenerationState(flag)
    if state is not GenerationState.FINISHED:
        logger.warning(
            f"Transcription did not finish cleanly: {state.name} "
            f"(chunk {flag_chunk}, position {flag_pos}). The output may be incomplete."
        )

    # Convert the token sequence to a score
    try:
        if strict:
            score = token_sequence_to_partitura(tokens)
        else:
            score = token_sequence_to_partitura_permissive(tokens)
    except Exception as e:
        message = f"Failed to convert the token sequence to a score: {e}."
        if strict:
            message += " You can retry without --strict to use the permissive converter."
        logger.error(message)
        raise SystemExit(1)

    # Save the score
    output.parent.mkdir(parents=True, exist_ok=True)
    pt.save_musicxml(score, output)
    click.echo(f"Transcription saved to {output.as_posix()}")

    # Save the token sequence if requested
    if tokens_output is not None:
        tokens_output.parent.mkdir(parents=True, exist_ok=True)
        with tokens_output.open("w") as f:
            f.write("\n".join(map(str, tokens)))
        click.echo(f"Token sequence saved to {tokens_output.as_posix()}")


if __name__ == "__main__":
    transcribe()
