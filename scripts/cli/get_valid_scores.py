"""
This script creates a metadata file containing parsing information about all scores in the ASAP
dataset. The final file contains 3 columns:
    - folder: the folder of that piece, containing the score and all performances
    - valid: True or False, denoting if the score is usable or not
    - reason: only for invalid scores, a message indicating the reason of invalidity
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import pickle
import warnings
import csv

import click
import pandas as pd
import partitura as pt
import partitura.score
from tqdm.auto import tqdm

from models.automata.tokens import fill_rests
from models.automata.automata import ConstrainedAutomata
from models.automata.io import token_sequence_to_partitura, partitura_to_token_sequence
from models.automata.voice_reordering import reorder_voices

from scripts.cli.simpler_chunker_midi_scores import create_chunk_file_for_midi_score


@dataclass
class Entry:
    folder: Path
    _error_type: str | None = None
    _error_msg: str | None = None

    def mark_as_error(self, error_type: str, error_msg: str = None):
        self._error_type = error_type
        self._error_msg = error_msg

    @property
    def valid(self):
        return not self._error_type and not self._error_msg

    @property
    def error_type(self):
        return self._error_type

    @property
    def error_msg(self):
        return self._error_msg


# fmt: off
@click.command()
@click.option("asap_path", "-a", "--asap-path", type=Path, required=True, help="Path to the ASAP dataset root directory.")
@click.option("output_file", "-o", "--output-file", type=Path, required=True, help="Path to the output TSV file.")
@click.option("cache_path", "-c", "--cache-path", type=Path, default=None, help="Optional path to cache the tokenized scores as pickle files.")
# fmt: on
def main(
    asap_path: Path,
    output_file: Path,
    cache_path: Path | None = None,
):
    if cache_path == "":
        cache_path = None

    metadata = pd.read_csv(asap_path / "metadata.csv")
    annotations = pd.read_json(asap_path / "asap_annotations.json", orient="index")

    def report_error(error_type: str, error_msg: str | None = None):
        pbar.write(
            f"Error processing {entry.folder.relative_to(asap_path)}: {error_type}: {error_msg}"
        )
        entry.mark_as_error(str(error_type), str(error_msg))
        results.append(entry)
        return entry

    score_paths = sorted(map(lambda p: asap_path / p, metadata["xml_score"].unique()))
    print(f"Found {len(score_paths)} scores in {asap_path}")
    results: list[Entry] = []

    for score_path in (pbar := tqdm(score_paths, position=0, leave=True)):
        entry = Entry(folder=score_path.parent)
        if cache_path is not None:
            save_path = (cache_path / score_path.relative_to(asap_path)).with_suffix(".pkl")
            if save_path.exists():
                results.append(entry)
                continue

        num_errors = len([e for e in results if not e.valid])
        pbar.set_postfix({"score": score_path, "errors": num_errors})
        automata = ConstrainedAutomata(16)

        # Test reading the score with Partitura
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = pt.load_musicxml(score_path, force_note_ids=True)
        except Exception as e:
            report_error("Reading error", e)
            continue

        # Test the number of parts
        if len(score.parts) != 1:
            report_error("Too much parts", str(len(score.parts)))
            continue
        score = score.parts[0]

        # Test unfolding the score
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = pt.score.unfold_part_maximal(score, ignore_leaps=False)
        except Exception as e:
            report_error("Unfolding error", e)
            continue

        # Try to fill rests
        try:
            # pt.score.fill_rests(score, measurewise=True)
            fill_rests(score)
        except Exception as e:
            report_error("Rest filling error", e)
            continue

        # Try to reorder the voices
        try:
            score = reorder_voices(score)
        except Exception as e:
            report_error("Voice reordering error", e)
            continue

        # Try to create the chunk file performances
        try:
            create_chunk_file_for_midi_score(
                piece=entry.folder.relative_to(asap_path).as_posix(),
                metadata=metadata,
                annotations=annotations,
                asap_path=asap_path,
            )
        except ValueError as e:
            report_error("Chunking error", e)
            continue

        # Try to tokenize the score
        try:
            tokens = partitura_to_token_sequence(score)
        except Exception as e:
            report_error("Tokenization error", e)
            continue

        # Test the validity of the sequence
        try:
            automata.transition_sequence(tokens)
        except ValueError as e:
            report_error("Invalid token sequence", e)
            continue

        # Test converting the tokens sequence back into a score
        try:
            regenerated_score = token_sequence_to_partitura(tokens)
            # pt.save_musicxml(regenerated_score, "/tmp/regenerated_score.musicxml")
        except Exception as e:
            report_error("Detokenization error", e)
            continue

        retokenized_tokens = partitura_to_token_sequence(token_sequence_to_partitura(tokens))
        if not all(
            token == retokenized_token
            for token, retokenized_token in zip(tokens, retokenized_tokens)
        ):
            token, retokenized_token = next(
                (a, b) for a, b in zip(tokens, retokenized_tokens) if a != b
            )
            report_error("Detokenization inconsistency", f"{token} != {retokenized_token}")
            continue

        if cache_path is not None:
            for token in tokens:
                if hasattr(token, "_partitura_elements"):
                    del token._partitura_elements
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(tokens, f)

        results.append(entry)

    # Display some stats about errors
    grouped_by_error = defaultdict(list)
    for entry in results:
        grouped_by_error[entry.error_type if not entry.valid else ""].append(entry)
    grouped_by_error = dict(sorted(grouped_by_error.items()))
    for error_type, entries in grouped_by_error.items():
        if not error_type:
            print(f"Found {len(entries)} scores without errors")
        else:
            print(f"Found {len(entries)} scores with {error_type}:")
            for entry in entries:
                print(
                    f"  * {entry.folder}  |  {entry.error_msg.replace('\n', ' ') if entry.error_msg else ''}"
                )

    # Export the results file as CSV
    with output_file.open("w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["folder", "valid", "type", "msg"])
        for entry in results:
            writer.writerow(
                [
                    entry.folder.relative_to(asap_path).as_posix(),
                    entry.valid,
                    entry.error_type if not entry.valid else "",
                    entry.error_msg if not entry.valid else "",
                ]
            )


if __name__ == '__main__':
    main()
