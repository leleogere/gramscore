"""
This script creates a metadata file containing parsing information about all scores in the ASAP
dataset. The final file contains 3 columns:
    - folder: the folder of that piece, containing the score and all performances
    - valid: True or False, denoting if the score is usable or not
    - reason: only for invalid scores, a message indicating the reason of invalidity
"""

import csv
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import joblib
import lxml
import pandas as pd
import partitura as pt
import partitura.score
from loguru import logger
from lxml.etree import XMLSyntaxError
from tqdm.auto import tqdm

from models.automata.io import partitura_to_token_sequence
from models.automata.tokens import fill_rests
from models.automata.voice_reordering import reorder_voices
from scripts.cli.simpler_chunker_midi_scores import create_chunk_file_for_midi_score

sys.setrecursionlimit(2**31 - 1)


@dataclass
class Entry:
    folder: str
    _error_type: str | None = None
    _error_msg: str | None = None

    def mark_as_error(self, error_type: Any, error_msg: Any = None):
        self._error_type = str(error_type)
        self._error_msg = str(error_msg)

    @property
    def valid(self):
        return not self._error_type and not self._error_msg

    @property
    def error_type(self):
        return self._error_type

    @property
    def error_msg(self):
        return self._error_msg


def test_score(
    asap_path: Path,
    folder: Path | str,
    annotations: pd.DataFrame,
    metadata: pd.DataFrame,
) -> Entry:
    path = asap_path / folder / "xml_score.musicxml"
    entry = Entry(folder=folder)

    # Test loading the XML
    try:
        xml = lxml.etree.parse(path)
        root = xml.getroot()
    except XMLSyntaxError as e:
        entry.mark_as_error("XML syntax error", e)
        return entry

    # Test the number of parts
    num_parts = root.findall(".//part")
    if len(num_parts) != 1:
        entry.mark_as_error("Too much parts", str(len(num_parts)))
        return entry

    # Test number of staves
    staves = root.findall(".//part//staff")
    num_staves = sorted(set(int(s.text) for s in staves))
    if len(num_staves) > 2:
        entry.mark_as_error("Too many staves", str(num_staves))
        return entry

    # Test reading the score with Partitura
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = pt.load_musicxml(path, force_note_ids=True)
            score = score.parts[0]
    except Exception as e:
        entry.mark_as_error("Reading error", e)
        return entry

    # Test unfolding the score
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = pt.score.unfold_part_maximal(score, ignore_leaps=False)
    except Exception as e:
        entry.mark_as_error("Unfolding error", e)
        return entry

    # Try to fill rests
    try:
        fill_rests(score)
    except Exception as e:
        entry.mark_as_error("Rest filling error", e)
        return entry

    # Try to reorder the voices
    try:
        score = reorder_voices(score)
    except Exception as e:
        entry.mark_as_error("Voice reordering error", e)
        return entry

    # Try to create the chunk file performances
    try:
        create_chunk_file_for_midi_score(
            piece=entry.folder,
            metadata=metadata,
            annotations=annotations,
            asap_path=asap_path,
        )
    except ValueError as e:
        if "No aligned performances found" in str(e):
            entry.mark_as_error("Alignment error", "No aligned performances found")
        else:
            entry.mark_as_error("Chunking error", e)
        return entry

    # Try to tokenize the score
    try:
        tokens = partitura_to_token_sequence(score)
    except Exception as e:
        entry.mark_as_error("Tokenization error", e)
        return entry

    # # Test the validity of the sequence
    # try:
    #     automata = ConstrainedAutomata(16)
    #     automata.transition_sequence(tokens)
    # except ValueError as e:
    #     entry.mark_as_error("Invalid token sequence", e)
    #     return entry
    #
    # # Test converting the tokens sequence back into a score
    # try:
    #     regenerated_score = token_sequence_to_partitura(tokens)
    #     # pt.save_musicxml(regenerated_score, "/tmp/regenerated_score.musicxml")
    # except Exception as e:
    #     entry.mark_as_error("Detokenization error", e)
    #     return entry
    #
    # retokenized_tokens = partitura_to_token_sequence(regenerated_score)
    # if not all(
    #     token == retokenized_token for token, retokenized_token in zip(tokens, retokenized_tokens)
    # ):
    #     token, retokenized_token = next(
    #         (a, b) for a, b in zip(tokens, retokenized_tokens) if a != b
    #     )
    #     entry.mark_as_error("Detokenization inconsistency", f"{token} != {retokenized_token}")
    #     return entry

    return entry


NL = "\n"


# fmt: off
@click.command()
@click.option("asap_path", "-a", "--asap-path", type=Path, required=True, help="Path to the ASAP dataset root directory.")
@click.option("output_file_name", "-o", "--output-file-name", type=str, required=True, help="Name of the output file (will be placed in the ASAP directory).")
@click.option("n_jobs", "-j", "--njobs", type=int, help="Number of scores to test in parallel")
# fmt: on
def main(
    asap_path: Path,
    output_file_name: Path,
    n_jobs: int = 1,
    log_level: str = "INFO",
):
    # Set logging level to INFO
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    # Read metadata and annotations
    metadata = pd.read_csv(asap_path / "metadata.csv")
    annotations = pd.read_json(asap_path / "asap_annotations.json", orient="index")

    score_folders = sorted(map(lambda p: p, metadata["folder"].unique()))
    # score_folders = ['Beethoven/Piano_Sonatas/24-2']
    print(f"Found {len(score_folders)} scores in {asap_path}")

    tested_scores = joblib.Parallel(n_jobs=n_jobs, return_as="generator")(
        joblib.delayed(
            lambda f: test_score(
                folder=f,
                asap_path=asap_path,
                metadata=metadata[metadata["folder"] == f],
                annotations=annotations[annotations.index.str.startswith(f)],
            )
        )(folder)
        for folder in score_folders
    )
    results = [entry for entry in tqdm(tested_scores, total=len(score_folders))]

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
                    f"  * {entry.folder}  |  {entry.error_msg.replace(NL, ' ') if entry.error_msg else ''}"
                )

    # Export the results file as TSV
    output_file = (asap_path / output_file_name).with_suffix(".tsv")
    with output_file.open("w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["folder", "valid", "type", "msg"])
        for entry in results:
            writer.writerow(
                [
                    entry.folder,
                    entry.valid,
                    entry.error_type if not entry.valid else "",
                    entry.error_msg if not entry.valid else "",
                ]
            )


if __name__ == '__main__':
    main()
