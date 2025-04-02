"""
Create a chunk file that contains a mapping from XML score measures to MIDI played notes.

Only works with quantified midi_score.mid files for now.
"""

import copy
import json
import math
import warnings
from collections import UserList
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal
from typing import Self

from loguru import logger
import pandas as pd
import partitura as pt
import partitura.score
import symusic
from tqdm import tqdm


class DownbeatScoreMap(UserList):
    def __init__(self, raw_score_map: list[int | str]):
        # Change the score map to a list of lists of consecutive integers
        # We need consecutive as the repetitions will be unfolded to we want to convert
        # 0, 1, 2, 0, 1, 2, 3, 4, 5... into 0, 1, 2, 3, 4, 5, 6, 7, 8...
        score_map = []
        # Start with first index available (not 0 as there can be anacrusis, see later)
        if isinstance(raw_score_map[0], int):
            i = raw_score_map[0]
        elif isinstance(raw_score_map[0], str):
            i = int(raw_score_map[0].split("-")[0])
        else:
            raise ValueError(f"Invalid type in score map: {type(raw_score_map[0])}")
        # Increment the index for element in each group
        for group in raw_score_map:
            if isinstance(group, str):
                num_elements = len(group.split("-"))
                score_map.append(list(range(i, i + num_elements)))
                i += num_elements
            elif isinstance(group, int):
                score_map.append([i])
                i += 1
            else:
                raise ValueError(f"Invalid type in score map: {type(group)}")
            super().__init__(score_map)
            self.is_default = False

    @property
    def num_groups(self) -> int:
        return len(self)

    @property
    def num_measures(self) -> int:
        return sum(len(group) for group in self)

    @classmethod
    def create_default(cls, num_measures: int) -> Self:
        # Create a default score map with one group per measure
        instance = cls(list(range(num_measures)))
        instance.is_default = True
        return instance


def check_if_last_measure_empty_or_all_tied(
    score: pt.score.Part,
    num_measures: int = 1,
) -> tuple[bool, Literal["empty", "tied"] | None]:
    last_measure_notes: list[pt.score.Note] = list(
        score.iter_all(
            cls=pt.score.Note,
            start=score.measures[-num_measures].start,
            include_subclasses=True,  # For grace notes
        )
    )
    if len(last_measure_notes) == 0:
        return True, "empty"
    if all(note.tie_prev is not None for note in last_measure_notes):
        return True, "tied"
    return False, None


def create_chunk_file_for_midi_score(
    piece: str,
    metadata: pd.DataFrame,
    annotations: pd.DataFrame,
    asap_path: Path,
):
    # Get metadata and annotations for the piece
    # As there are multiple performances for one piece, try to take the first one which do have
    # the "downbeats_score_map" field filled
    performances = metadata.loc[metadata["folder"] == piece]
    for _, performance in performances.iterrows():
        performance_annotations = annotations.loc[performance["midi_performance"]]
        if performance_annotations["downbeats_score_map"] is not None:
            break

    # Get the midi score downbeats
    midi_score_downbeats = performance_annotations["midi_score_downbeats"]

    # Get the downbeats score map
    if performance_annotations["downbeats_score_map"] is not None:
        downbeats_score_map = DownbeatScoreMap(performance_annotations["downbeats_score_map"])
    else:
        logger.warning(
            f"{piece}: No downbeats_score_map found for {piece}, will use a default one."
        )
        downbeats_score_map = DownbeatScoreMap.create_default(len(midi_score_downbeats))

    # In case of anacrusis, add downbeats at the start to take into account the missing first bar
    if not math.isclose(midi_score_downbeats[0], 0.0):
        midi_score_downbeats.insert(0, 0.0)
        if downbeats_score_map.is_default:
            # For default, simply append at a new downbeat at the end as already starting at 0
            downbeats_score_map.append([len(downbeats_score_map)])
        else:
            # For non-default, we need to insert a 0 at the start as the anacrusis is not taken
            # into account in the score map
            assert (
                downbeats_score_map[0][0] != 0
            ), "First downbeat in score map should not be 0 as there is an anacrusis"
            downbeats_score_map.insert(0, list(range(downbeats_score_map[0][0])))

    # Load the XML score
    xml_score_path = asap_path / performance["xml_score"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xml_score: pt.score.Score = pt.load_musicxml(xml_score_path, force_note_ids=True)
        assert len(xml_score.parts) == 1
        xml_score: pt.score.Part = xml_score.parts[0]
        xml_score: pt.score.Part = pt.score.unfold_part_maximal(xml_score, ignore_leaps=False)
    num_xml_measures = len(xml_score.measures)

    # Debug some information
    logger.debug(f"{piece}: Num XML measures: {num_xml_measures}")
    logger.debug(f"{piece}: Num MIDI downbeats: {len(midi_score_downbeats)}")
    logger.debug(
        f"{piece}: Num groups in map: "
        f"{downbeats_score_map.num_groups if not downbeats_score_map.is_default else 'DEFAULT'}"
    )
    logger.debug(
        f"{piece}: Num measures in map: "
        f"{downbeats_score_map.num_measures if not downbeats_score_map.is_default else 'DEFAULT'}"
    )

    # Check consistency between the number of downbeats, measures and groups, and try to fix it
    if num_xml_measures != downbeats_score_map.num_measures:
        difference = num_xml_measures - downbeats_score_map.num_measures
        # If difference of 1 or 2, check if that's because last measure empty or all tied
        if difference == 1 or difference == 2:  # Max offset of 2 found in ASAP
            no_onset, reason = check_if_last_measure_empty_or_all_tied(
                xml_score,
                num_measures=difference,
            )
            if no_onset:
                reason = "Empty last measure" if (reason == "empty") else "Last measure all tied"
                logger.warning(f"{piece}: {reason}, adding it to the score map.")
                for i in range(0, difference, -1):
                    downbeats_score_map.append([num_xml_measures - 1 - i])
            else:
                raise ValueError(
                    f"Number of measures in XML score ({num_xml_measures}) close to number of "
                    f"measures in score map ({downbeats_score_map.num_measures}), but last measure "
                    f"of the score contains onsets so not fixable."
                )
        else:
            raise ValueError(
                f"Number of measures in XML score ({num_xml_measures}) does not match number of "
                f"measures in score map ({downbeats_score_map.num_measures}). Shouldn't happen for "
                f"non-default maps in ASAP, here we have default={downbeats_score_map.is_default}."
            )

    # Read the MIDI score (and merge tracks into the first one if needed)
    midi_score_path = asap_path / performance["midi_score"]
    midi_score = symusic.Score(midi_score_path, ttype="second")
    track = midi_score.tracks[0]
    if len(midi_score.tracks) > 1:
        for other in midi_score.tracks[1:]:
            track.notes.extend(other.notes)
        track.sort(inplace=True)

    # Build the final JSON, assigning the XML measures to the index of the MIDI notes
    threshold = 0.01  # seconds
    indices_and_notes = enumerate(track.notes)
    i, note = next(indices_and_notes, (None, None))
    mapping = defaultdict(list)
    # Assign notes to their group
    for next_db, xml_measures in zip(midi_score_downbeats, downbeats_score_map):
        # Take all notes before the next downbeat, and assign them to the current measure (or group
        # of measures)
        notes_in_current_group = []
        while note is not None and note.start < next_db - threshold:
            notes_in_current_group.append(i)
            i, note = next(indices_and_notes, (None, None))
        xml_measures_str = "-".join(str(m) for m in xml_measures)
        mapping[xml_measures_str] = notes_in_current_group
    # Take care of the last group if needed
    if note is not None:
        last_measure = [i]
        for i, note in indices_and_notes:
            last_measure.append(i)
        xml_measures_str = "-".join(str(m) for m in downbeats_score_map[-1])
        mapping[xml_measures_str] = last_measure

    logger.trace(piece + ":\n" + "\n".join(f"{k}: {len(v)} notes" for k, v in mapping.items()))

    # Save the mapping to a JSON file
    chunk_file_path = midi_score_path.with_name(f"{midi_score_path.stem}_chunks.json")
    with open(chunk_file_path, "w") as f:
        json.dump(mapping, f, indent=2)


def create_chunk_file_for_all_midi_scores(
    metadata: pd.DataFrame,
    annotations: pd.DataFrame,
    asap_path: Path,
):
    # Get all pieces
    pieces = metadata["folder"].unique()

    for piece in tqdm(pieces):
        try:
            create_chunk_file_for_midi_score(
                piece=piece,
                metadata=metadata,
                annotations=annotations,
                asap_path=asap_path,
            )
        except Exception as e:
            logger.error(f"Error processing {piece}: {e}")
            continue
            # raise


if __name__ == '__main__':
    import sys

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")

    asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
    metadata = pd.read_csv(asap_path / "metadata.csv")
    # metadata = metadata[metadata["folder"] == "Scriabin/Sonatas/5"]
    # metadata = metadata[metadata["folder"] == "Beethoven/Piano_Sonatas/17-1"]
    # metadata = metadata[metadata["folder"] == "Schumann/Kreisleriana/7"]
    # metadata = metadata[metadata["folder"] == "Bach/Fugue/bwv_865"]
    annotations = pd.read_json(asap_path / "asap_annotations.json", orient="index")
    create_chunk_file_for_all_midi_scores(
        metadata=metadata,
        annotations=annotations,
        asap_path=asap_path,
    )
