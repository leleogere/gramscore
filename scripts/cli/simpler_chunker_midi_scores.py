"""
Create a chunk file that contains a mapping from XML score measures to MIDI played notes.

Only works with quantified midi_score.mid files for now.
"""

import itertools
import json
import math
import warnings
from collections import Counter
from collections import UserList
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Literal
from typing import Self

import pandas as pd
import partitura as pt
import partitura.score
import symusic
from loguru import logger
from tqdm import tqdm

from utils.midi import merge_into_single_track
from utils.midi import midi_pitch_to_note_name


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
    # Get all performances for this piece
    performances_metadata = metadata.loc[metadata["folder"] == piece]
    performances_annotations = [
        annotations.loc[p["midi_performance"]] for _, p in performances_metadata.iterrows()
    ]

    # Try to find a performance with a score downbeats map
    # As there are multiple performances for one piece, try to take the first one which does have
    # the "downbeats_score_map" field filled
    # QUESTION: do we need to check all performances or they share the same map?
    for (_, performance_metadata), performance_annotations in zip(
        performances_metadata.iterrows(), performances_annotations
    ):
        if performance_annotations["downbeats_score_map"] is not None:
            break

    # Get the downbeat lists
    midi_score_downbeats = performance_annotations["midi_score_downbeats"]
    performances_downbeats = [p["performance_downbeats"] for p in performances_annotations]
    assert all(len(p) == len(midi_score_downbeats) for p in performances_downbeats), (
        f"{piece}: Number of downbeats in performances ({len(performances_downbeats)}) "
        f"does not match number of downbeats in MIDI score ({len(midi_score_downbeats)})."
    )

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
        for p in performances_downbeats:
            p.insert(0, 0.0)
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
    xml_score_path = asap_path / performance_metadata["xml_score"]
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
                for i in range(difference, 0, -1):
                    downbeats_score_map.append([num_xml_measures - i])
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

    # Create chunks for the MIDI score
    midi_score_path = asap_path / performance_metadata["midi_score"]
    midi_score = merge_into_single_track(midi=symusic.Score(midi_score_path, ttype="second"))
    midi_score_mapping = assign_midi_score_notes_to_chunks(
        midi_score=midi_score,
        midi_score_downbeats=midi_score_downbeats,
        downbeats_score_map=downbeats_score_map,
    )
    logger.trace(
        piece + ":\n" + "\n".join(f"{k}: {len(v)} notes" for k, v in midi_score_mapping.items())
    )
    # Save the mapping to a JSON file (remove the pitches before saving as not needed)
    midi_score_mapping_indices = {
        measures: [i for i, _ in group] for measures, group in midi_score_mapping.items()
    }
    chunk_file_path = midi_score_path.with_name(f"{midi_score_path.stem}_chunks.json")
    with open(chunk_file_path, "w") as f:
        json.dump(midi_score_mapping_indices, f, indent=2)

    # Create chunks for the performances
    for (_, p_metadata), p_downbeats in zip(
        performances_metadata.iterrows(), performances_downbeats
    ):
        p_path = asap_path / p_metadata["midi_performance"]
        p = merge_into_single_track(midi=symusic.Score(p_path, ttype="second"))
        performance_mapping = assign_performance_notes_to_chunks(
            performance=p,
            midi_score=midi_score,
            performance_downbeats=p_downbeats,
            midi_score_downbeats=midi_score_downbeats,
            downbeats_score_map=downbeats_score_map,
            midi_score_mapping=midi_score_mapping,
        )
        logger.trace(
            piece
            + ":\n"
            + "\n".join(f"{k}: {len(v)} notes" for k, v in performance_mapping.items())
        )
        # Save the mapping to a JSON file (remove the pitches before saving as not needed)
        performance_mapping_indices = {
            measures: [i for i, _ in group] for measures, group in performance_mapping.items()
        }
        chunk_file_path = p_path.with_name(f"{p_path.stem}_chunks.json")
        with open(chunk_file_path, "w") as f:
            json.dump(performance_mapping_indices, f, indent=2)


def assign_midi_score_notes_to_chunks(
    midi_score: symusic.core.ScoreSecond,
    midi_score_downbeats: list[float],
    downbeats_score_map: DownbeatScoreMap,
    rounding_error_tolerance: float = 0.01,
) -> dict[str, list[tuple[int, symusic.core.NoteSecond]]]:
    """
    Returns a mapping from XML measure to index of MIDI notes with the symusic Note object.
    """
    assert len(midi_score.tracks) == 1
    track = midi_score.tracks[0]
    # Assign notes to their group
    mapping = defaultdict(list)
    indices_and_notes = enumerate(track.notes)
    i, note = next(indices_and_notes, (None, None))
    for next_db, xml_measures in zip_longest(
        midi_score_downbeats[1:],
        downbeats_score_map,
        fillvalue=math.inf,
    ):
        # Assign all notes before the next downbeat to the current measure (or group of measures)
        notes_in_current_group = []
        while note is not None and note.start < next_db - rounding_error_tolerance:
            notes_in_current_group.append((i, note))
            i, note = next(indices_and_notes, (None, None))
        xml_measures_str = "-".join(str(m) for m in xml_measures)
        mapping[xml_measures_str] = notes_in_current_group
    return mapping


def assign_performance_notes_to_chunks(
    performance: symusic.core.ScoreSecond,
    midi_score: symusic.core.ScoreSecond,
    performance_downbeats: list[float],
    midi_score_downbeats: list[float],
    downbeats_score_map: DownbeatScoreMap,
    midi_score_mapping: dict[str, list[tuple[int, symusic.core.NoteSecond]]],
    swapping_tolerance: float = 0.05,  # As in Beyer's
) -> dict[str, list[tuple[int, symusic.core.NoteSecond]]]:
    assert len(performance.tracks) == 1
    assert len(midi_score.tracks) == 1
    performance_track = performance.tracks[0]
    # Start by simply assigning performance notes to measure naively like midi_score
    indices_and_notes = enumerate(performance_track.notes)
    i, note = next(indices_and_notes, (None, None))
    mapping = defaultdict(list)
    for next_db, xml_measures in itertools.zip_longest(
        performance_downbeats[1:],
        downbeats_score_map,
        fillvalue=math.inf,
    ):
        # Assign all notes before the next downbeat to the current measure (or group of measures)
        notes_in_current_group = []
        while note is not None and note.start < next_db - swapping_tolerance:
            notes_in_current_group.append((i, note))
            i, note = next(indices_and_notes, (None, None))
        xml_measures_str = "-".join(str(m) for m in xml_measures)
        mapping[xml_measures_str] = notes_in_current_group

    # If no swapping allowed, simply return now
    if swapping_tolerance == 0:
        return mapping

    # Iterate over notes to see if the alignement would benefit from moving a note from one measure
    # to the next one (we check the pitches present in each measure to proxy the alignemnt quality
    # as it's done by Beyer)
    n_swaps = 0
    for (group, downbeat), (next_group, next_downbeat) in itertools.pairwise(
        zip(midi_score_mapping.keys(), performance_downbeats)
    ):
        notes_to_swap_remaining = True
        while notes_to_swap_remaining:
            notes_to_swap_remaining = False  # will be flipped back to True later if needed
            # Figure out which MIDI pitches can be moved forward/backward by one beat
            # because that would yield better alignment. Only considers notes within
            # 0.5s of the beat boundary.
            counter_performance = Counter([n.pitch for _, n in mapping[group]])
            counter_midi_score = Counter([n.pitch for _, n in midi_score_mapping[group]])

            counter_performance_next = Counter([n.pitch for _, n in mapping[next_group]])
            counter_midi_score_next = Counter([n.pitch for _, n in midi_score_mapping[next_group]])

            too_much = counter_performance - counter_midi_score
            lacking = counter_midi_score - counter_performance

            too_much_next = counter_performance_next - counter_midi_score_next
            lacking_next = counter_midi_score_next - counter_performance_next

            can_be_moved_forward = too_much & lacking_next
            can_be_moved_backward = too_much_next & lacking

            for pitch in can_be_moved_forward:
                last_i, last_n = mapping[group][-1]
                # Only swap notes if they are within swapping_tolerance seconds of the beat boundary
                if last_n.pitch == pitch and next_downbeat - last_n.start < swapping_tolerance:
                    print(
                        f" * Forward {last_i}|{midi_pitch_to_note_name(pitch)} ({last_n.start}) from {group} ({downbeat}) to {next_group} ({next_downbeat})"
                    )
                    mapping[next_group].insert(0, mapping[group].pop())
                    n_swaps += 1
                    notes_to_swap_remaining = True
                    break

            for pitch in can_be_moved_backward:
                first_i, first_n = mapping[next_group][0]
                # Only swap notes if they are within swapping_tolerance seconds of the beat boundary
                if first_n.pitch == pitch and first_n.start - downbeat < swapping_tolerance:
                    print(
                        f" * Backward {first_i}|{midi_pitch_to_note_name(pitch)} ({first_n.start}) from {next_group} ({next_downbeat}) to {group} ({downbeat})"
                    )
                    mapping[group].append(mapping[next_group].pop(0))
                    n_swaps += 1
                    notes_to_swap_remaining = True
                    break

    # If we had to swap a lot of notes, we probably messed up and should just return
    # the raw alignment...
    if n_swaps / len(performance_track.notes) > 0.1:
        return assign_performance_notes_to_chunks(
            performance=performance,
            midi_score=midi_score,
            performance_downbeats=performance_downbeats,
            midi_score_downbeats=midi_score_downbeats,
            downbeats_score_map=downbeats_score_map,
            midi_score_mapping=midi_score_mapping,
            swapping_tolerance=0.0,  # No swapping allowed
        )

    return mapping


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
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser(description="Create chunk files for all MIDI scores in the ASAP dataset.")
    parser.add_argument("asap_path", type=Path, help="Path to the ASAP dataset.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level (TRACE, DEBUG, INFO, WARNING, ERROR).")
    args = parser.parse_args()
    # fmt: on

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), level=args.log_level.upper())

    metadata = pd.read_csv(args.asap_path / "metadata.csv")
    # metadata = metadata[metadata["composer"] == "Bach"]
    # metadata = metadata[metadata["folder"] == "Scriabin/Sonatas/5"]
    # metadata = metadata[metadata["folder"] == "Beethoven/Piano_Sonatas/17-1"]
    # metadata = metadata[metadata["folder"] == "Schumann/Kreisleriana/7"]
    # metadata = metadata[metadata["folder"] == "Bach/Fugue/bwv_865"]
    # metadata = metadata[metadata["folder"] == "Chopin/Scherzos/31"]
    metadata = metadata[metadata["folder"] == "Beethoven/Piano_Sonatas/32-1"]
    annotations = pd.read_json(args.asap_path / "asap_annotations.json", orient="index")
    create_chunk_file_for_all_midi_scores(
        metadata=metadata,
        annotations=annotations,
        asap_path=args.asap_path,
    )
