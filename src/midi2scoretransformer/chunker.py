"""
This module contains code for chunking MIDI and MusicXML files into pseudo-measures using
beat-level annotations from the ASAP dataset to align the MIDI and MusicXML.

We use a greedy algorithm to align the MIDI and MusicXML files by moving notes that are
close to the beat boundaries to the next/previous measure if that improves the alignment.
The resulting chunks are saved as JSON files next to the performance-MIDI files.
"""

import json
from collections.abc import Iterable
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
import argparse

import pandas as pd
import pretty_midi
import symusic
from joblib import Parallel, delayed
from partitura.utils.music import midi_pitch_to_pitch_spelling
from tqdm import tqdm

from tokenizer import MultistreamTokenizer


def make_measures(midi, midi_score: pretty_midi.PrettyMIDI, mxl, annots, swap=True):
    swap_tol = 0.05
    measures = {"midi": [], "mxl": []}
    i, j = (0, 0)
    for sb_seconds in annots["midi_score_beats"]:
        # Convert the score beats from seconds to musical time using
        # the provided MIDI scores
        sb = midi_score.time_to_tick(sb_seconds) / midi_score.resolution
        # print(sb, sb_seconds)
        measures["mxl"].append([])
        while j < len(mxl) and mxl[j].offset < sb:
            # print(" ", mxl[j].offset, "\t", mxl[j].pitch.unicodeNameWithOctave)
            measures["mxl"][-1].append((j, mxl[j].pitch.midi))
            j += 1

    # Use the performance beats to split the MIDI into pseudo-measures of beat length
    for pb in annots["performance_beats"]:
        measures["midi"].append([])
        while i < len(midi) and midi[i].start < pb - swap_tol:
            measures["midi"][-1].append((i, midi[i].pitch))
            i += 1

    measures["midi"].append([])
    measures["mxl"].append([])
    while i < len(midi):
        measures["midi"][-1].append((i, midi[i].pitch))
        i += 1
    while j < len(mxl):
        measures["mxl"][-1].append((j, mxl[j].pitch.midi))
        j += 1

    if not swap:
        for i in range(len(measures["midi"])):
            measures["midi"][i] = [i_ for i_, p in measures["midi"][i]]
            measures["mxl"][i] = [i_ for i_, p in measures["mxl"][i]]
        measures["swapped"] = False
        return measures
    from collections import Counter

    n_swaps = 0
    swap_tol = 0.5
    for i in range(len(measures["midi"]) - 1):
        swapped = True
        while swapped:
            swapped = False
            # Figure out which MIDI pitches can be moved forward/backward by one beat
            # because that would yield better alignment. Only considers notes within
            # 0.5s of the beat boundary.
            c_midi = Counter([p for _, p in measures["midi"][i]])
            c_mxl = Counter([p for _, p in measures["mxl"][i]])

            c_midi_next = Counter([p for _, p in measures["midi"][i + 1]])
            c_mxl_next = Counter([p for _, p in measures["mxl"][i + 1]])
            too_much = c_midi - c_mxl
            lacking = c_mxl - c_midi
            too_much_next = c_midi_next - c_mxl_next
            lacking_next = c_mxl_next - c_midi_next
            can_be_moved_forward = too_much & lacking_next
            can_be_moved_backward = too_much_next & lacking
            for pitch in can_be_moved_forward:
                last_j, last_p = measures["midi"][i][-1]
                # Only swap notes if they are within 0.5s of the beat boundary
                if (
                    last_p == pitch
                    and annots["performance_beats"][i] - midi[last_j].start < swap_tol
                ):
                    measures["midi"][i + 1].insert(0, measures["midi"][i].pop())
                    n_swaps += 1
                    swapped = True
                    break

            for pitch in can_be_moved_backward.keys():
                first_j, first_p = measures["midi"][i + 1][0]
                # ---only swap notes if they are within 0.5s of the beat boundary
                if (
                    first_p == pitch
                    and midi[first_j].start - annots["performance_beats"][i] < swap_tol
                ):
                    measures["midi"][i].append(measures["midi"][i + 1].pop(0))
                    n_swaps += 1
                    swapped = True
                    break

    # print("Swaps", n_swaps, n_swaps/len(midi))
    # If we had to swap a lot of notes, we probably messed up and should just return
    # the raw alignment...
    if n_swaps / len(midi) > 0.1:
        return make_measures(midi, midi_score, mxl, annots, swap=False)
    for i in range(len(measures["midi"])):
        measures["midi"][i] = sorted([j for j, _ in measures["midi"][i]])
        measures["mxl"][i] = sorted([j for j, _ in measures["mxl"][i]])
        # measures["midi"][i] = sorted([(j, p) for j, p in measures["midi"][i]], key=lambda x: x[0])
        # measures["mxl"][i] = sorted([(j, p) for j, p in measures["mxl"][i]], key=lambda x: x[0])
        # for midi_or_mxl in [measures["midi"][i], measures["mxl"][i]]:
        #     for j in range(len(midi_or_mxl)):
        #         k, p = midi_or_mxl[j]
        #         step, alter, octave = midi_pitch_to_pitch_spelling(p)
        #         alter = "#" * alter if alter > 0 else "b" * -alter
        #         midi_or_mxl[j] = f"{k}: {step}{alter}{octave} ({p})"
    assert max([max(m + [0]) for m in measures["midi"]]) == len(midi) - 1
    assert sum(len(m) for m in measures["midi"]) == len(midi)
    assert max([max(m + [0]) for m in measures["mxl"]]) == len(mxl) - 1
    assert sum(len(m) for m in measures["mxl"]) == len(mxl)
    measures["swapped"] = True
    return measures


def handle_file(midi_path: Path, mxl_path: Path, save_path: Path, annots: dict[str, dict] | None):
    # # If midi_score, no need to run complex alignment fixer as with real performances
    # if midi_path.stem == "midi_score":
    #     # midi = symusic.Score(midi_path, ttype="tick")  # second results in rounding errors?
    #     midi = symusic.Score(midi_path, ttype="second")
    #     track = midi.tracks[0]
    #     if len(midi.tracks) > 1:
    #         for other_track in midi.tracks[1:]:
    #             track.notes.extend(other_track.notes)
    #         track.sort()
    #     measures = {"midi": []}
    #
    #     beats = iter(annots["midi_score_beats"])
    #     next(beats)  # Skip first beat as we check for the next one
    #     next_beat = next(beats)
    #
    #     def tick2second(tick: int, tpq: int, tempos: symusic.core.TempoTickList) -> Fraction:
    #         assert len(tempos) > 0
    #         tick_times_mspq = 0
    #         tempos_iter = iter(tempos)
    #         tempo = next(tempos_iter)
    #         next_tempo = next(tempos_iter, None)
    #         while next_tempo is not None and tick >= next_tempo.time:
    #             tick_times_mspq += (next_tempo.time - tempo.time) * tempo.mspq
    #             tempo = next_tempo
    #             next_tempo = next(tempos_iter, None)
    #             tick -= tempo.time
    #         tick_times_mspq += tick * tempo.mspq
    #         seconds = Fraction(tick_times_mspq, 1_000_000 * tpq)
    #         return seconds
    #         # return Fraction(mspq * tick, 1_000_000 * tpq)
    #
    #     notes_in_current_beat = []
    #     for i, note in enumerate(track.notes):
    #         note_start = tick2second(note.start, midi.tpq, midi.tempos)
    #         print(note.start, float(note_start))
    #         while note_start >= next_beat:
    #             print(
    #                 f"Starting beat {len(measures['midi']) + 1} (M{(len(measures['midi']) + 1 )// 3})"
    #             )
    #             measures["midi"].append(notes_in_current_beat)
    #             notes_in_current_beat = []
    #             next_beat = next(beats, float("inf"))
    #         print(
    #             f"P:{note.pitch} \t T:{note_start} \t NB:{next_beat} \t B:{len(measures['midi'])}"
    #         )  # TODO: check that this is not an error of tick2second conversion of symusic
    #         """
    #         Starting beat 20 (M6)
    #         P:58 	 T:4.199999809265137 	 NB:4.2 	 B:20
    #         P:60 	 T:4.199999809265137 	 NB:4.2 	 B:20
    #         P:63 	 T:4.199999809265137 	 NB:4.2 	 B:20
    #         P:66 	 T:4.199999809265137 	 NB:4.2 	 B:20
    #         P:70 	 T:4.199999809265137 	 NB:4.2 	 B:20
    #         P:75 	 T:4.199999809265137 	 NB:4.2 	 B:20
    #         P:84 	 T:4.199999809265137 	 NB:4.2 	 B:20
    #         P:87 	 T:4.199999809265137 	 NB:4.2 	 B:20
    #         """
    #         notes_in_current_beat.append(i)
    #
    #     # Same for xml as exactly aligned
    #     measures["mxl"] = measures["midi"]
    #     measures["swapped"] = False

    # Else, run the alignment fixer

    if not annots["score_and_performance_aligned"]:
        return
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, s = MultistreamTokenizer.mxl_to_list(str(mxl_path))
        measures = make_measures(
            MultistreamTokenizer.midi_to_list(str(midi_path)),
            pretty_midi.PrettyMIDI(str(mxl_path.with_name("midi_score.mid"))),
            m,
            annots,
        )

    # Write to disk
    if save_path.exists():
        try:
            with open(save_path, "r") as f:
                prev = json.load(f)
        except json.decoder.JSONDecodeError:
            print(save_path)
            raise
        # Print differences
        if prev["midi"] != measures["midi"]:
            print("midi", prev["midi"], measures["midi"], sep="\n")
        if prev["mxl"] != measures["mxl"]:
            print("mxl", prev["mxl"], measures["mxl"], sep="\n")
    with save_path.open("w") as f:
        json.dump(measures, f)


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path, help="Path to the ASAP dataset")
    parser.add_argument("--composer", type=str, help="Specific composer to process")
    args = parser.parse_args()
    dataset_path = args.dataset_path.absolute()

    # Read metadata
    with (dataset_path / "metadata.csv").open("r") as f:
        metadata = pd.read_csv(f)
    if args.composer:
        metadata = metadata[metadata.composer == args.composer]

    # Read annotations
    with (dataset_path / "asap_annotations.json").open("r") as f:
        annotations = json.load(f)

    # Find all scores and performances
    paths = []
    for _, row in metadata.iterrows():
        score_path = Path(dataset_path / row.xml_score)
        performance_path = Path(dataset_path / row.midi_performance)
        chunks = Path(dataset_path / row.midi_performance.replace(".mid", "_chunks.json"))
        paths.append((performance_path, score_path, chunks))

    # paths = [
    #     (
    #         Path("/home/gerel/Documents/datasets/asap-dataset/Chopin/Scherzos/31/midi_score.mid"),
    #         Path(
    #             "/home/gerel/Documents/datasets/asap-dataset/Chopin/Scherzos/31/xml_score.musicxml"
    #         ),
    #         Path(
    #             "/home/gerel/Documents/datasets/asap-dataset/Chopin/Scherzos/31/midi_score_chunks.json"
    #         ),
    #     )
    # ]

    # Create additional annotations for the midi_score.mid files
    for _, row in metadata.drop_duplicates("xml_score").iterrows():
        performance = row.midi_performance
        performance_beats = annotations[performance]["midi_score_beats"]
        midi_score_beats = annotations[performance]["midi_score_beats"]
        annotations[row.midi_score] = {
            "score_and_performance_aligned": True,
            "performance_beats": performance_beats,
            "midi_score_beats": midi_score_beats,
        }

    # Only process midi_score.mid files
    paths = [p for p in paths if p[0].name == "midi_score.mid"]

    # Chunk the files
    results = Parallel(n_jobs=min(12, len(paths)), return_as="generator")(
        delayed(handle_file)(
            midi_path,
            mxl_path,
            save_path,
            annotations[str(midi_path.relative_to(dataset_path))],
        )
        for midi_path, mxl_path, save_path in paths
    )

    # Track chunking
    for _ in tqdm(results, total=len(paths)):
        # for _ in results:
        pass

# if __name__ == '__main__':
#     annotations = json.load(open("data/asap-dataset/asap_annotations.json"))
#     xml = "./data/asap-dataset/Bach/Prelude/bwv_868/xml_score.musicxml"
#     midi = os.path.join(os.path.dirname(xml), "GonzalezJ05M.mid")
#     save_path = os.path.join(os.path.dirname(xml), os.path.basename(midi).replace(".mid", "_chunks.json"))
#     handle_file(midi, xml, save_path)
