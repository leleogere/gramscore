"""
Some old version of MuseScore has a weird behaviour in MusicXML export, where if one note of a
chord is tied, then all the other notes of the chord are marked as tied too. In addition to that,
the next occurrence of this note in the score will have the <tie type="stop"> tag, even if the note
is located several measures later (probably to avoid hanging <tie type="start"> tags).

Now, when parsing a MusicXML file, MuseScore verifies a certain number of conditions to ensure that
a tie is valid, mainly that: a tied note must be resolved in the same measure, or directly in
succession of the tied note if in the next measure. In all other cases, the tie is considered
invalid and dropped.

This script tries to estimate the amount of invalid ties in the ASAP dataset, according to MuseScore
criteria.
"""

import warnings
from pathlib import Path
import sys

import partitura as pt
import partitura.score
from tqdm import tqdm

asap_path = Path("/home/gerel/Documents/datasets/asap-dataset")


def check_ties_musescore(part: pt.score.Part) -> tuple[int, int, list[int]]:
    """Count the ties, and return a tuple containing the number of valid and invalid ties."""
    valid_ties = 0
    invalid_ties = 0
    invalid_measures = set()
    time_to_measure = part.measure_number_map

    for note in part.iter_all(cls=pt.score.GenericNote, include_subclasses=True):
        note: pt.score.GenericNote
        # If not tied skip
        if note.tie_next is None:
            continue

        current_measure = int(time_to_measure(note.start.t))
        next_measure = int(time_to_measure(note.tie_next.start.t))

        # If in same measure: OK
        if current_measure == next_measure:
            valid_ties += 1

        # If in consecutive measure and directly successive: OK
        elif current_measure + 1 == next_measure and note.tie_next.start.t == note.end.t:
            valid_ties += 1

        # Otherwise: invalid
        else:
            invalid_ties += 1
            invalid_measures.add(current_measure)

    return valid_ties, invalid_ties, sorted(invalid_measures)


if __name__ == '__main__':
    import joblib

    total_valid = 0
    total_invalid = 0

    # for score in asap_path.rglob("xml_score.musicxml"):
    #     print(score)
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         score = pt.load_musicxml(score)
    #         score = pt.score.unfold_part_maximal(score)
    #
    #     for part in score.parts:
    #         valid, invalid, measures = check_ties_musescore(part)
    #         print(
    #             f"  - Valid: {valid} | Invalid: {invalid}",
    #             f"[{', '.join(map(str, measures))}]" if measures else "",
    #         )

    def load_score_and_check_ties(path: Path) -> tuple[int, int, list[int]]:
        # print(path)
        sys.setrecursionlimit(1_000_000_000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = pt.load_musicxml(path)
            try:
                score = pt.score.unfold_part_maximal(score)
            except RecursionError:
                print(f"Recursion error in {path}")
                return 0, 0, []
        tot_valids = 0
        tot_invalids = 0
        tot_measures: set = set()
        for part in score.parts:
            valid, invalid, measures = check_ties_musescore(part)
            tot_valids += valid
            tot_invalids += invalid
            tot_measures.update(measures)
        return tot_valids, tot_invalids, sorted(tot_measures)

    # Parallel version
    scores = list(asap_path.rglob("xml_score.musicxml"))[::-1]
    # scores = list(
    #     map(
    #         Path,
    #         [
    #             # "/home/gerel/Documents/datasets/asap-dataset/Scriabin/Etudes_op_8/11/xml_score.musicxml",
    #             # "/home/gerel/Documents/datasets/asap-dataset/Debussy/Images_Book_1/1_Reflets_dans_lEau/xml_score.musicxml",
    #             # "/home/gerel/Documents/datasets/asap-dataset/Debussy/Pour_le_Piano/1/xml_score.musicxml",
    #             # "/home/gerel/Documents/datasets/asap-dataset/Liszt/Sonata/xml_score.musicxml",
    #             "/home/gerel/Documents/datasets/asap-dataset/Liszt/Mephisto_Waltz/xml_score.musicxml",
    #             # "/home/gerel/Documents/datasets/asap-dataset/Liszt/Sonata/xml_score.musicxml",
    #         ],
    #     )
    # )
    results = joblib.Parallel(n_jobs=4, return_as="generator", backend="loky")(
        joblib.delayed(load_score_and_check_ties)(score) for score in scores
    )

    for path, (valid, invalid, measures) in (
        pbar := tqdm(zip(scores, results), total=len(scores), file=sys.stdout)
    ):
        components = [
            f"Valid: {valid}",
            f"Invalid: {invalid}",
            str(path),
            f"[{', '.join(map(str, measures))}]" if measures else "",
        ]
        pbar.write("\t".join(components))
        total_valid += valid
        total_invalid += invalid

    print(f"Total valid: {total_valid} | Total invalid: {total_invalid}")
