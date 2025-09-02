import json
import warnings
from collections import defaultdict
from fractions import Fraction

import joblib
import tabulate
from lxml import etree
from pathlib import Path

import partitura as pt
import partitura.score
from tqdm import tqdm

import scripts.score_encoding_verification.simpler_tokens_with_measure_rests as tk

# import scripts.score_encoding_verification.simpler_tokens as tk


def check_automata(
    tokens: list[tk.Token],
    allow_longer_measures: bool,
    allow_duplicate_notes_in_chord: bool,
) -> dict[str, list[Exception]]:
    automata = tk.ConstrainedAutomata(
        num_voices=16,
        allow_longer_measures=allow_longer_measures,
        allow_duplicate_notes_in_chord=allow_duplicate_notes_in_chord,
    )

    exceptions = defaultdict(list)

    measure_number = None
    malformed_measure = False

    for token in tokens:
        if isinstance(token, tk.Bar):
            measure_number = token._partitura_elements[0].name
            malformed_measure = False

        # If malformed, skip until next Bar token
        if not malformed_measure:
            try:
                automata.transition(token)
            except tk.TransitionError as e:
                exceptions[measure_number].append(e)

                # Reset the automata
                malformed_measure = True
                automata.state.reset_to_clean_state()

    return exceptions


class TokenizationError(Exception):
    pass


def check_path(
    path: Path,
    allow_longer_measures: bool,
    allow_duplicate_notes_in_chord: bool,
) -> dict[str, list[Exception]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pt.load_musicxml(path, force_note_ids=True)
        if len(score.parts) != 1:
            return {"global": [ValueError(f"More than 1 part: {len(score.parts)}")]}
        score = score.parts[0]
        score = pt.score.unfold_part_maximal(score, ignore_leaps=False)

    tk.fill_rests(score)
    # tokens = partitura_to_token_sequence(score, include_start_token=False)
    try:
        tokens = tk.partitura_to_token_sequence(score)
    except Exception as e:
        raise TokenizationError(*e.args)
    # for token in tokens:
    #     print(token)
    try:
        exceptions = check_automata(
            tokens,
            allow_longer_measures=allow_longer_measures,
            allow_duplicate_notes_in_chord=allow_duplicate_notes_in_chord,
        )
    except Exception as e:
        exceptions = {"global": [e]}

    return exceptions


# if __name__ == '__main__':
#     path = Path("/home/gerel/Documents/papiers/asap_fixes/scores/bad_measure_rest/example.musicxml")
#     excs = check_path(path, allow_longer_measures=False, allow_duplicate_notes_in_chord=False)
#     for measure, mexcs in excs.items():
#         print(measure)
#         for exc in mexcs:
#             print(f"  - {exc}")

if __name__ == '__main__':
    # fmt: off
    # asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
    asap_path = Path("~/Documents/datasets/asap-dataset-original").expanduser()
    # fmt: on

    allow_longer_measures = True
    allow_duplicate_notes_in_chord = False

    invalid_file = asap_path / f"{asap_path.stem}_exceptions.json"
    results_file = (
        asap_path / f"{asap_path.stem}_transition_errors"
        f"_{allow_longer_measures=}"
        f"_{allow_duplicate_notes_in_chord=}"
        f".json"
    )

    with invalid_file.open("r") as f:
        invalid_paths = json.load(f)

    score_paths = sorted(
        filter(
            lambda p: p.relative_to(asap_path).as_posix() not in invalid_paths,
            asap_path.rglob("xml_score.musicxml"),
        )
    )
    print(f"Scores to check: {len(score_paths)}")
    results = joblib.Parallel(n_jobs=-1, return_as="generator")(
        joblib.delayed(check_path)(
            path=path,
            allow_longer_measures=allow_longer_measures,
            allow_duplicate_notes_in_chord=allow_duplicate_notes_in_chord,
        )
        for path in score_paths
    )

    all_exceptions = {}
    for path, exceptions in tqdm(zip(score_paths, results), total=len(score_paths)):
        if exceptions:
            print(f"Parsing errors in {path.relative_to(asap_path)}:")
            for measure, measure_exceptions in exceptions.items():
                print(f" * Measure {measure}:")
                for exception in measure_exceptions:
                    print(f"   - {exception}")
            all_exceptions[path.relative_to(asap_path).as_posix()] = exceptions
    pass

    all_exceptions = {
        path: {
            measure: [str(e) for e in measure_exceptions]
            for measure, measure_exceptions in path_exceptions.items()
        }
        for path, path_exceptions in all_exceptions.items()
    }
    with results_file.open("w") as f:
        json.dump(all_exceptions, f, indent=4)
    pass

    # Compute some statistics about the results

    # Number of paths with errors
    n_score_errors = len(all_exceptions)
    print(f"Number of scores with errors: {n_score_errors}")

    # We exclude global error scores for per-measure analysis
    global_error_scores = [
        path for path, measures in all_exceptions.items() if "global" in measures
    ]
    print(f"Exclusing {len(global_error_scores)} scores for global errors:")
    for s in global_error_scores:
        print(f" * {s}")

    # Total number of measures with errors
    n_measure_errors = sum(
        len(measures)
        for path, measures in all_exceptions.items()
        if path not in global_error_scores
    )
    print(f"Number of measures with errors: {n_measure_errors}")

    # Average number of erroneous measure per score
    avg_measure_error_per_score = n_measure_errors / (n_score_errors - len(global_error_scores))
    print(f"Average measure errors per score: {avg_measure_error_per_score}")

    # Median number of erroneous measure per score
    median_measure_error_per_score = sorted(
        len(measures)
        for path, measures in all_exceptions.items()
        if path not in global_error_scores
    )[n_score_errors // 2]
    print(f"Median measure errors per score: {median_measure_error_per_score}")

    # For each path, proportion of measures affected
    proportions = []
    for path, path_exceptions in all_exceptions.items():
        if path in global_error_scores:
            proportions.append((path, "global", "X", "X"))
            continue
        with open(asap_path / path, "r") as f:
            num_m = f.read().count("<measure")
        proportions.append((path, len(path_exceptions), num_m, len(path_exceptions) / num_m))
    proportions = sorted(
        proportions, key=lambda p: p[-1] if not isinstance(p[-1], str) else -1, reverse=True
    )
    print(tabulate.tabulate(proportions))
