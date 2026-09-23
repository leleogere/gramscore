import warnings
from pathlib import Path

from scripts.score_encoding_verification.check_divisions import (
    check_path as gather_divisions_exceptions,
)
import scripts.score_encoding_verification.simpler_tokens_with_measure_rests as tk
import partitura as pt
import partitura.score


class ReadingError(Exception):
    pass


class MultiPartError(Exception):
    pass


class UnfoldingError(Exception):
    pass


class RestFillingError(Exception):
    pass


class TokenizationError(Exception):
    pass


class DivisionsError(Exception):
    pass


def check_automata(
    path: Path,
    allow_longer_measures: bool,
    allow_duplicate_notes_in_chord: bool,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            score = pt.load_musicxml(path, force_note_ids=True)
        except Exception as e:
            raise ReadingError() from e

        if len(score.parts) != 1:
            raise MultiPartError()

        score = score.parts[0]

        try:
            score = pt.score.unfold_part_maximal(score, ignore_leaps=False)
        except Exception as e:
            raise UnfoldingError() from e

    try:
        tk.fill_rests(score)
    except Exception as e:
        raise RestFillingError() from e

    try:
        tokens = tk.partitura_to_token_sequence(score)
    except Exception as e:
        raise TokenizationError() from e

    automata = tk.ConstrainedAutomata(
        num_voices=16,
        allow_longer_measures=allow_longer_measures,
        allow_duplicate_notes_in_chord=allow_duplicate_notes_in_chord,
    )
    automata.transition_sequence(tokens)

    return None


def check_divisions(path: Path) -> None:
    exceptions = gather_divisions_exceptions(path)
    if len(exceptions) > 0:
        raise DivisionsError(exceptions)


def check_full(path) -> Exception | None:
    try:
        check_divisions(path)
        check_automata(path, allow_longer_measures=True, allow_duplicate_notes_in_chord=True)
    except Exception as e:
        return e
    return None
