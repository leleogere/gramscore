"""
This script simply takes a MusicXML file as input and tokenize it into a sequence of tokens,
printed to stdout.
"""

import warnings
from pathlib import Path

import click
import partitura as pt
import partitura.score

import scripts.score_encoding_verification.simpler_tokens_with_measure_rests as tk


def tokenize_score(path: Path) -> list[tk.AnyToken]:
    # Read score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pt.load_musicxml(path, force_note_ids=True)
        assert len(score.parts) == 1
        score = score.parts[0]
        score = pt.score.unfold_part_maximal(score, ignore_leaps=False)
        tk.fill_rests(score)

    # Tokenize the score
    tokens = tk.partitura_to_token_sequence(score)

    return tokens


def simplified_token_repr(token: tk.AnyToken) -> str:
    if isinstance(token, tk.Bar):
        return f"Bar(numerator={token.numerator}, denominator={token.denominator})"
    elif isinstance(token, tk.Note):
        pitch_str = (
            f"{token.note_class}{'♯' * token.accidental}{'♭' * -token.accidental}{token.octave}"
        )
        simpler_bool = {True: "T", False: "F"}
        return f"Note(voice={token.voice_idx}, pitch={pitch_str}, duration={token.duration}, is_tied={simpler_bool[token.tied]}, is_grace={simpler_bool[token.grace]})"
    elif isinstance(token, tk.Rest):
        return f"Rest(voice={token.voice_idx}, duration={token.duration})"
    elif isinstance(token, tk.ChordStart):
        return f"ChordStart(voice={token.voice_idx}, duration={token.duration})"
    elif isinstance(token, tk.ChordEnd):
        return f"ChordEnd(voice={token.voice_idx})"
    elif isinstance(token, tk.TupletStart):
        return f"TupletStart(voice={token.voice_idx}, n_actual={token.n_actual}, n_normal={token.n_normal}, unit_dur={token.unit_duration})"
    elif isinstance(token, tk.TupletEnd):
        return f"TupletEnd(voice={token.voice_idx})"
    elif isinstance(token, tk.EndOfScore):
        return str(token)
    else:
        raise ValueError(f"Unknown token: {token}")


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--simple", is_flag=True)
def main(path: Path, simple: bool):
    tokens = tokenize_score(path)
    for token in tokens:
        if not simple:
            print(token)
        else:
            print(simplified_token_repr(token))


if __name__ == '__main__':
    main()
