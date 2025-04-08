import itertools
from fractions import Fraction

import torch

from src.models.automata import tokens as tk


TOKEN_INDEXER = {token: i for i, token in enumerate(tk.TOKEN_TYPES)}
TOKEN_INDEXER_INV = {i: token for token, i in TOKEN_INDEXER.items()}

TS_DENOMINATOR_INDEXER = {den: i for i, den in enumerate([1, 2, 4, 8, 16, 32])}
TS_DENOMINATOR_INDEXER_INV = {i: den for den, i in TS_DENOMINATOR_INDEXER.items()}

DURATION_INDEXER = {dur.value: i for i, dur in enumerate(tk.SymbolicDurationType)}
DURATION_INDEXER_INV = {i: dur for dur, i in DURATION_INDEXER.items()}

NOTE_INDEXER = {note: i for i, note in enumerate("CDEFGAB")}
NOTE_INDEXER_INV = {i: note for note, i in NOTE_INDEXER.items()}

DIMS = {
    "token_type": len(TOKEN_INDEXER),
    "bar_numerator": 1,
    "bar_denominator": len(TS_DENOMINATOR_INDEXER),
    "bar_is_nominal": 1,  # True or False
    "key_signature": 15,  # -7 to 7
    "clef_type": 1,  # G or F
    "staff_idx": 4,  # 0 to 3 (max 4 staves)
    "voice_idx": 16,  # 0 to 15 (max 4 voices per staff)
    "duration_type": len(DURATION_INDEXER),
    "duration_dots": 4,  # Max 3 dots
    "note_class": 7,  # A to G
    "note_octave": 9,  # 0 to 8 (max 8 octaves)
    "note_accidental": 5,  # -2 to 2 (double flat to double sharp)
    "note_tied": 1,  # True or False
    "note_stem_direction": 3,  # up, down, or none
    "tuplet_actual_notes": 1,
    "tuplet_normal_notes": 1,
}

CUM_DIMS = {k: v for k, v in zip(DIMS.keys(), itertools.accumulate(DIMS.values(), initial=0))}

TOTAL_DIMENSIONS = sum(DIMS.values())


def clip_cast(x, lower=None, upper=None, type_=None):
    assert lower is None or upper is None or lower <= upper
    if lower is not None:
        x = max(x, lower)
    if upper is not None:
        x = min(x, upper)
    if type_ is not None:
        x = type_(x)
    return x


def vectorize_token(token: tk.Token):
    vector = torch.zeros(TOTAL_DIMENSIONS)

    # Vectorize token type
    vector[TOKEN_INDEXER[type(token)]] = 1

    # Bar
    if isinstance(token, tk.Bar):
        vector[CUM_DIMS["bar_numerator"]] = token.numerator
        vector[CUM_DIMS["bar_denominator"] + TS_DENOMINATOR_INDEXER[token.denominator]] = 1
        vector[CUM_DIMS["bar_is_nominal"]] = token.is_nominal

    # KeySignature
    if isinstance(token, tk.KeySignature):
        fifths = token.accidentals + 7
        vector[CUM_DIMS["key_signature"] + fifths] = 1

    # Clef
    if isinstance(token, tk.Clef):
        clef_type = 0 if token.clef == "G" else 1
        vector[CUM_DIMS["clef_type"]] = clef_type

    # HasStaffMixin
    if isinstance(token, tk.HasStaffMixin):
        vector[CUM_DIMS["staff_idx"] + token.staff_idx] = 1

    # HasVoiceMixin
    if isinstance(token, tk.HasVoiceMixin):
        vector[CUM_DIMS["voice_idx"] + token.voice_idx] = 1

    # HasDurationMixin
    if isinstance(token, tk.HasDurationMixin):
        vector[CUM_DIMS["duration_type"] + DURATION_INDEXER[token.duration.type.value]] = 1
        vector[CUM_DIMS["duration_dots"] + token.duration.dots] = 1

    # Note
    if isinstance(token, tk.Note):
        vector[CUM_DIMS["note_class"] + NOTE_INDEXER[token.note_class]] = 1
        vector[CUM_DIMS["note_octave"] + token.octave] = 1
        vector[CUM_DIMS["note_accidental"] + token.accidental + 2] = 1
        vector[CUM_DIMS["note_tied"]] = 1 if token.tied else 0
        stem = 0 if token.stem_direction == "up" else 1 if token.stem_direction == "down" else 2
        vector[CUM_DIMS["note_stem_direction"] + stem] = 1

    # TupletStart
    if isinstance(token, tk.TupletStart):
        vector[CUM_DIMS["tuplet_actual_notes"]] = token.n_actual
        vector[CUM_DIMS["tuplet_normal_notes"]] = token.n_normal
        vector[CUM_DIMS["duration_type"] + DURATION_INDEXER[token.unit_duration.type.value]] = 1
        vector[CUM_DIMS["duration_dots"] + token.unit_duration.dots] = 1

    return vector


def unvectorize_token(vector: torch.Tensor):
    def unvectorize_ohe(vector: torch.Tensor, key: str) -> int:
        start, length = CUM_DIMS[key], DIMS[key]
        return vector[start : start + length].argmax().item()

    token_type = tk.TOKEN_TYPES[vector[: DIMS["token_type"]].argmax().item()]

    # StartOfScore
    if token_type == tk.StartOfScore:
        return tk.StartOfScore()

    # EndOfScore
    if token_type == tk.EndOfScore:
        return tk.EndOfScore()

    # Bar
    if token_type == tk.Bar:
        numerator = vector[CUM_DIMS["bar_numerator"]]
        denominator_idx = unvectorize_ohe(vector, "bar_denominator")
        denominator = TS_DENOMINATOR_INDEXER_INV[denominator_idx]
        is_nominal = bool(vector[CUM_DIMS["bar_is_nominal"]] > 0.5)
        return tk.Bar(
            numerator=clip_cast(numerator, 1, type_=int),
            denominator=denominator,
            is_nominal=is_nominal,
        )

    # KeySignature
    if token_type == tk.KeySignature:
        fifths = unvectorize_ohe(vector, "key_signature") - 7
        return tk.KeySignature(accidentals=fifths)

    # Get eventual staff
    staff_idx = unvectorize_ohe(vector, "staff_idx")

    # Clef
    if token_type == tk.Clef:
        clef_idx = int(vector[CUM_DIMS["clef_type"]] > 0.5)
        clef_type = "G" if clef_idx == 0 else "F"
        return tk.Clef(clef=clef_type, staff_idx=staff_idx)

    # Get eventual voice
    voice_idx = unvectorize_ohe(vector, "voice_idx")

    # Get eventual duration
    duration_type_idx = unvectorize_ohe(vector, "duration_type")
    duration_type = DURATION_INDEXER_INV[duration_type_idx]
    duration_dots = unvectorize_ohe(vector, "duration_dots")

    # Note
    if token_type == tk.Note:
        note_class = NOTE_INDEXER_INV[unvectorize_ohe(vector, "note_class")]
        octave = unvectorize_ohe(vector, "note_octave")
        accidental = unvectorize_ohe(vector, "note_accidental") - 2
        tied = bool(vector[CUM_DIMS["note_tied"]] > 0.5)  # Or rather 0 for sigmoid?
        stem_direction_idx = unvectorize_ohe(vector, "note_stem_direction")
        stem_direction = ["up", "down", None][stem_direction_idx]
        return tk.Note(
            note_class=note_class,
            octave=octave,
            accidental=accidental,
            tied=tied,
            stem_direction=stem_direction,
            staff_idx=staff_idx,
            voice_idx=voice_idx,
            duration=tk.SymbolicDuration(
                type=tk.SymbolicDurationType(duration_type), dots=duration_dots
            ),
        )

    # Rest
    if token_type == tk.Rest:
        return tk.Rest(
            staff_idx=staff_idx,
            voice_idx=voice_idx,
            duration=tk.SymbolicDuration(
                type=tk.SymbolicDurationType(duration_type), dots=duration_dots
            ),
        )

    # ChordStart
    if token_type == tk.ChordStart:
        return tk.ChordStart(
            voice_idx=voice_idx,
            duration=tk.SymbolicDuration(
                type=tk.SymbolicDurationType(duration_type), dots=duration_dots
            ),
        )

    # ChordEnd
    if token_type == tk.ChordEnd:
        return tk.ChordEnd(voice_idx=voice_idx)

    # TupletStart
    if token_type == tk.TupletStart:
        n_actual = clip_cast(vector[CUM_DIMS["tuplet_actual_notes"]], 1, type_=int)
        n_normal = clip_cast(vector[CUM_DIMS["tuplet_normal_notes"]], 1, type_=int)
        return tk.TupletStart(
            n_actual=n_actual,
            n_normal=n_normal,
            unit_duration=tk.SymbolicDuration(
                type=tk.SymbolicDurationType(duration_type),
                dots=duration_dots,
            ),
            voice_idx=voice_idx,
        )

    # TupletEnd
    if token_type == tk.TupletEnd:
        return tk.TupletEnd(voice_idx=voice_idx)


if __name__ == '__main__2':
    tokens = [
        tk.StartOfScore(),
        tk.EndOfScore(),
        tk.Bar(numerator=6, denominator=8, actual_duration=Fraction(3, 1)),
        tk.KeySignature(accidentals=-2),
        tk.Clef(clef="G", staff_idx=1),
        tk.Note(
            note_class="C",
            octave=4,
            accidental=0,
            tied=False,
            stem_direction="up",
            staff_idx=1,
            voice_idx=5,
            duration=tk.SymbolicDuration(type=tk.SymbolicDurationType.EIGHTH, dots=0),
        ),
        tk.TupletStart(
            n_actual=3,
            n_normal=2,
            unit_duration=tk.SymbolicDuration(type=tk.SymbolicDurationType.QUARTER, dots=0),
            voice_idx=1,
        ),
        tk.ChordStart(
            voice_idx=1, duration=tk.SymbolicDuration(type=tk.SymbolicDurationType.HALF, dots=2)
        ),
        tk.ChordEnd(voice_idx=3),
        tk.Rest(
            staff_idx=2,
            voice_idx=1,
            duration=tk.SymbolicDuration(type=tk.SymbolicDurationType.WHOLE, dots=1),
        ),
        tk.TupletEnd(voice_idx=1),
    ]

    for token in tokens:
        print(str(token))
        vector = vectorize_token(token)
        print(vector)
        unvectorized = unvectorize_token(vector)
        print(str(unvectorized))
        assert token == unvectorized
        print()

if __name__ == '__main__':
    torch.manual_seed(0)
    random_vectors = torch.randn(10000, TOTAL_DIMENSIONS) * 2 + 0.5

    for vector in random_vectors:
        # print()
        # print(vector)
        unvectorized = unvectorize_token(vector)
        # print(str(unvectorized))
        assert unvectorize_token(vectorize_token(unvectorized)) == unvectorized
