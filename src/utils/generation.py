import copy
import heapq
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Self, Callable

import torch
from loguru import logger
from pretty_midi import note_number_to_name

import models.automata.tokens as tk
from models.automata.vectorization_torch import TokenTensor, ProblemType, clip
from utils.musicxml import ARTICULATIONS_AND_ORNAMENTS


class GenerationState(Enum):
    UNFINISHED = 0
    FINISHED = 1
    EXCEEDED_MAX_TOKENS = 2
    EXCEED_CONSTRAINED_ITERATIONS = 3
    NO_ALLOWED_TOKENS = 4
    CHUNK_ALIGNMENT_ERROR = 5


@dataclass
class PitchCounter:
    count = 0
    could_be_more = False

    def increment(self, count: int = 1):
        self.count += count

    def mark_minimum(self):
        self.could_be_more = True

    def unmark_minimum(self):
        self.could_be_more = False


class PitchTracker:
    def __init__(self):
        self.counters: dict[int | tuple[int, ...], PitchCounter] = defaultdict(PitchCounter)
        self.merged_pitches: dict[int, tuple[int, ...]] = {}

    def add(self, pitch: int | tuple[int, ...], count: int = 1):
        pitches = self.merged_pitches.get(pitch, pitch)
        self.counters[pitches].increment(count)

    def merge(self, first_pitch: int, *other_pitches: int):
        new_key = tuple(sorted([first_pitch, *other_pitches]))
        for pitch in new_key:
            # If already merged, merge with existing merged key
            if pitch in self.merged_pitches:
                previous_merge = self.merged_pitches[pitch]
                all_pitches = set(previous_merge).union(set(new_key))
                new_key = tuple(sorted(all_pitches))
                for p in previous_merge:
                    self.merged_pitches[p] = new_key
        self.counters[new_key].increment(sum(self.counters[pitch].count for pitch in new_key))
        for pitch in new_key:
            self.merged_pitches[pitch] = new_key
            if pitch in self.counters:
                del self.counters[pitch]

    def mark_minimum(self, pitch: int):
        pitches = self.merged_pitches.get(pitch, pitch)
        self.counters[pitches].mark_minimum()

    def __getitem__(self, pitch: int) -> PitchCounter:
        return self.counters[pitch]

    def __setitem__(self, pitch: int, count: int):
        self.counters[pitch].count = count

    def __str__(self):
        elements = []
        for p, c in sorted(
            self.counters.items(),
            key=lambda p_c: (p_c[0] if isinstance(p_c[0], int) else p_c[0][0]),
        ):
            if isinstance(p, tuple):
                element = f"<{'|'.join(note_number_to_name(p) for p in p)}>:{c.count}"
            else:
                element = f"{note_number_to_name(p)}:{c.count}"
            if c.could_be_more:
                element += "+"
            elements.append(element)
        return " / ".join(elements)

    def __delitem__(self, pitch: int):
        del self.counters[pitch]

    @staticmethod
    def subscript_number(s: str | int) -> str:
        return str(s).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))

    @classmethod
    def note_number_to_name(cls, pitch: int) -> str:
        pitch_str = cls.subscript_number(pitch)
        return f"{note_number_to_name(pitch)}{pitch_str}"

    def compute_score(self, other: Self) -> int:
        diff = 0
        assert self.merged_pitches == other.merged_pitches
        unique_keys = sorted(
            set(self.counters.keys()).union(set(other.counters.keys())),
            key=lambda x: x if isinstance(x, int) else x[0],
        )
        for key in unique_keys:
            counter = self.counters.get(key, PitchCounter())
            other_counter = other.counters.get(key, PitchCounter())
            if (
                (counter.could_be_more and other_counter.could_be_more)
                or (counter.could_be_more and counter.count <= other_counter.count)
                or (other_counter.could_be_more and other_counter.count <= counter.count)
            ):
                pass  # Diff not counted in that case
            else:
                diff += abs(counter.count - other_counter.count)
        return diff

    def copy(self, deep: bool = False):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    @classmethod
    def from_merged_pitches(cls, merged_pitches: dict[int, tuple[int, ...]]):
        tracker = cls()
        tracker.merged_pitches = merged_pitches
        return tracker


def get_last_aligned_downbeats(
    performance_pitches: list[int],  # L
    transcribed_tokens: list[tk.AnyToken],  # N
    downbeats_to_fetch: list[int] | None,
    min_onsets: int = 3,
) -> list[tuple[int, int, float]]:
    """
    Find the position of the last downbeats in the performance.

    :param performance_pitches: List of performance pitches.
    :param transcribed_tokens: List of transcribed tokens.
    :param downbeats_to_fetch: List of (reversed) indices of downbeats to fetch. Defaults to [0],
        which is the last *valid* downbeat. The indices start there, meaning that passing [0, 2]
        will fetch the first *valid* downbeats, and the 3rd valid one after that. Passing only [2]
        will only fetch the 3rd valid, but still from the first valid downbeat.
    :param min_onsets: Minimum number of onsets to mark a downbeat valid (allows avoiding bars with
        only 1 note, that could easily result in a bad alignment in case of noise).
    :return: A list of tuples containing, for each downbeat fetched:
        - the index of a Bar token in the transcribed token sequence
        - the index of the first note of that bar in the performed notes sequence
        Note that the list is sorted by ascending time, i.e. the last item corresponds to the last
        downbeat, aka the one marked as [0] in downbeats_to_fetch.
    """
    if downbeats_to_fetch is None:
        downbeats_to_fetch = [0]
    downbeats_to_fetch = sorted(set(downbeats_to_fetch))  # Make sure unique and sorted
    bar_token_indices = [
        i for i, token in enumerate(transcribed_tokens) if isinstance(token, tk.Bar)
    ]
    if max(downbeats_to_fetch) >= len(bar_token_indices):
        logger.warning(
            f"Only {len(bar_token_indices)} bars in transcription, ignoring additional ones"
        )
        downbeats_to_fetch = [i for i in downbeats_to_fetch if i < len(bar_token_indices)]
        if downbeats_to_fetch[-1] != len(bar_token_indices) - 1:
            downbeats_to_fetch.append(len(bar_token_indices) - 1)  # Complete context if needed

    # Go through all notes and set their previous tied note
    open_ties: dict[int, int] = {}  # From pitch to sequence index
    for note in transcribed_tokens:
        if isinstance(note, tk.Note):
            if note.pitch in open_ties:
                note._tied_prev = open_ties.pop(note.pitch)
                if note.tied:
                    open_ties[note.pitch] = note
            if note.tied:
                open_ties[note.pitch] = note
    # for note in transcribed_tokens:
    #     if isinstance(note, tk.Note) and hasattr(note, "_tied_prev"):
    #         print(f"* {note}")
    #         print(f"  {note._tied_prev}")

    # Prepare the reversed performance notes and tokens
    transcribed_tokens_reversed = transcribed_tokens[::-1]
    performances_pitches_reversed = performance_pitches[::-1]

    # Find the first adequate measure to run alignment on (no tied note)
    downbeat_counters = {}  # Store the counters at each downbeat when additional_downbeats is used
    transcription_counter = PitchTracker()
    found_onsets = 0
    valid_downbeats_counter = 0
    bar_counter = 0
    valid_downbeat_to_bar_idx = {}
    for token in transcribed_tokens_reversed:
        # print("###", token, transcription_counter, found_onsets, sep="\n")
        if isinstance(token, tk.Bar):
            if found_onsets >= min_onsets:
                # This is a valid downbeat, we store the counter if needed
                if valid_downbeats_counter in downbeats_to_fetch:
                    downbeat_counters[valid_downbeats_counter] = transcription_counter.copy(
                        deep=True
                    )
                valid_downbeat_to_bar_idx[valid_downbeats_counter] = bar_counter
                valid_downbeats_counter += 1
                found_onsets = 0  # Make sure that the next db has at least min_onsets onsets too
                if len(downbeat_counters) >= len(downbeats_to_fetch):
                    break
            bar_counter += 1
        elif isinstance(token, tk.Note):
            if hasattr(token, "_tied_prev"):
                logger.trace(f"Ignoring tied note {note_number_to_name(token.pitch)}")
                continue

            # Increment pitch
            transcription_counter.add(token.pitch)
            found_onsets += 1

            # Tweaks for ornament counting
            # For trills, at least one of main pitch, and one of either half or full next step
            if token.trill_mark:
                transcription_counter.mark_minimum(token.pitch)
                # Half/Full step above
                transcription_counter.merge(token.pitch + 1, token.pitch + 2)
                transcription_counter.add(token.pitch + 1)  # Also update the +2 as linked
                transcription_counter.mark_minimum(token.pitch + 1)  # Also update the +2 as linked
            # For mordents, one more occurrence of main note, and one more occurrence of half or full step above
            if token.mordent:
                transcription_counter.add(token.pitch)
                transcription_counter.merge(token.pitch + 1, token.pitch + 2)
                transcription_counter.add(token.pitch + 1)  # Also update the +2 as linked
            # For inverted mordents, one more occurrence of main note, and one more occurrence of half or full step below
            if token.inverted_mordent:
                transcription_counter.add(token.pitch)
                transcription_counter.merge(token.pitch - 1, token.pitch - 2)
                transcription_counter.add(token.pitch - 1)  # Also update the -2 as linked
            # For turns, one more occurrence of main note, and one more occurrence of half or full step above and below
            if token.turn:
                transcription_counter.add(token.pitch)
                transcription_counter.merge(token.pitch + 1, token.pitch + 2)
                transcription_counter.add(token.pitch + 1)  # Also update the +2 as linked
                transcription_counter.merge(token.pitch - 1, token.pitch - 2)
                transcription_counter.add(token.pitch - 1)  # Also update the -2 as linked
    # Make sure the counter on each downbeat has the same merged pitches (needed for the alignment)
    # They can differ if an ornament happens after the first detected downbeat
    for db_idx, counter in downbeat_counters.items():
        if counter.merged_pitches != transcription_counter.merged_pitches:
            # print(f"Updating {counter.merged_pitches} from {transcription_counter.merged_pitches}")
            for pitch, linked_pitches in transcription_counter.merged_pitches.items():
                counter: PitchTracker
                counter.merge(pitch, *linked_pitches)
            # print(f"  to {counter.merged_pitches}")

    # print(f"Downbeat counters:")
    # for db_idx, counter in downbeat_counters.items():
    #     print(f"  Downbeat {db_idx}: {counter}")

    # Now go over the (reversed) notes in the performance, and get the best candidate for each
    # transcription counters
    # QUESTION: is it better to keep the same performance_counter over the different downbeats?
    # or to reset it?
    #   * reset it:
    #      + avoid merging pitches not needed for that measure
    #      - impacted by an error in the previous downbeat detection
    #   * not reset it:  -> solution chosen for now
    #      + not impacted by a previous error
    #      - merges pitches eventually not needed
    performance_counter = PitchTracker.from_merged_pitches(transcription_counter.merged_pitches)
    score_histories = {db_idx: [] for db_idx in downbeat_counters.keys()}
    last_db_score = performance_counter.compute_score(transcription_counter)  # The one from last db
    max_score = last_db_score + 5
    max_notes = 100
    for i, pitch in enumerate(performances_pitches_reversed):
        if last_db_score > max_score or i > max_notes:
            logger.trace(f"Reached max score {max_score} or max notes {max_notes}")
            break
        performance_counter.add(pitch)
        for db_idx, counter in downbeat_counters.items():
            score = counter.compute_score(performance_counter)
            score_histories[db_idx].append(score)
        last_db_score = score
    num_notes_consumed = i

    # # Print a table of the scores history for each downbeat (one column for each)
    # import tabulate
    #
    # table = tabulate.tabulate(
    #     [
    #         (i, note_number_to_name(pitch), *(s[i] for s in score_histories.values()))
    #         for i, pitch in enumerate(performances_pitches_reversed[:num_notes_consumed])
    #     ],
    #     headers=["Note", "Pitch", *score_histories.keys()],
    #     tablefmt="github",
    # )
    # print(table)

    token_idx_to_note_idx = []
    best_score_idx_last_db = 0
    for db_idx, history in score_histories.items():
        # The best score is the minimum one
        # best_score_idx = len(history) - history[::-1].index(min(history)) - 1  # prefer later in case of tie
        best_score_idx = history.index(min(history))  # prefer earlier in case of tie
        if best_score_idx <= best_score_idx_last_db:
            logger.warning(
                f"Best score from db {db_idx} has a lower index than previous one "
                f"({best_score_idx_last_db} vs {best_score_idx}), forcing to look "
                f"for minimum from last best score."
            )
            clipped_history = history[best_score_idx_last_db + 1 :]
            best_score_idx = (
                clipped_history.index(min(clipped_history)) + best_score_idx_last_db + 1
            )
        # print(best_score_idx)
        token_idx = bar_token_indices[
            len(bar_token_indices) - valid_downbeat_to_bar_idx[db_idx] - 1
        ]
        pitch_idx = len(performances_pitches_reversed) - best_score_idx - 1
        token_idx_to_note_idx.append((token_idx, pitch_idx))
        best_score_idx_last_db = best_score_idx
    # Check that the list is sorted (both values)
    token_idx_to_note_idx = token_idx_to_note_idx[::-1]
    if not all(
        t1 < t2 and n1 < p2
        for (t1, n1), (t2, p2) in zip(token_idx_to_note_idx, token_idx_to_note_idx[1:])
    ):
        raise ValueError(
            f"token_idx_to_start_time is not sorted, something must have gone wrong: "
            f"{token_idx_to_note_idx}"
        )
    return token_idx_to_note_idx


def old_mask_unallowed(
    tensor: torch.Tensor,
    allowed_indices: list[int],
    mask_value: float = -torch.inf,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Mask the tensor at the specified indices with the given mask value.
    """
    if not inplace:
        tensor = tensor.clone()
    mask = torch.zeros(tensor.shape[-1], dtype=torch.bool, device=tensor.device)
    mask[allowed_indices] = True
    tensor[..., ~mask] = mask_value
    return tensor


def old_decode_next_token_constrained(
    token_type: type[tk.AnyToken],
    expert_outputs: dict[str, torch.Tensor],
    state: tk.TranscriptionState,
    max_constraint_iterations: int | None = None,
) -> tk.AnyToken:
    # # StartOfScore
    # if token_type == tk.StartOfScore:
    #     return tk.StartOfScore()

    # First, we deal with all tokens without interactions between attributes:
    # EOS, Bar, KS, Clef, ChordEnd, TupletEnd

    # EndOfScore
    if token_type == tk.EndOfScore:
        return tk.EndOfScore()

    # Bar
    if token_type == tk.Bar:
        # numerator = expert_outputs["bar_numerator"][0]
        # denominator_idx = expert_outputs["bar_denominator"].argmax().item()
        # denominator = TokenTensor.TS_DENOMINATOR_INDEXER_INV[denominator_idx]
        time_signature_idx = expert_outputs["bar_time_signature"].argmax().item()
        numerator, denominator = TokenTensor.TS_INDEXER_INV[time_signature_idx]
        is_nominal = bool(expert_outputs["bar_is_nominal"].argmax().item())
        return tk.Bar(
            # numerator=round(clip(numerator.item(), lower=1)),
            numerator=numerator,
            denominator=denominator,
            is_nominal=is_nominal,
        )

    # KeySignature
    if token_type == tk.KeySignature:
        fifths = expert_outputs["key_signature"].argmax().item() - 7
        return tk.KeySignature(accidentals=fifths)

    # Clef
    if token_type == tk.Clef:
        staff_idx = expert_outputs["staff_idx"].argmax().item()
        clef_idx = expert_outputs["clef_type"].argmax().item()
        clef_type = "G" if clef_idx == 0 else "F"
        return tk.Clef(clef=clef_type, staff_idx=staff_idx)

    # Get voices (all subsequent tokens should have a voice_idx)
    voice_outputs = expert_outputs["voice_idx"]

    # ChordEnd
    if token_type == tk.ChordEnd:
        # Find voices that are currently in a chord
        allowed_voices = [
            i for i, v in enumerate(state.voices) if v.in_chord and v.chord.can_resolve
        ]
        voice_idx = (
            old_mask_unallowed(voice_outputs, allowed_indices=allowed_voices).argmax().item()
        )
        return tk.ChordEnd(voice_idx=voice_idx)

    # TupletEnd
    if token_type == tk.TupletEnd:
        # Find voices that are currently in a tuplet
        allowed_voices = [i for i, v in enumerate(state.voices) if v.in_tuplet]
        voice_idx = (
            old_mask_unallowed(voice_outputs, allowed_indices=allowed_voices).argmax().item()
        )
        return tk.TupletEnd(voice_idx=voice_idx)

    # Now, we deal with tokens with interactions between attributes (for example, we can't
    # select the voice without it having an impact on the allowed durations).
    # We do that by computing the combination of attributes with the maximum probability,
    # while also being allowed by the automata

    # Note
    if token_type is tk.Note:
        classification_dimensions = [
            dim
            for dim in expert_outputs.keys()
            if TokenTensor.PROBLEM_TYPES[dim] == ProblemType.CLASSIFICATION
        ]
        classification_distributions = [
            torch.softmax(value, dim=-1)
            for name, value in expert_outputs.items()
            if name in classification_dimensions
        ]

        def build_token(indices: tuple[int, ...]) -> tk.Note:
            dim_to_index = dict(zip(classification_dimensions, indices, strict=True))
            voice = dim_to_index["voice_idx"]
            staff = dim_to_index["staff_idx"]
            duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
            duration_dots = dim_to_index["duration_dots"]
            duration = tk.SymbolicDuration(type=duration_type, dots=duration_dots)
            step = TokenTensor.NOTE_INDEXER_INV[dim_to_index["note_class"]]
            octave = dim_to_index["note_octave"]
            accidental = dim_to_index["note_accidental"] - 2
            tied = dim_to_index["note_tied"] == 1
            stem_direction = TokenTensor.STEM_INDEXER_INV[dim_to_index["note_stem_direction"]]
            fermata = dim_to_index["fermata"] == 1
            arts_and_orns = {
                art_or_orn: dim_to_index[art_or_orn] == 1
                for art_or_orn in ARTICULATIONS_AND_ORNAMENTS
            }
            return tk.Note(
                voice_idx=voice,
                staff_idx=staff,
                duration=duration,
                note_class=step,
                octave=octave,
                accidental=accidental,
                stem_direction=stem_direction,
                tied=tied,
                fermata=fermata,
                **arts_and_orns,
            )

        best_combination, proba = old_find_most_probable_combinations(
            distributions=[dist.tolist() for dist in classification_distributions],
            predicate=lambda x: build_token(x).is_token_allowed(state),
            max_iterations=max_constraint_iterations,
        )

        best_token = build_token(best_combination)
        return best_token

    # Rest
    if token_type == tk.Rest:
        classification_dimensions = [
            dim
            for dim in expert_outputs.keys()
            if TokenTensor.PROBLEM_TYPES[dim] != ProblemType.REGRESSION
        ]
        classification_distributions = [
            torch.softmax(value, dim=-1)
            for name, value in expert_outputs.items()
            if name in classification_dimensions
        ]

        def build_token(indices: tuple[int, ...]) -> tk.Rest:
            dim_to_index = dict(zip(classification_dimensions, indices, strict=True))
            voice = dim_to_index["voice_idx"]
            staff = dim_to_index["staff_idx"]
            measure_rest = dim_to_index["measure_rest"] == 1
            if measure_rest:
                duration_type = tk.SymbolicDurationType.WHOLE
                duration_dots = 0
            else:
                duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
                duration_dots = dim_to_index["duration_dots"]
            duration = tk.SymbolicDuration(type=duration_type, dots=duration_dots)
            fermata = dim_to_index["fermata"] == 1
            return tk.Rest(
                voice_idx=voice,
                staff_idx=staff,
                duration=duration,
                measure_rest=measure_rest,
                fermata=fermata,
            )

        best_combination, proba = old_find_most_probable_combinations(
            distributions=[dist.tolist() for dist in classification_distributions],
            predicate=lambda x: build_token(x).is_token_allowed(state),
            max_iterations=max_constraint_iterations,
        )

        best_token = build_token(best_combination)
        return best_token

    # ChordStart
    if token_type == tk.ChordStart:
        classification_dimensions = [
            dim
            for dim in expert_outputs.keys()
            if TokenTensor.PROBLEM_TYPES[dim] != ProblemType.REGRESSION
        ]
        classification_distributions = [
            torch.softmax(value, dim=-1)
            for name, value in expert_outputs.items()
            if name in classification_dimensions
        ]

        def build_token(indices: tuple[int, ...]) -> tk.ChordStart:
            dim_to_index = dict(zip(classification_dimensions, indices, strict=True))
            voice = dim_to_index["voice_idx"]
            duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
            duration_dots = dim_to_index["duration_dots"]
            return tk.ChordStart(
                voice_idx=voice,
                duration=tk.SymbolicDuration(type=duration_type, dots=duration_dots),
            )

        best_combination, proba = old_find_most_probable_combinations(
            distributions=[dist.tolist() for dist in classification_distributions],
            predicate=lambda x: build_token(x).is_token_allowed(state),
            max_iterations=max_constraint_iterations,
        )

        best_token = build_token(best_combination)
        return best_token

    # TupletStart
    if token_type == tk.TupletStart:
        # raise NotImplementedError("TupletStart not implemented")
        classification_dimensions = [
            dim
            for dim in expert_outputs.keys()
            if TokenTensor.PROBLEM_TYPES[dim] != ProblemType.REGRESSION
        ]
        classification_distributions = [
            (
                torch.softmax(value, dim=-1)
                if TokenTensor.PROBLEM_TYPES[name] == ProblemType.CLASSIFICATION
                # TODO: remove binary classification as everything is now multiclass
                else torch.concat([(x := torch.sigmoid(value)), 1 - x], axis=-1)  # binary classif.
            )
            for name, value in expert_outputs.items()
            if name in classification_dimensions
        ]

        def build_token(indices: tuple[int, ...]) -> tk.TupletStart:
            dim_to_index = dict(zip(classification_dimensions, indices, strict=True))
            voice = dim_to_index["voice_idx"]
            duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
            duration_dots = dim_to_index["duration_dots"]
            n_actual = round(clip(expert_outputs["tuplet_actual_notes"].item(), 1))
            n_normal = round(clip(expert_outputs["tuplet_normal_notes"].item(), 1))
            return tk.TupletStart(
                n_actual=n_actual,
                n_normal=n_normal,
                unit_duration=tk.SymbolicDuration(
                    type=tk.SymbolicDurationType(duration_type),
                    dots=duration_dots,
                ),
                voice_idx=voice,
            )

        best_combination, proba = old_find_most_probable_combinations(
            distributions=[dist.tolist() for dist in classification_distributions],
            predicate=lambda x: build_token(x).is_token_allowed(state),
            max_iterations=max_constraint_iterations,
        )

        best_token = build_token(best_combination)
        return best_token

    elif token_type == tk.TupletEnd:
        # Find voices that are currently in a tuplet
        allowed_voices = [
            i
            for i, v in enumerate(state.voices)
            if v.in_tuplet and not v.in_chord and v.tuplet.completed
        ]
        voice_idx = (
            old_mask_unallowed(voice_outputs, allowed_indices=allowed_voices).argmax().item()
        )
        return tk.TupletEnd(voice_idx=voice_idx)

    raise ValueError(f"Unknown token type {token_type}")


def old_find_most_probable_combinations(
    distributions: list[list[float]],
    predicate: Callable[[tuple[int, ...]], bool],
    max_iterations: int | None = None,
) -> tuple[tuple[int, ...], float]:
    # Validate distributions
    for distribution in distributions:
        if not isinstance(distribution, list):
            raise ValueError(
                f"Expected distribution to be a list of floats, got {type(distribution)}"
            )
        if not all(0 <= p <= 1 for p in distribution):
            raise ValueError(f"All probabilities must lay between 0 and 1: {distribution}")

    # Add indices to each distribution and sort them by decreasing probabilities
    # (so that we first explore highly probable combinations)
    sorted_idxs = [
        sorted(range(len(dist)), key=lambda i: dist[i], reverse=True) for dist in distributions
    ]
    n = len(distributions)

    # Function to compute the probability of a combination
    def proba_combination(indices: tuple[int, ...]) -> float:
        p = 1.0
        for dim, pos in enumerate(indices):
            orig_idx = sorted_idxs[dim][pos]
            p *= distributions[dim][orig_idx]
        return p

    # Create the max-heap (min-heap with negative probabilities)
    heap: list[tuple[float, tuple[int, ...]]] = []

    # Keep track of already visited combinations
    visited = set()

    # We start the heap with the most probable item
    # (index 0 in all sorted distributions)
    init = tuple([0] * n)
    init_prob = proba_combination(init)
    heapq.heappush(heap, (-init_prob, init))
    visited.add(init)

    # While there are still items in the heap, yield them
    i = -1
    while heap:
        i += 1
        # Get most probable combination
        neg_p, sorted_tuple = heapq.heappop(heap)
        p = -neg_p
        # Map sorted‐space positions to original indices
        orig_tuple = tuple(sorted_idxs[dim][pos] for dim, pos in enumerate(sorted_tuple))
        # If the predicate is true, return the combination
        if predicate(orig_tuple):
            return orig_tuple, p
        # Generate neighbors by incrementing the position in each dimension
        for dim in range(n):
            if sorted_tuple[dim] + 1 < len(sorted_idxs[dim]):
                neighbor = list(sorted_tuple)
                neighbor[dim] += 1
                neighbor = tuple(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    heapq.heappush(heap, (-proba_combination(neighbor), neighbor))

        if max_iterations is not None and i >= max_iterations:
            raise StopIteration("max_iterations reached")

    raise StopIteration("No combination found")


def get_most_probable_paths(
    router_distributions: torch.Tensor,  # B * K
    expert_distributions: list[list[torch.Tensor]],  # K-list of Ni-list of B * Fij
    num_paths_to_retrieve: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    params:
        router_distributions: Router probabilities
        expert_distributions: A list of length K. E[i] is a list of length N_i containing
           tensors of shape (B, F_i_j) representing attribute probabilities
        num_paths_to_retrieve: Number of top combinations to return
    returns:
        best_scores: (B, M) Probabilities of the top M tokens
        best_types:  (B, M) The router type index (0 to K-1) for each top token
        best_paths:  (B, M, max_N) The attribute indices chosen (padded with -1)
    """
    batch_size, num_ttypes = router_distributions.shape
    device = router_distributions.device

    # Avoid log(0)
    log_router_distributions = torch.log(torch.clamp(router_distributions, min=1e-12))

    all_type_scores = []
    all_type_paths = []
    all_type_indices = []

    max_num_attributes = max(len(attrs) for attrs in expert_distributions)

    for i in range(num_ttypes):
        # Initial score: log(p_i)
        beam_scores = log_router_distributions[:, i : i + 1]  # B * 1

        # Initial empty paths
        beam_paths = torch.empty((batch_size, 1, 0), dtype=torch.long, device=device)  # B * 1 * 0

        num_attributes = len(expert_distributions[i])
        for j in range(num_attributes):
            # log(q_i_j) / N_i
            attr_log_probs = (1.0 / num_attributes) * torch.log(
                torch.clamp(expert_distributions[i][j], min=1e-12)
            )  # B * F_ij
            F_ij = attr_log_probs.size(-1)

            # Combine paths
            # beam_scores: (B, current_M, 1)
            # attr_log_probs: (B, 1, F_ij)
            # expanded_scores: (B, current_M, F_ij)
            expanded_scores = beam_scores.unsqueeze(-1) + attr_log_probs.unsqueeze(1)

            # Flatten to find top M among (current_M * F_ij) candidates
            flat_scores = expanded_scores.view(batch_size, -1)
            current_M = min(num_paths_to_retrieve, flat_scores.size(-1))

            beam_scores, top_idx = torch.topk(flat_scores, current_M, dim=-1)

            prefix_idx = top_idx // F_ij
            attr_idx = top_idx % F_ij

            # Gather historical paths
            expanded_prefix = prefix_idx.unsqueeze(-1).expand(-1, -1, beam_paths.size(-1))
            gathered_paths = torch.gather(beam_paths, dim=1, index=expanded_prefix)

            # Concatenate new attribute
            beam_paths = torch.cat([gathered_paths, attr_idx.unsqueeze(-1)], dim=-1)

        # Pad paths up to max_N
        if num_attributes < max_num_attributes:
            padding = torch.full(
                (batch_size, beam_paths.size(1), max_num_attributes - num_attributes),
                -1,
                dtype=torch.long,
                device=device,
            )
            beam_paths = torch.cat([beam_paths, padding], dim=-1)

        all_type_scores.append(beam_scores)
        all_type_paths.append(beam_paths)
        all_type_indices.append(torch.full_like(beam_scores, i, dtype=torch.long))

    # Combine results from all K types
    final_scores = torch.cat(all_type_scores, dim=-1)  # (B, K*M)
    final_types = torch.cat(all_type_indices, dim=-1)  # (B, K*M)
    final_paths = torch.cat(all_type_paths, dim=1)  # (B, K*M, max_N)

    # Final top M reduction
    final_M = min(num_paths_to_retrieve, final_scores.size(-1))
    best_scores, best_idx = torch.topk(final_scores, final_M, dim=-1)

    best_types = torch.gather(final_types, dim=-1, index=best_idx)

    best_idx_expanded = best_idx.unsqueeze(-1).expand(-1, -1, max_num_attributes)
    best_paths = torch.gather(final_paths, dim=1, index=best_idx_expanded)

    return torch.exp(best_scores), best_types, best_paths


def get_token_from_path_if_valid(
    token_type: type[tk.AnyToken],
    classification_dims_indices: torch.LongTensor,
    regression_dims_values: dict[str, float],
    state: tk.TranscriptionState,
) -> tk.AnyToken | None:
    classification_dims = [
        dim
        for dim in TokenTensor.TOKEN_TYPE_TO_DIMS[token_type]
        if dim != "token_type" and TokenTensor.PROBLEM_TYPES[dim] == ProblemType.CLASSIFICATION
    ]
    assert (
        len(classification_dims) == (classification_dims_indices != -1).sum().item()
    ), f"Mismatching dimensions and paths: {len(classification_dims)} dimensions with path {classification_dims_indices}"
    dim_to_index = dict(
        zip(classification_dims, classification_dims_indices.tolist())
    )  # no strict as path padded with -1

    # StartOfScore
    if token_type == tk.StartOfScore:
        token = tk.StartOfScore()

    # EndOfScore
    elif token_type == tk.EndOfScore:
        token = tk.EndOfScore()

    # Bar
    elif token_type == tk.Bar:
        time_signature_idx = dim_to_index["bar_time_signature"]
        numerator, denominator = TokenTensor.TS_INDEXER_INV[time_signature_idx]
        is_nominal = dim_to_index["bar_is_nominal"] == 1
        token = tk.Bar(numerator=numerator, denominator=denominator, is_nominal=is_nominal)

    # KeySignature
    elif token_type == tk.KeySignature:
        fifths = dim_to_index["key_signature"] - 7
        token = tk.KeySignature(accidentals=fifths)

    # Clef
    elif token_type == tk.Clef:
        staff_idx = dim_to_index["staff_idx"]
        clef_idx = dim_to_index["clef_type"]
        clef_type = "G" if clef_idx == 0 else "F"
        token = tk.Clef(clef=clef_type, staff_idx=staff_idx)

    # ChordEnd
    elif token_type == tk.ChordEnd:
        voice_idx = dim_to_index["voice_idx"]
        token = tk.ChordEnd(voice_idx=voice_idx)

    # TupletEnd
    elif token_type == tk.TupletEnd:
        voice_idx = dim_to_index["voice_idx"]
        token = tk.TupletEnd(voice_idx=voice_idx)

    # Note
    elif token_type == tk.Note:
        voice_idx = dim_to_index["voice_idx"]
        staff_idx = dim_to_index["staff_idx"]
        duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
        duration_dots = dim_to_index["duration_dots"]
        duration = tk.SymbolicDuration(type=duration_type, dots=duration_dots)
        step = TokenTensor.NOTE_INDEXER_INV[dim_to_index["note_class"]]
        octave = dim_to_index["note_octave"]
        accidental = dim_to_index["note_accidental"] - 2
        tied = dim_to_index["note_tied"] == 1
        stem_direction = TokenTensor.STEM_INDEXER_INV[dim_to_index["note_stem_direction"]]
        fermata = dim_to_index["fermata"] == 1
        arts_and_orns = {
            art_or_orn: dim_to_index[art_or_orn] == 1 for art_or_orn in ARTICULATIONS_AND_ORNAMENTS
        }
        token = tk.Note(
            voice_idx=voice_idx,
            staff_idx=staff_idx,
            duration=duration,
            note_class=step,
            octave=octave,
            accidental=accidental,
            stem_direction=stem_direction,
            tied=tied,
            fermata=fermata,
            **arts_and_orns,
        )

    # Rest
    elif token_type == tk.Rest:
        voice_idx = dim_to_index["voice_idx"]
        staff_idx = dim_to_index["staff_idx"]
        duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
        duration_dots = dim_to_index["duration_dots"]
        duration = tk.SymbolicDuration(type=duration_type, dots=duration_dots)
        measure_rest = dim_to_index["measure_rest"] == 1
        fermata = dim_to_index["fermata"] == 1
        token = tk.Rest(
            voice_idx=voice_idx,
            staff_idx=staff_idx,
            duration=duration,
            measure_rest=measure_rest,
            fermata=fermata,
        )

    # ChordStart
    elif token_type == tk.ChordStart:
        voice_idx = dim_to_index["voice_idx"]
        duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
        duration_dots = dim_to_index["duration_dots"]
        duration = tk.SymbolicDuration(type=duration_type, dots=duration_dots)
        token = tk.ChordStart(voice_idx=voice_idx, duration=duration)

    # TupletStart
    elif token_type == tk.TupletStart:
        voice_idx = dim_to_index["voice_idx"]
        n_actual = round(clip(regression_dims_values["tuplet_actual_notes"], 1))
        n_normal = round(clip(regression_dims_values["tuplet_normal_notes"], 1))
        unit_duration_type = TokenTensor.DURATION_INDEXER_INV[dim_to_index["duration_type"]]
        unit_duration_dots = dim_to_index["duration_dots"]
        unit_duration = tk.SymbolicDuration(type=unit_duration_type, dots=unit_duration_dots)
        token = tk.TupletStart(
            n_actual=n_actual,
            n_normal=n_normal,
            unit_duration=unit_duration,
            voice_idx=voice_idx,
        )

    else:
        raise ValueError(f"Unknown token type: {token_type}")

    if token.is_token_allowed(state):
        return token
    else:
        return None


def get_most_probable_valid_tokens(
    router_distributions: torch.Tensor,
    expert_outputs: dict[str, dict[str, torch.Tensor]],
    states: list[tk.TranscriptionState],
    max_depth: int = 10,
) -> tuple[list[tk.AnyToken | None], list[float | None]]:
    """
    Find the most probable valid token for a batch of predictions.
    The probability computation is vectorized, but not the validity check.

    params:
        router_distribution: Router output probabilities, shape (B, K)
        expert_outputs: Output of the expert heads, dict (head) of dict (attr) of tensors (B, Fij)
        states: Transcription states, used to check whether a token is allowed
        max_depth: The maximum number of tokens to check per element of the batch
    returns:
        best_tokens: List of the best token for each element of the batch (or None if exceed depth)
        best_probas: Probability of each token in best_tokens (None if exceed depth)
    """
    B, K = router_distributions.shape

    expert_classification_distributions_lists = [
        [
            expert_outputs[ttype.as_str()][attr].softmax(-1)
            for attr in TokenTensor.TOKEN_TYPE_TO_DIMS[ttype]
            if attr != "token_type"
            and TokenTensor.PROBLEM_TYPES[attr] == ProblemType.CLASSIFICATION
        ]
        for ttype in tk.TOKEN_TYPES
    ]

    # We start by fetching exactly the top `max_depth` candidates
    # It might be theorically faster to use a heap and check the validity as we build the paths,
    # but it's not expensive to compute all top M paths anyway due to torch vectorization
    scores, types, paths = get_most_probable_paths(
        router_distributions=router_distributions,
        expert_distributions=expert_classification_distributions_lists,
        num_paths_to_retrieve=max_depth,
    )

    # Prepare output lists
    best_tokens: list[tk.AnyToken | None] = [None] * B
    best_probas: list[float | None] = [None] * B

    # # Track successes
    # resolved = torch.zeros(B, dtype=torch.bool, device=device)

    # Move to CPU once to avoid doing it at each `.item()`
    types_cpu = types.cpu()
    paths_cpu = paths.cpu()
    scores_cpu = scores.cpu()

    # Iterate through batch elements
    for b in range(B):
        # Check candidates in descending order of probability
        for m in range(max_depth):
            t_val = TokenTensor.TOKEN_INDEXER_INV[types_cpu[b, m].item()]
            p_val = paths_cpu[b, m]
            s_val = scores_cpu[b, m].item()

            # Check candidate's validity
            valid_token = get_token_from_path_if_valid(
                token_type=t_val,
                classification_dims_indices=p_val,
                regression_dims_values={
                    attr: expert_outputs[t_val.as_str()][attr][b, 0].item()
                    for attr in TokenTensor.TOKEN_TYPE_TO_DIMS[t_val]
                    if attr != "token_type"
                    and TokenTensor.PROBLEM_TYPES[attr] == ProblemType.REGRESSION
                },
                state=states[b],
            )
            # If valid, store it (along with its prob) and break
            if valid_token is not None:
                best_tokens[b] = valid_token
                best_probas[b] = s_val
                # resolved[b] = True
                break

    # # Inform about unsolved elements
    # if not resolved.all():
    #     failed_count = (~resolved).sum().item()
    #     logger.warning(
    #         f"{failed_count} elements failed to find a valid token within max_depth={max_depth}."
    #     )

    return best_tokens, best_probas
