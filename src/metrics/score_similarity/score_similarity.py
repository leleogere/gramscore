import pprint
import music21 as m21
from enum import IntEnum

import numpy as np
from numba import jit


class ScoreErrors(IntEnum):
    Clef = 0
    KeySignature = 1
    TimeSignature = 2
    NoteDeletion = 3
    NoteInsertion = 4
    NoteSpelling = 5
    NoteDuration = 6
    StemDirection = 7
    Beams = 8
    Tie = 9
    StaffAssignment = 10
    Voice = 11

    TrillTP = 12
    TrillFP = 13
    TrillTN = 14
    TrillFN = 15

    StaccatoTP = 16
    StaccatoFP = 17
    StaccatoTN = 18
    StaccatoFN = 19

    GraceTP = 20
    GraceFP = 21
    GraceTN = 22
    GraceFN = 23

    RestDeletion = 24
    RestInsertion = 25
    RestDuration = 26


def score_alignment(score_a, score_b):
    """Compare two musical scores.

    Parameters:

        aScore/bScore: music21.stream.Score objects

    Return value:

        (path, d):
            path is a list of tuples containing pairs of matching offsets
            d is the alignment matrix
    """

    def score_to_pitch_offset_list(s):
        """Convert a piano score into a list of tuples containing pitches

        Parameters
        ----------
            s: a music21.Stream containing two music21.stream.PartStaff

        Returns
        -------
            np.ndarray, shape=(N,) offsets
            np.ndarray, shape=(N, 128) a binary array to indicate the presence of a
                pitch onset at that location
        """
        my_dict = {}
        for n in s.flatten().notes:
            if n.offset not in my_dict:
                my_dict[n.offset] = np.zeros(128, dtype=np.uint16)

            if n.isChord:
                for p in n.pitches:
                    my_dict[n.offset][p.midi] = 1
            else:
                my_dict[n.offset][n.pitch.midi] = 1
        offsets = list(my_dict.keys())
        pitches = list(my_dict.values())
        return np.array(offsets), np.array(pitches)

    @jit(nopython=True, cache=True)
    def cost_matrix(s: np.ndarray, t: np.ndarray) -> np.ndarray:
        d = np.zeros((s.shape[0] + 1, t.shape[0] + 1))
        d[1:, 0] = np.inf
        d[0, 1:] = np.inf
        for j in range(t.shape[0]):
            for i in range(s.shape[0]):
                d[i + 1, j + 1] = (
                    min(d[i, j + 1], d[i + 1, j], d[i, j]) + np.bitwise_xor(s[i], t[j]).sum()
                )
        return d

    offsets_a, pitches_a = score_to_pitch_offset_list(score_a)
    offsets_b, pitches_b = score_to_pitch_offset_list(score_b)
    if not pitches_a.shape[0] or not pitches_b.shape[0]:
        return [(0, 0)]
    d = cost_matrix(pitches_a, pitches_b)

    # Uncomment for alignment path visual
    # d_img = d
    # d_img[np.isinf(d_img)] = d_img[np.logical_not(np.isinf(d_img))].max()
    # d_img = (d_img / d_img.max() * 255).astype(np.uint8)

    i, j = (d.shape[0] - 1, d.shape[1] - 1)
    path = []
    while i and j:
        # d_img[i - 1, j - 1] = 255
        path.insert(0, (offsets_a[i - 1], offsets_b[j - 1]))

        idx = np.argmin([d[i - 1, j], d[i, j - 1], d[i - 1, j - 1]])
        if idx == 0:
            i = i - 1
        elif idx == 1:
            j = j - 1
        else:
            i, j = i - 1, j - 1
    return path


def score_similarity(
    estScore: m21.stream.Score | str,
    gtScore: m21.stream.Score | str,
    partMapping={0: "right", 1: "left"},
    full=False,
    strip_ties=True,
) -> dict[str, float | int | None]:
    """Compare two musical scores.

    Parameters
    ----------
    estScore: music21.stream.Score
        Predicted piano score or path to its location.
    gtScore: music21.stream.Score
        Ground truth piano score or path to its location.
    partMapping: dict[int, str]
        A dictionary mapping part indices to hands. The keys are the part indices
        (0-based) and the values are either "right" or "left".
    full: bool
        Whether to run a full comparison including all attributes or not.
        Will run significantly faster if not full.
    strip_ties: bool
        Whether to strip ties from the scores or not.
    Returns
    -------
    Dict[str, float | int | None]:
        A dictionary containing the number of errors for each type of error.
    """
    if isinstance(estScore, str):
        estScore = m21.converter.parse(estScore, forceSource=True)
    estScore = estScore.expandRepeats()
    if strip_ties:
        estScore = estScore.stripTies(preserveVoices=False)
    if isinstance(gtScore, str):
        gtScore = m21.converter.parse(gtScore, forceSource=True)
    gtScore = gtScore.expandRepeats()
    if strip_ties:
        gtScore = gtScore.stripTies(preserveVoices=False)
    assert isinstance(estScore, m21.stream.Stream)
    assert isinstance(gtScore, m21.stream.Stream)

    def score_to_list(aScore):
        """Convert a piano score into a list of tuples

        Parameters
        ----------
        aScore: a music21.Stream containing two music21.stream.PartStaff

        Returns
        -------
        List[Tuple[float, int, music21 object]]
            offset is a real number indicating the offset of an object in music21 terms
            staff is an integer indicating the staff (0 = top, 1 = bottom)
            object is a music21 object
        """

        # Classes to consider
        CLASSES = (m21.bar.Barline, m21.note.Note, m21.chord.Chord, m21.note.Rest)

        def stream_to_list(s, hand) -> list[tuple[float, tuple[int, m21.base.Music21Object]]]:
            elements_with_offsets = {}
            for el in s.recurse():
                if isinstance(el, CLASSES):
                    offset = el.getOffsetInHierarchy(s)
                    elements_with_offsets.setdefault(offset, []).append((hand, el))
            return sorted(elements_with_offsets.items())

        def flatten_stream(s: m21.stream.Stream):
            new_stream = m21.stream.Stream()
            for el in s.flatten():
                if isinstance(el, m21.note.Note):
                    new_stream.insert(el)
                elif isinstance(el, m21.chord.Chord):
                    voice_ctx = el.getContextByClass('Voice')
                    for n in el.notes:
                        # Propagate attributes from Chord to Note if missing on Note
                        if voice_ctx:
                            n._voice_id_override = voice_ctx.id
                        if el.articulations and not n.articulations:
                            n.articulations = [a for a in el.articulations]
                        if el.expressions and not n.expressions:
                            n.expressions = [e for e in el.expressions]
                        if el.tie and not n.tie:
                            n.tie = el.tie

                        # Insert at the Chord's offset
                        new_stream.insert(el.offset, n)
                elif isinstance(el, m21.note.Rest):
                    new_stream.insert(el)
                elif isinstance(
                    el,
                    (
                        m21.clef.Clef,
                        m21.key.Key,
                        m21.key.KeySignature,
                        m21.meter.TimeSignature,
                    ),
                ):
                    new_stream.insert(el)

            for el in s.recurse():
                if isinstance(el, (m21.stream.Measure, m21.bar.Barline)):
                    new_stream.insert(el.getOffsetInHierarchy(s), m21.bar.Barline())
            return new_stream

        # We only report staff/hand scores if we are sure which part corresponds to
        # which part
        validParts = True
        parts = aScore.getElementsByClass([m21.stream.PartStaff, m21.stream.Part])
        topStaffList = []
        bottomStaffList = []
        kept_streams = []  # list to hold streams to prevent GC

        for i, part in enumerate(parts):
            if i not in partMapping:
                # All notes for both scores will be merged into the top staff later
                fs = flatten_stream(part)
                kept_streams.append(fs)
                topStaffList += stream_to_list(fs, 0)
                validParts = False
            elif partMapping[i] == "right":
                fs = flatten_stream(part)
                kept_streams.append(fs)
                topStaffList += stream_to_list(fs, 0)
            else:
                fs = flatten_stream(part)
                kept_streams.append(fs)
                bottomStaffList += stream_to_list(fs, 1)

        elTimes = []
        elList = []
        tIterator = iter(topStaffList)
        bIterator = iter(bottomStaffList)
        tEl = next(tIterator, None)
        bEl = next(bIterator, None)

        while tEl or bEl:
            if not tEl:
                elTimes.append(bEl[0])
                elList.append(bEl[1])
                bEl = next(bIterator, None)
            elif not bEl:
                elTimes.append(tEl[0])
                elList.append(tEl[1])
                tEl = next(tIterator, None)
            else:
                if tEl[0] < bEl[0]:
                    elTimes.append(tEl[0])
                    elList.append(tEl[1])
                    tEl = next(tIterator, None)
                elif tEl[0] > bEl[0]:
                    elTimes.append(bEl[0])
                    elList.append(bEl[1])
                    bEl = next(bIterator, None)
                else:
                    elTimes.append(tEl[0])
                    elList.append(tEl[1] + bEl[1])
                    tEl = next(tIterator, None)
                    bEl = next(bIterator, None)
        if elTimes and elList:
            elTimes, elList = zip(*sorted(zip(elTimes, elList), key=lambda x: x[0]))

        return np.array(elTimes), elList, validParts, kept_streams

    def count_objects_by_type(a_set):
        """Count objects in a set by type

        Parameters
        ----------
        a_set: list of tuples (staff, object)

        Returns
        -------
            dict: {type: count}
        """
        counts = {"Note": 0, "Rest": 0}
        for obj in a_set:
            if isinstance(obj[1], m21.note.Note):
                counts["Note"] += 1
            elif isinstance(obj[1], m21.chord.Chord):
                counts["Note"] += len(obj[1].pitches)
            elif isinstance(obj[1], m21.note.Rest):
                counts["Rest"] += 1
        return counts

    def compare_sets(a_set, b_set):
        """Compare two sets of concurrent musical objects.

        Parameters
        ----------
        a_set/b_set: list of tuples (staff, object)
            staff is an integer indicating the staff (1 = top, 2 = bottom)
            object is a music21 object

        Returns
        -------
            np.ndarray: An array with the differences between the two sets
                (see definition of errors below)
        """

        def pitchComp(pA, pB):
            """Compare two pitches
            In contrast to music21.pitch.Pitch.__eq__, this function allows
            no accidental to match a natural accidental.
            """
            if (pA.octave == pB.octave) and (pA.step == pB.step) and (pA.microtone == pB.microtone):
                if pA.accidental == pB.accidental:
                    return True
                if pA.accidental is None and getattr(pB.accidental, 'name', None) == 'natural':
                    return True
                if getattr(pA.accidental, 'name', None) == 'natural' and pB.accidental is None:
                    return True
            return False

        def findEnharmonicEquivalent(note, a_set):
            """Find the first enharmonic equivalent in a set

            Parameters
            ----------
            note: a music21.note.Note object
            a_set: list of tuples (staff, object)
                staff is an integer indicating the staff (0 = top, 1 = bottom)
                object is a music21 object

            Returns
            -------
                int: index of the first enharmonic equivalent of note in a_set -1 otherwise
            """
            for i, obj in enumerate(a_set):
                if isinstance(obj[1], m21.note.Note) and obj[1].pitch.ps == note.pitch.ps:
                    return i
            return -1

        def compare_obj(aObj, bObj):
            """
            Compare two music21 objects and return True if they are similar, False otherwise.

            Parameters
            ----------
                aObj: The first music21 object to compare.
                bObj: The second music21 object to compare.

            Returns
            -------
                bool: True if the two objects are similar, False otherwise.
            """
            if aObj == bObj:
                return True
            if type(aObj) != type(bObj):
                return False
            if isinstance(aObj, m21.note.Rest):
                if abs(aObj.duration.quarterLength - bObj.duration.quarterLength) < 1e-3:
                    return True
            if isinstance(aObj, m21.key.Key) or isinstance(aObj, m21.key.KeySignature):
                return aObj.sharps == bObj.sharps
            if type(aObj) != type(bObj):
                return False
            if isinstance(aObj, m21.note.Note):
                if (
                    pitchComp(aObj.pitch, bObj.pitch)
                    and abs(aObj.duration.quarterLength - bObj.duration.quarterLength) < 1e-3
                    and aObj.stemDirection == bObj.stemDirection
                ):
                    return True
            if isinstance(aObj, m21.bar.Barline):
                return True
            if isinstance(aObj, m21.stream.Measure):
                return True
            if isinstance(aObj, m21.clef.Clef):
                return True
            if isinstance(aObj, m21.meter.TimeSignature):
                if aObj.numerator == bObj.numerator and aObj.beatCount == bObj.beatCount:
                    return True
            return False

        def get_beams(note_obj: m21.note.Note):
            return '_'.join(
                ['-'.join([b.type, b.direction]) if b.direction else b.type for b in note_obj.beams]
            )

        def get_grace(note_obj: m21.note.Note):
            return note_obj.duration.isGrace

        def get_staccato(note_obj: m21.note.Note):
            if not hasattr(note_obj, 'articulations'):
                return False
            staccatos = (m21.articulations.Staccatissimo, m21.articulations.Staccato)
            return any(isinstance(e, staccatos) for e in note_obj.articulations)

        def get_clef(note_obj: m21.note.Note):
            context = note_obj.getContextByClass('Clef')
            return context.name if context is not None else ''

        def get_key_sig(note_obj: m21.note.Note):
            keyObj = note_obj.getContextByClass('Key') or note_obj.getContextByClass('KeySignature')
            return keyObj.sharps if keyObj else 0

        def get_tie(note_obj: m21.note.Note):
            return note_obj.tie.type if note_obj.tie is not None else ''

        def get_time_sig(note_obj: m21.note.Note):
            context = note_obj.getContextByClass(m21.meter.TimeSignature)
            return context.numerator / context.denominator if context else ""

        def get_trill(note_obj: m21.note.Note):
            if not hasattr(note_obj, 'expressions'):
                return False
            trills = (
                m21.expressions.Trill,
                m21.expressions.InvertedMordent,
                m21.expressions.Mordent,
                m21.expressions.Turn,
            )
            return any(isinstance(e, trills) for e in note_obj.expressions)

        def get_voice(note_obj: m21.note.Note):
            if hasattr(note_obj, '_voice_id_override') and note_obj._voice_id_override is not None:
                return note_obj._voice_id_override
            context = note_obj.getContextByClass('Voice')
            return context.id if context is not None else '1'

        errors = np.zeros((len(ScoreErrors.__members__)), int)

        a = a_set.copy()
        b = b_set.copy()

        def track_note_errors(a, b):
            if isinstance(a, m21.note.Rest) and isinstance(b, m21.note.Rest):
                if abs(a.duration.quarterLength - b.duration.quarterLength) > 1e-3:
                    errors[ScoreErrors.RestDuration] += 1
                if get_voice(a) != get_voice(b):
                    errors[ScoreErrors.Voice] += 1
                return

            if not isinstance(a, (m21.note.Note, m21.chord.Chord)):
                return
            if not isinstance(b, (m21.note.Note, m21.chord.Chord)):
                return

            if abs(a.duration.quarterLength - b.duration.quarterLength) > 1e-3:
                errors[ScoreErrors.NoteDuration] += 1

            if a.stemDirection != b.stemDirection:
                errors[ScoreErrors.StemDirection] += 1

            if get_beams(a) != get_beams(b):
                errors[ScoreErrors.Beams] += 1

            if get_grace(a):
                if get_grace(b):
                    errors[ScoreErrors.GraceTP] += 1
                else:
                    errors[ScoreErrors.GraceFN] += 1
            else:
                if get_grace(b):
                    errors[ScoreErrors.GraceFP] += 1
                else:
                    errors[ScoreErrors.GraceTN] += 1

            if get_staccato(a):
                if get_staccato(b):
                    errors[ScoreErrors.StaccatoTP] += 1
                else:
                    errors[ScoreErrors.StaccatoFN] += 1
            else:
                if get_staccato(b):
                    errors[ScoreErrors.StaccatoFP] += 1
                else:
                    errors[ScoreErrors.StaccatoTN] += 1

            if get_tie(a) != get_tie(b):
                errors[ScoreErrors.Tie] += 1

            if get_trill(a):
                if get_trill(b):
                    errors[ScoreErrors.TrillTP] += 1
                else:
                    errors[ScoreErrors.TrillFN] += 1
            else:
                if get_trill(b):
                    errors[ScoreErrors.TrillFP] += 1
                else:
                    errors[ScoreErrors.TrillTN] += 1

            # Only run slow checks if full is set to True
            if full:
                if get_clef(a) != get_clef(b):
                    errors[ScoreErrors.Clef] += 1
                if get_key_sig(a) != get_key_sig(b):
                    errors[ScoreErrors.KeySignature] += 1
                if get_time_sig(a) != get_time_sig(b):
                    errors[ScoreErrors.TimeSignature] += 1
                if get_voice(a) != get_voice(b):
                    errors[ScoreErrors.Voice] += 1

        # Step 2: Find exact matches
        # We filter a and b by removing matches from b,
        # and building a new 'a' that does not include the matches
        a_temp = []
        for pair in a:
            for bPair in b:
                if pair[0] == bPair[0] and compare_obj(pair[1], bPair[1]):
                    track_note_errors(bPair[1], pair[1])
                    b.remove(bPair)
                    break
            else:
                a_temp.append(pair)
        a = a_temp

        # Step 3
        a_temp = []
        for obj in a:
            if isinstance(obj[1], m21.note.Note):
                for bObj in b:
                    if isinstance(bObj[1], m21.note.Note) and pitchComp(
                        bObj[1].pitch, obj[1].pitch
                    ):
                        if bObj[0] != obj[0]:
                            errors[ScoreErrors.StaffAssignment] += 1
                        else:  # TODO: probably remove the else, only left here for backwards compatibility
                            track_note_errors(bObj[1], obj[1])
                        b.remove(bObj)
                        break
                else:
                    a_temp.append(obj)
            else:
                a_temp.append(obj)
        a = a_temp

        # Step 4: Find matching Rests (regardless of duration, to catch Duration errors)
        a_temp = []
        for obj in a:
            if isinstance(obj[1], m21.note.Rest):
                for bObj in b:
                    if isinstance(bObj[1], m21.note.Rest):
                        if bObj[0] != obj[0]:
                            errors[ScoreErrors.StaffAssignment] += 1
                        track_note_errors(bObj[1], obj[1])
                        b.remove(bObj)
                        break
                else:
                    a_temp.append(obj)  # No Rest match found in b
            else:
                a_temp.append(obj)
        a = a_temp

        # Find enharmonic equivalents and report spelling mistakes and duration mistakes
        a_temp = []
        for obj in a:
            if isinstance(obj[1], m21.note.Note):
                idx = findEnharmonicEquivalent(obj[1], b)
                if idx != -1:
                    if b[idx][0] != obj[0]:
                        errors[ScoreErrors.StaffAssignment] += 1
                    track_note_errors(b[idx][1], obj[1])
                    del b[idx]
                    errors[ScoreErrors.NoteSpelling] += 1
                    continue
            a_temp.append(obj)
        a = a_temp

        counts_a = count_objects_by_type(a)
        counts_b = count_objects_by_type(b)

        errors[ScoreErrors.NoteInsertion] = counts_a["Note"]
        errors[ScoreErrors.NoteDeletion] = counts_b["Note"]
        errors[ScoreErrors.RestInsertion] = counts_a["Rest"]
        errors[ScoreErrors.RestDeletion] = counts_b["Rest"]

        return errors

    def get_set(elements, times, start, end):
        endIdx = np.searchsorted(times, end, side='left')
        set = []
        for el in elements[start:endIdx]:
            set += el
        return set, endIdx

    path = score_alignment(estScore, gtScore)

    aTimes, aElements, est_parts_valid, _est_streams = score_to_list(estScore)
    bTimes, bElements, gt_parts_valid, _gt_streams = score_to_list(gtScore)
    if not (est_parts_valid and gt_parts_valid):
        for i, aElement in enumerate(aElements):
            for j, el in enumerate(aElement):
                aElements[i][j] = (0, el[1])
        for i, bElement in enumerate(bElements):
            for j, el in enumerate(bElement):
                bElements[i][j] = (0, el[1])
    errors = np.zeros((len(ScoreErrors.__members__)), float)
    aIdx, aEnd = 0, 0.0
    bIdx, bEnd = 0, 0.0
    for pair in path:
        if pair[0] != aEnd and pair[1] != bEnd:
            aEnd = pair[0]
            bEnd = pair[1]
            aSet, aIdx = get_set(aElements, aTimes, aIdx, aEnd)
            bSet, bIdx = get_set(bElements, bTimes, bIdx, bEnd)
            errors += compare_sets(aSet, bSet)
        elif pair[0] == aEnd:
            bEnd = pair[1]
        else:
            aEnd = pair[0]

    aSet, _ = get_set(aElements, aTimes, aIdx, float('inf'))
    bSet, _ = get_set(bElements, bTimes, bIdx, float('inf'))
    errors += compare_sets(aSet, bSet)

    results = {k: int(v) for k, v in zip(ScoreErrors.__members__.keys(), errors)}

    def update_f1(d, key):
        prec = (
            d[key + "TP"] / (d[key + "TP"] + d[key + "FP"])
            if d[key + "TP"] + d[key + "FP"]
            else None
        )
        rec = (
            d[key + "TP"] / (d[key + "TP"] + d[key + "FN"])
            if d[key + "TP"] + d[key + "FN"]
            else None
        )
        if prec is None or rec is None:
            f1 = None
        else:
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else None
        d.update({key + "Prec": prec, key + "Rec": rec, key + "F1": f1})
        return prec, rec, f1

    update_f1(results, "Trill")
    update_f1(results, "Staccato")
    update_f1(results, "Grace")

    # Count notes (pitches), chords, and rests
    # n_Note should count individual pitches to be invariant to Chord/Note representation
    notes_list = gtScore.flatten().notesAndRests
    n_note_count = 0
    n_chord_count = 0
    n_rest_count = 0

    for n in notes_list:
        if n.isRest:
            n_rest_count += 1
        elif n.isChord:
            n_chord_count += 1
            n_note_count += len(n.pitches)
        else:  # Note
            n_note_count += 1

    results.update({"n_Note": n_note_count, "n_Chord": n_chord_count, "n_Rest": n_rest_count})

    if not (est_parts_valid and gt_parts_valid):
        results['StaffAssignment'] = None

    return results


def score_similarity_normalized(est, gt, full=False, strip_ties: bool = False):
    sim = score_similarity(est, gt, full=full, strip_ties=strip_ties)
    new_sim = {}
    for k, v in sim.items():
        match k:
            # Metric unavailable
            case None:
                new_sim[k] = None

            # Normalize by note count
            case (
                "Beams"
                | "NoteInsertion"
                | "NoteDeletion"
                | "NoteDuration"
                | "NoteSpelling"
                | "Tie"
                | "StemDirection"
            ):
                new_sim[k] = v / (sim["n_Note"] + 1e-8)

            # Normalize by rest count
            case "RestInsertion" | "RestDeletion" | "RestDuration":
                new_sim[k] = v / (sim["n_Rest"] + 1e-8)

            # Normalize by both note count and rest count
            case "StaffAssignment" | "Voice":
                new_sim[k] = v / ((sim["n_Note"] + sim["n_Rest"]) + 1e-8)

            # Do not normalize
            case (
                "Clef"
                | "GraceFN"
                | "GraceTN"
                | "GraceFP"
                | "GraceTP"
                | "GracePrec"
                | "GraceRec"
                | "GraceF1"
                | "KeySignature"
                | "StaccatoFN"
                | "StaccatoTN"
                | "StaccatoFP"
                | "StaccatoTP"
                | "StaccatoPrec"
                | "StaccatoRec"
                | "StaccatoF1"
                | "TimeSignature"
                | "TrillFN"
                | "TrillTN"
                | "TrillFP"
                | "TrillTP"
                | "TrillPrec"
                | "TrillRec"
                | "TrillF1"
                | "n_Note"
                | "n_Rest"
                | "n_Chord"
            ):
                new_sim[k] = v

            # Error
            case _:
                raise ValueError(f"Unknown metric key: {k}")

    return new_sim  # TODO: change that to account for rests
