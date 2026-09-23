from music21 import chord, expressions, key, note, stream


def realize_spanners(s):
    to_remove = []
    for sp in s.recurse().getElementsByClass(expressions.TremoloSpanner):
        l = sp.getSpannedElements()
        if len(l) != 2:
            print("Not sure what to do with this spanner", sp, l)
            continue
        start, end = l

        offset = start.offset
        start_chord = None
        end_chord = None
        startActiveSite = start.activeSite
        endActiveSite = end.activeSite
        if start.activeSite is None:
            start_chord: chord.Chord = start._chordAttached
            offset = start_chord.offset
            startActiveSite = start._chordAttached.activeSite
        if end.activeSite is None:
            end_chord: chord.Chord = end._chordAttached
            endActiveSite = end._chordAttached.activeSite

        # We insert a tremolo expression on the start note
        # realize it, and then change every second note to have the pitch of the end note
        trem = expressions.Tremolo()
        trem.measured = sp.measured
        trem.numberOfMarks = sp.numberOfMarks
        start.expressions.append(trem)
        out = trem.realize(start, inPlace=True)[0]
        if start_chord:
            if len(start_chord.notes) == 1:
                startActiveSite.remove(start_chord)
            else:
                start_chord.remove(start)
        else:
            startActiveSite.remove(start)
        if end_chord:
            if len(end_chord.notes) == 1:
                endActiveSite.remove(end_chord)
            else:
                end_chord.remove(end)
        else:
            endActiveSite.remove(end)
        for i, n2 in enumerate(out):
            if i % 2 == 1:
                n2.pitch = end.pitch
            startActiveSite.insert(offset, n2)
            offset += n2.duration.quarterLength
        to_remove.append(sp)
    for sp in s.recurse().getElementsByClass(expressions.TrillExtension):
        l = sp.getSpannedElements()
        start = l[0]
        exp = [l.expressions for l in l]
        if not any(isinstance(e, expressions.Trill) for ex in exp for e in ex):
            if len(l) != 1:
                print("Not sure what to do with this spanner", sp, l)
                continue
            start.expressions.append(expressions.Trill())
            to_remove.append(sp)
    s.remove(to_remove, recurse=True)
    return s
