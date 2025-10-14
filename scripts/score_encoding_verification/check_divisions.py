import json
import zipfile
from collections import defaultdict
from fractions import Fraction

import joblib
# import tabulate
from lxml import etree
from pathlib import Path

from tqdm import tqdm

NOTE_TYPES = ["whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "128th", "256th"]
NOTE_TYPE_TO_QUARTER_LENGTH = {
    ntype: Fraction(1, 2) ** i for i, ntype in enumerate(NOTE_TYPES, start=-2)
}


class ParsingException(Exception):
    pass


def compute_theoretical_note_duration_ql(note_type: str, dots: int = 0) -> Fraction:
    return NOTE_TYPE_TO_QUARTER_LENGTH[note_type] * (2 - Fraction(1, 2**dots))


def compute_theoretical_note_element_duration_ql(note: etree._Element) -> Fraction:
    """
    Compute the theoretical note duration in quarter lengths of a <note> element.
    """
    quarter_length = compute_theoretical_note_duration_ql(
        note_type=note.find("type").text,
        dots=len(note.findall("dot")),
    )

    # Take into account eventual tuplet
    if (time_modification := note.find("time-modification")) is not None:
        time_modification_ratio = compute_time_modification_ratio(time_modification)
        quarter_length *= time_modification_ratio

    return quarter_length


def split_composite_duration(dur: Fraction) -> list[tuple[str, Fraction]]:
    """
    Split a duration (in quarter length) into a list of atomic valid durations.
    """
    remaining_duration = dur
    types_and_durations = iter(
        sorted(NOTE_TYPE_TO_QUARTER_LENGTH.items(), key=lambda x: x[1], reverse=True)
    )
    current_type, current_duration = next(types_and_durations)
    atomic_parts = []
    while remaining_duration > 0:
        if current_duration <= remaining_duration:
            remaining_duration -= current_duration
            atomic_parts.append((current_type, current_duration))
        else:
            try:
                current_type, current_duration = next(types_and_durations)
            except StopIteration:
                raise ParsingException(f"Duration {dur} not decomposable.")
    return atomic_parts


def compute_time_modification_ratio(time_modification: etree._Element) -> Fraction:
    actual_notes = int(time_modification.find("actual-notes").text)
    normal_notes = int(time_modification.find("normal-notes").text)
    # normal_type = time_modification.find("normal-type").text
    # normal_dots = len(note.findall("normal-dot"))
    return Fraction(normal_notes, actual_notes)


def check_divisions(xml: etree._ElementTree, _debug: str = "") -> dict[str, list[ParsingException]]:
    """
    This only modify the <divisions> and <duration> tags of the notes, without any smart
    fixes. It will print a warning when the new duration is not an integer, and round it.
    """
    root = xml.getroot()

    exceptions = defaultdict(list)

    divisions = None
    for measure in root.iter('measure'):
        measure_number = measure.attrib["number"]

        # Look for divisions
        if (attributes := measure.find('attributes')) is not None and (  # Check if has attributes
            divisions_el := attributes.find("divisions")  # Check if has divisions
        ) is not None:
            divisions = int(divisions_el.text)
        if divisions is None:
            exceptions[measure_number].append(
                ParsingException("No divisions found in the first measure's attributes.")
            )
            break

        for element in measure:

            # If it's a note, check that its duration is coherent
            if element.tag == "note":

                # Skip grace notes
                if element.find("grace") is not None:
                    continue

                # Skip measure-long rests
                note_type_el = element.find("type")
                is_rest = element.find("rest") is not None
                if is_rest and note_type_el is None:
                    continue

                # # Ignore chords to see
                # if element.find("chord") is not None:
                #     continue

                # Get actual duration in divisions
                actual_duration = int(element.find("duration").text)
                actual_quarter_length = Fraction(actual_duration, divisions)

                # Get some debugging information
                voice = int(element.find("voice").text) if element.find("voice") is not None else 1
                dots = len(element.findall("dot"))
                dur = note_type_el.text + "." * dots
                if is_rest:
                    note_info = f"rest/V{voice}/{dur}"
                else:
                    pitch_el = element.find("pitch")
                    if pitch_el is not None:
                        step = pitch_el.find("step").text
                        alter_el = pitch_el.find("alter")
                        octave = int(pitch_el.find("octave").text)
                        alter_int = 0 if alter_el is None else int(alter_el.text)
                        alter = "#" * alter_int + "#" * -alter_int
                        note_info = f"note/V{voice}/{dur}/{step}{alter}{octave}"
                    else:
                        note_info = f"note/V{voice}/{dur}"

                # Compute the theoretical duration of the note
                try:
                    theoretical_quarter_length = compute_theoretical_note_element_duration_ql(
                        element
                    )
                except KeyError:
                    exceptions[measure_number].append(
                        ParsingException(
                            f"<{note_info}>: Non-existing note type '{note_type_el.text}'"
                        )
                    )
                    continue
                theoretical_duration = theoretical_quarter_length * divisions

                if theoretical_duration != actual_duration:
                    exceptions[measure_number].append(
                        ParsingException(
                            f"<{note_info}>: Actual and theoretical durations differ: "
                            f"{actual_duration} != {theoretical_duration} "
                            f"({actual_quarter_length} != {theoretical_quarter_length}) "
                            f"[{float(actual_quarter_length)} != {float(theoretical_quarter_length)}]"
                        )
                    )

            if element.tag == "backup" or element.tag == "forward":
                # Check that the backup duration can be expressed as a combinations of valid times
                duration_ql = Fraction(int(element.find("duration").text), divisions)
                try:
                    split_composite_duration(duration_ql)
                except ParsingException as e:
                    exceptions[measure_number].append(ParsingException(f"<{element.tag}>: {e}"))

    return dict(exceptions.items())


def check_path(path: Path) -> dict[str, list[ParsingException]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zip_ref:
            with zip_ref.open("META-INF/container.xml") as container:
                container_tree: etree._ElementTree = etree.parse(container)
                rootfile_path = rootfile_path = container_tree.find(".//{*}rootfile").get(
                    "full-path"
                )
            with zip_ref.open(rootfile_path) as f:
                tree = etree.parse(f)
    else:
        # print(path.as_posix())
        tree = etree.parse(path)
    exceptions = check_divisions(tree, _debug=path.as_posix())
    return exceptions


if __name__ == '__main__':
    path = Path("/home/gerel/Documents/papiers/asap_fixes/scores/bad_measure_rest/example.musicxml")
    check_path(path)

if __name__ == '__main__':
    # fmt: off
    # asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
    asap_path = Path("~/Documents/datasets/asap-dataset-original").expanduser()
    # asap_path = Path("~/Téléchargements/georges_collier/musicxml").expanduser()
    results_file = asap_path / f"{asap_path.stem}_exceptions.json"
    # fmt: on

    score_paths = sorted(asap_path.rglob("xml_score.musicxml"))
    # score_paths = sorted(asap_path.rglob("*.musicxml"))
    results = joblib.Parallel(n_jobs=1, return_as="generator")(
        joblib.delayed(check_path)(path) for path in score_paths
    )

    all_exceptions = {}
    for path, exceptions in tqdm(zip(score_paths, results), total=len(score_paths)):
        if exceptions:
            # print(f"Parsing errors in {path.relative_to(asap_path)}:")
            # for measure, measure_exceptions in exceptions.items():
            #     print(f" * Measure {measure}:")
            #     for exception in measure_exceptions:
            #         print(f"   - {exception}")
            all_exceptions[path.relative_to(asap_path).as_posix()] = exceptions

    all_exceptions = {
        path: {
            measure: [str(e.args[0]) for e in measure_exceptions]
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

    # Total number of measures with errors
    n_measure_errors = sum(len(measures) for measures in all_exceptions.values())
    print(f"Number of measures with errors: {n_measure_errors}")

    # Average number of erroneous measure per score
    avg_measure_error_per_score = n_measure_errors / n_score_errors
    print(f"Average measure errors per score: {avg_measure_error_per_score}")

    # Median number of erroneous measure per score
    median_measure_error_per_score = sorted(len(measures) for measures in all_exceptions.values())[
        n_score_errors // 2
    ]
    print(f"Median measure errors per score: {median_measure_error_per_score}")

    # For each path, proportion of measures affected
    proportions = []
    for path, path_exceptions in all_exceptions.items():
        with open(asap_path / path, "r") as f:
            num_m = f.read().count("<measure")
        proportions.append((path, len(path_exceptions), num_m, len(path_exceptions) / num_m))
    proportions = sorted(proportions, key=lambda p: p[-1], reverse=True)
    print(proportions)
    # print(tabulate.tabulate(proportions))
