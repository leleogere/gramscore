"""
Fix the <division> tag of a score, as well as all the related elements
(<duration>, <forward>, <backward>...)
"""

import math
from fractions import Fraction
from pathlib import Path
from lxml import etree as ET
import re


# note_type_to_divisions = {
#     "long": 1,
#     "breve": 1,
#     "whole": 1,
#     "half": 1,
#     "quarter": 1,
#     "eighth": 2,
#     "16th": 4,
#     "32nd": 8,
#     "64th": 16,
#     "128th": 32,
#     "256th": 64,
#     "512th": 128,
#     "1024th": 256,
# }

note_type_to_duration = {
    "long": Fraction(8),
    "breve": Fraction(4),
    "whole": Fraction(4),
    "half": Fraction(2),
    "quarter": Fraction(1),
    "eighth": Fraction(1 / 2),
    "16th": Fraction(1 / 4),
    "32nd": Fraction(1 / 8),
    "64th": Fraction(1 / 16),
    "128th": Fraction(1 / 32),
    "256th": Fraction(1 / 64),
    "512th": Fraction(1 / 128),
    "1024th": Fraction(1 / 256),
}


def get_best_divisions(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()

    divisors = {1}

    # Iterate over <note> elements
    for note in root.findall(".//note"):
        # Find note type get associated duration in quarter notes
        note_type = note.find("type")
        if note_type is not None:
            duration = note_type_to_duration[note_type.text]
            # Count dots
            dots = note.findall("dot")
            for _ in dots:
                duration += duration / 2
            # If has attribute <time-modification>, take it into account
            time_modification = note.find("time-modification")
            if time_modification is not None:
                actual_notes = int(time_modification.find("actual-notes").text)
                normal_notes = int(time_modification.find("normal-notes").text)
                duration *= Fraction(normal_notes, actual_notes)
            # Add to divisors
            divisors.add(duration.denominator)

    # Return LCM of divisors
    lcm = math.lcm(*divisors)
    print(f"LCM={lcm} {sorted(divisors)}")
    return lcm


def naive_modify_divisions_of_score(path: Path, new_divisions: int):
    """
    This only modify the <divisions> and <duration> tags of the notes, without any smart
    fixes. It will print a warning when the new duration is not an integer, and round it.
    """
    tree: ET = ET.parse(path)
    root = tree.getroot()

    # Modify divisions
    divisions = root.find(".//divisions")
    old_divisions = int(divisions.text)
    divisions.text = str(new_divisions)
    print(f"Replacing {old_divisions=} with {new_divisions=}")

    ratio = Fraction(new_divisions, old_divisions)
    print(f"Multiplying all durations by {ratio}")

    # Modify all related elements
    measure_nb = 0
    for element in root.iter():
        # Keep track of measure number for logging
        if element.tag == "measure":
            measure_nb = element.get("number")

        # Warn about <forward> tags with duration 1
        if element.tag == "forward" and int(element.find("duration").text) == 1:
            print(f"Found <forward> tag with duration 1 in measure {measure_nb}")

        # Fix all duration tags
        if element.tag == "duration":
            parent = element.getparent()
            init_duration = int(element.text)
            updated_duration = init_duration * ratio
            if updated_duration.denominator != 1:
                print(
                    f"Computed duration of {parent.tag} in measure {measure_nb} is not an integer: "
                    f"{updated_duration} ({init_duration})"
                )
            updated_duration = int(updated_duration)
            element.text = str(updated_duration)

    # Export to string and fix some issues with lxml exporter to stick to original file
    new_tree_string = ET.tostring(
        tree,
        encoding='utf-8',
        xml_declaration=True,
        pretty_print=True,
    ).decode()
    with path.open("r") as f:
        original_tree_string = f.read()

    # Fix the xml declaration and the midi-device tag
    new_tree_string = fix_final_xml(original_tree_string, new_tree_string)

    # Export to new file
    new_path = path.with_name(path.stem + "_divisions_fixed.musicxml")
    with new_path.open("w") as f:
        f.write(new_tree_string)

    return new_path


def modify_divisions_of_score(path: Path, new_divisions: int):
    tree: ET = ET.parse(path)
    root = tree.getroot()

    # Modify divisions
    divisions = root.find(".//divisions")
    old_divisions = int(divisions.text)
    divisions.text = str(new_divisions)
    print(f"Replacing {old_divisions=} with {new_divisions=}")

    ratio = Fraction(new_divisions, old_divisions)

    # Modify all related elements
    measure_nb = 0
    for element in root.iter():
        # Keep track of measure number for logging
        if element.tag == "measure":
            measure_nb = element.get("number")

        # Fix all duration tags
        if element.tag == "duration":
            parent = element.getparent()
            init_duration = int(element.text)
            updated_duration = init_duration * ratio

            # If we have a note with explicit type, we can compute the duration
            if parent.tag == "note" and (note_type := parent.find("type")) is not None:
                computed_duration = note_type_to_duration[note_type.text] * new_divisions

                # Dots needs to be taken into account
                dots = parent.findall("dot")
                for _ in dots:
                    computed_duration *= Fraction(3, 2)

                # Time modification needs to be taken into account
                time_modification = parent.find("time-modification")
                if time_modification is not None:
                    actual_notes = int(time_modification.find("actual-notes").text)
                    normal_notes = int(time_modification.find("normal-notes").text)
                    computed_duration *= Fraction(normal_notes, actual_notes)

                # This will warn that the initial duration was not correct
                # assert abs(computed_duration - updated_duration) < 0.05 * new_divisions, (
                #     f"Computed duration of {parent.tag} (V{parent.find('voice').text}) in measure "
                #     f"{measure_nb} different than the naive update: {updated_duration} "
                #     f"({Fraction(updated_duration, new_divisions)}) -> "
                #     f"{computed_duration} ({Fraction(computed_duration, new_divisions)})"
                # )
                if int(computed_duration) != updated_duration:
                    # Try with
                    print(
                        f"Computed duration of {parent.tag} in measure {measure_nb} different than the "
                        f"naive update: {updated_duration} -> {computed_duration}"
                    )
                    pass
                updated_duration = computed_duration

            # Remove spurious forward tags (of duration 1 or just before end of measure)
            if parent.tag == "forward" and (
                updated_duration <= 0.01 * new_divisions or parent.getnext() is None
            ):
                parent.getparent().remove(parent)
                print(
                    f"Removed spurious <forward> tag [{updated_duration}] in measure {measure_nb}"
                )
                continue

            # Try to fix broken backups/forwards (due to rounding errors)
            if parent.tag in ["backup", "forward"] and not updated_duration.is_integer():
                # Accumulate duration since start of measure
                # Does not work in case of partial backup?
                fixed_duration = 0
                measure = parent.getparent()
                for child in measure.iter():
                    if child == parent:
                        break
                    if child.tag == "duration" and child.find("chord") is None:
                        if child.getparent().tag == "backup":
                            fixed_duration -= int(child.text)
                        else:
                            fixed_duration += int(child.text)
                # Make sure small difference with the computed one
                error = abs(fixed_duration - updated_duration) / updated_duration
                assert (
                    error < 0.05
                ), f"{parent.tag} duration in measure {measure_nb} is not consistent: {fixed_duration=} {updated_duration=} (error ~{error:.3f})"
                print(
                    f"Fixed {parent.tag} duration in measure {measure_nb} from "
                    f"{float(updated_duration)} to {fixed_duration} (error ~{error:.3f})"
                )
                updated_duration = fixed_duration

            # Make sure the new duration is an integer
            assert updated_duration.is_integer(), (
                f"New duration of {parent.tag} in measure {measure_nb} is not an integer: "
                f"{updated_duration=} ({init_duration=})"
            )
            updated_duration = int(updated_duration)
            element.text = str(updated_duration)

    # Try to fix invisible rests at the end of a measure
    # We assume that invisible rests at the end of a measure are an error and should be removed
    # We check invisible rests both at the end of a measure or before a backup element (in case
    # of multiple voices)
    measure = None
    for element in list(root.iter()):
        # Keep track of measure for logging
        if element.tag == "measure":
            measure = element
            measure_nb = measure.get("number")
        elif element.tag == "backup":
            # While the previous element is an invisible rest, remove it
            prev = element.getprevious()
            while (
                prev is not None
                and prev.tag == "note"
                and prev.find("rest") is not None
                and prev.get("print-object") == "no"
                and int(prev.find("duration").text) <= 0.5 * new_divisions  # max length eighth rest
            ):
                print(f"Removing invisible rest in measure {measure_nb}")
                # In case of backup, we need to remove the deleted duration
                duration = element.find("duration")
                rest_duration = int(prev.find("duration").text)
                duration.text = str(int(duration.text) - rest_duration)
                print(f"Removed {rest_duration=} from backup in measure {measure_nb}")
                measure.remove(prev)
                prev = element.getprevious()
        elif element.tag == "note" and element.getnext() is None:
            prev = element
            while (
                prev is not None
                and prev.tag == "note"
                and prev.find("rest") is not None
                and prev.get("print-object") == "no"
            ):
                print(f"Removing invisible rest in measure {measure_nb}")
                measure.remove(prev)
                prev = element.getprevious()

    # Try to fix backup close to getting back to measure start but not quite
    pos_in_measure = 0
    measure = 0
    for element in root.iter():
        if element.tag == "measure":
            pos_in_measure = 0
            measure = element.get("number")
            print(f"M{measure}")
        elif element.tag == "duration":
            parent = element.getparent()
            if parent.tag == "forward" or (parent.tag == "note" and parent.find("chord") is None):
                pos_in_measure += int(element.text)
            elif parent.tag == "backup":
                pos_in_measure -= int(element.text)
                magnet_increment = new_divisions / 2  # eighth note
                closest_position = int(round(pos_in_measure / magnet_increment) * magnet_increment)
                if 0 < abs(pos_in_measure - closest_position) < 0.05 * new_divisions:
                    print(
                        f"Clamping backup in measure {measure} to from {pos_in_measure} to {closest_position}"
                    )
                    element.text = str(closest_position)
                    pos_in_measure = closest_position
                # if 0 < (pos_in_measure % new_divisions) / new_divisions < 0.05:
                #     clamped_pos_in_measure = round(pos_in_measure / new_divisions) * new_divisions
                #     print(f"Clamping backup in measure {measure} to 0 {pos_in_measure=}")
                #     element.text = str(int(element.text) + pos_in_measure)
                #     pos_in_measure = 0
            print(f"{parent.tag} | {element.text} | {pos_in_measure}")

    # Export to string and fix some issues with lxml exporter to stick to original file
    new_tree_string = ET.tostring(
        tree,
        encoding='utf-8',
        xml_declaration=True,
        pretty_print=True,
    ).decode()
    with path.open("r") as f:
        original_tree_string = f.read()

    # Fix the xml declaration and the midi-device tag
    new_tree_string = fix_final_xml(original_tree_string, new_tree_string)

    # Export to new file
    new_path = path.with_name(path.stem + "_divisions_fixed.musicxml")
    with new_path.open("w") as f:
        f.write(new_tree_string)
    return new_path


def fix_final_xml(original_xml: str, fixed_xml: str) -> str:
    # Replace the xml declaration with the original one
    fixed_xml = re.sub(r"<\?xml .*>", original_xml.split("\n")[0], fixed_xml, count=1)

    # Replace the <midi-device> tag with the original one
    # Often, the original is "<midi-device ...></midi-device>", while lxml exports it as a self
    # closing tag "<midi-device ... />"
    original_midi_device = re.search(r"<midi-device.*", original_xml)
    if original_midi_device is not None:
        fixed_xml = re.sub(
            r"<midi-device.*",
            original_midi_device.group(),
            fixed_xml,
            count=1,
        )

    return fixed_xml


if __name__ == '__main__':
    import partitura as pt
    from src.models.automata.tokens import fill_rests

    # path = Path(
    #     # "/home/gerel/Documents/datasets/asap-dataset/Beethoven/Piano_Sonatas/17-2/xml_score.musicxml"
    #     "/home/gerel/Documents/datasets/asap-dataset/Beethoven/Piano_Sonatas/24-1_no_repeat/xml_score.musicxml"
    # )
    paths = """
# /home/gerel/Documents/datasets/asap-dataset/Beethoven/Piano_Sonatas/24-1_no_repeat/xml_score.musicxml
# /home/gerel/Documents/datasets/asap-dataset/Beethoven/Piano_Sonatas/7-4/xml_score.musicxml
# /home/gerel/Documents/datasets/asap-dataset/Chopin/Barcarolle/xml_score.musicxml
# /home/gerel/Documents/datasets/asap-dataset/Chopin/Etudes_op_25/11/xml_score.musicxml
# /home/gerel/Documents/datasets/asap-dataset/Debussy/Pour_le_Piano/1/xml_score.musicxml
# /home/gerel/Documents/datasets/asap-dataset/Haydn/Keyboard_Sonatas/48-1/xml_score.musicxml
# /home/gerel/Documents/datasets/asap-dataset/Liszt/Gran_Etudes_de_Paganini/2_La_campanella/xml_score.musicxml
# /home/gerel/Documents/datasets/asap-dataset/Liszt/Mephisto_Waltz/xml_score.musicxml
# /home/gerel/Documents/datasets/asap-dataset/Liszt/Transcendental_Etudes/1/xml_score.musicxml
# /home/gerel/Documents/datasets/asap-dataset/Liszt/Transcendental_Etudes/11/xml_score.musicxml
# /home/gerel/Documents/datasets/asap-dataset/Ravel/Miroirs/3_Une_Barque/xml_score.musicxml
# /home/gerel/Documents/datasets/asap-dataset/Ravel/Pavane/xml_score.musicxml
# /home/gerel/Documents/datasets/asap-dataset/Schumann/Kreisleriana/1_no_first_repeat/xml_score.musicxml
/home/gerel/Documents/datasets/asap-dataset/Liszt/Mephisto_Waltz/xml_score1.musicxml
"""
    # paths = list(map(Path, paths.strip().split("\n")))
    paths = [Path(p) for p in paths.strip().split("\n") if not p.startswith("#")]

    for path in paths:
        print("=" * 80)
        print(path)
        try:
            score = pt.load_musicxml(path)
            assert len(score.parts) == 1
            score = score.parts[0]
            fill_rests(score)
            print("ORIGINAL: OK filling rests")
        except Exception as e:
            print("ORIGINAL: ERROR filling rests:", e)

        # input(f"Press Enter to process {path}")
        try:
            divs = get_best_divisions(path)
            # new_path = modify_divisions_of_score(path, divs)
            new_path = naive_modify_divisions_of_score(path, divs)
            try:
                score = pt.load_musicxml(new_path)
                assert len(score.parts) == 1
                score = score.parts[0]
                fill_rests(score)
                print("FIXED: OK filling rests")
            except Exception as e:
                print("FIXED: ERROR filling rests:", e)
        except Exception as e:
            print("=> ERROR MODIFYING DIVISIONS:", e)
            # raise
            continue
