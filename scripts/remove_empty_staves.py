"""
Remap/delete empty staves from a MusicXML score
"""

from pathlib import Path

from lxml import etree as ET
from lxml.etree import _Element


def voice_absolute_into_voice_relative_to_staff(
    voice_idx: int,
    one_based: bool = False,
) -> tuple[int, int]:
    """
    Converts an absolute voice_index into a voice index relative to a staff and its associated
    staff. Following MusicXML/MuseScore standards, the four first voices are mapped to the first
    staff, the four next voices to the second staff, and so on.

    :param voice_idx: Voice index (0-based or 1-based)
    :param one_based: Whether the voice and staff indices are 1-based (True) or 0-based (False)

    :return: Relative voice index and associated staff index (0 or 1-based depending on the
        parameter `one_based`)
    """
    if one_based:
        voice_idx -= 1
    staff_idx = voice_idx // 4
    voice_in_staff_idx = voice_idx % 4
    if one_based:
        staff_idx += 1
        voice_in_staff_idx += 1
    return voice_in_staff_idx, staff_idx


def voice_relative_to_staff_into_voice_absolute(
    voice_in_staff_idx: int,
    staff_idx: int,
    one_based: bool = False,
) -> int:
    """
    Converts a voice index relative into a staff to an absolute voice index. Following MusicXML/
    MuseScore standards, the four first voices are mapped to the first staff, the four next voices
    to the second staff, and so on.

    :param voice_in_staff_idx: Voice index in the staff (0-based or 1-based)
    :param staff_idx: Staff index (0 or 1-based depending on the parameter `one_based`)
    :param one_based: Whether the voice and staff indices are 1-based (True) or 0-based (False)

    :return: Voice index (0 or 1-based depending on the parameter `one_based`)
    """
    if one_based:
        staff_idx -= 1
        voice_in_staff_idx -= 1
    voice_idx = voice_in_staff_idx + staff_idx * 4
    return voice_idx + 1 if one_based else voice_idx


def remove_staves(input_file: Path, output_file: Path, staves_mapping: dict[int, int | None]):
    # Check that staves_mapping contains no duplicate values
    non_none_values = [v for v in staves_mapping.values() if v is not None]
    if len(non_none_values) != len(set(non_none_values)):
        raise ValueError("staves_mapping contains duplicate values")
    # Check that there are no holes in the values
    if len(non_none_values) != max(non_none_values):
        raise ValueError("staves_mapping values contains holes")
    # Check that staves_mapping is in increasing order, both for keys and values
    if sorted(staves_mapping.keys()) != list(range(1, max(staves_mapping.keys()) + 1)):
        raise ValueError("staves_mapping keys are not in increasing order")
    if sorted(non_none_values) != list(range(1, max(non_none_values) + 1)):
        raise ValueError("staves_mapping values are not in increasing order")

    num_staves = len(non_none_values)

    tree = ET.parse(input_file)
    root: _Element = tree.getroot()
    root.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

    if len(root.findall(".//part")) != 1:
        raise ValueError("Multi-part scores are not supported")

    # Change number of staves
    for staves in root.findall(".//staves"):
        staves.text = str(num_staves)

    # Remove <note> elements with sub-elements <rest/> on a staff to delete
    for note in root.findall(".//note"):
        note: _Element
        # print(note)
        is_rest = note.find("rest") is not None
        staff_el = note.find("staff")
        if staff_el is None:
            raise ValueError(f"Note {note} has no staff element")
        staff = int(staff_el.text)
        new_staff = staves_mapping[staff]

        # If new_staff is None, make sure that this is a rest and delete
        if new_staff is None:  # then delete the rest
            if not is_rest:
                raise ValueError(f"Note {note} is not a rest but has staff {staff}")

            # Check if there is a backup before this rest (or after for staff 1) to remove it too
            measure = note.getparent()
            if staff == 1:
                get_direction = lambda e: e.getnext()
            else:
                get_direction = lambda e: e.getprevious()
            sibling = get_direction(note)
            # Try to remove backups as long as we do not encounter notes or measure boundary
            while sibling is not None and sibling.tag != "note":
                if sibling.tag == "backup":
                    measure.remove(sibling)
                sibling = get_direction(sibling)

            # Remove the note
            measure.remove(note)

    # Modify <voice> elements to match new staves
    # We keep the MuseScore convention of 4 voices per staff
    for voice_el in root.findall(".//voice"):
        abs_voice = int(voice_el.text)
        rel_voice, default_staff_of_that_voice = voice_absolute_into_voice_relative_to_staff(
            abs_voice, one_based=True
        )
        new_default_staff_of_that_voice = staves_mapping[default_staff_of_that_voice]
        new_abs_voice = voice_relative_to_staff_into_voice_absolute(
            rel_voice, new_default_staff_of_that_voice, one_based=True
        )
        voice_el.text = str(new_abs_voice)

    # Modify <staff> elements (we do not do it in the first loop as it can be present on other
    # elements than notes, such as directions for example)
    for staff_el in root.findall(".//staff"):
        staff_el.text = str(staves_mapping[int(staff_el.text)])

    # Remove unneeded <clef> elements or update their number
    for clef in root.findall(".//clef"):
        number = int(clef.get("number"))
        if staves_mapping[number] is None:
            clef.getparent().remove(clef)
        else:
            clef.set("number", str(staves_mapping[number]))

    # Remove unneeded <staff-layout> elements or update their number
    for staff_layout_el in root.findall(".//staff-layout"):
        number = staff_layout_el.get("number")
        if number is not None:
            if staves_mapping[int(number)] is None:
                staff_layout_el.getparent().remove(staff_layout_el)
            else:
                staff_layout_el.set("number", str(staves_mapping[int(number)]))

    # Hack to conserve </midi-device> closing tag
    for midi_device in root.findall(".//midi-device"):
        midi_device.text = ''

    # Save modified XML
    tree.write(
        output_file,
        encoding='utf-8',
        xml_declaration=True,
    )

    # Manually copy the first three lines from input to output as issues with formatting
    n = 3
    with open(input_file, 'r', encoding='utf-8') as f:
        first_lines = [f.readline() for _ in range(n)]
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(first_lines)
        f.writelines(lines[n:])


if __name__ == '__main__':
    piece, mapping = "Ravel/Miroirs/3_Une_Barque", {1: 1, 2: 2, 3: None}
    # piece, mapping = "Ravel/Miroirs/4_Alborada_del_gracioso", {1: 1, 2: 2, 3: None}

    full_path = Path("/home/gerel/Documents/datasets/asap-dataset") / piece / "xml_score.musicxml"
    backup_path = full_path.with_name(full_path.name + ".bak")
    if not backup_path.exists():
        backup_path.write_text(full_path.read_text())
    remove_staves(backup_path, full_path, staves_mapping=mapping)
