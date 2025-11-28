import zipfile
from pathlib import Path

ARTICULATIONS = [
    "accent",
    "strong_accent",
    "staccato",
    "staccatissimo",
    "tenuto",
    "detached_legato",
]

ORNAMENTS = [
    "trill_mark",
    "turn",
    "mordent",
    "inverted_mordent",
    "tremolo",
]

ARTICULATIONS_DASH = [a.replace("_", "-") for a in ARTICULATIONS]
ORNAMENTS_DASH = [o.replace("_", "-") for o in ORNAMENTS]

ARTICULATIONS_AND_ORNAMENTS = ARTICULATIONS + ORNAMENTS
ARTICULATIONS_AND_ORNAMENTS_DASH = ARTICULATIONS_DASH + ORNAMENTS_DASH


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


def read_musicxml(path: Path) -> str:
    """
    Reads a (maybe zipped) MusicXML file and returns its content as a string.

    :param path: Path to the MusicXML file.
    :return: Content of the MusicXML file as a string.
    """
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zipped_xml:
            contained_xml_name = zipped_xml.namelist()[-1]
            xml = zipped_xml.open(contained_xml_name)
            xml = xml.read().decode("utf-8")
    else:
        with open(path, "r") as f:
            xml = f.read()
    return xml
