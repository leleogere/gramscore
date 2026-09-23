"""
This file is used to try to get a cleaner interface for abc2xml and xml2abc conversion (as the
current one is quite bad, only usable from CLI).
This file does not comply to the project's code style and good practices, I did what I could to get
a somewhat usable interface.
"""

import os
import subprocess
from pathlib import Path

from lightning_utilities import StrEnum
from loguru import logger


class ConversionDirection(StrEnum):
    XML2ABC = "xml2abc"
    ABC2XML = "abc2xml"


# Compute the directory of the script
DIRECTORY = os.path.dirname(os.path.realpath(__file__))


def _convert_file(path: Path, direction: ConversionDirection) -> str:
    """
    Convert a MusicXML file to an ABC string or the other way around.

    :param path: Path to the XML/ABC file.
    :param direction: The direction of the conversion. Either "xml2abc" or "abc2xml".
    :return: The ABC/XML string.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    assert direction in ConversionDirection, f"Invalid direction: {direction}"
    p = subprocess.run(
        ["python", f"{direction.value}.py", str(path.absolute())],
        text=True,
        capture_output=True,
        cwd=DIRECTORY,
    )
    if p.stderr:
        logger.trace(p.stderr)
    return p.stdout


def _convert(input_: str, direction: ConversionDirection) -> str:
    """
    Convert a MusicXML string to an ABC string or the other way around.

    :param input_: The XML/ABC string.
    :param direction: The direction of the conversion. Either "xml2abc" or "abc2xml".
    :return: The ABC/XML string.
    """
    assert direction in ConversionDirection, f"Invalid direction: {direction}"
    option_stdin = "-i" if direction == "xml2abc" else "-"

    # Does not work with the dataloader, the program receive the stdin empty, why ???
    # Workaround, let the program directly read the file
    p = subprocess.run(
        ["python", f"{direction.value}.py", option_stdin],
        input=input_,
        text=True,
        capture_output=True,
        cwd=DIRECTORY,
    )
    if p.stderr:
        logger.trace(p.stderr)
    return p.stdout


def xml2abc(xml: str) -> str:
    """
    Convert MusicXML to ABC.

    :param xml: The XML string.
    :return: The ABC string.
    """
    return _convert(xml, ConversionDirection.XML2ABC)


def abc2xml(abc: str) -> str:
    """
    Convert ABC to MusicXML.

    :param abc: The ABC string.
    :return: The XML string.
    """
    return _convert(abc, ConversionDirection.ABC2XML)


def xmlfile2abc(path: Path) -> str:
    return _convert_file(path, ConversionDirection.XML2ABC)


def abcfile2xml(path: Path) -> str:
    return _convert_file(path, ConversionDirection.ABC2XML)


if __name__ == '__main__':
    abc = xmlfile2abc(Path("data/nasap/Beethoven/Piano_Sonatas/7-1/xml_score.musicxml"))
    print(abc)
    with open("/tmp/test.abc", "w") as f:
        f.write(abc)
    # print("#" * 80)
    xml = abc2xml(abc)
    # print(xml)
