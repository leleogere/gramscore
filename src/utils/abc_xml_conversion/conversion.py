"""
This file is used to try to get a cleaner interface for abc2xml and xml2abc conversion (as the
current one is quite bad, only usable from CLI).
This file does not comply to the project's code style and good practices, I did what I could to get
a somewhat usable interface.
"""

import subprocess
import os

from loguru import logger

# Compute the directory of the script
DIRECTORY = os.path.dirname(os.path.realpath(__file__))


def _convert(input: str, direction: str) -> str:
    """
    Convert MusicXML to ABC or the opposite.

    :param input: The XML/ABC string.
    :param direction: The direction of the conversion. Either "xml2abc" or "abc2xml".
    :return: The ABC/XML string.
    """
    assert direction in ["xml2abc", "abc2xml"], f"Invalid direction: {direction}"
    option_stdin = "-i" if direction == "xml2abc" else "-"
    p = subprocess.Popen(
        ["python", f"{direction}.py", option_stdin],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=DIRECTORY,
    )
    output, err = p.communicate(input)
    if err:
        logger.warning(err)
    return output


def xml2abc(xml: str) -> str:
    """
    Convert MusicXML to ABC.

    :param xml: The XML string.
    :return: The ABC string.
    """
    return _convert(xml, "xml2abc")


def abc2xml(abc: str) -> str:
    """
    Convert ABC to MusicXML.

    :param abc: The ABC string.
    :return: The XML string.
    """
    return _convert(abc, "abc2xml")


if __name__ == '__main__':
    with open(
        "/home/gerel/Documents/datasets/asap-dataset/Chopin/Etudes_op_10/12/xml_score.musicxml", "r"
    ) as f:
        xml = f.read()
    abc = xml2abc(xml)
    print(abc)
    print("#" * 80)
    xml = abc2xml(abc)
    print(xml)
