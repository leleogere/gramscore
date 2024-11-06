import io
import time
from pathlib import Path

import verovio
from PIL import Image
from cairosvg import svg2png


def render_score(
    path: Path | None = None,
    as_text: str | None = None,
    output_path: Path | None = None,
    input_type: str | None = None,
) -> Image.Image | None:
    """
    Render a score to an image.

    :param path: Path to the score file. If None, use `as_text`.
    :param as_text: Score as a string. If None, use `path`.
    :param output_path: Path to the output image file. If None, return a PIL Image.
    :param input_type: The type of the input file. If None, let verovio guess. Note that verovio
        ABC input does not support multi-voice.
    """
    if not (path is None) ^ (as_text is None):
        raise ValueError("Exactly one of `path` and `as_text` must be provided.")

    tk = verovio.toolkit()
    if input_type is not None:
        tk.setInputFrom(input_type)
    try:
        if as_text is None:
            tk.loadFile(str(path))
        else:
            tk.loadData(as_text)
    except Exception as e:
        print(f"Error rendering score: {e}")
        raise e

    try:
        svg = tk.renderToSVG().replace(
            "overflow=\"inherit\"",
            "overflow=\"visible\"",
        )  # https://github.com/Kozea/CairoSVG/issues/300
    except Exception as e:
        print(f"Error rendering score SVG: {e}")
        raise e
    png = svg2png(
        bytestring=svg.encode(),
        write_to=None,
        background_color="white",
    )
    if output_path is None:
        return Image.open(io.BytesIO(png))
    else:
        with open(output_path, "wb") as f:
            f.write(png)
            return None


if __name__ == '__main__':
    xml_path = Path(
        "/home/gerel/Documents/transcription/MIDI2ScoreTransformer/midi2scoretransformer/data/asap-dataset/Debussy/Images_Book_1/1_Reflets_dans_lEau/xml_score.musicxml",
    )
    abc_path = Path("/tmp/test.abc")
    # abc_path = Path("/tmp/test_corrupted.abc")

    # # Read from MusicXML path
    # output = render_score(xml_path)  # , output_path=Path("/tmp/test.png"))
    # if output is not None:
    #     output.show()
    #     time.sleep(1)
    #
    # # Read from MusicXML string
    # with open(xml_path, "r") as f:
    #     output = render_score(as_text=f.read())  # , output_path=Path("/tmp/test.png"))
    #     if output is not None:
    #         output.show()
    #         time.sleep(1)

    # Read from ABC path
    output = render_score(abc_path, input_type="abc")  # , output_path=Path("/tmp/test.png"))
    if output is not None:
        output.show()
        time.sleep(1)

    # # Read from ABC string
    # with open(abc_path, "r") as f:
    #     output = render_score(as_text=f.read())  # , output_path=Path("/tmp/test.png"))
    #     if output is not None:
    #         output.show()
    #         time.sleep(1)

    # Read from XML from ABC
    from utils.abc_xml_conversion import conversion

    xml = conversion.abcfile2xml(abc_path)
    output = render_score(as_text=xml)  # , output_path=Path("/tmp/test.png"))
    if output is not None:
        output.show()
        time.sleep(1)
