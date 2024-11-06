import io
import tempfile
import time
from pathlib import Path

import verovio

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from cairosvg import svg2png
from PIL import Image
import wand.image
import music21 as m21


def render_score(
    path: Path | None = None,
    as_text: str | None = None,
    output_path: Path | None = None,
) -> Image.Image | None:
    """
    Render a score to an image.

    :param path: Path to the score file. If None, use `as_text`.
    :param as_text: Score as a string. If None, use `path`.
    :param output_path: Path to the output image file. If None, return a PIL Image.
    """
    if not (path is None) ^ (as_text is None):
        raise ValueError("Exactly one of `path` and `as_text` must be provided.")

    tk = verovio.toolkit()
    if as_text is None:
        tk.loadFile(str(path))
    else:
        tk.loadData(as_text.encode())

    svg = tk.renderToSVG().replace(
        "overflow=\"inherit\"",
        "overflow=\"visible\"",
    )  # https://github.com/Kozea/CairoSVG/issues/300
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
    output = render_score(
        Path(
            "/home/gerel/Documents/transcription/MIDI2ScoreTransformer/midi2scoretransformer/data/asap-dataset/Debussy/Images_Book_1/1_Reflets_dans_lEau/xml_score.musicxml"
        ),
        output_path=Path("/tmp/test.png"),
        # output_path=None,
    )
    if output is not None:
        output.show()
        time.sleep(1)
