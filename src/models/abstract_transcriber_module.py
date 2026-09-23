from pathlib import Path

import lightning as L
import verovio


class AbstractTranscriberModule(L.LightningModule):
    @staticmethod
    def render_score(path: Path):
        """
        Render a score to an image.
        """
        tk = verovio.toolkit()
        tk.loadFile(str(path))
        tk.renderToSVG()
