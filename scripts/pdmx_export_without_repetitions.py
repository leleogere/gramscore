"""
NOT IDEAL AS RUNNING ONE MuseScore INSTANCE FOR EACH SCORE IS QUITE INEFFICIENT...
This script reexport the MIDI files of the PDMX scores using MuseScore, allowing to better control
the output.
Note that to export WITHOUT repetitions, one must first export a MIDI file in the UI, unticking the
"Unroll repetitions" in the MIDI-export dialog, so that the preference is saved. There is currently
no way to specify this option via CLI (unfortunately).
"""

from pathlib import Path
import subprocess
import joblib
from tqdm import tqdm

MUSESCORE_PATH = Path(
    "~/Applications/MuseScore-Studio-4.5.2.251141401-x86_64.AppImage"
).expanduser()

PDMX_PATH = Path("~/Documents/datasets/pdmx").expanduser()

MXL_PATH = PDMX_PATH / "mxl"

NEW_MID_PATH = PDMX_PATH / "mid_no_rep"


NEW_MID_PATH.mkdir(parents=True, exist_ok=True)

mxl_paths = sorted(MXL_PATH.rglob("*.mxl"))


def export_single_score(input_path: Path, output_path: Path) -> None:
    assert input_path.suffix == ".mxl"
    assert output_path.suffix == ".mid"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            MUSESCORE_PATH,
            input_path.as_posix(),
            "-o",
            output_path.as_posix(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


generator = joblib.Parallel(n_jobs=10, return_as="generator_unordered")(
    joblib.delayed(export_single_score)(
        input_path=input_path,
        output_path=NEW_MID_PATH / input_path.relative_to(MXL_PATH).with_suffix(".mid"),
    )
    for input_path in mxl_paths
)
for _ in tqdm(generator, total=len(mxl_paths)):
    pass
