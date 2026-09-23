"""
Count number of notes in each MIDI files of PDMX.
"""

import multiprocessing as mp
from pathlib import Path

import pandas as pd
import symusic
from tqdm import tqdm

tqdm.pandas()

PDMX_PATH = Path("~/Documents/datasets/pdmx").expanduser()


def count_notes(path: Path) -> int:
    path = PDMX_PATH / path
    return sum(len(track.notes) for track in symusic.Score(path).tracks)


pdmx_csv_path = PDMX_PATH / "PDMX_piano_tokenizable.csv"
pdmx = pd.read_csv(pdmx_csv_path, index_col=0)
pdmx = pdmx[pdmx["subset:all_valid"]]

# with mp.Pool(12) as pool:
#     pdmx["n_notes_midi"] = list(tqdm(pool.imap(count_notes, pdmx["mid"]), total=len(pdmx)))
pdmx["n_notes_midi"] = pdmx["mid"].progress_apply(count_notes)

pdmx.to_csv(pdmx_csv_path.with_stem(pdmx_csv_path.stem + "_midi-notes"))

# print(len(pdmx[pdmx["n_notes_midi"] <= 10]))
