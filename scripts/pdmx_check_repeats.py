"""
Add a column to PDMX to check if a score has repeated sections (~1m30).
"""

import zipfile
from pathlib import Path

import pandas as pd
from lxml import etree
from tqdm import tqdm
import multiprocessing as mp

tqdm.pandas()

pdmx_path = Path("~/Documents/datasets/pdmx").expanduser()
metadata_file = pdmx_path / "PDMX_piano_tokenizable.csv"


def has_repeats(path: Path):
    path = pdmx_path / path
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zip_ref:
            with zip_ref.open("META-INF/container.xml") as container:
                container_tree: etree._ElementTree = etree.parse(container)
                rootfile_path = rootfile_path = container_tree.find(".//{*}rootfile").get(
                    "full-path"
                )
            with zip_ref.open(rootfile_path) as f:
                tree = etree.parse(f)
    else:
        tree = etree.parse(path)

    root = tree.getroot()

    # Elements to check directly
    direct_elements = [
        "repeat",
        "measure-repeat",
        "beat-repeat",
        "slash",
        "ending",
        "coda",
        "segno",
    ]

    # Check for direct elements
    for el in direct_elements:
        if root.xpath(f".//{el}"):
            return True

    # Check <sound> attributes
    sound_flags = ["dacapo", "dalsegno", "fine", "segno", "tocoda", "forward-repeat"]
    sounds = root.xpath(".//sound")
    for s in sounds:
        for flag in sound_flags:
            if s.get(flag) == "yes":
                return True

    return False


pdmx_piano = pd.read_csv(metadata_file, index_col=0)

col_name = "has_repeats"
if col_name in pdmx_piano.columns:
    print(f"Column {col_name} already present in dataframe, not replacing it.")
else:
    with mp.Pool(processes=10) as pool:
        pdmx_piano["has_repeats"] = list(
            tqdm(pool.imap(has_repeats, pdmx_piano["mxl"]), total=len(pdmx_piano))
        )
    pdmx_piano.to_csv(metadata_file, index=True)
