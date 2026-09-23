"""
Count pieces in PDMX both usable and tokenizable
"""

from pathlib import Path

import pandas as pd

pdmx_path = Path("~/Documents/datasets/pdmx").expanduser()
metadata_file_path = pdmx_path / "PDMX_extended_metadata.csv"
tokenizable_file_path = pdmx_path / "PDMX_piano_tokenizable.csv"

metadata = pd.read_csv(metadata_file_path)
tokenizable = pd.read_csv(tokenizable_file_path)

print(metadata.shape)
print(tokenizable.shape)

metadata = metadata.merge(tokenizable["mxl"], on="mxl")


pass
