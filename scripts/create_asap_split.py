"""
This script attempts at adding split information into ASAP metadata, following the method used by
MIDI2ScoreTransformer (based on ACPAS split dataset)
We do not use the same version of ASAP so some files have been moved around, therefore the initial
script is not working as intended.
"""

from pathlib import Path

import pandas as pd

# increase the number of columns displayed in pandas
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


# fmt: off
METADATA_R = Path("~/Documents/transcription/MIDI2ScoreTransformer/midi2scoretransformer/data/ACPAS-dataset/metadata_R.csv").expanduser()
METADATA_S = Path("~/Documents/transcription/MIDI2ScoreTransformer/midi2scoretransformer/data/ACPAS-dataset/metadata_S.csv").expanduser()
METADATA_ASAP = Path("~/Documents/datasets/asap-dataset/metadata.csv").expanduser()
# fmt: on

# Read ACPAS dataset
metadata_r = pd.read_csv(METADATA_R)
metadata_s = pd.read_csv(METADATA_S)
metadata_acpas = pd.concat([metadata_r, metadata_s], ignore_index=True)

# Keep only ASAP entries
metadata_acpas = metadata_acpas[metadata_acpas["source"] == "ASAP"]

# Remove "ASAP_" in title
metadata_acpas["title"] = metadata_acpas["title"].str.replace("ASAP_", "")

# Drop duplicated "performance_MIDI_external" (some performances were resynthesized in ACPAS)
metadata_acpas = metadata_acpas.drop_duplicates(subset=["performance_MIDI_external"])

metadata_asap = pd.read_csv(METADATA_ASAP)

print(metadata_acpas)

print(metadata_asap)
