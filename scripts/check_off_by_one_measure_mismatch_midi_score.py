"""
This script reads an extended PDMX metadata file, and computes some statistics about small errors in
the reported number of measures in the MIDI files compared to the score.
"""

import matplotlib.pyplot as plt

from pathlib import Path
import pandas as pd

# %%
pdmx_path = Path("~/Documents/datasets/pdmx")
pdmx_metadata_path = pdmx_path / "PDMX_extended_metadata.csv"

df = pd.read_csv(pdmx_metadata_path)
df = df[df["usable"] & (df["n_notes_midi"] > 10)]

# %%
df["gap"] = df["n_measures_midi"] - df["n_measures_score"]
total = len(df["gap"])

# %%
df["gap"][df["gap"] != 0].hist()
plt.show()


values = [0, 1, 2]
for value in values:
    matching = df['gap'][df['gap'].abs() == value]
    print(f"Number with gap={value}: {len(matching):>6} ({len(matching) / total * 100:.2f}%)")
matching = df['gap'][df['gap'].abs() > values[-1]]
print(f"Number with gap>{values[-1]}: {len(matching):>6} ({len(matching) / total * 100:.2f}%)")
print(f"Total num. scores: {total:>6}")
