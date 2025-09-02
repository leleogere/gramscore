"""
Count scores available for training in PDMX dataset
"""

from pathlib import Path

import pandas as pd


pdmx_path = Path("~/Documents/datasets/pdmx").expanduser()

valid_scores = pd.read_csv(
    pdmx_path / "PDMX_piano_>4*_<500n_tokenizable.csv",
    index_col=0,
)
with (pdmx_path / "overlapping_asap.csv").open("r") as f:
    excluded_idx = [int(s.strip()) for s in f.readlines()]

training_scores = valid_scores[~valid_scores.index.isin(excluded_idx)]

print(f"Number of valid scores: {len(valid_scores)}")
print(f"Number of training scores: {len(training_scores)}")
