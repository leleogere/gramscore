"""
This scripts gather all existing TS in PDMX dataset
"""

import matplotlib.pyplot as plt

import numpy as np

import tqdm

import pandas as pd

from pathlib import Path

from loguru import logger
from lxml import etree
from matplotlib import colors
from tqdm import tqdm

from utils.musicxml import read_musicxml

tqdm.pandas()

# %%

pdmx_path = Path("~/Documents/datasets/pdmx").expanduser()
metadata_path = pdmx_path / "PDMX_extended_metadata.csv"


metadata = pd.read_csv(metadata_path)

metadata = metadata[(metadata["n_notes_midi"] >= 3) & (metadata["usable"])]

# %%
scores = [pdmx_path / rpath for rpath in metadata["mxl"]]; dataset = "PDMX"  # fmt: skip
# scores = list((pdmx_path.parent / "asap-dataset").rglob("xml_score.musicxml")); dataset = "ASAP"  # fmt: skip


# %%
def get_time_signatures(musicxml_path: Path) -> set[tuple[int, int]]:
    # Read XML
    xml = read_musicxml(musicxml_path)
    root = etree.fromstring(xml.encode("utf-8"))
    # Finding all <time> element
    time_elements = root.findall(".//time")
    time_signatures = set()
    for time in time_elements:
        try:
            time_signatures.add((int(time.findtext("beats")), int(time.findtext("beat-type"))))
        except ValueError as e:
            logger.error(e)
            continue
    return time_signatures


# %%
import joblib

time_signatures_gen = joblib.Parallel(n_jobs=8, return_as="generator_unordered")(
    joblib.delayed(get_time_signatures)(s) for s in scores
)
time_signatures = [ts for ts in tqdm(time_signatures_gen, total=len(scores))]

# %%
import numpy as np
from itertools import chain

time_signatures_df = pd.DataFrame(
    list(chain.from_iterable(time_signatures)),
    columns=["numerator", "denominator"],
)
unique_time_signatures = (
    time_signatures_df.groupby(by=["numerator", "denominator"]).size().reset_index(name='count')
)
pivot = (
    unique_time_signatures.pivot(index='denominator', columns='numerator', values='count')
    .fillna(np.nan)
    .reindex(columns=range(time_signatures_df["numerator"].max() + 1), fill_value=None)
)
print(pivot)

import matplotlib.pyplot as plt
import numpy as np

matrix = pivot.values

plt.figure(figsize=(25, 5))
plt.imshow(matrix, aspect="auto", cmap='viridis', zorder=1)
plt.colorbar(label='Count')

# Label axes
plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
plt.xticks(ticks=unique_time_signatures["numerator"].unique(), rotation=90)
plt.xlabel('Numerator')
plt.ylabel('Denominator')
plt.title(f'Number of each time signatures in {dataset} dataset')
# plt.grid(color="#eaeaea")
plt.tight_layout()
for _, row in unique_time_signatures.iterrows():
    n = row["numerator"]
    d = row["denominator"]
    c = row["count"]
    plt.text(
        n, pivot.index.get_loc(d), c, ha='center', va='center', color='white', rotation=90, size=8
    )
plt.show()

# %% Print a cumulative number of scores affected by choosing n&d up to a certain value
from tqdm import trange

# It is clearly possible to make this efficient rather than this long double loop,
# it's just so that it was easier to do it like that
cum_matrix = np.zeros_like(matrix).T
for numerator in trange(unique_time_signatures["numerator"].max()):
    for denominator in sorted(unique_time_signatures["denominator"].unique()):
        cum_matrix[numerator, pivot.index.get_loc(denominator)] = len(
            [
                ts_in_that_score
                for ts_in_that_score in time_signatures
                if all(d <= numerator and n <= denominator for (n, d) in ts_in_that_score)
            ]
        )

# %%
total = cum_matrix.max()

plt.figure(figsize=(25, 5))
img = plt.imshow(cum_matrix.T, aspect='auto', cmap='viridis', zorder=1)
# pcm = plt.pcolor(total - cumsum_2d, norm=colors.LogNorm())
# plt.colorbar(pcm)
plt.colorbar()
plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
plt.xticks(ticks=unique_time_signatures["numerator"].unique(), rotation=90)
plt.xlabel('Numerator')
plt.ylabel('Denominator')
plt.title(f'Cumulative number of each signatures in {dataset} dataset')
plt.tight_layout()
for _, row in unique_time_signatures.iterrows():
    n = row["numerator"]
    d = pivot.index.get_loc(row["denominator"])
    c = f"{float(cum_matrix[n, d] / total * 100):.3f}%"
    # c = int(cum_matrix[n, d])
    plt.text(n, d, c, ha='center', va='center', color='black', rotation=90, size=8)
plt.show()
