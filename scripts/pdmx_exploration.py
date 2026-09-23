"""
Exploring the PDMX dataset.

The columns of the PDMX.csv file are:
    path: Path to the data (MusicRender JSON) file.
    metadata: Path to the associated metadata (JSON) file. The basename of each file matches the basename of the corresponding file in the path column.
    mxl: Path to the associated compressed MusicXML (MXL) file. The basename of each file matches the basename of the corresponding file in the path column. Values may be N/A, since some of the original MuseScore files are corrupted and thus cannot be converted to compressed MusicXML.
    pdf: Path to the associated sheet music (PDF) file. The basename of each file matches the basename of the corresponding file in the path column. Values may be N/A, since some of the original MuseScore files are corrupted and thus cannot be converted to PDF.
    mid: Path to the associated MIDI (MID) file. The basename of each file matches the basename of the corresponding file in the path column. Values may be N/A, since some of the original MuseScore files are corrupted and thus cannot be converted to MID.
    version: Version of the original MuseScore file.
    is_user_pro: Whether the user who posted the original MuseScore file is a "pro" user (pays for a MuseScore subscription).
    is_user_publisher: Whether the user who posted the original MuseScore file is a music publisher.
    is_user_staff: Whether the user who posted the original MuseScore file is part of MuseScore staff.
    has_paywall: Whether the original MuseScore file had a paywall.
    is_rated: Whether the original MuseScore file had any ratings.
    is_official: Whether the original MuseScore file was an "official" score, a title decided by MuseScore.
    is_original: Whether the original MuseScore file was an original work.
    is_draft: Whether the original MuseScore file was marked as a draft by the user who posted it.
    has_custom_audio: Whether the original MuseScore file has an associated custom audio file (must retrieve the actual audio from the metadata).
    has_custom_video: Whether the original MuseScore file has an associated custom video file (must retrieve the actual video from the metadata).
    n_comments: Number of comments on the original MuseScore file.
    n_favorites: Number of users who favorited the original MuseScore file.
    n_views: Number of views on the original MuseScore file.
    n_ratings: Number of ratings on the original MuseScore file.
    rating: Average rating (out of five stars) of the original MuseScore file. A rating of zero indicates that a song is unrated.
    license: Creative Commons license of the original MuseScore file.
    license_url: Link to the Creative Commons license of the original MuseScore file. Directly related to the license column.
    license_conflict: Whether the song's public-facing copyright metadata license disagrees with the internal copyright license data of the original MuseScore file (i.e. publicy, the song holds a public domain license, but internally, the copyright license data of the original MuseScore file says elsewise).
    genres: Genre(s) associated with the original MuseScore file, separated with a "-" if there are multiple.
    groups: MuseScore group(s) associated with the original MuseScore file, separated with a "-" if there are multiple.
    tags: MuseScore tag(s) associated with the original MuseScore file, separated with a "-" if there are multiple.
    song_name: If available, the name of the song.
    title: If available, the title of the song, oftentimes the same as song_name.
    subtitle: If available, the subtitle of the song.
    artist_name: If available, the name of the artist who created the song.
    composer_name: If available, the name of the composer who created the song, oftentimes the same as artist_name.
    publisher: If available, the publisher of the song.
    complexity: The MuseScore complexity score of the original MuseScore file. Ranges from 0-3.
    n_tracks: The number of tracks (parts) in the original MuseScore file.
    tracks: Track{s} from the original MuseScore file, separated with a "-" if there are multiple.
    song_length: Length of the song, in metrical MusPy time steps. There are usually 12 time steps per beat, though the actual resolution of each song is determined as the least common multiple of all divisions. To be practical, this statistic is best calculated by loading the file in path as a MusicRender object, and obtaining the song length there.
    song_length.seconds: Length of the song, in seconds.
    song_length.bars: Length of the song, in bars.
    song_length.beats: Length of the song, in beats.
    n_notes: Number of notes in the song.
    notes_per_bar: Average number of notes per bar in the song.
    n_annotations: Number of annotations in the song.
    has_annotations: Whether the song has any annotations.
    n_lyrics: Number of lyrics in the song.
    has_lyrics: Whether the song has any lyrics.
    n_tokens: Number of tokens (n_notes + n_annotations + n_lyrics) in the song.
    pitch_class_entropy: Pitch Class Entropy of the song, as calculated by the MusPy Package.
    scale_consistency: Scale Consistency of the song, as calculated by the MusPy Package. Ranges from 0-1.
    groove_consistency: Groove Consistency of the song, as calculated by the MusPy Package. Ranges from 0-1.
    best_path: Best filepath in the song's title duplicate grouping (see paper for full description).
    is_best_path: Whether the song is the best_path in the title duplicate grouping.
    best_arrangement: Best filepath in the song's title-instrumentation duplicate grouping (see paper for full description).
    is_best_arrangement: Whether the song is the best_arrangement in the title-instrumentation duplicate grouping.
    best_unique_arrangement: Best filepath in the song's title-instrumentation-arrangement duplicate grouping (see paper for full description).
    is_best_unique_arrangement: Whether the song is the best_unique_arrangement in the title-instrumentation-arrangement duplicate grouping. All songs for which this value is true are part of the Deduplicated subset.
    subset:all: Whether the song is part of the All subset (all True).
    subset:deduplicated: Whether the song is part of the Deduplicated subset (the same as is_best_unique_arrangement).
    subset:rated: Whether the song is part of the Rated subset (the song has a non-zero rating).
    subset:rated_deduplicated: Whether the song is both part of the Rated and Deduplicated subsets.
    subset:no_license_conflict: Whether the song's public-facing copyright metadata license agrees with the internal copyright license data of the original MuseScore file; that is, publicy and internally, the song holds a public domain license (the negation of the license_conflict column).
    subset:all_valid: Whether the song's associated compressed MusicXML (MXL), sheet music (PDF), and MIDI (MID) files are all valid (non-N/A).
"""

import json
import subprocess
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm import tqdm

tqdm.pandas()
pd.options.plotting.backend = "plotly"
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


# %% Setup paths
pdmx_path = Path("~/Documents/datasets/pdmx").expanduser()
pdmx_metadata_path = pdmx_path / "PDMX.csv"
pdmx_subsets_path = pdmx_path / "subset_paths"
pdmx_mxl_path = pdmx_path / "mxl"


# %% Read dataframe
pdmx = pd.read_csv(pdmx_metadata_path)  # .head(10000)
print(f"Number of scores: {len(pdmx)}")
print(f"Number of valid scores: {len(pdmx[pdmx['subset:all_valid']])}")
print(f"Number of deduplicated scores: {len(pdmx[pdmx['subset:deduplicated']])}")
print(f"Number of rated scores: {len(pdmx[pdmx['subset:rated']])}")
print(f"Number of rated deduplicated scores: {len(pdmx[pdmx['subset:rated_deduplicated']])}")
# Number of scores: 254077
# Number of valid scores: 254035
# Number of deduplicated scores: 102635
# Number of rated scores: 14182
# Number of rated deduplicated scores: 13187

# pdmx_sample_metadata = pd.read_json(pdmx_path / "metadata/5/5740212.json")


# %% Extract instruments from metadata
def extract_instrument(relative_file_path: str, root_path: Path):
    file_path = root_path / relative_file_path
    try:
        with file_path.open('r') as file:
            data = json.load(file)
            # Extract the instrument information
            instruments = [i["name"] for i in data["data"]["score"]["instruments"]]
            return instruments
    except KeyError as e:
        print(f"Error reading {file_path}: {e}")
        return None


# Prevent recomputing if already computed
if "instruments" in pdmx:
    print("Instruments already extracted, skipping.")
else:
    pdmx["instruments"] = pdmx["metadata"].progress_apply(
        partial(extract_instrument, root_path=pdmx_path)
    )


# %% Keep only single piano samples
pdmx_piano = pdmx[pdmx["instruments"].apply(lambda i: i == ["Piano"])]
pdmx_piano.to_csv(pdmx_path / "PDMX_piano.csv", index=False)
print(f"Number of piano scores: {len(pdmx_piano)}")
print(f"Number of piano scores with R>=4: {len(pdmx_piano[pdmx['rating'] >= 4.0])}")
print(f"Number of piano scores with <=500 notes: {len(pdmx_piano[pdmx['n_notes'] <= 500])}")
print(
    f"Number of piano scores with <=500 notes and R>=4: {len(pdmx_piano[(pdmx['n_notes'] <= 500) & (pdmx['rating'] >= 4.0)])}"
)
print(
    f"Number of deduplicated and rated piano scores: {len(pdmx_piano[pdmx_piano['subset:rated_deduplicated']])}"
)
print(
    f"Number of deduplicated and rated piano scores with <=500 notes: {len(pdmx_piano[pdmx_piano['subset:rated_deduplicated'] & (pdmx['n_notes'] <= 500)])}"
)
print(
    f"Number of deduplicated and rated piano scores with R>=4: {len(pdmx_piano[pdmx_piano['subset:rated_deduplicated'] & (pdmx_piano['rating'] >= 4.0)])}"
)
print(
    f"Number of deduplicated and rated piano scores with <=500 notes and R>=4: {len(pdmx_piano[pdmx_piano['subset:rated_deduplicated'] & (pdmx['n_notes'] <= 500) & (pdmx_piano['rating'] >= 4.0)])}"
)
# Number of piano scores: 178923
# Number of deduplicated and rated piano scores: 4385

# %% Display a few piano scores
pdmx_piano_clean = pdmx_piano[pdmx_piano['subset:rated_deduplicated']]
pdmx_piano_clean_good_rating = pdmx_piano_clean[pdmx_piano_clean["rating"] >= 4.0]
processes = []


def open_in_musescore(path: Path):
    MUSESCORE_PATH = Path(
        "~/Applications/MuseScore-Studio-4.5.2.251141401-x86_64.AppImage"
    ).expanduser()
    return subprocess.Popen(
        [MUSESCORE_PATH.as_posix(), path.as_posix()],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


for i in range(0, 0):
    piece = pdmx_piano_clean.iloc[i]
    mxl = pdmx_path / piece["mxl"]
    mid = pdmx_path / piece["mid"]
    infos = [
        f"{piece['title']} / {piece['song_name']}",
        f"{piece['artist_name']} / {piece['composer_name']}",
        mxl.as_posix(),
        mid.as_posix(),
    ]
    print("\n".join(infos + [""]))
    processes.append(open_in_musescore(mxl))
[p.wait() for p in processes]


# %% Display plots showing histograms of number of measures and number of notes
import matplotlib.pyplot as pl
import plotly.express as px
import math
import time

print(pdmx_piano_clean.iloc[pdmx_piano_clean["song_length.bars"].argmax()]["mxl"])
# open_in_musescore(
#     pdmx_path / pdmx_piano_clean.iloc[pdmx_piano_clean["song_length.bars"].argmax()]["mxl"]
# ).wait()

# pdmx_piano_clean["song_length.bars"].hist(bins=list(range(0, 500, 10)), nbins=500)
# plt.gca().set_xscale("log")
# plt.show()
# pdmx_piano_clean["n_notes"].hist(bins=list(range(0, 10000, 20)), nbins=500)
# plt.show()

h = px.histogram(
    pdmx_piano_clean["song_length.bars"],
    nbins=math.ceil(pdmx_piano_clean["song_length.bars"].max() / 10),
    title="Distribution of number of bars",
)
h.show()

h = px.histogram(
    pdmx_piano_clean["n_notes"],
    nbins=math.ceil(pdmx_piano_clean["n_notes"].max() / 500),
    title="Distribution of number of notes",
)
h.show()


# %% Prepare comparison with ASAP test pieces
import re

# from data.components.score_performance_dataset import TEST_PIECES
from data.components.score_performance_dataset import AsapDataset

# Load ASAP and compute textual descriptions
asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
asap_dataset_val_test = AsapDataset(asap_path, split="validation")
asap_dataset_val_test.entries += AsapDataset(asap_path, split="test").entries
# Drop duplicated over composer and title
unique_entries = {}
for entry in asap_dataset_val_test.entries:
    if (key := (entry.composer, entry.title)) not in unique_entries:
        unique_entries[key] = entry
asap_dataset_val_test.entries = list(unique_entries.values())
print(len(asap_dataset_val_test))


def create_descriptor(title: str, composer: str):
    descriptor = f"{title}; by {composer}."
    descriptor = re.sub(pattern=r'[^ a-zA-Z0-9,.?!;:()&-]', repl=" ", string=descriptor)
    descriptor = " ".join(descriptor.split())  # remove wierd whitespace
    return descriptor


descriptions = [create_descriptor(e.title, e.composer) for e in asap_dataset_val_test.entries]

# Load embedding model
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

device = "cpu"
model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2", device=device)
asap_embeddings = model.encode(
    sentences=descriptions,
    batch_size=64,
    show_progress_bar=True,
    output_value="sentence_embedding",
    device=device,
    convert_to_tensor=True,
).to(torch.float64)
asap_magnitudes = torch.from_numpy(
    np.linalg.norm(asap_embeddings, axis=1)
)  # probably OK to directly use torch

# Load pdmx embeddings (only for clean dataset)
pdmx_embeddings_path = Path(
    "~/Documents/datasets/pdmx_tests/.deduplicate_intermediate_data/embeddings.csv"
).expanduser()
pdmx_magnitudes_path = pdmx_embeddings_path.parent / "magnitudes.csv"
pdmx_embeddings = torch.from_numpy(
    pd.read_csv(pdmx_embeddings_path, sep=",", header=None, index_col=False).values[
        pdmx_piano_clean.index
    ]
)
pdmx_magnitudes = torch.from_numpy(
    pd.read_csv(pdmx_magnitudes_path, sep=",", header=None, index_col=False).values[
        pdmx_piano_clean.index
    ]
).squeeze(-1)

# %% Find similar pieces in ASAP and PDMX
idx_to_remove = set()
for query_description, query_embedding, query_magnitude in zip(
    descriptions, asap_embeddings, asap_magnitudes
):
    similarities = torch.matmul(input=pdmx_embeddings, other=query_embedding)
    similarities = similarities / (pdmx_magnitudes * query_magnitude)
    similarities = (similarities + 1) / 2

    SIMILARITY_THRESHOLD = 0.8
    print("*", query_description)
    sim = torch.where(similarities >= SIMILARITY_THRESHOLD)[0]
    tmp = pdmx_piano_clean.iloc[sim][
        ["song_name", "title", "subtitle", "artist_name", "composer_name"]
    ]
    tmp["sim"] = similarities[sim]
    tmp = tmp.sort_values("sim", ascending=False)
    tmp = tmp[tmp.columns.tolist()[-1:] + tmp.columns.tolist()[:-1]]
    idx_to_remove.update(set(tmp.index))
    print(tmp)
    print()

print(f"Number of scores to remove among piano dedup/rated: {len(idx_to_remove)}")
for i in idx_to_remove:
    item = pdmx_piano_clean.loc[i]
    for field in ["song_name", "title", "subtitle"]:
        if not isinstance(item[field], float):
            name = item[field]
            break
    for field in ["artist_name", "composer_name"]:
        if not isinstance(item[field], float):
            composer = item[field]
            break
    print(f" * {i}: '{name}' - '{composer}'")

print(
    f"Number of scores to remove among piano dedup/rated>4: {len([i for i in idx_to_remove if i in pdmx_piano_clean_good_rating.index])}"
)
print(
    f"""Number of scores to remove among piano dedup/rated>4/notes<500: {
    len(
        [
            i for i in idx_to_remove
            if i in pdmx_piano_clean_good_rating[pdmx_piano_clean_good_rating["n_notes"] <= 500].index
        ]
    )
    }"""
)

# %% Export excluded scores
idx_to_remove = sorted(idx_to_remove)
with (pdmx_path / "overlapping_asap.csv").open("w") as f:
    for idx in idx_to_remove:
        f.write(f"{idx}\n")
