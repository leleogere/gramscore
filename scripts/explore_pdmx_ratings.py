import ast
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from tqdm import tqdm

from models.automata.vectorization_torch import TokenTensor

tqdm.pandas()

# Increase max column width and other display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", 1000)


# %% Read PDMX
pdmx_path = Path("~/Documents/datasets/pdmx").expanduser()

df = pd.read_csv(
    pdmx_path / "PDMX_extended_metadata.csv",
    converters={
        "allowed_transpositions": lambda transp_list: (
            ast.literal_eval(transp_list)
            if isinstance(transp_list, str) and transp_list != ""
            else []
        ),
        "time_signatures": lambda set_str: (
            ast.literal_eval(set_str)
            if isinstance(set_str, str) and set_str != "{}" and set_str != ""
            else set()
        ),
        "n_staves": lambda list_staves: (
            tuple(ast.literal_eval(list_staves))
            if isinstance(list_staves, str) and list_staves != ""
            else ()
        ),
        "instruments": ast.literal_eval,
    },
)
print(f"Initial dataset size: {len(df)}")
# df["instruments"] = df["instruments"].apply(ast.literal_eval)
# df["n_staves"] = df["n_staves"].apply(
#     lambda s: ast.literal_eval(s) if not isinstance(s, float) else []
# )

# # Only keep usable scores
# df = df[df["usable"]]
# df = df[df["n_notes_midi"] >= 3]
# df = df[df["n_staves"].isin([(1,), (2,)])]
# df = df[(df["n_measures_score"] == df["n_measures_midi"]) & (df["n_measures_score"] != -1)]
# df = df[
#     (df["time_signatures"].apply(len) > 0)
#     & (df["time_signatures"].map(lambda tss: all(ts in TokenTensor.TS_INDEXER for ts in tss)))
# ]
# print(f"After filtering: {len(df)}")


# %% Check some stats about ratings
rated_df = df[df["is_rated"]]
print(f"Number of rated pieces: {len(rated_df)}")

# Plot ratings histogram
# rated_df["rating"].hist(bins=np.arange(2.5, 5.1, 0.1))
# plt.gca().set_yscale("log")
# plt.show()

# Plot a matrix of rating vs number of rates (2D histogram)
rating_bins = np.arange(2.5, 5.05, 0.1)
print(rated_df["n_ratings"].min())
n_ratings_bins = np.logspace(
    np.log10(rated_df["n_ratings"].min()), np.log10(rated_df["n_ratings"].max() + 5000), 30
)
hist, xedges, yedges = np.histogram2d(
    rated_df["rating"], rated_df["n_ratings"], bins=[rating_bins, n_ratings_bins]
)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.pcolormesh(xedges, yedges, hist.T, cmap="viridis", shading="auto", norm=LogNorm())
ax.set_yscale("log")
ax.set_xlabel("Rating")
ax.set_ylabel("Number of ratings")
ax.set_title("2D Histogram: rating vs number of ratings")
plt.colorbar(im, ax=ax, label="Count")

# Add secondary y-axis showing bin edges
ax2 = ax.twinx()
ax2.set_yscale("log")
ax2.set_ylim(ax.get_ylim())
# Set ticks at bin edges
ax2.set_yticks(n_ratings_bins)
ax2.set_yticklabels([f'{math.ceil(val)}' for val in n_ratings_bins], fontsize=6)
# Disable minor ticks on secondary y-axis to show only bin edges
ax2.minorticks_off()

# Add text annotations for cells with small counts
for i in range(len(rating_bins) - 1):
    for j in range(len(n_ratings_bins) - 1):
        count = hist[i, j]
        if count > 0:
            x_center = (xedges[i] + xedges[i + 1]) / 2
            y_center = np.sqrt(yedges[j] * yedges[j + 1])  # Geometric mean for log scale
            ax.text(
                x_center,
                y_center,
                f'{int(count)}',
                ha='center',
                va='center',
                fontsize=7,
                color='white',
                fontweight='bold',
            )

# Add horizontal lines with labels indicating the total number of pieces above the line
# y_thresholds = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
y_thresholds = n_ratings_bins[::2]
for threshold in y_thresholds:
    if threshold >= rated_df["n_ratings"].min() and threshold <= rated_df["n_ratings"].max():
        pieces_above = rated_df[rated_df["n_ratings"] >= threshold]
        n_pieces_above = len(pieces_above)
        avg_rating = pieces_above["rating"].mean()
        median_rating = pieces_above["rating"].median()
        min_rating = pieces_above["rating"].min()
        max_rating = pieces_above["rating"].max()
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.3)
        ax.text(
            xedges[0] + 0.01,
            threshold,
            "\n".join(
                [
                    f"nR ≥ {math.ceil(threshold)} ({n_pieces_above})",
                    f"Avg: {avg_rating:.2f}",
                    f"Med: {median_rating:.2f}",
                    f"MinMax: [{min_rating:.2f}, {max_rating:.2f}]",
                ]
            ),
            verticalalignment='bottom',
            color='red',
            fontsize=5.5,
            fontweight='bold',
        )

# Add vertical lines with labels indicating the total number of pieces above the line
x_thresholds = rating_bins[6::2]
for threshold in x_thresholds:
    if threshold >= rated_df["rating"].min() and threshold <= rated_df["rating"].max():
        pieces_above = rated_df[rated_df["rating"] >= threshold]
        n_pieces_above = len(pieces_above)
        avg_n_ratings = pieces_above["n_ratings"].mean()
        median_n_ratings = pieces_above["n_ratings"].median()
        min_n_ratings = pieces_above["n_ratings"].min()
        max_n_ratings = pieces_above["n_ratings"].max()
        ax.axvline(x=threshold, color='green', linestyle='--', linewidth=1.5, alpha=0.3)
        ax.text(
            threshold + 0.01,
            yedges[-1] - 500,
            "\n".join(
                [
                    f"R ≥ {threshold:.2f}★",
                    f"({n_pieces_above})",
                    f"Avg: {avg_n_ratings:.2f}",
                    f"Med: {median_n_ratings:.2f}",
                    f"Min: {int(min_n_ratings):}",
                    f"Max: {int(max_n_ratings)}",
                ]
            ),
            horizontalalignment='left',
            verticalalignment='top',
            rotation_mode="anchor",
            color='green',
            fontsize=5.5,
            fontweight='bold',
            rotation=0,
        )

plt.tight_layout()
plt.show()


# %% Same matrix as above, but using a cumulative histogram, to see the cummulated counts when
# increasing both the number of ratings and the rating threshold

# Create cumulative histogram: for each bin (i,j), sum all bins where rating >= rating_i and n_ratings >= n_ratings_j
cumulative_hist = np.zeros_like(hist)
for i in range(len(rating_bins) - 1):
    for j in range(len(n_ratings_bins) - 1):
        # Sum all bins with rating >= current rating and n_ratings >= current n_ratings
        cumulative_hist[i, j] = hist[i:, j:].sum()

# Plot cumulative histogram
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.pcolormesh(
    xedges, yedges, cumulative_hist.T, cmap="viridis", shading="auto", norm=LogNorm()
)
ax.set_yscale("log")
ax.set_xlabel("Rating")
ax.set_ylabel("Number of ratings")
ax.set_title("Cumulative 2D Histogram: rating vs number of ratings")
plt.colorbar(im, ax=ax, label="Cumulative Count")

# Add secondary y-axis showing bin edges
ax2 = ax.twinx()
ax2.set_yscale("log")
ax2.set_ylim(ax.get_ylim())
# Set ticks at bin edges
ax2.set_yticks(n_ratings_bins)
ax2.set_yticklabels([f'{math.ceil(val)}' for val in n_ratings_bins], fontsize=6)
# Disable minor ticks on secondary y-axis to show only bin edges
ax2.minorticks_off()

# Add text annotations for cells with cumulative counts
for i in range(len(rating_bins) - 1):
    for j in range(len(n_ratings_bins) - 1):
        count = cumulative_hist[i, j]
        if count > 0:
            x_center = (xedges[i] + xedges[i + 1]) / 2
            y_center = np.sqrt(yedges[j] * yedges[j + 1])  # Geometric mean for log scale
            color = 'white' if count < 1000 else 'black'
            ax.text(
                x_center,
                y_center,
                f'{int(count)}',
                ha='center',
                va='center',
                fontsize=7,
                color=color,
                fontweight='bold',
                rotation=45,
            )

plt.tight_layout()
plt.show()


# %% Create an artificial matrix in which you compute a weight for each rating/n_ratings
# combination. This score is computed as rate_ij * n_ratings_ij / total_n_ratings

# Configure manual scaling factor and minimum weight threshold
scaling_factor = 1.0
min_weight = 0.0  # Minimum weight for sampling (weights below this will be clamped to this value)

# Calculate total number of ratings across all pieces
total_n_ratings = rated_df["n_ratings"].sum()

# Create weighted matrix
weighted_hist = np.zeros_like(hist)
for i in range(len(rating_bins) - 1):
    for j in range(len(n_ratings_bins) - 1):
        # Get the rating value (center of bin)
        rating_value = (rating_bins[i] + rating_bins[i + 1]) / 2
        # Get the n_ratings value (geometric mean of bin edges for log scale)
        n_ratings_value = np.sqrt(n_ratings_bins[j] * n_ratings_bins[j + 1])
        # Compute weight: rating * count * n_ratings / total_n_ratings * scaling_factor
        # weight = rating_value * hist[i, j] * n_ratings_value / total_n_ratings * scaling_factor
        # weight = np.log(np.exp(rating_value) * np.log(n_ratings_value + 1) * scaling_factor)
        R = 4  # Base rating
        W = 3  # Base weight
        weight = (R * W + rating_value * n_ratings_value) / (n_ratings_value + R)
        # Apply minimum weight clamping (only if count > 0)
        weighted_hist[i, j] = np.maximum(weight, min_weight)  # if hist[i, j] > 0 else 0

# Plot weighted histogram
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.pcolormesh(
    xedges,
    yedges,
    weighted_hist.T,
    cmap="viridis",
    shading="auto",
    # norm=LogNorm(vmin=min_weight if min_weight > 0 else 1e-3),
)
ax.set_yscale("log")
ax.set_xlabel("Rating")
ax.set_ylabel("Number of ratings")
ax.set_title("Sampling weights matrix")
plt.colorbar(im, ax=ax, label="Weight")

# Add secondary y-axis showing bin edges
ax2 = ax.twinx()
ax2.set_yscale("log")
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(n_ratings_bins)
ax2.set_yticklabels([f'{math.ceil(val)}' for val in n_ratings_bins], fontsize=6)
ax2.minorticks_off()

# Add text annotations for cells with weights
for i in range(len(rating_bins) - 1):
    for j in range(len(n_ratings_bins) - 1):
        weight = weighted_hist[i, j]
        if weight > 0:
            x_center = (xedges[i] + xedges[i + 1]) / 2
            y_center = np.sqrt(yedges[j] * yedges[j + 1])  # Geometric mean for log scale
            color = 'white' if weight < 4.3 else 'black'
            ax.text(
                x_center,
                y_center,
                # np.format_float_scientific(weight, precision=1, exp_digits=1),
                # f"{int(weight)}",
                f"{weight:.1f}\n{weight/min(weighted_hist.flatten()):.1f}x",
                ha='center',
                va='center',
                fontsize=6,
                color=color,
                fontweight='bold',
                rotation=0,
            )

plt.tight_layout()
plt.show()
