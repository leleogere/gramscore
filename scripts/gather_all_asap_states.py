"""
Gather all transcription states in ASAP scores
"""

from PIL import Image

import data.components.transforms as t

import torch

import pandas as pd
from pathlib import Path

import partitura as pt
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

from models.automata.automata import ConstrainedAutomata
from models.automata.io import partitura_to_token_sequence
from models.automata.tokens import fill_rests, VoiceCompletionState
from models.automata.vectorization_torch import TokenTensor
from models.automata.voice_reordering import reorder_voices


def save_states_as_image(states: torch.Tensor, save_path: Path, cmap: str = "jet"):
    assert states.ndim == 2
    states = states.numpy().T
    states_normalized = (states - states.min()) / (states.max() - states.min())
    cmap = plt.get_cmap(cmap)
    vector_colored = (cmap(states_normalized) * 255).astype(np.uint8)
    img = Image.fromarray(vector_colored)
    img.save(save_path)


asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
metadata_path = asap_path / "metadata.csv"
valid_scores_file = asap_path / "valid.tsv"

metadata = pd.read_csv(metadata_path)
metadata.drop_duplicates("folder", inplace=True)
print(f"Number of unique ASAP scores: {len(metadata)}")

valid = pd.read_csv(valid_scores_file, sep="\t")

metadata = metadata.merge(valid, on="folder")
metadata = metadata[metadata["valid"]]
print(f"Number of valid ASAP scores: {len(metadata)}")

cache_file = asap_path / "asap_states.pt"

if not cache_file.exists():
    states = []
    tokens_all = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        score = pt.load_musicxml(asap_path / row["folder"] / "xml_score.musicxml")
        assert len(score) == 1
        score = score[0]
        fill_rests(score)
        score = reorder_voices(score)
        tokens = partitura_to_token_sequence(score)
        automata = ConstrainedAutomata(num_voices=8)
        current_score_states = []
        for token in tokens:
            automata.transition(token)
            current_score_states.append(automata.state.vectorize())
        current_score_states = torch.stack(current_score_states)
        states.append(current_score_states)
        save_states_as_image(current_score_states, asap_path / row["folder"] / "states.png")
        tokens_all.append(tokens)
    torch.save((states, metadata["folder"].tolist(), tokens_all), cache_file)
else:
    states, _, tokens_all = torch.load(cache_file, weights_only=False)

assert len(states) == 141

concatenated_states = torch.cat(states, dim=0)
print(f"Concatenated states shape: {concatenated_states.shape}")

# Perform PCA
pca = PCA(n_components=100)
pca.fit(concatenated_states.numpy())
print(f"PCA number of components: {pca.n_components_}")

# Plot variance explained by eigenvalues
plt.figure(figsize=(10, 6))
plt.semilogy(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Variance explained')
plt.title('PCA: Variance Explained by Each Eigenvalue')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot cumulated variance explained by eigenvalues
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), 'o-'
)
plt.xlabel('Principal Component')
plt.ylabel('Cumulated variance explained')
plt.title('PCA: Cumulated variance explained by each eigenvalue')
plt.grid(True)
plt.tight_layout()
plt.show()

# Project rows into 20 first PC
pca_transformed = pca.transform(concatenated_states.numpy())
print(f"Shape of transformed PCA: {pca_transformed.shape}")
# color by token type
token_types = [type(t) for s in tokens_all for t in s]
voices = [t.voice_idx if hasattr(t, "voice_idx") else 9 for s in tokens_all for t in s]
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i in range(10):
    ax = axes[i]
    pc_x = 2 * i
    pc_y = 2 * i + 1
    for token_type in TokenTensor.TOKEN_INDEXER:
        items = np.array([tt == token_type for tt in token_types])
        ax.scatter(
            pca_transformed[items, pc_x],
            pca_transformed[items, pc_y],
            alpha=0.5,
            s=1,
            label=token_type.__name__,
        )
    # for voice in range(9):
    #     items = np.array([v == voice for v in voices])
    #     ax.scatter(
    #         pca_transformed[items, pc_x],
    #         pca_transformed[items, pc_y],
    #         alpha=0.5,
    #         s=1,
    #         label=f"V{voice}" if voice != 8 else "X",
    #     )
    ax.set_xlabel(f'PC{pc_x + 1} (exp var: {pca.explained_variance_ratio_[pc_x]:.3f})')
    ax.set_ylabel(f'PC{pc_y + 1} (exp var: {pca.explained_variance_ratio_[pc_y]:.3f})')
    ax.set_title(f'PC{pc_x + 1} vs PC{pc_y + 1}')
    ax.grid(True)
    if i == 9:
        # Put legend on the right, on the exterior, with larger dots that in the plot
        ax.legend(
            loc='upper right',
            bbox_to_anchor=(1.2, 1),
            ncol=1,
            fontsize='small',
            markerscale=2,
        )
plt.suptitle('First 10 pairs of principal components', fontsize=16, y=1.00)
plt.tight_layout()
plt.show()


def get_variable_name(idx: int) -> str:
    # fmt: off
    if idx < 8 * VoiceCompletionState.n_vec_dims:
        voice_idx = idx // VoiceCompletionState.n_vec_dims
        dim_idx_in_voice = idx % VoiceCompletionState.n_vec_dims
        name = f"V{voice_idx}"
        if dim_idx_in_voice == 0: name += "D"
        elif dim_idx_in_voice == 1: name += "B"
        elif dim_idx_in_voice == 2: name += "P"
        elif dim_idx_in_voice == 3: name += "F"
        elif dim_idx_in_voice == 4: name += "C"
        elif dim_idx_in_voice == 5: name += "T"
        elif dim_idx_in_voice == 6: name += "CD"
        elif dim_idx_in_voice == 7: name += "CN"
        elif 8 <= dim_idx_in_voice < 136: name += f"CP{dim_idx_in_voice - 8}"
        elif dim_idx_in_voice == 136: name += "TND"
        elif dim_idx_in_voice == 137: name += "TAD"
        elif dim_idx_in_voice == 138: name += "TNU"
        elif dim_idx_in_voice == 139: name += "TAU"
        elif dim_idx_in_voice == 140: name += "TU"
        elif dim_idx_in_voice == 141: name += "TP"
        elif dim_idx_in_voice == 142: name += "TF"
    elif idx == 1144: name = "MC"
    elif 1145 <= idx < 1273: name = f"LP{idx - 1145}"
    elif 1273 <= idx < 1401: name = f"LM{idx - 1273}"
    else: raise ValueError(f"Unknown index: {idx}")
    # fmt: on
    return name


print("Variable names:")
for idx in range(1401):
    print(f"{idx}: {get_variable_name(idx)}")


# Project variables (columns) into 20 first PC
loadings = pca.components_.T  # Shape: (num_features, num_components)
print(f"Loadings shape: {loadings.shape}")

fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
axes2 = axes2.flatten()
num_variables = loadings.shape[0]
colors = plt.cm.viridis(np.linspace(0, 1, num_variables))
for i in range(10):
    ax = axes2[i]
    pc_x = 2 * i
    pc_y = 2 * i + 1
    ax.scatter(
        loadings[:, pc_x],
        loadings[:, pc_y],
        c=colors,
        alpha=0.6,
        s=10,
    )

    # Add labels for variables with norm > 0.3
    for var_idx in range(num_variables):
        norm = np.sqrt(loadings[var_idx, pc_x] ** 2 + loadings[var_idx, pc_y] ** 2)
        if norm > 0.3:
            ax.text(
                loadings[var_idx, pc_x],
                loadings[var_idx, pc_y],
                get_variable_name(var_idx),
                fontsize=6,
                ha='center',
                va='bottom',
            )

    ax.set_xlabel(f'PC{pc_x + 1} (exp var: {pca.explained_variance_ratio_[pc_x]:.3f})')
    ax.set_ylabel(f'PC{pc_y + 1} (exp var: {pca.explained_variance_ratio_[pc_y]:.3f})')
    ax.set_title(f'Variable loadings: PC{pc_x + 1} vs PC{pc_y + 1}')
    ax.grid(True)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)

    # Add unit circle
    circle = plt.Circle((0, 0), 1, color='red', fill=False, linestyle='--', linewidth=1.5)
    ax.add_patch(circle)

    # Center view on unit circle
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal', adjustable='box')

plt.suptitle('Variable projections on first 10 pairs of principal components', fontsize=16, y=1.00)
plt.tight_layout()
plt.show()
