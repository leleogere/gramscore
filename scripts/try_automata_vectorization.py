"""
Play with transcription state vectorization
"""

import torch
import partitura as pt
from pathlib import Path
import matplotlib.pyplot as plt

from models.automata.automata import ConstrainedAutomata
from models.automata.io import partitura_to_token_sequence
from models.automata.tokens import Bar
from models.automata.voice_reordering import reorder_voices
from scripts.score_encoding_verification.simpler_tokens_with_measure_rests import fill_rests

asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
score_path = asap_path / "Beethoven/Piano_Sonatas/1-1/xml_score.musicxml"

score = pt.load_musicxml(score_path)[0]
fill_rests(score)
score = reorder_voices(score)

tokens = partitura_to_token_sequence(score)

automata = ConstrainedAutomata(num_voices=8)
vector_states = [automata.state.vectorize()]
measures = []
print(automata.state)

for i, token in enumerate(tokens):
    if isinstance(token, Bar):
        measures.append(i)
    print()
    print(" > ", token)
    automata.transition(token)
    vector_states.append(automata.state.vectorize())
    print(automata.state)

vector_states = torch.stack(vector_states)

from PIL import Image
import numpy as np

# Normalize vector_states to 0-1 range
vector_array = vector_states.numpy().T
vector_normalized = (vector_array - vector_array.min()) / (vector_array.max() - vector_array.min())

cmap = plt.get_cmap("jet")
vector_colored = (cmap(vector_normalized) * 255).astype(np.uint8)

# Save as PNG (RGB mode)
img = Image.fromarray(vector_colored)
img.save("state_evolution_raw.png")
print(f"Saved raw matrix image: state_evolution_raw.png (shape: {vector_normalized.shape})")


# # Plot matrix with time on x-axis
# fig = plt.gcf()
# DPI = fig.get_dpi()
# fig.set_size_inches(vector_states.shape[0] / float(DPI), vector_states.shape[1] / float(DPI))
#
# plt.imshow(vector_states.T, aspect='equal', cmap='viridis', interpolation='none')
# plt.colorbar(label='Value')
# plt.xlabel('Token')
# plt.ylabel('Vector dimension')
# plt.title('Transcription state evolution')
#
# # Set x-ticks to show bar numbers
# bar_numbers = list(range(1, 1 + len(measures)))
# plt.xticks(measures, bar_numbers)
#
# # Display only 1 label every 5 bars
# ax = plt.gca()
# for i, label in enumerate(ax.xaxis.get_ticklabels()):
#     if i % 5 != 0:
#         label.set_visible(False)
#
# plt.tight_layout()
# plt.savefig("state_evolution.png", dpi="figure")
# plt.show()
