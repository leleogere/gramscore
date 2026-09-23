"""
Experiences to make sure padding left a batch of sequences is OK for a model.
"""

import torch.nn.functional as F

import itertools

import pretty_midi as pm

import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import partitura as pt
import torch
from loguru import logger
from pretty_midi import note_number_to_name
from torch.nn.utils.rnn import pad_sequence

import data.components.transforms as t
import models.automata.tokens as tk
import utils.debug
from data.base_tokens_datamodule import BaseTokensDataModule
from data.components.score_performance_dataset import ScorePerformanceEntry, ScorePerformanceDataset
from data.base_tokens_datamodule import BaseTokensDataModule
from models.automata.io import token_sequence_to_partitura, token_sequence_to_partitura_permissive
from models.automata.vectorization_torch import TokenTensor
from utils.generation import GenerationState
from models.tokens_transcriber_module import TokensTranscriberModule
from utils.constants import PADDING_VALUE
from utils.debug import as_pycharm_link, performance_streams_to_midi


# %% Parameters
model_path = Path(
    "~/Documents/transcription/gramscore2/outputs/server/2026-02-16_16-41-28_spring-sunset-330/checkpoints/last.ckpt"
).expanduser()
NOTES_PER_CHUNK = 100
TOKENS_LIMIT_PER_CHUNK = 10


# %% Load the model
model: TokensTranscriberModule = TokensTranscriberModule.load_from_checkpoint(
    model_path,
    map_location="cpu",
    weights_only=False,
)
model.eval()


# %% Generate random performances
torch.random.manual_seed(10)
n = 2
performances = []
for i in range(n):
    n_notes = torch.randint(20, 50, (1,)).item()
    performance = {
        "performance_streams_raw": {
            "pitch": torch.randint(0, 127, (n_notes,)),
            "duration": torch.randint(1, 10, (n_notes,)),
            "onset": torch.sort(torch.randint(0, 100, (n_notes,))).values,
            "velocity": torch.randint(0, 127, (n_notes,)),
        },
    }
    performance["performance_onset_times"] = torch.clone(
        performance["performance_streams_raw"]["onset"]
    )
    performance["performance_streams"] = t.VectorizedPerformanceStreamsBucketizer().process(
        performance["performance_streams_raw"]
    )
    performances.append(performance)
print("Performances:")
for performance in performances:
    print(f"  - {performance['performance_onset_times'].shape}")


# %% Generate random prefixes
torch.random.manual_seed(11)
prefixes = [
    TokenTensor(3 * torch.rand(n_tokens, TokenTensor.TOTAL_DIMENSIONS)).quantize()
    for n_tokens in [5, 7]  # torch.randint(10, 30, (n,))
]
print("Prefixes:")
for prefix in prefixes:
    print(f"  - {prefix.shape}: {' | '.join([t.__class__.__name__ for t in prefix.to_tokens()])}")


# %% Predict performances sequentially
sequential_predictions = []
for i, (performance, prefix) in enumerate(zip(performances, prefixes)):
    streams = performance["performance_streams"]
    with torch.no_grad():
        performance_tokens, _ = model.transcriber_model.generate(
            performance_streams={k: v.unsqueeze(0) for k, v in streams.items() if k != "pad"},
            performance_onset_times=performance["performance_onset_times"].unsqueeze(0),
            performance_attention_mask=performance["performance_onset_times"].unsqueeze(0)
            != PADDING_VALUE,
            prefix=prefix.unsqueeze(0),
            constraint=False,
            max_constraint_iterations=None,
            max_generated_length=TOKENS_LIMIT_PER_CHUNK,
            use_cache=True,
            progress_bar=False,
            return_logits_and_mask=False,
        )
    sequential_predictions.append(performance_tokens)
print("Sequential predictions:")
for performance_tokens in sequential_predictions:
    print(
        f"  - {performance_tokens.shape}: "
        f"{' | '.join([t.__class__.__name__ for t in performance_tokens[0].to_tokens()])}"
    )


# %% Predict performances batched
batched_performances = BaseTokensDataModule.collate_fn(performances, keep_fields="all")
batched_prefixes = pad_sequence(
    prefixes, batch_first=True, padding_value=PADDING_VALUE, padding_side="left"
)
prefix_lengths = torch.LongTensor([prefix.shape[0] for prefix in prefixes])
performance_mask = batched_performances["performance_onset_times"] != PADDING_VALUE
with torch.no_grad():
    batched_predictions, _ = model.transcriber_model.generate(
        performance_streams=batched_performances["performance_streams"],
        performance_onset_times=batched_performances["performance_onset_times"],
        performance_attention_mask=performance_mask,
        prefix=batched_prefixes,
        constraint=False,
        max_constraint_iterations=None,
        max_generated_length=TOKENS_LIMIT_PER_CHUNK,
        progress_bar=False,
        use_cache=True,
        return_logits_and_mask=False,
    )
print(f"Batched predictions: {batched_predictions.shape}")
for performance_tokens in batched_predictions:
    print(f"  - {' | '.join([t.__class__.__name__ for t in performance_tokens.to_tokens()])}")


# %% Compare
padded_sequential_predictions = pad_sequence(
    [p.squeeze(0) for p in sequential_predictions],
    batch_first=True,
    padding_value=PADDING_VALUE,
    padding_side="left",
)
print(f"Sequential predictions: {padded_sequential_predictions.shape}")
print(f"Batched predictions: {batched_predictions.shape}")
diff = batched_predictions - padded_sequential_predictions
print(f"Equal: {torch.all(diff).item()}")
utils.debug.show_matrix(diff[0])
utils.debug.show_matrix(diff[1])
