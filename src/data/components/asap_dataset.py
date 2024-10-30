from __future__ import annotations

import csv
import itertools
import json
import traceback
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import pandas as pd
import partitura as pt
from loguru import logger
from tqdm import tqdm

from data.components.score_performance_dataset import ScorePerformanceDataset
from data.components.score_performance_dataset import ScorePerformanceEntry
from data.components.chunking_utils import chunk_partitura_score_and_performances
from utils.constants import SKIP
from utils.constants import SUCCESS_FILENAME
from utils.constants import TEST_PIECE_IDS
from utils.constants import TO_IGNORE_INDICES

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    import partitura.score


if __name__ == '__main__':
    ds = AsapDataset(Path("/tmp/asap-dataset"), "test")
    ds.chunk_nasap_dataset(
        new_dataset_path=Path("/tmp/chunk_test_asap"),
        num_measures_per_chunk=8,
        overlap=True,
        skip_already_chunked=True,
        n_jobs=-1,
        progress=True,
    )
