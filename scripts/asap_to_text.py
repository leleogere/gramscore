from pathlib import Path
import joblib
from data.components.score_performance_dataset import ScorePerformanceDataset
from data.components.score_performance_dataset import ScorePerformanceEntry
from datasets import tqdm

from src.data.nasap_abc_lm_datamodule import AsapABCLanguageModelDataModule
from src.data.components.chunking_utils import ChunkingConfig

# fmt: off
SEPARATOR = "\n\n" + "#" * 60 + "\n" + "#" * 60 + "\n\n"
NASAP_PATH = Path("/local/DEEPLEARNING/EPR/nasap-dataset")
# SPLIT_NASAP_PATH = Path("/local/DEEPLEARNING/EPR/nasap-dataset_chunks=32_overlap=True_min-ratio=0.6")
SPLIT_NASAP_PATH = Path("/local/DEEPLEARNING/EPR/nasap-dataset_chunks=16_overlap=True_min-ratio=0.6")
TEXT_DATASET_PATH = SPLIT_NASAP_PATH.parent / (SPLIT_NASAP_PATH.name + "_TEXT")
N_JOBS = 63
# fmt: on

# %% Instantiate datamodule
asap_datamodule = AsapABCLanguageModelDataModule(
    data_dir=Path(NASAP_PATH),
    tokenizer="mistralai/Mistral-7B-v0.3",
    chunking_config=ChunkingConfig(
        new_dataset_path=SPLIT_NASAP_PATH,
        num_measures_per_chunk=32,
        overlap=True,
        min_matched_notes_ratio=0.6,
        skip_already_chunked=True,
    ),
    batch_size=1,
    num_workers=1,
)
asap_datamodule.prepare_data()
asap_datamodule.setup()

# %% Save all paths as a text file
TEXT_DATASET_PATH.mkdir(exist_ok=True, parents=True)


def save_item(
    metadata: ScorePerformanceEntry,
    item: dict[str, str],
    split: str,
    base_dir: Path,
    sep: str = SEPARATOR,
) -> str:
    text = sep.join([item["performance"], item["score"]])
    path = base_dir / metadata.midi_performance.relative_to(SPLIT_NASAP_PATH).with_suffix(".txt")
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        f.write(text)


for split in ["train", "val", "test"]:
    print(f"Saving {split} split...")
    dataset: ScorePerformanceDataset = getattr(asap_datamodule, f"data_{split}")
    rows = joblib.Parallel(n_jobs=N_JOBS, return_as="generator")(
        joblib.delayed(save_item)(metadata, item, split, TEXT_DATASET_PATH)
        for metadata, item in zip(dataset.entries, dataset)
    )
    for row in tqdm(rows, total=len(getattr(asap_datamodule, f"data_{split}"))):
        continue
