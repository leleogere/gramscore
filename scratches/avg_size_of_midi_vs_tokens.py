"""
Compute the average number of notes per MIDI files, and average number of tokens per score in ASAP
"""

import joblib
from tqdm import tqdm

import data.components.transforms as t

from pathlib import Path

from data.components.asap_dataset import AsapDataset
from data.components.score_performance_dataset import ScorePerformanceEntry
from utils.midi import compute_midi_end_without_tempo_changes

asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()


dataset = AsapDataset(directory=asap_path, split="all", transforms=[t.ToDict()])


# %% Compute average number of notes and tokens
transform = t.TransformChainer(
    [
        t.ScoreReader(),
        t.PerformanceReader(),
        t.PerformanceToPerformedNoteList(),
        t.ScoreVectorizer(),
    ]
)


def get_num_notes_and_tokens_and_duration(
    entry: ScorePerformanceEntry, transform: t.TransformChainer
):
    try:
        item = transform(entry)
    except Exception as e:
        return None, None, None
    notes = item["performance_notes"]
    tokens = item["score_tokens"]
    midi = item["performance"]
    return len(notes), len(tokens), midi.get_end_time()


def compute_metrics(
    num_notes: list[int],
    num_tokens: list[int],
    durations: list[float],
    num_items_ratios: list[float],
    durations_ratios: list[float],
) -> dict[str, str]:
    max_ratio_items_idx = num_items_ratios.index(max(num_items_ratios))
    min_ratio_items_idx = num_items_ratios.index(min(num_items_ratios))
    max_ratio_duration_idx = durations_ratios.index(max(durations_ratios))
    min_ratio_duration_idx = durations_ratios.index(min(durations_ratios))
    max_num_notes_idx = num_notes.index(max(num_notes))
    max_num_tokens_idx = num_tokens.index(max(num_tokens))
    max_duration_idx = durations.index(max(durations))
    return {
        "num_notes": f"{sum(num_notes) / len(num_notes):.2f}",
        "max_num_notes": f"{num_notes[max_num_notes_idx]}[{num_tokens[max_num_notes_idx]:.2f}t|{durations[max_num_notes_idx]:.2f}s]",
        "num_tokens": f"{sum(num_tokens) / len(num_tokens):.2f}",
        "max_num_tokens": f"{num_tokens[max_num_tokens_idx]}[{num_notes[max_num_tokens_idx]}n|{durations[max_num_tokens_idx]:.2f}s]",
        "duration": f"{sum(durations) / len(durations):.2f}",
        "max_duration": f"{durations[max_duration_idx]:.2f}[{num_tokens[max_duration_idx]:.2f}t|{num_notes[max_duration_idx]}n]",
        "ratio_num_items": f"{sum(num_tokens) / sum(num_notes):.2f}",
        "ratio_duration": f"{sum(num_tokens) / sum(durations):.2f}",
        "max_ratio_num_items": f"{num_items_ratios[max_ratio_items_idx]:.2f}[{num_tokens[max_ratio_items_idx]}/{num_notes[max_ratio_items_idx]}]",
        "min_ratio_num_items": f"{num_items_ratios[min_ratio_items_idx]:.2f}[{num_tokens[min_ratio_items_idx]}/{num_notes[min_ratio_items_idx]}]",
        "max_ratio_durations": f"{durations_ratios[max_ratio_duration_idx]:.2f}[{num_tokens[max_ratio_duration_idx]}/{durations[max_ratio_duration_idx]:.2f}]",
        "min_ratio_durations": f"{durations_ratios[min_ratio_duration_idx]:.2f}[{num_tokens[min_ratio_duration_idx]}/{durations[min_ratio_duration_idx]:.2f}]",
    }


num_notes_and_tokens_iterator = joblib.Parallel(n_jobs=12, return_as="generator")(
    joblib.delayed(get_num_notes_and_tokens_and_duration)(item, transform) for item in dataset
)

num_notes = []
num_tokens = []
durations = []
num_items_ratios = []
durations_ratios = []
for i, item in enumerate(pbar := tqdm(num_notes_and_tokens_iterator, total=len(dataset))):
    notes, tokens, duration = item
    if notes is None or tokens is None or duration is None:
        continue
    num_notes.append(notes)
    num_tokens.append(tokens)
    durations.append(duration)
    num_items_ratios.append(tokens / notes)
    durations_ratios.append(tokens / duration)
    metrics = compute_metrics(num_notes, num_tokens, durations, num_items_ratios, durations_ratios)
    pbar.set_postfix(**metrics)
    if (i + 1) % 100 == 0:
        print(f"Step {i}")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")


# %% Print results
for metric, value in compute_metrics(
    num_notes, num_tokens, durations, num_items_ratios, durations_ratios
).items():
    print(f"{metric}: {value}")


"""
num_notes: 3078.07
max_num_notes: 15979[24111.00t|1282.90s]
num_tokens: 4885.42
max_num_tokens: 24111[15979n|1282.90s]
duration: 287.39
max_duration: 1282.90[24111.00t|15979n]
ratio_num_items: 1.59
ratio_duration: 17.00
max_ratio_num_items: 2.49[3622/1455]
min_ratio_num_items: 1.00[2822/2808]
max_ratio_durations: 53.54[3622/67.65]
min_ratio_durations: 3.34[821/245.49]
"""
