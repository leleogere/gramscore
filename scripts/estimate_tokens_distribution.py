"""
Estimate the distribution of tokens types in the dataset to correctly weight the loss function.
"""

import itertools
import pickle
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from data.asap_tokens_datamodule import AsapTokensDataModule

asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()

cache_path = Path("/tmp/asap_cache")
# cache_path.unlink(missing_ok=True)

if cache_path.exists():
    # Read from disk
    with open(cache_path, "rb") as f:
        token_types = pickle.load(f)
else:
    n_repetitions = 10
    data_module = AsapTokensDataModule(
        data_dir=asap_path,
        batch_size=1,
        filter_fn=lambda e: e.composer == "Bach" and e.midi_performance.stem == "midi_score",
        include_midi_scores=True,
        num_workers=4,
    )
    data_module.prepare_data()
    data_module.setup()
    token_types = Counter(
        itertools.chain.from_iterable(
            map(lambda t: t.__class__, entry["score_tokens"])
            for entry in tqdm(
                itertools.chain.from_iterable(data_module.data_train for _ in range(n_repetitions)),
                total=len(data_module.data_train) * n_repetitions,
                desc="Counting token types",
            )
        )
    )
    # Write to disk
    with open(cache_path, "wb") as f:
        pickle.dump(token_types, f)

print("Token types counts:")
for k, v in token_types.items():
    print(f"{v:7} | {k.__name__}")

print()
total = sum(token_types.values())
print("Token types distribution:")
for k, v in token_types.items():
    print(f"{v / total:7.2%} | {k.__name__}")

print()
print("Class weights:")
# Taken from https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
for k, v in token_types.items():
    print(f"{total / (len(token_types) * v):7.3f} | {k.__name__}")


# Token types counts:
#     530 | StartOfScore
#   12145 | Bar
#    1184 | Clef
#     530 | KeySignature
#  240501 | Note
#   20788 | Rest
#    2447 | ChordStart
#    2447 | ChordEnd
#     530 | EndOfScore
#    1199 | TupletStart
#    1199 | TupletEnd
#
# Token types distribution:
#   0.19% | StartOfScore
#   4.28% | Bar
#   0.42% | Clef
#   0.19% | KeySignature
#  84.83% | Note
#   7.33% | Rest
#   0.86% | ChordStart
#   0.86% | ChordEnd
#   0.19% | EndOfScore
#   0.42% | TupletStart
#   0.42% | TupletEnd
#
# Class weights:
#  48.628 | StartOfScore
#   2.122 | Bar
#  21.768 | Clef
#  48.628 | KeySignature
#   0.107 | Note
#   1.240 | Rest
#  10.532 | ChordStart
#  10.532 | ChordEnd
#  48.628 | EndOfScore
#  21.495 | TupletStart
#  21.495 | TupletEnd
