from collections import defaultdict
from pathlib import Path
import pickle
import warnings
import csv
import random

import partitura as pt
import partitura.score
from tqdm.auto import tqdm

from src.models.automata.tokens import fill_rests
from src.models.automata.automata import ConstrainedAutomata
from src.models.automata.io import token_sequence_to_partitura, partitura_to_token_sequence
from src.models.automata.voice_reordering import reorder_voices

# asap_path = Path("/home/gerel/Documents/datasets/asap-dataset")
# output_path = Path("/home/gerel/Documents/datasets/asap-dataset-tokenized")
asap_path = Path("/home/gerel/Documents/datasets/bach-dataset")
output_path = Path("/home/gerel/Documents/datasets/bach-dataset-tokenized")

CACHE = False

random.seed(42)

score_paths = list(asap_path.rglob("xml_score.musicxml"))
random.shuffle(score_paths)
print(f"Found {len(score_paths)} scores in {asap_path}")
errors: dict[str, list[Path]] = defaultdict(list)  # err_msg -> paths

for score_path in (pbar := tqdm(score_paths, position=0, leave=True)):
    save_path = (output_path / score_path.relative_to(asap_path)).with_suffix(".pkl")
    if save_path.exists() and CACHE:
        continue

    num_errors = sum(len(paths) for paths in errors.values())
    pbar.set_postfix({"score": score_path, "errors": num_errors})
    automata = ConstrainedAutomata(16)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = pt.load_musicxml(score_path, force_note_ids=True)
    except Exception as e:
        pbar.write(f"Failed to load {score_path}: {e}")
        errors[f"Loading error [{str(e)}]"].append(score_path)
        continue

    if len(score.parts) != 1:
        pbar.write(f"Skipping score with {len(score.parts)} parts: {score_path}")
        errors["More than 1 part"].append(score_path)
        continue

    score = score.parts[0]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = pt.score.unfold_part_maximal(score, ignore_leaps=False)
    except Exception as e:
        pbar.write(f"Failed to unfold {score_path}: {e}")
        errors[f"Unfolding error [{str(e)}]"].append(score_path)
        continue

    try:
        # pt.score.fill_rests(score, measurewise=True)
        fill_rests(score)
    except Exception as e:
        pbar.write(f"Failed to fill rests {score_path}: {e}")
        errors[f"Rest filling error [{str(e)}]"].append(score_path)
        continue

    score = reorder_voices(score)

    try:
        tokens = partitura_to_token_sequence(score)
    except Exception as e:
        pbar.write(f"Failed to tokenize {score_path}: {e}")
        errors[f"Tokenization error [{str(e)}]"].append(score_path)
        continue

    try:
        automata.transition_sequence(tokens)
    except ValueError as e:
        pbar.write(f"Failed to transition {score_path}: {e}")
        errors[f"Transition error [{str(e)}]"].append(score_path)
        continue

    # regenerated_score = token_sequence_to_partitura(tokens)
    # pt.save_musicxml(regenerated_score, "/tmp/regenerated_score.musicxml")

    for token in tokens:
        if hasattr(token, "_partitura_elements"):
            del token._partitura_elements

    # print(score_path)
    # for token, retbug with okenized_token in zip(
    #     tokens, partitura_to_token_sequence(token_sequence_to_partitura(tokens))
    # ):
    #     if token == retokenized_token:
    #         print("  ", end="")
    #     else:
    #         print("* ", end="")
    #     print(f"{token}     {retokenized_token}")
    # if token != retokenized_token:
    #     print(f"* {token} [{str(token._partitura_elements)[0]}]")
    #     print(f"  {retokenized_token} [{str(retokenized_token._partitura_elements[0])}]")

    assert all(
        token == retokenized_token
        for token, retokenized_token in zip(
            tokens,
            partitura_to_token_sequence(token_sequence_to_partitura(tokens)),
        )
    ), "Retokenization failed"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(tokens, f)

num_errors = sum(len(paths) for paths in errors.values())
print(f"Tokenized {len(score_paths) - num_errors} scores, {num_errors} errors")

# Sort errors
errors_tuples = list(errors.items())
errors_tuples.sort(key=lambda x: x[0])
for _, paths in errors_tuples:
    paths.sort()

# for err_msg, paths in errors_tuples:
#     print(f"  > {err_msg}: {len(paths)}")

# print()
# for err_msg, paths in errors.items():
#     print(f"{err_msg}:")
#     for path in paths:
#         print(f"  > {path}")

# Save errors to a file
errors_path = output_path / "errors.csv"
with open(errors_path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["error", "path"])
    for err_msg, paths in errors_tuples:
        for path in paths:
            writer.writerow([err_msg, path])
