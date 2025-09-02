"""
Test tokenizing PDMX scores
"""

import json
import subprocess
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Literal

import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

tqdm.pandas()
pandarallel.initialize(progress_bar=True)
pd.options.plotting.backend = "plotly"
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


# %% Setup paths
pdmx_path = Path("~/Documents/datasets/pdmx").expanduser()
pdmx_metadata_path = pdmx_path / "PDMX.csv"
pdmx_subsets_path = pdmx_path / "subset_paths"
pdmx_mxl_path = pdmx_path / "mxl"


# %% Read PDMX piano scores
pdmx = pd.read_csv(pdmx_metadata_path)  # .head(10000)
pdmx = pdmx[pdmx["subset:all_valid"]]


# %%
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

# pdmx_clean = pdmx[
#     pdmx['subset:rated_deduplicated'] & (pdmx["rating"] >= 4.0) & (pdmx["n_notes"] <= 500)
# ]
pdmx_clean = pdmx[pdmx['subset:all_valid']]
pdmx_piano_clean = pdmx_clean[pdmx_clean["instruments"].apply(lambda i: i == ["Piano"])]


def open_in_musescore(path: Path):
    MUSESCORE_PATH = Path(
        "~/Applications/MuseScore-Studio-4.5.2.251141401-x86_64.AppImage"
    ).expanduser()
    return subprocess.Popen(
        [MUSESCORE_PATH.as_posix(), path.as_posix()],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# %% Function to test a score
from scripts.score_encoding_verification.check_divisions import (
    check_path as gather_divisions_exceptions,
)
import scripts.score_encoding_verification.simpler_tokens_with_measure_rests as tk
import partitura as pt
import partitura.score


class ReadingError(Exception):
    pass


class MultiPartError(Exception):
    pass


class UnfoldingError(Exception):
    pass


class RestFillingError(Exception):
    pass


class TokenizationError(Exception):
    pass


class DivisionsError(Exception):
    pass


def check_automata(
    path: Path,
    allow_longer_measures: bool,
    allow_duplicate_notes_in_chord: bool,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            score = pt.load_musicxml(path, force_note_ids=True)
        except Exception as e:
            raise ReadingError() from e

        if len(score.parts) != 1:
            raise MultiPartError()

        score = score.parts[0]

        try:
            score = pt.score.unfold_part_maximal(score, ignore_leaps=False)
        except Exception as e:
            raise UnfoldingError() from e

    try:
        tk.fill_rests(score)
    except Exception as e:
        raise RestFillingError() from e

    try:
        tokens = tk.partitura_to_token_sequence(score)
    except Exception as e:
        raise TokenizationError() from e

    automata = tk.ConstrainedAutomata(
        num_voices=16,
        allow_longer_measures=allow_longer_measures,
        allow_duplicate_notes_in_chord=allow_duplicate_notes_in_chord,
    )
    automata.transition_sequence(tokens)

    return None


def check_divisions(path: Path) -> None:
    exceptions = gather_divisions_exceptions(path)
    if len(exceptions) > 0:
        raise DivisionsError(exceptions)


def check_full(path) -> str:
    path = pdmx_path / path
    try:
        check_divisions(path)
        check_automata(path, allow_longer_measures=True, allow_duplicate_notes_in_chord=True)
    except Exception as e:
        exc = type(e).__name__
        print(f"{exc}: {path.as_posix()}")
        return exc
    # except IndexError:
    #     return "IndexError (likely unfolding)"
    # except StopIteration:
    #     return "StopIteration (likely reading)"
    # except KeyError:
    #     return "KeyError (likely micro-tones)"
    # except ValueError:
    #     return "ValueError (likely rest filling)"
    # except AttributeError:
    #     return "AttributeError (likely reading on a fingerprint)"
    # except tk.TransitionError:
    #     return "TransitionError"
    # except UnfoldingError:
    #     return "UnfoldingError"
    # except ReadingError:
    #     return "ReadingError"
    # except TokenizationError:
    #     return "TokenizationError"
    # except MultiPartError:
    #     return "MultiPartError"
    # except

    return ""


# pdmx_piano_clean["errors"] = pdmx_piano_clean["mxl"].parallel_apply(check_full)


# %% Setup parallelization
import multiprocessing as mp

if "errors" in pdmx_piano_clean.columns:
    print("Column 'errors' already existing, not re-running the computation.")
else:
    with mp.Pool(10) as pool:
        pdmx_piano_clean["errors"] = list(
            tqdm(pool.imap(check_full, pdmx_piano_clean["mxl"]), total=len(pdmx_piano_clean))
        )


# %% Explore results
print(
    "Number of scores with error:",
    len(pdmx_piano_clean[pdmx_piano_clean["errors"] != ""]),
    "/",
    len(pdmx_piano_clean),
)
# Group by error and display numbers
grouped = pdmx_piano_clean.groupby("errors").size()
print(grouped)

# %% Export PDMX tokenizable
# pdmx_piano_clean[pdmx_piano_clean["errors"] == ""][
#     [c for c in pdmx_piano_clean if c != "errors"]
# ].to_csv(pdmx_path / "PDMX_piano_>4*_<500n_tokenizable.csv")
pdmx_piano_clean[pdmx_piano_clean["errors"] == ""][
    [c for c in pdmx_piano_clean if c != "errors"]
].to_csv(pdmx_path / "PDMX_piano_tokenizable.csv")
