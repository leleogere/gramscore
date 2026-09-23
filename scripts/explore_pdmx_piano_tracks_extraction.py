"""
Explore extracting PDMX piano tracks from multitracks pieces.
"""

import partitura as pt

import ast
from collections import defaultdict
from functools import partial
from pathlib import Path

import joblib
import lxml
import pandas as pd
from lxml import etree
from tqdm import tqdm

from utils.musicxml import read_musicxml

tqdm.pandas()

# Increase max column width and other display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", 1000)


# %% Read PDMX
pdmx_path = Path("~/Documents/datasets/pdmx").expanduser()

df = pd.read_csv(pdmx_path / "PDMX_extended_metadata.csv")
df["instruments"] = df["instruments"].apply(ast.literal_eval)
df["n_staves"] = df["n_staves"].apply(
    lambda s: ast.literal_eval(s) if not isinstance(s, float) else []
)
df = df.head(1000)


# %% Extract instruments directly in XML file to try a 3rd method


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def extract_instruments_info_from_xml(
    base_path: Path, mxl_path: Path
) -> dict[str, list[dict[str, str | None]]] | None:
    """Parse MusicXML file, and extract: instrument name, midi channel and midi program."""
    if isinstance(mxl_path, float):
        return None
    try:
        xml = read_musicxml(base_path / mxl_path)
        root = etree.fromstring(xml.encode("utf-8"))
    except (lxml.etree.XMLSyntaxError, TypeError) as e:
        print(f"Error parsing {mxl_path}: {e}")
        return None
    instruments = defaultdict(lambda: defaultdict(list))
    for i, part in enumerate(root.findall(".//score-part")):
        part_name_el = part.find(".//part-name")
        part_name = (
            part_name_el.text if part_name_el is not None and part_name_el.text else f"UKN{i}"
        )
        midi_device_el = part.find(".//midi-device")
        midi_device_port = midi_device_el.get("port", None) if midi_device_el is not None else None

        instruments[part_name]["midi_device_port"] = midi_device_port

        for score_instrument_el in part.findall(".//score-instrument"):
            instrument_id = score_instrument_el.get("id", None)
            instrument_name_el = score_instrument_el.find(".//instrument-name")
            instrument_name = instrument_name_el.text if instrument_name_el is not None else None
            instrument_sound_el = score_instrument_el.find(".//instrument-sound")
            instrument_sound = instrument_sound_el.text if instrument_sound_el is not None else None
            instruments[part_name]["score_instruments"].append(
                {
                    "instrument_id": instrument_id,
                    "instrument_name": instrument_name,
                    "instrument_sound": instrument_sound,
                }
            )

        for midi_instrument_el in part.findall(".//midi-instrument"):
            instrument_id = midi_instrument_el.get("id", None)
            midi_channel_el = midi_instrument_el.find(".//midi-channel")
            if midi_channel_el is not None:
                midi_channel = midi_channel_el.text if midi_channel_el is not None else None
                midi_program_el = midi_instrument_el.find(".//midi-program")
                midi_program = midi_program_el.text if midi_program_el is not None else None
                midi_unpitched_el = midi_instrument_el.find(".//midi-unpitched")
                midi_unpitched = midi_unpitched_el.text if midi_unpitched_el is not None else None
            else:
                midi_channel = None
                midi_program = None
                midi_unpitched = None
            instruments[part_name]["midi_instruments"].append(
                {
                    "instrument_id": instrument_id,
                    "midi_channel": midi_channel,
                    "midi_program": midi_program,
                    "midi_unpitched": midi_unpitched,
                }
            )

    return default_to_regular(instruments)


# instruments_info = df["mxl"].progress_apply(lambda x: extract_instruments_info_from_xml(x))
instruments_info = [
    _
    for _ in tqdm(
        joblib.Parallel(n_jobs=-1, return_as="generator")(
            joblib.delayed(partial(extract_instruments_info_from_xml, base_path=pdmx_path))(
                mxl_path=path
            )
            for path in df["mxl"]
        ),
        total=len(df["mxl"]),
    )
]


# %% Merge into df
df["instruments_info"] = instruments_info

# %%
import partitura as pt

pdmx_dir = Path("~/Documents/datasets/pdmx").expanduser()

p = df["mxl"].iloc[2]  # No info
# p = "./mxl/1/5/Qmb6XhSUsGMzLacW2FNnrJojcL3cB7B7v4BFw5V57R6rrw.mxl"  # drumset
# p = "./mxl/1/11/QmbbUxwDmVsaQw6AUay5fAjKfpnfxJcpPPmiXWK1ximCsY.mxl"  # piano
# p = "./mxl/1/11/QmbbH46ccvGJkoM3N2HMJiCrHzufPvDBKkTHEsPLXJPYZT.mxl"  # piano and choir
score = pt.load_musicxml(pdmx_dir / p)
for part in score.parts:
    print(f"name={part.part_name} | abbr={part.part_abbreviation} | id={part.id}")

# %% Different cell to avoid
df["xml_instrument_names"] = [
    "|".join([i["instrument_name"] if i["instrument_name"] is not None else "XXX" for i in x])
    for x in instruments_info
]
df["xml_midi_channels"] = [
    "|".join([i["midi_channel"] if i["midi_channel"] is not None else "XXX" for i in x])
    for x in instruments_info
]
df["xml_midi_programs"] = [
    "|".join([i["midi_program"] if i["midi_program"] is not None else "XXX" for i in x])
    for x in instruments_info
]

df_instr = df[
    [
        "mxl",
        "mid",
        "n_tracks",
        "n_parts",
        "n_staves",
        "tracks",
        "instruments",
        "xml_instrument_names",
        "xml_midi_channels",
        "xml_midi_programs",
    ]
]
df_instr_broken = df_instr[~df["_instrument_matched"]]

example = df_instr_broken.loc[96]
example2 = df_instr_broken.loc[110]
