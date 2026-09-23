from operator import getitem

import data.components.transforms as t
import ast
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


# %% Are n_tracks and n_parts identical?
print(
    "'n_tracks' and 'n_parts' identical?",
    all((df["n_parts"] == df["n_tracks"]) | df["n_parts"].isna()),
)
# => Yes => remove n_parts from metadata extraction

# %% What is in tracks
# print(df[["mxl", "mid", "metadata", "tracks", "instruments"]])
# => Instruments (separated by "-"), with 0 being piano
# Check if number of 0 always mean Piano
#  => Count the number of 0, check if column "instruments" contains a string "Piano (x)"
number_of_zeros_in_track = df["tracks"].str.split("-").map(lambda x: x.count("0"))
df["_instrument_matched"] = [
    (
        (f"Piano ({x})" if x >= 2 else "Piano") in arr
        if x >= 1
        else all("Piano" not in i for i in arr)
    )
    for arr, x in zip(df["instruments"], number_of_zeros_in_track)
]
print("'tracks' and 'instruments' equivalents?", all(df["_instrument_matched"]))
print(f"Number of matched: {sum(df['_instrument_matched'])}")
print(f"Number of unmatched: {len(df) - sum(df['_instrument_matched'])}")
# => NO => Some scores metadata files lack one or multiple instruments, track numbers do not seem
# consistent with instruments...

# %% Count the number of piano solo pieces (both new and old method)
print("Number of piano solo pieces:")
old_piano_solo = df[df["instruments"].apply(lambda i: i == ["Piano"])]
new_piano_solo = df[df["tracks"].apply(lambda t: t == "0")]
print(f"  Old: {len(old_piano_solo)}")
print(f"  New: {len(new_piano_solo)}")
common_pieces = old_piano_solo.merge(new_piano_solo, on="mxl", how="inner")
print(f"  Common: {len(common_pieces)}")
only_in_new_pieces = new_piano_solo[~new_piano_solo["mxl"].isin(old_piano_solo["mxl"])]
print(f"  Only in new: {len(only_in_new_pieces)}")
for i, (_, row) in enumerate(only_in_new_pieces.iterrows()):
    print(f"   > {row['mxl']} \t{row['tracks']}\t{row['instruments']}")
    if i > 5:
        break
only_in_old_pieces = old_piano_solo[~old_piano_solo["mxl"].isin(new_piano_solo["mxl"])]
print(f"  Only in old: {len(only_in_old_pieces)}")
for i, (_, row) in enumerate(only_in_old_pieces.iterrows()):
    print(f"   > {row['mxl']} \t{row['tracks']}\t{row['instruments']}")
    if i > 5:
        break
common_with_single_staff = common_pieces[
    (common_pieces["n_staves_x"].str.len() > 0) & (common_pieces["n_staves_x"].str.get(0) == 1)
]
print(f"  Common with single staff: {len(common_with_single_staff)}")
for i, (_, row) in enumerate(common_with_single_staff.iterrows()):
    print(f"   > {row['mxl']} \t{row['n_staves_x']}")
    if i > 5:
        break
common_with_double_staff = common_pieces[
    (common_pieces["n_staves_x"].str.len() > 0) & (common_pieces["n_staves_x"].str.get(0) == 2)
]
print(f"  Common with double staff: {len(common_with_double_staff)}")
for i, (_, row) in enumerate(common_with_double_staff.iterrows()):
    print(f"   > {row['mxl']} \t{row['n_staves_x']}")
    if i > 5:
        break
common_with_more_than_two_staves = common_pieces[
    (common_pieces["n_staves_x"].str.len() > 0) & (common_pieces["n_staves_x"].str.get(0) > 2)
]
print(f"  Common with more than two staves: {len(common_with_more_than_two_staves)}")
for i, (_, row) in enumerate(common_with_more_than_two_staves.iterrows()):
    print(f"   > {row['mxl']} \t{row['n_staves_x']}")
    if i > 5:
        break
"""
Number of piano solo pieces:
  Old: 178923
  New: 178099
  Common: 177596
  Only in new: 509
   > ./mxl/1/11/QmbbQYi9qzja3G9ymsqLo3zdo3gsDNhfYPwodCH6TLcQ39.mxl 	0	['Drum group']
   > ./mxl/1/5/Qmb6XhSUsGMzLacW2FNnrJojcL3cB7B7v4BFw5V57R6rrw.mxl 	0	['Drum group']
   > ./mxl/1/35/QmboQHNEobnJHHb66e5NddkVE7Peb6widRnDXVipw3Ue9t.mxl 	0	['Drum group']
   > ./mxl/1/13/QmbcEBa5EpcpLb5T6BcUxcMH6B69PiANUCFHiaJX3tueTN.mxl 	0	['Drum group']
   > ./mxl/1/32/QmbMsdLfVtAtyyWdGccTABe9JUWk2zRF2LPWnTs9urvoEp.mxl 	0	['Drum group']
   > ./mxl/1/10/QmbAngpENoq52iiwb1LxBgcuvcuPsEykJNedidfVrHH58P.mxl 	0	['Soprano']
   > ./mxl/1/38/QmbqrBzsRMVapKGdpuUtLBL16X3ESoXfsPYirQ2N9qfvyi.mxl 	0	['Soprano']
  Only in old: 1337
   > ./mxl/1/11/QmbbUxwDmVsaQw6AUay5fAjKfpnfxJcpPPmiXWK1ximCsY.mxl 	6	['Piano']
   > ./mxl/1/11/QmbbH46ccvGJkoM3N2HMJiCrHzufPvDBKkTHEsPLXJPYZT.mxl 	54-63-79-89	['Piano']
   > ./mxl/1/30/QmbLKi3SRMF1N8UStdEUM6gFseidA94TvR3SQwsGS5DnFy.mxl 	0-42	['Piano']
   > ./mxl/1/30/QmbLqugeSVP2vEuqGBpBvpb8W4gVM1hsrAWhABWBbiLtMq.mxl 	6	['Piano']
   > ./mxl/1/5/Qmb6w2kdM337Lq6DTEtkz1B6d3LLZXZvenwEwfP6hN6Dra.mxl 	6	['Piano']
   > ./mxl/1/5/Qmb6nVD2GsiytZ5vwAPSBQDAx7wE81WwMkX5keEeuMtARC.mxl 	6	['Piano']
   > ./mxl/1/44/QmbtL11QkGvBg51f5Td9id55PRnbL8H1ywQ93u4poHBqnW.mxl 	0-0	['Piano']
  Common with single staff: 156764
   > ./mxl/1/11/QmbbhLwwmuKqA9MC1ULhojEMGxnxDobsZPJEo69N8nbgTw.mxl 	[1]
   > ./mxl/1/11/Qmbbh58jD1ZLT1vevNsnyEZkexnx2USTFz6FSHX3KUiAzC.mxl 	[1]
   > ./mxl/1/11/Qmbbmnu3Eg81nQ3WHnBFrAj5gsr3D13ysSQkBNcxWLTFHu.mxl 	[1]
   > ./mxl/1/11/QmbbMfooxpp22GzyuYhezebYmaeff5ksWYcxxZXmJpdwWw.mxl 	[1]
   > ./mxl/1/11/Qmbbp9arXoEvT9FRiwYiRRuEm6gHv9VVZgPhpiu37FNVxv.mxl 	[1]
   > ./mxl/1/11/Qmbb7u9yVnSszEMvXfWeZxRHwEGedrqEx4BfwoMKtWProt.mxl 	[1]
   > ./mxl/1/11/Qmbb1ywKwWtz2EojinWwv9Lu8Um7Q6h2FNsWmVmeetNfwz.mxl 	[1]
  Common with double staff: 20651
   > ./mxl/1/11/QmbbmCapzTHpsmd3eLRuefXJ94Ms7avxPBy7v6tXSxYDkX.mxl 	[2]
   > ./mxl/1/11/QmbbFEQzNihEnR2EvTumtWeYgcCWcqUvEeCWPCk5GZA7GQ.mxl 	[2]
   > ./mxl/1/11/QmbbVvE6fJsDaNNQQLS3xSEgiznfcHj9hvKcD7QZvM3Ed7.mxl 	[2]
   > ./mxl/1/11/Qmbbh7fYdg9E1nsQvCgj9Egv2aygYGEtfL7QSv6BL8VHwG.mxl 	[2]
   > ./mxl/1/11/QmbbNRWP3gr5ooxoEcNCNfyB7aBYw9k7CKBTD2Rvj4dp9k.mxl 	[2]
   > ./mxl/1/11/QmbbgLmmDrUM6z7SHaDAroq3euADMXhex7h1XCbvjGpwh3.mxl 	[2]
   > ./mxl/1/11/QmbbCoyosnMgyAsivgrBzw2TQVhvx6QvJP2zcwKJk6PMjZ.mxl 	[2]
  Common with more than two staves: 169
   > ./mxl/1/20/QmbFLVFe7EhmKceRcf5bwDkbtDtrN4koYGww2VXvNQURR1.mxl 	[3]
   > ./mxl/1/3/Qmb44MRU4pUW2FU4UWrpVLzfKabpK3uQChFgyJUQwX6vDU.mxl 	[3]
   > ./mxl/1/6/Qmb72dkFsqp5K7sn47ZNgdE1QCRbFMUe7vfF3YQjpUUWfh.mxl 	[3]
   > ./mxl/1/31/QmbmidSUocFqPzYBPUaauFruS1yUNw9gaHioi4HUTh1yKz.mxl 	[3]
   > ./mxl/1/41/QmbRnZDck2XUvpk1uFkpGQMhLcyWppYHZ9WRb1qtqPX55N.mxl 	[3]
   > ./mxl/1/2/Qmb3TdvE21azQCaKve5SGrA5zVr8aXS9Mynhz8f1R1bbWd.mxl 	[6]
   > ./mxl/1/2/Qmb3aXX9gRizw8Zm3BZ4tcCQmdAsa9XTwAFSNFTS6T55Y9.mxl 	[3]
"""


# %% Extract instruments directly in XML file to try a 3rd method
def extract_instruments_info_from_xml(base_path: Path, mxl_path: Path) -> list[dict]:
    """Parse MusicXML file, and extract: instrument name, midi channel and midi program."""
    if isinstance(mxl_path, float):
        return []
    try:
        xml = read_musicxml(base_path / mxl_path)
        root = etree.fromstring(xml.encode("utf-8"))
    except (lxml.etree.XMLSyntaxError, TypeError) as e:
        print(f"Error parsing {mxl_path}: {e}")
        return []
    instruments = []
    for part in root.findall(".//score-part"):
        instrument_name_el = part.find(".//instrument-name")
        instrument_name = instrument_name_el.text if instrument_name_el is not None else None
        midi_instrument_el = part.find(".//midi-instrument")
        if midi_instrument_el is not None:
            midi_channel_el = midi_instrument_el.find(".//midi-channel")
            midi_channel = midi_channel_el.text if midi_channel_el is not None else None
            midi_program_el = midi_instrument_el.find(".//midi-program")
            midi_program = midi_program_el.text if midi_program_el is not None else None
        else:
            midi_channel = None
            midi_program = None
        instruments.append(
            {
                "instrument_name": instrument_name,
                "midi_channel": midi_channel,
                "midi_program": midi_program,
            }
        )
    return instruments


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
