"""
Realign the ASAP dataset after regenerating all MIDI files.
It uses the same alignment pipeline as F. Foscarin's code, most of the code is taken as it from
the notebooks `full_ann_pipeline.ipynb` and `create_json.ipynb`.
"""

import argparse
import json
import math
from importlib.resources import files
from pathlib import Path
from subprocess import PIPE
from subprocess import Popen
from typing import Any

import music21 as m21
import numpy as np
import pandas as pd
import pretty_midi as pm
from tqdm import tqdm

import scripts

tqdm.pandas()


def align_asap(asap_path: Path, nakamura_path: Path):
    metadata = pd.read_csv(asap_path / "metadata.csv")
    metadata = metadata[metadata.midi_score == "Bach/Fugue/bwv_846/midi_score.mid"]
    # metadata = metadata[metadata.midi_score == "Bach/Prelude/bwv_860/midi_score.mid"]
    # metadata = metadata[metadata.composer == "Bach"]
    # metadata = metadata.drop_duplicates(
    #     subset="midi_score",
    #     keep="first",
    # )  # OK temporarly as working only with midi_score

    # Generate the alignments with Nakamura's algorithm
    print("Generating alignments with Nakamura's algorithm")
    metadata.progress_apply(generate_nak_align, axis=1, args=(asap_path, nakamura_path))

    # Create the annotations files for the midi scores
    print("Creating annotations for the midi scores")
    quantified_metadata = metadata.drop_duplicates(subset="midi_score_annotations", keep="first")
    quantified_metadata.progress_apply(quantified_row_to_annotations, axis=1, args=(asap_path,))

    # Create the annotations files for the performances
    print("Creating annotations for the performances")
    metadata.progress_apply(perform_row_to_annotations, axis=1, args=(asap_path,))

    # Create the JSON annotations
    print("Creating JSON annotations")
    annotations = create_json(metadata, base_path=asap_path)
    with open(asap_path / "asap_annotations.json", "w") as f:
        json.dump(annotations, f)

    # Check dataset quality
    print("Checking dataset quality")
    check_dataset_quality(metadata, asap_path)


def check_dataset_quality(metadata: pd.DataFrame, asap_path: Path):
    #####################
    # Check MIDI scores #
    #####################
    quantified_df = metadata.drop_duplicates(subset="midi_score_annotations", keep="first")
    # Check early/late db/b
    quantified_df.progress_apply(
        lambda row: check_late_early_annot(
            asap_path / row["midi_score"],
            asap_path / row["midi_score_annotations"],
        ),
        axis=1,
    )
    # Check correct type
    quantified_df.progress_apply(
        lambda row: check_annotation_text(asap_path / row["midi_score_annotations"]),
        axis=1,
    )
    # Check correct ratio beats downbeats (according to the local time signature)
    quantified_df.progress_apply(
        lambda row: check_b_db_ratio(asap_path / row["midi_score_annotations"]),
        axis=1,
    )
    # Check if the number of db in annotations is the same of the (corrected) number of measures
    # in the score
    quantified_df.progress_apply(
        lambda row: same_number_of_measures_with_repetitions(
            row["xml_score"],
            asap_path / row["midi_score_annotations"],
            base_path=asap_path,
        ),
        axis=1,
    )

    ######################
    # Check performances #
    ######################
    # Check early/late db/b
    metadata.progress_apply(
        lambda row: check_late_early_annot(
            asap_path / row["midi_performance"],
            asap_path / row["performance_annotations"],
        ),
        axis=1,
    )
    # Check correct type
    metadata.progress_apply(
        lambda row: check_annotation_text(
            asap_path / row["performance_annotations"],
            allow_W_flag=False,
        ),
        axis=1,
    )
    # Check b,db correct ratio
    metadata.progress_apply(
        lambda row: check_b_db_ratio(asap_path / row["performance_annotations"]),
        axis=1,
    )
    # Check inverted annotations
    metadata.progress_apply(
        lambda row: check_inverted_annotations(asap_path / row["performance_annotations"]),
        axis=1,
    )
    # Check if performances and midi scores have the same number and type of annotations
    metadata.progress_apply(
        lambda row: midi_score_and_perf_aligned(
            asap_path / row["performance_annotations"],
            asap_path / row["midi_score_annotations"],
            verbose=True,
        ),
        axis=1,
    )

    ###################
    # Check all files #
    ###################
    for i, row in metadata.iterrows():
        fields_to_check = [
            "xml_score",
            "midi_score",
            "midi_performance",
            "performance_annotations",
            "midi_score_annotations",
        ]
        for f in fields_to_check:
            my_file = asap_path / row[f]
            if not my_file.exists():
                print(f"File {my_file} does not exist")


def create_json(dataframe: pd.DataFrame, base_path: Path) -> dict[str, Any]:
    asap_ann = {}
    for i, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        asap_ann[row["midi_performance"]] = {
            # fmt: off
            "performance_beats": get_beats_from_txt(base_path / row["performance_annotations"]),
            "performance_downbeats": get_downbeats_from_txt(base_path / row["performance_annotations"]),
            "performance_beats_type": get_beats_db_dict_from_txt(base_path / row["performance_annotations"]),
            "perf_time_signatures": get_ts_from_txt(base_path / row["performance_annotations"]),
            "perf_key_signatures": get_key_from_txt(base_path / row["performance_annotations"]),
            "midi_score_beats": get_beats_from_txt(base_path / row["midi_score_annotations"]),
            "midi_score_downbeats": get_downbeats_from_txt(base_path / row["midi_score_annotations"]),
            "midi_score_beats_type": get_beats_db_dict_from_txt(base_path / row["midi_score_annotations"]),
            "midi_score_time_signatures": get_ts_from_txt(base_path / row["midi_score_annotations"]),
            "midi_score_key_signatures": get_key_from_txt(base_path / row["midi_score_annotations"]),
            "downbeats_score_map": same_number_of_measures_with_repetitions(
                base_path / row["xml_score"],
                str(base_path / row["midi_score_annotations"]),
                base_path=base_path,
            ),
            "score_and_performance_aligned": midi_and_score_aligned(row, base_path),
            #         "manually_checked" : is_manually_checked(row)
            # fmt: on
        }
    return asap_ann


def quantified_row_to_annotations(row, base_path: Path):
    midi_data = pm.PrettyMIDI(str(base_path / row["midi_score"]))

    dbs = quant_downbeat_to_annotations_from_midi(midi_data)
    bs = quant_beat_to_annotations_from_midi(midi_data)
    anns = aggregate_annotations(bs, dbs)

    # Remove beats before/after first/last onset
    first_onset = midi_data.get_onsets()[0]
    last_onset = midi_data.get_onsets()[-1]
    anns = {
        time: ann for time, ann in anns.items() if first_onset - 0.025 <= time <= last_onset + 0.035
    }  # 35 ms accepted window

    beat_downbeat_to_text(anns, base_path / row["midi_score_annotations"])

    # Add TS changes
    ts_changes = align_ts_to_beat(row, anns, base_path)
    row2file_annotations(row, anns, ts_changes, base_path)

    # Add KS changes
    save_key_in_txt_annotations(row, base_path)


def quant_downbeat_to_annotations_from_midi(mididata):
    downbeats = mididata.get_downbeats()
    # the downbeat duration (in theory)
    db_len = downbeats[2] - downbeats[1]
    downbeat = {}
    for i, db in enumerate(downbeats[:-1]):
        assert db not in downbeat  # check if all values are uniques
        if math.isclose(downbeats[i + 1] - downbeats[i], db_len, rel_tol=1e-2):
            downbeat[db] = "db"
        else:
            downbeat[db] = "dbW"
    # Add last db
    if not downbeats[-1] in downbeat:
        downbeat[downbeats[-1]] = "db"
    # sort the dictionary just in case there are weird values
    ord_downbeat = {k: v for k, v in sorted(downbeat.items(), key=lambda item: item[1])}
    return ord_downbeat


def quant_beat_to_annotations_from_midi(mididata):
    beats = mididata.get_beats()
    # the beat duration (in theory)
    b_len = beats[40] - beats[39]  # multiple short beats in the beginning sometimes
    beat = {}
    for i, b in enumerate(beats[:-1]):
        assert b not in beat  # check if all values are uniques
        if math.isclose(beats[i + 1] - beats[i], b_len, rel_tol=1e-2):
            beat[b] = "b"
        else:
            beat[b] = "bW"
    # sort the dictionary just in case there are weird values
    ord_beat = {k: v for k, v in sorted(beat.items(), key=lambda item: item[1])}
    return ord_beat


def aggregate_annotations(beat, downbeat):
    # merge the downbeat and the beat, with downbeat overwriting beat information
    annotations = {**beat, **downbeat}
    return annotations


def beat_downbeat_to_text(annotations, out_path: Path):
    file_content = ""
    for a in annotations:
        file_content += str(a) + "\t" + str(a) + "\t" + annotations[a] + "\n"
    with open(out_path, "w") as f:
        f.write(file_content)


def score2ts_changes(score_path, remove_duplicates=True):
    score = m21.converter.parse(score_path)
    ts_changes = [
        str(ts.numerator) + "/" + str(ts.denominator)
        for ts in score.parts[0].recurse().getElementsByClass(m21.meter.TimeSignature)
    ]
    if remove_duplicates:
        # remove the duplicates
        ts_changes = [
            ts for i, ts in enumerate(ts_changes) if i == 0 or (i != 0 and ts_changes[i - 1] != ts)
        ]
    return ts_changes


def midi2ts_changes(midi_path, remove_duplicates=True):
    midi = pm.PrettyMIDI(midi_path)
    ts_changes = [
        (ts.time, str(ts.numerator) + "/" + str(ts.denominator))
        for ts in midi.time_signature_changes
    ]
    if remove_duplicates:
        # remove the duplicates
        ts_changes = [
            ts
            for i, ts in enumerate(ts_changes)
            if i == 0 or (i != 0 and ts_changes[i - 1][1] != ts[1])
        ]
    return ts_changes


def align_score_ts_with_midi_ts(score_path, midi_path):
    midi_ts_changes = midi2ts_changes(midi_path)
    score_ts_changes = score2ts_changes(score_path)
    out = []
    for si, sts in enumerate(score_ts_changes):
        found_at_index = None
        for mi, mts in enumerate(midi_ts_changes):
            if sts == mts[1]:
                out.append((sts, mts[0]))
                found_at_index = mi
                break
        if found_at_index is None:
            raise (ValueError("The algorithm of ts aligning is not working"))
        if found_at_index < len(midi_ts_changes):
            midi_ts_changes = midi_ts_changes[found_at_index + 1 :]
    return out


def row_to_ts_align(row, base_path: Path):
    try:
        return align_score_ts_with_midi_ts(
            str(base_path / row["xml_score"]), str(base_path / row["midi_score"])
        )
    except Exception as e:
        print("Processing piece", row["midi_score"], "..................Exception!!")
        print(e)
        return None


def align_ts_to_beat(row, anns, base_path: Path):
    # print("Aligning opus", row.midi_score)

    ann_list = list(anns.items())
    downbeats = [b[0] for b in ann_list if b[1] == "db"]

    # compute ts_changes
    ts_changes = row_to_ts_align(row, base_path=base_path)
    # align ts
    aligned_ts = []
    for ts in ts_changes:
        # find beat in the window
        close_db = [db for db in downbeats if (db >= ts[1] - 0.0175) and (db <= ts[1] + 0.0175)]
        if ts[1] == 0:  # initial time signature, can not coincide with downbeats, so don't align it
            aligned_ts.append(ts)
        elif len(close_db) != 1:  # if no close note found (beat on a rest)
            print("Some problems for ts", ts)
        else:
            aligned_ts.append((ts[0], close_db[0]))  # align to the close downbeat
            shift = np.abs(ts[1] - close_db[0])
            if shift > 0:
                print("ts", ts, "moved of ", shift)

    # move the first time signature to the first downbeat position
    first_db = downbeats[0]
    first_ts = aligned_ts[0][1]
    if first_db != first_ts:
        aligned_ts[0] = (aligned_ts[0][0], first_db)

    return aligned_ts


def row2file_annotations(row, anns, ts_changes, base_path: Path):
    # in case beats are missing, exit
    path = base_path / row["midi_score_annotations"]
    # produce the txt
    file_content = ""
    for a in list(anns.items()):
        file_content += str(a[0]) + "\t" + str(a[0]) + "\t" + a[1]
        # add the ts change if needed
        if a[0] in [ts[1] for ts in ts_changes]:
            file_content += "," + str([ts[0] for ts in ts_changes if ts[1] == a[0]][0])
        file_content += "\n"
    with open(path, "w") as f:
        f.write(file_content)


def score2ks_changes(score_path: Path, remove_duplicates: bool = True, enharmonic=True):
    score = m21.converter.parse(score_path)
    #     keys = [ ks.sharps for ks in score.parts[0].recurse().getElementsByClass(m21.key.KeySignature)]
    keys = [
        (ks.asKey(mode='major').tonic.pitchClass, ks.measureNumber, ks.sharps)
        for ks in score.parts[0].recurse().getElementsByClass(m21.key.KeySignature)
    ]
    if remove_duplicates:
        # remove the duplicates
        if enharmonic:  # we condider only the key as an integer [0,11]
            keys = [
                ks for i, ks in enumerate(keys) if i == 0 or (i != 0 and keys[i - 1][0] != ks[0])
            ]
        else:  # we consider the number of sharps and flats
            keys = [
                ks for i, ks in enumerate(keys) if i == 0 or (i != 0 and keys[i - 1][2] != ks[2])
            ]
    return keys


def midi2ks_changes(midi_path: Path, remove_duplicates: bool = True):
    midi = pm.PrettyMIDI(str(midi_path))
    # 0 is C Major, 12 is C minor.
    keys = [(k.key_number, k.time) for k in midi.key_signature_changes]
    if remove_duplicates:
        # remove the duplicates
        keys = [ks for i, ks in enumerate(keys) if i == 0 or (i != 0 and keys[i - 1][0] != ks[0])]
    return keys


def save_key_in_txt_annotations(row, base_path: Path):
    # get the keys informations
    score_keys = score2ks_changes(base_path / row["xml_score"])
    midi_keys = midi2ks_changes(base_path / row["midi_score"])
    #     assert([sk[0] for sk in score_keys]  == [mk[0] for mk in midi_keys]  ) #check if they coincide with the score
    if [sk[0] for sk in score_keys] != [mk[0] for mk in midi_keys]:
        print("Not working for", row.score_midi, score_keys, midi_keys)
        return

    # get the annotations file
    quant_ann_df = pd.read_csv(
        base_path / row["midi_score_annotations"],
        header=None,
        names=["time", "time2", "type"],
        sep='\t',
    )

    ks_out_dict = {}
    # check if there is a db close to every key change time (except for the first)
    for mk, sk in zip(midi_keys, score_keys):
        if mk[1] == 0:
            # it's the first of the piece, append it to the first annotation
            close_ann = [r["time"] for i, r in quant_ann_df.iterrows()][0]
            ks_out_dict[close_ann] = sk[2]
        else:
            # find the close beat
            #             db_list = [row["time"] for i,row in quant_ann_df.iterrows() if row["type"].split(",")[0]=="b" or ]
            close_anns = [
                r["time"]
                for i, r in quant_ann_df.iterrows()
                if (r["time"] >= mk[1] - 0.0175) and (r["time"] <= mk[1] + 0.0175)
            ]
            if len(close_anns) != 1:
                # we take the first annotation on the right
                close_ann = [r["time"] for i, r in quant_ann_df.iterrows() if (r["time"] >= mk[1])][
                    0
                ]
            else:
                close_ann = close_anns[0]
            ks_out_dict[close_ann] = sk[2]

    # now write the file
    file_content = ""
    counter_written = 0
    for i, r in quant_ann_df.iterrows():
        if r["time"] in ks_out_dict.keys():
            if len(r["type"].split(",")) == 1:  # no time signature
                file_content += (
                    str(r["time"])
                    + "\t"
                    + str(r["time"])
                    + "\t"
                    + str(r["type"])
                    + ",,"
                    + str(ks_out_dict[r["time"]])
                    + "\n"
                )
            elif len(r["type"].split(",")) == 2:  # time signature
                file_content += (
                    str(r["time"])
                    + "\t"
                    + str(r["time"])
                    + "\t"
                    + str(r["type"])
                    + ","
                    + str(ks_out_dict[r["time"]])
                    + "\n"
                )
            else:
                raise Exception("Bad annotations type")
        else:
            file_content += str(r["time"]) + "\t" + str(r["time"]) + "\t" + str(r["type"]) + "\n"
    with open(base_path / row["midi_score_annotations"], "w") as f:
        f.write(file_content)

    return


def generate_nak_align(row, base_path: Path, nakamura_path: Path):
    alignment_script = Path(files(scripts).joinpath("align.sh"))
    common_path = (base_path / row["midi_score"]).parent
    assert common_path == (base_path / row["midi_performance"]).parent
    midi_score_stem = row["midi_score"].split("/")[-1][:-4]  # No ext bc issue with midi2pianoroll
    midi_performance_stem = row["midi_performance"].split("/")[-1][:-4]
    p = Popen(
        [
            str(alignment_script),
            midi_score_stem,
            midi_performance_stem,
            str(common_path),
            str(nakamura_path),
        ],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        cwd=str((base_path / row["midi_score"]).parent),
    )
    # print(" ".join(p.args))
    output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    rc = p.returncode
    print(output.decode(), err.decode())


def cleaned_m2m_df(midi2midi_path):
    # read the nakamura output file and delete useless lines
    m2mdf = pd.read_csv(midi2midi_path, sep="\t", skiprows=1, header=None, index_col=False)
    m2mdf.columns = [
        "alignID",
        "alignOntime",
        "alignSitch",
        "alignPitch",
        "alignOnvel",
        "refID",
        "refOntime",
        "refSitch",
        "refPitch",
        "refOnvel",
        "empty",
    ]
    # drop the last column "empty"
    m2mdf = m2mdf.drop(columns=["empty"])
    # drop the rows where the alignOntime is -1 (note not existing in both files midi)
    return m2mdf[m2mdf["alignOntime"] != -1]


def generate_performance_annotations(m2m_df, cl_quant_ann):
    # create the list with the time of all annotations
    quant_ann_time_list = cl_quant_ann["time"].tolist()
    quant_ann_type_list = cl_quant_ann["type"].tolist()
    perf_ann_list = []
    missing_list = []
    for i, at in enumerate(quant_ann_time_list):
        p_ons = list(
            m2m_df[(m2m_df["refOntime"] > at - 0.020) & (m2m_df["refOntime"] < at + 0.020)][
                "alignOntime"
            ]
        )  # 40 ms accepted window
        if len(p_ons) == 0:  # no note matching the ann position in the performance
            perf_ann_list.append(None)
            #             print("missing corrispondency at quant time", at)
            missing_list.append(("missing", at))
        else:
            perf_ann_list.append(np.median(p_ons))  # median against outliers

    # clean the values that are smaller than the value before
    for ind, at in enumerate(perf_ann_list[1:]):
        if (
            (ind > 0)
            and (at is not None)
            and (perf_ann_list[ind - 1] is not None)
            and (at < perf_ann_list[ind - 1])
        ):
            #             print("Unexpected value",at,"smaller than the previous",perf_ann_list[ind-1])
            perf_ann_list[ind] = None
            missing_list.append(("smaller", at))

    # save the index for the none values
    none_indices = [index for index, value in enumerate(perf_ann_list) if value is None]

    # check if there are no None at the end (there should no, because of the manual corrections)
    if perf_ann_list[-1] is None:
        missing_list.append(("None at end", 0))
    # interpolate to fill the None values in the middle
    filled_at = pd.Series(perf_ann_list).interpolate().tolist()
    # fill the values at the beginning
    ind = 0
    while math.isnan(filled_at[ind]):  # find the number of None at the beginning
        ind += 1
    for i in range(ind)[::-1]:  # fill, starting from the later one
        proposed_value = filled_at[i + 1] - (filled_at[i + 2] - filled_at[i + 1])
        if proposed_value < 0:  # cap at 0
            proposed_value = 0
        filled_at[i] = proposed_value

    # associate the type of annotation
    assert len(quant_ann_type_list) == len(filled_at)
    perd_ann_dict = dict(zip(filled_at, quant_ann_type_list))
    # add the Warning for the None values
    warnings = False
    for val in none_indices:
        perd_ann_dict[filled_at[val]] += "W"
        warnings = True

    return perd_ann_dict, missing_list


def perform_row_to_annotations(row, base_path: Path):
    ann_quant_df = pd.read_csv(
        base_path / row["midi_score_annotations"],
        header=None,
        names=["time", "time2", "type"],
        sep='\t',
    )
    # load the m2m alignment
    # corresp_path = base_path / (row["midi_performance"][:-4] + "_corresp.txt")
    # if not corresp_path.exists():
    #     print(f"File {corresp_path} not found")
    #     return None
    m2m = cleaned_m2m_df(base_path / (row["midi_performance"][:-4] + "_corresp.txt"))
    # generate the annotations
    ann_dict, missing_list = generate_performance_annotations(m2m, ann_quant_df)
    # write the annotations to txt
    beat_downbeat_to_text(ann_dict, base_path / (row["midi_performance"][:-4] + "_annotations.txt"))
    # write the audio
    # annotations_to_audio(str(Path(BASE_PATH,row["midi_performance"][:-4]+".mid")),ann_dict,Path(BASE_PATH,row["midi_performance"][:-4] + "_click.wav"))
    return missing_list


def ts2n_of_beats(ts):
    """Get the number of beats in a given time signature according to music theory

    Arguments:
        ts {string} -- the time signature [numerator]/[denominator]

    Raises:
        TypeError: if the ts string is not in the correct format

    Returns:
        int -- the number of beats
    """
    num = int(ts.split("/")[0])
    if num == 1:
        return 1
    elif num == 2 or num == 6:  # duple meter
        return 2
    elif num == 3 or num == 9:  # triple meter
        return 3
    elif num == 4 or num == 12:  # quadruple meter
        return 4
    elif num == 5:
        return 5
    elif num == 24:  # my choice
        return 8
    else:  # complex meter
        raise TypeError("The meter is not supported")


def check_annotation_text(annotations_path, allow_W_flag=False):
    """Check the type of annotation for a text file (name, key signature, time signature) and print warnings if there are problems

    Arguments:
        annotations_path {string} -- the path of the annotation text file
        allow_W_flag {boolean} -- ignore the W flag used for annotation cleaning in name
    """
    ann_df = pd.read_csv(annotations_path, header=None, names=["time", "time2", "type"], sep='\t')
    allowed_names = ["db", "b", "bR"]

    warnings = []
    for i, r in ann_df.iterrows():
        # check the ann type (db,b or bR)
        if allow_W_flag:
            name = (
                r["type"].split(",")[0]
                if r["type"].split(",")[0][-1] != "W"
                else r["type"].split(",")[0][:-1]
            )
        else:
            name = r["type"].split(",")[0]
        if not name in allowed_names:
            warnings.append("Wrong name " + str(r["type"].split(",")[0]) + " at " + str(r["time"]))
        # check the time signature
        if len(r["type"].split(",")) == 2 or (
            len(r["type"].split(",")) == 3 and r["type"].split(",")[1] != ""
        ):
            try:
                ts2n_of_beats(r["type"].split(",")[1])
            except:
                warnings.append("Wrong ts" + str(r["type"].split(",")[1]))
        # check the key signature
        if len(r["type"].split(",")) == 3:
            if not int(r["type"].split(",")[2]) in list(range(-7, 8)):
                warnings.append("Wrong ks" + str(r["type"].split(",")[2]))

    if len(warnings) > 0:
        print("Annotation text problems in", annotations_path)
        print(warnings)


def check_b_db_ratio(annotations_path):
    """Check if the ratio beats, downbeats is correct according to the time signature and print warnings if there are problems
    The ration can be different of what expected if at least one "bR" is in the measure

    Arguments:
        annotations_path {string} -- the path of the annotation text file
    """
    warnings = []

    ann_df = pd.read_csv(annotations_path, header=None, names=["time", "time2", "type"], sep='\t')
    ann_df = ann_df.sort_values(by=['time'])
    # clean the W flag (used for annotations cleaning)
    merged_annotations = [
        (row["time"], row["type"]) if row["type"][-1] != "W" else (row["time"], row["type"][:-1])
        for i, row in ann_df.iterrows()
    ]

    try:
        # check if the correct number of beat every downbeat
        counter = 1
        pickup = True
        rubato = False
        for ann in merged_annotations:
            # set the eventual rubato flag
            if ann[1].split(",")[0][-1] == "R":
                rubato = True
            # start checking the beats and downbeats
            if ann[1].split(",")[0] == "db":
                if pickup:  # first db of the non pickup, don't check
                    pass
                elif rubato:  # first db after rubato, don't check
                    pass
                else:  # we check the counter
                    if counter != number_of_beats:
                        warnings.append("Checking ratio for opus " + str(annotations_path))
                        warnings.append(
                            "Wrong number of beats: ("
                            + str(counter)
                            + ") in ann"
                            + str(ann[0] // 60)
                            + "m"
                            + str(ann[0] % 60)
                            + ". Expecting"
                            + str(number_of_beats)
                        )
                pickup = False
                rubato = False
                counter = 1
            elif ann[1][0] == "b":  # count "b" and "bR"
                counter += 1
                if (not pickup) and (not rubato):
                    if counter > number_of_beats:
                        warnings.append("Checking ratio for opus" + str(annotations_path))
                        warnings.append(
                            "Wrong number of beats: ("
                            + str(counter)
                            + ") in ann"
                            + str(ann[0] // 60)
                            + "m"
                            + str(ann[0] % 60)
                            + ". Expecting"
                            + str(number_of_beats)
                        )

            else:
                counter = 1
                warnings.append("Wrong annotation kind", ann[1], "in time", ann[0])
            # set the time signature and number of beats
            if len(ann[1].split(",")) > 1 and ann[1].split(",")[1] != "":
                number_of_beats = ts2n_of_beats(ann[1].split(",")[1])

        if len(warnings) > 0:
            print("Beats downbeats ratio problems in", annotations_path)
            print(warnings)
    except:
        print("Exception for", annotations_path)


def check_inverted_annotations(annotations_path):
    "Check if annotations are saved in chronological order."
    ann_df = pd.read_csv(annotations_path, header=None, names=["time", "time2", "type"], sep='\t')
    for i, row in ann_df.iterrows():
        if row["time"] != row["time2"]:
            print("Time different from time2 in", annotations_path)
    time_list = ann_df["time"]
    for i, t in enumerate(time_list):
        if i != 0:
            if t < time_list[i - 1]:
                print("Inverted annotations for", annotations_path, "at time", t // 60, "m", t % 60)


exception_dict = {
    "Beethoven/Piano_Sonatas/29-2/xml_score.musicxml": {112: [113]},
    "Beethoven/Piano_Sonatas/29-4/xml_score.musicxml": {0: [1], 2: [3, 4], 12: [13, 14, 15]},
    "Beethoven/Piano_Sonatas/30-1/xml_score.musicxml": {15: [16], 66: [67]},
    "Beethoven/Piano_Sonatas/31-3_4/xml_score.musicxml": {4: [5, 6], 7: [8, 9]},
    "Haydn/Keyboard_Sonatas/49-1/xml_score.musicxml": {131: [132]},
    "Liszt/Gran_Etudes_de_Paganini/2_La_campanella/xml_score.musicxml": {97: [98], 99: [100]},
    "Liszt/Mephisto_Waltz/xml_score.musicxml": {857: [858], 198: [199], 808: [809, 910]},
    "Liszt/Transcendental_Etudes/4/xml_score.musicxml": {23: [24, 25], 56: [57, 58, 59]},
    "Liszt/Transcendental_Etudes/9/xml_score.musicxml": {45: [46], 75: [76]},
    "Mozart/Fantasie_475/xml_score.musicxml": {84: [85]},
    "Schumann/Kreisleriana/2/xml_score.musicxml": {38: [39], 57: [58], 96: [97]},
}

repetition_not_working = {
    "Beethoven/Piano_Sonatas/11-3/xml_score.musicxml": list(range(0, 9)) * 2
    + list(range(9, 32)) * 2
    + list(range(32, 41)) * 2
    + list(range(41, 50))
    + list(range(41, 49))
    + [50]
    + list(range(0, 32)),
    "Beethoven/Piano_Sonatas/28-2/xml_score.musicxml": list(range(0, 9))
    + list(range(1, 8))
    + list(range(9, 56))
    + list(range(13, 55))
    + [56, 57]
    + list(range(58, 68)) * 2
    + list(range(68, 98))
    + list(range(1, 8))
    + list(range(9, 55))
    + [56],
    "Beethoven/Piano_Sonatas/7-3/xml_score.musicxml": list(range(0, 17)) * 2
    + list(range(17, 56)) * 2
    + list(range(56, 89))
    + list(range(0, 56)),
}


def same_number_of_measures_with_repetitions(
    score_xml_path, quantized_annotations_path, base_path="./"
):
    """Check if the number of db in the annotations is the same of the (corrected) number of measures in the xml score.
    Many correction has to be done to the number of measure in the scores.
    In some extreme cases, the correction are hardcoded in exception_dict (connected measures) and repetition_not_working (dc al fine music21 exception).

    Arguments:
        score_xml_path {string} -- path of the xml score
        quantized_annotations_path {string} -- path of the midi score annotations

    Returns:
        [list] -- the corresponding measure number for each db in annotation
    """
    complete_score_path = str(Path(base_path, score_xml_path))
    # compute the measure map in the score
    score = m21.converter.parse(complete_score_path)
    # consider both hands because of some durations problems some measures has duration 0
    score_measures_r = score.parts[0].recurse().getElementsByClass(m21.stream.Measure)
    score_measures_l = score.parts[1].recurse().getElementsByClass(m21.stream.Measure)
    if len(score.parts[0].recurse().getElementsByClass(m21.repeat.RepeatMark)) == 0:
        # no repetitions, we are going linearly
        m_map = list(range(len(score_measures_r)))
    elif score_xml_path in repetition_not_working.keys():
        # in case of dc al fine, music21 does not work. We did it manually
        m_map = repetition_not_working[score_xml_path]
    else:
        try:
            e = m21.repeat.Expander(streamObj=score.parts[0])
            m_map = e.measureMap()
        except:
            print("Processing", score_xml_path)
            print("Expansion Exception")
            return np.nan

    # consider pickup measure in the score
    if score_measures_r[0].paddingLeft > 0 or score_measures_l[0].paddingLeft > 0:
        m_map = m_map[1:]
    else:  # or if there is a pause as first event
        some_notes_on_first_db = False
        for part in score.parts:
            measure = part.recurse().getElementsByClass(m21.stream.Measure)[0]
            notes_on_db = [n for n in measure.recurse().notes if n.beat == 1]
            if len(notes_on_db) != 0:
                some_notes_on_first_db = True
                break
        if not some_notes_on_first_db:
            m_map = m_map[1:]

    score_measures_n = len(m_map)

    # compute number of measures in the midi score (e.g. in the midi score annotations)
    quant_ann_df = pd.read_csv(
        quantized_annotations_path, header=None, names=["time", "time2", "type"], sep='\t'
    )
    len_ann = len(
        [
            db_tp.split(",")[0]
            for db_tp in quant_ann_df["type"].tolist()
            if db_tp.split(",")[0] == "db"
        ]
    )

    if (
        len_ann == score_measures_n
    ):  # if it's already aligned, finish here to spare computation time
        return list(m_map)
    else:  # consider the splitted measures
        new_map = []
        i = 0
        while i < len(m_map):
            # first check if the measure is an exception
            if (not exception_dict.get(score_xml_path) is None) and (
                m_map[i] in exception_dict[score_xml_path].keys()
            ):
                connected_to = exception_dict[score_xml_path][m_map[i]]
                if len(connected_to) == 1:
                    new_map.append(str(m_map[i]) + "-" + str(connected_to[0]))
                    i += 2
                elif len(connected_to) == 2:
                    new_map.append(
                        str(m_map[i]) + "-" + str(connected_to[0]) + "-" + str(connected_to[1])
                    )
                    i += 3
                elif len(connected_to) == 3:
                    new_map.append(
                        str(m_map[i])
                        + "-"
                        + str(connected_to[0])
                        + "-"
                        + str(connected_to[1])
                        + "-"
                        + str(connected_to[2])
                    )
                    i += 4
                else:
                    print("connected too with too many elements")
            else:
                measure_r = score_measures_r[m_map[i]]
                measure_l = score_measures_l[m_map[i]]
                m_dur = max(
                    [measure_r.duration.quarterLength, measure_l.duration.quarterLength]
                )  # actual lenght
                m_ts_dur = max(
                    [measure_r.barDuration.quarterLength, measure_l.barDuration.quarterLength]
                )  # lenght from the ts
                if m_ts_dur == m_dur:
                    new_map.append(m_map[i])
                    i += 1
                elif (m_ts_dur > m_dur) and i != len(m_map) - 1:
                    next_measure_r = score_measures_r[m_map[i + 1]]
                    next_measure_l = score_measures_l[m_map[i + 1]]
                    next_m_dur = max(
                        [
                            next_measure_r.duration.quarterLength,
                            next_measure_l.duration.quarterLength,
                        ]
                    )
                    next_m_ts_dur = max(
                        [
                            next_measure_r.barDuration.quarterLength,
                            next_measure_l.barDuration.quarterLength,
                        ]
                    )
                    #                 if (m_dur + next_m_dur == m_ts_dur) and (next_m_ts_dur == m_ts_dur) : #two splitted measures
                    if (m_dur + next_m_dur == m_ts_dur) and (
                        next_m_dur < next_m_ts_dur
                    ):  # two splitted measures, good also if the tempo change if the duration is correct
                        new_map.append(str(m_map[i]) + "-" + str(m_map[i + 1]))
                        i += 2
                    else:
                        new_map.append(m_map[i])
                        i += 1
                else:
                    new_map.append(m_map[i])
                    i += 1

    # consider empty measures at the end of the score or measures with tied chords
    end_index = 0
    found_note = False
    while not found_note:
        for part in score.parts:
            notes = list(
                part.recurse()
                .getElementsByClass(m21.stream.Measure)[new_map[-end_index - 1]]
                .recurse()
                .notes
            )
            if len(notes) != 0:
                for n in notes:
                    if n.isChord:
                        for note in n.notes:
                            if note.tie is None or note.tie.type == 'start':
                                found_note = True
                                break
                        if found_note:
                            break
                    else:  # it's a note
                        if n.tie is None or n.tie.type == 'start':
                            found_note = True
                            break
        if not found_note:
            end_index += 1
    if end_index > 0:
        new_map = new_map[:-end_index]

    if len_ann == len(new_map):
        #         print("Equal")
        return list(new_map)
    else:
        print("Different", score_xml_path)
        print(
            "Corrected number of measures:", len(new_map), ". Number of db in annotations:", len_ann
        )
        return np.nan


def check_late_early_annot(midi_path: Path, annotations_path: Path):
    """Check if the first annotation time is not earlier than the first note and the last annotation time is not later than last note. Print a warning otherwise

    Arguments:
        midi_path {string} -- path of midi file
        annotationsa_path {string} -- path of the annotations corresponding to the same midi file
    """
    # load annotation file
    ann_df = pd.read_csv(annotations_path, header=None, names=["time", "time2", "type"], sep='\t')
    first_ann = sorted(ann_df["time"].tolist())[0]
    last_ann = sorted(ann_df["time"].tolist())[-1]

    # load midi file
    midi = pm.PrettyMIDI(str(midi_path))
    # extract the first note position
    note_ons = midi.get_onsets()
    first_ons = note_ons[0]
    last_ons = note_ons[-1]

    warnings = []
    # check if the the first annotation is not before the first onset
    if first_ann < first_ons - 0.035:  # 35 ms accepted window
        warnings.append("Wrong first beat at time " + str(first_ann))

    # check if the the last annotation is not after the last onset
    if last_ann > last_ons + 0.035:  # 35 ms accepted window
        warnings.append("Wrong last beat at time " + str(last_ann))

    if len(warnings) > 0:
        print("Early late annotations in", annotations_path)
        print(warnings)


def get_beats_from_txt(ann_path):
    """Get the beats time from the text annotations

    Arguments:
        ann_path {string} -- the path of the text annotations

    Returns:
        [list] -- a list of beat onsets
    """
    ann_df = pd.read_csv(Path(ann_path), header=None, names=["time", "time2", "type"], sep='\t')
    return ann_df["time"].tolist()


def get_downbeats_from_txt(ann_path):
    """Get the downbeats time from the text annotations

    Arguments:
        ann_path {string} -- the path of the text annotations

    Returns:
        [list] -- a list of downbeat onsets
    """
    ann_df = pd.read_csv(Path(ann_path), header=None, names=["time", "time2", "type"], sep='\t')
    downbeats = [a["time"] for i, a in ann_df.iterrows() if a["type"].split(",")[0] == "db"]
    return downbeats


def get_beats_db_dict_from_txt(ann_path):
    """Get the position of beats and downbeats as dictionary from the text annotations

    Arguments:
        ann_path {string} -- the path of the text annotations

    Returns:
        [dict] -- a dictionary where keys are the time (string format) and the labels are either "db","b", or "bR"
    """
    ann_df = pd.read_csv(Path(ann_path), header=None, names=["time", "time2", "type"], sep='\t')
    out_dict = {str(a["time"]): a["type"].split(",")[0] for i, a in ann_df.iterrows()}
    return out_dict


def get_key_from_txt(ann_path):
    print(ann_path)
    ann_df = pd.read_csv(Path(ann_path), header=None, names=["time", "time2", "type"], sep='\t')
    keys = {}
    for i, r in ann_df.iterrows():
        if len(r["type"].split(",")) == 3:
            number_of_sharps = int(r["type"].split(",")[2])
            key_number = key_number_from_number_of_sharps(number_of_sharps)
            keys[str(r["time"])] = [key_number, number_of_sharps]
    return keys


def midi_score_and_perf_aligned(perf_annotations_path, midi_score_annotations_path, verbose=False):
    """Check if the performance and the midi score have the same number and type of annotations

    Arguments:
        perf_annotations_path {string} -- the path of the text annotations of the performance
        midi_score_annotations_path {string} -- the path of the text annotations of the midi score

    Returns:
        boolean -- True if performance and midi score have the same annotations, false otherwise
    """
    midi_score_anns_df = pd.read_csv(
        Path(midi_score_annotations_path), header=None, names=["time", "time2", "type"], sep='\t'
    ).sort_values(by=['time'])
    perf_anns_df = pd.read_csv(
        Path(perf_annotations_path), header=None, names=["time", "time2", "type"], sep='\t'
    ).sort_values(by=['time'])
    midi_score_anns_type = [
        r["type"] if r["type"][-1] != "W" else r["type"][:-1]
        for i, r in midi_score_anns_df.iterrows()
    ]
    perf_anns_type = [
        r["type"] if r["type"][-1] != "W" else r["type"][:-1] for i, r in perf_anns_df.iterrows()
    ]
    if midi_score_anns_df.shape[0] != perf_anns_df.shape[0]:
        if verbose:
            print(
                "Different length of annotations for {}: {} (ms) vs {} (perf) ".format(
                    perf_annotations_path, len(midi_score_anns_type), len(perf_anns_type)
                )
            )
        return False
    elif midi_score_anns_type == perf_anns_type:
        return True
    else:
        if verbose:
            for ms, ms_time, perf, perf_time in zip(
                midi_score_anns_type,
                midi_score_anns_df["time"].tolist(),
                perf_anns_type,
                perf_anns_df["time"].tolist(),
            ):
                if ms != perf:
                    print(
                        "Different for perf {} at time {},{} : {},{}".format(
                            perf_annotations_path, ms_time, perf_time, ms, perf
                        )
                    )
        return False


def files_exist(row, base_path):
    fields_to_check = [
        'xml_score',
        'midi_score',
        'midi_performance',
        'performance_annotations',
        'midi_score_annotations',
    ]
    for f in fields_to_check:
        my_file = Path(base_path, row[f])
        if not my_file.is_file():
            print("File not found", my_file)


def xmlscore_parsable_music21(score_xml_path):
    """Check if we can parse the score with music21"""
    try:
        score = m21.converter.parse(score_xml_path)
    except:
        print("Problems parsing the xml score", score_xml_path)


def is_manually_checked(row):
    # check based on the audio click files
    if Path(BASE_PATH, row["midi_performance"][:-4] + "_click.wav").exists():
        return False
    else:
        return True


def key_number_from_number_of_sharps(nos):
    if nos >= 0:
        return int((nos * 7) % 12)
    else:
        return int((np.abs(nos) * 5) % 12)


def get_ts_from_txt(ann_path):
    ann_df = pd.read_csv(ann_path, header=None, names=["time", "time2", "type"], sep='\t')
    tss = {}
    for i, r in ann_df.iterrows():
        if len(r["type"].split(",")) > 1 and r["type"].split(",")[1] != "":
            ts = r["type"].split(",")[1]
            beat_n = int(ts2n_of_beats(ts))
            tss[str(r["time"])] = [ts, beat_n]
    return tss


def midi_and_score_aligned(row, base_path: Path):
    return len(get_beats_from_txt(base_path / row["performance_annotations"])) == len(
        get_beats_from_txt(base_path / row["midi_score_annotations"])
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Align the ASAP dataset")
    # fmt: off
    parser.add_argument("asap_dir", type=Path, help="Path to the ASAP dataset directory")
    parser.add_argument("--musescore", type=Path, default="musescore", help="Path to the MuseScore executable")
    parser.add_argument("--nakamura", type=Path, required=True, help="Path to the Nakamura alignment algorithm directory")
    # fmt: on

    args = parser.parse_args()

    align_asap(asap_path=args.asap_dir, nakamura_path=args.nakamura)
