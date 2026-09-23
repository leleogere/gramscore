"""
This script reexport the MIDI files of the PDMX scores using MuseScore, allowing to better control
the output.
Note that to export WITHOUT repetitions, one must first export a MIDI file in the UI, unticking the
"Unroll repetitions" in the MIDI-export dialog, so that the preference is saved. There is currently
no way to specify this option via CLI (unfortunately).
"""

import json
import time
from pathlib import Path
import subprocess
import joblib
from tqdm import tqdm
import multiprocessing as mp

MUSESCORE_PATH = Path(
    "~/Applications/MuseScore-Studio-4.5.2.251141401-x86_64.AppImage"
).expanduser()

PDMX_PATH = Path("~/Documents/datasets/pdmx").expanduser()

MXL_PATH = PDMX_PATH / "mxl"

NEW_MID_PATH = PDMX_PATH / "mid_no_rep"


def create_job_list(mxl_dir: Path, new_mid_dir: Path) -> list[dict[str, Path]]:
    return [
        {
            "in": mxl_path,
            "out": new_mid_dir / mxl_path.relative_to(mxl_dir).with_suffix(".mid"),
        }
        for mxl_path in sorted(MXL_PATH.rglob("*.mxl"))
    ]


def create_output_structure(job_list: list[dict[str, Path]]):
    for job in job_list:
        job["out"].parent.mkdir(parents=True, exist_ok=True)


def split_list(lst: list, n: int):
    k = len(lst) // n
    return [
        lst[i * k + min(i, len(lst) % n) : (i + 1) * k + min(i + 1, len(lst) % n)] for i in range(n)
    ]


def export_job_list(
    job_list: list[dict[str, Path]], output_dir: Path, n_splits: int = 1
) -> list[Path]:
    str_job_list = [
        [{key: value.as_posix() for key, value in job.items()} for job in subjob_list]
        for subjob_list in split_list(job_list, n=n_splits)
    ]
    assert sum(len(jobs) for jobs in str_job_list) == len(job_list)
    output_paths = [output_dir / f"jobs_{i}.json" for i in range(1, n_splits + 1)]
    for output_path, sublist in zip(output_paths, str_job_list):
        with output_path.open("w") as f:
            json.dump(sublist, f)
    return output_paths


def run_musescore(job_list_location: Path):
    assert job_list_location.suffix == ".json"
    # print(
    command = [MUSESCORE_PATH, "-f", "-j", job_list_location.as_posix()]
    # print(command)
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process


def export_all(mxl_path: Path, new_mid_dir: Path):
    jobs = create_job_list(mxl_path, new_mid_dir)
    create_output_structure(jobs)
    jobs_directory = Path("/tmp/jobs")
    jobs_directory.mkdir(parents=True, exist_ok=True)
    n_jobs = 12
    jobs_locations = export_job_list(jobs, jobs_directory, n_splits=n_jobs)
    # joblib.Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
    #     joblib.delayed(run_musescore)(partial_job) for partial_job in jobs_locations
    # )
    processes = [run_musescore(sublist) for sublist in jobs_locations]
    monitor_jobs(new_mid_dir, total=len(jobs), processes=processes)
    for process in processes:
        process.wait()


def monitor_jobs(new_mid_dir: Path, total: int, processes: list[subprocess.Popen]):
    done = 0
    with tqdm(total=total) as pbar:
        while done != total:
            new_done = len(list(new_mid_dir.rglob("*.mid")))
            # print(f"{new_done = }")
            if new_done > done:
                pbar.update(new_done - done)
                done = new_done
            pbar.set_description(f"Processes return codes: {[p.poll() for p in processes]}")
            time.sleep(1)


if __name__ == '__main__':
    export_all(MXL_PATH, NEW_MID_PATH)
