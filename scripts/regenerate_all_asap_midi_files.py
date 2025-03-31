"""
This script regenerates the `midi_score.mid` files for all ASAP entries in the metadata.
It looks for every `xml_score.musicxml` file in the provided directory, and converts it to MIDI
using MuseScore.
~ 5 minutes with 8 parallel processes.
"""

import argparse
from pathlib import Path
import subprocess


def generate_midi_subprocess(
    score_path: Path,
    musescore_path: Path,
    overwrite: bool = False,
    dry_run: bool = False,
):
    """
    Create a subprocess to generate a MIDI file from a MusicXML file using MuseScore.
    Run the process in background and return the process object.
    """
    midi_path = score_path.with_name("midi_score.mid")
    if midi_path.exists():
        if not overwrite:
            print(f"Already existing: {midi_path}")
            return None
        else:
            print(f"Overwriting: {midi_path}")
    cmd = [str(musescore_path), "-o", str(midi_path), str(score_path)]
    if dry_run:
        print(" ".join(cmd))
        return None
    else:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
        )


def generate_all_midi_files(
    asap_dir: Path,
    musescore: Path,
    overwrite: bool = False,
    dry_run: bool = False,
    njobs: int = 1,
):
    """
    Generate all the processes, with a maximum of `n_process` processes running in parallel.
    It will wait for one process to finish before starting a new one.
    """
    processes = []
    scores = list(asap_dir.rglob("xml_score.musicxml"))
    for musicxml_file in scores:
        process = generate_midi_subprocess(
            musicxml_file, musescore, overwrite=overwrite, dry_run=dry_run
        )
        if process is not None:
            processes.append(process)
        while len(processes) >= njobs:
            for process in processes:
                if process.poll() is not None:
                    processes.remove(process)
    for process in processes:
        process.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Regenerate all ASAP MIDI files")
    # fmt: off
    parser.add_argument("asap_dir", type=Path, help="Path to the ASAP dataset directory")
    parser.add_argument("--musescore", type=Path, default="musescore", help="Path to the MuseScore executable")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing MIDI files")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--njobs", type=int, default=1, help="Number of parallel processes")
    # fmt: on

    args = parser.parse_args()

    generate_all_midi_files(
        args.asap_dir,
        musescore=args.musescore,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        njobs=args.njobs,
    )
