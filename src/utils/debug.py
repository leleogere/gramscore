import pretty_midi as pm
import subprocess
import time
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch


def show_matrix(
    matrix: torch.Tensor | np.ndarray,
    figsize: tuple[int, int] | None = None,
    colorbar: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    origin: str | None = None,
    extent: tuple[float, float, float, float] | None = None,
):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    mat = ax.imshow(
        matrix,
        aspect="auto",
        cmap="viridis",
        interpolation="none",
        extent=extent,
        origin=origin,
    )

    if colorbar:
        plt.colorbar(mat, ax=ax)

    if title is not None:
        title = f"{title}\n"
    else:
        title = ""
    title += f"shape={matrix.shape} min={matrix.min()} max={matrix.max()}"
    ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    plt.show()


def open_in_musescore(path: Path, only_if_not_open: bool = True):
    MUSESCORE_PATH = Path("~/.local/bin/musescore4portable").expanduser()
    if only_if_not_open:
        # Check if MuseScore is already running with the given file
        try:
            import psutil

            for proc in psutil.process_iter(['cmdline']):
                cmdline = proc.info['cmdline']
                if cmdline and MUSESCORE_PATH.as_posix() in cmdline and path.as_posix() in cmdline:
                    return None  # Already open
        except ImportError:
            pass  # psutil not available, skip the check
    return subprocess.Popen(
        [MUSESCORE_PATH.as_posix(), path.as_posix()],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def open_in_midieditor(path: Path):
    MIDIEDITOR_PATH = Path("~/Applications/midieditor/MidiEditor").expanduser()
    return subprocess.Popen(
        [MIDIEDITOR_PATH.as_posix(), path.as_posix()],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def open_in_meld(text_or_path1: str | Path, text_or_path2: str | Path):
    if isinstance(text_or_path1, Path):
        path1 = text_or_path1
    else:
        path1 = Path("/tmp/meld_temp1.txt")
        with path1.open("w") as f:
            f.write(text_or_path1)

    if isinstance(text_or_path2, Path):
        path2 = text_or_path2
    else:
        path2 = Path("/tmp/meld_temp2.txt")
        with path2.open("w") as f:
            f.write(text_or_path2)

    return subprocess.Popen(
        [
            "meld",
            path1.as_posix(),
            path2.as_posix(),
        ]
    )


def concat_streams(
    streams: dict[str, torch.Tensor],
    separate_by_none: bool = False,
    normalize: bool = False,
) -> np.ndarray:
    streams = {k: v.cpu().numpy() if v is not None else None for k, v in streams.items()}
    streams = {
        k: v if v.ndim == 2 else v[:, None] if v is not None else None for k, v in streams.items()
    }
    if normalize:
        streams = {
            k: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8) if v is not None else None
            for k, v in streams.items()
        }
    if separate_by_none:
        streams = {
            k: np.concat([v, np.full((v.shape[0], 1), np.nan)], axis=-1) if v is not None else None
            for k, v in streams.items()
        }
    return np.concat([v for v in streams.values() if v is not None], axis=-1)


def as_pycharm_link(path: Path | str) -> str:
    if isinstance(path, Path):
        path = path.absolute().as_posix()
    return f"file://{path}"


def write_compare_to_file(
    value,
    filename: str,
    wait_for_previous: bool = False,
    base_path: Path = Path("/tmp/debug_files"),
    if_different: Literal["error", "breakpoint", "meld"] = "breakpoint",
    disable: bool = True,
):
    if disable:
        return
    file_path = base_path / filename
    # First process to write, simply write the file
    if not wait_for_previous:
        base_path.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as f:
            f.write(str(value))
    # Second process to write, wait for the file to be written first, then compare it
    else:
        # Wait for the file to be written
        while not file_path.exists():
            time.sleep(0.1)
        # Write the second value
        with file_path.with_suffix(".2.txt").open("w") as f:
            f.write(str(value))
        # Load it
        with file_path.open("r") as f:
            first_value = f.read()
        # Compare
        if first_value != str(value):
            print(f"Values differ:\n{first_value} != {value}")
            if if_different == "breakpoint":
                breakpoint()
            if if_different == "error":
                raise ValueError(f"Values differ:\n{first_value} != {value}")
            if if_different == "meld":
                open_in_meld(first_value, str(value))
            else:
                raise ValueError(f"Unknown if_different value: {if_different}")


def performance_streams_to_midi(streams: dict[str, torch.Tensor]) -> pm.PrettyMIDI:
    """
    Converts back performance streams into a MIDI file.

    :param streams: Dictionary of performance streams.
    """
    notes = []
    for pitch, onset, duration, velocity in zip(
        streams["pitch"], streams["onset"], streams["duration"], streams["velocity"]
    ):
        notes.append(
            pm.Note(
                pitch=int(pitch),
                start=float(onset),
                end=float(onset + duration),
                velocity=int(velocity),
            )
        )
    midi = pm.PrettyMIDI()
    midi.instruments.append(pm.Instrument(program=0))
    midi.instruments[0].notes = notes
    return midi


def plot_pianoroll(
    midi: pm.PrettyMIDI,
    downbeats: list[float] | None = None,
    downbeats_labels: list["str"] | None = None,
    start: float = 0.0,
    end: float | None = None,
    fs: int = 100,
    pedal_threshold: int | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
):
    if end is None:
        end = midi.get_end_time()
    pianoroll = midi.get_piano_roll(
        fs=fs, times=np.arange(start, end, 1.0 / fs), pedal_threshold=pedal_threshold
    )

    # Find the pitch range with actual notes
    pitch_sum = pianoroll.sum(axis=1)
    active_pitches = np.where(pitch_sum > 0)[0]
    if len(active_pitches) == 0:
        min_pitch, max_pitch = 0, 127
    else:
        min_pitch = max(0, active_pitches.min() - 2)
        max_pitch = min(127, active_pitches.max() + 2)

    # Slice the piano roll to the active pitch range
    pianoroll_slice = pianoroll[min_pitch : max_pitch + 1, :]

    if ax is None:
        fig, ax = plt.subplots(figsize=(end - start, (max_pitch - min_pitch) / 10))

    # Plot piano roll
    ax.imshow(
        pianoroll_slice,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        extent=[start, end, min_pitch, max_pitch + 1],
    )

    # Set up axes
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MIDI Pitch")

    # Add downbeat markers
    if downbeats is not None:
        if downbeats_labels is None:
            downbeats_labels = map(str, range(len(downbeats)))
        for downbeat, label in zip(downbeats, downbeats_labels):
            if start <= downbeat <= end:
                ax.axvline(x=downbeat, color='red', linestyle='--', alpha=0.7, linewidth=1)
                ax.text(
                    downbeat,
                    max_pitch,
                    label,
                    rotation=0,
                    verticalalignment='bottom',
                    horizontalalignment='center',
                    color='red',
                )

    # plt.colorbar(im, ax=ax, label="Velocity")
    if show:
        plt.tight_layout()
        plt.show()
