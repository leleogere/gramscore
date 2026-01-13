from pathlib import Path
from pprint import pprint

from metrics.score_similarity import score_similarity_normalized
from metrics.muster.muster.muster import muster

# fmt: off
base_path = Path("~/Documents/datasets/pdmx").expanduser()
# reference_score_path = base_path / "mid/0/0/Qma1a1L2c1vSEuGfdPVSEEWfuYAEoBmjpVESX1ok2imu4b.mid"
# reference_score_path = base_path / "mid/0/0/Qma1a2YHEJ8j9gxLCxgALAfjYaiYJQogh1bgyFSwNSRPCs.mid"  # Polyphonic, long

reference_score_path = Path("/home/gerel/Documents/transcription/metrics/ScoreTransformer/tests/resources/piano1.musicxml")  # Polyphonic, chords
transcribed_score_path = Path("/home/gerel/Documents/transcription/metrics/ScoreTransformer/tests/resources/piano2.musicxml")

# reference_score_path = base_path / "mxl/1/11/Qmbbh58jD1ZLT1vevNsnyEZkexnx2USTFz6FSHX3KUiAzC.mxl"  # Monophonic
# transcribed_score_path = Path("/tmp/mid_1_11_Qmbbh58jD1ZLT1vevNsnyEZkexnx2USTFz6FSHX3KUiAzC.musicxml")

# fmt: on
print(f"Reference score path: {reference_score_path.as_posix()}")
print(f"Transcribed path: {transcribed_score_path.as_posix()}")


sim = {
    "mxl <-> gt_mxl": score_similarity_normalized(
        transcribed_score_path.as_posix(),
        reference_score_path.as_posix(),
        full=False,
    ),
    "muster": muster(transcribed_score_path, reference_score_path, clean_scores=False),
}
pprint(sim)
