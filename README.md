# Transciber model

## Install

```bash
uv sync
```

## Setup env variables

```bash
export DATA_DIR="..."  # must contain the `asap-dataset` folder
export PYTHONPATH="path/to/root_dir"
export CUDA_VISIBLE_DEVICES=0  # or any other GPU you want to use, multi-GPU does not work well
```

## Assign performed notes to measures

To compute the `[piece]_chunks.json` file for each piece, use the script `./scripts/cli/simpler_chunker_midi_scores.py`:
```bash
python ./scripts/cli/simpler_chunker_midi_scores.py "path/to/asap"
```

## Run training script

The training script is `slurm_train.sh`, that you can submit to a slurm cluster as:
```bash
sbatch -J pdmx_single_staff_states slurm_train.sh
```
with `-J` being the job name, as well as the experiment yaml config file in the `configs/experiment`.

To run the training without slurm, adapt the script to use `uv run` rather than `srun`.

## Evaluation and analyze

All evaluation and analyze scripts that I had to write during my explorations are in the folders `scripts` and `scratches`.
While `scratches` are supposed to be one-off scripts to try something, there are also some old such scripts in the `scripts` folder due to `scratches` being created way later.

The most important one is `scripts/full_transcribe_asap_test_and_evaluate.py`, that transcribes the ASAP dataset, and run the evaluation on the transcribed dataset.
