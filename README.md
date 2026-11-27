# Transcription

This repo contains all the code I developed to support the chapter 5 of my thesis about automatic performance-MIDI-to-score piano transcription under generation constraints.
More details about this approach are presented in my thesis (to be published soon, contact me to get access sooner if needed).

## Disclaimer

As those experiments did not end up giving satisfactory results, the repository has not really been cleaned up.
The code contains the different attempts and research directions I explored, and should be referred to as reference to try other approaches, rather than as a basis to build upon.

## Install

I used `uv` as environment manager.
See the `pyproject.toml` file for dependencies.

```bash
uv sync
```

## Setup env variables

```bash
export DATA_DIR="path/to/data_dir"  # will contain both ASAP and PDMX datasets
export PYTHONPATH="path/to/root_dir"
export CUDA_VISIBLE_DEVICES=0  # or any other GPU you want to use, multi-GPU does not work well with Partitura
```

## ASAP dataset

### Download and setup

Experiments were run with the ASAP dataset, on the develop branch as I fixed some malformed scores I came across during my experiments.
The validation and test sets are available in the folder `asap-splits` of this project.

```bash
cd $DATA_DIR
git clone https://github.com/fosfrancesco/asap-dataset.git "$DATA_DIR/asap-dataset"
git checkout a0370322baa81ffe86c81c10edeeca0df286cfcf  # ASAP version used
cp "asap_splits/*.txt" "$DATA_DIR/asap-dataset"
```

### Compute measure chunking information

To compute the `[piece]_chunks.json` file for each piece, use the script `./scripts/cli/simpler_chunker_midi_scores.py`:
```bash
uv run python ./scripts/cli/simpler_chunker_midi_scores.py "$DATA_DIR/asap-dataset"
```

## Download and setup PDMX dataset

This dataset is used for pre-training the model. It is not needed to simply use the model for inference.

```
TODO
```

## Transcribe a performance

First, make sure to download the model's weights from the `Releases` tab this repository.

```bash
uv run python "./scripts/transcribe.py" --model "path/to/weights.ckpt" --output transcription.musicxml performance.mid
```

## Reproduce manuscript results table

The following command should reproduce the results presented in the chapter 5 of my thesis manuscript.
The evaluation is done using my [amt-evaluation-metrics](https://github.com/leleogere/amt-evaluation-metrics) repository, cleaner than the metrics implementations in this repository.

First, adapt the `scripts/full_transcribe_asap_test_and_evaluate.py` file with the correct paths for the model weights, the ASAP dataset and the output transcription directory.
Make sure you also have copied the `test_pieces.txt` file in the ASAP dataset folder (see above).

```bash
# Transcribe ASAP test set (evaluation is voluntarily disabled in the script due to some implementation changes)
uv run python "./scripts/full_transcribe_asap_test_and_evaluate.py"
# Evaluate with the amt-evaluation-metrics repository
cd "/path/to/amt-evaluation-metrics"
uv run python main.py "path/to/transcribed-asap" -m SS --score-file-pattern "^xml_score.musicxml$" --transcription-file-pattern "^.*flag=FINISHED.*_perm.musicxml$"  # Score similarity computation
uv run python main.py "path/to/transcribed-asap" -m MUnc --score-file-pattern "^xml_score.musicxml$" --transcription-file-pattern "^.*flag=FINISHED.*_perm.musicxml$"  # MUSTER computation
# Display the metrics with csvstats
for f in "path/to/transcribed-asap/*.csv"; do echo -e "\n$f"; csvstat --mean --decimal-format "%.2f" $f; done
```

## Run training script

The training script is `slurm_train.sh`, that you can submit to a slurm cluster as:
```bash
sbatch -J pdmx_single_staff_states slurm_train.sh
```
with `-J` being the job name, as well as the experiment yaml config file in the `configs/experiment`.

To run the training without slurm, adapt the script to use `uv run` rather than `srun`.

