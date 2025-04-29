# Transciber model

## Create environement

```bash
# TODO
```

## Clone ASAP
```bash
## TODO
```

## Setup env variables

```bash
export DATA_DIR="..."  # must contain the `asap-dataset` folder
export PYTHONPATH="path/to/root_dir"
export CUDA_VISIBLE_DEVICES=0  # or any other GPU you want to use
```

## Assign performed notes to measures

To compute the `[piece]_chunks.json` file for each piece,
use the script `./scripts/cli/simpler_chunker_midi_scores.py`:
```bash
python ./scripts/cli/simpler_chunker_midi_scores.py "path/to/asap"
```

