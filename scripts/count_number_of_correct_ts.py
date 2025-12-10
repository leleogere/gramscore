"""
Try to load and run inference on a model
"""

import pandas as pd

import csv
import sys
from pprint import pprint
from pathlib import Path

from lxml import etree

import torch
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.components.score_performance_dataset import ScorePerformanceEntry
from data.pdmx_tokens_datamodule import PdmxTokensDataModule
from models.automata.vectorization_torch import TokenTensor
from models.tokens_transcriber_module import TokensTranscriberModule
from models.automata import tokens as tk
from utils.musicxml import read_musicxml

# %% Set paths
# fmt: off
# model_path = Path("~/Documents/transcription/gramscore/outputs/server/2025-12-02_15-26-21_clean-salad-313/checkpoints/last.ckpt").expanduser()
model_path = Path("~/Documents/transcription/gramscore/outputs/server/2025-12-09_13-57-48_sage-serenity-318/checkpoints/last.ckpt").expanduser()

pdmx_path = Path("~/Documents/datasets/pdmx").expanduser()
# performance_path = pdmx_path / "mid/0/0/Qma1a1L2c1vSEuGfdPVSEEWfuYAEoBmjpVESX1ok2imu4b.mid"
# performance_path = pdmx_path / "mid/0/0/Qma1a2YHEJ8j9gxLCxgALAfjYaiYJQogh1bgyFSwNSRPCs.mid"

output_path = Path(f"~/Documents/transcription/gramscore/bar_predictions_{model_path.parent.parent.stem}.csv").expanduser()
# fmt: on

logger.remove()
logger.add(sys.stdout, level="INFO")

# %% Load the model
model: TokensTranscriberModule = TokensTranscriberModule.load_from_checkpoint(
    model_path,
    map_location="cpu",
)
model.global_metrics.train(mode=False)
model.eval()

# %% Load the dataloader
datamodule = PdmxTokensDataModule(
    data_dir=pdmx_path,
    batch_size=25,
    num_workers=4,
    prefetch_factor=2,
    asap_path=pdmx_path.parent / "asap-dataset",
)
datamodule.prepare_data()
datamodule.setup(stage="validate")

# Make sure the validation pieces are the ones in the training
validation_pieces_file = model_path.parent.parent / "validation.txt"
with validation_pieces_file.open("r") as f:
    validation_piece_ids = [line.strip() for line in f.readlines()]
    print("\n".join(validation_piece_ids))
datamodule.data_val.entries = [
    ScorePerformanceEntry(
        xml_score=pdmx_path / piece_xml_path,
        midi_performance=pdmx_path / piece_xml_path.replace("mxl", "mid"),
        base_path=pdmx_path,
    )
    for piece_xml_path in validation_piece_ids
]

dataloader = datamodule.val_dataloader()


# %% Ground truth extractor function
def get_first_time_signature(musicxml_path: Path) -> tuple[int, int]:
    # Read XML
    xml = read_musicxml(musicxml_path)
    root = etree.fromstring(xml.encode("utf-8"))
    # Finding first <time> element
    time_el = root.find(".//time")
    if time_el is None:
        raise ValueError("No <time> element found")
    # Getting the time signature attributes
    beats = time_el.findtext("beats")
    beat_type = time_el.findtext("beat-type")
    numerator = int(beats)
    denominator = int(beat_type)
    return numerator, denominator


# %% Make predictions
bar_predictions = []
bar_expert = model.transcriber_model.expert_heads["Bar"]
with torch.no_grad():
    for batch in (pbar := tqdm(dataloader)):
        score_tensor = batch["score_tensor"]
        performance_streams = batch["performance_streams"]
        performance_onset_times = batch["performance_onset_times"]
        performance_no_padded_mask = ~batch["performance_padding_mask"]

        # Pass performance thought encoder
        encoder_outputs = model.transcriber_model.forward_encoder(
            input_streams=performance_streams,
            onset_times=performance_onset_times,
            attention_mask=performance_no_padded_mask,
        )

        # We want to only predict the first bar, so only pass the SOS token
        decoder_outputs = model.transcriber_model.forward_decoder(
            input_tokens=score_tensor[:, :1],
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=performance_no_padded_mask,
        )

        # Decode thought router (not needed, just to check that the model
        # indeed wants to produce a Bar token)
        router_logits = model.transcriber_model.forward_router(decoder_outputs.last_hidden_state)
        predicted_token_types = router_logits.argmax(dim=-1)
        if any(predicted_token_types[:, 0] != TokenTensor.TOKEN_INDEXER[tk.Bar]):
            logger.warning("At least one token does not want to predict a Bar as first token")

        # Pass thought Bar expert
        outputs = bar_expert(decoder_outputs.last_hidden_state)
        ts_idxs = outputs["bar_time_signature"].argmax(-1)
        are_nominal = outputs["bar_is_nominal"][:, 0].argmax(-1)

        for entry, ts_idx, is_nominal in zip(batch["entries"], ts_idxs, are_nominal):
            pred_ts = TokenTensor.TS_INDEXER_INV[ts_idx.item()]
            pred_is_nominal = bool(is_nominal.item())

            true_numerator, true_denominator = get_first_time_signature(entry.xml_score)

            is_correct = pred_ts == (true_numerator, true_denominator)

            bar_predictions.append(
                {
                    "pred_numerator": pred_ts[0],
                    "pred_denominator": pred_ts[1],
                    "pred_is_nominal": pred_is_nominal,
                    "true_numerator": true_numerator,
                    "true_denominator": true_denominator,
                    "correct": is_correct,
                    "score": entry.xml_score_relative,
                }
            )

            # breakpoint()


# %%
with output_path.open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(bar_predictions[0].keys()))
    writer.writeheader()
    for row in bar_predictions:
        writer.writerow(row)


# %% Play with predictions
import pandas as pd

predictions = pd.read_csv(output_path)

equivalence_ts = [
    ((4, 4), (2, 2), (2, 4)),
    ((3, 4), (6, 8)),
]
equivalence_ts_dict = {
    ts1: [ts2 for ts2 in group if ts2 != ts1] for group in equivalence_ts for ts1 in group
}

# Count the number of exactly correct time signatures
predictions["exactly_correct"] = predictions.apply(
    lambda row: (row.pred_numerator == row.true_numerator)
    and (row.pred_denominator == row.true_denominator),
    axis=1,
)

# Count the number of time signature in the correct equivalence group
predictions["approx_correct"] = predictions.apply(
    lambda row: (row.pred_numerator, row.pred_denominator)
    in equivalence_ts_dict.get((row.true_numerator, row.true_denominator), []),
    axis=1,
)

# Print stats
total = len(predictions)
print(f"Number of correct predictions: {sum(predictions['exactly_correct']):>4}/{total}")
print(f"Number of inexact predictions: {sum(predictions['approx_correct']):>4}/{total}")
print(
    f"Correct + inexact predictions: "
    f"{sum(predictions['exactly_correct'])+sum(predictions['approx_correct']):>4}/{total}"
)
