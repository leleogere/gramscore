"""
Visualize cross attention
"""

import torch

import partitura as pt

import matplotlib.pyplot as plt

from pathlib import Path

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPast,
)

import data.components.transforms as t
from data.components.score_performance_dataset import ScorePerformanceEntry, ScorePerformanceDataset
from models.automata.io import token_sequence_to_partitura_permissive
from utils.generation import GenerationState
from models.tokens_transcriber_module import TokensTranscriberModule
import models.automata.tokens as tk
from utils.constants import PADDING_VALUE

# fmt: off
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-03-16_15-49-56.653967_dulcet-dew-374/checkpoints/last.ckpt")
# model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-04-01_00-32-12.791876_whole-oath-391/checkpoints/last.ckpt")
# model_path = Path("~/Documents/transcription/gramscore2/outputs/2026-04/2026-04-08_11-20-19.491437/checkpoints/last.ckpt")
model_path = Path("~/Documents/transcription/gramscore2/outputs/server/2026-04-08_09-09-02.955002_solar-sound-393/checkpoints/last.ckpt")

pdmx_path = Path("~/Documents/datasets/pdmx").expanduser()
asap_path = Path("~/Documents/datasets/asap-dataset").expanduser()
performance_path = (
    # pdmx_path / "mid/0/0/Qma1a1L2c1vSEuGfdPVSEEWfuYAEoBmjpVESX1ok2imu4b.mid"
    # pdmx_path / "mid/1/24/QmbHzrXuDGNG3HyzLpeBNeZ9wAj4bTDGWsZ5xz56abM8D8.mid"  # Test unfinished
    # pdmx_path / "mid/13/29/QmVKam2PcfGjsMF1QtwWEwSgG5bqFyiBRGiF5btFQX6vmV.mid"  # Test grace notes
    # pdmx_path / "mid/0/0/Qma1a2YHEJ8j9gxLCxgALAfjYaiYJQogh1bgyFSwNSRPCs.mid"  # Polyphonic, long
    # pdmx_path / "mid/1/11/Qmbbh58jD1ZLT1vevNsnyEZkexnx2USTFz6FSHX3KUiAzC.mid"  # Monophonic
    # pdmx_path / "mid/15/12/QmXB8B4F9cbHUrnrP4mRGyL3vyhTkyDattentionjuVEbM3VdFu2EY9.mid"  # Polyphonic, chords
    # pdmx_path / "mid/1/32/QmbMQQPWGZpGiTV5gngSwmjTbqd7XosW8yyPwkSHnXbhY2.mid"  # Polyphonic, chords
    # Path("/home/gerel/Documents/Perso/Compositions/La Quête d'Ewilan/La Quête d'Ewilan.mid")
    # Path("/home/gerel/Téléchargements/2026-02-25_21-01-14_729n_67s_STARRED.mid")
    asap_path / "Prokofiev/Toccata/Teo05.mid"
)
if "pdmx" in performance_path.parts:
    reference_score_path = pdmx_path / "mxl" / performance_path.relative_to(pdmx_path / "mid").with_suffix(".mxl")
    base_path = pdmx_path
elif "asap-dataset" in performance_path.parts:
    reference_score_path = performance_path.parent / "xml_score.musicxml"
    base_path = asap_path
else:
    reference_score_path = performance_path.parent / "xml_score_clipped.musicxml"
# fmt: on
print(f"Reference score path: {reference_score_path}")
print(f"Performance path: {performance_path}")


NOTES_PER_CHUNK = 500
TOKENS_LIMIT_PER_CHUNK = 1500
GENERATE_INSTEAD_OF_GT = True
# Show only small figures, else close
SHOW_FIGS = NOTES_PER_CHUNK * TOKENS_LIMIT_PER_CHUNK < 50_000

output_dir = Path(
    f"/tmp/attentions/{performance_path.stem}/GENERATED={GENERATE_INSTEAD_OF_GT}_n={NOTES_PER_CHUNK}/"
)
output_dir.mkdir(exist_ok=True, parents=True)


# Load the model
model: TokensTranscriberModule = TokensTranscriberModule.load_from_checkpoint(
    model_path,
    map_location="cpu",
    weights_only=False,
)
model.eval()


dataset = ScorePerformanceDataset(
    entries=[
        ScorePerformanceEntry(
            xml_score=reference_score_path,
            midi_performance=performance_path,
            base_path=base_path,
        )
    ],
    transforms=[
        t.ToDict(),
        t.ScoreReader(),
        t.PerformanceReader(),
        (
            t.QuantizedMusicChunker(num_performed_notes=NOTES_PER_CHUNK, mode=0)
            if base_path == pdmx_path
            else t.AsapMusicChunker(
                num_performed_notes=NOTES_PER_CHUNK,
                annotations=asap_path / "asap_annotations.json",
                mode=0,
            )
        ),
        t.PerformanceToPerformedNoteList(),
        t.PerformedNoteListVectorizer(),
        t.VectorizedPerformanceStreamsBucketizer(),
        t.ScoreVectorizer(),
    ],
)

performance = next(iter(dataset))


encoder_outputs, encoder_pre_norm_outputs = model.transcriber_model.forward_encoder(
    input_streams={
        k: v.unsqueeze(0) for k, v in performance["performance_streams"].items() if k != "pad"
    },
    onset_times=performance["performance_streams_raw"]["onset"].unsqueeze(0),
    attention_mask=None,
    also_return_pre_norm_encoder_outputs=True,
    output_attentions=True,
)
print(f"Encoders self-attention shape: {encoder_pre_norm_outputs.attentions[-1].shape}")

if GENERATE_INSTEAD_OF_GT:
    input_tokens, flag = model.transcriber_model.generate(
        performance_streams={
            k: v.unsqueeze(0) for k, v in performance["performance_streams"].items() if k != "pad"
        },
        performance_onset_times=performance["performance_streams_raw"]["onset"].unsqueeze(0),
        performance_attention_mask=None,
        prefix=None,
        constraint=False,
        max_constraint_iterations=None,
        max_generated_length=TOKENS_LIMIT_PER_CHUNK,
        use_cache=True,
        progress_bar=True,
        return_logits_and_mask=False,
    )
    if GenerationState(flag[0][0].item()) != GenerationState.FINISHED:
        print("WARNING: Generation did not finished as expected")
    input_tokens = input_tokens.quantize().squeeze(0)
    input_tokens_decoded = input_tokens.to_tokens()
    transcribed_score = token_sequence_to_partitura_permissive(input_tokens_decoded)
    save_path = output_dir / "transcription.musicxml"
    pt.save_musicxml(transcribed_score, save_path)
    print(f"Transcription saved to {save_path}")
else:
    input_tokens = performance["score_tensor"]
    input_tokens_decoded = performance["score_tokens"]

decoder_outputs = model.transcriber_model.forward_decoder(
    input_tokens=input_tokens[:-1].unsqueeze(0),  # Remove EOS
    encoder_hidden_states=encoder_outputs,
    encoder_attention_mask=None,
    use_cache=False,
    output_attentions=True,
)

print(f"Decoder self-attention shape: {decoder_outputs.attentions[-1].shape}")
print(f"Decoder cross-attention shape: {decoder_outputs.cross_attentions[-1].shape}")
# exit(0)


# %% Define the attention plotting function
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import itertools
from typing import Iterable


def flatten(x):
    return itertools.chain.from_iterable([i] if not isinstance(i, Iterable) else i for i in x)


def plot_attentions(
    outputs: BaseModelOutputWithPastAndCrossAttentions | BaseModelOutputWithPast,
    cross: bool = False,
    average_over_heads: bool = True,
    tokens_names: list[str] | None = None,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    attentions = outputs.attentions if not cross else outputs.cross_attentions
    assert all(a.shape[0] == 1 for a in attentions), "Only batch size 1 is supported"

    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    src_length, tgt_length = attentions[0].shape[-2:]

    if average_over_heads:
        fig, axes = plt.subplots(
            1,
            n_layers,
            figsize=(int(0.1 * n_layers * src_length), int(0.12 * tgt_length)),
        )
        for l in range(n_layers):
            axes[l].imshow(
                attentions[l][0].mean(dim=0).detach().cpu().numpy().T,
                interpolation="nearest",
                aspect="equal",
            )
            axes[l].set_title(f"Layer {l}")
    else:
        fig, axes = plt.subplots(
            n_layers,
            n_heads,
            figsize=(int(0.1 * n_heads * tgt_length), int(0.06 * n_layers * src_length)),
        )
        for l in range(n_layers):
            for h in range(n_heads):
                axes[l, h].imshow(
                    attentions[l][0, h].detach().cpu().numpy().T,
                    interpolation="nearest",
                    aspect="equal",
                )
                if l == 0:
                    axes[l, h].set_title(f"Head {h}")
    for ax in flatten(axes):
        if tokens_names:
            ax.set_xticks(range(len(tokens_names)))
            ax.set_xticklabels(tokens_names, rotation=90, size=7)
            if not cross:
                ax.set_yticks(range(len(tokens_names)))
                ax.set_yticklabels(tokens_names, rotation=0, size=7)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


plot_attentions(
    decoder_outputs,
    cross=False,
    average_over_heads=True,
    tokens_names=[f"{t.short_name()} {i}" for i, t in enumerate(input_tokens_decoded[1:])],
    output_path=output_dir / f"avg_self_attention.pdf",
    show=SHOW_FIGS,
)
plot_attentions(
    decoder_outputs,
    cross=True,
    average_over_heads=True,
    tokens_names=[f"{t.short_name()} {i}" for i, t in enumerate(input_tokens_decoded[1:])],
    output_path=output_dir / f"avg_cross_attention.pdf",
    show=SHOW_FIGS,
)
plot_attentions(
    encoder_pre_norm_outputs,
    cross=False,
    average_over_heads=True,
    output_path=output_dir / f"avg_encoder_self_attention.pdf",
    show=SHOW_FIGS,
)


# # %% Plot the head-averaged cross-attention maps
# import matplotlib.pyplot as plt
# import matplotlib.ticker as tck
#
# n_layers = len(decoder_outputs.cross_attentions)
# n_heads = decoder_outputs.cross_attentions[0].shape[1]
# performance_length = len(performance["performance_notes"])
# score_length = len(input_tokens_decoded)
#
# fig, axes = plt.subplots(
#     n_layers, 1, figsize=(int(0.2 * score_length), int(0.1 * n_layers * performance_length))
# )
# for l in range(n_layers):
#     axes[l].imshow(
#         decoder_outputs.cross_attentions[l][0].mean(dim=0).detach().cpu().numpy().T,
#         interpolation="nearest",
#         aspect="equal",
#     )
#     # axes[l].yaxis.set_major_locator(tck.MultipleLocator())
#     axes[l].set_title(f"Layer {l}")
#     xticks = [f"{t.short_name()} {i}" for i, t in enumerate(input_tokens_decoded[1:])]
#     axes[l].set_xticks(range(len(xticks)))
#     axes[l].set_xticklabels(xticks, rotation=90, size=7)
#     axes[l].grid(which="major", axis="both", alpha=0.05)
#     axes[l].grid(which="minor", axis="both", alpha=0.05)
# # plt.tight_layout()
# plt.savefig(output_dir / "cross_attention_maps_headavg.pdf", bbox_inches="tight")
# # plt.show()
# plt.close()
#

# # %% Plot the head-averaged self-attention maps
# fig, axes = plt.subplots(
#     n_layers, 1, figsize=(int(0.1 * score_length), int(0.1 * n_layers * score_length))
# )
#
# for l in range(n_layers):
#     axes[l].imshow(
#         decoder_outputs.attentions[l][0].mean(dim=0).detach().cpu().numpy().T,
#         interpolation="nearest",
#         aspect="equal",
#     )
#     axes[l].set_title(f"Layer {l}")
#     ticks = [f"{t.short_name()} {i}" for i, t in enumerate(input_tokens_decoded)]
#     axes[l].set_xticks(range(len(ticks) - 1))
#     axes[l].set_xticklabels(ticks[1:], rotation=90, size=7)
#     axes[l].set_yticks(range(len(ticks) - 1))
#     axes[l].set_yticklabels(ticks[:-1], size=7)
#     axes[l].grid(which="major", axis="both", alpha=0.05)
#     axes[l].grid(which="minor", axis="both", alpha=0.05)
# plt.tight_layout()
# plt.savefig(output_dir / "self_attention_maps_headavg.pdf", bbox_inches="tight")
# plt.close()
#
#
# # %% Plot the cross-attention maps
# n_heads = decoder_outputs.cross_attentions[0].shape[1]
# performance_length = len(performance["performance_notes"])
# score_length = len(input_tokens_decoded)
#
# fig, axes = plt.subplots(
#     n_layers,
#     n_heads,
#     figsize=(int(0.1 * n_heads * score_length), int(0.2 * n_layers * performance_length)),
# )
# for l in range(n_layers):
#     for h in range(n_heads):
#         axes[l, h].imshow(
#             decoder_outputs.cross_attentions[l][0, h].detach().cpu().numpy().T,
#             interpolation="nearest",
#             aspect="equal",
#         )
#         axes[l, h].yaxis.set_major_locator(tck.MultipleLocator())
#         if l == 0:
#             axes[l, h].set_title(f"Head {h}")
#         if h == 0:
#             axes[l, h].set_ylabel(f"Layer {l}")
#         # if l == n_layers - 1:
#         xticks = [f"{t.short_name()} {i}" for i, t in enumerate(input_tokens_decoded[1:])]
#         axes[l, h].set_xticks(range(len(xticks)))
#         axes[l, h].set_xticklabels(xticks, rotation=90, size=7)
#         axes[l, h].grid(which="major", axis="both", alpha=0.05)
#         axes[l, h].grid(which="minor", axis="both", alpha=0.05)
# plt.tight_layout()
# plt.savefig(output_dir / "cross_attention_maps.pdf", bbox_inches="tight")
# plt.close()
# # plt.show()
#
#
# # %% Also plot the self-attention maps
# fig, axes = plt.subplots(
#     n_layers,
#     n_heads,
#     figsize=(int(0.15 * n_heads * score_length), int(0.15 * n_layers * score_length)),
# )
# for l in range(n_layers):
#     for h in range(n_heads):
#         axes[l, h].imshow(
#             decoder_outputs.attentions[l][0, h].detach().cpu().numpy().T,
#             interpolation="nearest",
#             aspect="equal",
#         )
#         if l == 0:
#             axes[l, h].set_title(f"Head {h}")
#         if h == 0:
#             axes[l, h].set_ylabel(f"Layer {l}")
#         # if l == n_layers - 1:
#         ticks = [f"{t.short_name()} {i}" for i, t in enumerate(input_tokens_decoded)]
#         axes[l, h].set_xticks(range(len(ticks) - 1))
#         axes[l, h].set_xticklabels(ticks[1:], rotation=90, size=7)
#         axes[l, h].set_yticks(range(len(ticks) - 1))
#         axes[l, h].set_yticklabels(ticks[:-1], size=7)
#         axes[l, h].grid(which="major", axis="both", alpha=0.05)
#         axes[l, h].grid(which="minor", axis="both", alpha=0.05)
# plt.tight_layout()
# plt.savefig(output_dir / "self_attention_maps.pdf", bbox_inches="tight")
# plt.close()
# # plt.show()
#
#
# # %% Print the index of Bar tokens in the sequence
# print(
#     f"Bar tokens: {[i for i, t in enumerate(performance['score_tokens']) if isinstance(t, tk.Bar)]}"
# )
