import torch
from pathlib import Path


def display_structure(obj, indent=0, max_items=1000000):
    """Recursively display the structure of nested dictionaries and lists."""
    prefix = "  " * indent

    if isinstance(obj, dict):
        print(f"{prefix}Dictionary with {len(obj)} keys:")
        for i, (key, value) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{prefix}  ... ({len(obj) - max_items} more keys)")
                break
            print(f"{prefix}  '{key}':")
            display_structure(value, indent + 2, max_items)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} with {len(obj)} items")
        if len(obj) > 0:
            print(f"{prefix}  First item type: {type(obj[0])}")
            if len(obj) <= 3:
                for i, item in enumerate(obj):
                    print(f"{prefix}  [{i}]:")
                    display_structure(item, indent + 2, max_items)
    elif isinstance(obj, torch.Tensor):
        print(f"{prefix}Tensor: shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{prefix}{type(obj).__name__}: {str(obj)[:100]}")


checkpoint_path = Path(
    "/home/gerel/Documents/transcription/gramscore2/outputs/server/2026-05-27_09-17-04.791128_ethereal-plasma-1405_modified/checkpoints/last.ckpt"
)

print(f"Loading checkpoint from: {checkpoint_path}\n")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
display_structure(checkpoint)

checkpoint["state_dict"][
    'transcriber_model.expert_heads.TupletStart.output_layers.tuplet_actual_to_normal_ratio.weight'
] = checkpoint["state_dict"].pop(
    'transcriber_model.expert_heads.TupletStart.output_layers.tuplet_normal_notes.weight'
)
checkpoint["state_dict"][
    'transcriber_model.expert_heads.TupletStart.output_layers.tuplet_actual_to_normal_ratio.bias'
] = checkpoint["state_dict"].pop(
    'transcriber_model.expert_heads.TupletStart.output_layers.tuplet_normal_notes.bias'
)
checkpoint["hyper_parameters"]["loss_weights"]["TupletStart"]["tuplet_actual_to_normal_ratio"] = (
    checkpoint["hyper_parameters"]["loss_weights"]["TupletStart"].pop("tuplet_normal_notes")
)

print("Checkpoint structure:")
print("=" * 80)

torch.save(checkpoint, checkpoint_path.with_stem("last_fixed"))
