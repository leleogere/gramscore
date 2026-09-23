"""
Sample the same PDMX file with data augmentations
"""

import partitura as pt
from pathlib import Path

from data.pdmx_tokens_datamodule import PdmxTokensDataModule
from models.automata.io import token_sequence_to_partitura_permissive
from utils.debug import performance_streams_to_midi

datasets_path = Path("~/Documents/datasets").expanduser()
asap_path = datasets_path / "asap-dataset"
pdmx_path = datasets_path / "pdmx"

pdmx = PdmxTokensDataModule(
    data_dir=pdmx_path,
    batch_size=1,
    num_workers=1,
    asap_test_pieces_path=asap_path / "test_pieces.txt",
    allowed_staffs=(1,),
    include_states=True,
    state_num_voices=8,
)
pdmx.prepare_data()
pdmx.setup("fit")
train_dataloader = pdmx.train_dataloader()

output_dir = Path("/tmp/augmentations")
output_dir.mkdir(exist_ok=True)

for i in range(10):
    dir = output_dir / f"sample_{i}"
    dir.mkdir(exist_ok=True)
    for batch in train_dataloader:
        for j in range(len(batch["entry"])):
            entry = batch["entry"][j]
            score_tensor = batch["score_tensor"][j]
            performance_streams = {
                s: batch["performance_streams_raw"][s][j]
                for s in batch["performance_streams_raw"].keys()
            }
            score_tokens = score_tensor.to_tokens()
            with open(dir / f"{entry.xml_score.stem}_{entry.xml_score.stem}.tokens", "w") as f:
                f.write("\n".join(map(str, score_tokens)))
            score = token_sequence_to_partitura_permissive(score_tokens)
            pt.save_musicxml(
                score,
                dir / f"{entry.xml_score.stem}_{entry.xml_score.stem}.musicxml",
            )
            midi = performance_streams_to_midi(performance_streams)
            midi.write(dir / f"{entry.xml_score.stem}_{entry.xml_score.stem}.mid")
            break
        break
