"""
Make a pass on a complete dataset to make sure it works
"""

from pathlib import Path

from tqdm import tqdm

from data.pdmx_tokens_datamodule import PdmxTokensDataModule


pdmx_path = Path("~/Documents/datasets/pdmx").expanduser()
asap_path = pdmx_path.parent / "asap-dataset"

datamodule = PdmxTokensDataModule(
    data_dir=pdmx_path,
    batch_size=128,
    num_workers=4,
    asap_path=asap_path,
)
datamodule.prepare_data()
datamodule.setup("fit")

# print("Testing validation set")
# for batch in tqdm(datamodule.val_dataloader()):
#     pass

# datamodule.data_train.entries = [
#     e
#     for e in datamodule.data_train.entries
#     if e.xml_score_relative.as_posix()
#     == "mxl/13/44/QmVtVpmQKg6GpnoaUFWLfwTXe1LHGdNkgPk6GBgfE5CDrW.mxl"
# ]
print(len(datamodule.data_train.entries))
print("Testing train set")
for batch in tqdm(datamodule.train_dataloader()):
    # print(batch.keys())
    pass

# FIXED!
# 2025-12-01 18:57:30.021 | DEBUG    | data.components.transforms:_process:374 - Clipping mxl/13/44/QmVtVpmQKg6GpnoaUFWLfwTXe1LHGdNkgPk6GBgfE5CDrW.mxl from (10, 11)
# 2025-12-01 18:57:30.021 | ERROR    | data.components.score_performance_dataset:apply_transforms:125 - Failed to apply transform <data.components.transforms.PerformanceSpeedAugmenter object at 0x79bcd0f9d810> on mxl/13/44/QmVtVpmQKg6GpnoaUFWLfwTXe1LHGdNkgPk6GBgfE5CDrW.mxl: index 0 is out of bounds for axis 0 with size 0
