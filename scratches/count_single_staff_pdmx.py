import ast
from pathlib import Path

import pandas as pd


# Set pandas display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_colwidth", 200)
# Left-align columns
pd.set_option("display.colheader_justify", "left")


datasets_path = Path("~/Documents/datasets").expanduser()
pdmx_path = datasets_path / "pdmx"
metadata_path = pdmx_path / "PDMX_extended_metadata.csv"

metadata = pd.read_csv(
    metadata_path,
    converters={
        "instruments": lambda instruments: (
            tuple(ast.literal_eval(instruments))
            if isinstance(instruments, str) and instruments != ""
            else ()
        )
    },
)
print(metadata.columns)

# %%
print(
    metadata[(metadata["n_parts"] == 1) & (metadata["instruments"] == ("Piano",))]
    .groupby(["n_staves", "instruments"])
    .count()
)
