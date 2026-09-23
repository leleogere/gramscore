"""
Print metadata about WandB runs
"""

import wandb
import tabulate
import datetime
from tqdm import tqdm


# "%Y-%m-%dT%H:%M:%SZ"
def utc_to_local_time(utc_time: str) -> str:
    return (
        datetime.datetime.strptime(utc_time, "%Y-%m-%dT%H:%M:%SZ")
        .replace(tzinfo=datetime.timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%d %H:%M:%S")
    )


# Initialize the API
api = wandb.Api()

# Replace with your 'entity/project' (e.g., 'my-team/my-project')
project_path = "leleogere/gramscore_tokens"

# Fetch newest runs
runs = api.runs(project_path, order="-created_at")

fields_getters = {
    "name": lambda r: r.name,
    "id": lambda r: r.id,
    "notes": lambda r: r.notes,
    "output_dir": lambda r: r.config["paths"]["output_dir"].split("/")[
        -1
    ],  # 2026-03-02_10-19-42.877278  or  2026-03-02_10-19-42
    # Created date is in format 2026-02-24T18:32:38Z
    "created_date": lambda r: utc_to_local_time(r.created_at),
    "heartbeat_date": lambda r: utc_to_local_time(r.heartbeat_at),
    # Compute duration from output dir and heartbeat date
    "duration": lambda r: (
        datetime.datetime.strptime(utc_to_local_time(r.heartbeat_at), "%Y-%m-%d %H:%M:%S")
        - datetime.datetime.strptime(
            r.config["paths"]["output_dir"].split("/")[-1],
            (
                "%Y-%m-%d_%H-%M-%S.%f"
                if "." in r.config["paths"]["output_dir"].split("/")[-1]
                else "%Y-%m-%d_%H-%M-%S"
            ),
        )
    ),
    "job_name": lambda r: r.metadata["slurm"]["job_name"] if r.metadata else "",
}

num_runs = 40
table = []
i = 0
pbar = tqdm(total=num_runs)
while i < num_runs:
    run = next(runs)
    if "loss_weights_sweep" in run.tags:
        print(f"Skipping sweep run {run.id}")
        continue
    table.append([getter(run) for getter in fields_getters.values()])
    i += 1
    pbar.update(1)

print(tabulate.tabulate(table, headers=list(fields_getters.keys())))
