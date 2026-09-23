import wandb

ENTITY = "leleogere"
PROJECT = "gramscore_tokens"

api = wandb.Api()

# First collect every DA=xxx tag used by the selected runs.
runs = api.runs(
    f"{ENTITY}/{PROJECT}",
    filters={"tags": {"$in": ["DA-tests"]}},
)

da_tags = set()

for run in runs:
    print(run.name)
    da_tags.update(tag for tag in run.tags if tag.startswith("DA="))

print(f"Found {len(da_tags)} DA tags: {sorted(da_tags)}")

# Update each run's config.
for run in runs:
    da_config = {tag: tag in run.tags for tag in sorted(da_tags)}

    run.config.update(da_config)
    run.update()

    print(f"Updated {run.id}: {da_config}")
