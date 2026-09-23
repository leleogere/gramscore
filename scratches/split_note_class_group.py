"""
Split a parameters group into their own weights
"""

import data.components.transforms as t

import optuna

study = optuna.load_study(
    study_name="loss_weights_optimization", storage="sqlite:////tmp/optuna.db"
)
print(study.trials_dataframe())


# %% Explore how trials are stored in the study
trial = study.trials[0]
# print(trial.params)
# print(trial.distributions)

distributions = trial.distributions
if all(t.distributions == distributions for t in study.trials):
    print("All trials have the same distributions")
else:
    print("Not all trials have the same distributions")

user_attrs = trial.user_attrs
if all(t.user_attrs == user_attrs for t in study.trials):
    print("All trials have the same user_attrs")
else:
    print("Not all trials have the same user_attrs")

system_attrs = trial.system_attrs
if all(t.system_attrs == system_attrs for t in study.trials):
    print("All trials have the same system_attrs")
else:
    print("Not all trials have the same system_attrs")

intermediate_values = trial.intermediate_values
if all(t.intermediate_values == intermediate_values for t in study.trials):
    print("All trials have the same intermediate_values")
else:
    print("Not all trials have the same intermediate_values")


# %% First delete the new study if needed
# optuna.delete_study(
#     study_name="loss_weights_optimization_split_pitch",
#     storage="sqlite:////tmp/optuna.db",
# )

# %% Create new study and ungroup parameters
new_study = optuna.create_study(
    storage="sqlite:////tmp/optuna.db",
    study_name="loss_weights_optimization_split_pitch",
    directions=study.directions,
    sampler=study.sampler,
    load_if_exists=False,
)

for trial in study.trials:
    if trial.state != optuna.trial.TrialState.COMPLETE:
        print(f"Skipping trial {trial.number} with state {trial.state}.")
        continue
    weight = trial.params["model.loss_weights.Note.note_class"]
    distributions = trial.distributions["model.loss_weights.Note.note_class"]
    for param_name in [
        "model.loss_weights.Note.note_octave",
        "model.loss_weights.Note.note_accidental",
    ]:
        trial.params[param_name] = weight
        trial.distributions[param_name] = distributions

    new_study.add_trial(trial)


# %% Delete old study
optuna.delete_study(
    study_name="loss_weights_optimization",
    storage="sqlite:////tmp/optuna.db",
)
