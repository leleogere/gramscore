"""
Manually create a baseline trial in an existing optuna study
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

# %% Create the trial
baseline_trial = optuna.trial.create_trial(
    state=optuna.trial.TrialState.COMPLETE,
    values=[56.97858810424805, 0.7313761711120605],
    params={param: 0.01 for param in distributions},
    distributions=distributions,
    user_attrs={**user_attrs, "baseline": True},
    system_attrs=system_attrs,
    intermediate_values=intermediate_values,
)

# %% Add the trial to the study
study.add_trial(baseline_trial)
