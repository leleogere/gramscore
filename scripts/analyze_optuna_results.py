"""
Analyze Optuna optimization results and generate reports.
"""

import argparse
from pathlib import Path

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)


def analyze_study(study_name: str, storage: str = "sqlite:///optuna.db"):
    """Analyze and visualize Optuna study results."""

    # Load study
    print(f"Loading study '{study_name}' from {storage}")
    study = optuna.load_study(study_name=study_name, storage=storage)

    print("\n" + "=" * 80)
    print("STUDY SUMMARY")
    print("=" * 80)
    print(f"Study name: {study.study_name}")
    print(f"Direction: {study.direction}")
    print(f"Number of trials: {len(study.trials)}")
    print(
        f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
    )
    print(
        f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
    )
    print(
        f"Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}"
    )

    if len(study.trials) == 0:
        print("\nNo trials found in the study.")
        return

    # Best trial
    print("\n" + "=" * 80)
    print("BEST TRIAL")
    print("=" * 80)
    best_trial = study.best_trial
    print(f"Trial number: {best_trial.number}")
    print(f"Validation loss: {best_trial.value:.6f}")
    print("\nBest parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    # Export best parameters to YAML
    output_dir = Path("optuna_results")
    output_dir.mkdir(exist_ok=True)

    yaml_path = output_dir / "best_loss_weights.yaml"
    with open(yaml_path, "w") as f:
        f.write("# Best loss weights found by Optuna optimization\n")
        f.write(f"# Trial number: {best_trial.number}\n")
        f.write(f"# Validation loss: {best_trial.value:.6f}\n\n")

        # Write grouped weights (as they appear in the config)
        for key, value in best_trial.params.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"\nBest parameters exported to: {yaml_path}")

    # Parameter importances
    print("\n" + "=" * 80)
    print("PARAMETER IMPORTANCES")
    print("=" * 80)
    try:
        importances = optuna.importance.get_param_importances(study)
        for param, importance in importances.items():
            print(f"  {param}: {importance:.4f}")
    except Exception as e:
        print(f"Could not compute parameter importances: {e}")

    # Export trials to CSV
    df = study.trials_dataframe()
    csv_path = output_dir / "trials.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nAll trials exported to: {csv_path}")

    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    try:
        fig = plot_optimization_history(study)
        fig.write_html(output_dir / "optimization_history.html")
        print(f"  - Optimization history: {output_dir / 'optimization_history.html'}")
    except Exception as e:
        print(f"  - Could not create optimization history: {e}")

    try:
        fig = plot_param_importances(study)
        fig.write_html(output_dir / "param_importances.html")
        print(f"  - Parameter importances: {output_dir / 'param_importances.html'}")
    except Exception as e:
        print(f"  - Could not create param importances plot: {e}")

    try:
        fig = plot_parallel_coordinate(study)
        fig.write_html(output_dir / "parallel_coordinate.html")
        print(f"  - Parallel coordinate: {output_dir / 'parallel_coordinate.html'}")
    except Exception as e:
        print(f"  - Could not create parallel coordinate plot: {e}")

    try:
        fig = plot_slice(study)
        fig.write_html(output_dir / "slice_plot.html")
        print(f"  - Slice plot: {output_dir / 'slice_plot.html'}")
    except Exception as e:
        print(f"  - Could not create slice plot: {e}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna optimization results")
    parser.add_argument(
        "--study-name",
        type=str,
        default="loss_weights_optimization",
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna.db",
        help="Storage URL for Optuna study",
    )

    args = parser.parse_args()
    analyze_study(args.study_name, args.storage)


if __name__ == "__main__":
    main()
