import numpy as np
import matplotlib.pyplot as plt

# # plt.rcParams.update(plt.rcParamsDefault)
# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": "Iwona",
#     }
# )

# 1. Configuration & Data Generation
N_TRAIN_POINTS = 20
N_TEST_POINTS = 10
X_MIN, X_MAX = 0, 10
Y_MIN, Y_MAX = -1, 13
NOISE_LEVEL = 0.6
SEED = 15


def true_function(x):
    return x + 4 * np.sin(x)


def calc_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


np.random.seed(SEED)

# Generate random training data
X_train = np.linspace(X_MIN, X_MAX, N_TRAIN_POINTS)
X_train += np.random.normal(0, 0.1, N_TRAIN_POINTS)
y_train = true_function(X_train) + np.random.normal(0, NOISE_LEVEL, N_TRAIN_POINTS)

# Generate random test data
X_test = np.linspace(X_MIN, X_MAX, N_TEST_POINTS)
X_test += np.random.normal(0, 0.1, N_TEST_POINTS)
y_test = true_function(X_test) + np.random.normal(0, NOISE_LEVEL, N_TEST_POINTS)

# Smooth X values for drawing the curves
X_curve = np.linspace(X_MIN - 1, X_MAX + 1, 10000)

# 2. Define the Models

# A. Underfitting: Linear Regression (Degree 1)
coeffs_under = np.polyfit(X_train, y_train, 1)
model_underfit = np.poly1d(coeffs_under)

# B. Good Fit: The True Function
model_good = true_function

# C. Overfitting: High-dimensional Polynomial (Degree N-1 perfectly interpolates N points)
coeffs_over = np.polyfit(X_train, y_train, N_TRAIN_POINTS - 1)
model_overfit = np.poly1d(coeffs_over)

# 3. Calculate Errors (MSE)
mse_train = {
    'Underfit': calc_mse(y_train, model_underfit(X_train)),
    'Good': calc_mse(y_train, model_good(X_train)),
    'Overfit': calc_mse(y_train, model_overfit(X_train)),
}

mse_test = {
    'Underfit': calc_mse(y_test, model_underfit(X_test)),
    'Good': calc_mse(y_test, model_good(X_test)),
    'Overfit': calc_mse(y_test, model_overfit(X_test)),
}

# 4. Plotting
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))

plot_configs = [
    {
        # 'ax': axes[0],
        'y_curve': model_underfit(X_curve),
        'color': '#d62728',
        'label': 'Underfitting model',
        'key': 'Underfit',
    },
    {
        # 'ax': axes[1],
        'y_curve': model_good(X_curve),
        'color': '#2ca02c',
        'label': 'Balanced model',
        'key': 'Good',
    },
    {
        # 'ax': axes[2],
        'y_curve': model_overfit(X_curve),
        'color': '#9467bd',
        'label': 'Overfitting model',
        'key': 'Overfit',
    },
]

for config in plot_configs:
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = plt.gca()

    # ax = config['ax']
    key = config['key']

    # Plot curves
    ax.plot(
        X_curve, config['y_curve'], color=config['color'], linewidth=2, label="Model"
    )  # , label=config['label'])

    # Plot points
    ax.scatter(
        X_train, y_train, color='lightgray', s=30, edgecolor='black', label='Train Data', zorder=5
    )
    ax.scatter(X_test, y_test, color='black', s=30, edgecolor='black', label='Test Data', zorder=5)

    # Format axes limits (locking Y prevents the overfit plot from zooming out to infinity)
    ax.set_xlim(X_MIN - 1, X_MAX + 1)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')

    # Create title with Error Metrics
    # Overfit test error can be massive, use scientific notation if it exceeds 9999
    test_err = mse_test[key]
    test_err_str = f"{test_err:.2e}" if test_err > 999999 else f"{test_err:.2f}"

    ax.set_xlabel("Model's input $x$")
    ax.set_ylabel("Model's output $y$")
    ax.set_title(
        f"Train MSE: {mse_train[key]:.2f}\n  Test MSE: {test_err_str}",
        fontsize=12,
        pad=10,
        loc='left',
    )

    plt.tight_layout()
    plt.savefig(
        f'/home/gerel/Documents/Manuscrit/bias_variance_tradeoff_{key}.pdf', bbox_inches='tight'
    )
    plt.show()
