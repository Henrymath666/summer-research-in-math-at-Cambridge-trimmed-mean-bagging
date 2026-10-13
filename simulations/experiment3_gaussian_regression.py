"""Experiment 3: Gaussian one-dimensional linear-regression stability.

Running this file creates only a PNG named with the current radius.
"""

import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trimmed_mean_mpl_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


N = 100
M = 20
B = 1_000
TRIALS = 50_000
RADIUS = 0.05
BETA_0 = 1.0
BETA_1 = 2.0
TEST_X = 0.5
SEED = 2026072603
BATCH_SIZE = 20

# The paper treats the covariates as fixed.
FIXED_Z = np.linspace(-1.0, 1.0, N)
ALPHAS = np.arange(0.0, 0.50, 0.01)
TRIM_COUNTS = np.rint(ALPHAS * B).astype(np.int64)
OUTPUT_PATH = Path(__file__).with_name(
    f"experiment3_gaussian_regression_{RADIUS:g}.png"
)


def sample_remaining_subbags(
    rng: np.random.Generator, batch_size: int
) -> np.ndarray:
    """Draw B independent uniform M-subsets from each set of N-1 rows."""
    random_keys = rng.random((batch_size, B, N - 1))
    return np.argpartition(random_keys, kth=M - 1, axis=2)[:, :, :M]


def split_at_deleted_row(
    data: np.ndarray, deleted_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = data.shape[0]
    keep = np.arange(N)[None, :] != deleted_index[:, None]
    remaining = data[keep].reshape(batch_size, N - 1)
    deleted = data[np.arange(batch_size), deleted_index]
    return remaining, deleted


def common_coupling(
    rng: np.random.Generator, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the exact fixed-size common coupling after deletion."""
    include_deleted = rng.random((batch_size, B)) < (M / N)
    replacement_position = rng.integers(0, M, size=(batch_size, B))
    return include_deleted, replacement_position


def regression_prediction(
    sum_z: np.ndarray,
    sum_y: np.ndarray,
    sum_z2: np.ndarray,
    sum_zy: np.ndarray,
) -> np.ndarray:
    denominator = M * sum_z2 - sum_z * sum_z
    slope = (M * sum_zy - sum_z * sum_y) / denominator
    return sum_y / M + slope * (TEST_X - sum_z / M)


def all_trimmed_means(outputs: np.ndarray) -> np.ndarray:
    ordered = np.sort(outputs, axis=1)
    prefix = np.pad(np.cumsum(ordered, axis=1), ((0, 0), (1, 0)))
    central_sums = prefix[:, B - TRIM_COUNTS] - prefix[:, TRIM_COUNTS]
    return central_sums / (B - 2 * TRIM_COUNTS)


def simulate() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    exceedances = np.zeros(ALPHAS.size, dtype=np.int64)

    for start in range(0, TRIALS, BATCH_SIZE):
        batch_size = min(BATCH_SIZE, TRIALS - start)
        z = np.broadcast_to(FIXED_Z, (batch_size, N))
        errors = rng.normal(0.0, 1.0, size=(batch_size, N))
        y = BETA_0 + BETA_1 * z + errors
        deleted_index = rng.integers(0, N, size=batch_size)
        remaining_z, deleted_z = split_at_deleted_row(z, deleted_index)
        remaining_y, deleted_y = split_at_deleted_row(y, deleted_index)

        bag_indices = sample_remaining_subbags(rng, batch_size)
        rows = np.arange(batch_size)[:, None, None]
        bag_z = remaining_z[rows, bag_indices]
        bag_y = remaining_y[rows, bag_indices]

        sum_z = bag_z.sum(axis=2)
        sum_y = bag_y.sum(axis=2)
        sum_z2 = np.square(bag_z).sum(axis=2)
        sum_zy = (bag_z * bag_y).sum(axis=2)
        deleted_outputs = regression_prediction(sum_z, sum_y, sum_z2, sum_zy)

        include_deleted, replacement_position = common_coupling(rng, batch_size)
        trial_rows = np.arange(batch_size)[:, None]
        bag_rows = np.arange(B)[None, :]
        removed_z = bag_z[trial_rows, bag_rows, replacement_position]
        removed_y = bag_y[trial_rows, bag_rows, replacement_position]

        full_sum_z = sum_z + include_deleted * (
            deleted_z[:, None] - removed_z
        )
        full_sum_y = sum_y + include_deleted * (
            deleted_y[:, None] - removed_y
        )
        full_sum_z2 = sum_z2 + include_deleted * (
            np.square(deleted_z[:, None]) - np.square(removed_z)
        )
        full_sum_zy = sum_zy + include_deleted * (
            deleted_z[:, None] * deleted_y[:, None] - removed_z * removed_y
        )
        full_outputs = regression_prediction(
            full_sum_z, full_sum_y, full_sum_z2, full_sum_zy
        )

        differences = np.abs(
            all_trimmed_means(full_outputs)
            - all_trimmed_means(deleted_outputs)
        )
        exceedances += np.sum(differences > RADIUS, axis=0)

    return exceedances / TRIALS


def save_graph(delta: np.ndarray) -> None:
    standard_error = np.sqrt(delta * (1.0 - delta) / TRIALS)
    lower = np.clip(delta - 1.96 * standard_error, 0.0, 1.0)
    upper = np.clip(delta + 1.96 * standard_error, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    color = "#059669"
    ax.fill_between(
        ALPHAS, lower, upper, color=color, alpha=0.16, label="95% Monte Carlo CI"
    )
    ax.plot(
        ALPHAS,
        delta,
        color=color,
        linewidth=2.4,
        marker="o",
        markersize=4.8,
        label=r"Empirical $\delta$",
    )
    ax.set_title(
        "Experiment 3: Gaussian one-dimensional linear regression\n"
        f"n={N}, m={M}, B={B:,}, trials={TRIALS:,}, x={TEST_X:g}, "
        rf"fixed radius $\epsilon={RADIUS:.3f}$"
    )
    ax.set_xlabel(r"Trimming level $\alpha$")
    ax.set_ylabel(
        r"$\widehat{\delta}=\Pr\!\left("
        r"|T_{\alpha,B}(D)-T_{\alpha,B}(D_{\backslash i})|>\epsilon\right)$"
    )
    ax.set_xlim(ALPHAS[0], ALPHAS[-1])
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.49])
    ax.set_ylim(0.0, min(1.0, max(0.05, float(upper.max()) * 1.18)))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    save_graph(simulate())
