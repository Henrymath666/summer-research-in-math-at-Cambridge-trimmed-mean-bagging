"""Experiment 2: empirical leave-one-out stability of Gaussian bag medians.

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
M = 21
B = 1000
TRIALS = 50_000
RADIUS = 0.010
THETA = 0.0
SEED = 2026072602
BATCH_SIZE = 20

ALPHAS = np.arange(0.0, 0.50, 0.01)
TRIM_COUNTS = np.rint(ALPHAS * B).astype(np.int64)
OUTPUT_PATH = Path(__file__).with_name(
    f"experiment2_gaussian_median_{RADIUS:g}.png"
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


def all_trimmed_means(outputs: np.ndarray) -> np.ndarray:
    ordered = np.sort(outputs, axis=1)
    prefix = np.pad(np.cumsum(ordered, axis=1), ((0, 0), (1, 0)))
    central_sums = prefix[:, B - TRIM_COUNTS] - prefix[:, TRIM_COUNTS]
    return central_sums / (B - 2 * TRIM_COUNTS)


def simulate() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    exceedances = np.zeros(ALPHAS.size, dtype=np.int64)
    middle = M // 2

    for start in range(0, TRIALS, BATCH_SIZE):
        batch_size = min(BATCH_SIZE, TRIALS - start)
        data = rng.normal(THETA, 1.0, size=(batch_size, N))
        deleted_index = rng.integers(0, N, size=batch_size)
        remaining, deleted_value = split_at_deleted_row(data, deleted_index)

        bag_indices = sample_remaining_subbags(rng, batch_size)
        rows = np.arange(batch_size)[:, None, None]
        bag_values = remaining[rows, bag_indices]
        deleted_outputs = np.partition(bag_values, middle, axis=2)[:, :, middle]

        include_deleted, replacement_position = common_coupling(rng, batch_size)
        full_bag_values = bag_values.copy()
        trial_rows = np.arange(batch_size)[:, None]
        bag_rows = np.arange(B)[None, :]
        old_value = full_bag_values[
            trial_rows, bag_rows, replacement_position
        ]
        full_bag_values[trial_rows, bag_rows, replacement_position] = np.where(
            include_deleted, deleted_value[:, None], old_value
        )
        full_outputs = np.partition(
            full_bag_values, middle, axis=2
        )[:, :, middle]

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
    color = "#7C3AED"
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
        "Experiment 2: Gaussian bag median\n"
        f"n={N}, m={M}, B={B:,}, trials={TRIALS:,}, fixed radius "
        rf"$\epsilon={RADIUS:.3f}$"
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
