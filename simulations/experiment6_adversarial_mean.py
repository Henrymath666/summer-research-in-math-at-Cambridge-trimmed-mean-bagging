"""Experiment 6: finite-B adversarial contamination for bagged means.

This matches Experiment 4 except that each bag returns its mean instead of its
median. Running this file creates only a PNG named with gamma and epsilon.
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
B = 500
TRIALS = 10_000
K = 20  # Number of independent runs/PNG graphs to produce.
RADIUS = 0.05
THETA = 0.0
CONTAMINATION_FRACTION = 0.10
CONTAMINATED_ROWS = int(CONTAMINATION_FRACTION * N)
ADVERSARIAL_VALUE = THETA + 10.0
BASE_SEED = 2026072606
BATCH_SIZE = 20

ALPHAS = np.arange(0.0, 0.50, 0.01)
TRIM_COUNTS = np.rint(ALPHAS * B).astype(np.int64)


def output_path(run_number: int) -> Path:
    return Path(__file__).with_name(
        "experiment6_adversarial_mean_"
        f"gamma_{CONTAMINATION_FRACTION:g}_epsilon_{RADIUS:g}_"
        f"run_{run_number}.png"
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
    """Compute the alpha-trimmed aggregate of the B bag means."""
    ordered = np.sort(outputs, axis=1)
    prefix = np.pad(np.cumsum(ordered, axis=1), ((0, 0), (1, 0)))
    central_sums = prefix[:, B - TRIM_COUNTS] - prefix[:, TRIM_COUNTS]
    return central_sums / (B - 2 * TRIM_COUNTS)


def simulate(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    exceedances = np.zeros(ALPHAS.size, dtype=np.int64)

    for start in range(0, TRIALS, BATCH_SIZE):
        batch_size = min(BATCH_SIZE, TRIALS - start)
        data = rng.uniform(THETA - 1.0, THETA + 1.0, size=(batch_size, N))
        # By exchangeability, fixing which rows the adversary replaces is
        # equivalent here to randomly labelling exactly the same number.
        data[:, :CONTAMINATED_ROWS] = ADVERSARIAL_VALUE
        deleted_index = rng.integers(0, N, size=batch_size)
        remaining, deleted_value = split_at_deleted_row(data, deleted_index)

        bag_indices = sample_remaining_subbags(rng, batch_size)
        rows = np.arange(batch_size)[:, None, None]
        bag_values = remaining[rows, bag_indices]
        deleted_outputs = bag_values.mean(axis=2)

        include_deleted, replacement_position = common_coupling(rng, batch_size)
        trial_rows = np.arange(batch_size)[:, None]
        bag_rows = np.arange(B)[None, :]
        removed_value = bag_values[
            trial_rows, bag_rows, replacement_position
        ]
        full_outputs = deleted_outputs + include_deleted * (
            deleted_value[:, None] - removed_value
        ) / M

        differences = np.abs(
            all_trimmed_means(full_outputs)
            - all_trimmed_means(deleted_outputs)
        )
        exceedances += np.sum(differences > RADIUS, axis=0)

    return exceedances / TRIALS


def save_graph(delta: np.ndarray, run_number: int, seed: int) -> None:
    standard_error = np.sqrt(delta * (1.0 - delta) / TRIALS)
    lower = np.clip(delta - 1.96 * standard_error, 0.0, 1.0)
    upper = np.clip(delta + 1.96 * standard_error, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    color = "#0F766E"
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
        "Experiment 6: adversarial contamination for bagged means\n"
        f"n={N}, m={M}, B={B:,}, trials={TRIALS:,}, "
        f"run={run_number}/{K}, seed={seed}\n"
        rf"$\gamma={CONTAMINATION_FRACTION:.2f}$, replacement="
        rf"{ADVERSARIAL_VALUE:g}, fixed radius $\epsilon={RADIUS:.3f}$"
    )
    ax.set_xlabel(r"Trimming level $\alpha$")
    ax.set_ylabel(
        r"$\widehat{\delta}=\Pr\!\left("
        r"|T_{\alpha,B}(D)-T_{\alpha,B}(D_{\backslash i})|>\epsilon\right)$"
    )
    ax.set_xlim(ALPHAS[0], ALPHAS[-1])
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.49])
    y_max = min(1.0, max(0.05, float(upper.max()) * 1.18))
    ax.set_ylim(0.0, y_max)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path(run_number), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if K < 1:
        raise ValueError("K must be at least 1.")

    for run_number in range(1, K + 1):
        run_seed = BASE_SEED + run_number - 1
        save_graph(simulate(run_seed), run_number, run_seed)
