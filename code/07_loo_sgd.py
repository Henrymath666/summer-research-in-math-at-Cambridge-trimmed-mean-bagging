#!/usr/bin/env python3
r"""Common-coupling stability-radius comparison for linear regression.

Methods:
1. Coordinatewise alpha-trimmed B-bag SGD, final iterate.
2. Sampling-with-replacement SGD, final iterate (Hardt).
3. Bassily sampling-with-replacement SGD on R^d, with the
   step-size-weighted averaged iterate as output. Projection onto R^d
   is the identity and is therefore not evaluated explicitly.

The dataset D is fixed. Each Monte Carlo replication redraws
I ~ Unif([n]) and the common-coupled algorithmic randomness.

The plotted empirical stability radius is

    r_hat(delta)
      = inf{r : empirical P(||A(D)-A(D\I)||_2 > r) <= delta}.

This is a same-update-count comparison, not a gradient-budget-matched
comparison. One trimmed update evaluates B*m gradients, whereas one
ordinary SGD update evaluates one gradient.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------

N = 1_000
D = 10
M_BAG = 50

# Matches the preceding regression simulations.
B = 250
ALPHA = 0.10

# T iterates means T-1 stochastic updates, matching Bassily's notation.
T = 100

# Start with 100-200 for a quick test. Use at least 1,000 for the final
# figure. Reliable empirical tails require roughly delta >= 5/TRIALS.
TRIALS = 300
MC_BATCH_SIZE = 2

RADIUS_MIN = 0.001
RADIUS_MAX = 0.01

RIDGE = 0.0
NOISE_STD = 1.0
DATA_SEED = 2026080601
MC_SEED = 2026080602

OUTPUT_STEM = (
    f"07_loo_sgd_n{N}_d{D}_m{M_BAG}_B{B}"
    + f"_alpha_{ALPHA:g}".replace(".", "p")
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------

def trimmed_weights(bags: int, alpha: float) -> np.ndarray:
    """Quantile-integral empirical alpha-trimmed-mean weights."""
    lower = np.arange(bags, dtype=np.float64) / bags
    upper = np.arange(1, bags + 1, dtype=np.float64) / bags

    weights = np.maximum(
        0.0,
        np.minimum(upper, 1.0 - alpha) - np.maximum(lower, alpha),
    )
    weights /= 1.0 - 2.0 * alpha

    if not np.isclose(weights.sum(), 1.0):
        raise AssertionError("Trimmed-mean weights do not sum to one.")
    return weights


TRIM_WEIGHTS = trimmed_weights(B, ALPHA)


def coordinatewise_trimmed_mean(reports: np.ndarray) -> np.ndarray:
    """Trim B reports coordinatewise.

    reports has shape (replications, B, d).
    """
    ordered = np.sort(reports, axis=1)
    return np.einsum(
        "rbd,b->rd",
        ordered,
        TRIM_WEIGHTS,
        optimize=True,
    )


def draw_ordered_fixed_size_bags(
    rng: np.random.Generator,
    replications: int,
    bags: int,
    bag_size: int,
    population_size: int,
) -> np.ndarray:
    """Draw exact uniform fixed-size samples without replacement.

    The rejection construction samples ordered tuples uniformly
    conditional on all coordinates being distinct. Since the bag
    gradient is symmetric, this is equivalent to a uniform subset.
    """
    if bag_size > population_size:
        raise ValueError("bag_size cannot exceed population_size")

    total = replications * bags
    output = np.empty((total, bag_size), dtype=np.int32)
    pending = np.arange(total)

    while pending.size:
        candidate = rng.integers(
            0,
            population_size,
            size=(pending.size, bag_size),
            dtype=np.int32,
        )
        ordered = np.sort(candidate, axis=1)
        accepted = np.all(
            ordered[:, 1:] != ordered[:, :-1],
            axis=1,
        )
        output[pending[accepted]] = candidate[accepted]
        pending = pending[~accepted]

    return output.reshape(replications, bags, bag_size)


def point_gradients(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """All per-observation ridge-square-loss gradients.

    theta has shape (replications, d).
    Output has shape (replications, n, d).
    """
    residuals = theta @ x.T - y[None, :]
    return (
        residuals[:, :, None] * x[None, :, :]
        + RIDGE * theta[:, None, :]
    )


def one_sample_gradients(
    theta: np.ndarray,
    indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """One stochastic gradient per replication."""
    selected_x = x[indices]
    residual = (
        np.einsum("rd,rd->r", selected_x, theta)
        - y[indices]
    )
    return residual[:, None] * selected_x + RIDGE * theta


# ---------------------------------------------------------------------
# Exact common couplings
# ---------------------------------------------------------------------

def coupled_one_sample_indices(
    rng: np.random.Generator,
    deleted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximal deletion coupling for one-sample SGD."""
    replications = deleted.size

    full_indices = rng.integers(
        0,
        N,
        size=replications,
        dtype=np.int32,
    )

    replacement_local = rng.integers(
        0,
        N - 1,
        size=replications,
        dtype=np.int32,
    )
    replacements = replacement_local + (
        replacement_local >= deleted
    )

    deleted_indices = np.where(
        full_indices == deleted,
        replacements,
        full_indices,
    ).astype(np.int32)

    return full_indices, deleted_indices


def coupled_trimmed_reports(
    rng: np.random.Generator,
    deleted: np.ndarray,
    theta_full: np.ndarray,
    theta_deleted: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the common-coupled B bag-gradient reports.

    We first sample each deleted-data bag uniformly from the n-1
    retained rows. Independently with probability m/n, one uniformly
    selected retained row is replaced by the deleted row to construct
    the full-data bag. This is exactly the common-permutation coupling.
    """
    replications = deleted.size

    # Local indices 0,...,n-2 in the leave-one-out dataset.
    local_bags = draw_ordered_fixed_size_bags(
        rng,
        replications,
        B,
        M_BAG,
        N - 1,
    )

    # Map local leave-one-out indices back to original-data indices.
    deleted_bags = local_bags + (
        local_bags >= deleted[:, None, None]
    )
    deleted_bags = deleted_bags.astype(np.int32)

    # Under the common coupling, a full-data bag contains I with m/n.
    include_deleted = (
        rng.random((replications, B)) < (M_BAG / N)
    )
    replacement_position = rng.integers(
        0,
        M_BAG,
        size=(replications, B),
    )

    full_point_gradients = point_gradients(
        theta_full,
        x,
        y,
    )
    deleted_point_gradients = point_gradients(
        theta_deleted,
        x,
        y,
    )

    trial_rows = np.arange(replications)[:, None, None]

    # Both coupled bags initially use the retained rows.
    full_reports = full_point_gradients[
        trial_rows,
        deleted_bags,
    ].mean(axis=2)

    deleted_reports = deleted_point_gradients[
        trial_rows,
        deleted_bags,
    ].mean(axis=2)

    # If the full bag contains I, replace one retained observation.
    trial_matrix = np.arange(replications)[:, None]
    bag_matrix = np.arange(B)[None, :]

    removed_indices = deleted_bags[
        trial_matrix,
        bag_matrix,
        replacement_position,
    ]
    removed_gradients = full_point_gradients[
        trial_matrix,
        removed_indices,
    ]
    inserted_gradients = full_point_gradients[
        np.arange(replications),
        deleted,
    ][:, None, :]

    full_reports += (
        include_deleted[:, :, None]
        * (inserted_gradients - removed_gradients)
        / M_BAG
    )

    return full_reports, deleted_reports


# ---------------------------------------------------------------------
# Monte Carlo experiment
# ---------------------------------------------------------------------

def simulate(
    x: np.ndarray,
    y: np.ndarray,
    eta: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(MC_SEED)

    trimmed_distances = np.empty(TRIALS)
    hardt_distances = np.empty(TRIALS)
    bassily_distances = np.empty(TRIALS)

    for start in range(0, TRIALS, MC_BATCH_SIZE):
        stop = min(start + MC_BATCH_SIZE, TRIALS)
        replications = stop - start

        deleted = rng.integers(
            0,
            N,
            size=replications,
            dtype=np.int32,
        )

        # Same initialization within every coupled pair.
        trim_full = np.zeros((replications, D))
        trim_deleted = np.zeros((replications, D))

        # Hardt: ordinary last-iterate SGD, without projection.
        hardt_full = np.zeros((replications, D))
        hardt_deleted = np.zeros((replications, D))

        # Bassily output on R^d: projection is the identity, and the
        # returned parameter is the step-size-weighted iterate average.
        bassily_full = np.zeros((replications, D))
        bassily_deleted = np.zeros((replications, D))

        bassily_full_sum = eta * bassily_full.copy()
        bassily_deleted_sum = eta * bassily_deleted.copy()
        bassily_weight_sum = eta

        # T iterates x_1,...,x_T correspond to T-1 updates.
        for _ in range(T - 1):
            # Proposed coordinatewise trimmed-bag method.
            full_reports, deleted_reports = coupled_trimmed_reports(
                rng,
                deleted,
                trim_full,
                trim_deleted,
                x,
                y,
            )

            trim_full_gradient = coordinatewise_trimmed_mean(
                full_reports
            )
            trim_deleted_gradient = coordinatewise_trimmed_mean(
                deleted_reports
            )

            trim_full -= eta * trim_full_gradient
            trim_deleted -= eta * trim_deleted_gradient

            # The same coupled sampled indices are used for the Hardt
            # and Bassily trajectories.
            full_indices, deleted_indices = (
                coupled_one_sample_indices(rng, deleted)
            )

            # Hardt last-iterate SGD.
            hardt_full -= eta * one_sample_gradients(
                hardt_full,
                full_indices,
                x,
                y,
            )
            hardt_deleted -= eta * one_sample_gradients(
                hardt_deleted,
                deleted_indices,
                x,
                y,
            )

            # The same SGD update, with a weighted-average output.
            bassily_full -= eta * one_sample_gradients(
                bassily_full,
                full_indices,
                x,
                y,
            )
            bassily_deleted -= eta * one_sample_gradients(
                bassily_deleted,
                deleted_indices,
                x,
                y,
            )

            bassily_full_sum += eta * bassily_full
            bassily_deleted_sum += eta * bassily_deleted
            bassily_weight_sum += eta

        bassily_full_output = (
            bassily_full_sum / bassily_weight_sum
        )
        bassily_deleted_output = (
            bassily_deleted_sum / bassily_weight_sum
        )

        trimmed_distances[start:stop] = np.linalg.norm(
            trim_full - trim_deleted,
            axis=1,
        )
        hardt_distances[start:stop] = np.linalg.norm(
            hardt_full - hardt_deleted,
            axis=1,
        )
        bassily_distances[start:stop] = np.linalg.norm(
            bassily_full_output - bassily_deleted_output,
            axis=1,
        )

        if stop % 50 == 0 or stop == TRIALS:
            print(f"completed {stop:,}/{TRIALS:,} replications")

    return {
        "trimmed": trimmed_distances,
        "hardt": hardt_distances,
        "bassily": bassily_distances,
    }


def empirical_survival_curve(
    distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return radii and empirical strict-tail probabilities."""
    ordered = np.sort(distances)
    sample_size = ordered.size

    radii = np.unique(np.concatenate(([0.0], ordered)))
    failures = (
        sample_size
        - np.searchsorted(ordered, radii, side="right")
    ) / sample_size
    return radii, failures


def save_results(
    distances: dict[str, np.ndarray],
) -> None:
    colors = {
        "trimmed": "#059669",
        "hardt": "#2563eb",
        "bassily": "#dc6b2f",
    }
    labels = {
        "trimmed": (
            rf"Trimmed bags: $m={M_BAG}$, $B={B}$, "
            rf"$\alpha={ALPHA:g}$ (final)"
        ),
        "hardt": "One-sample SGD (Hardt final iterate)",
        "bassily": "One-sample SGD (Bassily weighted average)",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.4, 6.0))

    for name in ("trimmed", "hardt", "bassily"):
        radii, failures = empirical_survival_curve(distances[name])
        ax.step(
            radii,
            failures,
            where="post",
            color=colors[name],
            linewidth=2.4,
            label=labels[name],
        )

    ax.set_xlim(RADIUS_MIN, RADIUS_MAX)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel(r"stability radius $r$")
    ax.set_ylabel(
        r"empirical failure probability "
        r"$\widehat{\Pr}\{\Delta_T>r\}$"
    )
    ax.set_title(
        "Common-coupling parameter stability in Gaussian linear regression\n"
        f"n={N:,}, d={D}, m={M_BAG}, B={B:,}, "
        f"T={T}, trials={TRIALS:,}"
    )

    ax.grid(True, which="both", alpha=0.27)
    ax.legend(frameon=False, fontsize=9)

    ax.text(
        0.02,
        0.025,
        "Same update count, not equal gradient work: "
        f"trimmed={B * M_BAG:,} gradients/update; SGD=1.",
        transform=ax.transAxes,
        fontsize=8.3,
        color="#555555",
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{OUTPUT_STEM}.png",
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    data_rng = np.random.default_rng(DATA_SEED)

    # Standard Gaussian linear-regression design: x_i ~ N(0, I_d).
    x = data_rng.normal(size=(N, D))

    beta_true = data_rng.normal(size=D)
    beta_true /= np.linalg.norm(beta_true)

    noise = data_rng.normal(
        0.0,
        NOISE_STD,
        size=N,
    )
    y = x @ beta_true + noise

    # Each squared loss has Hessian x_i x_i^T.
    smoothness = (
        np.max(np.sum(np.square(x), axis=1)) + RIDGE
    )

    # Satisfies eta < 2/M. The same eta is used for all methods.
    eta = 0.5 / smoothness

    print(f"n={N}, d={D}, m={M_BAG}, B={B}")
    print(f"alpha={ALPHA}, T={T}, trials={TRIALS}")
    print(f"smoothness upper bound={smoothness:.6g}")
    print(f"step size eta={eta:.6g}")
    print(
        "trimmed gradient evaluations per update="
        f"{B * M_BAG:,}"
    )

    distances = simulate(x, y, eta)
    save_results(distances)

    print("\nEmpirical probability masses at radius zero:")
    for name, values in distances.items():
        print(
            f"  {name:9s}: "
            f"{np.mean(values == 0.0):.4f}"
        )

    print("\nCreated:")
    print(OUTPUT_DIR / f"{OUTPUT_STEM}.png")


if __name__ == "__main__":
    main()
