#!/usr/bin/env python3
"""Target-accuracy simulations under adversarial and Huber contamination.

This script is deliberately separate from ``finite_B_stability.py``.  It
plots the empirical target-error survival function

    P(|T_{alpha,B} - f_P| > tau)

for two nondegenerate clean models.  Every bag samples exactly m distinct
rows out of n, uniformly without replacement.  Bags are independent
conditional on the finite data set.

At the vertical reference line in each panel, the model constants make the
target-accuracy failure event exactly equal to the event that more than
floor(alpha B) bags meet contaminated rows.  The script verifies this
identity in every Monte Carlo replication and compares the empirical value
with the corresponding exact finite-n probability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import binom

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = ("#202124", "#3973ac", "#e76f51")


def alpha_weight_matrix(bags: int, alphas: np.ndarray) -> np.ndarray:
    """Return quantile-integral weights for each trimming level."""
    lower = np.arange(bags, dtype=np.float64) / bags
    upper = np.arange(1, bags + 1, dtype=np.float64) / bags
    columns = []
    for alpha in alphas:
        weights = np.maximum(
            0.0,
            np.minimum(upper, 1.0 - alpha) - np.maximum(lower, alpha),
        )
        columns.append(weights / (1.0 - 2.0 * alpha))
    return np.stack(columns, axis=1)


def draw_fixed_size_bags(
    rng: np.random.Generator,
    replications: int,
    bags: int,
    bag_size: int,
    sample_size: int,
) -> np.ndarray:
    """Draw independent ordered uniform m-subsets for every replication.

    Each new index is drawn uniformly until it differs from the preceding
    indices in its bag.  Thus each ordered tuple is uniform over all tuples
    of distinct indices, which is equivalent to uniform fixed-size sampling
    without replacement for the symmetric base algorithms used below.
    """
    indices = np.empty(
        (replications, bags, bag_size), dtype=np.int32
    )
    for position in range(bag_size):
        candidate = rng.integers(
            0,
            sample_size,
            size=(replications, bags),
            dtype=np.int32,
        )
        if position:
            duplicate = np.any(
                candidate[..., None] == indices[..., :position], axis=-1
            )
            while np.any(duplicate):
                candidate[duplicate] = rng.integers(
                    0,
                    sample_size,
                    size=int(np.sum(duplicate)),
                    dtype=np.int32,
                )
                duplicate = np.any(
                    candidate[..., None] == indices[..., :position],
                    axis=-1,
                )
        indices[..., position] = candidate
    return indices


def aggregate_reports(
    reports: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Evaluate all requested coordinatewise alpha-trimmed means."""
    return np.sort(reports, axis=1) @ weights


def simulate_adversarial_regression(
    seed_sequences: list[np.random.SeedSequence],
    *,
    replications: int,
    blocks: int,
    batch_size: int,
    bags: int,
    bag_size: int,
    sample_size: int,
    alphas: np.ndarray,
    weights: np.ndarray,
    target: float,
    replacement_cross_moment: float,
    theorem_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one adversarial high-leverage regression replacement."""
    per_block = replications // blocks
    errors = np.empty((replications, alphas.size), dtype=np.float64)
    hit_counts = np.empty(replications, dtype=np.int16)

    for block_index, block_seed in enumerate(seed_sequences):
        rng = np.random.default_rng(block_seed)
        block_start = block_index * per_block
        for local_start in range(0, per_block, batch_size):
            current = min(batch_size, per_block - local_start)
            start = block_start + local_start
            stop = start + current

            signs = rng.choice(
                np.asarray((-1.0, 1.0)), size=(current, sample_size)
            )
            noise = rng.uniform(-1.0, 1.0, (current, sample_size))
            responses = target * signs + noise
            cross_moments = signs * responses

            # After drawing the clean data, replace row zero by (X,Y)=(1,M).
            cross_moments[:, 0] = replacement_cross_moment

            indices = draw_fixed_size_bags(
                rng, current, bags, bag_size, sample_size
            )
            rows = np.arange(current)[:, None, None]
            reports = cross_moments[rows, indices].mean(axis=2)
            estimates = aggregate_reports(reports, weights)
            errors[start:stop] = np.abs(estimates - target)
            hit_counts[start:stop] = np.sum(
                np.any(indices == 0, axis=2), axis=1
            )

    for alpha_index, alpha in enumerate(alphas):
        removed = int(np.floor(alpha * bags + 1.0e-12))
        observed = errors[:, alpha_index] > theorem_threshold
        predicted = hit_counts > removed
        if not np.array_equal(observed, predicted):
            mismatches = int(np.sum(observed != predicted))
            raise AssertionError(
                "adversarial target-error identity failed in "
                f"{mismatches} replications for alpha={alpha}"
            )
    return errors, hit_counts


def simulate_huber_second_moment(
    seed_sequences: list[np.random.SeedSequence],
    *,
    replications: int,
    blocks: int,
    batch_size: int,
    bags: int,
    bag_size: int,
    sample_size: int,
    alphas: np.ndarray,
    weights: np.ndarray,
    target: float,
    contamination_probability: float,
    replacement_value: float,
    theorem_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate Huber row contamination for a second-moment target."""
    per_block = replications // blocks
    errors = np.empty((replications, alphas.size), dtype=np.float64)
    hit_counts = np.empty(replications, dtype=np.int16)
    marked_counts = np.empty(replications, dtype=np.int16)

    for block_index, block_seed in enumerate(seed_sequences):
        rng = np.random.default_rng(block_seed)
        block_start = block_index * per_block
        for local_start in range(0, per_block, batch_size):
            current = min(batch_size, per_block - local_start)
            start = block_start + local_start
            stop = start + current

            observations = rng.uniform(
                -1.0, 1.0, (current, sample_size)
            )
            marked = (
                rng.random((current, sample_size))
                < contamination_probability
            )
            observations[marked] = replacement_value
            squared_observations = observations**2

            indices = draw_fixed_size_bags(
                rng, current, bags, bag_size, sample_size
            )
            rows = np.arange(current)[:, None, None]
            reports = squared_observations[rows, indices].mean(axis=2)
            estimates = aggregate_reports(reports, weights)
            errors[start:stop] = np.abs(estimates - target)
            hit_counts[start:stop] = np.sum(
                np.any(marked[rows, indices], axis=2), axis=1
            )
            marked_counts[start:stop] = np.sum(marked, axis=1)

    for alpha_index, alpha in enumerate(alphas):
        removed = int(np.floor(alpha * bags + 1.0e-12))
        observed = errors[:, alpha_index] > theorem_threshold
        predicted = hit_counts > removed
        if not np.array_equal(observed, predicted):
            mismatches = int(np.sum(observed != predicted))
            raise AssertionError(
                "Huber target-error identity failed in "
                f"{mismatches} replications for alpha={alpha}"
            )
    return errors, hit_counts, marked_counts


def finite_n_huber_failure(
    *,
    sample_size: int,
    bag_size: int,
    bags: int,
    gamma: float,
    removed: int,
) -> float:
    """Evaluate the exact finite-n Huber bag-hit mixture."""
    marked = np.arange(sample_size + 1, dtype=np.float64)
    miss_probability = np.ones(sample_size + 1, dtype=np.float64)
    for offset in range(bag_size):
        numerator = np.maximum(sample_size - marked - offset, 0.0)
        miss_probability *= numerator / (sample_size - offset)
    hit_probability = 1.0 - miss_probability
    return float(
        np.sum(
            binom.pmf(marked, sample_size, gamma)
            * binom.sf(removed, bags, hit_probability)
        )
    )


def empirical_survival(
    errors: np.ndarray,
    thresholds: np.ndarray,
    blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full-sample survival and central 80% block envelopes."""
    replications, alpha_count = errors.shape
    per_block = replications // blocks
    survival = np.empty((alpha_count, thresholds.size))
    block_survival = np.empty((blocks, alpha_count, thresholds.size))
    for alpha_index in range(alpha_count):
        survival[alpha_index] = np.mean(
            errors[:, alpha_index, None] > thresholds[None, :], axis=0
        )
        for block_index in range(blocks):
            start = block_index * per_block
            stop = start + per_block
            block_survival[block_index, alpha_index] = np.mean(
                errors[start:stop, alpha_index, None]
                > thresholds[None, :],
                axis=0,
            )
    lower = np.quantile(block_survival, 0.10, axis=0)
    upper = np.quantile(block_survival, 0.90, axis=0)
    return survival, lower, upper


def theorem_point_summary(
    errors: np.ndarray,
    *,
    threshold: float,
    alphas: np.ndarray,
    bags: int,
    exact: list[float],
    blocks: int,
) -> dict[str, object]:
    """Summarize empirical and exact failures at the theorem threshold."""
    replications = errors.shape[0]
    per_block = replications // blocks
    result: dict[str, object] = {}
    for alpha_index, (alpha, exact_value) in enumerate(zip(alphas, exact)):
        indicators = errors[:, alpha_index] > threshold
        block_values = np.asarray(
            [
                np.mean(
                    indicators[
                        block_index * per_block :
                        (block_index + 1) * per_block
                    ]
                )
                for block_index in range(blocks)
            ]
        )
        empirical = float(np.mean(indicators))
        standard_error = float(
            np.sqrt(exact_value * (1.0 - exact_value) / replications)
        )
        result[f"alpha={alpha:.12g}"] = {
            "floor_alpha_B": int(
                np.floor(alpha * bags + 1.0e-12)
            ),
            "empirical_failure": empirical,
            "exact_finite_n_failure": float(exact_value),
            "monte_carlo_standard_error": standard_error,
            "central_80_percent_across_blocks": [
                float(np.quantile(block_values, 0.10)),
                float(np.quantile(block_values, 0.90)),
            ],
            "absolute_error_from_exact": abs(empirical - exact_value),
        }
    return result


def create_figure(
    output_dir: Path,
    *,
    seed: int,
    replications: int,
    blocks: int,
    batch_size: int,
) -> dict[str, object]:
    bags = 21
    bag_size = 5
    alphas = np.asarray((0.0, 1.0 / bags, 3.0 / bags))
    weights = alpha_weight_matrix(bags, alphas)
    thresholds = np.geomspace(0.02, 15.0, 150)

    adversarial_sample_size = 100
    regression_target = 2.0
    replacement_cross_moment = 500.0
    adversarial_theorem_threshold = 3.0

    huber_sample_size = 400
    second_moment_target = 1.0 / 3.0
    contamination_probability = 0.01
    replacement_value = 16.0
    huber_theorem_threshold = 5.0 / 3.0

    master_seed = np.random.SeedSequence(seed)
    spawned = master_seed.spawn(2 * blocks)
    adversarial_seeds = spawned[:blocks]
    huber_seeds = spawned[blocks:]

    adversarial_errors, adversarial_hits = (
        simulate_adversarial_regression(
            adversarial_seeds,
            replications=replications,
            blocks=blocks,
            batch_size=batch_size,
            bags=bags,
            bag_size=bag_size,
            sample_size=adversarial_sample_size,
            alphas=alphas,
            weights=weights,
            target=regression_target,
            replacement_cross_moment=replacement_cross_moment,
            theorem_threshold=adversarial_theorem_threshold,
        )
    )
    huber_errors, huber_hits, marked_counts = simulate_huber_second_moment(
        huber_seeds,
        replications=replications,
        blocks=blocks,
        batch_size=batch_size,
        bags=bags,
        bag_size=bag_size,
        sample_size=huber_sample_size,
        alphas=alphas,
        weights=weights,
        target=second_moment_target,
        contamination_probability=contamination_probability,
        replacement_value=replacement_value,
        theorem_threshold=huber_theorem_threshold,
    )

    adversarial_survival = empirical_survival(
        adversarial_errors, thresholds, blocks
    )
    huber_survival = empirical_survival(huber_errors, thresholds, blocks)

    adversarial_exact = [
        float(
            binom.sf(
                int(np.floor(alpha * bags + 1.0e-12)),
                bags,
                bag_size / adversarial_sample_size,
            )
        )
        for alpha in alphas
    ]
    huber_exact = [
        finite_n_huber_failure(
            sample_size=huber_sample_size,
            bag_size=bag_size,
            bags=bags,
            gamma=contamination_probability,
            removed=int(np.floor(alpha * bags + 1.0e-12)),
        )
        for alpha in alphas
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(11.1, 4.25), constrained_layout=True
    )
    panels = (
        (
            axes[0],
            adversarial_survival,
            adversarial_theorem_threshold,
            adversarial_exact,
            r"One adversarial row ($n=100$)",
            (
                r"$X\in\{-1,1\}$, $Y=2X+\varepsilon$, "
                r"$f_P=\mathrm{E}[XY]=2$"
            ),
        ),
        (
            axes[1],
            huber_survival,
            huber_theorem_threshold,
            huber_exact,
            r"Huber contamination ($n=400$, $\gamma=0.01$)",
            (
                r"$Z\sim\mathrm{Unif}[-1,1]$, "
                r"$f_P=\mathrm{E}[Z^2]=1/3$"
            ),
        ),
    )

    plotting_floor = 0.5 / (replications // blocks)
    for ax, survival_data, reference, exact_values, title, model_text in panels:
        survival, lower, upper = survival_data
        for alpha_index, (alpha, color) in enumerate(zip(alphas, COLORS)):
            ax.fill_between(
                thresholds,
                np.maximum(lower[alpha_index], plotting_floor),
                np.maximum(upper[alpha_index], plotting_floor),
                color=color,
                alpha=0.13,
            )
            ax.plot(
                thresholds,
                np.maximum(survival[alpha_index], 0.5 / replications),
                color=color,
                linewidth=2.25,
                label=(
                    r"$T_{0,B}$"
                    if alpha_index == 0
                    else rf"$T_{{{int(round(alpha * bags))}/B,B}}$"
                ),
            )
            ax.scatter(
                [reference],
                [exact_values[alpha_index]],
                color=color,
                marker="x",
                s=52,
                linewidth=1.7,
                zorder=5,
            )

        ax.axvline(
            reference,
            color="#777777",
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
        )
        ax.text(
            reference * 1.18,
            0.92,
            r"clean radius $+R$",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8.2,
            color="#666666",
        )
        ax.text(
            0.025,
            0.035,
            model_text + "\n" + r"$\times$: exact finite-$n$ value",
            transform=ax.transAxes,
            fontsize=8.1,
            color="#555555",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(thresholds[0], thresholds[-1])
        ax.set_ylim(0.5 / replications, 1.05)
        ax.set_xlabel(r"target-accuracy threshold $\tau$")
        ax.set_ylabel("empirical target-error probability")
        ax.set_title(title)
        ax.grid(True, which="both", linewidth=0.55, alpha=0.28)
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.suptitle(
        r"Exact finite-$n$ target-accuracy checks under contamination "
        r"($B=21$, exact $m=5$ out of $n$)",
        fontsize=12,
    )
    for extension in ("pdf", "png"):
        fig.savefig(
            output_dir / f"target_accuracy_contamination.{extension}",
            dpi=320,
            bbox_inches="tight",
        )
    plt.close(fig)

    summary: dict[str, object] = {
        "seed": seed,
        "replications": replications,
        "independent_blocks": blocks,
        "B": bags,
        "m": bag_size,
        "alphas": alphas.tolist(),
        "fixed_m_sampling": (
            "Each bag samples exactly m distinct rows uniformly without "
            "replacement; bags are conditionally independent."
        ),
        "adversarial_regression_moment": {
            "n": adversarial_sample_size,
            "clean_model": (
                "X is Rademacher; Y=2X+epsilon; "
                "epsilon is Uniform(-1,1)."
            ),
            "base_algorithm": "bag average of X*Y",
            "target": regression_target,
            "changed_rows": 1,
            "replacement": "(X,Y)=(1,500)",
            "clean_one_bag_box": [1.0, 3.0],
            "clean_radius": 1.0,
            "box_width_R": 2.0,
            "theorem_target_threshold": adversarial_theorem_threshold,
            "target_failure_equals_hit_tail_in_every_replication": True,
            "mean_bag_hits": float(np.mean(adversarial_hits)),
            "failure_at_theorem_threshold": theorem_point_summary(
                adversarial_errors,
                threshold=adversarial_theorem_threshold,
                alphas=alphas,
                bags=bags,
                exact=adversarial_exact,
                blocks=blocks,
            ),
        },
        "huber_second_moment": {
            "n": huber_sample_size,
            "clean_model": "Z is Uniform(-1,1).",
            "base_algorithm": "bag average of Z^2",
            "target": second_moment_target,
            "gamma": contamination_probability,
            "replacement_value": replacement_value,
            "clean_one_bag_box": [0.0, 1.0],
            "clean_radius": 2.0 / 3.0,
            "box_width_R": 1.0,
            "theorem_target_threshold": huber_theorem_threshold,
            "target_failure_equals_hit_tail_in_every_replication": True,
            "mean_marked_rows": float(np.mean(marked_counts)),
            "mean_bag_hits": float(np.mean(huber_hits)),
            "failure_at_theorem_threshold": theorem_point_summary(
                huber_errors,
                threshold=huber_theorem_threshold,
                alphas=alphas,
                bags=bags,
                exact=huber_exact,
                blocks=blocks,
            ),
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--replications", type=int, default=240_000)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replications % args.blocks != 0:
        raise ValueError("replications must be divisible by blocks")
    if (args.replications // args.blocks) % args.batch_size != 0:
        raise ValueError("replications per block must be divisible by batch size")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = create_figure(
        args.output_dir,
        seed=args.seed,
        replications=args.replications,
        blocks=args.blocks,
        batch_size=args.batch_size,
    )
    summary_path = args.output_dir / "target_accuracy_contamination.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
