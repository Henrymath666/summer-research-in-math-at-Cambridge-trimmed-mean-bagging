#!/usr/bin/env python3
"""Target-accuracy simulations for finite-ensemble trimmed-mean bagging.

Every replication draws one exact population-bag ensemble and records

    |T_{alpha,B}(P,x) - f_P(x)|.

Thus the plotted (1-rho)-quantile is the empirical accuracy radius in
Definition 7, rather than a two-run stability radius.  The experiment
uses one preassigned reproducibility seed; no seeds are screened or
selected.  All Monte Carlo blocks and all models are generated from
independent child streams fixed before simulation.

The two models are genuinely different:

1. Cauchy(0,1) rows, one-bag sample mean, m=5, f_P=0.  By Cauchy
   stability, each population one-bag report is exactly Cauchy(0,1).
2. Symmetric-Lomax rows with P(|Z|>t)=(1+t)^(-3/4), one-bag sample
   median, m=3, f_P=0.  Its one-bag report has Pareto index 3/2.

Both use finite B=21 and the quantile-integral definition of
T_{alpha,B}; no n-to-infinity approximation is simulated internally.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = ("#202124", "#3973ac", "#e76f51")
DEFAULT_SEED = 20260801


def alpha_weight_matrix(bags: int, alphas: np.ndarray) -> np.ndarray:
    """Return the exact quantile-integral weights for T_{alpha,B}."""
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


def cauchy_bag_mean_reports(
    rng: np.random.Generator, replications: int, bags: int
) -> np.ndarray:
    """Draw exact population one-bag means for Cauchy(0,1) rows."""
    return rng.standard_cauchy((replications, bags))


def symmetric_lomax_bag_median_reports(
    rng: np.random.Generator,
    replications: int,
    bags: int,
    *,
    bag_size: int = 3,
    row_tail_index: float = 0.75,
) -> np.ndarray:
    """Draw exact population one-bag medians from symmetric Lomax rows."""
    uniforms = rng.random((replications, bags, bag_size))
    magnitudes = np.expm1(-np.log(uniforms) / row_tail_index)
    signs = 2.0 * rng.integers(
        0, 2, size=(replications, bags, bag_size)
    ) - 1.0
    rows = signs * magnitudes
    return np.partition(rows, bag_size // 2, axis=2)[:, :, bag_size // 2]


def simulate_accuracy_errors(
    rng: np.random.Generator,
    report_sampler: Callable[[np.random.Generator, int, int], np.ndarray],
    *,
    replications: int,
    bags: int,
    weights: np.ndarray,
    chunk_size: int,
    target: float = 0.0,
) -> np.ndarray:
    """Simulate |T_{alpha,B}(P,x)-target| for one population ensemble."""
    errors = np.empty((replications, weights.shape[1]), dtype=np.float64)
    for start in range(0, replications, chunk_size):
        stop = min(start + chunk_size, replications)
        reports = report_sampler(rng, stop - start, bags)
        aggregates = np.sort(reports, axis=1) @ weights
        errors[start:stop] = np.abs(aggregates - target)
    return errors


def empirical_radii(
    errors: np.ndarray,
    failure_probabilities: np.ndarray,
    blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pooled radii and the central 80% of preassigned blocks."""
    if errors.shape[0] % blocks != 0:
        raise ValueError("replications must be divisible by blocks")
    per_block = errors.shape[0] // blocks
    pooled = np.quantile(errors, 1.0 - failure_probabilities, axis=0)
    block_radii = np.empty(
        (blocks, failure_probabilities.size, errors.shape[1]),
        dtype=np.float64,
    )
    for block in range(blocks):
        start = block * per_block
        stop = start + per_block
        block_radii[block] = np.quantile(
            errors[start:stop], 1.0 - failure_probabilities, axis=0
        )
    lower = np.quantile(block_radii, 0.10, axis=0)
    upper = np.quantile(block_radii, 0.90, axis=0)
    return pooled, lower, upper


def alpha_label(alpha: float, bags: int) -> str:
    fully_trimmed = int(np.floor(alpha * bags + 1.0e-12))
    if fully_trimmed == 0:
        return r"$T_{0,B}$"
    return rf"$T_{{{fully_trimmed}/B,B}}$"


def draw_model_panel(
    ax: plt.Axes,
    errors: np.ndarray,
    *,
    alphas: np.ndarray,
    bags: int,
    blocks: int,
    failure_probabilities: np.ndarray,
    one_bag_tail_index: float,
    title: str,
    detail: str,
) -> dict[str, object]:
    """Plot target-accuracy radii and return numerical diagnostics."""
    radii, lower, upper = empirical_radii(
        errors, failure_probabilities, blocks
    )
    summary: dict[str, object] = {}
    anchor = int(np.argmin(np.abs(failure_probabilities - 0.02)))
    tail_region = failure_probabilities <= 0.04
    fit_region = (failure_probabilities >= 0.001) & (
        failure_probabilities <= 0.01
    )

    for alpha_index, (alpha, color) in enumerate(zip(alphas, COLORS)):
        ax.fill_between(
            failure_probabilities,
            lower[:, alpha_index],
            upper[:, alpha_index],
            color=color,
            alpha=0.13,
            linewidth=0.0,
        )
        ax.loglog(
            failure_probabilities,
            radii[:, alpha_index],
            color=color,
            linewidth=2.25,
            label=alpha_label(float(alpha), bags),
        )

        first_untrimmed_count = (
            int(np.floor(alpha * bags + 1.0e-12)) + 1
        )
        predicted_power = -1.0 / (
            first_untrimmed_count * one_bag_tail_index
        )
        guide = radii[anchor, alpha_index] * (
            failure_probabilities / failure_probabilities[anchor]
        ) ** predicted_power
        ax.loglog(
            failure_probabilities[tail_region],
            guide[tail_region],
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.60,
        )

        fitted_power = np.polyfit(
            np.log(failure_probabilities[fit_region]),
            np.log(radii[fit_region, alpha_index]),
            1,
        )[0]
        summary[f"alpha={alpha:.12g}"] = {
            "radius_at_rho_0.01": float(
                np.quantile(errors[:, alpha_index], 0.99)
            ),
            "radius_at_rho_0.001": float(
                np.quantile(errors[:, alpha_index], 0.999)
            ),
            "predicted_loglog_power": float(predicted_power),
            "fitted_loglog_power_rho_0.001_to_0.01": float(
                fitted_power
            ),
        }

    ax.set_xlabel(r"failure probability $\rho$")
    ax.set_ylabel("empirical accuracy threshold")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.grid(True, which="both", linewidth=0.55, alpha=0.28)
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0.025,
        0.035,
        detail + "\nDashed: predicted tail power",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#555555",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 1.5,
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate heavy-tail target-accuracy radii."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--replications", type=int, default=2_400_000)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replications % args.blocks != 0:
        raise ValueError("replications must be divisible by blocks")
    if args.replications <= 0 or args.blocks <= 0 or args.chunk_size <= 0:
        raise ValueError("replications, blocks, and chunk size must be positive")

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bags = 21
    cauchy_bag_size = 5
    lomax_bag_size = 3
    lomax_row_tail_index = 0.75
    lomax_one_bag_tail_index = 2.0 * lomax_row_tail_index
    alphas = np.asarray((0.0, 1.0 / bags, 3.0 / bags))
    weights = alpha_weight_matrix(bags, alphas)
    failure_probabilities = np.geomspace(1.0e-3, 0.20, 75)

    seed_sequence = np.random.SeedSequence(args.seed)
    cauchy_seed, lomax_seed = seed_sequence.spawn(2)

    plt.rcParams.update(
        {
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "legend.fontsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.35), constrained_layout=False
    )

    cauchy_errors = simulate_accuracy_errors(
        np.random.default_rng(cauchy_seed),
        cauchy_bag_mean_reports,
        replications=args.replications,
        bags=bags,
        weights=weights,
        chunk_size=args.chunk_size,
        target=0.0,
    )
    cauchy_summary = draw_model_panel(
        axes[0],
        cauchy_errors,
        alphas=alphas,
        bags=bags,
        blocks=args.blocks,
        failure_probabilities=failure_probabilities,
        one_bag_tail_index=1.0,
        title=rf"Cauchy location: bag mean ($m={cauchy_bag_size}$)",
        detail=r"Target $f_P=0$; one-bag index $1$",
    )
    del cauchy_errors

    def lomax_sampler(
        rng: np.random.Generator, replications: int, number_of_bags: int
    ) -> np.ndarray:
        return symmetric_lomax_bag_median_reports(
            rng,
            replications,
            number_of_bags,
            bag_size=lomax_bag_size,
            row_tail_index=lomax_row_tail_index,
        )

    lomax_errors = simulate_accuracy_errors(
        np.random.default_rng(lomax_seed),
        lomax_sampler,
        replications=args.replications,
        bags=bags,
        weights=weights,
        chunk_size=args.chunk_size,
        target=0.0,
    )
    lomax_summary = draw_model_panel(
        axes[1],
        lomax_errors,
        alphas=alphas,
        bags=bags,
        blocks=args.blocks,
        failure_probabilities=failure_probabilities,
        one_bag_tail_index=lomax_one_bag_tail_index,
        title=rf"Symmetric Lomax: bag median ($m={lomax_bag_size}$)",
        detail=(
            r"Target $f_P=0$; one-bag index $3/2$"
        ),
    )
    del lomax_errors

    fig.suptitle(
        rf"Heavy-tail target accuracy: exact population bags, finite $B={bags}$",
        fontsize=12.0,
        y=0.97,
    )
    layout = {
        "left": 0.075,
        "right": 0.985,
        "bottom": 0.15,
        "top": 0.82,
        "wspace": 0.16,
    }
    fig.set_layout_engine(None)
    fig.subplots_adjust(**layout)
    png_path = args.output_dir / "08_accuracy_heavy_tails_B21.png"
    fig.savefig(png_path, dpi=320)
    fig.subplots_adjust(**layout)
    plt.close(fig)

    runtime = time.perf_counter() - started
    summary = {
        "design": "exact population-bag target accuracy",
        "metric": "empirical (1-rho)-quantile of |T_{alpha,B}-f_P|",
        "seed": args.seed,
        "seed_selection": False,
        "B": bags,
        "alphas": alphas.tolist(),
        "replications_per_model": args.replications,
        "preassigned_independent_blocks": args.blocks,
        "failure_probability_range": [
            float(failure_probabilities.min()),
            float(failure_probabilities.max()),
        ],
        "models": {
            "cauchy_location_bag_mean": {
                "row_distribution": "standard Cauchy",
                "bag_size": cauchy_bag_size,
                "population_target": 0.0,
                "one_bag_tail_index": 1.0,
                "radii": cauchy_summary,
            },
            "symmetric_lomax_bag_median": {
                "absolute_tail": "P(|Z|>t)=(1+t)^(-3/4)",
                "bag_size": lomax_bag_size,
                "population_target": 0.0,
                "one_bag_tail_index": lomax_one_bag_tail_index,
                "radii": lomax_summary,
            },
        },
        "runtime_seconds": runtime,
        "outputs": {"png": str(png_path)},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
