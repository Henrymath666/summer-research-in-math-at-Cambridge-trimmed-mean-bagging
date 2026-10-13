#!/usr/bin/env python3
"""Simulations for the heavy-tail and contamination stability results.

The two procedure calls in every comparison use independent ensembles.
The command-line seed makes the complete experiment reproducible; it is
never reused to couple the two runs being compared.
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


def alpha_trimmed_mean(values: np.ndarray, alpha: float) -> np.ndarray:
    """Evaluate the quantile-integral definition of T_{alpha,B}."""
    ordered = np.sort(values, axis=-1)
    bags = ordered.shape[-1]
    lower = np.arange(bags, dtype=np.float64) / bags
    upper = np.arange(1, bags + 1, dtype=np.float64) / bags
    weights = np.maximum(
        0.0,
        np.minimum(upper, 1.0 - alpha) - np.maximum(lower, alpha),
    )
    weights /= 1.0 - 2.0 * alpha
    return np.sum(ordered * weights, axis=-1)


def alpha_weight_matrix(bags: int, alphas: np.ndarray) -> np.ndarray:
    """Return one quantile-integral weight vector for every alpha."""
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


def heavy_tail_figure(
    seed_sequence: np.random.SeedSequence,
    output_dir: Path,
    *,
    replications: int,
    batches: int,
    cauchy_replications: int,
) -> dict[str, object]:
    bags = 21
    bag_size = 5
    large_n = 1_000_000_000
    alphas = (0.0, 1.0 / bags, 3.0 / bags)
    deltas = np.geomspace(5.0e-4, 0.2, 75)
    if replications % batches != 0:
        raise ValueError("heavy-tail replications must be divisible by batches")
    per_batch = replications // batches
    spawned_seeds = seed_sequence.spawn(len(alphas) + 1)
    alpha_seeds = spawned_seeds[: len(alphas)]

    fig, axes = plt.subplots(
        1, 2, figsize=(10.9, 4.25), constrained_layout=True
    )
    ax = axes[0]
    summary: dict[str, object] = {
        "B": bags,
        "m": bag_size,
        "comparison_n": large_n,
        "coupling_error_bound": float(
            (2 * bags) * (2 * bags - 1) / 2
            * bag_size**2
            / (large_n - 1)
        ),
        "replications": replications,
        "independent_batches": batches,
        "radii": {},
    }

    for alpha, color, alpha_seed in zip(alphas, COLORS, alpha_seeds):
        rng = np.random.default_rng(alpha_seed)
        batch_radii = []
        all_differences = []
        for _ in range(batches):
            first_uniform = rng.random((per_batch, bags))
            second_uniform = rng.random((per_batch, bags))
            first = 1.0 / (1.0 - first_uniform ** (1.0 / bag_size))
            second = 1.0 / (1.0 - second_uniform ** (1.0 / bag_size))
            differences = np.abs(
                alpha_trimmed_mean(first, alpha)
                - alpha_trimmed_mean(second, alpha)
            )
            all_differences.append(differences)
            batch_radii.append(np.quantile(differences, 1.0 - deltas))

        combined = np.concatenate(all_differences)
        radii = np.quantile(combined, 1.0 - deltas)
        batch_radii_array = np.asarray(batch_radii)
        lower = np.quantile(batch_radii_array, 0.10, axis=0)
        upper = np.quantile(batch_radii_array, 0.90, axis=0)
        ax.fill_between(deltas, lower, upper, color=color, alpha=0.13)
        ax.loglog(
            deltas,
            radii,
            color=color,
            linewidth=2.25,
            label=rf"$\alpha={alpha:.3f}$",
        )

        first_untrimmed_count = int(np.floor(alpha * bags + 1.0e-12)) + 1
        anchor_delta = 0.02
        anchor_index = int(np.argmin(np.abs(deltas - anchor_delta)))
        slope_guide = radii[anchor_index] * (
            deltas / deltas[anchor_index]
        ) ** (-1.0 / first_untrimmed_count)
        ax.loglog(
            deltas,
            slope_guide,
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.60,
        )

        fit_mask = (deltas >= 0.001) & (deltas <= 0.01)
        fitted_slope = np.polyfit(
            np.log(deltas[fit_mask]), np.log(radii[fit_mask]), 1
        )[0]
        summary["radii"][f"alpha={alpha:.12g}"] = {
            "radius_at_delta_0.01": float(np.quantile(combined, 0.99)),
            "fitted_loglog_slope": float(fitted_slope),
            "predicted_slope": -1.0 / first_untrimmed_count,
        }

    ax.set_xlabel(r"failure probability $\delta$")
    ax.set_ylabel("stability threshold")
    ax.set_title(r"Pareto data and bag maximum")
    ax.invert_xaxis()
    ax.grid(True, which="both", linewidth=0.55, alpha=0.28)
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0.02,
        0.03,
        "Central 80% across preassigned batches",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#555555",
    )

    cauchy_bags = 31
    cauchy_bag_size = 5
    cauchy_large_n = 1_000_000_000
    cauchy_alphas = np.asarray(
        (0.0, 1.0 / cauchy_bags, 5.0 / cauchy_bags)
    )
    cauchy_deltas = np.geomspace(1.0e-3, 0.2, 75)
    if cauchy_replications % batches != 0:
        raise ValueError("Cauchy replications must be divisible by batches")
    cauchy_per_batch = cauchy_replications // batches
    cauchy_weights = alpha_weight_matrix(cauchy_bags, cauchy_alphas)
    cauchy_differences = np.empty(
        (cauchy_replications, cauchy_alphas.size), dtype=np.float64
    )
    cauchy_rng = np.random.default_rng(spawned_seeds[-1])
    for batch_index in range(batches):
        start = batch_index * cauchy_per_batch
        stop = start + cauchy_per_batch
        # The mean of m independent standard Cauchy observations is again
        # standard Cauchy, so these are exact population bag-mean outputs.
        first = cauchy_rng.standard_cauchy(
            (cauchy_per_batch, cauchy_bags)
        )
        second = cauchy_rng.standard_cauchy(
            (cauchy_per_batch, cauchy_bags)
        )
        cauchy_differences[start:stop] = np.abs(
            np.sort(first, axis=1) @ cauchy_weights
            - np.sort(second, axis=1) @ cauchy_weights
        )

    ax = axes[1]
    cauchy_summary: dict[str, object] = {}
    for alpha_index, (alpha, color) in enumerate(
        zip(cauchy_alphas, COLORS)
    ):
        differences = cauchy_differences[:, alpha_index]
        radii = np.quantile(differences, 1.0 - cauchy_deltas)
        batch_radii = np.asarray(
            [
                np.quantile(
                    cauchy_differences[
                        batch_index * cauchy_per_batch :
                        (batch_index + 1) * cauchy_per_batch,
                        alpha_index,
                    ],
                    1.0 - cauchy_deltas,
                )
                for batch_index in range(batches)
            ]
        )
        lower = np.quantile(batch_radii, 0.10, axis=0)
        upper = np.quantile(batch_radii, 0.90, axis=0)
        ax.fill_between(
            cauchy_deltas, lower, upper, color=color, alpha=0.13
        )
        ax.loglog(
            cauchy_deltas,
            radii,
            color=color,
            linewidth=2.25,
            label=rf"$\alpha={alpha:.3f}$",
        )
        first_untrimmed_count = (
            int(np.floor(alpha * cauchy_bags + 1.0e-12)) + 1
        )
        cauchy_summary[f"alpha={alpha:.12g}"] = {
            "radius_at_delta_0.01": float(
                np.quantile(differences, 0.99)
            ),
            "radius_at_delta_0.001": float(
                np.quantile(differences, 0.999)
            ),
            "predicted_delta_power": -1.0 / first_untrimmed_count,
        }

    ax.set_xlabel(r"failure probability $\delta$")
    ax.set_ylabel("stability threshold")
    ax.set_title(r"Cauchy data and bag mean")
    ax.invert_xaxis()
    ax.grid(True, which="both", linewidth=0.55, alpha=0.28)
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0.02,
        0.03,
        "Central 80% across preassigned batches",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#555555",
    )

    summary["cauchy_bag_mean_population"] = {
        "B": cauchy_bags,
        "m": cauchy_bag_size,
        "comparison_n": cauchy_large_n,
        "coupling_error_bound": float(
            (2 * cauchy_bags) * (2 * cauchy_bags - 1) / 2
            * cauchy_bag_size**2
            / (cauchy_large_n - 1)
        ),
        "replications": cauchy_replications,
        "independent_batches": batches,
        "radii": cauchy_summary,
    }

    fig.suptitle("Two different heavy-tailed mechanisms", fontsize=12)
    for extension in ("pdf", "png"):
        fig.savefig(
            output_dir / f"heavy_tail_expected_aloo.{extension}",
            dpi=320,
            bbox_inches="tight",
        )
    plt.close(fig)
    return summary


def fixed_set_hit_probability(
    sample_sizes: np.ndarray, bag_size: int, changed_rows: int
) -> np.ndarray:
    probability_no_hit = np.ones_like(sample_sizes, dtype=np.float64)
    for offset in range(bag_size):
        probability_no_hit *= (
            sample_sizes - changed_rows - offset
        ) / (sample_sizes - offset)
    return 1.0 - probability_no_hit


def adversarial_figure(
    seed_sequence: np.random.SeedSequence,
    output_dir: Path,
    *,
    replications: int,
) -> dict[str, object]:
    bags = 21
    bag_size = 5
    changed_rows = 1
    alphas = (0.0, 1.0 / bags, 2.0 / bags)
    alpha_seeds = seed_sequence.spawn(2 * len(alphas))

    fig, axes = plt.subplots(
        1, 2, figsize=(10.8, 4.15), constrained_layout=True
    )

    sample_sizes = np.unique(
        np.geomspace(1.0e3, 1.0e7, 240).astype(np.int64)
    ).astype(np.float64)
    hit_probabilities = fixed_set_hit_probability(
        sample_sizes, bag_size, changed_rows
    )
    fixed_points = np.asarray(
        [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000],
        dtype=np.float64,
    )
    fixed_point_hits = fixed_set_hit_probability(
        fixed_points, bag_size, changed_rows
    )

    ax = axes[0]
    fixed_summary: dict[str, object] = {}
    for alpha, color, alpha_seed in zip(alphas, COLORS, alpha_seeds[:3]):
        fully_removed = int(np.floor(alpha * bags + 1.0e-12))
        curve = binom.sf(fully_removed, bags, hit_probabilities)
        ax.loglog(
            sample_sizes,
            curve,
            color=color,
            linewidth=2.25,
            label=rf"$\alpha={alpha:.3f}$",
        )

        rng = np.random.default_rng(alpha_seed)
        empirical = []
        exact = binom.sf(fully_removed, bags, fixed_point_hits)
        for hit_probability in fixed_point_hits:
            counts = rng.binomial(bags, hit_probability, size=replications)
            empirical.append(float(np.mean(counts > fully_removed)))
        empirical_array = np.asarray(empirical)
        visible = (exact * replications >= 20.0) & (
            (1.0 - exact) * replications >= 20.0
        )
        ax.scatter(
            fixed_points[visible],
            empirical_array[visible],
            color=color,
            s=24,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        exact_at_reference = float(
            binom.sf(fully_removed, bags, bag_size / 100_000.0)
        )
        fixed_summary[f"alpha={alpha:.12g}"] = {
            "failure_at_n_100000": exact_at_reference,
            "predicted_n_power": -(fully_removed + 1),
        }

    ax.set_xlabel(r"sample size $n$ (one changed row)")
    ax.set_ylabel("failure probability")
    ax.set_title(r"Fixed $m$-out-of-$n$: polynomial separation")
    ax.grid(True, which="both", linewidth=0.55, alpha=0.28)
    ax.legend(frameon=False, fontsize=9)

    gammas = np.geomspace(1.0e-5, 0.10, 240)
    huber_hits = 1.0 - (1.0 - gammas) ** bag_size
    gamma_points = np.asarray(
        [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.08]
    )
    gamma_point_hits = 1.0 - (1.0 - gamma_points) ** bag_size

    ax = axes[1]
    huber_summary: dict[str, object] = {}
    for alpha, color, alpha_seed in zip(alphas, COLORS, alpha_seeds[3:]):
        fully_removed = int(np.floor(alpha * bags + 1.0e-12))
        curve = binom.sf(fully_removed, bags, huber_hits)
        ax.loglog(
            gammas,
            curve,
            color=color,
            linewidth=2.25,
            label=rf"$\alpha={alpha:.3f}$",
        )

        rng = np.random.default_rng(alpha_seed)
        empirical = []
        exact = binom.sf(fully_removed, bags, gamma_point_hits)
        for hit_probability in gamma_point_hits:
            counts = rng.binomial(bags, hit_probability, size=replications)
            empirical.append(float(np.mean(counts > fully_removed)))
        empirical_array = np.asarray(empirical)
        visible = (exact * replications >= 20.0) & (
            (1.0 - exact) * replications >= 20.0
        )
        ax.scatter(
            gamma_points[visible],
            empirical_array[visible],
            color=color,
            s=24,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        reference_gamma = 0.001
        reference_hit = 1.0 - (1.0 - reference_gamma) ** bag_size
        huber_summary[f"alpha={alpha:.12g}"] = {
            "failure_at_gamma_0.001": float(
                binom.sf(fully_removed, bags, reference_hit)
            ),
            "predicted_small_gamma_power": fully_removed + 1,
        }

    ax.set_xlabel(r"Huber row-contamination probability $\gamma$")
    ax.set_ylabel("failure probability")
    ax.set_title(r"Large-$n$ Huber contamination")
    ax.grid(True, which="both", linewidth=0.55, alpha=0.28)
    ax.legend(frameon=False, fontsize=9)

    for extension in ("pdf", "png"):
        fig.savefig(
            output_dir / f"adversarial_fixed_m.{extension}",
            dpi=320,
            bbox_inches="tight",
        )
    plt.close(fig)

    return {
        "B": bags,
        "m": bag_size,
        "changed_rows": changed_rows,
        "replications_per_marker": replications,
        "fixed_m_adversarial": fixed_summary,
        "huber": huber_summary,
    }


def huber_model_figure(
    seed_sequence: np.random.SeedSequence,
    output_dir: Path,
    *,
    replications: int,
) -> dict[str, object]:
    """Compare two genuinely different base algorithms under Huber noise."""
    rng = np.random.default_rng(seed_sequence)
    bags = 31
    bag_size = 3
    contamination_value = 20.0
    alphas = np.asarray((0.0, 0.05, 0.15))
    weights = alpha_weight_matrix(bags, alphas)
    gammas = np.asarray(
        (0.0, 0.01, 0.025, 0.05, 0.075, 0.10,
         0.125, 0.15, 0.175, 0.20, 0.30, 0.40)
    )
    plotted = gammas <= 0.20
    radius_quantile = 0.95
    batch_size = 5_000
    independent_blocks = 5
    if replications % independent_blocks != 0:
        raise ValueError(
            "Huber-model replications must be divisible by five"
        )
    per_block = replications // independent_blocks
    model_radii = {
        "bag mean": np.empty((gammas.size, alphas.size)),
        "bag median": np.empty((gammas.size, alphas.size)),
    }
    model_block_radii = {
        "bag mean": np.empty(
            (gammas.size, independent_blocks, alphas.size)
        ),
        "bag median": np.empty(
            (gammas.size, independent_blocks, alphas.size)
        ),
    }

    for gamma_index, gamma in enumerate(gammas):
        differences = {
            "bag mean": np.empty((replications, alphas.size)),
            "bag median": np.empty((replications, alphas.size)),
        }
        for start in range(0, replications, batch_size):
            stop = min(start + batch_size, replications)
            current = stop - start
            clean_rows = rng.uniform(-1.0, 1.0, (current, bags, bag_size))
            contaminated_rows = rng.uniform(
                -1.0, 1.0, (current, bags, bag_size)
            )
            contamination_mask = (
                rng.random((current, bags, bag_size)) < gamma
            )
            contaminated_rows[contamination_mask] = contamination_value

            clean_mean = clean_rows.mean(axis=2)
            contaminated_mean = contaminated_rows.mean(axis=2)
            clean_median = np.median(clean_rows, axis=2)
            contaminated_median = np.median(contaminated_rows, axis=2)

            differences["bag mean"][start:stop] = np.abs(
                np.sort(clean_mean, axis=1) @ weights
                - np.sort(contaminated_mean, axis=1) @ weights
            )
            differences["bag median"][start:stop] = np.abs(
                np.sort(clean_median, axis=1) @ weights
                - np.sort(contaminated_median, axis=1) @ weights
            )

        for model_name in model_radii:
            model_radii[model_name][gamma_index] = np.quantile(
                differences[model_name], radius_quantile, axis=0
            )
            for block_index in range(independent_blocks):
                block_start = block_index * per_block
                block_stop = block_start + per_block
                model_block_radii[model_name][
                    gamma_index, block_index
                ] = np.quantile(
                    differences[model_name][block_start:block_stop],
                    radius_quantile,
                    axis=0,
                )

    fig, axes = plt.subplots(
        1, 2, figsize=(10.8, 4.05), constrained_layout=True
    )
    for ax, model_name in zip(axes, ("bag mean", "bag median")):
        for alpha_index, (alpha, color, radii) in enumerate(zip(
            alphas, COLORS, model_radii[model_name].T
        )):
            block_radii = model_block_radii[model_name][
                :, :, alpha_index
            ]
            lower = np.quantile(block_radii, 0.10, axis=1)
            upper = np.quantile(block_radii, 0.90, axis=1)
            ax.fill_between(
                gammas[plotted],
                lower[plotted],
                upper[plotted],
                color=color,
                alpha=0.13,
            )
            ax.plot(
                gammas[plotted],
                radii[plotted],
                color=color,
                linewidth=2.2,
                label=rf"$\alpha={alpha:.2f}$",
            )
        ax.set_xlabel(r"Huber contamination probability $\gamma$")
        ax.set_ylabel("95th percentile stability threshold")
        ax.set_title(model_name.capitalize())
        ax.grid(True, linewidth=0.55, alpha=0.28)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        r"Uniform clean data, point contamination ($B=31$, $m=3$)",
        fontsize=12,
    )
    for extension in ("pdf", "png"):
        fig.savefig(
            output_dir / f"huber_base_models.{extension}",
            dpi=320,
            bbox_inches="tight",
        )
    plt.close(fig)

    selected_radii: dict[str, object] = {}
    for gamma in (0.0, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        gamma_index = int(np.argmin(np.abs(gammas - gamma)))
        selected_radii[f"gamma={gamma:.3g}"] = {
            "bag_mean": model_radii["bag mean"][gamma_index].tolist(),
            "bag_median": model_radii["bag median"][gamma_index].tolist(),
        }
    return {
        "B": bags,
        "m": bag_size,
        "comparison_n": 1_000_000_000,
        "coupling_error_bound": float(
            (2 * bags) * (2 * bags - 1) / 2
            * bag_size**2
            / (1_000_000_000 - 1)
        ),
        "clean_distribution": "Uniform(-1,1)",
        "contamination_value": contamination_value,
        "replications_per_gamma": replications,
        "independent_blocks": independent_blocks,
        "radius_quantile": radius_quantile,
        "alphas": alphas.tolist(),
        "selected_radii": selected_radii,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--heavy-replications", type=int, default=4_000_000)
    parser.add_argument("--heavy-batches", type=int, default=16)
    parser.add_argument(
        "--cauchy-replications", type=int, default=2_000_000
    )
    parser.add_argument("--contamination-replications", type=int, default=400_000)
    parser.add_argument(
        "--huber-model-replications", type=int, default=100_000
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heavy_seed, adversarial_seed, huber_model_seed = np.random.SeedSequence(
        args.seed
    ).spawn(3)
    summary = {
        "seed": args.seed,
        "independent_runs": True,
        "heavy_tail": heavy_tail_figure(
            heavy_seed,
            args.output_dir,
            replications=args.heavy_replications,
            batches=args.heavy_batches,
            cauchy_replications=args.cauchy_replications,
        ),
        "adversarial_and_huber": adversarial_figure(
            adversarial_seed,
            args.output_dir,
            replications=args.contamination_replications,
        ),
        "nondegenerate_huber_models": huber_model_figure(
            huber_model_seed,
            args.output_dir,
            replications=args.huber_model_replications,
        ),
    }
    summary_path = args.output_dir / "finite_B_summary.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
