# Result Index

This index identifies the available historical figures and their relationship to the retained source code. The artifact manifest contains a complete file inventory.

## Shared-coupling scalar LOO experiments

| Source script | Figure matching current defaults | Other retained historical variants |
| --- | --- | --- |
| [experiment1_gaussian_mean.py](../simulations/experiment1_gaussian_mean.py) | [experiment1_gaussian_mean_0.005.png](../simulations/experiment1_gaussian_mean_0.005.png) | [experiment1_gaussian_mean_0.02.png](../simulations/experiment1_gaussian_mean_0.02.png) |
| [experiment2_gaussian_median.py](../simulations/experiment2_gaussian_median.py) | [experiment2_gaussian_median_0.01.png](../simulations/experiment2_gaussian_median_0.01.png) | [experiment2_gaussian_median_0.02.png](../simulations/experiment2_gaussian_median_0.02.png) |
| [experiment3_gaussian_regression.py](../simulations/experiment3_gaussian_regression.py) | [experiment3_gaussian_regression_0.05.png](../simulations/experiment3_gaussian_regression_0.05.png) | [experiment3_gaussian_regression_0.02.png](../simulations/experiment3_gaussian_regression_0.02.png) |
| [experiment4_adversarial_median.py](../simulations/experiment4_adversarial_median.py) | [experiment4_adversarial_median_gamma_0.3_epsilon_0.05.png](../simulations/experiment4_adversarial_median_gamma_0.3_epsilon_0.05.png) | [experiment4_adversarial_median_0.02.png](../simulations/experiment4_adversarial_median_0.02.png) |
| [experiment5_cauchy_mean.py](../simulations/experiment5_cauchy_mean.py) | [experiment5_cauchy_mean_0.02.png](../simulations/experiment5_cauchy_mean_0.02.png) | [experiment5_cauchy_mean_0.1.png](../simulations/experiment5_cauchy_mean_0.1.png) |
| [experiment6_adversarial_mean.py](../simulations/experiment6_adversarial_mean.py) | Runs 1–20 (`experiment6_adversarial_mean_gamma_0.1_epsilon_0.05_run_*.png`) | [experiment6_adversarial_mean_gamma_0.1_epsilon_0.05.png](../simulations/experiment6_adversarial_mean_gamma_0.1_epsilon_0.05.png), [experiment6_adversarial_mean_gamma_0.2_epsilon_0.05.png](../simulations/experiment6_adversarial_mean_gamma_0.2_epsilon_0.05.png) |

The retained scripts reproduce their current configuration, not every historical parameter variant. Earlier tolerance/contamination variants have no separate historical source revision in this archive. Their filenames and embedded plot labels record the settings.

## Shared-coupling SGD experiment

[Source script](../sgd/experiment7_common_coupling_stability_n1000.py) currently uses B=250, n=1,000, m=50, d=10, alpha=0.10 and 300 Monte Carlo trials. It records 100 iterates, including initialization, hence 99 updates.

- [Current B=250 result](../sgd/experiment7_common_coupling_stability_radius_n1000_m50_B250_d10.png).
- [Historical B=200 result](../sgd/experiment7_common_coupling_stability_radius_n1000_m50_B200_d10.png). The corresponding historical code revision is unavailable.

The dataset is fixed across replications, and the deleted index and algorithmic randomness are resampled. The comparison uses equal update counts and different gradient costs.

## Target-accuracy experiments

| Script | PNG / PDF | Recorded numerical summary |
| --- | --- | --- |
| [heavy_tail_accuracy.py](../simulations/heavy_tail_accuracy.py) | [PNG](../simulations/heavy_tail_accuracy.png) / [PDF](../simulations/heavy_tail_accuracy.pdf) | [heavy_tail_accuracy_summary.json](../simulations/heavy_tail_accuracy_summary.json) |
| [target_accuracy_contamination.py](../simulations/target_accuracy_contamination.py) | [PNG](../simulations/target_accuracy_contamination.png) / [PDF](../simulations/target_accuracy_contamination.pdf) | [target_accuracy_contamination.json](../simulations/target_accuracy_contamination.json) |

The heavy-tail script draws independent population-bag ensembles and measures absolute error from the location target. The contamination script constructs finite datasets, samples fixed-size bags without replacement, and measures estimation error from a clean target. Its JSON includes exact finite-n failure probabilities and Monte Carlo standard errors. Neither protocol is the scalar shared-coupling LOO experiment.

## Historical independent-run experiments

[finite_B_stability.py](../simulations/finite_B_stability.py) explicitly uses independent ensembles in each paired comparison. These results are retained as historical experiments and do not establish shared-randomness leave-one-out stability. Its [JSON summary](../simulations/finite_B_summary.json) records the seed, replication counts, and comparison settings.

| Figure stem | PNG | PDF |
| --- | --- | --- |
| heavy_tail_expected_aloo | [PNG](../simulations/heavy_tail_expected_aloo.png) | [PDF](../simulations/heavy_tail_expected_aloo.pdf) |
| adversarial_fixed_m | [PNG](../simulations/adversarial_fixed_m.png) | [PDF](../simulations/adversarial_fixed_m.pdf) |
| huber_base_models | [PNG](../simulations/huber_base_models.png) | [PDF](../simulations/huber_base_models.pdf) |

## Additional historical figures

| Figure stem | PNG | PDF | Source availability |
| --- | --- | --- | --- |
| heavy_tail_finite_B | [PNG](../simulations/heavy_tail_finite_B.png) | [PDF](../simulations/heavy_tail_finite_B.pdf) | Exact generating source revision unavailable |
| contamination_finite_B | [PNG](../simulations/contamination_finite_B.png) | [PDF](../simulations/contamination_finite_B.pdf) | Exact generating source revision unavailable |

These earlier figure names are preserved for completeness. Do not assume they are generated by the current `finite_B_stability.py` or treat them as evidence of the final shared-coupling protocol without the matching original source.

## Preservation and validation

The archive contains 10 original Python scripts, 41 PNG figures, 7 figure PDFs and 3 numerical JSON summaries. It includes all 20 retained experiment-6 run figures. Python source and figure contents are unchanged. Only the two local absolute paths in `heavy_tail_accuracy_summary.json` were normalized; both original and packaged hashes are recorded in [artifact_manifest.json](artifact_manifest.json).

Packaging validation uses reduced simulation counts to exercise the code and rendering. It does not regenerate the historical full-run results. See [VALIDATION.md](VALIDATION.md) for the checks performed.
