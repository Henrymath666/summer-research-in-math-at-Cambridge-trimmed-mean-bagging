# Packaging Validation

Validated on 13 October 2026. All checks below passed.

## Environment

- Python: 3.12.14
- NumPy: 2.5.4
- SciPy: 1.18.1
- Matplotlib: 3.11.2

These are the packaging-check versions, not a claim about the original historical environment.

## Checks

- Compiled and imported all ten original Python scripts.
- Verified all 41 archived PNGs decode and all JSON summaries parse.
- Matched every archived artifact against its packaged SHA-256 hash.
- Verified repository-relative links in README and the result index.
- Executed all six scalar LOO simulations at 20 trials each, retaining their original n, m, B and alpha grids; checked probability shapes and ranges, and generated a figure from each.
- Executed the SGD comparison at 2 trials and 7 updates, retaining its original n, d, m, B and alpha; checked distances and generated its figure.
- Executed heavy_tail_accuracy.py at 12,000 replications per model, 12 blocks and chunk size 1,000; generated PNG, PDF and JSON.
- Executed target_accuracy_contamination.py at 1,200 replications, 12 blocks and batch size 100; generated PNG, PDF and JSON, including the built-in target-error versus bag-hit identities.
- Executed finite_B_stability.py at 16,000 heavy-tail replications, 16,000 Cauchy replications, 2,000 contamination replications and 1,000 Huber-model replications; generated PNGs, PDFs and JSON.

Historical full-run figures and numerical values were preserved rather than regenerated. Reduced runs confirm execution and plotting; they do not validate the full-run estimates or theoretical claims. Smoke-run outputs are excluded from the repository.
