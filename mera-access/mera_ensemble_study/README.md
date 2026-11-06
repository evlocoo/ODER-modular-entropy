# Eight-site MERA ensemble study

This package evaluates 64 random MERA encodings, their paired identity-disentangler trees, and 64 independent Haar-isometry controls. It asks separately whether contiguous access has a positive mean advantage, whether disentanglers increase that gap, and how the structured encodings compare with Haar controls.

Start with [the report](results/REPORT.md), [the descriptive statistics](results/descriptive_statistics.csv), or [the paired results](results/per_pair.csv). Figures display every encoding. The separate seed 20260918 appears only as a separately marked reference and regression check.

## Reproduce

Use Python 3.12. Install dependencies with `python -m pip install -r requirements.txt`, then run:

```sh
python study.py --output reproduction
```

This command generates the declared tensors, runs the ensemble invariant and acquisition calculations, performs the 20,000-resample analysis, and writes the figures and report. It processes seeds sequentially and checkpoints each completed seed. Use a new output directory; completed results are preserved.

For full fresh-process execution with the tolerance-aware companion comparison:

```sh
python ../notebook_support/verify_regeneration.py --output ../outputs/reproduction_check
```

The resource budget is 30 minutes and 2 GiB per study process, checked between seeds. A budget stop preserves completed seed checkpoints. `--resume` can continue a matching plan from verified checkpoints; an incomplete or failed seed directory remains available for diagnosis and is never silently replaced.

## Results and evidence

- `results/plan_before_execution.json`: complete seeds, generation conventions, estimands, resampling rules, tolerances, and failure policy.
- `results/encodings/`: tensors and hashes for every seed, complete subset scores, raw eigenvalues, density checks, and validation records.
- `results/per_encoding.csv`, `per_pair.csv`, and `all_subset_scores.csv`: all ensemble outcomes, including negative results.
- `results/descriptive_statistics.csv` and `structured_vs_haar.csv`: encoding-level summaries and approximate Monte Carlo intervals.
- `results/bootstrap_results.npz`: resampled means and the exact pair/Haar resampling indices.
- `results/exploratory_regression/` and `exploratory_comparison.csv`: excluded original-seed regression and descriptive placement.
- `inputs/` and `input_identity.json`: numerical methods, frozen tensors, generation conventions, and input hashes.

MERA/tree pairing is preserved in the analysis. Haar samples are independent. The overlapping subsets are not statistical replicates. The intervals describe the declared random-encoding distributions; this study does not optimize a physical ground state or establish an Ising benchmark.

The notebook also executes the frozen-pair causal-cone and exact-null checks. From the companion directory, they can be run separately with `python -m notebook_support.validation --output outputs/validation_check`.
