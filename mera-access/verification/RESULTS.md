# Results and reproducibility

Both collection protocols acquire three of eight sites and include each site with probability 3/8. Their difference is which sites arrive together. The recorded ensemble contains 64 MERA encodings, their paired identity-disentangler trees, and 64 independent Haar controls.

| Ensemble | Mean contiguous score | Mean uniform score | Mean gap |
| --- | ---: | ---: | ---: |
| MERA | 0.298845 | 0.253736 | +0.045110 |
| Paired tree | 0.348675 | 0.327693 | +0.020983 |
| Independent Haar | 0.133276 | 0.133782 | -0.000506 |

All 64 MERA gaps are positive. The MERA gap exceeds its paired-tree gap in 59 pairs and is smaller in five. Removing the disentanglers reverses the protocol preference in three pairs. The notebook explains these results alongside acquisition levels, complementary access, and the collection-sufficiency criterion.

## Execution checks

The notebook passed all 14 code cells in a fresh Windows Jupyter kernel in saved-results mode. It reconstructs the worked ensemble pair, recomputes ensemble and bootstrap summaries from all 10,752 subset rows, and executes the frozen exploratory pair's architecture and exact-null checks. Saved-results mode does not regenerate the full 64-pair ensemble.

| Check | Executed scope | Largest residual |
| --- | --- | ---: |
| Causal-cone ascent versus full matrices | 33 observables for each frozen encoder; all three encoding implementations | 6.062e-16 |
| Tensor, encoding, state, and regression invariants | Both frozen encoders; unchanged isometries and identity tree gates | 2.189e-13 |
| Reference depolarization | All 256 boundary subsets for each encoder; boundary marginal preserved | 4.419e-14 |
| Complement-balanced four-site access | All 70 subsets and eight cyclic windows for each encoder | 1.088e-14 |
| Boundary permutation average | All 40,320 permutations; each triple occurs exactly 5,760 times | 0.000e+00 |
| Full ensemble comparison (supplied Windows regeneration) | 91,000 arrays and 88,852 density-state records per dataset | 0.000e+00 |
| Independent-process reproduction | 2,037 arrays compared with fresh notebook execution | 0.000e+00 |

All 22 MERA tests passed: 15 comparator tests, five tests of ordered periodic legs, structural support, identity gates, and partial traces, and two tests of independent output directories and fresh-kernel saved-results execution that preserves supplied files. The numerical tolerance remains `1e-10`; operational sign resolution remains `1e-8`. Raw eigenvalues retain every positive value in entropy sums, with only tolerated negative values excluded. No input tensors or physical parameters are adjusted.

The 66 causal-cone contraction checks use complete logical 2×2 operators. The ascending path contracts only structurally retained tensors and cancels the others by isometry or unitarity; it does not construct the full encoding matrix. Per-observable support traces, numerical operators, exact-null scores, and raw eigenvalues are in [finite_checks/](finite_checks/). [reproduction.json](finite_checks/reproduction.json) records the separate-process comparison. [saved_execution.json](saved_execution.json) records notebook analysis, and [notebook_execution.json](notebook_execution.json) records fresh-kernel completion.

## Reproduce

From `Black_Hole_Companion/`, enter the MERA companion:

```sh
cd mera-access
```

Then run:

```sh
python -m notebook_support.validation --output outputs/validation_check
python -m unittest notebook_support.test_verifier notebook_support.test_validation
```

The supplied [regeneration record](current_regeneration.json) reports the full-ensemble comparison. To regenerate and compare:

```sh
python notebook_support/verify_regeneration.py --output outputs/reproduction
```

Each command-line output target must be new. The notebook executes the finite checks in both saved-results and regeneration modes. Full regeneration evaluates the ensemble again and compares the generated arrays with the supplied reference results.

The comparator requires exact scientific identifiers, seeds, counts, labels, schemas, shapes, and data types. Numerical comparisons use the declared `1e-10` tolerance. Raw eigenvalue arrays are compared numerically; negative-eigenvalue counts are checked against each run's own finite eigenvalues and aggregate total. Values below `-1e-10` fail validation. Tensor and checkpoint hashes are verified within each run. The reference source hash must match the supplied sampling plan; a regenerated run must record the hash of the distributed `study.py`. The scientific plan fields remain exact.

The [MERA README](../README.md) describes notebook modes and the offline HTML reading edition. The [computational notebooks](../../computational-notebooks/README.md) cover propagation and receiving channels.
