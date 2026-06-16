# ODER-BH Verification Report

- artifact: `ODER_BH_verification_artifact_v2`
- version: `2.0`
- analysis preset: `ODER_BH_RETRIEVAL_V2`
- run mode: `publication`
- bootstrap traces per observer: `200`
- observer-label permutations: `2000`

## Thresholds
- bootstrap_traces_publication: `200`
- retrieval_horizon_order: `['free_falling', 'accelerating', 'stationary']`
- inverse_recovery_median_abs_error_lt: `0.005`
- ordering_preservation_ge: `0.95`
- matched_null_allowed_full_signatures: `0`
- non_gap_max_stable_inverses: `1`
- label_permutation_baseline_tolerance: `0.05`
- inverse_finite_fraction_min: `0.55`
- inverse_negative_fraction_max: `0.02`
- inverse_gamma_cv_max: `1.25`
- inverse_roughness_max: `0.035`
- bond_dimensions: `[4, 8]`

## Observer Summary
- free_falling: 90% Retrieval Horizon=12.9, final S=1.0000, inverse gamma MAE=0.0031777
- accelerating: 90% Retrieval Horizon=20.5, final S=0.9999, inverse gamma MAE=0.00092367
- stationary: 90% Retrieval Horizon=30.5, final S=0.9811, inverse gamma MAE=0.0003643

## Adversarial Summary
- matched saturation families passing full signature: `0`
- Shared-Observer Envelope Null order reproduced: `False`
- observer-label permutation p-value: `0.1734`
- proper-time jitter ordering survival: `1.000`
- Non-Gap Dynamics Null stable inverse count: `0`
- nulls executed: `Matched Saturating Envelope Null, Shared-Observer Envelope Null, Observer-Label Permutation Null, Proper-Time Jitter Null, Non-Gap Dynamics Null`

## Pass Flags
- preset_integrity: PASS
- core_bounded: PASS
- core_monotone: PASS
- core_order: PASS
- core_inverse_recovery: PASS
- analytic_constant_gamma_recovery: PASS
- bootstrap_bounded: PASS
- d4_d8_resolution_robustness: PASS
- matched_saturating_null_rejected: PASS
- shared_observer_envelope_null_rejected: PASS
- observer_label_permutation_quantified: PASS
- proper_time_jitter_robust: PASS
- non_gap_dynamics_null_rejected: PASS

## Claim Map
- PASS: Core ODER traces remain bounded and monotone. -> core_observer_summary
- PASS: Observer retrieval horizons follow free_falling < accelerating < stationary. -> core_observer_summary
- PASS: The ODER inverse map recovers gamma(tau) with low error on generated retrieval traces. -> inverse_gamma_profiles
- PASS: Bootstrap confidence bands use 200 traces per observer in publication mode. -> entropy_retrieval_bootstrap_bands
- PASS: D=4/D=8 finite-resolution proxy settings preserve the retrieval structure. -> d4_d8_resolution_robustness
- PASS: Matched generic saturation does not reproduce the full ODER signature set. -> matched_saturating_envelope_null
- PASS: A Shared-Observer Envelope Null does not reproduce observer-class separation. -> shared_observer_envelope_null
- PASS: Observer-Label Permutation Null reports the chance ordering baseline. -> observer_label_permutation_null
- PASS: Proper-time jitter preserves retrieval-horizon ordering. -> proper_time_jitter_null
- PASS: A Non-Gap Dynamics Null degrades ODER inverse-map diagnostics. -> non_gap_dynamics_null

## Boundary Notes
- Synthetic/proxy verification only.
- No external astrophysical data are ingested.
- D=4/D=8 is treated as a finite-resolution proxy check.
- Adversarial nulls test generic saturation and structure-breaking alternatives without adding new physical models.
- Individual nulls may pass isolated diagnostics, but no adversarial family reproduces the full joint ODER signature across observer classes.