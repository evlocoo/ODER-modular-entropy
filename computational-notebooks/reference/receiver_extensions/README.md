# Receiver and record extension evidence

These files are computed by the three distributed notebooks, with all trials retained. The notebooks recompute the values from supplied inputs and reusable modules.

| File | Evidence |
| --- | --- |
| `gaussian_noise_and_recovery.json` | Seven independent characteristic-function quadrature checks, 64 inherited thermal-loss cases, the Gaussian calibration witness, all 32-source recovery examples, and the finite-matrix N=2 extraction witness. |
| `observer_trajectories_and_scales.json` | Stationary physical scales, asymptotic LC reference-distance calculation, all 32-source effective trajectory cases, and numerical refinements. |
| `joint_record_access.json` | Exact task scores, 4,096 decoder-pair enumeration, calibration ranks, and communication/source controls. |
| `execution.json` | Supplied process-run record, package versions, and elapsed times. |
| `attempted_kernel_execution.json` | Supplied kernel-startup record. See [execution provenance](../../VERIFICATION.md) for its scope. |
| [checkout_validation.json](checkout_validation.json) | Recorded checkout comparison; [verification instructions](../../VERIFICATION.md) describe the current checks. |
| `validation_summary.json` | Compact numerical residuals, 31 tests at the time of that run, and scope limits. |
| `baseline_preservation.json` | Comparison of all 1,742 files in the reviewed original archive; no notebooks or data changed. |

The only added numeric inputs are two hash-verified NPZ files totaling approximately 436 KB. The source/packet provenance is recorded in `inputs/receiver_extensions/`. No unrelated exploratory detector backend is included. The preexisting core, progressive, and MERA materials remain in place.
