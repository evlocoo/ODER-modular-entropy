# Reproducing the calculations

The notebooks follow information from source preparation through collection, recovery, and storage. They evaluate the retained results of *From Preservation to Possession: The Physical Conditions for Black-Hole Information Recovery* (November 5, 2026).

## Run the checks

From `computational-notebooks/`:

```sh
python run_notebooks.py --run-name reproduction_01
python -m unittest discover -s code -p "test_*.py"
python code/verify_event_results.py reference/observer
python code/verify_minimax_results.py reference/observer
python code/verify_progressive_results.py
```

Use a new run name. The runner saves executed copies in `outputs/runs/<run>/notebooks/`; `--update-originals` explicitly replaces the distributed notebooks. The seven source, propagation, and receiver notebooks contain 43 code cells. The three receiver/record notebooks add 20 code cells, for 63 computational cells in ten notebooks. The separate MERA notebook contains 14 code cells. The test command runs 32 core tests: 19 channel/event/minimax tests, ten progressive-receiver tests, and three runner tests. [reference/execution.json](reference/execution.json) records the earlier seven-notebook run.

## What each calculation checks

| Calculation and notebook | Expected result and numerical check |
| --- | --- |
| [Observer access](Observer_Access_Reading_Guide.ipynb) | Aggregate and branch-resolved recovery bounds, marginal constraints, joint-law certificates, and the complete 34-event ranking. E7 leads under clean and noisy readout. |
| [Upstream regeneration](Upstream_Regeneration.ipynb) | Source states, radial scattering, propagated packets, and detector probabilities are generated before reference comparisons. Source information, flux, detector balance, and grid refinements are checked. |
| [Receiving channels](Receiver_and_Storage.ipynb) | Controlled catcher survival, flagged transfer identities, deadline allocation, recorded progressive gates, and reference-assisted verification. Channel identities and independent evolution are compared. |
| [Propagation controls](Schwarzschild_Propagation_Controls.ipynb) | Fixed-preparation transfer controls compare flux, energy, prewindow behavior, and grid refinements. |
| [Causal phase and delay](Causal_Phase_and_Delay.ipynb) | Causal dispersive and matched-delay arms use the same source preparation and upstream prehistory. Matching, energy, access, and refinement residuals are reported. |
| [Modular realization](Modular_Realization.ipynb) | Packet overlap and normalized activation satisfy the analytic construction. Quadrature, density-state checks, and fixed-variance generator comparisons test the implementation. |
| [Finite receiver](Finite_Receiver.ipynb) | Finite-chain and finite-window receivers satisfy the stated uniform error bounds. Sampled trajectory errors are compared with the analytic certificates. |
| [Receiver and storage](Receiver_and_Storage.ipynb) | Finite switching, acquisition loss, noisy storage, operating bounds, and protocol-ordering reversal. Independent evolution, channel limits, explicit branches, and polynomial expressions are compared. |
| [Gaussian noise and recovery](Gaussian_Noise_and_Recovery.ipynb) | Exact Gaussian rail reduction, deterministic source certificates, independent quadrature, matched-calibration and additional-sector checks. |
| [Observer trajectories and physical scales](Observer_Trajectories_and_Physical_Scales.ipynb) | Three resource conventions for infall, stationary passive realization, reference-safe packet error, Hawking flux, mode occupations, and clock distinctions. |
| [Joint record access](Joint_Record_Access.ipynb) | All local decoder pairs, sufficient joint calibration, and communication-dependent protocol ordering for a fixed classical source. |
| [MERA access and encoding](../mera-access/MERA_Access_and_Encoding.ipynb) | Flagged access in unoptimized eight-site encodings, 66 causal-cone contraction checks, and three exact-null controls. See the [MERA verification guide](../mera-access/verification/RESULTS.md). |

The propagation controls retain the same source phases across transfer arms; phase removal is a separate diagnostic. The three-site receiver uses fixed links and the stated switching taper. Memory occupies zero-indexed sites 1 and 2, the second and third sites.

## Numerical conventions and recovery bounds

Channel comparisons use absolute scalar differences, Frobenius matrix norms, and Euclidean amplitude norms with tolerance `1e-10`. The three-site midpoint refinement retains its raw second-order residuals and independently extrapolated amplitude residual, approximately `3.66e-14`. Propagation uses its own stated solver and acceptance tolerances, detailed in the upstream notebook.

The observer notebook evaluates aggregate information and branch-resolved recovery on the same 32 source tables. The receiving notebook applies branch bounds to catcher survival probabilities. Statewise extrema are distinct from ensemble information means. The conservative bound pairs are `0.3572/0.8155`, `0.3308/0.8476`, and `0.3306/0.8472`; upper bounds round upward and lower bounds downward.

Generated propagation gives branch extrema near `0.330718446767` and `0.847608366618`. The reference detector arrays give approximately `0.330718445960` and `0.847608364688`. The respective differences are about `8.07e-10` and `1.93e-9`. These comparisons use the propagation acceptance criteria, rather than the separate channel-identity tolerance. Both support the conservative bounds.

## Joint collection and event selection

The seven-outcome law in the Observer notebook fixes all ten component marginals and gives maximum statewise upper certificate approximately `0.446947387373`, conservatively reported as **0.4470**. The limiting source is the eleventh in the ordered ensemble (zero-indexed source 10, seed 2026082110). Its weights are determined directly by the late probabilities and early marginal 0.65.

The primal minimax program minimizes the largest sourcewise upper certificate. The dual optimizes eleven unrestricted equality multipliers and a probability distribution over the 32 sources. Both are solved with HiGHS dual simplex and interior point, using `1e-9` LP feasibility settings. Witness acceptance uses `1e-10`: maximum absolute component for marginal, feasibility, and complementary-slackness residuals, and absolute scalar objective differences. Different solvers or platforms can return different optimal mask laws and dual multipliers. Accept a witness only if its entries are finite, it satisfies the original primal and dual feasibility constraints and the applicable residual and primal-dual gap tolerances, and it certifies the same conclusion. Raw-array equality is unnecessary; agreement on the objective alone is insufficient. The printed dual witness puts all source weight on the eleventh source. Numerical optimality is for this upper-certificate problem, not for the quantum decoding problem.

[The minimax reference result](reference/observer/minimax_obstruction.json) records the seven weights, all 32 upper certificates, both solver objectives, and raw residuals. [The independent replay](reference/observer/minimax_verification.json) checks the printed and computed witnesses using a separate bisection inversion of the binary-divergence bound, without invoking an optimizer. The witness arrays are in [minimax_witnesses.npz](reference/observer/minimax_witnesses.npz). The supplied Windows evaluation gives `0.4469473873729411`, differing from the manuscript's printed value by approximately `3.33e-16`. The two primal solver objectives differ by `8.88e-16`; their largest primal-dual gap is `9.44e-16`. The independent replay's largest raw residual is `5.44e-15`.

For comparison, the uniform-variable law collects late carrier $j$ when $U<p_j$ and all early carriers when $U\geq0.35$. It gives `0.48048102040658347` under the same marginals. Adding the joint-count constraint gives lower certificate `0.5190306127037225`, conservatively reported as **0.5190**.

The event catalog contains 25 conjunctions with both count thresholds from 1 through 5 and ten total-count thresholds from 1 through 10. Removing the duplicate all-ten event leaves 34 indicators. For ranking, headroom within `1e-8` of zero is tied and ordered by event name; raw headroom values are retained. Each indicator is evaluated for clean flags and independent symmetric flag flips of probability 0.02. For each source, maximize the event rate over compatible collection laws whose lower recovery certificate is at most one half. Take the largest of these thresholds across the sources in the group. Headroom is the feasible event maximum minus that threshold. The full source group, even-indexed source group, and odd-indexed source group are evaluated separately. The complete rankings select E7 in each group under both readout models.

The noisy full-family headroom is `0.34384971512254914`; the even-source value is `0.34414608114460626`. The clean/noisy sufficient rates 0.3759 and 0.3410 also pass direct lower-certificate minimization. For each candidate, directly minimize the lower recovery certificate at a feasible event rate `1e-6` above the group threshold. Candidates without that much feasible headroom are marked accordingly.

`reference/observer/` includes complete sourcewise thresholds, rankings, mask laws, dual multipliers, and raw LP residuals. Dual-simplex and interior-point calculations check the selected event and the dependence certificates. The LP feasibility tolerance remains `1e-8`. Independent replay verified 3,524 event witnesses; its largest raw residual was `8.497004527882514e-09`. The search and witness checker apply the same zero-headroom tie rule; E7 ranks first in all six clean/noisy source-split comparisons.


## Progressive recorded acquisition

[The progressive results](reference/progressive/summary.csv) use all 32 frozen source tables and all three supplied staggered propagation grids. A separate rail-independent time-recorded instrument has the same fixed-time acceptance probabilities as the coherent catcher; its event records are part of the apparatus definition. The CDF is piecewise linear on the saved time grid. Each source's bound is evaluated after integration and before taking extrema.

For count gates one through five, loading durations 0, 2, 5, 10 and 20M are evaluated. The common constant-hazard mode is distinct from a finite Jacobi chain reset at loading completion. The finite chain uses the existing h=1, N=62, Omega=0.025/M links, source isolation after at most 57.15M and lossless holding until 120M. The 10M-loading two-carrier gate has worst-source floor above 0.6393 with five levels per site. These are decoder-existence lower certificates; decoder execution time is separate from the acquisition deadline.

Gauss-Legendre orders four and eight are compared after splitting intervals at loading and source-isolation boundaries. Gate-density integrals independently match product-CDF count probabilities. [Metadata](reference/progressive/metadata.json) reports residuals; [sourcewise values](reference/progressive/statewise_scores.npz) retain branch weights, gate laws and all source scores. [The progressive execution record](reference/progressive/execution.json) identifies the clean-process evaluation of all receiving-notebook code cells. That supplied evaluation used a host without Jupyter socket transport and does not claim fresh-kernel execution.

The replay rejects nonfinite inputs, saved and recomputed scores, and comparison residuals before applying its unchanged absolute tolerance of `1e-11`. A regression runs the clean reference and rejects a copy containing a NaN score. The nine channel and timing tests check explicit entangled-state channel information, common-clock differentiation, record deletion, reset-age failure, gate-density mass, defective and missing arrivals, finite cutoff behavior, quadrature and validation of arrival records. Run `python code/progressive_receivers.py --output outputs/runs/progressive_01` for the standalone calculation, using a new output directory. The default also creates a new run directory; reference and input directories are protected. The supplied Windows record in `reference/execution.json` remains the record of that execution; the progressive section has its own execution record.

An [independent replay](reference/progressive/independent_replay.json) uses a separate 12-point implementation of the gate integrals, without calling the progressive module. It checks all 32 sourcewise scores for five gates, three displayed loading times and all three propagation grids. Run `python code/verify_progressive_results.py` to repeat it.


## Gaussian noise, observer trajectories, and joint classical records

The supplied extension outputs contain all 20 new code cells executed in fresh isolated Python processes. Their host blocked Jupyter startup before scientific cells ran. [The supplied execution record](reference/receiver_extensions/execution.json) retains this provenance; it is not a fresh-kernel result. Saved tables and PNG figures let readers inspect the calculations before execution.

| Notebook | Computed checks and scope |
| --- | --- |
| `Gaussian_Noise_and_Recovery.ipynb` | Seven independent characteristic-function quadratures verify the exact Gaussian moments; 64 thermal-loss cases compare inherited predecessor formulas. Both optional witnesses are checked from finite matrices or physically admissible channel parameters. The new logical weights are applied to every frozen source. These are specified modal interfaces, not an assumed local Hawking-noise model. |
| `Observer_Trajectories_and_Physical_Scales.ipynb` | Recomputes scalar flux integrals and mode occupations, the passive LC packet comparison and reference-safe error, plus three complete effective-trajectory runs. Solver-step refinement changes probabilities by at most 2.09e-13; physical-input refinement changes them by at most 3.41e-8. Maximum balance residual is 1.41e-15. Adaptive infall and equal-support static probabilities agree within 2.15e-13. |
| `Joint_Record_Access.ipynb` | Enumerates all 4,096 deterministic local decoder pairs at each of 11 mixture points, verifies LP endpoints, checks ranks 7 and 8 by exact rational elimination, and computes the communication reversal and source-independence/coherence controls. |

For fixed hardware and equal proper duration, the full-early infalling floor exceeds the static ceiling on all 32 sources: `0.6101378681 > 0.5509975450` at `r=6M` and `0.6525925001 > 0.5948956793` at `r=4M`. Equal retarded support changes the ranking; adaptive controls reproduce the equal-support static calculation. The distinction between these budgets is part of the result. The passive LC distance `0.0397401435` includes common greybody/aperture loss and applies only to the specified stationary packets, not an arbitrary-spectrum diamond distance or finite-deadline memory.

[Compact numerical evidence](reference/receiver_extensions/README.md) is retained independently of run folders. Input hashes are verified before each notebook. Two portable input files contain the frozen source/packet arrays and scalar flux grid; no unretained full-field detector sweeps are needed. Gaussian channels and finite matrices are computed in the notebooks rather than loaded as result tables. The fresh-process fallback is `run_extensions_process.py`; normal fresh-kernel reproduction remains available through the extended `run_notebooks.py` list.

The supplied [validation summary](reference/receiver_extensions/validation_summary.json) records 31 passing core tests and numerical checks. The [archive comparison](reference/receiver_extensions/baseline_preservation.json) describes preservation within the supplied package. Checkout validation is recorded separately so the two execution environments remain distinguishable.

## Reproduce the four receiver and record notebooks

From `computational-notebooks/`, install `requirements.txt` in a clean Python 3.12 environment and run:

```sh
python run_notebooks.py --run-name receiver_check --notebooks Gaussian_Noise_and_Recovery.ipynb Joint_Record_Access.ipynb Observer_Trajectories_and_Physical_Scales.ipynb Receiver_and_Storage.ipynb
python code/verify_progressive_results.py
```

The [Windows checkout validation](reference/receiver_extensions/checkout_validation.json) records the earlier 31-test suite and 30 code cells executed in four fresh Jupyter kernels. The three extension result files agreed with the supplied reference values within `4.45e-16` (absolute differences, comparison tolerance `1e-10`). All 612 progressive arrays agreed within `2.23e-16`; the independent 12-point replay agreed with the 32-source scores across 45 cases within `6.67e-16`. Executed copies and generated results went to a new run directory. The distributed notebooks retain the supplied tables and figures.
