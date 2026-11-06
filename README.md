# From Preservation to Possession

[![Verification](https://github.com/evlocoo/ODER-modular-entropy/actions/workflows/black-hole-verification.yml/badge.svg?branch=main)](https://github.com/evlocoo/ODER-modular-entropy/actions/workflows/black-hole-verification.yml)

## The Physical Conditions for Black-Hole Information Recovery

A quantum diary falls into a black hole. Suppose its information survives in the outgoing radiation. What would it take for an observer to recover it, bring it into memory, and keep it?

That question is the starting point for this work. Observer-Dependent Entropy Retrieval (ODER) follows information through the physical steps between global preservation and retained possession: which correlations reach an observer, which arrive together, and how much time remains to acquire them. Each step can change the recovery outcome, even when the underlying radiation state stays the same.

**A physical account of recovery must include the observer's access and receiving dynamics.** Establishing that information survives globally leaves those conditions to be determined. The calculations here show how those conditions change recovery and how their consequences can be tested.

The companion materials introduce the physical question before its mathematics. You can read the explanations and saved results, inspect an individual construction, or run the calculations yourself.

## What the work shows

**Access can obstruct recovery even with unrestricted decoding computation.** In the finite black-hole diary calculation, the source, late-radiation detector, and collection lifetime stay fixed. Across all 32 tested source states, optimal deterministic entanglement-recovery fidelity is at most 0.3308 without early radiation and at least 0.8476 with it. The upper bound applies to every decoder acting on the available registers. The difference comes from which correlations the observer possesses.

**Individual collection efficiencies leave joint access unresolved.** Two observers can receive each carrier equally often and still acquire different information because different combinations arrive together. With every component efficiency fixed, one explicit law caps recovery fidelity at 0.4470 for all 32 source states. An additional joint-count measurement instead guarantees at least 0.5190. The paper gives sharp access envelopes and an exact criterion for deciding when the available calibration statistics determine the information value. It also constructs a reference-assisted verification task in which preservation and erasure give identical accessible channels below the required joint access, while sufficient access makes them distinguishable. Storage noise can even reverse which collection protocol performs better, although the individual terminal efficiencies remain matched.

**The finite MERA tensor-network study makes the role of correlation structure concrete.** Both collection protocols receive exactly three of eight sites, and each site has the same inclusion probability. Yet contiguous collection yields more reference information in all 64 sampled MERA encodings. Removing the disentanglers raises mean acquisition under both protocols while reducing their mean difference. The comparison shows how an access pattern interacts with the encoding, even after collection size and individual efficiencies have been matched.

**Acquisition and storage have physical requirements of their own.** A controlled catcher carries the black-hole access separation into retained memory. A worked two-site receiver preserves that separation through finite switching, acquisition loss, and noisy storage. Its operating bounds show when the chosen recovery target remains guaranteed and when every decoder must fall short. An exact modular construction realizes the sharp activation envelope within the strip-holomorphic class, while finite receivers approximate its dynamics with explicit coupling, size, duration, and accuracy requirements.

**Progressive collection can feed a smaller receiving code.** A separately specified time-recorded instrument admits the first selected number of late carriers. Exact cohort dynamics retain their partial Hayden–Preskill (HP) correlations, while a common capture clock gives scalar closure. On the frozen sources and staggered propagation profiles, a two-carrier gate uses five levels per receiving site and guarantees at least 0.6731 with ideal routing, or 0.6393 with a supplied 10M loading interval. The final decoder acts after acquisition. These values describe a receiving-register tradeoff; recording, routing, buffering and early storage remain separate resources.

**Physical noise can be connected to the recovery task without discarding failed trials.** An exact reduction maps specified independent Gaussian rail channels into logical Pauli errors and recorded erasures. The Gaussian noise and recovery notebook composes those weights with all 32 source tables, exhibits two Gaussian channels with matched acceptance and Bell fidelity but different binary-task performance, and shows an achievable recovery gain from coherent access to an additional number sector.

**Observer comparisons need an explicit resource budget.** The effective infalling calculation separates equal proper duration, equal incoming retarded support, and adaptive coupling. A passive stationary implementation has a quantified packet-restricted reference error, while its moving extension remains separately specified. Stationary Hawking flux, field multiplicity, mode occupation, and receiver-clock tables clarify how the prepared source relates to black-hole scales. A classical-record example also shows that the same individual access statistics can rank collection protocols differently when communication is permitted.

Together, these results make preservation, access, and possession separately calculable. They also turn the framework toward experiment: the manuscript develops programmable-receiver tests and a staged program for Bose–Einstein condensate analogs.

## Find your way through the companion materials

Choose a notebook by the physical question or calculation you want to follow.

| Material | What it helps you explore |
| --- | --- |
| [Observer access](computational-notebooks/Observer_Access_Reading_Guide.ipynb) | Recovery bounds and joint collection laws from supplied source inputs. |
| [Upstream regeneration](computational-notebooks/Upstream_Regeneration.ipynb) | Source generation, scattering, packet propagation, and causal detection. |
| [Propagation controls](computational-notebooks/Schwarzschild_Propagation_Controls.ipynb) | Schwarzschild, unit-transmission, and magnitude-only comparisons with fixed source phases and refinement checks. |
| [Modular realization](computational-notebooks/Modular_Realization.ipynb) | Packet preparation, transfer, and normalized activation. |
| [Finite receiver](computational-notebooks/Finite_Receiver.ipynb) | Finite-chain acquisition, error bounds, resource requirements, and retention. |
| [Receiving and storage](computational-notebooks/Receiver_and_Storage.ipynb) | Controlled capture, finite switching, noisy storage, deadline allocation, recorded progressive collection, and verification. |
| [Causal phase and delay](computational-notebooks/Causal_Phase_and_Delay.ipynb) | Matched spectral transmission and mean delay, with different finite-window access. |
| [Gaussian noise and recovery](computational-notebooks/Gaussian_Noise_and_Recovery.ipynb) | Exact logical noise weights, all-trial source certificates, matched-calibration and additional-sector witnesses. |
| [Observer trajectories and scales](computational-notebooks/Observer_Trajectories_and_Physical_Scales.ipynb) | Three-budget infall, stationary passive comparison, reference error, and source/mode/clock reconciliation. |
| [Joint classical records](computational-notebooks/Joint_Record_Access.ipynb) | Exact decoder optimization, sufficient joint calibration, and a communication-dependent ranking reversal. |
| [`MERA_Access_and_Encoding.ipynb`](mera-access/MERA_Access_and_Encoding.ipynb) | If observers collect the same number of sites equally often, does it matter which sites they receive together? Explicit tensor-network encodings, matched collection protocols, paired removal of disentanglers, 66 causal-cone contraction checks, exact nulls, and sufficient calibration. |

The MERA companion also includes an [HTML reading edition](mera-access/MERA_Access_and_Encoding.html) with embedded figures and offline equation rendering. Download the complete companion folder before opening the HTML so its included equation renderer is available. Its saved-results mode reconstructs the worked encoding and recomputes the ensemble summaries; its regeneration mode rebuilds the full study and compares the outputs with the supplied results.

The [computational companion](computational-notebooks/README.md) follows source preparation, propagation, collection, and receiving dynamics. The MERA companion evaluates the separate finite tensor-network study. The [coverage and execution record](computational-notebooks/VERIFICATION.md) identifies the calculations performed, numerical checks, and their scope.

## Running the calculations

Both packages use Python 3.12. For the computational companion, work from `computational-notebooks/` and install `requirements.txt`. For the MERA companion, work from `mera-access/` and install `requirements-notebook.txt`. Create a virtual environment in the chosen package:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS or Linux. For the computational companion:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

For the MERA companion:

```sh
python -m pip install -r requirements-notebook.txt
python -m jupyterlab
```

Select that environment's kernel and choose **Restart Kernel and Run All Cells**. Keep each notebook with its supporting code, inputs, and verification files. The [core README](computational-notebooks/README.md) and [MERA README](mera-access/README.md) explain output locations and execution modes.

Numerical checks appear beside the results they support. The materials preserve the source inputs and distinguish numerical agreement from byte-for-byte identity, so the calculations can be checked across computing environments.

## Citation and discussion

The updated paper is *From Preservation to Possession: The Physical Conditions for Black-Hole Information Recovery*, by Evlondo Cooper III, November 5, 2026. When using these materials, please cite the manuscript and software version you actually used.

Questions, corrections, and extensions are welcome through the [issue tracker](https://github.com/evlocoo/ODER-modular-entropy/issues). For questions about a calculation, include the notebook, settings, and result.

Both packages use the MIT license: [computational notebooks](computational-notebooks/LICENSE) and [MERA companion](mera-access/LICENSE).

Earlier versions are available in the [release archive](https://github.com/evlocoo/ODER-modular-entropy/releases).
