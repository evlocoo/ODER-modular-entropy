# Black-hole computational companion

How does information preserved in radiation become information an observer can recover and keep? These notebooks follow the question through source preparation, propagation, joint collection, receiving dynamics, and storage.

Start with [Observer access](Observer_Access_Reading_Guide.ipynb) for the recovery argument. It uses supplied source information and propagated waves so that the collection question can be examined directly. Choose [Upstream regeneration](Upstream_Regeneration.ipynb) to generate the source and follow the calculation through scattering and detection first.

## Follow a calculation

| Notebook | Calculation and inputs | What is generated |
| --- | --- | --- |
| [Observer access](Observer_Access_Reading_Guide.ipynb) | Supplied Hayden–Preskill (HP) subset information and detector probabilities; readable collection flags, deterministic recovery, and joint collection laws. | Aggregate and branch-resolved certificates, sharp envelopes, attaining laws, the seven-outcome minimax obstruction with primal and dual witnesses, and complete clean/noisy rankings of 34 count events. |
| [Upstream regeneration](Upstream_Regeneration.ipynb) | Seeded source preparation, scalar Schwarzschild scattering, packet propagation, and a passive causal detector. Reference arrays enter afterward as comparison targets. | Source tables, waveforms, detector arrays, all three refinement settings, and recovery certificates. |
| [Propagation controls](Schwarzschild_Propagation_Controls.ipynb) | Newly generated sources and scattering; fixed preparation, emission phases, detector, and deadline. | Full transmission, unit transmission, and magnitude-only diagnostics, with energy, prewindow, and refinement checks. |
| [Causal phase and delay](Causal_Phase_and_Delay.ipynb) | Supplied, hash-verified source tables and transmission grids; a routed dispersive element and a matched pure delay. | Packetwise and statewise comparisons, independent resonator evolution, and prehistory/energy accounting. |
| [Modular realization](Modular_Realization.ipynb) | Explicit coherent-label packets, a fixed source/memory partition, and translated preparations. | Overlap and information checks, activation, strip-class diagnostics, and matched-generator-variance comparisons. |
| [Finite receiver](Finite_Receiver.ipynb) | Specified finite-chain Hamiltonians and pulse windows. | Acquisition trajectories, uniform error certificates, lossless retention after disconnection, and pulse-endpoint checks. |
| [Receiving and storage](Receiver_and_Storage.ipynb) | Source tables and propagated detector waves; a controlled catcher, whole-register capture, switched exchange, and storage loss. | Stored-state certificates, two/three-site controls, operating curves, fixed-encoding noise reversal, deadline allocation, recorded progressive gates with smaller receiving codes, and reference-assisted discrimination. |
| [Gaussian noise and recovery](Gaussian_Noise_and_Recovery.ipynb) | Specified independent identical Gaussian rail maps, coherent total-number readout, and the 32 frozen source tables. | Exact Pauli/erasure weights, independent quadrature checks, 64 thermal-loss antecedent checks, deterministic noisy recovery bounds, matched-calibration and additional-sector witnesses. |
| [Observer trajectories and physical scales](Observer_Trajectories_and_Physical_Scales.ipynb) | Frozen transmitted packet envelopes on two grids, scalar transmission quadrature, and three explicit resource budgets. | All-source infall certificates, solver/input refinements, static passive scattering comparison and reference error, Hawking throughput, mode occupations, and alternative clocks. |
| [Joint record access](Joint_Record_Access.ipynb) | A fixed classical-record source and two matched-marginal collection laws. | Enumeration of 4,096 local decoder pairs, exact calibration ranks, joint correctness, communication reversal, and source-independence/coherence controls. |

The separate [MERA companion](../mera-access/README.md) evaluates finite tensor-network access at matched collection size and individual efficiencies. It includes the inputs, methods, and verification needed for its saved-results and regeneration modes.

## Run the notebooks

Use Python 3.12. From this folder, create an environment:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS or Linux. Install the dependencies:

```sh
python -m pip install -r requirements.txt
```

Open JupyterLab:

```sh
python -m jupyterlab
```

Select the environment's Python kernel, open a notebook, and choose **Restart Kernel and Run All Cells**. Keep the notebooks beside `code/`, `inputs/`, and `reference/`. Each core notebook can use the supplied inputs independently. Running upstream first with the same `ODER_RUN_NAME` lets the storage notebook consume that run's generated source and propagation arrays.

For a complete scripted execution, use a new run name:

```sh
python run_notebooks.py --run-name reproduction_01
```

The runner starts a fresh Jupyter kernel for each notebook and limits BLAS to one thread. Executed notebook copies are saved in `outputs/runs/reproduction_01/notebooks/`; the distributed notebooks are preserved. Add `--update-originals` explicitly to replace them with successful executed copies. It preserves prior run directories. New arrays, tables, figures, and the execution record are written under `outputs/runs/reproduction_01/`; scratch runs and caches are ignored by Git. The compact reference results are in `reference/`. The notebooks also retain executed outputs for reading.

Run the core tests and independent certificate checks from this folder:

```sh
python -m unittest discover -s code -p "test_*.py"
python code/verify_event_results.py reference/observer
python code/verify_minimax_results.py reference/observer
python code/verify_progressive_results.py
```

## The receiver and record calculations

The Gaussian noise, observer trajectory, and joint record notebooks can be run independently using compact, hash-verified inputs in `inputs/receiver_extensions/`. They compute their numerical evidence rather than reading a precomputed result table. Reusable implementations are in `code/receiver_extensions/`; saved reference evidence and an explicit execution record are in [reference/receiver_extensions](reference/receiver_extensions/README.md).

The Gaussian notebook specifies its noise channel and coherent total-number instrument. Its illustrative thermal-loss examples are not asserted to be locally derived Hawking detectors. The trajectory notebook concerns an effective one-way vacuum channel and compares three declared budgets; it does not claim a completed local scalar-field apparatus model. The LC certificate is asymptotic and packet restricted. The classical-record notebook concerns actual acquisition and communication, while holding the record source fixed.

To run these three notebooks in separate Python processes when Jupyter kernels are unavailable:

```sh
python run_extensions_process.py --run-name extensions_01
```

The fallback supports these three plain-Python notebooks and preserves previous runs. The normal `run_notebooks.py` remains the preferred Jupyter execution path where kernel transport is available. Neither execution mode changes distributed reading notebooks unless `--update-originals` is explicitly supplied.

## Read the recovery certificates

| Procedure | No-early ceiling | Full-early floor | Accessible late register |
| --- | ---: | ---: | --- |
| Aggregate information | 0.3572 | 0.8155 | Accumulated detector output and current cavity |
| Branch-resolved recovery | 0.3308 | 0.8476 | The same register, with collection outcomes kept resolved when applying the bound |
| Controlled catcher | 0.3306 | 0.8472 | The actual retained catcher memory |

Ceilings are rounded upward and floors downward. Each is an extremum over the 32 source states after the appropriate statewise calculation. The underlying mean information is reported separately. These are certificates for optimal deterministic Bell-overlap recovery; the notebooks do not optimize or implement a recovery decoder.

The progressive recorded receiver transports the first selected number of late carriers while retaining early radiation separately. A two-carrier gate uses five levels per site and gives a fidelity floor of at least 0.6731 under ideal routing, or 0.6393 with a supplied 10M loading interval. The finite chain gives an age-dependent terminal certificate; exact scalar closure is separately derived for a common absolute-time hazard. Arrival records and conditional packing are explicit apparatus resources.

The switched receiver adds acquisition and storage loss to the retained channel. Its operating curve separates a guaranteed region, an unresolved interval, and an upper-bound obstruction. Reference-assisted verification has a different allowed interface from recovery. The BEC and programmable-receiver experiments remain proposed tests.

## Reproducibility and license

Source and propagation conventions are specified beside their calculations. The [verification record](VERIFICATION.md) gives the numerical tolerances, execution evidence, and cross-method checks. Supplied inputs and saved witnesses allow the retained results to be checked independently.

The code retains its [MIT license](LICENSE). When using the companion, cite the November 5, 2026 manuscript, *From Preservation to Possession: The Physical Conditions for Black-Hole Information Recovery*, and the applicable software version.
