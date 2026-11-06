# MERA access and encoding

This companion to the November 5, 2026 version of Evlondo Cooper III's *From Preservation to Possession: The Physical Conditions for Black-Hole Information Recovery* asks whether two observers acquire the same diary information when they collect the same number of sites and receive every individual site equally often.

Read the [executed notebook](MERA_Access_and_Encoding.ipynb) or download and open the [HTML reading edition](MERA_Access_and_Encoding.html). Both include the worked example, all 56 three-site scores, the 64 MERA/tree pairs, independent Haar controls, the collection-sufficiency calculation, 66 causal-cone contraction checks, and three exact-null controls. The [verification results](verification/RESULTS.md) explain the expected results and numerical checks.

## Run the notebook

Use Python 3.12. From the repository or extracted companion root, enter the companion directory:

```sh
cd mera-access
```

Create an environment:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS or Linux. Then run:

```sh
python -m pip install -r requirements-notebook.txt
python -m jupyterlab
```

Select the environment's kernel and choose **Restart Kernel and Run All Cells**.

The configuration cell selects `saved` by default: it loads the supplied tensors and results and executes the explanatory calculations and consistency checks offline. To regenerate the full study, set `MODE = "regenerate"`, choose a new `RUN_NAME`, and run all cells. The notebook then analyzes the newly generated outputs. Each execution writes its figures and verification records into a new `outputs/analysis_*/` directory, printed in the setup cell. The supplied figures and verification records remain available for comparison.

For command-line regeneration and comparison:

```sh
python notebook_support/verify_regeneration.py --output outputs/reproduction
```

The target must be new. To compare an existing completed run, use `--compare-only outputs/reproduction`. Scientific identifiers, seeds, counts, shapes, and types remain exact; finite numerical values use the `1e-10` tolerance. The two raw-negative-eigenvalue count diagnostics must match each run's saved eigenvalues exactly. Their differences between runs are reported separately.

Run the 22 MERA tests and rebuild the HTML after saving a notebook run:

```sh
python -m unittest notebook_support.test_verifier notebook_support.test_validation
python notebook_support/export_html.py
```

The causal-cone and exact-null checks also run independently:

```sh
python -m notebook_support.validation --output outputs/validation_check
```

These use the supplied exploratory MERA/tree tensors and leave the ensemble unchanged.

## Included files

| Path | Purpose |
| --- | --- |
| `MERA_Access_and_Encoding.ipynb`, `.html` | Executable explanation and offline reading edition. |
| `mera_ensemble_study/` | Ensemble implementation, frozen inputs, tensors, and complete reference results required for comparison. |
| `notebook_support/` | Notebook adapters, verifier, tests, and offline HTML renderer. |
| `figures/` | PNG figures for inspection. Notebook execution also exports SVG and PDF. |
| `verification/` | Source identities and compact numerical and execution evidence. |

The encodings are random and unoptimized. The calculation measures available reference information under flagged collection. The notebook explains the subsequent receiving task without executing a receiver composition. The paper's finite-switch, lossy two-site receiver and five-carrier storage-noise reversal are separate constructions. The latter changes the storage channel at fixed encoding; the MERA/tree intervention changes the encoding at fixed collection laws.

The code is released under the [MIT license](LICENSE). The included MathJax renderer has its own license in `notebook_support/MathJax-LICENSE.txt`.
