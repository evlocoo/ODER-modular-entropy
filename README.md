# ODER: A Modular Framework for Observer-Dependent Entropy Retrieval

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evlocoo/ODER-modular-retrieval/blob/main/ODER_Black_Hole_Framework_Complete_Simulation_(V2).ipynb)

Implementation of the Observer-Dependent Entropy Retrieval (ODER) framework for black hole information recovery, as described in:

> **Cooper, Evlondo.** (2025). *Modular entropy retrieval in black-hole information recovery: A proper-time saturation model*. Preprints. https://doi.org/10.20944/preprints202503.2057.v3

---

## Quick Start

1. Open `ODER_Black_Hole_Framework_Complete_Simulation_(V2).ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Run all cells to generate all visualizations.
3. All output figures will be saved to the `figures/` directory.

To install dependencies:

```bash
pip install -r requirements.txt
````

---

## Reuse and Exploration Encouraged

This framework is designed to be explored and extended. You are encouraged to:

* Modify parameters such as bond dimension, number of qubits, or τ intervals
* Add new observer classes or spacetime backgrounds
* Extend the modular retrieval law to include back-reaction
* Adapt for analog black hole platforms (BECs, photonic lattices, etc.)

If you find ways to improve, stress-test, or generalize the framework, please share them. That's the spirit of falsifiable physics.

---

## Theoretical Background

The ODER framework models entropy retrieval in black hole spacetimes as an **observer-dependent**, **Lorentzian-causal** process. It replaces global Page-curve bookkeeping with a local proper-time recovery law derived from Tomita–Takesaki modular flow.

Key distinctions from other approaches:

* **No replica wormholes or Euclidean saddles**
* **No island prescriptions or ensemble averaging**
* **Retrieval occurs in proper time, not in bulk-extremal coordinates**

The modular retrieval law is:

$$
\frac{dS_{\text{ret}}}{d\tau} = \gamma(\tau)[S_{\text{max}} - S_{\text{ret}}(\tau)]\tanh(\tau / \tau_{\text{char}})
$$

---

## Key Visualizations

The notebook produces the following figures:

* **Retrieval Rate Profiles** — γ(τ) for different observers
* **Entropy Retrieval Curves** — S(τ) for stationary, accelerating, and free-falling observers
* **Bootstrap Confidence Bands** — Statistical robustness via 200-trace resampling
* **g²(τ) Correlation** — Shows tanh-modulated oscillatory patterns for accelerating observers
* **g²(t₁,t₂) Heatmap** — Diagonal tanh-fringed retrieval envelope
* **Bond Dimension Comparisons** — D=4 vs D=8 impact on entropy and correlation

These visualizations directly map to figures in the paper and are reproducible with a single notebook run.

---

## Simulation Model

* Uses a 48-qubit lattice
* Bond dimension D = 4 and D = 8 simulate modular resolution depth
* Retrieval modeled as a bounded flow toward Sₘₐₓ
* g² correlations derived from retrieval rate and tanh onset
* No explicit tensor contractions — MERA is used as a geometric anchor

---

## File Overview

* `ODER_Black_Hole_Framework_Complete_Simulation_(V2).ipynb` — Main notebook
* `requirements.txt` — Python dependencies
* `LICENSE` — MIT open-source license
* `figures/` — Folder for plots and visual outputs

---

## Citation

If you use this code or base further work on ODER, please cite:

```
Cooper, Evlondo. (2025). Modular entropy retrieval in black-hole information recovery: A proper-time saturation model. Preprints. https://doi.org/10.20944/preprints202503.2057.v3
```

---

## License

MIT License. See `LICENSE` file for full terms.

---

## Contact

**Evlondo Cooper**
Email: [evlocoo@pm.me](mailto:evlocoo@pm.me)

