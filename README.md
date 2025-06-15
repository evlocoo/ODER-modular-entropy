```markdown
> Version: **v1.1** | Added: τ_char fitting, γ(τ) inversion, envelope diagnostics

# ODER: A Modular Framework for Observer-Dependent Entropy Retrieval

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evlocoo/ODER-modular-retrieval/blob/main/ODER_Black_Hole_Framework_Complete_Simulation_(V2).ipynb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15654115.svg)](https://doi.org/10.5281/zenodo.15654115)



Implementation of the Observer-Dependent Entropy Retrieval (ODER) framework for black hole information recovery, as described in:

> **Cooper, Evlondo.** (2025). *Modular entropy retrieval in black-hole information recovery: A proper-time saturation model*. Preprints. https://doi.org/10.20944/preprints202503.2057.v3

---

## Quick Start

### Main Framework Simulation
1. Open `ODER_Black_Hole_Framework_Complete_Simulation_(V2).ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Run all cells to generate comprehensive visualizations of the ODER framework.
3. All output figures will be saved to the `figures/` directory.

### Retrieval Fitting and Validation  
For entropy retrieval data fitting and envelope validation:

📓 **[ODER Retrieval Inversion & Validation](./ODER_Retrieval_Inversion_And_Validation.ipynb)**  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evlocoo/ODER-modular-retrieval/blob/main/ODER_Retrieval_Inversion_And_Validation.ipynb)

This notebook provides:
- **Modular retrieval fitting**: Extract τ_char and γ from S_ret(τ) data
- **Retrieval inversion**: Reconstruct γ(τ) profiles from entropy curves  
- **Horizon computation**: Calculate τ_RH and failure window Δ_fail
- **Diagnostic validation**: Null tests, SNR sensitivity, self-consistency checks
- **Envelope visualization**: Verify Lemma C.5 retrieval envelope structure

### Installation

```bash
pip install -r requirements.txt
```

---

## Notebooks Overview

| Notebook | Purpose | Key Features |
|----------|---------|--------------|
| **Main Simulation** | Complete ODER framework demonstration | Observer classes, retrieval curves, correlation analysis |
| **Retrieval Inversion** | Data fitting and validation tools | Parameter extraction, diagnostic tests, envelope verification |

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

Where:
- τ_char: Characteristic retrieval timescale
- γ(τ): Observer-dependent retrieval rate
- S_max: Maximum retrievable entropy

---

## Key Visualizations

The framework produces comprehensive visualizations including:

### Main Simulation Outputs
* **Retrieval Rate Profiles** — γ(τ) for different observers
* **Entropy Retrieval Curves** — S(τ) for stationary, accelerating, and free-falling observers
* **Bootstrap Confidence Bands** — Statistical robustness via 200-trace resampling
* **g²(τ) Correlation** — Shows tanh-modulated oscillatory patterns for accelerating observers
* **g²(t₁,t₂) Heatmap** — Diagonal tanh-fringed retrieval envelope
* **Bond Dimension Comparisons** — D=4 vs D=8 impact on entropy and correlation

### Retrieval Inversion Diagnostics
* **Parameter Recovery Tests** — Self-consistency validation with <1% error
* **SNR Sensitivity Analysis** — Fitting robustness across noise levels
* **Null Case Validation** — Proper failure modes for γ = 0 scenarios
* **γ(τ) Reconstruction** — Direct inversion from S_ret(τ) data

---

## Simulation Model

* Uses a 48-qubit lattice
* Bond dimension D = 4 and D = 8 simulate modular resolution depth
* Retrieval modeled as a bounded flow toward S_max
* g² correlations derived from retrieval rate and tanh onset
* No explicit tensor contractions — MERA is used as a geometric anchor

---

## Reuse and Exploration Encouraged

This framework is designed to be explored and extended. You are encouraged to:

* Modify parameters such as bond dimension, number of qubits, or τ intervals
* Add new observer classes or spacetime backgrounds
* Extend the modular retrieval law to include back-reaction
* Adapt for analog black hole platforms (BECs, photonic lattices, etc.)
* Use the fitting tools for experimental or simulated entropy data

If you find ways to improve, stress-test, or generalize the framework, please share them. That's the spirit of falsifiable physics.

---

## File Overview

* `ODER_Black_Hole_Framework_Complete_Simulation_(V2).ipynb` — Main framework simulation
* `ODER_Retrieval_Inversion_And_Validation.ipynb` — Parameter fitting and validation tools
* `requirements.txt` — Python dependencies
* `LICENSE` — MIT open-source license
* `figures/` — Folder for plots and visual outputs

---

## Citation

If you use this framework, please cite both:

**Cooper, Evlondo.** (2025). *Modular entropy retrieval in black-hole information recovery: A proper-time saturation model*.  
https://doi.org/10.5281/zenodo.15654115


**Cooper, Evlondo.** (2025). *ODER modular entropy simulation (Version 1.0)* [Software]. Zenodo.  
https://doi.org/10.5281/zenodo.15428312

---

## License

MIT License. See `LICENSE` file for full terms.

---

## Contact

**Evlondo Cooper**  
Email: [evlocoo@pm.me](mailto:evlocoo@pm.me)
```
