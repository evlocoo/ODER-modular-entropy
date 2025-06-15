# ODER: Observer-Dependent Entropy Retrieval Framework

**Version 1.1** — Adds `τ_char` fitting, `γ(τ)` inversion, `Δ_fail` diagnostics, and envelope validation (Lemma C.5).

This repository provides simulation and diagnostic tools for modular entropy retrieval in black hole spacetimes. It accompanies the 2025 preprint:

> **Cooper, Evlondo.** (2025). *Modular entropy retrieval in black-hole information recovery: A proper-time saturation model*.  
> https://doi.org/10.5281/zenodo.15654115

## Quick Start

### Main Framework Simulation

Use this notebook to explore how entropy retrieval evolves across different observers and modular rates.

**Notebook:** `ODER_Black_Hole_Framework_Complete_Simulation_(V2).ipynb`

- **Run** all cells to generate retrieval curves, MERA traces, and correlation diagnostics
- **View** plots saved to the `figures/` directory
- **Reproduce** Figures 1–5 from the paper

### Retrieval Fitting and Validation

Use this notebook to extract parameters from retrieval data and validate against theoretical predictions.

**Notebook:** `ODER_Retrieval_Inversion_And_Validation.ipynb`

- **Extract** `τ_char` and `γ(τ)` from retrieval curves
- **Compute** retrieval horizon and failure thresholds
- **Validate** modular signatures and envelope structure
- **Test** sensitivity and self-consistency

## Installation

```bash
pip install -r requirements.txt
```

## Theoretical Framework

ODER models entropy retrieval in black hole spacetimes as an **observer-dependent**, **Lorentzian-causal** process. It replaces global Page-curve bookkeeping with a local proper-time recovery law derived from Tomita–Takesaki modular flow.

### Key Distinctions

- No replica wormholes or Euclidean saddles
- No island prescriptions or ensemble averaging
- Retrieval occurs in proper time, not bulk-extremal coordinates

### Modular Retrieval Law

```
dS_ret/dτ = γ(τ)[S_max - S_ret(τ)]tanh(τ/τ_char)
```

**Parameters:**
- `τ_char`: Characteristic retrieval timescale
- `γ(τ)`: Observer-dependent retrieval rate
- `S_max`: Maximum retrievable entropy

## Simulation Features

**Model Specifications:**
- 48-qubit lattice simulation
- Bond dimensions D = 4 and D = 8 for modular resolution depth
- Bounded flow toward `S_max` with `g²` correlations
- MERA used as geometric anchor (no explicit tensor contractions)

**Generated Visualizations:**
- Retrieval rate profiles `γ(τ)` across observer types
- Entropy retrieval curves `S(τ)` for stationary, accelerating, and free-falling observers
- Bootstrap confidence bands with 200-trace resampling
- `g²(τ)` correlation showing tanh-modulated oscillatory patterns
- `g²(t₁,t₂)` heatmaps with diagonal tanh-fringed retrieval envelopes
- Bond dimension impact comparisons

## Repository Structure

| File | Description |
|------|-------------|
| `ODER_Black_Hole_Framework_Complete_Simulation_(V2).ipynb` | Main framework simulation |
| `ODER_Retrieval_Inversion_And_Validation.ipynb` | Parameter fitting and validation tools |
| `requirements.txt` | Python dependencies |
| `figures/` | Output directory for plots and visualizations |

## Extending the Framework

This framework encourages exploration and extension:

- **Modify parameters:** Bond dimension, qubit count, `τ` intervals
- **Add observer classes:** New spacetime backgrounds and reference frames
- **Extend retrieval law:** Include back-reaction effects
- **Adapt platforms:** BECs, photonic lattices, analog black holes
- **Apply fitting tools:** Experimental or simulated entropy data

Contributions that improve, stress-test, or generalize the framework are welcome.

## Citation

**Cooper, Evlondo.** (2025). *Modular entropy retrieval in black-hole information recovery: A proper-time saturation model*.  
https://doi.org/10.5281/zenodo.15654115



**Cooper, Evlondo.** (2025). *ODER modular entropy simulation* (Version 1.0) [Software]. Zenodo.  
https://doi.org/10.5281/zenodo.15428312

## License

MIT License - see `LICENSE` for details.

## Contact

**Evlondo Cooper**  
Email: [evlocoo@pm.me](mailto:evlocoo@pm.me)
