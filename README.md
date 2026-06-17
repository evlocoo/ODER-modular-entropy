# ODER Black-Hole Retrieval Verification

This repository contains the v2 verification artifact for the ODER black-hole retrieval paper.

Manuscript: [Modular Entropy Retrieval in Black-Hole Information Recovery: A Proper-Time Saturation Model](https://doi.org/10.20944/preprints202503.2057.v10)

The notebook is intended to make the numerical claims auditable. It runs the retrieval-law checks, inverse-gamma recovery, finite-resolution proxy checks, and adversarial-null diagnostics used in the manuscript.

Canonical notebook:

```text
notebooks/ODER_BH_verification_artifact_v2.ipynb
```

Canonical outputs:

```text
outputs/verification_report.md
outputs/validation_manifest.json
outputs/figures/
outputs/tables/
```

## Validation Scope

This repository supports synthetic and numerical verification of observer-indexed black-hole entropy retrieval under bounded modular access. It does not claim experimental confirmation, explicit tensor-network contractions, or a microscopic simulation of black-hole geometry.

The v2 artifact reproduces the retrieval-law checks, inverse-gamma recovery, finite-resolution robustness, and adversarial-null diagnostics reported in the manuscript.

## What The Notebook Runs

The notebook uses the publication preset:

| Setting | Value |
| --- | --- |
| Artifact | `ODER_BH_verification_artifact_v2` |
| Version | `2.0` |
| Run mode | `publication` |
| Bootstrap traces | `200` per observer class |
| Bond dimensions analyzed | `D=4`, `D=8` |
| Observer classes | free-falling, accelerating, stationary |
| External data | none |

The validation suite includes:

- bounded and monotone retrieval-law checks
- 90% Retrieval Horizon ordering
- inverse-gamma recovery
- D=4/D=8 finite-resolution proxy robustness
- Matched Saturating Envelope Null
- Shared-Observer Envelope Null
- Observer-Label Permutation Null
- Proper-Time Jitter Null
- Non-Gap Dynamics Null

The final report includes a claim-to-artifact map. Each reported claim points to a table, figure, or manifest entry produced by the notebook.

## Repository Layout

```text
ODER-modular-entropy/
  README.md
  notebooks/
    ODER_BH_verification_artifact_v2.ipynb
  outputs/
    verification_report.md
    validation_manifest.json
    figures/
    tables/
  archive/
    v1.1/
      ODER_Black_Hole_Framework_Complete_Simulation_(V2).ipynb
      ODER_Retrieval_Inversion_And_Validation.ipynb
      Figures/
  environment.yml
  requirements.txt
  LICENSE
  CITATION.cff
```

The `archive/v1.1/` folder preserves the earlier simulation and inversion notebooks that document the MERA-inspired finite-resolution proxy lineage, including the 48-qubit parameterization and D=4/D=8 comparison structure. These archived notebooks are retained for provenance and continuity with earlier manuscript versions. The current v2 release is driven by `notebooks/ODER_BH_verification_artifact_v2.ipynb`, the canonical adversarial verification artifact for the present manuscript.

This repository separates the supporting proxy lineage from the current verification suite: v1.1 documents the MERA-inspired finite-resolution proxy history, while v2 provides the final retrieval-law checks, inverse-gamma recovery, finite-resolution robustness, and adversarial-null diagnostics. Neither artifact claims experimental confirmation, explicit tensor-network contractions, or a microscopic simulation of black-hole geometry.

## Run The Artifact

Create an environment:

```bash
conda env create -f environment.yml
conda activate oder-bh-verification
```

Or install the minimal Python requirements:

```bash
pip install -r requirements.txt
```

Then open:

```text
notebooks/ODER_BH_verification_artifact_v2.ipynb
```

Run the notebook top to bottom. It regenerates the report, manifest, figures, and tables in `outputs/`.

## Expected Outputs

The notebook writes:

- `outputs/verification_report.md`
- `outputs/validation_manifest.json`
- CSV tables under `outputs/tables/`
- PNG figures under `outputs/figures/`

The manifest records the version, run mode, seed, bootstrap count, bond dimensions, thresholds, nulls executed, pass flags, and generated artifacts.

## Boundary Notes

This is a reproducibility artifact for the numerical validation suite. It uses synthetic/proxy traces and does not ingest external astrophysical data.

The finite-resolution check is scoped to the bond dimensions listed in the manifest. No additional resolution-scaling claims are made in this release.

## Citation

If you use this repository, cite the manuscript and the archived software release.

```bibtex
@software{cooper_oder_bh_v2,
  author = {Cooper, Evlondo},
  title = {ODER Black-Hole Retrieval Verification Artifact},
  version = {2.0},
  year = {2026},
  note = {v2 adversarial verification artifact}
}
```

## License

MIT License. See `LICENSE`.
