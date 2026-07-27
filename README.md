# dMRI Microstructural Factors

Code and data layout accompanying the publication on diffusion MRI microstructural
factor analysis and temporal lobe epilepsy asymmetry.

## Repository layout

```
dMRI_microstructural_factors/
  code/                 # All pipelines and analyses
  data/
    open/               # Tier 1 — publishable post-GAM products (no age/sex/IDs)
    controlled/         # Tier 2 — pre-CovBat inputs (gitignored; DUA only)
  config.example.yaml
  README.md
```

## Quick start

```bash
git clone git@github.com:marcjaskir/dMRI_microstructural_factors.git
cd dMRI_microstructural_factors
cp config.example.yaml config.yaml
# Edit config.yaml: set workspace_root; point controlled roots at local data

conda env create -f environment.yml
conda activate dmri_microstructural_factors

export PYTHONPATH="$PWD/code:$PYTHONPATH"
python code/tests/golden/run_golden_tests.py
```

## Configuration

All filesystem roots are defined in [`config.yaml`](config.yaml) (see
[`config.example.yaml`](config.example.yaml)):

| Key | Purpose |
|-----|---------|
| `workspace_root` | This repository root |
| `code_dir` | `code/` (scripts) |
| `data_open_dir` | Tier 1 open products |
| `data_controlled_dir` | Tier 2 controlled inputs |
| `gam_dir` | Post-GAM residual z-scores |
| `analysis_dir` | Factor / asymmetry / gradient outputs |
| `atlas_dir` | Atlas metadata (labels, tract tables) |
| `inclusion_dir` | Cohort inclusion tables |
| `controlled_derivatives_dir` | Early pipeline + pre-GAM derivatives |
| `controlled_metadata_dir` | Age/sex/scanner/clinical (controlled) |

Python scripts import [`code/lib/paths.py`](code/lib/paths.py). Shell scripts
resolve `$CODE_ROOT` / `$BASE` from the same config.

## Data tiers (Nature Neuroscience)

**Tier 1 (`data/open/`)** — minimum open dataset to reproduce manuscript figures
from GAM z-scores onward: anonymized GAM CSVs, factor/asymmetry products, atlas
metadata, inclusion with `laterality`/`lobe` only. No age, sex, or real IDs.

Populate from a local controlled workspace with:

```bash
python -u code/lib/export_tier1_open.py
```

Small products (`atlases/`, `metadata/`, `inclusion/`) ship in-repo. Large trees
(`gam/`, `analysis/`) are gitignored — keep them local or deposit on Zenodo.

**Tier 2 (`data/controlled/`)** — pre-CovBat/GAM features and covariates (age,
sex, scanner). Gitignored; share under controlled access / DUA if recomputing
harmonization from scratch.

See [`data/open/README.md`](data/open/README.md) and
[`data/controlled/README.md`](data/controlled/README.md).

## Pipeline overview

Cohort-specific ingress differs before the shared tractography path:

- **Penn + HCP-Aging:** BIDS → qsiprep → freesurfer → qsirecon
- **HCP-YA:** → qsirecon (skip qsiprep/freesurfer; use existing HCP derivatives)

```
Penn / HCP-Aging:  BIDS  →  qsiprep  →  freesurfer  →  qsirecon
HCP-YA:                                     →  qsirecon
                              ↓
                         acpc_mni_xfm
                              ↓
              trekker  →  bundleseg  →  pyafq / mni_micro
                              ↓
                    [Tier 2] covbat  →  gam
                              ↓
                         [Tier 1 open products]
              factor_analysis  →  factor_z-scores  →  gradients
                              ↓
         tract/region asymmetry  →  microstructural asymmetry reports
```

HCP Young Adult / Aging BIDS conversion uses
[HCPLifespan2BIDS](https://github.com/ellisdg/HCPLifespan2BIDS) (not vendored here).

Docs: [`code/docs/pipeline.md`](code/docs/pipeline.md),
[`code/docs/analysis.md`](code/docs/analysis.md),
[`code/docs/reproducibility.md`](code/docs/reproducibility.md).

## Privacy

No subject identifiers, demographics, or clinical outcomes are stored in open
source or `data/open/`. Controlled metadata stay in `data/controlled/` (local /
DUA only).

## Citation

> *[Publication citation to be added]*
