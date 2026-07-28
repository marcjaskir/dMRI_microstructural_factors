# dMRI Microstructural Factors

Code and data for reproducing results of Jaskir et. al 2026, Mapping Whole-Brain
Factors of Microstructural Similarity with Diffusion MRI.

## Repository layout

```
dMRI_microstructural_factors/
  code/                 # All pipelines and analyses
  data/
    open/               # Publishable post-GAM products (no age/sex/IDs)
  config.example.yaml
  README.md
```

## Quick start

```bash
git clone git@github.com:marcjaskir/dMRI_microstructural_factors.git
cd dMRI_microstructural_factors
cp config.example.yaml config.yaml
# Edit config.yaml: set workspace_root

conda env create -f environment.yml
conda activate dmri_microstructural_factors

export PYTHONPATH="$PWD/code:$PYTHONPATH"
python code/tests/golden/run_golden_tests.py
```

All filesystem roots are defined in `config.yaml` (see
[`config.example.yaml`](config.example.yaml)).

## Data availability and privacy

Data is provided to reproduce manuscript figures from GAM z-scores onward:
anonymized GAM CSVs, factor/asymmetry products, atlas metadata, inclusion with
laterality/lobe only. These data do not include participant age, sex, or IDs.
Populate from a local controlled workspace with
`python -u code/lib/export_tier1_open.py`. See
[`data/open/README.md`](data/open/README.md) for more information.
Pre-CovBat/GAM features and covariates (age, sex, scanner) are omitted for
protection of patient privacy.

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
                      covbat  →  gam
                              ↓
              factor_analysis  →  factor_z-scores  →  gradients
```

HCP Young Adult / Aging BIDS conversion uses
[HCPLifespan2BIDS](https://github.com/ellisdg/HCPLifespan2BIDS) (not vendored here).

Docs: [`code/docs/pipeline.md`](code/docs/pipeline.md),
[`code/docs/analysis.md`](code/docs/analysis.md),
[`code/docs/reproducibility.md`](code/docs/reproducibility.md).

## Citation

> *[Publication citation to be added]*
