# dMRI Microstructural Factors

Code and data for reproducing results of Jaskir et al. 2026, Mapping Whole-Brain
Factors of Microstructural Similarity with Diffusion MRI.

## Repository layout

```
dMRI_microstructural_factors/
  code/                 # All pipelines and analyses
  data/
    open/               # Small committed trees + OSF-synced analysis/gam
  config.example.yaml
  README.md
```

**Open data for manuscript reproduction:** sync the OSF **storage** tree into
`data/open/` so `analysis/` and `gam/` exist beside the committed
`atlases/` / `metadata/` / `inclusion/` stubs. No unpack step.

## Quick start

```bash
git clone git@github.com:marcjaskir/dMRI_microstructural_factors.git
cd dMRI_microstructural_factors
cp config.example.yaml config.yaml
# Edit config.yaml: set workspace_root
# After OSF upload: set open_osf_url (project/storage link)

conda env create -f environment.yml
conda activate dmri_microstructural_factors

export PYTHONPATH="$PWD/code:$PYTHONPATH"

# Download OSF files into data/open/ (merge with committed stubs), then:
python code/tests/golden/run_golden_tests.py
python code/tests/test_factor_labels.py
python code/tests/test_no_phi.py
```

All filesystem roots are defined in `config.yaml` (see
[`config.example.yaml`](config.example.yaml)).

## Data availability and privacy

Manuscript reproduction uses an **OSF-hosted `data/open/` directory** of
anonymized tabular products: inclusion (`anon_id`, laterality/lobe), a minimal
GAM residual-z sample (or fuller export), factor loadings/scores/z,
Laplacian-eigenmap gradient CSVs, and asymmetry digests. No participant age,
sex, scanner covariates, or real IDs.

See [`data/open/README.md`](data/open/README.md) and
[`code/docs/reproducibility.md`](code/docs/reproducibility.md) for layout and
what is intentionally omitted (raw dMRI, CovBat covariates). Controlled export
(lab only): `python -u code/lib/export_tier1_open.py --core` from
`structural_tractometry`.

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
   factor_analysis  →  factor_z-scores  →  Laplacian eigenmaps
                              ↓
              tract/region asymmetry  →  microstructural_asymmetries
```

HCP Young Adult / Aging BIDS conversion uses
[HCPLifespan2BIDS](https://github.com/ellisdg/HCPLifespan2BIDS) (not vendored here).

Docs: [`code/docs/pipeline.md`](code/docs/pipeline.md),
[`code/docs/analysis.md`](code/docs/analysis.md),
[`code/docs/reproducibility.md`](code/docs/reproducibility.md).

## Citation

> *[Publication citation to be added]*
