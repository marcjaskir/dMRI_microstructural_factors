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

**Open data for manuscript reproduction:** download from OSF
([https://osf.io/xsr7y](https://osf.io/xsr7y)) and place under `data/open/` so
`analysis/` and `gam/` exist beside the committed `atlases/` / `metadata/` /
`inclusion/` stubs. Copy `config.example.yaml` → `config.yaml` and set
`workspace_root` (see [`config.example.yaml`](config.example.yaml)).

## Data availability and privacy

Manuscript reproduction uses an **OSF-hosted `data/open/` archive**
([https://osf.io/xsr7y](https://osf.io/xsr7y)) of anonymized tabular products:
inclusion (`anon_id`, laterality/lobe), manuscript-allowlisted GAM residual-z
tables (`pyafq/` + `mni_micro/`), factor loadings/z, Laplacian-eigenmap gradient
CSVs, and asymmetry digests. No participant age, sex, scanner covariates, or
real IDs.

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
