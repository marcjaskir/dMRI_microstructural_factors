# dMRI Microstructural Factors

Code and data for reproducing results of Jaskir et al. 2026, Mapping Whole-Brain
Factors of Microstructural Similarity with Diffusion MRI.

## Repository layout

```
dMRI_microstructural_factors/
  code/                 # All pipelines and analyses
  data/
    open/               # Small committed trees + OSF HDF5 (see below)
  config.example.yaml
  README.md
```

**HDF5 location (required for manuscript reproduction):**

```
data/open/dmri_microstructural_factors_open_v1.h5
```

Download from OSF into that path (or run `fetch_open_data.py`, which does it).
`data/open/gam/` and `data/open/analysis/` are **not** shipped in git — unpack
the HDF5 to create them. You may delete those trees anytime and re-run
`python -u code/lib/fetch_open_data.py --unpack-only`.

## Quick start

```bash
git clone git@github.com:marcjaskir/dMRI_microstructural_factors.git
cd dMRI_microstructural_factors
cp config.example.yaml config.yaml
# Edit config.yaml: set workspace_root
# After OSF upload: set open_h5_osf_url (or export DMRI_MICRO_OSF_URL)

conda env create -f environment.yml
conda activate dmri_microstructural_factors

export PYTHONPATH="$PWD/code:$PYTHONPATH"

# Downloads → data/open/dmri_microstructural_factors_open_v1.h5, then unpacks gam/ + analysis/
python -u code/lib/fetch_open_data.py

python code/tests/golden/run_golden_tests.py
python code/tests/test_factor_labels.py
python code/tests/test_no_phi.py
```

If you already have the HDF5 file, copy it to
`data/open/dmri_microstructural_factors_open_v1.h5` and run:

```bash
python -u code/lib/fetch_open_data.py --unpack-only
```

If the OSF project is not public yet, pack locally from an existing open tree:

```bash
python -u code/lib/pack_open_h5.py pack --profile core
# writes data/open/dmri_microstructural_factors_open_v1.h5
python -u code/lib/pack_open_h5.py unpack
```

All filesystem roots are defined in `config.yaml` (see
[`config.example.yaml`](config.example.yaml)).

## Data availability and privacy

Manuscript reproduction uses **one OSF-hosted HDF5**
(`dmri_microstructural_factors_open_v1.h5`) containing anonymized tabular
products: inclusion (`anon_id`, laterality/lobe), GAM residual z tables needed
for kept analyses, factor loadings/scores/z, Laplacian-eigenmap gradient CSVs,
and asymmetry digests. No participant age, sex, scanner covariates, or real IDs.

See [`data/open/README.md`](data/open/README.md) for the HDF5 schema and
[`code/docs/reproducibility.md`](code/docs/reproducibility.md) for what is
intentionally omitted (raw dMRI, CovBat covariates). Controlled export (lab
only): `python -u code/lib/export_tier1_open.py` from `structural_tractometry`.

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
