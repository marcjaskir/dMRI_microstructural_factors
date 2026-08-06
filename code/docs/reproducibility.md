# Reproducibility

## Data layout

```
workspace_root/
  code/                  # This repository's scripts
  data/
    open/                # Publishable products (small trees + unpacked OSF HDF5)
      atlases/ metadata/ inclusion/ gam/ analysis/
      dmri_microstructural_factors_open_v1.h5   # gitignored cache
```

### Open products (`data/open/`)

Enough to reproduce factor analysis digests, **Laplacian eigenmaps** gradients,
and asymmetry reports:

- Post-GAM residual z-scores (`anon_id`, `group`, `*_z`) for kept analyses
  (manuscript scalars/tracts only; see `lib/manuscript_features.py`)
- Factor loadings / scores / factor z
- Control LE gradient products (G1/G2, neuroaxis correlations)
- Tract/region asymmetry summaries and group Cohen’s d / Mahalanobis digests
- Atlas label tables and anonymized inclusion (`laterality`, `lobe`; no age/sex)

**Intentionally omitted:** raw dMRI, CovBat covariates (age/sex/scanner),
reversible ID maps, NIfTI volumes, HTML dumps, diffusion-map gradient trees,
and microstructural scalars / WM tracts outside the manuscript analysis set.

### OSF HDF5 (preferred distribution)

```bash
# Set open_h5_osf_url in config.yaml or DMRI_MICRO_OSF_URL
python -u code/lib/fetch_open_data.py
```

Pack / unpack / PHI schema check (lab):

```bash
python -u code/lib/pack_open_h5.py pack --profile core
python -u code/lib/pack_open_h5.py check
python -u code/lib/pack_open_h5.py unpack
```

Placeholder until the OSF project exists: `OSF_URL=https://osf.io/XXXXX/`.

### Controlled export (lab only)

From a local controlled workspace (e.g. `structural_tractometry`):

```bash
python -u code/lib/export_tier1_open.py
```

Maps real `sub` → `anon_id`, drops demographics, writes under `data/open/`. The
reversible map stays in `data/controlled/` (gitignored).

## Configuration

Copy `config.example.yaml` → `config.yaml` at the workspace root and set
`workspace_root`. Defaults point all manuscript-reproduction roots at
`data/open/`. Optional controlled_* keys are unused for open figure
replication.

Environment overrides:

- `DMRI_MICRO_CONFIG` — path to config file
- `DMRI_MICRO_ROOT` — override `workspace_root`
- `DMRI_MICRO_OPEN` — override `data_open_dir`
- `DMRI_MICRO_OSF_URL` — OSF download URL for the open HDF5

## Manuscript analysis DAG

```
factor_analysis → factor_z-scores → gradients_group-controls (LE)
                                           ↘ gradients_tle_z
factor_analysis → factor_representation / factor_analysis_voxelwise (supp)
tract_asymmetry (+ normative) ┐
region_asymmetry_tle (+ normative) ├→ microstructural_asymmetries
```

Factor labels (F1 overall, F2 non-Gaussian, F3 anisotropic) are centralized in
`code/lib/factor_labels.py`.

## Golden-output tests

```bash
export PYTHONPATH="$PWD/code:$PYTHONPATH"
python code/tests/golden/run_golden_tests.py capture  # first run / justified refresh
python code/tests/golden/run_golden_tests.py
python code/tests/test_factor_labels.py
python code/tests/test_no_phi.py
```

Baselines under `code/tests/golden/baseline/` (gitignored) cover profile means,
tract asymmetry summaries, and manuscript table digests (`rtol≈1e-6`).

## Singularity images

QSIPrep, QSIRecon, and Trekker `.sif` files are not shipped. Configure local
image paths in your environment.

## Python / R environment

Create the conda env from the repo root:

```bash
conda env create -f environment.yml
conda activate dmri_microstructural_factors
```

Then install the CovBat R package (not on conda-forge):

```bash
Rscript -e "remotes::install_github('andy1764/ComBatFamily')"
```
