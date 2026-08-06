# Reproducibility

## Data layout

```
workspace_root/
  code/                  # This repository's scripts
  data/
    open/                # Publishable products
      atlases/ metadata/ inclusion/   # small; committed
      analysis/ gam/                  # from OSF storage; gitignored
```

### Open products (`data/open/`)

Enough to reproduce factor analysis digests, **Laplacian eigenmaps** gradients,
and asymmetry reports. Prefer the **OSF directory share** of `data/open/`:

1. Clone the repo (committed stubs under `atlases/`, `metadata/`, `inclusion/`).
2. Sync OSF storage into `data/open/` so `gam/` and `analysis/` CSVs exist for
   `lib/paths.py` (`gam_dir`, `analysis_dir`).

Typical contents after sync:

- Post-GAM residual z-scores (`anon_id`, `group`, `*_z`) for kept analyses
  (manuscript scalars/tracts only; see `lib/manuscript_features.py`)
- Factor loadings / scores / factor z
- Control LE gradient products (G1/G2, neuroaxis correlations)
- Tract/region asymmetry summaries and group Cohen’s d / Mahalanobis digests
- Atlas label tables and anonymized inclusion (`laterality`, `lobe`; no age/sex)

**Intentionally omitted:** raw dMRI, CovBat covariates (age/sex/scanner),
reversible ID maps, NIfTI volumes, HTML dumps, diffusion-map gradient trees,
and microstructural scalars / WM tracts outside the manuscript analysis set.

You may delete local `data/open/gam/` and `data/open/analysis/` to save disk;
re-download those folders from OSF into `data/open/`.

### OSF directory (preferred distribution)

```bash
# Set open_osf_url in config.yaml (project or storage link)
# Download/sync OSF files into data/open/ so analysis/ and gam/ exist
```

Placeholder until the OSF project exists: `https://osf.io/XXXXX/`.

### Controlled export (lab only)

From a local controlled workspace (e.g. `structural_tractometry`):

```bash
python -u code/lib/export_tier1_open.py --core   # manuscript core OSF share
# or without --core for fuller manuscript-allowlisted GAM CSVs
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
- `DMRI_MICRO_OSF_URL` — OSF project/storage URL for the open directory

## Manuscript analysis DAG

```
factor_analysis → factor_z-scores → gradients_group-controls (LE)
                                           ↘ gradients_tle_z
factor_analysis → factor_representation / factor_analysis_voxelwise (supp)
tract_asymmetry ┐
region_asymmetry_tle ├→ microstructural_asymmetries
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
