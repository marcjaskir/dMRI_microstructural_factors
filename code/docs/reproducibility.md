# Reproducibility

## Data layout

```
workspace_root/
  code/                  # This repository's scripts
  data/
    open/                # Publishable products
      gam/ analysis/ atlases/ metadata/ inclusion/
```

### Open products (`data/open/`)

Enough to reproduce factor analysis, gradients, and asymmetry reports:

- Post-GAM residual z-scores (`anon_id`, `group`, `*_z`)
- Factor loadings / scores / asymmetry products
- Atlas label tables and tract metadata
- Anonymized inclusion (`laterality`, `lobe`; no age/sex)

Export from a local controlled workspace (e.g. `structural_tractometry`):

```bash
python -u code/lib/export_tier1_open.py
```

This writes de-identified products under `data/open/`. Small trees
(`atlases/`, `metadata/`, `inclusion/`) can be committed; `gam/` and
`analysis/` stay local / Zenodo (gitignored).

## Configuration

Copy `config.example.yaml` → `config.yaml` at the workspace root and set
`workspace_root`. Defaults point all manuscript-reproduction roots at
`data/open/`. Optional controlled_* keys are unused for open figure
replication.

Environment overrides:

- `DMRI_MICRO_CONFIG` — path to config file
- `DMRI_MICRO_ROOT` — override `workspace_root`
- `DMRI_MICRO_OPEN` — override `data_open_dir`

## Golden-output tests

```bash
export PYTHONPATH="$PWD/code:$PYTHONPATH"
python code/tests/golden/run_golden_tests.py capture  # first run
python code/tests/golden/run_golden_tests.py
```

Baselines are written under `code/tests/golden/baseline/` (gitignored) and use
`anon_id` rather than real subject identifiers.

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
