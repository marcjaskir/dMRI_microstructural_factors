# Reproducibility

## Data layout

```
workspace_root/
  code/                  # This repository's scripts
  data/
    open/                # Publishable products
      gam/ analysis/ atlases/ metadata/ inclusion/
    controlled/          # Local / DUA only (gitignored)
      metadata/ derivatives/ inclusion/
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

This writes de-identified products under `data/open/` and the reversible
`sub`→`anon_id` map under `data/controlled/anon_id_map.csv` (gitignored).
Small trees (`atlases/`, `metadata/`, `inclusion/`) can be committed; `gam/` and
`analysis/` stay local / Zenodo (gitignored).

### Controlled inputs (`data/controlled/`)

Required only to **recompute** CovBat/GAM:

- Subject-level pyAFQ profiles / mni_micro HDF5
- `age`, `sex`, scanner batch IDs
- Identifiable inclusion / clinical tables

## Configuration

Copy `config.example.yaml` → `config.yaml` at the workspace root.

For local development without copying terabytes of data, point controlled roots
at an existing workspace:

```yaml
data_controlled_dir: /path/to/structural_tractometry
gam_dir: ${data_controlled_dir}/derivatives/gam
analysis_dir: ${data_controlled_dir}/derivatives/analysis
atlas_dir: ${data_controlled_dir}/data/atlases
inclusion_dir: ${data_controlled_dir}/results/1_inclusion
```

Environment overrides:

- `DMRI_MICRO_CONFIG` — path to config file
- `DMRI_MICRO_ROOT` — override `workspace_root`
- `DMRI_MICRO_OPEN` / `DMRI_MICRO_CONTROLLED` — override open / controlled roots

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
