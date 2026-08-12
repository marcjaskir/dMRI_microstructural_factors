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

### Data distribution

Data necessary for replication is available here: [https://osf.io/xsr7y](https://osf.io/xsr7y)  

## Configuration

Copy `config.example.yaml` → `config.yaml` at the workspace root and set
`workspace_root`. Defaults point all manuscript-reproduction roots at
`data/open/`. Optional controlled_* keys are unused for open figure
replication.

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
