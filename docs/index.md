# dMRI Microstructural Factors — Documentation

Code for diffusion MRI preprocessing, tract microstructure profiling, harmonization, factor analysis, and epilepsy asymmetry reporting.

## Contents

- [Pipeline](pipeline.md) — imaging preprocessing and feature extraction
- [Analysis](analysis.md) — statistical analysis and reporting modules
- [Reproducibility](reproducibility.md) — configuration, data layout, and golden tests

## Getting started

1. Clone the repository and copy `config.example.yaml` to `config.yaml`.
2. Point `project_root` at a workspace containing `data/`, `derivatives/`, and `results/`.
3. Install the conda environment from `environment.yml`.
4. Run pipeline stages in order (see [pipeline.md](pipeline.md)).

## Repository structure

| Directory | Role |
|-----------|------|
| `lib/` | Shared path resolution (`config.yaml` loader) |
| `acpc_mni_xfm/` … `trekker/` | Imaging pipeline stages |
| `covbat/`, `gam/` | Harmonization and normative modeling |
| `analysis/` | Downstream statistical analyses |
| `tests/golden/` | Regression tests against known outputs |
