# dMRI Microstructural Factors — Documentation

Code for diffusion MRI preprocessing, tract microstructure profiling,
harmonization, factor analysis, and epilepsy asymmetry reporting.

## Contents

- [Pipeline](pipeline.md) — imaging preprocessing and feature extraction (under `code/`)
- [Analysis](analysis.md) — statistical analysis modules (under `code/analysis/`)
- [Reproducibility](reproducibility.md) — data layout and config

Paths in this docs set refer to modules under `code/` unless noted as `data/open/`.

## Getting started

1. Clone the repository and copy `config.example.yaml` to `config.yaml` at the
   workspace root; set `workspace_root`.
2. Download open data from [https://osf.io/xsr7y](https://osf.io/xsr7y) into
   `data/open/` (zip extract or storage sync) so `analysis/` and `gam/` exist
   beside the committed stubs (see `data/open/README.md`).
3. Install the conda environment from `environment.yml` and set `PYTHONPATH` to
   include `code/`.

## Repository structure

| Path | Role |
|------|------|
| `code/` | Pipelines, analyses, `lib/paths.py`, docs |
| `code/lib/export_tier1_open.py` | Lab-only de-identify export into `data/open/` |
| `code/lib/factor_labels.py` | Canonical F1/F2/F3 paper labels |
| `data/open/` | Committed stubs + OSF-synced `analysis/` / `gam/` |
