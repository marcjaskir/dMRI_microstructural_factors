# dMRI Microstructural Factors — Documentation

Code for diffusion MRI preprocessing, tract microstructure profiling,
harmonization, factor analysis, and epilepsy asymmetry reporting.

## Contents

- [Pipeline](pipeline.md) — imaging preprocessing and feature extraction (under `code/`)
- [Analysis](analysis.md) — statistical analysis modules (under `code/analysis/`)
- [Reproducibility](reproducibility.md) — data layout, config, golden tests

Paths in this docs set refer to modules under `code/` unless noted as `data/open/`.

## Getting started

1. Clone the repository and copy `config.example.yaml` to `config.yaml` at the
   workspace root; set `workspace_root`.
2. Sync OSF storage into `data/open/` so `analysis/` and `gam/` exist beside the
   committed `atlases/` / `metadata/` / `inclusion/` stubs (see
   `data/open/README.md`). Set `open_osf_url` in `config.yaml` to the project
   or storage link.
3. Install the conda environment from `environment.yml`.
4. Set `PYTHONPATH` to include `code/` and run golden tests under `code/tests/golden/`.

## Repository structure

| Path | Role |
|------|------|
| `code/` | Pipelines, analyses, `lib/paths.py`, docs, tests |
| `code/lib/export_tier1_open.py` | Lab-only de-identify export into `data/open/` |
| `code/lib/factor_labels.py` | Canonical F1/F2/F3 paper labels |
| `data/open/` | Committed stubs + OSF-synced `analysis/` / `gam/` |
