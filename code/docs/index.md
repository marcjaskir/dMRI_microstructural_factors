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
2. Place the OSF HDF5 at `data/open/dmri_microstructural_factors_open_v1.h5`
   and unpack: `python -u code/lib/fetch_open_data.py` (or `--unpack-only`).
   That creates `data/open/gam/` and `data/open/analysis/` (not in git).
3. Install the conda environment from `environment.yml`.
4. Set `PYTHONPATH` to include `code/` and run golden tests under `code/tests/golden/`.

## Repository structure

| Path | Role |
|------|------|
| `code/` | Pipelines, analyses, `lib/paths.py`, docs, tests |
| `code/lib/pack_open_h5.py` | Pack/unpack manuscript-reproduction HDF5 |
| `code/lib/fetch_open_data.py` | Download OSF HDF5 → `data/open/*.h5`, unpack `gam/`/`analysis/` |
| `code/lib/export_tier1_open.py` | Lab-only de-identify export into `data/open/` |
| `code/lib/factor_labels.py` | Canonical F1/F2/F3 paper labels |
| `data/open/` | Small committed trees + HDF5; `gam/`/`analysis/` from unpack only |
