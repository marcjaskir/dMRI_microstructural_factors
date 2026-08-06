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
2. Download/unpack the OSF HDF5: `python -u code/lib/fetch_open_data.py`
   (or pack locally with `python -u code/lib/pack_open_h5.py pack`).
3. Install the conda environment from `environment.yml`.
4. Set `PYTHONPATH` to include `code/` and run golden tests under `code/tests/golden/`.

## Repository structure

| Path | Role |
|------|------|
| `code/` | Pipelines, analyses, `lib/paths.py`, docs, tests |
| `code/lib/pack_open_h5.py` | Pack/unpack manuscript-reproduction HDF5 |
| `code/lib/fetch_open_data.py` | Download OSF HDF5 into `data/open/` |
| `code/lib/export_tier1_open.py` | Lab-only de-identify export into `data/open/` |
| `code/lib/factor_labels.py` | Canonical F1/F2/F3 paper labels |
| `data/open/` | Publishable products (small committed trees + OSF unpack) |
