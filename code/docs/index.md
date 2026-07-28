# dMRI Microstructural Factors — Documentation

Code for diffusion MRI preprocessing, tract microstructure profiling,
harmonization, factor analysis, and epilepsy asymmetry reporting.

## Contents

- [Pipeline](pipeline.md) — imaging preprocessing and feature extraction (under `code/`)
- [Analysis](analysis.md) — statistical analysis modules (under `code/analysis/`)
- [Reproducibility](reproducibility.md) — data layout, config, golden tests

Paths in this docs set refer to modules under `code/` unless noted as `data/open/` or `data/controlled/`.

## Getting started

1. Clone the repository and copy `config.example.yaml` to `config.yaml` at the
   workspace root.
2. Point `data_controlled_dir` (and related keys) at your imaging workspace, or
   populate `data/open/` via `python -u code/lib/export_tier1_open.py`.
3. Install the conda environment from `environment.yml`.
4. Set `PYTHONPATH` to include `code/`.

## Repository structure

| Path | Role |
|------|------|
| `code/` | Pipelines, analyses, `lib/paths.py`, docs, tests |
| `code/lib/export_tier1_open.py` | De-identify and populate `data/open/` |
| `data/open/` | Publishable analysis products |
| `data/controlled/` | Local controlled inputs (gitignored) |
