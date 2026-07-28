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
2. Populate `data/open/` via `python -u code/lib/export_tier1_open.py` (or use
   shipped/local open products).
3. Install the conda environment from `environment.yml`.
4. Set `PYTHONPATH` to include `code/`.

## Repository structure

| Path | Role |
|------|------|
| `code/` | Pipelines, analyses, `lib/paths.py`, docs, tests |
| `code/lib/export_tier1_open.py` | De-identify and populate `data/open/` |
| `data/open/` | Publishable analysis products |
