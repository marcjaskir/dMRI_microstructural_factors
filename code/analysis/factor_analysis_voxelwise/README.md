# Voxelwise factors

## Purpose

Voxelwise factor analysis from QSIRecon MNI maps (supplement). ROI Laplacian
eigenmaps live in `gradients_group-controls`.

## Entry points

`factor_analysis_voxelwise.py`, `factor_analysis_group_mean.py`

## Inputs

qsirecon MNI scalar maps; factor loadings from `factor_analysis`

## Outputs

`derivatives/analysis/factor_analysis_voxelwise`

## Configuration

Paths are resolved via [`config.yaml`](../../config.example.yaml) and [`lib/paths.py`](../../lib/paths.py).
Shared helpers for group-mean factor maps live under `gradient_lib/`.
