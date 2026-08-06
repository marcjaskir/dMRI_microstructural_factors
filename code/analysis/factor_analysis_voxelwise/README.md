# Voxelwise factors

## Purpose

Voxelwise factor analysis from QSIRecon MNI maps + neuromaps correlations
(supplement). ROI Laplacian eigenmaps live in `gradients_group-controls`.

## Entry points

`factor_analysis_voxelwise.py`, `factor_analysis_group_mean.py`,
`compute_neuromaps_correlations.py`

## Inputs

qsirecon MNI scalar maps; factor loadings from `factor_analysis`

## Outputs

`derivatives/analysis/factor_analysis_voxelwise`

## Configuration

Paths are resolved via [`config.yaml`](../../config.example.yaml) and [`lib/paths.py`](../../lib/paths.py).
Neuromaps helpers are vendored under `gradient_lib/` in this module.
