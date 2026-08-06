# Factor analysis

## Purpose

Control factor analysis / PCA on GAM residual **z-scores** (GM parcels + WM
thirds). Factors interpret as overall, non-Gaussian, and anisotropic diffusivity
(F1/F2/F3; see `lib/factor_labels.py`).

## Entry points

`factor_analysis.py`

## Inputs

`gam`

## Outputs

`derivatives/analysis/factor_analysis`

## Configuration

Paths are resolved via [`config.yaml`](../../config.example.yaml) and [`lib/paths.py`](../../lib/paths.py).
