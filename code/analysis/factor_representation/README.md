# Factor representation

## Purpose

Measure how well model statistics (NODDI, MAP-MRI, DKI) represent whole-brain
factor-score gradients (overall / non-Gaussian / anisotropic diffusivity).

## Entry points

`factor_representation.py`

## Inputs

`factor_analysis`, `factor_z-scores`

## Outputs

`derivatives/analysis/factor_representation` — similarity / triplet CSVs and
factor-bar figures

## Configuration

Paths are resolved via [`config.yaml`](../../config.example.yaml) and [`lib/paths.py`](../../lib/paths.py).
