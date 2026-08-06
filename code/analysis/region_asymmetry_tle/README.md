# Region asymmetry TLE

## Purpose

GM region asymmetry from GAM mni_micro z-scores, plus normative covariance for
Mahalanobis asymmetry.

## Entry points

| Script | Role |
|--------|------|
| `cli.py` / `python -m region_asymmetry_tle` | Per-subject asymmetry CSVs |
| `compute_region_covariance.py` | Control normative cov / invcov |

## Inputs

`gam` (mni_micro)

## Outputs

| Path | Contents |
|------|----------|
| `derivatives/analysis/region_asymmetry_tle` | Subject asymmetry CSVs |
| `derivatives/analysis/region_asymmetry_tle_normative` | Per-region cov matrices |

## Configuration

Paths are resolved via [`config.yaml`](../../config.example.yaml) and [`lib/paths.py`](../../lib/paths.py).
