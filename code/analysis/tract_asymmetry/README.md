# Tract asymmetry

## Purpose

WM tract scalar asymmetry (Cohen's d) from GAM pyAFQ z-scores, plus normative
covariance for Mahalanobis asymmetry.

## Entry points

| Script | Role |
|--------|------|
| `cli.py` / `python -m tract_asymmetry` | Per-subject asymmetry CSVs |
| `compute_tract_covariance.py` | Control normative cov / invcov |

## Inputs

`gam` (pyAFQ)

## Outputs

| Path | Contents |
|------|----------|
| `derivatives/analysis/tract_asymmetry` | Subject asymmetry CSVs |
| `derivatives/analysis/tract_asymmetry_normative` | Segment/node cov matrices |

## Configuration

Paths are resolved via [`config.yaml`](../../config.example.yaml) and [`lib/paths.py`](../../lib/paths.py).
