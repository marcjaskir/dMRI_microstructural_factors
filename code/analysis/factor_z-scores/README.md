# Factor z-scores

## Purpose

Apply All4_Combined control loadings to GAM residual z-scores to produce:

- wide **factor scores** (`factor_scores/{controls,epilepsy}_F{1,2,3}_scores.csv`)
- control-normative **factor z** (`factor_z_scores/{controls,epilepsy}_F{1,2,3}_z_scores.csv`)
- per-scalar GAM z tables (`scalar_z-scores/*.csv`) for factor representation

This module is a CSV pipeline only (no HTML / PNG / NIfTI brain maps).

## Entry points

```bash
# from code/analysis/factor_z-scores/
python factor_z-scores.py
python -m factor_z_scores

# reuse existing factor_scores; rewrite factor z only
python -m factor_z_scores --skip-scores --skip-scalar-z
```

Package layout: `factor_z_scores/{config,io,scores,zscores,scalar_z,cli}.py`.

## Inputs

`gam` (mni_micro Glasser + 4S156; pyAFQ HCP1065), `factor_analysis` loadings,
inclusion metadata.

## Outputs

`derivatives/analysis/factor_z-scores` (or `data/open/analysis/factor_z-scores`
when configured for open):

| Path | Consumers |
|------|-----------|
| `factor_scores/*.csv` | LE gradients, asymmetry, voxelwise subject list |
| `factor_z_scores/controls_F*_z_scores.csv` | golden digests |
| `factor_z_scores/epilepsy_F*_z_scores.csv` | `gradients_tle_z`, asymmetry |
| `scalar_z-scores/*.csv` | `factor_representation` |

## Configuration

Paths via [`config.yaml`](../../config.example.yaml) and [`lib/paths.py`](../../lib/paths.py).
Node thirds: end1 1–34, core 35–66, end2 67–100. Factor z: `(x − mean) / sd`
with `MIN_CONTROLS_FOR_ROI_Z = 2` (population SD, `ddof=0`).
