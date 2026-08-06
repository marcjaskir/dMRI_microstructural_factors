# Analysis modules

Downstream analyses consuming GAM residual **z-scores**, factor loadings, and
ipsilateral–contralateral asymmetry derivatives for the manuscript.

Terminology: **overall / non-Gaussian / anisotropic diffusivity** (F1/F2/F3);
GAM normative z-scores; **Laplacian eigenmaps** (G1/G2); Cohen’s d and
**Mahalanobis** asymmetry. Datasets: Penn, HCP-YA, HCP-Aging, TLE.

## QC

| Module | Purpose | Entry points |
|--------|---------|--------------|
| [qc](../analysis/qc/) | QC correlations; BundleSeg QC | `qc.py`, `bundleseg_qc.py` |

## Factor analysis and gradients

| Module | Purpose | Entry points |
|--------|---------|--------------|
| [factor_analysis](../analysis/factor_analysis/) | Control FA/PCA on GAM z (GM + WM) | `factor_analysis.py` |
| [factor_z-scores](../analysis/factor_z-scores/) | Apply loadings → wide factor scores / normative z / scalar z (CSV only) | `factor_z-scores.py` / `python -m factor_z_scores` |
| [factor_representation](../analysis/factor_representation/) | Statistic vs factor-score representation | `factor_representation.py` |
| [factor_analysis_voxelwise](../analysis/factor_analysis_voxelwise/) | Voxelwise FA (supplement) | `factor_analysis_voxelwise.py` |
| [gradients_group-controls](../analysis/gradients_group-controls/) | Control **Laplacian eigenmaps** (BrainSpace) | `compute_gradients.py` |
| [gradients_tle_z](../analysis/gradients_tle_z/) | TLE z-aligned gradients on control LE axes | `compute_gradients_tle_z.py` |

## Asymmetry pipeline

| Module | Purpose | Entry points |
|--------|---------|--------------|
| [tract_asymmetry](../analysis/tract_asymmetry/) | WM tract scalar asymmetry (Cohen's d) | `cli.py` |
| [tract_asymmetry_normative](../analysis/tract_asymmetry_normative/) | Normative WM covariance for Mahalanobis | `compute_tract_covariance.py` |
| [region_asymmetry_tle](../analysis/region_asymmetry_tle/) | GM region asymmetry (TLE) | `cli.py` |
| [region_asymmetry_tle_normative](../analysis/region_asymmetry_tle_normative/) | Normative GM covariance | `compute_region_covariance.py` |
| [microstructural_asymmetries](../analysis/microstructural_asymmetries/) | Group Cohen's d / Mahalanobis reports & LaTeX | `microstructural_asymmetry_report_*.py` |

## Examples and visualization

| Module | Purpose |
|--------|---------|
| [covbat_example](../analysis/covbat_example/) | CovBat effect illustration (Methods schematic) |
| [profile_thirds_example](../analysis/profile_thirds_example/) | Normative along-tract thirds example |

## Typical analysis order

1. `factor_analysis` → `factor_z-scores` → `gradients_group-controls` (LE) / `gradients_tle_z`
2. `tract_asymmetry` + `region_asymmetry_tle` (+ normative covariance modules)
3. `microstructural_asymmetries` group reports

## Outputs

Analysis outputs are written under `derivatives/analysis/<module_name>/` (or the
open mirror `data/open/analysis/`). Paths are resolved via `lib/paths.py`.
Canonical factor labels live in `lib/factor_labels.py`.
