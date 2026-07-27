# Analysis modules

Downstream analyses consuming GAM z-scores, CovBat outputs, and asymmetry derivatives.

## Cohort and QC

| Module | Purpose | Entry points |
|--------|---------|--------------|
| [inclusion](../analysis/inclusion/) | Penn epilepsy / control inclusion criteria | `penn_epilepsy_inclusion.py`, `controls_inclusion.py` |
| [qc](../analysis/qc/) | QC correlations; BundleSeg QC | `qc.py`, `bundleseg_qc.py` |

## Factor analysis and gradients

| Module | Purpose | Entry points |
|--------|---------|--------------|
| [factor_analysis](../analysis/factor_analysis/) | Control FA/PCA on GAM z (GM + WM) | `factor_analysis.py` |
| [factor_z-scores](../analysis/factor_z-scores/) | Apply loadings; patient/control factor z | `factor_z-scores.py` |
| [factor_representation](../analysis/factor_representation/) | Factor representation Upset plots | `factor_representation_upset.py` |
| [factor_analysis_voxelwise](../analysis/factor_analysis_voxelwise/) | Voxelwise FA + neuromaps | `factor_analysis_voxelwise.py` |
| [gradients_group-controls](../analysis/gradients_group-controls/) | Control DM/LE gradients (BrainSpace) | `compute_gradients.py` |
| [gradients_tle_z](../analysis/gradients_tle_z/) | TLE z-aligned gradients | `compute_gradients_tle_z.py` |
| [gradients_voxelwise](../analysis/gradients_voxelwise/) | Voxelwise gradient analysis | `gradient_lib/` |

## Asymmetry pipeline

| Module | Purpose | Entry points |
|--------|---------|--------------|
| [tract_asymmetry](../analysis/tract_asymmetry/) | WM tract scalar asymmetry (Cohen's d) | `cli.py` |
| [tract_asymmetry_normative](../analysis/tract_asymmetry_normative/) | Normative WM covariance for Mahalanobis | `compute_tract_covariance.py` |
| [region_asymmetry_tle](../analysis/region_asymmetry_tle/) | GM region asymmetry (TLE) | `cli.py` |
| [region_asymmetry_tle_normative](../analysis/region_asymmetry_tle_normative/) | Normative GM covariance | `compute_region_covariance.py` |
| [asymmetry_tle](../analysis/asymmetry_tle/) | GAM-z asymmetry HTML maps | `asymmetry_tle.py` |
| [asymmetry_tle_covbat_pyafq](../analysis/asymmetry_tle_covbat_pyafq/) | Pre/post CovBat profile asymmetry | `asymmetry_tle_covbat_pyafq.py` |
| [asymmetry_tle_region](../analysis/asymmetry_tle_region/) | Region/tract asymmetry summaries | `asymmetry_tle_region.py` |
| [microstructural_asymmetries](../analysis/microstructural_asymmetries/) | Group Cohen's d / Mahalanobis HTML reports | `microstructural_asymmetry_report_*.py` |
| [asymmetry_correlations](../analysis/asymmetry_correlations/) | Cross-ROI asymmetry correlation matrices | `asymmetry_*_correlation_matrix.py` |
| [within_patient_scalar_asymmetries](../analysis/within_patient_scalar_asymmetries/) | Per-patient scalar asymmetry panels | `within_patient_scalar_asymmetry.py` |

## Examples and visualization

| Module | Purpose |
|--------|---------|
| [covbat_example](../analysis/covbat_example/) | CovBat effect illustration notebook |
| [profile_thirds_example](../analysis/profile_thirds_example/) | Normative along-tract thirds example |
| [intervention_scalar_viz](../analysis/intervention_scalar_viz/) | Intervention scalar visualization |

## Typical analysis order

1. `inclusion` → subject lists in `results/<inclusion_subdir>/`
2. `factor_analysis` → `factor_z-scores` → `gradients_*`
3. `tract_asymmetry` + `region_asymmetry_tle` (+ normative modules)
4. `microstructural_asymmetries`, `asymmetry_correlations`, `within_patient_scalar_asymmetries`

## Outputs

Analysis outputs are written under `derivatives/analysis/<module_name>/`. Paths are resolved via `lib/paths.py`.
