#!/usr/bin/env python3
"""
Epilepsy TLE factor z-score coloring on controls Laplacian G2-vs-G1 gradient scatters.

Reuses controls gradient positions from ``gradients_group-controls`` Laplacian CSVs and
colors each ROI by the epilepsy group-mean factor z-score (signed or mean |z|).

Outputs under ``derivatives/analysis/gradients_tle_z/laplacian_eigenmodes/figures/gradients-2/``:

* ``gradients_by-tle-z_signed_cohort-epilepsy.png``
* ``gradients_by-tle-z_absolute_cohort-epilepsy.png``
* ``gradients_by-tle-z_signed_cohort-epilepsy_roi-markers.png``
* ``gradients_by-tle-z_absolute_cohort-epilepsy_roi-markers.png``
* ``gradients_by-tle-z_signed_cohort-epilepsy_roi-markers_LR.png``
* ``gradients_by-tle-z_absolute_cohort-epilepsy_roi-markers_LR.png``
* ``legend-tle-z_signed.png``
* ``legend-tle-z_absolute.png``
* ``legend-roi-markers.png``
* ``legend-roi-markers_LR.png``
* ``factor_z_correlations_lollipop_cohort-epilepsy.png``
* ``gradients_by-tle-z_summary.png``
* ``factor_z_correlations_cohort-epilepsy.csv`` (under ``csv/gradients-2/``)
* ``epilepsy_F2_mean_z_gt0_bin.nii.gz`` (under ``nii/``; Non-Gaussian mean z > 0, binary)
* ``epilepsy_F2_mean_z_gt0.nii.gz`` (under ``nii/``; same ROIs with continuous mean z)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Georgia", "DejaVu Serif", "serif"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.titlesize": 16,
    }
)

from gradient_lib.config import (  # noqa: E402
    DEFAULT_GRADIENTS_CONTROLS_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRACTOMETRY_ROOT,
    DEFAULT_Z_SCORES_DIR,
    GRADIENTS_K,
    METHOD_TAG,
)
from gradient_lib.correlations import build_factor_z_correlation_table  # noqa: E402
from gradient_lib.io import (  # noqa: E402
    build_scatter_rows,
    parse_epilepsy_z_files,
    save_aggregated_z_csvs,
)
from gradient_lib.nii_positive_z import (  # noqa: E402
    default_mean_z_csv,
    default_nii_out_dir,
    write_positive_mean_z_binary_nii,
    write_positive_mean_z_continuous_nii,
)
from gradient_lib.plots_lollipop import plot_factor_z_correlation_lollipop  # noqa: E402
from gradient_lib.plots_scatter import (  # noqa: E402
    plot_gradients_by_tle_z_scatter,
    save_standalone_legend_roi_markers,
    save_standalone_legend_roi_markers_lr,
    save_standalone_legend_tle_z,
)
from gradient_lib.plots_summary import plot_gradients_by_tle_z_summary  # noqa: E402
from gradient_lib.roi_markers import DEFAULT_ROI_MARKERS  # noqa: E402


def _default_gradients_csv_dir(gradients_controls_dir: Path) -> Path:
    nested = (
        gradients_controls_dir
        / METHOD_TAG
        / "csv"
        / f"gradients-{GRADIENTS_K}"
    )
    probe = "F1_principal_gradient1_scores_cohort-controls.csv"
    if (nested / probe).is_file():
        return nested
    # Flattened open layout: CSVs live directly under gradients_group-controls/
    if (gradients_controls_dir / probe).is_file():
        return gradients_controls_dir
    return nested


def _default_figures_dir(output_dir: Path) -> Path:
    return output_dir / METHOD_TAG / "figures" / f"gradients-{GRADIENTS_K}"


def _default_csv_out_dir(output_dir: Path) -> Path:
    return output_dir / METHOD_TAG / "csv" / f"gradients-{GRADIENTS_K}"


def _parse_factors(arg: str | None) -> set[int] | None:
    if arg is None or not str(arg).strip():
        return None
    return {int(x.strip()) for x in str(arg).split(",") if x.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot controls Laplacian G2 vs G1 scatters colored by epilepsy "
            "group-mean factor z-scores."
        ),
    )
    parser.add_argument(
        "--gradients-csv-dir",
        type=Path,
        default=None,
        help=(
            "Directory with F{k}_principal_gradient{1,2}_scores_cohort-controls.csv "
            f"(default: {METHOD_TAG}/csv/gradients-{GRADIENTS_K} under gradients_group-controls)"
        ),
    )
    parser.add_argument(
        "--z-scores-dir",
        type=Path,
        default=DEFAULT_Z_SCORES_DIR,
        help="Directory containing epilepsy_F{n}_z_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for gradients_tle_z derivatives",
    )
    parser.add_argument(
        "--gradients-controls-dir",
        type=Path,
        default=DEFAULT_GRADIENTS_CONTROLS_DIR,
        help="Root of gradients_group-controls outputs (used when --gradients-csv-dir is omitted)",
    )
    parser.add_argument(
        "--factors",
        type=str,
        default=None,
        help="Comma-separated factor indices to include (default: all epilepsy_F* found)",
    )
    parser.add_argument(
        "--no-aggregated-csv",
        action="store_true",
        help="Skip writing per-ROI aggregated epilepsy z-score CSVs",
    )
    args = parser.parse_args(argv)

    factors_filter = _parse_factors(args.factors)
    z_dir = Path(args.z_scores_dir)
    output_dir = Path(args.output_dir)
    gradients_csv_dir = (
        Path(args.gradients_csv_dir)
        if args.gradients_csv_dir is not None
        else _default_gradients_csv_dir(Path(args.gradients_controls_dir))
    )
    figures_dir = _default_figures_dir(output_dir)

    factor_z_paths = parse_epilepsy_z_files(z_dir, factors_filter)
    if not factor_z_paths:
        print(f"No epilepsy_F*_z_scores.csv found under {z_dir}")
        return 1

    print(f"Factors: {[t for t, _ in factor_z_paths]}")
    print(f"Gradient CSVs: {gradients_csv_dir}")
    print(f"Z-scores: {z_dir}")
    print(f"Figures: {figures_dir}")

    if not args.no_aggregated_csv:
        save_aggregated_z_csvs(
            z_dir,
            factor_z_paths,
            _default_csv_out_dir(output_dir),
        )

    for color_mode, stem in (("signed", "signed"), ("absolute", "absolute")):
        rows = build_scatter_rows(
            gradients_csv_dir=gradients_csv_dir,
            z_dir=z_dir,
            factor_z_paths=factor_z_paths,
            absolute=(color_mode == "absolute"),
        )
        scatter_path = figures_dir / f"gradients_by-tle-z_{stem}_cohort-epilepsy.png"
        scatter_roi_path = (
            figures_dir / f"gradients_by-tle-z_{stem}_cohort-epilepsy_roi-markers.png"
        )
        scatter_roi_lr_path = (
            figures_dir / f"gradients_by-tle-z_{stem}_cohort-epilepsy_roi-markers_LR.png"
        )
        legend_path = figures_dir / f"legend-tle-z_{stem}.png"
        plot_gradients_by_tle_z_scatter(rows, scatter_path, color_mode=color_mode)
        plot_gradients_by_tle_z_scatter(
            rows,
            scatter_roi_path,
            color_mode=color_mode,
            roi_markers=DEFAULT_ROI_MARKERS,
        )
        plot_gradients_by_tle_z_scatter(
            rows,
            scatter_roi_lr_path,
            color_mode=color_mode,
            roi_markers_lr=True,
        )
        save_standalone_legend_tle_z(rows, legend_path, color_mode=color_mode)
        print(f"Wrote {scatter_path}")
        print(f"Wrote {scatter_roi_path}")
        print(f"Wrote {scatter_roi_lr_path}")
        print(f"Wrote {legend_path}")

    roi_legend_path = figures_dir / "legend-roi-markers.png"
    save_standalone_legend_roi_markers(roi_legend_path)
    print(f"Wrote {roi_legend_path}")
    roi_lr_legend_path = figures_dir / "legend-roi-markers_LR.png"
    save_standalone_legend_roi_markers_lr(roi_lr_legend_path)
    print(f"Wrote {roi_lr_legend_path}")

    corr_df = build_factor_z_correlation_table(
        gradients_csv_dir=gradients_csv_dir,
        factor_z_paths=factor_z_paths,
        tractometry_root=DEFAULT_TRACTOMETRY_ROOT,
    )
    corr_csv_path = _default_csv_out_dir(output_dir) / "factor_z_correlations_cohort-epilepsy.csv"
    corr_csv_path.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(corr_csv_path, index=False)
    print(f"Wrote {corr_csv_path}")

    lollipop_path = figures_dir / "factor_z_correlations_lollipop_cohort-epilepsy.png"
    plot_factor_z_correlation_lollipop(corr_df, lollipop_path)
    print(f"Wrote {lollipop_path}")

    signed_rows = build_scatter_rows(
        gradients_csv_dir=gradients_csv_dir,
        z_dir=z_dir,
        factor_z_paths=factor_z_paths,
        absolute=False,
    )
    summary_path = figures_dir / "gradients_by-tle-z_summary.png"
    plot_gradients_by_tle_z_summary(
        signed_rows,
        corr_df,
        summary_path,
        color_mode="signed",
        roi_markers=DEFAULT_ROI_MARKERS,
    )
    print(f"Wrote {summary_path}")

    # Non-Gaussian (F2) group-mean z > 0 → binary MNI mask (Glasser + 4S156 + tract thirds).
    f2_mean_csv = default_mean_z_csv(output_dir, "F2")
    if f2_mean_csv.is_file():
        nii_dir = default_nii_out_dir(output_dir)
        bin_path = nii_dir / "epilepsy_F2_mean_z_gt0_bin.nii.gz"
        cont_path = nii_dir / "epilepsy_F2_mean_z_gt0.nii.gz"
        _, bin_counts = write_positive_mean_z_binary_nii(f2_mean_csv, bin_path)
        _, cont_counts = write_positive_mean_z_continuous_nii(f2_mean_csv, cont_path)
        print(
            f"Wrote {bin_path} "
            f"(rois={bin_counts['n_rois']}, voxels={bin_counts['n_voxels']}, "
            f"glasser={bin_counts['n_glasser']}, 4S={bin_counts['n_4s156']}, "
            f"tracts={bin_counts['n_tract_thirds']}, unmatched={bin_counts['n_unmatched']})"
        )
        print(
            f"Wrote {cont_path} "
            f"(rois={cont_counts['n_rois']}, voxels={cont_counts['n_voxels']}, "
            f"glasser={cont_counts['n_glasser']}, 4S={cont_counts['n_4s156']}, "
            f"tracts={cont_counts['n_tract_thirds']}, unmatched={cont_counts['n_unmatched']})"
        )
    else:
        print(f"Skipping F2 positive-z NIfTI; missing {f2_mean_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
