"""Emit all voxelwise gradient figures (probseg + atlas-encoded)."""

from __future__ import annotations

from pathlib import Path

from .config import COHORT_TAG, DEFAULT_MASK_NII, DEFAULT_TRACTOMETRY_ROOT, tissue_classes_for_csf_mode
from .parcel_gradients import save_parcel_gradient_csvs, voxel_rows_to_parcel_gradient_run_rows
from .plots_bars_voxelwise import plot_groups_axes_bars
from .plots_scatter_voxelwise import (
    plot_gradient1_summary,
    plot_gradient_summary,
    plot_gradients_by_tissue,
    plot_gradients_by_yeo_mesulam,
    save_standalone_legend_gradient1,
    save_standalone_legend_tissue,
)
from .types import VoxelGradientRunRow
from .yeo_mesulam_voxelwise import (
    save_standalone_legend_mesulam_voxelwise,
    save_standalone_legend_yeo_voxelwise,
)


def emit_all_figures(
    rows: list[VoxelGradientRunRow],
    figures_dir: Path,
    output_dir: Path,
    *,
    mask_nii: Path | None = None,
    csf_mode: str | None = None,
    tractometry_root: Path | None = None,
) -> None:
    """Write probseg tissue, atlas region-group, Yeo/Mesulam, and summary figures."""
    cache_dir = output_dir / "_cache"
    figures_dir.mkdir(parents=True, exist_ok=True)
    root = tractometry_root or DEFAULT_TRACTOMETRY_ROOT
    mask_path = Path(mask_nii) if mask_nii is not None else DEFAULT_MASK_NII
    tissue_classes = tissue_classes_for_csf_mode(csf_mode)

    from .parcel_gradients import clear_atlas_cache, get_atlas_context

    clear_atlas_cache()
    atlas = get_atlas_context(cache_dir=cache_dir, mask_nii=mask_path)

    save_parcel_gradient_csvs(
        rows, output_dir / "csv", atlas=atlas, cohort_tag=COHORT_TAG
    )
    parcel_rows = voxel_rows_to_parcel_gradient_run_rows(rows, atlas=atlas)

    tissue_p = figures_dir / f"gradients_by-tissue_cohort-{COHORT_TAG}.png"
    plot_gradients_by_tissue(
        rows,
        tissue_p,
        cache_dir=cache_dir,
        cohort_tag=COHORT_TAG,
        mask_nii=mask_path,
        tissue_classes=tissue_classes,
    )
    print(f"  saved {tissue_p}")

    ym_p = figures_dir / f"gradients_by-yeo-mesulam_cohort-{COHORT_TAG}.png"
    plot_gradients_by_yeo_mesulam(
        rows, ym_p, cache_dir=cache_dir,
        tractometry_root=root, cohort_tag=COHORT_TAG, atlas=atlas,
    )
    print(f"  saved {ym_p}")

    g1_b = figures_dir / f"gradient1_by-groups-axes_cohort-{COHORT_TAG}.png"
    plot_groups_axes_bars(
        rows, g1_b, gradient_index=0, cache_dir=cache_dir,
        tractometry_root=root, cohort_tag=COHORT_TAG, mask_nii=mask_path,
    )
    print(f"  saved {g1_b}")

    g2_b = figures_dir / f"gradient2_by-groups-axes_cohort-{COHORT_TAG}.png"
    plot_groups_axes_bars(
        rows, g2_b, gradient_index=1, cache_dir=cache_dir,
        tractometry_root=root, cohort_tag=COHORT_TAG, mask_nii=mask_path,
    )
    print(f"  saved {g2_b}")

    sum_g1 = figures_dir / f"gradient1_summary_cohort-{COHORT_TAG}.png"
    plot_gradient1_summary(
        rows,
        sum_g1,
        cache_dir=cache_dir,
        tractometry_root=root,
        cohort_tag=COHORT_TAG,
        mask_nii=mask_path,
        tissue_classes=tissue_classes,
    )
    print(f"  saved {sum_g1}")

    sum_g2 = figures_dir / f"gradient2_summary_cohort-{COHORT_TAG}.png"
    plot_gradient_summary(
        rows,
        sum_g2,
        gradient_index=1,
        cache_dir=cache_dir,
        tractometry_root=root,
        cohort_tag=COHORT_TAG,
        mask_nii=mask_path,
        tissue_classes=tissue_classes,
    )
    print(f"  saved {sum_g2}")

    lg = figures_dir / "legend-gradient1.png"
    save_standalone_legend_gradient1(rows, lg, tissue_classes=tissue_classes)
    print(f"  saved {lg}")

    lt = figures_dir / "legend-tissue.png"
    save_standalone_legend_tissue(lt, tissue_classes=tissue_classes)
    print(f"  saved {lt}")

    ly = figures_dir / "legend-yeo.png"
    save_standalone_legend_yeo_voxelwise(atlas, ly, tractometry_root=root)
    print(f"  saved {ly}")

    lm = figures_dir / "legend-mesulam.png"
    save_standalone_legend_mesulam_voxelwise(atlas, lm, tractometry_root=root)
    print(f"  saved {lm}")

    emit_neuromaps_figures_if_available(
        output_dir, figures_dir, factors=[r[0] for r in rows], rows=rows,
        cache_dir=cache_dir, tractometry_root=root,
        mask_nii=mask_path, csf_mode=csf_mode,
    )


def emit_neuromaps_figures_if_available(
    output_dir: Path,
    figures_dir: Path,
    *,
    factors: list[str],
    gradient_indices: list[int] | None = None,
    rows: list[VoxelGradientRunRow] | None = None,
    cache_dir: Path | None = None,
    tractometry_root: Path | None = None,
    mask_nii: Path | None = None,
    csf_mode: str | None = None,
) -> None:
    """Plot neuromaps lollipop and combined summary figures when CSVs exist."""
    from .neuromaps_correlations import neuromaps_correlation_csv_path
    from .plots_neuromaps_lollipop import plot_neuromaps_figures_from_output
    from .plots_summary_with_neuromaps import plot_summary_with_neuromaps_figures

    gradients = gradient_indices or [1, 2]
    if not all(
        neuromaps_correlation_csv_path(output_dir, f, gi, cohort_tag=COHORT_TAG).is_file()
        for f in factors
        for gi in gradients
    ):
        return
    paths = plot_neuromaps_figures_from_output(
        output_dir,
        factors=factors,
        gradient_indices=gradients,
        cohort_tag=COHORT_TAG,
        figures_dir=figures_dir,
    )
    for p in paths:
        print(f"  saved {p}")

    if rows is not None:
        summary_paths = plot_summary_with_neuromaps_figures(
            rows,
            output_dir,
            gradient_indices=gradients,
            cohort_tag=COHORT_TAG,
            figures_dir=figures_dir,
            cache_dir=cache_dir,
            tractometry_root=tractometry_root,
            mask_nii=mask_nii,
            csf_mode=csf_mode,
        )
        for p in summary_paths:
            print(f"  saved {p}")
