"""Combined summary figures: scatter, region groups, neuromaps, neuroaxis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec

from .config import (
    COHORT_TAG,
    DEFAULT_TRACTOMETRY_ROOT,
    NEUROMAPS_ANNOTATION_INFO_CSV,
    NEUROMAPS_ROW_FSLR,
    NEUROMAPS_ROW_MNI,
    NEUROMAPS_SPACE_FSLR,
    NEUROMAPS_SPACE_MNI,
    NEUROMAPS_TOP_K,
)
from .embedding import gradient_from_row
from .neuromaps_correlations import (
    dedupe_correlations_by_description_brief,
    load_annotation_description_lookup,
    load_annotation_metadata,
    neuromaps_correlation_csv_path,
)
from .neuroaxis_voxelwise import (
    NEUROAXIS_AXES,
    compute_neuroaxis_ranks,
    gradient_values_in_mask_order,
    pearson_r_gradient_vs_coordinate_ranks,
)
from .parcel_gradients import voxel_rows_to_parcel_gradient_run_rows
from .plots_bars_voxelwise import (
    _BAR_FIGURE_FONT_RCPARAMS,
    _neuroaxis_corr_ylim,
    _plot_neuroaxis_lollipop_ax,
    _plot_region_group_bars_ax,
    _prepare_region_group_bars,
    load_cortical_lobe_region_group_by_roi,
    load_tract_label_to_type_group,
)
from .plots_neuromaps_lollipop import _plot_lollipop_panel
from .plots_scatter_voxelwise import _factor_display_name, _g1_g2_arrays, _plot_tissue_panel
from .region_groups_voxelwise import build_parcel_to_region_group
from .tissue_gmwmcsf import load_tissue_masks_inclusive
from .types import VoxelGradientRunRow

_FACTOR_WSPACE = 0.28
_YLIM_PAD_FRAC = 0.08
_SUMMARY_WITH_NEUROMAPS_FIG_W = 14.0
_SUMMARY_WITH_NEUROMAPS_FIG_H = 11.5
# Column 1 (region groups / neuroaxis) needs more width than scatter / neuromaps.
_SUMMARY_WITH_NEUROMAPS_WIDTH_RATIOS: tuple[float, float] = (1.0, 2.1)
_SUMMARY_WITH_NEUROMAPS_WSPACE = 0.38

NEUROMAPS_SPACE_FILENAME_SLUG: dict[str, str] = {
    NEUROMAPS_SPACE_MNI: "MNI",
    NEUROMAPS_SPACE_FSLR: "fsLR",
}
NEUROMAPS_SPACE_ROW_TITLE: dict[str, str] = {
    NEUROMAPS_SPACE_MNI: NEUROMAPS_ROW_MNI,
    NEUROMAPS_SPACE_FSLR: NEUROMAPS_ROW_FSLR,
}
SUMMARY_WITH_NEUROMAPS_SPACES: tuple[str, ...] = (NEUROMAPS_SPACE_MNI, NEUROMAPS_SPACE_FSLR)


def _pad_ylim(lo: float, hi: float, *, pad_frac: float = _YLIM_PAD_FRAC) -> tuple[float, float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (-1.0, 1.0)
    if lo == hi:
        margin = max(abs(lo) * 0.1, 0.05)
        return lo - margin, hi + margin
    span = hi - lo
    return lo - pad_frac * span, hi + pad_frac * span


def _top_row_ylim(
    g1: np.ndarray,
    g_series: pd.Series,
    parcel_to_group: dict[str, str],
) -> tuple[float, float]:
    y_vals: list[np.ndarray] = []
    g1f = g1[np.isfinite(g1)]
    if g1f.size:
        y_vals.append(g1f)
    _, means, sems = _prepare_region_group_bars(g_series, parcel_to_group)
    if len(means):
        y_vals.append(means - sems)
        y_vals.append(means + sems)
    if not y_vals:
        return (-1.0, 1.0)
    merged = np.concatenate(y_vals)
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return (-1.0, 1.0)
    return _pad_ylim(float(np.min(merged)), float(np.max(merged)))


def _neuromaps_top_r_values(
    df: pd.DataFrame,
    *,
    gradient_num: int,
    space: str,
    metadata,
) -> np.ndarray:
    sub = df[
        (df["gradient"].astype(int) == int(gradient_num))
        & (df["space"].astype(str) == space)
    ]
    if sub.empty:
        return np.asarray([], dtype=np.float64)
    sub = dedupe_correlations_by_description_brief(sub, metadata)
    sub = sub.sort_values("abs_r", ascending=False).head(NEUROMAPS_TOP_K)
    return sub["pearson_r"].to_numpy(dtype=np.float64)


def _bottom_row_ylim(
    neuromaps_df: pd.DataFrame,
    *,
    gradient_num: int,
    gradient_index: int,
    neuromaps_space: str,
    g_values: np.ndarray,
    ranks: dict[str, np.ndarray],
    metadata,
) -> tuple[float, float]:
    vals: list[np.ndarray] = []
    r = _neuromaps_top_r_values(
        neuromaps_df, gradient_num=gradient_num, space=neuromaps_space, metadata=metadata
    )
    if r.size:
        vals.append(r)

    tab = pearson_r_gradient_vs_coordinate_ranks(g_values, ranks, axes=NEUROAXIS_AXES)
    neuro = tab["pearson_r"].to_numpy(dtype=np.float64)
    neuro = neuro[np.isfinite(neuro)]
    if neuro.size:
        vals.append(neuro)

    nax_lo, nax_hi = _neuroaxis_corr_ylim(gradient_index)
    abs_candidates = [0.05, abs(nax_lo), abs(nax_hi)]
    if vals:
        merged = np.concatenate(vals)
        merged = merged[np.isfinite(merged)]
        if merged.size:
            abs_candidates.append(float(np.nanmax(np.abs(merged))))
    ymax = max(abs_candidates) * 1.15
    return (-ymax, ymax)


def _summary_with_neuromaps_out_path(
    figures_dir: Path,
    factor: str,
    gradient_num: int,
    space_slug: str,
    cohort_tag: str,
) -> Path:
    return (
        figures_dir
        / f"{factor}_G{gradient_num}_summary_with_neuromaps-{space_slug}_cohort-{cohort_tag}.png"
    )


def _paint_factor_summary_with_neuromaps(
    fig: plt.Figure,
    gs,
    *,
    vrow: VoxelGradientRunRow,
    prow,
    nm_df: pd.DataFrame,
    gradient_index: int,
    neuromaps_space: str,
    tissue_masks: dict[str, np.ndarray],
    tract_to_type: dict[str, str],
    cortical_by_roi: dict,
    description_lookup: dict[tuple[str, str], str],
    metadata,
    show_left_ylabels: bool = True,
    tissue_classes: tuple[str, ...] | None = None,
) -> None:
    """Paint one factor's 2×2 summary panels into ``gs``."""
    from .config import TISSUE_CLASSES

    plot_tissues = tissue_classes or TISSUE_CLASSES
    g_num = gradient_index + 1
    row_title = NEUROMAPS_SPACE_ROW_TITLE[neuromaps_space]
    factor = vrow[0]

    g_series = gradient_from_row(prow, gradient_index)
    parcel_to_group = build_parcel_to_region_group(
        g_series,
        tract_to_type=tract_to_type,
        cortical_by_roi=cortical_by_roi,
    )
    g1, g2 = _g1_g2_arrays(vrow)
    ranks = compute_neuroaxis_ranks(vrow[4])
    g_vals = gradient_values_in_mask_order(vrow, gradient_index)

    top_ylim = _top_row_ylim(g1, g_series, parcel_to_group)
    bottom_ylim = _bottom_row_ylim(
        nm_df,
        gradient_num=g_num,
        gradient_index=gradient_index,
        neuromaps_space=neuromaps_space,
        g_values=g_vals,
        ranks=ranks,
        metadata=metadata,
    )

    inner = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=gs,
        height_ratios=[1.0, 1.0],
        width_ratios=list(_SUMMARY_WITH_NEUROMAPS_WIDTH_RATIOS),
        hspace=0.38,
        wspace=_SUMMARY_WITH_NEUROMAPS_WSPACE,
    )
    ax_scatter = fig.add_subplot(inner[0, 0])
    ax_rg = fig.add_subplot(inner[0, 1], sharey=ax_scatter)
    ax_nm = fig.add_subplot(inner[1, 0])
    ax_nax = fig.add_subplot(inner[1, 1], sharey=ax_nm)

    _plot_tissue_panel(
        ax_scatter,
        g1,
        g2,
        tissue_masks,
        title=_factor_display_name(factor),
        ylim=top_ylim,
        tissue_classes=plot_tissues,
    )
    _plot_region_group_bars_ax(
        ax_rg,
        g_series,
        parcel_to_group,
        show_ylabel=False,
        ylabel=f"Gradient {g_num}",
        title="Region groups",
        ylim=top_ylim,
    )

    sub = nm_df[
        (nm_df["gradient"].astype(int) == int(g_num))
        & (nm_df["space"].astype(str) == neuromaps_space)
    ]
    _plot_lollipop_panel(
        ax_nm,
        sub,
        title=row_title,
        show_ylabel=show_left_ylabels,
        ylabel="Pearson r" if show_left_ylabels else "",
        description_lookup=description_lookup,
        metadata=metadata,
        ylim=bottom_ylim,
    )

    _plot_neuroaxis_lollipop_ax(
        ax_nax,
        g_vals,
        ranks,
        neuroaxis_ylim=bottom_ylim,
        show_ylabel=False,
        ylabel="Pearson r",
        title=f"Gradient {g_num} axis",
    )

    plt.setp(ax_rg.get_yticklabels(), visible=False)
    plt.setp(ax_nax.get_yticklabels(), visible=False)


def plot_factor_gradient_summary_with_neuromaps(
    vrow: VoxelGradientRunRow,
    neuromaps_csv: Path,
    out_path: Path,
    *,
    gradient_index: int,
    neuromaps_space: str,
    cache_dir: Path | None = None,
    tractometry_root: Path | None = None,
    cohort_tag: str | None = "controls",
    metadata_csv: Path | None = None,
    mask_nii: Path | None = None,
    tissue_classes: tuple[str, ...] | None = None,
    csf_mode: str | None = None,
) -> Path:
    """One factor: scatter | region groups (top), neuromaps | neuroaxis (bottom)."""
    from .config import tissue_classes_for_csf_mode

    if neuromaps_space not in NEUROMAPS_SPACE_FILENAME_SLUG:
        raise ValueError(
            f"Unknown neuromaps_space {neuromaps_space!r}; "
            f"expected one of {list(NEUROMAPS_SPACE_FILENAME_SLUG)}"
        )
    _ = cohort_tag
    if not neuromaps_csv.is_file():
        raise FileNotFoundError(f"Missing neuromaps CSV: {neuromaps_csv}")

    tissue_masks = load_tissue_masks_inclusive(cache_dir=cache_dir, mask_nii=mask_nii)
    parcel_results = voxel_rows_to_parcel_gradient_run_rows(
        [vrow], cache_dir=cache_dir, mask_nii=mask_nii
    )
    plot_tissues = tissue_classes or tissue_classes_for_csf_mode(csf_mode)
    prow = parcel_results[0]
    root = tractometry_root or DEFAULT_TRACTOMETRY_ROOT
    tract_to_type = load_tract_label_to_type_group()
    cortical_by_roi = load_cortical_lobe_region_group_by_roi(root)

    description_lookup = load_annotation_description_lookup(
        metadata_csv or NEUROMAPS_ANNOTATION_INFO_CSV
    )
    metadata = load_annotation_metadata(metadata_csv or NEUROMAPS_ANNOTATION_INFO_CSV)
    nm_df = pd.read_csv(neuromaps_csv)

    fig_w = _SUMMARY_WITH_NEUROMAPS_FIG_W
    fig_h = _SUMMARY_WITH_NEUROMAPS_FIG_H

    with plt.rc_context(_BAR_FIGURE_FONT_RCPARAMS):
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(1, 1)
        _paint_factor_summary_with_neuromaps(
            fig,
            gs[0, 0],
            vrow=vrow,
            prow=prow,
            nm_df=nm_df,
            gradient_index=gradient_index,
            neuromaps_space=neuromaps_space,
            tissue_masks=tissue_masks,
            tract_to_type=tract_to_type,
            cortical_by_roi=cortical_by_roi,
            description_lookup=description_lookup,
            metadata=metadata,
            show_left_ylabels=True,
            tissue_classes=plot_tissues,
        )
        fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.10)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


def plot_gradient_summary_with_neuromaps(
    results: list[VoxelGradientRunRow],
    neuromaps_csvs: dict[str, Path],
    out_path: Path,
    *,
    gradient_index: int,
    neuromaps_space: str,
    cache_dir: Path | None = None,
    tractometry_root: Path | None = None,
    cohort_tag: str | None = "controls",
    metadata_csv: Path | None = None,
) -> Path:
    """Multi-factor layout (legacy); prefer ``plot_factor_gradient_summary_with_neuromaps``."""
    if neuromaps_space not in NEUROMAPS_SPACE_FILENAME_SLUG:
        raise ValueError(
            f"Unknown neuromaps_space {neuromaps_space!r}; "
            f"expected one of {list(NEUROMAPS_SPACE_FILENAME_SLUG)}"
        )
    _ = cohort_tag
    tissue_masks = load_tissue_masks_inclusive(cache_dir=cache_dir)
    parcel_results = voxel_rows_to_parcel_gradient_run_rows(results, cache_dir=cache_dir)
    root = tractometry_root or DEFAULT_TRACTOMETRY_ROOT
    tract_to_type = load_tract_label_to_type_group()
    cortical_by_roi = load_cortical_lobe_region_group_by_roi(root)

    description_lookup = load_annotation_description_lookup(
        metadata_csv or NEUROMAPS_ANNOTATION_INFO_CSV
    )
    metadata = load_annotation_metadata(metadata_csv or NEUROMAPS_ANNOTATION_INFO_CSV)

    neuromaps_tables: dict[str, pd.DataFrame] = {}
    for factor, path in neuromaps_csvs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing neuromaps CSV: {path}")
        neuromaps_tables[factor] = pd.read_csv(path)

    n = len(results)
    fig_w = _SUMMARY_WITH_NEUROMAPS_FIG_W * n + 0.8
    fig_h = _SUMMARY_WITH_NEUROMAPS_FIG_H

    with plt.rc_context(_BAR_FIGURE_FONT_RCPARAMS):
        fig = plt.figure(figsize=(fig_w, fig_h))
        outer = fig.add_gridspec(1, n, wspace=_FACTOR_WSPACE)

        for f, (vrow, prow) in enumerate(zip(results, parcel_results)):
            factor = vrow[0]
            _paint_factor_summary_with_neuromaps(
                fig,
                outer[0, f],
                vrow=vrow,
                prow=prow,
                nm_df=neuromaps_tables[factor],
                gradient_index=gradient_index,
                neuromaps_space=neuromaps_space,
                tissue_masks=tissue_masks,
                tract_to_type=tract_to_type,
                cortical_by_roi=cortical_by_roi,
                description_lookup=description_lookup,
                metadata=metadata,
                show_left_ylabels=(f == 0),
            )

        fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.07)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


def plot_summary_with_neuromaps_figures(
    results: list[VoxelGradientRunRow],
    output_dir: Path,
    *,
    gradient_indices: list[int] | None = None,
    cohort_tag: str = COHORT_TAG,
    figures_dir: Path | None = None,
    cache_dir: Path | None = None,
    tractometry_root: Path | None = None,
    mask_nii: Path | None = None,
    csf_mode: str | None = None,
) -> list[Path]:
    """Write per-factor gradient summary figures with neuromaps lollipop panels."""
    factors = [r[0] for r in results]
    gradients = gradient_indices or [1, 2]
    fig_dir = figures_dir or (output_dir / "figures")
    cache = cache_dir or (output_dir / "_cache")
    csv_map = {
        f: neuromaps_correlation_csv_path(output_dir, f, gi, cohort_tag=cohort_tag)
        for f in factors
        for gi in gradients
    }
    if not all(p.is_file() for p in csv_map.values()):
        return []

    saved: list[Path] = []
    for vrow in results:
        factor = vrow[0]
        for gi in gradients:
            csv_path = neuromaps_correlation_csv_path(
                output_dir, factor, gi, cohort_tag=cohort_tag
            )
            for space in SUMMARY_WITH_NEUROMAPS_SPACES:
                slug = NEUROMAPS_SPACE_FILENAME_SLUG[space]
                out = _summary_with_neuromaps_out_path(
                    fig_dir, factor, gi, slug, cohort_tag
                )
                plot_factor_gradient_summary_with_neuromaps(
                    vrow,
                    csv_path,
                    out,
                    gradient_index=gi - 1,
                    neuromaps_space=space,
                    cache_dir=cache,
                    tractometry_root=tractometry_root,
                    cohort_tag=cohort_tag,
                    mask_nii=mask_nii,
                    csf_mode=csf_mode,
                )
                saved.append(out)
    return saved
