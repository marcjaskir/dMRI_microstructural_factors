"""GM/WM/CSF scatter, Yeo/Mesulam, and summary figures for voxelwise gradients."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from .config import (
    GRADIENT_AXIS_LABEL_COLOR,
    SCATTER_MAX_POINTS,
    TISSUE_CLASSES,
    TISSUE_COLORS,
    TISSUE_MARKERS,
    TISSUE_POINT_EDGEWIDTH,
    TISSUE_SCATTER_POINT_ALPHA,
    TISSUE_SCATTER_POINT_SIZE,
)
from .embedding import gradient_from_row
from .parcel_gradients import voxel_rows_to_parcel_gradient_run_rows
from .plots_bars_voxelwise import (
    _BAR_FIGURE_FONT_RCPARAMS,
    _reflow_bar_pair_axes,
    paint_groups_axes_bars_row,
)
from .tissue_gmwmcsf import (
    load_tissue_masks_inclusive,
    subsample_voxels_stratified,
    tissue_centroids_g1_g2,
)
from .types import VoxelGradientRunRow

_gc_scatter = None


def _gc_plots_scatter():
    global _gc_scatter
    if _gc_scatter is None:
        from .gc_imports import gc_plots_scatter
        _gc_scatter = gc_plots_scatter()
    return _gc_scatter

_FACTOR_WSPACE = 0.22
_AXIS_LABEL_FS = 22.0
_AXIS_TICK_FS = 16.0


def _factor_display_name(factor_tag: str) -> str:
    m = re.match(r"^F(\d+)$", str(factor_tag).strip(), re.I)
    if m:
        return f"Factor {int(m.group(1))}"
    return str(factor_tag)


def _g1_g2_arrays(row: VoxelGradientRunRow) -> tuple[np.ndarray, np.ndarray]:
    flat = row[5].astype(str)
    g1 = gradient_from_row(row, 0).reindex(flat).to_numpy(dtype=np.float64)
    g2 = gradient_from_row(row, 1).reindex(flat).to_numpy(dtype=np.float64) if len(row[2]) > 1 else np.zeros_like(g1)
    return g1, g2


def _plot_tissue_panel(
    ax: plt.Axes,
    g1: np.ndarray,
    g2: np.ndarray,
    tissue_masks: dict[str, np.ndarray],
    *,
    title: str,
    ylim: tuple[float, float] | None = None,
    tissue_classes: tuple[str, ...] = TISSUE_CLASSES,
) -> None:
    g1p, g2p, labels = subsample_voxels_stratified(
        g1, g2, tissue_masks, max_points=SCATTER_MAX_POINTS, tissue_classes=tissue_classes
    )
    for tissue in tissue_classes:
        m = labels == tissue
        if not np.any(m):
            continue
        ax.scatter(
            g2p[m], g1p[m],
            c=TISSUE_COLORS[tissue],
            marker=TISSUE_MARKERS[tissue],
            s=TISSUE_SCATTER_POINT_SIZE,
            alpha=TISSUE_SCATTER_POINT_ALPHA,
            edgecolors="k",
            linewidths=TISSUE_POINT_EDGEWIDTH * 0.5,
            label=tissue,
        )

    cents = tissue_centroids_g1_g2(g1, g2, tissue_masks, tissue_classes=tissue_classes)
    for tissue in tissue_classes:
        cx, cy = cents[tissue]  # (mean G1, mean G2)
        if not np.isfinite(cx) or not np.isfinite(cy):
            continue
        ax.scatter(
            [cy], [cx],
            s=180,
            c=[TISSUE_COLORS[tissue]],
            marker=TISSUE_MARKERS[tissue],
            edgecolors="k",
            linewidths=2.0,
            zorder=5,
        )

    ax.set_xlabel("Gradient 2", fontsize=_AXIS_LABEL_FS, color=GRADIENT_AXIS_LABEL_COLOR)
    ax.set_ylabel("Gradient 1", fontsize=_AXIS_LABEL_FS, color=GRADIENT_AXIS_LABEL_COLOR)
    ax.set_title(title, fontsize=_AXIS_LABEL_FS, color=GRADIENT_AXIS_LABEL_COLOR)
    ax.tick_params(axis="both", labelsize=_AXIS_TICK_FS, colors=GRADIENT_AXIS_LABEL_COLOR)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_gradients_by_tissue(
    results: list[VoxelGradientRunRow],
    out_path: Path,
    *,
    cache_dir: Path | None = None,
    cohort_tag: str | None = "controls",
    mask_nii: Path | None = None,
    tissue_classes: tuple[str, ...] = TISSUE_CLASSES,
) -> Path:
    _ = cohort_tag
    tissue_masks = load_tissue_masks_inclusive(cache_dir=cache_dir, mask_nii=mask_nii)
    n = len(results)
    fig_w = max(5.5 * n, 8.0)
    with plt.rc_context(_BAR_FIGURE_FONT_RCPARAMS):
        fig, axes = plt.subplots(1, n, figsize=(fig_w, 5.5), squeeze=False)
        for f, row in enumerate(results):
            g1, g2 = _g1_g2_arrays(row)
            _plot_tissue_panel(
                axes[0, f], g1, g2, tissue_masks,
                title=_factor_display_name(row[0]),
                tissue_classes=tissue_classes,
            )
        fig.tight_layout(pad=0.35)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


def plot_gradients_by_yeo_mesulam(
    results: list[VoxelGradientRunRow],
    out_path: Path,
    *,
    cache_dir: Path | None = None,
    tractometry_root: Path | None = None,
    cohort_tag: str | None = "controls",
    atlas=None,
) -> Path:
    """Voxelwise Yeo/Mesulam scatter: cortical voxels colored by MMP parcel community."""
    _ = cohort_tag
    from .yeo_mesulam_voxelwise import plot_gradients_by_yeo_mesulam_voxelwise

    return plot_gradients_by_yeo_mesulam_voxelwise(
        results,
        out_path,
        cache_dir=cache_dir,
        tractometry_root=tractometry_root,
        atlas=atlas,
    )


def plot_gradient_summary(
    results: list[VoxelGradientRunRow],
    out_path: Path,
    *,
    gradient_index: int,
    cache_dir: Path | None = None,
    tractometry_root: Path | None = None,
    cohort_tag: str | None = "controls",
    mask_nii: Path | None = None,
    tissue_classes: tuple[str, ...] = TISSUE_CLASSES,
) -> Path:
    """Row 1: GM/WM/CSF tissue scatter; row 2: region groups + neuroaxis."""
    _ = cohort_tag
    tissue_masks = load_tissue_masks_inclusive(cache_dir=cache_dir, mask_nii=mask_nii)
    parcel_results = voxel_rows_to_parcel_gradient_run_rows(
        results, cache_dir=cache_dir, mask_nii=mask_nii
    )
    n = len(results)
    row_h = 5.75
    fig_w = 6.2 * n + 0.8
    fig_h = row_h * 2.1 + 0.9

    with plt.rc_context(_BAR_FIGURE_FONT_RCPARAMS):
        fig = plt.figure(figsize=(fig_w, fig_h))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.32)
        gs_top = GridSpecFromSubplotSpec(1, n, subplot_spec=outer[0, 0], wspace=_FACTOR_WSPACE)
        gs_bot = GridSpecFromSubplotSpec(1, n, subplot_spec=outer[1, 0], wspace=_FACTOR_WSPACE)

        for f, row in enumerate(results):
            ax = fig.add_subplot(gs_top[0, f])
            g1, g2 = _g1_g2_arrays(row)
            _plot_tissue_panel(
                ax, g1, g2, tissue_masks,
                title=_factor_display_name(row[0]),
                tissue_classes=tissue_classes,
            )

        ax_pairs = paint_groups_axes_bars_row(
            fig, gs_bot, results, parcel_results,
            gradient_index=gradient_index,
            tractometry_root=tractometry_root,
            cache_dir=cache_dir,
        )
        fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08, hspace=0.35)
        for ax_rg, ax_corr in ax_pairs:
            _reflow_bar_pair_axes(ax_rg, ax_corr)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


def plot_gradient1_summary(
    results: list[VoxelGradientRunRow],
    out_path: Path,
    *,
    cache_dir: Path | None = None,
    tractometry_root: Path | None = None,
    cohort_tag: str | None = "controls",
    mask_nii: Path | None = None,
    tissue_classes: tuple[str, ...] = TISSUE_CLASSES,
) -> Path:
    return plot_gradient_summary(
        results, out_path, gradient_index=0,
        cache_dir=cache_dir, tractometry_root=tractometry_root, cohort_tag=cohort_tag,
        mask_nii=mask_nii, tissue_classes=tissue_classes,
    )


def save_standalone_legend_tissue(
    out_path: Path,
    *,
    tissue_classes: tuple[str, ...] = TISSUE_CLASSES,
) -> Path:
    handles = [
        Line2D(
            [0], [0], marker=TISSUE_MARKERS[t], color="w",
            markerfacecolor=TISSUE_COLORS[t], markeredgecolor="k",
            markersize=10, label=t,
        )
        for t in tissue_classes
    ]
    handles.append(
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="#808080", markeredgecolor="k",
            markersize=12, markeredgewidth=2, label="Tissue centroid",
        ),
    )
    fig, ax = plt.subplots(figsize=(7.8, 1.2))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=max(len(tissue_classes), 1), frameon=False, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_standalone_legend_gradient1(
    results: list[VoxelGradientRunRow],
    out_path: Path,
    *,
    tissue_classes: tuple[str, ...] = TISSUE_CLASSES,
) -> Path:
    """Minimal G1 colorbar placeholder (voxelwise tissue plots are not G1-colored)."""
    _ = results
    label = " / ".join(tissue_classes) + " encoding"
    fig, ax = plt.subplots(figsize=(4.0, 0.6))
    ax.axis("off")
    ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
