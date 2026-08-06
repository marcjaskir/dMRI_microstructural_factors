"""Voxelwise Yeo / Mesulam scatter helpers (Glasser/MMP parcel → community)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec

from .atlas_voxelwise import VoxelwiseAtlasContext, glasser_parcel_label_per_inmask_voxel
from .config import (
    DEFAULT_TRACTOMETRY_ROOT,
    GRADIENT_AXIS_LABEL_COLOR,
    SCATTER_MAX_POINTS,
    TISSUE_POINT_EDGEWIDTH,
    TISSUE_SCATTER_POINT_ALPHA,
    TISSUE_SCATTER_POINT_SIZE,
)
from .embedding import gradient_from_row
from .gc_imports import gc_groupings
from .parcel_gradients import get_atlas_context
from .types import VoxelGradientRunRow

_WSPACE_INNER = 0.07
_WSPACE_OUTER = 0.32
_COL_W = 6.2
_ROW_H = 5.75
_HEADER_RATIO = 0.09
_AXIS_LABEL_FS = 22.0
_AXIS_TICK_FS = 16.0
_HEADER_FS = 20.0
_CENTROID_SIZE = 104.0
_CENTROID_EDGE_W = 2.45


def _lookup_community(parcel_name: str, label_map: Mapping[str, str]) -> str:
    if not parcel_name:
        return ""
    lab = label_map.get(str(parcel_name), "")
    if not lab:
        return ""
    s = str(lab).strip()
    if s.lower() in ("nan", "n/a", "none"):
        return ""
    return s


def community_labels_for_parcels(
    parcel_names: np.ndarray,
    label_map: Mapping[str, str],
) -> np.ndarray:
    """Map each in-mask voxel's Glasser parcel to a community label (or '')."""
    out = np.empty(len(parcel_names), dtype=object)
    for i, pname in enumerate(parcel_names):
        out[i] = _lookup_community(str(pname), label_map)
    return out


def communities_seen_on_glasser_voxels(
    parcel_names: np.ndarray,
    label_map: Mapping[str, str],
    color_fn: Callable[[str], str],
) -> dict[str, str]:
    """Community label → color for all communities present on cortical voxels."""
    seen: dict[str, str] = {}
    for pname in parcel_names:
        lab = _lookup_community(str(pname), label_map)
        if lab and lab not in seen:
            seen[lab] = color_fn(lab)
    return {lab: seen[lab] for lab in sorted(seen)}


def subsample_voxels_by_label(
    g1: np.ndarray,
    g2: np.ndarray,
    labels: np.ndarray,
    *,
    max_points: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample (G1, G2) stratified by non-empty community label."""
    gen = rng if rng is not None else np.random.default_rng(0)
    valid_idx = np.flatnonzero(np.array([bool(l) for l in labels], dtype=bool))
    if valid_idx.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, np.array([], dtype=object)

    unique = sorted({str(labels[i]) for i in valid_idx})
    per_group = max(1, max_points // max(len(unique), 1))
    picked: list[int] = []
    picked_labels: list[str] = []
    for lab in unique:
        idx = np.flatnonzero(labels == lab)
        if idx.size == 0:
            continue
        n = min(per_group, idx.size)
        choice = gen.choice(idx, size=n, replace=False)
        picked.extend(int(i) for i in choice)
        picked_labels.extend([lab] * n)

    pick = np.asarray(picked, dtype=np.int64)
    lab_arr = np.asarray(picked_labels, dtype=object)
    if pick.size > max_points:
        sel = gen.choice(pick.size, size=max_points, replace=False)
        pick = pick[sel]
        lab_arr = lab_arr[sel]
    return g1[pick], g2[pick], lab_arr


def _decorate_axes(
    ax: plt.Axes,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    if show_xlabel:
        ax.set_xlabel("Gradient 2", fontsize=_AXIS_LABEL_FS, color=GRADIENT_AXIS_LABEL_COLOR)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Gradient 1", fontsize=_AXIS_LABEL_FS, color=GRADIENT_AXIS_LABEL_COLOR)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=_AXIS_TICK_FS, colors=GRADIENT_AXIS_LABEL_COLOR)
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_community_voxel_scatter(
    ax: plt.Axes,
    g1: np.ndarray,
    g2: np.ndarray,
    community_labels: np.ndarray,
    color_fn: Callable[[str], str],
    *,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    g1p, g2p, labs = subsample_voxels_by_label(
        g1, g2, community_labels, max_points=SCATTER_MAX_POINTS
    )
    if g1p.size:
        colors = [color_fn(str(lab)) for lab in labs]
        ax.scatter(
            g2p,
            g1p,
            c=colors,
            marker="o",
            s=TISSUE_SCATTER_POINT_SIZE,
            alpha=TISSUE_SCATTER_POINT_ALPHA,
            edgecolors="k",
            linewidths=TISSUE_POINT_EDGEWIDTH * 0.5,
            zorder=1,
        )
    _decorate_axes(ax, show_xlabel=show_xlabel, show_ylabel=show_ylabel)


def _plot_community_voxel_centroids(
    ax: plt.Axes,
    g1: np.ndarray,
    g2: np.ndarray,
    community_labels: np.ndarray,
    color_fn: Callable[[str], str],
    *,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    for lab in sorted({str(l) for l in community_labels if l}):
        m = community_labels == lab
        if not np.any(m):
            continue
        cx = float(np.nanmean(g2[m]))
        cy = float(np.nanmean(g1[m]))
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        ax.scatter(
            [cx],
            [cy],
            s=_CENTROID_SIZE,
            c=[color_fn(lab)],
            marker="o",
            edgecolors="black",
            linewidths=_CENTROID_EDGE_W,
            alpha=1.0,
            zorder=5,
        )
    _decorate_axes(ax, show_xlabel=show_xlabel, show_ylabel=show_ylabel)


def _factor_display_name(factor_tag: str) -> str:
    import re

    m = re.match(r"^F(\d+)$", str(factor_tag).strip(), re.I)
    if m:
        return f"Factor {int(m.group(1))}"
    return str(factor_tag)


def plot_gradients_by_yeo_mesulam_voxelwise(
    results: list[VoxelGradientRunRow],
    out_path: Path,
    *,
    cache_dir: Path | None = None,
    tractometry_root: Path | None = None,
    atlas: VoxelwiseAtlasContext | None = None,
) -> Path:
    """Yeo / Mesulam scatter: cortical voxels colored by MMP parcel community."""
    n = len(results)
    if n == 0:
        return out_path

    root = tractometry_root or DEFAULT_TRACTOMETRY_ROOT
    grp = gc_groupings()
    yeo_by = grp.load_yeo_labels(root)
    mesu_by = grp.load_mesulam_labels(root)
    yeo_color = grp.yeo_network_color
    mesu_color = grp.mesulam_type_color

    if atlas is None:
        atlas = get_atlas_context(cache_dir=cache_dir)
    n_inmask = int(atlas.analysis_mask.sum())
    parcel_names = glasser_parcel_label_per_inmask_voxel(atlas.glasser, n_inmask)
    yeo_labels = community_labels_for_parcels(parcel_names, yeo_by)
    mesu_labels = community_labels_for_parcels(parcel_names, mesu_by)

    fig_w = _COL_W * (2 * n) + 0.8
    fig_h = _ROW_H * (_HEADER_RATIO + 2.0) + 0.45
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        3,
        n,
        width_ratios=[1.0] * n,
        height_ratios=[_HEADER_RATIO, 1.0, 1.0],
        hspace=0.14,
        wspace=_WSPACE_OUTER,
    )

    for f, row in enumerate(results):
        ax_h = fig.add_subplot(gs[0, f])
        ax_h.set_axis_off()
        ax_h.text(
            0.5,
            0.98,
            _factor_display_name(row[0]),
            ha="center",
            va="top",
            transform=ax_h.transAxes,
            fontsize=_HEADER_FS,
            color=GRADIENT_AXIS_LABEL_COLOR,
        )

        flat = row[5].astype(str)
        g1 = gradient_from_row(row, 0).reindex(flat).to_numpy(dtype=np.float64)
        g2 = (
            gradient_from_row(row, 1).reindex(flat).to_numpy(dtype=np.float64)
            if len(row[2]) > 1
            else np.zeros_like(g1)
        )

        inner_y = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[1, f], wspace=_WSPACE_INNER
        )
        inner_m = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[2, f], wspace=_WSPACE_INNER
        )
        ax_y = fig.add_subplot(inner_y[0, 0])
        ax_yc = fig.add_subplot(inner_y[0, 1], sharex=ax_y, sharey=ax_y)
        ax_m = fig.add_subplot(inner_m[0, 0])
        ax_mc = fig.add_subplot(inner_m[0, 1], sharex=ax_m, sharey=ax_m)
        ax_yc.tick_params(labelleft=False)
        ax_mc.tick_params(labelleft=False)

        _plot_community_voxel_scatter(
            ax_y, g1, g2, yeo_labels, yeo_color,
            show_xlabel=False, show_ylabel=False,
        )
        _plot_community_voxel_centroids(
            ax_yc, g1, g2, yeo_labels, yeo_color,
            show_xlabel=False, show_ylabel=False,
        )
        _plot_community_voxel_scatter(
            ax_m, g1, g2, mesu_labels, mesu_color,
            show_xlabel=True, show_ylabel=False,
        )
        _plot_community_voxel_centroids(
            ax_mc, g1, g2, mesu_labels, mesu_color,
            show_xlabel=True, show_ylabel=False,
        )
        ax_y.set_title("Markers", fontsize=_AXIS_LABEL_FS, color=GRADIENT_AXIS_LABEL_COLOR)
        ax_yc.set_title("Centroids", fontsize=_AXIS_LABEL_FS, color=GRADIENT_AXIS_LABEL_COLOR)

    fig.tight_layout(pad=0.35, h_pad=0.45, w_pad=0.35, rect=[0.0, 0.04, 1.0, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_standalone_legend_yeo_voxelwise(
    atlas: VoxelwiseAtlasContext,
    out_path: Path,
    *,
    tractometry_root: Path | None = None,
) -> Path:
    from .plots_scatter_voxelwise import _gc_plots_scatter

    root = tractometry_root or DEFAULT_TRACTOMETRY_ROOT
    grp = gc_groupings()
    gc_ps = _gc_plots_scatter()
    n_inmask = int(atlas.analysis_mask.sum())
    parcel_names = glasser_parcel_label_per_inmask_voxel(atlas.glasser, n_inmask)
    yeo_by = grp.load_yeo_labels(root)
    seen = communities_seen_on_glasser_voxels(parcel_names, yeo_by, grp.yeo_network_color)
    fig = plt.figure(figsize=(4.8, 4.5), facecolor="white")
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.84])
    gc_ps._add_cortex_community_legend_top_row(
        ax, seen, bbox_y=0.5, bbox_x=0.5, loc="center", ncol=2
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def save_standalone_legend_mesulam_voxelwise(
    atlas: VoxelwiseAtlasContext,
    out_path: Path,
    *,
    tractometry_root: Path | None = None,
) -> Path:
    from .plots_scatter_voxelwise import _gc_plots_scatter

    root = tractometry_root or DEFAULT_TRACTOMETRY_ROOT
    grp = gc_groupings()
    gc_ps = _gc_plots_scatter()
    n_inmask = int(atlas.analysis_mask.sum())
    parcel_names = glasser_parcel_label_per_inmask_voxel(atlas.glasser, n_inmask)
    mesu_by = grp.load_mesulam_labels(root)
    seen = communities_seen_on_glasser_voxels(parcel_names, mesu_by, grp.mesulam_type_color)
    fig = plt.figure(figsize=(4.2, 3.4), facecolor="white")
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])
    gc_ps._add_cortex_community_legend_top_row(
        ax, seen, bbox_y=0.5, bbox_x=0.5, loc="center", ncol=2
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
