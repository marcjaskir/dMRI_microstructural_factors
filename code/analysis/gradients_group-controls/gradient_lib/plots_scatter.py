"""Embedding-coordinate scatter figures for controls: 2D (G2 vs G1) and 3D (G2, G3, G1).

For 3D, axes are arranged as ``x = G2``, ``y = G3``, ``z = G1``.

``plot_gradients_by_gradient1_scatter`` lays out factors as **columns** in a single row
(``1 × N``). G1 turbo scaling matches ``save_standalone_legend_gradient1``; the main
figure has no embedded colorbar.

``plot_gradient_by_tissue_scatter`` and ``plot_gradient_by_yeo_mesulam_scatter`` use a
transposed layout: factors are **column pairs** (e.g. regions vs tissue-class centroids
for tissue encoding, or markers vs centroids for Yeo/Mesulam) left to right with
a small **header row** for factor names (no suptitle, no in-figure legends). Encoding
is documented in ``save_standalone_legend_tissue``, ``save_standalone_legend_yeo``, and
``save_standalone_legend_mesulam``. Tissue-only is one data row; Yeo/Mesulam is two rows
(Glasser cortex only). Within a factor, the two panels use a tight horizontal gap; a
wider gap separates factors. Marker and centroid panels share limits; across factors
limits are independent.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd

# mpl_toolkits.mplot3d registers the '3d' projection on import.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .config import (
    DEFAULT_TRACTOMETRY_ROOT,
    GRADIENT_AXIS_LABEL_COLOR,
    TISSUE_CENTROID_EDGE_COLOR,
    TISSUE_CENTROID_FILL_COLOR,
    TISSUE_CORTICAL_GM,
    TISSUE_POINT_EDGEWIDTH,
    TISSUE_SCATTER_POINT_ALPHA,
)
from .embedding import gradient_from_row
from .groupings import (
    load_mesulam_labels,
    load_yeo_labels,
    mesulam_type_color,
    yeo_network_color,
)
from .io import (
    glasser_parcel_name_set,
    load_hcp1065_tract_metadata,
    region_is_white_matter_column,
    region_is_wm_core_segment,
    region_is_wm_end_segment,
    subcortical_grey_matter_column_names,
    wm_end_loc_class_from_metadata,
)
from .figure_style import apply_figure_font_rcparams, figure_font_context
from .types import GradientRunRow

apply_figure_font_rcparams()

ColorBy = Literal["tissue", "yeo", "mesulam"]
Dims = Literal[2, 3]

# Nested column layout for ``plot_gradient_by_tissue_scatter`` /
# ``plot_gradient_by_yeo_mesulam_scatter``: tight spacing between the two panels of one
# factor; ``_GRADIENT_BY_FACTOR_WSPACE_OUTER`` is spacing between factors (matches legacy
# flat ``wspace`` between factor groups).
_GRADIENT_BY_FACTOR_WSPACE_INNER = 0.07
_GRADIENT_BY_FACTOR_WSPACE_OUTER = 0.32
# Tighter factor-column spacing for stacked tissue + bar summary figures only.
_GRADIENT_SUMMARY_FACTOR_WSPACE_OUTER = 0.22

# ``gradients_by-*`` figures: axis labels, tick labels, panel titles, factor header row.
_GRADIENT_BY_FIGURE_AXIS_LABEL_FS_2D = 22.0
_GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D = 16.0
_GRADIENT_BY_FIGURE_AXIS_LABEL_FS_3D = 16.0
_GRADIENT_BY_FIGURE_AXIS_TICK_FS_3D = 16.0
_GRADIENT_BY_FIGURE_COLUMN_HEADER_FS = 22.0
_GRADIENT_BY_FIGURE_SUPTITLE_FS = 16.0

# ``legend-tissue.png``: compact 1x5 horizontal strip (tighter than top-row legend strips).
_TISSUE_STANDALONE_LEGEND_FIG_WIDTH = 7.8
_TISSUE_STANDALONE_LEGEND_FIG_HEIGHT = 1.2
_TISSUE_STANDALONE_LEGEND_HANDLELENGTH = 1.0
_TISSUE_STANDALONE_LEGEND_HANDLETEXTPAD = 0.3
_TISSUE_STANDALONE_LEGEND_COLUMNSPACING = 0.75

# 3D panels are visually busier than 2D; use a slightly smaller marker area than ``s=26`` 2D scatters.
SCATTER_POINT_SIZE_3D = 16.0

# Shared scale for the paired-column top legend strips (tissue, Yeo, Mesulam).
FIGURE_TOP_LEGEND_STYLE_SCALE = 1.58
# Legend text size (pt) is ``FIGURE_TOP_LEGEND_FONT_PT * scale``; handles use ``6.2 * scale``.
FIGURE_TOP_LEGEND_FONT_PT = 11.75
# Legend center y within each top-row axes (transAxes); >0.5 shifts legends toward the figure top.
FIGURE_TOP_LEGEND_BBOX_Y = 0.82


def _factor_display_name(factor_tag: str) -> str:
    """``F1`` -> ``Factor 1`` for subplot titles; otherwise return ``factor_tag`` unchanged."""
    m = re.match(r"^F(\d+)$", str(factor_tag).strip(), re.I)
    if m:
        return f"Factor {int(m.group(1))}"
    return str(factor_tag)


def _turbo_norm_from_color_values(arrays: list[np.ndarray]) -> mcolors.Normalize:
    """Single Normalize for turbo across subplots (finite values only)."""
    parts = [a[np.isfinite(a)] for a in arrays if a.size]
    if not parts:
        return mcolors.Normalize(0.0, 1.0)
    c = np.concatenate(parts)
    vmin, vmax = float(np.min(c)), float(np.max(c))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return mcolors.Normalize(0.0, 1.0)
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1e-12
    return mcolors.Normalize(vmin, vmax)


def _print_colorbar_range(
    out_path: Path, norm: mcolors.Normalize, *, quantity: str
) -> None:
    """Stdout: vmin/vmax used for the shared turbo scale."""
    print(
        f"{out_path.name}: colorbar {quantity} range "
        f"[{norm.vmin:g}, {norm.vmax:g}] (shared across subplots)"
    )


def _tissue_quadrant_masks(
    regions: np.ndarray,
    subcortical_labels: frozenset[str],
    *,
    tract_metadata: pd.DataFrame | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Boolean masks (same length as ``regions``) for:
    cortical GM, subcortical GM, WM end at cortex (or other) loc, WM end at subcortex loc, WM core.
    """
    r = regions.astype(str)
    wm = np.array([region_is_white_matter_column(x) for x in r], dtype=bool)
    sub_gm = np.array([str(x) in subcortical_labels for x in r], dtype=bool)
    cort_gm = ~wm & ~sub_gm
    sub_only = ~wm & sub_gm
    wm_end = np.array([region_is_wm_end_segment(x) for x in r], dtype=bool)
    wm_core = np.array([region_is_wm_core_segment(x) for x in r], dtype=bool)

    meta = tract_metadata if tract_metadata is not None else pd.DataFrame()
    n = len(r)
    wm_end_cortex = np.zeros(n, dtype=bool)
    wm_end_subcortex = np.zeros(n, dtype=bool)
    for i in range(n):
        if not wm_end[i]:
            continue
        loc = wm_end_loc_class_from_metadata(str(r[i]), meta)
        if loc == "subcortex":
            wm_end_subcortex[i] = True
        else:
            wm_end_cortex[i] = True
    return cort_gm, sub_only, wm_end_cortex, wm_end_subcortex, wm_core


def _in_glasser_parcel_mask(
    regions: np.ndarray, glasser_names: frozenset[str]
) -> np.ndarray:
    return np.array([str(r) in glasser_names for r in regions], dtype=bool)


_TISSUE_MARKERS: tuple[tuple[str, str, bool], ...] = (
    # (mask_key, marker, white_fill)
    ("cort_gm", "o", False),
    ("sub_only", "s", False),
    ("wm_end_cx", "^", True),
    ("wm_end_sctx", "v", True),
    ("wm_core", "D", True),
)


def _tissue_legend_handles(
    *,
    markersize: float = 9.0,
    markeredgewidth: float = 0.9,
    scatter_point_alpha: float = TISSUE_SCATTER_POINT_ALPHA,
) -> list[Line2D]:
    """Tissue marker legend handles; ``scatter_point_alpha`` matches :data:`TISSUE_SCATTER_POINT_ALPHA` on scatters."""
    return [
        Line2D(
            [0], [0], marker="o", color="k",
            markerfacecolor=TISSUE_CORTICAL_GM, markeredgecolor="black",
            markeredgewidth=markeredgewidth, markersize=markersize,
            linestyle="None", label="Cortex",
            alpha=scatter_point_alpha,
        ),
        Line2D(
            [0], [0], marker="s", color="k",
            markerfacecolor=TISSUE_CORTICAL_GM, markeredgecolor="black",
            markeredgewidth=markeredgewidth, markersize=markersize,
            linestyle="None", label="Subcortex",
            alpha=scatter_point_alpha,
        ),
        Line2D(
            [0], [0], marker="^", color="k",
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=markeredgewidth, markersize=markersize,
            linestyle="None", label="Tract end (cortex)",
            alpha=scatter_point_alpha,
        ),
        Line2D(
            [0], [0], marker="D", color="k",
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=markeredgewidth, markersize=markersize,
            linestyle="None", label="Tract core",
            alpha=scatter_point_alpha,
        ),
        Line2D(
            [0], [0], marker="v", color="k",
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=markeredgewidth, markersize=markersize,
            linestyle="None", label="Tract end (subcortex)",
            alpha=scatter_point_alpha,
        ),
    ]


def _tissue_centroid_patch(*, linewidth: float = 2.1) -> Patch:
    return Patch(
        facecolor=TISSUE_CENTROID_FILL_COLOR,
        edgecolor=TISSUE_CENTROID_EDGE_COLOR,
        linewidth=linewidth,
        label="Tissue class centroid",
    )


def _tissue_legend_handles_with_centroid(
    *,
    markersize: float = 9.0,
    markeredgewidth: float = 0.9,
    centroid_patch_linewidth: float = 2.1,
    scatter_point_alpha: float = TISSUE_SCATTER_POINT_ALPHA,
) -> list[Line2D | Patch]:
    return [
        _tissue_centroid_patch(linewidth=centroid_patch_linewidth),
        *_tissue_legend_handles(
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            scatter_point_alpha=scatter_point_alpha,
        ),
    ]


def _add_by_tissue_figure_legend_top_row(
    ax: plt.Axes,
    *,
    with_centroid: bool = True,
    bbox_y: float | None = None,
    bbox_x: float = 0.5,
    loc: str = "center",
    ncol: int = 2,
) -> None:
    """Tissue shape legend on a dedicated axes (no colorbar entries).

    ``bbox_y`` defaults to :data:`FIGURE_TOP_LEGEND_BBOX_Y` (top-of-axes placement for a
    top legend strip). Pass ``bbox_y=0.5`` to center vertically (left-column legend use).
    ``ncol`` defaults to ``2`` (top-row strip); pass ``ncol=1`` for a narrow left-column
    legend that stacks entries vertically. ``bbox_x`` / ``loc`` allow anchoring the
    legend to the left of its axes (``bbox_x=0.0, loc="center left"``).
    """
    s = FIGURE_TOP_LEGEND_STYLE_SCALE
    ax.set_axis_off()
    handles = (
        _tissue_legend_handles_with_centroid(
            markersize=6.2 * s,
            markeredgewidth=0.8 * s,
            centroid_patch_linewidth=2.1 * s,
        )
        if with_centroid
        else _tissue_legend_handles(
            markersize=6.2 * s, markeredgewidth=0.8 * s
        )
    )
    y = FIGURE_TOP_LEGEND_BBOX_Y if bbox_y is None else float(bbox_y)
    ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=(float(bbox_x), y),
        bbox_transform=ax.transAxes,
        ncol=int(ncol),
        frameon=True,
        fontsize=FIGURE_TOP_LEGEND_FONT_PT * s,
        handlelength=2.2 * s,
        handletextpad=0.6 * s,
        columnspacing=1.25 * s,
        borderpad=0.5 * s,
        labelspacing=0.4 * s,
    )


def _set_xy_ticks_tissue_panel(
    ax: plt.Axes,
    *,
    tick_label_pad_x: float = 6.0,
    tick_labelsize: float = 12.0,
    major_nbins_y: int = 6,
    major_nbins_x: int = 3,
    minor_subdivs: int = 2,
) -> None:
    """Ticks / labels for 2D gradient scatter panels (G2 on x, G1 on y).

    ``major_nbins_x`` defaults to half of ``major_nbins_y`` so long G2 decimals do not overlap.
    """
    ax.xaxis.set_major_locator(MaxNLocator(nbins=major_nbins_x))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=major_nbins_y))
    ax.xaxis.set_minor_locator(AutoMinorLocator(minor_subdivs))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_subdivs))
    for axis in (ax.xaxis, ax.yaxis):
        sf = ScalarFormatter()
        sf.set_scientific(False)
        sf.set_useOffset(False)
        axis.set_major_formatter(sf)
    ax.tick_params(axis="x", which="major", pad=tick_label_pad_x)
    plt.setp(ax.get_xticklabels(minor=False), va="top")
    ax.tick_params(axis="both", which="major", length=4.0, width=0.8, labelsize=tick_labelsize)
    ax.tick_params(axis="both", which="minor", length=1.5, width=0.5)


def _build_tissue_masks_for_row(
    row: GradientRunRow,
    *,
    subcortical_labels: frozenset[str],
    tract_metadata: pd.DataFrame,
    k: int,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
]:
    """Return (tissue masks dict, g1, g2, g3 or None, regions) for a single row, dims=k."""
    g1 = gradient_from_row(row, 0)
    g2 = gradient_from_row(row, 1)
    idx = g1.index.intersection(g2.index)
    if k >= 3:
        g3 = gradient_from_row(row, 2)
        idx = idx.intersection(g3.index)
    g1v = g1.reindex(idx).to_numpy(dtype=np.float64)
    g2v = g2.reindex(idx).to_numpy(dtype=np.float64)
    mask = np.isfinite(g1v) & np.isfinite(g2v)
    if k >= 3:
        g3v = g3.reindex(idx).to_numpy(dtype=np.float64)
        mask &= np.isfinite(g3v)
    g1v = g1v[mask]
    g2v = g2v[mask]
    g3v = g3v[mask] if k >= 3 else None
    regions = idx[mask].astype(str).to_numpy()

    cort_gm, sub_only, wm_end_cx, wm_end_sctx, wm_core = _tissue_quadrant_masks(
        regions,
        subcortical_labels,
        tract_metadata=tract_metadata,
    )
    masks = {
        "cort_gm": cort_gm,
        "sub_only": sub_only,
        "wm_end_cx": wm_end_cx,
        "wm_end_sctx": wm_end_sctx,
        "wm_core": wm_core,
    }
    return masks, g1v, g2v, g3v, regions


def _community_face_colors_for_cortex(
    regions: np.ndarray,
    cort_gm: np.ndarray,
    label_map: Mapping[str, str],
    color_fn: Callable[[str], str],
    *,
    fallback: str,
) -> list[str]:
    cort_ix = np.where(cort_gm)[0]
    if not cort_ix.size:
        return []
    return [
        (
            color_fn(label_map[str(regions[i])])
            if label_map.get(str(regions[i]))
            else fallback
        )
        for i in cort_ix
    ]


def _community_edge_colors_for_cortex(
    regions: np.ndarray,
    cort_gm: np.ndarray,
    label_map: Mapping[str, str],
    color_fn: Callable[[str], str],
    *,
    fallback: str = "black",
) -> list[str]:
    cort_ix = np.where(cort_gm)[0]
    if not cort_ix.size:
        return []
    return [
        (
            color_fn(label_map[str(regions[i])])
            if label_map.get(str(regions[i]))
            else fallback
        )
        for i in cort_ix
    ]


def _cortex_community_legend_handles_from_seen(
    seen: Mapping[str, str],
    *,
    edge_legend: bool = False,
    markersize: float = 5.0,
    markeredgewidth_filled: float = 0.6,
    markeredgewidth_edge: float = 1.4,
    title_case_labels: bool = False,
) -> list[Line2D]:
    """Filled (or edge-only) circle handles for sorted community keys in ``seen``."""

    def _lab_disp(lab: str) -> str:
        t = str(lab).strip()
        return t.title() if title_case_labels else t

    if not seen:
        return []
    order = sorted(seen.keys())
    if edge_legend:
        return [
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor="white",
                markeredgecolor=seen[lab],
                markeredgewidth=markeredgewidth_edge, markersize=markersize,
                linestyle="None", label=_lab_disp(lab),
            )
            for lab in order
        ]
    return [
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=seen[lab],
            markeredgecolor="0.2",
            markeredgewidth=markeredgewidth_filled, markersize=markersize,
            linestyle="None", label=_lab_disp(lab),
        )
        for lab in order
    ]


def _union_glasser_community_label_colors(
    results: list[GradientRunRow],
    *,
    subcortical_labels: frozenset[str],
    tract_metadata: pd.DataFrame,
    glasser_parcel_names: frozenset[str],
    label_map: Mapping[str, str],
    color_fn: Callable[[str], str],
    k: int,
) -> dict[str, str]:
    """Map each community label (Yeo or Mesulam) seen on Glasser cortex in any row → color."""
    out: dict[str, str] = {}
    for row in results:
        masks, _g1, _g2, _g3, regions = _build_tissue_masks_for_row(
            row,
            subcortical_labels=subcortical_labels,
            tract_metadata=tract_metadata,
            k=k,
        )
        gc = masks["cort_gm"] & _in_glasser_parcel_mask(regions, glasser_parcel_names)
        for i in np.flatnonzero(gc):
            lab = label_map.get(str(regions[i]))
            if not lab or not str(lab).strip():
                continue
            sk = str(lab).strip()
            if sk not in out:
                out[sk] = color_fn(sk)
    return {lab: out[lab] for lab in sorted(out.keys())}


def _add_cortex_community_legend_top_row(
    ax: plt.Axes,
    seen: Mapping[str, str],
    *,
    bbox_y: float | None = None,
    bbox_x: float = 0.5,
    loc: str = "center",
    ncol: int = 2,
) -> None:
    """Legend strip (axis off) for Yeo / Mesulam; matches tissue top-row legend styling.

    ``bbox_y`` defaults to :data:`FIGURE_TOP_LEGEND_BBOX_Y` (top-of-axes placement). Pass
    ``bbox_y=0.5`` to center vertically for a left-column legend. ``ncol`` defaults to
    ``2`` for the top-row strip; pass ``ncol=1`` for the narrow left-column variant.
    ``bbox_x`` / ``loc`` allow anchoring to the left of the axes
    (``bbox_x=0.0, loc="center left"``).
    """
    s = FIGURE_TOP_LEGEND_STYLE_SCALE
    ax.set_axis_off()
    markersize = 6.2 * s
    mew = 0.8 * s
    handles = _cortex_community_legend_handles_from_seen(
        seen,
        edge_legend=False,
        markersize=markersize,
        markeredgewidth_filled=mew,
        markeredgewidth_edge=1.4 * s,
        title_case_labels=True,
    )
    if not handles:
        return
    y = FIGURE_TOP_LEGEND_BBOX_Y if bbox_y is None else float(bbox_y)
    ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=(float(bbox_x), y),
        bbox_transform=ax.transAxes,
        ncol=int(ncol),
        frameon=True,
        fontsize=FIGURE_TOP_LEGEND_FONT_PT * s,
        handlelength=2.2 * s,
        handletextpad=0.6 * s,
        columnspacing=1.25 * s,
        borderpad=0.5 * s,
        labelspacing=0.4 * s,
    )


def _add_cortex_community_legend(
    ax: plt.Axes,
    regions: np.ndarray,
    cort_gm: np.ndarray,
    label_map: Mapping[str, str],
    color_fn: Callable[[str], str],
    *,
    edge_legend: bool = False,
) -> None:
    seen: dict[str, str] = {}
    for r0 in regions[cort_gm]:
        lab = label_map.get(str(r0))
        if not lab or not str(lab).strip():
            continue
        sk = str(lab).strip()
        if sk not in seen:
            seen[sk] = color_fn(sk)
    handles = _cortex_community_legend_handles_from_seen(seen, edge_legend=edge_legend)
    if not handles:
        return
    ax.legend(
        handles=handles,
        loc="lower right",
        framealpha=0.9,
        fontsize=6.5,
        handletextpad=0.4,
        borderpad=0.3,
    )


def _indices_by_assigned_label(
    regions: np.ndarray,
    gc: np.ndarray,
    label_map: Mapping[str, str],
) -> dict[str, np.ndarray]:
    """Glasser-cortex indices ``gc`` grouped by trimmed non-empty ``label_map`` assignment."""
    pools: defaultdict[str, list[int]] = defaultdict(list)
    for i in np.flatnonzero(gc):
        lab = label_map.get(str(regions[i]))
        if not lab or not str(lab).strip():
            continue
        pools[str(lab).strip()].append(int(i))
    return {k: np.array(v, dtype=np.int64) for k, v in sorted(pools.items())}


def _hide_top_right_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _equalize_tissue_scatter_axes_boxes(tissue_axes: list[plt.Axes]) -> None:
    """Give every row-1 tissue panel the same axes width/height (figure coords).

    Per-factor ``xlim`` / ``ylim`` are unchanged. Avoids ``set_aspect(..., adjustable='box')``,
    which shrinks panels differently when gradient ranges differ across factors.
    """
    if not tissue_axes:
        return
    positions = [ax.get_position() for ax in tissue_axes]
    w = max(p.width for p in positions)
    h = max(p.height for p in positions)
    for ax in tissue_axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, w, h])
        ax.set_aspect("auto")


def _decorate_gradient_axes_2d(
    ax: plt.Axes,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
    show_title: bool,
    row: GradientRunRow,
    axis_label_fs: float,
    axis_tick_pad_x: float,
    axis_tick_fs: float,
) -> None:
    if show_title:
        ax.set_title(
            _factor_display_name(row[0]),
            fontsize=axis_label_fs,
            color=GRADIENT_AXIS_LABEL_COLOR,
        )
    lp = 5.5
    xlab_kw = {
        "color": GRADIENT_AXIS_LABEL_COLOR,
        "labelpad": lp,
        "fontsize": axis_label_fs,
    }
    ylab_kw = {
        "color": GRADIENT_AXIS_LABEL_COLOR,
        "labelpad": lp,
        "fontsize": axis_label_fs,
    }
    if show_xlabel:
        ax.set_xlabel("Gradient 2", **xlab_kw)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Gradient 1", **ylab_kw)
    else:
        ax.set_ylabel("")
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    _set_xy_ticks_tissue_panel(
        ax,
        tick_label_pad_x=axis_tick_pad_x,
        tick_labelsize=axis_tick_fs,
    )
    _hide_top_right_spines(ax)


def _plot_panel_2d_glasser_community_centroids(
    ax: plt.Axes,
    row: GradientRunRow,
    *,
    regions: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
    gc: np.ndarray,
    label_map: Mapping[str, str],
    color_fn: Callable[[str], str],
    show_xlabel: bool,
    show_ylabel: bool,
    show_title: bool,
    axis_label_fs: float = _GRADIENT_BY_FIGURE_AXIS_LABEL_FS_2D,
    axis_tick_pad_x: float = 6.0,
    axis_tick_fs: float = _GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D,
) -> None:
    """Mean (G2, G1) per assigned Yeo / Mesulam label among Glasser cortical ROIs."""
    s_plot = 26.0
    centroid_s = max(s_plot * 4.0, 70.0)
    edge_w = 2.05 + 0.4 * (s_plot / 26.0)
    for _lab, ix in _indices_by_assigned_label(regions, gc, label_map).items():
        if ix.size == 0:
            continue
        cx = float(np.mean(xv[ix]))
        cy = float(np.mean(yv[ix]))
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        fc = color_fn(_lab)
        ax.scatter(
            [cx],
            [cy],
            s=centroid_s,
            c=[fc],
            marker="o",
            edgecolors="black",
            linewidths=edge_w,
            alpha=1.0,
            zorder=5,
        )
    _decorate_gradient_axes_2d(
        ax,
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        show_title=show_title,
        row=row,
        axis_label_fs=axis_label_fs,
        axis_tick_pad_x=axis_tick_pad_x,
        axis_tick_fs=axis_tick_fs,
    )


def _decorate_gradient_axes_3d(
    ax,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
    show_zlabel: bool,
    show_title: bool,
    row: GradientRunRow,
    axis_label_fs: float,
    axis_tick_fs: float,
) -> None:
    if show_title:
        ax.set_title(
            _factor_display_name(row[0]),
            fontsize=axis_label_fs,
            color=GRADIENT_AXIS_LABEL_COLOR,
        )
    xlab_kw = {"color": GRADIENT_AXIS_LABEL_COLOR, "fontsize": axis_label_fs}
    ylab_kw = {"color": GRADIENT_AXIS_LABEL_COLOR, "fontsize": axis_label_fs}
    zlab_kw = {"color": GRADIENT_AXIS_LABEL_COLOR, "fontsize": axis_label_fs}
    ax.set_xlabel("Gradient 2" if show_xlabel else "", **xlab_kw)
    ax.set_ylabel("Gradient 3" if show_ylabel else "", **ylab_kw)
    ax.set_zlabel("Gradient 1" if show_zlabel else "", **zlab_kw)
    ax.tick_params(axis="x", labelsize=axis_tick_fs)
    ax.tick_params(axis="y", labelsize=axis_tick_fs)
    ax.tick_params(axis="z", labelsize=axis_tick_fs)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except (AttributeError, NotImplementedError):
        pass


def _plot_panel_3d_glasser_community_centroids(
    ax,
    row: GradientRunRow,
    *,
    regions: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
    zv: np.ndarray,
    gc: np.ndarray,
    label_map: Mapping[str, str],
    color_fn: Callable[[str], str],
    show_xlabel: bool,
    show_ylabel: bool,
    show_zlabel: bool,
    show_title: bool,
    axis_label_fs: float = _GRADIENT_BY_FIGURE_AXIS_LABEL_FS_3D,
    axis_tick_fs: float = _GRADIENT_BY_FIGURE_AXIS_TICK_FS_3D,
) -> None:
    """Mean (G2, G3, G1) per assigned Yeo / Mesulam label among Glasser cortical ROIs."""
    s_ref = SCATTER_POINT_SIZE_3D
    centroid_s = max(float(s_ref) * 4.0, 70.0)
    edge_w = 2.05 + 0.4 * (float(s_ref) / 26.0)
    z_ord = 50.0
    for _lab, ix in _indices_by_assigned_label(regions, gc, label_map).items():
        if ix.size == 0:
            continue
        cx = float(np.mean(xv[ix]))
        cy = float(np.mean(yv[ix]))
        cz = float(np.mean(zv[ix]))
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(cz)):
            continue
        fc = color_fn(_lab)
        ax.scatter(
            [cx],
            [cy],
            [cz],
            s=centroid_s,
            c=[fc],
            marker="o",
            edgecolors="black",
            linewidths=edge_w,
            alpha=1.0,
            depthshade=False,
            zorder=z_ord,
        )
    _decorate_gradient_axes_3d(
        ax,
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        show_zlabel=show_zlabel,
        show_title=show_title,
        row=row,
        axis_label_fs=axis_label_fs,
        axis_tick_fs=axis_tick_fs,
    )


# ----------------------------------------------------------------------
# 2D panel painter
# ----------------------------------------------------------------------


def _plot_panel_2d(
    ax: plt.Axes,
    row: GradientRunRow,
    *,
    subcortical_labels: frozenset[str],
    tract_metadata: pd.DataFrame,
    color_by: ColorBy,
    yeo_by_roi: Mapping[str, str] | None,
    mesulam_by_roi: Mapping[str, str] | None,
    show_xlabel: bool,
    show_ylabel: bool,
    show_title: bool,
    axis_label_fs: float = _GRADIENT_BY_FIGURE_AXIS_LABEL_FS_2D,
    axis_tick_pad_x: float = 6.0,
    axis_tick_fs: float = _GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D,
    glasser_parcel_names: frozenset[str] | None = None,
    draw_tissue_marker_points: bool = True,
    draw_tissue_class_centroids: bool = True,
    tissue_centroids_use_legend_style: bool = False,
) -> None:
    masks, g1v, g2v, _g3v, regions = _build_tissue_masks_for_row(
        row,
        subcortical_labels=subcortical_labels,
        tract_metadata=tract_metadata,
        k=2,
    )
    xv, yv = g2v, g1v
    s_plot = 26.0
    lw = TISSUE_POINT_EDGEWIDTH
    gm = TISSUE_CORTICAL_GM

    # Glasser cortical columns (Yeo or Mesulam): ``cort_gm`` ∩ Glasser atlas parcels only.
    if glasser_parcel_names is not None:
        label_map: Mapping[str, str] | None = None
        color_fn: Callable[[str], str] | None = None
        if color_by == "yeo" and yeo_by_roi is not None:
            label_map, color_fn = yeo_by_roi, yeo_network_color
        elif color_by == "mesulam" and mesulam_by_roi is not None:
            label_map, color_fn = mesulam_by_roi, mesulam_type_color
        if label_map is not None and color_fn is not None:
            gc = masks["cort_gm"] & _in_glasser_parcel_mask(regions, glasser_parcel_names)
            if np.any(gc):
                face = _community_face_colors_for_cortex(
                    regions, gc, label_map, color_fn, fallback=gm
                )
                ax.scatter(
                    xv[gc],
                    yv[gc],
                    c=face,
                    marker="o",
                    edgecolors="black",
                    linewidths=lw,
                    alpha=TISSUE_SCATTER_POINT_ALPHA,
                    s=s_plot,
                    zorder=1,
                )
            _decorate_gradient_axes_2d(
                ax,
                show_xlabel=show_xlabel,
                show_ylabel=show_ylabel,
                show_title=show_title,
                row=row,
                axis_label_fs=axis_label_fs,
                axis_tick_pad_x=axis_tick_pad_x,
                axis_tick_fs=axis_tick_fs,
            )
            return

    cort_face: list[str] | str = gm
    if color_by == "yeo" and yeo_by_roi is not None:
        face = _community_face_colors_for_cortex(
            regions, masks["cort_gm"], yeo_by_roi, yeo_network_color, fallback=gm
        )
        if face:
            cort_face = face
    elif color_by == "mesulam" and mesulam_by_roi is not None:
        face = _community_face_colors_for_cortex(
            regions, masks["cort_gm"], mesulam_by_roi, mesulam_type_color, fallback=gm
        )
        if face:
            cort_face = face

    if draw_tissue_marker_points:
        for mkey, marker, white in _TISSUE_MARKERS:
            m = masks[mkey]
            if not np.any(m):
                continue
            if mkey == "cort_gm":
                ax.scatter(
                    xv[m], yv[m],
                    c=cort_face if isinstance(cort_face, list) else cort_face,
                    marker=marker, edgecolors="black",
                    linewidths=lw, alpha=TISSUE_SCATTER_POINT_ALPHA, s=s_plot, zorder=1,
                )
            elif mkey == "sub_only":
                ax.scatter(
                    xv[m], yv[m],
                    c=gm, marker=marker, edgecolors="black",
                    linewidths=lw, alpha=TISSUE_SCATTER_POINT_ALPHA, s=s_plot, zorder=2,
                )
            else:
                ax.scatter(
                    xv[m], yv[m],
                    facecolors="white" if white else gm,
                    edgecolors="black",
                    marker=marker, linewidths=lw,
                    alpha=TISSUE_SCATTER_POINT_ALPHA, s=s_plot,
                    zorder=3 if mkey != "wm_core" else 4,
                )

    if draw_tissue_class_centroids:
        centroid_s = max(s_plot * 4.0, 70.0)
        edge_w = 2.05 + 0.4 * (s_plot / 26.0)
        for mkey, marker, white in _TISSUE_MARKERS:
            m = masks[mkey]
            if not np.any(m):
                continue
            cx, cy = float(np.mean(xv[m])), float(np.mean(yv[m]))
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue
            if tissue_centroids_use_legend_style:
                if mkey == "cort_gm":
                    c_face = gm if isinstance(cort_face, list) else cort_face
                    ax.scatter(
                        [cx],
                        [cy],
                        s=centroid_s,
                        marker=marker,
                        c=c_face,
                        edgecolors="black",
                        linewidths=edge_w,
                        alpha=TISSUE_SCATTER_POINT_ALPHA,
                        zorder=32.0,
                    )
                elif mkey == "sub_only":
                    ax.scatter(
                        [cx],
                        [cy],
                        s=centroid_s,
                        marker=marker,
                        c=gm,
                        edgecolors="black",
                        linewidths=edge_w,
                        alpha=TISSUE_SCATTER_POINT_ALPHA,
                        zorder=32.0,
                    )
                else:
                    fc = "white" if white else gm
                    ax.scatter(
                        [cx],
                        [cy],
                        s=centroid_s,
                        marker=marker,
                        facecolors=fc,
                        edgecolors="black",
                        linewidths=edge_w,
                        alpha=TISSUE_SCATTER_POINT_ALPHA,
                        zorder=32.0,
                    )
            else:
                ax.scatter(
                    [cx],
                    [cy],
                    s=centroid_s,
                    marker=marker,
                    facecolors=TISSUE_CENTROID_FILL_COLOR,
                    edgecolors=TISSUE_CENTROID_EDGE_COLOR,
                    linewidths=edge_w,
                    zorder=32.0,
                )

    _decorate_gradient_axes_2d(
        ax,
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        show_title=show_title,
        row=row,
        axis_label_fs=axis_label_fs,
        axis_tick_pad_x=axis_tick_pad_x,
        axis_tick_fs=axis_tick_fs,
    )


def _add_tissue_class_centroids_3d(
    ax,
    xv: np.ndarray,
    yv: np.ndarray,
    zv: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    s_point_ref: float,
    use_legend_point_style: bool = False,
    gm_fill: str | None = None,
) -> None:
    """Per-tissue-class mean position in (G2, G3, G1).

    Default: translucent yellow fill and orange edges (matches legacy 2D centroids).
    With ``use_legend_point_style=True``, face/edge colors follow ``_TISSUE_MARKERS`` /
    ``legend-tissue.png`` (cortex and subcortex filled grey, tract classes white fill).
    """
    centroid_s = max(float(s_point_ref) * 4.0, 70.0)
    edge_w = 2.05 + 0.4 * (float(s_point_ref) / 26.0)
    z_centroid = 50.0
    gm = gm_fill if gm_fill is not None else TISSUE_CORTICAL_GM
    for mkey, marker, white in _TISSUE_MARKERS:
        m = masks[mkey]
        if not np.any(m):
            continue
        cx = float(np.mean(xv[m]))
        cy = float(np.mean(yv[m]))
        cz = float(np.mean(zv[m]))
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(cz)):
            continue
        if use_legend_point_style:
            fc = "white" if white else gm
            ax.scatter(
                [cx],
                [cy],
                [cz],
                s=centroid_s,
                marker=marker,
                facecolors=fc,
                edgecolors="black",
                linewidths=edge_w,
                alpha=TISSUE_SCATTER_POINT_ALPHA,
                depthshade=False,
                zorder=z_centroid,
            )
        else:
            ax.scatter(
                [cx],
                [cy],
                [cz],
                s=centroid_s,
                marker=marker,
                facecolors=TISSUE_CENTROID_FILL_COLOR,
                edgecolors=TISSUE_CENTROID_EDGE_COLOR,
                linewidths=edge_w,
                depthshade=False,
                zorder=z_centroid,
            )


# ----------------------------------------------------------------------
# 3D panel painter
# ----------------------------------------------------------------------


def _apply_shared_3d_lims(
    axes: Sequence,
    xv: np.ndarray,
    yv: np.ndarray,
    zv: np.ndarray,
    *,
    pad_frac: float = 0.04,
) -> None:
    """Set matching x/y/z limits on each 3D axis (mpl 3D sharing is unreliable)."""
    m = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(zv)
    if not np.any(m):
        return
    xr = (float(np.min(xv[m])), float(np.max(xv[m])))
    yr = (float(np.min(yv[m])), float(np.max(yv[m])))
    zr = (float(np.min(zv[m])), float(np.max(zv[m])))
    span = max(xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0], 1e-12)
    pad = pad_frac * span
    for ax in axes:
        ax.set_xlim(xr[0] - pad, xr[1] + pad)
        ax.set_ylim(yr[0] - pad, yr[1] + pad)
        ax.set_zlim(zr[0] - pad, zr[1] + pad)


def _plot_panel_3d(
    ax,
    row: GradientRunRow,
    *,
    subcortical_labels: frozenset[str],
    tract_metadata: pd.DataFrame,
    color_by: ColorBy,
    yeo_by_roi: Mapping[str, str] | None,
    mesulam_by_roi: Mapping[str, str] | None,
    g1_norm: mcolors.Normalize | None,
    show_xlabel: bool,
    show_ylabel: bool,
    show_zlabel: bool,
    show_title: bool,
    axis_label_fs: float = _GRADIENT_BY_FIGURE_AXIS_LABEL_FS_3D,
    axis_tick_fs: float = _GRADIENT_BY_FIGURE_AXIS_TICK_FS_3D,
    color_face_by_gradient1: bool = True,
    glasser_parcel_names: frozenset[str] | None = None,
    draw_tissue_marker_points: bool = True,
    draw_tissue_class_centroids: bool = True,
    tissue_centroids_use_legend_style: bool = False,
) -> None:
    """3D panel: ``x=G2``, ``y=G3``, ``z=G1``.

    When ``color_face_by_gradient1`` is True (standalone by-G1 tissue 3D), face color
    encodes ``G1`` via ``g1_norm`` / turbo and marker shape encodes tissue (optional
    ``draw_tissue_class_centroids``; the standalone figure passes ``False``).

    When False (Yeo/Mesulam quad), face colors follow the 2D convention: tissue column
    uses fixed tissue fills; Yeo / Mesulam columns show **Glasser cortical** parcels only,
    with Yeo / Mesulam face colors (no G1 turbo). Use ``draw_tissue_marker_points`` /
    ``draw_tissue_class_centroids`` to show tissue ROI markers without centroid overlay, or
    centroid markers only (tissue centroid column in the six-column Yeo/Mesulam figure).
    """
    masks, g1v, g2v, g3v, regions = _build_tissue_masks_for_row(
        row,
        subcortical_labels=subcortical_labels,
        tract_metadata=tract_metadata,
        k=3,
    )
    assert g3v is not None  # k=3 above
    xv, yv, zv = g2v, g3v, g1v

    s_plot = SCATTER_POINT_SIZE_3D
    lw = TISSUE_POINT_EDGEWIDTH
    gm = TISSUE_CORTICAL_GM

    def _finish_axes() -> None:
        if show_title:
            ax.set_title(
                _factor_display_name(row[0]),
                fontsize=axis_label_fs,
                color=GRADIENT_AXIS_LABEL_COLOR,
            )
        xlab_kw = {"color": GRADIENT_AXIS_LABEL_COLOR, "fontsize": axis_label_fs}
        ylab_kw = {"color": GRADIENT_AXIS_LABEL_COLOR, "fontsize": axis_label_fs}
        zlab_kw = {"color": GRADIENT_AXIS_LABEL_COLOR, "fontsize": axis_label_fs}
        ax.set_xlabel("Gradient 2" if show_xlabel else "", **xlab_kw)
        ax.set_ylabel("Gradient 3" if show_ylabel else "", **ylab_kw)
        ax.set_zlabel("Gradient 1" if show_zlabel else "", **zlab_kw)
        ax.tick_params(axis="x", labelsize=axis_tick_fs)
        ax.tick_params(axis="y", labelsize=axis_tick_fs)
        ax.tick_params(axis="z", labelsize=axis_tick_fs)
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except (AttributeError, NotImplementedError):
            pass

    if not color_face_by_gradient1:
        if color_by == "tissue":
            z_ord = 1
            if draw_tissue_marker_points:
                for mkey, marker, white in _TISSUE_MARKERS:
                    m = masks[mkey]
                    if not np.any(m):
                        continue
                    fc = "white" if white else gm
                    ax.scatter(
                        xv[m],
                        yv[m],
                        zv[m],
                        facecolors=fc,
                        edgecolors="black",
                        marker=marker,
                        linewidths=lw,
                        alpha=TISSUE_SCATTER_POINT_ALPHA,
                        s=s_plot,
                        depthshade=False,
                        zorder=z_ord,
                    )
                    z_ord += 1
            if draw_tissue_class_centroids:
                _add_tissue_class_centroids_3d(
                    ax,
                    xv,
                    yv,
                    zv,
                    masks,
                    s_point_ref=s_plot,
                    use_legend_point_style=tissue_centroids_use_legend_style,
                    gm_fill=gm,
                )
        elif color_by == "yeo" and yeo_by_roi is not None and glasser_parcel_names is not None:
            gc = masks["cort_gm"] & _in_glasser_parcel_mask(regions, glasser_parcel_names)
            if np.any(gc):
                face = _community_face_colors_for_cortex(
                    regions,
                    gc,
                    yeo_by_roi,
                    yeo_network_color,
                    fallback=gm,
                )
                ax.scatter(
                    xv[gc],
                    yv[gc],
                    zv[gc],
                    c=face,
                    marker="o",
                    edgecolors="black",
                    linewidths=lw,
                    alpha=TISSUE_SCATTER_POINT_ALPHA,
                    s=s_plot,
                    depthshade=False,
                    zorder=1,
                )
        elif (
            color_by == "mesulam"
            and mesulam_by_roi is not None
            and glasser_parcel_names is not None
        ):
            gc = masks["cort_gm"] & _in_glasser_parcel_mask(regions, glasser_parcel_names)
            if np.any(gc):
                face = _community_face_colors_for_cortex(
                    regions,
                    gc,
                    mesulam_by_roi,
                    mesulam_type_color,
                    fallback=gm,
                )
                ax.scatter(
                    xv[gc],
                    yv[gc],
                    zv[gc],
                    c=face,
                    marker="o",
                    edgecolors="black",
                    linewidths=lw,
                    alpha=TISSUE_SCATTER_POINT_ALPHA,
                    s=s_plot,
                    depthshade=False,
                    zorder=1,
                )
        _finish_axes()
        return

    if g1_norm is None:
        raise ValueError("g1_norm is required when color_face_by_gradient1 is True")

    def _scatter_subset(
        mask: np.ndarray,
        marker: str,
        *,
        edge_color="black",
        zorder: int = 1,
    ) -> None:
        if not np.any(mask):
            return
        ax.scatter(
            xv[mask],
            yv[mask],
            zv[mask],
            c=zv[mask],
            cmap="turbo",
            norm=g1_norm,
            marker=marker,
            edgecolors=edge_color,
            linewidths=lw,
            alpha=TISSUE_SCATTER_POINT_ALPHA,
            s=s_plot,
            depthshade=False,
            zorder=zorder,
        )

    if color_by == "yeo" and yeo_by_roi is not None:
        cort_edges = _community_edge_colors_for_cortex(
            regions, masks["cort_gm"], yeo_by_roi, yeo_network_color
        )
    elif color_by == "mesulam" and mesulam_by_roi is not None:
        cort_edges = _community_edge_colors_for_cortex(
            regions, masks["cort_gm"], mesulam_by_roi, mesulam_type_color
        )
    else:
        cort_edges = None

    cort_mask = masks["cort_gm"]
    if cort_edges is not None and cort_edges and np.any(cort_mask):
        ax.scatter(
            xv[cort_mask],
            yv[cort_mask],
            zv[cort_mask],
            c=zv[cort_mask],
            cmap="turbo",
            norm=g1_norm,
            marker="o",
            edgecolors=cort_edges,
            linewidths=lw * 2.0,
            alpha=TISSUE_SCATTER_POINT_ALPHA,
            s=s_plot,
            depthshade=False,
            zorder=1,
        )
    else:
        _scatter_subset(cort_mask, "o", edge_color="black", zorder=1)

    _scatter_subset(masks["sub_only"], "s", edge_color="black", zorder=2)
    _scatter_subset(masks["wm_end_cx"], "^", edge_color="black", zorder=3)
    _scatter_subset(masks["wm_end_sctx"], "v", edge_color="black", zorder=3)
    _scatter_subset(masks["wm_core"], "D", edge_color="black", zorder=4)

    if color_by == "tissue" and draw_tissue_class_centroids:
        _add_tissue_class_centroids_3d(
            ax,
            xv,
            yv,
            zv,
            masks,
            s_point_ref=s_plot,
            use_legend_point_style=tissue_centroids_use_legend_style,
            gm_fill=gm,
        )

    _finish_axes()


# ----------------------------------------------------------------------
# Figure assembly
# ----------------------------------------------------------------------


def _figure_method_label(method_tag: str) -> str:
    return {
        "diffusion_embedding": "Diffusion-map",
        "laplacian_eigenmodes": "Laplacian-eigenmap",
    }.get(method_tag, method_tag)


def _shared_g1_norm(results: list[GradientRunRow], *, k_dims: int) -> mcolors.Normalize:
    """Shared turbo Normalize across rows: built from G1 values on the finite ROI set used
    by each panel (which depends on ``k_dims`` because the 3D panel intersects G3 too)."""
    cvals: list[np.ndarray] = []
    for row in results:
        g1 = gradient_from_row(row, 0)
        g2 = gradient_from_row(row, 1)
        idx = g1.index.intersection(g2.index)
        if k_dims >= 3:
            g3 = gradient_from_row(row, 2)
            idx = idx.intersection(g3.index)
        a1 = g1.reindex(idx).to_numpy(dtype=np.float64)
        a2 = g2.reindex(idx).to_numpy(dtype=np.float64)
        m = np.isfinite(a1) & np.isfinite(a2)
        if k_dims >= 3:
            a3 = g3.reindex(idx).to_numpy(dtype=np.float64)
            m &= np.isfinite(a3)
        cvals.append(a1[m])
    return _turbo_norm_from_color_values(cvals)


def _add_g1_colorbar_left(
    fig: plt.Figure,
    cax: plt.Axes,
    norm: mcolors.Normalize,
    *,
    label: str = "Gradient 1 (turbo)",
    label_fontsize: float | None = None,
    tick_labelsize: float | None = None,
) -> None:
    """Vertical turbo colorbar to the left of the data column(s).

    Tick labels and the axis title are placed on the **left** so the colorbar reads as
    a true left margin rather than crowding column 1.
    """
    sm = cm.ScalarMappable(norm=norm, cmap="turbo")
    sm.set_array([])
    ls = label_fontsize if label_fontsize is not None else float(plt.rcParams.get("axes.labelsize", 15))
    ts = tick_labelsize if tick_labelsize is not None else float(plt.rcParams.get("ytick.labelsize", 13))
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.set_label(label, fontsize=ls)
    cbar.ax.tick_params(labelsize=ts)


def save_standalone_legend_gradient1(
    results: list[GradientRunRow],
    out_path: Path,
    *,
    k_dims: int,
) -> Path:
    """Vertical turbo colorbar for Gradient 1 (same norm as ``plot_gradients_by_gradient1_scatter``)."""
    if len(results) == 0:
        return out_path
    norm = _shared_g1_norm(results, k_dims=k_dims)
    _print_colorbar_range(
        out_path, norm, quantity=f"Gradient 1 (standalone legend, k_dims={k_dims})"
    )
    fig = plt.figure(figsize=(2.4, 5.2), facecolor="white")
    cax = fig.add_axes([0.42, 0.12, 0.22, 0.72])
    _add_g1_colorbar_left(fig, cax, norm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def save_standalone_legend_tissue(out_path: Path) -> Path:
    """Tissue marker legend in one horizontal row (1x5).

    Uses the same legend styling as ``_add_by_tissue_figure_legend_top_row``, with serif
    text matching ``compute_gradients`` (Georgia first, then DejaVu Serif). Handle order:
    Cortex, Subcortex, Tract end (cortex), Tract core, Tract end (subcortex).
    """
    s = FIGURE_TOP_LEGEND_STYLE_SCALE
    ms = 6.2 * s
    mew = 0.8 * s
    handles = _tissue_legend_handles(markersize=ms, markeredgewidth=mew)

    with figure_font_context():
        fig = plt.figure(
            figsize=(
                _TISSUE_STANDALONE_LEGEND_FIG_WIDTH,
                _TISSUE_STANDALONE_LEGEND_FIG_HEIGHT,
            ),
            facecolor="white",
        )
        ax = fig.add_axes([0.02, 0.12, 0.96, 0.76])
        ax.set_axis_off()
        ax.legend(
            handles=handles,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            bbox_transform=ax.transAxes,
            ncol=5,
            frameon=True,
            fontsize=FIGURE_TOP_LEGEND_FONT_PT * s,
            handlelength=_TISSUE_STANDALONE_LEGEND_HANDLELENGTH,
            handletextpad=_TISSUE_STANDALONE_LEGEND_HANDLETEXTPAD,
            columnspacing=_TISSUE_STANDALONE_LEGEND_COLUMNSPACING,
            borderpad=0.35,
            labelspacing=0.3,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return out_path


def save_standalone_legend_yeo(
    results: list[GradientRunRow],
    out_path: Path,
    *,
    tractometry_root: Path | None,
    k_dims: int,
) -> Path:
    """Yeo network legend (union of Glasser-cortex labels across factors)."""
    if len(results) == 0:
        return out_path
    root = tractometry_root if tractometry_root is not None else DEFAULT_TRACTOMETRY_ROOT
    subcortical_labels = subcortical_grey_matter_column_names(root)
    tract_meta = load_hcp1065_tract_metadata(root)
    glasser_names = glasser_parcel_name_set(root)
    yeo_by = load_yeo_labels(root)
    yeo_seen = _union_glasser_community_label_colors(
        results,
        subcortical_labels=subcortical_labels,
        tract_metadata=tract_meta,
        glasser_parcel_names=glasser_names,
        label_map=yeo_by,
        color_fn=yeo_network_color,
        k=k_dims,
    )
    fig = plt.figure(figsize=(4.8, 4.5), facecolor="white")
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.84])
    _add_cortex_community_legend_top_row(
        ax, yeo_seen, bbox_y=0.5, bbox_x=0.5, loc="center", ncol=2
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def save_standalone_legend_mesulam(
    results: list[GradientRunRow],
    out_path: Path,
    *,
    tractometry_root: Path | None,
    k_dims: int,
) -> Path:
    """Mesulam community legend (union of Glasser-cortex labels across factors)."""
    if len(results) == 0:
        return out_path
    root = tractometry_root if tractometry_root is not None else DEFAULT_TRACTOMETRY_ROOT
    subcortical_labels = subcortical_grey_matter_column_names(root)
    tract_meta = load_hcp1065_tract_metadata(root)
    glasser_names = glasser_parcel_name_set(root)
    mesu_by = load_mesulam_labels(root)
    mes_seen = _union_glasser_community_label_colors(
        results,
        subcortical_labels=subcortical_labels,
        tract_metadata=tract_meta,
        glasser_parcel_names=glasser_names,
        label_map=mesu_by,
        color_fn=mesulam_type_color,
        k=k_dims,
    )
    fig = plt.figure(figsize=(4.2, 3.4), facecolor="white")
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])
    _add_cortex_community_legend_top_row(
        ax, mes_seen, bbox_y=0.5, bbox_x=0.5, loc="center", ncol=2
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Public API (scatter figures)
# ----------------------------------------------------------------------


def plot_gradients_by_gradient1_scatter(
    results: list[GradientRunRow],
    out_path: Path,
    *,
    dims: Dims = 2,
    tractometry_root: Path | None = None,
    method_tag: str = "laplacian_eigenmodes",
    cohort_tag: str | None = "controls",
) -> Path:
    """Standalone G2–G1 embedding scatters; **point color = Gradient 1** (turbo).

    Factors are arranged **as columns** in a single row of panels. The turbo color scale
    is emitted separately as ``legend-gradient1.png`` (see ``save_standalone_legend_gradient1``).

    * ``dims=2``: ``1 × N`` grid of G2 vs G1 panels; only the leftmost panel shows the
      ``Gradient 1`` y-axis label / ticks.
    * ``dims=3``: ``1 × N`` grid of 3D panels (x=G2, y=G3, z=G1) with G1 turbo face color
      and tissue marker shapes (no tissue-class centroid overlay). Tissue marker encoding
      is documented in ``legend-tissue.png``.
    """
    n = len(results)
    if n == 0:
        return out_path
    root = tractometry_root if tractometry_root is not None else DEFAULT_TRACTOMETRY_ROOT
    subcortical_labels = subcortical_grey_matter_column_names(root)
    tract_meta = load_hcp1065_tract_metadata(root)

    col_w = 6.2
    row_h = 5.75
    axis_label_fs = _GRADIENT_BY_FIGURE_AXIS_LABEL_FS_2D
    axis_tick_pad_x = 6.0
    axis_tick_fs = _GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D
    lp = 5.5
    xlab_kw = {
        "color": GRADIENT_AXIS_LABEL_COLOR,
        "labelpad": lp,
        "fontsize": axis_label_fs,
    }
    ylab_kw = {
        "color": GRADIENT_AXIS_LABEL_COLOR,
        "labelpad": lp,
        "fontsize": axis_label_fs,
    }

    if dims == 2:
        g1_norm = _shared_g1_norm(results, k_dims=2)
        _print_colorbar_range(out_path, g1_norm, quantity="Gradient 1 (2D scatter)")
        fig = plt.figure(figsize=(col_w * n + 0.6, row_h))
        gs = fig.add_gridspec(1, n, wspace=0.18)
        for c, row in enumerate(results):
            ax = fig.add_subplot(gs[0, c])
            g1 = gradient_from_row(row, 0)
            g2 = gradient_from_row(row, 1)
            idx = g1.index.intersection(g2.index)
            xv = g2.reindex(idx).to_numpy(dtype=np.float64)
            yv = g1.reindex(idx).to_numpy(dtype=np.float64)
            m = np.isfinite(xv) & np.isfinite(yv)
            xv, yv = xv[m], yv[m]
            ax.scatter(
                xv,
                yv,
                c=yv,
                cmap="turbo",
                norm=g1_norm,
                alpha=0.78,
                s=26.0,
                edgecolors="none",
            )
            ax.set_title(
                _factor_display_name(row[0]),
                fontsize=axis_label_fs,
                color=GRADIENT_AXIS_LABEL_COLOR,
            )
            ax.set_xlabel("Gradient 2", **xlab_kw)
            if c == 0:
                ax.set_ylabel("Gradient 1", **ylab_kw)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
            ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
            _set_xy_ticks_tissue_panel(
                ax,
                tick_label_pad_x=axis_tick_pad_x,
                tick_labelsize=axis_tick_fs,
            )
        fig.tight_layout(
            pad=0.35,
            h_pad=0.35,
            w_pad=0.38,
            rect=[0.0, 0.04, 1.0, 0.94 if cohort_tag else 0.98],
        )
        if cohort_tag:
            fig.suptitle(
                f"{_figure_method_label(method_tag)} G2 vs G1 (color = Gradient 1) | "
                f"cohort={cohort_tag}",
                fontsize=_GRADIENT_BY_FIGURE_SUPTITLE_FS,
                color=GRADIENT_AXIS_LABEL_COLOR,
                y=0.995,
            )
    else:
        fig = plt.figure(figsize=(col_w * n + 0.6, row_h))
        gs = fig.add_gridspec(1, n, wspace=0.18)
        g1_norm = _shared_g1_norm(results, k_dims=3)
        _print_colorbar_range(out_path, g1_norm, quantity="Gradient 1 (3D scatter)")
        for c, row in enumerate(results):
            ax = fig.add_subplot(gs[0, c], projection="3d")
            _plot_panel_3d(
                ax,
                row,
                subcortical_labels=subcortical_labels,
                tract_metadata=tract_meta,
                color_by="tissue",
                yeo_by_roi=None,
                mesulam_by_roi=None,
                g1_norm=g1_norm,
                show_xlabel=True,
                show_ylabel=True,
                show_zlabel=True,
                show_title=True,
                axis_label_fs=_GRADIENT_BY_FIGURE_AXIS_LABEL_FS_3D,
                axis_tick_fs=_GRADIENT_BY_FIGURE_AXIS_TICK_FS_3D,
                draw_tissue_class_centroids=False,
            )
        fig.tight_layout(
            pad=0.4,
            h_pad=0.5,
            w_pad=0.4,
            rect=[0.0, 0.04, 1.0, 0.96 if cohort_tag else 0.99],
        )
        if cohort_tag:
            fig.suptitle(
                f"{_figure_method_label(method_tag)} G2, G3, G1 by tissue (face color = G1) | "
                f"cohort={cohort_tag} (3D)",
                fontsize=_GRADIENT_BY_FIGURE_SUPTITLE_FS,
                color=GRADIENT_AXIS_LABEL_COLOR,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _add_factor_pair_header_row(
    fig: plt.Figure,
    gs: object,
    results: list[GradientRunRow],
    *,
    data_col_offset: int = 0,
) -> None:
    """Header row (``gs`` row 0): one axes per factor (``gs`` has **one column per factor**).

    Each factor's two data panels live in a nested sub-gridspec below; the header cell
    ``gs[0, data_col_offset + f]`` spans that factor's full width.

    ``data_col_offset`` is the first outer-column index for Factor 1 (``0`` when there
    is no left legend column).
    """
    for f, row in enumerate(results):
        ax_h = fig.add_subplot(gs[0, data_col_offset + f])
        ax_h.set_axis_off()
        ax_h.text(
            0.5,
            0.98,
            _factor_display_name(row[0]),
            ha="center",
            va="top",
            transform=ax_h.transAxes,
            fontsize=_GRADIENT_BY_FIGURE_COLUMN_HEADER_FS,
            color=GRADIENT_AXIS_LABEL_COLOR,
        )


def _paint_tissue_by_factor_gridspec(
    fig: plt.Figure,
    gs: object,
    results: list[GradientRunRow],
    *,
    dims: Dims,
    root: Path,
) -> list[plt.Axes]:
    """Paint the data row (row index ``1``) of a ``(2, n_factors)`` gridspec for tissue encoding.

    Expects row ``0`` to already contain factor headers. Inner columns use equal width
    (regions | tissue-class centroids); does not run ``tight_layout``.

    Returns 2D tissue scatter axes (empty for 3D).
    """
    subcortical_labels = subcortical_grey_matter_column_names(root)
    tract_meta = load_hcp1065_tract_metadata(root)
    tissue_axes: list[plt.Axes] = []

    if dims == 2:
        axis_label_fs = _GRADIENT_BY_FIGURE_AXIS_LABEL_FS_2D
        axis_tick_fs = _GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D
        for f, row in enumerate(results):
            inner = GridSpecFromSubplotSpec(
                1,
                2,
                subplot_spec=gs[1, f],
                wspace=_GRADIENT_BY_FACTOR_WSPACE_INNER,
            )
            ax_t = fig.add_subplot(inner[0, 0])
            ax_tc = fig.add_subplot(inner[0, 1], sharex=ax_t, sharey=ax_t)
            ax_tc.tick_params(labelleft=False)
            _plot_panel_2d(
                ax_t, row,
                subcortical_labels=subcortical_labels,
                tract_metadata=tract_meta,
                color_by="tissue",
                yeo_by_roi=None, mesulam_by_roi=None,
                show_xlabel=True, show_ylabel=True, show_title=False,
                axis_label_fs=axis_label_fs,
                axis_tick_fs=axis_tick_fs,
                draw_tissue_class_centroids=False,
            )
            _plot_panel_2d(
                ax_tc, row,
                subcortical_labels=subcortical_labels,
                tract_metadata=tract_meta,
                color_by="tissue",
                yeo_by_roi=None, mesulam_by_roi=None,
                show_xlabel=True, show_ylabel=False, show_title=False,
                axis_label_fs=axis_label_fs,
                axis_tick_fs=axis_tick_fs,
                draw_tissue_marker_points=False,
                draw_tissue_class_centroids=True,
                tissue_centroids_use_legend_style=True,
            )
            ax_t.set_title(
                "Regions", fontsize=axis_label_fs, color=GRADIENT_AXIS_LABEL_COLOR
            )
            ax_tc.set_title(
                "Tissue class centroids",
                fontsize=axis_label_fs,
                color=GRADIENT_AXIS_LABEL_COLOR,
            )
            tissue_axes.extend([ax_t, ax_tc])
        return tissue_axes
    else:
        glasser_names = glasser_parcel_name_set(root)
        axis_label_fs_3d = _GRADIENT_BY_FIGURE_AXIS_LABEL_FS_3D
        axis_tick_fs_3d = _GRADIENT_BY_FIGURE_AXIS_TICK_FS_3D
        for f, row in enumerate(results):
            inner = GridSpecFromSubplotSpec(
                1,
                2,
                subplot_spec=gs[1, f],
                wspace=_GRADIENT_BY_FACTOR_WSPACE_INNER,
            )
            ax_t = fig.add_subplot(inner[0, 0], projection="3d")
            ax_tc = fig.add_subplot(inner[0, 1], projection="3d")
            _plot_panel_3d(
                ax_t, row,
                subcortical_labels=subcortical_labels,
                tract_metadata=tract_meta,
                color_by="tissue",
                yeo_by_roi=None, mesulam_by_roi=None,
                g1_norm=None,
                show_xlabel=True, show_ylabel=True, show_zlabel=True,
                show_title=False,
                axis_label_fs=axis_label_fs_3d,
                axis_tick_fs=axis_tick_fs_3d,
                color_face_by_gradient1=False,
                glasser_parcel_names=glasser_names,
                draw_tissue_class_centroids=False,
            )
            _plot_panel_3d(
                ax_tc, row,
                subcortical_labels=subcortical_labels,
                tract_metadata=tract_meta,
                color_by="tissue",
                yeo_by_roi=None, mesulam_by_roi=None,
                g1_norm=None,
                show_xlabel=True, show_ylabel=False, show_zlabel=False,
                show_title=False,
                axis_label_fs=axis_label_fs_3d,
                axis_tick_fs=axis_tick_fs_3d,
                color_face_by_gradient1=False,
                glasser_parcel_names=glasser_names,
                draw_tissue_marker_points=False,
                draw_tissue_class_centroids=True,
                tissue_centroids_use_legend_style=True,
            )
            masks, _g1v, g2v, g3v, regions = _build_tissue_masks_for_row(
                row,
                subcortical_labels=subcortical_labels,
                tract_metadata=tract_meta,
                k=3,
            )
            xv, yv, zv = g2v, g3v, _g1v
            _apply_shared_3d_lims([ax_t, ax_tc], xv, yv, zv)
            ax_t.set_title(
                "Regions",
                fontsize=axis_label_fs_3d,
                color=GRADIENT_AXIS_LABEL_COLOR,
            )
            ax_tc.set_title(
                "Tissue class centroids",
                fontsize=axis_label_fs_3d,
                color=GRADIENT_AXIS_LABEL_COLOR,
            )
    return tissue_axes


def plot_gradient_by_tissue_scatter(
    results: list[GradientRunRow],
    out_path: Path,
    *,
    dims: Dims = 2,
    tractometry_root: Path | None = None,
    method_tag: str = "laplacian_eigenmodes",
    cohort_tag: str | None = "controls",
) -> Path:
    """One data row: tissue markers + tissue class centroids per factor (factors as column pairs).

    Tissue encoding legend is saved separately as ``legend-tissue.png`` (no in-figure legend).
    """
    n = len(results)
    if n == 0:
        return out_path
    root = tractometry_root if tractometry_root is not None else DEFAULT_TRACTOMETRY_ROOT

    col_w = 6.2
    row_h = 5.75
    header_row_ratio = 0.09
    fig_w_total = col_w * (2 * n) + 0.8
    fig_h_total = row_h * (header_row_ratio + 1.0) + 0.45
    fig = plt.figure(figsize=(fig_w_total, fig_h_total))
    gs = fig.add_gridspec(
        2,
        n,
        width_ratios=[1.0] * n,
        height_ratios=[header_row_ratio, 1.0],
        hspace=0.14,
        wspace=_GRADIENT_BY_FACTOR_WSPACE_OUTER,
    )
    _add_factor_pair_header_row(fig, gs, results, data_col_offset=0)
    tissue_axes = _paint_tissue_by_factor_gridspec(fig, gs, results, dims=dims, root=root)

    if dims == 2:
        fig.tight_layout(
            pad=0.35, h_pad=0.45, w_pad=0.35,
            rect=[0.0, 0.04, 1.0, 1.0],
        )
        _equalize_tissue_scatter_axes_boxes(tissue_axes)
    else:
        fig.tight_layout(
            pad=0.35, h_pad=0.45, w_pad=0.4,
            rect=[0.0, 0.04, 1.0, 1.0],
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_gradient_by_yeo_mesulam_scatter(
    results: list[GradientRunRow],
    out_path: Path,
    *,
    dims: Dims = 2,
    tractometry_root: Path | None = None,
    method_tag: str = "laplacian_eigenmodes",
    cohort_tag: str | None = "controls",
) -> Path:
    """Two data rows: Yeo (Glasser cortex) + Mesulam (Glasser cortex), markers and centroids per factor.

    Yeo and Mesulam legends are saved as ``legend-yeo.png`` and ``legend-mesulam.png`` (no in-figure legends).
    """
    n = len(results)
    if n == 0:
        return out_path
    root = tractometry_root if tractometry_root is not None else DEFAULT_TRACTOMETRY_ROOT
    subcortical_labels = subcortical_grey_matter_column_names(root)
    tract_meta = load_hcp1065_tract_metadata(root)
    yeo_by = load_yeo_labels(root)
    mesu_by = load_mesulam_labels(root)
    glasser_names = glasser_parcel_name_set(root)

    col_w = 6.2
    row_h = 5.75
    header_row_ratio = 0.09
    fig_w_total = col_w * (2 * n) + 0.8
    fig_h_total = row_h * (header_row_ratio + 2.0) + 0.45
    fig = plt.figure(figsize=(fig_w_total, fig_h_total))
    gs = fig.add_gridspec(
        3,
        n,
        width_ratios=[1.0] * n,
        height_ratios=[header_row_ratio, 1.0, 1.0],
        hspace=0.14,
        wspace=_GRADIENT_BY_FACTOR_WSPACE_OUTER,
    )
    _add_factor_pair_header_row(fig, gs, results, data_col_offset=0)

    if dims == 2:
        axis_label_fs = _GRADIENT_BY_FIGURE_AXIS_LABEL_FS_2D
        axis_tick_fs = _GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D
        for f, row in enumerate(results):
            masks, g1v, g2v, _g3v, regions = _build_tissue_masks_for_row(
                row,
                subcortical_labels=subcortical_labels,
                tract_metadata=tract_meta,
                k=2,
            )
            xv, yv = g2v, g1v
            gc_glasser = masks["cort_gm"] & _in_glasser_parcel_mask(
                regions, glasser_names
            )

            inner_y = GridSpecFromSubplotSpec(
                1,
                2,
                subplot_spec=gs[1, f],
                wspace=_GRADIENT_BY_FACTOR_WSPACE_INNER,
            )
            inner_m = GridSpecFromSubplotSpec(
                1,
                2,
                subplot_spec=gs[2, f],
                wspace=_GRADIENT_BY_FACTOR_WSPACE_INNER,
            )
            ax_y = fig.add_subplot(inner_y[0, 0])
            ax_yc = fig.add_subplot(inner_y[0, 1], sharex=ax_y, sharey=ax_y)
            ax_m = fig.add_subplot(inner_m[0, 0])
            ax_mc = fig.add_subplot(inner_m[0, 1], sharex=ax_m, sharey=ax_m)
            ax_yc.tick_params(labelleft=False)
            ax_mc.tick_params(labelleft=False)

            _plot_panel_2d(
                ax_y, row,
                subcortical_labels=subcortical_labels,
                tract_metadata=tract_meta,
                color_by="yeo",
                yeo_by_roi=yeo_by, mesulam_by_roi=None,
                show_xlabel=False, show_ylabel=False, show_title=False,
                glasser_parcel_names=glasser_names,
                axis_label_fs=axis_label_fs,
                axis_tick_fs=axis_tick_fs,
            )
            _plot_panel_2d_glasser_community_centroids(
                ax_yc,
                row,
                regions=regions,
                xv=xv,
                yv=yv,
                gc=gc_glasser,
                label_map=yeo_by,
                color_fn=yeo_network_color,
                show_xlabel=False,
                show_ylabel=False,
                show_title=False,
                axis_label_fs=axis_label_fs,
                axis_tick_fs=axis_tick_fs,
            )
            _plot_panel_2d(
                ax_m, row,
                subcortical_labels=subcortical_labels,
                tract_metadata=tract_meta,
                color_by="mesulam",
                yeo_by_roi=None, mesulam_by_roi=mesu_by,
                show_xlabel=True, show_ylabel=False, show_title=False,
                glasser_parcel_names=glasser_names,
                draw_tissue_class_centroids=False,
                axis_label_fs=axis_label_fs,
                axis_tick_fs=axis_tick_fs,
            )
            _plot_panel_2d_glasser_community_centroids(
                ax_mc,
                row,
                regions=regions,
                xv=xv,
                yv=yv,
                gc=gc_glasser,
                label_map=mesu_by,
                color_fn=mesulam_type_color,
                show_xlabel=True,
                show_ylabel=False,
                show_title=False,
                axis_label_fs=axis_label_fs,
                axis_tick_fs=axis_tick_fs,
            )
            ax_y.set_title("Cortical regions", fontsize=axis_label_fs, color=GRADIENT_AXIS_LABEL_COLOR)
            ax_yc.set_title(
                "Community centroids", fontsize=axis_label_fs, color=GRADIENT_AXIS_LABEL_COLOR
            )

        fig.tight_layout(
            pad=0.35, h_pad=0.45, w_pad=0.35,
            rect=[0.0, 0.04, 1.0, 1.0],
        )
    else:
        axis_label_fs_3d = _GRADIENT_BY_FIGURE_AXIS_LABEL_FS_3D
        axis_tick_fs_3d = _GRADIENT_BY_FIGURE_AXIS_TICK_FS_3D
        for f, row in enumerate(results):
            inner_y = GridSpecFromSubplotSpec(
                1,
                2,
                subplot_spec=gs[1, f],
                wspace=_GRADIENT_BY_FACTOR_WSPACE_INNER,
            )
            inner_m = GridSpecFromSubplotSpec(
                1,
                2,
                subplot_spec=gs[2, f],
                wspace=_GRADIENT_BY_FACTOR_WSPACE_INNER,
            )
            ax_y = fig.add_subplot(inner_y[0, 0], projection="3d")
            ax_yc = fig.add_subplot(inner_y[0, 1], projection="3d")
            ax_m = fig.add_subplot(inner_m[0, 0], projection="3d")
            ax_mc = fig.add_subplot(inner_m[0, 1], projection="3d")
            _plot_panel_3d(
                ax_y, row,
                subcortical_labels=subcortical_labels,
                tract_metadata=tract_meta,
                color_by="yeo",
                yeo_by_roi=yeo_by, mesulam_by_roi=None,
                g1_norm=None,
                show_xlabel=True, show_ylabel=True, show_zlabel=False,
                show_title=False,
                axis_label_fs=axis_label_fs_3d,
                axis_tick_fs=axis_tick_fs_3d,
                color_face_by_gradient1=False,
                glasser_parcel_names=glasser_names,
            )
            _plot_panel_3d(
                ax_m, row,
                subcortical_labels=subcortical_labels,
                tract_metadata=tract_meta,
                color_by="mesulam",
                yeo_by_roi=None, mesulam_by_roi=mesu_by,
                g1_norm=None,
                show_xlabel=True, show_ylabel=True, show_zlabel=False,
                show_title=False,
                axis_label_fs=axis_label_fs_3d,
                axis_tick_fs=axis_tick_fs_3d,
                color_face_by_gradient1=False,
                glasser_parcel_names=glasser_names,
            )
            masks, _g1v, g2v, g3v, regions = _build_tissue_masks_for_row(
                row,
                subcortical_labels=subcortical_labels,
                tract_metadata=tract_meta,
                k=3,
            )
            xv, yv, zv = g2v, g3v, _g1v
            gc_glasser = masks["cort_gm"] & _in_glasser_parcel_mask(
                regions, glasser_names
            )
            _plot_panel_3d_glasser_community_centroids(
                ax_yc,
                row,
                regions=regions,
                xv=xv,
                yv=yv,
                zv=zv,
                gc=gc_glasser,
                label_map=yeo_by,
                color_fn=yeo_network_color,
                show_xlabel=True,
                show_ylabel=False,
                show_zlabel=False,
                show_title=False,
                axis_label_fs=axis_label_fs_3d,
                axis_tick_fs=axis_tick_fs_3d,
            )
            _plot_panel_3d_glasser_community_centroids(
                ax_mc,
                row,
                regions=regions,
                xv=xv,
                yv=yv,
                zv=zv,
                gc=gc_glasser,
                label_map=mesu_by,
                color_fn=mesulam_type_color,
                show_xlabel=True,
                show_ylabel=False,
                show_zlabel=False,
                show_title=False,
                axis_label_fs=axis_label_fs_3d,
                axis_tick_fs=axis_tick_fs_3d,
            )
            if np.any(gc_glasser):
                gxv, gyv, gzv = xv[gc_glasser], yv[gc_glasser], zv[gc_glasser]
                _apply_shared_3d_lims([ax_y, ax_yc], gxv, gyv, gzv)
                _apply_shared_3d_lims([ax_m, ax_mc], gxv, gyv, gzv)
            else:
                _apply_shared_3d_lims([ax_y, ax_yc], xv, yv, zv)
                _apply_shared_3d_lims([ax_m, ax_mc], xv, yv, zv)
            ax_y.set_title("Cortical regions", fontsize=axis_label_fs_3d, color=GRADIENT_AXIS_LABEL_COLOR)
            ax_yc.set_title(
                "Community centroids", fontsize=axis_label_fs_3d, color=GRADIENT_AXIS_LABEL_COLOR
            )

        fig.tight_layout(
            pad=0.35, h_pad=0.45, w_pad=0.4,
            rect=[0.0, 0.04, 1.0, 1.0],
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_gradient_summary(
    results: list[GradientRunRow],
    out_path: Path,
    *,
    gradient_index: int,
    dims: Dims = 2,
    tractometry_root: Path | None = None,
    method_tag: str = "laplacian_eigenmodes",
    cohort_tag: str | None = "controls",
) -> Path:
    """Stack tissue scatter (row 1) and region-group / axis bars for one gradient (row 2).

    Row 1 matches ``gradients_by-tissue`` (tissue-colored regions vs centroids; factor
    titles above each column). Row 2 matches ``gradient{K}_by-groups-axes`` for the
    chosen ``gradient_index`` (``K = gradient_index + 1``): region-group bars and
    neuroaxis Pearson bars, without repeating factor titles.

    ``dims`` controls the tissue half only (2D vs 3D); the bar half is always 2D.
    ``method_tag`` / ``cohort_tag`` are accepted for API parity with other emitters.
    """
    from .plots_bars import (
        _paint_groups_axes_bars_data_row,
        _reflow_bar_pair_axes,
    )

    _ = (method_tag, cohort_tag)

    if gradient_index < 0:
        raise ValueError(f"gradient_index must be >= 0; got {gradient_index}")

    n = len(results)
    if n == 0:
        return out_path
    root = tractometry_root if tractometry_root is not None else DEFAULT_TRACTOMETRY_ROOT

    header_row_ratio = 0.09
    row_h = 5.75
    col_w_tissue = 6.2
    # Width follows the tissue row (two equal panels per factor), not the bar row.
    fig_w_total = col_w_tissue * (2 * n) + 0.8
    fig_h_total = row_h * (header_row_ratio + 1.0) + row_h * 1.0 + 0.9
    h_top = header_row_ratio + 1.0
    h_bot = 1.0

    with figure_font_context():
        fig = plt.figure(figsize=(fig_w_total, fig_h_total))
        outer = fig.add_gridspec(2, 1, height_ratios=[h_top, h_bot], hspace=0.32)
        gs_top = GridSpecFromSubplotSpec(
            2,
            n,
            subplot_spec=outer[0, 0],
            height_ratios=[header_row_ratio, 1.0],
            width_ratios=[1.0] * n,
            hspace=0.14,
            wspace=_GRADIENT_SUMMARY_FACTOR_WSPACE_OUTER,
        )
        gs_bot = GridSpecFromSubplotSpec(
            1,
            n,
            subplot_spec=outer[1, 0],
            width_ratios=[1.0] * n,
            wspace=_GRADIENT_SUMMARY_FACTOR_WSPACE_OUTER,
        )
        _add_factor_pair_header_row(fig, gs_top, results, data_col_offset=0)
        tissue_axes = _paint_tissue_by_factor_gridspec(
            fig, gs_top, results, dims=dims, root=root
        )
        _, bar_ax_pairs = _paint_groups_axes_bars_data_row(
            fig,
            gs_bot,
            results,
            gradient_index=gradient_index,
            tractometry_root=root,
            gs_data_row=0,
        )

        if dims == 2:
            fig.tight_layout(
                pad=0.35,
                h_pad=0.56,
                w_pad=0.35,
                rect=[0.0, 0.04, 1.0, 1.0],
            )
        else:
            fig.tight_layout(
                pad=0.35,
                h_pad=0.56,
                w_pad=0.4,
                rect=[0.0, 0.04, 1.0, 1.0],
            )

        for ax_rg, ax_axis in bar_ax_pairs:
            _reflow_bar_pair_axes(ax_rg, ax_axis)
        if dims == 2:
            _equalize_tissue_scatter_axes_boxes(tissue_axes)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return out_path


def plot_gradient1_summary(
    results: list[GradientRunRow],
    out_path: Path,
    *,
    dims: Dims = 2,
    tractometry_root: Path | None = None,
    method_tag: str = "laplacian_eigenmodes",
    cohort_tag: str | None = "controls",
) -> Path:
    """Same as :func:`plot_gradient_summary` with ``gradient_index=0`` (gradient 1 bars)."""
    return plot_gradient_summary(
        results,
        out_path,
        gradient_index=0,
        dims=dims,
        tractometry_root=tractometry_root,
        method_tag=method_tag,
        cohort_tag=cohort_tag,
    )
