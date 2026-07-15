"""G2 vs G1 scatters colored by epilepsy group-mean factor z-scores."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .config import FACTOR_PANEL_LABELS
from .gc_imports import _load_gc_module, gc_plots_scatter
from .io import TleZScatterRow
from .roi_markers import (
    DEFAULT_ROI_MARKERS,
    ROI_MARKER_EDGE_COLOR,
    ROI_MARKER_BOX_EDGE_WIDTH,
    ROI_MARKER_EDGE_WIDTH,
    ROI_MARKER_LEFT,
    ROI_MARKER_RIGHT,
    ROI_MARKER_SIZE_PT2,
    RoiMarkerSpec,
    add_roi_marker_boxes,
    add_roi_marker_lr_triangles,
)

_gc = gc_plots_scatter()
_gc_config = _load_gc_module("config")
GRADIENT_AXIS_LABEL_COLOR = _gc_config.GRADIENT_AXIS_LABEL_COLOR
_print_colorbar_range = _gc._print_colorbar_range
_set_xy_ticks_tissue_panel = _gc._set_xy_ticks_tissue_panel
_hide_top_right_spines = _gc._hide_top_right_spines
_GRADIENT_BY_FIGURE_AXIS_LABEL_FS_2D = _gc._GRADIENT_BY_FIGURE_AXIS_LABEL_FS_2D
_GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D = _gc._GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D

ColorMode = Literal["signed", "absolute"]

# Text in ``gradients_by-tle-z_*`` scatter figures only (legends unchanged).
_SCATTER_TEXT_SCALE = 0.75

COLORBAR_LABEL = "Factor z-scores"


def _finite_panel_arrays(
    row: TleZScatterRow,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, g1, g2, z_color = row
    idx = g1.index.intersection(g2.index).intersection(z_color.index)
    xv = g2.reindex(idx).to_numpy(dtype=np.float64)
    yv = g1.reindex(idx).to_numpy(dtype=np.float64)
    cv = z_color.reindex(idx).to_numpy(dtype=np.float64)
    m = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(cv)
    return xv[m], yv[m], cv[m]


def _shared_signed_z_norm(rows: list[TleZScatterRow]) -> mcolors.Normalize:
    parts: list[np.ndarray] = []
    for row in rows:
        _, _, cv = _finite_panel_arrays(row)
        if cv.size:
            parts.append(cv)
    if not parts:
        return mcolors.Normalize(-1.0, 1.0)
    c = np.concatenate(parts)
    abs_max = float(max(abs(np.min(c)), abs(np.max(c))))
    if not np.isfinite(abs_max) or abs_max < 1e-15:
        abs_max = 1.0
    return mcolors.Normalize(-abs_max, abs_max)


def _shared_absolute_z_norm(rows: list[TleZScatterRow]) -> mcolors.Normalize:
    parts: list[np.ndarray] = []
    for row in rows:
        _, _, cv = _finite_panel_arrays(row)
        if cv.size:
            parts.append(cv)
    if not parts:
        return mcolors.Normalize(0.0, 1.0)
    c = np.concatenate(parts)
    vmax = float(np.max(c))
    if not np.isfinite(vmax) or vmax < 1e-15:
        vmax = 1.0
    return mcolors.Normalize(0.0, vmax)


def _shared_z_norm(
    rows: list[TleZScatterRow],
    *,
    color_mode: ColorMode,
) -> mcolors.Normalize:
    if color_mode == "signed":
        return _shared_signed_z_norm(rows)
    return _shared_absolute_z_norm(rows)


def _factor_panel_title(factor_tag: str) -> str:
    return FACTOR_PANEL_LABELS.get(str(factor_tag).strip(), str(factor_tag))


def _cmap_name(color_mode: ColorMode) -> str:
    return "RdBu_r" if color_mode == "signed" else "Reds"


def _add_z_colorbar_horizontal(
    fig: plt.Figure,
    cax: plt.Axes,
    norm: mcolors.Normalize,
    *,
    color_mode: ColorMode,
    label_fontsize: float | None = None,
    tick_labelsize: float | None = None,
) -> None:
    sm = cm.ScalarMappable(norm=norm, cmap=_cmap_name(color_mode))
    sm.set_array([])
    ls = label_fontsize if label_fontsize is not None else 15.0
    ts = tick_labelsize if tick_labelsize is not None else 13.0
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label(COLORBAR_LABEL, fontsize=ls, labelpad=6)
    cbar.ax.tick_params(axis="x", which="both", labelsize=ts, pad=2)


def save_standalone_legend_tle_z(
    rows: list[TleZScatterRow],
    out_path: Path,
    *,
    color_mode: ColorMode,
) -> Path:
    """Horizontal colorbar for factor z-score coloring (shared norm across panels)."""
    if len(rows) == 0:
        return out_path
    norm = _shared_z_norm(rows, color_mode=color_mode)
    _print_colorbar_range(
        out_path,
        norm,
        quantity=f"Factor z-scores ({color_mode}, standalone legend)",
    )
    fig = plt.figure(figsize=(6.4, 0.72), facecolor="white")
    cax = fig.add_axes([0.164, 0.06, 0.672, 0.34])
    _add_z_colorbar_horizontal(fig, cax, norm, color_mode=color_mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _roi_marker_legend_handles(
    markers: Sequence[RoiMarkerSpec] = DEFAULT_ROI_MARKERS,
) -> list[Line2D]:
    ms = float(ROI_MARKER_SIZE_PT2) ** 0.5
    return [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=spec.color,
            markeredgecolor=ROI_MARKER_EDGE_COLOR,
            markeredgewidth=ROI_MARKER_BOX_EDGE_WIDTH,
            markersize=ms,
            linestyle="None",
            label=spec.label,
        )
        for spec in markers
    ]


def save_standalone_legend_roi_markers_lr(
    out_path: Path,
    *,
    markers: Sequence[RoiMarkerSpec] = DEFAULT_ROI_MARKERS,
) -> Path:
    """Legend for L/R triangle markers plus ROI colors."""
    ms = float(ROI_MARKER_SIZE_PT2) ** 0.5
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker=ROI_MARKER_LEFT,
            color="none",
            markerfacecolor="0.55",
            markeredgecolor=ROI_MARKER_EDGE_COLOR,
            markeredgewidth=ROI_MARKER_EDGE_WIDTH,
            markersize=ms,
            linestyle="None",
            label="Left ROI",
        ),
        Line2D(
            [0],
            [0],
            marker=ROI_MARKER_RIGHT,
            color="none",
            markerfacecolor="0.55",
            markeredgecolor=ROI_MARKER_EDGE_COLOR,
            markeredgewidth=ROI_MARKER_EDGE_WIDTH,
            markersize=ms,
            linestyle="None",
            label="Right ROI",
        ),
    ]
    fig = plt.figure(figsize=(9.6, 1.55), facecolor="white")
    ax_top = fig.add_axes([0.02, 0.52, 0.96, 0.42])
    ax_bot = fig.add_axes([0.02, 0.06, 0.96, 0.42])
    ax_top.axis("off")
    ax_bot.axis("off")
    ax_top.legend(
        handles=shape_handles,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=13,
        handlelength=1.2,
        handletextpad=0.55,
        columnspacing=2.0,
    )
    ax_bot.legend(
        handles=_roi_marker_legend_handles(markers),
        loc="center",
        ncol=len(markers),
        frameon=False,
        fontsize=13,
        handlelength=1.2,
        handletextpad=0.55,
        columnspacing=1.4,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def save_standalone_legend_roi_markers(
    out_path: Path,
    *,
    markers: Sequence[RoiMarkerSpec] = DEFAULT_ROI_MARKERS,
) -> Path:
    """Horizontal legend for fixed-color bilateral ROI centroid markers."""
    fig = plt.figure(figsize=(9.2, 1.15), facecolor="white")
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.84])
    ax.axis("off")
    ax.legend(
        handles=_roi_marker_legend_handles(markers),
        loc="center",
        ncol=len(markers),
        frameon=False,
        fontsize=13,
        handlelength=1.2,
        handletextpad=0.55,
        columnspacing=1.4,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def paint_tle_z_scatter_ax(
    ax: plt.Axes,
    row: TleZScatterRow,
    *,
    color_mode: ColorMode,
    z_norm: mcolors.Normalize,
    axis_label_fs: float,
    axis_tick_fs: float,
    show_ylabel: bool = True,
    hide_top_right: bool = False,
    title: str | None = None,
    roi_markers: Sequence[RoiMarkerSpec] | None = None,
    roi_markers_lr: bool = False,
) -> None:
    """Paint one G2-vs-G1 panel colored by epilepsy group-mean factor z-scores."""
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
    xv, yv, cv = _finite_panel_arrays(row)
    ax.scatter(
        xv,
        yv,
        c=cv,
        cmap=_cmap_name(color_mode),
        norm=z_norm,
        alpha=0.78,
        s=26.0,
        edgecolors="none",
    )
    if roi_markers is not None:
        add_roi_marker_boxes(ax, row, markers=roi_markers)
    if roi_markers_lr:
        lr_markers = (
            roi_markers if roi_markers is not None else DEFAULT_ROI_MARKERS
        )
        add_roi_marker_lr_triangles(ax, row, markers=lr_markers)
    ax.set_title(
        _factor_panel_title(row[0]) if title is None else title,
        fontsize=axis_label_fs,
        color=GRADIENT_AXIS_LABEL_COLOR,
    )
    ax.set_xlabel("Gradient 2", **xlab_kw)
    if show_ylabel:
        ax.set_ylabel("Gradient 1", **ylab_kw)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    _set_xy_ticks_tissue_panel(
        ax,
        tick_label_pad_x=6.0,
        tick_labelsize=axis_tick_fs,
    )
    if hide_top_right:
        _hide_top_right_spines(ax)


def plot_gradients_by_tle_z_scatter(
    rows: list[TleZScatterRow],
    out_path: Path,
    *,
    color_mode: ColorMode,
    roi_markers: Sequence[RoiMarkerSpec] | None = None,
    roi_markers_lr: bool = False,
) -> Path:
    """G2 vs G1 panels colored by epilepsy group-mean factor z-scores."""
    n = len(rows)
    if n == 0:
        return out_path

    z_norm = _shared_z_norm(rows, color_mode=color_mode)
    _print_colorbar_range(
        out_path,
        z_norm,
        quantity=f"Epilepsy z ({color_mode}, 2D scatter)",
    )

    col_w = 6.2
    row_h = 5.75
    axis_label_fs = _GRADIENT_BY_FIGURE_AXIS_LABEL_FS_2D * _SCATTER_TEXT_SCALE
    axis_tick_fs = _GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D * _SCATTER_TEXT_SCALE

    fig = plt.figure(figsize=(col_w * n + 0.6, row_h))
    gs = fig.add_gridspec(1, n, wspace=0.18)
    for c, row in enumerate(rows):
        ax = fig.add_subplot(gs[0, c])
        paint_tle_z_scatter_ax(
            ax,
            row,
            color_mode=color_mode,
            z_norm=z_norm,
            axis_label_fs=axis_label_fs,
            axis_tick_fs=axis_tick_fs,
            show_ylabel=(c == 0),
            roi_markers=roi_markers,
            roi_markers_lr=roi_markers_lr,
        )

    fig.tight_layout(
        pad=0.35,
        h_pad=0.35,
        w_pad=0.38,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
