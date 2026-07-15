"""Combined 1x6 summary: signed ROI-marker scatters interleaved with correlation lollipops."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec

from .correlations import symmetric_ylim
from .gc_imports import _load_gc_module
from .io import TleZScatterRow
from .plots_lollipop import (
    _lollipop_panel_width_inches,
    _plot_factor_lollipop_ax,
)
from .plots_scatter import (
    ColorMode,
    _shared_z_norm,
    paint_tle_z_scatter_ax,
)
from .roi_markers import DEFAULT_ROI_MARKERS, RoiMarkerSpec

_gc_scatter = _load_gc_module("plots_scatter")
_gc_figure_style = _load_gc_module("figure_style")
_GRADIENT_BY_FIGURE_AXIS_LABEL_FS_2D = _gc_scatter._GRADIENT_BY_FIGURE_AXIS_LABEL_FS_2D
_GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D = _gc_scatter._GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D
figure_font_context = _gc_figure_style.figure_font_context

# Match group-controls tissue panel sizing (row_h=5.75; one scatter ~ half of 6.2 pair).
_ROW_HEIGHT_IN = 5.75
_SCATTER_PANEL_WIDTH_IN = 5.5
_SUMMARY_DPI = 200
# Nested spacing: within-factor (scatter|lollipop) < between-factor pairs.
_SUMMARY_WSPACE_INNER = 0.32
_SUMMARY_WSPACE_OUTER = 0.32

_SCATTER_TITLE = "Regions"
_LOLLIPOP_TITLE = "Factor z-score axis"
_LOLLIPOP_YLABEL = "Pearson's $r$"


def _align_panel_axes_boxes(
    scatter_axes: list[plt.Axes],
    lollipop_axes: list[plt.Axes],
) -> None:
    """Match y-spine length/placement across panels; keep scatters square."""
    axes = list(scatter_axes) + list(lollipop_axes)
    if not axes:
        return
    fig = axes[0].figure
    fig_w, fig_h = fig.get_size_inches()
    positions = [ax.get_position() for ax in axes]

    # Shared vertical span for every y-axis spine.
    heights_in = [p.height * fig_h for p in positions]
    scatter_widths_in = [p.width * fig_w for p in positions[: len(scatter_axes)]]
    # Square scatter side also sets the common spine height.
    side_in = min(min(scatter_widths_in), min(heights_in))
    shared_h = side_in / fig_h
    # Bottom-align spines on the lowest current baseline among panels.
    shared_y0 = min(p.y0 for p in positions)

    for ax, box in zip(scatter_axes, positions[: len(scatter_axes)]):
        side_w = side_in / fig_w
        # Keep left edge so the y-spine stays where the subplot slot put it.
        ax.set_position([box.x0, shared_y0, side_w, shared_h])
        ax.set_aspect("auto")

    for ax, box in zip(lollipop_axes, positions[len(scatter_axes) :]):
        ax.set_position([box.x0, shared_y0, box.width, shared_h])
        ax.set_aspect("auto")


def plot_gradients_by_tle_z_summary(
    rows: list[TleZScatterRow],
    corr_df: pd.DataFrame,
    out_path: Path,
    *,
    color_mode: ColorMode = "signed",
    roi_markers: tuple[RoiMarkerSpec, ...] = DEFAULT_ROI_MARKERS,
) -> Path:
    """1x6 figure: for each factor, signed ROI-marker scatter then correlation lollipop.

    Panel order: F1 scatter | F1 lollipop | F2 scatter | F2 lollipop | F3 scatter | F3 lollipop.
    Scatter panels are square; fonts match group-controls summary (labels 22 pt, ticks 16 pt).
    """
    if not rows:
        return out_path

    n_factors = len(rows)
    n_lollipops = int(corr_df.loc[corr_df["factor"] == rows[0][0], "comparison"].nunique())
    if n_lollipops <= 0:
        n_lollipops = 5
    scatter_w = _SCATTER_PANEL_WIDTH_IN
    lollipop_w = _lollipop_panel_width_inches(n_lollipops)
    pair_w = scatter_w + lollipop_w
    # Extra figure width so nested outer/inner gaps have room to show.
    fig_w = pair_w * n_factors + 3.2

    z_norm = _shared_z_norm(rows, color_mode=color_mode)
    ylim = symmetric_ylim(corr_df["pearson_r"])
    axis_label_fs = _GRADIENT_BY_FIGURE_AXIS_LABEL_FS_2D
    axis_tick_fs = _GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D

    with figure_font_context():
        fig = plt.figure(figsize=(fig_w, _ROW_HEIGHT_IN))
        outer = fig.add_gridspec(
            1,
            n_factors,
            wspace=_SUMMARY_WSPACE_OUTER,
            left=0.04,
            right=0.995,
            bottom=0.16,
            top=0.90,
        )
        scatter_axes: list[plt.Axes] = []
        lollipop_axes: list[plt.Axes] = []
        for i, row in enumerate(rows):
            factor_tag = row[0]
            inner = GridSpecFromSubplotSpec(
                1,
                2,
                subplot_spec=outer[0, i],
                width_ratios=[scatter_w, lollipop_w],
                wspace=_SUMMARY_WSPACE_INNER,
            )
            ax_scatter = fig.add_subplot(inner[0, 0])
            ax_lollipop = fig.add_subplot(inner[0, 1])
            scatter_axes.append(ax_scatter)
            lollipop_axes.append(ax_lollipop)

            paint_tle_z_scatter_ax(
                ax_scatter,
                row,
                color_mode=color_mode,
                z_norm=z_norm,
                axis_label_fs=axis_label_fs,
                axis_tick_fs=axis_tick_fs,
                show_ylabel=True,
                hide_top_right=True,
                title=_SCATTER_TITLE,
                roi_markers=roi_markers,
            )
            ax_scatter.spines["top"].set_visible(False)
            ax_scatter.spines["right"].set_visible(False)

            factor_df = corr_df.loc[corr_df["factor"] == factor_tag].copy()
            _plot_factor_lollipop_ax(
                ax_lollipop,
                factor_df,
                factor_tag=factor_tag,
                ylim=ylim,
                show_ylabel=True,
                axis_tick_fs=axis_tick_fs,
                axis_label_fs=axis_label_fs,
                title=_LOLLIPOP_TITLE,
                ylabel=_LOLLIPOP_YLABEL,
            )

        # Nested GridSpec spacing is authoritative; skip tight_layout (flattens gaps).
        _align_panel_axes_boxes(scatter_axes, lollipop_axes)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            dpi=_SUMMARY_DPI,
            bbox_inches="tight",
            pad_inches=0.15,
            facecolor="white",
        )
        plt.close(fig)
    return out_path
