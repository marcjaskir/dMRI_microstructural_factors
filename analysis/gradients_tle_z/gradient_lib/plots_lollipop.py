"""Lollipop plots: factor z-score correlations with gradients and neuroaxis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec

from .config import FACTOR_PANEL_LABELS
from .correlations import COMPARISON_SPECS, symmetric_ylim
from .gc_imports import _load_gc_module, gc_plots_bars

_gc_scatter = _load_gc_module("plots_scatter")
_gc_figure_style = _load_gc_module("figure_style")
_GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D = _gc_scatter._GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D
figure_font_context = _gc_figure_style.figure_font_context

_gc_bars = gc_plots_bars()
_style_bar_axes = _gc_bars._style_bar_axes
_LOLLIPOP_COLOR = _gc_bars._NEUROAXIS_LOLLIPOP_COLOR
_LOLLIPOP_LINEWIDTH = _gc_bars._NEUROAXIS_LOLLIPOP_LINEWIDTH
_LOLLIPOP_MARKERSIZE = _gc_bars._NEUROAXIS_LOLLIPOP_MARKERSIZE
_LOLLIPOP_EDGEWIDTH = _gc_bars._NEUROAXIS_LOLLIPOP_EDGEWIDTH
_LOLLIPOP_X_MARGIN = _gc_bars._NEUROAXIS_LOLLIPOP_X_MARGIN
_gc_config = _load_gc_module("config")
GRADIENT_AXIS_LABEL_COLOR = _gc_config.GRADIENT_AXIS_LABEL_COLOR

# Width of the right-hand lollipop column in gradient-by-groups bar figures.
_REF_LOLLIPOP_PANEL_WIDTH_IN = 3.6  # col_w_unit * 1.0 / _BAR_PAIR_WIDTH_SUM
_REF_LOLLIPOP_N = 3
_ROW_HEIGHT_IN = 5.75
_LOLLIPOP_DPI = 400
# Horizontal gap between factor panels (wspace in GridSpec units).
_LOLLIPOP_FACTOR_WSPACE = 0.6
# Stem-to-stem pitch; edge margin ``_LOLLIPOP_X_MARGIN`` is unchanged.
_LOLLIPOP_X_SPACING = 0.32
# ``$r$`` renders as Georgia italic via ``GEORGIA_MATHTEXT_RCPARAMS`` in figure_font_context.
_YLABEL = "Factor z-score\nPearson's $r$"
_PEARSONS_R_YLABEL = "Pearson's $r$"


def _lollipop_x_positions(n_lollipops: int) -> np.ndarray:
    return np.arange(n_lollipops, dtype=np.float64) * _LOLLIPOP_X_SPACING


def _lollipop_x_span(n_lollipops: int) -> float:
    if n_lollipops <= 1:
        return 2.0 * _LOLLIPOP_X_MARGIN
    return (n_lollipops - 1) * _LOLLIPOP_X_SPACING + 2.0 * _LOLLIPOP_X_MARGIN


def _lollipop_panel_width_inches(n_lollipops: int) -> float:
    """Scale panel width with data span so edge buffers stay visually constant."""
    ref_span = (_REF_LOLLIPOP_N - 1) + 2.0 * _LOLLIPOP_X_MARGIN
    span = _lollipop_x_span(n_lollipops)
    return _REF_LOLLIPOP_PANEL_WIDTH_IN * (span / ref_span)


def _plot_factor_lollipop_ax(
    ax: plt.Axes,
    factor_df: pd.DataFrame,
    *,
    factor_tag: str,
    ylim: tuple[float, float],
    show_ylabel: bool,
    axis_tick_fs: float,
    axis_label_fs: float | None = None,
    title: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Five-stem lollipop styled like group-controls ``Gradient K axis`` panel."""
    label_fs = axis_tick_fs if axis_label_fs is None else axis_label_fs
    labels = [spec[0] for spec in COMPARISON_SPECS]
    label_to_r = dict(zip(factor_df["comparison"], factor_df["pearson_r"]))
    values = np.asarray(
        [float(label_to_r.get(lab, np.nan)) for lab in labels],
        dtype=np.float64,
    )

    n = len(labels)
    x = _lollipop_x_positions(n)
    ax.vlines(
        x,
        0.0,
        values,
        colors=_LOLLIPOP_COLOR,
        linewidth=_LOLLIPOP_LINEWIDTH,
        zorder=2,
    )
    ax.scatter(
        x,
        values,
        s=_LOLLIPOP_MARKERSIZE,
        c=_LOLLIPOP_COLOR,
        edgecolors=_LOLLIPOP_COLOR,
        linewidths=_LOLLIPOP_EDGEWIDTH,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        fontsize=axis_tick_fs,
    )
    ax.set_xlim(-_LOLLIPOP_X_MARGIN, x[-1] + _LOLLIPOP_X_MARGIN)
    ax.set_ylim(ylim)
    ax.margins(x=0, y=0)

    panel_title = (
        title if title is not None else FACTOR_PANEL_LABELS.get(factor_tag, factor_tag)
    )
    panel_ylabel = _YLABEL if ylabel is None else ylabel
    _style_bar_axes(
        ax,
        axis_label_fs=label_fs,
        axis_tick_fs=axis_tick_fs,
        show_ylabel=show_ylabel,
        ylabel=panel_ylabel,
        title=panel_title,
    )


def plot_factor_z_correlation_lollipop(
    corr_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """One row of lollipop panels (one per factor) comparing five predictors."""
    factors = list(dict.fromkeys(corr_df["factor"].tolist()))
    if not factors:
        raise ValueError("Correlation table is empty.")

    n_lollipops = len(COMPARISON_SPECS)
    n_factors = len(factors)
    panel_w = _lollipop_panel_width_inches(n_lollipops)
    fig_w = panel_w * n_factors + 0.25
    ylim = symmetric_ylim(corr_df["pearson_r"])

    axis_tick_fs = _GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D

    with figure_font_context():
        fig = plt.figure(figsize=(fig_w, _ROW_HEIGHT_IN))
        gs = GridSpecFromSubplotSpec(
            1,
            n_factors,
            subplot_spec=fig.add_gridspec(1, 1)[0, 0],
            width_ratios=[1.0] * n_factors,
            wspace=_LOLLIPOP_FACTOR_WSPACE,
        )
        for i, factor_tag in enumerate(factors):
            ax = fig.add_subplot(gs[0, i])
            factor_df = corr_df.loc[corr_df["factor"] == factor_tag].copy()
            _plot_factor_lollipop_ax(
                ax,
                factor_df,
                factor_tag=factor_tag,
                ylim=ylim,
                show_ylabel=(i == 0),
                axis_tick_fs=axis_tick_fs,
            )

        fig.tight_layout(
            pad=0.2,
            h_pad=0.4,
            w_pad=0.12,
            rect=[0.0, 0.06, 1.0, 1.0],
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=_LOLLIPOP_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
