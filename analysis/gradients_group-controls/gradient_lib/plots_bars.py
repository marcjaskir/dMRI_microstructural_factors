"""Region-group and anatomical-axis bar summaries for G1/G2 of controls.

For each factor (column pair), draws two bar panels:

* Left: per-ROI gradient mean +/- SEM within each region group (HCP1065 tract
  families, 4S subcortex groups, Glasser cortical lobes), sorted ascending by
  mean.
* Right: Pearson r between the chosen gradient (G1 or G2) across ROIs and each
  anatomical-axis rank (M-L, A-P, D-V) from the whole-brain centroid table.

Layout uses the same outer column spacing as
:func:`plots_scatter.plot_gradient_by_tissue_scatter` (one outer column per
factor, ``_GRADIENT_BY_FACTOR_WSPACE_OUTER``), with an asymmetric inner sub-grid per
factor (region groups : anatomical axis) spaced by ``_BAR_PAIR_WSPACE_INNER``. Inner
``width_ratios`` stay at ``_BAR_PAIR_WIDTH_RATIOS`` (2.5:1) so each factor column keeps
the same total width as before; optional post-layout reflow widens region groups vs the
axis lollipop without resizing the outer column.
Header row reuses :func:`plots_scatter._add_factor_pair_header_row`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec

from .config import (
    DEFAULT_TRACTOMETRY_ROOT,
    GRADIENT_AXIS_LABEL_COLOR,
    TISSUE_CORTICAL_GM,
)
from .embedding import gradient_from_row
from .groupings import load_neuroaxis_ranks
from .neuroaxis_correlations import (
    NEUROAXIS_AXES,
    collect_gradient_roi_labels,
    pearson_r_gradient_vs_neuroaxis,
)
from .io import subcortical_grey_matter_column_names
from .plots_scatter import (
    _GRADIENT_BY_FACTOR_WSPACE_OUTER,
    _GRADIENT_BY_FIGURE_AXIS_LABEL_FS_2D,
    _GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D,
    _add_factor_pair_header_row,
    _hide_top_right_spines,
)
from .region_groups import (
    REGION_GROUP_BAR_EDGE_WIDTH,
    build_roi_to_region_group,
    load_cortical_lobe_region_group_by_roi,
    region_group_bar_facecolor,
    tract_base_to_functional_group_map,
)
from .types import GradientRunRow

# Solid bar color for neuroaxis correlations; matches the cortex GM tone used in tissue scatters.
_NEUROAXIS_BAR_FACECOLOR = TISSUE_CORTICAL_GM
_NEUROAXIS_BAR_EDGECOLOR = "black"
_NEUROAXIS_BAR_EDGEWIDTH = 0.8

# Short keys must match ``NEUROAXIS_LABELS`` values in ``groupings.py`` (ROI dict keys).
_NEUROAXIS_DISPLAY_ORDER: tuple[str, ...] = ("M-L", "A-P", "D-V")
_NEUROAXIS_SHORT_TO_TICK_LABEL: dict[str, str] = {
    "M-L": "Mesial-Lateral",
    "A-P": "Anterior-Posterior",
    "D-V": "Dorsal-Ventral",
}
# Pearson-r y-limits for the anatomical-axis bar panel (per gradient index).
_NEUROAXIS_YLIM_BY_GRADIENT_INDEX: dict[int, tuple[float, float]] = {
    0: (-0.5, 0.5),  # G1
    1: (-0.6, 0.6),  # G2
}


def _neuroaxis_corr_ylim(gradient_index: int) -> tuple[float, float]:
    """Y-axis limits for Pearson r on the anatomical-axis panel."""
    return _NEUROAXIS_YLIM_BY_GRADIENT_INDEX.get(
        gradient_index, _NEUROAXIS_YLIM_BY_GRADIENT_INDEX[1]
    )


def _lollipop_ylim_display(neuroaxis_ylim: tuple[float, float]) -> tuple[float, float]:
    """Y limits shown on the lollipop panel (G1 uses exact ``-0.5``..``0.5``, no padding)."""
    ylo, yhi = neuroaxis_ylim
    if ylo == -0.5 and yhi == 0.5:
        return (-0.5, 0.5)
    pad = _NEUROAXIS_LOLLIPOP_Y_PAD_FRAC * (yhi - ylo)
    return (ylo - pad, yhi + pad)
# Inner wspace between the (region-groups | anatomical-axis) pair. Looser than the
# scatter pair constant because both bar panels carry their own y-tick labels.
_BAR_PAIR_WSPACE_INNER = 0.48
# GridSpec inner split (region groups | axis). Sum 3.5 matches standalone bar fig width.
_BAR_PAIR_WIDTH_RATIOS: tuple[float, float] = (2.5, 1.0)
_BAR_PAIR_WIDTH_SUM = sum(_BAR_PAIR_WIDTH_RATIOS)
# After ``tight_layout``, repartition the pair bbox (left fraction -> region groups).
_BAR_PAIR_REFLOW_LEFT_FRAC = 0.86
_BAR_PAIR_REFLOW_INNER_GAP_FRAC = 0.11

# Gradient-axis lollipop styling (Pearson r vs anatomical ranks).
_NEUROAXIS_LOLLIPOP_COLOR = "black"
_NEUROAXIS_LOLLIPOP_LINEWIDTH = 3.5
_NEUROAXIS_LOLLIPOP_MARKERSIZE = 140.0
_NEUROAXIS_LOLLIPOP_EDGEWIDTH = 1.2
_NEUROAXIS_LOLLIPOP_X_MARGIN = 0.42
_NEUROAXIS_LOLLIPOP_Y_PAD_FRAC = 0.1

from .figure_style import figure_font_context


# ---------------------------------------------------------------------------
# Bar prep helpers (port from gradients/gradient_lib/bars.py)
# ---------------------------------------------------------------------------


def _prepare_region_group_bars(
    g: pd.Series,
    roi_to_group: dict[str, str],
) -> tuple[list[str], np.ndarray, np.ndarray, list[int]]:
    """Mean +/- SEM of ``g`` within each region-group label; sorted asc by mean."""
    grouped: dict[str, list[float]] = {}
    for roi, val in g.items():
        lab = roi_to_group.get(str(roi))
        if lab is None:
            continue
        v = float(val)
        if not np.isfinite(v):
            continue
        grouped.setdefault(lab, []).append(v)
    if not grouped:
        return [], np.asarray([]), np.asarray([]), []
    labels_all = sorted(grouped.keys())
    means_all = np.array(
        [float(np.mean(grouped[lab])) for lab in labels_all], dtype=np.float64
    )
    sems_all = np.array(
        [
            float(np.std(grouped[lab], ddof=1) / np.sqrt(len(grouped[lab])))
            if len(grouped[lab]) > 1
            else 0.0
            for lab in labels_all
        ],
        dtype=np.float64,
    )
    counts_all = [len(grouped[lab]) for lab in labels_all]
    order = np.argsort(means_all)
    labels = [labels_all[i] for i in order]
    means = means_all[order]
    sems = sems_all[order]
    counts = [counts_all[i] for i in order]
    return labels, means, sems, counts


def _prepare_correlation_bars(
    g: pd.Series,
    mapping: dict[str, dict[str, float]],
    columns: tuple[str, ...],
) -> tuple[list[str], np.ndarray, list[int]]:
    """Pearson r between ``g`` and each column in ``mapping``; sorted desc by r."""
    tab = pearson_r_gradient_vs_neuroaxis(g, mapping, axes=columns)
    if tab.empty:
        return [], np.asarray([]), []
    valid = tab["pearson_r"].notna()
    if not valid.any():
        return [], np.asarray([]), []
    sub = tab.loc[valid].copy()
    names_sorted = sorted(
        sub["neuroaxis_axis"].astype(str).tolist(),
        key=lambda k: -float(sub.loc[sub["neuroaxis_axis"] == k, "pearson_r"].iloc[0]),
    )
    values = np.array(
        [float(sub.loc[sub["neuroaxis_axis"] == k, "pearson_r"].iloc[0]) for k in names_sorted],
        dtype=np.float64,
    )
    counts = [
        int(sub.loc[sub["neuroaxis_axis"] == k, "n_rois"].iloc[0]) for k in names_sorted
    ]
    return names_sorted, values, counts


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _reflow_bar_pair_axes(
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    *,
    left_frac: float = _BAR_PAIR_REFLOW_LEFT_FRAC,
    inner_gap_frac: float = _BAR_PAIR_REFLOW_INNER_GAP_FRAC,
) -> None:
    """Widen the left panel and narrow the right within the existing factor-column bbox."""
    if not (0.0 < left_frac < 1.0):
        raise ValueError(f"left_frac must be in (0, 1); got {left_frac}")
    box_l = ax_left.get_position()
    box_r = ax_right.get_position()
    x0 = min(box_l.x0, box_r.x0)
    x1 = max(box_l.x1, box_r.x1)
    width = x1 - x0
    gap = inner_gap_frac * width
    usable = width - gap
    left_w = usable * left_frac
    right_w = usable * (1.0 - left_frac)
    ax_left.set_position([x0, box_l.y0, left_w, box_l.height])
    ax_right.set_position([x0 + left_w + gap, box_r.y0, right_w, box_r.height])


# ---------------------------------------------------------------------------
# Axes painters
# ---------------------------------------------------------------------------


def _style_bar_axes(
    ax: plt.Axes,
    *,
    axis_label_fs: float,
    axis_tick_fs: float,
    show_ylabel: bool,
    ylabel: str | None,
    title: str,
) -> None:
    ax.tick_params(axis="x", labelsize=axis_tick_fs)
    ax.tick_params(axis="y", labelsize=axis_tick_fs)
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.45, zorder=0)
    ax.set_title(title, fontsize=axis_label_fs, color=GRADIENT_AXIS_LABEL_COLOR)
    if show_ylabel and ylabel:
        ax.set_ylabel(
            ylabel,
            fontsize=axis_label_fs,
            color=GRADIENT_AXIS_LABEL_COLOR,
        )
    else:
        ax.set_ylabel("")
    _hide_top_right_spines(ax)


def _plot_region_group_bars_ax(
    ax: plt.Axes,
    g: pd.Series,
    roi_to_group: dict[str, str],
    *,
    axis_label_fs: float,
    axis_tick_fs: float,
    show_ylabel: bool,
    ylabel: str | None,
    title: str,
) -> None:
    labels, means, sems, _ = _prepare_region_group_bars(g, roi_to_group)
    if not labels:
        ax.text(
            0.5, 0.5,
            "(no region-group overlap)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=axis_label_fs * 0.6, color=GRADIENT_AXIS_LABEL_COLOR,
        )
        _style_bar_axes(
            ax,
            axis_label_fs=axis_label_fs,
            axis_tick_fs=axis_tick_fs,
            show_ylabel=show_ylabel,
            ylabel=ylabel,
            title=title,
        )
        return
    x = np.arange(len(labels))
    facecolors = [region_group_bar_facecolor(lab) for lab in labels]
    ax.bar(
        x,
        means,
        yerr=sems,
        color=facecolors,
        edgecolor="k",
        linewidth=REGION_GROUP_BAR_EDGE_WIDTH,
        capsize=3,
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    _style_bar_axes(
        ax,
        axis_label_fs=axis_label_fs,
        axis_tick_fs=axis_tick_fs,
        show_ylabel=show_ylabel,
        ylabel=ylabel,
        title=title,
    )


def _plot_neuroaxis_corr_lollipop_ax(
    ax: plt.Axes,
    g: pd.Series,
    neuroaxis_by_roi: dict[str, dict[str, float]],
    *,
    neuroaxis_ylim: tuple[float, float],
    axis_label_fs: float,
    axis_tick_fs: float,
    show_ylabel: bool,
    ylabel: str | None,
    title: str,
) -> None:
    """Pearson r vs anatomical-axis ranks as a lollipop (stem + marker)."""
    names, values, _ = _prepare_correlation_bars(
        g, neuroaxis_by_roi, NEUROAXIS_AXES
    )
    y_disp_lo, y_disp_hi = _lollipop_ylim_display(neuroaxis_ylim)

    if not names:
        ax.text(
            0.5, 0.5,
            "(no anatomical-axis overlap)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=axis_label_fs * 0.6, color=GRADIENT_AXIS_LABEL_COLOR,
        )
        _style_bar_axes(
            ax,
            axis_label_fs=axis_label_fs,
            axis_tick_fs=axis_tick_fs,
            show_ylabel=show_ylabel,
            ylabel=ylabel,
            title=title,
        )
        ax.set_ylim(y_disp_lo, y_disp_hi)
        return
    # Force the fixed left-to-right display order (M-L, A-P, D-V) regardless of
    # the r-value sort returned by ``_prepare_correlation_bars``.
    name_to_value = dict(zip(names, values.tolist()))
    fixed_names = [n for n in _NEUROAXIS_DISPLAY_ORDER if n in name_to_value]
    fixed_values = np.asarray(
        [name_to_value[n] for n in fixed_names], dtype=np.float64
    )
    x = np.arange(len(fixed_names))
    ax.vlines(
        x,
        0.0,
        fixed_values,
        colors=_NEUROAXIS_LOLLIPOP_COLOR,
        linewidth=_NEUROAXIS_LOLLIPOP_LINEWIDTH,
        zorder=2,
    )
    ax.scatter(
        x,
        fixed_values,
        s=_NEUROAXIS_LOLLIPOP_MARKERSIZE,
        c=_NEUROAXIS_LOLLIPOP_COLOR,
        edgecolors=_NEUROAXIS_LOLLIPOP_COLOR,
        linewidths=_NEUROAXIS_LOLLIPOP_EDGEWIDTH,
        zorder=3,
    )
    ax.set_xticks(x)
    tick_labels = [_NEUROAXIS_SHORT_TO_TICK_LABEL[n] for n in fixed_names]
    ax.set_xticklabels(
        tick_labels,
        rotation=45,
        ha="right",
        fontsize=axis_tick_fs,
    )
    ax.set_xlim(-_NEUROAXIS_LOLLIPOP_X_MARGIN, len(fixed_names) - 1 + _NEUROAXIS_LOLLIPOP_X_MARGIN)
    ax.set_ylim(y_disp_lo, y_disp_hi)
    ax.margins(x=0, y=0)
    _style_bar_axes(
        ax,
        axis_label_fs=axis_label_fs,
        axis_tick_fs=axis_tick_fs,
        show_ylabel=show_ylabel,
        ylabel=ylabel,
        title=title,
    )


def _paint_groups_axes_bars_data_row(
    fig: plt.Figure,
    gs: object,
    results: list[GradientRunRow],
    *,
    gradient_index: int,
    tractometry_root: Path,
    gs_data_row: int = 1,
) -> tuple[int, list[tuple[plt.Axes, plt.Axes]]]:
    """Paint one row of per-factor bar pairs: region groups then anatomical-axis Pearson r.

    ``gs_data_row`` is the row index in ``gs`` where each factor's inner
    ``GridSpecFromSubplotSpec`` is anchored (``_BAR_PAIR_WIDTH_RATIOS`` on that inner
    pair only). Use ``1`` when ``gs`` is ``(2, n)`` with factor headers in row ``0``;
    use ``0`` when ``gs`` is ``(1, n)`` and there is no header row.

    Does not run ``tight_layout``, reflow, or ``savefig``.

    Returns ``(gradient_number, [(ax_region, ax_axis), ...])`` per factor.
    """
    if gradient_index < 0:
        raise ValueError(f"gradient_index must be >= 0; got {gradient_index}")
    g_num = gradient_index + 1
    n = len(results)

    subcortical_labels = subcortical_grey_matter_column_names(tractometry_root)
    tract_to_group = tract_base_to_functional_group_map()
    cortical_by_roi = load_cortical_lobe_region_group_by_roi(tractometry_root)
    neuroaxis_by_roi = load_neuroaxis_ranks(
        tractometry_root, roi_labels=collect_gradient_roi_labels(results)
    )
    neuroaxis_corr_ylim = _neuroaxis_corr_ylim(gradient_index)

    axis_label_fs = _GRADIENT_BY_FIGURE_AXIS_LABEL_FS_2D
    axis_tick_fs = _GRADIENT_BY_FIGURE_AXIS_TICK_FS_2D
    ax_pairs: list[tuple[plt.Axes, plt.Axes]] = []

    for f, row in enumerate(results):
        inner = GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=gs[gs_data_row, f],
            width_ratios=list(_BAR_PAIR_WIDTH_RATIOS),
            wspace=_BAR_PAIR_WSPACE_INNER,
        )
        ax_rg = fig.add_subplot(inner[0, 0])
        ax_corr = fig.add_subplot(inner[0, 1])

        g_series = gradient_from_row(row, gradient_index)
        roi_to_group = build_roi_to_region_group(
            g_series,
            tract_to_group=tract_to_group,
            cortical_by_roi=cortical_by_roi,
            subcortical_matched_columns=subcortical_labels,
        )

        _plot_region_group_bars_ax(
            ax_rg,
            g_series,
            roi_to_group,
            axis_label_fs=axis_label_fs,
            axis_tick_fs=axis_tick_fs,
            show_ylabel=True,
            ylabel=f"Gradient {g_num}",
            title="Region groups",
        )
        _plot_neuroaxis_corr_lollipop_ax(
            ax_corr,
            g_series,
            neuroaxis_by_roi,
            neuroaxis_ylim=neuroaxis_corr_ylim,
            axis_label_fs=axis_label_fs,
            axis_tick_fs=axis_tick_fs,
            show_ylabel=True,
            ylabel="Pearson r",
            title=f"Gradient {g_num} axis",
        )
        ax_pairs.append((ax_rg, ax_corr))
    return g_num, ax_pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_gradient_by_groups_axes_bars(
    results: list[GradientRunRow],
    out_path: Path,
    *,
    gradient_index: int,
    tractometry_root: Path | None = None,
    method_tag: str = "laplacian_eigenmodes",
    cohort_tag: str | None = "controls",
) -> Path:
    """One data row: ``[region groups | anatomical axis]`` per factor (column pairs).

    ``gradient_index=0`` selects G1, ``gradient_index=1`` selects G2 (etc.); the same
    encoding-free ROI table is used for both panels (no in-figure legend).
    """
    n = len(results)
    if n == 0:
        return out_path
    if gradient_index < 0:
        raise ValueError(f"gradient_index must be >= 0; got {gradient_index}")

    root = tractometry_root if tractometry_root is not None else DEFAULT_TRACTOMETRY_ROOT

    with figure_font_context():
        # Inner pair: region groups (left, wider) then gradient-axis lollipop (right).
        # One "unit" of width; each factor's inner pair spans ``_BAR_PAIR_WIDTH_SUM``.
        col_w_unit = 3.6
        row_h = 5.75
        header_row_ratio = 0.09
        fig_w_total = col_w_unit * (_BAR_PAIR_WIDTH_SUM * n) + 0.8
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
        _, ax_pairs = _paint_groups_axes_bars_data_row(
            fig, gs, results, gradient_index=gradient_index, tractometry_root=root
        )

        fig.tight_layout(
            pad=0.35,
            h_pad=0.45,
            w_pad=0.35,
            rect=[0.0, 0.04, 1.0, 1.0],
        )
        for ax_rg, ax_axis in ax_pairs:
            _reflow_bar_pair_axes(ax_rg, ax_axis)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path
