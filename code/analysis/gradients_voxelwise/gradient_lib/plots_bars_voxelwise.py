"""Region-group + neuroaxis bar figures for voxelwise gradients."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec

from .config import DEFAULT_TRACTOMETRY_ROOT, GRADIENT_AXIS_LABEL_COLOR
from .embedding import gradient_from_row
from .gc_imports import gc_region_groups
from .neuroaxis_voxelwise import (
    NEUROAXIS_AXES,
    compute_neuroaxis_ranks,
    gradient_values_in_mask_order,
    pearson_r_gradient_vs_coordinate_ranks,
)
from .parcel_gradients import voxel_rows_to_parcel_gradient_run_rows
from .region_groups_voxelwise import (
    build_parcel_to_region_group,
    load_tract_label_to_type_group,
    region_group_bar_facecolor,
)
from .types import VoxelGradientRunRow

load_cortical_lobe_region_group_by_roi = gc_region_groups().load_cortical_lobe_region_group_by_roi

_BAR_FIGURE_FONT_RCPARAMS: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Georgia", "DejaVu Serif", "serif"],
}

_NEUROAXIS_DISPLAY_ORDER: tuple[str, ...] = ("M-L", "A-P", "D-V")
_NEUROAXIS_SHORT_TO_TICK_LABEL: dict[str, str] = {
    "M-L": "Mesial-Lateral",
    "A-P": "Anterior-Posterior",
    "D-V": "Dorsal-Ventral",
}
_NEUROAXIS_YLIM_BY_GRADIENT_INDEX: dict[int, tuple[float, float]] = {
    0: (-0.5, 0.5),
    1: (-0.6, 0.6),
}
_NEUROAXIS_LOLLIPOP_COLOR = "black"
_NEUROAXIS_LOLLIPOP_LINEWIDTH = 3.5
_NEUROAXIS_LOLLIPOP_MARKERSIZE = 140.0
_NEUROAXIS_LOLLIPOP_EDGEWIDTH = 1.2
_NEUROAXIS_LOLLIPOP_X_MARGIN = 0.42
_BAR_PAIR_WSPACE_INNER = 0.48
_BAR_PAIR_WIDTH_RATIOS: tuple[float, float] = (2.5, 1.0)
_BAR_PAIR_REFLOW_LEFT_FRAC = 0.86
_BAR_PAIR_REFLOW_INNER_GAP_FRAC = 0.11
REGION_GROUP_BAR_EDGE_WIDTH = 2.0
_AXIS_LABEL_FS = 22.0
_AXIS_TICK_FS = 16.0


def _neuroaxis_corr_ylim(gradient_index: int) -> tuple[float, float]:
    return _NEUROAXIS_YLIM_BY_GRADIENT_INDEX.get(
        gradient_index, _NEUROAXIS_YLIM_BY_GRADIENT_INDEX[1]
    )


def _hide_top_right_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _prepare_region_group_bars(
    g: pd.Series,
    parcel_to_group: dict[str, str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    grouped: dict[str, list[float]] = {}
    for parcel, val in g.items():
        lab = parcel_to_group.get(str(parcel))
        if lab is None:
            continue
        v = float(val)
        if not np.isfinite(v):
            continue
        grouped.setdefault(lab, []).append(v)
    if not grouped:
        return [], np.asarray([]), np.asarray([])
    labels_all = sorted(grouped.keys())
    means_all = np.array([float(np.mean(grouped[lab])) for lab in labels_all])
    sems_all = np.array(
        [
            float(np.std(grouped[lab], ddof=1) / np.sqrt(len(grouped[lab])))
            if len(grouped[lab]) > 1
            else 0.0
            for lab in labels_all
        ]
    )
    order = np.argsort(means_all)
    return (
        [labels_all[i] for i in order],
        means_all[order],
        sems_all[order],
    )


def _style_bar_axes(
    ax: plt.Axes,
    *,
    show_ylabel: bool,
    ylabel: str | None,
    title: str,
) -> None:
    ax.tick_params(axis="x", labelsize=_AXIS_TICK_FS)
    ax.tick_params(axis="y", labelsize=_AXIS_TICK_FS)
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.45, zorder=0)
    ax.set_title(title, fontsize=_AXIS_LABEL_FS, color=GRADIENT_AXIS_LABEL_COLOR)
    if show_ylabel and ylabel:
        ax.set_ylabel(ylabel, fontsize=_AXIS_LABEL_FS, color=GRADIENT_AXIS_LABEL_COLOR)
    else:
        ax.set_ylabel("")
    _hide_top_right_spines(ax)


def _plot_region_group_bars_ax(
    ax: plt.Axes,
    g: pd.Series,
    parcel_to_group: dict[str, str],
    *,
    show_ylabel: bool,
    ylabel: str | None,
    title: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    labels, means, sems = _prepare_region_group_bars(g, parcel_to_group)
    if not labels:
        ax.text(
            0.5, 0.5, "(no region-group overlap)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=_AXIS_LABEL_FS * 0.6, color=GRADIENT_AXIS_LABEL_COLOR,
        )
        _style_bar_axes(ax, show_ylabel=show_ylabel, ylabel=ylabel, title=title)
        return
    x = np.arange(len(labels))
    facecolors = [region_group_bar_facecolor(lab) for lab in labels]
    ax.bar(
        x, means, yerr=sems, color=facecolors, edgecolor="k",
        linewidth=REGION_GROUP_BAR_EDGE_WIDTH, capsize=3, zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    _style_bar_axes(ax, show_ylabel=show_ylabel, ylabel=ylabel, title=title)


def _plot_neuroaxis_lollipop_ax(
    ax: plt.Axes,
    g_values: np.ndarray,
    ranks: dict[str, np.ndarray],
    *,
    neuroaxis_ylim: tuple[float, float],
    show_ylabel: bool,
    ylabel: str,
    title: str,
) -> None:
    tab = pearson_r_gradient_vs_coordinate_ranks(g_values, ranks, axes=NEUROAXIS_AXES)
    name_to_value = dict(zip(tab["neuroaxis_axis"].astype(str), tab["pearson_r"]))
    fixed_names = [n for n in _NEUROAXIS_DISPLAY_ORDER if n in name_to_value]
    fixed_values = np.asarray([name_to_value[n] for n in fixed_names], dtype=np.float64)
    ylo, yhi = neuroaxis_ylim

    if not fixed_names:
        ax.text(
            0.5, 0.5, "(no overlap)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=_AXIS_LABEL_FS * 0.6, color=GRADIENT_AXIS_LABEL_COLOR,
        )
    else:
        x = np.arange(len(fixed_names))
        ax.vlines(x, 0.0, fixed_values, colors=_NEUROAXIS_LOLLIPOP_COLOR,
                  linewidth=_NEUROAXIS_LOLLIPOP_LINEWIDTH, zorder=2)
        ax.scatter(
            x, fixed_values, s=_NEUROAXIS_LOLLIPOP_MARKERSIZE,
            c=_NEUROAXIS_LOLLIPOP_COLOR, edgecolors=_NEUROAXIS_LOLLIPOP_COLOR,
            linewidths=_NEUROAXIS_LOLLIPOP_EDGEWIDTH, zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_NEUROAXIS_SHORT_TO_TICK_LABEL[n] for n in fixed_names],
            rotation=45, ha="right", fontsize=_AXIS_TICK_FS,
        )
        ax.set_xlim(-_NEUROAXIS_LOLLIPOP_X_MARGIN, len(fixed_names) - 1 + _NEUROAXIS_LOLLIPOP_X_MARGIN)

    ax.set_ylim(ylo, yhi)
    _style_bar_axes(ax, show_ylabel=show_ylabel, ylabel=ylabel, title=title)


def _reflow_bar_pair_axes(ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    box_l = ax_left.get_position()
    box_r = ax_right.get_position()
    x0 = min(box_l.x0, box_r.x0)
    x1 = max(box_l.x1, box_r.x1)
    width = x1 - x0
    gap = _BAR_PAIR_REFLOW_INNER_GAP_FRAC * width
    usable = width - gap
    left_w = usable * _BAR_PAIR_REFLOW_LEFT_FRAC
    right_w = usable * (1.0 - _BAR_PAIR_REFLOW_LEFT_FRAC)
    ax_left.set_position([x0, box_l.y0, left_w, box_l.height])
    ax_right.set_position([x0 + left_w + gap, box_r.y0, right_w, box_r.height])


def paint_groups_axes_bars_row(
    fig: plt.Figure,
    gs_row,
    voxel_results: list[VoxelGradientRunRow],
    parcel_results: list,
    *,
    gradient_index: int,
    tractometry_root: Path | None = None,
    cache_dir: Path | None = None,
) -> list[tuple[plt.Axes, plt.Axes]]:
    """Paint [region groups | coordinate neuroaxis] per factor."""
    _ = cache_dir
    g_num = gradient_index + 1
    root = tractometry_root or DEFAULT_TRACTOMETRY_ROOT
    tract_to_type = load_tract_label_to_type_group()
    cortical_by_roi = load_cortical_lobe_region_group_by_roi(root)
    ylim = _neuroaxis_corr_ylim(gradient_index)
    ax_pairs: list[tuple[plt.Axes, plt.Axes]] = []

    for f, (vrow, prow) in enumerate(zip(voxel_results, parcel_results)):
        inner = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_row[0, f],
            width_ratios=list(_BAR_PAIR_WIDTH_RATIOS),
            wspace=_BAR_PAIR_WSPACE_INNER,
        )
        ax_rg = fig.add_subplot(inner[0, 0])
        ax_corr = fig.add_subplot(inner[0, 1])

        g_series = gradient_from_row(prow, gradient_index)
        parcel_to_group = build_parcel_to_region_group(
            g_series,
            tract_to_type=tract_to_type,
            cortical_by_roi=cortical_by_roi,
        )
        _plot_region_group_bars_ax(
            ax_rg, g_series, parcel_to_group,
            show_ylabel=True,
            ylabel=f"Gradient {g_num}",
            title="Region groups",
        )
        ranks = compute_neuroaxis_ranks(vrow[4])
        g_vals = gradient_values_in_mask_order(vrow, gradient_index)
        _plot_neuroaxis_lollipop_ax(
            ax_corr, g_vals, ranks,
            neuroaxis_ylim=ylim,
            show_ylabel=True,
            ylabel="Pearson r",
            title=f"Gradient {g_num} axis",
        )
        ax_pairs.append((ax_rg, ax_corr))
    return ax_pairs


def plot_groups_axes_bars(
    voxel_results: list[VoxelGradientRunRow],
    out_path: Path,
    *,
    gradient_index: int,
    cache_dir: Path | None = None,
    tractometry_root: Path | None = None,
    cohort_tag: str | None = "controls",
    mask_nii: Path | None = None,
) -> Path:
    """Standalone [region groups | neuroaxis] figure per factor."""
    _ = cohort_tag
    n = len(voxel_results)
    parcel_results = voxel_rows_to_parcel_gradient_run_rows(
        voxel_results, cache_dir=cache_dir, mask_nii=mask_nii
    )
    fig_w = max(7.0 * n, 10.0)
    with plt.rc_context(_BAR_FIGURE_FONT_RCPARAMS):
        fig = plt.figure(figsize=(fig_w, 5.75))
        gs = fig.add_gridspec(1, n, wspace=0.32)
        ax_pairs = paint_groups_axes_bars_row(
            fig, gs, voxel_results, parcel_results,
            gradient_index=gradient_index,
            tractometry_root=tractometry_root,
            cache_dir=cache_dir,
        )
        fig.tight_layout(pad=0.35, h_pad=0.56, w_pad=0.35, rect=[0.0, 0.04, 1.0, 1.0])
        for ax_rg, ax_corr in ax_pairs:
            _reflow_bar_pair_axes(ax_rg, ax_corr)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path
