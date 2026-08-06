"""Lollipop summary figures for neuromaps spatial correlations."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    COHORT_TAG,
    GRADIENT_AXIS_LABEL_COLOR,
    NEUROMAPS_ANNOTATION_INFO_CSV,
    NEUROMAPS_ROW_FSLR,
    NEUROMAPS_ROW_MNI,
    NEUROMAPS_SPACE_FSLR,
    NEUROMAPS_SPACE_MNI,
    NEUROMAPS_TOP_K,
)
from .neuromaps_correlations import (
    AnnotationMetadataIndex,
    annotation_display_label,
    dedupe_correlations_by_description_brief,
    load_annotation_description_lookup,
    load_annotation_metadata,
    neuromaps_correlation_csv_path,
)
from .plots_bars_voxelwise import _BAR_FIGURE_FONT_RCPARAMS

_LOLLIPOP_COLOR = "black"
_LOLLIPOP_LINEWIDTH = 3.5
_LOLLIPOP_MARKERSIZE = 140.0
_LOLLIPOP_EDGEWIDTH = 1.2
_LOLLIPOP_X_MARGIN = 0.42
_AXIS_LABEL_FS = 22.0
_AXIS_TICK_FS = 14.0
_ROW_TITLE_FS = 20.0
_P_SIG = 0.05


def _factor_display_name(factor_tag: str) -> str:
    m = re.match(r"^F(\d+)$", str(factor_tag).strip(), re.I)
    if m:
        return f"Factor {int(m.group(1))}"
    return str(factor_tag)


def _short_label(key: str, max_len: int = 32) -> str:
    s = str(key).replace("_", " ")
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _plot_lollipop_panel(
    ax: plt.Axes,
    panel_df: pd.DataFrame,
    *,
    title: str,
    show_ylabel: bool,
    ylabel: str,
    description_lookup: dict[tuple[str, str], str],
    metadata: AnnotationMetadataIndex | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    if panel_df.empty:
        ax.text(
            0.5,
            0.5,
            "(no correlations)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=_AXIS_LABEL_FS * 0.6,
            color=GRADIENT_AXIS_LABEL_COLOR,
        )
        ax.set_title(title, fontsize=_AXIS_LABEL_FS, color=GRADIENT_AXIS_LABEL_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return

    panel_df = dedupe_correlations_by_description_brief(panel_df, metadata)
    panel_df = panel_df.sort_values("abs_r", ascending=False).head(NEUROMAPS_TOP_K)
    labels = [
        _short_label(
            annotation_display_label(
                row["source"],
                row["desc"],
                description_lookup,
                fallback=str(row.get("annotation_key", "")),
            )
        )
        for _, row in panel_df.iterrows()
    ]
    values = panel_df["pearson_r"].to_numpy(dtype=np.float64)
    x = np.arange(len(labels))

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

    if "p_null" in panel_df.columns:
        for xi, (_, row) in enumerate(panel_df.iterrows()):
            p = row.get("p_null")
            if pd.notna(p) and float(p) < _P_SIG:
                y = float(row["pearson_r"])
                ax.text(
                    xi,
                    y + (0.03 if y >= 0 else -0.03),
                    "*",
                    ha="center",
                    va="bottom" if y >= 0 else "top",
                    fontsize=_AXIS_TICK_FS,
                    color=GRADIENT_AXIS_LABEL_COLOR,
                )

    ax.axhline(0.0, color="k", lw=0.6, alpha=0.45, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=_AXIS_TICK_FS - 3)
    ax.set_xlim(-_LOLLIPOP_X_MARGIN, len(labels) - 1 + _LOLLIPOP_X_MARGIN)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ymax = max(0.05, float(np.nanmax(np.abs(values))) * 1.15)
        ax.set_ylim(-ymax, ymax)
    ax.set_title(title, fontsize=_AXIS_LABEL_FS, color=GRADIENT_AXIS_LABEL_COLOR)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=_AXIS_LABEL_FS, color=GRADIENT_AXIS_LABEL_COLOR)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=_AXIS_TICK_FS, colors=GRADIENT_AXIS_LABEL_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_neuromaps_lollipop_summary(
    csv_paths: dict[str, Path],
    out_path: Path,
    *,
    gradient_index: int,
    cohort_tag: str = COHORT_TAG,
    metadata_csv: Path | None = None,
) -> Path:
    """2 rows (MNI / fsLR) × N factor columns of top-|r| neuromaps lollipops."""
    factors = sorted(csv_paths.keys(), key=lambda f: int(re.sub(r"\D", "", f) or 0))
    if not factors:
        raise ValueError("No factor CSV paths provided")

    description_lookup = load_annotation_description_lookup(
        metadata_csv or NEUROMAPS_ANNOTATION_INFO_CSV
    )
    metadata = load_annotation_metadata(metadata_csv or NEUROMAPS_ANNOTATION_INFO_CSV)

    tables: dict[str, pd.DataFrame] = {}
    for factor, path in csv_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing neuromaps CSV: {path}")
        tables[factor] = pd.read_csv(path)

    n = len(factors)
    fig_w = max(6.0 * n, 10.0)
    fig_h = 10.5
    row_titles = (NEUROMAPS_ROW_MNI, NEUROMAPS_ROW_FSLR)
    space_keys = (NEUROMAPS_SPACE_MNI, NEUROMAPS_SPACE_FSLR)

    with plt.rc_context(_BAR_FIGURE_FONT_RCPARAMS):
        fig, axes = plt.subplots(2, n, figsize=(fig_w, fig_h), squeeze=False)
        for col, factor in enumerate(factors):
            df = tables[factor]
            df = df[df["gradient"].astype(int) == int(gradient_index)]
            for row, (space, row_title) in enumerate(zip(space_keys, row_titles)):
                ax = axes[row, col]
                sub = df[df["space"].astype(str) == space]
                _plot_lollipop_panel(
                    ax,
                    sub,
                    title=_factor_display_name(factor) if row == 0 else "",
                    show_ylabel=(col == 0),
                    ylabel="Pearson r" if col == 0 else "",
                    description_lookup=description_lookup,
                    metadata=metadata,
                )
                if col == 0:
                    ax.text(
                        -0.28,
                        0.5,
                        row_title,
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=_ROW_TITLE_FS,
                        color=GRADIENT_AXIS_LABEL_COLOR,
                    )
        fig.suptitle(
            f"Gradient {gradient_index} — top {NEUROMAPS_TOP_K} |r| neuromaps annotations",
            fontsize=_AXIS_LABEL_FS + 2,
            color=GRADIENT_AXIS_LABEL_COLOR,
            y=0.98,
        )
        fig.tight_layout(rect=[0.06, 0.03, 1.0, 0.95])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


def plot_neuromaps_figures_from_output(
    output_dir: Path,
    *,
    factors: list[str],
    gradient_indices: list[int],
    cohort_tag: str = COHORT_TAG,
    figures_dir: Path | None = None,
) -> list[Path]:
    fig_dir = figures_dir or (output_dir / "figures")
    saved: list[Path] = []
    for gi in gradient_indices:
        csv_map = {
            f: neuromaps_correlation_csv_path(output_dir, f, gi, cohort_tag=cohort_tag)
            for f in factors
        }
        out = fig_dir / f"gradient{gi}_neuromaps_lollipop_cohort-{cohort_tag}.png"
        plot_neuromaps_lollipop_summary(csv_map, out, gradient_index=gi, cohort_tag=cohort_tag)
        saved.append(out)
    return saved
