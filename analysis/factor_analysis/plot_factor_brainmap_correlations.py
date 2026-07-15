#!/usr/bin/env python3
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
Pearson r between control-mean Glasser factor scores and brainmaps_glasser.csv columns.

Reads per-ROI factor scores from factor_z-scores (control cohort mean), correlates each
factor's spatial profile across Glasser cortex with every column in
``data/atlases/S-A_ArchetypalAxis/Glasser360_MMP/brainmaps_glasser.csv``, and writes
one lollipop plot per factor plus a combined figure and CSV tables.

Example:
  python analysis/factor_analysis/plot_factor_brainmap_correlations.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("{project_root()}")
DEFAULT_GM_FACTOR_SCORES = (
    PROJECT_ROOT
    / "derivatives/analysis/factor_z-scores/roi_factor_scores/gm_regions_factor_scores.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "derivatives/analysis/factor_analysis/All4_Combined"
DEFAULT_FILE_PREFIX = "controls_All4_Combined"

BRAINMAPS_PATH = (
    PROJECT_ROOT
    / "data/atlases/S-A_ArchetypalAxis/Glasser360_MMP/brainmaps_glasser.csv"
)
GLASSER_REGIONS_PATH = (
    PROJECT_ROOT
    / "data/atlases/S-A_ArchetypalAxis/Glasser360_MMP/glasser_regions.csv"
)

# Column order and human-readable labels (subset of GRADIENT_ATLAS_DISPLAY + Evolution.Expansion).
BRAINMAPS_GLASSER_PANEL_SPEC: list[tuple[str, str]] = [
    ("T1T2ratio", "Anatomical Hierarchy"),
    ("G1.fMRI", "Functional Hierarchy"),
    ("Evolution.Expansion", "Evolutionary Expansion"),
    ("AllometricScaling.PNC20mm", "Allometric Scaling"),
    ("PET.AG", "Aerobic Glycolysis"),
    ("CBF", "Cerebral Blood Flow"),
    ("PC1.AHBA", "Gene Expression"),
    ("PC1.Neurosynth", "Neurosynth"),
    ("Externopyramidisation", "Externopyramidization"),
    ("Cortical.Thickness", "Cortical Thickness"),
]
BRAINMAPS_GLASSER_DISPLAY: dict[str, str] = dict(BRAINMAPS_GLASSER_PANEL_SPEC)
BRAINMAPS_GLASSER_COLUMNS: tuple[str, ...] = tuple(c for c, _ in BRAINMAPS_GLASSER_PANEL_SPEC)

_LOLLIPOP_COLOR = "black"
_LOLLIPOP_LINEWIDTH = 2.5
_LOLLIPOP_MARKERSIZE = 72.0
_LOLLIPOP_EDGEWIDTH = 1.0
_LOLLIPOP_X_MARGIN = 0.38
_LOLLIPOP_YLIM = (-0.5, 0.5)
# Inches per x tick — primary control for horizontal density.
_LOLLIPOP_WIDTH_PER_TICK = 0.32
_LOLLIPOP_FIG_HEIGHT_SINGLE = 3.6
_LOLLIPOP_FIG_HEIGHT_PER_ROW = 2.55


def _lollipop_figsize(n_maps: int, *, n_rows: int = 1) -> tuple[float, float]:
    # Width spans inner ticks only; outer buffer comes from ``_LOLLIPOP_X_MARGIN`` on xlim.
    width = max(3.4, n_maps * _LOLLIPOP_WIDTH_PER_TICK)
    if n_rows <= 1:
        height = _LOLLIPOP_FIG_HEIGHT_SINGLE
    else:
        height = max(3.2, n_rows * _LOLLIPOP_FIG_HEIGHT_PER_ROW)
    return (width, height)


def load_brainmaps_by_roi() -> dict[str, dict[str, float]]:
    """ROI -> {map column: value} from brainmaps_glasser.csv aligned to glasser_regions.csv."""
    maps_df = pd.read_csv(BRAINMAPS_PATH)
    regions_df = pd.read_csv(GLASSER_REGIONS_PATH)
    if len(maps_df) != len(regions_df):
        raise ValueError(
            f"Row count mismatch: {BRAINMAPS_PATH.name} ({len(maps_df)}) vs "
            f"{GLASSER_REGIONS_PATH.name} ({len(regions_df)})"
        )
    missing = [c for c in BRAINMAPS_GLASSER_COLUMNS if c not in maps_df.columns]
    if missing:
        raise ValueError(f"brainmaps_glasser.csv missing expected columns: {missing}")

    regions = regions_df.iloc[:, 0].astype(str).tolist()
    out: dict[str, dict[str, float]] = {}
    for i, region in enumerate(regions):
        rec: dict[str, float] = {}
        for col in BRAINMAPS_GLASSER_COLUMNS:
            val = maps_df.iloc[i][col]
            if pd.isna(val):
                continue
            rec[col] = float(val)
        if not rec:
            continue
        out[region] = rec
        out[f"Left_{region}"] = rec
        out[f"Right_{region}"] = rec
    return out


def glasser_cortex_factor_series(factor_scores: pd.DataFrame, factor: str) -> pd.Series:
    """Control-mean factor scores restricted to Left_/Right_ Glasser cortex ROIs."""
    if factor not in factor_scores.columns:
        raise KeyError(f"Factor {factor!r} not in factor scores columns: {list(factor_scores.columns)}")
    s = factor_scores[factor].copy()
    s.index = s.index.astype(str)
    glasser_mask = s.index.str.startswith("Left_") | s.index.str.startswith("Right_")
    return s.loc[glasser_mask].dropna()


def pearson_r_vs_maps(
    roi_values: pd.Series,
    maps_by_roi: dict[str, dict[str, float]],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Pearson r between ``roi_values`` and each brainmap column across overlapping ROIs."""
    rows: list[dict[str, object]] = []
    for col in columns:
        xs: list[float] = []
        ys: list[float] = []
        for roi, yv in roi_values.items():
            rec = maps_by_roi.get(str(roi))
            if rec is None or col not in rec:
                continue
            xv = rec[col]
            if not np.isfinite(xv) or not np.isfinite(yv):
                continue
            xs.append(float(xv))
            ys.append(float(yv))
        if len(xs) < 3:
            r, n = np.nan, len(xs)
        else:
            xa = np.asarray(xs, dtype=np.float64)
            ya = np.asarray(ys, dtype=np.float64)
            if xa.std(ddof=0) == 0 or ya.std(ddof=0) == 0:
                r, n = np.nan, len(xs)
            else:
                r = float(np.corrcoef(xa, ya)[0, 1])
                n = len(xs)
        rows.append({"map": col, "pearson_r": r, "n_rois": n})
    out = pd.DataFrame(rows)
    out["abs_pearson_r"] = out["pearson_r"].abs()
    return out.sort_values("abs_pearson_r", ascending=False, na_position="last").reset_index(drop=True)


def _plot_lollipop_ax(
    ax: plt.Axes,
    corr_df: pd.DataFrame,
    *,
    title: str,
    show_ylabel: bool,
) -> None:
    """Lollipop of Pearson r vs brainmap columns, sorted by |r| descending."""
    valid = corr_df.loc[corr_df["pearson_r"].notna()].copy()
    if valid.empty:
        ax.text(0.5, 0.5, "(no map overlap)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_ylim(_LOLLIPOP_YLIM)
        return

    names = valid["map"].astype(str).tolist()
    values = valid["pearson_r"].to_numpy(dtype=np.float64)
    x = np.arange(len(names))
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4, zorder=1)
    ax.vlines(x, 0.0, values, colors=_LOLLIPOP_COLOR, linewidth=_LOLLIPOP_LINEWIDTH, zorder=2)
    ax.scatter(
        x,
        values,
        s=_LOLLIPOP_MARKERSIZE,
        c=_LOLLIPOP_COLOR,
        edgecolors=_LOLLIPOP_COLOR,
        linewidths=_LOLLIPOP_EDGEWIDTH,
        zorder=3,
    )
    tick_labels = [BRAINMAPS_GLASSER_DISPLAY.get(n, n) for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=55, ha="right", fontsize=9)
    ax.set_xlim(-_LOLLIPOP_X_MARGIN, len(names) - 1 + _LOLLIPOP_X_MARGIN)
    ax.set_ylim(_LOLLIPOP_YLIM)
    ax.margins(x=0)
    ax.tick_params(axis="x", pad=1)
    ax.set_title(title, fontsize=13)
    if show_ylabel:
        ax.set_ylabel("Pearson r", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_factor_brainmap_lollipops(
    factor_scores_path: Path,
    output_dir: Path,
    file_prefix: str,
) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Georgia", "DejaVu Serif", "serif"]

    factor_scores = pd.read_csv(factor_scores_path, index_col=0)
    maps_by_roi = load_brainmaps_by_roi()
    factors = [c for c in factor_scores.columns if str(c).startswith("F")]
    if not factors:
        raise ValueError(f"No factor columns found in {factor_scores_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_corr: dict[str, pd.DataFrame] = {}

    for factor in factors:
        roi_series = glasser_cortex_factor_series(factor_scores, factor)
        corr_df = pearson_r_vs_maps(roi_series, maps_by_roi, BRAINMAPS_GLASSER_COLUMNS)
        all_corr[factor] = corr_df

        csv_path = output_dir / f"{file_prefix}_{factor}_brainmaps_glasser_correlations.csv"
        corr_df.to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=_lollipop_figsize(len(BRAINMAPS_GLASSER_COLUMNS)))
        _plot_lollipop_ax(
            ax,
            corr_df,
            title=f"{factor} vs Glasser360 multimodal maps (n={int(corr_df['n_rois'].max())} ROIs)",
            show_ylabel=True,
        )
        fig.subplots_adjust(left=0.13, right=0.99, bottom=0.34, top=0.88)
        png_path = output_dir / f"{file_prefix}_{factor}_brainmaps_glasser_lollipop.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        print(f"Saved {png_path}")

    n_f = len(factors)
    fig, axes = plt.subplots(
        n_f, 1, figsize=_lollipop_figsize(len(BRAINMAPS_GLASSER_COLUMNS), n_rows=n_f)
    )
    axes = np.atleast_1d(axes)
    for ax, factor in zip(axes, factors):
        corr_df = all_corr[factor]
        _plot_lollipop_ax(
            ax,
            corr_df,
            title=f"{factor} (n={int(corr_df['n_rois'].max())} Glasser ROIs)",
            show_ylabel=True,
        )
    axes[-1].set_xlabel("Glasser360 multimodal map", fontsize=11)
    fig.suptitle("Factor spatial profile vs brainmaps_glasser.csv", fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.14, top=0.93, hspace=0.38)
    combined_path = output_dir / f"{file_prefix}_brainmaps_glasser_lollipops_combined.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {combined_path}")

    stacked = pd.concat(
        [df.assign(factor=f) for f, df in all_corr.items()],
        ignore_index=True,
    )
    stacked_path = output_dir / f"{file_prefix}_brainmaps_glasser_correlations_all_factors.csv"
    stacked.to_csv(stacked_path, index=False)
    print(f"Saved {stacked_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lollipop plots: control-mean factor scores vs brainmaps_glasser.csv"
    )
    parser.add_argument(
        "--factor-scores",
        type=Path,
        default=DEFAULT_GM_FACTOR_SCORES,
        help="CSV of control-mean GM factor scores (rows=ROI, cols=F1..Fn)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG/CSV outputs",
    )
    parser.add_argument(
        "--file-prefix",
        default=DEFAULT_FILE_PREFIX,
        help="Filename prefix for outputs",
    )
    args = parser.parse_args()
    plot_factor_brainmap_lollipops(
        args.factor_scores,
        args.output_dir,
        args.file_prefix,
    )


if __name__ == "__main__":
    main()
