import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
"""
Scatter of tract-profile MD at a single node vs age (females, control cohorts), plus
an along-tract mean ILF profile with end1/core/end2 node shading.

Data: derivatives/gam/pyafq (GAM-adjusted / normative outputs per subject-node).
Styling: cohort colors and order aligned with pyafq_covbat_example/plot_covbat_example.ipynb.
"""

import json
import os
from os.path import join as ospj

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.transforms import blended_transform_factory
from scipy.interpolate import UnivariateSpline


def _register_georgia() -> None:
    """Register Georgia TTFs with matplotlib (FSL/bundled matplotlib often skips system fonts)."""
    home = os.path.expanduser("~")
    for name in (
        "georgia.ttf",
        "georgiab.ttf",
        "georgiai.ttf",
        "georgiaz.ttf",
        "Georgia.ttf",
    ):
        for d in (
            "/usr/share/fonts/truetype",
            "/usr/share/fonts/truetype/msttcorefonts",
            "/usr/share/fonts/TTF",
            ospj(home, ".fonts"),
            ospj(home, ".local/share/fonts"),
        ):
            path = ospj(d, name)
            if os.path.isfile(path):
                try:
                    font_manager.fontManager.addfont(path)
                except (OSError, ValueError):
                    pass


_register_georgia()
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Georgia", "DejaVu Serif", "DejaVu Sans"]

# =============================================================================
# CONFIGURATION
# =============================================================================

GAM_PYAFQ_DIR = ospj(str(gam_dir()), "pyafq")
OUTPUT_DIR = ospj(str(analysis_dir()), "profile_thirds_example")

WM_ATLAS = "HCP1065"
TRACT = "ILF_L"
SCALAR = "dti_md"
NODE_INDEX = 50
NODE_COL = f"node{NODE_INDEX}"

# HCP1065 along-tract thirds (same as factor_analysis / factor_z-scores)
N_NODES_PROFILE = 100
# end1: nodes 1–34, core: 35–66, end2: 67–100
NODE_SEGMENTS_END1 = (1, 34)
NODE_SEGMENTS_CORE = (35, 66)
NODE_SEGMENTS_END2 = (67, 100)

# ILF mean profile: wide aspect + large type for slides / posters
ILF_PROFILE_FIGSIZE = (32, 16)
ILF_PROFILE_AXIS_LABEL_FONTSIZE = 96
ILF_PROFILE_TICK_FONTSIZE = 72
ILF_PROFILE_SEGMENT_LABEL_FONTSIZE = 96
ILF_PROFILE_LINEWIDTH = 12
SEGMENT_LABELS = False

# Cohort draw order and point colors (match plot_covbat_example.ipynb)
CONTROL_GROUP_ORDER = ["penn_controls", "hcpya", "hcpaging"]

GROUP_COLORS = {
    "penn_controls": "#0173B2",
    "hcpya": "#029E73",
    "hcpaging": "#DE8F05",
}

SCALAR_COLORS_JSON = ospj(PROJECT_ROOT, "data/metadata/scalar_labels_to_colors.json")
SCALAR_HUMAN_JSON = ospj(PROJECT_ROOT, "data/metadata/scalar_labels_to_human.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _scalar_human_and_color(scalar_key: str) -> tuple[str, str]:
    with open(SCALAR_HUMAN_JSON, encoding="utf-8") as f:
        human_map = json.load(f)
    with open(SCALAR_COLORS_JSON, encoding="utf-8") as f:
        color_map = json.load(f)
    raw = human_map.get(scalar_key, scalar_key.replace("_", " "))
    if raw.endswith(")") and " (" in raw:
        label = raw.rsplit(" (", 1)[0].strip()
    else:
        label = raw
    color = color_map.get(scalar_key, "#000000")
    return label, color


def load_ilf_gam_mean(scalar: str = SCALAR) -> pd.DataFrame:
    """Load subject-level GAM table for ILF_L mean statistic (all nodes + preds)."""
    csv_path = ospj(
        GAM_PYAFQ_DIR,
        WM_ATLAS,
        TRACT,
        f"{TRACT}_{scalar}_stat-mean_gam.csv",
    )
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"GAM pyAFQ file not found: {csv_path}")
    return pd.read_csv(csv_path)


def plot_node_md_vs_age_females(
    output_path: str,
    *,
    node_col: str = NODE_COL,
    scalar: str = SCALAR,
) -> None:
    """
    Age (x) vs observed MD at ``node_col`` (y), females only, control cohorts, colored by group.

    Overlays GAM predicted mean (grey curve) and a band of ± one pooled residual SD across
    those controls (observed minus predicted at this node).
    """
    df = load_ilf_gam_mean(scalar)
    if node_col not in df.columns:
        raise KeyError(f"Column {node_col!r} not in GAM table (check node index).")
    pred_col = f"{node_col}_pred"
    if pred_col not in df.columns:
        raise KeyError(f"Column {pred_col!r} not in GAM table (expected GAM prediction).")

    df = df.loc[df["sex"] == "F"].copy()
    df = df.loc[df["group"] != "penn_epilepsy"].copy()

    obs = df[node_col].astype(float)
    pred = df[pred_col].astype(float)
    residuals = obs - pred
    residual_sd = float(residuals.std()) if len(df) > 1 else 0.0

    y_label, y_color = _scalar_human_and_color(scalar)

    fig, ax = plt.subplots(figsize=(14, 10))

    df_sorted = df.sort_values("age")
    ages = df_sorted["age"].astype(float).to_numpy()
    preds_sorted = df_sorted[pred_col].astype(float).to_numpy()

    if len(df_sorted) > 3:
        spline = UnivariateSpline(ages, preds_sorted, s=len(df_sorted) * 0.1)
        age_smooth = np.linspace(float(ages.min()), float(ages.max()), 200)
        pred_smooth = spline(age_smooth)
    else:
        age_smooth = ages
        pred_smooth = preds_sorted

    ax.fill_between(
        age_smooth,
        pred_smooth - residual_sd,
        pred_smooth + residual_sd,
        color="grey",
        alpha=0.28,
        zorder=1,
        linewidth=0,
    )
    ax.plot(age_smooth, pred_smooth, color="grey", linewidth=2.0, zorder=2)

    for g in CONTROL_GROUP_ORDER:
        sub = df.loc[df["group"] == g]
        if sub.empty:
            continue
        color = GROUP_COLORS.get(g, "#666666")
        ax.scatter(
            sub["age"].astype(float),
            sub[node_col].astype(float),
            c=color,
            s=100,
            alpha=0.65,
            edgecolors="none",
            zorder=3,
        )

    ax.set_xlabel("Age (years)", fontsize=48)
    ax.set_ylabel(y_label, fontsize=48, color=y_color)
    ax.tick_params(axis="y", labelcolor="black")
    ax.set_title(
        f"Left ILF — {y_label} (node {NODE_INDEX})\n"
        f"n = {len(df)} females, control cohorts",
        fontsize=50,
    )
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.tick_params(axis="x", labelsize=48)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="y", labelsize=48)
    ax.grid(True, alpha=0.3, which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_ilf_mean_profile_with_node_segments(
    output_path: str,
    *,
    scalar: str = SCALAR,
    segment_labels: bool = SEGMENT_LABELS,
) -> None:
    """
    Along-tract mean profile (harmonized GAM table): pool all subjects and average observed
    values at each node, then shade end1 / core / end2 with distinct backgrounds.

    When ``segment_labels`` is True, draw End 1 / Core / End 2 text on each third.

    Node thirds match ``factor_analysis.py`` / ``factor_z-scores.py`` (100 nodes per tract).
    """
    df = load_ilf_gam_mean(scalar)
    node_cols = [f"node{i}" for i in range(1, N_NODES_PROFILE + 1)]
    missing = [c for c in node_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing node columns in GAM table (first few): {missing[:5]!r}")

    # Grand mean across all rows (all cohorts / groups in the harmonized table)
    profile = df[node_cols].astype(float).mean(axis=0).to_numpy()
    x = np.arange(1, N_NODES_PROFILE + 1, dtype=float)

    y_label, y_color = _scalar_human_and_color(scalar)

    fig, ax = plt.subplots(figsize=ILF_PROFILE_FIGSIZE)

    # End1/End2: light grey; core: white
    _grey_end = "#e8e8e8"
    _white_core = "#ffffff"
    segments = [
        ("End 1", NODE_SEGMENTS_END1[0], NODE_SEGMENTS_END1[1], _grey_end),
        ("Core", NODE_SEGMENTS_CORE[0], NODE_SEGMENTS_CORE[1], _white_core),
        ("End 2", NODE_SEGMENTS_END2[0], NODE_SEGMENTS_END2[1], _grey_end),
    ]
    for _name, lo, hi, face in segments:
        ax.axvspan(
            lo - 0.5,
            hi + 0.5,
            facecolor=face,
            edgecolor="none",
            alpha=1.0,
            zorder=0,
        )

    ax.plot(x, profile, color=y_color, linewidth=ILF_PROFILE_LINEWIDTH, zorder=3)

    if segment_labels:
        # x in data coords, y in axes coords — large labels centered in each segment band
        x_y_axes = blended_transform_factory(ax.transData, ax.transAxes)
        for name, lo, hi, _face in segments:
            xc = 0.5 * (lo + hi)
            ax.text(
                xc,
                0.94,
                name,
                transform=x_y_axes,
                ha="center",
                va="top",
                fontsize=ILF_PROFILE_SEGMENT_LABEL_FONTSIZE,
                fontweight="bold",
                color="#333333",
                zorder=2,
            )

    ax.set_xlim(0.5, N_NODES_PROFILE + 0.5)
    ax.set_xlabel("Along-tract segment", fontsize=ILF_PROFILE_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=ILF_PROFILE_AXIS_LABEL_FONTSIZE, color=y_color)
    ax.tick_params(axis="y", labelcolor="black")
    if segment_labels:
        _ymin, _ = ax.get_ylim()
        ax.set_ylim(_ymin, 0.84)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.tick_params(
        axis="x",
        labelsize=ILF_PROFILE_TICK_FONTSIZE,
    )
    ax.tick_params(
        axis="y",
        labelsize=ILF_PROFILE_TICK_FONTSIZE,
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, alpha=0.35, which="major", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    layout_rect = [0, 0, 1, 0.92] if segment_labels else None
    plt.tight_layout(rect=layout_rect)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    out = ospj(
        OUTPUT_DIR,
        f"{TRACT}_{SCALAR}_node{NODE_INDEX}_females_age_pyafq_gam.png",
    )
    plot_node_md_vs_age_females(out)
    out_nodes = ospj(
        OUTPUT_DIR,
        f"{TRACT}_{SCALAR}_mean_profile_nodes_harmonized_pyafq_gam.png",
    )
    plot_ilf_mean_profile_with_node_segments(
        out_nodes, segment_labels=SEGMENT_LABELS
    )


if __name__ == "__main__":
    main()
