#!/usr/bin/env python3
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
Correlate control factor z-scores (per ROI) with DWI QC, stratified by dataset.

Three QC measures:
  - Neighbor correlation: qsiprep raw_neighbor_corr (penn, hcpaging); HCP-YA raw_dwi_neighbor_corr.
  - Head motion: qsiprep mean_fd (penn, hcpaging); HCP-YA meanRMS_rel-lastvol.
  - T1post-DWI contrast: qsiprep t1post_dwi_contrast (penn, hcpaging only; not available for HCP-YA).

Output per measure: CSV of correlations, summary strip plots (|r| and raw r), and QC distribution.
"""

import argparse
import os
from os.path import join as ospj
import glob
import numpy as np
import pandas as pd
from scipy import stats

FACTOR_Z_SCORES_DIR = ospj(PROJECT_ROOT, "derivatives/analysis/factor_z-scores/factor_z_scores")
QSIPREP_DIR = ospj(PROJECT_ROOT, "derivatives/qsiprep")
HCPYA_QC_CSV = ospj(PROJECT_ROOT, "data/hcpya/qc/hcpya_dwi_qc.csv")
OUTPUT_DIR = ospj(PROJECT_ROOT, "derivatives/analysis/qc")
DEFAULT_FACTORS = ["F1", "F2", "F3"]
CONTROL_GROUPS = ["penn_controls", "hcpya", "hcpaging"]


def load_qc_penn() -> pd.DataFrame:
    """Load QC from qsiprep TSVs: penn_controls, first session per subject; neighbor corr and mean_fd."""
    pattern = ospj(QSIPREP_DIR, "penn_controls", "sub-*", "ses-*", "dwi", "*_space-ACPC_desc-image_qc.tsv")
    files = sorted(glob.glob(pattern))
    rows = []
    seen_subjects = set()
    for path in files:
        parts = path.split(os.sep)
        sub = next((p for p in parts if p.startswith("sub-") and "ses-" not in p), None)
        ses = next((p for p in parts if p.startswith("ses-")), None)
        if not sub or not ses:
            continue
        if sub in seen_subjects:
            continue
        try:
            df = pd.read_csv(path, sep="\t")
            if "raw_neighbor_corr" not in df.columns:
                continue
            val = df["raw_neighbor_corr"].iloc[0]
            if pd.isna(val):
                continue
            motion = float(df["mean_fd"].iloc[0]) if "mean_fd" in df.columns else np.nan
            contrast = (
                float(df["t1post_dwi_contrast"].iloc[0])
                if "t1post_dwi_contrast" in df.columns
                else np.nan
            )
            rows.append({
                "subject": sub,
                "group": "penn_controls",
                "qc_value": float(val),
                "qc_motion": motion,
                "qc_t1post_contrast": contrast,
            })
            seen_subjects.add(sub)
        except Exception:
            continue
    return pd.DataFrame(rows)


def load_qc_hcpaging() -> pd.DataFrame:
    """Load QC from qsiprep TSVs: hcpaging; neighbor corr and mean_fd."""
    pattern = ospj(QSIPREP_DIR, "hcpaging", "sub-*", "dwi", "*_space-ACPC_desc-image_qc.tsv")
    files = sorted(glob.glob(pattern))
    rows = []
    for path in files:
        parts = path.split(os.sep)
        sub = next((p for p in parts if p.startswith("sub-")), None)
        if not sub:
            continue
        try:
            df = pd.read_csv(path, sep="\t")
            if "raw_neighbor_corr" not in df.columns:
                continue
            val = df["raw_neighbor_corr"].iloc[0]
            if pd.isna(val):
                continue
            motion = float(df["mean_fd"].iloc[0]) if "mean_fd" in df.columns else np.nan
            contrast = (
                float(df["t1post_dwi_contrast"].iloc[0])
                if "t1post_dwi_contrast" in df.columns
                else np.nan
            )
            rows.append({
                "subject": sub,
                "group": "hcpaging",
                "qc_value": float(val),
                "qc_motion": motion,
                "qc_t1post_contrast": contrast,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def load_qc_hcpya() -> pd.DataFrame:
    """Load QC from hcpya_dwi_qc.csv: sub -> sub-{id}; raw_dwi_neighbor_corr and meanRMS_rel-lastvol."""
    df = pd.read_csv(HCPYA_QC_CSV)
    if "raw_dwi_neighbor_corr" not in df.columns:
        return pd.DataFrame()
    df = df.rename(columns={"raw_dwi_neighbor_corr": "qc_value"})
    if "meanRMS_rel-lastvol" in df.columns:
        df["qc_motion"] = df["meanRMS_rel-lastvol"]
    else:
        df["qc_motion"] = np.nan
    if "sub" in df.columns:
        df["subject"] = df["sub"].astype(int).astype(str).map(lambda x: f"sub-{x}")
    else:
        return pd.DataFrame()
    df["group"] = "hcpya"
    df["qc_t1post_contrast"] = np.nan
    return df[["subject", "group", "qc_value", "qc_motion", "qc_t1post_contrast"]].dropna(subset=["qc_value"])


def load_qc_all() -> pd.DataFrame:
    """Concatenate QC from penn, hcpaging, hcpya."""
    penn = load_qc_penn()
    hcpaging = load_qc_hcpaging()
    hcpya = load_qc_hcpya()
    out = pd.concat([penn, hcpaging, hcpya], ignore_index=True)
    if "qc_motion" not in out.columns:
        out["qc_motion"] = np.nan
    if "qc_t1post_contrast" not in out.columns:
        out["qc_t1post_contrast"] = np.nan
    return out


def load_z_scores(factor: str) -> pd.DataFrame:
    """Load controls_{factor}_z_scores.csv (subject, group, ROI columns)."""
    path = ospj(FACTOR_Z_SCORES_DIR, f"controls_{factor}_z_scores.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    id_cols = [c for c in df.columns if c in ("subject", "group")]
    roi_cols = [c for c in df.columns if c not in id_cols]
    long = df.melt(id_vars=id_cols, value_vars=roi_cols, var_name="roi", value_name="z_score")
    long["factor"] = factor
    return long.dropna(subset=["z_score"])


def run_correlations(
    qc_df: pd.DataFrame,
    factors: list[str],
    qc_col: str = "qc_value",
) -> pd.DataFrame:
    """Merge z-scores with QC by subject/group; for each (group, ROI, factor) compute Pearson r, p, n."""
    if qc_col not in qc_df.columns:
        return pd.DataFrame()
    qc_df = qc_df.dropna(subset=[qc_col])
    results = []
    for factor in factors:
        z_df = load_z_scores(factor)
        if z_df.empty:
            continue
        merged = z_df.merge(
            qc_df[["subject", "group", qc_col]],
            on=["subject", "group"],
            how="inner",
        )
        if merged.empty:
            continue
        for (g, roi), grp in merged.groupby(["group", "roi"]):
            if grp["z_score"].nunique() < 2 or grp[qc_col].nunique() < 2:
                continue
            r, p = stats.pearsonr(grp["z_score"], grp[qc_col])
            n = len(grp)
            results.append({"factor": factor, "group": g, "roi": roi, "r": r, "p": p, "n": n})
    return pd.DataFrame(results)


def _add_mean_sd_indicators(
    ax,
    plot_df: pd.DataFrame,
    group_order: list,
    factor_order: list,
    factor_offsets: np.ndarray | None = None,
    x_centers: dict | None = None,
    y_col: str = "plot_y",
) -> None:
    """Add minimalist mean (horizontal) and SEM (vertical) black lines per strip.
    If x_centers is provided, it should map (gi, fi) -> x position for overlay on points.
    y_col: column name for the plotted metric (mean/SEM computed from it)."""
    summary = plot_df.groupby(["group", "factor"], as_index=False).agg(
        mean_y=(y_col, "mean"),
        sd_y=(y_col, "std"),
        n_y=(y_col, "count"),
    )
    summary["sd_y"] = summary["sd_y"].fillna(0)
    summary["sem_y"] = summary["sd_y"] / np.sqrt(summary["n_y"].clip(lower=1))

    n_factors = len(factor_order)
    if factor_offsets is None:
        dodge_width = 0.8
        factor_offsets = np.linspace(-dodge_width / 2, dodge_width / 2, n_factors, endpoint=(n_factors == 1))

    for gi, g in enumerate(group_order):
        for fi, factor in enumerate(factor_order):
            row = summary[(summary["group"] == g) & (summary["factor"] == factor)]
            if len(row) == 0:
                continue
            mean_val = row["mean_y"].iloc[0]
            sem_val = row["sem_y"].iloc[0]
            if x_centers is not None and (gi, fi) in x_centers:
                x_center = x_centers[(gi, fi)]
            else:
                x_center = gi + factor_offsets[fi]
            seg_half = 0.04
            ax.plot(
                [x_center - seg_half, x_center + seg_half],
                [mean_val, mean_val],
                color="black",
                linewidth=1.5,
                zorder=5,
            )
            y_lo = mean_val - sem_val
            y_hi = mean_val + sem_val
            ax.plot(
                [x_center, x_center],
                [y_lo, y_hi],
                color="black",
                linewidth=1,
                zorder=5,
            )


def _add_significance_asterisks(
    ax,
    corr_df: pd.DataFrame,
    group_order: list,
    factor_order: list,
    x_centers: dict,
) -> None:
    """For raw r plot: one-sample t-test of r vs 0 per (group, factor); add * ** *** above strip if significant."""
    n_factors = len(factor_order)
    default_offset = (np.arange(n_factors) - (n_factors - 1) / 2) * (0.8 / max(n_factors, 1))
    y_top = 0.82  # lower so asterisk sits nearer the strip
    for gi, g in enumerate(group_order):
        for fi, factor in enumerate(factor_order):
            subset = corr_df[(corr_df["group"] == g) & (corr_df["factor"] == factor)]
            if len(subset) < 2:
                continue
            r_vals = subset["r"].values
            try:
                _, p = stats.ttest_1samp(r_vals, 0)
            except Exception:
                continue
            if p >= 0.05:
                continue
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            else:
                star = "*"
            x_center = x_centers.get((gi, fi), gi + default_offset[fi])
            ax.text(x_center, y_top, star, ha="center", va="bottom", fontsize=10, zorder=6)


def _one_stripplot(
    corr_df: pd.DataFrame,
    output_path: str,
    use_abs: bool,
    groups: list[str] | None = None,
) -> None:
    """Single strip plot: y = |r| or r, x = group, hue = factor. use_abs=True -> |r|."""
    import matplotlib.pyplot as plt

    if corr_df.empty or "factor" not in corr_df.columns or "group" not in corr_df.columns or "r" not in corr_df.columns:
        return

    plot_df = corr_df.copy()
    plot_df["abs_r"] = plot_df["r"].abs()
    plot_df["plot_y"] = plot_df["abs_r"] if use_abs else plot_df["r"]
    if groups is None:
        groups = CONTROL_GROUPS
    else:
        groups = [g for g in groups if g in plot_df["group"].unique()]
        if not groups:
            return
    factors = [f for f in DEFAULT_FACTORS if f in plot_df["factor"].unique()] or sorted(plot_df["factor"].unique())
    # Sort by group (same order as stripplot) so collection point order matches (group, factor) indexing
    group_order = dict(zip(groups, range(len(groups))))
    plot_df = plot_df.sort_values(by="group", key=lambda c: c.map(group_order)).reset_index(drop=True)

    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.stripplot(
            data=plot_df,
            x="group",
            y="plot_y",
            hue="factor",
            order=groups,
            hue_order=factors,
            dodge=True,
            jitter=0.2,
            alpha=0.35,
            size=3,
            ax=ax,
        )
        # Use mean x from collection per (group, factor). First group aligns; use its dodge
        # offset for others: x(gi, fi) = x(0, fi) + gi (categories at 0, 1, 2).
        x_centers = {}
        n_g = len(groups)
        for fi, factor in enumerate(factors):
            if fi >= len(ax.collections):
                break
            col = ax.collections[fi]
            xy = col.get_offsets()
            if len(xy) == 0:
                continue
            try:
                xy_data = ax.transData.inverted().transform(col.get_offset_transform().transform(xy))
            except Exception:
                xy_data = xy
            x_coords = xy_data[:, 0]
            group_id = np.clip(np.round(x_coords).astype(int), 0, n_g - 1)
            for gi in range(n_g):
                in_cat = (group_id == gi)
                if in_cat.any():
                    x_centers[(gi, fi)] = float(x_coords[in_cat].mean())
        # If first group is correct but others wrong (e.g. transform/scale), fix: same dodge per factor.
        for fi in range(len(factors)):
            if (0, fi) not in x_centers:
                continue
            x0 = x_centers[(0, fi)]
            for gi in range(1, n_g):
                x_centers[(gi, fi)] = x0 + gi
        _add_mean_sd_indicators(ax, plot_df, groups, factors, x_centers=x_centers, y_col="plot_y")
    except ImportError:
        fig, ax = plt.subplots(figsize=(8, 5))
        n_factors = len(factors)
        width = 0.8 / max(n_factors, 1)
        factor_offsets = (np.arange(n_factors) - (n_factors - 1) / 2) * width
        x_centers = {(gi, fi): gi + factor_offsets[fi] for gi in range(len(groups)) for fi in range(len(factors))}
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_factors, 10)))[:n_factors]
        for fi, factor in enumerate(factors):
            for gi, g in enumerate(groups):
                subset = plot_df[(plot_df["group"] == g) & (plot_df["factor"] == factor)]
                if subset.empty:
                    continue
                x_jitter = np.random.uniform(-width / 2, width / 2, size=len(subset))
                x_pos = gi + (fi - (n_factors - 1) / 2) * width + x_jitter
                ax.scatter(
                    x_pos,
                    subset["plot_y"],
                    c=[colors[fi]],
                    alpha=0.35,
                    s=15,
                    label=factor if gi == 0 else None,
                )
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=15, ha="right")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Factor", bbox_to_anchor=(1.02, 1), loc="upper left")
        _add_mean_sd_indicators(ax, plot_df, groups, factors, factor_offsets=factor_offsets, y_col="plot_y")

    if not use_abs:
        _add_significance_asterisks(ax, corr_df, groups, factors, x_centers)

    ax.set_ylabel("|r| (absolute correlation)" if use_abs else "r (correlation)")
    ax.set_xlabel("Dataset")
    ax.set_title("Factor z-scores vs DWI QC: " + ("|r|" if use_abs else "r") + " per ROI")
    if use_abs:
        ax.set_ylim(0, None)
    else:
        ax.set_ylim(-1, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_summary_stripplots(
    corr_df: pd.DataFrame,
    base_path: str,
    groups: list[str] | None = None,
) -> None:
    """Write two summary plots: one with |r|, one with raw r. base_path is full path for |r| version."""
    base = base_path.replace(".png", "").rstrip("_abs_r").rstrip("_r")
    _one_stripplot(corr_df, f"{base}_abs_r.png", use_abs=True, groups=groups)
    _one_stripplot(corr_df, f"{base}_r.png", use_abs=False, groups=groups)


def groups_with_qc_data(qc_df: pd.DataFrame, qc_col: str) -> list[str]:
    """Return CONTROL_GROUPS subsets that have at least one non-null value for qc_col."""
    return [
        g for g in CONTROL_GROUPS
        if g in qc_df["group"].values and qc_df.loc[qc_df["group"] == g, qc_col].notna().any()
    ]


def plot_qc_summary(
    qc_df: pd.DataFrame,
    output_path: str,
    y_col: str = "qc_value",
    y_label: str = "QC (neighbor correlation)",
    title: str = "DWI QC by dataset",
    y_lim: tuple = (0.6, None),
    groups: list[str] | None = None,
) -> None:
    """Strip plot of QC measure by group (one point per subject), mean ± SEM per group."""
    import matplotlib.pyplot as plt

    if qc_df.empty or "group" not in qc_df.columns or y_col not in qc_df.columns:
        return
    if groups is None:
        groups = CONTROL_GROUPS
    plot_df = qc_df[qc_df["group"].isin(groups)].dropna(subset=[y_col]).copy()
    if plot_df.empty:
        return

    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.stripplot(
            data=plot_df,
            x="group",
            y=y_col,
            order=groups,
            jitter=0.2,
            alpha=0.35,
            size=3,
            color="steelblue",
            ax=ax,
        )
        summary = plot_df.groupby("group", as_index=False).agg(
            mean_y=(y_col, "mean"),
            sd_y=(y_col, "std"),
            n_y=(y_col, "count"),
        )
        summary["sem_y"] = summary["sd_y"].fillna(0) / np.sqrt(summary["n_y"].clip(lower=1))
        for gi, g in enumerate(groups):
            row = summary[summary["group"] == g]
            if len(row) == 0:
                continue
            mean_val = row["mean_y"].iloc[0]
            sem_val = row["sem_y"].iloc[0]
            x_center = gi
            seg_half = 0.04
            ax.plot(
                [x_center - seg_half, x_center + seg_half],
                [mean_val, mean_val],
                color="black",
                linewidth=1.5,
                zorder=5,
            )
            ax.plot(
                [x_center, x_center],
                [mean_val - sem_val, mean_val + sem_val],
                color="black",
                linewidth=1,
                zorder=5,
            )
    except ImportError:
        fig, ax = plt.subplots(figsize=(8, 5))
        for gi, g in enumerate(groups):
            subset = plot_df[plot_df["group"] == g]
            if subset.empty:
                continue
            x_jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
            ax.scatter(gi + x_jitter, subset[y_col], alpha=0.35, s=15, color="steelblue")
        summary = plot_df.groupby("group", as_index=False).agg(
            mean_y=(y_col, "mean"),
            sd_y=(y_col, "std"),
            n_y=(y_col, "count"),
        )
        summary["sem_y"] = summary["sd_y"].fillna(0) / np.sqrt(summary["n_y"].clip(lower=1))
        for gi, g in enumerate(groups):
            row = summary[summary["group"] == g]
            if len(row) == 0:
                continue
            mean_val = row["mean_y"].iloc[0]
            sem_val = row["sem_y"].iloc[0]
            x_center = gi
            seg_half = 0.04
            ax.plot([x_center - seg_half, x_center + seg_half], [mean_val, mean_val], color="black", linewidth=1.5, zorder=5)
            ax.plot([x_center, x_center], [mean_val - sem_val, mean_val + sem_val], color="black", linewidth=1, zorder=5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=15, ha="right")

    ax.set_ylabel(y_label)
    ax.set_xlabel("Dataset")
    ax.set_title(title)
    if y_lim[0] is not None or y_lim[1] is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_qc_correlation(qc_df: pd.DataFrame, output_path: str) -> None:
    """Scatter plot per group: DWI neighbor correlation vs head motion, with line of best fit and Spearman r/p."""
    import matplotlib.pyplot as plt

    if qc_df.empty or "qc_value" not in qc_df.columns or "qc_motion" not in qc_df.columns:
        return
    plot_df = qc_df.dropna(subset=["qc_value", "qc_motion"]).copy()
    if plot_df.empty:
        return
    groups = CONTROL_GROUPS
    n_g = len(groups)
    fig, axes = plt.subplots(1, n_g, figsize=(4 * n_g, 5), squeeze=False)
    axes = axes.flatten()
    for gi, g in enumerate(groups):
        ax = axes[gi]
        sub = plot_df[plot_df["group"] == g]
        if sub.empty:
            ax.set_visible(False)
            continue
        x = sub["qc_value"].values
        y = sub["qc_motion"].values
        ax.scatter(x, y, alpha=0.35, s=15, color="steelblue", edgecolors="none")
        # Spearman correlation and p-value
        r, p = stats.spearmanr(x, y)
        # Line of best fit (linear regression)
        slope, intercept, _, _, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5, zorder=5)
        p_str = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.05, 0.95, f"r = {r:.3f}\n{p_str}", transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("DWI neighbor correlation")
        ax.set_ylabel("Head motion (mean FD / meanRMS rel-lastvol)")
        ax.set_title(g)
        # Rescale x-axis per group to min/max of that group's neighbor correlation
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            margin = (x_max - x_min) * 0.02
            ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(0, None)
    fig.suptitle("DWI neighbor correlation vs head motion by dataset", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Correlate control factor z-scores with QC (stratified by dataset)."
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=DEFAULT_FACTORS,
        help=f"Factors to use (default: {DEFAULT_FACTORS})",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory for CSV and plots",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    qc_df = load_qc_all()
    print(f"Loaded QC for {len(qc_df)} subjects ({qc_df.groupby('group').size().to_dict()})")

    qc_measures = [
        {
            "qc_col": "qc_value",
            "csv_stem": "factor_z_scores_vs_qc_correlations",
            "summary_stem": "qc_summary",
            "y_label": "QC (neighbor correlation)",
            "title": "DWI QC by dataset (neighbor correlation)",
            "y_lim": (0.6, None),
            "groups": None,
        },
        {
            "qc_col": "qc_motion",
            "csv_stem": "factor_z_scores_vs_qc_motion",
            "summary_stem": "qc_motion_summary",
            "y_label": "Head motion (mean FD / meanRMS rel-lastvol)",
            "title": "Head motion by dataset",
            "y_lim": (0, None),
            "groups": None,
        },
        {
            "qc_col": "qc_t1post_contrast",
            "csv_stem": "factor_z_scores_vs_qc_t1post_contrast",
            "summary_stem": "qc_t1post_contrast_summary",
            "y_label": "T1post-DWI contrast",
            "title": "T1post-DWI contrast by dataset",
            "y_lim": (None, None),
            "groups": ["penn_controls", "hcpaging"],
        },
    ]

    for m in qc_measures:
        qc_col = m["qc_col"]
        n_valid = qc_df[qc_col].notna().sum()
        if n_valid == 0:
            print(f"Skipping {qc_col}: no valid values")
            continue
        groups = m.get("groups") or groups_with_qc_data(qc_df, qc_col)
        corr_df = run_correlations(qc_df, args.factors, qc_col=qc_col)
        csv_path = ospj(args.output_dir, f"{m['csv_stem']}.csv")
        corr_df.to_csv(csv_path, index=False)
        print(f"Saved {qc_col} correlations to {csv_path}")

        for (g, f), sub in corr_df.groupby(["group", "factor"]):
            n_sig = (sub["p"] < 0.05).sum()
            print(f"  {g} {f}: {n_sig}/{len(sub)} ROIs with p < 0.05")

        plot_base = ospj(args.output_dir, f"{m['csv_stem']}_summary")
        plot_summary_stripplots(corr_df, f"{plot_base}.png", groups=groups)
        print(f"Saved summary plots to {plot_base}_abs_r.png and {plot_base}_r.png")

        summary_path = ospj(args.output_dir, f"{m['summary_stem']}.png")
        plot_qc_summary(
            qc_df,
            summary_path,
            y_col=qc_col,
            y_label=m["y_label"],
            title=m["title"],
            y_lim=m["y_lim"],
            groups=groups,
        )
        print(f"Saved QC summary to {summary_path}")

    qc_corr_path = ospj(args.output_dir, "qc_correlation.png")
    plot_qc_correlation(qc_df, qc_corr_path)
    print(f"Saved QC correlation plot to {qc_corr_path}")


if __name__ == "__main__":
    main()
