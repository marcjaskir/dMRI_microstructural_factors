import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
"""Generate asymmetry_tle_covbat_pyafq.html: dataset summary, CovBat-GAM controls, and TLE tract profiles with flipping and effect sizes."""
from __future__ import annotations

import argparse
import html as html_module
import sys
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

# Import from sibling asymmetry_tle for config
_script_dir = Path(__file__).resolve().parents[0]
_sibling_dir = _script_dir.parent / "asymmetry_tle"
sys.path.insert(0, str(_sibling_dir))
import config as cfg

DEFAULT_PROJECT_ROOT = project_root()
DEFAULT_TRACTS = ["F", "UF", "C_PHP", "C_PH", "ILF", "TR_A", "CST"]
DEFAULT_SCALARS = ["dti_md", "dti_fa"]
NODES = list(range(1, 101))

GROUP_COLORS = {
    "penn_controls": "tab:green",
    "hcpya": "tab:orange",
    "hcpaging": "tab:purple",
}
EPILEPSY_GROUP = "penn_epilepsy"
TLE_COLORS = {"Left TLE": "tab:green", "Right TLE": "tab:blue"}
# GAM z-scores (rows 2–3) use red to indicate abnormality
TLE_Z_COLOR = "red"


# ---------------------------------------------------------------------------
# Helpers: tract metadata, end labels, flipping
# ---------------------------------------------------------------------------


def _get_tract_ends(
    tract_meta: pd.DataFrame, tract_label: str
) -> Tuple[Optional[str], Optional[str]]:
    """Return (end1, end2) anatomical direction labels for a tract, or (None, None)."""
    if tract_meta.empty or "label" not in tract_meta.columns:
        return None, None
    row = tract_meta[tract_meta["label"] == tract_label]
    if row.empty:
        return None, None
    r = row.iloc[0]
    end1 = None
    end2 = None
    if "end1" in r.index and pd.notna(r["end1"]) and str(r["end1"]).strip().upper() not in ("NA", ""):
        end1 = str(r["end1"]).strip()
    if "end2" in r.index and pd.notna(r["end2"]) and str(r["end2"]).strip().upper() not in ("NA", ""):
        end2 = str(r["end2"]).strip()
    return end1, end2


def _get_tract_label_human(tract_meta: pd.DataFrame, tract_label: str) -> str:
    """Return human-readable name for a tract label, e.g. F_L -> 'Fornix (left)', F_R -> 'Fornix (right)'."""
    if tract_meta.empty or "label" not in tract_meta.columns or "name" not in tract_meta.columns:
        return tract_label
    row = tract_meta[tract_meta["label"] == tract_label]
    if row.empty:
        return tract_label
    name = str(row.iloc[0]["name"])
    if name.endswith("_L"):
        base = name[:-2]
    elif name.endswith("_R"):
        base = name[:-2]
    else:
        base = name
    base = base.replace("_", " ").strip()
    if tract_label.endswith("_L"):
        return f"{base} (left)" if base else tract_label
    if tract_label.endswith("_R"):
        return f"{base} (right)" if base else tract_label
    return base or tract_label


def _get_tract_base_human(tract_meta: pd.DataFrame, tract_base: str) -> str:
    """Return human-readable name for a tract base, e.g. F -> 'Fornix'."""
    if tract_meta.empty or "label" not in tract_meta.columns or "name" not in tract_meta.columns:
        return tract_base
    left_label = f"{tract_base}_L"
    row = tract_meta[tract_meta["label"] == left_label]
    if row.empty:
        return tract_base
    name = str(row.iloc[0]["name"])
    for suffix in ("_L", "_R"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("_", " ")


def get_flip_left_right(
    tract_meta: pd.DataFrame, tract_base: str
) -> Tuple[bool, bool]:
    """Return (flip_L, flip_R) for tract profile plotting so anatomical ends face center.
    Rules from structural_tractometry_context: (1) one subcortex + cortex/WM -> left flipped;
    (2) both cortex -> right flipped; (3) both subcortex -> left flipped.
    """
    left_label = f"{tract_base}_L"
    right_label = f"{tract_base}_R"
    info_l = tract_meta[tract_meta["label"] == left_label]
    info_r = tract_meta[tract_meta["label"] == right_label]
    if info_l.empty or info_r.empty:
        return False, False
    r_l = info_l.iloc[0]
    r_r = info_r.iloc[0]
    e1_loc = str(r_l.get("end1_loc", "") or "").strip().lower()
    e2_loc = str(r_l.get("end2_loc", "") or "").strip().lower()
    sub = "subcortex"
    ctx = "cortex"
    wm = "wm"
    if e1_loc == sub and (e2_loc == ctx or e2_loc == wm):
        return True, False
    if (e1_loc == ctx or e1_loc == wm) and e2_loc == sub:
        return True, False
    if e1_loc == ctx and e2_loc == ctx:
        return False, True
    if e1_loc == sub and e2_loc == sub:
        return True, False
    return False, False


# Single-letter anatomical direction -> human-readable (from structural_tractometry_context)
_END_LABEL_TO_HUMAN = {
    "A": "Anterior",
    "P": "Posterior",
    "I": "Inferior",
    "S": "Superior",
    "M": "Medial",
    "L": "Lateral",
}


def _end_label_human(letter: str) -> str:
    """Return human-readable anatomical label for a single letter, or the letter if unknown."""
    if not letter:
        return letter
    return _END_LABEL_TO_HUMAN.get(str(letter).strip().upper(), letter)


def _apply_end_ticks(
    ax: plt.Axes,
    end_labels: Optional[Tuple[str, str]],
    flip: bool = False,
) -> None:
    """Set x-axis ticks with anatomical end labels at nodes 1 and 100 (human-readable, bold)."""
    ticks = [1, 25, 50, 75, 100]
    if end_labels and end_labels[0] and end_labels[1]:
        e1 = _end_label_human(end_labels[0])
        e2 = _end_label_human(end_labels[1])
        if flip:
            tick_labels = [f"100\n{e2}", "75", "50", "25", f"1\n{e1}"]
        else:
            tick_labels = [f"1\n{e1}", "25", "50", "75", f"100\n{e2}"]
    else:
        tick_labels = [str(t) for t in ticks]
    ax.set_xticks(ticks)
    labels = ax.set_xticklabels(tick_labels, fontsize=7)
    if end_labels and end_labels[0] and end_labels[1] and len(labels) >= 5:
        labels[0].set_fontweight("bold")
        labels[4].set_fontweight("bold")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _node_cols_before(scalar: str) -> List[str]:
    return [f"{scalar}_node{i}" for i in NODES]


def _node_cols_after() -> List[str]:
    return [f"node{i}" for i in NODES]


def _node_cols_z() -> List[str]:
    return [f"node{i}_z" for i in NODES]


def load_control_profiles_by_group(
    pyafq_gam_dir: Path,
    preharm_dir: Path,
    tract_label: str,
    scalar: str,
    before: bool,
    use_z: bool = False,
) -> Optional[Dict[str, Tuple[List[str], np.ndarray]]]:
    """Load profiles for control subjects only, grouped by group.
    Returns {group: (subject_ids, array shape (n, 100))} or None.
    If use_z=True, load GAM z-scores (node*_z) from the GAM CSV; before is ignored.
    """
    gam_path = pyafq_gam_dir / tract_label / f"{tract_label}_{scalar}_gam.csv"
    if not gam_path.exists():
        return None
    gam_df = pd.read_csv(gam_path, usecols=["sub", "group"])
    gam_df["sub"] = gam_df["sub"].astype(str)
    sub_to_group = dict(zip(gam_df["sub"], gam_df["group"]))
    control_subs = [s for s, g in sub_to_group.items() if g != EPILEPSY_GROUP]
    if not control_subs:
        return None

    if use_z:
        csv_path = gam_path
        cols = _node_cols_z()
    elif before:
        csv_path = preharm_dir / tract_label / f"{tract_label}_{scalar}_data.csv"
        cols = _node_cols_before(scalar)
    else:
        csv_path = gam_path
        cols = _node_cols_after()

    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["sub"] = df["sub"].astype(str)
    df = df[df["sub"].isin(set(control_subs))]
    if df.empty:
        return None
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None

    df["_group"] = df["sub"].map(sub_to_group).fillna("unknown")
    result: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for grp, grp_df in df.groupby("_group"):
        result[str(grp)] = (grp_df["sub"].tolist(), grp_df[cols].values.astype(float))
    return result if result else None


def load_tract_profiles(
    pyafq_gam_dir: Path,
    preharm_dir: Path,
    tract_label: str,
    scalar: str,
    subjects: List[str],
    before: bool,
    use_z: bool = False,
) -> Optional[Tuple[List[str], np.ndarray]]:
    """Load node profiles for given subjects. Returns (subject_ids, (n, 100)) or None.
    If use_z=True, load GAM z-scores from the GAM CSV; before is ignored.
    """
    if use_z:
        csv_path = pyafq_gam_dir / tract_label / f"{tract_label}_{scalar}_gam.csv"
        cols = _node_cols_z()
    elif before:
        csv_path = preharm_dir / tract_label / f"{tract_label}_{scalar}_data.csv"
        cols = _node_cols_before(scalar)
    else:
        csv_path = pyafq_gam_dir / tract_label / f"{tract_label}_{scalar}_gam.csv"
        cols = _node_cols_after()
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["sub"] = df["sub"].astype(str)
    df = df[df["sub"].isin(set(subjects))]
    if df.empty:
        return None
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None
    return df["sub"].tolist(), df[cols].values.astype(float)


def cohens_d_paired_per_node(ipsi: np.ndarray, contra: np.ndarray) -> np.ndarray:
    """Paired Cohen's d per node. ipsi/contra shape (n_subjects, 100). Returns (100,) with NaN where n<2 or std=0."""
    diff = ipsi - contra
    n = np.sum(np.isfinite(diff), axis=0)
    mean_d = np.nanmean(diff, axis=0)
    std_d = np.nanstd(diff, axis=0, ddof=1)
    d = np.full(diff.shape[1], np.nan)
    valid = (n >= 2) & (std_d > 0)
    d[valid] = mean_d[valid] / std_d[valid]
    return d


def _intersect_paired(
    left: Tuple[List[str], np.ndarray],
    right: Tuple[List[str], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Align (subjects, profiles) to shared subjects. Returns (left_arr, right_arr) matching row order."""
    left_subs, left_arr = left
    right_subs, right_arr = right
    left_idx = {s: i for i, s in enumerate(left_subs)}
    right_idx = {s: i for i, s in enumerate(right_subs)}
    shared = [s for s in left_subs if s in right_idx]
    if not shared:
        return np.empty((0, left_arr.shape[1])), np.empty((0, right_arr.shape[1]))
    l_rows = [left_idx[s] for s in shared]
    r_rows = [right_idx[s] for s in shared]
    return left_arr[l_rows], right_arr[r_rows]


def asymmetry_index_per_node(ipsi: np.ndarray, contra: np.ndarray) -> np.ndarray:
    """Asymmetry index (ipsi - contra) / (|ipsi| + |contra|) per node. Shape (n_subjects, n_nodes)."""
    denom = np.abs(ipsi) + np.abs(contra)
    with np.errstate(divide="ignore", invalid="ignore"):
        asym = np.where(denom > 0, (ipsi - contra) / denom, np.nan)
    return asym


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------


def _make_summary_section(
    pyafq_gam_dir: Path,
    left_tle_subs: List[str],
    right_tle_subs: List[str],
) -> str:
    """Build HTML with group table and age density plots (controls top, TLE bottom)."""
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return '<h2 id="summary">Dataset summary</h2><p>scipy required for density plots.</p>'

    gam_csvs = list(pyafq_gam_dir.glob("*/*_gam.csv"))
    sub_group: Dict[str, str] = {}
    sub_age: Dict[str, float] = {}
    for csv_path in gam_csvs:
        try:
            df = pd.read_csv(csv_path, usecols=["sub", "group", "age"])
            df["sub"] = df["sub"].astype(str)
            for _, row in df.iterrows():
                s = row["sub"]
                if s not in sub_group:
                    sub_group[s] = str(row["group"])
                if s not in sub_age and pd.notna(row["age"]):
                    sub_age[s] = float(row["age"])
        except Exception:
            continue

    ltle_set = set(left_tle_subs)
    rtle_set = set(right_tle_subs)
    groups = [
        ("Left TLE", ltle_set),
        ("Right TLE", rtle_set),
        ("penn_controls", {s for s, g in sub_group.items() if g == "penn_controls"}),
        ("hcpya", {s for s, g in sub_group.items() if g == "hcpya"}),
        ("hcpaging", {s for s, g in sub_group.items() if g == "hcpaging"}),
    ]

    table_rows = ""
    for label, subs in groups:
        ages = [sub_age[s] for s in subs if s in sub_age]
        n = len(subs)
        if ages:
            age_str = (
                f"{np.mean(ages):.1f} &plusmn; {np.std(ages, ddof=1):.1f} "
                f"({np.min(ages):.0f}&ndash;{np.max(ages):.0f})"
            )
        else:
            age_str = "N/A"
        table_rows += (
            f"<tr><td>{html_module.escape(label)}</td>"
            f"<td>{n}</td><td>{age_str}</td></tr>\n"
        )

    table_html = (
        '<table class="summary-table">\n'
        "<thead><tr><th>Group</th><th>n</th>"
        "<th>Age (mean &plusmn; SD, range)</th></tr></thead>\n"
        f"<tbody>\n{table_rows}</tbody>\n</table>\n"
    )

    colors = {
        "Left TLE": TLE_COLORS["Left TLE"],
        "Right TLE": TLE_COLORS["Right TLE"],
        "penn_controls": GROUP_COLORS["penn_controls"],
        "hcpya": GROUP_COLORS["hcpya"],
        "hcpaging": GROUP_COLORS["hcpaging"],
    }
    control_groups = [g for g in groups if g[0] not in ("Left TLE", "Right TLE")]
    patient_groups = [g for g in groups if g[0] in ("Left TLE", "Right TLE")]

    fig, (ax_ctrl, ax_pat) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    all_ages = [sub_age[s] for s in sub_age]
    x_lo = min(all_ages) - 5 if all_ages else 0
    x_hi = max(all_ages) + 5 if all_ages else 100
    xs = np.linspace(x_lo, x_hi, 300)

    for label, subs in control_groups:
        ages = np.array([sub_age[s] for s in subs if s in sub_age])
        if len(ages) < 2:
            continue
        kde = gaussian_kde(ages, bw_method=0.3)
        ax_ctrl.plot(xs, kde(xs), color=colors.get(label, "gray"), label=f"{label} (n={len(ages)})")
        ax_ctrl.fill_between(xs, kde(xs), color=colors.get(label, "gray"), alpha=0.15)
    ax_ctrl.set_ylabel("Density", fontsize=10)
    ax_ctrl.set_title("Controls", fontsize=11, fontweight="bold")
    ax_ctrl.legend(fontsize=8)

    for label, subs in patient_groups:
        ages = np.array([sub_age[s] for s in subs if s in sub_age])
        if len(ages) < 2:
            continue
        kde = gaussian_kde(ages, bw_method=0.3)
        ax_pat.plot(xs, kde(xs), color=colors.get(label, "gray"), label=f"{label} (n={len(ages)})")
        ax_pat.fill_between(xs, kde(xs), color=colors.get(label, "gray"), alpha=0.15)
    ax_pat.set_xlabel("Age (years)", fontsize=10)
    ax_pat.set_ylabel("Density", fontsize=10)
    ax_pat.set_title("TLE", fontsize=11, fontweight="bold")
    ax_pat.legend(fontsize=8)

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    age_b64 = "data:image/png;base64," + b64encode(buf.read()).decode()

    return (
        '<h2 id="summary">Dataset summary</h2>\n'
        + table_html
        + f'<img src="{age_b64}" alt="Age distributions" style="max-width:700px; height:auto;" />\n'
    )


# ---------------------------------------------------------------------------
# Control section: one row per tract, 4 columns (Before traces, After traces, Before mean±SEM, After mean±SEM)
# ---------------------------------------------------------------------------


def _plot_traces_by_group(
    ax: plt.Axes,
    group_data: Optional[Dict[str, Tuple[List[str], np.ndarray]]],
    ylabel: str,
    end_labels: Optional[Tuple[str, str]],
) -> None:
    x = np.arange(1, 101)
    if group_data:
        for grp in sorted(group_data):
            subs, profiles = group_data[grp]
            color = GROUP_COLORS.get(grp, "gray")
            for i in range(profiles.shape[0]):
                ax.plot(x, profiles[i], color=color, alpha=0.2, linewidth=0.5, label=grp if i == 0 else None)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=6, loc="best")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlim(1, 100)
    _apply_end_ticks(ax, end_labels, flip=False)


def _plot_mean_sem_by_group(
    ax: plt.Axes,
    group_data: Optional[Dict[str, Tuple[List[str], np.ndarray]]],
    ylabel: str,
    end_labels: Optional[Tuple[str, str]],
) -> None:
    x = np.arange(1, 101)
    if group_data:
        for grp in sorted(group_data):
            profiles = group_data[grp][1]
            if profiles.shape[0] == 0:
                continue
            color = GROUP_COLORS.get(grp, "gray")
            mean = np.nanmean(profiles, axis=0)
            sem = np.nanstd(profiles, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(profiles), axis=0).clip(min=1))
            ax.plot(x, mean, color=color, label=grp)
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlim(1, 100)
    _apply_end_ticks(ax, end_labels, flip=False)
    ax.legend(fontsize=6, loc="best")


def make_control_row_figure(
    pyafq_gam_dir: Path,
    preharm_dir: Path,
    tract_label: str,
    scalar: str,
    scalar_human: str,
    end_labels: Optional[Tuple[str, str]],
    tract_meta: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """One row: 4 columns — Before traces, After traces, Before mean±SEM, After mean±SEM (controls only)."""
    before_traces = load_control_profiles_by_group(
        pyafq_gam_dir, preharm_dir, tract_label, scalar, before=True
    )
    after_traces = load_control_profiles_by_group(
        pyafq_gam_dir, preharm_dir, tract_label, scalar, before=False
    )
    if not before_traces and not after_traces:
        return None

    tract_title = _get_tract_label_human(tract_meta, tract_label) if tract_meta is not None else tract_label
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(tract_title, fontsize=11, fontweight="bold", y=1.02)

    _plot_traces_by_group(
        axes[0], before_traces, f"{scalar_human}\nBefore CovBat", end_labels
    )
    axes[0].set_title("Before CovBat (traces)", fontsize=9)

    _plot_traces_by_group(
        axes[1], after_traces, f"{scalar_human}\nAfter CovBat", end_labels
    )
    axes[1].set_title("After CovBat (traces)", fontsize=9)

    _plot_mean_sem_by_group(
        axes[2], before_traces, f"{scalar_human}\nBefore CovBat", end_labels
    )
    axes[2].set_title("Before CovBat", fontsize=9)

    _plot_mean_sem_by_group(
        axes[3], after_traces, f"{scalar_human}\nAfter CovBat", end_labels
    )
    axes[3].set_title("After CovBat", fontsize=9)

    for ax in axes:
        ax.set_xlim(1, 100)
    ymins = [ax.get_ylim()[0] for ax in axes if ax.get_ylim()[0] != 0 or ax.get_ylim()[1] != 1]
    ymaxs = [ax.get_ylim()[1] for ax in axes if ax.get_ylim()[0] != 0 or ax.get_ylim()[1] != 1]
    if ymins and ymaxs:
        ylo, yhi = min(ymins), max(ymaxs)
        for ax in axes:
            ax.set_ylim(ylo, yhi)

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# TLE section: 4x4 figure per tract_base (no traces; cols 0-1 CovBat, cols 2-3 GAM z; flip per block)
# ---------------------------------------------------------------------------


def _apply_flip(profiles: np.ndarray, flip: bool) -> np.ndarray:
    if not flip or profiles.size == 0:
        return profiles
    return profiles[:, ::-1]


def _fdr_significant_nodes(
    ipsi: np.ndarray, contra: np.ndarray, q: float = 0.05
) -> np.ndarray:
    """Paired t-test per node (ipsi vs contra), FDR correction. Returns bool (100,) True where q < 0.05."""
    if ipsi.shape[0] < 2 or ipsi.shape[1] != 100:
        return np.zeros(100, dtype=bool)
    try:
        from scipy.stats import ttest_rel
    except ImportError:
        return np.zeros(100, dtype=bool)
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return np.zeros(100, dtype=bool)
    pvals = np.ones(100)
    for j in range(100):
        a, b = ipsi[:, j], contra[:, j]
        valid = np.isfinite(a) & np.isfinite(b)
        if np.sum(valid) >= 2:
            stat, p = ttest_rel(a[valid], b[valid])
            if np.isfinite(p):
                pvals[j] = p
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    return (qvals < q) & np.isfinite(qvals)


def _draw_sig_spans(
    ax: plt.Axes, sig_mask: np.ndarray, flip: bool, color: str = "yellow", alpha: float = 0.35
) -> None:
    """Draw vertical spans for significant nodes. sig_mask length 100 (tract node order). flip matches axis display."""
    if sig_mask is None or len(sig_mask) != 100:
        return
    n = 100
    for i in range(n):
        if not sig_mask[i]:
            continue
        if flip:
            x_lo, x_hi = (n - i) - 0.5, (n - i) + 0.5
        else:
            x_lo, x_hi = (i + 1) - 0.5, (i + 1) + 0.5
        ax.axvspan(x_lo, x_hi, facecolor=color, alpha=alpha, zorder=0)


def _is_mesial_temporal(clinical_df: Optional[pd.DataFrame], sub: str) -> bool:
    """True if resection_details or ablation_target contains 'mesial_temporal' or 'mesial temporal' (case-insensitive)."""
    if clinical_df is None or clinical_df.empty:
        return False
    for col in ("resection_details", "ablation_target"):
        if col not in clinical_df.columns:
            continue
        row = clinical_df[clinical_df["sub"].astype(str) == str(sub)]
        if row.empty:
            continue
        val = row.iloc[0][col]
        if pd.isna(val):
            continue
        s = str(val).strip().lower()
        if "mesial_temporal" in s or "mesial temporal" in s:
            return True
    return False


def make_all_scalar_summary_figure(
    pyafq_gam_dir: Path,
    tract_base: str,
    scalars: List[str],
    left_tle_subs: List[str],
    right_tle_subs: List[str],
    tract_meta: pd.DataFrame,
    tract_end_labels: Dict[str, Tuple[str, str]],
    scalar_to_human: Dict[str, str],
    scalar_to_color: Dict[str, str],
    model_to_color: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """2x2 summary: row0 = frequency of significant nodes across scalars (Left TLE, Right TLE);
    row1 = Cohen's d at significant nodes only, colored by scalar, with label at rightmost sig node.
    Returns base64 PNG or None if no data."""
    left_tract = f"{tract_base}_L"
    right_tract = f"{tract_base}_R"
    x = np.arange(1, 101)
    n_scalars = len(scalars)
    if n_scalars == 0:
        return None

    # Collect per-scalar: sig_ltle_z, sig_rtle_z, cohens_d_ltle, cohens_d_rtle (all length 100)
    sig_ltle_list: List[np.ndarray] = []
    sig_rtle_list: List[np.ndarray] = []
    d_ltle_list: List[np.ndarray] = []
    d_rtle_list: List[np.ndarray] = []

    for scalar in scalars:
        ltle_L_z = load_tract_profiles(
            pyafq_gam_dir, pyafq_gam_dir, left_tract, scalar, left_tle_subs, before=False, use_z=True
        )
        ltle_R_z = load_tract_profiles(
            pyafq_gam_dir, pyafq_gam_dir, right_tract, scalar, left_tle_subs, before=False, use_z=True
        )
        rtle_L_z = load_tract_profiles(
            pyafq_gam_dir, pyafq_gam_dir, left_tract, scalar, right_tle_subs, before=False, use_z=True
        )
        rtle_R_z = load_tract_profiles(
            pyafq_gam_dir, pyafq_gam_dir, right_tract, scalar, right_tle_subs, before=False, use_z=True
        )
        sig_ltle = np.zeros(100, dtype=bool)
        sig_rtle = np.zeros(100, dtype=bool)
        d_ltle = np.full(100, np.nan)
        d_rtle = np.full(100, np.nan)
        if ltle_L_z and ltle_R_z:
            ipsi, contra = _intersect_paired(ltle_L_z, ltle_R_z)
            if ipsi.shape[0] >= 2:
                sig_ltle = _fdr_significant_nodes(ipsi, contra)
                d_ltle = cohens_d_paired_per_node(ipsi, contra)
        if rtle_R_z and rtle_L_z:
            ipsi, contra = _intersect_paired(rtle_R_z, rtle_L_z)
            if ipsi.shape[0] >= 2:
                sig_rtle = _fdr_significant_nodes(ipsi, contra)
                d_rtle = cohens_d_paired_per_node(ipsi, contra)
        sig_ltle_list.append(sig_ltle)
        sig_rtle_list.append(sig_rtle)
        d_ltle_list.append(d_ltle)
        d_rtle_list.append(d_rtle)

    # Colors by scalar model (e.g. DTI, DKI); fallback to scalar_to_color or tab10
    tab10 = plt.cm.tab10
    def _scalar_color(i: int, s: str):
        model = s.split("_")[0] if "_" in s else s
        if model_to_color and model in model_to_color:
            return model_to_color[model]
        c = scalar_to_color.get(s)
        if c is not None and c != "":
            return c if isinstance(c, str) else tuple(tab10(i % 10)[:3])
        return tuple(tab10(i % 10)[:3])
    colors = [_scalar_color(i, s) for i, s in enumerate(scalars)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    tract_title = _get_tract_base_human(tract_meta, tract_base)
    fig.suptitle(f"{tract_title} GAM z-score asymmetries (q<0.05) across scalars", fontsize=12, fontweight="bold")

    end_labels = tract_end_labels.get(tract_base)

    # Row 0: Frequency per node (how many scalars have this node significant)
    freq_ltle = np.sum(np.stack(sig_ltle_list, axis=0), axis=0).astype(int)
    freq_rtle = np.sum(np.stack(sig_rtle_list, axis=0), axis=0).astype(int)
    axes[0, 0].bar(x, freq_ltle, color="tab:green", alpha=0.7, width=0.8)
    axes[0, 0].set_ylabel("Number of scalars")
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 0].set_title("Left TLE")
    axes[0, 0].set_xlim(0.5, 100.5)
    _apply_end_ticks(axes[0, 0], end_labels, flip=False)
    axes[0, 1].bar(x, freq_rtle, color="tab:blue", alpha=0.7, width=0.8)
    axes[0, 1].set_ylabel("Number of scalars")
    axes[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 1].set_title("Right TLE")
    axes[0, 1].set_xlim(0.5, 100.5)
    _apply_end_ticks(axes[0, 1], end_labels, flip=False)

    # Row 1: Cohen's d only at significant nodes, colored by scalar; label at rightmost sig node per scalar
    for ax, d_list, sig_list, col_title in [
        (axes[1, 0], d_ltle_list, sig_ltle_list, "Left TLE"),
        (axes[1, 1], d_rtle_list, sig_rtle_list, "Right TLE"),
    ]:
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        for idx, (d_arr, sig_arr) in enumerate(zip(d_list, sig_list)):
            d_masked = np.where(sig_arr, d_arr, np.nan)
            lab = scalar_to_human.get(scalars[idx], scalars[idx])
            ax.plot(x, d_masked, color=colors[idx], linewidth=1.2, label=lab)
            # Rightmost significant node for this scalar
            sig_inds = np.where(sig_arr)[0]
            if len(sig_inds) > 0:
                rightmost = int(sig_inds[-1])
                node_1based = rightmost + 1
                val = d_arr[rightmost]
                if np.isfinite(val):
                    ax.annotate(
                        lab,
                        xy=(node_1based, val),
                        xytext=(5, 0),
                        textcoords="offset points",
                        fontsize=7,
                        color=colors[idx],
                        ha="left",
                        va="center",
                    )
        ax.set_ylabel("Ipsi − Contra Cohen's d")
        ax.set_title(col_title)
        ax.set_xlim(0.5, 100.5)
        _apply_end_ticks(ax, end_labels, flip=False)

    for ax in axes.flat:
        ax.set_xlabel("Node")
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return b64encode(buf.read()).decode("utf-8")


def make_within_subject_summary_figure(
    pyafq_gam_dir: Path,
    tract_base: str,
    scalars: List[str],
    left_tle_subs: List[str],
    right_tle_subs: List[str],
    tract_meta: pd.DataFrame,
    tract_end_labels: Dict[str, Tuple[str, str]],
) -> Optional[str]:
    """2x2 figure: per-(subject, node) Cohen's d across scalars.
    Row 0: frequency of subjects with significant effect sizes per node.
    Row 1: mean +/- SEM of Cohen's d across subjects.
    Returns base64 PNG or None."""
    from scipy.stats import ttest_1samp

    left_tract = f"{tract_base}_L"
    right_tract = f"{tract_base}_R"
    x = np.arange(1, 101)

    def _compute_per_subject(tle_subs: List[str], ipsi_tract: str, contra_tract: str):
        """Returns dict {sub: d_array(100,)} and dict {sub: sig_bool(100,)}."""
        # Gather per-scalar aligned ipsi/contra profiles
        per_scalar_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        for scalar in scalars:
            ipsi_data = load_tract_profiles(
                pyafq_gam_dir, pyafq_gam_dir, ipsi_tract, scalar, tle_subs, before=False, use_z=True
            )
            contra_data = load_tract_profiles(
                pyafq_gam_dir, pyafq_gam_dir, contra_tract, scalar, tle_subs, before=False, use_z=True
            )
            if ipsi_data is None or contra_data is None:
                continue
            ipsi_subs, ipsi_arr = ipsi_data
            contra_subs, contra_arr = contra_data
            ipsi_idx = {s: i for i, s in enumerate(ipsi_subs)}
            contra_idx = {s: i for i, s in enumerate(contra_subs)}
            shared = [s for s in ipsi_subs if s in contra_idx]
            for sub in shared:
                if sub not in per_scalar_data:
                    per_scalar_data[sub] = {}
                per_scalar_data[sub][scalar] = (
                    ipsi_arr[ipsi_idx[sub]],
                    contra_arr[contra_idx[sub]],
                )

        d_out: Dict[str, np.ndarray] = {}
        sig_out: Dict[str, np.ndarray] = {}
        for sub, scalar_dict in per_scalar_data.items():
            if len(scalar_dict) < 2:
                continue
            diffs = np.stack([
                ipsi_row - contra_row
                for ipsi_row, contra_row in scalar_dict.values()
            ], axis=0)  # (n_scalars, 100)
            n_finite = np.sum(np.isfinite(diffs), axis=0)
            mean_d = np.nanmean(diffs, axis=0)
            std_d = np.nanstd(diffs, axis=0, ddof=1)
            d = np.full(100, np.nan)
            valid = (n_finite >= 2) & (std_d > 0)
            d[valid] = mean_d[valid] / std_d[valid]
            d_out[sub] = d

            pvals = np.ones(100)
            for j in range(100):
                col = diffs[:, j]
                fin = col[np.isfinite(col)]
                if len(fin) >= 2:
                    _, p = ttest_1samp(fin, 0)
                    if np.isfinite(p):
                        pvals[j] = p
            try:
                from statsmodels.stats.multitest import multipletests as mt
                _, qvals, _, _ = mt(pvals, method="fdr_bh")
                sig_out[sub] = (qvals < 0.05) & np.isfinite(qvals)
            except ImportError:
                sig_out[sub] = pvals < 0.05
        return d_out, sig_out

    ltle_d, ltle_sig = _compute_per_subject(left_tle_subs, left_tract, right_tract)
    rtle_d, rtle_sig = _compute_per_subject(right_tle_subs, right_tract, left_tract)

    if not ltle_d and not rtle_d:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    tract_title = _get_tract_base_human(tract_meta, tract_base)
    fig.suptitle(f"{tract_title} within-subject GAM z-score asymmetry effect sizes", fontsize=12, fontweight="bold")
    end_labels = tract_end_labels.get(tract_base)

    # Row 0: frequency of significant subjects per node
    for col, d_dict, sig_dict, color, title in [
        (0, ltle_d, ltle_sig, "tab:green", "Left TLE"),
        (1, rtle_d, rtle_sig, "tab:blue", "Right TLE"),
    ]:
        if sig_dict:
            freq = np.sum(np.stack(list(sig_dict.values()), axis=0).astype(int), axis=0)
        else:
            freq = np.zeros(100, dtype=int)
        axes[0, col].bar(x, freq, color=color, alpha=0.7, width=0.8)
        axes[0, col].set_ylabel("Number of subjects")
        axes[0, col].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0, col].set_title(title)
        axes[0, col].set_xlim(0.5, 100.5)
        _apply_end_ticks(axes[0, col], end_labels, flip=False)

    # Row 1: mean +/- SEM of Cohen's d across subjects
    for col, d_dict, color, title in [
        (0, ltle_d, "tab:green", "Left TLE"),
        (1, rtle_d, "tab:blue", "Right TLE"),
    ]:
        axes[1, col].axhline(0, color="grey", linewidth=0.5, linestyle="--")
        if d_dict:
            d_stack = np.stack(list(d_dict.values()), axis=0)  # (n_subjects, 100)
            n_fin = np.sum(np.isfinite(d_stack), axis=0).clip(min=1)
            mean_d = np.nanmean(d_stack, axis=0)
            sem_d = np.nanstd(d_stack, axis=0, ddof=1) / np.sqrt(n_fin)
            axes[1, col].plot(x, mean_d, color=color, linewidth=1.2)
            axes[1, col].fill_between(x, mean_d - sem_d, mean_d + sem_d, color=color, alpha=0.25)
        axes[1, col].set_ylabel("Ipsi - Contra Cohen's d (mean ± SEM)")
        axes[1, col].set_title(title)
        axes[1, col].set_xlim(0.5, 100.5)
        _apply_end_ticks(axes[1, col], end_labels, flip=False)

    for ax in axes.flat:
        ax.set_xlabel("Node")
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return b64encode(buf.read()).decode("utf-8")


def _overlay_control_mean(
    ax: plt.Axes,
    ctrl_profiles: Optional[Tuple[List[str], np.ndarray]],
    flip: bool,
) -> None:
    if ctrl_profiles is None:
        return
    profiles = ctrl_profiles[1]
    if profiles.shape[0] == 0:
        return
    x = np.arange(1, 101)
    mean = np.nanmean(profiles, axis=0)
    if flip:
        mean = mean[::-1]
    ax.plot(x, mean, color="black", linewidth=1.2, linestyle="--", label="Control mean", zorder=5)


def make_tle_figure(
    pyafq_gam_dir: Path,
    preharm_dir: Path,
    tract_base: str,
    scalar: str,
    scalar_human: str,
    left_tle_subs: List[str],
    right_tle_subs: List[str],
    tract_meta: pd.DataFrame,
    tract_end_labels: Dict[str, Tuple[str, str]],
    clinical_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """5 rows x 4 columns: rows 0–3 as before; row 4 = GAM z traces (purple=mesial temporal, black=other). Cols 0&1 = Left TLE, 2&3 = Right TLE."""
    flip_L, flip_R = get_flip_left_right(tract_meta, tract_base)
    left_tract = f"{tract_base}_L"
    right_tract = f"{tract_base}_R"
    end_L = tract_end_labels.get(tract_base)

    def _concat_control(g):
        if not g:
            return None
        all_subs, all_profs = [], []
        for (subs, profs) in g.values():
            all_subs.extend(subs)
            all_profs.append(profs)
        return (all_subs, np.concatenate(all_profs, axis=0)) if all_profs else None

    g_after_L = load_control_profiles_by_group(pyafq_gam_dir, preharm_dir, left_tract, scalar, before=False)
    g_after_R = load_control_profiles_by_group(pyafq_gam_dir, preharm_dir, right_tract, scalar, before=False)
    g_z_L = load_control_profiles_by_group(pyafq_gam_dir, preharm_dir, left_tract, scalar, before=False, use_z=True)
    g_z_R = load_control_profiles_by_group(pyafq_gam_dir, preharm_dir, right_tract, scalar, before=False, use_z=True)
    ctrl_L_after = _concat_control(g_after_L)
    ctrl_R_after = _concat_control(g_after_R)
    ctrl_L_z = _concat_control(g_z_L)
    ctrl_R_z = _concat_control(g_z_R)

    ltle_L_after = load_tract_profiles(pyafq_gam_dir, preharm_dir, left_tract, scalar, left_tle_subs, before=False)
    ltle_R_after = load_tract_profiles(pyafq_gam_dir, preharm_dir, right_tract, scalar, left_tle_subs, before=False)
    rtle_L_after = load_tract_profiles(pyafq_gam_dir, preharm_dir, left_tract, scalar, right_tle_subs, before=False)
    rtle_R_after = load_tract_profiles(pyafq_gam_dir, preharm_dir, right_tract, scalar, right_tle_subs, before=False)
    ltle_L_z = load_tract_profiles(pyafq_gam_dir, preharm_dir, left_tract, scalar, left_tle_subs, before=False, use_z=True)
    ltle_R_z = load_tract_profiles(pyafq_gam_dir, preharm_dir, right_tract, scalar, left_tle_subs, before=False, use_z=True)
    rtle_L_z = load_tract_profiles(pyafq_gam_dir, preharm_dir, left_tract, scalar, right_tle_subs, before=False, use_z=True)
    rtle_R_z = load_tract_profiles(pyafq_gam_dir, preharm_dir, right_tract, scalar, right_tle_subs, before=False, use_z=True)

    any_data = (
        ltle_L_after or ltle_R_after or rtle_L_after or rtle_R_after
        or ltle_L_z or ltle_R_z or rtle_L_z or rtle_R_z
    )
    if not any_data:
        return None

    # Per-quadrant FDR-corrected paired t-tests (ipsi vs contra); sig mask (100,) bool per quadrant
    sig_ltle_covbat = np.zeros(100, dtype=bool)
    if ltle_L_after and ltle_R_after:
        ipsi, contra = _intersect_paired(ltle_L_after, ltle_R_after)
        if ipsi.shape[0] >= 2:
            sig_ltle_covbat = _fdr_significant_nodes(ipsi, contra)
    sig_rtle_covbat = np.zeros(100, dtype=bool)
    if rtle_R_after and rtle_L_after:
        ipsi, contra = _intersect_paired(rtle_R_after, rtle_L_after)
        if ipsi.shape[0] >= 2:
            sig_rtle_covbat = _fdr_significant_nodes(ipsi, contra)
    sig_ltle_z = np.zeros(100, dtype=bool)
    if ltle_L_z and ltle_R_z:
        ipsi_z, contra_z = _intersect_paired(ltle_L_z, ltle_R_z)
        if ipsi_z.shape[0] >= 2:
            sig_ltle_z = _fdr_significant_nodes(ipsi_z, contra_z)
    sig_rtle_z = np.zeros(100, dtype=bool)
    if rtle_R_z and rtle_L_z:
        ipsi_z, contra_z = _intersect_paired(rtle_R_z, rtle_L_z)
        if ipsi_z.shape[0] >= 2:
            sig_rtle_z = _fdr_significant_nodes(ipsi_z, contra_z)

    tract_title = _get_tract_base_human(tract_meta, tract_base)
    fig, axes = plt.subplots(5, 4, figsize=(16, 14))
    fig.suptitle(tract_title, fontsize=14, fontweight="bold", y=0.995)
    x = np.arange(1, 101)
    fig.text(0.25, 0.94, "Left TLE", fontsize=11, fontweight="bold", ha="center")
    fig.text(0.75, 0.94, "Right TLE", fontsize=11, fontweight="bold", ha="center")
    # Column titles above first row: Left hemi, Right hemi, Left hemi, Right hemi
    for col, label in enumerate(["Left hemisphere", "Right hemisphere", "Left hemisphere", "Right hemisphere"]):
        fig.text(0.125 + 0.25 * col, 0.90, label, fontsize=9, ha="center", va="bottom")

    # Row 0: Top row — Left TLE CovBat (0,1), Right TLE CovBat (2,3)
    if ltle_L_after:
        subs, profs = ltle_L_after
        profs = _apply_flip(profs, flip_L)
        mean = np.nanmean(profs, axis=0)
        sem = np.nanstd(profs, axis=0, ddof=1) / np.sqrt(profs.shape[0])
        axes[0, 0].plot(x, mean, color="tab:green", label="Left TLE")
        axes[0, 0].fill_between(x, mean - sem, mean + sem, color="tab:green", alpha=0.2)
        _overlay_control_mean(axes[0, 0], ctrl_L_after, flip_L)
        # axes[0, 0].set_title("Left TLE mean ± SEM (L hemi)", fontsize=8)
        axes[0, 0].set_ylabel(scalar_human, fontsize=8)
        _apply_end_ticks(axes[0, 0], end_L, flip_L)
    axes[0, 0].set_xlim(1, 100)

    if ltle_R_after:
        subs, profs = ltle_R_after
        profs = _apply_flip(profs, flip_R)
        mean = np.nanmean(profs, axis=0)
        sem = np.nanstd(profs, axis=0, ddof=1) / np.sqrt(profs.shape[0])
        axes[0, 1].plot(x, mean, color="tab:green", label="Left TLE")
        axes[0, 1].fill_between(x, mean - sem, mean + sem, color="tab:green", alpha=0.2)
        _overlay_control_mean(axes[0, 1], ctrl_R_after, flip_R)
        # axes[0, 1].set_title("Left TLE mean ± SEM (R hemi)", fontsize=8)
        axes[0, 1].set_ylabel(scalar_human, fontsize=8)
        _apply_end_ticks(axes[0, 1], end_L, flip_R)
    axes[0, 1].set_xlim(1, 100)

    if rtle_L_after:
        subs, profs = rtle_L_after
        profs = _apply_flip(profs, flip_L)
        mean = np.nanmean(profs, axis=0)
        sem = np.nanstd(profs, axis=0, ddof=1) / np.sqrt(profs.shape[0])
        axes[0, 2].plot(x, mean, color=TLE_COLORS["Right TLE"], label="Right TLE")
        axes[0, 2].fill_between(x, mean - sem, mean + sem, color=TLE_COLORS["Right TLE"], alpha=0.2)
        _overlay_control_mean(axes[0, 2], ctrl_L_after, flip_L)
        # axes[0, 2].set_title("Right TLE mean ± SEM (L hemi)", fontsize=8)
        axes[0, 2].set_ylabel(scalar_human, fontsize=8)
        _apply_end_ticks(axes[0, 2], end_L, flip_L)
    axes[0, 2].set_xlim(1, 100)

    if rtle_R_after:
        subs, profs = rtle_R_after
        profs = _apply_flip(profs, flip_R)
        mean = np.nanmean(profs, axis=0)
        sem = np.nanstd(profs, axis=0, ddof=1) / np.sqrt(profs.shape[0])
        axes[0, 3].plot(x, mean, color=TLE_COLORS["Right TLE"], label="Right TLE")
        axes[0, 3].fill_between(x, mean - sem, mean + sem, color=TLE_COLORS["Right TLE"], alpha=0.2)
        _overlay_control_mean(axes[0, 3], ctrl_R_after, flip_R)
        # axes[0, 3].set_title("Right TLE mean ± SEM (R hemi)", fontsize=8)
        axes[0, 3].set_ylabel(scalar_human, fontsize=8)
        _apply_end_ticks(axes[0, 3], end_L, flip_R)
    axes[0, 3].set_xlim(1, 100)

    # Row 1: Left TLE effect (0,1), Right TLE effect (2,3) — both CovBat
    axes[1, 0].axhline(0, color="grey", linewidth=0.5, linestyle="--")
    axes[1, 0].set_ylabel("Cohen's d (ipsi − contra)", fontsize=8)
    _apply_end_ticks(axes[1, 0], end_L, flip_L)
    if ltle_L_after and ltle_R_after:
        ipsi, contra = _intersect_paired(ltle_L_after, ltle_R_after)
        if ipsi.shape[0] >= 2:
            d_after = cohens_d_paired_per_node(ipsi, contra)
            d_plot = d_after[::-1] if flip_L else d_after
            axes[1, 0].plot(x, d_plot, color="tab:green", linewidth=1)
        if ipsi.shape[0] >= 1:
            asym = asymmetry_index_per_node(ipsi, contra)
            asym_display = _apply_flip(asym, flip_R)
            mean_asym = np.nanmean(asym_display, axis=0)
            sem_asym = np.nanstd(asym_display, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(asym_display), axis=0).clip(min=1))
            axes[1, 1].axhline(0, color="grey", linewidth=0.5, linestyle="--")
            axes[1, 1].plot(x, mean_asym, color="tab:green", linewidth=1)
            axes[1, 1].fill_between(x, mean_asym - sem_asym, mean_asym + sem_asym, color="tab:green", alpha=0.2)
            axes[1, 1].set_ylabel("Asymmetry", fontsize=8)
            _apply_end_ticks(axes[1, 1], end_L, flip_R)
    axes[1, 0].set_xlim(1, 100)
    axes[1, 1].set_xlim(1, 100)

    axes[1, 2].axhline(0, color="grey", linewidth=0.5, linestyle="--")
    axes[1, 2].set_ylabel("Cohen's d (ipsi − contra)", fontsize=8)
    _apply_end_ticks(axes[1, 2], end_L, flip_L)
    if rtle_R_after and rtle_L_after:
        ipsi, contra = _intersect_paired(rtle_R_after, rtle_L_after)
        if ipsi.shape[0] >= 2:
            d_after = cohens_d_paired_per_node(ipsi, contra)
            d_plot = d_after[::-1] if flip_L else d_after
            axes[1, 2].plot(x, d_plot, color=TLE_COLORS["Right TLE"], linewidth=1)
        if ipsi.shape[0] >= 1:
            asym = asymmetry_index_per_node(ipsi, contra)
            asym_display = _apply_flip(asym, flip_R)
            mean_asym = np.nanmean(asym_display, axis=0)
            sem_asym = np.nanstd(asym_display, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(asym_display), axis=0).clip(min=1))
            axes[1, 3].axhline(0, color="grey", linewidth=0.5, linestyle="--")
            axes[1, 3].plot(x, mean_asym, color=TLE_COLORS["Right TLE"], linewidth=1)
            axes[1, 3].fill_between(x, mean_asym - sem_asym, mean_asym + sem_asym, color=TLE_COLORS["Right TLE"], alpha=0.2)
            axes[1, 3].set_ylabel("Asymmetry", fontsize=8)
            _apply_end_ticks(axes[1, 3], end_L, flip_R)
    axes[1, 2].set_xlim(1, 100)
    axes[1, 3].set_xlim(1, 100)

    # Row 2: Left TLE z (0,1), Right TLE z (2,3)
    if ltle_L_z:
        subs, profs = ltle_L_z
        profs = _apply_flip(profs, flip_L)
        mean = np.nanmean(profs, axis=0)
        sem = np.nanstd(profs, axis=0, ddof=1) / np.sqrt(profs.shape[0])
        axes[2, 0].plot(x, mean, color=TLE_Z_COLOR, label="Left TLE (z)")
        axes[2, 0].fill_between(x, mean - sem, mean + sem, color=TLE_Z_COLOR, alpha=0.2)
        _overlay_control_mean(axes[2, 0], ctrl_L_z, flip_L)
        # axes[2, 0].set_title("Left TLE mean ± SEM (L hemi)", fontsize=8)
        axes[2, 0].set_ylabel("GAM z", fontsize=8)
        _apply_end_ticks(axes[2, 0], end_L, flip_L)
    axes[2, 0].set_xlim(1, 100)

    if ltle_R_z:
        subs, profs = ltle_R_z
        profs = _apply_flip(profs, flip_R)
        mean = np.nanmean(profs, axis=0)
        sem = np.nanstd(profs, axis=0, ddof=1) / np.sqrt(profs.shape[0])
        axes[2, 1].plot(x, mean, color=TLE_Z_COLOR, label="Left TLE (z)")
        axes[2, 1].fill_between(x, mean - sem, mean + sem, color=TLE_Z_COLOR, alpha=0.2)
        _overlay_control_mean(axes[2, 1], ctrl_R_z, flip_R)
        # axes[2, 1].set_title("Left TLE mean ± SEM (R hemi)", fontsize=8)
        axes[2, 1].set_ylabel("GAM z", fontsize=8)
        _apply_end_ticks(axes[2, 1], end_L, flip_R)
    axes[2, 1].set_xlim(1, 100)

    if rtle_L_z:
        subs, profs = rtle_L_z
        profs = _apply_flip(profs, flip_L)
        mean = np.nanmean(profs, axis=0)
        sem = np.nanstd(profs, axis=0, ddof=1) / np.sqrt(profs.shape[0])
        axes[2, 2].plot(x, mean, color=TLE_Z_COLOR, label="Right TLE (z)")
        axes[2, 2].fill_between(x, mean - sem, mean + sem, color=TLE_Z_COLOR, alpha=0.2)
        _overlay_control_mean(axes[2, 2], ctrl_L_z, flip_L)
        # axes[2, 2].set_title("Right TLE mean ± SEM (L hemi)", fontsize=8)
        axes[2, 2].set_ylabel("GAM z", fontsize=8)
        _apply_end_ticks(axes[2, 2], end_L, flip_L)
    axes[2, 2].set_xlim(1, 100)

    if rtle_R_z:
        subs, profs = rtle_R_z
        profs = _apply_flip(profs, flip_R)
        mean = np.nanmean(profs, axis=0)
        sem = np.nanstd(profs, axis=0, ddof=1) / np.sqrt(profs.shape[0])
        axes[2, 3].plot(x, mean, color=TLE_Z_COLOR, label="Right TLE (z)")
        axes[2, 3].fill_between(x, mean - sem, mean + sem, color=TLE_Z_COLOR, alpha=0.2)
        _overlay_control_mean(axes[2, 3], ctrl_R_z, flip_R)
        # axes[2, 3].set_title("Right TLE mean ± SEM (R hemi)", fontsize=8)
        axes[2, 3].set_ylabel("GAM z", fontsize=8)
        _apply_end_ticks(axes[2, 3], end_L, flip_R)
    axes[2, 3].set_xlim(1, 100)

    # Row 3: Left TLE effect z (0,1), Right TLE effect z (2,3)
    axes[3, 0].axhline(0, color="grey", linewidth=0.5, linestyle="--")
    axes[3, 0].set_ylabel("Cohen's d (ipsi − contra)", fontsize=8)
    _apply_end_ticks(axes[3, 0], end_L, flip_L)
    if ltle_L_z and ltle_R_z:
        ipsi_z, contra_z = _intersect_paired(ltle_L_z, ltle_R_z)
        if ipsi_z.shape[0] >= 2:
            d_z = cohens_d_paired_per_node(ipsi_z, contra_z)
            d_plot_z = d_z[::-1] if flip_L else d_z
            axes[3, 0].plot(x, d_plot_z, color=TLE_Z_COLOR, linewidth=1)
        if ipsi_z.shape[0] >= 1:
            asym_z = asymmetry_index_per_node(ipsi_z, contra_z)
            asym_display_z = _apply_flip(asym_z, flip_R)
            mean_asym_z = np.nanmean(asym_display_z, axis=0)
            sem_asym_z = np.nanstd(asym_display_z, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(asym_display_z), axis=0).clip(min=1))
            axes[3, 1].axhline(0, color="grey", linewidth=0.5, linestyle="--")
            axes[3, 1].plot(x, mean_asym_z, color=TLE_Z_COLOR, linewidth=1)
            axes[3, 1].fill_between(x, mean_asym_z - sem_asym_z, mean_asym_z + sem_asym_z, color=TLE_Z_COLOR, alpha=0.2)
            axes[3, 1].set_ylabel("Asymmetry", fontsize=8)
            _apply_end_ticks(axes[3, 1], end_L, flip_R)
    axes[3, 0].set_xlim(1, 100)
    axes[3, 1].set_xlim(1, 100)

    axes[3, 2].axhline(0, color="grey", linewidth=0.5, linestyle="--")
    axes[3, 2].set_ylabel("Cohen's d (ipsi − contra)", fontsize=8)
    _apply_end_ticks(axes[3, 2], end_L, flip_L)
    if rtle_R_z and rtle_L_z:
        ipsi_z, contra_z = _intersect_paired(rtle_R_z, rtle_L_z)
        if ipsi_z.shape[0] >= 2:
            d_z = cohens_d_paired_per_node(ipsi_z, contra_z)
            d_plot_z = d_z[::-1] if flip_L else d_z
            axes[3, 2].plot(x, d_plot_z, color=TLE_Z_COLOR, linewidth=1)
        if ipsi_z.shape[0] >= 1:
            asym_z = asymmetry_index_per_node(ipsi_z, contra_z)
            asym_display_z = _apply_flip(asym_z, flip_R)
            mean_asym_z = np.nanmean(asym_display_z, axis=0)
            sem_asym_z = np.nanstd(asym_display_z, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(asym_display_z), axis=0).clip(min=1))
            axes[3, 3].axhline(0, color="grey", linewidth=0.5, linestyle="--")
            axes[3, 3].plot(x, mean_asym_z, color=TLE_Z_COLOR, linewidth=1)
            axes[3, 3].fill_between(x, mean_asym_z - sem_asym_z, mean_asym_z + sem_asym_z, color=TLE_Z_COLOR, alpha=0.2)
            axes[3, 3].set_ylabel("Asymmetry", fontsize=8)
            _apply_end_ticks(axes[3, 3], end_L, flip_R)
    axes[3, 2].set_xlim(1, 100)
    axes[3, 3].set_xlim(1, 100)

    # Row 4: GAM z traces (all patients); purple = mesial temporal, grey = other
    for (ax, data, fl) in [
        (axes[4, 0], ltle_L_z, flip_L),
        (axes[4, 1], ltle_R_z, flip_R),
        (axes[4, 2], rtle_L_z, flip_L),
        (axes[4, 3], rtle_R_z, flip_R),
    ]:
        if data is None:
            ax.set_xlim(1, 100)
            continue
        subs, profs = data
        profs = _apply_flip(profs, fl)
        for i, sub in enumerate(subs):
            color = "purple" if _is_mesial_temporal(clinical_df, sub) else "grey"
            ax.plot(x, profs[i], color=color, alpha=0.75, linewidth=0.8)
        ax.set_ylabel("GAM z", fontsize=8)
        _apply_end_ticks(ax, end_L, fl)
        ax.set_xlim(1, 100)

    # Highlight FDR-significant nodes (q < 0.05) in yellow per quadrant
    _sig_config = [
        (0, 0, sig_ltle_covbat, flip_L), (0, 1, sig_ltle_covbat, flip_R),
        (0, 2, sig_rtle_covbat, flip_L), (0, 3, sig_rtle_covbat, flip_R),
        (1, 0, sig_ltle_covbat, flip_L), (1, 1, sig_ltle_covbat, flip_R),
        (1, 2, sig_rtle_covbat, flip_L), (1, 3, sig_rtle_covbat, flip_R),
        (2, 0, sig_ltle_z, flip_L), (2, 1, sig_ltle_z, flip_R),
        (2, 2, sig_rtle_z, flip_L), (2, 3, sig_rtle_z, flip_R),
        (3, 0, sig_ltle_z, flip_L), (3, 1, sig_ltle_z, flip_R),
        (3, 2, sig_rtle_z, flip_L), (3, 3, sig_rtle_z, flip_R),
        (4, 0, sig_ltle_z, flip_L), (4, 1, sig_ltle_z, flip_R),
        (4, 2, sig_rtle_z, flip_L), (4, 3, sig_rtle_z, flip_R),
    ]
    for r, c, mask, fl in _sig_config:
        _draw_sig_spans(axes[r, c], mask, fl)

    # Row 0: all columns share the same y-axis limits
    row0_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3]]
    ylims0 = [(ax.get_ylim()[0], ax.get_ylim()[1]) for ax in row0_axes if ax.lines]
    if ylims0:
        ymin, ymax = min(y[0] for y in ylims0), max(y[1] for y in ylims0)
        for ax in row0_axes:
            ax.set_ylim(ymin, ymax)

    # Row 1: cols 0 & 2 match and zero-centered; cols 1 & 3 match and zero-centered
    for ax_list in ([axes[1, 0], axes[1, 2]], [axes[1, 1], axes[1, 3]]):
        M = 0.0
        for ax in ax_list:
            lo, hi = ax.get_ylim()
            M = max(M, abs(lo), abs(hi))
        M = max(M, 0.05)
        for ax in ax_list:
            ax.set_ylim(-M, M)

    # Row 2 & 4 (GAM z mean±SEM and traces): all columns share the same y-axis limits
    row2_4_axes = [axes[2, 0], axes[2, 1], axes[2, 2], axes[2, 3], axes[4, 0], axes[4, 1], axes[4, 2], axes[4, 3]]
    ylims2 = [(ax.get_ylim()[0], ax.get_ylim()[1]) for ax in row2_4_axes if ax.lines]
    if ylims2:
        ymin, ymax = min(y[0] for y in ylims2), max(y[1] for y in ylims2)
        for ax in row2_4_axes:
            ax.set_ylim(ymin, ymax)

    # Row 3: cols 0 & 2 match and zero-centered; cols 1 & 3 match and zero-centered
    for ax_list in ([axes[3, 0], axes[3, 2]], [axes[3, 1], axes[3, 3]]):
        M = 0.0
        for ax in ax_list:
            lo, hi = ax.get_ylim()
            M = max(M, abs(lo), abs(hi))
        M = max(M, 0.05)
        for ax in ax_list:
            ax.set_ylim(-M, M)

    # Figure legend: colors and lines used in the TLE grid
    legend_handles = [
        Line2D([0], [0], color="tab:green", linewidth=2, label="Left TLE"),
        Line2D([0], [0], color=TLE_COLORS["Right TLE"], linewidth=2, label="Right TLE"),
        Line2D([0], [0], color=TLE_Z_COLOR, linewidth=2, label="GAM z-scores"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Control mean"),
        Line2D([0], [0], color="grey", linestyle="--", linewidth=0.8, label="Zero reference"),
        Patch(facecolor="yellow", alpha=0.35, edgecolor="none", label="Significant (FDR q<0.05)"),
        Line2D([0], [0], color="purple", linewidth=2, alpha=0.75, label="Mesial temporal"),
        Line2D([0], [0], color="black", linewidth=2, alpha=0.75, label="Other"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_handles),
        fontsize=8,
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# Main: build HTML (TLE subjects omitted per plot when missing tract/scalar or one hemisphere)
# ---------------------------------------------------------------------------


def run(
    base_dir: Optional[Path] = None,
    tracts: Optional[List[str]] = None,
    scalars_filter: Optional[List[str]] = None,
    all_scalar_summary: bool = False,
) -> None:
    base_dir = Path(base_dir or DEFAULT_PROJECT_ROOT)
    paths = cfg.get_paths(base_dir)
    pyafq_gam_dir = paths["pyafq_gam_dir"]
    preharm_dir = base_dir / "derivatives" / "covbat" / "inputs" / "pyafq" / "HCP1065"
    output_dir = base_dir / "derivatives" / "analysis" / "asymmetry_tle_covbat_pyafq"
    output_dir.mkdir(parents=True, exist_ok=True)

    clinical_df = cfg.load_clinical(paths["clinical_path"])
    left_tle_subs, right_tle_subs = cfg.get_left_right_tle_subjects(clinical_df)
    if not left_tle_subs and not right_tle_subs:
        print("No left/right TLE subjects found. Exiting.")
        return

    all_scalars = cfg.get_scalar_labels(paths)
    if all_scalar_summary:
        scalars = list(all_scalars)  # all non-excluded scalars
    else:
        scalars = scalars_filter if scalars_filter else [s for s in DEFAULT_SCALARS if s in all_scalars]
    if not scalars:
        print("No scalars found. Exiting.")
        return

    scalar_meta = cfg.load_scalar_metadata(paths)
    scalar_to_human = scalar_meta.get("scalar_to_human", {})

    tract_meta_path = paths["tract_metadata_path"]
    if tract_meta_path.exists():
        tract_meta = pd.read_csv(tract_meta_path)
    else:
        print("Warning: tract metadata not found. Proceeding without end labels.")
        tract_meta = pd.DataFrame()

    tract_bases = list(tracts) if tracts else list(DEFAULT_TRACTS)
    tract_end_labels: Dict[str, Tuple[str, str]] = {}
    tract_labels_for_controls: List[str] = []  # F_L, F_R, UF_L, ...

    for tb in list(tract_bases):
        left_dir = pyafq_gam_dir / f"{tb}_L"
        right_dir = pyafq_gam_dir / f"{tb}_R"
        if not left_dir.exists() and not right_dir.exists():
            tract_bases.remove(tb)
            continue
        l_end1, l_end2 = _get_tract_ends(tract_meta, f"{tb}_L")
        r_end1, r_end2 = _get_tract_ends(tract_meta, f"{tb}_R")
        if (l_end1, l_end2) != (None, None) and (r_end1, r_end2) != (None, None):
            if (l_end1, l_end2) != (r_end1, r_end2):
                print(
                    f"ERROR: end1/end2 mismatch for {tb}: "
                    f"{tb}_L=({l_end1},{l_end2}) vs {tb}_R=({r_end1},{r_end2}). Exiting."
                )
                return
            tract_end_labels[tb] = (l_end1, l_end2)
        if left_dir.exists():
            tract_labels_for_controls.append(f"{tb}_L")
        if right_dir.exists():
            tract_labels_for_controls.append(f"{tb}_R")

    if not tract_bases:
        print("No valid tracts. Exiting.")
        return

    print(f"Left TLE n={len(left_tle_subs)}, Right TLE n={len(right_tle_subs)}")

    summary_html = _make_summary_section(pyafq_gam_dir, left_tle_subs, right_tle_subs)

    _PAGE_STYLE = """
body { font-family: sans-serif; margin: 20px; max-width: 1400px; }
h1 { font-size: 1.5em; }
h2 { font-size: 1.2em; margin-top: 32px; padding-top: 8px; border-top: 1px solid #ccc; }
nav.toc { margin: 20px 0 32px 0; padding: 16px; background: #f8f8f8; border: 1px solid #ddd; max-width: 500px; }
nav.toc h2 { margin: 0 0 10px 0; font-size: 1.1em; border: none; padding: 0; }
nav.toc ul { list-style: none; padding-left: 0; margin: 0; }
nav.toc a { color: #0066cc; text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }
img { display: block; margin: 12px 0 24px 0; }
table.summary-table { border-collapse: collapse; margin: 12px 0; }
table.summary-table th, table.summary-table td { border: 1px solid #ccc; padding: 6px 12px; text-align: left; }
table.summary-table th { background: #f0f0f0; }
"""

    index_links: List[Tuple[str, str]] = []  # (filename, human_name) for index

    for tract_base in tqdm(tract_bases, desc="Tract reports"):
        toc_entries: List[Tuple[str, str]] = [("summary", "Dataset summary")]
        sections_html: List[str] = []

        # Controls: only this tract's L and R
        tract_labels_this = [tl for tl in (f"{tract_base}_L", f"{tract_base}_R") if tl in tract_labels_for_controls]
        for scalar in scalars:
            scalar_human = scalar_to_human.get(scalar, scalar)
            anchor_ctl = f"controls_{scalar}"
            toc_entries.append((anchor_ctl, f"CovBat-GAM controls — {scalar_human}"))
            section_parts = [f'<h2 id="{html_module.escape(anchor_ctl)}">CovBat-GAM controls — {html_module.escape(scalar_human)}</h2>\n']
            for tract_label in tract_labels_this:
                end_lbl = tract_end_labels.get(tract_base)
                b64 = make_control_row_figure(
                    pyafq_gam_dir, preharm_dir, tract_label, scalar, scalar_human, end_lbl, tract_meta
                )
                if b64:
                    section_parts.append(
                        f'<img src="{b64}" alt="Control {tract_label} {scalar_human}" style="max-width:100%; height:auto;" />\n'
                    )
            sections_html.append("".join(section_parts))

        # TLE: only this tract_base
        for scalar in scalars:
            scalar_human = scalar_to_human.get(scalar, scalar)
            anchor_tle = f"tle_{scalar}"
            toc_entries.append((anchor_tle, f"CovBat-GAM TLE — {scalar_human}"))
            section_parts = [f'<h2 id="{html_module.escape(anchor_tle)}">CovBat-GAM TLE — {html_module.escape(scalar_human)}</h2>\n']
            b64 = make_tle_figure(
                pyafq_gam_dir,
                preharm_dir,
                tract_base,
                scalar,
                scalar_human,
                left_tle_subs,
                right_tle_subs,
                tract_meta,
                tract_end_labels,
                clinical_df,
            )
            if b64:
                section_parts.append(
                    f'<img src="{b64}" alt="TLE {tract_base} {scalar_human}" style="max-width:100%; height:auto;" />\n'
                )
            sections_html.append("".join(section_parts))

        toc_html = (
            '<nav class="toc"><h2>Contents</h2><ul>\n'
            + "\n".join(
                f'<li><a href="#{html_module.escape(a)}">{html_module.escape(t)}</a></li>'
                for a, t in toc_entries
            )
            + "\n</ul></nav>"
        )

        tract_title = _get_tract_base_human(tract_meta, tract_base)
        page_title = f"Asymmetry TLE CovBat-GAM pyAFQ — {tract_title}"
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{html_module.escape(page_title)}</title>
<style>
{_PAGE_STYLE}
</style>
</head>
<body>
<h1>{html_module.escape(page_title)}</h1>
<p>Left TLE n={len(left_tle_subs)}, Right TLE n={len(right_tle_subs)}. In each tract/scalar figure, subjects are included only when both hemispheres have data for that tract and scalar.</p>

{toc_html}

{summary_html}

{"".join(sections_html)}

</body>
</html>
"""
        out_name = f"asymmetry_tle_covbat_pyafq_{tract_base}.html"
        out_path = output_dir / out_name
        out_path.write_text(html)
        index_links.append((out_name, tract_title))

    # Master report: dataset summary + clickable index; optionally per-tract all-scalar summary figures
    index_items = "".join(
        f'  <li><a href="{html_module.escape(out_name)}">{html_module.escape(human_name)}</a></li>\n'
        for out_name, human_name in index_links
    )
    scalar_to_color = scalar_meta.get("scalar_to_color", {})
    model_to_color = cfg.get_model_to_color(scalar_to_color) if scalar_to_color else {}
    scalar_summary_sections: List[str] = []
    scalar_summary_toc: List[Tuple[str, str]] = []  # (anchor, title)
    if all_scalar_summary:
        for tract_base in tqdm(tract_bases, desc="All-scalar summary figures"):
            tract_title = _get_tract_base_human(tract_meta, tract_base)
            tract_anchor = f"scalar_summary_{tract_base}"
            wg_anchor = f"allscalar_{tract_base}"
            ws_anchor = f"withinsubj_{tract_base}"

            wg_b64 = make_all_scalar_summary_figure(
                pyafq_gam_dir,
                tract_base,
                scalars,
                left_tle_subs,
                right_tle_subs,
                tract_meta,
                tract_end_labels,
                scalar_to_human,
                scalar_to_color,
                model_to_color,
            )
            ws_b64 = make_within_subject_summary_figure(
                pyafq_gam_dir,
                tract_base,
                scalars,
                left_tle_subs,
                right_tle_subs,
                tract_meta,
                tract_end_labels,
            )

            if wg_b64 or ws_b64:
                scalar_summary_toc.append((tract_anchor, tract_title))
                parts = [
                    f'<section class="tract-section" id="{html_module.escape(tract_anchor)}">\n'
                    f'<h2>{html_module.escape(tract_title)}</h2>\n'
                ]
                if wg_b64:
                    parts.append(
                        f'<h3 id="{html_module.escape(wg_anchor)}">Within-group effect sizes — GAM z-score asymmetries (q&lt;0.05) across scalars</h3>\n'
                        f'<img src="data:image/png;base64,{wg_b64}" alt="Within-group {tract_title}" style="max-width:100%; height:auto;" />\n'
                    )
                if ws_b64:
                    parts.append(
                        f'<h3 id="{html_module.escape(ws_anchor)}">Within-subject effect sizes across scalars</h3>\n'
                        f'<img src="data:image/png;base64,{ws_b64}" alt="Within-subject {tract_title}" style="max-width:100%; height:auto;" />\n'
                    )
                parts.append("</section>\n")
                scalar_summary_sections.append("".join(parts))

    scalar_summary_html = "\n".join(scalar_summary_sections)
    toc_items = ""
    if scalar_summary_toc:
        toc_items = "\n".join(
            f'  <li><a href="#{html_module.escape(a)}">{html_module.escape(t)}</a></li>'
            for a, t in scalar_summary_toc
        )
        toc_items = (
            '<nav class="toc" aria-label="Scalar summary contents">\n'
            '<h2>Effect sizes across scalars</h2>\n<ul>\n'
            f'{toc_items}\n</ul>\n</nav>\n'
        )

    master_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Asymmetry TLE CovBat-GAM pyAFQ — Master report</title>
<style>
{_PAGE_STYLE}
nav.tract-index {{ margin: 24px 0 32px 0; padding: 20px; background: #f0f4ff; border: 1px solid #b8c8e8; max-width: 600px; }}
nav.tract-index h2 {{ margin: 0 0 12px 0; font-size: 1.1em; border: none; padding: 0; }}
nav.tract-index ul {{ list-style: none; padding-left: 0; margin: 0; }}
nav.tract-index li {{ margin: 6px 0; }}
nav.tract-index a {{ color: #0044aa; text-decoration: none; font-weight: 500; }}
nav.tract-index a:hover {{ text-decoration: underline; }}
section.tract-section {{ margin: 32px 0; padding: 24px; background: #fafbff; border: 1px solid #c4d0e8; border-radius: 6px; }}
section.tract-section h2 {{ margin: 0 0 20px 0; padding: 0 0 12px 0; border-bottom: 2px solid #7a9bd4; font-size: 1.25em; }}
section.tract-section h3 {{ margin: 20px 0 10px 0; font-size: 1.05em; }}
</style>
</head>
<body>
<h1>Asymmetry TLE CovBat-GAM pyAFQ</h1>
<p>Master report. Left TLE n={len(left_tle_subs)}, Right TLE n={len(right_tle_subs)}. Use the index below to open tract-specific reports (dataset summary, CovBat-GAM controls, and TLE figures per tract).</p>

{toc_items}

<nav class="tract-index" aria-label="Tract reports">
<h2 id="tract-index">Tract-specific reports</h2>
<ul>
{index_items}</ul>
</nav>

{summary_html}
{scalar_summary_html}
</body>
</html>
"""
    master_path = output_dir / "asymmetry_tle_covbat_pyafq.html"
    master_path.write_text(master_html)
    print(f"Wrote {len(index_links)} tract reports and master report {master_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tract-specific asymmetry TLE CovBat-GAM pyAFQ reports (one HTML per tract plus index)."
    )
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--tracts", nargs="*", default=None, help=f"Tract base names (default: {DEFAULT_TRACTS})")
    parser.add_argument("--scalars", nargs="*", default=None, help="Scalars to include (default: dti_md dti_fa)")
    parser.add_argument(
        "--all-scalar-summary",
        action="store_true",
        help="Use all non-excluded scalars and add per-tract 2x2 GAM z asymmetry summary figures to the master report.",
    )
    args = parser.parse_args()
    run(
        base_dir=args.base_dir,
        tracts=args.tracts,
        scalars_filter=args.scalars,
        all_scalar_summary=args.all_scalar_summary,
    )


if __name__ == "__main__":
    main()
