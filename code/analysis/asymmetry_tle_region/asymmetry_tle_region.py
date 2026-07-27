import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
# Asymmetry TLE region: ipsi vs contra z-score analysis per scalar for temporal lobe epilepsy subjects,
# plus a factor_z–styled bar plot of signed Cohen's d: F1–F3 (when available) left, microstructural scalars right.
from __future__ import annotations

import argparse
import html as html_module
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from . import config as cfg
except ImportError:
    _pkg_dir = Path(__file__).resolve().parents[0]
    sys.path.insert(0, str(_pkg_dir))
    import config as cfg

# Shared strip/bar styling with microstructural_asymmetry_report_scalars.
_ANALYSIS_DIR = Path(__file__).resolve().parent.parent
_MRS_DIR = _ANALYSIS_DIR / "microstructural_asymmetries"
if str(_MRS_DIR) not in sys.path:
    sys.path.insert(0, str(_MRS_DIR))
try:
    import microstructural_asymmetry_report_scalars as mrs  # noqa: E402
except Exception:  # pragma: no cover
    mrs = None  # type: ignore

try:
    import microstructural_asymmetry_report_factor_z as fz  # noqa: E402
except Exception:  # pragma: no cover
    fz = None  # type: ignore

if plt is not None:
    if mrs is not None and hasattr(mrs, "_configure_matplotlib_georgia"):
        mrs._configure_matplotlib_georgia()
    else:
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = [
            "Georgia",
            "DejaVu Serif",
            "Liberation Serif",
            "Times New Roman",
            "Times",
            "Nimbus Roman",
        ]
        matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
    matplotlib.rcParams["axes.unicode_minus"] = False

DEFAULT_PROJECT_ROOT = project_root()
# Bar typography aligned with microstructural_asymmetry_report_scalars (plot1 / plot2 bars).
_BAR_XTICK_FONTSIZE = 13.5
_BAR_YTICK_FONTSIZE = 13.5
_BAR_YLABEL_FONTSIZE = 15.5
_BAR_TITLE_FONTSIZE = 14
_BAR_FIG_HEIGHT = 4.25
_BAR_CAPSIZE = 2.5
_BAR_EDGE_COLOR = "0.25"
_BAR_EDGE_LW = 0.8
_BAR_ERROR_EC = "black"
_BAR_ERROR_LW = 1.2
_FACTOR_BAR_COLOR = "#6e6e6e"
_FACTOR_BAR_COLOR_NA = "#bdbdbd"
# Fixed y-axis for signed Cohen's d bar figures (factor + microstructural panels).
COHENS_D_BAR_YLIM: Tuple[float, float] = (-0.9, 0.7)
IPSI_Z_BAR_YLIM: Tuple[float, float] = (-1.6, 1.6)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_temporal_subjects(inclusion_path: Path) -> List[str]:
    """Subject IDs where lobe == 'temporal' from the inclusion metadata CSV."""
    df = pd.read_csv(inclusion_path)
    mask = df["lobe"].astype(str).str.strip().str.lower() == "temporal"
    return df.loc[mask, "sub"].astype(str).tolist()


def load_gm_asymmetry(
    region_asym_dir: Path,
    subjects: List[str],
    region: str,
    stat: str = "mean",
) -> pd.DataFrame:
    """Load ipsi/contra z-scores from region_asymmetry_tle/{sub}/{sub}_asym_regions.csv.

    Filters by region and stat.  Returns DataFrame[sub, scalar, ipsi_mean_z, contra_mean_z].
    """
    frames: List[pd.DataFrame] = []
    for sub in subjects:
        csv_path = region_asym_dir / sub / f"{sub}_asym_regions.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            df["sub"] = df["sub"].astype(str)
            df["region"] = df["region"].astype(str)
            mask = (df["region"] == str(region)) & (df["stat"] == stat)
            part = df.loc[mask, ["sub", "scalar", "ipsi_mean_z", "contra_mean_z"]].copy()
            if not part.empty:
                frames.append(part)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["sub", "scalar", "ipsi_mean_z", "contra_mean_z"])
    return pd.concat(frames, ignore_index=True)


def load_wm_asymmetry(
    tract_asym_dir: Path,
    subjects: List[str],
    tract: str,
    segment: str,
) -> pd.DataFrame:
    """Load ipsi/contra z-scores from tract_asymmetry/{sub}/{sub}_asym_scalars.csv.

    Filters by tract and segment.  Returns DataFrame[sub, scalar, ipsi_mean_z, contra_mean_z].
    """
    frames: List[pd.DataFrame] = []
    for sub in subjects:
        csv_path = tract_asym_dir / sub / f"{sub}_asym_scalars.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            df["sub"] = df["sub"].astype(str)
            mask = (df["tract"] == tract) & (df["segment"] == segment)
            part = df.loc[mask, ["sub", "scalar", "ipsi_mean_z", "contra_mean_z"]].copy()
            if not part.empty:
                frames.append(part)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["sub", "scalar", "ipsi_mean_z", "contra_mean_z"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cohens_d(a_vals: List[float], b_vals: List[float]) -> float:
    """Paired Cohen's d: mean(A - B) / std(A - B).  Returns NaN if n<2 or std==0."""
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    diff = a[valid] - b[valid]
    n = len(diff)
    if n < 2:
        return float("nan")
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    if std_diff <= 0:
        return float("nan")
    return mean_diff / std_diff


def cohens_d_jackknife_se(ipsi: List[float], contra: List[float]) -> Tuple[float, float]:
    """Paired Cohen's d and jackknife standard error of d (for mean ± uncertainty bars)."""
    d = cohens_d(ipsi, contra)
    n = len(ipsi)
    if n < 3 or not np.isfinite(d):
        return d, 0.0
    vals: List[float] = []
    for i in range(n):
        ii = [float(ipsi[j]) for j in range(n) if j != i]
        cc = [float(contra[j]) for j in range(n) if j != i]
        vals.append(cohens_d(ii, cc))
    arr = np.asarray(vals, dtype=float)
    if not np.all(np.isfinite(arr)):
        return d, 0.0
    theta_dot = float(np.mean(arr))
    se = float(np.sqrt(((n - 1) / n) * np.sum((arr - theta_dot) ** 2)))
    return d, se


def mean_sem(values: List[float]) -> Tuple[float, float]:
    """Sample mean and standard error of the mean."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    mu = float(np.mean(arr))
    if n < 2:
        return mu, float("nan")
    return mu, float(np.std(arr, ddof=1) / np.sqrt(n))


def compute_scalar_ipsi_mean_sem(
    data_df: pd.DataFrame,
    scalars: List[str],
) -> Tuple[List[float], List[float]]:
    """Per scalar: mean ± SEM of ipsilateral z-scores across temporal TLE subjects."""
    means: List[float] = []
    sems: List[float] = []
    for scalar in scalars:
        sdf = data_df[data_df["scalar"] == scalar].dropna(subset=["ipsi_mean_z"])
        mu, se = mean_sem(sdf["ipsi_mean_z"].tolist())
        means.append(mu)
        sems.append(se)
    return means, sems


def _sort_bars_by_abs_mean(
    labels: List,
    means: List[float],
    sems: List[float],
) -> Tuple[List, List[float], List[float]]:
    """Reorder bars descending by |mean| (NaN means sort last)."""
    if not labels:
        return labels, means, sems
    order = sorted(
        range(len(labels)),
        key=lambda i: abs(float(means[i])) if np.isfinite(means[i]) else -1.0,
        reverse=True,
    )
    return (
        [labels[i] for i in order],
        [means[i] for i in order],
        [sems[i] for i in order],
    )


def _sort_scalars_by_reconstruction_model(scalars: List[str]) -> List[str]:
    """Order scalars NODDI → MAPMRI → DKI → DTI → GQI; within-model order from factor loadings CSV."""
    if mrs is not None:
        return mrs.sort_scalars_by_reconstruction_model(scalars)
    return sorted(scalars)


def save_ipsi_z_bar_means_csv(
    output_path: Path,
    *,
    atlas: str,
    roi_label: str,
    factor_indices: List[int],
    factor_means: List[float],
    factor_sems: List[float],
    sorted_scalars: List[str],
    scalar_means: List[float],
    scalar_sems: List[float],
    scalar_to_human: Dict[str, str],
) -> None:
    """Write bar means ± SEM used in ``ipsi_z_bar.png`` (factors left, scalars right)."""
    rows: List[dict] = []
    for i, fk in enumerate(factor_indices):
        rows.append(
            {
                "atlas": atlas,
                "roi": roi_label,
                "panel": "factor",
                "label": f"F{fk}",
                "display_label": f"F{fk}",
                "mean": factor_means[i] if i < len(factor_means) else float("nan"),
                "sem": factor_sems[i] if i < len(factor_sems) else float("nan"),
                "bar_order": i,
            }
        )
    scalar_offset = len(factor_indices)
    for i, scalar in enumerate(sorted_scalars):
        rows.append(
            {
                "atlas": atlas,
                "roi": roi_label,
                "panel": "scalar",
                "label": scalar,
                "display_label": scalar_to_human.get(scalar, scalar),
                "mean": scalar_means[i] if i < len(scalar_means) else float("nan"),
                "sem": scalar_sems[i] if i < len(scalar_sems) else float("nan"),
                "bar_order": scalar_offset + i,
            }
        )
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _scalar_abbrev(scalar: str) -> str:
    """Short x-axis tick (matches microstructural_asymmetry_report_scalars)."""
    if mrs is not None:
        return mrs._scalar_abbrev(scalar)
    if scalar == "map_ngpar":
        return r"$\mathrm{NG}\parallel$"
    if scalar == "map_ngperp":
        return r"$\mathrm{NG}\perp$"
    if "_" in scalar:
        return scalar.split("_", 1)[-1].upper()
    return scalar.upper()


def _scalar_color_for_bar(scalar: str, scalar_to_color: Dict[str, str]) -> str:
    """Bar color from JSON + model prefix fallback (matches _scalar_color in scalar report)."""
    model_fallback = {"dki": "#7A297F", "dti": "#C43031", "gqi": "#FAA51A", "noddi": "#38489E", "map": "#289144", "rdi": "#C43031"}
    if mrs is not None:
        return mrs._scalar_color(scalar, scalar_to_color, model_fallback)
    if scalar in scalar_to_color:
        return scalar_to_color[scalar]
    for prefix, color in model_fallback.items():
        if scalar.startswith(prefix):
            return color
    return "#333333"


# Fixed figure size for all per-scalar strip/line panels (inches × dpi → consistent PNG dimensions).
_STRIP_FIGSIZE_IN = (3.35, 4.2)
_STRIP_DPI = 200


def plot_ipsi_contra_strip(
    ipsi_zs: List[float],
    contra_zs: List[float],
    scalar_label: str,
    cohens_d_val: float,
    output_path: Path,
    title_color: Optional[str] = None,
) -> None:
    """Strip plot: two strips (ipsilateral, contralateral), y = z-score.

    ``scalar_label`` is shown above the panel (typically abbreviated scalar name, not full human text).

    Y-axis autoscales per scalar. Figure uses a fixed figsize and no tight bbox crop so every
    saved PNG has the same pixel dimensions.
    """
    if plt is None:
        return
    n = len(ipsi_zs)
    if n == 0:
        return
    ipsi_label, contra_label = "Ipsilateral", "Contralateral"
    df = pd.DataFrame({
        "side": [ipsi_label] * n + [contra_label] * n,
        "z": list(ipsi_zs) + list(contra_zs),
    })
    fig, ax = plt.subplots(figsize=_STRIP_FIGSIZE_IN)
    order = [ipsi_label, contra_label]
    for i in range(n):
        ax.plot([0, 1], [ipsi_zs[i], contra_zs[i]], color="gray", alpha=0.25, linewidth=0.9, zorder=0)
    if sns is not None:
        try:
            sns.stripplot(data=df, x="side", y="z", order=order, ax=ax, color="0.3", size=7, alpha=0.7, jitter=False)
        except Exception:
            ax.scatter([0] * n, ipsi_zs, color="0.3", alpha=0.7, s=55)
            ax.scatter([1] * n, contra_zs, color="0.3", alpha=0.7, s=55)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(order)
    else:
        ax.scatter([0] * n, ipsi_zs, color="0.3", alpha=0.7, s=55)
        ax.scatter([1] * n, contra_zs, color="0.3", alpha=0.7, s=55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(order)
    ax.set_xlabel("")
    ax.set_ylabel("z-score", fontsize=16)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)

    d_str = f"Cohen's d = {cohens_d_val:.3f}" if pd.notna(cohens_d_val) else "Cohen's d = N/A"
    title_c = title_color or "black"
    # Scalar name: large + model color; Cohen's d line: smaller, always black
    ax.text(
        0.5,
        1.085,
        scalar_label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=17,
        color=title_c,
    )
    ax.text(
        0.5,
        1.01,
        d_str,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="black",
    )
    # Fixed margins so every PNG is the same size (avoid bbox_inches="tight" per-figure crop).
    # Large ``top`` pulls the axes up so title text sits near the figure edge (less white above).
    fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_STRIP_DPI)
    plt.close(fig)


def plot_cohens_d_bar(
    sorted_scalars: List[str],
    d_values: List[float],
    sem_values: List[float],
    scalar_to_color: Dict[str, str],
    output_path: Path,
    title: str,
    *,
    match_bar_width_to_n_full: Optional[int] = None,
) -> None:
    """Legacy scalar-only bar plot (|Cohen's d|); prefer plot_combined_factor_microstructural_cohens_d_bars.

    Mean = |Cohen's d| (paired ipsi vs contra across subjects); error bars = jackknife SE of d.
    X-axis: abbreviated scalar names only; y-axis label |Cohen's d|.
    If match_bar_width_to_n_full is set (e.g. total scalar count for a top-k subset), bar width
    matches the full plot; figure width is a slightly padded crop of the full layout.
    """
    if plt is None or not sorted_scalars:
        return
    d_arr = np.asarray(d_values, dtype=float)
    sem_arr = np.asarray(sem_values, dtype=float)
    n = len(sorted_scalars)
    n_ref = max(int(match_bar_width_to_n_full), 1) if match_bar_width_to_n_full is not None else max(n, 1)
    means = np.abs(d_arr)
    means = np.where(np.isfinite(means), means, 0.0)
    sems = np.where(np.isfinite(sem_arr), sem_arr, 0.0)

    y_top_bar = float(np.nanmax(means + sems)) * 1.05 if n else 0.1
    y_max = max(y_top_bar, 0.1)
    y_min = 0.0

    bar_w = _mrs_bar_width(n_ref)
    fig_w = _scalar_panel_fig_width(n, n_ref if match_bar_width_to_n_full is not None else None)
    fig, ax = plt.subplots(figsize=(fig_w, _BAR_FIG_HEIGHT))
    x = np.arange(n, dtype=float)
    bar_palette = [_scalar_color_for_bar(s, scalar_to_color) for s in sorted_scalars]
    ax.bar(
        x,
        means,
        yerr=sems,
        width=bar_w,
        capsize=_BAR_CAPSIZE,
        color=bar_palette,
        edgecolor=_BAR_EDGE_COLOR,
        linewidth=_BAR_EDGE_LW,
        error_kw={"ecolor": _BAR_ERROR_EC, "linewidth": _BAR_ERROR_LW},
    )
    ax.set_xticks(x)
    tick_labels = [_scalar_abbrev(s) for s in sorted_scalars]
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=_BAR_XTICK_FONTSIZE)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("|Cohen's d|", fontsize=_BAR_YLABEL_FONTSIZE, fontweight="normal")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=_BAR_XTICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=_BAR_YTICK_FONTSIZE)
    ax.set_title(title, fontsize=_BAR_TITLE_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Factor z-scores (control-referenced), ipsi vs contra Cohen's d — style from
# microstructural_asymmetry_report_factor_z (tissue-panel PNGs).
# ---------------------------------------------------------------------------

def _load_laterality_map(inclusion_path: Path) -> Dict[str, str]:
    """sub -> 'left' | 'right' for temporal lobe rows (same rules as factor_z / mahalanobis)."""
    out: Dict[str, str] = {}
    if not inclusion_path.exists():
        return out
    try:
        df = pd.read_csv(inclusion_path)
        if "sub" not in df.columns or "laterality" not in df.columns:
            return out
        if "lobe" in df.columns:
            df = df[df["lobe"].astype(str).str.strip().str.lower() == "temporal"]
        for _, row in df.iterrows():
            sub = row.get("sub")
            if pd.isna(sub):
                continue
            lat = str(row.get("laterality", "")).strip().lower()
            if lat in ("left", "right"):
                out[str(sub)] = lat
    except Exception:
        pass
    return out


def resolve_factor_score_columns(
    atlas: str,
    region_spec: Optional[str],
    segment_spec: Optional[str],
    paths: Dict[str, Path],
) -> Optional[Tuple[str, str]]:
    """Map report ROI to (left_hemi_col, right_hemi_col) in wide factor score CSVs."""
    if atlas in ("4S156", "Glasser"):
        if not region_spec:
            return None
        pair = cfg.resolve_region_to_pair(str(region_spec).strip(), paths, atlas)
        if pair is None:
            return None
        return pair[0], pair[1]
    if atlas == "HCP1065":
        if not region_spec:
            return None
        resolved = cfg.resolve_wm_tract_segment(
            str(region_spec).strip(), segment_spec or "core", paths
        )
        if resolved is None:
            return None
        tract_left, tract_right, segment_name, _nodes = resolved
        meta = cfg.load_tract_metadata(paths.get("tract_metadata_path"))
        anat = cfg.segment_to_anatomical(tract_left, segment_name, meta)
        return f"{tract_left}_{anat}", f"{tract_right}_{anat}"
    return None


def _normalize_sub(s: object) -> str:
    if fz is not None:
        return fz.normalize_subject_id(s)
    t = str(s).strip()
    if not t.startswith("sub-"):
        t = "sub-" + t
    return t


def compute_factor_z_ipsi_contra_cohens_d(
    paths: Dict[str, Path],
    left_col: str,
    right_col: str,
    temporal_subjects: List[str],
    factor_indices: Tuple[int, ...] = (1, 2, 3),
) -> Tuple[List[float], List[float], str]:
    """Per factor: paired Cohen's d (ipsi − contra z) and jackknife SE of d; empty if unusable."""
    if fz is None:
        return [], [], "microstructural_asymmetry_report_factor_z not importable"
    factor_dir = paths.get("factor_scores_dir")
    if factor_dir is None or not Path(factor_dir).exists():
        return [], [], "factor_scores_dir missing"
    inclusion = paths["inclusion_metadata"]
    lat_map = _load_laterality_map(inclusion)
    temporal_set = {_normalize_sub(s) for s in temporal_subjects}

    d_out: List[float] = []
    se_out: List[float] = []
    for fk in factor_indices:
        ctrl_p = Path(factor_dir) / f"controls_F{fk}_scores.csv"
        epi_p = Path(factor_dir) / f"epilepsy_F{fk}_scores.csv"
        if not ctrl_p.is_file() or not epi_p.is_file():
            d_out.append(float("nan"))
            se_out.append(float("nan"))
            continue
        try:
            ctrl = pd.read_csv(ctrl_p)
            epi = pd.read_csv(epi_p)
        except Exception:
            d_out.append(float("nan"))
            se_out.append(float("nan"))
            continue
        if left_col not in ctrl.columns or right_col not in ctrl.columns:
            d_out.append(float("nan"))
            se_out.append(float("nan"))
            continue
        if left_col not in epi.columns or right_col not in epi.columns:
            d_out.append(float("nan"))
            se_out.append(float("nan"))
            continue
        roi_cols = [left_col, right_col]
        zdf = fz.zscore_epilepsy_vs_controls(ctrl, epi, roi_cols)
        ipsi_vals: List[float] = []
        contra_vals: List[float] = []
        for _, row in zdf.iterrows():
            sub = str(row["subject"])
            if sub not in temporal_set:
                continue
            lat = lat_map.get(sub)
            if lat not in ("left", "right"):
                continue
            lv = row[left_col]
            rv = row[right_col]
            if not (np.isfinite(lv) and np.isfinite(rv)):
                continue
            if lat == "left":
                ipsi_vals.append(float(lv))
                contra_vals.append(float(rv))
            else:
                ipsi_vals.append(float(rv))
                contra_vals.append(float(lv))
        d_val, se_val = cohens_d_jackknife_se(ipsi_vals, contra_vals)
        d_out.append(d_val)
        se_out.append(se_val)

    if not any(np.isfinite(d) for d in d_out):
        return d_out, se_out, "no finite Cohen's d (check columns / subjects / laterality)"
    return d_out, se_out, ""


def compute_factor_z_ipsi_mean_sem(
    paths: Dict[str, Path],
    left_col: str,
    right_col: str,
    temporal_subjects: List[str],
    factor_indices: Tuple[int, ...] = (1, 2, 3),
) -> Tuple[List[float], List[float], str]:
    """Per factor: mean ± SEM of ipsilateral (seizure-side) control-referenced z-scores."""
    if fz is None:
        return [], [], "microstructural_asymmetry_report_factor_z not importable"
    factor_dir = paths.get("factor_scores_dir")
    if factor_dir is None or not Path(factor_dir).exists():
        return [], [], "factor_scores_dir missing"
    inclusion = paths["inclusion_metadata"]
    lat_map = _load_laterality_map(inclusion)
    temporal_set = {_normalize_sub(s) for s in temporal_subjects}

    mean_out: List[float] = []
    sem_out: List[float] = []
    for fk in factor_indices:
        ctrl_p = Path(factor_dir) / f"controls_F{fk}_scores.csv"
        epi_p = Path(factor_dir) / f"epilepsy_F{fk}_scores.csv"
        if not ctrl_p.is_file() or not epi_p.is_file():
            mean_out.append(float("nan"))
            sem_out.append(float("nan"))
            continue
        try:
            ctrl = pd.read_csv(ctrl_p)
            epi = pd.read_csv(epi_p)
        except Exception:
            mean_out.append(float("nan"))
            sem_out.append(float("nan"))
            continue
        if left_col not in ctrl.columns or right_col not in ctrl.columns:
            mean_out.append(float("nan"))
            sem_out.append(float("nan"))
            continue
        if left_col not in epi.columns or right_col not in epi.columns:
            mean_out.append(float("nan"))
            sem_out.append(float("nan"))
            continue
        roi_cols = [left_col, right_col]
        zdf = fz.zscore_epilepsy_vs_controls(ctrl, epi, roi_cols)
        ipsi_vals: List[float] = []
        for _, row in zdf.iterrows():
            sub = str(row["subject"])
            if sub not in temporal_set:
                continue
            lat = lat_map.get(sub)
            if lat not in ("left", "right"):
                continue
            lv = row[left_col]
            rv = row[right_col]
            if not (np.isfinite(lv) and np.isfinite(rv)):
                continue
            if lat == "left":
                ipsi_vals.append(float(lv))
            else:
                ipsi_vals.append(float(rv))
        mu, se = mean_sem(ipsi_vals)
        mean_out.append(mu)
        sem_out.append(se)

    if not any(np.isfinite(m) for m in mean_out):
        return mean_out, sem_out, "no finite ipsilateral factor z (check columns / subjects / laterality)"
    return mean_out, sem_out, ""


def _factor_display_label(factor_index: int) -> str:
    if fz is not None:
        return fz.FACTOR_DISPLAY_LABELS.get(factor_index, f"F{factor_index}")
    return f"F{factor_index}"


def _mrs_bar_width(n_bars: int) -> float:
    """Bar width rule from microstructural_asymmetry_report_scalars plot1_whole_brain_bars."""
    return min(0.74, 20.0 / max(n_bars, 1))


def _fz_bar_width(n_bars: int) -> float:
    if fz is not None:
        return float(fz._bar_width_for_n(n_bars))
    return min(0.58, 16.0 / max(n_bars, 1))


def _scalar_panel_fig_width(n_bars: int, n_ref: Optional[int] = None) -> float:
    n = max(n_bars, 1)
    n_ref_n = max(int(n_ref), 1) if n_ref is not None else n
    fig_w_full = max(7.0, 0.32 * n_ref_n)
    if n_ref is not None and n_ref_n > n:
        return fig_w_full * (n / n_ref_n) * 1.14
    return max(7.0, 0.32 * n)


def _factor_panel_width_inches(n_bars: int) -> float:
    """Factor-axis width (matches microstructural_asymmetry_report_factor_z)."""
    if fz is not None:
        return float(fz._panel_width_inches(n_bars))
    return max(2.0, 0.20 * max(n_bars, 1))


def _side_by_side_gridspec_kw() -> Dict[str, object]:
    """Factor : scalar axis width = 1 : 8, same as factor_z tissue panels."""
    ratio = int(fz.SCALAR_PANEL_WIDTH_RATIO) if fz is not None else 8
    wspace = float(fz._FZ_INNER_WSPACE) if fz is not None else 0.04
    return {"width_ratios": [1, ratio], "wspace": wspace}


def _side_by_side_figsize_inches(n_factor_bars: int) -> float:
    w_factor = _factor_panel_width_inches(n_factor_bars)
    if fz is not None:
        return float(fz._side_by_side_figsize_inches(w_factor))
    return w_factor * 9.0


def _side_by_side_fig_height_inches() -> float:
    if fz is not None:
        return float(fz._FZ_TISSUE_PANEL_HEIGHT_IN)
    return 9.0


def _side_by_side_figsize(n_factor_bars: int) -> Tuple[float, float]:
    """(width, height) in inches — matches factor_z tissue side-by-side PNGs."""
    return _side_by_side_figsize_inches(n_factor_bars), _side_by_side_fig_height_inches()


def _side_by_side_subplots_adjust(fig) -> None:
    wspace = float(fz._FZ_INNER_WSPACE) if fz is not None else 0.04
    fig.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.22, wspace=wspace)


def _factor_only_subplots_adjust(fig) -> None:
    """Margins for narrow factor-only figure (factor_z per-quadrant factor panel)."""
    fig.subplots_adjust(left=0.12, right=0.99, top=0.98, bottom=0.22)


def _fz_tissue_panel_rc() -> Dict[str, object]:
    if fz is not None:
        return dict(fz.PLOT_RC_TISSUE_PANELS)
    return {}


_FZ_TISSUE_BAR_EDGEWIDTH = 3.0


def _decorate_signed_cohens_bar_ax(
    ax: "plt.Axes",
    x: np.ndarray,
    x_labels: List[str],
    m_plot: List[float],
    sems: List[float],
    colors: List[str],
    *,
    ylabel: Optional[str] = None,
    hide_yticklabels: bool = False,
    bar_width_n_ref: Optional[int] = None,
    bar_width: Optional[float] = None,
) -> None:
    n = len(x_labels)
    n_ref = max(int(bar_width_n_ref), 1) if bar_width_n_ref is not None else max(n, 1)
    bar_w = float(bar_width) if bar_width is not None else _mrs_bar_width(n_ref)
    ax.bar(
        x,
        m_plot,
        bar_w,
        yerr=sems,
        capsize=_BAR_CAPSIZE,
        color=colors,
        edgecolor=_BAR_EDGE_COLOR,
        linewidth=_BAR_EDGE_LW,
        error_kw={"ecolor": _BAR_ERROR_EC, "linewidth": _BAR_ERROR_LW},
    )
    ax.axhline(0, color="k", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=_BAR_XTICK_FONTSIZE)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=_BAR_XTICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=_BAR_YTICK_FONTSIZE)
    if hide_yticklabels:
        ax.tick_params(labelleft=False)
    elif ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=_BAR_YLABEL_FONTSIZE, fontweight="normal")
    if fz is not None:
        pad = float(fz._xlim_side_pad(bar_w))
    else:
        pad = max(0.35, 0.5 * bar_w + 0.14)
    ax.set_xlim(-pad, max(n - 1, 0) + pad)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_factor_panel_on_ax(
    ax: "plt.Axes",
    factor_indices: List[int],
    d_factor: List[float],
    se_factor: List[float],
    *,
    ylabel: str = "Cohen's d",
) -> None:
    n_f = len(factor_indices)
    x = np.arange(n_f, dtype=float)
    m_plot = [float(d) if np.isfinite(d) else 0.0 for d in d_factor]
    sems = [float(s) if np.isfinite(s) else 0.0 for s in se_factor]
    colors = [
        _FACTOR_BAR_COLOR if np.isfinite(d) else _FACTOR_BAR_COLOR_NA for d in d_factor
    ]
    x_labels = [_factor_display_label(k) for k in factor_indices]
    if fz is not None:
        fz._decorate_bar_axes(
            ax,
            x,
            x_labels,
            m_plot,
            sems,
            colors,
            title=None,
            ylabel=ylabel,
            bar_edgewidth=_FZ_TISSUE_BAR_EDGEWIDTH,
            bar_width=_fz_bar_width(n_f),
        )
    else:
        _decorate_signed_cohens_bar_ax(
            ax,
            x,
            x_labels,
            m_plot,
            sems,
            colors,
            ylabel=ylabel,
            bar_width=_fz_bar_width(n_f),
        )


def _plot_scalar_panel_on_ax(
    ax: "plt.Axes",
    sorted_scalars: List[str],
    d_scalar: List[float],
    se_scalar: List[float],
    scalar_to_color: Dict[str, str],
    *,
    bar_width_n_ref: Optional[int] = None,
    use_fz_bar_layout: bool = False,
) -> None:
    n_s = len(sorted_scalars)
    x = np.arange(n_s, dtype=float)
    m_plot = [float(d) if np.isfinite(d) else 0.0 for d in d_scalar]
    sems = [float(s) if np.isfinite(s) else 0.0 for s in se_scalar]
    colors = [_scalar_color_for_bar(s, scalar_to_color) for s in sorted_scalars]
    tick_lab = [_scalar_abbrev(s) for s in sorted_scalars]
    n_ref = max(int(bar_width_n_ref), 1) if bar_width_n_ref is not None else max(n_s, 1)
    if use_fz_bar_layout and fz is not None:
        fz._decorate_bar_axes(
            ax,
            x,
            tick_lab,
            m_plot,
            sems,
            colors,
            title=None,
            ylabel=None,
            bar_edgewidth=_FZ_TISSUE_BAR_EDGEWIDTH,
            bar_width=_fz_bar_width(n_ref),
        )
    else:
        _decorate_signed_cohens_bar_ax(
            ax,
            x,
            tick_lab,
            m_plot,
            sems,
            colors,
            ylabel=None,
            hide_yticklabels=True,
            bar_width_n_ref=bar_width_n_ref,
        )


def plot_combined_factor_microstructural_cohens_d_bars(
    factor_indices: List[int],
    d_factor: List[float],
    se_factor: List[float],
    sorted_scalars: List[str],
    d_scalar: List[float],
    se_scalar: List[float],
    scalar_to_color: Dict[str, str],
    output_path: Path,
    *,
    dpi: int = 150,
    match_scalar_bar_width_to_n_full: Optional[int] = None,
    ylabel: str = "Cohen's d",
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Signed Cohen's d ± jackknife SE: factor z-scores (left) and microstructural scalars (right).

    Combined layout matches microstructural_asymmetry_report_factor_z tissue panels
    (PLOT_RC_TISSUE_PANELS typography, bar width, spacing).
    """
    if plt is None or not sorted_scalars:
        return
    n_f = len(factor_indices)
    n_s = len(sorted_scalars)
    if n_f > 0 and (len(d_factor) != n_f or len(se_factor) != n_f):
        return
    if len(d_scalar) != n_s or len(se_scalar) != n_s:
        return
    ylim = COHENS_D_BAR_YLIM if ylim is None else ylim
    scalar_bar_ref = (
        max(int(match_scalar_bar_width_to_n_full), 1)
        if match_scalar_bar_width_to_n_full is not None
        else n_s
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if n_f > 0:
        fig_w, fig_h = _side_by_side_figsize(n_f)
        with plt.rc_context(rc=_fz_tissue_panel_rc()):
            fig, (ax_f, ax_s) = plt.subplots(
                1,
                2,
                figsize=(fig_w, fig_h),
                sharey=True,
                gridspec_kw=_side_by_side_gridspec_kw(),
            )
            _plot_factor_panel_on_ax(ax_f, factor_indices, d_factor, se_factor, ylabel=ylabel)
            _plot_scalar_panel_on_ax(
                ax_s,
                sorted_scalars,
                d_scalar,
                se_scalar,
                scalar_to_color,
                bar_width_n_ref=scalar_bar_ref,
                use_fz_bar_layout=True,
            )
            ax_f.set_ylim(ylim)
            ax_s.set_ylim(ylim)
            ax_s.tick_params(labelleft=False)
            _side_by_side_subplots_adjust(fig)
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return
    else:
        fig_w = _scalar_panel_fig_width(n_s, scalar_bar_ref)
        fig, ax_s = plt.subplots(figsize=(fig_w, _BAR_FIG_HEIGHT))
        _plot_scalar_panel_on_ax(
            ax_s,
            sorted_scalars,
            d_scalar,
            se_scalar,
            scalar_to_color,
            bar_width_n_ref=scalar_bar_ref,
        )
        ax_s.set_ylabel(ylabel, fontsize=_BAR_YLABEL_FONTSIZE, fontweight="normal")
        ax_s.tick_params(axis="y", labelsize=_BAR_YTICK_FONTSIZE)
        ax_s.set_ylim(ylim)
        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_factor_only_cohens_d_bars(
    factor_indices: List[int],
    d_factor: List[float],
    se_factor: List[float],
    output_path: Path,
    *,
    ylim: Tuple[float, float] = COHENS_D_BAR_YLIM,
    ylabel: str = "Cohen's d",
    dpi: int = 150,
) -> None:
    """Factor panel only: same styling and axis width as the left panel in cohens_d_bar.png."""
    if plt is None or not factor_indices:
        return
    n_f = len(factor_indices)
    if len(d_factor) != n_f or len(se_factor) != n_f:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = _factor_panel_width_inches(n_f)
    fig_h = _side_by_side_fig_height_inches()
    with plt.rc_context(rc=_fz_tissue_panel_rc()):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        _plot_factor_panel_on_ax(ax, factor_indices, d_factor, se_factor, ylabel=ylabel)
        ax.set_ylim(ylim)
        _factor_only_subplots_adjust(fig)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sanitize_slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    return s.strip("_") or "region"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    base_dir: Optional[Path] = None,
    region: Optional[str] = None,
    atlas: str = "4S156",
    segment: Optional[str] = None,
) -> None:
    base_dir = Path(base_dir or DEFAULT_PROJECT_ROOT)
    paths = cfg.get_paths(base_dir)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- subjects ---
    inclusion_path = paths["inclusion_metadata"]
    if not inclusion_path.exists():
        print(f"Inclusion metadata not found: {inclusion_path}. Exiting.")
        return
    subjects = load_temporal_subjects(inclusion_path)
    if not subjects:
        print("No temporal lobe subjects found. Exiting.")
        return
    n_subjects = len(subjects)

    # --- region / tract identification ---
    is_wm = atlas == "HCP1065"
    if is_wm:
        tract = region
        seg = segment or "core"
        if not tract:
            print("For HCP1065, --region (tract name e.g. AF) is required. Exiting.")
            return
        slug = _sanitize_slug(f"{tract}_{seg}")
        roi_label = f"{tract} ({seg})"
    else:
        if not region:
            region = "Hippocampus"
        slug = _sanitize_slug(region)
        roi_label = region

    # Region-specific subfolder: figures and CSV only; HTML report in parent (atlas dir)
    region_subfolder = output_dir / atlas / slug
    figures_dir = region_subfolder / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- load pre-computed asymmetry data ---
    if is_wm:
        data_df = load_wm_asymmetry(paths["tract_asymmetry_dir"], subjects, tract, seg)
    else:
        data_df = load_gm_asymmetry(paths["region_asymmetry_dir"], subjects, region)

    if data_df.empty:
        print(f"No asymmetry data found for {roi_label}. Exiting.")
        return

    # --- scalars: config order intersected with data ---
    config_scalars = cfg.get_scalar_labels(paths)
    data_scalars = set(data_df["scalar"].unique())
    scalars = [s for s in config_scalars if s in data_scalars]
    if not scalars:
        scalars = sorted(data_scalars)
    if not scalars:
        print("No scalars available. Exiting.")
        return

    scalar_meta = cfg.load_scalar_metadata(paths)
    scalar_to_human: Dict[str, str] = scalar_meta.get("scalar_to_human", {})
    scalar_to_color: Dict[str, str] = scalar_meta.get("scalar_to_color", {})

    # --- Cohen's d (ipsi − contra) per scalar + jackknife SE ---
    scalar_to_data: Dict[str, Tuple[List[float], List[float], float, float]] = {}
    for scalar in scalars:
        sdf = data_df[data_df["scalar"] == scalar].dropna(subset=["ipsi_mean_z", "contra_mean_z"])
        ipsi = sdf["ipsi_mean_z"].tolist()
        contra = sdf["contra_mean_z"].tolist()
        d, d_se = cohens_d_jackknife_se(ipsi, contra)
        scalar_to_data[scalar] = (ipsi, contra, d, d_se)

    sorted_scalars = sorted(
        scalars,
        key=lambda s: abs(scalar_to_data[s][2]) if pd.notna(scalar_to_data[s][2]) else -1.0,
        reverse=True,
    )

    # --- CSV output ---
    rows: List[dict] = []
    for scalar in sorted_scalars:
        sdf = data_df[data_df["scalar"] == scalar].dropna(subset=["ipsi_mean_z", "contra_mean_z"])
        for _, r in sdf.iterrows():
            rows.append({
                "sub": r["sub"], "scalar": scalar,
                "ipsi_mean_z": r["ipsi_mean_z"], "contra_mean_z": r["contra_mean_z"],
            })
    if rows:
        csv_path = region_subfolder / f"z_asymmetry_tle_region_{slug}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    # --- Signed Cohen's d bars: factors (when available) + microstructural scalars (scalar-report bar style) ---
    factor_msg = ""
    fc_pair: Optional[Tuple[str, str]] = None
    plot_factors = False
    d_f: List[float] = []
    se_f: List[float] = []
    factor_indices_plot: List[int] = []

    if atlas in ("4S156", "Glasser", "HCP1065"):
        if is_wm:
            fc_pair = resolve_factor_score_columns(atlas, tract, seg, paths)
        else:
            fc_pair = resolve_factor_score_columns(atlas, region, None, paths)
    if fc_pair is None:
        if atlas in ("4S156", "Glasser", "HCP1065"):
            factor_msg = (
                "Factor bars omitted: could not resolve ROI columns in wide "
                "factor score CSVs (check region / tract spelling for this atlas)."
            )
    else:
        lc_fc, rc_fc = fc_pair
        d_f, se_f, fz_err = compute_factor_z_ipsi_contra_cohens_d(
            paths,
            lc_fc,
            rc_fc,
            subjects,
            factor_indices=tuple(
                fz.DEFAULT_FACTOR_INDICES if fz is not None else (1, 2, 3)
            ),
        )
        if fz_err:
            factor_msg = f"Factor bars omitted: {fz_err}"
        else:
            plot_factors = True
            factor_indices_plot = (
                list(fz.DEFAULT_FACTOR_INDICES)
                if fz is not None
                else [1, 2, 3]
            )

    d_vals = [scalar_to_data[s][2] for s in sorted_scalars]
    sem_vals = [scalar_to_data[s][3] for s in sorted_scalars]
    bar_png = "cohens_d_bar.png"
    plot_combined_factor_microstructural_cohens_d_bars(
        factor_indices_plot if plot_factors else [],
        d_f if plot_factors else [],
        se_f if plot_factors else [],
        sorted_scalars,
        d_vals,
        sem_vals,
        scalar_to_color,
        figures_dir / bar_png,
    )
    if (figures_dir / bar_png).is_file():
        print(f"Wrote {figures_dir / bar_png}")

    model_grouped_scalars = _sort_scalars_by_reconstruction_model(scalars)
    bar_grouped_png = "cohens_d_bar_grouped-model.png"
    plot_combined_factor_microstructural_cohens_d_bars(
        factor_indices_plot if plot_factors else [],
        d_f if plot_factors else [],
        se_f if plot_factors else [],
        model_grouped_scalars,
        [scalar_to_data[s][2] for s in model_grouped_scalars],
        [scalar_to_data[s][3] for s in model_grouped_scalars],
        scalar_to_color,
        figures_dir / bar_grouped_png,
    )
    if (figures_dir / bar_grouped_png).is_file():
        print(f"Wrote {figures_dir / bar_grouped_png}")

    # --- Ipsilateral mean z (seizure-side region only): factors + microstructural scalars ---
    ipsi_scalar_means, ipsi_scalar_sems = compute_scalar_ipsi_mean_sem(data_df, scalars)
    ipsi_mean_by_scalar = dict(zip(scalars, ipsi_scalar_means))
    ipsi_sem_by_scalar = dict(zip(scalars, ipsi_scalar_sems))
    ipsi_sorted_scalars, ipsi_scalar_means, ipsi_scalar_sems = _sort_bars_by_abs_mean(
        scalars, ipsi_scalar_means, ipsi_scalar_sems
    )
    plot_ipsi_factors = False
    ipsi_factor_means: List[float] = []
    ipsi_factor_sems: List[float] = []
    ipsi_factor_msg = ""
    if fc_pair is not None:
        lc_fc, rc_fc = fc_pair
        ipsi_factor_means, ipsi_factor_sems, ipsi_fz_err = compute_factor_z_ipsi_mean_sem(
            paths,
            lc_fc,
            rc_fc,
            subjects,
            factor_indices=tuple(
                fz.DEFAULT_FACTOR_INDICES if fz is not None else (1, 2, 3)
            ),
        )
        if ipsi_fz_err:
            ipsi_factor_msg = f"Ipsilateral factor bars omitted: {ipsi_fz_err}"
        else:
            plot_ipsi_factors = True
    elif atlas in ("4S156", "Glasser", "HCP1065"):
        ipsi_factor_msg = factor_msg or (
            "Ipsilateral factor bars omitted: could not resolve ROI columns in wide "
            "factor score CSVs (check region / tract spelling for this atlas)."
        )

    ipsi_z_png = "ipsi_z_bar.png"
    ipsi_factor_order = (
        list(fz.DEFAULT_FACTOR_INDICES) if fz is not None else [1, 2, 3]
    )
    plot_combined_factor_microstructural_cohens_d_bars(
        ipsi_factor_order if plot_ipsi_factors else [],
        ipsi_factor_means if plot_ipsi_factors else [],
        ipsi_factor_sems if plot_ipsi_factors else [],
        ipsi_sorted_scalars,
        ipsi_scalar_means,
        ipsi_scalar_sems,
        scalar_to_color,
        figures_dir / ipsi_z_png,
        ylabel="z",
        ylim=IPSI_Z_BAR_YLIM,
    )
    if (figures_dir / ipsi_z_png).is_file():
        print(f"Wrote {figures_dir / ipsi_z_png}")

    ipsi_model_grouped = _sort_scalars_by_reconstruction_model(scalars)
    ipsi_z_grouped_png = "ipsi_z_bar_grouped-model.png"
    plot_combined_factor_microstructural_cohens_d_bars(
        ipsi_factor_order if plot_ipsi_factors else [],
        ipsi_factor_means if plot_ipsi_factors else [],
        ipsi_factor_sems if plot_ipsi_factors else [],
        ipsi_model_grouped,
        [ipsi_mean_by_scalar[s] for s in ipsi_model_grouped],
        [ipsi_sem_by_scalar[s] for s in ipsi_model_grouped],
        scalar_to_color,
        figures_dir / ipsi_z_grouped_png,
        ylabel="z",
        ylim=IPSI_Z_BAR_YLIM,
    )
    if (figures_dir / ipsi_z_grouped_png).is_file():
        print(f"Wrote {figures_dir / ipsi_z_grouped_png}")

    ipsi_z_csv = figures_dir / "ipsi_z_bar_means.csv"
    save_ipsi_z_bar_means_csv(
        ipsi_z_csv,
        atlas=atlas,
        roi_label=roi_label,
        factor_indices=ipsi_factor_order if plot_ipsi_factors else [],
        factor_means=ipsi_factor_means if plot_ipsi_factors else [],
        factor_sems=ipsi_factor_sems if plot_ipsi_factors else [],
        sorted_scalars=ipsi_sorted_scalars,
        scalar_means=ipsi_scalar_means,
        scalar_sems=ipsi_scalar_sems,
        scalar_to_human=scalar_to_human,
    )
    if ipsi_z_csv.is_file():
        print(f"Wrote {ipsi_z_csv}")

    ipsi_z_factor_png = f"ipsi_z_factor_{slug}.png"
    if plot_ipsi_factors:
        plot_factor_only_cohens_d_bars(
            ipsi_factor_order,
            ipsi_factor_means,
            ipsi_factor_sems,
            figures_dir / ipsi_z_factor_png,
            ylim=IPSI_Z_BAR_YLIM,
            ylabel="Factor z-score",
        )
        if (figures_dir / ipsi_z_factor_png).is_file():
            print(f"Wrote {figures_dir / ipsi_z_factor_png}")

    bar_factors_png = "cohens_d_bar_factors.png"
    if plot_factors:
        plot_factor_only_cohens_d_bars(
            factor_indices_plot,
            d_f,
            se_f,
            figures_dir / bar_factors_png,
        )
        if (figures_dir / bar_factors_png).is_file():
            print(f"Wrote {figures_dir / bar_factors_png}")

    top_n = 10
    top_scalars = sorted_scalars[: min(top_n, len(sorted_scalars))]
    bar_top_png = "cohens_d_bar_top10.png"
    plot_combined_factor_microstructural_cohens_d_bars(
        factor_indices_plot if plot_factors else [],
        d_f if plot_factors else [],
        se_f if plot_factors else [],
        top_scalars,
        [scalar_to_data[s][2] for s in top_scalars],
        [scalar_to_data[s][3] for s in top_scalars],
        scalar_to_color,
        figures_dir / bar_top_png,
        match_scalar_bar_width_to_n_full=len(sorted_scalars),
    )
    if (figures_dir / bar_top_png).is_file():
        print(f"Wrote {figures_dir / bar_top_png}")

    # --- strip plots per scalar (raw z-scores); fixed fig size → identical PNG dimensions ---
    cells: List[str] = []
    for scalar in sorted_scalars:
        ipsi, contra, d_val, _d_se = scalar_to_data[scalar]
        scalar_display = scalar_to_human.get(scalar, scalar)
        png_name = f"{scalar}.png"
        plot_ipsi_contra_strip(
            ipsi,
            contra,
            _scalar_abbrev(scalar),
            d_val,
            figures_dir / png_name,
            title_color=scalar_to_color.get(scalar, "#333333"),
        )
        d_str = f"{d_val:.3f}" if pd.notna(d_val) else "N/A"
        cells.append(
            f'<div class="cell">'
            f'<img src="{slug}/figures/{png_name}" alt="{html_module.escape(scalar_display)}" />'
            f'<p class="caption">{html_module.escape(scalar_display)} &mdash; Cohen\'s d = {d_str}</p>'
            f'</div>'
        )

    # --- HTML report (in atlas dir, named {slug}.html; figures/csv in {slug}/) ---
    raw_grid_html = "".join(cells)
    bar_fig_path = figures_dir / bar_png
    ipsi_z_fig_path = figures_dir / ipsi_z_png
    bar_summary = (
        f"Signed paired Cohen&apos;s <em>d</em> (ipsilateral &minus; contralateral) with jackknife SE of "
        f"<em>d</em>. When available, control-referenced factor scores (Overall, Non-Gaussian, Anisotropic) "
        f"appear in the left panel; microstructural scalars sorted by |<em>d</em>| are in the right panel "
        f"(bar colors by model; <code>gqi_iso</code> excluded). "
        f"Temporal TLE cohort, n={n_subjects}."
    )
    if factor_msg:
        bar_summary += " " + html_module.escape(factor_msg)
    ipsi_z_summary = (
        f"Mean control-referenced z-scores in each patient&apos;s <strong>ipsilateral</strong> "
        f"region only (left hemisphere for left TLE, right for right TLE), ± SEM across "
        f"n={n_subjects} temporal TLE subjects. Factor scores (Overall, Non-Gaussian, "
        f"Anisotropic) appear in the left panel when available; microstructural scalars "
        f"are sorted by |z| (descending)."
    )
    if ipsi_factor_msg:
        ipsi_z_summary += " " + html_module.escape(ipsi_factor_msg)
    bars_block = (
        f'<h2 id="cohens-d-bars" class="section-heading">Ipsi&ndash;contra asymmetry (Cohen&apos;s <em>d</em>)</h2>\n'
        f'<p class="summary">{bar_summary}</p>\n'
    )
    if bar_fig_path.is_file():
        bars_block += (
            f'<div class="barplot-wrapper"><img src="{slug}/figures/{bar_png}" '
            f'alt="Signed Cohen d: factors and microstructural scalars" class="barplot-img" /></div>\n'
        )
        bar_grouped_path = figures_dir / bar_grouped_png
        if bar_grouped_path.is_file():
            bars_block += (
                f'<div class="barplot-wrapper"><img src="{slug}/figures/{bar_grouped_png}" '
                f'alt="Signed Cohen d: factors and microstructural scalars (grouped by model)" '
                f'class="barplot-img" /></div>\n'
            )
        bar_factors_path = figures_dir / bar_factors_png
        if bar_factors_path.is_file():
            bars_block += (
                f'<div class="barplot-wrapper"><img src="{slug}/figures/{bar_factors_png}" '
                f'alt="Signed Cohen d: factor scores only" class="barplot-img" /></div>\n'
            )
        bars_block += (
            f'<div class="barplot-wrapper"><img src="{slug}/figures/{bar_top_png}" '
            f'alt="Signed Cohen d: top 10 microstructural scalars by |d|" class="barplot-img" /></div>\n'
        )
    bars_block += (
        f'<h2 id="ipsi-z-bars" class="section-heading">Ipsilateral region (mean z)</h2>\n'
        f'<p class="summary">{ipsi_z_summary}</p>\n'
    )
    if ipsi_z_fig_path.is_file():
        bars_block += (
            f'<div class="barplot-wrapper"><img src="{slug}/figures/{ipsi_z_png}" '
            f'alt="Mean ipsilateral z: factors and microstructural scalars" class="barplot-img" /></div>\n'
        )
        ipsi_z_grouped_path = figures_dir / ipsi_z_grouped_png
        if ipsi_z_grouped_path.is_file():
            bars_block += (
                f'<div class="barplot-wrapper"><img src="{slug}/figures/{ipsi_z_grouped_png}" '
                f'alt="Mean ipsilateral z: factors and microstructural scalars (grouped by model)" '
                f'class="barplot-img" /></div>\n'
            )

    report_path = output_dir / atlas / f"{slug}.html"
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Asymmetry TLE &mdash; {html_module.escape(roi_label)}</title>
<style>
body {{ font-family: sans-serif; margin: 20px; }}
h1 {{ font-size: 1.5em; }}
h2.section-heading {{ font-size: 1.15em; margin-top: 32px; padding-top: 16px; border-top: 2px solid #333; }}
h2.section-heading:first-of-type {{ margin-top: 16px; padding-top: 0; border-top: none; }}
.summary {{ margin-bottom: 20px; }}
.barplot-wrapper {{ margin: 16px 0; }}
.barplot-wrapper .barplot-img {{ max-width: 100%; height: auto; }}
.grid-6col {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 16px; margin-top: 12px; }}
.grid-6col .cell {{ text-align: center; border: 1px solid #ddd; padding: 8px; }}
.grid-6col .cell img {{ max-width: 100%; height: auto; }}
.grid-6col .caption {{ font-size: 0.85em; margin: 8px 0 0 0; }}
</style>
</head>
<body>
<h1>Asymmetry TLE &mdash; {html_module.escape(roi_label)}</h1>
<p class="summary">Atlas: {html_module.escape(atlas)}. n={n_subjects} temporal lobe TLE subjects with documented seizure laterality. Summary bar charts show <strong>signed</strong> Cohen&apos;s <em>d</em> (ipsi &minus; contra) and mean ipsilateral z-scores (seizure-side region only), plus the top 10 microstructural scalars by |<em>d</em>| at matched bar width. Per-scalar panels below show raw ipsilateral vs contralateral z-scores.</p>

{bars_block}
<h2 id="raw" class="section-heading">Raw z-scores (ipsilateral vs contralateral)</h2>
<div class="grid-6col">{raw_grid_html}</div>

</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    print(f"Wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Asymmetry TLE region: ipsi vs contra z-score analysis per scalar for temporal lobe epilepsy subjects.",
    )
    parser.add_argument("--base-dir", type=Path, default=None, help="Project base directory")
    parser.add_argument(
        "--region", default=None,
        help="Region name matching the pre-computed asymmetry CSVs "
             "(GM: e.g. Hippocampus, V1; WM/HCP1065: tract base e.g. AF). "
             "Default for GM: Hippocampus.",
    )
    parser.add_argument(
        "--atlas", default="4S156",
        help="Atlas: 4S156 or Glasser (GM, loads from region_asymmetry_tle), "
             "or HCP1065 (WM, loads from tract_asymmetry). Default: 4S156.",
    )
    parser.add_argument(
        "--segment", default="core",
        help="For HCP1065 only: segment label (e.g. core, A, P, I, S). Default: core.",
    )
    args = parser.parse_args()
    run(
        base_dir=args.base_dir,
        region=args.region,
        atlas=args.atlas,
        segment=args.segment,
    )


if __name__ == "__main__":
    main()
