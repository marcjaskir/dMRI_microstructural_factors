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
Factor score asymmetry report: z-score epilepsy factor scores per ROI vs controls,
then paired ipsi-contra Cohen's d per ROI; summarize by tissue quadrant (1×4 bar plot).

Per tissue quadrant, factor F1–F3 bars (left) and microstructural scalar bars (right)
are shown side-by-side with a shared y-axis (Cohen's d labeled on the factor panel only).
Scalars are signed Cohen's d (mean ± SEM across ROIs), ordered per quadrant by descending
|mean signed d| (ties: mean |d|), when tract/region asymmetry data are available.

Reads wide CSVs from derivatives/analysis/factor_z-scores/factor_scores/.

Writes ``factor_score_z_ipsi_contra_cohens_d_by_tissue.csv``: mean factor- and
scalar ipsi-contra Cohen's d per tissue class (cortex, subcortex, association,
projection), matching the combined bar plots.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from microstructural_asymmetry_report_mahalanobis import (  # noqa: E402
    EXCLUDED_VOLUMETRIC_ASYMMETRY_TRACT_BASES,
    INCLUSION_PATH,
    _cohens_d_paired,
    _get_4s_subcortical_base_to_labels,
    _get_glasser_bases,
    _load_tract_metadata,
    _wm_roi_to_tract_segment,
    _wm_tract_base_key,
)

import microstructural_asymmetry_report_scalars as mrs  # noqa: E402

PROJECT_ROOT = project_root()
FACTOR_SCORES_DIR = analysis_dir() / "factor_z-scores" / "factor_scores"
OUTPUT_DIR_DEFAULT = analysis_dir() / "microstructural_asymmetries"

# Figure width (factor-only legacy figures).
_FIG_WIDTH_MULT = 1.2
# Side-by-side layout: scalar axis width = this × factor axis width.
SCALAR_PANEL_WIDTH_RATIO = 8
# Compact layout for factor_score_z_ipsi_contra_cohens_d_* (tight bars + tight subplot gaps).
_FZ_BAR_WIDTH_MAX = 0.58
_FZ_BAR_WIDTH_PER_BAR = 16.0  # cap width for many-bar scalar panels
_FZ_BAR_XLIM_SIDE_PAD = 0.35  # minimum x-axis margin (see _xlim_side_pad)
_FZ_OUTER_WSPACE = 0.07
_FZ_INNER_WSPACE = 0.04
_FZ_PANEL_WIDTH_PER_BAR_IN = 0.20
_FZ_PANEL_WIDTH_MIN_IN = 2.0
_FZ_1X4_HEIGHT_IN = 9.0
_FZ_TISSUE_PANEL_HEIGHT_IN = 9.0
# Fixed Cohen's d y-axis for all factor/scalar bar figures.
BAR_PLOT_YLIM: Tuple[float, float] = (-0.285, .25)


def _configure_georgia_font() -> None:
    """Prefer Georgia for all figures from this script; register TTF on Linux if needed."""
    try:
        fm.findfont("Georgia", fallback_to_default=False)
    except Exception:
        for _d in (
            PROJECT_ROOT / "data" / "fonts",
            Path("/usr/share/fonts/truetype/msttcorefonts"),
            Path("/usr/share/fonts/truetype/microsoft"),
            Path("/usr/local/share/fonts/truetype/msttcorefonts"),
        ):
            if not _d.is_dir():
                continue
            for _f in _d.iterdir():
                if _f.suffix.lower() == ".ttf" and "georgia" in _f.name.lower() and "bold" not in _f.name.lower():
                    try:
                        fm.fontManager.addfont(str(_f))
                        break
                    except Exception:
                        pass
            else:
                continue
            break
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_georgia_font()

DEFAULT_FACTOR_INDICES: List[int] = [1, 2, 3]
DEPRECATED_FACTOR_INDICES = frozenset({4})

FACTOR_DISPLAY_LABELS: Dict[int, str] = {
    1: "Overall",
    2: "Non-Gaussian",
    3: "Anisotropic",
}

QUADRANT_ORDER = ("glasser_cortex", "4s_subcortex", "wm_association", "wm_projection")
QUADRANT_TITLES = {
    "glasser_cortex": "Glasser cortex (GM)",
    "4s_subcortex": "4S156 subcortex (GM)",
    "wm_association": "HCP1065 association WM (thirds)",
    "wm_projection": "HCP1065 projection WM (thirds)",
}

# Short tissue labels for summary CSV (matches plot quadrants).
TISSUE_CLASS_LABELS: Dict[str, str] = {
    "glasser_cortex": "cortex",
    "4s_subcortex": "subcortex",
    "wm_association": "association",
    "wm_projection": "projection",
}

# Bar colors: grey for valid Cohen's d, lighter grey for missing (bar height 0)
BAR_COLOR = "#6e6e6e"
BAR_COLOR_NA = "#bdbdbd"

MODEL_FALLBACK_COLORS = {
    "dki": "#7A297F",
    "dti": "#C43031",
    "gqi": "#FAA51A",
    "noddi": "#38489E",
    "map": "#289144",
    "rdi": "#C43031",
}

# Matplotlib style: Georgia (see _configure_georgia_font), larger type
PLOT_FONT_SIZE = 30
PLOT_RC = {
    "font.family": ["Georgia", "DejaVu Serif", "serif"],
    "font.serif": ["Georgia", "DejaVu Serif", "Liberation Serif", "Nimbus Roman", "Times New Roman"],
    "font.size": PLOT_FONT_SIZE,
    "axes.labelsize": PLOT_FONT_SIZE,
    "axes.titlesize": PLOT_FONT_SIZE,
    "xtick.labelsize": PLOT_FONT_SIZE,
    "ytick.labelsize": PLOT_FONT_SIZE,
    "legend.fontsize": PLOT_FONT_SIZE,
}

# Per-quadrant PNGs (same type scale as combined 1×4)
PLOT_RC_TISSUE_PANELS = {
    **PLOT_RC,
}

# Columns that are not ROI measurements (controls CSVs may include e.g. group)
NON_ROI_COLUMNS = frozenset({"subject", "group"})


def normalize_subject_id(s: object) -> str:
    t = str(s).strip()
    if not t.startswith("sub-"):
        t = "sub-" + t
    return t


def load_laterality_map() -> Dict[str, str]:
    """sub -> 'left' or 'right' (temporal lobe rows only, matching mahalanobis)."""
    out: Dict[str, str] = {}
    if not INCLUSION_PATH.exists():
        return out
    try:
        df = pd.read_csv(INCLUSION_PATH)
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


def zscore_epilepsy_vs_controls(
    ctrl: pd.DataFrame, epi: pd.DataFrame, roi_cols: List[str]
) -> pd.DataFrame:
    """Per ROI column: z = (epi - mean_ctrl) / std_ctrl (ddof=1)."""
    ctrl_num = ctrl[roi_cols].apply(pd.to_numeric, errors="coerce")
    epi_num = epi[roi_cols].apply(pd.to_numeric, errors="coerce")
    data: Dict[str, np.ndarray] = {
        "subject": epi["subject"].map(normalize_subject_id).to_numpy(),
    }
    for c in roi_cols:
        v = ctrl_num[c].to_numpy(dtype=float)
        mu = np.nanmean(v)
        sd = np.nanstd(v, ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            data[c] = np.full(len(epi), np.nan, dtype=float)
        else:
            data[c] = (epi_num[c].to_numpy(dtype=float) - mu) / sd
    return pd.DataFrame(data, index=epi.index)


def build_roi_pairs(
    columns: Set[str],
    glasser_bases: Set[str],
    subcortical_base_to_labels: Dict[str, Tuple[str, str]],
    tract_base_to_type: Dict[str, str],
) -> Tuple[List[Dict[str, object]], Set[str]]:
    pairs: List[Dict[str, object]] = []
    used: Set[str] = set()

    for base in sorted(glasser_bases):
        lc, rc = f"Left_{base}", f"Right_{base}"
        if lc in columns and rc in columns:
            pairs.append({"quadrant": "glasser_cortex", "left_col": lc, "right_col": rc})
            used.add(lc)
            used.add(rc)

    for _base, (lh, rh) in sorted(subcortical_base_to_labels.items()):
        if lh in columns and rh in columns:
            pairs.append({"quadrant": "4s_subcortex", "left_col": lh, "right_col": rh})
            used.add(lh)
            used.add(rh)

    seen_wm: Set[Tuple[str, str, str]] = set()
    for col in sorted(columns):
        if "_L_" not in col:
            continue
        rc = col.replace("_L_", "_R_", 1)
        if rc not in columns:
            continue
        tract_hemi, segment = _wm_roi_to_tract_segment(col)
        if not tract_hemi.endswith("_L"):
            continue
        base_key = _wm_tract_base_key(tract_hemi)
        if base_key in EXCLUDED_VOLUMETRIC_ASYMMETRY_TRACT_BASES:
            continue
        ttype = tract_base_to_type.get(base_key)
        if ttype == "association":
            q = "wm_association"
        elif ttype == "projection":
            q = "wm_projection"
        else:
            continue
        dedupe_key = (base_key, segment, col)
        if dedupe_key in seen_wm:
            continue
        seen_wm.add(dedupe_key)
        pairs.append({"quadrant": q, "left_col": col, "right_col": rc})
        used.add(col)
        used.add(rc)

    return pairs, used


def _bar_series_for_quadrant(
    d_by_factor: Dict[int, List[Tuple[str, float]]],
    factor_indices: List[int],
    q: str,
) -> Tuple[np.ndarray, List[float], List[float], List[str]]:
    """Return x positions and bar(), yerr, color inputs."""
    means: List[float] = []
    sems: List[float] = []
    for fk in sorted(factor_indices):
        mean_d, sem, _n = aggregate_quadrant_factor(d_by_factor[fk], q)
        means.append(mean_d)
        sems.append(0.0 if not np.isfinite(sem) else sem)
    m_plot = [m if np.isfinite(m) else 0.0 for m in means]
    colors = [BAR_COLOR if np.isfinite(m) else BAR_COLOR_NA for m in means]
    return np.arange(len(factor_indices)), m_plot, sems, colors


def _omit_top_right_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _decorate_bar_axes(
    ax: plt.Axes,
    x: np.ndarray,
    x_labels: List[str],
    m_plot: List[float],
    sems: List[float],
    colors: List[str],
    *,
    title: str | None,
    ylabel: str | None = "Cohen's d",
    bar_edgewidth: float = 0.6,
    bar_width: float = 0.65,
) -> None:
    ax.bar(
        x,
        m_plot,
        bar_width,
        yerr=sems,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=bar_edgewidth,
        error_kw={"ecolor": "black", "linewidth": 1.2},
    )
    ax.axhline(0, color="k", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    pad = _xlim_side_pad(bar_width)
    ax.set_xlim(-pad, max(len(x) - 1, 0) + pad)
    _omit_top_right_spines(ax)


def _bar_width_for_n(n_bars: int) -> float:
    return min(_FZ_BAR_WIDTH_MAX, _FZ_BAR_WIDTH_PER_BAR / max(n_bars, 1))


def _xlim_side_pad(bar_width: float) -> float:
    """Margin so outer bars + error caps are not clipped at axis edges."""
    return max(_FZ_BAR_XLIM_SIDE_PAD, 0.5 * bar_width + 0.14)


def _panel_width_inches(n_bars: int) -> float:
    return max(_FZ_PANEL_WIDTH_MIN_IN, _FZ_PANEL_WIDTH_PER_BAR_IN * max(n_bars, 1))


def _side_by_side_gridspec_kw(*, wspace: float | None = None) -> Dict[str, object]:
    if wspace is None:
        wspace = _FZ_INNER_WSPACE
    return {"width_ratios": [1, SCALAR_PANEL_WIDTH_RATIO], "wspace": wspace}


def _side_by_side_figsize_inches(w_factor: float) -> float:
    return w_factor * (1 + SCALAR_PANEL_WIDTH_RATIO)


def build_ipsi_contra_by_tissue_table(
    d_by_factor: Dict[int, List[Tuple[str, float]]],
    factor_indices: List[int],
    cohens_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Long table of mean ipsi-contra Cohen's d by tissue class for factors and scalars.

    Matches bar heights in ``factor_score_z_ipsi_contra_cohens_d_*`` figures:
    factors = mean across ROI pairs in quadrant (SEM across pairs); scalars = mean
    signed d across ROIs in quadrant (SEM across ROIs), per scalar.
    """
    rows: List[dict[str, object]] = []
    for q in QUADRANT_ORDER:
        tissue_class = TISSUE_CLASS_LABELS.get(q, q)
        for fk in sorted(factor_indices):
            mean_d, sem, n_roi = aggregate_quadrant_factor(d_by_factor[fk], q)
            rows.append(
                {
                    "tissue_class": tissue_class,
                    "tissue_quadrant": q,
                    "measure_type": "factor",
                    "name": f"F{fk}",
                    "display_name": FACTOR_DISPLAY_LABELS.get(fk, f"F{fk}"),
                    "mean_cohens_d": mean_d,
                    "sem": sem,
                    "n_rois": n_roi,
                }
            )
        if cohens_df.empty:
            continue
        sub = filter_cohens_for_factor_quadrant(cohens_df, q)
        if sub.empty:
            continue
        scalars_ord = sort_scalars_in_quadrant_by_desc_abs_mean_signed_d(sub)
        for scalar in scalars_ord:
            v = (
                sub.loc[sub["scalar"] == scalar, "cohens_d"]
                .dropna()
                .astype(float)
                .to_numpy()
            )
            n = int(v.size)
            if n == 0:
                mean_d, sem = float("nan"), float("nan")
            elif n == 1:
                mean_d, sem = float(v[0]), float("nan")
            else:
                mean_d = float(np.mean(v))
                sem = float(np.std(v, ddof=1) / np.sqrt(n))
            rows.append(
                {
                    "tissue_class": tissue_class,
                    "tissue_quadrant": q,
                    "measure_type": "scalar",
                    "name": scalar,
                    "display_name": mrs._scalar_abbrev(scalar),
                    "mean_cohens_d": mean_d,
                    "sem": sem,
                    "n_rois": n,
                }
            )
    cols = [
        "tissue_class",
        "tissue_quadrant",
        "measure_type",
        "name",
        "display_name",
        "mean_cohens_d",
        "sem",
        "n_rois",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def aggregate_quadrant_factor(
    d_list: List[Tuple[str, float]],
    quadrant: str,
) -> Tuple[float, float, int]:
    vals = [d for q, d in d_list if q == quadrant and np.isfinite(d)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(vals[0]), float("nan"), 1
    mean_d = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(n))
    return mean_d, sem, n


def shared_ylim_for_tissue_panel_pngs(
    d_by_factor: Dict[int, List[Tuple[str, float]]],
    factor_indices: List[int],
) -> Tuple[float, float]:
    """y-axis limits spanning mean ± SEM for every bar across all tissue quadrants.

    Used so standalone ``factor_score_z_ipsi_contra_cohens_d_<quadrant>.png`` files
    are directly comparable.
    """
    lows: List[float] = []
    highs: List[float] = []
    for q in QUADRANT_ORDER:
        for fk in sorted(factor_indices):
            mean_d, sem, _n = aggregate_quadrant_factor(d_by_factor[fk], q)
            if not np.isfinite(mean_d):
                continue
            err = float(sem) if np.isfinite(sem) else 0.0
            lows.append(mean_d - err)
            highs.append(mean_d + err)
    if not lows:
        return (-0.5, 0.5)
    y0 = min(min(lows), 0.0)
    y1 = max(max(highs), 0.0)
    span = y1 - y0
    pad = 0.06 * span if span > 0 else 0.12
    return y0 - pad, y1 + pad


def _load_scalar_labels_and_colors() -> Tuple[Dict[str, str], Dict[str, str]]:
    labels: Dict[str, str] = {}
    if mrs.SCALAR_LABELS_PATH.exists():
        try:
            labels = json.loads(mrs.SCALAR_LABELS_PATH.read_text())
        except Exception:
            pass
    colors = mrs._load_scalar_colors()
    return labels, colors


def filter_cohens_for_factor_quadrant(
    cohens_df: pd.DataFrame, factor_q: str
) -> pd.DataFrame:
    """Restrict microstructural cohens_df to the same tissue bucket as factor_z quadrant."""
    if cohens_df.empty or "quadrant" not in cohens_df.columns:
        return pd.DataFrame()
    d = cohens_df.copy()
    if factor_q == "glasser_cortex":
        has_atlas = "atlas" in d.columns
        if has_atlas:
            sub = d[(d["quadrant"] == "cortex") & (d["atlas"].astype(str) == "glasser")]
        else:
            sub = d[d["quadrant"] == "cortex"]
    elif factor_q == "4s_subcortex":
        sub = d[d["quadrant"] == "subcortex"]
    elif factor_q == "wm_association":
        sub = d[d["quadrant"] == "association_wm"]
    elif factor_q == "wm_projection":
        sub = d[d["quadrant"] == "projection_wm"]
    else:
        sub = pd.DataFrame()
    if sub.empty:
        return sub
    sub = sub[~sub["scalar"].isin(mrs.EXCLUDED_SCALARS)].copy()
    sub["abs_cohens_d"] = np.abs(sub["cohens_d"])
    sub = sub.dropna(subset=["abs_cohens_d"])
    return sub


def sort_scalars_in_quadrant_by_desc_abs_mean_signed_d(sub: pd.DataFrame) -> List[str]:
    """
    Per tissue quadrant only: scalars that have rows in ``sub``, ordered by descending
    absolute value of the mean signed Cohen's d (same aggregation as the bar height).
    Ties: larger mean |d| across ROIs first, then scalar name.
    """
    if sub.empty or "scalar" not in sub.columns:
        return []
    present: List[str] = []
    seen: Set[str] = set()
    for s in sub["scalar"].dropna().unique():
        t = str(s)
        if t in seen:
            continue
        v = sub.loc[sub["scalar"] == t, "cohens_d"].dropna().astype(float)
        if len(v) == 0:
            continue
        seen.add(t)
        present.append(t)
    scored: List[Tuple[float, float, str]] = []
    for s in present:
        v = sub.loc[sub["scalar"] == s, "cohens_d"].dropna().astype(float).to_numpy()
        m_signed = float(np.mean(v))
        m_abs = float(np.mean(np.abs(v)))
        scored.append((-abs(m_signed), -m_abs, s))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[2] for t in scored]


def _collect_barplot_heights_and_error_extents(
    d_by_factor: Dict[int, List[Tuple[str, float]]],
    factor_indices: List[int],
    cohens_df: pd.DataFrame,
) -> Tuple[List[float], List[float], List[float]]:
    """Bar heights and mean ± SEM extents matching plot_*_bars_tissue_quadrant inputs."""
    heights: List[float] = []
    lows: List[float] = []
    highs: List[float] = []
    for q in QUADRANT_ORDER:
        for fk in sorted(factor_indices):
            mean_d, sem, _n = aggregate_quadrant_factor(d_by_factor[fk], q)
            h = float(mean_d) if np.isfinite(mean_d) else 0.0
            err = float(sem) if np.isfinite(sem) else 0.0
            heights.append(h)
            lows.append(h - err)
            highs.append(h + err)
        if cohens_df.empty:
            continue
        sub = filter_cohens_for_factor_quadrant(cohens_df, q)
        if sub.empty:
            continue
        scalars_ord = sort_scalars_in_quadrant_by_desc_abs_mean_signed_d(sub)
        means, sems = mrs._mean_sem_abs_per_scalar_values(
            sub, scalars_ord, value_col="cohens_d"
        )
        for i in range(len(scalars_ord)):
            h = float(means[i])
            err = float(sems[i]) if np.isfinite(sems[i]) else 0.0
            heights.append(h)
            lows.append(h - err)
            highs.append(h + err)
    return heights, lows, highs


def print_barplot_value_ranges(
    d_by_factor: Dict[int, List[Tuple[str, float]]],
    factor_indices: List[int],
    cohens_df: pd.DataFrame,
) -> None:
    """Print min/max of Cohen's d bar heights (and error-bar extents) to stdout."""
    heights, lows, highs = _collect_barplot_heights_and_error_extents(
        d_by_factor, factor_indices, cohens_df
    )
    if not heights:
        print("Bar plot values: no finite bar heights.")
        return
    print(
        "Bar plot bar heights (Cohen's d means): "
        f"min={min(heights):.6g} max={max(heights):.6g} (n={len(heights)})"
    )
    print(
        "Bar plot with error bars (mean ± SEM): "
        f"min={min(lows):.6g} max={max(highs):.6g}"
    )


def shared_ylim_combined(
    d_by_factor: Dict[int, List[Tuple[str, float]]],
    factor_indices: List[int],
    cohens_df: pd.DataFrame,
) -> Tuple[float, float]:
    """Global y limits from factor bars and signed microstructural bars (mean ± SEM) in every quadrant."""
    lows: List[float] = []
    highs: List[float] = []
    for q in QUADRANT_ORDER:
        for fk in sorted(factor_indices):
            mean_d, sem, _n = aggregate_quadrant_factor(d_by_factor[fk], q)
            if not np.isfinite(mean_d):
                continue
            err = float(sem) if np.isfinite(sem) else 0.0
            lows.append(mean_d - err)
            highs.append(mean_d + err)
        sub = filter_cohens_for_factor_quadrant(cohens_df, q)
        if sub.empty:
            continue
        scalars_ord = sort_scalars_in_quadrant_by_desc_abs_mean_signed_d(sub)
        means, sems = mrs._mean_sem_abs_per_scalar_values(
            sub, scalars_ord, value_col="cohens_d"
        )
        for i in range(len(scalars_ord)):
            m = float(means[i])
            s = float(sems[i])
            if not np.isfinite(m):
                continue
            lows.append(m - s)
            highs.append(m + s)
    if not lows:
        return (-0.5, 0.5)
    y0 = min(min(lows), 0.0)
    y1 = max(max(highs), 0.0)
    span = y1 - y0
    pad = 0.06 * span if span > 0 else 0.12
    return y0 - pad, y1 + pad


def plot_factor_bars_tissue_quadrant(
    ax: plt.Axes,
    factor_q: str,
    d_by_factor: Dict[int, List[Tuple[str, float]]],
    factor_indices: List[int],
    *,
    panel_title: Optional[str] = None,
    ylabel: Optional[str] = "Cohen's d",
    bar_edgewidth: float = 3.0,
) -> int:
    """Factor ipsi-contra Cohen's d bars for one tissue quadrant."""
    x_labels_f = [FACTOR_DISPLAY_LABELS[k] for k in sorted(factor_indices)]
    m_plot_f: List[float] = []
    sems_f: List[float] = []
    colors_f: List[str] = []
    for fk in sorted(factor_indices):
        mean_d, sem, _n = aggregate_quadrant_factor(d_by_factor[fk], factor_q)
        m_plot_f.append(float(mean_d) if np.isfinite(mean_d) else 0.0)
        sems_f.append(float(sem) if np.isfinite(sem) else 0.0)
        colors_f.append(BAR_COLOR if np.isfinite(mean_d) else BAR_COLOR_NA)
    n_f = len(factor_indices)
    x = np.arange(n_f)
    _decorate_bar_axes(
        ax,
        x,
        x_labels_f,
        m_plot_f,
        sems_f,
        colors_f,
        title=panel_title,
        ylabel=ylabel,
        bar_edgewidth=bar_edgewidth,
        bar_width=_bar_width_for_n(n_f),
    )
    return n_f


def plot_scalar_bars_tissue_quadrant(
    ax: plt.Axes,
    factor_q: str,
    cohens_df: pd.DataFrame,
    scalar_colors: Dict[str, str],
    *,
    panel_title: Optional[str] = None,
    bar_edgewidth: float = 3.0,
    scalar_sort: str = "abs_mean_signed",
) -> int:
    """Microstructural scalar ipsi-contra Cohen's d bars for one tissue quadrant."""
    sub = filter_cohens_for_factor_quadrant(cohens_df, factor_q)
    if sub.empty:
        ax.set_visible(False)
        return 0
    if scalar_sort == "reconstruction_model":
        scalars_ord = mrs.sort_scalars_in_quadrant_by_reconstruction_model(sub)
    else:
        scalars_ord = sort_scalars_in_quadrant_by_desc_abs_mean_signed_d(sub)
    means_s, sems_s = mrs._mean_sem_abs_per_scalar_values(
        sub, scalars_ord, value_col="cohens_d"
    )
    colors_s = [
        mrs._scalar_color(s, scalar_colors, MODEL_FALLBACK_COLORS) for s in scalars_ord
    ]
    tick_lab_s = [mrs._scalar_abbrev(s) for s in scalars_ord]
    m_plot = [float(means_s[j]) for j in range(len(scalars_ord))]
    sems = [float(sems_s[j]) for j in range(len(scalars_ord))]
    n_s = len(scalars_ord)
    x = np.arange(n_s)
    _decorate_bar_axes(
        ax,
        x,
        tick_lab_s,
        m_plot,
        sems,
        colors_s,
        title=panel_title,
        ylabel=None,
        bar_edgewidth=bar_edgewidth,
        bar_width=_bar_width_for_n(n_s),
    )
    return n_s


def plot_side_by_side_factor_scalar_quadrant(
    ax_factor: plt.Axes,
    ax_scalar: plt.Axes,
    factor_q: str,
    d_by_factor: Dict[int, List[Tuple[str, float]]],
    factor_indices: List[int],
    cohens_df: pd.DataFrame,
    scalar_colors: Dict[str, str],
    *,
    panel_title: Optional[str] = None,
    ylabel_left: Optional[str] = "Cohen's d",
    bar_edgewidth: float = 3.0,
    scalar_sort: str = "abs_mean_signed",
) -> Tuple[int, int]:
    """Factor bars (left) and scalar bars (right) with shared y limits; y-label on left only."""
    n_f = plot_factor_bars_tissue_quadrant(
        ax_factor,
        factor_q,
        d_by_factor,
        factor_indices,
        panel_title=panel_title,
        ylabel=ylabel_left,
        bar_edgewidth=bar_edgewidth,
    )
    n_s = plot_scalar_bars_tissue_quadrant(
        ax_scalar,
        factor_q,
        cohens_df,
        scalar_colors,
        panel_title=None,
        bar_edgewidth=bar_edgewidth,
        scalar_sort=scalar_sort,
    )
    if n_s == 0:
        ax_scalar.set_visible(False)
    else:
        ax_scalar.tick_params(labelleft=False)
    return n_f, n_s


def _hide_yticklabels_except_first(axes: List[plt.Axes]) -> None:
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)


def _write_combined_factor_scalar_bar_figures(
    d_by_factor: Dict[int, List[Tuple[str, float]]],
    factor_indices: List[int],
    cohens_df: pd.DataFrame,
    scalar_colors: Dict[str, str],
    output_dir: Path,
    dpi: int,
    ylim_tissue: Tuple[float, float],
    *,
    filename_suffix: str = "",
    scalar_sort: str = "abs_mean_signed",
    height_scale: float = 1.0,
) -> None:
    """Write 1×4 and per-quadrant side-by-side factor/scalar bar PNGs."""
    w_factor = _panel_width_inches(len(factor_indices))
    height_1x4 = _FZ_1X4_HEIGHT_IN * height_scale
    height_tissue = _FZ_TISSUE_PANEL_HEIGHT_IN * height_scale
    supt = (
        "Mean ipsilateral-contralateral Cohen's d: factors (control-z-scored factor scores per ROI, left) "
        "and microstructural scalars (raw z asymmetry, right), per tissue quadrant.\n"
        "Factors: mean across ROIs in quadrant (error bars = SEM across ROIs). "
        "Scalars: signed d, mean ± SEM across ROIs in quadrant."
    )
    with plt.rc_context(rc=PLOT_RC):
        fig = plt.figure(
            figsize=(4 * _side_by_side_figsize_inches(w_factor), height_1x4)
        )
        outer = fig.add_gridspec(1, 4, wspace=_FZ_OUTER_WSPACE)
        axes: List[plt.Axes] = []
        ax_share_y: Optional[plt.Axes] = None
        for i, q in enumerate(QUADRANT_ORDER):
            inner = outer[i].subgridspec(
                1,
                2,
                width_ratios=[1, SCALAR_PANEL_WIDTH_RATIO],
                wspace=_FZ_INNER_WSPACE,
            )
            if ax_share_y is None:
                ax_f = fig.add_subplot(inner[0])
                ax_share_y = ax_f
            else:
                ax_f = fig.add_subplot(inner[0], sharey=ax_share_y)
            ax_s = fig.add_subplot(inner[1], sharey=ax_share_y)
            axes.extend([ax_f, ax_s])
            plot_side_by_side_factor_scalar_quadrant(
                ax_f,
                ax_s,
                q,
                d_by_factor,
                factor_indices,
                cohens_df,
                scalar_colors,
                panel_title=QUADRANT_TITLES[q],
                ylabel_left="Cohen's d" if i == 0 else None,
                scalar_sort=scalar_sort,
            )
        for ax in axes:
            ax.set_ylim(ylim_tissue)
        _hide_yticklabels_except_first(axes)
        fig.suptitle(supt, fontsize=PLOT_FONT_SIZE)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.14, wspace=_FZ_OUTER_WSPACE)
        png_path = output_dir / f"factor_score_z_ipsi_contra_cohens_d_1x4{filename_suffix}.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print(f"Wrote {png_path}")

    with plt.rc_context(rc=PLOT_RC_TISSUE_PANELS):
        w_factor_tissue = _panel_width_inches(len(factor_indices))
        for q in QUADRANT_ORDER:
            fig1, (ax_f, ax_s) = plt.subplots(
                1,
                2,
                figsize=(
                    _side_by_side_figsize_inches(w_factor_tissue),
                    height_tissue,
                ),
                sharey=True,
                gridspec_kw=_side_by_side_gridspec_kw(),
            )
            plot_side_by_side_factor_scalar_quadrant(
                ax_f,
                ax_s,
                q,
                d_by_factor,
                factor_indices,
                cohens_df,
                scalar_colors,
                panel_title=None,
                ylabel_left="Cohen's d",
                scalar_sort=scalar_sort,
            )
            ax_f.set_ylim(ylim_tissue)
            ax_s.set_ylim(ylim_tissue)
            _hide_yticklabels_except_first([ax_f, ax_s])
            fig1.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.22, wspace=_FZ_INNER_WSPACE)
            single_path = output_dir / f"factor_score_z_ipsi_contra_cohens_d_{q}{filename_suffix}.png"
            fig1.savefig(single_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig1)
            print(f"Wrote {single_path}")


def run(factor_indices: List[int], output_dir: Path, dpi: int) -> None:
    glasser_bases = _get_glasser_bases()
    subcortical_base_to_labels = _get_4s_subcortical_base_to_labels()
    _meta, tract_base_to_type = _load_tract_metadata()
    lat_map = load_laterality_map()

    probe = FACTOR_SCORES_DIR / "epilepsy_F1_scores.csv"
    if not probe.exists():
        print(f"Missing {probe}", file=sys.stderr)
        sys.exit(1)
    all_cols = set(pd.read_csv(probe, nrows=0).columns) - NON_ROI_COLUMNS

    pairs, used = build_roi_pairs(all_cols, glasser_bases, subcortical_base_to_labels, tract_base_to_type)
    unclassified = sorted(all_cols - used)
    if unclassified:
        print(
            f"Columns not in any tissue quadrant (n={len(unclassified)}); "
            f"first few: {unclassified[:5]}"
        )

    d_by_factor: Dict[int, List[Tuple[str, float]]] = {k: [] for k in factor_indices}

    for fk in factor_indices:
        cpath = FACTOR_SCORES_DIR / f"controls_F{fk}_scores.csv"
        epath = FACTOR_SCORES_DIR / f"epilepsy_F{fk}_scores.csv"
        if not cpath.exists() or not epath.exists():
            print(f"Skip F{fk}: missing {cpath.name} or {epath.name}", file=sys.stderr)
            continue
        ctrl = pd.read_csv(cpath)
        epi = pd.read_csv(epath)
        roi_cols = sorted((set(ctrl.columns) & set(epi.columns)) - NON_ROI_COLUMNS)
        z_df = zscore_epilepsy_vs_controls(ctrl, epi, roi_cols)
        z_df = z_df.drop_duplicates(subset=["subject"], keep="first")

        subj_list = sorted(s for s in z_df["subject"].unique() if s in lat_map)
        if len(subj_list) < 2:
            print(f"F{fk}: fewer than 2 epilepsy subjects with laterality; skipping.", file=sys.stderr)
            continue

        rowby = z_df.set_index("subject")
        for p in pairs:
            q = str(p["quadrant"])
            lc = str(p["left_col"])
            rc = str(p["right_col"])
            if lc not in z_df.columns or rc not in z_df.columns:
                continue
            ipsi_all: List[float] = []
            contra_all: List[float] = []
            for s in subj_list:
                laterality = lat_map.get(s)
                if laterality not in ("left", "right"):
                    continue
                ipsi_col = lc if laterality == "left" else rc
                contra_col = rc if laterality == "left" else lc
                i = rowby.at[s, ipsi_col]
                c = rowby.at[s, contra_col]
                if np.isfinite(i) and np.isfinite(c):
                    ipsi_all.append(float(i))
                    contra_all.append(float(c))
            d = _cohens_d_paired(ipsi_all, contra_all)
            if np.isfinite(d):
                d_by_factor[fk].append((q, float(d)))

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in QUADRANT_ORDER:
        for fk in sorted(factor_indices):
            mean_d, sem, n_roi = aggregate_quadrant_factor(d_by_factor[fk], q)
            rows.append(
                {
                    "quadrant": q,
                    "factor": f"F{fk}",
                    "mean_cohens_d": mean_d,
                    "sem": sem,
                    "n_rois": n_roi,
                }
            )
    summary = pd.DataFrame(rows)
    csv_path = output_dir / "factor_score_z_ipsi_contra_cohens_d_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    _scalar_labels, scalar_colors = _load_scalar_labels_and_colors()
    cohens_df: pd.DataFrame = pd.DataFrame()
    try:
        tract_df = mrs.load_tract_asymmetry()
        subcortical_bases = mrs._get_4s_subcortical_bases()
        cortical_bases_4s = mrs._get_4s_cortical_bases()
        glasser_bases_mrs = mrs._get_glasser_bases()
        region_df = mrs.load_region_asymmetry(
            subcortical_bases, cortical_bases_4s, glasser_bases_mrs
        )
        cohens_df = mrs.compute_cohens_d_per_roi_scalar(tract_df, region_df)
        cohens_df = mrs._exclude_volumetric_asymmetry_tracts(cohens_df)
        _, tract_bt = mrs._load_tract_metadata()
        cohens_df = mrs.add_quadrant_column(cohens_df, tract_bt)
    except Exception as exc:
        print(
            f"Microstructural Cohen's d not loaded ({exc}); factor-only bar plots.",
            file=sys.stderr,
        )

    by_tissue = build_ipsi_contra_by_tissue_table(
        d_by_factor, factor_indices, cohens_df
    )
    by_tissue_path = output_dir / "factor_score_z_ipsi_contra_cohens_d_by_tissue.csv"
    by_tissue.to_csv(by_tissue_path, index=False)
    print(f"Wrote {by_tissue_path}")

    use_combined = not cohens_df.empty
    if not use_combined:
        print(
            "No tract/region asymmetry data: writing factor-only tissue bar figures.",
            file=sys.stderr,
        )
    ylim_tissue = BAR_PLOT_YLIM

    print_barplot_value_ranges(d_by_factor, factor_indices, cohens_df)

    x_labels = [FACTOR_DISPLAY_LABELS[k] for k in sorted(factor_indices)]

    if use_combined:
        _write_combined_factor_scalar_bar_figures(
            d_by_factor,
            factor_indices,
            cohens_df,
            scalar_colors,
            output_dir,
            dpi,
            ylim_tissue,
        )
        _write_combined_factor_scalar_bar_figures(
            d_by_factor,
            factor_indices,
            cohens_df,
            scalar_colors,
            output_dir,
            dpi,
            ylim_tissue,
            filename_suffix="_grouped-model",
            scalar_sort="reconstruction_model",
        )
        _write_combined_factor_scalar_bar_figures(
            d_by_factor,
            factor_indices,
            cohens_df,
            scalar_colors,
            output_dir,
            dpi,
            ylim_tissue,
            filename_suffix="_grouped-model_shorter",
            scalar_sort="reconstruction_model",
            height_scale=0.75,
        )
    else:
        w_factor_only = _panel_width_inches(len(factor_indices))
        bar_w = _bar_width_for_n(len(factor_indices))
        with plt.rc_context(rc=PLOT_RC):
            fig, axes = plt.subplots(
                1,
                4,
                figsize=(4 * w_factor_only, _FZ_1X4_HEIGHT_IN),
                sharey=True,
            )
            for ax, q in zip(np.ravel(axes), QUADRANT_ORDER):
                x, m_plot, sems, colors = _bar_series_for_quadrant(
                    d_by_factor, factor_indices, q
                )
                _decorate_bar_axes(
                    ax,
                    x,
                    x_labels,
                    m_plot,
                    sems,
                    colors,
                    title=QUADRANT_TITLES[q],
                    bar_width=bar_w,
                )
                ax.set_ylim(ylim_tissue)

            fig.suptitle(
                "Mean ipsilateral-contralateral Cohen's d (control-z-scored factor scores)\n"
                "per ROI, averaged across ROIs (error bars = SEM across ROIs)",
                fontsize=PLOT_FONT_SIZE,
            )
            fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.14, wspace=_FZ_OUTER_WSPACE)
            png_path = output_dir / "factor_score_z_ipsi_contra_cohens_d_1x4.png"
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {png_path}")

        with plt.rc_context(rc=PLOT_RC_TISSUE_PANELS):
            bar_w_tissue = _bar_width_for_n(len(factor_indices))
            for q in QUADRANT_ORDER:
                fig1, ax1 = plt.subplots(figsize=(w_factor_only, _FZ_TISSUE_PANEL_HEIGHT_IN))
                x, m_plot, sems, colors = _bar_series_for_quadrant(
                    d_by_factor, factor_indices, q
                )
                _decorate_bar_axes(
                    ax1,
                    x,
                    x_labels,
                    m_plot,
                    sems,
                    colors,
                    title=None,
                    bar_edgewidth=3.0,
                    bar_width=bar_w_tissue,
                )
                ax1.set_ylim(ylim_tissue)
                fig1.subplots_adjust(left=0.12, right=0.99, top=0.98, bottom=0.22)
                single_path = output_dir / f"factor_score_z_ipsi_contra_cohens_d_{q}.png"
                fig1.savefig(single_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig1)
                print(f"Wrote {single_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="Directory for PNG and summary CSV",
    )
    p.add_argument(
        "--factors",
        type=str,
        default=",".join(str(k) for k in DEFAULT_FACTOR_INDICES),
        help="Comma-separated factor indices (default: 1,2,3; F4 is deprecated and omitted)",
    )
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    factors = [int(x.strip()) for x in args.factors.split(",") if x.strip()]
    dropped = [f for f in factors if f in DEPRECATED_FACTOR_INDICES]
    if dropped:
        print(
            f"Ignoring deprecated factor(s): {', '.join(f'F{f}' for f in dropped)}",
            file=sys.stderr,
        )
    factors = [f for f in factors if f not in DEPRECATED_FACTOR_INDICES]
    if not factors:
        factors = list(DEFAULT_FACTOR_INDICES)
    run(factors, args.output_dir, args.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
