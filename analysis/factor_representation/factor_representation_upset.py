#!/usr/bin/env python3
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
Factor representation via UpSet plots.

Question: For each single multi-shell model (NODDI, MAP-MRI, DKI), are 3
well-selected statistics representative of the whole-brain factor-score
gradients (F1, F2, F3) in healthy controls, and how does this correspondence
differ across triplets of statistics?

Approach
--------
For every triplet of statistics within a model we measure how well the 3 stats
jointly represent the 3 factor gradients:

  1. For each (stat, factor) pair, compute the per-subject cosine similarity
     between the stat gradient and the factor gradient across all whole-brain
     ROIs (NaN-masked per subject), take |cosine|, then average across the
     control subjects -> an ``n_stats x 3`` similarity matrix.
  2. Each statistic is assigned to a single factor: the factor onto which it
     loads most strongly (argmax of |loading| in the All4_Combined factor
     loadings). A valid triplet then picks one statistic from each factor's
     candidate set (one per F1, F2, F3, all distinct). If a factor has no
     statistic whose max loading is that factor, its candidate set falls back
     to the single statistic with the highest |loading| on that factor.
  3. For each triplet, the score = mean of the 3 |cosine| values between each
     chosen statistic and its assigned factor.

Statistics are ordered (UpSet category rows, triplet labels) by their column
order in the All4_Combined factor loadings CSV.

Each model gets one UpSet plot (bar height = triplet score) plus CSV exports.
"""

import os
from io import BytesIO
from os.path import join as ospj
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.transforms import offset_copy
from PIL import Image

import upsetplot


def _patch_upsetplot_for_pandas3() -> None:
    """Work around upsetplot 0.9 incompatibilities with pandas 3 and matplotlib 3.10.

    - pandas 3 Copy-on-Write: in-place ``fillna`` on a DataFrame column is a no-op
      (UpSetPlot #303), leaving NaN matrix styles that crash matplotlib.
    - ``show_counts=False`` because upsetplot's count labels break on mpl 3.10.
    """
    UpSet = upsetplot.UpSet
    if getattr(UpSet, "_pandas3_cow_patch", False):
        return

    def plot_matrix(self, ax):
        ax = self._reorient(ax)
        data = self.intersections
        n_cats = data.index.nlevels

        inclusion = data.index.to_frame().values

        styles = [
            [
                self.subset_styles[i]
                if inclusion[i, j]
                else {"facecolor": self._other_dots_color, "linewidth": 0}
                for j in range(n_cats)
            ]
            for i in range(len(data))
        ]
        styles = sum(styles, [])
        style_columns = {
            "facecolor": "facecolors",
            "edgecolor": "edgecolors",
            "linewidth": "linewidths",
            "linestyle": "linestyles",
            "hatch": "hatch",
        }
        styles = (
            pd.DataFrame(styles)
            .reindex(columns=style_columns.keys())
            .astype(
                {
                    "facecolor": "O",
                    "edgecolor": "O",
                    "linewidth": float,
                    "linestyle": "O",
                    "hatch": "O",
                }
            )
        )
        styles["linewidth"] = styles["linewidth"].fillna(1)
        styles["facecolor"] = styles["facecolor"].fillna(self._facecolor)
        styles["edgecolor"] = styles["edgecolor"].fillna(styles["facecolor"])
        styles["linestyle"] = styles["linestyle"].fillna("solid")
        del styles["hatch"]

        x = np.repeat(np.arange(len(data)), n_cats)
        y = np.tile(np.arange(n_cats), len(data))

        if self._element_size is not None:
            s = (self._element_size * 0.35) ** 2
        else:
            s = 200
        ax.scatter(
            *self._swapaxes(x, y),
            s=s,
            zorder=10,
            **styles.rename(columns=style_columns),
        )

        if self._with_lines:
            idx = np.flatnonzero(inclusion)
            line_data = (
                pd.Series(y[idx], index=x[idx])
                .groupby(level=0)
                .aggregate(["min", "max"])
            )
            colors = pd.Series(
                [
                    style.get("edgecolor", style.get("facecolor", self._facecolor))
                    for style in self.subset_styles
                ],
                name="color",
            )
            line_data = line_data.join(colors)
            ax.vlines(
                line_data.index.values,
                line_data["min"],
                line_data["max"],
                lw=2,
                colors=line_data["color"],
                zorder=5,
            )

        tick_axis = ax.yaxis
        tick_axis.set_ticks(np.arange(n_cats))
        tick_labels = [
            "" if (name or "").startswith("_pad_row_") else name
            for name in data.index.names
        ]
        tick_axis.set_ticklabels(
            tick_labels, rotation=0 if self._horizontal else -90
        )
        ax.xaxis.set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)
        if not self._horizontal:
            ax.yaxis.set_ticks_position("top")
        ax.set_frame_on(False)
        ax.set_xlim(-0.5, x[-1] + 0.5, auto=False)
        ax.grid(False)

    UpSet.plot_matrix = plot_matrix
    UpSet._pandas3_cow_patch = True


_patch_upsetplot_for_pandas3()


GEORGIA_FONT_DIR = "/usr/share/fonts/truetype"
GEORGIA_FONT_FILES = (
    "georgia.ttf",
    "georgiab.ttf",
    "georgiai.ttf",
    "georgiaz.ttf",
)


def register_georgia_fonts() -> None:
    """Register system Georgia fonts so matplotlib can resolve the family name."""
    for filename in GEORGIA_FONT_FILES:
        path = ospj(GEORGIA_FONT_DIR, filename)
        if os.path.isfile(path):
            fm.fontManager.addfont(path)


register_georgia_fonts()

# Scale all plot text relative to matplotlib defaults.
PLOT_FONT_SCALE = 1.75
BASE_FONT_SIZE = 10

# Match the project plotting style used in factor_analysis.py
plt.rcParams.update(
    {
        "font.family": "Georgia",
        "font.size": BASE_FONT_SIZE * PLOT_FONT_SCALE,
        "axes.labelsize": BASE_FONT_SIZE * PLOT_FONT_SCALE,
        "xtick.labelsize": (BASE_FONT_SIZE - 1) * PLOT_FONT_SCALE,
        "ytick.labelsize": (BASE_FONT_SIZE - 1) * PLOT_FONT_SCALE,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Georgia",
        "mathtext.it": "Georgia:italic",
        "mathtext.bf": "Georgia:bold",
    }
)


def georgia_fp(size: float | None = None) -> fm.FontProperties:
    fp = fm.FontProperties(family="Georgia")
    fp.set_size(size if size is not None else plt.rcParams["font.size"])
    return fp


GEORGIA_FP = georgia_fp()

# ============================================================================
# CONFIGURATION
# ============================================================================

FZ_DIR = f"{PROJECT_ROOT}/derivatives/analysis/factor_z-scores"
FACTOR_DIR = f"{FZ_DIR}/factor_scores"
SCALAR_DIR = f"{FZ_DIR}/scalar_z-scores"

OUTPUT_DIR = f"{PROJECT_ROOT}/derivatives/analysis/factor_representation"

# Factor loadings used to (a) order statistics and (b) assign each statistic to
# the factor it loads onto most strongly.
LOADINGS_PATH = (
    f"{PROJECT_ROOT}/derivatives/analysis/factor_analysis/All4_Combined/"
    "controls_All4_Combined_scalar_factor_loadings_ordered.csv"
)

GROUP = "controls"
FACTORS: List[str] = ["F1", "F2", "F3"]
FACTOR_DISPLAY_LABELS: Dict[str, str] = {
    "F1": "Overall",
    "F2": "Non-Gaussian",
    "F3": "Anisotropic",
}
MULTIBAR_FACTOR_ALPHAS = [0.45, 0.7, 1.0]

# Single multi-shell models and their statistics (mirrors scalar_z-scores files).
MODELS: Dict[str, List[str]] = {
    "NODDI": ["icvf", "isovf", "od"],
    "MAP-MRI": ["ng", "ngpar", "ngperp", "pa", "path", "rtap", "rtop", "rtpp"],
    "DKI": ["ad", "ak", "fa", "kfa", "md", "mk", "mkt", "rd", "rk"],
}

# Filename prefix for each model's scalar files (controls_{prefix}_{stat}_z_scores.csv).
MODEL_FILE_PREFIX: Dict[str, str] = {
    "NODDI": "noddi",
    "MAP-MRI": "map",
    "DKI": "dki",
}

# Model colors from factor_analysis.py RECONSTRUCTION_MODEL_LEGEND.
MODEL_COLORS: Dict[str, str] = {
    "NODDI": "#38489E",
    "MAP-MRI": "#289144",
    "DKI": "#7A297F",
}

# Optional per-model statistic order overrides (UpSet category rows, top to bottom).
MODEL_STAT_ORDER_OVERRIDE: Dict[str, List[str]] = {
    "MAP-MRI": ["rtpp", "rtop", "rtap", "ng", "ngperp", "ngpar", "pa", "path"],
}

# Panel order for the combined 1x3 figure.
MODEL_PLOT_ORDER: List[str] = ["NODDI", "MAP-MRI", "DKI"]
SIMILARITY_SPECS: Dict[str, Dict[str, str]] = {
    "cosine": {
        "label": "Factor-matched\n|cosine|",
        "file": "similarity-cosine",
    },
    "pearson": {
        "label": "Matched factor score\n|Pearson's $r$|",
        "file": "similarity-pearson",
    },
}
FACTOR_BAR_YLABELS: Dict[str, str] = {
    "cosine": "Factor score\n|cosine|",
    "pearson": "Factor score\n|Pearson's $r$|",
}

# Stat pairs that must not appear in the same triplet.
EXCLUSIVE_STAT_PAIRS: List[frozenset] = [frozenset({"pa", "path"})]

# Non-ROI columns present in every CSV.
META_COLS = ["subject", "group"]

# ============================================================================
# DATA LOADING
# ============================================================================


def _load_matrix(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load a subject x ROI CSV, drop meta columns, return (array, roi_cols)."""
    df = pd.read_csv(path)
    roi_cols = [c for c in df.columns if c not in META_COLS]
    return df[roi_cols].to_numpy(dtype=float), roi_cols


def load_factor_gradients() -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Load control factor-score matrices keyed by factor id."""
    factors: Dict[str, np.ndarray] = {}
    ref_cols: List[str] = []
    for factor in FACTORS:
        path = ospj(FACTOR_DIR, f"{GROUP}_{factor}_scores.csv")
        arr, cols = _load_matrix(path)
        if not ref_cols:
            ref_cols = cols
        elif cols != ref_cols:
            raise ValueError(f"ROI columns for {factor} do not match reference")
        factors[factor] = arr
    return factors, ref_cols


def load_scalar_gradients(
    model: str, stats: List[str], ref_cols: List[str]
) -> Dict[str, np.ndarray]:
    """Load control scalar matrices for a model keyed by statistic name."""
    prefix = MODEL_FILE_PREFIX[model]
    scalars: Dict[str, np.ndarray] = {}
    for stat in stats:
        path = ospj(SCALAR_DIR, f"{GROUP}_{prefix}_{stat}_z_scores.csv")
        arr, cols = _load_matrix(path)
        if cols != ref_cols:
            raise ValueError(f"ROI columns for {model} {stat} do not match factors")
        scalars[stat] = arr
    return scalars


def load_loadings() -> pd.DataFrame:
    """Load the All4_Combined factor loadings (factors x scalar columns)."""
    df = pd.read_csv(LOADINGS_PATH).set_index("factor")
    return df.loc[FACTORS]


def ordered_model_stats(model: str, loadings: pd.DataFrame) -> List[str]:
    """Return the model's statistics ordered by their loadings-column order."""
    if model in MODEL_STAT_ORDER_OVERRIDE:
        return list(MODEL_STAT_ORDER_OVERRIDE[model])

    prefix = MODEL_FILE_PREFIX[model]
    declared = MODELS[model]
    ordered: List[str] = []
    for col in loadings.columns:
        if col.startswith(prefix + "_"):
            stat = col[len(prefix) + 1:]
            if stat in declared and stat not in ordered:
                ordered.append(stat)
    # Append any declared stats absent from the loadings file (preserve order).
    for stat in declared:
        if stat not in ordered:
            ordered.append(stat)
    return ordered


# Loading magnitude threshold for the fallback candidate rule (group-level).
FALLBACK_LOADING_THRESHOLD = 0.5


def candidate_sets(
    model: str, stats: List[str], loadings: pd.DataFrame
) -> Dict[str, List[str]]:
    """Map each factor to its candidate statistics.

    A statistic is a candidate for the factor onto which it loads most strongly
    (argmax of |loading|). If a factor ends up with no candidate, it falls back
    to all statistics whose |loading| on that factor exceeds
    ``FALLBACK_LOADING_THRESHOLD`` (group-level loadings); if none exceed it,
    the single statistic with the highest |loading| on that factor is used.
    """
    prefix = MODEL_FILE_PREFIX[model]
    col_for = {s: f"{prefix}_{s}" for s in stats}
    abs_load = loadings[[col_for[s] for s in stats]].abs()

    candidates: Dict[str, List[str]] = {f: [] for f in FACTORS}
    for stat in stats:
        primary = abs_load[col_for[stat]].idxmax()
        candidates[primary].append(stat)

    for factor in FACTORS:
        if not candidates[factor]:
            over_thresh = [
                s for s in stats
                if abs_load.loc[factor, col_for[s]] > FALLBACK_LOADING_THRESHOLD
            ]
            if over_thresh:
                candidates[factor] = over_thresh
            else:
                best_col = abs_load.loc[factor].idxmax()
                candidates[factor] = [next(s for s in stats if col_for[s] == best_col)]
    return candidates


# ============================================================================
# CORE COMPUTATION
# ============================================================================


def per_subject_abs_cosine(
    stat_mat: np.ndarray, factor_mat: np.ndarray
) -> np.ndarray:
    """Per-subject |cosine| between stat and factor gradients.

    Each row is one subject's gradient across ROIs. NaN entries are masked
    pairwise within each subject before computing the cosine. Returns a length
    n_subjects array (NaN where a subject has <2 valid ROIs or a zero vector).
    """
    out = np.full(stat_mat.shape[0], np.nan, dtype=float)
    for i, (s_row, f_row) in enumerate(zip(stat_mat, factor_mat)):
        mask = np.isfinite(s_row) & np.isfinite(f_row)
        if mask.sum() < 2:
            continue
        a = s_row[mask]
        b = f_row[mask]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            continue
        out[i] = abs(float(np.dot(a, b) / denom))
    return out


def per_subject_abs_pearson(
    stat_mat: np.ndarray, factor_mat: np.ndarray
) -> np.ndarray:
    """Per-subject |Pearson r| between stat and factor gradients."""
    out = np.full(stat_mat.shape[0], np.nan, dtype=float)
    for i, (s_row, f_row) in enumerate(zip(stat_mat, factor_mat)):
        mask = np.isfinite(s_row) & np.isfinite(f_row)
        if mask.sum() < 2:
            continue
        a = s_row[mask] - np.mean(s_row[mask])
        b = f_row[mask] - np.mean(f_row[mask])
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            continue
        out[i] = abs(float(np.dot(a, b) / denom))
    return out


SIMILARITY_FUNCS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "cosine": per_subject_abs_cosine,
    "pearson": per_subject_abs_pearson,
}


def build_per_subject(
    scalars: Dict[str, np.ndarray],
    factors: Dict[str, np.ndarray],
    stats: List[str],
    similarity: str,
) -> Dict[Tuple[str, str], np.ndarray]:
    """Map (stat, factor) -> per-subject absolute similarity array."""
    sim_func = SIMILARITY_FUNCS[similarity]
    per_subject: Dict[Tuple[str, str], np.ndarray] = {}
    for stat in stats:
        for factor in FACTORS:
            per_subject[(stat, factor)] = sim_func(scalars[stat], factors[factor])
    return per_subject


def similarity_matrix(
    per_subject: Dict[Tuple[str, str], np.ndarray], stats: List[str]
) -> pd.DataFrame:
    """n_stats x n_factors mean absolute similarity DataFrame."""
    data = np.empty((len(stats), len(FACTORS)), dtype=float)
    for i, stat in enumerate(stats):
        for j, factor in enumerate(FACTORS):
            data[i, j] = float(np.nanmean(per_subject[(stat, factor)]))
    return pd.DataFrame(data, index=stats, columns=FACTORS)


def similarity_sem_matrix(
    per_subject: Dict[Tuple[str, str], np.ndarray], stats: List[str]
) -> pd.DataFrame:
    """n_stats x n_factors across-subject SEM of the absolute similarity."""
    data = np.empty((len(stats), len(FACTORS)), dtype=float)
    for i, stat in enumerate(stats):
        for j, factor in enumerate(FACTORS):
            valid = per_subject[(stat, factor)]
            valid = valid[np.isfinite(valid)]
            n = valid.size
            data[i, j] = (
                float(valid.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            )
    return pd.DataFrame(data, index=stats, columns=FACTORS)


def score_triplets(
    per_subject: Dict[Tuple[str, str], np.ndarray],
    candidates: Dict[str, List[str]],
    stats: List[str],
    similarity: str,
) -> pd.DataFrame:
    """Score every valid triplet (one stat per factor from its candidate set).

    Each statistic can only match the factor it loads onto most strongly, so a
    triplet is a choice of one distinct statistic per factor. For each subject
    the triplet score is the mean of the 3 |cosine| values between each chosen
    statistic and its factor; the reported ``score`` is the across-subject mean
    and ``sem`` its standard error.
    """
    # A stat-set may be reachable via multiple assignments when candidate sets
    # overlap (e.g. a fallback stat that also belongs to another factor); keep
    # the best-scoring assignment per unique stat-set.
    best_rows: Dict[frozenset, dict] = {}
    for s1 in candidates["F1"]:
        for s2 in candidates["F2"]:
            for s3 in candidates["F3"]:
                if len({s1, s2, s3}) < 3:
                    continue
                triplet_stats_set = {s1, s2, s3}
                if any(pair.issubset(triplet_stats_set) for pair in EXCLUSIVE_STAT_PAIRS):
                    continue
                assignment = {"F1": s1, "F2": s2, "F3": s3}
                # n_subjects x 3 per-subject similarities, keep subjects valid in all 3.
                mat = np.column_stack(
                    [per_subject[(assignment[f], f)] for f in FACTORS]
                )
                valid = np.all(np.isfinite(mat), axis=1)
                sub = mat[valid]
                per_subj_score = sub.mean(axis=1)
                score = float(per_subj_score.mean())
                n = per_subj_score.size
                sem = (
                    float(per_subj_score.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                )
                key = frozenset((s1, s2, s3))
                if key in best_rows and best_rows[key]["score"] >= score:
                    continue
                factor_scores = {
                    f: float(sub[:, j].mean()) for j, f in enumerate(FACTORS)
                }
                factor_sems = {
                    f: (
                        float(sub[:, j].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                    )
                    for j, f in enumerate(FACTORS)
                }
                triplet_stats = [s1, s2, s3]
                row = {
                    "triplet": "+".join(sorted(triplet_stats, key=stats.index)),
                    "score": score,
                    "sem": sem,
                    "n_subjects": int(n),
                }
                for factor in FACTORS:
                    row[f"{factor}_stat"] = assignment[factor]
                    row[f"{factor}_{similarity}"] = factor_scores[factor]
                    row[f"{factor}_{similarity}_sem"] = factor_sems[factor]
                    row[f"{factor}_similarity"] = factor_scores[factor]
                    row[f"{factor}_similarity_sem"] = factor_sems[factor]
                # Boolean membership for each stat (used to build the UpSet index).
                for stat in stats:
                    row[stat] = stat in triplet_stats
                best_rows[key] = row
    df = (
        pd.DataFrame(list(best_rows.values()))
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return df


# ============================================================================
# PLOTTING
# ============================================================================

# Display overrides for statistic labels. Default is the uppercased abbreviation;
# parallel/perpendicular MAP-MRI metrics use the same mathtext pattern as
# factor_analysis.py so symbols render with Georgia.
_U2225_PARALLEL = "\u2225"
_U22A5_PERP = "\u22a5"
STAT_LABEL_OVERRIDES: Dict[str, str] = {
    "ngpar": r"$\mathdefault{NG\text{" + _U2225_PARALLEL + r"}}$",
    "ngperp": r"$\mathdefault{NG\text{" + _U22A5_PERP + r"}}$",
    "path": "PAth",
}


# Fixed grid geometry for the combined 1x3 figure (pad smaller models to these).
PANEL_ELEMENT_SIZE = 38
# Keep this fraction of upsetplot's default label-gutter width (0.5 = half as wide).
LABEL_GUTTER_WIDTH_FRACTION = 0.5
# Intersection bar band height as a fraction of the matrix (stat row) height.
BAR_BAND_MATRIX_FRACTION = 0.75
# Shared y-axis limits for all bar plots (extra bottom margin for tick labels).
BAR_YLIM = (0.45, 1.0)

# Grouped per-factor bar figure geometry (one subplot per factor).
FACTOR_BAR_WIDTH = 0.82
FACTOR_BAR_WITHIN_STEP = 1.2
FACTOR_BAR_GROUP_GAP = 0.45
FACTOR_BAR_INCHES_PER_SLOT = 0.58
FACTOR_BAR_FIG_HEIGHT = 5.0
FACTOR_BAR_FIG_MARGIN_IN = 1.25
FACTOR_BAR_STRIP_ALPHA = 0.4
FACTOR_STRIP_MARKER_SIZE = 5.0
FACTOR_STRIP_JITTER = 0.14


def bar_band_rows(max_stats: int) -> int:
    """Grid rows allocated to the intersection bar plot in combined panels."""
    return max(1, round(BAR_BAND_MATRIX_FRACTION * max_stats))


def uniform_fig_height_in(max_stats: int) -> float:
    """Figure height (inches) for a uniform combined panel."""
    return (PANEL_ELEMENT_SIZE / 72) * (max_stats + bar_band_rows(max_stats))


def fix_intersection_scales(
    int_ax: plt.Axes, matrix_ax: plt.Axes, n_bars: int, *, bar_width: float = 0.5
) -> None:
    """Lock shared x limits and a fixed bar width in data coordinates."""
    x_right = n_bars - 0.5
    int_ax.set_xlim(-0.5, x_right)
    matrix_ax.set_xlim(-0.5, x_right)
    for patch in int_ax.patches:
        patch.set_width(bar_width)


def remove_default_intersection_bars(int_ax: plt.Axes) -> None:
    """Remove upsetplot's summary bars before drawing factor-specific bars."""
    for patch in list(int_ax.patches):
        patch.remove()


def _redraw_row_shading(shading_ax: plt.Axes, n_rows: int, facecolor: str) -> None:
    """Replace upsetplot's partial-height shading bands with uniform full-row stripes."""
    for patch in list(shading_ax.patches):
        patch.remove()
    stripe = mcolors.to_rgba(facecolor, 0.05)
    for i in range(n_rows):
        if i % 2 == 0:
            shading_ax.axhspan(
                i - 0.5,
                i + 0.5,
                color=stripe,
                zorder=0,
                linewidth=0,
            )


def _lock_equal_matrix_rows(
    matrix_ax: plt.Axes, shading_ax: plt.Axes, n_rows: int
) -> None:
    """Force uniform row height in data and display coordinates."""
    ylim = (-0.5, n_rows - 0.5)
    for ax in (matrix_ax, shading_ax):
        ax.set_autoscale_on(False)
        ax.set_ylim(ylim)
        ax.margins(y=0)
        ax.tick_params(axis="y", which="both", length=0)


def apply_uniform_panel_geometry(
    fig: plt.Figure,
    axes: dict,
    *,
    max_stats: int,
    n_bars: int,
    model: str,
) -> None:
    """Uniform row heights, taller bar band, and aligned matrix/intersection x."""
    bar_rows = bar_band_rows(max_stats)
    matrix_rows = max_stats

    fig.set_size_inches(fig.get_figwidth(), uniform_fig_height_in(max_stats))

    matrix_ax = axes["matrix"]
    shading_ax = axes["shading"]
    int_ax = axes["intersections"]

    # upsetplot uses hspace=1 between the bar band and matrix, which skews row
    # proportions; re-pack axes to an exact bar:matrix = 3:4 split (75% of rows).
    fig.canvas.draw()
    int_pos = int_ax.get_position()
    mat_pos = matrix_ax.get_position()
    shade_pos = shading_ax.get_position()
    bottom = mat_pos.y0
    top = int_pos.y1
    total_h = top - bottom
    bar_frac = bar_rows / (bar_rows + matrix_rows)
    bar_h = total_h * bar_frac
    mat_h = total_h - bar_h
    int_ax.set_position([int_pos.x0, bottom + mat_h, int_pos.width, bar_h])
    matrix_ax.set_position([mat_pos.x0, bottom, mat_pos.width, mat_h])
    shading_ax.set_position([shade_pos.x0, bottom, shade_pos.width, mat_h])

    # Narrow the statistic-label gutter (upsetplot over-allocates text columns).
    mat_pos = matrix_ax.get_position()
    shade_pos = shading_ax.get_position()
    int_pos = int_ax.get_position()
    label_gutter = mat_pos.x0 - shade_pos.x0
    gutter_trim = label_gutter * (1.0 - LABEL_GUTTER_WIDTH_FRACTION)
    if gutter_trim > 0:
        matrix_ax.set_position(
            [mat_pos.x0 - gutter_trim, bottom, mat_pos.width, mat_h]
        )
        shading_ax.set_position(
            [shade_pos.x0, bottom, shade_pos.width - gutter_trim, mat_h]
        )
        int_ax.set_position(
            [int_pos.x0 - gutter_trim, bottom + mat_h, int_pos.width, bar_h]
        )

    _redraw_row_shading(shading_ax, matrix_rows, MODEL_COLORS[model])
    _lock_equal_matrix_rows(matrix_ax, shading_ax, matrix_rows)
    fix_intersection_scales(int_ax, matrix_ax, n_bars)


def _sync_matrix_shading_positions(
    matrix_ax: plt.Axes, shading_ax: plt.Axes
) -> None:
    """Keep matrix and shading vertically identical after tick labels resize margins."""
    mat_pos = matrix_ax.get_position()
    shade_pos = shading_ax.get_position()
    shading_ax.set_position(
        [shade_pos.x0, mat_pos.y0, shade_pos.width, mat_pos.height]
    )


def set_matrix_row_labels(matrix_ax: plt.Axes, stats: List[str], model: str) -> None:
    """Set visible matrix row labels, hiding padded placeholder rows."""
    labels = ["" if s.startswith("_pad_row_") else stat_label(s) for s in stats]
    # upsetplot sort_categories_by="-input" reverses rows (first stat at top).
    labels = labels[::-1]
    matrix_ax.set_yticks(np.arange(len(stats)))
    matrix_ax.set_yticklabels(labels)
    matrix_ax.tick_params(axis="y", pad=1)
    model_color = MODEL_COLORS[model]
    for label in matrix_ax.get_yticklabels():
        if label.get_text().strip():
            label.set_horizontalalignment("right")
            label.set_color(model_color)


def scrub_pad_text_artists(fig: plt.Figure) -> None:
    """Remove any stray _pad_row_* text artists upsetplot may place on the figure."""
    for text in fig.findobj(mtext.Text):
        if "_pad_row_" in text.get_text():
            text.set_visible(False)


def stat_label(stat: str) -> str:
    """Return the display label for a statistic abbreviation."""
    if stat.startswith("_pad_row_"):
        return stat  # unique MultiIndex level name; hidden on the plot axis
    return STAT_LABEL_OVERRIDES.get(stat, stat.upper())


def uniform_panel_dims(
    panels: Dict[str, Tuple[pd.DataFrame, List[str]]],
) -> Tuple[int, int]:
    """Max stat rows and max triplet columns across models."""
    max_stats = max(len(stats) for _, stats in panels.values())
    max_triplets = max(len(df) for df, _ in panels.values())
    return max_stats, max_triplets


def pad_for_uniform_layout(
    triplet_df: pd.DataFrame,
    stats: List[str],
    max_stats: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """Pad stat rows with empty placeholders so every panel shares the same grid height."""
    stats_out = list(stats)
    for i in range(max_stats - len(stats)):
        stats_out.append(f"_pad_row_{i}")

    df = triplet_df.copy()
    for stat in stats_out:
        if stat not in df.columns:
            df[stat] = False

    return df, stats_out


def apply_georgia_font(fig: plt.Figure) -> None:
    """Apply Georgia to all text artists in a figure (including upsetplot labels)."""
    for text in fig.findobj(mtext.Text):
        if not text.get_text().strip():
            continue
        size = text.get_fontsize()
        if not size:
            size = plt.rcParams["font.size"]
        text.set_fontproperties(georgia_fp(size))
    for ax in fig.axes:
        ax.xaxis.label.set_fontproperties(georgia_fp(ax.xaxis.label.get_fontsize()))
        ax.yaxis.label.set_fontproperties(georgia_fp(ax.yaxis.label.get_fontsize()))
        ax.title.set_fontproperties(georgia_fp(ax.title.get_fontsize()))
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            size = label.get_fontsize()
            if not size:
                size = plt.rcParams["ytick.labelsize"]
            label.set_fontproperties(georgia_fp(size))


def panel_figsize(n_triplets: int, n_stats: int) -> Tuple[float, float]:
    """Figure size for one UpSet panel, scaled to triplet count and stat rows."""
    width = max(3.5, 0.22 * n_triplets + 1.8)
    height = max(4.0, 0.32 * n_stats + 1.5)
    return (width, height)


def build_upset_series(
    triplet_df: pd.DataFrame, stats: List[str]
) -> pd.Series:
    """Build the upsetplot Series for a model's scored triplets."""
    display_names = [stat_label(s) for s in stats]
    index = pd.MultiIndex.from_frame(triplet_df[stats]).set_names(display_names)
    return pd.Series(triplet_df["score"].to_numpy(), index=index, name="score")


def render_upset_figure(
    model: str,
    triplet_df: pd.DataFrame,
    stats: List[str],
    *,
    similarity: str,
    multibar: bool = False,
    show_ylabel: bool = True,
    panel_title: str | None = None,
    n_real_triplets: int | None = None,
    max_stats: int | None = None,
    uniform_layout: bool = False,
) -> Tuple[plt.Figure, dict]:
    """Draw one UpSet panel and return its figure (caller must close or save)."""
    series = build_upset_series(triplet_df, stats)
    if uniform_layout:
        fig = plt.figure(figsize=(1, 1))  # upsetplot overwrites size from grid geometry
    else:
        fig = plt.figure(figsize=panel_figsize(len(triplet_df), len(stats)))

    upset = upsetplot.UpSet(
        series,
        sort_by="input",
        sort_categories_by="-input",
        facecolor=MODEL_COLORS[model],
        show_counts=False,  # upsetplot count labels break on mpl 3.10
        element_size=PANEL_ELEMENT_SIZE,
        intersection_plot_elements=(
            bar_band_rows(max_stats) if uniform_layout and max_stats else 6
        ),
        totals_plot_elements=0,
        with_lines=False,
    )
    axes = upset.plot(fig=fig)
    int_ax = axes["intersections"]
    matrix_ax = axes["matrix"]
    int_ax.set_ylabel(SIMILARITY_SPECS[similarity]["label"] if show_ylabel else "")

    n_bars = n_real_triplets if n_real_triplets is not None else len(triplet_df)
    scores = triplet_df["score"].to_numpy()[:n_bars]
    sems = triplet_df["sem"].to_numpy()[:n_bars]
    x = np.arange(n_bars)
    if multibar:
        remove_default_intersection_bars(int_ax)
        factor_values = triplet_df[[f"{f}_similarity" for f in FACTORS]].to_numpy()[
            :n_bars
        ]
        factor_sems = triplet_df[
            [f"{f}_similarity_sem" for f in FACTORS]
        ].to_numpy()[:n_bars]
    else:
        int_ax.errorbar(
            x,
            scores,
            yerr=sems,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=2.5,
            zorder=5,
        )
    int_ax.set_ylim(*BAR_YLIM)

    if uniform_layout and max_stats is not None:
        apply_uniform_panel_geometry(
            fig,
            axes,
            max_stats=max_stats,
            n_bars=n_bars,
            model=model,
        )

    if uniform_layout:
        scrub_pad_text_artists(fig)
        shading_ax = axes["shading"]
        _sync_matrix_shading_positions(matrix_ax, shading_ax)
        _redraw_row_shading(shading_ax, len(stats), MODEL_COLORS[model])
        _lock_equal_matrix_rows(matrix_ax, shading_ax, len(stats))

    set_matrix_row_labels(matrix_ax, stats, model)

    if multibar:
        bar_width = 0.5 / len(FACTORS)
        offsets = (np.arange(len(FACTORS)) - (len(FACTORS) - 1) / 2) * bar_width
        for j, factor in enumerate(FACTORS):
            int_ax.bar(
                x + offsets[j],
                factor_values[:, j],
                width=bar_width,
                color=MODEL_COLORS[model],
                alpha=MULTIBAR_FACTOR_ALPHAS[j],
                align="center",
                zorder=4,
                label=factor,
            )
            int_ax.errorbar(
                x + offsets[j],
                factor_values[:, j],
                yerr=factor_sems[:, j],
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=1.8,
                zorder=6,
            )

    if panel_title:
        fig.suptitle(panel_title, fontproperties=GEORGIA_FP, y=0.98)

    apply_georgia_font(fig)
    return fig, axes


def figure_to_image(
    fig: plt.Figure,
    dpi: int = 300,
    *,
    tight: bool = True,
    pad_inches: float = 0.02,
) -> Image.Image:
    """Rasterize a matplotlib figure to a PIL image."""
    buf = BytesIO()
    save_kw: dict = {"format": "png", "dpi": dpi, "facecolor": "white"}
    if tight:
        save_kw["bbox_inches"] = "tight"
        save_kw["pad_inches"] = pad_inches
    fig.savefig(buf, **save_kw)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def trim_image(
    im: Image.Image, padding: int = 6, *, keep_left: int | None = None
) -> Image.Image:
    """Crop excess white border around panel content."""
    arr = np.asarray(im)
    mask = np.any(arr < 252, axis=2)
    if not mask.any():
        return im
    ys, xs = np.where(mask)
    left = keep_left if keep_left is not None else max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)
    right = min(im.width, int(xs.max()) + 1 + padding)
    bottom = min(im.height, int(ys.max()) + 1 + padding)
    return im.crop((left, top, right, bottom))


def trim_image_right_only(im: Image.Image, padding: int = 4) -> Image.Image:
    """Trim trailing horizontal whitespace without changing row pixel height."""
    arr = np.asarray(im)
    mask = np.any(arr < 252, axis=2)
    if not mask.any():
        return im
    xs = np.where(mask)[1]
    right = min(im.width, int(xs.max()) + 1 + padding)
    return im.crop((0, 0, right, im.height))


def stitch_images_horizontally(
    images: List[Image.Image], *, gap: int = 16
) -> Image.Image:
    """Concatenate panels left-to-right; pad height without scaling."""
    target_h = max(im.height for im in images)
    total_w = sum(im.width for im in images) + gap * (len(images) - 1)
    combined = Image.new("RGB", (total_w, target_h), "white")
    x = 0
    for i, im in enumerate(images):
        if im.height < target_h:
            padded = Image.new("RGB", (im.width, target_h), "white")
            padded.paste(im, (0, 0))
            im = padded
        combined.paste(im, (x, 0))
        x += im.width + (gap if i < len(images) - 1 else 0)
    return combined


def similarity_file_label(similarity: str) -> str:
    """Filename-safe label for a similarity output."""
    return SIMILARITY_SPECS[similarity]["file"]


def plot_upset(
    model: str, triplet_df: pd.DataFrame, stats: List[str], similarity: str
) -> None:
    """Render and save the UpSet plot for a single model."""
    fig, _ = render_upset_figure(
        model, triplet_df, stats, similarity=similarity, show_ylabel=True
    )
    out_png = ospj(
        OUTPUT_DIR,
        f"upset_{MODEL_FILE_PREFIX[model]}_{similarity_file_label(similarity)}.png",
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    if similarity == "cosine":
        legacy_out_png = ospj(OUTPUT_DIR, f"upset_{MODEL_FILE_PREFIX[model]}.png")
        fig.savefig(legacy_out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out_png}")


def plot_combined_upset(
    panels: Dict[str, Tuple[pd.DataFrame, List[str]]],
    *,
    similarity: str,
    multibar: bool = False,
) -> None:
    """Render NODDI, MAP-MRI, and DKI UpSet panels in one aligned 1x3 figure."""
    max_stats, _ = uniform_panel_dims(panels)
    images: List[Image.Image] = []

    for i, model in enumerate(MODEL_PLOT_ORDER):
        triplet_df, stats = panels[model]
        triplet_pad, stats_pad = pad_for_uniform_layout(
            triplet_df, stats, max_stats
        )
        fig, _ = render_upset_figure(
            model,
            triplet_pad,
            stats_pad,
            similarity=similarity,
            multibar=multibar,
            show_ylabel=(i == 0),
            n_real_triplets=len(triplet_df),
            max_stats=max_stats,
            uniform_layout=True,
        )
        # Fixed height, natural width (one element_size column per bar).
        im = figure_to_image(fig, tight=True, pad_inches=0.02)
        images.append(trim_image(im, padding=4))

    combined = stitch_images_horizontally(images, gap=16)
    suffix = "_multibar" if multibar else ""
    out_png = ospj(
        OUTPUT_DIR, f"upset_{similarity_file_label(similarity)}{suffix}.png"
    )
    combined.save(out_png, dpi=(300, 300))
    print(f"  saved {out_png}")
    if similarity == "cosine" and not multibar:
        legacy_out_png = ospj(OUTPUT_DIR, "upset_simarity-cosine.png")
        combined.save(legacy_out_png, dpi=(300, 300))


def plot_multibar_legend() -> None:
    """Save a standalone legend for multibar UpSet figures.

    Rows are models; columns are factors. Each cell uses the model color at the
    factor alpha used in the multibar plots.
    """
    cell_in = 0.30
    gap_in = 0.05
    label_gap_in = 0.06
    row_label_w_in = 0.72
    top_margin_in = 0.85
    right_margin_in = 0.08
    bottom_margin_in = 0.10

    n_models = len(MODEL_PLOT_ORDER)
    n_factors = len(FACTORS)
    grid_w_in = n_factors * cell_in + (n_factors - 1) * gap_in
    grid_h_in = n_models * cell_in + (n_models - 1) * gap_in

    fig_w_in = row_label_w_in + label_gap_in + grid_w_in + right_margin_in
    fig_h_in = bottom_margin_in + grid_h_in + label_gap_in + top_margin_in
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def fx(inches: float) -> float:
        return inches / fig_w_in

    def fy(inches: float) -> float:
        return inches / fig_h_in

    grid_x0_in = row_label_w_in + label_gap_in
    grid_y0_in = bottom_margin_in
    grid_y1_in = grid_y0_in + grid_h_in
    cell_w = fx(cell_in)
    cell_h = fy(cell_in)

    for i, model in enumerate(MODEL_PLOT_ORDER):
        y_in = grid_y1_in - (i + 1) * cell_in - i * gap_in
        ax.text(
            fx(row_label_w_in),
            fy(y_in + cell_in / 2),
            model,
            ha="right",
            va="center",
            fontproperties=GEORGIA_FP,
            transform=ax.transAxes,
        )
        for j, factor in enumerate(FACTORS):
            x_in = grid_x0_in + j * (cell_in + gap_in)
            ax.add_patch(
                mpatches.Rectangle(
                    (fx(x_in), fy(y_in)),
                    cell_w,
                    cell_h,
                    transform=ax.transAxes,
                    facecolor=MODEL_COLORS[model],
                    alpha=MULTIBAR_FACTOR_ALPHAS[j],
                    edgecolor="none",
                )
            )

    for j, factor in enumerate(FACTORS):
        x_in = grid_x0_in + j * (cell_in + gap_in)
        ax.text(
            fx(x_in),
            fy(grid_y1_in + label_gap_in),
            FACTOR_DISPLAY_LABELS[factor],
            ha="left",
            va="bottom",
            rotation=45,
            rotation_mode="anchor",
            fontproperties=GEORGIA_FP,
            transform=offset_copy(
                ax.transAxes,
                fig=fig,
                x=cell_in / 2,
                y=0,
                units="inches",
            ),
        )

    apply_georgia_font(fig)
    out_png = ospj(OUTPUT_DIR, "upset_multibar_legend.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    print(f"  saved {out_png}")


def _build_factor_bar_layout(
    panels: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]],
    factor: str,
) -> Tuple[List[dict], float]:
    """Lay out one factor subplot: one contiguous group of bar positions per model.

    Within each model group the candidate statistics for ``factor`` are ordered by
    descending mean absolute similarity with the factor score. Returns the group
    descriptors and the total x extent (position just past the final bar).
    """
    groups: List[dict] = []
    pos = 0.0
    for model in MODEL_PLOT_ORDER:
        mean_df, _sem_df, candidates = panels[model]
        stats_sorted = sorted(
            candidates[factor],
            key=lambda s: float(mean_df.loc[s, factor]),
            reverse=True,
        )
        positions: List[float] = []
        for _ in stats_sorted:
            positions.append(pos)
            pos += FACTOR_BAR_WITHIN_STEP
        groups.append(
            {"model": model, "stats": stats_sorted, "positions": positions}
        )
        pos += FACTOR_BAR_GROUP_GAP
    extent = pos - FACTOR_BAR_GROUP_GAP
    return groups, extent


def plot_factor_bars(
    panels: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]],
    *,
    similarity: str,
    stripplot: bool = False,
    per_subject_panels: Dict[str, Dict[Tuple[str, str], np.ndarray]] | None = None,
) -> None:
    """Render one grouped-bar subplot per factor.

    Each subplot corresponds to a microstructural factor and contains three bar
    groups (NODDI, MAP-MRI, DKI). Each group holds that model's statistics that
    load most strongly onto the factor (its candidate set), sorted within-model
    by descending mean absolute similarity with the factor score.

    When ``stripplot`` is True, bars are semi-transparent and subject-level
    points are overlaid with small horizontal jitter; mean ± SEM markers are
    drawn on top of the points so they remain visible.
    """
    if stripplot and per_subject_panels is None:
        raise ValueError("per_subject_panels is required when stripplot=True")

    layouts = {f: _build_factor_bar_layout(panels, f) for f in FACTORS}
    extents = [layouts[f][1] for f in FACTORS]
    rng = np.random.default_rng(42)

    ymax = 0.0
    for factor in FACTORS:
        groups, _ = layouts[factor]
        for g in groups:
            mean_df, sem_df, _ = panels[g["model"]]
            per_subject = (
                per_subject_panels[g["model"]] if stripplot else None
            )
            for stat in g["stats"]:
                ymax = max(
                    ymax,
                    float(mean_df.loc[stat, factor])
                    + float(sem_df.loc[stat, factor]),
                )
                if stripplot and per_subject is not None:
                    values = per_subject[(stat, factor)]
                    valid = values[np.isfinite(values)]
                    if valid.size:
                        ymax = max(ymax, float(valid.max()))
    ymax = ymax * 1.12 if ymax > 0 else 1.0

    fig_w = sum(extents) * FACTOR_BAR_INCHES_PER_SLOT + FACTOR_BAR_FIG_MARGIN_IN
    fig, axes = plt.subplots(
        1,
        len(FACTORS),
        figsize=(fig_w, FACTOR_BAR_FIG_HEIGHT),
        sharey=True,
        gridspec_kw={"width_ratios": extents},
    )

    for k, factor in enumerate(FACTORS):
        ax = axes[k]
        groups, extent = layouts[factor]
        xticks: List[float] = []
        xticklabels: List[str] = []
        tick_colors: List[str] = []
        for g in groups:
            model = g["model"]
            mean_df, sem_df, _ = panels[model]
            color = MODEL_COLORS[model]
            per_subject = (
                per_subject_panels[model] if stripplot else None
            )
            for stat, xpos in zip(g["stats"], g["positions"]):
                val = float(mean_df.loc[stat, factor])
                err = float(sem_df.loc[stat, factor])
                ax.bar(
                    xpos,
                    val,
                    width=FACTOR_BAR_WIDTH,
                    color=color,
                    alpha=FACTOR_BAR_STRIP_ALPHA if stripplot else 1.0,
                    align="center",
                    zorder=3,
                )
                if stripplot and per_subject is not None:
                    values = per_subject[(stat, factor)]
                    valid_mask = np.isfinite(values)
                    if valid_mask.any():
                        jitter = rng.uniform(
                            -FACTOR_STRIP_JITTER,
                            FACTOR_STRIP_JITTER,
                            size=int(valid_mask.sum()),
                        )
                        ax.scatter(
                            xpos + jitter,
                            values[valid_mask],
                            s=FACTOR_STRIP_MARKER_SIZE,
                            color=color,
                            alpha=0.75,
                            linewidths=0,
                            zorder=5,
                        )
                    # Draw SEM after scatter with plain line artists so zorder is
                    # reliable (errorbar LineCollections can sit under PathCollections).
                    cap = 0.12
                    for ecolor, lw, zo in (("white", 3.2, 9), ("black", 1.2, 10)):
                        ax.vlines(
                            xpos,
                            val - err,
                            val + err,
                            colors=ecolor,
                            linewidth=lw,
                            zorder=zo,
                            clip_on=False,
                        )
                        ax.hlines(
                            [val - err, val + err],
                            xpos - cap,
                            xpos + cap,
                            colors=ecolor,
                            linewidth=lw,
                            zorder=zo,
                            clip_on=False,
                        )
                else:
                    ax.errorbar(
                        xpos,
                        val,
                        yerr=err,
                        fmt="none",
                        ecolor="black",
                        elinewidth=0.8,
                        capsize=1.8,
                        zorder=6,
                    )
                xticks.append(xpos)
                xticklabels.append(stat_label(stat))
                tick_colors.append(color)
            if g["positions"] and k == 0:
                center = sum(g["positions"]) / len(g["positions"])
                ax.text(
                    center,
                    -0.30,
                    model,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    color=color,
                    fontproperties=GEORGIA_FP,
                )
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            xticklabels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        for label, color in zip(ax.get_xticklabels(), tick_colors):
            label.set_color(color)
        ax.set_xlim(-0.5, extent - 0.35)
        ax.set_ylim(0, ymax)
        ax.set_title(FACTOR_DISPLAY_LABELS[factor])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color="0.85", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        if k == 0:
            ax.set_ylabel(FACTOR_BAR_YLABELS[similarity])

    fig.subplots_adjust(bottom=0.28, wspace=0.04)
    apply_georgia_font(fig)
    suffix = "_strip" if stripplot else ""
    out_png = ospj(
        OUTPUT_DIR,
        f"factorbars_{similarity_file_label(similarity)}{suffix}.png",
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    print(f"  saved {out_png}")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    factors, ref_cols = load_factor_gradients()
    print(f"Loaded factor gradients: {len(factors)} factors x {len(ref_cols)} ROIs")

    loadings = load_loadings()
    panels_by_similarity: Dict[str, Dict[str, Tuple[pd.DataFrame, List[str]]]] = {
        similarity: {} for similarity in SIMILARITY_SPECS
    }
    factor_bar_panels: Dict[
        str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]]
    ] = {similarity: {} for similarity in SIMILARITY_SPECS}
    factor_bar_per_subject: Dict[
        str, Dict[str, Dict[Tuple[str, str], np.ndarray]]
    ] = {similarity: {} for similarity in SIMILARITY_SPECS}

    for model in MODELS:
        print(f"\n=== {model} ===")
        stats = ordered_model_stats(model, loadings)
        scalars = load_scalar_gradients(model, stats, ref_cols)
        print(f"  {len(stats)} statistics: {', '.join(stats)}")

        candidates = candidate_sets(model, stats, loadings)
        for factor in FACTORS:
            print(f"  {factor} candidates: {', '.join(candidates[factor])}")

        for similarity in SIMILARITY_SPECS:
            print(f"  -- {similarity} --")
            per_subject = build_per_subject(scalars, factors, stats, similarity)
            sim = similarity_matrix(per_subject, stats)
            sem = similarity_sem_matrix(per_subject, stats)
            factor_bar_panels[similarity][model] = (sim, sem, candidates)
            factor_bar_per_subject[similarity][model] = per_subject
            sim_path = ospj(
                OUTPUT_DIR,
                (
                    f"similarity_matrix_{MODEL_FILE_PREFIX[model]}_"
                    f"{similarity_file_label(similarity)}.csv"
                ),
            )
            sim.to_csv(sim_path)
            print(f"  saved {sim_path}")
            if similarity == "cosine":
                legacy_sim_path = ospj(
                    OUTPUT_DIR, f"similarity_matrix_{MODEL_FILE_PREFIX[model]}.csv"
                )
                sim.to_csv(legacy_sim_path)

            triplet_df = score_triplets(per_subject, candidates, stats, similarity)
            print(f"  {len(triplet_df)} valid triplets")
            export_cols = (
                ["triplet", "score", "sem", "n_subjects"]
                + [f"{f}_stat" for f in FACTORS]
                + [f"{f}_{similarity}" for f in FACTORS]
                + [f"{f}_{similarity}_sem" for f in FACTORS]
            )
            triplet_path = ospj(
                OUTPUT_DIR,
                (
                    f"triplet_scores_{MODEL_FILE_PREFIX[model]}_"
                    f"{similarity_file_label(similarity)}.csv"
                ),
            )
            triplet_df[export_cols].to_csv(triplet_path, index=False)
            print(f"  saved {triplet_path}")
            if similarity == "cosine":
                legacy_triplet_path = ospj(
                    OUTPUT_DIR, f"triplet_scores_{MODEL_FILE_PREFIX[model]}.csv"
                )
                triplet_df[export_cols].to_csv(legacy_triplet_path, index=False)

            best = triplet_df.iloc[0]
            print(
                f"  best triplet: {best['triplet']} (score={best['score']:.3f})"
            )

            plot_upset(model, triplet_df, stats, similarity)
            panels_by_similarity[similarity][model] = (triplet_df, stats)

    for similarity, panels in panels_by_similarity.items():
        plot_combined_upset(panels, similarity=similarity, multibar=False)
        plot_combined_upset(panels, similarity=similarity, multibar=True)

    for similarity in SIMILARITY_SPECS:
        plot_factor_bars(factor_bar_panels[similarity], similarity=similarity)
        plot_factor_bars(
            factor_bar_panels[similarity],
            similarity=similarity,
            stripplot=True,
            per_subject_panels=factor_bar_per_subject[similarity],
        )

    plot_multibar_legend()


if __name__ == "__main__":
    main()
