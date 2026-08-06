#!/usr/bin/env python3
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
Factor representation of whole-brain microstructural factors by model statistics.

For each single multi-shell model (NODDI, MAP-MRI, DKI), measure how well
selected statistics represent control factor-score gradients (F1–F3):

  1. Per-subject |cosine| / |Pearson r| between each statistic gradient and
     each factor gradient across ROIs.
  2. Assign each statistic to the factor with strongest |loading|; score
     valid one-per-factor triplets.
  3. Export similarity / triplet CSVs and factor-bar figures.
"""

import os
from os.path import join as ospj
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

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

FZ_DIR = f"{analysis_dir()}/factor_z-scores"
FACTOR_DIR = f"{FZ_DIR}/factor_scores"
SCALAR_DIR = f"{FZ_DIR}/scalar_z-scores"

OUTPUT_DIR = f"{analysis_dir()}/factor_representation"

# Factor loadings used to (a) order statistics and (b) assign each statistic to
# the factor it loads onto most strongly.
LOADINGS_PATH = (
    f"{analysis_dir()}/factor_analysis/"
    "controls_All4_Combined_scalar_factor_loadings_ordered.csv"
)

GROUP = "controls"
FACTORS: List[str] = ["F1", "F2", "F3"]
from lib.factor_labels import FACTOR_SHORT_LABELS as FACTOR_DISPLAY_LABELS  # noqa: E402

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

# Optional per-model statistic order overrides.
MODEL_STAT_ORDER_OVERRIDE: Dict[str, List[str]] = {
    "MAP-MRI": ["rtpp", "rtop", "rtap", "ng", "ngperp", "ngpar", "pa", "path"],
}

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
META_COLS = ["subject", "group", "anon_id"]

# Open / manuscript subject-level export (all models, factor-matched |loading| > 0.5).
FACTOR_MATCHED_SUBJECT_CSV = "factor_matched_subject_similarity.csv"

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
    """Load the combined GM+WM factor loadings (factors × scalar columns)."""
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


def factor_matched_stats(
    loadings: pd.DataFrame,
    *,
    min_abs_loading: float = FALLBACK_LOADING_THRESHOLD,
) -> List[dict]:
    """Return manuscript figure stats: factor-matched with |loading| > threshold.

    Each entry has keys ``model``, ``statistic``, ``matched_factor``,
    ``abs_loading``, ``loading_col``.
    """
    rows: List[dict] = []
    for model in MODEL_PLOT_ORDER:
        prefix = MODEL_FILE_PREFIX[model]
        stats = ordered_model_stats(model, loadings)
        for stat in stats:
            col = f"{prefix}_{stat}"
            if col not in loadings.columns:
                continue
            abs_series = loadings[col].abs()
            matched = str(abs_series.idxmax())
            abs_loading = float(abs_series.loc[matched])
            if abs_loading <= min_abs_loading:
                continue
            rows.append(
                {
                    "model": model,
                    "statistic": stat,
                    "matched_factor": matched,
                    "abs_loading": abs_loading,
                    "loading_col": col,
                }
            )
    return rows


def _subject_id_series(df: pd.DataFrame) -> pd.Series:
    for col in ("anon_id", "subject", "sub"):
        if col in df.columns:
            return df[col].astype(str)
    raise KeyError("Expected anon_id/subject/sub column in factor-score CSV")


def build_factor_matched_subject_similarity(
    *,
    loadings: pd.DataFrame | None = None,
    factors: Dict[str, np.ndarray] | None = None,
    subject_ids: List[str] | None = None,
    ref_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Per-subject factor-matched |cosine| / |Pearson r| for all manuscript models.

    Long-form table with one row per (subject, model, statistic).
    """
    if loadings is None:
        loadings = load_loadings()
    if factors is None or ref_cols is None or subject_ids is None:
        factors_loaded, ref_cols_loaded = load_factor_gradients()
        factors = factors if factors is not None else factors_loaded
        ref_cols = ref_cols if ref_cols is not None else ref_cols_loaded
        if subject_ids is None:
            sample = pd.read_csv(ospj(FACTOR_DIR, f"{GROUP}_{FACTORS[0]}_scores.csv"))
            subject_ids = _subject_id_series(sample).tolist()

    assert factors is not None and ref_cols is not None and subject_ids is not None
    matched = factor_matched_stats(loadings)
    rows: List[dict] = []
    # Cache scalar matrices per model to avoid reloading.
    scalars_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    id_key = (
        "anon_id"
        if any(str(s).startswith("anon_") for s in subject_ids)
        else "subject"
    )
    for entry in matched:
        model = entry["model"]
        if model not in scalars_by_model:
            stats = ordered_model_stats(model, loadings)
            scalars_by_model[model] = load_scalar_gradients(model, stats, ref_cols)
        stat = entry["statistic"]
        factor = entry["matched_factor"]
        cos = per_subject_abs_cosine(
            scalars_by_model[model][stat], factors[factor]
        )
        pear = per_subject_abs_pearson(
            scalars_by_model[model][stat], factors[factor]
        )
        for i, sid in enumerate(subject_ids):
            rows.append(
                {
                    id_key: sid,
                    "model": model,
                    "statistic": stat,
                    "matched_factor": factor,
                    "abs_loading": entry["abs_loading"],
                    "abs_cosine": float(cos[i]) if np.isfinite(cos[i]) else np.nan,
                    "abs_pearson": float(pear[i]) if np.isfinite(pear[i]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_factor_matched_subject_similarity(
    df: pd.DataFrame | None = None,
    *,
    out_path: str | None = None,
) -> str:
    """Write the consolidated subject-level factor-representation CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if df is None:
        df = build_factor_matched_subject_similarity()
    path = out_path or ospj(OUTPUT_DIR, FACTOR_MATCHED_SUBJECT_CSV)
    df.to_csv(path, index=False)
    print(f"saved {path} ({len(df)} rows)")
    return path

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
# Intersection bar band height as a fraction of the matrix (stat row) height.
# Shared y-axis limits for all bar plots (extra bottom margin for tick labels).

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

def stat_label(stat: str) -> str:
    """Return the display label for a statistic abbreviation."""
    if stat.startswith("_pad_row_"):
        return stat  # unique MultiIndex level name; hidden on the plot axis
    return STAT_LABEL_OVERRIDES.get(stat, stat.upper())

def apply_georgia_font(fig: plt.Figure) -> None:
    """Apply Georgia to all text artists in a figure."""
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

def similarity_file_label(similarity: str) -> str:
    """Filename-safe label for a similarity output."""
    return SIMILARITY_SPECS[similarity]["file"]

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
            sim.to_csv(sim_path, index_label="statistic")
            print(f"  saved {sim_path}")
            if similarity == "cosine":
                legacy_sim_path = ospj(
                    OUTPUT_DIR, f"similarity_matrix_{MODEL_FILE_PREFIX[model]}.csv"
                )
                sim.to_csv(legacy_sim_path, index_label="statistic")

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

    for similarity in SIMILARITY_SPECS:
        plot_factor_bars(factor_bar_panels[similarity], similarity=similarity)
        plot_factor_bars(
            factor_bar_panels[similarity],
            similarity=similarity,
            stripplot=True,
            per_subject_panels=factor_bar_per_subject[similarity],
        )

    # Consolidated subject-level table for open / manuscript products.
    sample = pd.read_csv(ospj(FACTOR_DIR, f"{GROUP}_{FACTORS[0]}_scores.csv"))
    subject_ids = _subject_id_series(sample).tolist()
    save_factor_matched_subject_similarity(
        build_factor_matched_subject_similarity(
            loadings=loadings,
            factors=factors,
            subject_ids=subject_ids,
            ref_cols=ref_cols,
        )
    )


if __name__ == "__main__":
    main()
