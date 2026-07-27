"""Pearson r between principal gradients and whole-brain neuroaxis ranks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .embedding import gradient_from_row
from .groupings import load_neuroaxis_ranks
from .types import GradientRunRow

NEUROAXIS_AXES: tuple[str, ...] = ("M-L", "A-P", "D-V")
NEUROAXIS_AXIS_LABELS: dict[str, str] = {
    "M-L": "Mesial-Lateral",
    "A-P": "Anterior-Posterior",
    "D-V": "Dorsal-Ventral",
}

# Matches ``atlas`` values in ``wholebrain_centroids.csv``.
ATLAS_GLASSER = "glasser_cortex"
ATLAS_SUBCORTEX = "four_s156_subcortex"
ATLAS_WM = "hcp1065_tract_third"


def collect_gradient_roi_labels(results: list[GradientRunRow]) -> set[str]:
    """Union of ROI names across all gradient series in ``results``."""
    labels: set[str] = set()
    for row in results:
        for g in row[2]:
            labels.update(str(x) for x in g.index)
    return labels


def load_roi_atlas_class_map(tractometry_root: Path) -> dict[str, str]:
    """Map ROI label -> atlas class from ``wholebrain_centroids.csv``."""
    p = tractometry_root / "derivatives/atlas_centroids/wholebrain_centroids.csv"
    df = pd.read_csv(p)
    return dict(zip(df["label"].astype(str), df["atlas"].astype(str)))


def _tissue_counts_for_pairs(
    rois_used: list[str],
    atlas_by_roi: dict[str, str],
) -> dict[str, int]:
    c = Counter(atlas_by_roi.get(r, "unknown") for r in rois_used)
    return {
        "n_glasser_cortex": int(c.get(ATLAS_GLASSER, 0)),
        "n_four_s156_subcortex": int(c.get(ATLAS_SUBCORTEX, 0)),
        "n_hcp1065_wm_tract_third": int(c.get(ATLAS_WM, 0)),
    }


def pearson_r_gradient_vs_neuroaxis(
    g: pd.Series,
    neuroaxis_by_roi: dict[str, dict[str, float]],
    *,
    axes: tuple[str, ...] = NEUROAXIS_AXES,
    atlas_by_roi: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Pearson r between gradient scores and each neuroaxis rank column."""
    if atlas_by_roi is None:
        atlas_by_roi = {}
    rows: list[dict[str, object]] = []
    for col in axes:
        xs: list[float] = []
        ys: list[float] = []
        rois_used: list[str] = []
        for roi, gv in g.items():
            rec = neuroaxis_by_roi.get(str(roi))
            if rec is None or col not in rec:
                continue
            v = rec[col]
            if not np.isfinite(v) or not np.isfinite(gv):
                continue
            xs.append(float(v))
            ys.append(float(gv))
            rois_used.append(str(roi))
        tissue = _tissue_counts_for_pairs(rois_used, atlas_by_roi)
        base: dict[str, object] = {
            "neuroaxis_axis": col,
            "neuroaxis_label": NEUROAXIS_AXIS_LABELS.get(col, col),
            "n_rois": len(xs),
            **tissue,
        }
        if len(xs) < 3:
            rows.append({**base, "pearson_r": np.nan})
            continue
        xa = np.asarray(xs, dtype=np.float64)
        ya = np.asarray(ys, dtype=np.float64)
        if xa.std(ddof=0) == 0 or ya.std(ddof=0) == 0:
            rows.append({**base, "pearson_r": np.nan})
            continue
        r = float(np.corrcoef(xa, ya)[0, 1])
        rows.append({**base, "pearson_r": r})
    return pd.DataFrame(rows)


def build_neuroaxis_correlations_table(
    results: list[GradientRunRow],
    *,
    tractometry_root: Path,
    cohort_tag: str = "controls",
    n_gradients: int = 3,
) -> pd.DataFrame:
    """Long table: one row per (factor, principal gradient, neuroaxis axis)."""
    roi_labels = collect_gradient_roi_labels(results)
    neuroaxis_by_roi = load_neuroaxis_ranks(
        tractometry_root, roi_labels=roi_labels
    )
    atlas_by_roi = load_roi_atlas_class_map(tractometry_root)
    parts: list[pd.DataFrame] = []
    n_gradients = max(1, int(n_gradients))
    for row in results:
        factor_tag = row[0]
        n_avail = len(row[2])
        for gi in range(min(n_gradients, n_avail)):
            g = gradient_from_row(row, gi)
            if g.empty:
                continue
            sub = pearson_r_gradient_vs_neuroaxis(
                g, neuroaxis_by_roi, atlas_by_roi=atlas_by_roi
            )
            sub.insert(0, "factor", factor_tag)
            sub.insert(1, "principal_gradient", f"G{gi + 1}")
            sub.insert(2, "cohort", cohort_tag)
            parts.append(sub)
    col_order = [
        "factor",
        "principal_gradient",
        "cohort",
        "neuroaxis_axis",
        "neuroaxis_label",
        "pearson_r",
        "n_rois",
        "n_glasser_cortex",
        "n_four_s156_subcortex",
        "n_hcp1065_wm_tract_third",
    ]
    if not parts:
        return pd.DataFrame(columns=col_order)
    return pd.concat(parts, ignore_index=True)[col_order]


def save_neuroaxis_correlations_csv(
    results: list[GradientRunRow],
    out_path: Path,
    *,
    tractometry_root: Path,
    cohort_tag: str = "controls",
    n_gradients: int = 3,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_neuroaxis_correlations_table(
        results,
        tractometry_root=tractometry_root,
        cohort_tag=cohort_tag,
        n_gradients=n_gradients,
    )
    df.to_csv(out_path, index=False)
    return out_path
