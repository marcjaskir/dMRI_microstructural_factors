"""Voxel MNI coordinate ranks and neuroaxis Pearson correlations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .embedding import gradient_from_row
from .types import VoxelGradientRunRow

NEUROAXIS_AXES: tuple[str, ...] = ("M-L", "A-P", "D-V")
NEUROAXIS_AXIS_LABELS: dict[str, str] = {
    "M-L": "Mesial-Lateral",
    "A-P": "Anterior-Posterior",
    "D-V": "Dorsal-Ventral",
}


def compute_neuroaxis_ranks(mni_xyz: np.ndarray) -> dict[str, np.ndarray]:
    """
    Rank in-mask voxel coordinates (same rules as group-controls ROI centroids).

    * A-P: rank by y descending (anterior = 1)
    * D-V: rank by z descending (dorsal = 1)
    * M-L: rank by |x| ascending (mesial = 1)
    """
    x, y, z = mni_xyz[:, 0], mni_xyz[:, 1], mni_xyz[:, 2]
    rank_ap = pd.Series(y).rank(method="min", ascending=False).to_numpy()
    rank_dv = pd.Series(z).rank(method="min", ascending=False).to_numpy()
    rank_ml = pd.Series(np.abs(x)).rank(method="min", ascending=True).to_numpy()
    return {"A-P": rank_ap, "D-V": rank_dv, "M-L": rank_ml}


def pearson_r_gradient_vs_coordinate_ranks(
    g_values: np.ndarray,
    ranks: dict[str, np.ndarray],
    *,
    axes: tuple[str, ...] = NEUROAXIS_AXES,
) -> pd.DataFrame:
    """Pearson r between gradient and coordinate rank arrays."""
    rows: list[dict[str, object]] = []
    for col in axes:
        xs = ranks[col].astype(np.float64)
        ys = g_values.astype(np.float64)
        finite = np.isfinite(xs) & np.isfinite(ys)
        n = int(finite.sum())
        base: dict[str, object] = {
            "neuroaxis_axis": col,
            "neuroaxis_label": NEUROAXIS_AXIS_LABELS.get(col, col),
            "n_voxels": n,
        }
        if n < 3:
            rows.append({**base, "pearson_r": np.nan})
            continue
        xa = xs[finite]
        ya = ys[finite]
        if xa.std(ddof=0) == 0 or ya.std(ddof=0) == 0:
            rows.append({**base, "pearson_r": np.nan})
            continue
        r = float(np.corrcoef(xa, ya)[0, 1])
        rows.append({**base, "pearson_r": r})
    return pd.DataFrame(rows)


def gradient_values_in_mask_order(
    row: VoxelGradientRunRow,
    gradient_index: int,
) -> np.ndarray:
    """Gradient vector aligned with mni_xyz rows."""
    g = gradient_from_row(row, gradient_index)
    flat = row[5].astype(str)
    return g.reindex(flat).to_numpy(dtype=np.float64)


def build_neuroaxis_correlations_table(
    results: list[VoxelGradientRunRow],
    *,
    cohort_tag: str = "controls",
    n_gradients: int = 2,
) -> pd.DataFrame:
    """Long table: factor × Gk × neuroaxis axis."""
    parts: list[pd.DataFrame] = []
    for row in results:
        factor_tag = row[0]
        ranks = compute_neuroaxis_ranks(row[4])
        n_avail = len(row[2])
        for gi in range(min(n_gradients, n_avail)):
            g_vals = gradient_values_in_mask_order(row, gi)
            sub = pearson_r_gradient_vs_coordinate_ranks(g_vals, ranks)
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
        "n_voxels",
    ]
    if not parts:
        return pd.DataFrame(columns=col_order)
    return pd.concat(parts, ignore_index=True)[col_order]


def save_neuroaxis_correlations_csv(
    results: list[VoxelGradientRunRow],
    out_path: Path,
    *,
    cohort_tag: str = "controls",
    n_gradients: int = 2,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_neuroaxis_correlations_table(
        results, cohort_tag=cohort_tag, n_gradients=n_gradients
    )
    df.to_csv(out_path, index=False)
    return out_path
