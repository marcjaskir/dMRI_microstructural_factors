"""Per-subject scalar z-scoring and factor score computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_OUTLIER_IQR_MULTIPLIER, MIN_FINITE_VOXELS_FOR_IQR


def mask_subject_scalar_iqr_outliers(
    subject_matrix: np.ndarray,
    multiplier: float,
    min_finite_voxels: int = MIN_FINITE_VOXELS_FOR_IQR,
) -> np.ndarray:
    """Set per-subject, per-scalar IQR outliers to NaN."""
    out = np.array(subject_matrix, dtype=np.float32, copy=True)
    finite = np.isfinite(out)
    finite_counts = finite.sum(axis=0)
    if int(finite_counts.max()) < min_finite_voxels:
        return out

    with np.errstate(invalid="ignore"):
        q1, q3 = np.nanpercentile(out, [25, 75], axis=0)
    iqr = q3 - q1
    active = (iqr > 0) & (finite_counts >= min_finite_voxels)
    low = q1 - multiplier * iqr
    high = q3 + multiplier * iqr
    outlier = finite & active & ((out < low) | (out > high))
    out[outlier] = np.nan
    return out


def zscore_scalars_per_subject(subject_matrix: np.ndarray) -> np.ndarray:
    """Z-score each scalar column across in-mask voxels for one subject."""
    out = np.array(subject_matrix, dtype=np.float32, copy=True)
    n_voxels, n_scalars = out.shape
    for j in range(n_scalars):
        col = out[:, j]
        finite = np.isfinite(col)
        if finite.sum() < 2:
            out[:, j] = 0.0
            continue
        vals = col[finite]
        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        if std <= 0 or not np.isfinite(std):
            out[:, j] = 0.0
            continue
        z = (col - mean) / std
        z[~finite] = 0.0
        out[:, j] = z.astype(np.float32, copy=False)
    return out


def compute_factor_scores_from_z(
    z_matrix: np.ndarray,
    loadings_df: pd.DataFrame,
    scalar_labels: list[str],
    factors: list[str],
) -> dict[str, np.ndarray]:
    """Dot product of z-scored scalars with factor loadings per voxel."""
    scores: dict[str, np.ndarray] = {}
    for factor in factors:
        weights = loadings_df.loc[factor, scalar_labels].to_numpy(dtype=np.float64)
        scores[factor] = (z_matrix @ weights).astype(np.float32, copy=False)
    return scores


def apply_iqr_if_requested(
    subject_matrix: np.ndarray,
    *,
    use_iqr: bool,
    multiplier: float = DEFAULT_OUTLIER_IQR_MULTIPLIER,
) -> np.ndarray:
    if not use_iqr:
        return subject_matrix
    return mask_subject_scalar_iqr_outliers(subject_matrix, multiplier)
