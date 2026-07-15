"""Pearson r between epilepsy group-mean factor z-scores and gradients / neuroaxis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_TRACTOMETRY_ROOT
from .gc_imports import gc_groupings
from .io import (
    aggregate_epilepsy_z_scores,
    load_controls_gradients,
)

# Fixed left-to-right order on lollipop x-axis.
COMPARISON_SPECS: tuple[tuple[str, str, str], ...] = (
    ("Gradient 1", "gradient", "G1"),
    ("Gradient 2", "gradient", "G2"),
    ("Mesial-Lateral", "neuroaxis", "M-L"),
    ("Anterior-Posterior", "neuroaxis", "A-P"),
    ("Dorsal-Ventral", "neuroaxis", "D-V"),
)


def pearson_r_pair(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    """Pearson r between two ROI-indexed series (intersection, finite pairs only)."""
    idx = a.index.intersection(b.index)
    xa = pd.to_numeric(a.reindex(idx), errors="coerce").to_numpy(dtype=np.float64)
    xb = pd.to_numeric(b.reindex(idx), errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(xb)
    xa = xa[mask]
    xb = xb[mask]
    n = int(mask.sum())
    if n < 3 or xa.std() == 0.0 or xb.std() == 0.0:
        return np.nan, n
    return float(np.corrcoef(xa, xb)[0, 1]), n


def pearson_r_vs_neuroaxis(
    z: pd.Series,
    neuroaxis_by_roi: dict[str, dict[str, float]],
    axis_key: str,
) -> tuple[float, int]:
    """Pearson r between z-scores and one neuroaxis rank column."""
    xs: list[float] = []
    ys: list[float] = []
    for roi, zv in z.items():
        rec = neuroaxis_by_roi.get(str(roi))
        if rec is None or axis_key not in rec:
            continue
        rank = rec[axis_key]
        if not np.isfinite(rank) or not np.isfinite(zv):
            continue
        xs.append(float(rank))
        ys.append(float(zv))
    n = len(xs)
    if n < 3:
        return np.nan, n
    xa = np.asarray(xs, dtype=np.float64)
    xb = np.asarray(ys, dtype=np.float64)
    if xa.std() == 0.0 or xb.std() == 0.0:
        return np.nan, n
    return float(np.corrcoef(xa, xb)[0, 1]), n


def build_factor_z_correlation_table(
    *,
    gradients_csv_dir: Path,
    factor_z_paths: list[tuple[str, Path]],
    tractometry_root: Path = DEFAULT_TRACTOMETRY_ROOT,
) -> pd.DataFrame:
    """One row per (factor, comparison variable) with Pearson r and n ROIs."""
    groupings = gc_groupings()
    neuroaxis_by_roi = groupings.load_neuroaxis_ranks(tractometry_root)

    rows: list[dict[str, object]] = []
    for factor_tag, z_path in factor_z_paths:
        z = aggregate_epilepsy_z_scores(z_path, absolute=False)
        g1, g2 = load_controls_gradients(gradients_csv_dir, factor_tag)
        for label, source, key in COMPARISON_SPECS:
            if source == "gradient":
                predictor = g1 if key == "G1" else g2
                r, n = pearson_r_pair(z, predictor)
            else:
                r, n = pearson_r_vs_neuroaxis(z, neuroaxis_by_roi, key)
            rows.append(
                {
                    "factor": factor_tag,
                    "comparison": label,
                    "source": source,
                    "key": key,
                    "pearson_r": r,
                    "n_rois": n,
                }
            )
    return pd.DataFrame(rows)


def symmetric_ylim(values: pd.Series, *, pad_frac: float = 0.08) -> tuple[float, float]:
    """Shared symmetric y-limits for Pearson r lollipop panels."""
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return (-0.5, 0.5)
    abs_max = float(finite.abs().max())
    if abs_max == 0.0:
        return (-0.1, 0.1)
    pad = pad_frac * abs_max
    return (-abs_max - pad, abs_max + pad)
