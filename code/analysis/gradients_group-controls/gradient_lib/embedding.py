"""Gradient extraction helpers for controls-only gradient rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import GradientRunRow


def gradient_from_row(row: GradientRunRow, j: int) -> pd.Series:
    """Return the ``j``-th gradient (0-indexed) from a row.

    Pads with a zero Series indexed to G1's index if the row has fewer than ``j+1`` modes.
    """
    grads = row[2]
    if j < len(grads):
        return grads[j]
    if not grads:
        return pd.Series(dtype=np.float64)
    return pd.Series(0.0, index=grads[0].index)


def gradient_k_intersection_arrays(
    row: GradientRunRow,
    k: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Finite ``k`` gradient values on the shared ROI set + region names.

    Returns ``([g1, ..., gK], regions)`` where each ``gj`` is a 1d float array over the
    same ROI ordering. ROIs are kept only when **all** of g1..gK are finite.
    """
    k = max(1, int(k))
    series = [gradient_from_row(row, j) for j in range(k)]
    if not series:
        return [], np.array([], dtype=object)
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)
    arrs = [s.reindex(idx).to_numpy(dtype=np.float64) for s in series]
    if not arrs:
        return arrs, np.array([], dtype=object)
    finite = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        finite &= np.isfinite(a)
    arrs_f = [a[finite] for a in arrs]
    regions = np.asarray(idx[finite].astype(str), dtype=object)
    return arrs_f, regions
