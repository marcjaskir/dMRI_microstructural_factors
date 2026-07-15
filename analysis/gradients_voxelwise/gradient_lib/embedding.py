"""Gradient extraction helpers for voxelwise gradient rows."""

from __future__ import annotations

import pandas as pd

from .types import VoxelGradientRunRow


def gradient_from_row(row: VoxelGradientRunRow, j: int) -> pd.Series:
    """Return the ``j``-th gradient (0-indexed) from a row."""
    grads = row[2]
    if j < len(grads):
        return grads[j]
    if not grads:
        return pd.Series(dtype=np.float64)
    return pd.Series(0.0, index=grads[0].index)
