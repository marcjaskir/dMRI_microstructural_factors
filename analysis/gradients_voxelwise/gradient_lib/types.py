"""Type aliases for voxelwise gradient rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

# (factor_tag, mean_per_voxel, [G1, G2, ...], lambdas, mni_xyz, mask_flat_indices)
# Gradients are pd.Series indexed by flat voxel index (int as str).
VoxelGradientRunRow = tuple[
    str,
    pd.Series,
    list[pd.Series],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]
