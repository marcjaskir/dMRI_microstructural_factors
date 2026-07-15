"""Type aliases for controls-only group-level gradient rows (DM and Laplacian)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# (factor_tag, mean_per_roi, [G1, G2, ...] Series list, lambdas_).
# Gradients are pd.Series indexed by ROI name; ``grads[0]`` = principal (G1), ``grads[1]`` = G2, etc.
GradientRunRow = tuple[str, pd.Series, list[pd.Series], np.ndarray]
