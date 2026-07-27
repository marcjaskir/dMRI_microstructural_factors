"""CSV writers for voxelwise gradient tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _sorted_gradient_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_sorted{path.suffix}")


def write_voxel_gradient_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    gradient_index: int,
) -> tuple[Path, Path]:
    """Write gradient CSV and sorted copy."""
    score_col = f"principal_gradient{gradient_index}_score"
    if score_col not in df.columns:
        raise ValueError(f"Missing column {score_col!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    sorted_path = _sorted_gradient_path(path)
    df.sort_values(score_col, ascending=False, kind="mergesort").to_csv(
        sorted_path, index=False
    )
    return path, sorted_path


def build_voxel_gradient_dataframe(
    flat_indices: np.ndarray,
    mni_xyz: np.ndarray,
    gradient_values: np.ndarray,
    *,
    gradient_index: int,
) -> pd.DataFrame:
    """Build CSV table with voxel index, MNI coords, and one gradient column."""
    return pd.DataFrame(
        {
            "voxel_flat_index": flat_indices.astype(np.int64),
            "mni_x": mni_xyz[:, 0],
            "mni_y": mni_xyz[:, 1],
            "mni_z": mni_xyz[:, 2],
            f"principal_gradient{gradient_index}_score": gradient_values.astype(np.float64),
        }
    )
