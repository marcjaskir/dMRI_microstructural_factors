"""Load VoxelGradientRunRow tuples from saved outputs on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import COHORT_TAG, DEFAULT_GRADIENTS_VOXELWISE_DIR
from .io_voxelwise import load_analysis_mask, load_masked_nii_vector, voxel_mni_coordinates
from .types import VoxelGradientRunRow


def _gradient_csv(csv_dir: Path, tag: str, j: int, cohort_tag: str) -> Path | None:
    path = csv_dir / f"{tag}_principal_gradient{j}_scores_cohort-{cohort_tag}.csv"
    return path if path.is_file() else None


def load_voxel_rows_from_output(
    output_dir: Path | None = None,
    *,
    cohort_tag: str = COHORT_TAG,
    factors: list[str] | None = None,
    mask_nii: Path | None = None,
) -> list[VoxelGradientRunRow]:
    """Reconstruct voxel rows from saved CSVs and factor score means."""
    root = output_dir or DEFAULT_GRADIENTS_VOXELWISE_DIR
    csv_dir = root / "csv"
    mask_img, mask = load_analysis_mask(mask_nii)
    mni_xyz = voxel_mni_coordinates(mask_img, mask)
    flat_indices = np.flatnonzero(mask.ravel()).astype(np.int64)

    if factors is None:
        factors = sorted(
            {
                p.name.split("_principal_gradient")[0]
                for p in csv_dir.glob(f"F*_principal_gradient1_scores_cohort-{cohort_tag}.csv")
            }
        )

    rows: list[VoxelGradientRunRow] = []
    for tag in factors:
        means_path = csv_dir / f"{tag}_factor_score_means_cohort-{cohort_tag}.csv"
        if not means_path.is_file():
            continue
        means_df = pd.read_csv(means_path)
        mean_idx = means_df["voxel_flat_index"].astype(str)
        mean_per_voxel = pd.Series(
            means_df["mean_factor_score"].to_numpy(dtype=np.float64),
            index=mean_idx,
        )
        grads: list[pd.Series] = []
        for j in range(1, 10):
            gp = _gradient_csv(csv_dir, tag, j, cohort_tag)
            if gp is None:
                break
            gdf = pd.read_csv(gp)
            idx = gdf["voxel_flat_index"].astype(str)
            col = f"principal_gradient{j}_score"
            grads.append(pd.Series(gdf[col].to_numpy(dtype=np.float64), index=idx))
        if not grads:
            nii_g1 = root / "factor_gradient_nii" / f"{tag}_G1.nii.gz"
            if nii_g1.is_file():
                g1 = load_masked_nii_vector(nii_g1, mask)
                idx = flat_indices.astype(str)
                grads.append(pd.Series(g1, index=idx))
                nii_g2 = root / "factor_gradient_nii" / f"{tag}_G2.nii.gz"
                if nii_g2.is_file():
                    g2 = load_masked_nii_vector(nii_g2, mask)
                    grads.append(pd.Series(g2, index=idx))
        rows.append(
            (tag, mean_per_voxel, grads, np.array([], dtype=np.float64), mni_xyz, flat_indices)
        )
    return rows
