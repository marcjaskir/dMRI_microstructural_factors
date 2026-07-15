"""Laplacian runner and output writers for voxelwise factor maps."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .affinity_voxelwise import compute_voxelwise_laplacian_gradients
from .config import COHORT_TAG, N_GRADIENTS_TO_SAVE, DEFAULT_MAX_EMBED_VOXELS
from .csv_outputs_voxelwise import build_voxel_gradient_dataframe, write_voxel_gradient_csv
from .io_voxelwise import (
    gradient_nii_path,
    load_analysis_mask,
    load_masked_nii_vector,
    save_masked_vector_nii,
    voxel_mni_coordinates,
)
from .types import VoxelGradientRunRow


def load_subject_factor_matrix(
    manifest: pd.DataFrame,
    output_dir: Path,
    factor_tag: str,
    mask: np.ndarray,
    *,
    subject_filter: set[str] | None = None,
    group_filter: set[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Stack subject factor score maps -> (n_subjects, n_voxels)."""
    subjects = manifest.groupby("subject").first().reset_index()
    if group_filter is not None:
        allowed = {str(g) for g in group_filter}
        subjects = subjects[subjects["group"].astype(str).isin(allowed)]
    if subject_filter is not None:
        subjects = subjects[subjects["subject"].astype(str).isin(subject_filter)]
    n_voxels = int(mask.sum())
    x = np.zeros((len(subjects), n_voxels), dtype=np.float32)
    subject_ids: list[str] = []

    for i, row in enumerate(tqdm(subjects.itertuples(), total=len(subjects), desc=f"Loading {factor_tag}")):
        path = (
            output_dir
            / "factor_score_nii"
            / str(row.group)
            / str(row.sub)
            / f"{row.sub}_{factor_tag}.nii.gz"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing factor score map: {path}")
        x[i] = load_masked_nii_vector(path, mask)
        subject_ids.append(str(row.subject))

    return x, subject_ids


def save_voxel_laplacian_outputs(
    row: VoxelGradientRunRow,
    output_dir: Path,
    mask_img: nib.Nifti1Image,
    mask: np.ndarray,
    *,
    cohort_tag: str = COHORT_TAG,
    n_gradients_to_save: int = N_GRADIENTS_TO_SAVE,
) -> tuple[list[Path], Path]:
    """Write G1/G2 NIfTI and CSV tables."""
    factor_tag, mean_per_voxel, gradients, _lambdas, mni_xyz, flat_indices = row
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    idx = gradients[0].index.astype(str)
    mean_aligned = mean_per_voxel.reindex(idx)
    paths: list[Path] = []
    k_save = min(max(1, int(n_gradients_to_save)), len(gradients))
    coords = mni_xyz

    for j in range(k_save):
        g = gradients[j].reindex(idx).to_numpy(dtype=np.float64)
        nii_path = gradient_nii_path(output_dir, factor_tag, j + 1)
        save_masked_vector_nii(g, mask_img, mask, nii_path)

        csv_path = csv_dir / (
            f"{factor_tag}_principal_gradient{j + 1}_scores_cohort-{cohort_tag}.csv"
        )
        df = build_voxel_gradient_dataframe(
            flat_indices.astype(np.int64),
            coords,
            g,
            gradient_index=j + 1,
        )
        path_g, _ = write_voxel_gradient_csv(df, csv_path, gradient_index=j + 1)
        paths.append(path_g)

    mean_path = csv_dir / f"{factor_tag}_factor_score_means_cohort-{cohort_tag}.csv"
    pd.DataFrame(
        {
            "voxel_flat_index": flat_indices.astype(np.int64),
            "mni_x": coords[:, 0],
            "mni_y": coords[:, 1],
            "mni_z": coords[:, 2],
            "mean_factor_score": mean_aligned.to_numpy(dtype=np.float64),
        }
    ).to_csv(mean_path, index=False)
    return paths, mean_path


def compute_laplacian_voxel_row(
    factor_tag: str,
    manifest: pd.DataFrame,
    output_dir: Path,
    *,
    embed_stride: int,
    embed_top_k: int,
    max_embed_voxels: int = DEFAULT_MAX_EMBED_VOXELS,
    subject_filter: set[str] | None = None,
    group_filter: set[str] | tuple[str, ...] | None = None,
    factor_score_dir: Path | None = None,
    mask_nii: Path | None = None,
) -> VoxelGradientRunRow:
    """Run Laplacian embedding for one factor from saved subject factor score NIfTIs."""
    mask_img, mask = load_analysis_mask(mask_nii)
    mni_xyz = voxel_mni_coordinates(mask_img, mask)
    scores_dir = factor_score_dir or output_dir
    x, _subject_ids = load_subject_factor_matrix(
        manifest,
        scores_dir,
        factor_tag,
        mask,
        subject_filter=subject_filter,
        group_filter=group_filter,
    )
    flat_indices = np.flatnonzero(mask.ravel()).astype(str)
    mean_per_voxel = pd.Series(x.mean(axis=0), index=flat_indices)

    glist, lambdas, flat_indices_int = compute_voxelwise_laplacian_gradients(
        x.astype(np.float64),
        mask,
        mni_xyz,
        embed_stride=embed_stride,
        top_k=embed_top_k,
        max_embed_voxels=max_embed_voxels,
    )
    return factor_tag, mean_per_voxel, glist, lambdas, mni_xyz, flat_indices_int
