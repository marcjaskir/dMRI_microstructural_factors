"""Manifest, mask, probseg, and NIfTI I/O for voxelwise gradients."""

from __future__ import annotations

import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .config import (
    DEFAULT_LOADINGS_CSV,
    DEFAULT_MANIFEST_CSV,
    DEFAULT_MASK_NII,
    PROBSEG_PATHS,
    PROBSEG_THRESHOLD,
)


def load_factor_loadings(path: Path | None = None) -> pd.DataFrame:
    """Load factor loadings CSV (index F1–Fn)."""
    p = path or DEFAULT_LOADINGS_CSV
    df = pd.read_csv(p)
    if "factor" in df.columns:
        df = df.set_index("factor")
    df.index = df.index.astype(str)
    return df


def load_manifest(path: Path | None = None) -> pd.DataFrame:
    """Load scalar image manifest for reduced controls."""
    p = path or DEFAULT_MANIFEST_CSV
    return pd.read_csv(p)


def load_analysis_mask(
    path: Path | None = None,
) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load boolean analysis mask (966k voxels on scalar grid)."""
    p = path or DEFAULT_MASK_NII
    mask_img = nib.load(str(p))
    mask = np.asarray(mask_img.get_fdata() > 0)
    return mask_img, mask


def load_masked_scalar(path: str, mask_img: nib.Nifti1Image, mask: np.ndarray) -> np.ndarray:
    """Load one scalar NIfTI and return in-mask vector."""
    img = nib.load(path)
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim > 3:
        data = np.squeeze(data)
    if data.shape != mask.shape:
        raise ValueError(f"Image shape {data.shape} != mask shape {mask.shape}: {path}")
    if not np.allclose(img.affine, mask_img.affine, atol=1e-3):
        raise ValueError(f"Image affine mismatch: {path}")
    return data[mask].astype(np.float32, copy=False)


def save_masked_vector_nii(
    values: np.ndarray,
    mask_img: nib.Nifti1Image,
    mask: np.ndarray,
    output_path: Path,
    fill_value: float = 0.0,
) -> None:
    """Write a flat masked vector back to 3D NIfTI."""
    data = np.full(mask.shape, fill_value, dtype=np.float32)
    data[mask] = values.astype(np.float32, copy=False)
    header = mask_img.header.copy()
    header.set_data_dtype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, mask_img.affine, header), str(output_path))


def load_masked_nii_vector(
    path: Path,
    mask: np.ndarray,
) -> np.ndarray:
    """Load NIfTI and extract in-mask vector."""
    data = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim > 3:
        data = np.squeeze(data)
    return data[mask].astype(np.float32, copy=False)


def voxel_mni_coordinates(
    mask_img: nib.Nifti1Image,
    mask: np.ndarray,
) -> np.ndarray:
    """Return (n_voxels, 3) MNI mm coordinates for in-mask voxels."""
    ijk = np.column_stack(np.where(mask))
    xyz = nib.affines.apply_affine(mask_img.affine, ijk)
    return xyz.astype(np.float64)


def flat_indices_from_mask(mask: np.ndarray) -> np.ndarray:
    """Flat indices of in-mask voxels."""
    return np.flatnonzero(mask.ravel())


def factor_score_nii_path(
    output_dir: Path,
    group: str,
    sub: str,
    factor_tag: str,
) -> Path:
    """``factor_score_nii/{group}/{sub}/{sub}_F{n}.nii.gz``."""
    return output_dir / "factor_score_nii" / group / sub / f"{sub}_{factor_tag}.nii.gz"


def gradient_nii_path(output_dir: Path, factor_tag: str, gradient_index: int) -> Path:
    """``factor_gradient_nii/F{n}_G{j}.nii.gz`` (1-based gradient index)."""
    return output_dir / "factor_gradient_nii" / f"{factor_tag}_G{gradient_index}.nii.gz"


def check_duplicate_subs(manifest: pd.DataFrame) -> None:
    """Warn if the same ``sub`` appears under multiple subject sessions."""
    dup = manifest.groupby("sub")["subject"].nunique()
    dup = dup[dup > 1]
    if not dup.empty:
        warnings.warn(
            f"{len(dup)} sub ID(s) map to multiple sessions; "
            f"factor_score_nii/{{group}}/{{sub}}/ will overwrite: "
            f"{dup.index[:5].tolist()}{'...' if len(dup) > 5 else ''}",
            stacklevel=2,
        )


def reslice_probseg_maps(
    mask_img: nib.Nifti1Image,
    *,
    cache_dir: Path | None = None,
    threshold: float = PROBSEG_THRESHOLD,
    analysis_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Reslice GM/WM/CSF probseg to scalar grid; return in-mask bool vectors (prob >= threshold)."""
    from nibabel.processing import resample_from_to

    target = (mask_img.shape[:3], mask_img.affine)
    mask = analysis_mask if analysis_mask is not None else np.asarray(mask_img.get_fdata() > 0)
    mask_flat = mask.ravel()
    out: dict[str, np.ndarray] = {}

    for tissue, src_path in PROBSEG_PATHS.items():
        if not src_path.is_file():
            raise FileNotFoundError(
                f"Missing tissue probseg for {tissue}: {src_path}. "
                "Expected under data/atlases/MNI/."
            )
        cache_path = None
        if cache_dir is not None:
            cache_path = cache_dir / f"resliced_{tissue}_probseg_thr-{threshold:g}.nii.gz"
            if cache_path.is_file():
                cached = np.asarray(nib.load(str(cache_path)).get_fdata(), dtype=np.float32)
                out[tissue] = cached.ravel()[mask_flat] >= threshold
                continue

        resampled = resample_from_to(nib.load(str(src_path)), target, order=0)
        data = np.asarray(resampled.get_fdata(dtype=np.float32), dtype=np.float32)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(resampled, str(cache_path))
        out[tissue] = data.ravel()[mask_flat] >= threshold

    return out


def scalar_labels_from_loadings(loadings_df: pd.DataFrame) -> list[str]:
    """Column order for dot product with loadings."""
    return [str(c) for c in loadings_df.columns]
