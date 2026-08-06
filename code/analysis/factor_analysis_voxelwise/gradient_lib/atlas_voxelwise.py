"""Reslice and cache Glasser, 4S156 subcortex, and HCP1065 WM atlases on voxelwise grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    ATLAS_4S156_DSEG_NII,
    ATLAS_4S156_DSEG_TSV,
    DEFAULT_MASK_NII,
    DEFAULT_TRACTOMETRY_ROOT,
    FOUR_S_SUBCORTICAL_ATLAS_NAMES,
    GLASSER_DSEG_NII,
    GLASSER_DSEG_TSV,
    HCP1065_ALL_NII_BIN_DIR,
    HCP1065_TRACT_METADATA_CSV,
    MIN_PARCEL_VOXELS,
    atlas_cache_label,
)
from .io_voxelwise import load_analysis_mask


@dataclass
class ParcelCollection:
    """Named parcel masks as in-mask boolean vectors (mask order)."""

    masks: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return sorted(self.masks.keys())


@dataclass
class VoxelwiseAtlasContext:
    analysis_mask: np.ndarray
    glasser: ParcelCollection
    subcortex: ParcelCollection
    wm_tracts: ParcelCollection


def _reslice_to_reference(
    source_path: Path,
    reference_img: nib.Nifti1Image,
    cache_path: Path | None = None,
) -> np.ndarray:
    from nibabel.processing import resample_from_to

    if cache_path is not None and cache_path.is_file():
        return np.asarray(nib.load(str(cache_path)).get_fdata())

    target = (reference_img.shape[:3], reference_img.affine)
    resampled = resample_from_to(nib.load(str(source_path)), target, order=0)
    data = np.asarray(resampled.get_fdata())
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(resampled, str(cache_path))
    return data


def _label_map_to_parcels(
    label_data: np.ndarray,
    analysis_mask: np.ndarray,
    index_to_name: dict[int, str],
    *,
    min_voxels: int = MIN_PARCEL_VOXELS,
) -> ParcelCollection:
    flat_mask = analysis_mask.ravel()
    flat_labels = label_data.ravel()[flat_mask].astype(np.int64)
    out: dict[str, np.ndarray] = {}
    for idx, name in index_to_name.items():
        if idx <= 0:
            continue
        parcel_mask = flat_labels == int(idx)
        if int(parcel_mask.sum()) >= min_voxels:
            out[str(name)] = parcel_mask
    return ParcelCollection(masks=out)


def _load_glasser_parcels(
    reference_img: nib.Nifti1Image,
    analysis_mask: np.ndarray,
    cache_dir: Path,
) -> ParcelCollection:
    tsv = pd.read_csv(GLASSER_DSEG_TSV, sep="\t")
    index_to_name = dict(zip(tsv["index"].astype(int), tsv["label"].astype(str)))
    data = _reslice_to_reference(
        GLASSER_DSEG_NII,
        reference_img,
        cache_dir / "resliced_glasser_dseg.nii.gz",
    )
    return _label_map_to_parcels(data, analysis_mask, index_to_name)


def _load_subcortex_parcels(
    reference_img: nib.Nifti1Image,
    analysis_mask: np.ndarray,
    cache_dir: Path,
) -> ParcelCollection:
    tsv = pd.read_csv(ATLAS_4S156_DSEG_TSV, sep="\t")
    sub = tsv[tsv["atlas_name"].astype(str).isin(FOUR_S_SUBCORTICAL_ATLAS_NAMES)]
    index_to_name = dict(zip(sub["index"].astype(int), sub["label"].astype(str)))
    data = _reslice_to_reference(
        ATLAS_4S156_DSEG_NII,
        reference_img,
        cache_dir / "resliced_4s156_dseg.nii.gz",
    )
    return _label_map_to_parcels(data, analysis_mask, index_to_name)


def _load_hcp1065_tract_parcels(
    reference_img: nib.Nifti1Image,
    analysis_mask: np.ndarray,
    cache_dir: Path,
    *,
    min_voxels: int = MIN_PARCEL_VOXELS,
) -> ParcelCollection:
    if not HCP1065_ALL_NII_BIN_DIR.is_dir():
        raise FileNotFoundError(f"Missing HCP1065 masks: {HCP1065_ALL_NII_BIN_DIR}")

    meta = pd.read_csv(HCP1065_TRACT_METADATA_CSV)
    labels = meta["label"].astype(str).tolist()
    flat_mask = analysis_mask.ravel()
    out: dict[str, np.ndarray] = {}
    wm_cache = cache_dir / "hcp1065_whole"
    wm_cache.mkdir(parents=True, exist_ok=True)

    for label in tqdm(labels, desc="Reslicing HCP1065 tracts", leave=False):
        src = HCP1065_ALL_NII_BIN_DIR / f"{label}.nii.gz"
        if not src.is_file():
            continue
        cache_path = wm_cache / f"{label}.nii.gz"
        data = _reslice_to_reference(src, reference_img, cache_path)
        parcel_mask = (data.ravel()[flat_mask] > 0)
        if int(parcel_mask.sum()) >= min_voxels:
            out[label] = parcel_mask
    return ParcelCollection(masks=out)


def glasser_parcel_label_per_inmask_voxel(
    glasser: ParcelCollection,
    n_inmask: int,
) -> np.ndarray:
    """Glasser/MMP parcel name per in-mask voxel ('' if unassigned)."""
    out = np.full(n_inmask, "", dtype=object)
    for label, pmask in glasser.masks.items():
        out[pmask] = label
    return out


def load_voxelwise_atlas_context(
    *,
    cache_dir: Path | None = None,
    tractometry_root: Path | None = None,
    mask_nii: Path | None = None,
) -> VoxelwiseAtlasContext:
    """Load or build resliced atlas parcel masks on the voxelwise analysis grid."""
    _ = tractometry_root
    mask_path = Path(mask_nii) if mask_nii is not None else DEFAULT_MASK_NII
    mask_img, analysis_mask = load_analysis_mask(mask_path)
    if cache_dir is None:
        from .config import DEFAULT_GRADIENTS_VOXELWISE_DIR

        cache = DEFAULT_GRADIENTS_VOXELWISE_DIR / "_cache" / "atlas" / atlas_cache_label(mask_path)
    else:
        cache = cache_dir / "atlas" / atlas_cache_label(mask_path)
    cache.mkdir(parents=True, exist_ok=True)

    glasser = _load_glasser_parcels(mask_img, analysis_mask, cache)
    subcortex = _load_subcortex_parcels(mask_img, analysis_mask, cache)
    wm_tracts = _load_hcp1065_tract_parcels(mask_img, analysis_mask, cache)
    return VoxelwiseAtlasContext(
        analysis_mask=analysis_mask,
        glasser=glasser,
        subcortex=subcortex,
        wm_tracts=wm_tracts,
    )
