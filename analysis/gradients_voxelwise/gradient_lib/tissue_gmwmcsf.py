"""GM/WM/CSF tissue masks from resliced probseg maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import DEFAULT_MASK_NII, PROBSEG_THRESHOLD, TISSUE_CLASSES
from .io_voxelwise import load_analysis_mask, reslice_probseg_maps


def load_tissue_masks_inclusive(
    *,
    tractometry_root: Path | None = None,
    cache_dir: Path | None = None,
    threshold: float = PROBSEG_THRESHOLD,
    mask_nii: Path | None = None,
) -> dict[str, np.ndarray]:
    """Return in-mask bool vectors for GM, WM, CSF (prob >= threshold; inclusive)."""
    _ = tractometry_root
    mask_path = Path(mask_nii) if mask_nii is not None else DEFAULT_MASK_NII
    mask_img, mask = load_analysis_mask(mask_path)
    return reslice_probseg_maps(
        mask_img,
        cache_dir=cache_dir,
        threshold=threshold,
        analysis_mask=mask,
    )


def tissue_centroids_g1_g2(
    g1: np.ndarray,
    g2: np.ndarray,
    tissue_masks: dict[str, np.ndarray],
    *,
    tissue_classes: tuple[str, ...] = TISSUE_CLASSES,
) -> dict[str, tuple[float, float]]:
    """Mean G1/G2 per tissue class (inclusive assignment)."""
    out: dict[str, tuple[float, float]] = {}
    for tissue in tissue_classes:
        m = tissue_masks.get(tissue)
        if m is None or not np.any(m):
            out[tissue] = (np.nan, np.nan)
            continue
        out[tissue] = (float(np.nanmean(g1[m])), float(np.nanmean(g2[m])))
    return out


def subsample_voxels_stratified(
    g1: np.ndarray,
    g2: np.ndarray,
    tissue_masks: dict[str, np.ndarray],
    *,
    max_points: int,
    rng: np.random.Generator | None = None,
    tissue_classes: tuple[str, ...] = TISSUE_CLASSES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subsample scatter points up to max_points, stratified by tissue membership.

    Returns (g1_plot, g2_plot, tissue_label_per_point).
    """
    gen = rng if rng is not None else np.random.default_rng(0)
    labels: list[str] = []
    indices: list[int] = []
    for tissue in tissue_classes:
        m = tissue_masks.get(tissue)
        if m is None:
            continue
        idx = np.flatnonzero(m)
        if idx.size:
            labels.extend([tissue] * idx.size)
            indices.extend(idx.tolist())

    if not indices:
        n = min(max_points, g1.size)
        pick = gen.choice(g1.size, size=n, replace=False) if g1.size > n else np.arange(g1.size)
        return g1[pick], g2[pick], np.array(["GM"] * pick.size, dtype=object)

    indices_arr = np.asarray(indices, dtype=np.int64)
    labels_arr = np.asarray(labels, dtype=object)
    if indices_arr.size <= max_points:
        return g1[indices_arr], g2[indices_arr], labels_arr

    n_classes = max(1, len(tissue_classes))
    per_tissue = max(1, max_points // n_classes)
    picked: list[int] = []
    picked_labels: list[str] = []
    for tissue in tissue_classes:
        m = tissue_masks.get(tissue)
        if m is None:
            continue
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        n = min(per_tissue, idx.size)
        choice = gen.choice(idx, size=n, replace=False)
        picked.extend(choice.tolist())
        picked_labels.extend([tissue] * n)

    pick = np.asarray(picked, dtype=np.int64)
    lab = np.asarray(picked_labels, dtype=object)
    if pick.size > max_points:
        sel = gen.choice(pick.size, size=max_points, replace=False)
        pick = pick[sel]
        lab = lab[sel]
    return g1[pick], g2[pick], lab
