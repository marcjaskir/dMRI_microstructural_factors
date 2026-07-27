"""Sparse subject-correlation affinity and Laplacian embedding for voxels."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from brainspace.gradient.embedding import LaplacianEigenmaps
from scipy.spatial import cKDTree

from .config import (
    DEFAULT_EMBED_STRIDE,
    DEFAULT_INTERP_NEIGHBORS,
    DEFAULT_MAX_EMBED_VOXELS,
    DEFAULT_TOP_K,
    N_GRADIENTS_TO_COMPUTE,
)


def select_embed_voxel_indices(
    mask: np.ndarray,
    *,
    embed_stride: int = DEFAULT_EMBED_STRIDE,
    max_embed_voxels: int = DEFAULT_MAX_EMBED_VOXELS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Stride subsample in-mask voxels; cap count with random subsample."""
    stride = max(1, int(embed_stride))
    ijk = np.column_stack(np.where(mask))
    if stride > 1:
        keep = (
            (ijk[:, 0] % stride == 0)
            & (ijk[:, 1] % stride == 0)
            & (ijk[:, 2] % stride == 0)
        )
        ijk = ijk[keep]
    flat = np.ravel_multi_index((ijk[:, 0], ijk[:, 1], ijk[:, 2]), mask.shape)
    if flat.size > max_embed_voxels:
        gen = rng if rng is not None else np.random.default_rng(0)
        pick = gen.choice(flat.size, size=max_embed_voxels, replace=False)
        flat = np.sort(flat[pick])
    return flat.astype(np.int64, copy=False)


def _standardize_columns(x: np.ndarray) -> np.ndarray:
    """Standardize each column (voxel) across subjects."""
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, ddof=0, keepdims=True)
    std = np.where(std <= 0, 1.0, std)
    z = (x - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.astype(np.float64, copy=False)


def correlation_affinity_embed_subset(
    x_subjects_by_voxels: np.ndarray,
    embed_cols: np.ndarray,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> np.ndarray:
    """Nonnegative Pearson affinity among embed voxels (columns of x)."""
    z = _standardize_columns(x_subjects_by_voxels[:, embed_cols])
    n_sub = z.shape[0]
    if n_sub < 3:
        raise ValueError(f"Need >=3 subjects for correlation affinity, got {n_sub}")

    corr = (z.T @ z) / max(n_sub - 1, 1)
    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.maximum(corr, 0.0)

    n = corr.shape[0]
    k = min(max(1, int(top_k)), max(1, n - 1))
    sparse = np.zeros_like(corr)
    for i in range(n):
        row = corr[i].copy()
        row[i] = -np.inf
        if k >= n - 1:
            keep = np.arange(n) != i
            sparse[i, keep] = row[keep]
        else:
            idx = np.argpartition(row, -k)[-k:]
            sparse[i, idx] = row[idx]
    sparse = np.maximum(sparse, sparse.T)
    np.fill_diagonal(sparse, 0.0)
    return sparse


def fit_laplacian_on_affinity(
    w: np.ndarray,
    *,
    n_components: int = N_GRADIENTS_TO_COMPUTE,
) -> tuple[np.ndarray, np.ndarray]:
    """Run BrainSpace LaplacianEigenmaps on dense affinity matrix."""
    n = w.shape[0]
    k_req = min(max(1, int(n_components)), max(1, n - 1))
    d = w.sum(axis=1)
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        w = w + np.eye(n, dtype=np.float64) * 1e-10

    try:
        le = LaplacianEigenmaps(
            n_components=k_req,
            norm_laplacian=True,
            random_state=0,
        )
        le.fit(w)
        maps = np.asarray(le.maps_, dtype=np.float64)
        lambdas = np.asarray(le.lambdas_, dtype=np.float64).ravel().copy()
    except Exception as exc:
        warnings.warn(f"LaplacianEigenmaps failed ({exc!r}); returning zeros.")
        maps = np.zeros((n, k_req), dtype=np.float64)
        lambdas = np.full(k_req, np.nan, dtype=np.float64)
    return maps, lambdas


def interpolate_gradients_to_full_mask(
    embed_flat_indices: np.ndarray,
    embed_gradients: np.ndarray,
    all_flat_indices: np.ndarray,
    mni_xyz: np.ndarray,
    *,
    n_neighbors: int = DEFAULT_INTERP_NEIGHBORS,
) -> np.ndarray:
    """IDW interpolation of G1..Gk from embed set to all in-mask voxels."""
    n_voxels = all_flat_indices.size
    n_grads = embed_gradients.shape[1]
    out = np.zeros((n_voxels, n_grads), dtype=np.float64)

    flat_to_row = {int(f): i for i, f in enumerate(all_flat_indices.tolist())}
    embed_rows = np.asarray([flat_to_row[int(f)] for f in embed_flat_indices], dtype=np.int64)
    embed_coords = mni_xyz[embed_rows]

    tree = cKDTree(embed_coords)
    full_coords = mni_xyz
    k_query = min(max(1, int(n_neighbors)), embed_coords.shape[0])
    dist, idx = tree.query(full_coords, k=k_query)

    if k_query == 1:
        dist = dist[:, np.newaxis]
        idx = idx[:, np.newaxis]

    for vi in range(n_voxels):
        d = dist[vi].astype(np.float64)
        d = np.maximum(d, 1e-6)
        w = 1.0 / d
        w /= w.sum()
        out[vi] = (embed_gradients[idx[vi]] * w[:, np.newaxis]).sum(axis=0)

    return out


def compute_voxelwise_laplacian_gradients(
    x_subjects_by_voxels: np.ndarray,
    mask: np.ndarray,
    mni_xyz: np.ndarray,
    *,
    embed_stride: int = DEFAULT_EMBED_STRIDE,
    top_k: int = DEFAULT_TOP_K,
    max_embed_voxels: int = DEFAULT_MAX_EMBED_VOXELS,
    n_components: int = N_GRADIENTS_TO_COMPUTE,
    n_neighbors: int = DEFAULT_INTERP_NEIGHBORS,
) -> tuple[list[pd.Series], np.ndarray, np.ndarray]:
    """
    Fit Laplacian on sparse subject-correlation graph; return full-mask G1..Gk.

    Returns (gradient_series_list, lambdas, all_flat_indices).
    """
    all_flat = np.flatnonzero(mask.ravel()).astype(np.int64)
    embed_flat = select_embed_voxel_indices(
        mask,
        embed_stride=embed_stride,
        max_embed_voxels=max_embed_voxels,
    )
    flat_to_local = {int(f): i for i, f in enumerate(all_flat.tolist())}
    embed_local = np.asarray([flat_to_local[int(f)] for f in embed_flat], dtype=np.int64)

    w = correlation_affinity_embed_subset(
        x_subjects_by_voxels,
        embed_local,
        top_k=top_k,
    )
    maps, lambdas = fit_laplacian_on_affinity(w, n_components=n_components)

    full_grads = interpolate_gradients_to_full_mask(
        embed_flat,
        maps,
        all_flat,
        mni_xyz,
        n_neighbors=n_neighbors,
    )

    idx = pd.Index(all_flat.astype(str), dtype=str)
    series_list = [
        pd.Series(full_grads[:, j].copy(), index=idx)
        for j in range(maps.shape[1])
    ]
    return series_list, lambdas, all_flat
