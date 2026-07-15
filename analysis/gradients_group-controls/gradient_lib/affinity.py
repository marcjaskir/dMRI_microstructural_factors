"""Column-mean imputation, nonnegative Pearson affinity, BrainSpace DM and Laplacian gradients."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from brainspace.gradient import GradientMaps
from brainspace.gradient.embedding import LaplacianEigenmaps

from .config import (
    N_GRADIENTS_TO_COMPUTE,
    SPARSITY_BY_MODE,
)


def column_mean_impute_region_matrix(
    X: pd.DataFrame,
    regions: list[str],
) -> pd.DataFrame:
    sub = X[regions].copy()
    means = sub.mean(axis=0, skipna=True)
    out = sub.fillna(means)
    if out.isna().any().any():
        bad = [str(c) for c in out.columns if out[c].isna().any()]
        out = out.fillna(0.0)
        warnings.warn(
            "Column mean imputation: no finite values in column(s) "
            f"{bad[:8]}{'...' if len(bad) > 8 else ''}; filled with 0."
        )
    return out


def nonnegative_correlation_affinity_matrix(
    X: pd.DataFrame,
    region_cols: list[str],
) -> tuple[np.ndarray, list[str]] | None:
    """Pearson correlation across subjects, clamped to >=0, symmetric, zero diagonal."""
    regions = [c for c in region_cols if c in X.columns]
    if len(regions) < 3:
        warnings.warn(
            f"Affinity: need >=3 regions, got {len(regions)}; skipping embedding."
        )
        return None

    Xm = column_mean_impute_region_matrix(X, regions)
    C = Xm.corr(method="pearson")
    W = np.array(C.to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    np.fill_diagonal(W, 0.0)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    W = np.maximum(W, 0.0)
    W = (W + W.T) / 2.0

    d = W.sum(axis=1)
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        W = W + np.eye(len(regions), dtype=np.float64) * 1e-10
        d = W.sum(axis=1)

    if d.sum() == 0.0 or not np.all(np.isfinite(d)) or np.any(d <= 0):
        warnings.warn("Degenerate affinity graph; cannot build embedding.")
        return None

    return W, regions


def _min_pad_grads(grads: list[pd.Series], idx: pd.Index, min_n: int) -> list[pd.Series]:
    while len(grads) < min_n:
        grads.append(pd.Series(0.0, index=idx))
    return grads


def diffusion_map_gradients(
    X: pd.DataFrame,
    region_cols: list[str],
    *,
    alpha: float,
    sparsity_mode: str,
    n_components: int = N_GRADIENTS_TO_COMPUTE,
    min_components: int = 3,
) -> tuple[list[pd.Series], np.ndarray]:
    """Fit BrainSpace ``GradientMaps(approach='dm')`` on a nonnegative correlation affinity.

    Returns the leading ``n_components`` gradients (list of pd.Series, G1..Gk) plus the
    eigenvalues (``lambdas_``). The gradient list is padded with zero Series up to
    ``min_components`` so downstream code can always index G1..G3.
    """
    if sparsity_mode not in SPARSITY_BY_MODE:
        raise ValueError(
            f"Unknown sparsity_mode {sparsity_mode!r}; expected one of {list(SPARSITY_BY_MODE)}"
        )
    sparsity = SPARSITY_BY_MODE[sparsity_mode]

    aff = nonnegative_correlation_affinity_matrix(X, region_cols)
    if aff is None:
        regions = [c for c in region_cols if c in X.columns]
        idx = pd.Index(regions, dtype=str)
        return (
            _min_pad_grads([], idx, max(min_components, n_components)),
            np.full(n_components, np.nan, dtype=np.float64),
        )

    W, regions = aff
    idx = pd.Index(regions, dtype=str)
    max_modes = max(1, len(regions) - 1)
    k_req = min(max(1, int(n_components)), max_modes)

    try:
        gm = GradientMaps(
            n_components=k_req,
            approach="dm",
            kernel=None,
            alignment=None,
            random_state=0,
        )
        gm.fit(W, sparsity=sparsity, alpha=float(alpha))
        grads_arr = np.asarray(gm.gradients_, dtype=np.float64)
        lambdas = np.asarray(gm.lambdas_, dtype=np.float64).ravel().copy()
    except Exception as exc:
        warnings.warn(
            f"BrainSpace DiffusionMaps failed ({exc!r}); returning zero gradients."
        )
        return (
            _min_pad_grads([], idx, max(min_components, n_components)),
            np.full(n_components, np.nan, dtype=np.float64),
        )

    if grads_arr.ndim != 2 or grads_arr.shape[1] < 1:
        warnings.warn(
            f"DiffusionMaps returned unexpected gradient shape {grads_arr.shape}; "
            "returning zero gradients."
        )
        return (
            _min_pad_grads([], idx, max(min_components, n_components)),
            np.full(n_components, np.nan, dtype=np.float64),
        )

    grads: list[pd.Series] = [
        pd.Series(grads_arr[:, j].copy(), index=idx)
        for j in range(grads_arr.shape[1])
    ]
    grads = _min_pad_grads(grads, idx, max(min_components, n_components))

    if lambdas.size < n_components:
        lambdas = np.pad(
            lambdas,
            (0, int(n_components - lambdas.size)),
            constant_values=np.nan,
        )
    elif lambdas.size > n_components:
        lambdas = lambdas[:n_components]

    return grads, lambdas


def laplacian_nontrivial_gradients(
    X: pd.DataFrame,
    region_cols: list[str],
    k: int,
    *,
    min_components: int = 3,
) -> tuple[list[pd.Series], np.ndarray]:
    """Fit BrainSpace ``LaplacianEigenmaps`` on the same nonnegative-correlation affinity."""
    k = max(1, int(k))
    aff = nonnegative_correlation_affinity_matrix(X, region_cols)
    if aff is None:
        regions = [c for c in region_cols if c in X.columns]
        idx = pd.Index(regions, dtype=str)
        lam = np.full(k, np.nan, dtype=np.float64)
        return (
            [pd.Series(0.0, index=idx) for _ in range(max(min_components, k))],
            lam,
        )

    W, regions = aff
    idx = pd.Index(regions, dtype=str)
    max_modes = max(1, len(regions) - 1)
    k_req = min(k, max_modes)

    try:
        le = LaplacianEigenmaps(
            n_components=k_req,
            norm_laplacian=True,
            random_state=0,
        )
        le.fit(W)
        maps = np.asarray(le.maps_, dtype=np.float64)
        lambdas = np.asarray(le.lambdas_, dtype=np.float64).ravel().copy()
    except Exception as exc:
        warnings.warn(
            f"BrainSpace LaplacianEigenmaps failed ({exc!r}); padding with zeros."
        )
        lam = np.full(k, np.nan, dtype=np.float64)
        return (
            [pd.Series(0.0, index=idx) for _ in range(max(min_components, k))],
            lam,
        )

    n_col = maps.shape[1]
    series_list: list[pd.Series] = [
        pd.Series(maps[:, j].copy(), index=idx) for j in range(n_col)
    ]
    if lambdas.size < n_col:
        lambdas = np.pad(
            lambdas,
            (0, int(n_col - lambdas.size)),
            constant_values=np.nan,
        )
    elif lambdas.size > n_col:
        lambdas = lambdas[:n_col]

    series_list = _min_pad_grads(series_list, idx, max(min_components, k_req))
    if lambdas.size < k:
        lambdas = np.pad(
            lambdas,
            (0, int(k - lambdas.size)),
            constant_values=np.nan,
        )
    elif lambdas.size > k:
        lambdas = lambdas[:k]
    return series_list, lambdas
