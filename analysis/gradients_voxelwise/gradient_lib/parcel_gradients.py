"""Aggregate voxel G1/G2 to atlas parcel means; adapt to group-controls GradientRunRow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .atlas_voxelwise import ParcelCollection, VoxelwiseAtlasContext, load_voxelwise_atlas_context

_ATLAS_CACHES: dict[str, VoxelwiseAtlasContext] = {}


def get_atlas_context(
    cache_dir: Path | None = None,
    *,
    mask_nii: Path | None = None,
) -> VoxelwiseAtlasContext:
    from .config import DEFAULT_MASK_NII

    mask_path = Path(mask_nii) if mask_nii is not None else DEFAULT_MASK_NII
    key = str(mask_path.resolve())
    if key not in _ATLAS_CACHES:
        _ATLAS_CACHES[key] = load_voxelwise_atlas_context(
            cache_dir=cache_dir,
            mask_nii=mask_path,
        )
    return _ATLAS_CACHES[key]


def clear_atlas_cache() -> None:
    _ATLAS_CACHES.clear()


from .embedding import gradient_from_row
from .types import VoxelGradientRunRow

# group-controls GradientRunRow alias
GradientRunRow = tuple[str, pd.Series, list[pd.Series], np.ndarray]


def _mean_over_mask(values: np.ndarray, parcel_mask: np.ndarray) -> float:
    sel = values[parcel_mask]
    if sel.size == 0:
        return np.nan
    return float(np.nanmean(sel))


def aggregate_parcel_gradients(
    g_values: np.ndarray,
    collection: ParcelCollection,
) -> pd.Series:
    """Mean gradient per parcel label."""
    out: dict[str, float] = {}
    for label, pmask in collection.masks.items():
        val = _mean_over_mask(g_values, pmask)
        if np.isfinite(val):
            out[label] = val
    return pd.Series(out)


def aggregate_all_parcel_gradients(
    atlas: VoxelwiseAtlasContext,
    g_values: np.ndarray,
) -> pd.Series:
    """Combine Glasser + subcortex + WM tract parcel means into one Series."""
    parts: list[pd.Series] = []
    for coll in (atlas.glasser, atlas.subcortex, atlas.wm_tracts):
        s = aggregate_parcel_gradients(g_values, coll)
        if not s.empty:
            parts.append(s)
    if not parts:
        return pd.Series(dtype=np.float64)
    return pd.concat(parts)


def voxel_row_to_parcel_gradient_run_row(
    row: VoxelGradientRunRow,
    atlas: VoxelwiseAtlasContext | None = None,
    *,
    cache_dir: Path | None = None,
    mask_nii: Path | None = None,
) -> GradientRunRow:
    """Convert voxelwise row to parcel-level GradientRunRow for group-controls plots."""
    if atlas is None:
        atlas = get_atlas_context(cache_dir=cache_dir, mask_nii=mask_nii)

    flat = row[5].astype(str)
    mean_per_voxel = row[1].reindex(flat).to_numpy(dtype=np.float64)
    mean_per_parcel = aggregate_all_parcel_gradients(atlas, mean_per_voxel)

    parcel_grads: list[pd.Series] = []
    for gi in range(len(row[2])):
        g_vals = gradient_from_row(row, gi).reindex(flat).to_numpy(dtype=np.float64)
        parcel_grads.append(aggregate_all_parcel_gradients(atlas, g_vals))

    return row[0], mean_per_parcel, parcel_grads, row[3]


def voxel_rows_to_parcel_gradient_run_rows(
    rows: list[VoxelGradientRunRow],
    atlas: VoxelwiseAtlasContext | None = None,
    *,
    cache_dir: Path | None = None,
    mask_nii: Path | None = None,
) -> list[GradientRunRow]:
    if atlas is None:
        atlas = get_atlas_context(cache_dir=cache_dir, mask_nii=mask_nii)
    return [
        voxel_row_to_parcel_gradient_run_row(row, atlas)
        for row in rows
    ]


def save_parcel_gradient_csvs(
    rows: list[VoxelGradientRunRow],
    csv_dir: Path,
    atlas: VoxelwiseAtlasContext | None = None,
    *,
    cache_dir: Path | None = None,
    cohort_tag: str = "controls",
) -> list[Path]:
    """Optional parcel-level gradient CSVs for debugging/regeneration."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    if atlas is None:
        atlas = get_atlas_context(cache_dir=cache_dir)
    paths: list[Path] = []
    for row in rows:
        tag = row[0]
        for gi in range(min(2, len(row[2]))):
            flat = row[5].astype(str)
            g_vals = gradient_from_row(row, gi).reindex(flat).to_numpy(dtype=np.float64)
            s = aggregate_all_parcel_gradients(atlas, g_vals)
            path = csv_dir / (
                f"{tag}_parcel_gradient{gi + 1}_scores_cohort-{cohort_tag}.csv"
            )
            pd.DataFrame({"region": s.index.astype(str), f"gradient{gi + 1}": s.values}).to_csv(
                path, index=False
            )
            paths.append(path)
    return paths
