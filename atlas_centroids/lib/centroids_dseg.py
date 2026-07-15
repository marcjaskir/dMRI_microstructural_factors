"""Voxel-mean centroids (MNI mm) from a discrete-segmentation NIfTI + index/label TSV.

Mirrors the centroid math in ``code/atlases/gen_glasser_atlas_centroids.py``:
``argwhere`` of voxels with a given parcel id, ``+0.5`` voxel-center offset, then
``apply_affine`` to world (MNI mm) coordinates; output is the mean of those points.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.affines import apply_affine


def _load_index_to_label(dseg_tsv: Path) -> dict[int, str]:
    df = pd.read_csv(dseg_tsv, sep="\t")
    if "index" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"TSV must have 'index' and 'label' columns: {dseg_tsv}"
        )
    out: dict[int, str] = {}
    for _, row in df.iterrows():
        out[int(row["index"])] = str(row["label"]).strip()
    return out


def _load_filter_indices(
    dseg_tsv: Path,
    atlas_name_filter: frozenset[str] | None,
) -> set[int] | None:
    """Return the set of TSV indices whose ``atlas_name`` is in the filter (or ``None``)."""
    if atlas_name_filter is None:
        return None
    df = pd.read_csv(dseg_tsv, sep="\t")
    if "atlas_name" not in df.columns:
        raise ValueError(
            f"TSV missing 'atlas_name' column required for filtering: {dseg_tsv}"
        )
    mask = df["atlas_name"].astype(str).isin(atlas_name_filter)
    return {int(i) for i in df.loc[mask, "index"].tolist()}


def _voxel_mean_centroids(
    data: np.ndarray,
    affine: np.ndarray,
    keep_ids: set[int] | None,
) -> dict[int, np.ndarray]:
    lab = np.rint(data).astype(np.int32)
    ids = np.unique(lab)
    ids = ids[ids > 0]
    out: dict[int, np.ndarray] = {}
    for pid in ids:
        pid_i = int(pid)
        if keep_ids is not None and pid_i not in keep_ids:
            continue
        ijk = np.argwhere(lab == pid_i).astype(np.float64)
        if ijk.size == 0:
            continue
        ijk += 0.5
        xyz = apply_affine(affine, ijk)
        out[pid_i] = np.nanmean(xyz, axis=0)
    return out


def centroids_from_dseg(
    nifti_path: Path,
    dseg_tsv: Path,
    *,
    atlas_name_filter: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Compute MNI-mm centroids from a discrete-segmentation NIfTI.

    Parameters
    ----------
    nifti_path
        Path to the discrete-segmentation NIfTI (e.g. ``*_dseg.nii.gz``).
    dseg_tsv
        Path to the matching TSV (must have ``index`` and ``label`` columns; must
        also have ``atlas_name`` when ``atlas_name_filter`` is provided).
    atlas_name_filter
        Optional set of ``atlas_name`` values to retain (e.g.
        ``{ThalamusHCP, SubcorticalHCP, Cerebellum, CIT168Subcortical}``). When
        ``None``, every non-zero parcel id present in the NIfTI is returned.

    Returns
    -------
    DataFrame
        Columns ``label, x, y, z`` sorted by parcel id.
    """
    nifti_path = Path(nifti_path)
    dseg_tsv = Path(dseg_tsv)
    if not nifti_path.is_file():
        raise FileNotFoundError(nifti_path)
    if not dseg_tsv.is_file():
        raise FileNotFoundError(dseg_tsv)

    index_to_label = _load_index_to_label(dseg_tsv)
    keep_ids = _load_filter_indices(dseg_tsv, atlas_name_filter)

    img = nib.load(str(nifti_path))
    data = np.asanyarray(img.dataobj)
    affine = np.asarray(img.affine)

    id_to_centroid = _voxel_mean_centroids(data, affine, keep_ids)

    rows: list[tuple[int, str, float, float, float]] = []
    for pid, xyz in id_to_centroid.items():
        name = index_to_label.get(pid)
        if name is None:
            continue
        rows.append((pid, name, float(xyz[0]), float(xyz[1]), float(xyz[2])))
    rows.sort(key=lambda r: r[0])

    if not rows:
        return pd.DataFrame(columns=["label", "x", "y", "z"])
    return pd.DataFrame(
        [{"label": r[1], "x": r[2], "y": r[3], "z": r[4]} for r in rows]
    )
