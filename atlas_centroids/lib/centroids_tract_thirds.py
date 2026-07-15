"""HCP1065 tract-third positions (end1, core, end2) in MNI mm.

Each per-tract ``*_model_centroids.npy`` is a ``(100, 3)`` MNI-mm point cloud (one
row per pyAFQ node 1..100). Per ``code/prompt_context/structural_tractometry_context.md``
lines 80-82 the canonical thirds are nodes 1..34 (end1), 35..66 (core), 67..100 (end2).

For neuroanatomical axis ranks we use the **centermost node** in each third (guaranteed
on the atlas streamline), not the mean across nodes in the third:

    END1_CENTER_NODE = 17   (within nodes 1..34)
    CORE_CENTER_NODE = 50  (within nodes 35..66)
    END2_CENTER_NODE = 83  (within nodes 67..100)

Labels follow ``parse_wm_tract_end_column_name`` in
``gradients_group-controls/gradient_lib/io.py``: ``{tract}_end-{END1_CODE}``,
``{tract}_core``, ``{tract}_end-{END2_CODE}``, where the end codes come from the
``end1`` / ``end2`` columns of ``HCP1065_tract_metadata.csv`` (e.g. ``AF_L`` =>
``AF_L_end-A``, ``AF_L_core``, ``AF_L_end-P``).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 1-based pyAFQ node indices (centermost node per third).
END1_CENTER_NODE = 17
CORE_CENTER_NODE = 50
END2_CENTER_NODE = 83

_NPY_SUFFIX = "_model_centroids.npy"
_N_NODES = 100


def _norm_end_code(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s.upper() in ("", "NA", "N/A", "NAN"):
        return None
    return s


def _read_metadata(metadata_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    # Strip a possible UTF-8 BOM on the leading 'label' column.
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    required = {"label", "profilable", "end1", "end2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"HCP1065 metadata missing required columns {sorted(missing)}: {metadata_csv}"
        )
    return df


def _is_truthy(x: object) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("true", "t", "1", "yes", "y")


def _node_xyz(arr64: np.ndarray, node_1based: int) -> np.ndarray | None:
    """MNI mm coordinates at ``node_1based`` (1..100), or None if invalid."""
    if not (1 <= node_1based <= _N_NODES):
        return None
    xyz = arr64[node_1based - 1]
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        return None
    return xyz


def tract_third_centroids(
    centroids_dir: Path,
    metadata_csv: Path,
) -> pd.DataFrame:
    """MNI-mm position per tract-third from centermost pyAFQ nodes (17, 50, 83).

    Returns a DataFrame with columns ``label, x, y, z`` (one row per third) sorted
    by tract label. Tracts marked ``profilable != TRUE`` are skipped silently;
    tracts whose ``.npy`` is missing, malformed, or has missing ``end1``/``end2``
    codes are skipped with a ``warnings.warn`` message.
    """
    centroids_dir = Path(centroids_dir)
    metadata_csv = Path(metadata_csv)
    if not centroids_dir.is_dir():
        raise FileNotFoundError(centroids_dir)
    if not metadata_csv.is_file():
        raise FileNotFoundError(metadata_csv)

    meta = _read_metadata(metadata_csv)
    rows: list[dict[str, float | str]] = []

    for _, row in meta.iterrows():
        label = str(row["label"]).strip()
        if not label:
            continue
        if not _is_truthy(row["profilable"]):
            continue
        end1 = _norm_end_code(row["end1"])
        end2 = _norm_end_code(row["end2"])
        if end1 is None or end2 is None:
            warnings.warn(
                f"HCP1065 tract {label!r} is profilable but missing end1/end2 codes; skipping.",
                stacklevel=2,
            )
            continue

        npy_path = centroids_dir / f"{label}{_NPY_SUFFIX}"
        if not npy_path.is_file():
            warnings.warn(
                f"HCP1065 centroid file missing for {label!r}: {npy_path}",
                stacklevel=2,
            )
            continue

        try:
            arr = np.load(str(npy_path))
        except Exception as exc:  # corrupt or unreadable .npy
            warnings.warn(
                f"Failed to read HCP1065 centroid file {npy_path}: {exc}",
                stacklevel=2,
            )
            continue

        if arr.ndim != 2 or arr.shape != (_N_NODES, 3):
            warnings.warn(
                f"HCP1065 centroid array for {label!r} has unexpected shape {arr.shape}; "
                f"expected ({_N_NODES}, 3); skipping.",
                stacklevel=2,
            )
            continue

        arr64 = arr.astype(np.float64)
        third_specs: list[tuple[str, int]] = [
            (f"{label}_end-{end1}", END1_CENTER_NODE),
            (f"{label}_core", CORE_CENTER_NODE),
            (f"{label}_end-{end2}", END2_CENTER_NODE),
        ]
        for third_label, node_1based in third_specs:
            xyz = _node_xyz(arr64, node_1based)
            if xyz is None:
                continue
            rows.append(
                {
                    "label": third_label,
                    "x": float(xyz[0]),
                    "y": float(xyz[1]),
                    "z": float(xyz[2]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["label", "x", "y", "z"])
    out = pd.DataFrame(rows)
    out.sort_values("label", inplace=True, kind="stable")
    out.reset_index(drop=True, inplace=True)
    return out
