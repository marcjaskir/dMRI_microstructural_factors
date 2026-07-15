"""Paint epilepsy group-mean factor z onto a shared MNI NIfTI."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .config import DEFAULT_TRACTOMETRY_ROOT, GRADIENTS_K, METHOD_TAG

GLASSER_DSEG_NII = (
    DEFAULT_TRACTOMETRY_ROOT
    / "data/atlases/Glasser/atlas-Glasser_space-MNI152NLin2009cAsym_res-1_dseg.nii.gz"
)
GLASSER_DSEG_TSV = (
    DEFAULT_TRACTOMETRY_ROOT / "data/atlases/Glasser/atlas-Glasser_dseg.tsv"
)
ATLAS_4S156_DSEG_NII = (
    DEFAULT_TRACTOMETRY_ROOT
    / "data/atlases/4S/tpl-MNI152NLin2009cAsym_atlas-4S156Parcels_res-01_dseg.nii.gz"
)
ATLAS_4S156_DSEG_TSV = (
    DEFAULT_TRACTOMETRY_ROOT / "data/atlases/4S/atlas-4S156Parcels_dseg.tsv"
)
HCP1065_ENDPOINT_NII_BIN_DIR = (
    DEFAULT_TRACTOMETRY_ROOT / "data/atlases/HCP1065/endpoint_nii_bin"
)

AtlasKind = Literal["4s156", "glasser", "hcp1065"]
# Lowest → highest priority for overlapping voxels.
_ATLAS_PAINT_ORDER: tuple[AtlasKind, ...] = ("4s156", "glasser", "hcp1065")


def _region_lookup_keys(region: str) -> list[str]:
    keys = [region]
    if region.startswith("LH_"):
        keys.append("LH-" + region[3:])
    elif region.startswith("LH-"):
        keys.append("LH_" + region[3:])
    elif region.startswith("RH_"):
        keys.append("RH-" + region[3:])
    elif region.startswith("RH-"):
        keys.append("RH_" + region[3:])
    return keys


def _load_label_to_index(tsv_path: Path) -> dict[str, int]:
    df = pd.read_csv(tsv_path, sep="\t")
    return {
        str(lab): int(idx)
        for lab, idx in zip(df["label"].astype(str), df["index"].astype(int))
    }


def _is_tract_third_label(region: str) -> bool:
    return region.endswith("_core") or "_end-" in region


def _load_roi_scores(
    mean_z_csv: Path,
    *,
    z_col: str,
    min_z: float | None,
) -> dict[str, float]:
    df = pd.read_csv(mean_z_csv)
    if "region" not in df.columns or z_col not in df.columns:
        raise ValueError(f"Expected columns 'region' and '{z_col}' in {mean_z_csv}")
    scores: dict[str, float] = {}
    for region, z in zip(
        df["region"].astype(str),
        pd.to_numeric(df[z_col], errors="coerce"),
    ):
        if not np.isfinite(z):
            continue
        if min_z is not None and float(z) <= min_z:
            continue
        scores[str(region)] = float(z)
    if not scores:
        raise ValueError(
            f"No ROIs selected from {mean_z_csv} "
            f"(z_col={z_col!r}, min_z={min_z!r})"
        )
    return scores


def _classify_rois(
    scores: dict[str, float],
    glasser_lab2idx: dict[str, int],
    four_s_lab2idx: dict[str, int],
) -> tuple[dict[AtlasKind, dict[str, float]], list[str]]:
    """Assign each ROI to one atlas; HCP thirds preferred over Glasser/4S names."""
    by_atlas: dict[AtlasKind, dict[str, float]] = {
        "4s156": {},
        "glasser": {},
        "hcp1065": {},
    }
    unmatched: list[str] = []
    for region, z in scores.items():
        if _is_tract_third_label(region):
            mask_path = HCP1065_ENDPOINT_NII_BIN_DIR / f"{region}.nii.gz"
            if mask_path.is_file():
                by_atlas["hcp1065"][region] = z
                continue
            unmatched.append(region)
            continue

        placed = False
        for key in _region_lookup_keys(region):
            if key in glasser_lab2idx:
                by_atlas["glasser"][region] = z
                placed = True
                break
            if key in four_s_lab2idx:
                by_atlas["4s156"][region] = z
                placed = True
                break
        if not placed:
            unmatched.append(region)
    return by_atlas, unmatched


def write_mean_z_nii(
    mean_z_csv: Path,
    out_nii: Path,
    *,
    z_col: str = "mean_z",
    min_z: float | None = 0.0,
    binary: bool = False,
) -> tuple[Path, dict[str, int]]:
    """Paint ROI mean z onto the shared MNI grid.

    Overlap priority (highest wins): HCP1065 tract-thirds > Glasser > 4S156.
    When ``binary`` is True, painted voxels are 1; otherwise they receive ``mean_z``.
    ROIs with ``z <= min_z`` are dropped when ``min_z`` is not None.
    """
    import nibabel as nib

    scores = _load_roi_scores(mean_z_csv, z_col=z_col, min_z=min_z)

    for p in (GLASSER_DSEG_NII, GLASSER_DSEG_TSV, ATLAS_4S156_DSEG_NII, ATLAS_4S156_DSEG_TSV):
        if not p.is_file():
            raise FileNotFoundError(p)
    if not HCP1065_ENDPOINT_NII_BIN_DIR.is_dir():
        raise FileNotFoundError(HCP1065_ENDPOINT_NII_BIN_DIR)

    ref_img = nib.load(str(GLASSER_DSEG_NII))
    dtype = np.uint8 if binary else np.float32
    out = np.zeros(ref_img.shape, dtype=dtype)

    glasser = np.asarray(ref_img.get_fdata())
    glasser_lab2idx = _load_label_to_index(GLASSER_DSEG_TSV)
    four_s_img = nib.load(str(ATLAS_4S156_DSEG_NII))
    four_s = np.asarray(four_s_img.get_fdata())
    if four_s.shape != out.shape:
        raise ValueError(
            f"4S156 shape {four_s.shape} != Glasser/MNI grid {out.shape}; "
            "expected native (non-resliced) 4S dseg."
        )
    four_s_lab2idx = _load_label_to_index(ATLAS_4S156_DSEG_TSV)

    by_atlas, unmatched = _classify_rois(scores, glasser_lab2idx, four_s_lab2idx)
    counts = {
        "n_rois": len(scores),
        "n_glasser": len(by_atlas["glasser"]),
        "n_4s156": len(by_atlas["4s156"]),
        "n_tract_thirds": len(by_atlas["hcp1065"]),
        "n_unmatched": len(unmatched),
    }
    if unmatched:
        print(
            f"Warning: {len(unmatched)} ROIs unmatched to atlases "
            f"(first few: {unmatched[:8]})"
        )

    # Paint low → high priority so HCP1065 overwrites Glasser overwrites 4S156.
    for atlas in _ATLAS_PAINT_ORDER:
        for region, z in by_atlas[atlas].items():
            fill = np.dtype(dtype).type(1 if binary else z)
            if atlas == "hcp1065":
                mask_img = nib.load(str(HCP1065_ENDPOINT_NII_BIN_DIR / f"{region}.nii.gz"))
                mask = np.asarray(mask_img.get_fdata()) > 0
                if mask.shape != out.shape:
                    raise ValueError(
                        f"Tract mask shape {mask.shape} != reference {out.shape} "
                        f"for {region}"
                    )
                out[mask] = fill
            elif atlas == "glasser":
                for key in _region_lookup_keys(region):
                    if key in glasser_lab2idx:
                        out[glasser == glasser_lab2idx[key]] = fill
                        break
            else:
                for key in _region_lookup_keys(region):
                    if key in four_s_lab2idx:
                        out[four_s == four_s_lab2idx[key]] = fill
                        break

    out_nii.parent.mkdir(parents=True, exist_ok=True)
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(out, ref_img.affine, hdr), str(out_nii))
    counts["n_voxels"] = int(np.count_nonzero(out))
    return out_nii, counts


def write_positive_mean_z_binary_nii(
    mean_z_csv: Path,
    out_nii: Path,
    *,
    z_col: str = "mean_z",
    threshold: float = 0.0,
) -> tuple[Path, dict[str, int]]:
    """Binary MNI map: voxels in ROIs with ``z_col > threshold`` are 1."""
    return write_mean_z_nii(
        mean_z_csv,
        out_nii,
        z_col=z_col,
        min_z=threshold,
        binary=True,
    )


def write_positive_mean_z_continuous_nii(
    mean_z_csv: Path,
    out_nii: Path,
    *,
    z_col: str = "mean_z",
    threshold: float = 0.0,
) -> tuple[Path, dict[str, int]]:
    """Continuous MNI map: ROI voxels get ``mean_z`` where ``z_col > threshold``."""
    return write_mean_z_nii(
        mean_z_csv,
        out_nii,
        z_col=z_col,
        min_z=threshold,
        binary=False,
    )


def default_mean_z_csv(output_dir: Path, factor_tag: str = "F2") -> Path:
    return (
        output_dir
        / METHOD_TAG
        / "csv"
        / f"gradients-{GRADIENTS_K}"
        / f"epilepsy_{factor_tag}_mean_z_scores.csv"
    )


def default_nii_out_dir(output_dir: Path) -> Path:
    return output_dir / METHOD_TAG / "nii"
