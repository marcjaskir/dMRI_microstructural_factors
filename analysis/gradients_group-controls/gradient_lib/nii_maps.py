"""Map principal-gradient score CSVs onto 4S156 subcortical parcel NIfTIs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FOUR_S_SUBCORTICAL_ATLAS_NAMES

DEFAULT_ATLAS_NII = (
    "data/atlases/4S/tpl-MNI152NLin2009cAsym_atlas-4S156Parcels_res-01_dseg.nii.gz"
)
DEFAULT_ATLAS_TSV = "data/atlases/4S/atlas-4S156Parcels_dseg.tsv"


def _region_lookup_keys(region: str) -> list[str]:
    """Atlas label variants that may appear in wide tables."""
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


def load_region_gradient_scores(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    score_cols = [
        c
        for c in df.columns
        if c.startswith("principal_gradient") and c.endswith("_score")
    ]
    if len(score_cols) != 1:
        raise ValueError(
            f"Expected one principal_gradient*_score column in {csv_path.name}, "
            f"found {score_cols}"
        )
    col = score_cols[0]
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        region = str(row["region"])
        val = float(row[col])
        for key in _region_lookup_keys(region):
            out[key] = val
    return out


def _score_for_atlas_label(label: str, scores: dict[str, float]) -> float | None:
    for key in _region_lookup_keys(label):
        if key in scores:
            return scores[key]
    return scores.get(label)


def write_subcortex_gradient_nii(
    scores_csv: Path,
    out_nii: Path,
    *,
    tractometry_root: Path,
    threshold: float = 0.005,
    atlas_nii_rel: str = DEFAULT_ATLAS_NII,
    atlas_tsv_rel: str = DEFAULT_ATLAS_TSV,
) -> tuple[Path, int]:
    """
  Write a 4S156-space image: subcortical parcels with score >= threshold take that
  score as intensity; all other voxels are 0 (cortex / WM / below-threshold subcortex).
    """
    import nibabel as nib

    atlas_nii = tractometry_root / atlas_nii_rel
    atlas_tsv = tractometry_root / atlas_tsv_rel
    if not atlas_nii.exists():
        raise FileNotFoundError(atlas_nii)
    if not atlas_tsv.exists():
        raise FileNotFoundError(atlas_tsv)

    scores = load_region_gradient_scores(scores_csv)
    df_4s = pd.read_csv(atlas_tsv, sep="\t")
    sub = df_4s[df_4s["atlas_name"].astype(str).isin(FOUR_S_SUBCORTICAL_ATLAS_NAMES)]
    label_to_index = dict(
        zip(df_4s["label"].astype(str), df_4s["index"].astype(int))
    )

    img = nib.load(str(atlas_nii))
    data = np.asarray(img.get_fdata())
    out = np.zeros(data.shape, dtype=np.float32)
    n_painted = 0

    for label in sub["label"].astype(str):
        idx = label_to_index.get(label)
        if idx is None:
            continue
        raw = _score_for_atlas_label(label, scores)
        if raw is None or not np.isfinite(raw) or raw < threshold:
            continue
        mask = data == idx
        if not np.any(mask):
            continue
        out[mask] = np.float32(raw)
        n_painted += 1

    out_nii.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(out, img.affine, img.header)
    nib.save(out_img, str(out_nii))
    return out_nii, n_painted
