#!/usr/bin/env python3
import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
"""
Project group-mean voxelwise scalar maps onto regionwise factor loadings.

1. Z-score each of the 26 group-mean scalar maps across in-mask voxels.
2. Compute the voxelwise dot product of the z-scored scalars with each factor
   loading vector from All4_Combined.
3. Save F1–F3 factor-score NIfTIs under factor_nii/loadings-regionwise/.
4. Also save min-max normalized versions scaled to [0, 1] across in-mask voxels.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

_CODE_ANALYSIS_DIR = Path(__file__).resolve().parents[1]

PROJECT_ROOT = project_root()
OUTPUT_PROJECT_ROOT = PROJECT_ROOT / "derivatives/analysis/factor_analysis_voxelwise"
WITH_CSF_DIR = OUTPUT_PROJECT_ROOT / "Voxelwise_ReducedControls/all/with_csf"
FILE_PREFIX = "controls_Voxelwise_ReducedControls_all_with_csf"

DEFAULT_SCALAR_DIR = WITH_CSF_DIR / "scalar_nii"
DEFAULT_MASK_NII = WITH_CSF_DIR / f"{FILE_PREFIX}_mni_t1w_mask_used.nii.gz"
DEFAULT_LOADINGS_CSV = (
    PROJECT_ROOT
    / "derivatives/analysis/factor_analysis"
    / "controls_All4_Combined_scalar_factor_loadings.csv"
)
DEFAULT_OUTPUT_DIR = WITH_CSF_DIR / "factor_nii" / "loadings-regionwise"
NORM_MAP_VARIANT = "norm-0-1"

_GRADIENTS_DIR = Path(__file__).resolve().parent  # vendored gradient_lib
if str(_GRADIENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_GRADIENTS_DIR))

from gradient_lib.factor_scores import (  # noqa: E402
    compute_factor_scores_from_z,
    zscore_scalars_per_subject,
)
from gradient_lib.io_voxelwise import (  # noqa: E402
    load_analysis_mask,
    load_factor_loadings,
    load_masked_scalar,
    save_masked_vector_nii,
    scalar_labels_from_loadings,
)


def _parse_factors(arg: str, loadings_index: list[str]) -> list[str]:
    out: list[str] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        tag = part if part.upper().startswith("F") else f"F{part}"
        if tag not in loadings_index:
            raise SystemExit(f"Unknown factor {tag!r}; available: {loadings_index}")
        if tag not in out:
            out.append(tag)
    return out


def _scalar_mean_path(scalar_dir: Path, scalar: str) -> Path:
    return scalar_dir / f"{FILE_PREFIX}_scalar-{scalar}_mean.nii.gz"


def _validate_scalar_niis(scalar_dir: Path, scalar_labels: list[str]) -> None:
    missing = [s for s in scalar_labels if not _scalar_mean_path(scalar_dir, s).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} scalar mean NIfTI(s) under {scalar_dir}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )


def _load_scalar_matrix(
    scalar_dir: Path,
    scalar_labels: list[str],
    mask_img,
    mask: np.ndarray,
) -> np.ndarray:
    columns = [
        load_masked_scalar(str(_scalar_mean_path(scalar_dir, scalar)), mask_img, mask)
        for scalar in scalar_labels
    ]
    return np.column_stack(columns)


def _normalize_to_range(values: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Min-max scale finite values to [low, high]; non-finite values become 0."""
    out = np.zeros_like(values, dtype=np.float32)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return out
    vals = values[finite].astype(np.float64, copy=False)
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax <= vmin:
        return out
    scaled = low + (values.astype(np.float64, copy=False) - vmin) * (high - low) / (vmax - vmin)
    out[finite] = scaled[finite].astype(np.float32, copy=False)
    return out


def _write_loadings_source_metadata(
    output_dir: Path,
    *,
    loadings_csv: Path,
    scalar_dir: Path,
    mask_nii: Path,
    factors: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "loadings_source.json"
    payload = {
        "source": "regionwise",
        "run_label": "loadings-regionwise",
        "loadings_csv": str(loadings_csv.resolve()),
        "scalar_dir": str(scalar_dir.resolve()),
        "mask_nii": str(mask_nii.resolve()),
        "output_dir": str(output_dir.resolve()),
        "factors": factors,
        "file_prefix": FILE_PREFIX,
        "normalized_suffix": f"_factor-score_{NORM_MAP_VARIANT}",
        "normalization": "min-max to [0, 1] over in-mask finite voxels per factor map",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Group-mean voxelwise factor scores from regionwise loadings."
    )
    parser.add_argument("--scalar-dir", type=Path, default=DEFAULT_SCALAR_DIR)
    parser.add_argument("--loadings-csv", type=Path, default=DEFAULT_LOADINGS_CSV)
    parser.add_argument("--mask-nii", type=Path, default=DEFAULT_MASK_NII)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--factors",
        type=str,
        default="F1,F2,F3",
        help="Comma-separated factor tags (default: F1,F2,F3).",
    )
    args = parser.parse_args()

    loadings = load_factor_loadings(args.loadings_csv)
    scalar_labels = scalar_labels_from_loadings(loadings)
    factors = _parse_factors(args.factors, list(loadings.index))

    _validate_scalar_niis(args.scalar_dir, scalar_labels)

    mask_img, mask = load_analysis_mask(args.mask_nii)
    scalar_matrix = _load_scalar_matrix(args.scalar_dir, scalar_labels, mask_img, mask)
    z_matrix = zscore_scalars_per_subject(scalar_matrix)
    scores = compute_factor_scores_from_z(z_matrix, loadings, scalar_labels, factors)

    meta_path = _write_loadings_source_metadata(
        args.output_dir,
        loadings_csv=args.loadings_csv,
        scalar_dir=args.scalar_dir,
        mask_nii=args.mask_nii,
        factors=factors,
    )
    print(f"Wrote {meta_path}")

    for factor in tqdm(factors, desc="Saving factor-score NIfTIs"):
        values = scores[factor]
        out_path = args.output_dir / f"{FILE_PREFIX}_{factor}_factor-score.nii.gz"
        save_masked_vector_nii(values, mask_img, mask, out_path)
        print(f"  {out_path}")

        norm_values = _normalize_to_range(values)
        norm_path = args.output_dir / f"{FILE_PREFIX}_{factor}_factor-score_{NORM_MAP_VARIANT}.nii.gz"
        save_masked_vector_nii(norm_values, mask_img, mask, norm_path)
        print(f"  {norm_path}")

    print(
        f"Done. {len(factors)} raw and {len(factors)} normalized factor-score map(s) "
        f"under {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
