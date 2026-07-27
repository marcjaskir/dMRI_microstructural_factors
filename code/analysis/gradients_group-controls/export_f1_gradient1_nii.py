#!/usr/bin/env python3
"""Export F1 G1 scores as a thresholded 4S156 subcortical NIfTI (gradients-2/nii)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gradient_lib.config import DEFAULT_GRADIENTS_DIR, DEFAULT_TRACTOMETRY_ROOT
from gradient_lib.nii_maps import write_subcortex_gradient_nii


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write F1 principal gradient 1 subcortical NIfTI (4S156)."
    )
    parser.add_argument(
        "--tractometry-root",
        type=Path,
        default=DEFAULT_TRACTOMETRY_ROOT,
    )
    parser.add_argument(
        "--gradients-root",
        type=Path,
        default=DEFAULT_GRADIENTS_DIR,
    )
    parser.add_argument(
        "--scores-csv",
        type=Path,
        default=None,
        help="Defaults to laplacian_eigenmodes/.../F1_principal_gradient1_scores_cohort-controls.csv",
    )
    parser.add_argument(
        "--out-nii",
        type=Path,
        default=None,
    )
    parser.add_argument("--threshold", type=float, default=0.005)
    args = parser.parse_args()

    scores_csv = args.scores_csv
    if scores_csv is None:
        scores_csv = (
            args.gradients_root
            / "laplacian_eigenmodes"
            / "csv"
            / "gradients-2"
            / "F1_principal_gradient1_scores_cohort-controls.csv"
        )
    out_nii = args.out_nii
    if out_nii is None:
        thr_slug = f"{args.threshold:g}".replace(".", "p")
        out_nii = (
            args.gradients_root
            / "laplacian_eigenmodes"
            / "csv"
            / "gradients-2"
            / "nii"
            / f"F1_principal_gradient1_scores_cohort-controls_subcortex_thr-{thr_slug}.nii.gz"
        )

    path, n_painted = write_subcortex_gradient_nii(
        scores_csv,
        out_nii,
        tractometry_root=args.tractometry_root,
        threshold=args.threshold,
    )
    print(f"Wrote {path} ({n_painted} subcortical parcels >= {args.threshold})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
