import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
Convert HCP1065 tractograms (.trk) into NIfTI tract masks (all_nii, all_nii_bin).

Inputs:
  data/atlases/HCP1065/all_trk/<TRACT>.trk

Outputs:
  data/atlases/HCP1065/all_nii/<TRACT>.nii.gz      (float mask)
  data/atlases/HCP1065/all_nii_bin/<TRACT>.nii.gz  (binary mask)

Mask rasterization uses nibabel to load .trk streamlines and projects streamline
points into the voxel grid defined by the reference NIfTI:
  data/atlases/HCP1065/FSL_HCP1065_FA.nii

Because rasterization can be slow, streamline points are optionally subsampled
via --point-stride (default 2). The binary mask can also be dilated in voxel
space via --dilate-vox (default 1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import numpy as np


def _iter_trk_files(trk_dir: Path, tract_glob: str) -> Iterator[Path]:
    for p in sorted(trk_dir.glob(tract_glob)):
        if p.is_file():
            yield p


def _points_to_voxel_indices(
    points_world: np.ndarray,
    inv_ref_affine: np.ndarray,
    shape_xyz: tuple[int, int, int],
    point_stride: int,
) -> np.ndarray:
    """
    Convert (N, 3) world coordinates to integer voxel indices (K, 3).
    """
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        return np.zeros((0, 3), dtype=int)
    if point_stride > 1:
        points_world = points_world[::point_stride]
        if points_world.size == 0:
            return np.zeros((0, 3), dtype=int)

    xyz = points_world.T  # (3, N)
    vox = inv_ref_affine[:3, :3] @ xyz + inv_ref_affine[:3, 3:4]  # (3, N)
    ijk = np.round(vox).astype(int).T  # (N, 3)

    i = ijk[:, 0]
    j = ijk[:, 1]
    k = ijk[:, 2]
    m = (
        (i >= 0)
        & (i < shape_xyz[0])
        & (j >= 0)
        & (j < shape_xyz[1])
        & (k >= 0)
        & (k < shape_xyz[2])
    )
    return ijk[m]


def _rasterize_trk_to_binary_mask(
    trk_path: Path,
    ref_affine: np.ndarray,
    shape_xyz: tuple[int, int, int],
    point_stride: int,
    dilate_vox: int,
) -> np.ndarray:
    """Return boolean binary mask aligned to the reference voxel grid."""
    import nibabel as nib

    trk = nib.streamlines.load(str(trk_path))
    inv_affine = np.linalg.inv(ref_affine)

    mask = np.zeros(shape_xyz, dtype=bool)
    for sl in trk.streamlines:
        if sl is None:
            continue
        points_world = np.asarray(sl, dtype=float)
        ijk = _points_to_voxel_indices(
            points_world,
            inv_ref_affine=inv_affine,
            shape_xyz=shape_xyz,
            point_stride=point_stride,
        )
        if ijk.size == 0:
            continue
        mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    if dilate_vox > 0:
        try:
            from scipy.ndimage import binary_dilation

            structure = np.ones(
                (2 * dilate_vox + 1, 2 * dilate_vox + 1, 2 * dilate_vox + 1),
                dtype=bool,
            )
            mask = binary_dilation(mask, structure=structure)
        except Exception:
            # If SciPy is unavailable, skip dilation.
            pass

    return mask


def _save_mask_nifti(mask_bool: np.ndarray, ref_img, out_path: Path, dtype: np.dtype) -> None:
    import nibabel as nib

    data = mask_bool.astype(dtype, copy=False)
    img = nib.Nifti1Image(data, ref_img.affine, header=ref_img.header)
    try:
        img.set_data_dtype(dtype)
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Rasterize HCP1065 .trk into NIfTI masks.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="{project_root()}",
    )
    parser.add_argument(
        "--ref-nifti",
        type=str,
        default="data/atlases/HCP1065/FSL_HCP1065_FA.nii",
    )
    parser.add_argument(
        "--trk-dir",
        type=str,
        default="data/atlases/HCP1065/all_trk",
    )
    parser.add_argument(
        "--out-nii-dir",
        type=str,
        default="data/atlases/HCP1065/all_nii",
    )
    parser.add_argument(
        "--out-nii-bin-dir",
        type=str,
        default="data/atlases/HCP1065/all_nii_bin",
    )
    parser.add_argument("--tract-glob", type=str, default="*.trk")
    parser.add_argument("--point-stride", type=int, default=2, help="Subsample points along each streamline.")
    parser.add_argument("--dilate-vox", type=int, default=1, help="Binary dilation radius in voxels.")
    parser.add_argument("--max-tracts", type=int, default=0, help="If >0, process only first N tracts.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    ref_path = base_dir / args.ref_nifti
    trk_dir = base_dir / args.trk_dir
    out_nii_dir = base_dir / args.out_nii_dir
    out_bin_dir = base_dir / args.out_nii_bin_dir

    if not ref_path.exists():
        print(f"Missing reference NIfTI: {ref_path}", file=sys.stderr)
        return 1
    if not trk_dir.exists():
        print(f"Missing trk directory: {trk_dir}", file=sys.stderr)
        return 1

    import nibabel as nib

    ref_img = nib.load(str(ref_path))
    shape_xyz = ref_img.shape[:3]
    ref_affine = np.asarray(ref_img.affine, dtype=float)

    trk_files = list(_iter_trk_files(trk_dir, args.tract_glob))
    if args.max_tracts and args.max_tracts > 0:
        trk_files = trk_files[: args.max_tracts]

    if not trk_files:
        print("No .trk files found.", file=sys.stderr)
        return 1

    for idx, trk_path in enumerate(trk_files):
        stem = trk_path.stem
        print(f"[{idx + 1}/{len(trk_files)}] Rasterizing {stem}", file=sys.stderr)

        mask_bool = _rasterize_trk_to_binary_mask(
            trk_path=trk_path,
            ref_affine=ref_affine,
            shape_xyz=shape_xyz,
            point_stride=args.point_stride,
            dilate_vox=args.dilate_vox,
        )

        _save_mask_nifti(
            mask_bool=mask_bool,
            ref_img=ref_img,
            out_path=out_nii_dir / f"{stem}.nii.gz",
            dtype=np.float32,
        )
        _save_mask_nifti(
            mask_bool=mask_bool,
            ref_img=ref_img,
            out_path=out_bin_dir / f"{stem}.nii.gz",
            dtype=np.uint8,
        )

    print(f"Done. Wrote masks to:\n  - {out_nii_dir}\n  - {out_bin_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

