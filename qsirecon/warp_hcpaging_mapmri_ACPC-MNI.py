#!/usr/bin/env python3
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
Warp hcpaging MAPMRI scalars from ACPC to MNI152NLin2009cAsym using qsiprep transforms.

Reads ACPC-space dwimaps from:
  derivatives/qsirecon/hcpaging/derivatives/qsirecon-TORTOISE_model-MAPMRI/{sub}/dwi/
  e.g. {sub}_space-ACPC_model-mapmri_param-ng_dwimap.nii.gz

Uses subject-specific transform from qsiprep:
  derivatives/qsiprep/hcpaging/{sub}/anat/{sub}_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5

Writes MNI-space outputs to the same directory with space-MNI152NLin2009cAsym in the filename.

Uses each subject's MNI-space MD image (1.25 mm isotropic) as the warp reference grid, to match other MNI scalars:
  derivatives/qsirecon/hcpaging/derivatives/qsirecon-DSIStudio/{sub}/dwi/{sub}_space-MNI152NLin2009cAsym_model-tensor_param-md_dwimap.nii.gz
"""

import argparse
import glob
import re
import subprocess
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

_QSIRECON_MAPMRI = _PROJECT_ROOT / "derivatives" / "qsirecon" / "hcpaging" / "derivatives" / "qsirecon-TORTOISE_model-MAPMRI"
_QSIRECON_DSISTUDIO = _PROJECT_ROOT / "derivatives" / "qsirecon" / "hcpaging" / "derivatives" / "qsirecon-DSIStudio"
_QSIPREP_HCPAGING = _PROJECT_ROOT / "derivatives" / "qsiprep" / "hcpaging"

SPACE_ACPC = "space-ACPC"
SPACE_MNI = "space-MNI152NLin2009cAsym"


def _subject_md_mni_path(project_root, sub, md_dir=None):
    """Path to subject's MNI-space MD image (1.25 mm) for warp reference grid, or None if missing."""
    root = Path(project_root)
    base = md_dir if md_dir is not None else (root / "derivatives" / "qsirecon" / "hcpaging" / "derivatives" / "qsirecon-DSIStudio")
    p = base / sub / "dwi" / f"{sub}_space-MNI152NLin2009cAsym_model-tensor_param-md_dwimap.nii.gz"
    return str(p) if p.exists() else None


def _discover_acpc_mapmri(mapmri_dir):
    """
    Yield (sub, input_nii_path) for each ACPC-space MAPMRI dwimap.
    input_nii_path is under mapmri_dir/{sub}/dwi/*_space-ACPC_*mapmri*_dwimap.nii.gz
    """
    mapmri_dir = Path(mapmri_dir)
    if not mapmri_dir.is_dir():
        return
    for sub_dir in sorted(mapmri_dir.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        dwi_dir = sub_dir / "dwi"
        if not dwi_dir.is_dir():
            continue
        pattern = str(dwi_dir / "*_space-ACPC_*mapmri*_dwimap.nii.gz")
        for path in sorted(glob.glob(pattern)):
            yield sub_dir.name, Path(path)


def _xfm_path(qsiprep_dir, sub):
    """Path to ACPC→MNI transform."""
    return Path(qsiprep_dir) / sub / "anat" / f"{sub}_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5"


def _output_path(input_path, out_dir=None):
    """Replace space-ACPC with space-MNI152NLin2009cAsym in filename; same parent by default."""
    p = Path(input_path)
    out_name = re.sub(re.escape(SPACE_ACPC), SPACE_MNI, p.name, count=1)
    if out_dir is not None:
        return Path(out_dir) / p.parent.name / "dwi" / out_name
    return p.parent / out_name


def _run_ants_apply_transforms(moving, reference, transform, output, interpolation="Linear"):
    """Run antsApplyTransforms: moving (ACPC) image into reference (MNI) space using ACPC→MNI transform."""
    cmd = [
        "antsApplyTransforms",
        "-d", "3",
        "-i", str(moving),
        "-r", str(reference),
        "-o", str(output),
        "-t", str(transform),
        "-n", interpolation,
    ]
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(
        description="Warp hcpaging MAPMRI ACPC-space scalars to MNI152NLin2009cAsym using qsiprep transforms.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_PROJECT_ROOT,
        help="Project root (default: parent of code/).",
    )
    parser.add_argument(
        "--mapmri-dir",
        type=Path,
        default=None,
        help=f"Override MAPMRI directory (default: {{project_root}}/derivatives/qsirecon/hcpaging/derivatives/qsirecon-TORTOISE_model-MAPMRI).",
    )
    parser.add_argument(
        "--qsiprep-dir",
        type=Path,
        default=None,
        help=f"Override qsiprep hcpaging directory (default: {{project_root}}/derivatives/qsiprep/hcpaging).",
    )
    parser.add_argument(
        "--mni-ref",
        type=Path,
        default=None,
        help="Override MNI reference image (used for all subjects). Default: each subject's MNI-space MD image (1.25 mm) from qsirecon-DSIStudio.",
    )
    parser.add_argument(
        "--md-dir",
        type=Path,
        default=None,
        help="Override directory containing subject dwi/ with MNI MD images (default: .../qsirecon/hcpaging/derivatives/qsirecon-DSIStudio).",
    )
    parser.add_argument(
        "--subject",
        action="append",
        dest="subjects",
        metavar="SUB",
        help="Process only these subjects (e.g. sub-HCA6002236). Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running antsApplyTransforms.",
    )
    parser.add_argument(
        "--interpolation",
        default="Linear",
        choices=("Linear", "NearestNeighbor", "BSpline"),
        help="Interpolation for resampling (default: Linear).",
    )
    args = parser.parse_args()

    root = args.project_root
    mapmri_dir = args.mapmri_dir or (root / "derivatives" / "qsirecon" / "hcpaging" / "derivatives" / "qsirecon-TORTOISE_model-MAPMRI")
    qsiprep_dir = args.qsiprep_dir or (root / "derivatives" / "qsiprep" / "hcpaging")
    md_dir = args.md_dir or (root / "derivatives" / "qsirecon" / "hcpaging" / "derivatives" / "qsirecon-DSIStudio")
    global_mni_ref = args.mni_ref
    if global_mni_ref is not None and not Path(global_mni_ref).exists():
        print("Error: --mni-ref image not found:", global_mni_ref, file=sys.stderr)
        sys.exit(1)

    if not mapmri_dir.is_dir():
        print("Error: MAPMRI directory not found:", mapmri_dir, file=sys.stderr)
        sys.exit(1)

    subjects_filter = set(args.subjects) if args.subjects else None
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for sub, input_path in _discover_acpc_mapmri(mapmri_dir):
        if subjects_filter is not None and sub not in subjects_filter:
            continue
        xfm = _xfm_path(qsiprep_dir, sub)
        if not xfm.exists():
            print("Skip (no transform):", sub, xfm, file=sys.stderr)
            n_skip += 1
            continue
        if global_mni_ref is not None:
            mni_ref = str(global_mni_ref)
        else:
            mni_ref = _subject_md_mni_path(root, sub, md_dir)
            if not mni_ref:
                print("Skip (no MNI MD reference):", sub, file=sys.stderr)
                n_skip += 1
                continue
        out_path = _output_path(input_path)
        if out_path.exists() and out_path.stat().st_mtime >= input_path.stat().st_mtime:
            print("Skip (up-to-date):", out_path)
            n_skip += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            print("Would run: antsApplyTransforms -i", input_path, "-r", mni_ref, "-t", xfm, "-o", out_path)
            n_ok += 1
            continue
        try:
            _run_ants_apply_transforms(
                str(input_path),
                mni_ref,
                str(xfm),
                str(out_path),
                interpolation=args.interpolation,
            )
            print("Wrote:", out_path)
            n_ok += 1
        except subprocess.CalledProcessError as e:
            print("Error:", input_path, e.stderr or e, file=sys.stderr)
            n_fail += 1

    print("Done: %d written, %d skipped, %d failed." % (n_ok, n_skip, n_fail))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
