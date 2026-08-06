import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
"""Paths and constants for factor-score neuromaps screening."""

from __future__ import annotations

from pathlib import Path

_CODE_ANALYSIS_DIR = Path(__file__).resolve().parents[2]
_GRADIENTS_DIR = Path(__file__).resolve().parents[1]  # factor_analysis_voxelwise (vendored gradient_lib)

PROJECT_ROOT = project_root()
OUTPUT_PROJECT_ROOT = PROJECT_ROOT / "derivatives/analysis/factor_analysis_voxelwise"
WITH_CSF_DIR = OUTPUT_PROJECT_ROOT / "Voxelwise_ReducedControls/all/with_csf"
FILE_PREFIX = "controls_Voxelwise_ReducedControls_all_with_csf"

DEFAULT_MAP_VARIANT = "norm-0-1"
DEFAULT_INPUT_DIR = WITH_CSF_DIR / "factor_nii" / "loadings-regionwise"
DEFAULT_MASK_NII = WITH_CSF_DIR / f"{FILE_PREFIX}_mni_t1w_mask_used.nii.gz"

NEUROMAPS_SPACE_CHOICES = ("mni", "fslr", "both")
DEFAULT_NEUROMAPS_SPACES = "both"


def parse_neuromaps_spaces(arg: str) -> tuple[str, ...]:
    """Parse --spaces into canonical neuromaps space keys (MNI152, fsLR)."""
    from gradient_lib.config import NEUROMAPS_SPACE_FSLR, NEUROMAPS_SPACE_MNI

    raw = str(arg).strip().lower()
    if raw in ("both", "all"):
        return (NEUROMAPS_SPACE_MNI, NEUROMAPS_SPACE_FSLR)
    if raw in ("mni", "mni152", "volume"):
        return (NEUROMAPS_SPACE_MNI,)
    if raw in ("fslr", "surface", "fsaverage"):
        return (NEUROMAPS_SPACE_FSLR,)

    tokens = [t.strip().lower() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise ValueError(f"Empty --spaces value {arg!r}")

    out: list[str] = []
    for token in tokens:
        if token in ("mni", "mni152", "volume"):
            key = NEUROMAPS_SPACE_MNI
        elif token in ("fslr", "surface", "fsaverage"):
            key = NEUROMAPS_SPACE_FSLR
        else:
            raise ValueError(
                f"Unknown neuromaps space {token!r}; expected mni, fslr, or both"
            )
        if key not in out:
            out.append(key)
    return tuple(out)


def neuromaps_space_slug(space: str) -> str:
    """Filename slug for a neuromaps space key."""
    if space == "MNI152":
        return "MNI152"
    if space == "fsLR":
        return "fsLR"
    return str(space).replace(" ", "_")
