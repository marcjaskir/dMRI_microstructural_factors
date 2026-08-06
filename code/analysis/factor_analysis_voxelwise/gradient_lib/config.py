"""Paths and constants used by group-mean voxelwise factor-score helpers."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root  # noqa: E402

PROJECT_ROOT = project_root()

DEFAULT_OUTLIER_IQR_MULTIPLIER = 1.5
MIN_FINITE_VOXELS_FOR_IQR = 4
PROBSEG_THRESHOLD = 0.75

DEFAULT_TRACTOMETRY_ROOT = Path(
    os.environ.get(
        "STRUCTURAL_TRACTOMETRY_ROOT",
        str(PROJECT_ROOT),
    )
)

VOXELWISE_FA_ALL_DIR = DEFAULT_TRACTOMETRY_ROOT / (
    "derivatives/analysis/factor_analysis_voxelwise/Voxelwise_ReducedControls/all"
)
VOXELWISE_FA_ALL_WITH_CSF_DIR = VOXELWISE_FA_ALL_DIR / "with_csf"
VOXELWISE_FA_FACTORS_SUBDIR = "factors-3"
VOXELWISE_FA_DEFAULT_DIR = VOXELWISE_FA_ALL_WITH_CSF_DIR / VOXELWISE_FA_FACTORS_SUBDIR

DEFAULT_LOADINGS_CSV = (
    VOXELWISE_FA_DEFAULT_DIR
    / "controls_Voxelwise_ReducedControls_all_with_csf_scalar_factor_loadings.csv"
)
DEFAULT_MANIFEST_CSV = (
    VOXELWISE_FA_ALL_WITH_CSF_DIR
    / "controls_Voxelwise_ReducedControls_all_with_csf_scalar_image_manifest.csv"
)
DEFAULT_MASK_NII = (
    VOXELWISE_FA_ALL_WITH_CSF_DIR
    / "controls_Voxelwise_ReducedControls_all_with_csf_mni_t1w_mask_used.nii.gz"
)

MNI_ATLAS_DIR = DEFAULT_TRACTOMETRY_ROOT / "data/atlases/MNI"
PROBSEG_PATHS: dict[str, Path] = {
    "GM": MNI_ATLAS_DIR / "tpl-MNI152NLin2009cAsym_res-1mm_label-gm_probseg.nii.gz",
    "WM": MNI_ATLAS_DIR / "tpl-MNI152NLin2009cAsym_res-1mm_label-wm_probseg.nii.gz",
    "CSF": MNI_ATLAS_DIR / "tpl-MNI152NLin2009cAsym_res-1mm_label-csf_probseg.nii.gz",
}
