"""Paths and constants for the factor score / normative z CSV pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from lib.paths import (  # noqa: E402
    analysis_dir,
    atlas_dir,
    gam_dir,
    inclusion_dir,
    open_metadata_dir,
    project_root,
)

PROJECT_ROOT = project_root()
METADATA_DIR = str(open_metadata_dir())

MNI_MICRO_PROJECT_ROOT = f"{gam_dir()}/mni_micro"
GM_GLASSER_PROFILE_DIR = f"{MNI_MICRO_PROJECT_ROOT}/Glasser"
GM_4S156_PROFILE_DIR = f"{MNI_MICRO_PROJECT_ROOT}/4S156"
WM_PROFILE_DIR_PYAFQ = f"{gam_dir()}/pyafq/HCP1065"
FOUR_S156_DSEG_PATH = str(atlas_dir() / "4S" / "atlas-4S156Parcels_dseg.tsv")
HCP1065_TRACT_METADATA_PATH = str(atlas_dir() / "HCP1065" / "HCP1065_tract_metadata.csv")

OUTPUT_PROJECT_ROOT = f"{analysis_dir()}/factor_z-scores"
FACTOR_SCORES_DIR = f"{OUTPUT_PROJECT_ROOT}/factor_scores"
FACTOR_Z_SCORES_DIR = f"{OUTPUT_PROJECT_ROOT}/factor_z_scores"
SCALAR_Z_SCORES_OUTPUT_DIR = f"{OUTPUT_PROJECT_ROOT}/scalar_z-scores"

INCLUSION_METADATA_PATH = str(inclusion_dir() / "penn_epilepsy_included_basic_metadata.csv")
FACTOR_LOADINGS_PATH = (
    f"{analysis_dir()}/factor_analysis/All4_Combined/"
    "controls_All4_Combined_scalar_factor_loadings.csv"
)

N_NODES = 100
END1_NODES = list(range(1, 35))
CORE_NODES = list(range(35, 67))
END2_NODES = list(range(67, 101))

CONTROL_GROUPS = ["penn_controls", "hcpya", "hcpaging"]
PATIENT_GROUPS = ["penn_epilepsy"]

SCALAR_PREFIX_ORDER = ["dti", "rdi", "dki", "gqi", "noddi", "map"]

# Minimum control subjects with valid raw factor scores for per-ROI mean/SD
MIN_CONTROLS_FOR_ROI_Z = 2
