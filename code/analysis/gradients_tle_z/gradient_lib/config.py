import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
"""Paths and constants for the TLE z-score gradient scatter pipeline."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_TRACTOMETRY_ROOT = Path(
    os.environ.get(
        "STRUCTURAL_TRACTOMETRY_ROOT",
        str(project_root()),
    )
)

DEFAULT_GRADIENTS_CONTROLS_DIR = Path(
    os.environ.get(
        "GRADIENTS_CONTROLS_OUTPUT_DIR",
        str(DEFAULT_TRACTOMETRY_ROOT / "derivatives/analysis/gradients_group-controls"),
    )
)

DEFAULT_Z_SCORES_DIR = Path(
    os.environ.get(
        "FACTOR_Z_SCORES_DIR",
        str(
            DEFAULT_TRACTOMETRY_ROOT
            / "derivatives/analysis/factor_z-scores/factor_z_scores"
        ),
    )
)

DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "GRADIENTS_TLE_Z_OUTPUT_DIR",
        str(DEFAULT_TRACTOMETRY_ROOT / "derivatives/analysis/gradients_tle_z"),
    )
)

METHOD_TAG = "laplacian_eigenmodes"
GRADIENTS_K = 2
COHORT_TAG = "epilepsy"

from lib.factor_labels import FACTOR_DIFFUSIVITY_LABELS as FACTOR_PANEL_LABELS  # noqa: E402

NON_REGION_COLS = frozenset({"subject", "group"})
