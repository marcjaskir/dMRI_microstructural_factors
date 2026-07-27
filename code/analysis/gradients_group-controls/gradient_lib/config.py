import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
"""Paths and constants for the controls-only group-level gradient pipeline."""

from __future__ import annotations

import os
from pathlib import Path

NON_REGION_COLS = frozenset({"subject", "group"})

N_GRADIENTS_TO_COMPUTE = 10
# Subspace sizes for which we produce ``gradients-{K}/`` subdirectories. ``K`` controls
# both the number of gradient CSVs saved (G1..GK) and the dimensionality of scatter figures
# (2D for K==2, 3D for K>=3).
GRADIENT_SUBSPACE_CHOICES: tuple[int, ...] = (2, 3)

AFFINITY_SPARSITY_MODE: str = "full"
SPARSITY_BY_MODE: dict[str, float] = {"full": 0.0}

ALPHA_DEFAULT: float = 0.5

FOUR_S_SUBCORTICAL_ATLAS_NAMES = frozenset(
    {"ThalamusHCP", "SubcorticalHCP", "Cerebellum", "CIT168Subcortical"}
)

HCP1065_TRACT_METADATA_REL = Path("data/atlases/HCP1065/HCP1065_tract_metadata.csv")

# By-tissue scatter (matches the original factor_gradients styling).
TISSUE_CORTICAL_GM = "#5c5c5c"
TISSUE_POINT_EDGEWIDTH = 0.9
TISSUE_SCATTER_POINT_ALPHA = 0.5
TISSUE_CENTROID_FILL_COLOR = (1.0, 1.0, 0.0, 0.2)
TISSUE_CENTROID_EDGE_COLOR = "tab:orange"

GRADIENT_AXIS_LABEL_COLOR = "k"

DEFAULT_TRACTOMETRY_ROOT = Path(
    os.environ.get(
        "STRUCTURAL_TRACTOMETRY_ROOT",
        str(project_root()),
    )
)

DEFAULT_GRADIENTS_DIR = Path(
    os.environ.get(
        "GRADIENTS_CONTROLS_OUTPUT_DIR",
        str(DEFAULT_TRACTOMETRY_ROOT / "derivatives/analysis/gradients_group-controls"),
    )
)

DEFAULT_FACTOR_SCORES_DIR = Path(
    os.environ.get(
        "FACTOR_SCORES_DIR",
        str(
            DEFAULT_TRACTOMETRY_ROOT
            / "derivatives/analysis/factor_z-scores/factor_scores"
        ),
    )
)


def diffusion_embedding_dirs(
    gradients_root: Path | None = None,
) -> tuple[Path, Path]:
    """Return ``(csv_dir, figures_dir)`` under ``<root>/diffusion_embedding``."""
    root = DEFAULT_GRADIENTS_DIR if gradients_root is None else Path(gradients_root)
    d = root / "diffusion_embedding"
    return d / "csv", d / "figures"


def laplacian_eigenmodes_dirs(
    gradients_root: Path | None = None,
) -> tuple[Path, Path]:
    """Return ``(csv_dir, figures_dir)`` under ``<root>/laplacian_eigenmodes``."""
    root = DEFAULT_GRADIENTS_DIR if gradients_root is None else Path(gradients_root)
    d = root / "laplacian_eigenmodes"
    return d / "csv", d / "figures"
