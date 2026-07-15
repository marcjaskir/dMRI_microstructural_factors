"""Controls-only, group-level BrainSpace gradient pipeline (DM + Laplacian)."""

from .config import (
    AFFINITY_SPARSITY_MODE,
    ALPHA_DEFAULT,
    DEFAULT_FACTOR_SCORES_DIR,
    DEFAULT_GRADIENTS_DIR,
    DEFAULT_TRACTOMETRY_ROOT,
    GRADIENT_SUBSPACE_CHOICES,
    N_GRADIENTS_TO_COMPUTE,
    SPARSITY_BY_MODE,
    diffusion_embedding_dirs,
    laplacian_eigenmodes_dirs,
)
from .run_diffusion import (
    compute_diffusion_embedding_row,
    save_diffusion_gradient_outputs,
)
from .run_laplacian import (
    compute_laplacian_row,
    save_laplacian_gradient_outputs,
)
from .types import GradientRunRow

__all__ = [
    "AFFINITY_SPARSITY_MODE",
    "ALPHA_DEFAULT",
    "DEFAULT_FACTOR_SCORES_DIR",
    "DEFAULT_GRADIENTS_DIR",
    "DEFAULT_TRACTOMETRY_ROOT",
    "GRADIENT_SUBSPACE_CHOICES",
    "GradientRunRow",
    "N_GRADIENTS_TO_COMPUTE",
    "SPARSITY_BY_MODE",
    "compute_diffusion_embedding_row",
    "compute_laplacian_row",
    "diffusion_embedding_dirs",
    "laplacian_eigenmodes_dirs",
    "save_diffusion_gradient_outputs",
    "save_laplacian_gradient_outputs",
]
