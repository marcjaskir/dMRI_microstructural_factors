import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""Paths and constants for the voxelwise gradient pipeline."""

from __future__ import annotations

import os
from pathlib import Path

N_GRADIENTS_TO_COMPUTE = 10
N_GRADIENTS_TO_SAVE = 2
COHORT_TAG = "controls"

# Cohorts used for Laplacian gradient fitting (controls only).
GRADIENT_GROUPS: tuple[str, ...] = ("penn_controls", "hcpya", "hcpaging")
# Extra cohorts whose per-subject factor score NIfTIs are saved but excluded from gradients.
FACTOR_SCORE_EXTRA_GROUPS: tuple[str, ...] = ("penn_epilepsy",)

DEFAULT_EMBED_STRIDE = 2
DEFAULT_TOP_K = 200
DEFAULT_MAX_EMBED_VOXELS = 15000
DEFAULT_INTERP_NEIGHBORS = 8

DEFAULT_OUTLIER_IQR_MULTIPLIER = 1.5
MIN_FINITE_VOXELS_FOR_IQR = 4

PROBSEG_THRESHOLD = 0.75
TISSUE_CLASSES: tuple[str, ...] = ("GM", "WM", "CSF")
TISSUE_CLASSES_GMWM: tuple[str, ...] = ("GM", "WM")
TISSUE_COLORS: dict[str, str] = {
    "GM": "#808080",
    "WM": "#ffffff",
    "CSF": "#aec7e8",
}
TISSUE_MARKERS: dict[str, str] = {
    "GM": "o",
    "WM": "s",
    "CSF": "^",
}

TISSUE_POINT_EDGEWIDTH = 0.9
TISSUE_SCATTER_POINT_SIZE = 3.0
TISSUE_SCATTER_POINT_ALPHA = 0.35
TISSUE_CENTROID_FILL_COLOR = (1.0, 1.0, 0.0, 0.2)
TISSUE_CENTROID_EDGE_COLOR = "tab:orange"
GRADIENT_AXIS_LABEL_COLOR = "k"

SCATTER_MAX_POINTS = 50000

DEFAULT_TRACTOMETRY_ROOT = Path(
    os.environ.get(
        "STRUCTURAL_TRACTOMETRY_ROOT",
        "{project_root()}",
    )
)

DEFAULT_PENN_EPILEPSY_INCLUSION_CSV = (
    DEFAULT_TRACTOMETRY_ROOT
    / "results/inclusion/penn_epilepsy_included_basic_metadata.csv"
)

VOXELWISE_FA_ALL_DIR = DEFAULT_TRACTOMETRY_ROOT / (
    "derivatives/analysis/factor_analysis_voxelwise/Voxelwise_ReducedControls/all"
)
VOXELWISE_FA_ALL_WITH_CSF_DIR = VOXELWISE_FA_ALL_DIR / "with_csf"
VOXELWISE_FA_ALL_NO_CSF_DIR = VOXELWISE_FA_ALL_DIR / "no_csf"
VOXELWISE_FA_FACTORS_SUBDIR = "factors-3"
VOXELWISE_FA_DEFAULT_DIR = VOXELWISE_FA_ALL_WITH_CSF_DIR / VOXELWISE_FA_FACTORS_SUBDIR

CSF_MODES: tuple[str, ...] = ("with_csf", "no_csf")
DEFAULT_CSF_MODES: tuple[str, ...] = ("with_csf", "no_csf")

GRADIENTS_RUN_DIR_REGIONWISE = "regionwise"
GRADIENTS_RUN_DIR_VOXELWISE_CUSTOM = "voxelwise"

REGIONWISE_FA_ALL4_DIR = DEFAULT_TRACTOMETRY_ROOT / (
    "derivatives/analysis/factor_analysis/All4_Combined"
)

DEFAULT_GRADIENTS_VOXELWISE_DIR = Path(
    os.environ.get(
        "GRADIENTS_VOXELWISE_OUTPUT_DIR",
        str(DEFAULT_TRACTOMETRY_ROOT / "derivatives/analysis/gradients_voxelwise"),
    )
)

FACTOR_LOADING_SOURCES: tuple[str, ...] = ("voxelwise", "regionwise")
DEFAULT_FACTOR_LOADING_SOURCE = "voxelwise"

DEFAULT_LOADINGS_CSV = (
    VOXELWISE_FA_DEFAULT_DIR
    / "controls_Voxelwise_ReducedControls_all_with_csf_scalar_factor_loadings.csv"
)
REGIONWISE_DEFAULT_LOADINGS_CSV = (
    REGIONWISE_FA_ALL4_DIR / "controls_All4_Combined_scalar_factor_loadings.csv"
)

DEFAULT_LOADINGS_BY_SOURCE: dict[str, Path] = {
    "voxelwise": DEFAULT_LOADINGS_CSV,
    "regionwise": REGIONWISE_DEFAULT_LOADINGS_CSV,
}
DEFAULT_MANIFEST_CSV = (
    VOXELWISE_FA_ALL_WITH_CSF_DIR
    / "controls_Voxelwise_ReducedControls_all_with_csf_scalar_image_manifest.csv"
)
DEFAULT_MASK_NII = (
    VOXELWISE_FA_ALL_WITH_CSF_DIR
    / "controls_Voxelwise_ReducedControls_all_with_csf_mni_t1w_mask_used.nii.gz"
)


def voxelwise_fa_csf_dir(csf_mode: str) -> Path:
    """Return the voxelwise FA output directory for *with_csf* or *no_csf*."""
    mode = str(csf_mode).strip().lower()
    if mode == "with_csf":
        return VOXELWISE_FA_ALL_WITH_CSF_DIR
    if mode == "no_csf":
        return VOXELWISE_FA_ALL_NO_CSF_DIR
    raise ValueError(f"Unknown CSF mode {csf_mode!r}; expected one of {list(CSF_MODES)}")


def voxelwise_fa_file_prefix(csf_mode: str) -> str:
    mode = str(csf_mode).strip().lower()
    if mode == "with_csf":
        return "controls_Voxelwise_ReducedControls_all_with_csf"
    if mode == "no_csf":
        return "controls_Voxelwise_ReducedControls_all_no_csf"
    raise ValueError(f"Unknown CSF mode {csf_mode!r}; expected one of {list(CSF_MODES)}")


def voxelwise_loadings_csv(csf_mode: str) -> Path:
    prefix = voxelwise_fa_file_prefix(csf_mode)
    return (
        voxelwise_fa_csf_dir(csf_mode)
        / VOXELWISE_FA_FACTORS_SUBDIR
        / f"{prefix}_scalar_factor_loadings.csv"
    )


def voxelwise_manifest_csv(csf_mode: str) -> Path:
    prefix = voxelwise_fa_file_prefix(csf_mode)
    return voxelwise_fa_csf_dir(csf_mode) / f"{prefix}_scalar_image_manifest.csv"


def voxelwise_mask_nii(csf_mode: str) -> Path:
    prefix = voxelwise_fa_file_prefix(csf_mode)
    return voxelwise_fa_csf_dir(csf_mode) / f"{prefix}_mni_t1w_mask_used.nii.gz"


def tissue_classes_for_csf_mode(csf_mode: str | None) -> tuple[str, ...]:
    """Tissue classes shown in summary plots (no CSF when analysis excludes CSF voxels)."""
    if csf_mode == "no_csf":
        return TISSUE_CLASSES_GMWM
    return TISSUE_CLASSES


def atlas_cache_label(mask_nii: Path) -> str:
    """Subdirectory label for atlas reslicing cache keyed by analysis mask."""
    name = mask_nii.name
    if "no_csf" in name:
        return "no_csf"
    if "with_csf" in name:
        return "with_csf"
    return mask_nii.stem


def gradients_run_dir_name(source: str, csf_mode: str | None = None) -> str:
    """Top-level gradients_voxelwise subdirectory for one loading configuration."""
    src = str(source).strip().lower()
    if src == "regionwise":
        return GRADIENTS_RUN_DIR_REGIONWISE
    if csf_mode:
        mode = str(csf_mode).strip().lower()
        if mode not in CSF_MODES:
            raise ValueError(f"Unknown CSF mode {csf_mode!r}; expected one of {list(CSF_MODES)}")
        return f"voxelwise_{mode}"
    return GRADIENTS_RUN_DIR_VOXELWISE_CUSTOM

MNI_ATLAS_DIR = DEFAULT_TRACTOMETRY_ROOT / "data/atlases/MNI"
PROBSEG_PATHS: dict[str, Path] = {
    "GM": MNI_ATLAS_DIR / "tpl-MNI152NLin2009cAsym_res-1mm_label-gm_probseg.nii.gz",
    "WM": MNI_ATLAS_DIR / "tpl-MNI152NLin2009cAsym_res-1mm_label-wm_probseg.nii.gz",
    "CSF": MNI_ATLAS_DIR / "tpl-MNI152NLin2009cAsym_res-1mm_label-csf_probseg.nii.gz",
}

GLASSER_DSEG_NII = (
    DEFAULT_TRACTOMETRY_ROOT
    / "data/atlases/Glasser/atlas-Glasser_space-MNI152NLin2009cAsym_res-1_dseg.nii.gz"
)
GLASSER_DSEG_TSV = DEFAULT_TRACTOMETRY_ROOT / "data/atlases/Glasser/atlas-Glasser_dseg.tsv"

ATLAS_4S156_DSEG_NII = (
    DEFAULT_TRACTOMETRY_ROOT
    / "data/atlases/4S/tpl-MNI152NLin2009cAsym_atlas-4S156Parcels_res-01_dseg.nii.gz"
)
ATLAS_4S156_DSEG_TSV = (
    DEFAULT_TRACTOMETRY_ROOT / "data/atlases/4S/atlas-4S156Parcels_dseg.tsv"
)

HCP1065_ALL_NII_BIN_DIR = DEFAULT_TRACTOMETRY_ROOT / "data/atlases/HCP1065/all_nii_bin"
HCP1065_TRACT_METADATA_CSV = (
    DEFAULT_TRACTOMETRY_ROOT / "data/atlases/HCP1065/HCP1065_tract_metadata.csv"
)

FOUR_S_SUBCORTICAL_ATLAS_NAMES = frozenset(
    {"ThalamusHCP", "SubcorticalHCP", "Cerebellum", "CIT168Subcortical"}
)

MIN_PARCEL_VOXELS = 10

WM_TYPE_DISPLAY: dict[str, str] = {
    "association": "Association",
    "projection": "Projection",
    "commissural": "Commissural",
    "cerebellar": "Cerebellar",
    "cranial_nerves": "Cranial nerves",
}

WM_REGION_GROUP_NAMES: frozenset[str] = frozenset(WM_TYPE_DISPLAY.values())
GM_REGION_GROUP_NAMES: frozenset[str] = frozenset(
    (
        "Basal ganglia",
        "Thalamus",
        "Midbrain",
        "Limbic",
        "Cerebellum",
        "Frontal lobe",
        "Parietal lobe",
        "Temporal lobe",
        "Occipital lobe",
        "Insular lobe",
    )
)
REGION_GROUP_BAR_EDGE_WIDTH = 2.0
REGION_GROUP_WM_FACE = "#ffffff"
REGION_GROUP_GM_FACE = "#9e9e9e"

# neuromaps spatial-correlation screening
NEUROMAPS_TOP_K = 15
NEUROMAPS_MNI_DENSITY = "2mm"
NEUROMAPS_FSLR_DENSITY = "32k"
NEUROMAPS_DEFAULT_N_PERM_MNI = 100
NEUROMAPS_DEFAULT_N_PERM_FSLR = 1000
NEUROMAPS_NULL_SEED = 42
NEUROMAPS_SPACE_MNI = "MNI152"
NEUROMAPS_SPACE_FSLR = "fsLR"
NEUROMAPS_ROW_MNI = "MNI152"
NEUROMAPS_ROW_FSLR = "fsLR 32k"

NEUROMAPS_ANNOTATION_INFO_CSV = (
    DEFAULT_TRACTOMETRY_ROOT / "data/neuromaps/neuromaps_annotation_info.csv"
)

# Plot labels for annotations missing or mislabeled in metadata CSV.
NEUROMAPS_LABEL_OVERRIDES: dict[tuple[str, str], str] = {
    ("castrillon2023", "cmrglc"): "CMRGlu (PET)",
}
