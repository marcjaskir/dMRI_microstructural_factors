#!/usr/bin/env python3
import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
"""
Microstructural factor analysis using GAM-derived z-scores from `derivatives/gam/mni_micro`.

The script supports 4 tissue classes (Cortex GM, Subcortex GM, Association WM, Projection WM)
and an "all-4 combined" run by building scalar feature vectors that concatenate:
  - GM parcel z-scores (one value per parcel)
  - WM tract segment z-scores (three values per tract: end1, core, end2)
"""

import os
from os.path import join as ospj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, Set
from tqdm import tqdm


from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import FastICA, PCA
from matplotlib.patches import Rectangle, Patch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullLocator, FixedLocator

# Suppress sklearn deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# Use Georgia as the default font for all matplotlib text in this script
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Georgia"]

# ============================================================================
# CONFIGURATION
# ============================================================================

# Mode controls which groups are included and where results are written.
# - "epilepsy":   only `penn_epilepsy`
# - "controls":   combined controls: ["penn_controls", "hcpya", "hcpaging"]
GROUP_MODE = "controls"  # "epilepsy" or "controls"

GM_ATLAS = "Glasser + 4S156 (mni_micro)"
WM_ATLAS = "HCP1065 (mni_micro)"
USE_ABS = False  # Regular z-scores only
COMPUTE_FACTOR_SCORES_H5 = False  # Default: do not compute/save .h5 factor scores
N_NODES = 100  # Number of nodes per tract profile

# Define three segments: end1 (nodes 1-34), core (nodes 35-66), end2 (nodes 67-100)
END1_NODES = list(range(1, 35))  # nodes 1-34
CORE_NODES = list(range(35, 67))  # nodes 35-66
END2_NODES = list(range(67, 101))  # nodes 67-100

METADATA_DIR = f"{PROJECT_ROOT}/data/metadata"

MNI_MICRO_PROJECT_ROOT = f"{gam_dir()}/mni_micro"
GM_GLASSER_PROFILE_DIR = f"{MNI_MICRO_PROJECT_ROOT}/Glasser"
GM_4S156_PROFILE_DIR = f"{MNI_MICRO_PROJECT_ROOT}/4S156"
WM_HCP1065_PROFILE_DIR = f"{MNI_MICRO_PROJECT_ROOT}/HCP1065"
WM_PROFILE_DIR_PYAFQ = f"{gam_dir()}/pyafq/HCP1065"

FOUR_S156_DSEG_PATH = f"{PROJECT_ROOT}/data/atlases/4S/atlas-4S156Parcels_dseg.tsv"
HCP1065_TRACT_METADATA_PATH = f"{PROJECT_ROOT}/data/atlases/HCP1065/HCP1065_tract_metadata.csv"

OUTPUT_PROJECT_ROOT = f"{analysis_dir()}/factor_analysis"

EXCLUDED_SCALARS = ["map_li", "map_am", "dti_txx", "dti_txy", "dti_txz", "dti_tyy", "dti_tyz", "dti_tzz", "dti_ha", "rdi_rd1", "rdi_rd2", "gqi_iso"]
# COLINEAR_SCALRS = ["rdi_rd1", "rdi_rd2", "dki_mk", "dki_rd", "map_pa", "map_ngperp", "map_rtap", "dti_md", "dti_ad", "dti_fa", "dti_rd"]

# Tracts to exclude from analysis
TRACTS_TO_REMOVE = [
    "CBT_L", "CBT_R", "RST_L", "RST_R", "DRTT_L", "DRTT_R", 
    "EMC_L", "EMC_R", "C_PHP_L", "C_PHP_R", "AF_L", "AF_R", "SLF3_L", "SLF3_R",
    "SLF2_L", "SLF2_R", "FAT_L", "FAT_R"
]
# Some segmentation failures (lose 9 subjects), but included: "CPT_F_L", "CPT_F_R", "CST_L", "CST_R"

# Order microstructural statistics by these scalar prefixes when plotting
SCALAR_PREFIX_ORDER = ["dti", "rdi", "dki", "gqi", "noddi", "map"]

# Combined heatmap subset: DTI, DKI, GQI only (excludes NODDI, MAP-MRI ``map_*``, RDI ``rdi_*``).
COMBINED_HEATMAP_DTI_DKI_GQI_PREFIXES = frozenset({"dti", "dki", "gqi"})
# Further omit specific scalars from the DTI/DKI/GQI correlation-only figure (e.g. isotropic diffusion).
COMBINED_HEATMAP_DTI_DKI_GQI_EXCLUDE_SCALARS = frozenset({"gqi_iso"})

# DPI for ``*_factor_loading_ordered.png`` (Pairwise Correlations + … Factor loading ordered).
CORR_FACTOR_PCA_FACTOR_ORDERED_DPI = 600

# Human-readable factor names (F2 = non-Gaussian, F3 = anisotropic per loadings interpretation).
FACTOR_DIFFUSIVITY_LABELS: Dict[str, str] = {
    "F1": "Overall diffusivity",
    "F2": "Non-Gaussian diffusivity",
    "F3": "Anisotropic diffusivity",
    "F4": "SDF-based diffusivity",
}

# Short factor names for the bottom-layout corr + loadings heatmap.
FACTOR_SHORT_LABELS: Dict[str, str] = {
    "F1": "Overall",
    "F2": "Non-Gaussian",
    "F3": "Anisotropic",
    "F4": "SDF-based",
}

# Reconstruction model legend: NODDI → MAP-MRI → DKI → DTI → GQI.
RECONSTRUCTION_MODEL_LEGEND: Tuple[Tuple[str, str], ...] = (
    ("#38489E", "NODDI"),
    ("#289144", "MAP-MRI"),
    ("#7A297F", "DKI"),
    ("#C43031", "DTI"),
    ("#FAA51A", "GQI"),
)


def reconstruction_model_legend_handles() -> List[Patch]:
    return [
        Patch(facecolor=hex_color, edgecolor="none", label=label)
        for hex_color, label in RECONSTRUCTION_MODEL_LEGEND
    ]


CORR_FACTOR_LOADING_CBAR_LABEL = "Pearson correlation / Factor loading"


def factor_name_to_diffusivity_label(factor_name: str) -> str:
    """Map factor id (e.g. F2) to human-readable diffusivity label."""
    if factor_name in FACTOR_DIFFUSIVITY_LABELS:
        return FACTOR_DIFFUSIVITY_LABELS[factor_name]
    if factor_name.startswith("F") and factor_name[1:].isdigit():
        return f"Factor {factor_name[1:]}"
    return factor_name


def factor_name_to_short_label(factor_name: str) -> str:
    """Map factor id (e.g. F2) to short label for compact heatmap axes."""
    if factor_name in FACTOR_SHORT_LABELS:
        return FACTOR_SHORT_LABELS[factor_name]
    if factor_name.startswith("F") and factor_name[1:].isdigit():
        return f"F{factor_name[1:]}"
    return factor_name

# ICA: fit up to this many components; retain exactly `n_factors_rot` for correlations + variance plot
N_ICA_COMPONENTS_FULL = 26

# PCA: fit up to this many PCs for cumulative explained-variance elbow selection
N_PCA_COMPONENTS_FULL = 26


def _resolve_group_mode(mode: str) -> Tuple[List[str], str, str]:
    """
    Resolve GROUP_MODE to:
      - list of group labels to include from the GAM files
      - a short label used in filenames
      - the subdirectory name under OUTPUT_PROJECT_ROOT
    """
    mode = mode.lower()
    if mode == "epilepsy":
        groups = ["penn_epilepsy"]
        group_label = "penn_epilepsy"
        subdir = "penn_epilepsy"
    elif mode == "controls":
        groups = ["penn_controls", "hcpya", "hcpaging"]
        group_label = "controls"
        subdir = "controls"
    else:
        raise ValueError(f"Unsupported GROUP_MODE: {mode}")
    return groups, group_label, subdir


GROUPS, GROUP_LABEL, GROUP_SUBDIR = _resolve_group_mode(GROUP_MODE)
# Outputs are written per atlas run under OUTPUT_PROJECT_ROOT (no GROUP_SUBDIR in path).
os.makedirs(OUTPUT_PROJECT_ROOT, exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_scalar_labels() -> List[str]:
    """Load and filter scalar labels."""
    path = ospj(METADATA_DIR, "scalar_labels_to_filenames.json")
    with open(path) as f:
        all_labels = list(json.load(f).keys())
    return [label for label in all_labels if label not in EXCLUDED_SCALARS]

def _list_subdirs(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    return sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(ospj(base_dir, d))]
    )


def get_glasser_regions() -> List[str]:
    """Cortex GM parcel dirs from `mni_micro/Glasser`."""
    return _list_subdirs(GM_GLASSER_PROFILE_DIR)


def get_subcortex_4s156_regions() -> List[str]:
    """Subcortex GM parcel labels (network_label == 'n/a') from 4S156 dseg."""
    if not os.path.exists(FOUR_S156_DSEG_PATH):
        return []
    dseg = pd.read_csv(FOUR_S156_DSEG_PATH, sep="\t")
    if "label" not in dseg.columns or "network_label" not in dseg.columns:
        return []

    # `pandas` often parses literal "n/a" as NaN; treat NaN as subcortex.
    nl = dseg["network_label"]
    subcortex_mask = nl.isna()
    # Also handle cases where "n/a" survives as a string.
    subcortex_mask = subcortex_mask | (nl.astype(str).str.strip().str.lower() == "n/a")
    subcortex = dseg.loc[subcortex_mask, "label"].astype(str).tolist()
    existing = set(_list_subdirs(GM_4S156_PROFILE_DIR))
    return sorted([lab for lab in subcortex if lab in existing])


@lru_cache(maxsize=1)
def load_hcp1065_tract_metadata() -> pd.DataFrame:
    if not os.path.exists(HCP1065_TRACT_METADATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(HCP1065_TRACT_METADATA_PATH)


def get_tracts_by_type(tract_type: str) -> List[str]:
    """HCP1065 tract labels filtered by metadata column `type`."""
    meta = load_hcp1065_tract_metadata()
    if meta.empty or "label" not in meta.columns or "type" not in meta.columns:
        return []

    tracts = meta.loc[meta["type"].astype(str) == tract_type, "label"].astype(str).tolist()
    tracts = [t for t in tracts if t not in TRACTS_TO_REMOVE]

    # Further filter: keep only tract bases that exist in the pyAFQ pipeline outputs.
    # (We use pyafq node-wise GAM outputs for WM segments.)
    available_bases: Set[str] = set(_list_subdirs(WM_PROFILE_DIR_PYAFQ))
    return sorted([t for t in tracts if t in available_bases])


def get_mni_micro_gm_profile_dir_for_region(region: str) -> str:
    """Route GM region label to Glasser vs 4S156 base directories."""
    if os.path.isdir(ospj(GM_GLASSER_PROFILE_DIR, region)):
        return GM_GLASSER_PROFILE_DIR
    return GM_4S156_PROFILE_DIR


def tract_to_mni_micro_segments(tract: str, hcp_meta: pd.DataFrame) -> Tuple[str, str, str] | None:
    """
    Map a base HCP1065 tract label to (end1_dir, core_dir, end2_dir) under `mni_micro/HCP1065`.
    """
    if hcp_meta.empty or "label" not in hcp_meta.columns:
        return None
    if tract not in set(hcp_meta["label"].astype(str).tolist()):
        return None

    row = hcp_meta.loc[hcp_meta["label"].astype(str) == tract].iloc[0]
    end1 = str(row["end1"]) if "end1" in row else None
    end2 = str(row["end2"]) if "end2" in row else None
    if end1 in (None, "nan", "NA") or end2 in (None, "nan", "NA"):
        return None

    end1_dir = f"{tract}_end-{end1}"
    core_dir = f"{tract}_core"
    end2_dir = f"{tract}_end-{end2}"
    return end1_dir, core_dir, end2_dir


def load_region_scalar_data(
    region: str,
    scalar: str,
    groups: Sequence[str],
) -> pd.DataFrame | None:
    """
    Load z-score data for a specific region and scalar, restricted to the
    requested GAM `group` labels.

    Returns a DataFrame with index 'sub' and a single column '{scalar}_z',
    or None if data are unavailable.
    """
    gm_profile_dir = get_mni_micro_gm_profile_dir_for_region(region)
    gam_path = ospj(gm_profile_dir, region, f"{region}_{scalar}_stat-mean_gam.csv")
    if not os.path.exists(gam_path):
        gam_path_legacy = ospj(gm_profile_dir, region, f"{region}_{scalar}_gam.csv")
        if not os.path.exists(gam_path_legacy):
            return None
        gam_path = gam_path_legacy

    try:
        gam_data = pd.read_csv(gam_path)
        group_data = gam_data[gam_data["group"].isin(groups)].copy()
        if group_data.empty:
            return None
        z_col = f"{scalar}_z"
        if z_col not in group_data.columns:
            return None
        return group_data[["sub", z_col]].set_index("sub")
    except Exception as e:  # noqa: BLE001
        print(f"Error loading {region}_{scalar}: {e}")
        return None


def load_tract_scalar_data(
    tract: str,
    scalar: str,
    groups: Sequence[str],
) -> pd.DataFrame | None:
    """
    Load z-score data for a specific tract and scalar, restricted to the
    requested GAM `group` labels.
    
    Returns a DataFrame with index 'sub' and columns:
      - 'node1_z', 'node2_z', ..., 'node100_z'
    or None if data are unavailable.
    """
    try:
        gam_path = ospj(WM_PROFILE_DIR_PYAFQ, tract, f"{tract}_{scalar}_stat-mean_gam.csv")
        if not os.path.exists(gam_path):
            gam_path_legacy = ospj(WM_PROFILE_DIR_PYAFQ, tract, f"{tract}_{scalar}_gam.csv")
            if not os.path.exists(gam_path_legacy):
                return None
            gam_path = gam_path_legacy

        gam_data = pd.read_csv(gam_path)
        group_data = gam_data[gam_data["group"].isin(groups)].copy()
        if group_data.empty:
            return None

        z_cols = [f"node{i}_z" for i in range(1, N_NODES + 1)]
        missing_cols = [col for col in z_cols if col not in group_data.columns]
        if missing_cols:
            return None
        return group_data[["sub"] + z_cols].set_index("sub")
    except Exception as e:  # noqa: BLE001
        print(f"Error loading {tract}_{scalar}: {e}")
        return None


def get_segment_mean_z(z_scores: np.ndarray, segment_nodes: List[int]) -> float:
    """
    Compute mean z-score for a specific segment of nodes.
    
    Args:
        z_scores: Array of z-scores for all 100 nodes (0-indexed: indices 0-99 correspond to nodes 1-100)
        segment_nodes: List of node numbers (1-indexed: 1-100)
    
    Returns:
        Mean z-score for the segment
    """
    # Convert 1-indexed node numbers to 0-indexed array indices
    segment_indices = [node - 1 for node in segment_nodes]
    segment_values = z_scores[segment_indices]
    return float(np.nanmean(segment_values))


def build_combined_feature_vectors(
    subjects: Sequence[str],
    regions: Sequence[str],
    tracts: Sequence[str],
    scalar_labels: Sequence[str],
    groups: Sequence[str],
    use_abs: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Build feature vectors for each scalar by concatenating:
      - GM parcel z-scores (one value per parcel)
      - WM tract segment mean z-scores (three values per tract: end1, core, end2)
    
    Each scalar's feature vector is:
        [z(sub, region) for sub in subjects, for region in regions] +
        [mean(z(sub, tract), end1_nodes), mean(z(sub, tract), core_nodes), mean(z(sub, tract), end2_nodes)
         for sub in subjects, for tract in tracts]
    
    Shape per scalar: (n_subjects * n_regions + n_subjects * n_tracts * 3)
    """
    all_region_scalar_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    all_tract_scalar_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    
    # Load all region/scalar data once
    print("Loading GM region data...")
    for region in tqdm(sorted(regions), desc="Loading regions"):
        all_region_scalar_data[region] = {}
        for scalar in scalar_labels:
            data = load_region_scalar_data(region, scalar, groups)
            if data is not None:
                all_region_scalar_data[region][scalar] = data
    
    # Load all tract/scalar data once
    print("Loading WM tract data...")
    for tract in tqdm(sorted(tracts), desc="Loading tracts"):
        all_tract_scalar_data[tract] = {}
        for scalar in scalar_labels:
            data = load_tract_scalar_data(tract, scalar, groups)
            if data is not None:
                all_tract_scalar_data[tract][scalar] = data
    
    scalar_vectors: Dict[str, np.ndarray] = {}
    
    print("Building combined feature vectors...")
    for scalar in tqdm(scalar_labels, desc="Building vectors"):
        feature_vector: list[float] = []
        
        # Add GM region data
        for subject in subjects:
            for region in sorted(regions):
                if (
                    region in all_region_scalar_data
                    and scalar in all_region_scalar_data[region]
                ):
                    region_scalar_data = all_region_scalar_data[region][scalar]
                    if subject in region_scalar_data.index:
                        z_val = region_scalar_data.loc[subject, f"{scalar}_z"]
                        if use_abs:
                            z_val = np.abs(z_val)
                        feature_vector.append(float(z_val))
                    else:
                        feature_vector.append(np.nan)
                else:
                    feature_vector.append(np.nan)
        
    # Add WM tract data (3 features per tract segment: end1, core, end2)
        for subject in subjects:
            for tract in sorted(tracts):
                if (
                    tract in all_tract_scalar_data
                    and scalar in all_tract_scalar_data[tract]
                ):
                    tract_scalar_data = all_tract_scalar_data[tract][scalar]
                    if subject in tract_scalar_data.index:
                        z_scores = tract_scalar_data.loc[subject].values  # node1_z..node100_z
                        if use_abs:
                            # For abs-z analyses, segment means are computed from abs(node z).
                            z_scores = np.abs(z_scores)

                        mean_end1 = get_segment_mean_z(z_scores, END1_NODES)
                        mean_core = get_segment_mean_z(z_scores, CORE_NODES)
                        mean_end2 = get_segment_mean_z(z_scores, END2_NODES)

                        feature_vector.extend([mean_end1, mean_core, mean_end2])
                    else:
                        feature_vector.extend([np.nan, np.nan, np.nan])
                else:
                    feature_vector.extend([np.nan, np.nan, np.nan])
        
        scalar_vectors[scalar] = np.array(feature_vector, dtype=float)
    
    return scalar_vectors


def compute_correlation_matrix_scalars(
    feature_matrix: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Compute pairwise correlation matrix between scalars."""
    scalars = feature_matrix.index.tolist()
    n_scalars = len(scalars)
    correlation_matrix = np.full((n_scalars, n_scalars), np.nan, dtype=float)
    n_shared_features = np.zeros((n_scalars, n_scalars), dtype=int)

    for i, scalar1 in enumerate(scalars):
        for j, scalar2 in enumerate(scalars):
            if i == j:
                correlation_matrix[i, j] = 1.0
                n_shared_features[i, j] = feature_matrix.shape[1]
            elif i < j:
                vec1 = feature_matrix.loc[scalar1].values
                vec2 = feature_matrix.loc[scalar2].values
                valid_mask = ~(np.isnan(vec1) | np.isnan(vec2))

                if valid_mask.sum() > 0:
                    vec1_valid = vec1[valid_mask]
                    vec2_valid = vec2[valid_mask]

                    if (
                        len(vec1_valid) > 1
                        and np.std(vec1_valid) > 0
                        and np.std(vec2_valid) > 0
                    ):
                        corr = np.corrcoef(vec1_valid, vec2_valid)[0, 1]
                        correlation_matrix[i, j] = corr
                        correlation_matrix[j, i] = corr
                        n_shared_features[i, j] = valid_mask.sum()
                        n_shared_features[j, i] = valid_mask.sum()

    corr_df = pd.DataFrame(correlation_matrix, index=scalars, columns=scalars)
    shared_df = pd.DataFrame(n_shared_features, index=scalars, columns=scalars)
    return corr_df, shared_df, correlation_matrix


def order_scalars_by_prefix(scalars: Sequence[str]) -> List[str]:
    """Order scalar names by predefined prefixes, then alphabetically."""
    prefix_rank = {p: i for i, p in enumerate(SCALAR_PREFIX_ORDER)}

    def _get_prefix(name: str) -> str:
        return name.split("_", 1)[0] if "_" in name else name

    return sorted(
        list(scalars),
        key=lambda name: (prefix_rank.get(_get_prefix(name), len(prefix_rank)), name),
    )


def order_scalars_by_max_abs_factor_loading(loadings_df: pd.DataFrame) -> List[str]:
    """
    Order scalar columns by:
      (1) which factor has the largest |loading| for that scalar (F1 group, then F2, ...);
      (2) within each group, descending |loading| on that dominant factor.
    """
    scalars = loadings_df.columns.tolist()
    if not scalars:
        return []
    abs_mat = np.abs(loadings_df.values.T)  # (n_scalars, n_factors)
    win = abs_mat.argmax(axis=1).astype(int)
    win_val = abs_mat[np.arange(len(scalars)), win]
    order = np.lexsort((-win_val, win))
    return [scalars[i] for i in order]


# MAP-MRDI NG labels use ∥ (U+2225) and ⊥ (U+22A5); Georgia often lacks these glyphs in matplotlib.
_U2225_PARALLEL = "\u2225"
_U22A5_PERP = "\u22a5"


def format_scalar_tick_label_for_mixed_fonts(label: str) -> str:
    """
    Keep Georgia for scalar names where possible; render parallel/perpendicular with mathtext
    so symbols draw correctly (\\parallel, \\perp). Uses \\mathdefault{NG} so the letters match
    the document text font (Georgia set via rcParams).
    """
    # \text{∥/⊥} avoids \parallel/\perp mathrel spacing before closing ")".
    if label == f"Non-Gaussian Parallel (NG{_U2225_PARALLEL})":
        return r"Non-Gaussian Parallel $\mathdefault{(NG\text{" + _U2225_PARALLEL + r"})}$"
    if label == f"Non-Gaussian Perpendicular (NG{_U22A5_PERP})":
        return r"Non-Gaussian Perpendicular $\mathdefault{(NG\text{" + _U22A5_PERP + r"})}$"
    if label == "NG" + _U2225_PARALLEL:
        return r"$\mathdefault{NG\text{" + _U2225_PARALLEL + r"}}$"
    if label == "NG" + _U22A5_PERP:
        return r"$\mathdefault{NG\text{" + _U22A5_PERP + r"}}$"
    return label


def plot_corr_and_loadings_combined(
    corr_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    output_path: str,
    *,
    row_order: str = "prefix",
    allowed_prefixes: Optional[Set[str]] = None,
    dpi: int = 300,
) -> None:
    """
    Create a single heatmap showing:
      - Correlation matrix on the left
      - Factor loadings on the right (after a spacer column)

    row_order:
      - ``prefix``: order rows/columns by diffusion family prefix (default).
      - ``max_factor_loading``: order by dominant factor (largest |loading|), then |loading|.

    allowed_prefixes:
      - If set, keep only scalars whose prefix (before first ``_``) is in this set.
    """
    if allowed_prefixes is not None:
        def _pfx(n: str) -> str:
            return n.split("_", 1)[0] if "_" in n else n

        keep = [c for c in corr_df.columns if _pfx(c) in allowed_prefixes]
        keep = order_scalars_by_prefix(keep)
        if len(keep) < 2:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.axis("off")
            msg = f"Too few scalars after prefix filter (n={len(keep)}); need ≥2."
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=12)
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Warning: corr+loadings combined skipped ({msg}) -> {output_path}")
            return
        corr_df = corr_df.loc[keep, keep].copy()
        loadings_df = loadings_df[keep].copy()

    prefix_cols = order_scalars_by_prefix(corr_df.columns)
    corr_df = corr_df.loc[prefix_cols, prefix_cols]
    loadings_df = loadings_df[prefix_cols]

    if row_order == "max_factor_loading":
        ordered_cols = order_scalars_by_max_abs_factor_loading(loadings_df)
        corr_df = corr_df.loc[ordered_cols, ordered_cols]
        loadings_df = loadings_df[ordered_cols]
    elif row_order == "prefix":
        ordered_cols = prefix_cols
    else:
        raise ValueError(f"Unknown row_order: {row_order!r}")

    n_scalars = len(ordered_cols)
    n_factors = loadings_df.shape[0]

    # Transpose loadings so factors are columns (plotted to the right of the correlation block)
    loadings_df_T = loadings_df.T  # (n_scalars, n_factors) - factors as columns

    # Separator column between correlation matrix and factor loadings
    separator_col = pd.DataFrame(
        np.nan,
        index=ordered_cols,
        columns=[" "],  # Single space as column name
    )

    # Combine: correlation matrix (left) + spacer + factor loadings (right)
    combined_df = pd.concat([corr_df, separator_col, loadings_df_T], axis=1)

    # Calculate figure size
    fig_width = 18
    fig_height = 12

    # Create figure with GridSpec for main plot and horizontal colorbar at bottom
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='none', edgecolor='none', frameon=False)
    fig.patch.set_visible(False)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[20, 0.5], hspace=0.10)
    ax_main = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    # Create heatmap
    vmin = -1.0
    vmax = 1.0
    sns.heatmap(
        combined_df,
        ax=ax_main,
        cbar_ax=cax,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": CORR_FACTOR_LOADING_CBAR_LABEL, "orientation": "horizontal"},
        xticklabels=False,
        yticklabels=False,
    )
    
    # Move colorbar tick labels above the bar
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    # Load metadata for labels and colors
    try:
        colors_path = ospj(METADATA_DIR, "scalar_labels_to_colors.json")
        human_path = ospj(METADATA_DIR, "scalar_labels_to_human.json")
        with open(colors_path) as f:
            scalar_to_color = json.load(f)
        with open(human_path) as f:
            scalar_to_human = json.load(f)

        human_labels = [
            scalar_to_human.get(scalar_name, scalar_name)
            for scalar_name in ordered_cols
        ]

        # Abbreviated labels for x-axis
        abbr_labels: List[str] = []
        for label in human_labels:
            if "(" in label and ")" in label:
                start = label.rfind("(") + 1
                end = label.rfind(")")
                abbr = label[start:end].strip()
                abbr_labels.append(abbr if abbr else label)
            else:
                abbr_labels.append(label)

        human_labels = [format_scalar_tick_label_for_mixed_fonts(l) for l in human_labels]
        abbr_labels = [format_scalar_tick_label_for_mixed_fonts(l) for l in abbr_labels]

        # Y-AXIS: Set custom tick positions and labels
        y_tick_positions = np.arange(0.5, n_scalars + 0.5)
        ax_main.yaxis.set_major_locator(FixedLocator(y_tick_positions))
        ax_main.set_yticklabels(human_labels, rotation=0, fontsize=12)
        ax_main.tick_params(axis='y', which='major', length=0, width=0, left=False, labelleft=True)
        for tick, scalar_name in zip(ax_main.get_yticklabels(), ordered_cols):
            tick.set_color(scalar_to_color.get(scalar_name, "#000000"))

        # Right spine: scalar abbreviations (rows span correlation + loadings blocks)
        sec_y = ax_main.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
        sec_y.set_yticks(y_tick_positions)
        sec_y.set_yticklabels(abbr_labels, rotation=0, fontsize=12)
        sec_y.tick_params(axis="y", which="major", length=0, width=0, right=True, labelright=True)
        for tick, scalar_name in zip(sec_y.get_yticklabels(), ordered_cols):
            tick.set_color(scalar_to_color.get(scalar_name, "#000000"))

        # X-AXIS: correlation columns (left), then factor loadings columns (right)
        x_tick_positions = []
        x_tick_labels = []

        for i in range(n_scalars):
            x_tick_positions.append(i + 0.5)
            x_tick_labels.append(abbr_labels[i])

        factor_names = loadings_df.index.tolist()
        loadings_x0 = n_scalars + 1
        for i in range(n_factors):
            x_tick_positions.append(loadings_x0 + i + 0.5)
            x_tick_labels.append(factor_name_to_short_label(factor_names[i]))

        ax_main.xaxis.set_major_locator(FixedLocator(x_tick_positions))
        ax_main.set_xticklabels(x_tick_labels, rotation=45, fontsize=12, ha='left')
        ax_main.tick_params(axis='x', which='major', length=0, width=0, bottom=False, labelbottom=False, top=True, labeltop=True)

        # Color x-axis labels for the correlation matrix columns only
        for i in range(n_scalars):
            tick_idx = i
            tick = ax_main.get_xticklabels()[tick_idx]
            tick.set_color(scalar_to_color.get(ordered_cols[i], "#000000"))

    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not apply scalar color/human labels to combined plot: {e}")

    # Add text labels for loading weights in the factor loadings section (right block)
    loadings_x0 = n_scalars + 1
    loadings_values = loadings_df_T.values  # (n_scalars, n_factors)
    abs_loadings = np.abs(loadings_values)
    max_loading_cols = abs_loadings.argmax(axis=1)
    for j in range(n_scalars):
        for i in range(n_factors):
            val = loadings_values[j, i]
            if not np.isnan(val):
                # Position text in center of cell
                x_pos = loadings_x0 + i + 0.5
                y_pos = j + 0.5
                # Format value with 2 decimal places
                text = f"{val:.2f}"
                # Choose text color based on background
                text_color = 'black' if abs(val) < 0.3 else 'white'
                ax_main.text(
                    x_pos, y_pos, text,
                    ha='center', va='center',
                    fontsize=8,
                    color=text_color,
                    weight='bold' if i == max_loading_cols[j] else 'normal',
                )

    # Add text labels for correlation values in the lower triangle of the correlation matrix (left block)
    corr_values = corr_df.loc[ordered_cols, ordered_cols].values  # (n_scalars, n_scalars)
    for j in range(n_scalars):
        for i in range(n_scalars):
            # Only label lower triangle (j > i, meaning row > column, below diagonal) to avoid duplication
            if j > i:
                val = corr_values[j, i]
                if not np.isnan(val):
                    # Position text in center of cell
                    x_pos = i + 0.5
                    y_pos = j + 0.5
                    # Format value with 2 decimal places
                    text = f"{val:.2f}"
                    # Choose text color based on background (same logic as loadings)
                    text_color = 'black' if abs(val) < 0.3 else 'white'
                    ax_main.text(
                        x_pos, y_pos, text,
                        ha='center', va='center',
                        fontsize=8,
                        color=text_color,
                    )

    # Outline, for each scalar (row), the cell with the largest |loading| in the loadings section
    for j, i in enumerate(max_loading_cols):
        rect = Rectangle(
            (loadings_x0 + i, j),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
        ax_main.add_patch(rect)

    # Set axis labels
    ax_main.set_ylabel("Diffusion statistic", fontsize=14)
    ax_main.set_xlabel("")
    cax.set_xlabel(CORR_FACTOR_LOADING_CBAR_LABEL, fontsize=14)

    legend_handles = reconstruction_model_legend_handles()

    # Tight margins - room for right-side abbreviation labels and top factor column ticks
    fig.subplots_adjust(left=0.20, right=0.88, top=0.915, bottom=0.12, wspace=0.02)

    # Footer row: reconstruction-model legend (left) + colorbar (right)
    ax_pos = ax_main.get_position()
    cax_pos = cax.get_position()
    cbar_width = ax_pos.width * 0.4
    cbar_x0 = ax_pos.x0 + ax_pos.width - cbar_width
    cbar_height = cax_pos.height * 0.5
    cbar_y0 = cax_pos.y0
    cax.set_position([cbar_x0, cbar_y0, cbar_width, cbar_height])

    leg_gap = 0.015
    leg_width = max(cbar_x0 - ax_pos.x0 - leg_gap, 0.25)
    ax_leg = fig.add_axes([ax_pos.x0, cbar_y0 + cax_pos.height * 0.8, leg_width, cbar_height])
    ax_leg.set_axis_off()
    leg = ax_leg.legend(
        handles=legend_handles,
        loc="center left",
        ncol=5,
        fontsize=14,
        title="Reconstruction model",
        frameon=True,
        fancybox=True,
        borderpad=0.3,
        columnspacing=0.5,
        handletextpad=0.3,
    )
    if leg.get_title() is not None:
        leg.get_title().set_fontsize(14)
    leg.get_frame().set_facecolor("#F6F6FA")
    leg.get_frame().set_edgecolor("#CCCCCC")
    leg.get_frame().set_linewidth(1.5)

    fig.savefig(output_path, dpi=dpi, bbox_inches=None, facecolor='none', edgecolor='none', pad_inches=0)
    plt.close(fig)
    ord_note = "factor-loading ordered" if row_order == "max_factor_loading" else "prefix order"
    print(f"Saved combined correlation + factor loading plot ({ord_note}) to: {output_path}")


def _prepare_corr_and_loadings_blocks(
    corr_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    *,
    row_order: str = "prefix",
    allowed_prefixes: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
    """Shared ordering/filtering for corr + loadings combined heatmaps."""
    if allowed_prefixes is not None:
        def _pfx(n: str) -> str:
            return n.split("_", 1)[0] if "_" in n else n

        keep = [c for c in corr_df.columns if _pfx(c) in allowed_prefixes]
        keep = order_scalars_by_prefix(keep)
        if len(keep) < 2:
            raise ValueError(f"Too few scalars after prefix filter (n={len(keep)}); need ≥2.")
        corr_df = corr_df.loc[keep, keep].copy()
        loadings_df = loadings_df[keep].copy()

    prefix_cols = order_scalars_by_prefix(corr_df.columns)
    corr_df = corr_df.loc[prefix_cols, prefix_cols]
    loadings_df = loadings_df[prefix_cols]

    if row_order == "max_factor_loading":
        ordered_cols = order_scalars_by_max_abs_factor_loading(loadings_df)
        corr_df = corr_df.loc[ordered_cols, ordered_cols]
        loadings_df = loadings_df[ordered_cols]
    elif row_order == "prefix":
        ordered_cols = prefix_cols
    else:
        raise ValueError(f"Unknown row_order: {row_order!r}")

    ord_note = "factor-loading ordered" if row_order == "max_factor_loading" else "prefix order"
    return corr_df, loadings_df, ordered_cols, ord_note


def plot_corr_and_loadings_combined_bottom(
    corr_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    output_path: str,
    *,
    row_order: str = "max_factor_loading",
    allowed_prefixes: Optional[Set[str]] = None,
    dpi: int = 300,
) -> None:
    """
    Correlation matrix (lower triangle only) with factor loadings below (not to the right).

    * Correlation block: upper triangle + diagonal masked; no x-axis labels at top.
    * Loadings block: factor short names on y-axis (Overall, Non-Gaussian, Anisotropic);
      full statistic names with abbreviations in parentheses on x-axis (45°).
    """
    try:
        corr_df, loadings_df, ordered_cols, ord_note = _prepare_corr_and_loadings_blocks(
            corr_df,
            loadings_df,
            row_order=row_order,
            allowed_prefixes=allowed_prefixes,
        )
    except ValueError as exc:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes, fontsize=12)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Warning: corr+loadings bottom combined skipped ({exc}) -> {output_path}")
        return

    n_scalars = len(ordered_cols)
    n_factors = loadings_df.shape[0]

    corr_values = corr_df.to_numpy(dtype=float, copy=True)
    upper_mask = np.triu(np.ones((n_scalars, n_scalars), dtype=bool))
    corr_values[upper_mask] = np.nan
    corr_masked = pd.DataFrame(corr_values, index=ordered_cols, columns=ordered_cols)

    separator_row = pd.DataFrame(
        np.nan,
        index=[" "],
        columns=ordered_cols,
    )
    combined_df = pd.concat([corr_masked, separator_row, loadings_df], axis=0)

    fig_width = max(14.0, n_scalars * 0.55)
    fig_height = max(10.0, (n_scalars + n_factors + 2) * 0.42)
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="none", edgecolor="none", frameon=False)
    fig.patch.set_visible(False)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.4, 20], hspace=0.03)
    ax_header = fig.add_subplot(gs[0])
    ax_header.set_axis_off()
    ax_main = fig.add_subplot(gs[1])

    vmin = -1.0
    vmax = 1.0
    hm = sns.heatmap(
        combined_df,
        ax=ax_main,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )

    loadings_y0 = n_scalars + 1
    loadings_values = loadings_df.values
    abs_loadings = np.abs(loadings_values)
    max_loading_rows = abs_loadings.argmax(axis=0)

    try:
        colors_path = ospj(METADATA_DIR, "scalar_labels_to_colors.json")
        human_path = ospj(METADATA_DIR, "scalar_labels_to_human.json")
        with open(colors_path) as f:
            scalar_to_color = json.load(f)
        with open(human_path) as f:
            scalar_to_human = json.load(f)

        human_labels = [
            scalar_to_human.get(scalar_name, scalar_name)
            for scalar_name in ordered_cols
        ]
        human_labels_fmt = [format_scalar_tick_label_for_mixed_fonts(l) for l in human_labels]

        corr_y_positions = np.arange(0.5, n_scalars + 0.5)
        factor_names = loadings_df.index.tolist()
        factor_y_positions = np.arange(loadings_y0 + 0.5, loadings_y0 + n_factors + 0.5)
        factor_short_labels = [factor_name_to_short_label(name) for name in factor_names]
        all_y_positions = np.concatenate([corr_y_positions, factor_y_positions])
        y_labels = human_labels_fmt + factor_short_labels

        ax_main.yaxis.set_major_locator(FixedLocator(all_y_positions))
        ax_main.set_yticklabels(y_labels, rotation=0, fontsize=12)
        ax_main.tick_params(axis="y", which="major", length=0, width=0, left=False, labelleft=True)
        for tick, scalar_name in zip(ax_main.get_yticklabels()[:n_scalars], ordered_cols):
            tick.set_color(scalar_to_color.get(scalar_name, "#000000"))

        x_tick_positions = np.arange(0.5, n_scalars + 0.5)
        ax_main.xaxis.set_major_locator(FixedLocator(x_tick_positions))
        ax_main.set_xticklabels(human_labels_fmt, rotation=45, fontsize=12, ha="right")
        ax_main.tick_params(
            axis="x",
            which="major",
            length=0,
            width=0,
            bottom=True,
            labelbottom=True,
            top=False,
            labeltop=False,
        )
        for tick, scalar_name in zip(ax_main.get_xticklabels(), ordered_cols):
            tick.set_color(scalar_to_color.get(scalar_name, "#000000"))

    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not apply scalar color/human labels to bottom combined plot: {e}")

    for j in range(n_scalars):
        for i in range(n_factors):
            val = loadings_values[i, j]
            if not np.isnan(val):
                x_pos = j + 0.5
                y_pos = loadings_y0 + i + 0.5
                text_color = "black" if abs(val) < 0.3 else "white"
                ax_main.text(
                    x_pos,
                    y_pos,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    weight="bold" if i == max_loading_rows[j] else "normal",
                )

    corr_values = corr_df.loc[ordered_cols, ordered_cols].values
    for j in range(n_scalars):
        for i in range(n_scalars):
            if j > i:
                val = corr_values[j, i]
                if not np.isnan(val):
                    text_color = "black" if abs(val) < 0.3 else "white"
                    ax_main.text(
                        i + 0.5,
                        j + 0.5,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                    )

    for j, i in enumerate(max_loading_rows):
        rect = Rectangle(
            (j, loadings_y0 + i),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
        ax_main.add_patch(rect)

    ax_main.set_ylabel("Diffusion statistic", fontsize=14)
    ax_main.set_xlabel("")

    legend_handles = reconstruction_model_legend_handles()
    leg = ax_header.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.05),
        ncol=5,
        fontsize=14,
        title="Reconstruction model",
        frameon=True,
        fancybox=True,
        borderpad=0.3,
        columnspacing=0.5,
        handletextpad=0.3,
    )
    if leg.get_title() is not None:
        leg.get_title().set_fontsize(14)
    leg.get_frame().set_facecolor("#F6F6FA")
    leg.get_frame().set_edgecolor("#CCCCCC")
    leg.get_frame().set_linewidth(1.5)

    cbar_ax = ax_header.inset_axes([0.52, 0.18, 0.46, 0.64])
    cbar = fig.colorbar(hm.collections[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label(CORR_FACTOR_LOADING_CBAR_LABEL, fontsize=14)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")

    fig.subplots_adjust(left=0.20, right=0.95, top=0.98, bottom=0.22, hspace=0.02)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches=None, facecolor="none", edgecolor="none", pad_inches=0)
    plt.close(fig)
    print(f"Saved combined correlation + factor loading plot (bottom layout, {ord_note}) to: {output_path}")


def plot_corr_factor_loadings_and_pca_components_combined(
    corr_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    pca_loadings_df: pd.DataFrame,
    output_path: str,
    *,
    row_order: str = "prefix",
    allowed_prefixes: Optional[Set[str]] = None,
    dpi: int = 300,
    include_factor_pca_blocks: bool = True,
    axis_tick_fontsize: int = 12,
    corr_annotation_fontsize: int = 8,
    loadings_annotation_fontsize: int = 8,
    cbar_label_fontsize: int = 14,
    cbar_tick_fontsize: int = 14,
    exclude_scalar_names: Optional[Set[str]] = None,
    subplots_left: Optional[float] = None,
) -> None:
    """
    Combined heatmap (single figure) with left-to-right blocks:
      1) correlation matrix (scalars × scalars)
      2) empty spacer column (omitted if ``include_factor_pca_blocks`` is False)
      3) factor loadings (scalars × factors)
      4) empty spacer column
      5) PCA component loadings (scalars × PCs)

    When ``include_factor_pca_blocks`` is False, only the correlation matrix is drawn (same row
    ordering and optional prefix filter as otherwise).

    Formatting is designed to match the existing correlation+loadings combined plot:
    RdBu_r cmap, vmin/vmax in [-1, 1], scalar human labels on y-axis, factor/PC labels on x-axis,
    and annotations for correlation lower triangle + factor loadings values.

    row_order:
      - ``prefix``: order scalars by diffusion family prefix (default).
      - ``max_factor_loading``: order by dominant factor (largest |loading|), then by |loading|
        on that factor within each factor group.

    allowed_prefixes:
      - If set, keep only scalars whose name prefix (before first ``_``) is in this set
        (e.g. DTI / DKI / GQI only, excluding NODDI, MAP-MRI ``map_*``, and RDI ``rdi_*``).

    dpi:
      - Resolution for the saved PNG (default 300).

    include_factor_pca_blocks:
      - If False, omit factor-loadings and PCA-loadings panels.

    axis_tick_fontsize, corr_annotation_fontsize, loadings_annotation_fontsize:
      - Font sizes for axis ticks, correlation matrix annotations, and loading annotations.

    cbar_label_fontsize, cbar_tick_fontsize:
      - Color bar label and tick sizes.

    exclude_scalar_names:
      - Optional set of scalar column names to drop after other filters (e.g. ``gqi_iso``).

    subplots_left:
      - Matplotlib ``subplots_adjust(left=...)``. If None, uses a wider default when
        ``include_factor_pca_blocks`` is False so y-axis labels are not clipped.
    """
    if allowed_prefixes is not None:
        def _pfx(n: str) -> str:
            return n.split("_", 1)[0] if "_" in n else n

        keep = [c for c in corr_df.columns if _pfx(c) in allowed_prefixes]
        keep = order_scalars_by_prefix(keep)
        if len(keep) < 2:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.axis("off")
            msg = f"Too few scalars after prefix filter (n={len(keep)}); need ≥2."
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=12)
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Warning: combined heatmap subset skipped ({msg}) -> {output_path}")
            return
        corr_df = corr_df.loc[keep, keep].copy()
        loadings_df = loadings_df[keep].copy()
        pca_loadings_df = pca_loadings_df[keep].copy()

    if exclude_scalar_names:
        excl = set(exclude_scalar_names)
        keep = [c for c in corr_df.columns if c not in excl]
        keep = order_scalars_by_prefix(keep)
        if len(keep) < 2:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.axis("off")
            msg = f"Too few scalars after exclude_scalar_names (n={len(keep)}); need ≥2."
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=12)
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Warning: combined heatmap subset skipped ({msg}) -> {output_path}")
            return
        corr_df = corr_df.loc[keep, keep].copy()
        loadings_df = loadings_df[keep].copy()
        pca_loadings_df = pca_loadings_df[keep].copy()

    prefix_cols = order_scalars_by_prefix(corr_df.columns)
    corr_df = corr_df.loc[prefix_cols, prefix_cols]
    loadings_df = loadings_df[prefix_cols]
    pca_loadings_df = pca_loadings_df[prefix_cols]

    if row_order == "max_factor_loading":
        ordered_cols = order_scalars_by_max_abs_factor_loading(loadings_df)
        corr_df = corr_df.loc[ordered_cols, ordered_cols]
        loadings_df = loadings_df[ordered_cols]
        pca_loadings_df = pca_loadings_df[ordered_cols]
    elif row_order == "prefix":
        ordered_cols = prefix_cols
    else:
        raise ValueError(f"Unknown row_order: {row_order!r}")

    n_scalars = len(ordered_cols)
    n_factors = loadings_df.shape[0]
    n_pcs = pca_loadings_df.shape[0]

    # Transpose so factors/PCs are columns (matching heatmap blocks)
    loadings_df_T = loadings_df.T  # (n_scalars, n_factors)
    pca_loadings_T = pca_loadings_df.T  # (n_scalars, n_pcs)

    if include_factor_pca_blocks:
        # Spacer columns (NaN values)
        spacer1 = pd.DataFrame(np.nan, index=ordered_cols, columns=[" "])
        spacer2 = pd.DataFrame(np.nan, index=ordered_cols, columns=["  "])

        # Block order: corr | spacer1 | factor loadings | spacer2 | PCA loadings
        combined_df = pd.concat([corr_df, spacer1, loadings_df_T, spacer2, pca_loadings_T], axis=1)

        # Column start indices (0-based) for each block in the combined_df
        corr_start = 0
        factor_start = n_scalars + 1
        pca_start = n_scalars + 2 + n_factors

        fig_width = max(18, 8 + n_scalars * 0.35 + n_factors * 0.7 + n_pcs * 0.7)
    else:
        combined_df = corr_df
        corr_start = 0
        factor_start = -1
        pca_start = -1
        # Wider canvas so long y-axis (human) labels are not clipped at default margins.
        fig_width = max(16, 9 + n_scalars * 0.5)

    cbar_label_text = (
        CORR_FACTOR_LOADING_CBAR_LABEL if include_factor_pca_blocks else "Pearson correlation"
    )

    # Figure
    fig_height = max(12, 8 + n_scalars * 0.1)

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="none", edgecolor="none", frameon=False)
    fig.patch.set_visible(False)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[20, 0.5], hspace=0.10)
    ax_main = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    vmin = -1.0
    vmax = 1.0
    sns.heatmap(
        combined_df,
        ax=ax_main,
        cbar_ax=cax,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": cbar_label_text, "orientation": "horizontal"},
        xticklabels=False,
        yticklabels=False,
    )

    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")

    # Load metadata for labels and colors
    try:
        colors_path = ospj(METADATA_DIR, "scalar_labels_to_colors.json")
        human_path = ospj(METADATA_DIR, "scalar_labels_to_human.json")
        with open(colors_path) as f:
            scalar_to_color = json.load(f)
        with open(human_path) as f:
            scalar_to_human = json.load(f)

        human_labels = [scalar_to_human.get(scalar_name, scalar_name) for scalar_name in ordered_cols]
        abbr_labels: List[str] = []
        for label in human_labels:
            if "(" in label and ")" in label:
                start = label.rfind("(") + 1
                end = label.rfind(")")
                abbr = label[start:end].strip()
                abbr_labels.append(abbr if abbr else label)
            else:
                abbr_labels.append(label)

        human_labels = [format_scalar_tick_label_for_mixed_fonts(l) for l in human_labels]
        abbr_labels = [format_scalar_tick_label_for_mixed_fonts(l) for l in abbr_labels]

        # Y-axis
        y_tick_positions = np.arange(0.5, n_scalars + 0.5)
        ax_main.yaxis.set_major_locator(FixedLocator(y_tick_positions))
        ax_main.set_yticklabels(human_labels, rotation=0, fontsize=axis_tick_fontsize)
        ax_main.tick_params(axis="y", which="major", length=0, width=0, left=False, labelleft=True)
        for tick, scalar_name in zip(ax_main.get_yticklabels(), ordered_cols):
            tick.set_color(scalar_to_color.get(scalar_name, "#000000"))

        # Right spine: scalar abbreviations (aligned with factor loadings / PCA / correlation rows)
        sec_y = ax_main.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
        sec_y.set_yticks(y_tick_positions)
        sec_y.set_yticklabels(abbr_labels, rotation=0, fontsize=axis_tick_fontsize)
        sec_y.tick_params(axis="y", which="major", length=0, width=0, right=True, labelright=True)
        for tick, scalar_name in zip(sec_y.get_yticklabels(), ordered_cols):
            tick.set_color(scalar_to_color.get(scalar_name, "#000000"))

        # X-axis labels
        x_tick_positions: List[float] = []
        x_tick_labels: List[str] = []

        # Correlation matrix columns: scalar abbreviations
        for i in range(n_scalars):
            x_tick_positions.append(corr_start + i + 0.5)
            x_tick_labels.append(abbr_labels[i])

        if include_factor_pca_blocks:
            # Factor columns
            factor_names = loadings_df.index.tolist()
            for i in range(n_factors):
                x_tick_positions.append(factor_start + i + 0.5)
                factor_label = factor_names[i]
                if factor_label.startswith("F") and factor_label[1:].isdigit():
                    factor_num = factor_label[1:]
                    x_tick_labels.append(f"Factor {factor_num}")
                else:
                    x_tick_labels.append(factor_label)

            # PCA columns (PC1..PCk)
            for i in range(n_pcs):
                x_tick_positions.append(pca_start + i + 0.5)
                x_tick_labels.append(f"PC{i + 1}")

        ax_main.xaxis.set_major_locator(FixedLocator(x_tick_positions))
        ax_main.set_xticklabels(x_tick_labels, rotation=45, fontsize=axis_tick_fontsize, ha="left")
        ax_main.tick_params(axis="x", which="major", length=0, width=0, bottom=False, labelbottom=False, top=True, labeltop=True)

        # Color x-axis tick labels for the correlation matrix part only
        for i in range(n_scalars):
            tick_idx = i  # first n_scalars entries are corr matrix columns
            tick = ax_main.get_xticklabels()[tick_idx]
            tick.set_color(scalar_to_color.get(ordered_cols[i], "#000000"))
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not apply scalar color/human labels to combined PCA figure: {e}")

    if include_factor_pca_blocks:
        # Annotate factor loadings block (all values)
        loadings_values = loadings_df_T.values  # (n_scalars, n_factors)
        for j in range(n_scalars):
            for i in range(n_factors):
                val = loadings_values[j, i]
                if np.isnan(val):
                    continue
                x_pos = factor_start + i + 0.5
                y_pos = j + 0.5
                text = f"{val:.2f}"
                text_color = "black" if abs(val) < 0.3 else "white"
                ax_main.text(
                    x_pos,
                    y_pos,
                    text,
                    ha="center",
                    va="center",
                    fontsize=loadings_annotation_fontsize,
                    color=text_color,
                    weight="bold" if abs(val) > 0.5 else "normal",
                )

        # Annotate PCA component loadings block (all values) and outline max-|loading| per row
        pca_loadings_values = pca_loadings_T.values  # (n_scalars, n_pcs)
        for j in range(n_scalars):
            for i in range(n_pcs):
                val = pca_loadings_values[j, i]
                if np.isnan(val):
                    continue
                x_pos = pca_start + i + 0.5
                y_pos = j + 0.5
                text = f"{val:.2f}"
                text_color = "black" if abs(val) < 0.3 else "white"
                ax_main.text(
                    x_pos,
                    y_pos,
                    text,
                    ha="center",
                    va="center",
                    fontsize=loadings_annotation_fontsize,
                    color=text_color,
                    weight="bold" if abs(val) > 0.5 else "normal",
                )

        abs_pca_vals = np.abs(pca_loadings_values)
        max_pca_cols = abs_pca_vals.argmax(axis=1)
        for j, i in enumerate(max_pca_cols):
            rect = Rectangle(
                (pca_start + i, j),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
            )
            ax_main.add_patch(rect)

    # Annotate correlation matrix lower triangle (below diagonal)
    corr_values = corr_df.loc[ordered_cols, ordered_cols].values  # (n_scalars, n_scalars)
    for j in range(n_scalars):
        for i in range(n_scalars):
            if j <= i:
                continue
            val = corr_values[j, i]
            if np.isnan(val):
                continue
            x_pos = corr_start + i + 0.5
            y_pos = j + 0.5
            text = f"{val:.2f}"
            text_color = "black" if abs(val) < 0.3 else "white"
            ax_main.text(
                x_pos,
                y_pos,
                text,
                ha="center",
                va="center",
                fontsize=corr_annotation_fontsize,
                color=text_color,
                weight="bold" if abs(val) > 0.5 else "normal",
            )

    if include_factor_pca_blocks:
        # Outline the cell with largest |loading| per row in the factor-loadings block
        abs_vals = loadings_df_T.abs().values
        max_cols = abs_vals.argmax(axis=1)
        for j, i in enumerate(max_cols):
            rect = Rectangle(
                (factor_start + i, j),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
            )
            ax_main.add_patch(rect)

    # Room for y-axis labels (left) and right-side abbreviation labels
    _adj_left = subplots_left if subplots_left is not None else (
        0.28 if not include_factor_pca_blocks else 0.20
    )
    fig.subplots_adjust(left=_adj_left, right=0.88, top=0.93, bottom=0.12, wspace=0.02)

    # Reposition colorbar: bottom center, slightly wider, and match sizing to axis text
    try:
        ax_pos = ax_main.get_position()
        cax_pos = cax.get_position()
        cbar_width = ax_pos.width * 0.4
        cbar_x0 = ax_pos.x0 + (ax_pos.width - cbar_width) / 2.0
        cbar_height = cax_pos.height * 0.6
        # Nudge the colorbar slightly lower relative to the axes.
        # Nudge the colorbar slightly lower relative to the axes.
        cbar_y0 = cax_pos.y0 - (cax_pos.height * 0.05)
        cax.set_position([cbar_x0, cbar_y0, cbar_width, cbar_height])

        cax.tick_params(axis="x", labelsize=cbar_tick_fontsize)
        try:
            cax.xaxis.label.set_size(cbar_label_fontsize)
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not reposition colorbar: {e}")

    fig.savefig(output_path, dpi=dpi, bbox_inches=None, facecolor="none", edgecolor="none", pad_inches=0)
    plt.close(fig)
    ord_desc = "factor-dominant order" if row_order == "max_factor_loading" else "prefix order"
    if include_factor_pca_blocks:
        print(
            f"Saved combined correlation + factor loadings + PCA components plot ({ord_desc}) to: {output_path}"
        )
    else:
        print(f"Saved correlation matrix heatmap ({ord_desc}) to: {output_path}")


def plot_factor_pca_combined_summary(
    eigenvalues: np.ndarray,
    optimal_n_factors: int,
    pca_cum_pct: np.ndarray,
    optimal_pc: int,
    pca_corr_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Combined figure:
      - Left column: 2x1 stack of factor scree (top) and PCA cumulative variance (bottom)
      - Right column: PCA vs factor loadings correlation heatmap
    Left and right columns use equal overall dimensions.
    """
    # Keep equal-width columns so the right panel matches the left 2x1 block area.
    fig = plt.figure(figsize=(14, 7))
    gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.22)
    gs_left = gs_outer[0, 0].subgridspec(2, 1, hspace=0.40)
    ax1 = fig.add_subplot(gs_left[0, 0])
    ax2 = fig.add_subplot(gs_left[1, 0])
    ax3 = fig.add_subplot(gs_outer[0, 1])

    # 1) Factor scree
    n_factors_total = len(eigenvalues)
    x1 = np.arange(1, n_factors_total + 1)
    y1 = eigenvalues
    ax1.plot(x1, y1, "o-", linewidth=2, markersize=8, color="black", label="_nolegend_")
    if optimal_n_factors <= n_factors_total:
        ax1.axvline(x=optimal_n_factors, color="red", linestyle="--", linewidth=2, label="_nolegend_")
        ax1.plot(
            optimal_n_factors,
            y1[optimal_n_factors - 1],
            "ro",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=2,
            zorder=5,
            label="Optimal number of factors",
        )
    ax1.set_xticks(x1)
    ax1.set_xticklabels([str(i) for i in x1], fontsize=10)
    ax1.set_xlabel("Factor Number", fontsize=14)
    ax1.set_ylabel("Eigenvalue", fontsize=14)
    ax1.set_title("Factor analysis scree plot", fontsize=16)
    ax1.grid(True, alpha=0.3, axis="y")

    y_max1 = float(np.max(y1)) if np.any(np.isfinite(y1)) else 1.0
    ax1.set_ylim(0, y_max1 * 1.15)
    ax1.set_aspect("auto")

    # 2) PCA cumulative explained variance
    y2 = pca_cum_pct.astype(float)
    n_pcs = len(y2)
    x2 = np.arange(1, n_pcs + 1)
    optimal_pc = int(np.clip(optimal_pc, 1, n_pcs))
    ax2.plot(x2, y2, "o-", linewidth=2, markersize=8, color="black", label="_nolegend_")
    ax2.axvline(x=optimal_pc, color="red", linestyle="--", linewidth=2, label="_nolegend_")
    ax2.plot(
        optimal_pc,
        y2[optimal_pc - 1],
        "ro",
        markersize=10,
        markeredgecolor="black",
        markeredgewidth=2,
        zorder=5,
        label="Optimal number of PCs",
    )
    ax2.set_xticks(x2)
    ax2.set_xticklabels([str(i) for i in x2], fontsize=10)
    ax2.set_xlabel("PC Number", fontsize=14)
    ax2.set_ylabel("Cumulative explained variance (%)", fontsize=14)
    ax2.set_title("PCA cumulative variance explained", fontsize=16)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(fontsize=11, loc="lower right", bbox_to_anchor=(1.0, 0.03))
    y_max2 = float(np.max(y2)) if np.any(np.isfinite(y2)) else 1.0
    ax2.set_ylim(0, min(105.0, max(5.0, y_max2 * 1.08)))
    ax2.set_aspect("auto")

    # 3) PCA vs factor correlations (rows = PCA components, columns = factors)
    n_pc_h = pca_corr_df.shape[0]
    n_f_h = pca_corr_df.shape[1]
    # Make correlation annotations more readable across varying numbers of PCs/factors.
    annot_fs = max(11, min(16, int(240 / max(n_pc_h, n_f_h, 1))))
    axis_label_fs = 16
    tick_fs = 16
    sns.heatmap(
        pca_corr_df,
        ax=ax3,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": annot_fs},
        square=True,
        cbar=False,
    )
    ax3.set_title("Factor analysis vs PCA loading correlations", fontsize=16)
    ax3.set_xlabel("Factors", fontsize=axis_label_fs)
    ax3.set_ylabel("PCA components", fontsize=axis_label_fs)
    ax3.tick_params(axis="both", which="major", labelsize=tick_fs)
    # seaborn(square=True) already enforces square cells within the allocated axes box

    # Outline the PC with max |r| within each factor column
    try:
        corr_vals = pca_corr_df.values  # (n_pc, n_factors)
        abs_vals = np.abs(corr_vals)
        max_pc_row_per_factor = abs_vals.argmax(axis=0)
        for j in range(abs_vals.shape[1]):
            i = int(max_pc_row_per_factor[j])
            rect = Rectangle(
                (j, i),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
            )
            ax3.add_patch(rect)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not outline PC max-correlation cells: {e}")

    ax1.legend(fontsize=11, loc="lower right", bbox_to_anchor=(1.0, 0.03))
    plt.tight_layout(pad=0.15, w_pad=0.20, h_pad=0.20)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved combined factor/PCA summary plot to: {output_path}")


def load_tissue_pc1_variance_explained_percent(group_label: str, tissue_run: str) -> Optional[float]:
    """
    Return PCA PC1 percent variance explained for a tissue atlas run, if
    ``{group_label}_{tissue_run}_pca_explained_variance_ratio.csv`` exists under
    ``OUTPUT_PROJECT_ROOT / tissue_run``.
    """
    csv_path = ospj(
        OUTPUT_PROJECT_ROOT,
        tissue_run,
        f"{group_label}_{tissue_run}_pca_explained_variance_ratio.csv",
    )
    if not os.path.exists(csv_path):
        return None
    try:
        ev_df = pd.read_csv(csv_path)
        if ev_df.empty or "variance_percent" not in ev_df.columns:
            return None
        if "component" in ev_df.columns:
            row = ev_df[ev_df["component"].astype(str) == "PC1"]
            if not row.empty:
                return float(row["variance_percent"].iloc[0])
        return float(ev_df["variance_percent"].iloc[0])
    except Exception:  # noqa: BLE001
        return None


def _pearson_corr_series(a: pd.Series, b: pd.Series) -> float:
    """Pearson r between two vectors stored as pandas Series over scalar names."""
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return float("nan")
    a_vals = a[common].values.astype(float)
    b_vals = b[common].values.astype(float)
    mask = np.isfinite(a_vals) & np.isfinite(b_vals)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a_vals[mask], b_vals[mask])[0, 1])


def plot_tissue_pc1_correlations_with_wholebrain(
    *,
    group_label: str,
    all4_output_dir: str,
    all4_factor_loadings_csv_path: str,
    all4_pca_component_loadings_csv_path: str,
    tissue_run_order: List[str],
    tissue_display_names: List[str],
    output_path_factors: str,
    output_path_pcs: str,
    target_count: int = 4,
    force_ncols: int | None = None,
    use_absolute_correlations: bool = False,
    add_gm_wm_abs_diff_subplot: bool = False,
) -> None:
    """
    Build two plots (saved to `output_path_factors` and `output_path_pcs`):
      1) Tissue PC1 loadings correlated with whole-brain factors (F1..F4)
      2) Tissue PC1 loadings correlated with whole-brain PCs (PC1..PC4)
    Each plot is a 1xn grid (one subplot per tissue type).
    """
    whole_factors_df = pd.read_csv(all4_factor_loadings_csv_path, index_col=0)
    whole_pcs_df = pd.read_csv(all4_pca_component_loadings_csv_path, index_col=0)

    # Determine how many targets we actually have.
    whole_factor_labels = whole_factors_df.index.tolist()[:target_count]
    whole_pc_labels = whole_pcs_df.index.tolist()[:target_count]
    n_factors = len(whole_factor_labels)
    n_pcs = len(whole_pc_labels)

    if n_factors == 0 and n_pcs == 0:
        print("Warning: whole-brain factors/PCs missing; skipping tissue-PC1 plots.")
        return

    # Gather tissue PC1 vectors.
    tissue_pc1_by_run: Dict[str, pd.Series] = {}
    for tissue_run in tissue_run_order:
        tissue_output_dir = ospj(OUTPUT_PROJECT_ROOT, tissue_run)
        tissue_file_prefix = f"{group_label}_{tissue_run}"
        tissue_pca_csv = ospj(tissue_output_dir, f"{tissue_file_prefix}_pca_component_loadings.csv")
        if not os.path.exists(tissue_pca_csv):
            print(f"Warning: missing tissue PCA loadings CSV: {tissue_pca_csv}")
            continue
        tissue_pca_df = pd.read_csv(tissue_pca_csv, index_col=0)
        if "PC1" not in tissue_pca_df.index:
            print(f"Warning: tissue PCA CSV missing PC1 row: {tissue_pca_csv}")
            continue
        tissue_pc1_by_run[tissue_run] = tissue_pca_df.loc["PC1"]

    if not tissue_pc1_by_run:
        print("Warning: no tissue PC1 vectors found; skipping tissue-PC1 plots.")
        return

    n_tissues = len(tissue_run_order)
    extra_panels = 1 if add_gm_wm_abs_diff_subplot else 0
    n_panels = n_tissues + extra_panels
    ncols = force_ncols if force_ncols is not None else n_tissues
    ncols = max(1, min(ncols, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    # GM/WM combined + diff: single row (1×3) uses narrower columns; stacked layouts use taller figure.
    if add_gm_wm_abs_diff_subplot and nrows == 1:
        fig_width = max(7.0, 3.0 * ncols)
        # Taller to fit two-line panel titles (PC1 + % variance explained for GM/WM).
        fig_height = 5.0
    else:
        fig_width = max(6.0, 5.0 * ncols)
        fig_height = max(4.8, 3.8 * nrows)

    # Shared font sizing to match Factor/PCA Summary plots.
    title_fs = 16
    axis_fs = 14
    tick_fs = 10
    ylabel_fs = 14

    # Plot 1: tissue PC1 vs whole-brain factors
    fig_f, axes_f = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharey=not add_gm_wm_abs_diff_subplot)
    axes_f = np.atleast_1d(axes_f).ravel()
    for ax in axes_f:
        if use_absolute_correlations:
            ax.set_ylim(0.0, 1.0)
        else:
            ax.set_ylim(-1.0, 1.0)

    factor_vals_by_run: Dict[str, List[float]] = {}
    for i, tissue_run in enumerate(tissue_run_order):
        ax = axes_f[i]
        tissue_label = tissue_display_names[i] if i < len(tissue_display_names) else tissue_run
        factor_panel_title = f"{tissue_label} PC1"
        if add_gm_wm_abs_diff_subplot and tissue_run in ("GM_Combined", "WM_Combined"):
            vpct = load_tissue_pc1_variance_explained_percent(group_label, tissue_run)
            if vpct is not None:
                factor_panel_title = f"{tissue_label} PC1\n{vpct:.1f}% variance explained"
        if tissue_run not in tissue_pc1_by_run or n_factors == 0:
            ax.set_title(factor_panel_title, fontsize=title_fs)
            ax.axis("off")
            continue

        tissue_pc1 = tissue_pc1_by_run[tissue_run]
        vals = []
        for fac_lab in whole_factor_labels:
            vals.append(_pearson_corr_series(tissue_pc1, whole_factors_df.loc[fac_lab]))
        vals_arr = np.asarray(vals, dtype=float)
        if use_absolute_correlations:
            vals_arr = np.abs(vals_arr)
        factor_vals_by_run[tissue_run] = vals_arr.tolist()
        ax.bar(whole_factor_labels, vals_arr, color="#9E9E9E", edgecolor="black", linewidth=0.3, alpha=0.9)
        ax.set_title(factor_panel_title, fontsize=title_fs)
        ax.set_xlabel("Factor", fontsize=axis_fs)
        ax.tick_params(axis="x", labelsize=tick_fs)
        if i == 0:
            ax.set_ylabel("|Pearson correlation|" if use_absolute_correlations else "Pearson correlation", fontsize=ylabel_fs)
        else:
            ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.3)

    if add_gm_wm_abs_diff_subplot and len(axes_f) > n_tissues:
        ax_diff = axes_f[n_tissues]
        gm_vals = np.asarray(factor_vals_by_run.get("GM_Combined", []), dtype=float)
        wm_vals = np.asarray(factor_vals_by_run.get("WM_Combined", []), dtype=float)
        if gm_vals.size == len(whole_factor_labels) and wm_vals.size == len(whole_factor_labels):
            diff_vals = gm_vals - wm_vals
            ax_diff.bar(whole_factor_labels, diff_vals, color="#9E9E9E", edgecolor="black", linewidth=0.3, alpha=0.9)
            ax_diff.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
            ax_diff.set_title("Grey-white matter differences", fontsize=title_fs)
            ax_diff.set_xlabel("Factor", fontsize=axis_fs)
            ax_diff.set_ylabel("|Pearson correlation| difference", fontsize=ylabel_fs)
            ax_diff.tick_params(axis="x", labelsize=tick_fs)
            ymax = float(np.nanmax(np.abs(diff_vals))) if np.any(np.isfinite(diff_vals)) else 0.1
            ymax = max(0.1, ymax * 1.15)
            ax_diff.set_ylim(-ymax, ymax)
            ax_diff.grid(True, axis="y", alpha=0.3)
        else:
            ax_diff.axis("off")

    for k in range(n_panels, len(axes_f)):
        axes_f[k].axis("off")
    plt.tight_layout()
    fig_f.savefig(output_path_factors, dpi=150, bbox_inches="tight")
    plt.close(fig_f)

    # Plot 2: tissue PC1 vs whole-brain PCs
    fig_p, axes_p = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharey=not add_gm_wm_abs_diff_subplot)
    axes_p = np.atleast_1d(axes_p).ravel()
    for ax in axes_p:
        if use_absolute_correlations:
            ax.set_ylim(0.0, 1.0)
        else:
            ax.set_ylim(-1.0, 1.0)

    pca_vals_by_run: Dict[str, List[float]] = {}
    for i, tissue_run in enumerate(tissue_run_order):
        ax = axes_p[i]
        tissue_label = tissue_display_names[i] if i < len(tissue_display_names) else tissue_run
        pca_panel_title = f"{tissue_label} PC1 correlations\nwith whole-brain PCA loadings"
        if tissue_run not in tissue_pc1_by_run or n_pcs == 0:
            ax.set_title(pca_panel_title, fontsize=title_fs)
            ax.axis("off")
            continue

        tissue_pc1 = tissue_pc1_by_run[tissue_run]
        vals = []
        for pc_lab in whole_pc_labels:
            vals.append(_pearson_corr_series(tissue_pc1, whole_pcs_df.loc[pc_lab]))
        vals_arr = np.asarray(vals, dtype=float)
        if use_absolute_correlations:
            vals_arr = np.abs(vals_arr)
        pca_vals_by_run[tissue_run] = vals_arr.tolist()
        ax.bar(whole_pc_labels, vals_arr, color="#9E9E9E", edgecolor="black", linewidth=0.3, alpha=0.9)
        ax.set_title(pca_panel_title, fontsize=title_fs)
        ax.set_xlabel("PC", fontsize=axis_fs)
        ax.tick_params(axis="x", labelsize=tick_fs)
        if i == 0:
            ax.set_ylabel("|Pearson correlation|" if use_absolute_correlations else "Pearson correlation", fontsize=ylabel_fs)
        else:
            ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.3)

    if add_gm_wm_abs_diff_subplot and len(axes_p) > n_tissues:
        ax_diff = axes_p[n_tissues]
        gm_vals = np.asarray(pca_vals_by_run.get("GM_Combined", []), dtype=float)
        wm_vals = np.asarray(pca_vals_by_run.get("WM_Combined", []), dtype=float)
        if gm_vals.size == len(whole_pc_labels) and wm_vals.size == len(whole_pc_labels):
            diff_vals = gm_vals - wm_vals
            ax_diff.bar(whole_pc_labels, diff_vals, color="#9E9E9E", edgecolor="black", linewidth=0.3, alpha=0.9)
            ax_diff.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
            ax_diff.set_title("Grey matter-White matter PC1\nabsolute correlation differences", fontsize=title_fs)
            ax_diff.set_xlabel("PC", fontsize=axis_fs)
            ax_diff.set_ylabel("|Pearson Correlation| difference", fontsize=ylabel_fs)
            ax_diff.tick_params(axis="x", labelsize=tick_fs)
            ymax = float(np.nanmax(np.abs(diff_vals))) if np.any(np.isfinite(diff_vals)) else 0.1
            ymax = max(0.1, ymax * 1.15)
            ax_diff.set_ylim(-ymax, ymax)
            ax_diff.grid(True, axis="y", alpha=0.3)
        else:
            ax_diff.axis("off")

    for k in range(n_panels, len(axes_p)):
        axes_p[k].axis("off")
    plt.tight_layout()
    fig_p.savefig(output_path_pcs, dpi=150, bbox_inches="tight")
    plt.close(fig_p)


def plot_corr_and_ica_combined(
    corr_df: pd.DataFrame,
    ica_loadings_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create a single heatmap showing:
      - ICA component loadings as columns on the left (scalars × IC1–ICk)
      - Pairwise correlation matrix on the right
    ica_loadings_df has shape (n_components, n_scalars), index IC1, IC2, ..., columns = scalar names.
    """
    # Shared scalar ordering
    ordered_cols = order_scalars_by_prefix(corr_df.columns)
    corr_df = corr_df.loc[ordered_cols, ordered_cols]
    ica_loadings_df = ica_loadings_df[ordered_cols]

    n_scalars = len(ordered_cols)
    n_components = ica_loadings_df.shape[0]

    # Transpose so components are columns (to appear on the left)
    ica_loadings_T = ica_loadings_df.T  # (n_scalars, n_components)

    # Create separator column (NaN values) to add space between loadings and correlation matrix
    separator_col = pd.DataFrame(
        np.nan,
        index=ordered_cols,
        columns=[" "],
    )

    # Combine: ICA loadings (left) + separator + correlation matrix (right)
    combined_df = pd.concat([ica_loadings_T, separator_col, corr_df], axis=1)

    fig_width = max(18, 10 + n_components * 0.55 + n_scalars * 0.08)
    fig_height = max(12, 8 + n_scalars * 0.12)

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='none', edgecolor='none', frameon=False)
    fig.patch.set_visible(False)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[20, 0.5], hspace=0.10)
    ax_main = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    vmin = -1.0
    vmax = 1.0
    sns.heatmap(
        combined_df,
        ax=ax_main,
        cbar_ax=cax,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "PCA component correlations / correlations", "orientation": "horizontal"},
        xticklabels=False,
        yticklabels=False,
    )

    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    try:
        colors_path = ospj(METADATA_DIR, "scalar_labels_to_colors.json")
        human_path = ospj(METADATA_DIR, "scalar_labels_to_human.json")
        with open(colors_path) as f:
            scalar_to_color = json.load(f)
        with open(human_path) as f:
            scalar_to_human = json.load(f)

        human_labels = [
            scalar_to_human.get(scalar_name, scalar_name)
            for scalar_name in ordered_cols
        ]

        abbr_labels: List[str] = []
        for label in human_labels:
            if "(" in label and ")" in label:
                start = label.rfind("(") + 1
                end = label.rfind(")")
                abbr = label[start:end].strip()
                abbr_labels.append(abbr if abbr else label)
            else:
                abbr_labels.append(label)

        human_labels = [format_scalar_tick_label_for_mixed_fonts(l) for l in human_labels]
        abbr_labels = [format_scalar_tick_label_for_mixed_fonts(l) for l in abbr_labels]

        y_tick_positions = np.arange(0.5, n_scalars + 0.5)
        ax_main.yaxis.set_major_locator(FixedLocator(y_tick_positions))
        ax_main.set_yticklabels(human_labels, rotation=0, fontsize=12)
        ax_main.tick_params(axis='y', which='major', length=0, width=0, left=False, labelleft=True)
        for tick, scalar_name in zip(ax_main.get_yticklabels(), ordered_cols):
            tick.set_color(scalar_to_color.get(scalar_name, "#000000"))

        x_tick_positions = []
        x_tick_labels = []

        # Component names: use provided row index (e.g., PC1, PC2, ...)
        for i in range(n_components):
            x_tick_positions.append(i + 0.5)
            comp_label = ica_loadings_df.index[i]
            x_tick_labels.append(comp_label)

        for i in range(n_scalars):
            x_tick_positions.append(n_components + 1 + i + 0.5)
            x_tick_labels.append(abbr_labels[i])

        ax_main.xaxis.set_major_locator(FixedLocator(x_tick_positions))
        ax_main.set_xticklabels(x_tick_labels, rotation=45, fontsize=12, ha='left')
        ax_main.tick_params(axis='x', which='major', length=0, width=0, bottom=False, labelbottom=False, top=True, labeltop=True)

        for i in range(n_scalars):
            tick_idx = n_components + i
            tick = ax_main.get_xticklabels()[tick_idx]
            tick.set_color(scalar_to_color.get(ordered_cols[i], "#000000"))

    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not apply scalar color/human labels to ICA combined plot: {e}")

    # Annotate ICA loadings section
    loadings_values = ica_loadings_T.values  # (n_scalars, n_components)
    for j in range(n_scalars):
        for i in range(n_components):
            val = loadings_values[j, i]
            if not np.isnan(val):
                x_pos = i + 0.5
                y_pos = j + 0.5
                text = f"{val:.2f}"
                text_color = 'black' if abs(val) < 0.3 else 'white'
                ax_main.text(
                    x_pos, y_pos, text,
                    ha='center', va='center',
                    fontsize=8,
                    color=text_color,
                    weight='bold' if abs(val) > 0.5 else 'normal',
                )

    # Annotate lower triangle of correlation matrix
    corr_start_x = n_components + 1
    corr_values = corr_df.loc[ordered_cols, ordered_cols].values
    for j in range(n_scalars):
        for i in range(n_scalars):
            if j > i:
                val = corr_values[j, i]
                if not np.isnan(val):
                    x_pos = corr_start_x + i + 0.5
                    y_pos = j + 0.5
                    text = f"{val:.2f}"
                    text_color = 'black' if abs(val) < 0.3 else 'white'
                    ax_main.text(
                        x_pos, y_pos, text,
                        ha='center', va='center',
                        fontsize=8,
                        color=text_color,
                        weight='bold' if abs(val) > 0.5 else 'normal',
                    )

    # Outline cell with largest |loading| per row in ICA section
    abs_vals = ica_loadings_T.abs().values
    max_cols = abs_vals.argmax(axis=1)
    for j, i in enumerate(max_cols):
        rect = Rectangle(
            (i, j),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
        ax_main.add_patch(rect)

    ax_main.set_ylabel("Diffusion statistic", fontsize=14)
    ax_main.set_xlabel("")
    cax.set_xlabel("PCA component correlations / correlations", fontsize=14)

    legend_handles = reconstruction_model_legend_handles()

    leg = ax_main.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.10),
        ncol=5,
        fontsize=14,
        title="Reconstruction model",
        frameon=True,
        bbox_transform=ax_main.transAxes,
        fancybox=True,
        borderpad=0.3,
        columnspacing=0.5,
        handletextpad=0.3,
    )
    if leg.get_title() is not None:
        leg.get_title().set_fontsize(14)
    leg.get_frame().set_facecolor("#F6F6FA")
    leg.get_frame().set_edgecolor("#CCCCCC")
    leg.get_frame().set_linewidth(1.5)

    fig.subplots_adjust(left=0.20, right=0.95, top=0.93, bottom=0.12, wspace=0.02)

    ax_pos = ax_main.get_position()
    cax_pos = cax.get_position()
    cbar_width = ax_pos.width * 0.4
    cbar_x0 = ax_pos.x0 + ax_pos.width - cbar_width
    cbar_height = cax_pos.height * 0.5
    cbar_y0 = cax_pos.y0
    cax.set_position([cbar_x0, cbar_y0, cbar_width, cbar_height])

    fig.savefig(output_path, dpi=300, bbox_inches=None, facecolor='none', edgecolor='none', pad_inches=0)
    plt.close(fig)
    print(f"Saved combined correlation + PCA loadings plot to: {output_path}")


def plot_corr_matrix_minimal(
    corr_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create a minimal correlation matrix heatmap with:
    - Only the correlation matrix (no factor loadings)
    - Same colors as other heatmap (RdBu_r, vmin=-1.0, vmax=1.0)
    - No labeled correlation values
    - Only scalar abbreviations for x- and y-axes
    """
    # Order scalars by prefix
    ordered_cols = order_scalars_by_prefix(corr_df.columns)
    corr_df_ordered = corr_df.loc[ordered_cols, ordered_cols]
    
    n_scalars = len(ordered_cols)
    
    # Calculate figure size (square for correlation matrix)
    fig_width = 12
    fig_height = 12
    
    # Create figure with GridSpec for main plot and horizontal colorbar at bottom
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='none', edgecolor='none', frameon=False)
    fig.patch.set_visible(False)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[20, 0.5], hspace=0.10)
    ax_main = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])
    
    # Create heatmap with same settings as combined plot
    vmin = -1.0
    vmax = 1.0
    sns.heatmap(
        corr_df_ordered,
        ax=ax_main,
        cbar_ax=cax,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation", "orientation": "horizontal"},
        xticklabels=False,
        yticklabels=False,
        annot=False,  # No annotations
    )
    
    # Move colorbar tick labels above the bar
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    
    # Load metadata for labels and colors
    try:
        colors_path = ospj(METADATA_DIR, "scalar_labels_to_colors.json")
        human_path = ospj(METADATA_DIR, "scalar_labels_to_human.json")
        with open(colors_path) as f:
            scalar_to_color = json.load(f)
        with open(human_path) as f:
            scalar_to_human = json.load(f)
        
        human_labels = [
            scalar_to_human.get(scalar_name, scalar_name)
            for scalar_name in ordered_cols
        ]
        
        # Abbreviated labels for both axes
        abbr_labels: List[str] = []
        for label in human_labels:
            if "(" in label and ")" in label:
                start = label.rfind("(") + 1
                end = label.rfind(")")
                abbr = label[start:end].strip()
                abbr_labels.append(abbr if abbr else label)
            else:
                abbr_labels.append(label)

        abbr_labels = [format_scalar_tick_label_for_mixed_fonts(l) for l in abbr_labels]
        
        # Y-AXIS: Set custom tick positions and labels (abbreviations only)
        y_tick_positions = np.arange(0.5, n_scalars + 0.5)
        ax_main.yaxis.set_major_locator(FixedLocator(y_tick_positions))
        ax_main.set_yticklabels(abbr_labels, rotation=0, fontsize=12)
        ax_main.tick_params(axis='y', which='major', length=0, width=0, left=False, labelleft=True)
        for tick, scalar_name in zip(ax_main.get_yticklabels(), ordered_cols):
            tick.set_color(scalar_to_color.get(scalar_name, "#000000"))
        
        # X-AXIS: Set custom tick positions and labels (abbreviations only)
        x_tick_positions = np.arange(0.5, n_scalars + 0.5)
        ax_main.xaxis.set_major_locator(FixedLocator(x_tick_positions))
        ax_main.set_xticklabels(abbr_labels, rotation=45, fontsize=12, ha='left')
        ax_main.tick_params(axis='x', which='major', length=0, width=0, bottom=False, labelbottom=False, top=True, labeltop=True)
        
        # Color x-axis labels
        for tick, scalar_name in zip(ax_main.get_xticklabels(), ordered_cols):
            tick.set_color(scalar_to_color.get(scalar_name, "#000000"))
    
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not apply scalar color/human labels to minimal correlation plot: {e}")
        # Fallback: use scalar names directly
        y_tick_positions = np.arange(0.5, n_scalars + 0.5)
        ax_main.yaxis.set_major_locator(FixedLocator(y_tick_positions))
        ax_main.set_yticklabels(ordered_cols, rotation=0, fontsize=12)
        ax_main.tick_params(axis='y', which='major', length=0, width=0, left=False, labelleft=True)
        
        x_tick_positions = np.arange(0.5, n_scalars + 0.5)
        ax_main.xaxis.set_major_locator(FixedLocator(x_tick_positions))
        ax_main.set_xticklabels(ordered_cols, rotation=45, fontsize=12, ha='left')
        ax_main.tick_params(axis='x', which='major', length=0, width=0, bottom=False, labelbottom=False, top=True, labeltop=True)
    
    # Set axis labels
    ax_main.set_ylabel("Diffusion statistic", fontsize=14)
    ax_main.set_xlabel("Diffusion statistic", fontsize=14)
    cax.set_xlabel("Correlation", fontsize=14)
    
    # Tight margins
    fig.subplots_adjust(left=0.20, right=0.95, top=0.93, bottom=0.12, wspace=0.02)
    
    # Position colorbar at bottom
    ax_pos = ax_main.get_position()
    cax_pos = cax.get_position()
    cbar_width = ax_pos.width * 0.4
    cbar_x0 = ax_pos.x0 + ax_pos.width - cbar_width
    cbar_height = cax_pos.height * 0.5
    cbar_y0 = cax_pos.y0
    cax.set_position([cbar_x0, cbar_y0, cbar_width, cbar_height])
    
    fig.savefig(output_path, dpi=300, bbox_inches=None, facecolor='none', edgecolor='none', pad_inches=0)
    plt.close(fig)
    print(f"Saved minimal correlation matrix plot to: {output_path}")


def plot_factor_loadings_standalone(
    loadings_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create a standalone factor loadings heatmap with:
    - Rows = factors, Columns = diffusion statistics
    - Columns ordered by dominant factor (max |loading|), then |loading| within factor
    - Scalar labels on the x-axis (bottom), factor labels on the y-axis
    - Square cells with loading values annotated; max-|loading| cell per scalar outlined
    - Horizontal factor-loading colorbar at the top left of the heatmap
    """
    ordered_cols = order_scalars_by_max_abs_factor_loading(loadings_df)
    loadings_df_ordered = loadings_df[ordered_cols]
    
    n_scalars = len(ordered_cols)
    n_factors = loadings_df_ordered.shape[0]
    
    # Calculate figure size - wider for more scalars, taller for more factors
    # Increased base sizes to accommodate larger labels and prevent clipping
    base_width_per_scalar = 0.8  # Increased from 0.6
    base_height_per_factor = 0.8
    fig_width = max(20, n_scalars * base_width_per_scalar)  # Increased minimum from 16
    fig_height = max(8, n_factors * base_height_per_factor)  # Increased minimum from 6
    
    loadings_annotation_fontsize = 12
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='none', edgecolor='none', frameon=False)
    fig.patch.set_visible(False)
    ax_main = fig.add_subplot(111)
    cax = fig.add_axes([0.20, 0.88, 0.22, 0.025])  # repositioned after layout (top left)

    vmin = -1.0
    vmax = 1.0
    sns.heatmap(
        loadings_df_ordered,
        ax=ax_main,
        cbar_ax=cax,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Factor loading", "orientation": "horizontal"},
        xticklabels=False,
        yticklabels=False,
        annot=False,
    )

    cax.set_xticks([-1, 0, 1])
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cax.tick_params(axis="x", labelsize=20)
    
    # Load metadata for labels and colors
    try:
        colors_path = ospj(METADATA_DIR, "scalar_labels_to_colors.json")
        human_path = ospj(METADATA_DIR, "scalar_labels_to_human.json")
        with open(colors_path) as f:
            scalar_to_color = json.load(f)
        with open(human_path) as f:
            scalar_to_human = json.load(f)
        
        human_labels = [
            scalar_to_human.get(scalar_name, scalar_name)
            for scalar_name in ordered_cols
        ]
        
        # Use full human labels (not abbreviations) for x-axis
        x_labels = [format_scalar_tick_label_for_mixed_fonts(l) for l in human_labels]
        
        # Fixed font size for x-axis labels
        x_fontsize = 20
        
        # Y-AXIS: Factor labels
        y_tick_positions = np.arange(0.5, n_factors + 0.5)
        ax_main.yaxis.set_major_locator(FixedLocator(y_tick_positions))
        
        factor_labels = [
            factor_name_to_diffusivity_label(factor_name)
            for factor_name in loadings_df_ordered.index
        ]
        ax_main.set_yticklabels(factor_labels, rotation=0, fontsize=20)
        ax_main.tick_params(axis='y', which='major', length=0, width=0, left=False, labelleft=True)
        
        # X-AXIS: full scalar labels on bottom
        x_tick_positions = np.arange(0.5, n_scalars + 0.5)
        ax_main.xaxis.set_major_locator(FixedLocator(x_tick_positions))
        ax_main.set_xticklabels(x_labels, rotation=45, fontsize=x_fontsize, ha="right")
        ax_main.tick_params(axis="x", which="major", length=0, width=0, bottom=True, labelbottom=True, top=False, labeltop=False)
        
        # Color x-axis labels
        for tick, scalar_name in zip(ax_main.get_xticklabels(), ordered_cols):
            tick.set_color(scalar_to_color.get(scalar_name, "#000000"))
    
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not apply scalar color/human labels to factor loadings plot: {e}")
        # Fallback: use scalar names directly
        y_tick_positions = np.arange(0.5, n_factors + 0.5)
        ax_main.yaxis.set_major_locator(FixedLocator(y_tick_positions))
        
        factor_labels = [
            factor_name_to_diffusivity_label(factor_name)
            for factor_name in loadings_df_ordered.index
        ]
        ax_main.set_yticklabels(factor_labels, rotation=0, fontsize=20)
        ax_main.tick_params(axis='y', which='major', length=0, width=0, left=False, labelleft=True)
        
        x_tick_positions = np.arange(0.5, n_scalars + 0.5)
        ax_main.xaxis.set_major_locator(FixedLocator(x_tick_positions))
        x_fontsize = 20
        ax_main.set_xticklabels(ordered_cols, rotation=45, fontsize=x_fontsize, ha="right")
        ax_main.tick_params(axis="x", which="major", length=0, width=0, bottom=True, labelbottom=True, top=False, labeltop=False)

    # Annotate loading values in each cell
    loadings_values = loadings_df_ordered.values  # (n_factors, n_scalars)
    for i in range(n_factors):
        for j in range(n_scalars):
            val = loadings_values[i, j]
            if not np.isnan(val):
                text_color = "black" if abs(val) < 0.3 else "white"
                ax_main.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=loadings_annotation_fontsize,
                    color=text_color,
                    weight="bold" if abs(val) > 0.5 else "normal",
                )

    # Outline, for each scalar (column), the factor row with the largest |loading|
    abs_vals = np.abs(loadings_values)
    max_rows = abs_vals.argmax(axis=0)
    for j, i in enumerate(max_rows):
        rect = Rectangle(
            (j, i),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
        ax_main.add_patch(rect)

    # Remove axis labels
    ax_main.set_ylabel("", fontsize=18)
    ax_main.set_xlabel("", fontsize=18)
    cax.set_xlabel("Factor loading", fontsize=20)

    # Room for y-axis factor labels, bottom scalar labels, and top-left colorbar
    fig.subplots_adjust(left=0.20, right=0.95, top=0.88, bottom=0.28)

    ax_pos = ax_main.get_position()
    cbar_width = ax_pos.width * 0.28
    cbar_height = 0.022
    cbar_gap = 0.012
    cbar_x0 = ax_pos.x0
    cbar_y0 = ax_pos.y1 + cbar_gap
    cax.set_position([cbar_x0, cbar_y0, cbar_width, cbar_height])
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f"Saved standalone factor loadings heatmap to: {output_path}")


def find_elbow(eigenvalues: np.ndarray) -> int:
    """
    Find the elbow point in eigenvalues using the elbow method (scree plot).
    
    The elbow method finds the point where adding more factors provides diminishing returns.
    This is done by finding the point that maximizes the distance from the line connecting
    the first and last points.
    
    Args:
        eigenvalues: Array of eigenvalues in descending order
    
    Returns:
        Optimal number of factors = number of eigenvalues above the elbow (not including elbow itself)
        Returns the number of factors to retain (1-indexed)
    """
    n_points = len(eigenvalues)
    if n_points < 2:
        return 1
    
    y = eigenvalues
    x = np.arange(1, n_points + 1)
    
    # Line from first to last point
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    
    # Distance from each point to the line
    # Distance = |(y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1| / sqrt((y2-y1)^2 + (x2-x1)^2)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    if denominator == 0:
        return 1
    
    distances = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denominator
    
    # Find the point with maximum distance (elbow)
    elbow_idx = np.argmax(distances)  # 0-indexed position of elbow
    
    # Optimal number of factors = number of eigenvalues above the elbow (not including elbow itself)
    # If elbow is at position i (0-indexed), we keep factors 1 through i (i factors, 1-indexed)
    optimal_factors = elbow_idx  # This is the number of factors above the elbow (0-indexed count)
    
    # Ensure at least 1 factor
    if optimal_factors < 1:
        optimal_factors = 1
    
    return optimal_factors


def plot_scree_eigenvalues(
    eigenvalues: np.ndarray,
    optimal_n_factors: int,
    output_path: str,
) -> None:
    """
    Plot scree plot of eigenvalues in descending order.
    
    Args:
        eigenvalues: Array of eigenvalues in descending order
        optimal_n_factors: Optimal number of factors determined by elbow method
        output_path: Path to save the plot
    """
    n_factors = len(eigenvalues)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(1, n_factors + 1)
    y = eigenvalues
    
    # Plot line
    ax.plot(x, y, 'o-', linewidth=2, markersize=8, color='#4CAF50', label='Eigenvalues')
    
    # Highlight optimal number of factors
    if optimal_n_factors <= n_factors:
        ax.axvline(x=optimal_n_factors, color='red', linestyle='--', linewidth=2, 
                   label=f'Optimal: {optimal_n_factors} factors')
        # Mark the optimal point
        ax.plot(optimal_n_factors, y[optimal_n_factors - 1], 'ro', markersize=12, 
                markeredgecolor='black', markeredgewidth=2, zorder=5)
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=12)
    
    ax.set_xlabel('Factor Number', fontsize=16)
    ax.set_ylabel('Eigenvalue', fontsize=16)
    ax.set_title('Scree plot', fontsize=18, fontweight='bold')
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits
    ax.set_ylim(0, max(y) * 1.15)
    ax.set_xlim(0.5, n_factors + 0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved eigenvalues scree plot to: {output_path}")


def plot_scalar_std(
    X: np.ndarray,
    scalar_order: List[str],
    output_path: str,
) -> None:
    """
    Plot standard deviation of the feature vector per scalar, sorted ascending.
    Bars are colored by model (scalar_labels_to_colors) and labeled with human-readable names.
    X: (n_observations, n_scalars); each column is one scalar's feature vector.
    """
    if X.size == 0 or not scalar_order:
        return
    stds = np.nanstd(X, axis=0, ddof=1)
    stds = np.where(np.isfinite(stds), stds, 0.0)
    # Sort ascending: smallest std first
    sort_idx = np.argsort(stds)
    stds_sorted = stds[sort_idx]
    scalars_sorted = [scalar_order[i] for i in sort_idx]

    scalar_to_color: Dict[str, str] = {}
    scalar_to_human: Dict[str, str] = {}
    try:
        colors_path = ospj(METADATA_DIR, "scalar_labels_to_colors.json")
        human_path = ospj(METADATA_DIR, "scalar_labels_to_human.json")
        with open(colors_path) as f:
            scalar_to_color = json.load(f)
        with open(human_path) as f:
            scalar_to_human = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not load scalar colors/labels for std plot: {e}")

    human_labels = [format_scalar_tick_label_for_mixed_fonts(scalar_to_human.get(s, s)) for s in scalars_sorted]
    colors = [scalar_to_color.get(s, "#808080") for s in scalars_sorted]

    n = len(scalars_sorted)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.35), 6))
    x_pos = np.arange(n)
    ax.bar(x_pos, stds_sorted, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(human_labels, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Scalar", fontsize=14)
    ax.set_ylabel("Standard deviation (feature vector)", fontsize=14)
    ax.set_title("Standard deviation of feature vector per scalar (ascending)", fontsize=14, fontweight="bold")
    y_max = float(np.nanmax(stds_sorted)) if np.any(np.isfinite(stds_sorted)) else 1.0
    ax.set_ylim(0, y_max * 1.15)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scalar std plot to: {output_path}")


def plot_ica_factor_correlation(corr_df: pd.DataFrame, output_path: str) -> None:
    """
    Plot heatmap of correlations between ICA components (columns in corr_df) and factors (rows).
    After transpose: ICs on the x-axis, factors on the y-axis. vmin=-1, vmax=1.
    """
    n_ic = corr_df.shape[0]
    n_f = corr_df.shape[1]
    fig_w = max(7.0, min(22.0, 1.2 + n_ic * 0.42))
    fig_h = max(4.0, min(14.0, 1.0 + n_f * 0.65))
    annot_fs = max(5, min(9, int(140 / max(n_ic, n_f, 1))))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        corr_df.T,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": annot_fs},
        square=False,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("ICA vs factor loadings correlation (across scalars)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ICA-factor correlation heatmap to: {output_path}")


def ica_variance_percent_increment_per_component(ica, X: np.ndarray) -> np.ndarray:
    """
    Percent variance explained *cumulatively* using reconstruction error.

    For k=1..K, we reconstruct the centered data X_centered using the first k ICA components:
      X_hat_k_centered = S[:, :k] @ A[:, :k].T
    where S = ica.transform(X) and A = ica.mixing_.

    We compute cumulative explained percent:
      cum_k = (1 - SSE_k / SSE_0) * 100
    and return cum_k for k=1..K.
    """
    if X.size == 0:
        return np.array([])

    mean_ = getattr(ica, "mean_", None)
    if mean_ is None:
        mean_ = np.nanmean(X, axis=0)
    X_centered = X - mean_
    total_ss = float(np.sum(X_centered**2))
    if total_ss <= 0:
        return np.array([])

    sources = ica.transform(X)  # (n_samples, n_components)
    mixing = ica.mixing_
    if mixing is None:
        return np.array([])
    n_comp = sources.shape[1]
    if mixing.shape[1] != n_comp:
        # Fallback: truncate to the shared min dimensionality.
        n_comp = min(n_comp, mixing.shape[1])
        sources = sources[:, :n_comp]
        mixing = mixing[:, :n_comp]

    cum_explained = np.zeros(n_comp, dtype=float)
    for k in range(1, n_comp + 1):
        X_hat_k = sources[:, :k] @ mixing[:, :k].T
        sse_k = float(np.sum((X_centered - X_hat_k) ** 2))
        cum_explained[k - 1] = (1.0 - sse_k / total_ss) * 100.0

    return cum_explained


def plot_ica_variance_explained(
    variance_percent_cumulative: np.ndarray,
    output_path: str,
    retained_n_ic: int,
) -> None:
    """
    Scree-style plot for ICA cumulative reconstruction-variance explained (%).

    ICA section uses a fixed retained number of ICs (typically the same as the number
    of factor-analysis factors), so we plot and highlight that fixed cutoff.
    """
    if variance_percent_cumulative.size == 0:
        return

    y_full = variance_percent_cumulative.astype(float)
    retained_n_ic = int(np.clip(retained_n_ic, 1, len(y_full)))
    y = y_full[:retained_n_ic]
    n_comp = len(y)
    x = np.arange(1, n_comp + 1)

    fig_w = max(8, min(16, 6 + n_comp * 0.25))
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    ax.plot(
        x,
        y,
        "o-",
        linewidth=2,
        markersize=max(4, min(8, 120 // max(n_comp, 1))),
        color="#4CAF50",
        label="Cumulative variance explained (%)",
        zorder=2,
    )

    ax.axvline(
        x=retained_n_ic,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Retained: IC{retained_n_ic}",
        zorder=1,
    )
    ax.plot(
        retained_n_ic,
        y[retained_n_ic - 1],
        "ro",
        markersize=10,
        markeredgecolor="black",
        markeredgewidth=1.5,
        zorder=3,
        label="_nolegend_",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=max(8, min(11, 200 // max(n_comp, 1))))
    ax.set_xlabel("ICA Component Number", fontsize=16)
    ax.set_ylabel("Cumulative variance explained (%)", fontsize=16)
    ax.set_title("ICA Scree Plot (cumulative variance explained)", fontsize=18, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    y_max = float(np.max(y)) if np.any(np.isfinite(y)) else 1.0
    ax.set_ylim(0, min(105.0, max(5.0, y_max * 1.08)))
    ax.set_xlim(0.5, n_comp + 0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ICA variance explained (scree) plot to: {output_path}")


def plot_pca_cumulative_variance_explained(
    pca_cumulative_percent: np.ndarray,
    optimal_pc: int,
    output_path: str,
) -> None:
    """
    Scree-style line plot for PCA cumulative variance explained (%).

    Places the red marker at `optimal_pc` (PC index, 1-indexed) and labels it as `optimal`.
    """
    if pca_cumulative_percent.size == 0:
        return

    y = pca_cumulative_percent.astype(float)
    n_comp = len(y)
    optimal_pc = int(np.clip(optimal_pc, 1, n_comp))

    x = np.arange(1, n_comp + 1)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, y, "o-", linewidth=2, markersize=8, color="#4CAF50", label="Cumulative variance explained (%)")

    ax.axvline(
        x=optimal_pc,
        color="red",
        linestyle="--",
        linewidth=2,
        label="_nolegend_",
    )

    optimal_y = float(y[optimal_pc - 1])
    ax.plot(
        optimal_pc,
        optimal_y,
        "ro",
        markersize=12,
        markeredgecolor="black",
        markeredgewidth=2,
        zorder=5,
        label="Optimal number of PCs",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=12)
    ax.set_xlabel("PC Number", fontsize=16)
    ax.set_ylabel("Cumulative variance explained (%)", fontsize=16)
    ax.set_title("PCA cumulative variance explained", fontsize=18, fontweight="bold")
    ax.legend(fontsize=14, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    y_max = float(np.max(y)) if np.any(np.isfinite(y)) else 1.0
    ax.set_ylim(0, min(105.0, max(5.0, y_max * 1.08)))
    ax.set_xlim(0.5, n_comp + 0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA cumulative variance plot to: {output_path}")


def plot_pca_factor_correlation(corr_df: pd.DataFrame, output_path: str) -> None:
    """
    Heatmap of correlations between PCA components (rows in corr_df) and factor loadings (columns in corr_df).
    Display: PCA components on the y-axis, factors on the x-axis.
    """
    n_pc = corr_df.shape[0]
    n_f = corr_df.shape[1]

    # Wider with more factor columns; taller with more PC rows.
    fig_w = max(7.0, min(22.0, 1.2 + n_f * 0.42))
    fig_h = max(4.0, min(14.0, 1.0 + n_pc * 0.65))
    annot_fs = max(11, min(16, int(220 / max(n_pc, n_f, 1))))
    axis_label_fs = 16
    tick_fs = 16

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        corr_df,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": annot_fs},
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("PCA vs factor loadings correlation (across scalars)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Factors", fontsize=axis_label_fs)
    ax.set_ylabel("PCA components", fontsize=axis_label_fs)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA-factor correlation heatmap to: {output_path}")


def create_html_factor_report(
    output_path: str,
    group_label: str,
    group_mode: str,
    n_factors: int,
    eigenvalues: np.ndarray,
    optimal_n_factors: int,
    rotation_used: str,
    n_subjects: int,
    n_regions: int,
    n_tracts: int,
    n_scalars: int,
    excluded_tracts: List[str],
    loadings_csv_path: str,
    variance_plot_path: str,
    corr_heatmap_path: str,
    scalar_std_plot_path: str | None = None,
    pca_corr_and_loadings_plot_path: str | None = None,
    pca_corr_plot_path: str | None = None,
    pca_variance_plot_path: str | None = None,
    corr_factor_pca_components_plot_path: str | None = None,
    corr_factor_pca_components_plot_path_factor_ordered: str | None = None,
    corr_and_loadings_factor_loading_ordered_path: str | None = None,
    corr_factor_pca_components_plot_path_factor_ordered_dti_dki_gqi: str | None = None,
    factor_pca_combined_summary_plot_path: str | None = None,
    tissue_pc1_factors_plot_path: str | None = None,
    tissue_pc1_pcs_plot_path: str | None = None,
    tissue_pc1_factors_plot_path_combined: str | None = None,
    tissue_pc1_pcs_plot_path_combined: str | None = None,
    atlas_set_label: str | None = None,
) -> None:
    """Create an HTML report summarizing GAM mni_micro tissue-class factor analysis results.
    Images are referenced by filename (same directory as report) to keep HTML small and reliable to open.
    """
    # Use relative paths (basename) so the report stays small and opens in all browsers
    def image_src(image_path: str | None) -> str | None:
        if not image_path or not os.path.exists(image_path):
            return None
        return os.path.basename(image_path)

    # Calculate cumulative variance from eigenvalues
    total_eigenvalue = float(eigenvalues[optimal_n_factors - 1]) if optimal_n_factors <= len(eigenvalues) else 0.0
    total_var_retained = float(np.sum(eigenvalues[:optimal_n_factors]) / np.sum(eigenvalues) * 100.0) if np.sum(eigenvalues) > 0 else 0.0
    var_src = image_src(variance_plot_path)
    corr_src = image_src(corr_heatmap_path)
    corr_factor_pca_components_src = image_src(corr_factor_pca_components_plot_path) if corr_factor_pca_components_plot_path else None
    corr_factor_pca_components_factor_ordered_src = (
        image_src(corr_factor_pca_components_plot_path_factor_ordered)
        if corr_factor_pca_components_plot_path_factor_ordered
        else None
    )
    corr_and_loadings_factor_loading_ordered_src = (
        image_src(corr_and_loadings_factor_loading_ordered_path)
        if corr_and_loadings_factor_loading_ordered_path
        else None
    )
    corr_factor_pca_components_factor_ordered_dti_dki_gqi_src = (
        image_src(corr_factor_pca_components_plot_path_factor_ordered_dti_dki_gqi)
        if corr_factor_pca_components_plot_path_factor_ordered_dti_dki_gqi
        else None
    )
    factor_pca_summary_src = image_src(factor_pca_combined_summary_plot_path) if factor_pca_combined_summary_plot_path else None

    # Format excluded tracts for display
    excluded_tracts_str = ", ".join(excluded_tracts) if excluded_tracts else "None"

    # Conditional wording for atlas set and subject inclusion
    if atlas_set_label:
        if n_tracts == 0:
            atlas_set_display = f"GM only ({atlas_set_label})"
            subject_criteria = "Only subjects with complete data for all GM regions were included in the analysis."
        elif n_regions == 0:
            atlas_set_display = f"WM only ({atlas_set_label})"
            subject_criteria = "Only subjects with complete data for all remaining WM tracts were included in the analysis."
        else:
            atlas_set_display = f"Combined ({atlas_set_label})"
            subject_criteria = "Only subjects with complete data for all GM regions AND all remaining WM tracts were included in the analysis."
    else:
        atlas_set_display = ""
        subject_criteria = "Only subjects with complete data for all GM regions AND all remaining WM tracts were included in the analysis."

    n_features = n_subjects * (n_regions + n_tracts * 3)
    if n_tracts == 0:
        feature_construction_text = "Feature vectors use GM parcel z-scores only (no WM tract data)."
    elif n_regions == 0:
        feature_construction_text = "Feature vectors use pyafq-derived WM tract node z-scores only (no GM parcel data). Each tract contributes 3 segment mean z-scores (end1, core, end2)."
    else:
        feature_construction_text = "Feature vectors combine GM parcel z-scores (mni_micro) and pyafq-derived WM tract segment mean z-scores, concatenated as [GM parcels] + [WM tract segments]. Each tract contributes 3 features (end1, core, end2)."

    report_title = f"Factor Analysis Report — {atlas_set_display}" if atlas_set_display else "Combined mni_micro Scalar Factor Analysis Report"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report_title}</title>
    <style>
        body {{
            font-family: Georgia, serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
            font-size: 24px;
            margin: 20px 0;
        }}
        h2 {{
            color: #333;
            font-size: 18px;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        p {{
            font-size: 12px;
            line-height: 1.6;
        }}
        .section {{
            background-color: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-kv {{
            font-size: 12px;
            line-height: 1.8;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .summary-table th,
        .summary-table td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .summary-table th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            width: 40%;
        }}
        .summary-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .plot-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }}
        .plot-item {{
            flex: 1;
            min-width: 800px;
            max-width: 1600px;
            text-align: center;
        }}
        .plot-title {{
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        .plot-image {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .code-path {{
            font-family: monospace;
            font-size: 11px;
            background-color: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .excluded-tracts {{
            font-family: monospace;
            font-size: 11px;
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <h1>Combined mni_micro Downsampled Microstructural Factor Analysis</h1>

    <div class="section">
        <h2>Analysis Summary</h2>
        <table class="summary-table">
            <tr>
                <th>Group Mode</th>
                <td>{group_mode} ({group_label})</td>
            </tr>
            <tr>
                <th>Number of Factors Retained</th>
                <td>{n_factors} (determined by elbow method, optimal: {optimal_n_factors})</td>
            </tr>
            <tr>
                <th>Total Variance Explained</th>
                <td>{total_var_retained:.1f}%</td>
            </tr>
            <tr>
                <th>Rotation Method</th>
                <td>{rotation_used}</td>
            </tr>
            <tr>
                <th>Number of Subjects</th>
                <td>{n_subjects} ({subject_criteria})</td>
            </tr>
            <tr>
                <th>Number of GM Regions</th>
                <td>{n_regions}</td>
            </tr>
            <tr>
                <th>Number of WM Tracts</th>
                <td>{n_tracts}</td>
            </tr>
            <tr>
                <th>Number of Scalars</th>
                <td>{n_scalars}</td>
            </tr>
            <tr>
                <th>Feature Vector Length</th>
                <td>{n_subjects} subjects * ({n_regions} regions + {n_tracts} tracts * 3 segments) = {n_features} features per scalar</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>Data Processing Details</h2>
        <p class="summary-kv">
            <strong>GM Atlas:</strong> {GM_ATLAS}<br/>
            <strong>WM Atlas:</strong> {WM_ATLAS}<br/>
            {f'<strong>Atlas set:</strong> {atlas_set_display}<br/>' if atlas_set_display else ''}
        </p>
        <div class="excluded-tracts">
            <strong>Excluded Scalars ({len(EXCLUDED_SCALARS)}):</strong><br/>
            {', '.join(EXCLUDED_SCALARS)}
        </div>
        <div class="excluded-tracts">
            <strong>Excluded WM Tracts ({len(excluded_tracts)}):</strong><br/>
            {excluded_tracts_str}
        </div>
        <p class="summary-kv">
            <strong>Subject Inclusion Criteria:</strong> {subject_criteria}<br/>
            <strong>WM Tract Data:</strong> For each WM tract, pyafq GAM node z-scores are divided into segments (end1/core/end2) using the node definitions, and segment means are used as 3 features per tract.<br/>
            <strong>Feature Construction:</strong> {feature_construction_text}
        </p>
    </div>

    <div class="section">
        <h2>Pairwise Correlations + Factor Loadings + PCA Components</h2>
        <p>Single combined heatmap with blocks (left-to-right): correlation matrix, empty spacer, factor loadings, empty spacer, and PCA component loadings.</p>
        <div class="plot-container">
"""

    if corr_factor_pca_components_src:
        html += f"""            <div class="plot-item">
                <img src="{corr_factor_pca_components_src}" alt="Correlation matrix + factor loadings + PCA component loadings" class="plot-image"/>
            </div>
"""

    html += """        </div>
"""

    if corr_factor_pca_components_factor_ordered_src:
        html += """        <h3>Pairwise Correlations + Factor Loadings + PCA Components (Factor loading ordered)</h3>
        <p>Same heatmap as above, but rows (statistics) are ordered by which factor has the largest |loading| for that statistic (all statistics dominated by F1 first, then F2, …), then by descending |loading| on that dominant factor within each group.</p>
        <div class="plot-container">
"""
        html += f"""            <div class="plot-item">
                <img src="{corr_factor_pca_components_factor_ordered_src}" alt="Correlation matrix + factor loadings + PCA component loadings (factor loading ordered)" class="plot-image"/>
            </div>
"""
        html += """        </div>
"""
        if corr_and_loadings_factor_loading_ordered_src:
            html += """        <h3>Pairwise Correlations + Factor Loadings (Factor loading ordered)</h3>
        <p>Same row ordering as the figure above, but only the correlation matrix and factor loadings blocks (no PCA component loadings).</p>
        <div class="plot-container">
"""
            html += f"""            <div class="plot-item">
                <img src="{corr_and_loadings_factor_loading_ordered_src}" alt="Correlation matrix + factor loadings (factor loading ordered, no PCA)" class="plot-image"/>
            </div>
"""
            html += """        </div>
"""

    elif corr_and_loadings_factor_loading_ordered_src:
        html += """        <h3>Pairwise Correlations + Factor Loadings (Factor loading ordered)</h3>
        <p>Rows (statistics) ordered by dominant factor (largest |loading|), then by |loading| on that factor; correlation matrix and factor loadings only (no PCA block).</p>
        <div class="plot-container">
"""
        html += f"""            <div class="plot-item">
                <img src="{corr_and_loadings_factor_loading_ordered_src}" alt="Correlation matrix + factor loadings (factor loading ordered, no PCA)" class="plot-image"/>
            </div>
"""
        html += """        </div>
"""

    if corr_factor_pca_components_factor_ordered_dti_dki_gqi_src:
        html += """        <h3>Pairwise Correlations + Factor Loadings + PCA Components (Factor loading ordered; DTI, DKI, and GQI only)</h3>
        <p>Same layout as the factor-loading–ordered heatmap, but only <strong>DTI</strong> (<code>dti_*</code>), <strong>DKI</strong> (<code>dki_*</code>), and <strong>GQI</strong> (<code>gqi_*</code>) scalars. Excludes <strong>NODDI</strong> (<code>noddi_*</code>), <strong>MAP-MRI</strong> (<code>map_*</code>), and <strong>RDI</strong> (<code>rdi_*</code>).</p>
        <div class="plot-container">
"""
        html += f"""            <div class="plot-item">
                <img src="{corr_factor_pca_components_factor_ordered_dti_dki_gqi_src}" alt="Correlation matrix + factor loadings + PCA component loadings (factor loading ordered, DTI DKI GQI subset)" class="plot-image"/>
            </div>
"""
        html += """        </div>
"""

    html += """    </div>

    <div class="section">
        <h2>Factor/PCA Summary</h2>
        <p>Single combined figure with factor scree plot, PCA cumulative variance explained, and factor loadings vs PCA loadings correlations.</p>
        <div class="plot-container">
"""

    if factor_pca_summary_src:
        html += f"""            <div class="plot-item">
                <img src="{factor_pca_summary_src}" alt="Factor/PCA combined summary plot" class="plot-image"/>
            </div>
"""

    html += """        </div>
    </div>
"""

    if tissue_pc1_factors_plot_path or tissue_pc1_factors_plot_path_combined:
        html += """    <div class="section">
        <h2>Tissue-specific PC1 correlations with whole-brain factors</h2>
        <p>For each tissue type (PC1 loadings), Pearson correlation with whole-brain factor loadings (F1-F4) across shared scalars.</p>
"""

        if tissue_pc1_factors_plot_path_combined:
            html += """        <div class="plot-container">
"""
            html += f"""            <div class="plot-item">
                <img src="{image_src(tissue_pc1_factors_plot_path_combined)}" alt="Tissue PC1 correlations with whole-brain factors (GM_Combined and WM_Combined)" class="plot-image"/>
            </div>
"""
            html += """        </div>
"""

        if tissue_pc1_factors_plot_path:
            html += """        <div class="plot-container">
"""
            html += f"""            <div class="plot-item">
                <img src="{image_src(tissue_pc1_factors_plot_path)}" alt="Tissue PC1 correlations with whole-brain factors (4 tissue types)" class="plot-image"/>
            </div>
"""
            html += """        </div>
"""

        html += """    </div>
"""

    if tissue_pc1_pcs_plot_path or tissue_pc1_pcs_plot_path_combined:
        html += """    <div class="section">
        <h2>Tissue-specific PC1 correlations with whole-brain PCs</h2>
        <p>For each tissue type (PC1 loadings), Pearson correlation with whole-brain PCA loadings (PC1-PC4) across shared scalars.</p>
"""

        if tissue_pc1_pcs_plot_path_combined:
            html += """        <div class="plot-container">
"""
            html += f"""            <div class="plot-item">
                <img src="{image_src(tissue_pc1_pcs_plot_path_combined)}" alt="Tissue PC1 correlations with whole-brain PCs (GM_Combined and WM_Combined)" class="plot-image"/>
            </div>
"""
            html += """        </div>
"""

        if tissue_pc1_pcs_plot_path:
            html += """        <div class="plot-container">
"""
            html += f"""            <div class="plot-item">
                <img src="{image_src(tissue_pc1_pcs_plot_path)}" alt="Tissue PC1 correlations with whole-brain PCs (4 tissue types)" class="plot-image"/>
            </div>
"""
            html += """        </div>
"""

        html += """    </div>
"""

    html += f"""    <div class="section">
        <h2>Output Files</h2>
        <p class="summary-kv">
            <strong>Factor Loadings:</strong> <span class="code-path">{os.path.basename(loadings_csv_path)}</span><br/>
            <strong>Correlation Matrix:</strong> <span class="code-path">{os.path.basename(loadings_csv_path).replace('_scalar_factor_loadings.csv', '_scalar_correlations.csv')}</span><br/>
            <strong>Eigenvalues:</strong> <span class="code-path">{os.path.basename(loadings_csv_path).replace('_scalar_factor_loadings.csv', '_scalar_factor_eigenvalues.csv')}</span><br/>
        </p>
    </div>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Saved factor analysis HTML report to: {output_path}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main() -> None:
    print("\n" + "=" * 80)
    print(
        f"Processing GAM mni_micro tissue-class scalar correlations (GROUP_MODE={GROUP_MODE}, GROUPS={GROUPS})"
    )
    print("=" * 80 + "\n")

    # Load scalar labels and tissue-class elements
    scalar_labels = load_scalar_labels()
    cortex_gm_parcels = get_glasser_regions()
    subcortex_gm_parcels = get_subcortex_4s156_regions()
    assoc_wm_tracts = get_tracts_by_type("association")
    proj_wm_tracts = get_tracts_by_type("projection")

    all_gm_regions_master = sorted(list(dict.fromkeys(cortex_gm_parcels + subcortex_gm_parcels)))
    all_wm_tracts_master = sorted(list(dict.fromkeys(assoc_wm_tracts + proj_wm_tracts)))

    print(f"Found {len(scalar_labels)} scalar labels (after exclusions)")
    print(f"Found {len(cortex_gm_parcels)} Cortex GM (Glasser) parcels")
    print(f"Found {len(subcortex_gm_parcels)} Subcortex GM (4S156 subcortex) parcels")
    print(f"Found {len(assoc_wm_tracts)} Association WM (HCP1065) tracts")
    print(f"Found {len(proj_wm_tracts)} Projection WM (HCP1065) tracts")
    print(f"Global master elements: {len(all_gm_regions_master)} GM parcels + {len(all_wm_tracts_master)} WM tracts")

    # Determine subjects with complete data across all 4 tissue classes (global intersection)
    print("\nDetermining subjects with complete data across all 4 tissue classes (global intersection)...")
    example_scalar = scalar_labels[0] if scalar_labels else None
    regions_with_subjects: Dict[str, Set[str]] = {}

    for region in tqdm(sorted(all_gm_regions_master), desc="Checking GM parcels"):
        if example_scalar is None:
            continue
        data = load_region_scalar_data(region, example_scalar, GROUPS)
        if data is not None and not data.empty:
            regions_with_subjects[region] = set(data.index)

    if regions_with_subjects:
        subjects_with_all_regions = set.intersection(*regions_with_subjects.values())
    else:
        subjects_with_all_regions = set()

    print(f"Subjects with all GM parcels: {len(subjects_with_all_regions)}")

    # Determine subjects with all tracts
    print("\nDetermining subjects with all WM tracts...")
    tract_subjects: Dict[str, Set[str]] = {}

    for tract in tqdm(sorted(all_wm_tracts_master), desc="Checking WM tracts"):
        if example_scalar is None:
            continue
        data = load_tract_scalar_data(tract, example_scalar, GROUPS)
        if data is not None and not data.empty:
            tract_subjects[tract] = set(data.index)
        else:
            tract_subjects[tract] = set()

    if tract_subjects:
        subjects_with_all_tracts = set.intersection(*tract_subjects.values())
    else:
        subjects_with_all_tracts = set()

    print(f"Subjects with all WM tracts: {len(subjects_with_all_tracts)}")

    # Use the same subjects for all atlas runs: only those with complete GM and WM data
    subjects_with_complete_data = subjects_with_all_regions.intersection(subjects_with_all_tracts)
    subjects_used_global = sorted(list(subjects_with_complete_data))
    if not subjects_used_global:
        print("\nERROR: No subjects found with complete data (all GM regions + all WM tracts). Aborting.")
        return
    print(f"\nSubjects used for all atlas runs (complete GM + WM data): {len(subjects_used_global)}")

    # Master subject list: all subjects with complete data (used for all atlas sets)
    master_subjects = sorted(subjects_used_global)
    master_subjects_path = ospj(OUTPUT_PROJECT_ROOT, "subjects_included.csv")
    pd.DataFrame({"subject": master_subjects}).to_csv(master_subjects_path, index=False)
    print(f"\nSaved master subjects_included.csv ({len(master_subjects)} subjects) to: {master_subjects_path}")

    # Factor analysis: only the all-4 combined atlas run (GM Glasser+4S156 + WM HCP1065).
    atlas_runs_fa = [
        {"name": "All4_Combined", "regions": all_gm_regions_master, "tracts": all_wm_tracts_master},
    ]
    summary_lines: List[str] = []

    for run in atlas_runs_fa:
        run_name = run["name"]
        run_regions = run["regions"]
        run_tracts = run["tracts"]
        subjects_used = master_subjects

        run_output_dir = ospj(OUTPUT_PROJECT_ROOT, run_name)
        os.makedirs(run_output_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Factor analysis: {run_name} (n_regions={len(run_regions)}, n_tracts={len(run_tracts)}, n_subjects={len(subjects_used)})")
        print(f"{'='*80}")

        # All FA output filenames include atlas set name for clarity
        file_prefix = f"{GROUP_LABEL}_{run_name}"

        # Build feature vectors for this run (no per-atlas subjects_included.csv)
        desc = "GM only" if not run_tracts else ("WM only" if not run_regions else "combined")
        print(f"\nBuilding feature vectors ({desc})...")
        scalar_vectors = build_combined_feature_vectors(
            subjects_used,
            run_regions,
            run_tracts,
            scalar_labels,
            GROUPS,
            USE_ABS,
        )

        all_scalars = [s for s in scalar_labels if s in scalar_vectors]
        if not all_scalars:
            print(f"No scalars with data for run {run_name}; skipping.")
            continue

        feature_matrix = pd.DataFrame(
            {scalar: scalar_vectors[scalar] for scalar in all_scalars}
        ).T
        print(f"Feature matrix shape: {feature_matrix.shape} (scalars x features)")

        # Correlation matrix
        print("\nComputing correlation matrix between scalars...")
        corr_df, _, _ = compute_correlation_matrix_scalars(feature_matrix)
        corr_matrix_path = ospj(run_output_dir, f"{file_prefix}_scalar_correlations.csv")
        corr_df.to_csv(corr_matrix_path)
        print(f"Saved correlation matrix to: {corr_matrix_path}")
        corr_matrix_minimal_path = ospj(run_output_dir, f"{file_prefix}_corr_matrix_minimal.png")
        plot_corr_matrix_minimal(corr_df, corr_matrix_minimal_path)

        # Factor analysis (observations x variables)
        X = feature_matrix.T.values
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        X_imputed = X.copy()
        inds = np.where(np.isnan(X_imputed))
        if inds[0].size > 0:
            X_imputed[inds] = np.take(col_means, inds[1])

        n_vars = X_imputed.shape[1]
        corr_matrix_data = np.corrcoef(X_imputed.T)
        corr_matrix_data = np.nan_to_num(corr_matrix_data, nan=0.0)
        eigenvalues_all, _ = np.linalg.eig(corr_matrix_data)
        eigenvalues_all = np.real(eigenvalues_all)
        eigenvalues_all = np.sort(eigenvalues_all)[::-1]
        eigenvalues = eigenvalues_all

        optimal_n_factors = find_elbow(eigenvalues)
        print(f"Optimal number of factors (elbow): {optimal_n_factors}")
        n_factors = optimal_n_factors

        # Promax only: write directly to run_output_dir (no rot-promax subdir)
        rot_method = "promax"
        n_factors_rot = n_factors
        n_factors_for_rotation = min(n_factors, n_vars - 1) if n_vars > 1 else 1
        if n_factors_for_rotation < n_factors:
            print(f"Warning: Reducing factors for rotation from {n_factors} to {n_factors_for_rotation}.")
            n_factors_rot = n_factors_for_rotation

        try:
            factor_analyzer = FactorAnalyzer(
                n_factors=n_factors_rot,
                method="minres",
                rotation=rot_method,
                svd_method="lapack",
            )
            factor_analyzer.fit(X_imputed)
            loadings = factor_analyzer.loadings_
            rotation_used = rot_method
            print(f"Applied {rot_method} rotation to {n_factors_rot} factors.")
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Warning: {rot_method} rotation failed ({e}). Using unrotated loadings.")
            factor_analyzer = FactorAnalyzer(
                n_factors=n_factors_rot,
                method="minres",
                rotation=None,
                svd_method="lapack",
            )
            factor_analyzer.fit(X_imputed)
            loadings = factor_analyzer.loadings_
            rotation_used = "none (unrotated)"

        scalar_order = feature_matrix.index.tolist()
        factor_labels = [f"F{i + 1}" for i in range(n_factors_rot)]
        loadings_df_vars = pd.DataFrame(
            loadings,
            index=scalar_order,
            columns=factor_labels,
        )
        loadings_df = loadings_df_vars.T
        loadings_df.index.name = "factor"

        loadings_csv_path = ospj(run_output_dir, f"{file_prefix}_scalar_factor_loadings.csv")
        loadings_df.to_csv(loadings_csv_path)
        print(f"Saved factor loadings to: {loadings_csv_path}")

        loadings_ordered_cols = order_scalars_by_max_abs_factor_loading(loadings_df)
        loadings_ordered_csv_path = ospj(
            run_output_dir, f"{file_prefix}_scalar_factor_loadings_ordered.csv"
        )
        loadings_df[loadings_ordered_cols].to_csv(loadings_ordered_csv_path)
        print(f"Saved factor loadings (factor-dominant column order) to: {loadings_ordered_csv_path}")

        # Save uniquenesses and scalar means for downstream factor score computation (e.g. compute_factor_scores_h5)
        uniquenesses = factor_analyzer.get_uniquenesses()  # 1D, length n_scalars (same order as loadings rows = scalar_order)
        uniquenesses_df = pd.DataFrame({"uniqueness": uniquenesses}, index=scalar_order)
        uniquenesses_csv_path = ospj(run_output_dir, f"{file_prefix}_scalar_uniquenesses.csv")
        uniquenesses_df.to_csv(uniquenesses_csv_path)
        print(f"Saved scalar uniquenesses to: {uniquenesses_csv_path}")

        scalar_means = np.nanmean(X_imputed, axis=0)  # column means, same order as scalar_order
        scalar_means_df = pd.DataFrame({"mean": scalar_means}, index=scalar_order)
        scalar_means_csv_path = ospj(run_output_dir, f"{file_prefix}_scalar_means.csv")
        scalar_means_df.to_csv(scalar_means_csv_path)
        print(f"Saved scalar means to: {scalar_means_csv_path}")

        variance_plot_path = ospj(run_output_dir, f"{file_prefix}_scalar_factor_eigenvalues_scree.png")
        plot_scree_eigenvalues(eigenvalues, optimal_n_factors, variance_plot_path)

        scalar_std_plot_path = None  # Removed from reports (kept computation optional)

        cumulative_variance_from_eigen = np.cumsum(eigenvalues) / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else np.zeros_like(eigenvalues)
        variance_df = pd.DataFrame({
            "Factor": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "variance_fraction": eigenvalues / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else np.zeros_like(eigenvalues),
            "variance_percent": (eigenvalues / np.sum(eigenvalues) * 100.0) if np.sum(eigenvalues) > 0 else np.zeros_like(eigenvalues),
            "cumulative_fraction": cumulative_variance_from_eigen,
            "cumulative_percent": cumulative_variance_from_eigen * 100.0,
        })
        variance_csv_path = ospj(run_output_dir, f"{file_prefix}_scalar_factor_eigenvalues.csv")
        variance_df.to_csv(variance_csv_path, index=False)

        corr_and_loadings_path = ospj(run_output_dir, f"{file_prefix}_scalar_corr_and_factor_loadings.png")
        plot_corr_and_loadings_combined(corr_df, loadings_df, corr_and_loadings_path)

        corr_and_loadings_factor_loading_ordered_path = ospj(
            run_output_dir,
            f"{file_prefix}_scalar_corr_and_factor_loadings_factor_loading_ordered.png",
        )
        plot_corr_and_loadings_combined(
            corr_df,
            loadings_df,
            corr_and_loadings_factor_loading_ordered_path,
            row_order="max_factor_loading",
            dpi=CORR_FACTOR_PCA_FACTOR_ORDERED_DPI,
        )

        corr_and_loadings_factor_loading_ordered_bottom_path = ospj(
            run_output_dir,
            f"{file_prefix}_scalar_corr_and_factor_loadings_factor_loading_ordered_bottom.png",
        )
        plot_corr_and_loadings_combined_bottom(
            corr_df,
            loadings_df,
            corr_and_loadings_factor_loading_ordered_bottom_path,
            row_order="max_factor_loading",
            dpi=CORR_FACTOR_PCA_FACTOR_ORDERED_DPI,
        )

        loadings_standalone_path = ospj(run_output_dir, f"{file_prefix}_scalar_factor_loadings_standalone.png")
        plot_factor_loadings_standalone(loadings_df, loadings_standalone_path)

        # PCA for this run (same X_imputed as factor analysis: observations × scalars)
        pca_corr_and_loadings_path: str | None = None
        pca_corr_plot_path: str | None = None
        pca_variance_plot_path: str | None = None
        corr_factor_pca_components_plot_path: str | None = None
        corr_factor_pca_components_plot_path_factor_ordered: str | None = None
        corr_factor_pca_components_plot_path_factor_ordered_dti_dki_gqi: str | None = None
        factor_pca_combined_summary_plot_path: str | None = None
        pca_component_loadings_csv_path: str | None = None

        tissue_pc1_factors_plot_path: str | None = None
        tissue_pc1_pcs_plot_path: str | None = None
        tissue_pc1_factors_plot_path_combined: str | None = None
        tissue_pc1_pcs_plot_path_combined: str | None = None

        # PCA block
        n_samples = X_imputed.shape[0]
        n_pca_target = min(N_PCA_COMPONENTS_FULL, n_vars, n_samples)
        if n_vars >= 2 and n_pca_target >= 1:
            try:
                pca_full = PCA(n_components=n_pca_target)
                pca_full.fit(X_imputed)

                # Save full PCA component loadings for downstream (PC1 vs factors/PCs across tissues).
                pca_loadings_full_df = pd.DataFrame(
                    pca_full.components_[:n_pca_target, :],
                    index=[f"PC{i + 1}" for i in range(n_pca_target)],
                    columns=scalar_order,
                )
                pca_component_loadings_csv_path = ospj(
                    run_output_dir, f"{file_prefix}_pca_component_loadings.csv"
                )
                pca_loadings_full_df.to_csv(pca_component_loadings_csv_path)

                # Per-component explained variance (for tissue PC1 subtitles in GM/WM combined plots).
                ev_ratio = np.asarray(pca_full.explained_variance_ratio_, dtype=float)
                n_ev = len(ev_ratio)
                pd.DataFrame(
                    {
                        "component": [f"PC{i + 1}" for i in range(n_ev)],
                        "variance_ratio": ev_ratio,
                        "variance_percent": ev_ratio * 100.0,
                    }
                ).to_csv(
                    ospj(run_output_dir, f"{file_prefix}_pca_explained_variance_ratio.csv"),
                    index=False,
                )

                # Cumulative explained variance (%), length = n_pca_target.
                pca_cum_pct = np.cumsum(pca_full.explained_variance_ratio_) * 100.0
                if pca_cum_pct.size == 0:
                    raise ValueError("PCA cumulative variance array empty")

                # Requested selection rule:
                # optimal_pc = elbow-1 of the cumulative variance curve.
                elbow_pc = int(find_elbow(pca_cum_pct))
                elbow_pc = max(1, min(elbow_pc, len(pca_cum_pct)))
                optimal_pc = max(1, elbow_pc - 1)
                optimal_pc = min(optimal_pc, n_pca_target)

                pca_variance_plot_path = ospj(run_output_dir, f"{file_prefix}_pca_cumulative_variance_explained.png")
                plot_pca_cumulative_variance_explained(
                    pca_cum_pct,
                    optimal_pc=optimal_pc,
                    output_path=pca_variance_plot_path,
                )

                n_factor_cols = n_factors_rot
                pc_labels = [f"PC{i + 1}" for i in range(optimal_pc)]
                corr_pc_f = np.zeros((optimal_pc, n_factor_cols))

                for i in range(optimal_pc):
                    pc_vec = pca_full.components_[i, :]
                    for j in range(n_factor_cols):
                        f_vec = loadings_df.loc[f"F{j + 1}"].values
                        if np.std(pc_vec) > 0 and np.std(f_vec) > 0:
                            corr_pc_f[i, j] = np.corrcoef(pc_vec, f_vec)[0, 1]
                        else:
                            corr_pc_f[i, j] = np.nan

                factor_col_labels = [f"F{k + 1}" for k in range(n_factor_cols)]
                pca_corr_df = pd.DataFrame(
                    corr_pc_f,
                    index=pc_labels,
                    columns=factor_col_labels,
                )
                pca_corr_plot_path = ospj(run_output_dir, f"{file_prefix}_pca_factor_correlation.png")
                plot_pca_factor_correlation(pca_corr_df, pca_corr_plot_path)

                # Combined heatmap: PCA component loadings + pairwise scalar correlations (square cells)
                pca_loadings_df = pd.DataFrame(
                    pca_full.components_[:optimal_pc, :],
                    index=pc_labels,
                    columns=scalar_order,
                )
                pca_corr_and_loadings_path = ospj(run_output_dir, f"{file_prefix}_scalar_corr_and_pca_loadings.png")
                plot_corr_and_ica_combined(corr_df, pca_loadings_df, pca_corr_and_loadings_path)

                # Combined heatmap (requested): correlation matrix | spacer | factor loadings | spacer | PCA component loadings
                corr_factor_pca_components_plot_path = ospj(
                    run_output_dir,
                    f"{file_prefix}_scalar_corr_factor_loadings_pca_components_combined.png",
                )
                plot_corr_factor_loadings_and_pca_components_combined(
                    corr_df,
                    loadings_df,
                    pca_loadings_df,
                    corr_factor_pca_components_plot_path,
                )

                corr_factor_pca_components_plot_path_factor_ordered = ospj(
                    run_output_dir,
                    f"{file_prefix}_scalar_corr_factor_loadings_pca_components_combined_factor_loading_ordered.png",
                )
                plot_corr_factor_loadings_and_pca_components_combined(
                    corr_df,
                    loadings_df,
                    pca_loadings_df,
                    corr_factor_pca_components_plot_path_factor_ordered,
                    row_order="max_factor_loading",
                    dpi=CORR_FACTOR_PCA_FACTOR_ORDERED_DPI,
                )

                corr_factor_pca_components_plot_path_factor_ordered_dti_dki_gqi = ospj(
                    run_output_dir,
                    f"{file_prefix}_scalar_corr_factor_loadings_pca_components_combined_factor_loading_ordered_dti_dki_gqi.png",
                )
                plot_corr_factor_loadings_and_pca_components_combined(
                    corr_df,
                    loadings_df,
                    pca_loadings_df,
                    corr_factor_pca_components_plot_path_factor_ordered_dti_dki_gqi,
                    row_order="max_factor_loading",
                    allowed_prefixes=COMBINED_HEATMAP_DTI_DKI_GQI_PREFIXES,
                    exclude_scalar_names=COMBINED_HEATMAP_DTI_DKI_GQI_EXCLUDE_SCALARS,
                    include_factor_pca_blocks=False,
                    axis_tick_fontsize=16,
                    corr_annotation_fontsize=11,
                    cbar_label_fontsize=16,
                    cbar_tick_fontsize=15,
                )

                # Combined summary (requested): factor scree | PCA cumulative variance | factor loadings vs PCA correlations
                factor_pca_combined_summary_plot_path = ospj(
                    run_output_dir,
                    f"{file_prefix}_factor_pca_combined_scree_and_correlations.png",
                )
                plot_factor_pca_combined_summary(
                    eigenvalues=eigenvalues,
                    optimal_n_factors=optimal_n_factors,
                    pca_cum_pct=pca_cum_pct,
                    optimal_pc=optimal_pc,
                    pca_corr_df=pca_corr_df,
                    output_path=factor_pca_combined_summary_plot_path,
                )

            except (np.linalg.LinAlgError, ValueError, Exception) as e:
                print(f"Warning: PCA failed ({e}). Skipping PCA section in report.")
                pca_corr_plot_path = None
                pca_variance_plot_path = None
                pca_corr_and_loadings_path = None
                corr_factor_pca_components_plot_path = None
                corr_factor_pca_components_plot_path_factor_ordered = None
                corr_factor_pca_components_plot_path_factor_ordered_dti_dki_gqi = None
                factor_pca_combined_summary_plot_path = None

        # For the All4_Combined run, add tissue-specific PC1 correlations with whole-brain factors/PCs.
        if (
            run_name == "All4_Combined"
            and pca_component_loadings_csv_path
            and os.path.exists(loadings_csv_path)
        ):
            tissue_run_order = ["CortexGM_Glasser", "SubcortexGM_4S156", "AssocWM_HCP1065", "ProjWM_HCP1065"]
            tissue_display_names = [
                "Cortex GM only",
                "Subcortex GM only",
                "Association WM only",
                "Projection WM only",
            ]
            tissue_pc1_factors_plot_path = ospj(
                run_output_dir, f"{file_prefix}_tissue_pc1_correlations_with_wholebrain_factors.png"
            )
            tissue_pc1_pcs_plot_path = ospj(
                run_output_dir, f"{file_prefix}_tissue_pc1_correlations_with_wholebrain_pcs.png"
            )
            try:
                plot_tissue_pc1_correlations_with_wholebrain(
                    group_label=GROUP_LABEL,
                    all4_output_dir=run_output_dir,
                    all4_factor_loadings_csv_path=loadings_csv_path,
                    all4_pca_component_loadings_csv_path=pca_component_loadings_csv_path,
                    tissue_run_order=tissue_run_order,
                    tissue_display_names=tissue_display_names,
                    output_path_factors=tissue_pc1_factors_plot_path,
                    output_path_pcs=tissue_pc1_pcs_plot_path,
                    target_count=4,
                    force_ncols=4,
                )
            except Exception as e:  # noqa: BLE001
                print(f"Warning: failed to create tissue-PC1 correlation plots ({e}).")
                tissue_pc1_factors_plot_path = None
                tissue_pc1_pcs_plot_path = None

            # Second set: GM_Combined + WM_Combined + |GM|-|WM| diff (1×3 bar plots)
            tissue_run_order_combined = ["GM_Combined", "WM_Combined"]
            tissue_display_names_combined = [
                "Grey matter",
                "White matter",
            ]
            tissue_pc1_factors_plot_path_combined = ospj(
                run_output_dir, f"{file_prefix}_tissue_pc1_correlations_with_wholebrain_factors_gm_wm.png"
            )
            tissue_pc1_pcs_plot_path_combined = ospj(
                run_output_dir, f"{file_prefix}_tissue_pc1_correlations_with_wholebrain_pcs_gm_wm.png"
            )
            try:
                plot_tissue_pc1_correlations_with_wholebrain(
                    group_label=GROUP_LABEL,
                    all4_output_dir=run_output_dir,
                    all4_factor_loadings_csv_path=loadings_csv_path,
                    all4_pca_component_loadings_csv_path=pca_component_loadings_csv_path,
                    tissue_run_order=tissue_run_order_combined,
                    tissue_display_names=tissue_display_names_combined,
                    output_path_factors=tissue_pc1_factors_plot_path_combined,
                    output_path_pcs=tissue_pc1_pcs_plot_path_combined,
                    target_count=4,
                    force_ncols=3,
                    use_absolute_correlations=True,
                    add_gm_wm_abs_diff_subplot=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"Warning: failed to create tissue-PC1 GM/WM combined plots ({e}).")
                tissue_pc1_factors_plot_path_combined = None
                tissue_pc1_pcs_plot_path_combined = None

        html_report_path = ospj(run_output_dir, f"{file_prefix}_scalar_factor_analysis_report.html")
        create_html_factor_report(
            output_path=html_report_path,
            group_label=GROUP_LABEL,
            group_mode=GROUP_MODE,
            n_factors=n_factors_rot,
            eigenvalues=eigenvalues,
            optimal_n_factors=optimal_n_factors,
            rotation_used=rotation_used,
            n_subjects=len(subjects_used),
            n_regions=len(run_regions),
            n_tracts=len(run_tracts),
            n_scalars=len(all_scalars),
            excluded_tracts=TRACTS_TO_REMOVE,
            loadings_csv_path=loadings_csv_path,
            variance_plot_path=variance_plot_path,
            corr_heatmap_path=corr_and_loadings_path,
            scalar_std_plot_path=scalar_std_plot_path,
            pca_corr_and_loadings_plot_path=pca_corr_and_loadings_path,
            pca_corr_plot_path=pca_corr_plot_path,
            pca_variance_plot_path=pca_variance_plot_path,
            corr_factor_pca_components_plot_path=corr_factor_pca_components_plot_path,
            corr_factor_pca_components_plot_path_factor_ordered=corr_factor_pca_components_plot_path_factor_ordered,
            corr_and_loadings_factor_loading_ordered_path=corr_and_loadings_factor_loading_ordered_path,
            corr_factor_pca_components_plot_path_factor_ordered_dti_dki_gqi=corr_factor_pca_components_plot_path_factor_ordered_dti_dki_gqi,
            factor_pca_combined_summary_plot_path=factor_pca_combined_summary_plot_path,
            tissue_pc1_factors_plot_path=tissue_pc1_factors_plot_path,
            tissue_pc1_pcs_plot_path=tissue_pc1_pcs_plot_path,
            tissue_pc1_factors_plot_path_combined=tissue_pc1_factors_plot_path_combined,
            tissue_pc1_pcs_plot_path_combined=tissue_pc1_pcs_plot_path_combined,
            atlas_set_label=run_name,
        )

        summary_lines.append(f"  {run_name}: n_subjects={len(subjects_used)}, n_regions={len(run_regions)}, n_tracts={len(run_tracts)}, optimal_factors={optimal_n_factors}")

    # Optionally compute factor scores (.h5) per atlas/method.
    if COMPUTE_FACTOR_SCORES_H5:
        try:
            from compute_factor_scores_h5 import run_factor_scores_h5
            print("\nComputing factor scores (weighted_sum + regression) and writing .h5 files...")
            run_factor_scores_h5(
                base_dir=PROJECT_ROOT,
                output_base_dir=OUTPUT_PROJECT_ROOT,
                atlas_runs=atlas_runs_fa,
            )
        except Exception as e:
            print(f"Warning: Factor score .h5 computation failed: {e}")
    else:
        print("\nSkipping .h5 factor-score computation (COMPUTE_FACTOR_SCORES_H5=False).")

    print("\n" + "=" * 80)
    print("Combined mni_micro scalar factor-analysis processing complete!")
    print("=" * 80)
    print(f"\nOutput base directory: {OUTPUT_PROJECT_ROOT}")
    print(f"  - Master subjects_included.csv: {master_subjects_path}")
    for run in atlas_runs_fa:
        print(f"  - {run['name']}: {ospj(OUTPUT_PROJECT_ROOT, run['name'])}")
    if summary_lines:
        print("\nPer-run summary:")
        for line in summary_lines:
            print(line)


if __name__ == "__main__":
    main()

