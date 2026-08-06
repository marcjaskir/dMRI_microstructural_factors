import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
import glob
import os
from os.path import join as ospj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
import pickle
from typing import Any, Dict, List, Sequence, Set, Tuple, Optional
from tqdm import tqdm
from scipy import stats
import nibabel as nib
from nilearn import plotting
from base64 import b64encode

# Use Georgia as the default font for all matplotlib text in this script
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Georgia"]

# ============================================================================
# CONFIGURATION
# ============================================================================

METADATA_DIR = f"{PROJECT_ROOT}/data/metadata"
# GAM z-scores: Glasser + 4S156 subcortex under mni_micro; WM node profiles from pyAFQ (same as factor_analysis.py)
MNI_MICRO_PROJECT_ROOT = f"{gam_dir()}/mni_micro"
GM_GLASSER_PROFILE_DIR = f"{MNI_MICRO_PROJECT_ROOT}/Glasser"
GM_4S156_PROFILE_DIR = f"{MNI_MICRO_PROJECT_ROOT}/4S156"
WM_PROFILE_DIR_PYAFQ = f"{gam_dir()}/pyafq/HCP1065"
FOUR_S156_DSEG_PATH = f"{PROJECT_ROOT}/data/atlases/4S/atlas-4S156Parcels_dseg.tsv"
HCP1065_TRACT_METADATA_PATH = f"{PROJECT_ROOT}/data/atlases/HCP1065/HCP1065_tract_metadata.csv"
OUTPUT_PROJECT_ROOT = f"{analysis_dir()}/factor_z-scores"
SCALAR_Z_SCORES_OUTPUT_DIR = f"{OUTPUT_PROJECT_ROOT}/scalar_z-scores"
CLINICAL_METADATA_PATH = f"{PROJECT_ROOT}/derivatives/metadata/clinical_penn_epilepsy_qsirecon.csv"
INCLUSION_METADATA_PATH = str(inclusion_dir() / "penn_epilepsy_included_basic_metadata.csv")
FACTOR_LOADINGS_PATH = (
    f"{analysis_dir()}/factor_analysis/All4_Combined/"
    "controls_All4_Combined_scalar_factor_loadings.csv"
)
# Master control list from All4_Combined FA (intersection across GM+WM); used for per-ROI factor score mean/SD
CONTROLS_SUBJECTS_INCLUDED_PATH = f"{analysis_dir()}/factor_analysis/subjects_included.csv"
# 4S156 subcortex brain maps (same space as historical factor_z reports)
ATLAS_4S_NIFTI_PATH = (
    f"{PROJECT_ROOT}/data/atlases/4S/tpl-MNI152NLin2009cAsym_atlas-4S156Parcels_res-01_dseg_resliced-hcp1065.nii.gz"
)
ATLAS_4S_TSV_PATH = f"{PROJECT_ROOT}/data/atlases/4S/atlas-4S156Parcels_dseg.tsv"
# Glasser cortex brain maps
ATLAS_GLASSER_NIFTI_PATH = (
    f"{PROJECT_ROOT}/data/atlases/Glasser/atlas-Glasser_space-MNI152NLin2009cAsym_res-1_dseg.nii.gz"
)
ATLAS_GLASSER_TSV_PATH = f"{PROJECT_ROOT}/data/atlases/Glasser/atlas-Glasser_dseg.tsv"
# Legacy names (4S subcortex atlas) — prefer ATLAS_4S_* in new code
ATLAS_NIFTI_PATH = ATLAS_4S_NIFTI_PATH
ATLAS_TSV_PATH = ATLAS_4S_TSV_PATH

N_NODES = 100  # Number of nodes in pyAFQ tract profiles

# Define three segments: end1 (nodes 1-34), core (nodes 35-66), end2 (nodes 67-100)
END1_NODES = list(range(1, 35))  # nodes 1-34
CORE_NODES = list(range(35, 67))  # nodes 35-66
END2_NODES = list(range(67, 101))  # nodes 67-100

# Define groups
CONTROL_GROUPS = ["penn_controls", "hcpya", "hcpaging"]
PATIENT_GROUPS = ["penn_epilepsy"]


def get_group_from_subject_id(sub: str) -> Optional[str]:
    """Determine control group from subject ID pattern (penn_controls, hcpya, hcpaging)."""
    sub_clean = sub.replace("sub-", "") if sub.startswith("sub-") else sub
    if sub_clean.startswith("RID"):
        return "penn_controls"
    if sub_clean.startswith("HCA"):
        return "hcpaging"
    if sub_clean.isdigit() or (sub_clean.startswith("1") and len(sub_clean) == 6):
        return "hcpya"
    return None

# Scalar / tract exclusions — keep in sync with factor_analysis.py
EXCLUDED_SCALARS = [
    "map_li", "map_am", "dti_txx", "dti_txy", "dti_txz", "dti_tyy", "dti_tyz", "dti_tzz",
    "dti_ha", "rdi_rd1", "rdi_rd2", "gqi_iso"
]

# Factor labels: paper-aligned (F1 overall, F2 non-Gaussian, F3 anisotropic)
from lib.factor_labels import FACTOR_LABELS, get_factor_label  # noqa: E402

TRACTS_TO_REMOVE = [
    "CBT_L", "CBT_R", "RST_L", "RST_R", "DRTT_L", "DRTT_R",
    "EMC_L", "EMC_R", "C_PHP_L", "C_PHP_R", "AF_L", "AF_R", "SLF3_L", "SLF3_R",
    "SLF2_L", "SLF2_R", "FAT_L", "FAT_R",
]

# Order microstructural statistics by these scalar prefixes when plotting
SCALAR_PREFIX_ORDER = ["dti", "rdi", "dki", "gqi", "noddi", "map"]

# Create output directory
os.makedirs(OUTPUT_PROJECT_ROOT, exist_ok=True)

# Minimum number of control subjects with valid raw factor scores required for per-ROI mean/SD
MIN_CONTROLS_FOR_ROI_Z = 2


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_scalar_labels() -> List[str]:
    """Load and filter scalar labels."""
    path = ospj(METADATA_DIR, "scalar_labels_to_filenames.json")
    with open(path) as f:
        all_labels = list(json.load(f).keys())
    return [label for label in all_labels if label not in EXCLUDED_SCALARS]


def load_clinical_metadata() -> pd.DataFrame | None:
    """Load clinical metadata from CSV file."""
    if os.path.exists(CLINICAL_METADATA_PATH):
        return pd.read_csv(CLINICAL_METADATA_PATH)
    print(f"Warning: Clinical metadata file not found at {CLINICAL_METADATA_PATH}")
    return None


def load_included_temporal_subjects() -> set[str]:
    """Load subject IDs where lobe == 'temporal' from the inclusion metadata CSV.

    Returns both 'sub-RIDxxxx' and 'RIDxxxx' forms so ID matching is flexible.
    """
    included: set[str] = set()
    if not os.path.exists(INCLUSION_METADATA_PATH):
        print(f"Warning: Inclusion metadata not found at {INCLUSION_METADATA_PATH}")
        return included
    try:
        df = pd.read_csv(INCLUSION_METADATA_PATH)
        mask = df["lobe"].astype(str).str.strip().str.lower() == "temporal"
        for sub in df.loc[mask, "sub"].astype(str):
            included.add(sub)
            if sub.startswith("sub-"):
                included.add(sub[4:])
            else:
                included.add(f"sub-{sub}")
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not load inclusion metadata: {e}")
    return included


def load_controls_subjects_included() -> set[str]:
    """Load set of control subjects that should be included from CSV file.
    
    Returns a set containing both the original subject IDs and normalized versions
    (with and without 'sub-' prefix) to handle different ID formats.
    """
    included_subjects = set()
    if os.path.exists(CONTROLS_SUBJECTS_INCLUDED_PATH):
        try:
            df = pd.read_csv(CONTROLS_SUBJECTS_INCLUDED_PATH)
            if "subject" in df.columns:
                subjects = df["subject"].tolist()
                # Add both original and normalized versions to handle different formats
                for sub in subjects:
                    included_subjects.add(sub)
                    # Also add version without 'sub-' prefix if it exists
                    if sub.startswith("sub-"):
                        included_subjects.add(sub[4:])
                    # Also add version with 'sub-' prefix if it doesn't exist
                    else:
                        included_subjects.add(f"sub-{sub}")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Could not load controls subjects included file: {e}")
    else:
        print(f"Warning: Controls subjects included file not found at {CONTROLS_SUBJECTS_INCLUDED_PATH}")
    return included_subjects


def load_temporal_patient_subjects_ordered() -> List[str]:
    """Temporal lobe subjects from inclusion metadata, stable order (first occurrence in CSV)."""
    if not os.path.exists(INCLUSION_METADATA_PATH):
        return []
    try:
        df = pd.read_csv(INCLUSION_METADATA_PATH)
        mask = df["lobe"].astype(str).str.strip().str.lower() == "temporal"
        subs = df.loc[mask, "sub"].astype(str).tolist()
        seen: set[str] = set()
        out: List[str] = []
        for s in subs:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not load ordered temporal subjects: {e}")
        return []


def resolve_subject_key(subject: str, index: pd.Index) -> Any | None:
    """Map a subject ID to the matching index label (handles sub- vs no-prefix)."""
    if index is None or len(index) == 0:
        return None
    s = str(subject)
    str_to_key: Dict[str, Any] = {}
    for k in index:
        sk = str(k)
        str_to_key.setdefault(sk, k)
        if sk.startswith("sub-"):
            str_to_key.setdefault(sk[4:], k)
        else:
            str_to_key.setdefault(f"sub-{sk}", k)
    return str_to_key.get(s)


def load_tract_metadata() -> Dict[str, str]:
    """Load tract metadata and return mapping from label to name."""
    tract_metadata_path = ospj(PROJECT_ROOT, "data", "atlases", "HCP1065", "HCP1065_tract_metadata.csv")
    label_to_name = {}
    
    if os.path.exists(tract_metadata_path):
        try:
            metadata_df = pd.read_csv(tract_metadata_path)
            if "label" in metadata_df.columns and "name" in metadata_df.columns:
                label_to_name = dict(zip(metadata_df["label"], metadata_df["name"]))
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Could not load tract metadata: {e}")
    
    return label_to_name


def load_tract_metadata_full() -> pd.DataFrame:
    """Load full tract metadata DataFrame."""
    if os.path.exists(HCP1065_TRACT_METADATA_PATH):
        try:
            return pd.read_csv(HCP1065_TRACT_METADATA_PATH)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Could not load tract metadata: {e}")
    return pd.DataFrame()


def _list_subdirs(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    return sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(ospj(base_dir, d))]
    )


def get_glasser_regions() -> List[str]:
    """Cortex GM parcel dirs from mni_micro/Glasser (matches factor_analysis.py)."""
    return _list_subdirs(GM_GLASSER_PROFILE_DIR)


def get_subcortex_4s156_regions() -> List[str]:
    """Subcortex GM parcel labels (network_label == 'n/a') from 4S156 dseg, intersect mni_micro dirs."""
    if not os.path.exists(FOUR_S156_DSEG_PATH):
        return []
    dseg = pd.read_csv(FOUR_S156_DSEG_PATH, sep="\t")
    if "label" not in dseg.columns or "network_label" not in dseg.columns:
        return []
    nl = dseg["network_label"]
    subcortex_mask = nl.isna()
    subcortex_mask = subcortex_mask | (nl.astype(str).str.strip().str.lower() == "n/a")
    subcortex = dseg.loc[subcortex_mask, "label"].astype(str).tolist()
    existing = set(_list_subdirs(GM_4S156_PROFILE_DIR))
    return sorted([lab for lab in subcortex if lab in existing])


def get_mni_micro_gm_profile_dir_for_region(region: str) -> str:
    """Route GM region label to Glasser vs 4S156 base directory."""
    if os.path.isdir(ospj(GM_GLASSER_PROFILE_DIR, region)):
        return GM_GLASSER_PROFILE_DIR
    return GM_4S156_PROFILE_DIR


def get_tracts_by_type(tract_type: str) -> List[str]:
    """HCP1065 tract labels filtered by metadata column type (matches factor_analysis.py)."""
    meta = load_tract_metadata_full()
    if meta.empty or "label" not in meta.columns or "type" not in meta.columns:
        return []
    tracts = meta.loc[meta["type"].astype(str) == tract_type, "label"].astype(str).tolist()
    tracts = [t for t in tracts if t not in TRACTS_TO_REMOVE]
    available_bases: Set[str] = set(_list_subdirs(WM_PROFILE_DIR_PYAFQ))
    return sorted([t for t in tracts if t in available_bases])


def discover_all_gm_regions() -> List[str]:
    """All GM regions: Glasser cortex + 4S156 subcortex (mni_micro), de-duplicated."""
    all_gm = list(dict.fromkeys(get_glasser_regions() + get_subcortex_4s156_regions()))
    return sorted(all_gm)


def discover_all_wm_tracts() -> List[str]:
    """Association + projection WM tracts on pyAFQ, minus TRACTS_TO_REMOVE (matches All4_Combined)."""
    assoc = get_tracts_by_type("association")
    proj = get_tracts_by_type("projection")
    return sorted(list(dict.fromkeys(assoc + proj)))


def load_factor_loadings(scalar_labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Load All4_Combined factor loadings; columns restricted to scalars used in this pipeline."""
    if not os.path.exists(FACTOR_LOADINGS_PATH):
        print(f"Warning: Factor loadings file not found at {FACTOR_LOADINGS_PATH}")
        return pd.DataFrame()
    try:
        loadings_df = pd.read_csv(FACTOR_LOADINGS_PATH, index_col=0)
        if loadings_df.index.name is None and "factor" in loadings_df.columns:
            loadings_df = loadings_df.set_index("factor")
        labels = list(scalar_labels) if scalar_labels is not None else load_scalar_labels()
        use_cols = [c for c in loadings_df.columns if c in set(labels)]
        if not use_cols:
            print("Warning: No loadings columns overlap with scalar_labels.")
            return pd.DataFrame()
        return loadings_df[use_cols]
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not load factor loadings: {e}")
        return pd.DataFrame()


def compute_factor_scores(
    roi_data: Dict[str, pd.DataFrame],
    roi_name: str,
    scalar_labels: Sequence[str],
    subjects: Sequence[str],
    factor_loadings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute factor scores for a ROI (region or tract).
    
    For GM regions: uses single z-score column '{scalar}_z'
    For WM tracts: uses segment-based z-scores '{scalar}_z_end1', '{scalar}_z_core', '{scalar}_z_end2'
                  and averages factor scores across the three segments
    
    For each subject and factor:
        factor_score = sum(scalar_z_score * factor_loading) across all scalars
    
    Args:
        roi_data: Dict[scalar] -> DataFrame with index 'sub'
                  For GM: column '{scalar}_z'
                  For WM: columns '{scalar}_z_end1', '{scalar}_z_core', '{scalar}_z_end2'
        roi_name: Name of the ROI (for error messages)
        scalar_labels: List of scalar labels
        subjects: List of subjects
        factor_loadings: DataFrame with factors as rows, scalars as columns
    
    Returns:
        DataFrame with subjects as rows and factors as columns
    """
    if factor_loadings.empty:
        return pd.DataFrame()
    
    factor_scores = {}
    
    # Get available scalars in factor loadings
    available_scalars = set(factor_loadings.columns) & set(scalar_labels)
    
    # Check if this is a WM tract (has segment columns) or GM region (has single z column)
    is_wm_tract = False
    if available_scalars:
        sample_scalar = list(available_scalars)[0]
        if sample_scalar in roi_data:
            sample_data = roi_data[sample_scalar]
            if not sample_data.empty:
                # Check if segment columns exist
                z_col_end1 = f"{sample_scalar}_z_end1"
                if z_col_end1 in sample_data.columns:
                    is_wm_tract = True
    
    for subject in subjects:
        subject_scores = {}
        
        for factor in factor_loadings.index:
            score = 0.0
            n_scalars = 0
            
            if is_wm_tract:
                # For WM tracts: compute factor scores for each segment and average
                segment_scores = []
                for segment in ['end1', 'core', 'end2']:
                    segment_score = 0.0
                    segment_n_scalars = 0
                    
                    for scalar in available_scalars:
                        if scalar not in roi_data:
                            continue
                        
                        data = roi_data[scalar]
                        z_col = f"{scalar}_z_{segment}"
                        row_key = resolve_subject_key(subject, data.index)
                        if row_key is not None and z_col in data.columns:
                            z_score = data.loc[row_key, z_col]
                            if not np.isnan(z_score):
                                loading = factor_loadings.loc[factor, scalar]
                                if not np.isnan(loading):
                                    segment_score += z_score * loading
                                    segment_n_scalars += 1
                    
                    if segment_n_scalars > 0:
                        segment_scores.append(segment_score)
                
                # Average across segments
                if segment_scores:
                    score = np.mean(segment_scores)
                    n_scalars = len(segment_scores)
            else:
                # For GM regions: use single z-score column
                for scalar in available_scalars:
                    if scalar not in roi_data:
                        continue
                    
                    data = roi_data[scalar]
                    z_col = f"{scalar}_z"
                    row_key = resolve_subject_key(subject, data.index)
                    if row_key is not None:
                        z_score = data.loc[row_key, z_col]
                        if not np.isnan(z_score):
                            loading = factor_loadings.loc[factor, scalar]
                            if not np.isnan(loading):
                                score += z_score * loading
                                n_scalars += 1
            
            # Store score (or NaN if no valid scalars)
            if n_scalars > 0:
                subject_scores[factor] = score
            else:
                subject_scores[factor] = np.nan
        
        factor_scores[subject] = subject_scores
    
    return pd.DataFrame(factor_scores).T


def load_gm_region_scalar_data(
    region: str,
    scalar: str,
    groups: Sequence[str],
) -> pd.DataFrame | None:
    """
    Load z-score data for a specific GM region and scalar (mni_micro Glasser or 4S156).

    Returns a DataFrame with index 'sub' and a single column '{scalar}_z',
    or None if data are unavailable.
    """
    gm_profile_dir = get_mni_micro_gm_profile_dir_for_region(region)
    gam_path = ospj(gm_profile_dir, region, f"{region}_{scalar}_stat-mean_gam.csv")
    if not os.path.exists(gam_path):
        legacy = ospj(gm_profile_dir, region, f"{region}_{scalar}_gam.csv")
        if not os.path.exists(legacy):
            return None
        gam_path = legacy

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


def load_wm_tract_scalar_data(
    tract: str,
    scalar: str,
    groups: Sequence[str],
) -> pd.DataFrame | None:
    """
    Load z-score data for a specific WM tract and scalar (pyAFQ node-level GAM).

    Returns a DataFrame with index 'sub' and columns 'node1_z', 'node2_z', ..., 'node100_z',
    or None if data are unavailable.
    """
    gam_path = ospj(WM_PROFILE_DIR_PYAFQ, tract, f"{tract}_{scalar}_stat-mean_gam.csv")
    if not os.path.exists(gam_path):
        legacy = ospj(WM_PROFILE_DIR_PYAFQ, tract, f"{tract}_{scalar}_gam.csv")
        if not os.path.exists(legacy):
            return None
        gam_path = legacy

    try:
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


def compute_tract_averaged_z_scores(
    tract_node_data: pd.DataFrame,
) -> pd.Series:
    """
    Compute mean z-score across nodes for each subject.
    
    Args:
        tract_node_data: DataFrame with index 'sub' and columns 'node1_z', ..., 'node100_z'
    
    Returns:
        Series with index 'sub' and mean z-score values
    """
    z_cols = [f'node{i}_z' for i in range(1, N_NODES + 1)]
    return tract_node_data[z_cols].mean(axis=1)


def compute_tract_segment_z_scores(
    tract_node_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute mean z-scores for each of three segments (end1, core, end2) for each subject.
    
    Args:
        tract_node_data: DataFrame with index 'sub' and columns 'node1_z', ..., 'node100_z'
    
    Returns:
        DataFrame with index 'sub' and columns '{scalar}_z_end1', '{scalar}_z_core', '{scalar}_z_end2'
        Note: scalar name is extracted from the calling context, so this function assumes
        the scalar name is passed separately or stored in the DataFrame.
    """
    z_cols = [f'node{i}_z' for i in range(1, N_NODES + 1)]
    results = {}
    
    for subject in tract_node_data.index:
        z_scores = tract_node_data.loc[subject, z_cols].values
        mean_end1 = get_segment_mean_z(z_scores, END1_NODES)
        mean_core = get_segment_mean_z(z_scores, CORE_NODES)
        mean_end2 = get_segment_mean_z(z_scores, END2_NODES)
        results[subject] = {
            'end1': mean_end1,
            'core': mean_core,
            'end2': mean_end2,
        }
    
    return pd.DataFrame(results).T


def order_scalars_by_prefix(scalars: Sequence[str]) -> List[str]:
    """Order scalar names by predefined prefixes, then alphabetically."""
    prefix_rank = {p: i for i, p in enumerate(SCALAR_PREFIX_ORDER)}

    def _get_prefix(name: str) -> str:
        return name.split("_", 1)[0] if "_" in name else name

    return sorted(
        list(scalars),
        key=lambda name: (prefix_rank.get(_get_prefix(name), len(prefix_rank)), name),
    )


def _wm_tract_segment_roi_key(
    tract: str,
    segment: str,
    tract_to_end1: Dict[str, str],
    tract_to_end2: Dict[str, str],
) -> str:
    """WM column label: same as factor z-score exports (e.g. AF_L -> AF_L_end-A)."""
    end1_label = tract_to_end1.get(tract, "end1")
    end2_label = tract_to_end2.get(tract, "end2")
    segment_to_label = {
        "end1": f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
        "core": "core",
        "end2": f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
    }
    segment_label = segment_to_label.get(segment, segment)
    if tract.endswith("_L"):
        tract_base = tract[:-2]
        hemi = "L"
    elif tract.endswith("_R"):
        tract_base = tract[:-2]
        hemi = "R"
    else:
        tract_base = tract
        hemi = ""
    if hemi:
        return f"{tract_base}_{hemi}_{segment_label}"
    return f"{tract_base}_{segment_label}"


def _load_gm_scalar_z_series(
    region: str,
    scalar: str,
    groups: Sequence[str],
    subjects: Sequence[str],
) -> Optional[pd.Series]:
    """Per-subject GAM residual z for one GM region and scalar; reindexed to ``subjects``."""
    gm_base = get_mni_micro_gm_profile_dir_for_region(region)
    for fname in (f"{region}_{scalar}_stat-mean_gam.csv", f"{region}_{scalar}_gam.csv"):
        path = ospj(gm_base, region, fname)
        if not os.path.exists(path):
            continue
        try:
            gam = pd.read_csv(path)
            g = gam[gam["group"].isin(groups)]
            if g.empty:
                return None
            zc = f"{scalar}_z"
            if zc not in g.columns:
                return None
            s = g.set_index("sub")[zc]
            return s.reindex(list(subjects))
        except Exception:
            return None
    return None


def _load_wm_segment_z_series(
    tract: str,
    scalar: str,
    groups: Sequence[str],
    segment: str,
    subjects: Sequence[str],
) -> Optional[pd.Series]:
    """Mean of node*z within end1/core/end2 for one tract; reindexed to ``subjects``."""
    for fname in (f"{tract}_{scalar}_stat-mean_gam.csv", f"{tract}_{scalar}_gam.csv"):
        path = ospj(WM_PROFILE_DIR_PYAFQ, tract, fname)
        if not os.path.exists(path):
            continue
        try:
            gam = pd.read_csv(path)
            g = gam[gam["group"].isin(groups)]
            if g.empty:
                return None
            seg_nodes = {
                "end1": END1_NODES,
                "core": CORE_NODES,
                "end2": END2_NODES,
            }[segment]
            seg_cols = [f"node{i}_z" for i in seg_nodes]
            if not seg_cols or any(c not in g.columns for c in seg_cols):
                return None
            idxd = g.set_index("sub")
            means = idxd[seg_cols].mean(axis=1)
            return means.reindex(list(subjects))
        except Exception:
            return None
    return None


def save_scalar_z_scores(
    all_regions: Sequence[str],
    all_tracts: Sequence[str],
    patient_subjects: Sequence[str],
    control_subjects: Sequence[str],
    patient_groups: Sequence[str],
    control_groups: Sequence[str],
    tract_to_end1: Dict[str, str],
    tract_to_end2: Dict[str, str],
) -> None:
    """
    Save per-subject scalar-level GAM z-scores (GM ``{scalar}_z``; WM mean segment ``node*i*_z``)
    to ``SCALAR_Z_SCORES_OUTPUT_DIR`` as ``epilepsy_{scalar}_z_scores.csv`` and
    ``controls_{scalar}_z_scores.csv`` (controls with leading ``group`` column).
    """
    os.makedirs(SCALAR_Z_SCORES_OUTPUT_DIR, exist_ok=True)
    scalar_labels = order_scalars_by_prefix(load_scalar_labels())
    tracts_f = [t for t in all_tracts if t not in TRACTS_TO_REMOVE]

    def _one_table(
        scalar: str,
        groups: Sequence[str],
        subjects: Sequence[str],
        include_group: bool,
    ) -> Optional[pd.DataFrame]:
        if not groups or not subjects:
            return None
        subs = list(subjects)
        parts: List[Tuple[str, pd.Series]] = []
        for region in all_regions:
            s = _load_gm_scalar_z_series(region, scalar, groups, subs)
            if s is not None:
                parts.append((region, s))
        for tract in tracts_f:
            for seg in ("end1", "core", "end2"):
                roi_key = _wm_tract_segment_roi_key(tract, seg, tract_to_end1, tract_to_end2)
                s = _load_wm_segment_z_series(tract, scalar, groups, seg, subs)
                if s is not None:
                    parts.append((roi_key, s))
        if not parts:
            return None
        out = pd.concat([s.reindex(subs) for _, s in parts], axis=1, copy=False)
        out.columns = [n for n, _ in parts]
        out.index.name = "subject"
        if include_group:
            group_series = pd.Series(
                [get_group_from_subject_id(s) for s in out.index],
                index=out.index,
                name="group",
            )
            out = pd.concat([group_series, out], axis=1)
        return out

    for scalar in tqdm(scalar_labels, desc="scalar z-score CSVs"):
        ep_df = _one_table(scalar, list(patient_groups), list(patient_subjects), include_group=False)
        if ep_df is not None and not ep_df.empty:
            p = ospj(SCALAR_Z_SCORES_OUTPUT_DIR, f"epilepsy_{scalar}_z_scores.csv")
            ep_df.to_csv(p)
            print(f"  Saved scalar z-scores (epilepsy) to {p}")
        ct_df = _one_table(scalar, list(control_groups), list(control_subjects), include_group=True)
        if ct_df is not None and not ct_df.empty:
            p = ospj(SCALAR_Z_SCORES_OUTPUT_DIR, f"controls_{scalar}_z_scores.csv")
            ct_df.to_csv(p)
            print(f"  Saved scalar z-scores (controls) to {p}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data(
    regions: Sequence[str],
    tracts: Sequence[str],
    scalar_labels: Sequence[str],
    control_groups: Sequence[str],
    patient_groups: Sequence[str],
) -> Tuple[Dict, Dict, List[str]]:
    """
    Load all GM region and WM tract (averaged) data.

    Returns:
        gm_data: Dict[region][scalar] -> DataFrame with index 'sub', column '{scalar}_z'
        wm_tract_data: Dict[tract][scalar] -> DataFrame with index 'sub', column '{scalar}_z'
        union_subjects: Subjects who appear in **at least one** loaded GM or WM table (any
            region/scalar/tract). Factor scores are computed from available rows per ROI;
            we no longer require complete-case overlap across the full feature set.
    """
    gm_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for region in tqdm(regions, desc="GM regions"):
        gm_data[region] = {}
        for scalar in scalar_labels:
            # Load control data
            control_data = load_gm_region_scalar_data(region, scalar, control_groups)
            # Load patient data
            patient_data = load_gm_region_scalar_data(region, scalar, patient_groups)
            
            if control_data is not None and patient_data is not None:
                # Combine control and patient data
                combined = pd.concat([control_data, patient_data])
                gm_data[region][scalar] = combined
            elif control_data is not None:
                gm_data[region][scalar] = control_data
            elif patient_data is not None:
                gm_data[region][scalar] = patient_data
    
    wm_tract_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for tract in tqdm(tracts, desc="WM tracts (segments)"):
        wm_tract_data[tract] = {}
        for scalar in scalar_labels:
            # Load node-level data
            node_data = load_wm_tract_scalar_data(tract, scalar, control_groups + patient_groups)
            if node_data is not None:
                # Compute segment means (end1, core, end2)
                segment_data = compute_tract_segment_z_scores(node_data)
                # Create DataFrame with columns for each segment
                wm_tract_data[tract][scalar] = pd.DataFrame({
                    f"{scalar}_z_end1": segment_data['end1'],
                    f"{scalar}_z_core": segment_data['core'],
                    f"{scalar}_z_end2": segment_data['end2'],
                })
    
    # Union of subjects across all loaded tables (partial data allowed downstream)
    all_subject_sets = []
    
    # From GM data
    for region in regions:
        for scalar in scalar_labels:
            if region in gm_data and scalar in gm_data[region]:
                all_subject_sets.append(set(gm_data[region][scalar].index))
    
    # From WM tract data
    for tract in tracts:
        for scalar in scalar_labels:
            if tract in wm_tract_data and scalar in wm_tract_data[tract]:
                all_subject_sets.append(set(wm_tract_data[tract][scalar].index))
    
    if not all_subject_sets:
        print("Warning: No data found!")
        return {}, {}, []

    union_subjects = sorted(list(set.union(*all_subject_sets)))

    return gm_data, wm_tract_data, union_subjects


def get_lateralization_groups(
    subjects: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """
    Get subjects grouped by seizure lateralization.
    
    Returns:
        left_subjects: Subjects with lateralization in ["left", "left > right"]
        right_subjects: Subjects with lateralization in ["right", "right > left"]
    """
    clinical_metadata = load_clinical_metadata()
    
    left_lateralizations = ["left", "left > right"]
    right_lateralizations = ["right", "right > left"]
    
    left_subjects = []
    right_subjects = []
    
    if clinical_metadata is not None and 'seizure_lateralization' in clinical_metadata.columns:
        for sub in subjects:
            sub_data = clinical_metadata[clinical_metadata['sub'] == sub]
            if not sub_data.empty:
                lateralization = sub_data['seizure_lateralization'].iloc[0]
                if pd.notna(lateralization):
                    if lateralization in left_lateralizations:
                        left_subjects.append(sub)
                    elif lateralization in right_lateralizations:
                        right_subjects.append(sub)
    
    return left_subjects, right_subjects


def get_good_outcome_groups(
    subjects: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """
    Get subjects with good intervention outcomes grouped by seizure lateralization.
    Good outcomes: ilae_category_pecclinical in ["1a", "1b", "2"]
    
    Returns:
        left_subjects: Left-lateralized subjects with good outcomes
        right_subjects: Right-lateralized subjects with good outcomes
    """
    clinical_metadata = load_clinical_metadata()
    
    left_lateralizations = ["left", "left > right"]
    right_lateralizations = ["right", "right > left"]
    good_outcomes = ["1a", "1b", "2"]
    
    left_subjects = []
    right_subjects = []
    
    if clinical_metadata is not None:
        has_lateralization = 'seizure_lateralization' in clinical_metadata.columns
        has_ilae = 'ilae_category_pecclinical' in clinical_metadata.columns
        
        if has_lateralization and has_ilae:
            for sub in subjects:
                sub_data = clinical_metadata[clinical_metadata['sub'] == sub]
                if not sub_data.empty:
                    lateralization = sub_data['seizure_lateralization'].iloc[0]
                    ilae = sub_data['ilae_category_pecclinical'].iloc[0]
                    
                    if pd.notna(lateralization) and pd.notna(ilae):
                        if ilae in good_outcomes:
                            if lateralization in left_lateralizations:
                                left_subjects.append(sub)
                            elif lateralization in right_lateralizations:
                                right_subjects.append(sub)
    
    return left_subjects, right_subjects


def get_subject_groups(
    gm_data: Dict[str, Dict[str, pd.DataFrame]],
    wm_tract_data: Dict[str, Dict[str, pd.DataFrame]],
    regions: Sequence[str],
    tracts: Sequence[str],
    scalar_labels: Sequence[str],
    control_groups: Sequence[str],
    patient_groups: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """
    Determine control and patient cohorts for outputs.

    When the inclusion CSV exists, patients are **all temporal subjects** in that file
    (NaN where GAM is missing). Controls are **all non-patient IDs** present in the
    loaded GAM tables (union), not the smaller FA ``subjects_included.csv`` cohort.
    If the inclusion file is missing, patients are derived from loaded data as before.
    
    Returns:
        control_subjects: List of control subject IDs
        patient_subjects: List of patient subject IDs
    """
    clinical_metadata = load_clinical_metadata()
    all_patients_in_metadata: set[str] = set()
    if clinical_metadata is not None and "sub" in clinical_metadata.columns:
        all_patients_in_metadata = set(clinical_metadata["sub"].astype(str))

    temporal_ordered = load_temporal_patient_subjects_ordered()

    if temporal_ordered:
        patient_subjects = list(temporal_ordered)
    else:
        all_subjects_set: set[str] = set()
        for region in regions:
            for scalar in scalar_labels:
                if region in gm_data and scalar in gm_data[region]:
                    all_subjects_set.update(gm_data[region][scalar].index.astype(str))
        for tract in tracts:
            for scalar in scalar_labels:
                if tract in wm_tract_data and scalar in wm_tract_data[tract]:
                    all_subjects_set.update(wm_tract_data[tract][scalar].index.astype(str))
        included_temporal = load_included_temporal_subjects()
        patient_subjects = []
        for sub in sorted(all_subjects_set):
            if sub in all_patients_in_metadata and sub in included_temporal:
                patient_subjects.append(sub)

    all_subjects_set: set[str] = set()
    for region in regions:
        for scalar in scalar_labels:
            if region in gm_data and scalar in gm_data[region]:
                all_subjects_set.update(gm_data[region][scalar].index.astype(str))
    for tract in tracts:
        for scalar in scalar_labels:
            if tract in wm_tract_data and scalar in wm_tract_data[tract]:
                all_subjects_set.update(wm_tract_data[tract][scalar].index.astype(str))
    control_subjects = []
    for sub in sorted(all_subjects_set):
        if sub not in all_patients_in_metadata:
            control_subjects.append(sub)

    return control_subjects, patient_subjects


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def raincloud_plot(
    data: List[List[float]],
    positions: np.ndarray,
    ax,
    colors: List[str],
    widths: float = 0.6,
) -> None:
    """
    Create a raincloud plot combining half-violin, mean marker, and strip plot.
    
    Args:
        data: List of lists, one per position
        positions: X positions for each data array
        ax: Matplotlib axis
        colors: List of colors, one per position
        widths: Width of the plots
    """
    np.random.seed(42)  # For reproducible jitter
    
    for i, (values, pos, color) in enumerate(zip(data, positions, colors)):
        if len(values) == 0:
            continue
        
        # Remove NaN values
        values = np.array(values)
        values = values[~np.isnan(values)]
        
        if len(values) == 0:
            continue
        
        # Half-violin plot (density on left)
        try:
            if len(values) > 1:
                density = stats.gaussian_kde(values)
                y_range = np.linspace(values.min(), values.max(), 100)
                density_values = density(y_range)
                # Normalize density to fit within width
                if density_values.max() > 0:
                    density_values = density_values / density_values.max() * widths * 0.4
                
                # Plot half-violin
                ax.fill_betweenx(
                    y_range,
                    pos - density_values,
                    pos,
                    color=color,
                    alpha=0.6,
                )
        except Exception:
            # Fallback if KDE fails
            pass
        
        # Mean marker
        mean_val = np.mean(values)
        ax.plot(
            pos,
            mean_val,
            marker='o',
            markersize=6,
            color='black',
            markeredgecolor='white',
            markeredgewidth=1,
            zorder=15,
        )
        
        # Strip plot (jittered points, offset to the right)
        jitter = np.random.normal(0, widths * 0.03, len(values))
        # Offset to the right of the histogram
        offset = widths * 0.15
        ax.scatter(
            pos + offset + jitter,
            values,
            color=color,
            alpha=0.7,
            s=12,
            zorder=10,
        )


def plot_region_factor_scores_by_groups(
    factor_scores_left: pd.DataFrame,
    factor_scores_right: pd.DataFrame,
    region_base: str,
    all_patients: Sequence[str],
    left_lateralized: Sequence[str],
    right_lateralized: Sequence[str],
    output_path: str,
) -> None:
    """
    Plot factor scores for a GM region base in 2 rows x 2 columns layout.
    Row 1: All temporal patients
    Row 2: Lateralized patients (left/right)
    """
    if factor_scores_left.empty and factor_scores_right.empty:
        return
    
    # Get factors
    if not factor_scores_left.empty:
        factors = factor_scores_left.columns.tolist()
    elif not factor_scores_right.empty:
        factors = factor_scores_right.columns.tolist()
    else:
        return
    
    # Create figure with 2 rows x 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Row 1: All patients
    if not factor_scores_left.empty:
        ax = axes[0, 0]
        factor_data = []
        for factor in factors:
            patient_scores = factor_scores_left.loc[
                factor_scores_left.index.isin(all_patients), factor
            ].dropna().values
            factor_data.append(patient_scores.tolist())
        positions = np.arange(1, len(factors) + 1)
        colors = ['#C44E52'] * len(factors)
        raincloud_plot(factor_data, positions, ax, colors, widths=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(factors, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Factor Score', fontsize=11)
        ax.set_title(f'Left Hemisphere\nAll Temporal Patients', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    if not factor_scores_right.empty:
        ax = axes[0, 1]
        factor_data = []
        for factor in factors:
            patient_scores = factor_scores_right.loc[
                factor_scores_right.index.isin(all_patients), factor
            ].dropna().values
            factor_data.append(patient_scores.tolist())
        positions = np.arange(1, len(factors) + 1)
        colors = ['#C44E52'] * len(factors)
        raincloud_plot(factor_data, positions, ax, colors, widths=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(factors, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Factor Score', fontsize=11)
        ax.set_title(f'Right Hemisphere\nAll Temporal Patients', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Row 2: Lateralized patients
    if not factor_scores_left.empty:
        ax = axes[1, 0]
        factor_data_left = []
        factor_data_right = []
        for factor in factors:
            left_scores = factor_scores_left.loc[
                factor_scores_left.index.isin(left_lateralized), factor
            ].dropna().values
            right_scores = factor_scores_left.loc[
                factor_scores_left.index.isin(right_lateralized), factor
            ].dropna().values
            factor_data_left.append(left_scores.tolist())
            factor_data_right.append(right_scores.tolist())
        positions = np.arange(1, len(factors) + 1)
        left_positions = positions - 0.2
        right_positions = positions + 0.2
        raincloud_plot(factor_data_left, left_positions, ax, ['#4C72B0'] * len(factors), widths=0.3)
        raincloud_plot(factor_data_right, right_positions, ax, ['#C44E52'] * len(factors), widths=0.3)
        ax.set_xticks(positions)
        ax.set_xticklabels(factors, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Factor Score', fontsize=11)
        ax.set_title(f'Left Hemisphere\nBy Lateralization', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4C72B0', alpha=0.7, label='Left/Left>Right'),
            Patch(facecolor='#C44E52', alpha=0.7, label='Right/Right>Left'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    if not factor_scores_right.empty:
        ax = axes[1, 1]
        factor_data_left = []
        factor_data_right = []
        for factor in factors:
            left_scores = factor_scores_right.loc[
                factor_scores_right.index.isin(left_lateralized), factor
            ].dropna().values
            right_scores = factor_scores_right.loc[
                factor_scores_right.index.isin(right_lateralized), factor
            ].dropna().values
            factor_data_left.append(left_scores.tolist())
            factor_data_right.append(right_scores.tolist())
        positions = np.arange(1, len(factors) + 1)
        left_positions = positions - 0.2
        right_positions = positions + 0.2
        raincloud_plot(factor_data_left, left_positions, ax, ['#4C72B0'] * len(factors), widths=0.3)
        raincloud_plot(factor_data_right, right_positions, ax, ['#C44E52'] * len(factors), widths=0.3)
        ax.set_xticks(positions)
        ax.set_xticklabels(factors, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Factor Score', fontsize=11)
        ax.set_title(f'Right Hemisphere\nBy Lateralization', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4C72B0', alpha=0.7, label='Left/Left>Right'),
            Patch(facecolor='#C44E52', alpha=0.7, label='Right/Right>Left'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roi_means_rescaled_lr(
    roi_means_rescaled: pd.DataFrame,
    scalar: str,
    output_path: str,
    is_gm: bool = True,
) -> None:
    """
    Plot left-right comparison of roi_means_rescaled for a single scalar.
    
    Args:
        roi_means_rescaled: DataFrame with ROIs as rows, scalars as columns
        scalar: Scalar name to plot
        output_path: Path to save the plot
        is_gm: If True, plot GM regions; if False, plot WM tracts
    """
    if scalar not in roi_means_rescaled.columns:
        return
    
    # Separate left and right hemispheres
    left_rois = []
    right_rois = []
    left_values = []
    right_values = []
    
    for roi in roi_means_rescaled.index:
        value = roi_means_rescaled.loc[roi, scalar]
        if pd.isna(value):
            continue
        
        if is_gm:
            # For GM regions, check for LH- or RH- prefix
            if roi.startswith("LH-") or roi.startswith("LH_"):
                left_rois.append(roi.replace("LH-", "").replace("LH_", ""))
                left_values.append(value)
            elif roi.startswith("RH-") or roi.startswith("RH_"):
                right_rois.append(roi.replace("RH-", "").replace("RH_", ""))
                right_values.append(value)
        else:
            # For WM tracts, check for _L or _R suffix in tract name
            if isinstance(roi, tuple):
                tract, segment = roi
                if tract.endswith("_L"):
                    left_rois.append((tract.replace("_L", ""), segment))
                    left_values.append(value)
                elif tract.endswith("_R"):
                    right_rois.append((tract.replace("_R", ""), segment))
                    right_values.append(value)
    
    # Find matching regions/tracts between left and right
    if is_gm:
        common_regions = set(left_rois) & set(right_rois)
        if not common_regions:
            return
        
        left_matched = []
        right_matched = []
        for region in sorted(common_regions):
            left_idx = left_rois.index(region)
            right_idx = right_rois.index(region)
            left_matched.append(left_values[left_idx])
            right_matched.append(right_values[right_idx])
        
        region_labels = sorted(common_regions)
    else:
        # For WM tracts, match by tract base and segment
        common_tracts = {}
        for tract_seg in left_rois:
            if tract_seg in right_rois:
                common_tracts[tract_seg] = True
        
        if not common_tracts:
            return
        
        left_matched = []
        right_matched = []
        tract_labels = []
        for tract_seg in sorted(common_tracts.keys()):
            left_idx = left_rois.index(tract_seg)
            right_idx = right_rois.index(tract_seg)
            left_matched.append(left_values[left_idx])
            right_matched.append(right_values[right_idx])
            tract, segment = tract_seg
            tract_labels.append(f"{tract}_{segment}")
        
        region_labels = tract_labels
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(region_labels) * 0.3)))
    
    # Left hemisphere
    ax = axes[0]
    positions = np.arange(1, len(region_labels) + 1)
    ax.scatter(left_matched, positions, s=50, alpha=0.7, color='#4C72B0', label='Left')
    ax.set_yticks(positions)
    ax.set_yticklabels(region_labels, fontsize=8)
    ax.set_xlabel('Rescaled Mean', fontsize=11)
    ax.set_title(f'Left Hemisphere\n{scalar}', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    # Right hemisphere
    ax = axes[1]
    ax.scatter(right_matched, positions, s=50, alpha=0.7, color='#C44E52', label='Right')
    ax.set_yticks(positions)
    ax.set_yticklabels(region_labels, fontsize=8)
    ax.set_xlabel('Rescaled Mean', fontsize=11)
    ax.set_title(f'Right Hemisphere\n{scalar}', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tract_factor_scores_by_groups(
    factor_scores_left: pd.DataFrame,
    factor_scores_right: pd.DataFrame,
    tract_base: str,
    tract_label_to_name: Dict[str, str],
    all_patients: Sequence[str],
    left_lateralized: Sequence[str],
    right_lateralized: Sequence[str],
    output_path: str,
) -> None:
    """
    Plot factor scores for a WM tract base in 2 rows x 2 columns layout.
    Row 1: All temporal patients
    Row 2: Lateralized patients (left/right)
    """
    if factor_scores_left.empty and factor_scores_right.empty:
        return
    
    # Get factors
    if not factor_scores_left.empty:
        factors = factor_scores_left.columns.tolist()
    elif not factor_scores_right.empty:
        factors = factor_scores_right.columns.tolist()
    else:
        return
    
    # Get human-readable names
    left_tract_name = None
    right_tract_name = None
    if f"{tract_base}_L" in tract_label_to_name:
        left_tract_name = tract_label_to_name[f"{tract_base}_L"]
    if f"{tract_base}_R" in tract_label_to_name:
        right_tract_name = tract_label_to_name[f"{tract_base}_R"]
    
    # Create figure with 2 rows x 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Row 1: All patients
    if not factor_scores_left.empty:
        ax = axes[0, 0]
        factor_data = []
        for factor in factors:
            patient_scores = factor_scores_left.loc[
                factor_scores_left.index.isin(all_patients), factor
            ].dropna().values
            factor_data.append(patient_scores.tolist())
        positions = np.arange(1, len(factors) + 1)
        colors = ['#C44E52'] * len(factors)
        raincloud_plot(factor_data, positions, ax, colors, widths=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(factors, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Factor Score', fontsize=11)
        tract_display_name = left_tract_name.replace('_', ' ') if left_tract_name else tract_base
        ax.set_title(f'Left Hemisphere\nAll Temporal Patients\n{tract_display_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    if not factor_scores_right.empty:
        ax = axes[0, 1]
        factor_data = []
        for factor in factors:
            patient_scores = factor_scores_right.loc[
                factor_scores_right.index.isin(all_patients), factor
            ].dropna().values
            factor_data.append(patient_scores.tolist())
        positions = np.arange(1, len(factors) + 1)
        colors = ['#C44E52'] * len(factors)
        raincloud_plot(factor_data, positions, ax, colors, widths=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(factors, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Factor Score', fontsize=11)
        tract_display_name = right_tract_name.replace('_', ' ') if right_tract_name else tract_base
        ax.set_title(f'Right Hemisphere\nAll Temporal Patients\n{tract_display_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Row 2: Lateralized patients
    if not factor_scores_left.empty:
        ax = axes[1, 0]
        factor_data_left = []
        factor_data_right = []
        for factor in factors:
            left_scores = factor_scores_left.loc[
                factor_scores_left.index.isin(left_lateralized), factor
            ].dropna().values
            right_scores = factor_scores_left.loc[
                factor_scores_left.index.isin(right_lateralized), factor
            ].dropna().values
            factor_data_left.append(left_scores.tolist())
            factor_data_right.append(right_scores.tolist())
        positions = np.arange(1, len(factors) + 1)
        left_positions = positions - 0.2
        right_positions = positions + 0.2
        raincloud_plot(factor_data_left, left_positions, ax, ['#4C72B0'] * len(factors), widths=0.3)
        raincloud_plot(factor_data_right, right_positions, ax, ['#C44E52'] * len(factors), widths=0.3)
        ax.set_xticks(positions)
        ax.set_xticklabels(factors, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Factor Score', fontsize=11)
        tract_display_name = left_tract_name.replace('_', ' ') if left_tract_name else tract_base
        ax.set_title(f'Left Hemisphere\nBy Lateralization\n{tract_display_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4C72B0', alpha=0.7, label='Left/Left>Right'),
            Patch(facecolor='#C44E52', alpha=0.7, label='Right/Right>Left'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    if not factor_scores_right.empty:
        ax = axes[1, 1]
        factor_data_left = []
        factor_data_right = []
        for factor in factors:
            left_scores = factor_scores_right.loc[
                factor_scores_right.index.isin(left_lateralized), factor
            ].dropna().values
            right_scores = factor_scores_right.loc[
                factor_scores_right.index.isin(right_lateralized), factor
            ].dropna().values
            factor_data_left.append(left_scores.tolist())
            factor_data_right.append(right_scores.tolist())
        positions = np.arange(1, len(factors) + 1)
        left_positions = positions - 0.2
        right_positions = positions + 0.2
        raincloud_plot(factor_data_left, left_positions, ax, ['#4C72B0'] * len(factors), widths=0.3)
        raincloud_plot(factor_data_right, right_positions, ax, ['#C44E52'] * len(factors), widths=0.3)
        ax.set_xticks(positions)
        ax.set_xticklabels(factors, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Factor Score', fontsize=11)
        tract_display_name = right_tract_name.replace('_', ' ') if right_tract_name else tract_base
        ax.set_title(f'Right Hemisphere\nBy Lateralization\n{tract_display_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4C72B0', alpha=0.7, label='Left/Left>Right'),
            Patch(facecolor='#C44E52', alpha=0.7, label='Right/Right>Left'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_region_factor_scores_lateralization(
    factor_scores_left: pd.DataFrame,
    factor_scores_right: pd.DataFrame,
    region_base: str,
    left_lateralized_subjects: Sequence[str],
    right_lateralized_subjects: Sequence[str],
    output_path: str,
) -> None:
    """
    Plot factor scores comparison by lateralization for a GM region base.
    Left and right hemispheres side by side, with lateralization groups compared.
    """
    if factor_scores_left.empty and factor_scores_right.empty:
        return
    
    # Get factors
    if not factor_scores_left.empty:
        factors = factor_scores_left.columns.tolist()
    elif not factor_scores_right.empty:
        factors = factor_scores_right.columns.tolist()
    else:
        return
    
    # Create figure with two subplots (left and right hemispheres)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if factor_scores_left.empty:
        axes[0].set_visible(False)
        axes = [axes[1]]
    elif factor_scores_right.empty:
        axes[1].set_visible(False)
        axes = [axes[0]]
    
    plot_idx = 0
    
    # Plot left hemisphere
    if not factor_scores_left.empty:
        ax = axes[plot_idx]
        plot_idx += 1
        
        # Collect data for all factors and both groups
        left_group_data = []
        right_group_data = []
        
        for factor in factors:
            left_scores = factor_scores_left.loc[
                factor_scores_left.index.isin(left_lateralized_subjects), factor
            ].dropna().values
            right_scores = factor_scores_left.loc[
                factor_scores_left.index.isin(right_lateralized_subjects), factor
            ].dropna().values
            
            left_group_data.append(left_scores.tolist())
            right_group_data.append(right_scores.tolist())
        
        # Create raincloud plots for both groups
        positions = np.arange(1, len(factors) + 1)
        left_positions = positions - 0.2
        right_positions = positions + 0.2
        
        raincloud_plot(left_group_data, left_positions, ax, ['#4C72B0'] * len(factors), widths=0.3)
        raincloud_plot(right_group_data, right_positions, ax, ['#C44E52'] * len(factors), widths=0.3)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(factors, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Factor Score', fontsize=12)
        ax.set_title(f'Left Hemisphere\n{region_base}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4C72B0', alpha=0.7, label='Left/Left>Right'),
            Patch(facecolor='#C44E52', alpha=0.7, label='Right/Right>Left'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Plot right hemisphere
    if not factor_scores_right.empty:
        ax = axes[plot_idx]
        
        # Collect data for all factors and both groups
        left_group_data = []
        right_group_data = []
        
        for factor in factors:
            left_scores = factor_scores_right.loc[
                factor_scores_right.index.isin(left_lateralized_subjects), factor
            ].dropna().values
            right_scores = factor_scores_right.loc[
                factor_scores_right.index.isin(right_lateralized_subjects), factor
            ].dropna().values
            
            left_group_data.append(left_scores.tolist())
            right_group_data.append(right_scores.tolist())
        
        # Create raincloud plots for both groups
        positions = np.arange(1, len(factors) + 1)
        left_positions = positions - 0.2
        right_positions = positions + 0.2
        
        raincloud_plot(left_group_data, left_positions, ax, ['#4C72B0'] * len(factors), widths=0.3)
        raincloud_plot(right_group_data, right_positions, ax, ['#C44E52'] * len(factors), widths=0.3)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(factors, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Factor Score', fontsize=12)
        ax.set_title(f'Right Hemisphere\n{region_base}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4C72B0', alpha=0.7, label='Left/Left>Right'),
            Patch(facecolor='#C44E52', alpha=0.7, label='Right/Right>Left'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tract_factor_scores_lateralization(
    factor_scores_left: pd.DataFrame,
    factor_scores_right: pd.DataFrame,
    tract_base: str,
    tract_label_to_name: Dict[str, str],
    left_lateralized_subjects: Sequence[str],
    right_lateralized_subjects: Sequence[str],
    output_path: str,
) -> None:
    """
    Plot factor scores comparison by lateralization for a WM tract base.
    Left and right tracts side by side, with lateralization groups compared.
    """
    if factor_scores_left.empty and factor_scores_right.empty:
        return
    
    # Get factors
    if not factor_scores_left.empty:
        factors = factor_scores_left.columns.tolist()
    elif not factor_scores_right.empty:
        factors = factor_scores_right.columns.tolist()
    else:
        return
    
    # Get human-readable names
    left_tract_name = None
    right_tract_name = None
    if f"{tract_base}_L" in tract_label_to_name:
        left_tract_name = tract_label_to_name[f"{tract_base}_L"]
    if f"{tract_base}_R" in tract_label_to_name:
        right_tract_name = tract_label_to_name[f"{tract_base}_R"]
    
    # Create figure with two subplots (left and right tracts)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if factor_scores_left.empty:
        axes[0].set_visible(False)
        axes = [axes[1]]
    elif factor_scores_right.empty:
        axes[1].set_visible(False)
        axes = [axes[0]]
    
    plot_idx = 0
    
    # Plot left tract
    if not factor_scores_left.empty:
        ax = axes[plot_idx]
        plot_idx += 1
        
        # Collect data for all factors and both groups
        left_group_data = []
        right_group_data = []
        
        for factor in factors:
            left_scores = factor_scores_left.loc[
                factor_scores_left.index.isin(left_lateralized_subjects), factor
            ].dropna().values
            right_scores = factor_scores_left.loc[
                factor_scores_left.index.isin(right_lateralized_subjects), factor
            ].dropna().values
            
            left_group_data.append(left_scores.tolist())
            right_group_data.append(right_scores.tolist())
        
        # Create raincloud plots for both groups
        positions = np.arange(1, len(factors) + 1)
        left_positions = positions - 0.2
        right_positions = positions + 0.2
        
        raincloud_plot(left_group_data, left_positions, ax, ['#4C72B0'] * len(factors), widths=0.3)
        raincloud_plot(right_group_data, right_positions, ax, ['#C44E52'] * len(factors), widths=0.3)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(factors, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Factor Score', fontsize=12)
        tract_display_name = left_tract_name.replace('_', ' ') if left_tract_name else tract_base
        ax.set_title(f'Left Hemisphere\n{tract_display_name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4C72B0', alpha=0.7, label='Left/Left>Right'),
            Patch(facecolor='#C44E52', alpha=0.7, label='Right/Right>Left'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Plot right tract
    if not factor_scores_right.empty:
        ax = axes[plot_idx]
        
        # Collect data for all factors and both groups
        left_group_data = []
        right_group_data = []
        
        for factor in factors:
            left_scores = factor_scores_right.loc[
                factor_scores_right.index.isin(left_lateralized_subjects), factor
            ].dropna().values
            right_scores = factor_scores_right.loc[
                factor_scores_right.index.isin(right_lateralized_subjects), factor
            ].dropna().values
            
            left_group_data.append(left_scores.tolist())
            right_group_data.append(right_scores.tolist())
        
        # Create raincloud plots for both groups
        positions = np.arange(1, len(factors) + 1)
        left_positions = positions - 0.2
        right_positions = positions + 0.2
        
        raincloud_plot(left_group_data, left_positions, ax, ['#4C72B0'] * len(factors), widths=0.3)
        raincloud_plot(right_group_data, right_positions, ax, ['#C44E52'] * len(factors), widths=0.3)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(factors, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Factor Score', fontsize=12)
        tract_display_name = right_tract_name.replace('_', ' ') if right_tract_name else tract_base
        ax.set_title(f'Right Hemisphere\n{tract_display_name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4C72B0', alpha=0.7, label='Left/Left>Right'),
            Patch(facecolor='#C44E52', alpha=0.7, label='Right/Right>Left'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# Correlation computation functions removed - no longer needed


# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

def create_html_report(
    output_path: str,
    regions: Sequence[str],
    tracts: Sequence[str],
    scalar_labels: Sequence[str],
    region_factor_plot_paths_norm: Dict[str, str],
    tract_factor_plot_paths_norm: Dict[str, str],
) -> None:
    """Create an HTML report with all plots and correlations."""
    from base64 import b64encode

    def image_to_base64(image_path: str) -> str | None:
        if not image_path or not os.path.exists(image_path):
            return None
        with open(image_path, "rb") as img_f:
            img_data = img_f.read()
        img_base64 = b64encode(img_data).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".png":
            mime = "image/png"
        elif ext in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        else:
            mime = "image/png"
        return f"data:{mime};base64,{img_base64}"

    # Load tract metadata for human-readable names
    tract_label_to_name = load_tract_metadata()
    tract_names_display = [tract_label_to_name.get(t, t) for t in tracts]
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Micro Factor Z Analysis</title>
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
            margin: 10px 0;
        }}
        h2 {{
            color: #333;
            font-size: 18px;
            margin-top: 30px;
        }}
        .section {{
            background-color: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }}
        .plot-item {{
            flex: 1;
            min-width: 300px;
            max-width: 500px;
            text-align: center;
        }}
        .plot-image {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        .correlation-table {{
            margin: 20px 0;
            overflow-x: auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 11px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: #4C72B0;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    <h1>Micro Factor Z Analysis</h1>
    
    <div class="section">
        <h2>Summary</h2>
        <p><strong>GM Regions:</strong> {', '.join(regions)}</p>
        <p><strong>WM Tracts:</strong> {', '.join(tract_names_display)}</p>
        <p><strong>Number of Scalars:</strong> {len(scalar_labels)}</p>
    </div>
"""

    # Add region factor score plots
    if region_factor_plot_paths_norm:
        html += """
    <div class="section">
        <h2>GM Region Factor Abnormality Scores</h2>
        <div class="plot-container">
"""
        for region_base, plot_path in sorted(region_factor_plot_paths_norm.items()):
            img = image_to_base64(plot_path)
            if img:
                html += f"""
            <div class="plot-item" style="max-width: 1400px;">
                <img src="{img}" alt="{region_base}_factor_scores" class="plot-image"/>
            </div>
"""
        html += """
        </div>
    </div>
"""

    # Add tract factor score plots
    if tract_factor_plot_paths_norm:
        html += """
    <div class="section">
        <h2>WM Tract Factor Abnormality Scores</h2>
        <div class="plot-container">
"""
        for tract_base, plot_path in sorted(tract_factor_plot_paths_norm.items()):
            img = image_to_base64(plot_path)
            if img:
                html += f"""
            <div class="plot-item" style="max-width: 1400px;">
                <img src="{img}" alt="{tract_base}_factor_scores" class="plot-image"/>
            </div>
"""
        html += """
        </div>
    </div>
"""
    
    # Correlation plots are computed but not displayed in HTML
    # (Region-tract and profile correlations functionality kept for potential future use)

    html += """
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)


# ============================================================================
# FACTOR SCORE COMPUTATION FOR ALL REGIONS/TRACTS
# ============================================================================


def collect_control_subjects_union_from_gam(
    all_regions: Sequence[str],
    all_tracts: Sequence[str],
    scalar_labels: Sequence[str],
    control_groups: Sequence[str],
) -> List[str]:
    """
    Control subjects who have **any** GAM z-score row in at least one GM region×scalar
    or WM tract×scalar.

    Not intersected with ``subjects_included.csv`` (FA intersection cohort): that file
    lists fewer IDs than have GAM derivatives; we want one output row per control with
    any GAM data. Per-ROI ``compute_factor_scores`` already uses NaN for missing scalars.
    """
    subs: Set[str] = set()
    for region in all_regions:
        for scalar in scalar_labels:
            data = load_gm_region_scalar_data(region, scalar, control_groups)
            if data is not None and not data.empty:
                subs.update(data.index.astype(str))
    for tract in all_tracts:
        for scalar in scalar_labels:
            node_data = load_wm_tract_scalar_data(tract, scalar, control_groups)
            if node_data is not None and not node_data.empty:
                subs.update(node_data.index.astype(str))
    return sorted(subs)


def compute_and_save_all_factor_scores(
    scalar_labels: Sequence[str],
    patient_groups: Sequence[str],
    control_groups: Sequence[str],
    factor_loadings: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Compute factor scores for all epilepsy patients and control subjects across all GM regions and WM tracts.

    Writes one CSV per cohort and factor under ``output_dir``, matching the wide layout of
    ``epilepsy_{factor}_z_scores.csv`` / ``controls_{factor}_z_scores.csv``:

    - ``epilepsy_F1_scores.csv``, … — rows = subjects, columns = GM regions and WM ROI keys
    - ``controls_F1_scores.csv``, … — same, with a leading ``group`` column (control cohort)
    """
    # Discover all regions and tracts
    all_regions = discover_all_gm_regions()
    all_tracts = discover_all_wm_tracts()
    # Filter out excluded tracts
    all_tracts = [t for t in all_tracts if t not in TRACTS_TO_REMOVE]
    
    # Output directory (flat CSVs: epilepsy_Fk_scores.csv, controls_Fk_scores.csv)
    os.makedirs(output_dir, exist_ok=True)
    
    # Full temporal cohort from inclusion metadata (rows with NaN where GAM is missing)
    patient_subjects = load_temporal_patient_subjects_ordered()
    if not patient_subjects and all_regions:
        sample_region = all_regions[0]
        sample_scalar = scalar_labels[0] if scalar_labels else None
        if sample_scalar:
            sample_data = load_gm_region_scalar_data(sample_region, sample_scalar, patient_groups)
            if sample_data is not None:
                patient_subjects = sorted(list(sample_data.index.astype(str)))
    
    # All controls with any GAM table (union across Glasser, 4S156, pyAFQ); not FA-only list
    control_subjects = collect_control_subjects_union_from_gam(
        all_regions, all_tracts, scalar_labels, control_groups
    )
    
    if not patient_subjects and not control_subjects:
        print("Warning: No subjects found. Skipping factor score computation.")
        return
    
    if patient_subjects:
        print(f"Found {len(patient_subjects)} patient subjects")
    if control_subjects:
        print(f"Found {len(control_subjects)} control subjects")
    
    # Load tract metadata to get end labels (needed for both groups)
    tract_metadata_df = load_tract_metadata_full()
    tract_to_end1_label = {}
    tract_to_end2_label = {}
    if not tract_metadata_df.empty and "label" in tract_metadata_df.columns:
        if "end1" in tract_metadata_df.columns:
            tract_to_end1_label = dict(zip(tract_metadata_df["label"], tract_metadata_df["end1"]))
        if "end2" in tract_metadata_df.columns:
            tract_to_end2_label = dict(zip(tract_metadata_df["label"], tract_metadata_df["end2"]))
    
    # Process both groups — accumulate wide tables per factor (columns = ROI keys)
    for group_name, groups, subjects in [
        ("epilepsy", patient_groups, patient_subjects),
        ("controls", control_groups, control_subjects),
    ]:
        if not subjects:
            print(f"\nSkipping {group_name} group - no subjects found.")
            continue

        wide_by_factor: Dict[str, Dict[str, pd.Series]] = {
            str(f): {} for f in factor_loadings.index
        }

        for region in tqdm(all_regions, desc=f"GM regions ({group_name})"):
            roi_data = {}
            for scalar in scalar_labels:
                data = load_gm_region_scalar_data(region, scalar, groups)
                if data is not None:
                    roi_data[scalar] = data

            if not roi_data:
                continue

            subjects_for_region = list(subjects)
            factor_scores = compute_factor_scores(
                roi_data, region, scalar_labels,
                subjects_for_region, factor_loadings,
            )
            if factor_scores.empty:
                continue
            for fac in factor_scores.columns:
                wide_by_factor[str(fac)][region] = factor_scores[fac]

        for tract in tqdm(all_tracts, desc=f"WM tracts ({group_name})"):
            roi_data = {}
            for scalar in scalar_labels:
                node_data = load_wm_tract_scalar_data(tract, scalar, groups)
                if node_data is not None:
                    segment_data = compute_tract_segment_z_scores(node_data)
                    roi_data[scalar] = pd.DataFrame({
                        f"{scalar}_z_end1": segment_data['end1'],
                        f"{scalar}_z_core": segment_data['core'],
                        f"{scalar}_z_end2": segment_data['end2'],
                    })

            if not roi_data:
                continue

            for segment in ['end1', 'core', 'end2']:
                segment_roi_data = {}
                for scalar in scalar_labels:
                    if scalar in roi_data:
                        data = roi_data[scalar]
                        z_col = f"{scalar}_z_{segment}"
                        if z_col in data.columns:
                            segment_roi_data[scalar] = pd.DataFrame({f"{scalar}_z": data[z_col]})

                if not segment_roi_data:
                    continue

                segment_subjects = list(subjects)
                segment_factor_scores = compute_factor_scores(
                    segment_roi_data, f"{tract}_{segment}", scalar_labels,
                    segment_subjects, factor_loadings,
                )
                if segment_factor_scores.empty:
                    continue
                roi_key = _tract_segment_to_roi_key(
                    tract, segment, tract_to_end1_label, tract_to_end2_label,
                )
                for fac in segment_factor_scores.columns:
                    wide_by_factor[str(fac)][roi_key] = segment_factor_scores[fac]

        for fac, col_map in wide_by_factor.items():
            if not col_map:
                continue
            df = pd.DataFrame(col_map)
            df.index.name = "subject"
            if group_name == "controls":
                group_series = pd.Series(
                    [get_group_from_subject_id(s) for s in df.index],
                    index=df.index,
                    name="group",
                )
                df = pd.concat([group_series, df], axis=1)
            csv_path = ospj(output_dir, f"{group_name}_{fac}_scores.csv")
            df.to_csv(csv_path)
            print(f"  Saved {group_name} {fac} factor scores to {csv_path}")
    


# ============================================================================
# BRAIN MAP GENERATION
# ============================================================================


def load_atlas_metadata_for_tsv(tsv_path: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Load label -> index and label -> network (or placeholder) from a parcel TSV."""
    label_to_index: Dict[str, int] = {}
    label_to_network: Dict[str, str] = {}
    if not os.path.exists(tsv_path):
        return label_to_index, label_to_network
    try:
        atlas_df = pd.read_csv(tsv_path, sep="\t")
        if "label" in atlas_df.columns and "index" in atlas_df.columns:
            label_to_index = dict(zip(atlas_df["label"], atlas_df["index"]))
        if "label" in atlas_df.columns and "network_label" in atlas_df.columns:
            network_labels = atlas_df["network_label"].fillna("n/a").astype(str)
            network_labels = network_labels.replace("nan", "n/a")
            label_to_network = dict(zip(atlas_df["label"], network_labels))
        elif "label" in atlas_df.columns:
            # Glasser TSV: no network_label; treat all parcels as cortical for ctx/sctx split
            label_to_network = {str(lab): "ctx" for lab in atlas_df["label"].astype(str)}
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not load atlas metadata from {tsv_path}: {e}")
    return label_to_index, label_to_network


def _create_brain_map_single_atlas(
    region_scores: Dict[str, float],
    factor_name: str,
    lateralization: str,
    output_path: str,
    atlas_nifti_path: str,
    atlas_tsv_path: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_absolute: bool = False,
    *,
    force_all_cortex: bool = False,
) -> None:
    """
    Paint parcel-level scores onto one atlas (4S subcortex or Glasser cortex).

    Per-ROI factor scores are a projection ∑(z×loading); FA fit used stacked features — see module doc.
    """
    if not os.path.exists(atlas_nifti_path):
        print(f"Warning: Atlas NIfTI not found at {atlas_nifti_path}. Skipping brain map.")
        return

    label_to_index, label_to_network = load_atlas_metadata_for_tsv(atlas_tsv_path)
    if not label_to_index:
        print(f"Warning: Could not load atlas metadata from {atlas_tsv_path}. Skipping brain map.")
        return

    try:
        atlas_img = nib.load(atlas_nifti_path)
        atlas_data = atlas_img.get_fdata()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not load atlas nifti: {e}. Skipping brain map.")
        return
    
    # Create arrays for full, cortical, and subcortical maps
    # Initialize with zeros, matching the atlas data type
    # Use .copy() to ensure we have independent arrays, not views
    stat_map_data_full = np.zeros_like(atlas_data, dtype=atlas_data.dtype).copy()
    stat_map_data_ctx = np.zeros_like(atlas_data, dtype=atlas_data.dtype).copy()
    stat_map_data_sctx = np.zeros_like(atlas_data, dtype=atlas_data.dtype).copy()
    
    # Create a mask for background voxels (where atlas_data == 0)
    # This will be used to ensure background remains exactly 0
    background_mask = atlas_data == 0
    
    # Debug: Track label matching statistics
    matched_labels = []
    unmatched_labels = []
    cortical_count = 0
    subcortical_count = 0
    
    # Sort regions by absolute value (ascending) so that most abnormal regions
    # (highest absolute value) are processed last and have precedence when regions overlap.
    # But keep the signed score values (don't apply abs() unless use_absolute=True).
    sorted_regions = sorted(region_scores.items(), key=lambda x: abs(x[1]) if not pd.isna(x[1]) else 0)
    
    for region_label, score in sorted_regions:
        if pd.isna(score):
            continue
        
        # Store the original signed score for intensity
        signed_score = score
        
        # Apply absolute value only if requested (for display/intensity)
        if use_absolute:
            score = abs(score)
        else:
            # Keep signed value for intensity
            score = signed_score
        
        # Handle both "LH_/RH_" and "LH-/RH-" prefixes
        atlas_idx = None
        network_label = None
        matched_label = None
        
        # Try exact match first
        if region_label in label_to_index:
            atlas_idx = label_to_index[region_label]
            # Get network_label - use the value from the dictionary, or "n/a" if not found
            # Both dictionaries should have the same keys since they're built from the same TSV
            if region_label in label_to_network:
                network_label = label_to_network[region_label]
            else:
                network_label = "n/a"  # Default if somehow missing
            matched_label = region_label
        else:
            # Try with hyphen instead of underscore (for first separator only)
            # e.g., "LH_Hippocampus" -> "LH-Hippocampus"
            if region_label.startswith("LH_") or region_label.startswith("RH_"):
                region_label_alt = region_label.replace("_", "-", 1)  # Only replace first occurrence
                if region_label_alt in label_to_index:
                    atlas_idx = label_to_index[region_label_alt]
                    if region_label_alt in label_to_network:
                        network_label = label_to_network[region_label_alt]
                    else:
                        network_label = "n/a"  # Default if somehow missing
                    matched_label = region_label_alt
            # Try with underscore instead of hyphen (for first separator only)
            elif region_label.startswith("LH-") or region_label.startswith("RH-"):
                region_label_alt = region_label.replace("-", "_", 1)  # Only replace first occurrence
                if region_label_alt in label_to_index:
                    atlas_idx = label_to_index[region_label_alt]
                    if region_label_alt in label_to_network:
                        network_label = label_to_network[region_label_alt]
                    else:
                        network_label = "n/a"  # Default if somehow missing
                    matched_label = region_label_alt

        # Glasser-style labels (Left_*/Right_*) from 4S-style LH_*/RH_* prefixes
        if atlas_idx is None and region_label.startswith("LH_"):
            gl = "Left_" + region_label[3:]
            if gl in label_to_index:
                atlas_idx = label_to_index[gl]
                network_label = label_to_network.get(gl, "ctx")
                matched_label = gl
        if atlas_idx is None and region_label.startswith("RH_"):
            gl = "Right_" + region_label[3:]
            if gl in label_to_index:
                atlas_idx = label_to_index[gl]
                network_label = label_to_network.get(gl, "ctx")
                matched_label = gl

        if atlas_idx is not None:
            # Set all voxels with this index to the factor score
            # Only update voxels that are NOT background (atlas_idx != 0)
            mask = (atlas_data == atlas_idx) & (~background_mask)
            
            # Full map: all regions
            stat_map_data_full[mask] = score
            
            # Determine if cortical or subcortical based on network_label
            # Subcortical regions have network_label == "n/a"
            # Cortical regions have network_label with actual network names (Vis, Default, etc.)
            # Convert to string, strip whitespace, and handle None/NaN values
            if network_label is None:
                network_label_str = "n/a"
            else:
                network_label_str = str(network_label).strip()
                # Handle pandas NaN string representation
                if network_label_str.lower() == "nan" or network_label_str == "":
                    network_label_str = "n/a"

            if force_all_cortex:
                stat_map_data_ctx[mask] = score
                cortical_count += 1
                matched_labels.append((region_label, matched_label, atlas_idx, "cortical", "forced_ctx"))
            elif network_label_str == "n/a":
                stat_map_data_sctx[mask] = score
                subcortical_count += 1
                matched_labels.append((region_label, matched_label, atlas_idx, "subcortical", network_label_str))
            else:
                stat_map_data_ctx[mask] = score
                cortical_count += 1
                matched_labels.append((region_label, matched_label, atlas_idx, "cortical", network_label_str))
        else:
            unmatched_labels.append(region_label)
    
    # Only print warnings for unmatched labels
    if unmatched_labels and len(unmatched_labels) > 10:
        print(f"Warning: {len(unmatched_labels)} regions unmatched for {factor_name} ({lateralization})")
    
    # Explicitly set background voxels to exactly 0 to avoid floating point precision issues
    stat_map_data_full[background_mask] = 0.0
    stat_map_data_ctx[background_mask] = 0.0
    stat_map_data_sctx[background_mask] = 0.0
    
    # Debug: Check if negative values are present in the NIfTI data before saving
    if not use_absolute:
        ctx_nonzero = stat_map_data_ctx[stat_map_data_ctx != 0]
        sctx_nonzero = stat_map_data_sctx[stat_map_data_sctx != 0]
        if len(ctx_nonzero) > 0:
            ctx_has_negative = np.any(ctx_nonzero < 0)
            print(f"    {factor_name} {lateralization} cortex NIfTI: min={ctx_nonzero.min():.3f}, max={ctx_nonzero.max():.3f}, has_negative={ctx_has_negative}, nonzero_count={len(ctx_nonzero)}")
        if len(sctx_nonzero) > 0:
            sctx_has_negative = np.any(sctx_nonzero < 0)
            print(f"    {factor_name} {lateralization} subcortex NIfTI: min={sctx_nonzero.min():.3f}, max={sctx_nonzero.max():.3f}, has_negative={sctx_has_negative}, nonzero_count={len(sctx_nonzero)}")
    
    # Additional safety: set any values that are extremely close to zero (likely floating point errors)
    # to exactly zero. Use a very small threshold (1e-9) to catch precision issues without affecting real data
    threshold = 1e-9
    stat_map_data_full[np.abs(stat_map_data_full) < threshold] = 0.0
    stat_map_data_ctx[np.abs(stat_map_data_ctx) < threshold] = 0.0
    stat_map_data_sctx[np.abs(stat_map_data_sctx) < threshold] = 0.0
    
    # Copy the header to preserve all metadata exactly
    # Use header.copy() to avoid modifying the original
    header_full = atlas_img.header.copy()
    header_ctx = atlas_img.header.copy()
    header_sctx = atlas_img.header.copy()
    
    # Ensure the header's data type matches the data
    # Also ensure scl_slope and scl_inter are set correctly to avoid scaling issues
    header_full.set_data_dtype(stat_map_data_full.dtype)
    header_ctx.set_data_dtype(stat_map_data_ctx.dtype)
    header_sctx.set_data_dtype(stat_map_data_sctx.dtype)
    
    # Reset scaling to 1.0 and intercept to 0.0 to ensure no scaling is applied
    header_full['scl_slope'] = 1.0
    header_full['scl_inter'] = 0.0
    header_ctx['scl_slope'] = 1.0
    header_ctx['scl_inter'] = 0.0
    header_sctx['scl_slope'] = 1.0
    header_sctx['scl_inter'] = 0.0
    
    # Verify arrays are independent before creating NIfTI images
    # Check that ctx and sctx arrays are different (they should have different non-zero voxels)
    ctx_sctx_overlap = np.count_nonzero((stat_map_data_ctx != 0) & (stat_map_data_sctx != 0))
    if ctx_sctx_overlap > 0:
        print(f"  WARNING: Found {ctx_sctx_overlap} voxels with non-zero values in BOTH ctx and sctx arrays!")
    
    # Create nifti images with exact header copy
    # Use .copy() to ensure NIfTI images have their own copy of the data
    stat_map_img_full = nib.Nifti1Image(stat_map_data_full.copy(), atlas_img.affine, header_full)
    stat_map_img_ctx = nib.Nifti1Image(stat_map_data_ctx.copy(), atlas_img.affine, header_ctx)
    stat_map_img_sctx = nib.Nifti1Image(stat_map_data_sctx.copy(), atlas_img.affine, header_sctx)
    
    # Save nifti files with the specified naming convention
    # Extract factor number from factor_name (e.g., "Factor1" -> "1", "F1" -> "1", "1" -> "1")
    # Handle "Factor1", "F1", or just "1" formats
    if "Factor" in factor_name:
        factor_num = factor_name.replace("Factor", "")
    elif factor_name.startswith("F") and len(factor_name) > 1:
        # Handle "F1", "F2", etc. - extract just the number part
        factor_num = factor_name[1:]  # Remove the "F" prefix
    else:
        # If it's already just a number, use as-is
        factor_num = factor_name
    
    # Get the directory and base name from output_path
    output_dir = os.path.dirname(output_path)
    # The user wants: F{factor}_{left/right}_lateralized_brain_map.nii.gz for patient groups
    # For controls: F{factor}_controls_brain_map.nii.gz (no "lateralized")
    if lateralization == "controls":
        base_name = f"F{factor_num}_controls_brain_map"
    else:
        base_name = f"F{factor_num}_{lateralization}_lateralized_brain_map"
    
    nii_full_path = ospj(output_dir, f"{base_name}.nii.gz")
    nii_ctx_path = ospj(output_dir, f"{base_name}_ctx.nii.gz")
    nii_sctx_path = ospj(output_dir, f"{base_name}_sctx.nii.gz")
    
    nib.save(stat_map_img_full, nii_full_path)
    nib.save(stat_map_img_ctx, nii_ctx_path)
    nib.save(stat_map_img_sctx, nii_sctx_path)
    
    # Create brain map visualization: cortex on top, subcortex on bottom
    # Both use the same formatting as the whole-brain plot
    import tempfile
    from matplotlib import image as mpimg
    
    # Determine colorbar range if vmin/vmax provided
    # Use the same range for both cortex and subcortex within each lateralization group
    if vmin is not None and vmax is not None:
        if use_absolute:
            # For absolute values, range is 0 to max (vmin is already 0, vmax is already max absolute)
            vmin_symmetric = 0
            vmax_symmetric = abs(vmax)
        else:
            # Use symmetric range for better visualization
            abs_max = max(abs(vmin), abs(vmax))
            vmin_symmetric = -abs_max
            vmax_symmetric = abs_max
    else:
        # Compute from data if not provided
        ctx_values = stat_map_data_ctx[stat_map_data_ctx != 0]
        sctx_values = stat_map_data_sctx[stat_map_data_sctx != 0]
        all_values = np.concatenate([ctx_values.flatten(), sctx_values.flatten()])
        if len(all_values) > 0:
            if use_absolute:
                vmin_symmetric = 0
                vmax_symmetric = max(abs(all_values.min()), abs(all_values.max()))
            else:
                abs_max = max(abs(all_values.min()), abs(all_values.max()))
                vmin_symmetric = -abs_max
                vmax_symmetric = abs_max
        else:
            vmin_symmetric = None
            vmax_symmetric = None
    
    # Helper function to create and save a brain map with specific display mode
    def create_and_save_map(stat_map_img, display_mode_str, output_file):
        # Use Reds colormap for absolute values (lower limit 0), RdBu_r for raw values
        cmap_to_use = 'Reds' if use_absolute else 'RdBu_r'
        fig = plt.figure(figsize=(12, 6))
        # For raw values with negative range, use symmetric_cbar=True to ensure proper centering
        # For absolute values, symmetric_cbar doesn't matter since range is 0 to max
        use_symmetric = not use_absolute and (vmin_symmetric is not None and vmin_symmetric < 0)
        display = plotting.plot_glass_brain(
            stat_map_img,
            display_mode=display_mode_str,
            colorbar=False,  # We'll create our own horizontal colorbar
            cmap=cmap_to_use,
            symmetric_cbar=use_symmetric,
            title='',  # No title
            figure=fig,
            vmin=vmin_symmetric,
            vmax=vmax_symmetric,
            plot_abs=False,  # Explicitly set to False for signed values
        )
        # Set colorbar limits - ensure negative values are properly displayed
        if vmin_symmetric is not None and vmax_symmetric is not None:
            # Find and update the image limits in the figure
            for ax in fig.axes:
                for im in ax.get_images():
                    im.set_clim(vmin_symmetric, vmax_symmetric)
                # Also check for collections (surface plots)
                if hasattr(ax, 'collections'):
                    for coll in ax.collections:
                        if hasattr(coll, 'set_clim'):
                            coll.set_clim(vmin_symmetric, vmax_symmetric)
                        # For surface plots, also check if there's a colormap
                        if hasattr(coll, 'set_cmap'):
                            coll.set_cmap(cmap_to_use)
        
        # Make L/R labels larger and add labels to lr plots
        if display_mode_str == 'lr':
            # For lr plots, remove existing L/R labels and add single labels at top
            for ax in fig.axes:
                # Remove existing L/R text labels
                texts_to_remove = []
                for text in ax.texts:
                    text_str = text.get_text().strip()
                    if text_str in ['L', 'R', 'Left', 'Right', 'L.', 'R.']:
                        texts_to_remove.append(text)
                for text in texts_to_remove:
                    text.remove()
            
            # Add one L label at top, positioned at x=0.32
            fig.text(0.32, 0.98, 'L', fontsize=20, fontweight='bold', 
                    ha='center', va='top', color='black')
            # Add one R label at top, positioned at x=0.68
            fig.text(0.68, 0.98, 'R', fontsize=20, fontweight='bold', 
                    ha='center', va='top', color='black')
        else:
            # For other display modes (y), just enlarge existing L/R labels
            for ax in fig.axes:
                for text in ax.texts:
                    text_str = text.get_text().strip()
                    if text_str in ['L', 'R', 'Left', 'Right', 'L.', 'R.']:
                        text.set_fontsize(20)  # Increase from default
                        text.set_fontweight('bold')
        
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    # Create and save 4 separate images: cortex_y, cortex_lr, subcortex_y, subcortex_lr
    ctx_y_output_path = output_path.replace('.png', '_ctx_y.png')
    ctx_lr_output_path = output_path.replace('.png', '_ctx_lr.png')
    sctx_y_output_path = output_path.replace('.png', '_sctx_y.png')
    sctx_lr_output_path = output_path.replace('.png', '_sctx_lr.png')
    
    # Create cortex y-slice view
    create_and_save_map(stat_map_img_ctx, 'y', ctx_y_output_path)
    
    # Create cortex left/right view
    create_and_save_map(stat_map_img_ctx, 'lr', ctx_lr_output_path)
    
    # Create subcortex y-slice view
    create_and_save_map(stat_map_img_sctx, 'y', sctx_y_output_path)
    
    # Create subcortex left/right view
    create_and_save_map(stat_map_img_sctx, 'lr', sctx_lr_output_path)
    
    # Also save legacy combined images for backward compatibility
    ctx_output_path = output_path.replace('.png', '_ctx.png')
    sctx_output_path = output_path.replace('.png', '_sctx.png')
    
    # Save cortex image (using y view as default)
    ctx_img = mpimg.imread(ctx_y_output_path)
    fig_ctx_single = plt.figure(figsize=(12, 6))
    ax_ctx = plt.subplot(1, 1, 1)
    ax_ctx.imshow(ctx_img)
    ax_ctx.axis('off')
    plt.tight_layout()
    plt.savefig(ctx_output_path, dpi=150, bbox_inches="tight")
    plt.close(fig_ctx_single)
    
    # Save subcortex image (using y view as default)
    sctx_img = mpimg.imread(sctx_y_output_path)
    fig_sctx_single = plt.figure(figsize=(12, 6))
    ax_sctx = plt.subplot(1, 1, 1)
    ax_sctx.imshow(sctx_img)
    ax_sctx.axis('off')
    plt.tight_layout()
    plt.savefig(sctx_output_path, dpi=150, bbox_inches="tight")
    plt.close(fig_sctx_single)
    
    # Combine cortex (top) and subcortex (bottom) views for the combined output
    fig_combined = plt.figure(figsize=(16, 12))
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(ctx_img)
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.imshow(sctx_img)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig_combined)


def create_brain_map(
    region_scores: Dict[str, float],
    factor_name: str,
    lateralization: str,
    output_path: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_absolute: bool = False,
) -> None:
    """
    GM factor maps on Glasser (cortex) and 4S156 (subcortex) atlases, plus one stacked legacy PNG.

    Writes ``*_glasser.png`` and ``*_4s_subcortex.png`` (each with the same suffix layout as before:
    ``_ctx_y``, ``_ctx_lr``, ``_sctx_*`` within the single-atlas helper).
    """
    glasser_set = set(get_glasser_regions())
    sctx_set = set(get_subcortex_4s156_regions())
    scores_gl = {k: v for k, v in region_scores.items() if k in glasser_set}
    scores_4s = {k: v for k, v in region_scores.items() if k in sctx_set}

    sub_pngs: List[str] = []
    if scores_gl and os.path.exists(ATLAS_GLASSER_NIFTI_PATH):
        pg = output_path.replace(".png", "_glasser.png")
        _create_brain_map_single_atlas(
            scores_gl, factor_name, lateralization, pg,
            ATLAS_GLASSER_NIFTI_PATH, ATLAS_GLASSER_TSV_PATH,
            vmin, vmax, use_absolute, force_all_cortex=True,
        )
        if os.path.exists(pg):
            sub_pngs.append(pg)
    if scores_4s and os.path.exists(ATLAS_4S_NIFTI_PATH):
        p4 = output_path.replace(".png", "_4s_subcortex.png")
        _create_brain_map_single_atlas(
            scores_4s, factor_name, lateralization, p4,
            ATLAS_4S_NIFTI_PATH, ATLAS_4S_TSV_PATH,
            vmin, vmax, use_absolute, force_all_cortex=False,
        )
        if os.path.exists(p4):
            sub_pngs.append(p4)

    if not sub_pngs:
        print(f"Warning: No Glasser/4S subcortex GM scores for brain map ({factor_name}, {lateralization}).")
        return

    from matplotlib import image as mpimg

    stack: List[np.ndarray] = []
    for p in sub_pngs:
        if os.path.exists(p):
            try:
                stack.append(mpimg.imread(p))
            except Exception:
                pass
    if not stack:
        return
    n = len(stack)
    fig = plt.figure(figsize=(14, 6.5 * n))
    for i, im in enumerate(stack):
        ax = fig.add_subplot(n, 1, i + 1)
        ax.imshow(im)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_wm_tract_brain_map(
    tract_segment_scores: Dict[Tuple[str, str], float],
    factor_name: str,
    lateralization: str,
    output_path_association: str,
    output_path_projection: str,
    tract_metadata_df: pd.DataFrame,
    use_absolute: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Create brain maps showing factor scores for WM tract segments.
    Separates association and projection tracts.
    Builds maps iteratively, with highest loadings taking precedence.
    
    Args:
        tract_segment_scores: Dict mapping (tract, segment) tuple to average factor score
        factor_name: Name of the factor
        lateralization: "left" or "right"
        output_path_association: Path to save association tract brain map
        output_path_projection: Path to save projection tract brain map
        tract_metadata_df: DataFrame with tract metadata (to determine tract type)
        use_absolute: If True, use absolute values
        vmin: Minimum value for colorbar (optional)
        vmax: Maximum value for colorbar (optional)
    """
    endpoint_nii_dir = ospj(PROJECT_ROOT, "data", "atlases", "HCP1065", "endpoint_nii_bin")
    if not os.path.exists(endpoint_nii_dir):
        print(f"Warning: Endpoint NIfTI directory not found at {endpoint_nii_dir}. Skipping WM brain maps.")
        return
    
    # Get tract type mapping
    tract_to_type = {}
    if not tract_metadata_df.empty and "label" in tract_metadata_df.columns and "type" in tract_metadata_df.columns:
        tract_to_type = dict(zip(tract_metadata_df["label"], tract_metadata_df["type"]))
    
    # Separate association and projection tracts
    association_scores = {}
    projection_scores = {}
    
    for (tract, segment), score in tract_segment_scores.items():
        if pd.isna(score):
            continue
        
        # Store original signed score
        signed_score = score
        
        # Apply absolute if needed (for sorting/precedence, but we'll use signed for intensity)
        if use_absolute:
            score = abs(score)
        
        # Determine tract type
        tract_type = tract_to_type.get(tract, "unknown")
        if tract_type == "association":
            association_scores[(tract, segment)] = (signed_score, abs(signed_score))  # Store (signed, abs) tuple
        elif tract_type == "projection":
            projection_scores[(tract, segment)] = (signed_score, abs(signed_score))  # Store (signed, abs) tuple
    
    # Helper function to build a brain map for a set of tract segments
    def build_tract_map(scores_dict, output_path):
        if not scores_dict:
            print(f"Warning: No scores for {output_path}. Skipping.")
            return
        
        # Sort by absolute value (ascending) so most abnormal (highest |value|) are processed last
        # and have precedence when segments overlap. But use signed score for intensity.
        sorted_items = sorted(scores_dict.items(), key=lambda x: x[1][1] if isinstance(x[1], tuple) else abs(x[1]))
        
        # Get reference image from first mask to determine dimensions
        first_tract, first_segment = sorted_items[0][0]
        tract_to_end1 = {}
        tract_to_end2 = {}
        if not tract_metadata_df.empty and "label" in tract_metadata_df.columns:
            if "end1" in tract_metadata_df.columns:
                tract_to_end1 = dict(zip(tract_metadata_df["label"], tract_metadata_df["end1"]))
            if "end2" in tract_metadata_df.columns:
                tract_to_end2 = dict(zip(tract_metadata_df["label"], tract_metadata_df["end2"]))
        
        # Get segment label for first tract
        end1_label = tract_to_end1.get(first_tract, "end1")
        end2_label = tract_to_end2.get(first_tract, "end2")
        segment_to_label = {
            'end1': f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
            'core': 'core',
            'end2': f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
        }
        first_segment_label = segment_to_label.get(first_segment, first_segment)
        
        # Load first mask to get reference
        first_mask_path = ospj(endpoint_nii_dir, f"{first_tract}_{first_segment_label}.nii.gz")
        if not os.path.exists(first_mask_path):
            print(f"Warning: Mask not found: {first_mask_path}")
            return
        
        try:
            ref_img = nib.load(first_mask_path)
            ref_data = ref_img.get_fdata()
            ref_affine = ref_img.affine
            ref_header = ref_img.header.copy()
        except Exception as e:
            print(f"Warning: Could not load reference mask {first_mask_path}: {e}")
            return
        
        # Initialize output map with zeros
        output_map = np.zeros_like(ref_data, dtype=np.float32)
        
        # Iteratively add tract segments (lowest absolute to highest absolute, so highest |value| overwrites)
        for (tract, segment), score_tuple in sorted_items:
            # Extract signed score (for intensity) and absolute score (for sorting)
            if isinstance(score_tuple, tuple):
                signed_score, abs_score = score_tuple
                # Use signed score for intensity, but apply abs() if use_absolute=True
                score = abs(signed_score) if use_absolute else signed_score
            else:
                # Fallback for old format
                score = abs(score_tuple) if use_absolute else score_tuple
            # Get segment label
            end1_label = tract_to_end1.get(tract, "end1")
            end2_label = tract_to_end2.get(tract, "end2")
            segment_to_label = {
                'end1': f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
                'core': 'core',
                'end2': f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
            }
            segment_label = segment_to_label.get(segment, segment)
            
            # Load mask
            mask_path = ospj(endpoint_nii_dir, f"{tract}_{segment_label}.nii.gz")
            if not os.path.exists(mask_path):
                continue
            
            try:
                mask_img = nib.load(mask_path)
                mask_data = mask_img.get_fdata()
                
                # Check dimensions match
                if mask_data.shape != ref_data.shape:
                    print(f"Warning: Mask {mask_path} has different dimensions. Skipping.")
                    continue
                
                # Apply mask: set voxels in mask to this score (overwrites previous values)
                mask = mask_data > 0
                output_map[mask] = score
                
            except Exception as e:
                print(f"Warning: Could not load mask {mask_path}: {e}")
                continue
        
        # Create NIfTI image
        output_img = nib.Nifti1Image(output_map, ref_affine, ref_header)
        output_img.header.set_data_dtype(output_map.dtype)
        output_img.header['scl_slope'] = 1.0
        output_img.header['scl_inter'] = 0.0
        
        # Save NIfTI
        nii_output_path = output_path.replace('.png', '.nii.gz')
        nib.save(output_img, nii_output_path)
        
        # Determine colorbar range (use provided vmin/vmax for consistency across all maps)
        if vmin is not None and vmax is not None:
            if use_absolute:
                # For absolute values, range is 0 to max (vmin is already 0, vmax is already max absolute)
                vmin_symmetric = 0
                vmax_symmetric = abs(vmax)
            else:
                # Use symmetric range for better visualization
                abs_max = max(abs(vmin), abs(vmax))
                vmin_symmetric = -abs_max
                vmax_symmetric = abs_max
        else:
            non_zero_values = output_map[output_map != 0]
            if len(non_zero_values) > 0:
                if use_absolute:
                    vmin_symmetric = 0
                    vmax_symmetric = max(abs(non_zero_values))
                else:
                    abs_max = max(abs(non_zero_values.min()), abs(non_zero_values.max()))
                    vmin_symmetric = -abs_max
                    vmax_symmetric = abs_max
            else:
                vmin_symmetric = None
                vmax_symmetric = None
        
        # Helper function to create and save a brain map with specific display mode
        def create_and_save_map(stat_map_img, display_mode_str, output_file):
            # Use Reds colormap for absolute values (lower limit 0), RdBu_r for raw values
            cmap_to_use = 'Reds' if use_absolute else 'RdBu_r'
            fig = plt.figure(figsize=(12, 6))
            display = plotting.plot_glass_brain(
                stat_map_img,
                display_mode=display_mode_str,
                colorbar=False,  # We'll create our own horizontal colorbar
                cmap=cmap_to_use,
                symmetric_cbar=True if vmin_symmetric is None else False,
                title='',
                figure=fig,
                vmin=vmin_symmetric,
                vmax=vmax_symmetric,
                plot_abs=False,  # Explicitly set to False for signed values
            )
            # Set colorbar limits
            if vmin_symmetric is not None and vmax_symmetric is not None:
                for ax in fig.axes:
                    for im in ax.get_images():
                        im.set_clim(vmin_symmetric, vmax_symmetric)
                    if hasattr(ax, 'collections'):
                        for coll in ax.collections:
                            if hasattr(coll, 'set_clim'):
                                coll.set_clim(vmin_symmetric, vmax_symmetric)
            
            # Make L/R labels larger and add labels to lr plots
            if display_mode_str == 'lr':
                # For lr plots, remove existing L/R labels and add single labels at top
                for ax in fig.axes:
                    # Remove existing L/R text labels
                    texts_to_remove = []
                    for text in ax.texts:
                        text_str = text.get_text().strip()
                        if text_str in ['L', 'R', 'Left', 'Right', 'L.', 'R.']:
                            texts_to_remove.append(text)
                    for text in texts_to_remove:
                        text.remove()
                
                # Add one L label at top, positioned at x=0.32
                fig.text(0.32, 0.98, 'L', fontsize=20, fontweight='bold', 
                        ha='center', va='top', color='black')
                # Add one R label at top, positioned at x=0.68
                fig.text(0.68, 0.98, 'R', fontsize=20, fontweight='bold', 
                        ha='center', va='top', color='black')
            else:
                # For other display modes (y), just enlarge existing L/R labels
                for ax in fig.axes:
                    for text in ax.texts:
                        text_str = text.get_text().strip()
                        if text_str in ['L', 'R', 'Left', 'Right', 'L.', 'R.']:
                            text.set_fontsize(20)  # Increase from default
                            text.set_fontweight('bold')
            
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
        
        # Create and save separate y and lr views
        y_output_path = output_path.replace('.png', '_y.png')
        lr_output_path = output_path.replace('.png', '_lr.png')
        
        # Create y-slice view
        create_and_save_map(output_img, 'y', y_output_path)
        
        # Create left/right view
        create_and_save_map(output_img, 'lr', lr_output_path)
        
    
    # Build association and projection maps
    if association_scores:
        build_tract_map(association_scores, output_path_association)
    if projection_scores:
        build_tract_map(projection_scores, output_path_projection)


# ============================================================================
# MASTER REPORT GENERATION
# ============================================================================

def _tract_segment_to_roi_key(
    tract: str,
    segment: str,
    tract_to_end1_label: Dict[str, str],
    tract_to_end2_label: Dict[str, str],
) -> str:
    """WM ROI column label aligned with ``epilepsy_{factor}_z_scores.csv`` / per-patient z-score exports."""
    end1_label = tract_to_end1_label.get(tract, "end1")
    end2_label = tract_to_end2_label.get(tract, "end2")
    segment_to_label = {
        "end1": f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
        "core": "core",
        "end2": f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
    }
    segment_label = segment_to_label.get(segment, segment)
    if tract.endswith("_L"):
        tract_base = tract[:-2]
        hemi = "L"
    elif tract.endswith("_R"):
        tract_base = tract[:-2]
        hemi = "R"
    else:
        tract_base = tract
        hemi = ""
    if hemi:
        return f"{tract_base}_{hemi}_{segment_label}"
    return f"{tract_base}_{segment_label}"


def _parse_factor_from_cohort_scores_filename(filename: str, cohort: str) -> Optional[str]:
    base = os.path.basename(filename)
    prefix = f"{cohort}_"
    suffix = "_scores.csv"
    if not base.startswith(prefix) or not base.endswith(suffix):
        return None
    return base[len(prefix) : -len(suffix)]


def _consolidated_cohort_factor_score_paths(scores_root: str, cohort: str) -> List[str]:
    return sorted(glob.glob(ospj(scores_root, f"{cohort}_*_scores.csv")))


def _resolve_consolidated_cohort_factor_path(
    scores_root: str, cohort: str, factor_name: str
) -> Optional[str]:
    """Path to ``{cohort}_{factor}_scores.csv`` allowing F1 / Factor1 style names."""
    candidates = [factor_name]
    if factor_name.startswith("F") and len(factor_name) > 1:
        candidates.append(f"Factor{factor_name[1:]}")
    elif factor_name.startswith("Factor"):
        candidates.append(f"F{factor_name[6:]}")
    for c in candidates:
        p = ospj(scores_root, f"{cohort}_{c}_scores.csv")
        if os.path.exists(p):
            return p
    return None


def _infer_cohort_and_root_for_factor_scores_dir(factor_scores_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    From a legacy subdirectory (…/epilepsy/gm_regions, …/controls/wm_tracts) infer cohort and scores root.
    Returns (cohort, root) or (None, None) if not a recognized layout.
    """
    norm = os.path.normpath(factor_scores_dir)
    base = os.path.basename(norm)
    if base in ("gm_regions", "wm_tracts"):
        parent = os.path.dirname(norm)
        cohort = os.path.basename(parent)
        if cohort in ("epilepsy", "controls"):
            root = os.path.dirname(parent)
            return cohort, root
        # Consolidated wide tables live at …/factor_scores/{cohort}_Fk_scores.csv
        return None, parent
    if os.path.basename(norm) in ("epilepsy", "controls"):
        cohort = os.path.basename(norm)
        root = os.path.dirname(norm)
        return cohort, root
    return None, None


def _consolidated_scores_root(factor_scores_dir: str) -> str:
    """
    Directory containing ``{cohort}_F{k}_scores.csv`` wide tables.

    Handles ``factor_scores/``, ``factor_scores/wm_tracts``, and legacy
    ``factor_scores/epilepsy/wm_tracts`` layouts.
    """
    norm = os.path.normpath(factor_scores_dir)
    base = os.path.basename(norm)
    if base in ("gm_regions", "wm_tracts"):
        parent = os.path.dirname(norm)
        if os.path.basename(parent) in ("epilepsy", "controls"):
            return os.path.dirname(parent)
        return parent
    if base in ("epilepsy", "controls"):
        return os.path.dirname(norm)
    return norm


def _wm_factor_scores_dir(factor_scores_dir: str) -> str:
    """Per-ROI WM CSV dir if present; otherwise consolidated scores root."""
    tract_dir = ospj(factor_scores_dir, "wm_tracts")
    if os.path.isdir(tract_dir):
        return tract_dir
    return factor_scores_dir


def _scores_root_from_gm_wm_subdir(factor_scores_dir: str) -> Optional[str]:
    """Deprecated alias for :func:`_consolidated_scores_root`."""
    norm = os.path.normpath(factor_scores_dir)
    if os.path.basename(norm) in ("gm_regions", "wm_tracts"):
        return _consolidated_scores_root(factor_scores_dir)
    return None


# Cache wide consolidated tables: (scores_root, cohort) -> {factor: DataFrame}
_CONSOLIDATED_COHORT_TABLES: Dict[Tuple[str, str], Dict[str, pd.DataFrame]] = {}


def _load_consolidated_cohort_tables(scores_root: str, cohort: str) -> Dict[str, pd.DataFrame]:
    """Load and cache all ``{cohort}_{Fk}_scores.csv`` tables for a scores root."""
    key = (os.path.normpath(scores_root), cohort)
    if key in _CONSOLIDATED_COHORT_TABLES:
        return _CONSOLIDATED_COHORT_TABLES[key]
    tables: Dict[str, pd.DataFrame] = {}
    for path in _consolidated_cohort_factor_score_paths(scores_root, cohort):
        factor = _parse_factor_from_cohort_scores_filename(path, cohort)
        if not factor:
            continue
        try:
            tables[factor] = pd.read_csv(path, index_col=0)
        except Exception:
            continue
    _CONSOLIDATED_COHORT_TABLES[key] = tables
    return tables


def clear_consolidated_factor_scores_cache() -> None:
    """Drop cached wide factor-score tables (e.g. after regenerating CSVs)."""
    _CONSOLIDATED_COHORT_TABLES.clear()


def _get_consolidated_factor_table(
    scores_root: str, cohort: str, factor_name: str
) -> Optional[pd.DataFrame]:
    tables = _load_consolidated_cohort_tables(scores_root, cohort)
    for candidate in (factor_name,):
        if candidate in tables:
            return tables[candidate]
    if factor_name.startswith("F") and len(factor_name) > 1:
        alt = f"Factor{factor_name[1:]}"
        if alt in tables:
            return tables[alt]
    elif factor_name.startswith("Factor"):
        alt = f"F{factor_name[6:]}"
        if alt in tables:
            return tables[alt]
    return None


def _load_consolidated_factor_scores_for_roi(
    scores_root: str, cohort: str, roi_name: str
) -> pd.DataFrame:
    """Subjects × factors for one ROI from ``{cohort}_{Fk}_scores.csv`` wide tables."""
    cols: Dict[str, pd.Series] = {}
    for factor, df in _load_consolidated_cohort_tables(scores_root, cohort).items():
        if roi_name not in df.columns:
            continue
        cols[factor] = df[roi_name]
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols)


def load_factor_scores_from_csv(
    factor_scores_dir: str,
    roi_name: str,
    *,
    cohort: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load factor scores for one ROI (subjects × factors).

    Legacy: ``{roi_name}_factor_scores.csv`` under ``gm_regions`` / ``wm_tracts``.
    Consolidated: ``{cohort}_{F1}_scores.csv`` etc. at the factor_scores root (same column
    layout as ``{cohort}_{F1}_z_scores.csv``).
    """
    csv_path = ospj(factor_scores_dir, f"{roi_name}_factor_scores.csv")
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path, index_col=0)
        except Exception:
            return pd.DataFrame()

    inferred_cohort, root = _infer_cohort_and_root_for_factor_scores_dir(factor_scores_dir)
    cohort_eff = cohort if cohort is not None else inferred_cohort
    if cohort_eff is None:
        return pd.DataFrame()

    scores_root = root if root is not None else _consolidated_scores_root(factor_scores_dir)
    return _load_consolidated_factor_scores_for_roi(scores_root, cohort_eff, roi_name)


def compute_averaged_z_scores_from_individual_patients(
    patient_subjects: Sequence[str],
    factor_scores_dir: str,
    roi_name: str,
    factor_name: str,
    controls_factor_scores_dir: str,
    *,
    cohort: Optional[str] = None,
) -> float:
    """
    Compute z-scores for each individual patient, then average them.
    This is the correct approach: z-score per patient, then average z-scores.
    
    Args:
        patient_subjects: List of patient subject IDs
        factor_scores_dir: Base directory for patient factor scores
        roi_name: ROI name (region like "LH_Hippocampus" or tract_segment like "AF_L_end-A")
        factor_name: Factor name to load
        controls_factor_scores_dir: Base directory for control factor scores
    
    Returns:
        Average z-score (float) or np.nan if not available
    """
    # Get control mean and std
    control_mean, control_std = compute_control_stats(controls_factor_scores_dir, roi_name, factor_name)
    
    if pd.isna(control_mean) or pd.isna(control_std) or control_std == 0:
        return np.nan
    
    # Load factor scores for this ROI
    factor_scores = load_factor_scores_from_csv(factor_scores_dir, roi_name, cohort=cohort)
    if factor_scores.empty:
        return np.nan
    
    # Get factor column name
    factor_col = None
    if factor_name in factor_scores.columns:
        factor_col = factor_name
    elif factor_name.startswith("F") and len(factor_name) > 1:
        alt_factor = f"Factor{factor_name[1:]}"
        if alt_factor in factor_scores.columns:
            factor_col = alt_factor
    elif factor_name.startswith("Factor"):
        alt_factor = f"F{factor_name[6:]}"
        if alt_factor in factor_scores.columns:
            factor_col = alt_factor
    
    if factor_col is None:
        return np.nan
    
    # Compute z-score for each patient, then average
    z_scores = []
    for patient_id in patient_subjects:
        row_key = resolve_subject_key(patient_id, factor_scores.index)
        if row_key is None:
            continue
        patient_score = factor_scores.loc[row_key, factor_col]
        if not pd.isna(patient_score):
            # Compute z-score: (patient_score - control_mean) / control_std
            z_score = (patient_score - control_mean) / control_std
            z_scores.append(z_score)
    
    if not z_scores:
        return np.nan
    
    # Return average z-score
    return np.mean(z_scores)


def compute_per_patient_z_scores(
    patient_subjects: Sequence[str],
    factor_scores_dir: str,
    roi_name: str,
    factor_name: str,
    controls_factor_scores_dir: str,
    *,
    cohort: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute z-scores for each individual patient (relative to control mean/std).
    Returns a dict mapping patient_id -> z_score for the given ROI and factor.

    Args:
        patient_subjects: List of patient subject IDs
        factor_scores_dir: Base directory for patient factor scores
        roi_name: ROI name (region or tract_segment like "AF_L_end-A")
        factor_name: Factor name to load
        controls_factor_scores_dir: Base directory for control factor scores

    Returns:
        Dict mapping each patient_id to z_score; missing factor scores use NaN.
    """
    nan_out = {str(pid): float(np.nan) for pid in patient_subjects}

    control_mean, control_std = compute_control_stats(
        controls_factor_scores_dir, roi_name, factor_name
    )

    if pd.isna(control_mean) or pd.isna(control_std) or control_std == 0:
        return dict(nan_out)

    factor_scores = load_factor_scores_from_csv(factor_scores_dir, roi_name, cohort=cohort)
    if factor_scores.empty:
        return dict(nan_out)

    factor_col = None
    if factor_name in factor_scores.columns:
        factor_col = factor_name
    elif factor_name.startswith("F") and len(factor_name) > 1:
        alt_factor = f"Factor{factor_name[1:]}"
        if alt_factor in factor_scores.columns:
            factor_col = alt_factor
    elif factor_name.startswith("Factor"):
        alt_factor = f"F{factor_name[6:]}"
        if alt_factor in factor_scores.columns:
            factor_col = alt_factor

    if factor_col is None:
        return dict(nan_out)

    out: Dict[str, float] = {}
    for patient_id in patient_subjects:
        pid = str(patient_id)
        row_key = resolve_subject_key(pid, factor_scores.index)
        if row_key is None:
            out[pid] = float(np.nan)
            continue
        patient_score = factor_scores.loc[row_key, factor_col]
        if pd.isna(patient_score):
            out[pid] = float(np.nan)
        else:
            out[pid] = float((patient_score - control_mean) / control_std)
    return out


def compute_control_stats(
    factor_scores_dir: str,
    roi_name: str,
    factor_name: str,
    master_control_subjects: Optional[Set[str]] = None,
) -> Tuple[float, float]:
    """
    Mean and SD of **raw** factor scores for controls at this ROI.

    By default uses **all** subjects in the saved control factor-scores CSV for that ROI
    (every row with non-NaN scores contributes). Pass ``master_control_subjects`` to
    restrict to a subset (e.g. ``load_controls_subjects_included()`` for FA-matched stats).
    """
    scores_root = _consolidated_scores_root(factor_scores_dir)
    factor_scores_df = _get_consolidated_factor_table(scores_root, "controls", factor_name)
    if factor_scores_df is not None:
        try:
            if roi_name not in factor_scores_df.columns:
                return (np.nan, np.nan)
            if master_control_subjects is not None and len(master_control_subjects) > 0:
                factor_scores_df = factor_scores_df.loc[
                    factor_scores_df.index.astype(str).isin(master_control_subjects)
                ]
            factor_col = roi_name
            vals = factor_scores_df[factor_col].dropna()
            if len(vals) < MIN_CONTROLS_FOR_ROI_Z:
                return (np.nan, np.nan)
            mean_val = float(vals.mean())
            std_val = float(vals.std(ddof=0))
            if std_val == 0.0 or np.isnan(std_val):
                return (np.nan, np.nan)
            return (mean_val, std_val)
        except Exception as e:
            print(f"Error computing control stats for {roi_name} (consolidated): {e}")
            return (np.nan, np.nan)

    region_scores_dir = ospj(factor_scores_dir, "gm_regions")
    csv_path = ospj(region_scores_dir, f"{roi_name}_factor_scores.csv")

    if not os.path.exists(csv_path):
        tract_scores_dir = ospj(factor_scores_dir, "wm_tracts")
        csv_path = ospj(tract_scores_dir, f"{roi_name}_factor_scores.csv")
        if not os.path.exists(csv_path):
            return (np.nan, np.nan)

    try:
        factor_scores_df = pd.read_csv(csv_path, index_col=0)
        if master_control_subjects is not None and len(master_control_subjects) > 0:
            factor_scores_df = factor_scores_df.loc[
                factor_scores_df.index.astype(str).isin(master_control_subjects)
            ]

        factor_col = None
        if factor_name in factor_scores_df.columns:
            factor_col = factor_name
        elif factor_name.startswith("F") and len(factor_name) > 1:
            alt_factor = f"Factor{factor_name[1:]}"
            if alt_factor in factor_scores_df.columns:
                factor_col = alt_factor
        elif factor_name.startswith("Factor"):
            alt_factor = f"F{factor_name[6:]}"
            if alt_factor in factor_scores_df.columns:
                factor_col = alt_factor

        if factor_col is None:
            return (np.nan, np.nan)

        vals = factor_scores_df[factor_col].dropna()
        if len(vals) < MIN_CONTROLS_FOR_ROI_Z:
            return (np.nan, np.nan)
        mean_val = float(vals.mean())
        std_val = float(vals.std(ddof=0))
        if std_val == 0.0 or np.isnan(std_val):
            return (np.nan, np.nan)
        return (mean_val, std_val)
    except Exception as e:
        print(f"Error computing control stats for {roi_name}: {e}")
        return (np.nan, np.nan)


def compute_average_factor_scores_by_group(
    factor_scores_dir: str,
    roi_names: Sequence[str],
    all_patients: Sequence[str],
    left_lateralized: Sequence[str],
    right_lateralized: Sequence[str],
    *,
    cohort: str = "epilepsy",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute average factor scores per ROI for different patient groups.
    
    Returns:
        all_patients_avg: DataFrame with ROIs as rows, factors as columns (all patients)
        left_lateralized_avg: DataFrame with ROIs as rows, factors as columns (left lateralized)
        right_lateralized_avg: DataFrame with ROIs as rows, factors as columns (right lateralized)
    """
    all_scores = {}
    left_scores = {}
    right_scores = {}
    
    for roi in roi_names:
        factor_scores = load_factor_scores_from_csv(factor_scores_dir, roi, cohort=cohort)
        if factor_scores.empty:
            continue
        
        # Get available factors
        factors = factor_scores.columns.tolist()
        
        # Compute averages for each group
        all_patient_scores = factor_scores.loc[
            factor_scores.index.isin(all_patients)
        ].mean(axis=0)
        
        left_patient_scores = factor_scores.loc[
            factor_scores.index.isin(left_lateralized)
        ].mean(axis=0)
        
        right_patient_scores = factor_scores.loc[
            factor_scores.index.isin(right_lateralized)
        ].mean(axis=0)
        
        all_scores[roi] = all_patient_scores
        left_scores[roi] = left_patient_scores
        right_scores[roi] = right_patient_scores
    
    all_patients_df = pd.DataFrame(all_scores).T
    left_lateralized_df = pd.DataFrame(left_scores).T
    right_lateralized_df = pd.DataFrame(right_scores).T
    
    return all_patients_df, left_lateralized_df, right_lateralized_df


def compute_average_loadings_controls(
    roi_names: Sequence[str],
    factor_loadings: pd.DataFrame,
    scalar_labels: Sequence[str],
) -> pd.DataFrame:
    """
    Compute average factor loadings per ROI for controls.
    For each ROI and factor, averages the loadings across all scalars that contribute to that factor.
    
    Returns:
        controls_loadings: DataFrame with ROIs as rows, factors as columns (average loadings)
    """
    control_loadings = {}
    
    for roi in roi_names:
        # For each factor, compute average loading across all scalars
        factor_avg_loadings = {}
        
        for factor in factor_loadings.index:
            # Get loadings for this factor across all scalars
            scalar_loadings = []
            for scalar in scalar_labels:
                if scalar in factor_loadings.columns:
                    loading = factor_loadings.loc[factor, scalar]
                    if not pd.isna(loading):
                        scalar_loadings.append(loading)
            
            # Average the loadings
            if scalar_loadings:
                factor_avg_loadings[factor] = np.mean(scalar_loadings)
            else:
                factor_avg_loadings[factor] = np.nan
        
        control_loadings[roi] = pd.Series(factor_avg_loadings)
    
    controls_df = pd.DataFrame(control_loadings).T
    
    
    return controls_df


def compute_segment_loadings_controls(
    tract_names: Sequence[str],
    factor_loadings: pd.DataFrame,
    scalar_labels: Sequence[str],
) -> pd.DataFrame:
    """
    Compute average factor loadings per tract segment for controls.
    Since loadings are scalar-level (not segment-specific), the same loading applies to all segments.
    
    Returns:
        controls_loadings: DataFrame with (tract, segment) as MultiIndex, factors as columns
    """
    control_loadings = []
    
    for tract in tract_names:
        # For each segment, compute average loadings (same for all segments)
        for segment in ['end1', 'core', 'end2']:
            # For each factor, compute average loading across all scalars
            factor_avg_loadings = {}
            
            for factor in factor_loadings.index:
                # Get loadings for this factor across all scalars
                scalar_loadings = []
                for scalar in scalar_labels:
                    if scalar in factor_loadings.columns:
                        loading = factor_loadings.loc[factor, scalar]
                        if not pd.isna(loading):
                            scalar_loadings.append(loading)
                
                # Average the loadings
                if scalar_loadings:
                    factor_avg_loadings[factor] = np.mean(scalar_loadings)
                else:
                    factor_avg_loadings[factor] = np.nan
            
            control_loadings.append((tract, segment, pd.Series(factor_avg_loadings)))
    
    # Convert to DataFrame with MultiIndex
    if control_loadings:
        control_data = {(tract, segment): loadings for tract, segment, loadings in control_loadings}
        controls_df = pd.DataFrame(control_data).T
    else:
        controls_df = pd.DataFrame()
    
    return controls_df


def compute_roi_means_rescaled_controls(
    all_regions: Sequence[str],
    all_tracts: Sequence[str],
    scalar_labels: Sequence[str],
    control_groups: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each statistic, compute region- and tract segment-wise means across controls.
    Rescale such that the minimum mean is 0, and maximum mean is 1.
    
    Returns:
        roi_means_rescaled_gm: DataFrame with regions as rows, scalars as columns
        roi_means_rescaled_wm: DataFrame with (tract, segment) as MultiIndex, scalars as columns
    """
    print("  Computing roi_means_rescaled for controls...")
    print(f"    Processing {len(all_regions)} GM regions and {len(all_tracts)} WM tracts")
    print(f"    Using {len(scalar_labels)} scalars")
    
    # Step 1: Compute raw means for GM regions (using raw statistics, not z-scores)
    print("    Step 1: Computing raw means for GM regions...")
    gm_means = {}
    for region in all_regions:
        region_means = {}
        for scalar in scalar_labels:
            gm_base = get_mni_micro_gm_profile_dir_for_region(region)
            gam_path = ospj(gm_base, region, f"{region}_{scalar}_stat-mean_gam.csv")
            if not os.path.exists(gam_path):
                gam_path = ospj(gm_base, region, f"{region}_{scalar}_gam.csv")
            if os.path.exists(gam_path):
                try:
                    gam_data = pd.read_csv(gam_path)
                    group_data = gam_data[gam_data["group"].isin(control_groups)].copy()
                    if not group_data.empty and scalar in group_data.columns:
                        mean_val = group_data[scalar].mean()
                        region_means[scalar] = mean_val
                except Exception as e:  # noqa: BLE001
                    pass
        if region_means:
            gm_means[region] = pd.Series(region_means)
    
    gm_means_df = pd.DataFrame(gm_means).T if gm_means else pd.DataFrame()
    print(f"      Computed means for {len(gm_means)} GM regions")
    
    # Step 2: Compute raw means for WM tract segments
    print("    Step 2: Computing raw means for WM tract segments...")
    wm_means = []
    tract_metadata_df = load_tract_metadata_full()
    tract_to_end1_label = {}
    tract_to_end2_label = {}
    if not tract_metadata_df.empty and "label" in tract_metadata_df.columns:
        if "end1" in tract_metadata_df.columns:
            tract_to_end1_label = dict(zip(tract_metadata_df["label"], tract_metadata_df["end1"]))
        if "end2" in tract_metadata_df.columns:
            tract_to_end2_label = dict(zip(tract_metadata_df["label"], tract_metadata_df["end2"]))
    
    for tract in tqdm(all_tracts, desc="      Processing WM tracts", leave=False):
        segment_means_dict = {'end1': {}, 'core': {}, 'end2': {}}
        
        for scalar in scalar_labels:
            gam_path = ospj(WM_PROFILE_DIR_PYAFQ, tract, f"{tract}_{scalar}_stat-mean_gam.csv")
            if not os.path.exists(gam_path):
                gam_path = ospj(WM_PROFILE_DIR_PYAFQ, tract, f"{tract}_{scalar}_gam.csv")
            if os.path.exists(gam_path):
                try:
                    gam_data = pd.read_csv(gam_path)
                    group_data = gam_data[gam_data["group"].isin(control_groups)].copy()
                    if not group_data.empty:
                        # Get raw node columns (node1, node2, ..., node100)
                        node_cols = [f'node{i}' for i in range(1, N_NODES + 1)]
                        missing_cols = [col for col in node_cols if col not in group_data.columns]
                        if not missing_cols:
                            # Compute segment means from raw node data
                            for segment in ['end1', 'core', 'end2']:
                                if segment == 'end1':
                                    segment_nodes = END1_NODES
                                elif segment == 'core':
                                    segment_nodes = CORE_NODES
                                else:  # end2
                                    segment_nodes = END2_NODES
                                
                                # Get node columns for this segment (1-indexed to 0-indexed)
                                segment_node_cols = [f'node{i}' for i in segment_nodes]
                                segment_data = group_data[segment_node_cols]
                                # Compute mean across nodes for each subject, then mean across subjects
                                mean_val = segment_data.mean(axis=1).mean()
                                segment_means_dict[segment][scalar] = mean_val
                except Exception as e:  # noqa: BLE001
                    pass
        
        # Add to wm_means list
        for segment in ['end1', 'core', 'end2']:
            if segment_means_dict[segment]:
                wm_means.append((tract, segment, pd.Series(segment_means_dict[segment])))
    
    if wm_means:
        wm_means_data = {(tract, segment): means for tract, segment, means in wm_means}
        wm_means_df = pd.DataFrame(wm_means_data).T
    else:
        wm_means_df = pd.DataFrame()
    print(f"      Computed means for {len(wm_means)} WM tract segments")
    
    # Step 3: Rescale each scalar to 0-1 range (across all regions/tracts)
    print("    Step 3: Rescaling means to 0-1 range for each scalar...")
    gm_means_rescaled = pd.DataFrame(index=gm_means_df.index, columns=gm_means_df.columns)
    for scalar in scalar_labels:
        if scalar in gm_means_df.columns:
            values = gm_means_df[scalar].dropna()
            if len(values) > 0:
                min_val = values.min()
                max_val = values.max()
                if max_val != min_val:
                    gm_means_rescaled[scalar] = (gm_means_df[scalar] - min_val) / (max_val - min_val)
                else:
                    gm_means_rescaled[scalar] = 0.0
    
    wm_means_rescaled = pd.DataFrame(index=wm_means_df.index, columns=wm_means_df.columns)
    for scalar in scalar_labels:
        if scalar in wm_means_df.columns:
            values = wm_means_df[scalar].dropna()
            if len(values) > 0:
                min_val = values.min()
                max_val = values.max()
                if max_val != min_val:
                    wm_means_rescaled[scalar] = (wm_means_df[scalar] - min_val) / (max_val - min_val)
                else:
                    wm_means_rescaled[scalar] = 0.0
    
    print(f"    Completed: {len(gm_means_rescaled)} GM regions, {len(wm_means_rescaled)} WM segments")
    return gm_means_rescaled, wm_means_rescaled


def compute_roi_factor_scores_from_rescaled(
    roi_means_rescaled_gm: pd.DataFrame,
    roi_means_rescaled_wm: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    scalar_labels: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each factor, multiply roi_means_rescaled by the signed factor loadings,
    producing roi_factor_scores.
    
    Returns:
        roi_factor_scores_gm: DataFrame with regions as rows, factors as columns
        roi_factor_scores_wm: DataFrame with (tract, segment) as MultiIndex, factors as columns
    """
    print("  Computing roi_factor_scores from rescaled means and signed loadings...")
    print(f"    Processing {len(roi_means_rescaled_gm)} GM regions and {len(roi_means_rescaled_wm)} WM segments")
    print(f"    Using {len(factor_loadings.index)} factors")
    
    # Compute GM factor scores
    gm_factor_scores = {}
    for region in roi_means_rescaled_gm.index:
        factor_scores = {}
        for factor in factor_loadings.index:
            score = 0.0
            for scalar in scalar_labels:
                if scalar in roi_means_rescaled_gm.columns and scalar in factor_loadings.columns:
                    rescaled_mean = roi_means_rescaled_gm.loc[region, scalar]
                    loading = factor_loadings.loc[factor, scalar]
                    if not pd.isna(rescaled_mean) and not pd.isna(loading):
                        score += rescaled_mean * loading
            factor_scores[factor] = score
        gm_factor_scores[region] = pd.Series(factor_scores)
    
    gm_factor_scores_df = pd.DataFrame(gm_factor_scores).T if gm_factor_scores else pd.DataFrame()
    print(f"    Computed factor scores for {len(gm_factor_scores)} GM regions")
    
    # Compute WM factor scores
    wm_factor_scores = {}
    for (tract, segment) in roi_means_rescaled_wm.index:
        factor_scores = {}
        for factor in factor_loadings.index:
            score = 0.0
            for scalar in scalar_labels:
                if scalar in roi_means_rescaled_wm.columns and scalar in factor_loadings.columns:
                    rescaled_mean = roi_means_rescaled_wm.loc[(tract, segment), scalar]
                    loading = factor_loadings.loc[factor, scalar]
                    if not pd.isna(rescaled_mean) and not pd.isna(loading):
                        score += rescaled_mean * loading
            factor_scores[factor] = score
        wm_factor_scores[(tract, segment)] = pd.Series(factor_scores)
    
    if wm_factor_scores:
        wm_factor_scores_df = pd.DataFrame(wm_factor_scores).T
    else:
        wm_factor_scores_df = pd.DataFrame()
    
    print(f"    Computed factor scores for {len(wm_factor_scores)} WM segments")
    print("  Completed roi_factor_scores computation")
    return gm_factor_scores_df, wm_factor_scores_df


def compute_factor_z_scores(
    patient_factor_scores: pd.DataFrame,
    control_factor_scores_dir: str,
    roi_name: str,
    factor_name: str,
) -> float:
    """
    Compute average factor z-score for a group of patients.
    First computes z-scores for each individual patient, then averages them.
    
    Args:
        patient_factor_scores: DataFrame with patient factor scores (index: subjects, columns: factors)
        control_factor_scores_dir: Directory containing control factor scores
        roi_name: ROI name
        factor_name: Factor name
    
    Returns:
        Average z-score (float) or np.nan if not available
    """
    # Get control mean and std
    control_mean, control_std = compute_control_stats(control_factor_scores_dir, roi_name, factor_name)
    
    if pd.isna(control_mean) or pd.isna(control_std) or control_std == 0:
        return np.nan
    
    # Get factor column name
    factor_col = None
    if factor_name in patient_factor_scores.columns:
        factor_col = factor_name
    else:
        # Try alternative factor name
        if factor_name.startswith("F") and len(factor_name) > 1:
            alt_factor = f"Factor{factor_name[1:]}"
            if alt_factor in patient_factor_scores.columns:
                factor_col = alt_factor
        elif factor_name.startswith("Factor"):
            alt_factor = f"F{factor_name[6:]}"
            if alt_factor in patient_factor_scores.columns:
                factor_col = alt_factor
    
    if factor_col is None:
        return np.nan
    
    # Compute z-score for each individual patient, then average
    z_scores = []
    for patient_id in patient_factor_scores.index:
        patient_score = patient_factor_scores.loc[patient_id, factor_col]
        if not pd.isna(patient_score):
            # Compute z-score: (patient_score - control_mean) / control_std
            z_score = (patient_score - control_mean) / control_std
            z_scores.append(z_score)
    
    if not z_scores:
        return np.nan
    
    # Return average z-score
    return np.mean(z_scores)


def load_control_z_scores_for_roi(
    controls_factor_scores_dir: str,
    roi_name: str,
    factors: List[str],
    control_subjects: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """
    Load control z-scores for a ROI/region across all factors.
    
    Args:
        controls_factor_scores_dir: Base directory for control factor scores
        roi_name: ROI name (region like "LH_Hippocampus" or tract_segment like "AF_L_end-A")
        factors: List of factor names (e.g., ["F1", "F2", "F3"])
        control_subjects: Optional list of control subject IDs to filter
    
    Returns:
        numpy array of shape (n_controls, n_factors) with z-scores, or None if not available
    """
    scores_root = _consolidated_scores_root(controls_factor_scores_dir)
    dfs = [
        _get_consolidated_factor_table(scores_root, "controls", f) for f in factors
    ]
    if all(df is not None for df in dfs):
        try:
            for df in dfs:
                assert df is not None
                if roi_name not in df.columns:
                    return None

            def _matching_subject_strings(available: Set[str]) -> Set[str]:
                if not control_subjects:
                    return available
                control_subjects_set = {str(s) for s in control_subjects}
                matching = available.intersection(control_subjects_set)
                if matching:
                    return matching
                control_subjects_no_prefix = [
                    str(s).replace("sub-", "") if str(s).startswith("sub-") else str(s)
                    for s in control_subjects
                ]
                available_no_prefix = {
                    str(s).replace("sub-", "") if str(s).startswith("sub-") else str(s)
                    for s in available
                }
                match_no = set(control_subjects_no_prefix).intersection(available_no_prefix)
                if not match_no:
                    return set()
                return {
                    s
                    for s in available
                    if (
                        str(s).replace("sub-", "") if str(s).startswith("sub-") else str(s)
                    )
                    in match_no
                }

            match_sets = [
                _matching_subject_strings(set(d.index.astype(str))) for d in dfs
            ]
            if any(len(m) == 0 for m in match_sets):
                return None
            common_sub = set.intersection(*match_sets)
            if not common_sub:
                return None
            order = sorted(common_sub)

            control_z_scores = []
            for df in dfs:
                idx_by_str = {str(i): i for i in df.index}
                raw_vals = []
                for s in order:
                    if s not in idx_by_str:
                        return None
                    raw_vals.append(df.loc[idx_by_str[s], roi_name])
                col = pd.Series(raw_vals, index=order, dtype=float)
                control_mean = col.mean()
                control_std = col.std(ddof=0)
                if pd.isna(control_mean) or pd.isna(control_std) or control_std == 0:
                    return None
                z_scores = (col - control_mean) / control_std
                control_z_scores.append(z_scores.values)
            control_z_array = np.column_stack(control_z_scores)
            valid_rows = ~np.isnan(control_z_array).any(axis=1)
            if valid_rows.sum() == 0:
                return None
            return control_z_array[valid_rows]
        except Exception:
            pass

    # Legacy: per-ROI CSV under gm_regions / wm_tracts
    region_scores_dir = ospj(controls_factor_scores_dir, "gm_regions")
    csv_path = ospj(region_scores_dir, f"{roi_name}_factor_scores.csv")
    
    if not os.path.exists(csv_path):
        # Try tract segment
        tract_scores_dir = ospj(controls_factor_scores_dir, "wm_tracts")
        csv_path = ospj(tract_scores_dir, f"{roi_name}_factor_scores.csv")
        if not os.path.exists(csv_path):
            return None
    
    try:
        factor_scores_df = pd.read_csv(csv_path, index_col=0)
        
        # Filter to control subjects if provided
        if control_subjects:
            # Try to match subjects (handle "sub-" prefix variations)
            available_subjects = set(factor_scores_df.index)
            control_subjects_set = set(control_subjects)
            matching_subjects = available_subjects.intersection(control_subjects_set)
            
            if len(matching_subjects) == 0:
                # Try removing "sub-" prefix
                control_subjects_no_prefix = [s.replace("sub-", "") if s.startswith("sub-") else s for s in control_subjects]
                available_subjects_no_prefix = {s.replace("sub-", "") if s.startswith("sub-") else s for s in available_subjects}
                matching_subjects_no_prefix = set(control_subjects_no_prefix).intersection(available_subjects_no_prefix)
                if matching_subjects_no_prefix:
                    matching_subjects = {s for s in available_subjects 
                                       if (s.replace("sub-", "") if s.startswith("sub-") else s) in matching_subjects_no_prefix}
            
            if len(matching_subjects) > 0:
                factor_scores_df = factor_scores_df.loc[factor_scores_df.index.isin(matching_subjects)]
            else:
                return None
        
        # Collect z-scores for each factor
        control_z_scores = []
        for factor in factors:
            # Get the factor column name
            factor_col = None
            if factor in factor_scores_df.columns:
                factor_col = factor
            elif factor.startswith("F") and len(factor) > 1:
                alt_factor = f"Factor{factor[1:]}"
                if alt_factor in factor_scores_df.columns:
                    factor_col = alt_factor
            elif factor.startswith("Factor"):
                alt_factor = f"F{factor[6:]}"
                if alt_factor in factor_scores_df.columns:
                    factor_col = alt_factor
            
            if factor_col is None:
                return None
            
            # Compute z-scores for controls (normalize by control mean and std)
            control_mean = factor_scores_df[factor_col].mean()
            control_std = factor_scores_df[factor_col].std(ddof=0)
            
            if pd.isna(control_mean) or pd.isna(control_std) or control_std == 0:
                return None
            
            # Compute z-scores: (value - control_mean) / control_std
            z_scores = (factor_scores_df[factor_col] - control_mean) / control_std
            control_z_scores.append(z_scores.values)
        
        # Stack into array: (n_controls, n_factors)
        control_z_array = np.column_stack(control_z_scores)
        
        # Remove rows with any NaN values
        valid_rows = ~np.isnan(control_z_array).any(axis=1)
        if valid_rows.sum() == 0:
            return None
        
        return control_z_array[valid_rows]
        
    except Exception as e:
        return None


def compute_across_factor_abs_z_scores(
    all_factor_z_scores: Dict[str, Dict[str, float]],
    all_gm_factor_z_scores: Dict[str, Dict[str, float]],
    factors: List[str],
    controls_factor_scores_dir: Optional[str] = None,
    control_subjects: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute across-factor Mahalanobis distance of factor z-scores from control mean.
    
    Args:
        all_factor_z_scores: Dict mapping factor_name to dict of {roi: z_score} for WM tract segments
        all_gm_factor_z_scores: Dict mapping factor_name to dict of {region: z_score} for GM regions
        factors: List of factor names (e.g., ["F1", "F2", "F3"])
        controls_factor_scores_dir: Base directory for control factor scores (required for Mahalanobis distance)
        control_subjects: Optional list of control subject IDs to use for covariance estimation
    
    Returns:
        tuple: (across_factor_wm_scores, across_factor_gm_scores)
            - across_factor_wm_scores: Dict mapping ROI to Mahalanobis distance
            - across_factor_gm_scores: Dict mapping region to Mahalanobis distance
    """
    from scipy.linalg import LinAlgError
    
    across_factor_wm_scores = {}
    across_factor_gm_scores = {}
    
    # Collect all WM ROIs across all factors
    all_wm_rois = set()
    for factor_scores in all_factor_z_scores.values():
        all_wm_rois.update(factor_scores.keys())
    
    # Compute Mahalanobis distance for each WM ROI
    for roi in all_wm_rois:
        # Collect z-scores for this ROI across all factors
        roi_z_vector = []
        available_factors = []
        for factor in factors:
            if factor in all_factor_z_scores and roi in all_factor_z_scores[factor]:
                z_score = all_factor_z_scores[factor][roi]
                if not pd.isna(z_score):
                    roi_z_vector.append(z_score)
                    available_factors.append(factor)
                else:
                    # Skip this ROI if any factor is missing
                    roi_z_vector = None
                    break
        
        if roi_z_vector is None or len(roi_z_vector) == 0:
            continue
        
        # If we have control data, compute Mahalanobis distance
        if controls_factor_scores_dir and os.path.exists(controls_factor_scores_dir):
            # Load control z-scores for this ROI
            control_z_array = load_control_z_scores_for_roi(
                controls_factor_scores_dir, roi, available_factors, control_subjects
            )
            
            if control_z_array is not None and len(control_z_array) > 0:
                try:
                    # Control mean should be approximately zero (since z-scores are normalized)
                    control_mean = np.zeros(len(available_factors))
                    
                    # Compute covariance matrix from controls
                    if len(control_z_array) > len(available_factors):  # Need more samples than dimensions
                        cov_matrix = np.cov(control_z_array.T, ddof=0)
                        
                        # Check if covariance matrix is invertible
                        if np.linalg.cond(cov_matrix) < 1e12:  # Condition number check
                            # Compute Mahalanobis distance
                            roi_z_array = np.array(roi_z_vector)
                            diff = roi_z_array - control_mean
                            mahal_dist = np.sqrt(diff @ np.linalg.inv(cov_matrix) @ diff)
                            across_factor_wm_scores[roi] = mahal_dist
                        else:
                            # Fallback to Euclidean distance if covariance is singular
                            mahal_dist = np.linalg.norm(roi_z_vector)
                            across_factor_wm_scores[roi] = mahal_dist
                    else:
                        # Not enough samples, use Euclidean distance as fallback
                        mahal_dist = np.linalg.norm(roi_z_vector)
                        across_factor_wm_scores[roi] = mahal_dist
                except (LinAlgError, np.linalg.LinAlgError):
                    # If inversion fails, use Euclidean distance as fallback
                    mahal_dist = np.linalg.norm(roi_z_vector)
                    across_factor_wm_scores[roi] = mahal_dist
            else:
                # No control data available, use Euclidean distance as fallback
                mahal_dist = np.linalg.norm(roi_z_vector)
                across_factor_wm_scores[roi] = mahal_dist
        else:
            # No control data available, use Euclidean distance as fallback
            mahal_dist = np.linalg.norm(roi_z_vector)
            across_factor_wm_scores[roi] = mahal_dist
    
    # Collect all GM regions across all factors
    all_gm_regions = set()
    for gm_scores in all_gm_factor_z_scores.values():
        all_gm_regions.update(gm_scores.keys())
    
    # Compute Mahalanobis distance for each GM region
    for region in all_gm_regions:
        # Collect z-scores for this region across all factors
        region_z_vector = []
        available_factors = []
        for factor in factors:
            if factor in all_gm_factor_z_scores and region in all_gm_factor_z_scores[factor]:
                z_score = all_gm_factor_z_scores[factor][region]
                if not pd.isna(z_score):
                    region_z_vector.append(z_score)
                    available_factors.append(factor)
                else:
                    # Skip this region if any factor is missing
                    region_z_vector = None
                    break
        
        if region_z_vector is None or len(region_z_vector) == 0:
            continue
        
        # If we have control data, compute Mahalanobis distance
        if controls_factor_scores_dir and os.path.exists(controls_factor_scores_dir):
            # Load control z-scores for this region
            control_z_array = load_control_z_scores_for_roi(
                controls_factor_scores_dir, region, available_factors, control_subjects
            )
            
            if control_z_array is not None and len(control_z_array) > 0:
                try:
                    # Control mean should be approximately zero (since z-scores are normalized)
                    control_mean = np.zeros(len(available_factors))
                    
                    # Compute covariance matrix from controls
                    if len(control_z_array) > len(available_factors):  # Need more samples than dimensions
                        cov_matrix = np.cov(control_z_array.T, ddof=0)
                        
                        # Check if covariance matrix is invertible
                        if np.linalg.cond(cov_matrix) < 1e12:  # Condition number check
                            # Compute Mahalanobis distance
                            region_z_array = np.array(region_z_vector)
                            diff = region_z_array - control_mean
                            mahal_dist = np.sqrt(diff @ np.linalg.inv(cov_matrix) @ diff)
                            across_factor_gm_scores[region] = mahal_dist
                        else:
                            # Fallback to Euclidean distance if covariance is singular
                            mahal_dist = np.linalg.norm(region_z_vector)
                            across_factor_gm_scores[region] = mahal_dist
                    else:
                        # Not enough samples, use Euclidean distance as fallback
                        mahal_dist = np.linalg.norm(region_z_vector)
                        across_factor_gm_scores[region] = mahal_dist
                except (LinAlgError, np.linalg.LinAlgError):
                    # If inversion fails, use Euclidean distance as fallback
                    mahal_dist = np.linalg.norm(region_z_vector)
                    across_factor_gm_scores[region] = mahal_dist
            else:
                # No control data available, use Euclidean distance as fallback
                mahal_dist = np.linalg.norm(region_z_vector)
                across_factor_gm_scores[region] = mahal_dist
        else:
            # No control data available, use Euclidean distance as fallback
            mahal_dist = np.linalg.norm(region_z_vector)
            across_factor_gm_scores[region] = mahal_dist
    
    return across_factor_wm_scores, across_factor_gm_scores


def create_across_factor_raincloud(
    across_factor_wm_scores: Dict[str, float],
    across_factor_gm_scores: Dict[str, float],
    output_path: str,
    left_wm_scores: Optional[Dict[str, float]] = None,
    left_gm_scores: Optional[Dict[str, float]] = None,
    right_wm_scores: Optional[Dict[str, float]] = None,
    right_gm_scores: Optional[Dict[str, float]] = None,
) -> None:
    """
    Create a swarmplot with overlaid boxplot comparing across-factor Mahalanobis distances grouped by GM and WM.
    Creates 3 subplots: All temporal, Left temporal, Right temporal.
    Includes statistical comparisons using paired t-tests (when regions can be matched) or independent t-tests.
    Saves test statistics to a CSV file.
    
    Args:
        across_factor_wm_scores: Dict mapping ROI to Mahalanobis distance (WM, all temporal)
        across_factor_gm_scores: Dict mapping region to Mahalanobis distance (GM, all temporal)
        output_path: Path to save the plot PNG (statistics CSV will be saved to same location with _statistics.csv suffix)
        left_wm_scores: Optional dict for left temporal WM (if None, uses all)
        left_gm_scores: Optional dict for left temporal GM (if None, uses all)
        right_wm_scores: Optional dict for right temporal WM (if None, uses all)
        right_gm_scores: Optional dict for right temporal GM (if None, uses all)
    """
    # First pass: collect all data to compute global y-axis limits
    all_values = []
    
    # Collect data from all three groups
    datasets = [
        (across_factor_wm_scores, across_factor_gm_scores),
        (left_wm_scores if left_wm_scores is not None else across_factor_wm_scores,
         left_gm_scores if left_gm_scores is not None else across_factor_gm_scores),
        (right_wm_scores if right_wm_scores is not None else across_factor_wm_scores,
         right_gm_scores if right_gm_scores is not None else across_factor_gm_scores),
    ]
    
    for wm_scores_dict, gm_scores_dict in datasets:
        for region, score in gm_scores_dict.items():
            if not pd.isna(score):
                all_values.append(score)
        for roi_key, score in wm_scores_dict.items():
            if not pd.isna(score):
                all_values.append(score)
    
    # Compute global y-axis limits
    if all_values:
        global_ymin = 0  # Absolute values start at 0
        global_ymax = max(all_values)
        # Add padding
        global_ymax += global_ymax * 0.1
    else:
        global_ymin = 0
        global_ymax = 1
    
    # Store statistics for CSV export
    all_statistics = []
    
    # Helper function to create a single subplot
    def create_subplot(ax, wm_scores_dict, gm_scores_dict, title, group_name, fixed_y_bracket=None):
        from scipy.stats import ttest_rel, ttest_ind
        import seaborn as sns
        
        # Collect GM and WM scores with region/ROI identifiers for pairing
        gm_scores = []
        gm_regions = []
        wm_scores = []
        wm_rois = []
        
        # Process GM regions
        for region, score in gm_scores_dict.items():
            if not pd.isna(score):
                gm_scores.append(score)
                gm_regions.append(region)
        
        # Process WM tract segments
        for roi_key, score in wm_scores_dict.items():
            if not pd.isna(score):
                wm_scores.append(score)
                wm_rois.append(roi_key)
        
        if not gm_scores and not wm_scores:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(title, fontsize=18, fontweight='bold')
            ax.set_ylim(global_ymin, global_ymax)
            return {'stats': None, 'bracket_pos': None}
        
        # Prepare data for plotting
        plot_data = []
        categories = []
        
        for score in gm_scores:
            plot_data.append(score)
            categories.append('Gray Matter')
        
        for score in wm_scores:
            plot_data.append(score)
            categories.append('White Matter')
        
        df = pd.DataFrame({'Category': categories, 'Value': plot_data})
        
        # Perform paired t-test if we can match regions
        # Try to match GM regions to WM ROIs by name similarity
        paired_gm = []
        paired_wm = []
        
        # Simple matching: try to find WM ROIs that contain GM region names
        for i, gm_region in enumerate(gm_regions):
            gm_region_clean = gm_region.replace('LH-', '').replace('RH-', '').replace('LH_', '').replace('RH_', '')
            for j, wm_roi in enumerate(wm_rois):
                # Try to match by substring or similar name
                if gm_region_clean.lower() in wm_roi.lower() or wm_roi.lower() in gm_region_clean.lower():
                    paired_gm.append(gm_scores[i])
                    paired_wm.append(wm_scores[j])
                    break
        
        # If we have enough pairs, do paired t-test; otherwise do independent t-test
        if len(paired_gm) >= 3 and len(paired_wm) >= 3:
            try:
                t_stat, p_value = ttest_rel(paired_gm, paired_wm)
                df_degrees = len(paired_gm) - 1
                test_type = 'paired'
            except:
                # Fallback to independent t-test
                t_stat, p_value = ttest_ind(gm_scores, wm_scores)
                df_degrees = len(gm_scores) + len(wm_scores) - 2
                test_type = 'independent'
        else:
            # Use independent t-test
            if len(gm_scores) >= 2 and len(wm_scores) >= 2:
                t_stat, p_value = ttest_ind(gm_scores, wm_scores)
                df_degrees = len(gm_scores) + len(wm_scores) - 2
                test_type = 'independent'
            else:
                t_stat, p_value, df_degrees = np.nan, np.nan, np.nan
                test_type = 'insufficient_data'
        
        # Store statistics
        all_statistics.append({
            'Group': group_name,
            'Test_Type': test_type,
            'T_Statistic': t_stat,
            'Degrees_of_Freedom': df_degrees,
            'P_Value': p_value,
            'N_GM': len(gm_scores),
            'N_WM': len(wm_scores),
            'N_Paired': len(paired_gm) if len(paired_gm) >= 3 else 0
        })
        
        # Create swarm plot with seaborn
        if sns is not None:
            try:
                # Plot all points with swarmplot
                print(f"Creating swarmplot for 'Across Factor Mahalanobis Distance: Gray vs White Matter' - {group_name} subplot ({len(df)} points)")
                sns.swarmplot(data=df, x='Category', y='Value', ax=ax,
                            palette={'Gray Matter': 'white', 'White Matter': 'white'},
                            size=3.5, alpha=1.0, orient='v',
                            dodge=False, linewidth=0.5)
                
                # Color points: gray for GM, white with black outline for WM
                for collection in ax.collections:
                    offsets = collection.get_offsets()
                    if len(offsets) == 0:
                        continue
                    
                    facecolors = np.array(collection.get_facecolors(), copy=True)
                    if facecolors.ndim == 1:
                        facecolors = np.tile(facecolors, (len(offsets), 1))
                    
                    # Color based on category
                    for i, (x, y) in enumerate(offsets):
                        if i < len(df):
                            cat = df.iloc[i]['Category']
                            if cat == 'Gray Matter':
                                facecolors[i] = [0.5, 0.5, 0.5, 1.0]  # Gray
                            else:
                                facecolors[i] = [1.0, 1.0, 1.0, 1.0]  # White
                    
                    collection.set_facecolors(facecolors)
            except Exception as e:
                print(f"Warning: sns.swarmplot failed: {e}")
        
        # Overlay boxplot without whiskers or fliers (on top of points)
        bp = ax.boxplot([df[df['Category'] == cat]['Value'].values 
                         for cat in ['Gray Matter', 'White Matter']],
                        positions=[0, 1], widths=0.3, patch_artist=True,
                        showfliers=False, whis=0, zorder=10)
        
        for patch in bp['boxes']:
            patch.set_facecolor('none')  # Transparent fill
            patch.set_edgecolor('black')  # Black outline
            patch.set_linewidth(1.5)
        
        # Set color and linewidth for boxplot medians (visible, blue)
        if 'medians' in bp:
            for item in bp['medians']:
                item.set_color('blue')
                item.set_linewidth(1.5)
        
        # Initialize bracket_pos
        bracket_pos = None
        
        # Add significance annotation
        if not (np.isnan(p_value) or np.isnan(t_stat)):
            # Determine significance level
            if p_value < 0.001:
                sig_text = '***'
            elif p_value < 0.01:
                sig_text = '**'
            elif p_value < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            
            # Add significance bracket
            # Use fixed y_bracket if provided (from Left Temporal reference), otherwise calculate from data
            y_max_data = df['Value'].max() if not df.empty else 0
            data_range = y_max_data - df['Value'].min() if not df.empty and len(df) > 1 else y_max_data if not df.empty else 1
            
            if fixed_y_bracket is not None:
                y_bracket = fixed_y_bracket
            else:
                # Position bracket just above the maximum data value, with a small offset
                y_bracket = y_max_data + data_range * 0.08  # Small offset above max data point
                # Store bracket position for return (only if not fixed)
                bracket_pos = y_bracket
            
            # Draw bracket
            bracket_height = data_range * 0.02  # Small bracket height relative to data range
            ax.plot([0, 0, 1, 1], [y_bracket - bracket_height, 
                                   y_bracket, y_bracket, 
                                   y_bracket - bracket_height],
                   'k-', linewidth=1.5, zorder=15)
            # Add significance text
            ax.text(0.5, y_bracket + bracket_height * 0.3, sig_text,
                   ha='center', va='bottom', fontsize=16, fontweight='bold', zorder=15)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Gray Matter', 'White Matter'], fontsize=12)  # Keep x-tick labels smaller
        ax.set_xlabel('Tissue type', fontsize=18, fontweight='bold')
        ax.set_ylabel('Mahalanobis distance', fontsize=18, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold')
        for label in ax.get_yticklabels():
            label.set_fontsize(16)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set consistent y-axis limits
        ax.set_ylim(global_ymin, global_ymax)
        
        # Return statistics and bracket position (if calculated)
        return {'stats': all_statistics[-1] if all_statistics else None, 'bracket_pos': bracket_pos}
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Left temporal (use provided or fallback to all) - create first to get reference bracket position
    left_wm = left_wm_scores if left_wm_scores is not None else across_factor_wm_scores
    left_gm = left_gm_scores if left_gm_scores is not None else across_factor_gm_scores
    result_left = create_subplot(axes[1], left_wm, left_gm, 'Left Temporal', 'Left Temporal')
    
    # Get reference bracket position from Left Temporal
    reference_bracket_pos = result_left.get('bracket_pos') if result_left and result_left.get('bracket_pos') is not None else None
    
    # All temporal - use reference bracket position
    result_all = create_subplot(axes[0], across_factor_wm_scores, across_factor_gm_scores, 'All Temporal', 'All Temporal', fixed_y_bracket=reference_bracket_pos)
    
    # Right temporal (use provided or fallback to all) - use reference bracket position
    right_wm = right_wm_scores if right_wm_scores is not None else across_factor_wm_scores
    right_gm = right_gm_scores if right_gm_scores is not None else across_factor_gm_scores
    result_right = create_subplot(axes[2], right_wm, right_gm, 'Right Temporal', 'Right Temporal', fixed_y_bracket=reference_bracket_pos)
    
    # Add overall title
    fig.suptitle('Across Factor Mahalanobis Distance: Gray vs White Matter', fontsize=20, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(all_statistics)
    csv_path = output_path.replace('.png', '_statistics.csv')
    stats_df.to_csv(csv_path, index=False)
    
    print(f"Across factor Mahalanobis distance swarm plot saved: {output_path}")
    print(f"Statistics saved to: {csv_path}")
    
    # Also create custom version with y-axis max = 1 and lower statistical comparisons
    custom_plot_path = ospj(os.path.dirname(output_path), "Mahalanobis_Grey_White_Custom.png")
    create_across_factor_raincloud_custom(
        across_factor_wm_scores, across_factor_gm_scores, custom_plot_path,
        left_wm_scores=left_wm_scores if left_wm_scores is not None else None,
        left_gm_scores=left_gm_scores if left_gm_scores is not None else None,
        right_wm_scores=right_wm_scores if right_wm_scores is not None else None,
        right_gm_scores=right_gm_scores if right_gm_scores is not None else None,
    )


def create_across_factor_raincloud_custom(
    across_factor_wm_scores: Dict[str, float],
    across_factor_gm_scores: Dict[str, float],
    output_path: str,
    left_wm_scores: Optional[Dict[str, float]] = None,
    left_gm_scores: Optional[Dict[str, float]] = None,
    right_wm_scores: Optional[Dict[str, float]] = None,
    right_gm_scores: Optional[Dict[str, float]] = None,
) -> None:
    """
    Custom version of create_across_factor_raincloud with y-axis max set to 1 
    and statistical comparisons positioned even lower.
    Saves only the PNG file (not embedded in HTML).
    
    Args:
        across_factor_wm_scores: Dict mapping ROI to Mahalanobis distance (WM, all temporal)
        across_factor_gm_scores: Dict mapping region to Mahalanobis distance (GM, all temporal)
        output_path: Path to save the plot PNG
        left_wm_scores: Optional dict for left temporal WM (if None, uses all)
        left_gm_scores: Optional dict for left temporal GM (if None, uses all)
        right_wm_scores: Optional dict for right temporal WM (if None, uses all)
        right_gm_scores: Optional dict for right temporal GM (if None, uses all)
    """
    # Store statistics for CSV export (optional, but keep for consistency)
    all_statistics = []
    
    # Helper function to create a single subplot
    def create_subplot(ax, wm_scores_dict, gm_scores_dict, title, group_name, fixed_y_bracket=None):
        from scipy.stats import ttest_rel, ttest_ind
        import seaborn as sns
        
        # Collect GM and WM scores with region/ROI identifiers for pairing
        gm_scores = []
        gm_regions = []
        wm_scores = []
        wm_rois = []
        
        # Process GM regions
        for region, score in gm_scores_dict.items():
            if not pd.isna(score):
                gm_scores.append(score)
                gm_regions.append(region)
        
        # Process WM tract segments
        for roi_key, score in wm_scores_dict.items():
            if not pd.isna(score):
                wm_scores.append(score)
                wm_rois.append(roi_key)
        
        if not gm_scores and not wm_scores:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(title, fontsize=18, fontweight='bold')
            ax.set_ylim(0, 1)
            return {'stats': None, 'bracket_pos': None}
        
        # Prepare data for plotting
        plot_data = []
        categories = []
        
        for score in gm_scores:
            plot_data.append(score)
            categories.append('Gray Matter')
        
        for score in wm_scores:
            plot_data.append(score)
            categories.append('White Matter')
        
        df = pd.DataFrame({'Category': categories, 'Value': plot_data})
        
        # Perform paired t-test if we can match regions
        paired_gm = []
        paired_wm = []
        
        # Simple matching: try to find WM ROIs that contain GM region names
        for i, gm_region in enumerate(gm_regions):
            gm_region_clean = gm_region.replace('LH-', '').replace('RH-', '').replace('LH_', '').replace('RH_', '')
            for j, wm_roi in enumerate(wm_rois):
                if gm_region_clean.lower() in wm_roi.lower() or wm_roi.lower() in gm_region_clean.lower():
                    paired_gm.append(gm_scores[i])
                    paired_wm.append(wm_scores[j])
                    break
        
        # If we have enough pairs, do paired t-test; otherwise do independent t-test
        if len(paired_gm) >= 3 and len(paired_wm) >= 3:
            try:
                t_stat, p_value = ttest_rel(paired_gm, paired_wm)
                df_degrees = len(paired_gm) - 1
                test_type = 'paired'
            except:
                t_stat, p_value = ttest_ind(gm_scores, wm_scores)
                df_degrees = len(gm_scores) + len(wm_scores) - 2
                test_type = 'independent'
        else:
            if len(gm_scores) >= 2 and len(wm_scores) >= 2:
                t_stat, p_value = ttest_ind(gm_scores, wm_scores)
                df_degrees = len(gm_scores) + len(wm_scores) - 2
                test_type = 'independent'
            else:
                t_stat, p_value, df_degrees = np.nan, np.nan, np.nan
                test_type = 'insufficient_data'
        
        # Store statistics
        all_statistics.append({
            'Group': group_name,
            'Test_Type': test_type,
            'T_Statistic': t_stat,
            'Degrees_of_Freedom': df_degrees,
            'P_Value': p_value,
            'N_GM': len(gm_scores),
            'N_WM': len(wm_scores),
            'N_Paired': len(paired_gm) if len(paired_gm) >= 3 else 0
        })
        
        # Create swarm plot with seaborn
        if sns is not None:
            try:
                print(f"Creating swarmplot for 'Across Factor Mahalanobis Distance: Gray vs White Matter (Custom)' - {group_name} subplot ({len(df)} points)")
                sns.swarmplot(data=df, x='Category', y='Value', ax=ax,
                            palette={'Gray Matter': 'white', 'White Matter': 'white'},
                            size=3.5, alpha=1.0, orient='v',
                            dodge=False, linewidth=0.5)
                
                # Color points: gray for GM, white with black outline for WM
                for collection in ax.collections:
                    offsets = collection.get_offsets()
                    if len(offsets) == 0:
                        continue
                    
                    facecolors = np.array(collection.get_facecolors(), copy=True)
                    if facecolors.ndim == 1:
                        facecolors = np.tile(facecolors, (len(offsets), 1))
                    
                    for i, (x, y) in enumerate(offsets):
                        if i < len(df):
                            cat = df.iloc[i]['Category']
                            if cat == 'Gray Matter':
                                facecolors[i] = [0.5, 0.5, 0.5, 1.0]  # Gray
                            else:
                                facecolors[i] = [1.0, 1.0, 1.0, 1.0]  # White
                    
                    collection.set_facecolors(facecolors)
            except Exception as e:
                print(f"Warning: sns.swarmplot failed: {e}")
        
        # Overlay boxplot without whiskers or fliers (on top of points)
        bp = ax.boxplot([df[df['Category'] == cat]['Value'].values 
                         for cat in ['Gray Matter', 'White Matter']],
                        positions=[0, 1], widths=0.3, patch_artist=True,
                        showfliers=False, whis=0, zorder=10)
        
        for patch in bp['boxes']:
            patch.set_facecolor('none')
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        # Set color and linewidth for boxplot medians (visible, blue)
        if 'medians' in bp:
            for item in bp['medians']:
                item.set_color('blue')
                item.set_linewidth(1.5)
        
        # Initialize bracket_pos
        bracket_pos = None
        
        # Add significance annotation - positioned much lower
        if not (np.isnan(p_value) or np.isnan(t_stat)):
            if p_value < 0.001:
                sig_text = '***'
            elif p_value < 0.01:
                sig_text = '**'
            elif p_value < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            
            # Position bracket at 0.90 of y-axis max (which is 1.0)
            y_bracket = 0.90  # Fixed position at 90% of y-axis
            
            if fixed_y_bracket is not None:
                y_bracket = fixed_y_bracket
            else:
                bracket_pos = y_bracket
            
            # Draw bracket
            bracket_height = 0.015  # Small bracket height
            ax.plot([0, 0, 1, 1], [y_bracket - bracket_height, 
                                   y_bracket, y_bracket, 
                                   y_bracket - bracket_height],
                   'k-', linewidth=1.5, zorder=15)
            # Add significance text - positioned above the bracket
            ax.text(0.5, y_bracket + bracket_height * 2, sig_text,
                   ha='center', va='bottom', fontsize=16, fontweight='bold', zorder=15)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Gray Matter', 'White Matter'], fontsize=12)
        ax.set_xlabel('Tissue type', fontsize=18, fontweight='bold')
        ax.set_ylabel('Mahalanobis distance', fontsize=18, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold')
        for label in ax.get_yticklabels():
            label.set_fontsize(16)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits to 0-1
        ax.set_ylim(0, 1)
        
        return {'stats': all_statistics[-1] if all_statistics else None, 'bracket_pos': bracket_pos}
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Left temporal (use provided or fallback to all) - create first to get reference bracket position
    left_wm = left_wm_scores if left_wm_scores is not None else across_factor_wm_scores
    left_gm = left_gm_scores if left_gm_scores is not None else across_factor_gm_scores
    result_left = create_subplot(axes[1], left_wm, left_gm, 'Left Temporal', 'Left Temporal')
    
    # Get reference bracket position from Left Temporal
    reference_bracket_pos = result_left.get('bracket_pos') if result_left and result_left.get('bracket_pos') is not None else 0.90
    
    # All temporal - use reference bracket position
    result_all = create_subplot(axes[0], across_factor_wm_scores, across_factor_gm_scores, 'All Temporal', 'All Temporal', fixed_y_bracket=reference_bracket_pos)
    
    # Right temporal (use provided or fallback to all) - use reference bracket position
    right_wm = right_wm_scores if right_wm_scores is not None else across_factor_wm_scores
    right_gm = right_gm_scores if right_gm_scores is not None else across_factor_gm_scores
    result_right = create_subplot(axes[2], right_wm, right_gm, 'Right Temporal', 'Right Temporal', fixed_y_bracket=reference_bracket_pos)
    
    # Add overall title
    fig.suptitle('Across Factor Mahalanobis Distance: Gray vs White Matter (Custom)', fontsize=20, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Custom Mahalanobis distance plot saved: {output_path}")


def create_hemisphere_mahalanobis_plot(across_factor_wm_scores, across_factor_gm_scores,
                                      gm_intervention_dict, tract_intervention_dict,
                                      output_path):
    """
    Create a figure with 3 side-by-side swarm plots comparing left vs right hemispheres:
    1. All regions (GM + WM combined)
    2. Gray matter regions only
    3. White matter tract segments only
    
    Includes statistical comparisons between left and right.
    Colors points by intervention overlap proportion using Reds colormap.
    
    Args:
        across_factor_wm_scores: Dict mapping ROI to Mahalanobis distance (WM)
        across_factor_gm_scores: Dict mapping region to Mahalanobis distance (GM)
        gm_intervention_dict: Dict mapping region name to intervention proportion (0.0 to 1.0)
        tract_intervention_dict: Dict mapping (tract, segment) tuple to intervention proportion (0.0 to 1.0)
        output_path: Path to save the plot PNG
    """
    from matplotlib.cm import Reds
    from matplotlib.colors import Normalize
    from scipy.stats import ttest_rel, ttest_ind
    try:
        import seaborn as sns
    except ImportError:
        print("Warning: seaborn not available, using basic plot")
        sns = None
    
    # Helper function to determine hemisphere
    def get_hemisphere(roi_name):
        """Determine if ROI is left, right, or bilateral."""
        # Check for GM region prefixes (LH-/RH- or LH_/RH_)
        if roi_name.startswith("LH-") or roi_name.startswith("LH_"):
            return "Left"
        elif roi_name.startswith("RH-") or roi_name.startswith("RH_"):
            return "Right"
        # Check for WM tract hemisphere indicators (_L or _R, possibly followed by segment)
        # Format: {tract}_{L or R}_{segment} or {tract}_{L or R}
        parts = roi_name.split("_")
        if len(parts) >= 2:
            # Check if second-to-last or last part is L or R
            if parts[-1] in ["L", "R"] or (len(parts) >= 2 and parts[-2] in ["L", "R"]):
                hemi_idx = -1 if parts[-1] in ["L", "R"] else -2
                if parts[hemi_idx] == "L":
                    return "Left"
                elif parts[hemi_idx] == "R":
                    return "Right"
        # Check for ending patterns
        if roi_name.endswith("_L") or roi_name.endswith("_L."):
            return "Left"
        elif roi_name.endswith("_R") or roi_name.endswith("_R."):
            return "Right"
        return None  # Bilateral or unknown
    
    # Separate GM regions into left and right
    gm_left_scores = []
    gm_right_scores = []
    gm_left_intervention = []
    gm_right_intervention = []
    
    for region, score in across_factor_gm_scores.items():
        if pd.isna(score):
            continue
        hemi = get_hemisphere(region)
        intervention = gm_intervention_dict.get(region, 0) if gm_intervention_dict else 0
        if hemi == "Left":
            gm_left_scores.append(score)
            gm_left_intervention.append(intervention)
        elif hemi == "Right":
            gm_right_scores.append(score)
            gm_right_intervention.append(intervention)
    
    # Separate WM tracts into left and right
    wm_left_scores = []
    wm_right_scores = []
    wm_left_intervention = []
    wm_right_intervention = []
    
    for roi_key, score in across_factor_wm_scores.items():
        if pd.isna(score):
            continue
        hemi = get_hemisphere(roi_key)
        intervention = 0
        if tract_intervention_dict:
            parts = roi_key.rsplit("_", 2)
            if len(parts) == 3:
                tract_base, hemi_part, seg_label = parts
                tract_name = f"{tract_base}_{hemi_part}"
                intervention = tract_intervention_dict.get((tract_name, seg_label), 0)
        if hemi == "Left":
            wm_left_scores.append(score)
            wm_left_intervention.append(intervention)
        elif hemi == "Right":
            wm_right_scores.append(score)
            wm_right_intervention.append(intervention)
    
    # Combine for "All regions" plot
    all_left_scores = gm_left_scores + wm_left_scores
    all_right_scores = gm_right_scores + wm_right_scores
    all_left_intervention = gm_left_intervention + wm_left_intervention
    all_right_intervention = gm_right_intervention + wm_right_intervention
    
    # Create figure with 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Store statistics for CSV export
    all_statistics = []
    
    # Helper function to create a single subplot
    def create_subplot(ax, left_scores, right_scores, left_intervention, right_intervention, title, group_name):
        """Create a single subplot comparing left vs right."""
        from scipy.stats import ttest_rel, ttest_ind
        
        # Prepare data for plotting
        plot_data = []
        categories = []
        intervention_list = []
        
        for score, intervention in zip(left_scores, left_intervention):
            plot_data.append(score)
            categories.append('Left')
            intervention_list.append(intervention)
        
        for score, intervention in zip(right_scores, right_intervention):
            plot_data.append(score)
            categories.append('Right')
            intervention_list.append(intervention)
        
        df = pd.DataFrame({
            'Hemisphere': categories,
            'Value': plot_data,
            'Intervention': intervention_list
        })
        
        if df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(title, fontsize=14, fontweight='bold', fontfamily='Georgia')
            return None
        
        # Perform statistical test
        if len(left_scores) >= 2 and len(right_scores) >= 2:
            # Try to match regions for paired t-test (match by removing hemisphere prefix/suffix)
            paired_left = []
            paired_right = []
            
            # For GM: try to match by removing LH-/RH- prefix
            if group_name == "GM":
                left_regions = [r for r in across_factor_gm_scores.keys() if get_hemisphere(r) == "Left"]
                right_regions = [r for r in across_factor_gm_scores.keys() if get_hemisphere(r) == "Right"]
                for left_region in left_regions:
                    left_base = left_region.replace("LH-", "").replace("RH-", "").replace("LH_", "").replace("RH_", "")
                    for right_region in right_regions:
                        right_base = right_region.replace("LH-", "").replace("RH-", "").replace("LH_", "").replace("RH_", "")
                        if left_base == right_base:
                            if left_region in across_factor_gm_scores and right_region in across_factor_gm_scores:
                                left_val = across_factor_gm_scores[left_region]
                                right_val = across_factor_gm_scores[right_region]
                                if not pd.isna(left_val) and not pd.isna(right_val):
                                    paired_left.append(left_val)
                                    paired_right.append(right_val)
                            break
            # For WM: try to match by removing hemisphere indicator
            elif group_name == "WM":
                left_rois = [r for r in across_factor_wm_scores.keys() if get_hemisphere(r) == "Left"]
                right_rois = [r for r in across_factor_wm_scores.keys() if get_hemisphere(r) == "Right"]
                for left_roi in left_rois:
                    parts_left = left_roi.rsplit("_", 2)
                    if len(parts_left) == 3 and parts_left[1] == "L":
                        left_base = f"{parts_left[0]}_{parts_left[2]}"
                    elif len(parts_left) >= 2 and parts_left[-1] == "L":
                        left_base = "_".join(parts_left[:-1])
                    else:
                        left_base = left_roi
                    for right_roi in right_rois:
                        parts_right = right_roi.rsplit("_", 2)
                        if len(parts_right) == 3 and parts_right[1] == "R":
                            right_base = f"{parts_right[0]}_{parts_right[2]}"
                        elif len(parts_right) >= 2 and parts_right[-1] == "R":
                            right_base = "_".join(parts_right[:-1])
                        else:
                            right_base = right_roi
                        if left_base == right_base:
                            if left_roi in across_factor_wm_scores and right_roi in across_factor_wm_scores:
                                left_val = across_factor_wm_scores[left_roi]
                                right_val = across_factor_wm_scores[right_roi]
                                if not pd.isna(left_val) and not pd.isna(right_val):
                                    paired_left.append(left_val)
                                    paired_right.append(right_val)
                            break
            # For All: try to match both GM and WM
            elif group_name == "All":
                # Match GM regions
                left_gm_regions = [r for r in across_factor_gm_scores.keys() if get_hemisphere(r) == "Left"]
                right_gm_regions = [r for r in across_factor_gm_scores.keys() if get_hemisphere(r) == "Right"]
                for left_region in left_gm_regions:
                    left_base = left_region.replace("LH-", "").replace("RH-", "").replace("LH_", "").replace("RH_", "")
                    for right_region in right_gm_regions:
                        right_base = right_region.replace("LH-", "").replace("RH-", "").replace("LH_", "").replace("RH_", "")
                        if left_base == right_base:
                            if left_region in across_factor_gm_scores and right_region in across_factor_gm_scores:
                                left_val = across_factor_gm_scores[left_region]
                                right_val = across_factor_gm_scores[right_region]
                                if not pd.isna(left_val) and not pd.isna(right_val):
                                    paired_left.append(left_val)
                                    paired_right.append(right_val)
                            break
                # Match WM tracts
                left_wm_rois = [r for r in across_factor_wm_scores.keys() if get_hemisphere(r) == "Left"]
                right_wm_rois = [r for r in across_factor_wm_scores.keys() if get_hemisphere(r) == "Right"]
                for left_roi in left_wm_rois:
                    parts_left = left_roi.rsplit("_", 2)
                    if len(parts_left) == 3 and parts_left[1] == "L":
                        left_base = f"{parts_left[0]}_{parts_left[2]}"
                    elif len(parts_left) >= 2 and parts_left[-1] == "L":
                        left_base = "_".join(parts_left[:-1])
                    else:
                        left_base = left_roi
                    for right_roi in right_wm_rois:
                        parts_right = right_roi.rsplit("_", 2)
                        if len(parts_right) == 3 and parts_right[1] == "R":
                            right_base = f"{parts_right[0]}_{parts_right[2]}"
                        elif len(parts_right) >= 2 and parts_right[-1] == "R":
                            right_base = "_".join(parts_right[:-1])
                        else:
                            right_base = right_roi
                        if left_base == right_base:
                            if left_roi in across_factor_wm_scores and right_roi in across_factor_wm_scores:
                                left_val = across_factor_wm_scores[left_roi]
                                right_val = across_factor_wm_scores[right_roi]
                                if not pd.isna(left_val) and not pd.isna(right_val):
                                    paired_left.append(left_val)
                                    paired_right.append(right_val)
                            break
            
            # Perform test
            if len(paired_left) >= 3 and len(paired_right) >= 3:
                try:
                    t_stat, p_value = ttest_rel(paired_left, paired_right)
                    df_degrees = len(paired_left) - 1
                    test_type = 'paired'
                except:
                    t_stat, p_value = ttest_ind(left_scores, right_scores)
                    df_degrees = len(left_scores) + len(right_scores) - 2
                    test_type = 'independent'
            else:
                t_stat, p_value = ttest_ind(left_scores, right_scores)
                df_degrees = len(left_scores) + len(right_scores) - 2
                test_type = 'independent'
            
            # Store statistics
            all_statistics.append({
                'Group': group_name,
                'Test_Type': test_type,
                'T_Statistic': t_stat,
                'Degrees_of_Freedom': df_degrees,
                'P_Value': p_value,
                'N_Left': len(left_scores),
                'N_Right': len(right_scores),
                'N_Paired': len(paired_left) if len(paired_left) >= 3 else 0
            })
            
            # Add significance annotation
            if not (np.isnan(p_value) or np.isnan(t_stat)):
                # Determine significance level
                if p_value < 0.001:
                    sig_text = '***'
                elif p_value < 0.01:
                    sig_text = '**'
                elif p_value < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'ns'
                
                # Position bracket above the data
                y_max_data = df['Value'].max() if not df.empty else 0
                y_min_data = df['Value'].min() if not df.empty else 0
                data_range = y_max_data - y_min_data if not df.empty and len(df) > 1 else y_max_data if y_max_data != 0 else 1
                y_bracket = y_max_data + data_range * 0.08
                bracket_height = data_range * 0.02
                
                # Draw bracket
                ax.plot([0, 0, 1, 1], 
                       [y_bracket - bracket_height, y_bracket, y_bracket, y_bracket - bracket_height],
                       'k-', linewidth=1.5, zorder=15)
                # Add significance text
                ax.text(0.5, y_bracket + bracket_height * 0.3, sig_text,
                       ha='center', va='bottom', fontsize=16, fontweight='bold', zorder=15)
        
        # Create swarm plot
        if sns is not None:
            try:
                print(f"Creating swarmplot for 'Across Factor Mahalanobis Distances by Hemisphere' - {group_name} subplot ({len(df)} points)")
                sns.swarmplot(data=df, x='Hemisphere', y='Value', ax=ax,
                            palette={'Left': 'white', 'Right': 'white'},
                            size=3.5, alpha=1.0, orient='v',
                            dodge=False, linewidth=0.5)
                
                # Color intervention points with Reds colormap
                for collection in ax.collections:
                    offsets = collection.get_offsets()
                    if len(offsets) == 0:
                        continue
                    
                    facecolors_raw = collection.get_facecolors()
                    facecolors = np.array(facecolors_raw, copy=True, dtype=float)
                    
                    if facecolors.ndim == 1:
                        if len(facecolors) == 4:
                            facecolors = np.tile(facecolors, (len(offsets), 1))
                        else:
                            facecolors = np.tile(np.append(facecolors, 1.0), (len(offsets), 1))
                    elif facecolors.shape[0] != len(offsets):
                        if len(offsets) > 0:
                            default_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
                            facecolors = np.tile(default_color, (len(offsets), 1))
                        else:
                            continue
                    
                    if facecolors.shape[1] == 3:
                        alpha_channel = np.ones((facecolors.shape[0], 1), dtype=float)
                        facecolors = np.hstack([facecolors, alpha_channel])
                    
                    facecolors = np.array(facecolors, dtype=float, copy=True)
                    
                    matched_indices = set()
                    intervention_count = 0
                    
                    for idx, row in df.iterrows():
                        if row['Intervention'] <= 0:
                            continue
                        
                        hemi_idx = [0, 1][['Left', 'Right'].index(row['Hemisphere'])]
                        x_match = np.abs(offsets[:, 0] - hemi_idx) < 0.2
                        y_match = np.abs(offsets[:, 1] - row['Value']) < 0.1
                        matching = x_match & y_match
                        matching_indices = np.where(matching)[0]
                        
                        for pt_idx in matching_indices:
                            if pt_idx < len(facecolors) and pt_idx not in matched_indices:
                                norm = Normalize(vmin=0, vmax=1)
                                color_rgba = Reds(norm(row['Intervention']))
                                if isinstance(color_rgba, tuple):
                                    color_rgba = np.array(list(color_rgba), dtype=float)
                                else:
                                    color_rgba = np.array(color_rgba, dtype=float)
                                if len(color_rgba) == 3:
                                    color_rgba = np.append(color_rgba, 1.0)
                                elif len(color_rgba) == 4:
                                    color_rgba[3] = 1.0
                                facecolors[pt_idx, :] = color_rgba[:4]
                                matched_indices.add(pt_idx)
                                intervention_count += 1
                    
                    if intervention_count > 0:
                        collection.set_facecolors(facecolors)
            except Exception as e:
                print(f"Warning: sns.swarmplot failed: {e}")
        
        # Overlay boxplot
        bp = ax.boxplot([df[df['Hemisphere'] == h]['Value'].values 
                        for h in ['Left', 'Right']],
                       positions=[0, 1], widths=0.3, patch_artist=True,
                       showfliers=False, whis=0, zorder=10)
        
        for patch in bp['boxes']:
            patch.set_facecolor('none')
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        if 'medians' in bp:
            for item in bp['medians']:
                item.set_color('blue')
                item.set_linewidth(1.5)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Left', 'Right'], fontsize=12, fontfamily='Georgia')
        ax.set_xlabel('Hemisphere', fontsize=14, fontweight='bold', fontfamily='Georgia')
        ax.set_ylabel('Mahalanobis Distance', fontsize=14, fontweight='bold', fontfamily='Georgia')
        ax.set_title(title, fontsize=14, fontweight='bold', fontfamily='Georgia')
        for label in ax.get_yticklabels():
            label.set_fontfamily('Georgia')
            label.set_fontsize(12)
        ax.grid(True, alpha=0.3, axis='y')
        
        return all_statistics[-1] if all_statistics else None
    
    # Create subplots
    create_subplot(axes[0], all_left_scores, all_right_scores, 
                   all_left_intervention, all_right_intervention,
                   'All Regions', 'All')
    create_subplot(axes[1], gm_left_scores, gm_right_scores,
                   gm_left_intervention, gm_right_intervention,
                   'Gray Matter Regions', 'GM')
    create_subplot(axes[2], wm_left_scores, wm_right_scores,
                   wm_left_intervention, wm_right_intervention,
                   'White Matter Tract Segments', 'WM')
    
    # Add Intervention Overlap colorbar at the bottom
    from matplotlib.cm import ScalarMappable
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=Reds, norm=norm)
    sm.set_array([])
    
    # Create colorbar axis at the bottom
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Intervention Overlap Proportion', fontsize=12, fontweight='bold', fontfamily='Georgia')
    cbar.ax.tick_params(labelsize=10)
    for tick_label in cbar.ax.get_xticklabels():
        tick_label.set_fontfamily('Georgia')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Leave space for colorbar
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Save statistics to CSV
    if all_statistics:
        stats_df = pd.DataFrame(all_statistics)
        csv_path = output_path.replace('.png', '_statistics.csv')
        stats_df.to_csv(csv_path, index=False)
        print(f"Statistics saved to: {csv_path}")
    
    print(f"Hemisphere Mahalanobis distance plot saved: {output_path}")


def create_factor1_top10_loadings_plot(
    all_regions: Sequence[str],
    all_tracts: Sequence[str],
    all_patients: Sequence[str],
    left_lateralized: Sequence[str],
    right_lateralized: Sequence[str],
    output_path: str,
) -> None:
    """
    Create plot showing Factor 1 z-scores (signed, not absolute) for loadings >= 0.8.
    Creates a single figure with 2 subplots (rows) for "All TLE" cohort only:
    1. Grey Matter only
    2. White Matter only
    
    Uses swarm plots and colors x-axis labels by reconstruction model.
    
    Args:
        all_regions: List of all GM regions
        all_tracts: List of all WM tracts
        all_patients: List of all patient subjects (used for "All TLE" cohort)
        left_lateralized: List of left-lateralized patients (not used)
        right_lateralized: List of right-lateralized patients (not used)
        output_path: Path to save the plot PNG
    """
    import json
    from matplotlib.patches import Patch
    try:
        import seaborn as sns
    except ImportError:
        print("Warning: seaborn not available, using basic plot")
        sns = None
    
    # Load factor loadings
    factor_loadings = load_factor_loadings()
    if factor_loadings.empty or 'F1' not in factor_loadings.index:
        print("Warning: Could not load factor loadings or F1 not found. Skipping Factor 1 loadings plot.")
        return
    
    # Get F1 loadings and filter for loadings >= 0.8 (absolute value)
    f1_loadings = factor_loadings.loc['F1']
    # Filter for absolute loadings >= 0.8
    high_loadings = f1_loadings[f1_loadings.abs() >= 0.8]
    # Sort by absolute value for consistent ordering
    high_loadings_sorted = high_loadings.abs().sort_values(ascending=False)
    selected_scalars = high_loadings_sorted.index.tolist()
    
    if not selected_scalars:
        print("Warning: No scalars with Factor 1 loading >= 0.8 found. Skipping Factor 1 loadings plot.")
        return
    
    print(f"Found {len(selected_scalars)} scalars with Factor 1 loading >= 0.8: {', '.join(selected_scalars)}")
    
    # Load scalar metadata for abbreviations and colors
    scalar_to_color = {}
    scalar_to_human = {}
    scalar_to_abbrev = {}
    
    try:
        colors_path = ospj(METADATA_DIR, "scalar_labels_to_colors.json")
        human_path = ospj(METADATA_DIR, "scalar_labels_to_human.json")
        
        if os.path.exists(colors_path):
            with open(colors_path) as f:
                scalar_to_color = json.load(f)
        
        if os.path.exists(human_path):
            with open(human_path) as f:
                scalar_to_human = json.load(f)
                # Extract abbreviations from human labels
                for scalar, human_label in scalar_to_human.items():
                    if "(" in human_label and ")" in human_label:
                        start = human_label.rfind("(") + 1
                        end = human_label.rfind(")")
                        abbr = human_label[start:end].strip()
                        scalar_to_abbrev[scalar] = abbr if abbr else scalar
                    else:
                        scalar_to_abbrev[scalar] = scalar
    except Exception as e:
        print(f"Warning: Could not load scalar metadata: {e}")
    
    # Helper function to get scalar prefix (model)
    def get_scalar_prefix(scalar_name):
        """Get the prefix/model name from a scalar (e.g., 'dti' from 'dti_fa')."""
        if "_" in scalar_name:
            return scalar_name.split("_")[0]
        return scalar_name
    
    # Map prefixes to model names
    prefix_to_model = {
        "dti": "DTI",
        "dki": "DKI",
        "gqi": "GQI",
        "noddi": "NODDI",
        "map": "MAP-MRI",
        "rdi": "RDI"
    }
    
    # Create color legend mapping (from reference file)
    color_legend = {
        "#C43031": "DTI",
        "#7A297F": "DKI",
        "#FAA51A": "GQI",
        "#38489E": "NODDI",
        "#289144": "MAP-MRI",
    }
    
    # Helper function to collect region-level mean z-scores for a scalar (signed, not absolute)
    def collect_scalar_z_scores(patients: Sequence[str], scalar: str, tissue_type: str = "all"):
        """
        Collect region-level mean z-scores for a scalar (signed, not absolute).
        For each region/tract, compute the mean z-score across all patients.
        
        Args:
            patients: List of patient IDs
            scalar: Scalar name
            tissue_type: "all", "gm", or "wm"
        
        Returns:
            List of mean z-scores (one per region/tract, signed)
        """
        all_z_scores = []
        
        if tissue_type in ["all", "gm"]:
            # Process GM regions
            for region in all_regions:
                gm_base = get_mni_micro_gm_profile_dir_for_region(region)
                gam_path = ospj(gm_base, region, f"{region}_{scalar}_stat-mean_gam.csv")
                if not os.path.exists(gam_path):
                    gam_path = ospj(gm_base, region, f"{region}_{scalar}_gam.csv")
                if os.path.exists(gam_path):
                    try:
                        gam_data = pd.read_csv(gam_path)
                        patient_data = gam_data[gam_data['sub'].isin(patients)]
                        if not patient_data.empty and f'{scalar}_z' in patient_data.columns:
                            # Compute mean z-score across all patients for this region
                            z_scores = patient_data[f'{scalar}_z'].dropna()
                            if not z_scores.empty:
                                mean_z = z_scores.mean()
                                all_z_scores.append(mean_z)  # Keep signed, not absolute
                    except Exception as e:
                        pass
        
        if tissue_type in ["all", "wm"]:
            # Process WM tract segments (end1, core, end2)
            for tract in all_tracts:
                if tract in TRACTS_TO_REMOVE:
                    continue
                gam_path = ospj(WM_PROFILE_DIR_PYAFQ, tract, f"{tract}_{scalar}_stat-mean_gam.csv")
                if not os.path.exists(gam_path):
                    gam_path = ospj(WM_PROFILE_DIR_PYAFQ, tract, f"{tract}_{scalar}_gam.csv")
                if os.path.exists(gam_path):
                    try:
                        gam_data = pd.read_csv(gam_path)
                        patient_data = gam_data[gam_data['sub'].isin(patients)]
                        if not patient_data.empty:
                            z_cols = [f'node{i}_z' for i in range(1, N_NODES + 1)]
                            available_cols = [col for col in z_cols if col in patient_data.columns]
                            if available_cols:
                                # Process each segment separately (end1, core, end2)
                                segments = [
                                    ("end1", END1_NODES),
                                    ("core", CORE_NODES),
                                    ("end2", END2_NODES),
                                ]
                                for segment_name, segment_nodes in segments:
                                    # Get z-score columns for this segment's nodes
                                    segment_cols = [f'node{i}_z' for i in segment_nodes]
                                    available_segment_cols = [col for col in segment_cols if col in available_cols]
                                    if available_segment_cols:
                                        # For each patient, compute mean across nodes in this segment
                                        patient_segment_means = []
                                        for patient in patients:
                                            if patient in patient_data['sub'].values:
                                                patient_row = patient_data[patient_data['sub'] == patient]
                                                if not patient_row.empty:
                                                    segment_mean_z = patient_row[available_segment_cols].mean(axis=1).values[0]
                                                    if not pd.isna(segment_mean_z):
                                                        patient_segment_means.append(segment_mean_z)
                                        # Compute mean across all patients for this tract segment
                                        if patient_segment_means:
                                            segment_mean = np.mean(patient_segment_means)
                                            all_z_scores.append(segment_mean)  # Keep signed, not absolute
                    except Exception as e:
                        pass
        
        return all_z_scores
    
    # Create a single figure with 2 subplots side-by-side for "All TLE" cohort only
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Create subplots: GM, WM (side-by-side)
    subplot_configs = [
        ("gm", "Grey Matter", axes[0]),
        ("wm", "White Matter", axes[1]),
    ]
    
    for tissue_type, subplot_title, ax in subplot_configs:
        # Collect data for all scalars and compute median z-score for sorting
        scalar_median_abs_z = {}
        scalar_median_z = {}  # Signed median for labeling
        scalar_data = {}
        
        for scalar in selected_scalars:
            z_scores = collect_scalar_z_scores(all_patients, scalar, tissue_type)  # Use signed z-scores
            if z_scores:
                # Compute median absolute z-score for sorting
                abs_z_scores = [abs(z) for z in z_scores]
                scalar_median_abs_z[scalar] = np.median(abs_z_scores)
                # Compute signed median for labeling
                scalar_median_z[scalar] = np.median(z_scores)
                scalar_data[scalar] = z_scores
        
        if not scalar_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(subplot_title, fontsize=14, fontweight='bold', fontfamily='Georgia')
            continue
        
        # Sort scalars by median |z-score| (descending)
        sorted_scalars = sorted(scalar_median_abs_z.items(), key=lambda x: x[1], reverse=True)
        sorted_scalar_list = [scalar for scalar, _ in sorted_scalars]
        
        # Prepare data for plotting
        plot_data = []
        scalar_names = []
        
        for scalar in sorted_scalar_list:
            z_scores = scalar_data[scalar]
            for z in z_scores:
                plot_data.append(z)
                scalar_names.append(scalar)
        
        # Create DataFrame for swarm plot
        df = pd.DataFrame({
            'Scalar': scalar_names,
            'Value': plot_data
        })
        
        # Create swarm plot
        if sns is not None:
            try:
                # Use swarmplot with appropriate point size
                print(f"Creating swarmplot for 'Factor 1 loadings (>= 0.8)' - {subplot_title} subplot ({len(df)} points)")
                sns.swarmplot(data=df, x='Scalar', y='Value', ax=ax,
                            color='white', size=3.5, alpha=1.0, orient='v',
                            linewidth=0.5)
            except Exception as e:
                # Fallback to stripplot if swarmplot fails
                print(f"Warning: sns.swarmplot failed, using stripplot: {e}")
                try:
                    sns.stripplot(data=df, x='Scalar', y='Value', ax=ax,
                                color='white', size=3.5, alpha=0.7, orient='v',
                                linewidth=0.5, jitter=0.3)
                except Exception as e2:
                    print(f"Warning: sns.stripplot also failed: {e2}")
        
        # Overlay boxplot
        bp = ax.boxplot([df[df['Scalar'] == s]['Value'].values 
                       for s in sorted_scalar_list],
                      positions=range(len(sorted_scalar_list)), widths=0.3, patch_artist=True,
                      showfliers=False, whis=0, zorder=10)
        
        for patch in bp['boxes']:
            patch.set_facecolor('none')
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        if 'medians' in bp:
            for item in bp['medians']:
                item.set_color('blue')
                item.set_linewidth(1.5)
        
        # Get scalar abbreviations and compute median z-score for labels
        scalar_abbrevs_with_median = []
        scalar_colors_list = []
        
        for scalar in sorted_scalar_list:
            abbrev = scalar_to_abbrev.get(scalar, scalar)
            median_z = scalar_median_z[scalar]
            scalar_abbrevs_with_median.append(f"{abbrev}\n(med={median_z:.2f})")
            scalar_colors_list.append(scalar_to_color.get(scalar, "#000000"))
        
        # Set x-axis labels with medians and color them by model
        ax.set_xticks(range(len(sorted_scalar_list)))
        ax.set_xticklabels(scalar_abbrevs_with_median, rotation=45, ha='right', fontsize=12, fontfamily='Georgia')
        
        # Color x-axis labels by model
        for i, color in enumerate(scalar_colors_list):
            ax.get_xticklabels()[i].set_color(color)
        
        # Set labels and title
        ax.set_ylabel('Statistic z-score', fontsize=14, fontweight='bold', fontfamily='Georgia')
        ax.set_title(subplot_title, fontsize=14, fontweight='bold', fontfamily='Georgia')
        for label in ax.get_yticklabels():
            label.set_fontfamily('Georgia')
            label.set_fontsize(12)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Add overall title
    n_scalars = len(selected_scalars)
    fig.suptitle(f'Abnormalities of statistics with high (>=0.8) "Overall diffusivity" loadings (n={n_scalars})',
                 fontsize=16, fontweight='bold', fontfamily='Georgia', y=0.995)
    
    # Add legend for reconstruction models at the bottom center
    legend_handles = [
        Patch(facecolor=hex_color, edgecolor="none", label=label)
        for hex_color, label in color_legend.items()
    ]
    
    leg = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        fontsize=12,
        title="Reconstruction model",
        frameon=True,
        fancybox=True,
        borderpad=0.3,
        columnspacing=0.5,
        handletextpad=0.3,
    )
    if leg.get_title() is not None:
        leg.get_title().set_fontsize(12)
        leg.get_title().set_fontfamily('Georgia')
    leg.get_frame().set_facecolor("#F6F6FA")
    leg.get_frame().set_edgecolor("#CCCCCC")
    leg.get_frame().set_linewidth(1.5)
    for text in leg.get_texts():
        text.set_fontfamily('Georgia')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Leave space for legend
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Factor 1 loadings (>= 0.8) plot saved: {output_path}")


def create_factor1_top10_loadings_plot_sorted(
    all_regions: Sequence[str],
    all_tracts: Sequence[str],
    all_patients: Sequence[str],
    left_lateralized: Sequence[str],
    right_lateralized: Sequence[str],
    output_path: str,
) -> None:
    """
    Create plot showing Factor 1 z-scores (signed, not absolute) for top 10 loadings.
    Sorts measures by median absolute value of region-level z-scores.
    Creates a single figure with 2 subplots (rows) for "All TLE" cohort only:
    1. Grey Matter only
    2. White Matter only
    
    Includes labels showing the median absolute value for each scalar.
    
    Uses swarm plots and colors x-axis labels by reconstruction model.
    
    Args:
        all_regions: List of all GM regions
        all_tracts: List of all WM tracts
        all_patients: List of all patient subjects (used for "All TLE" cohort)
        left_lateralized: List of left-lateralized patients (not used)
        right_lateralized: List of right-lateralized patients (not used)
        output_path: Path to save the plot PNG
    """
    import json
    from matplotlib.patches import Patch
    try:
        import seaborn as sns
    except ImportError:
        print("Warning: seaborn not available, using basic plot")
        sns = None
    
    # Load factor loadings
    factor_loadings = load_factor_loadings()
    if factor_loadings.empty or 'F1' not in factor_loadings.index:
        print("Warning: Could not load factor loadings or F1 not found. Skipping Factor 1 top 10 loadings plot.")
        return
    
    # Get F1 loadings and find top 10 by absolute value
    f1_loadings = factor_loadings.loc['F1'].abs().sort_values(ascending=False)
    top10_scalars = f1_loadings.head(10).index.tolist()
    
    if not top10_scalars:
        print("Warning: No top 10 loadings found. Skipping Factor 1 top 10 loadings plot.")
        return
    
    # Load scalar metadata for abbreviations and colors
    scalar_to_color = {}
    scalar_to_human = {}
    scalar_to_abbrev = {}
    
    try:
        colors_path = ospj(METADATA_DIR, "scalar_labels_to_colors.json")
        human_path = ospj(METADATA_DIR, "scalar_labels_to_human.json")
        
        if os.path.exists(colors_path):
            with open(colors_path) as f:
                scalar_to_color = json.load(f)
        
        if os.path.exists(human_path):
            with open(human_path) as f:
                scalar_to_human = json.load(f)
                # Extract abbreviations from human labels
                for scalar, human_label in scalar_to_human.items():
                    if "(" in human_label and ")" in human_label:
                        start = human_label.rfind("(") + 1
                        end = human_label.rfind(")")
                        abbr = human_label[start:end].strip()
                        scalar_to_abbrev[scalar] = abbr if abbr else scalar
                    else:
                        scalar_to_abbrev[scalar] = scalar
    except Exception as e:
        print(f"Warning: Could not load scalar metadata: {e}")
    
    # Create color legend mapping (from reference file)
    color_legend = {
        "#C43031": "DTI",
        "#7A297F": "DKI",
        "#FAA51A": "GQI",
        "#38489E": "NODDI",
        "#289144": "MAP-MRI",
    }
    
    # Helper function to collect region-level mean z-scores for a scalar (signed, not absolute)
    def collect_scalar_z_scores(patients: Sequence[str], scalar: str, tissue_type: str = "all"):
        """
        Collect region-level mean z-scores for a scalar (signed, not absolute).
        For each region/tract, compute the mean z-score across all patients.
        
        Args:
            patients: List of patient IDs
            scalar: Scalar name
            tissue_type: "all", "gm", or "wm"
        
        Returns:
            List of mean z-scores (one per region/tract, signed)
        """
        all_z_scores = []
        
        if tissue_type in ["all", "gm"]:
            # Process GM regions
            for region in all_regions:
                gm_base = get_mni_micro_gm_profile_dir_for_region(region)
                gam_path = ospj(gm_base, region, f"{region}_{scalar}_stat-mean_gam.csv")
                if not os.path.exists(gam_path):
                    gam_path = ospj(gm_base, region, f"{region}_{scalar}_gam.csv")
                if os.path.exists(gam_path):
                    try:
                        gam_data = pd.read_csv(gam_path)
                        patient_data = gam_data[gam_data['sub'].isin(patients)]
                        if not patient_data.empty and f'{scalar}_z' in patient_data.columns:
                            # Compute mean z-score across all patients for this region
                            z_scores = patient_data[f'{scalar}_z'].dropna()
                            if not z_scores.empty:
                                mean_z = z_scores.mean()
                                all_z_scores.append(mean_z)  # Keep signed, not absolute
                    except Exception as e:
                        pass
        
        if tissue_type in ["all", "wm"]:
            # Process WM tracts - compute one value per tract segment (end1, core, end2)
            for tract in all_tracts:
                if tract in TRACTS_TO_REMOVE:
                    continue
                gam_path = ospj(WM_PROFILE_DIR_PYAFQ, tract, f"{tract}_{scalar}_stat-mean_gam.csv")
                if not os.path.exists(gam_path):
                    gam_path = ospj(WM_PROFILE_DIR_PYAFQ, tract, f"{tract}_{scalar}_gam.csv")
                if os.path.exists(gam_path):
                    try:
                        gam_data = pd.read_csv(gam_path)
                        patient_data = gam_data[gam_data['sub'].isin(patients)]
                        if not patient_data.empty:
                            # Process each segment separately
                            segments = [
                                ("end1", END1_NODES),
                                ("core", CORE_NODES),
                                ("end2", END2_NODES),
                            ]
                            
                            for segment_name, segment_nodes in segments:
                                # Get z-score columns for this segment's nodes
                                segment_cols = [f'node{i}_z' for i in segment_nodes]
                                available_cols = [col for col in segment_cols if col in patient_data.columns]
                                
                                if available_cols:
                                    # For each patient, compute mean across nodes in this segment
                                    patient_means = []
                                    for patient in patients:
                                        if patient in patient_data['sub'].values:
                                            patient_row = patient_data[patient_data['sub'] == patient]
                                            if not patient_row.empty:
                                                mean_z = patient_row[available_cols].mean(axis=1).values[0]
                                                if not pd.isna(mean_z):
                                                    patient_means.append(mean_z)
                                    
                                    # Compute mean across all patients for this tract-segment
                                    if patient_means:
                                        segment_mean = np.mean(patient_means)
                                        all_z_scores.append(segment_mean)  # Keep signed, not absolute
                    except Exception as e:
                        pass
        
        return all_z_scores
    
    # Compute median absolute values for each scalar and tissue type separately
    # This allows different sorting and median display per tissue type
    scalar_median_abs_gm = {}
    scalar_median_abs_wm = {}
    scalar_median_signed_gm = {}  # For display (signed, not absolute)
    scalar_median_signed_wm = {}  # For display (signed, not absolute)
    
    for scalar in top10_scalars:
        gm_scores = collect_scalar_z_scores(all_patients, scalar, "gm")
        wm_scores = collect_scalar_z_scores(all_patients, scalar, "wm")
        
        if gm_scores:
            scalar_median_abs_gm[scalar] = np.median(np.abs(gm_scores))
            scalar_median_signed_gm[scalar] = np.median(gm_scores)  # Signed median for display
        
        if wm_scores:
            scalar_median_abs_wm[scalar] = np.median(np.abs(wm_scores))
            scalar_median_signed_wm[scalar] = np.median(wm_scores)  # Signed median for display
    
    # Sort scalars by median absolute value separately for each tissue type
    sorted_scalars_gm = sorted(top10_scalars, key=lambda s: scalar_median_abs_gm.get(s, 0), reverse=True)
    sorted_scalars_wm = sorted(top10_scalars, key=lambda s: scalar_median_abs_wm.get(s, 0), reverse=True)
    
    # Create a single figure with 2 subplots for "All TLE" cohort only
    # Increased height to make subplots taller
    fig, axes = plt.subplots(2, 1, figsize=(14, 14))
    
    # Create subplots: GM, WM
    subplot_configs = [
        ("gm", "Grey Matter", axes[0], sorted_scalars_gm, scalar_median_signed_gm, scalar_median_abs_gm),
        ("wm", "White Matter", axes[1], sorted_scalars_wm, scalar_median_signed_wm, scalar_median_abs_wm),
    ]
    
    for tissue_type, subplot_title, ax, sorted_scalars, scalar_median_signed, scalar_median_abs in subplot_configs:
        # Collect data for all scalars
        plot_data = []
        scalar_names = []
        
        for scalar in sorted_scalars:
            z_scores = collect_scalar_z_scores(all_patients, scalar, tissue_type)  # Use signed z-scores
            for z in z_scores:
                plot_data.append(z)
                scalar_names.append(scalar)
        
        if not plot_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(subplot_title, fontsize=14, fontweight='bold', fontfamily='Georgia')
            continue
        
        # Create DataFrame for swarm plot
        df = pd.DataFrame({
            'Scalar': scalar_names,
            'Value': plot_data
        })
        
        # Create swarm plot (reduced point size to ensure all points can be placed)
        if sns is not None:
            try:
                print(f"Creating swarmplot for 'Factor 1 top 10 loadings (sorted)' - {subplot_title} subplot ({len(df)} points)")
                sns.swarmplot(data=df, x='Scalar', y='Value', ax=ax,
                            color='white', size=2.5, alpha=1.0, orient='v',
                            linewidth=0.5)
            except Exception as e:
                print(f"Warning: sns.swarmplot failed, using stripplot: {e}")
                try:
                    sns.stripplot(data=df, x='Scalar', y='Value', ax=ax,
                                color='white', size=2.5, alpha=0.7, orient='v',
                                linewidth=0.5, jitter=0.3)
                except Exception as e2:
                    print(f"Warning: sns.stripplot also failed: {e2}")
        
        # Overlay boxplot
        bp = ax.boxplot([df[df['Scalar'] == s]['Value'].values 
                       for s in sorted_scalars],
                      positions=range(len(sorted_scalars)), widths=0.3, patch_artist=True,
                      showfliers=False, whis=0, zorder=10)
        
        for patch in bp['boxes']:
            patch.set_facecolor('none')
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        if 'medians' in bp:
            for item in bp['medians']:
                item.set_color('blue')
                item.set_linewidth(1.5)
        
        # Set x-axis labels and color them by model, and add median labels
        ax.set_xticks(range(len(sorted_scalars)))
        tick_labels = []
        for i, scalar in enumerate(sorted_scalars):
            abbr = scalar_to_abbrev.get(scalar, scalar)
            # Use signed median for display (not absolute)
            median_signed = scalar_median_signed.get(scalar, 0)
            # Format median to 2 decimal places, keep sign
            median_str = f"{median_signed:.2f}"
            tick_labels.append(f"{abbr}\nMdn={median_str}")
        
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=11, fontfamily='Georgia')
        
        # Color x-axis labels by model
        for i, scalar in enumerate(sorted_scalars):
            color = scalar_to_color.get(scalar, "#000000")
            ax.get_xticklabels()[i].set_color(color)
        
        # Set labels and title
        ax.set_ylabel('Factor 1 z-scores', fontsize=14, fontweight='bold', fontfamily='Georgia')
        ax.set_xlabel('Diffusion statistic', fontsize=14, fontweight='bold', fontfamily='Georgia')
        ax.set_title(subplot_title, fontsize=14, fontweight='bold', fontfamily='Georgia')
        for label in ax.get_yticklabels():
            label.set_fontfamily('Georgia')
            label.set_fontsize(12)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Add overall title (two lines)
    fig.suptitle('"Overall diffusivity" Factor 1 z-scores for top loadings\nSorted by median |z-scores|',
                 fontsize=16, fontweight='bold', fontfamily='Georgia', y=0.995)
    
    # Add legend for reconstruction models at the bottom center
    legend_handles = [
        Patch(facecolor=hex_color, edgecolor="none", label=label)
        for hex_color, label in color_legend.items()
    ]
    
    leg = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        fontsize=12,
        title="Reconstruction model",
        frameon=True,
        fancybox=True,
        borderpad=0.3,
        columnspacing=0.5,
        handletextpad=0.3,
    )
    if leg.get_title() is not None:
        leg.get_title().set_fontsize(12)
        leg.get_title().set_fontfamily('Georgia')
    leg.get_frame().set_facecolor("#F6F6FA")
    leg.get_frame().set_edgecolor("#CCCCCC")
    leg.get_frame().set_linewidth(1.5)
    for text in leg.get_texts():
        text.set_fontfamily('Georgia')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Leave space for legend
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Factor 1 top 10 loadings plot (sorted) saved: {output_path}")


def create_factor_z_raincloud_plot(
    all_factor_z_scores: Dict[str, Dict[str, float]],
    all_gm_factor_z_scores: Dict[str, Dict[str, float]],
    factor_names: List[str],
    output_path: str,
    left_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
    left_gm_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
    right_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
    right_gm_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """
    Create raincloud plots showing signed z-scores per factor, with GM on left and WM on right (flipped).
    Creates 3 subplots: All temporal, Left temporal, Right temporal.
    
    Args:
        all_factor_z_scores: Dict mapping factor_name to dict of {roi: z_score} for WM tract segments (all temporal)
        all_gm_factor_z_scores: Dict mapping factor_name to dict of {region: z_score} for GM regions (all temporal)
        factor_names: List of factor names (e.g., ["F1", "F2", "F3"])
        output_path: Path to save the plot PNG
        left_factor_z_scores: Optional dict for left temporal WM (if None, uses all)
        left_gm_factor_z_scores: Optional dict for left temporal GM (if None, uses all)
        right_factor_z_scores: Optional dict for right temporal WM (if None, uses all)
        right_gm_factor_z_scores: Optional dict for right temporal GM (if None, uses all)
    """
    from scipy import stats
    from scipy.stats import ttest_rel, ttest_ind
    import pandas as pd
    import numpy as np
    
    # Helper function to create raincloud for a single factor
    def create_factor_raincloud(ax, gm_data, wm_data, factor_pos, factor_name, width=0.3):
        """Create raincloud plot for one factor with GM on left, WM on right (flipped)"""
        # GM raincloud on left
        if gm_data:
            gm_values = np.array([v for v in gm_data if not pd.isna(v)])
            if len(gm_values) > 0:
                # Density plot on left
                try:
                    if len(gm_values) > 1:
                        density = stats.gaussian_kde(gm_values)
                        y_range = np.linspace(gm_values.min(), gm_values.max(), 100)
                        density_values = density(y_range)
                        if density_values.max() > 0:
                            density_values = density_values / density_values.max() * width * 0.7  # Larger density plot
                        ax.fill_betweenx(
                            y_range,
                            factor_pos - width - density_values,
                            factor_pos - width,
                            color='gray',
                            alpha=0.6,
                        )
                except Exception:
                    pass
                
                # Mean marker
                mean_val = np.mean(gm_values)
                ax.plot(
                    factor_pos - width,
                    mean_val,
                    marker='o',
                    markersize=6,
                    color='black',
                    markeredgecolor='white',
                    markeredgewidth=1,
                    zorder=15,
                )
                
                # Jittered points to the right (more jitter, more space from density)
                np.random.seed(42)
                jitter = np.random.normal(0, width * 0.08, len(gm_values))  # More jitter
                offset = width * 0.3  # More space between density and points
                ax.scatter(
                    factor_pos - width + offset + jitter,
                    gm_values,
                    color='gray',
                    alpha=0.7,
                    s=8,
                    zorder=10,
                )
        
        # WM raincloud on right (flipped - density on right, points on left of density)
        if wm_data:
            wm_values = np.array([v for v in wm_data if not pd.isna(v)])
            if len(wm_values) > 0:
                # Density plot on right (flipped)
                try:
                    if len(wm_values) > 1:
                        density = stats.gaussian_kde(wm_values)
                        y_range = np.linspace(wm_values.min(), wm_values.max(), 100)
                        density_values = density(y_range)
                        if density_values.max() > 0:
                            density_values = density_values / density_values.max() * width * 0.7  # Larger density plot
                        # Flip: fill from factor_pos to factor_pos + density_values
                        ax.fill_betweenx(
                            y_range,
                            factor_pos + width,
                            factor_pos + width + density_values,
                            color='white',
                            edgecolor='black',
                            linewidth=1,
                            alpha=0.6,
                        )
                except Exception:
                    pass
                
                # Mean marker
                mean_val = np.mean(wm_values)
                ax.plot(
                    factor_pos + width,
                    mean_val,
                    marker='o',
                    markersize=6,
                    color='black',
                    markeredgecolor='white',
                    markeredgewidth=1,
                    zorder=15,
                )
                
                # Jittered points to the left of density (more jitter, more space from density)
                np.random.seed(42)
                jitter = np.random.normal(0, width * 0.08, len(wm_values))  # More jitter
                offset = width * 0.3  # More space between density and points
                ax.scatter(
                    factor_pos + width - offset + jitter,
                    wm_values,
                    color='white',
                    edgecolors='black',
                    linewidths=0.5,
                    alpha=0.8,
                    s=8,
                    zorder=10,
                )
    
    # Store statistics for CSV export
    all_statistics = []
    
    # Helper function to create a single subplot
    def create_subplot(ax, wm_scores_dict, gm_scores_dict, title, group_name):
        # Collect all individual z-scores per factor for GM and WM
        gm_data = {f: [] for f in factor_names}
        wm_data = {f: [] for f in factor_names}
        gm_regions = {f: [] for f in factor_names}
        wm_rois = {f: [] for f in factor_names}
        
        for factor_name in factor_names:
            # Collect individual z-scores for GM regions
            gm_scores = gm_scores_dict.get(factor_name, {})
            for region, z in gm_scores.items():
                if not pd.isna(z):
                    gm_data[factor_name].append(z)
                    gm_regions[factor_name].append(region)
            
            # Collect individual z-scores for WM tract segments
            wm_scores = wm_scores_dict.get(factor_name, {})
            for roi_key, z in wm_scores.items():
                if not pd.isna(z):
                    wm_data[factor_name].append(z)
                    wm_rois[factor_name].append(roi_key)
        
        # Create raincloud for each factor
        positions = np.arange(1, len(factor_names) + 1)
        width = 0.15  # Reduced width to bring GM and WM closer together
        
        for i, factor_name in enumerate(factor_names):
            create_factor_raincloud(
                ax, 
                gm_data[factor_name], 
                wm_data[factor_name], 
                positions[i], 
                factor_name, 
                width
            )
            
            # Perform statistical test between GM and WM for this factor
            gm_vals = np.array(gm_data[factor_name])
            wm_vals = np.array(wm_data[factor_name])
            
            if len(gm_vals) >= 2 and len(wm_vals) >= 2:
                # Try to match regions for paired t-test
                paired_gm = []
                paired_wm = []
                
                for j, gm_region in enumerate(gm_regions[factor_name]):
                    gm_region_clean = gm_region.replace('LH-', '').replace('RH-', '').replace('LH_', '').replace('RH_', '')
                    for k, wm_roi in enumerate(wm_rois[factor_name]):
                        if gm_region_clean.lower() in wm_roi.lower() or wm_roi.lower() in gm_region_clean.lower():
                            paired_gm.append(gm_vals[j])
                            paired_wm.append(wm_vals[k])
                            break
                
                # Perform test
                if len(paired_gm) >= 3 and len(paired_wm) >= 3:
                    try:
                        t_stat, p_value = ttest_rel(paired_gm, paired_wm)
                        df_degrees = len(paired_gm) - 1
                        test_type = 'paired'
                    except:
                        t_stat, p_value = ttest_ind(gm_vals, wm_vals)
                        df_degrees = len(gm_vals) + len(wm_vals) - 2
                        test_type = 'independent'
                else:
                    t_stat, p_value = ttest_ind(gm_vals, wm_vals)
                    df_degrees = len(gm_vals) + len(wm_vals) - 2
                    test_type = 'independent'
                
                # Store statistics
                all_statistics.append({
                    'Group': group_name,
                    'Factor': factor_name,
                    'Test_Type': test_type,
                    'T_Statistic': t_stat,
                    'Degrees_of_Freedom': df_degrees,
                    'P_Value': p_value,
                    'N_GM': len(gm_vals),
                    'N_WM': len(wm_vals),
                    'N_Paired': len(paired_gm) if len(paired_gm) >= 3 else 0
                })
                
                # Add significance annotation
                if not (np.isnan(p_value) or np.isnan(t_stat)):
                    # Determine significance level
                    if p_value < 0.001:
                        sig_text = '***'
                    elif p_value < 0.01:
                        sig_text = '**'
                    elif p_value < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = 'ns'
                    
                    # Position bracket above the data for this factor
                    factor_pos = positions[i]
                    y_max_data = max(gm_vals.max() if len(gm_vals) > 0 else 0, 
                                    wm_vals.max() if len(wm_vals) > 0 else 0)
                    y_min_data = min(gm_vals.min() if len(gm_vals) > 0 else 0, 
                                    wm_vals.min() if len(wm_vals) > 0 else 0)
                    data_range = y_max_data - y_min_data if y_max_data != y_min_data else abs(y_max_data) if y_max_data != 0 else 1
                    y_bracket = y_max_data + data_range * 0.08
                    bracket_height = data_range * 0.02
                    
                    # Draw bracket from GM position to WM position
                    gm_x = factor_pos - width
                    wm_x = factor_pos + width
                    ax.plot([gm_x, gm_x, wm_x, wm_x], 
                           [y_bracket - bracket_height, y_bracket, y_bracket, y_bracket - bracket_height],
                           'k-', linewidth=1.5, zorder=15)
                    # Add significance text
                    ax.text(factor_pos, y_bracket + bracket_height * 0.3, sig_text,
                           ha='center', va='bottom', fontsize=16, fontweight='bold', zorder=15)
        
        ax.set_xlabel('Factor', fontsize=18, fontweight='bold')
        ax.set_ylabel('Z-Score', fontsize=18, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels([get_factor_label(f) for f in factor_names], fontsize=12, rotation=0, ha='center')
        for label in ax.get_yticklabels():
            label.set_fontsize(16)
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # First pass: collect all data to compute global y-axis limits
    all_wm_data = {f: [] for f in factor_names}
    all_gm_data = {f: [] for f in factor_names}
    
    datasets = [
        (all_factor_z_scores, all_gm_factor_z_scores),
        (left_factor_z_scores if left_factor_z_scores is not None else all_factor_z_scores,
         left_gm_factor_z_scores if left_gm_factor_z_scores is not None else all_gm_factor_z_scores),
        (right_factor_z_scores if right_factor_z_scores is not None else all_factor_z_scores,
         right_gm_factor_z_scores if right_gm_factor_z_scores is not None else all_gm_factor_z_scores),
    ]
    
    for wm_scores_dict, gm_scores_dict in datasets:
        for factor_name in factor_names:
            gm_scores = gm_scores_dict.get(factor_name, {})
            for region, z in gm_scores.items():
                if not pd.isna(z):
                    all_gm_data[factor_name].append(z)
            wm_scores = wm_scores_dict.get(factor_name, {})
            for roi_key, z in wm_scores.items():
                if not pd.isna(z):
                    all_wm_data[factor_name].append(z)
    
    # Compute global y-axis limits
    all_values = []
    for f in factor_names:
        all_values.extend(all_gm_data[f])
        all_values.extend(all_wm_data[f])
    
    if all_values:
        global_ymin = min(all_values)
        global_ymax = max(all_values)
        y_range = global_ymax - global_ymin
        global_ymin -= y_range * 0.1
        global_ymax += y_range * 0.1
    else:
        global_ymin = -1
        global_ymax = 1
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # All temporal
    create_subplot(axes[0], all_factor_z_scores, all_gm_factor_z_scores, 'All Temporal', 'All Temporal')
    axes[0].set_ylim(global_ymin, global_ymax)
    
    # Left temporal
    left_wm = left_factor_z_scores if left_factor_z_scores is not None else all_factor_z_scores
    left_gm = left_gm_factor_z_scores if left_gm_factor_z_scores is not None else all_gm_factor_z_scores
    create_subplot(axes[1], left_wm, left_gm, 'Left Temporal', 'Left Temporal')
    axes[1].set_ylim(global_ymin, global_ymax)
    
    # Right temporal
    right_wm = right_factor_z_scores if right_factor_z_scores is not None else all_factor_z_scores
    right_gm = right_gm_factor_z_scores if right_gm_factor_z_scores is not None else all_gm_factor_z_scores
    create_subplot(axes[2], right_wm, right_gm, 'Right Temporal', 'Right Temporal')
    axes[2].set_ylim(global_ymin, global_ymax)
    
    # Add common legend for all subplots (place it lower to avoid blocking x-axis label)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Gray Matter'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='black', 
               markersize=10, label='White Matter'),
    ]
    # Position legend at bottom center, lower to avoid blocking x-axis label
    fig.legend(handles=legend_elements, fontsize=16, loc='lower center', 
               bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=True)
    
    # Add overall title
    fig.suptitle('Factor z-scores by tissue type', fontsize=20, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Save statistics to CSV
    if all_statistics:
        stats_df = pd.DataFrame(all_statistics)
        csv_path = output_path.replace('.png', '_statistics.csv')
        stats_df.to_csv(csv_path, index=False)
        print(f"Statistics saved to: {csv_path}")
    
    print(f"Factor z-score raincloud plot saved: {output_path}")


def create_across_factor_abs_z_raincloud_plot(
    all_factor_z_scores: Dict[str, Dict[str, float]],
    all_gm_factor_z_scores: Dict[str, Dict[str, float]],
    factor_names: List[str],
    output_path: str,
    left_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
    left_gm_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
    right_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
    right_gm_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """
    Create raincloud plots showing absolute z-scores per factor, combining GM and WM data into one raincloud per factor.
    Creates 3 subplots: All temporal, Left temporal, Right temporal.
    
    Args:
        all_factor_z_scores: Dict mapping factor_name to dict of {roi: z_score} for WM tract segments (all temporal)
        all_gm_factor_z_scores: Dict mapping factor_name to dict of {region: z_score} for GM regions (all temporal)
        factor_names: List of factor names (e.g., ["F1", "F2", "F3"])
        output_path: Path to save the plot PNG
        left_factor_z_scores: Optional dict for left temporal WM (if None, uses all)
        left_gm_factor_z_scores: Optional dict for left temporal GM (if None, uses all)
        right_factor_z_scores: Optional dict for right temporal WM (if None, uses all)
        right_gm_factor_z_scores: Optional dict for right temporal GM (if None, uses all)
    """
    from scipy.stats import ttest_rel, ttest_ind
    import seaborn as sns
    from itertools import combinations
    
    # Store statistics for CSV export
    all_statistics = []
    
    # Helper function to create a single subplot
    def create_subplot(ax, wm_scores_dict, gm_scores_dict, title, group_name):
        # Collect all individual absolute z-scores per factor, keeping track of region/ROI for pairing
        factor_data = {f: [] for f in factor_names}
        factor_regions = {f: [] for f in factor_names}  # Track region/ROI for each value
        
        for factor_name in factor_names:
            # Collect individual absolute z-scores for GM regions
            gm_scores = gm_scores_dict.get(factor_name, {})
            for region, z in gm_scores.items():
                if not pd.isna(z):
                    factor_data[factor_name].append(abs(z))
                    factor_regions[factor_name].append(('GM', region))
            
            # Collect individual absolute z-scores for WM tract segments
            wm_scores = wm_scores_dict.get(factor_name, {})
            for roi_key, z in wm_scores.items():
                if not pd.isna(z):
                    factor_data[factor_name].append(abs(z))
                    factor_regions[factor_name].append(('WM', roi_key))
        
        # Prepare data for plotting
        plot_data = []
        factor_categories = []
        
        for factor_name in factor_names:
            for val in factor_data[factor_name]:
                plot_data.append(val)
                factor_categories.append(factor_name)
        
        if not plot_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(title, fontsize=18, fontweight='bold')
            return None
        
        df = pd.DataFrame({'Factor': factor_categories, 'Value': plot_data})
        
        # Perform pairwise comparisons between all factors (paired by region/ROI)
        pairwise_stats = []
        
        for factor1, factor2 in combinations(factor_names, 2):
            # Try to pair values by matching region/ROI
            paired_vals1 = []
            paired_vals2 = []
            
            # Create dictionaries mapping region/ROI to value for each factor
            factor1_dict = {}
            for i, (tissue, region) in enumerate(factor_regions[factor1]):
                key = (tissue, region)
                if key not in factor1_dict:
                    factor1_dict[key] = []
                factor1_dict[key].append(factor_data[factor1][i])
            
            factor2_dict = {}
            for i, (tissue, region) in enumerate(factor_regions[factor2]):
                key = (tissue, region)
                if key not in factor2_dict:
                    factor2_dict[key] = []
                factor2_dict[key].append(factor_data[factor2][i])
            
            # Match pairs
            for key in factor1_dict:
                if key in factor2_dict:
                    # Use mean if multiple values for same region
                    val1 = np.mean(factor1_dict[key])
                    val2 = np.mean(factor2_dict[key])
                    paired_vals1.append(val1)
                    paired_vals2.append(val2)
            
            # Perform paired t-test if we have enough pairs
            if len(paired_vals1) >= 3:
                try:
                    t_stat, p_value = ttest_rel(paired_vals1, paired_vals2)
                    df_degrees = len(paired_vals1) - 1
                    test_type = 'paired'
                except:
                    # Fallback to independent t-test
                    t_stat, p_value = ttest_ind(factor_data[factor1], factor_data[factor2])
                    df_degrees = len(factor_data[factor1]) + len(factor_data[factor2]) - 2
                    test_type = 'independent'
            else:
                # Use independent t-test
                if len(factor_data[factor1]) >= 2 and len(factor_data[factor2]) >= 2:
                    t_stat, p_value = ttest_ind(factor_data[factor1], factor_data[factor2])
                    df_degrees = len(factor_data[factor1]) + len(factor_data[factor2]) - 2
                    test_type = 'independent'
                else:
                    t_stat, p_value, df_degrees = np.nan, np.nan, np.nan
                    test_type = 'insufficient_data'
            
            pairwise_stats.append({
                'Factor1': factor1,
                'Factor2': factor2,
                'Test_Type': test_type,
                'T_Statistic': t_stat,
                'Degrees_of_Freedom': df_degrees,
                'P_Value': p_value,
                'N_Paired': len(paired_vals1) if len(paired_vals1) >= 3 else 0,
                'N_Factor1': len(factor_data[factor1]),
                'N_Factor2': len(factor_data[factor2])
            })
        
        # Store statistics for this group
        for stat in pairwise_stats:
            stat['Group'] = group_name
            all_statistics.append(stat)
        
        # Create swarm plot with seaborn
        if sns is not None:
            try:
                # Plot all points with swarmplot
                print(f"Creating swarmplot for 'Factor |z-scores|' - {group_name} subplot ({len(df)} points)")
                sns.swarmplot(data=df, x='Factor', y='Value', ax=ax,
                            color='#808080', size=3.5, alpha=1.0, orient='v',
                            linewidth=0.5)
            except Exception as e:
                print(f"Warning: sns.swarmplot failed: {e}")
        
        # Overlay boxplot without whiskers or fliers (on top of points)
        bp = ax.boxplot([df[df['Factor'] == f]['Value'].values 
                         for f in factor_names],
                        positions=range(len(factor_names)), widths=0.3, patch_artist=True,
                        showfliers=False, whis=0, zorder=10)
        
        for patch in bp['boxes']:
            patch.set_facecolor('none')  # Transparent fill
            patch.set_edgecolor('black')  # Black outline
            patch.set_linewidth(1.5)
        
        # Set color and linewidth for boxplot medians (visible, blue)
        if 'medians' in bp:
            for item in bp['medians']:
                item.set_color('blue')
                item.set_linewidth(1.5)
        
        # Add significance annotations for all pairwise comparisons
        # Organize brackets to avoid overlap - use different y-positions
        # Position brackets much lower with more spacing between them
        y_max_data = df['Value'].max() if not df.empty else 0
        y_min_data = df['Value'].min() if not df.empty else 0
        data_range = y_max_data - y_min_data if not df.empty and len(df) > 1 else y_max_data
        bracket_spacing = (global_ymax - global_ymin) * 0.12  # Increased spacing to 0.12
        # Start brackets much lower, just above the maximum data value
        bracket_base = y_max_data + data_range * 0.1  # Small offset above max data point
        
        # Get factor positions
        factor_positions = {f: i for i, f in enumerate(factor_names)}
        
        # Draw brackets for each comparison, organized to avoid overlap
        bracket_levels = {}  # Track which y-level each bracket is on
        for i, stat in enumerate(pairwise_stats):
            if np.isnan(stat['P_Value']) or np.isnan(stat['T_Statistic']):
                continue
            
            factor1_idx = factor_positions[stat['Factor1']]
            factor2_idx = factor_positions[stat['Factor2']]
            
            # Determine significance level
            p_value = stat['P_Value']
            if p_value < 0.001:
                sig_text = '***'
            elif p_value < 0.01:
                sig_text = '**'
            elif p_value < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            
            # Find an available bracket level (avoid overlap with existing brackets)
            bracket_level = 0
            for level in range(10):  # Max 10 levels
                overlap = False
                for existing in bracket_levels.values():
                    if existing['level'] == level:
                        # Check if x-positions overlap
                        existing_x1, existing_x2 = existing['x_range']
                        if not (factor2_idx < existing_x1 or factor1_idx > existing_x2):
                            overlap = True
                            break
                if not overlap:
                    bracket_level = level
                    break
            
            y_bracket = bracket_base + bracket_level * bracket_spacing
            
            # Draw bracket
            ax.plot([factor1_idx, factor1_idx, factor2_idx, factor2_idx], 
                   [y_bracket - bracket_spacing * 0.5, y_bracket, y_bracket, 
                    y_bracket - bracket_spacing * 0.5],
                   'k-', linewidth=1.5, zorder=15)
            
            # Add significance text
            ax.text((factor1_idx + factor2_idx) / 2, y_bracket + bracket_spacing * 0.2, sig_text,
                   ha='center', va='bottom', fontsize=14, fontweight='bold', zorder=15)
            
            # Store bracket info
            bracket_levels[i] = {
                'level': bracket_level,
                'x_range': (factor1_idx, factor2_idx)
            }
        
        # Adjust y-axis limits to accommodate brackets
        if bracket_levels:
            max_level = max(b['level'] for b in bracket_levels.values())
            y_top = bracket_base + (max_level + 1) * bracket_spacing + bracket_spacing * 0.3
            ax.set_ylim(global_ymin, max(global_ymax, y_top))
        else:
            ax.set_ylim(global_ymin, global_ymax)
        
        ax.set_xticks(range(len(factor_names)))
        ax.set_xticklabels([get_factor_label(f) for f in factor_names], fontsize=12)  # Keep x-tick labels smaller
        ax.set_xlabel('Factor', fontsize=18, fontweight='bold')
        ax.set_ylabel('Factor |z-scores|', fontsize=18, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold')
        for label in ax.get_yticklabels():
            label.set_fontsize(16)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        return pairwise_stats
    
    # First pass: collect all data to compute global y-axis limits
    all_wm_data = {f: [] for f in factor_names}
    all_gm_data = {f: [] for f in factor_names}
    
    datasets = [
        (all_factor_z_scores, all_gm_factor_z_scores),
        (left_factor_z_scores if left_factor_z_scores is not None else all_factor_z_scores,
         left_gm_factor_z_scores if left_gm_factor_z_scores is not None else all_gm_factor_z_scores),
        (right_factor_z_scores if right_factor_z_scores is not None else all_factor_z_scores,
         right_gm_factor_z_scores if right_gm_factor_z_scores is not None else all_gm_factor_z_scores),
    ]
    
    for wm_scores_dict, gm_scores_dict in datasets:
        for factor_name in factor_names:
            gm_scores = gm_scores_dict.get(factor_name, {})
            for region, z in gm_scores.items():
                if not pd.isna(z):
                    all_gm_data[factor_name].append(abs(z))
            wm_scores = wm_scores_dict.get(factor_name, {})
            for roi_key, z in wm_scores.items():
                if not pd.isna(z):
                    all_wm_data[factor_name].append(abs(z))
    
    # Compute global y-axis limits
    all_values = []
    for f in factor_names:
        all_values.extend(all_gm_data[f])
        all_values.extend(all_wm_data[f])
    
    if all_values:
        global_ymin = 0
        global_ymax = max(all_values)
        global_ymax += global_ymax * 0.1
    else:
        global_ymin = 0
        global_ymax = 1
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # All temporal
    stats_all = create_subplot(axes[0], all_factor_z_scores, all_gm_factor_z_scores, 'All Temporal', 'All Temporal')
    
    # Left temporal
    left_wm = left_factor_z_scores if left_factor_z_scores is not None else all_factor_z_scores
    left_gm = left_gm_factor_z_scores if left_gm_factor_z_scores is not None else all_gm_factor_z_scores
    stats_left = create_subplot(axes[1], left_wm, left_gm, 'Left Temporal', 'Left Temporal')
    
    # Right temporal
    right_wm = right_factor_z_scores if right_factor_z_scores is not None else all_factor_z_scores
    right_gm = right_gm_factor_z_scores if right_gm_factor_z_scores is not None else all_gm_factor_z_scores
    stats_right = create_subplot(axes[2], right_wm, right_gm, 'Right Temporal', 'Right Temporal')
    
    # Add overall title
    fig.suptitle('Factor |z-scores|', fontsize=20, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(all_statistics)
    csv_path = output_path.replace('.png', '_statistics.csv')
    stats_df.to_csv(csv_path, index=False)
    
    print(f"Across factor Mahalanobis distance swarm plot saved: {output_path}")
    print(f"Statistics saved to: {csv_path}")


def create_factor_total_z_barplot_with_jitter(
    all_factor_z_scores: Dict[str, Dict[str, float]],
    all_gm_factor_z_scores: Dict[str, Dict[str, float]],
    factor_names: List[str],
    output_path: str,
    left_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
    left_gm_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
    right_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
    right_gm_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """
    Create a grouped barplot with error bars and jitter points showing signed z-scores per factor, with separate bars for GM and WM.
    Creates 3 subplots: All temporal, Left temporal, Right temporal.
    
    Args:
        all_factor_z_scores: Dict mapping factor_name to dict of {roi: z_score} for WM tract segments (all temporal)
        all_gm_factor_z_scores: Dict mapping factor_name to dict of {region: z_score} for GM regions (all temporal)
        factor_names: List of factor names (e.g., ["F1", "F2", "F3"])
        output_path: Path to save the plot PNG
        left_factor_z_scores: Optional dict for left temporal WM (if None, uses all)
        left_gm_factor_z_scores: Optional dict for left temporal GM (if None, uses all)
        right_factor_z_scores: Optional dict for right temporal WM (if None, uses all)
        right_gm_factor_z_scores: Optional dict for right temporal GM (if None, uses all)
    """
    # First pass: collect all data to compute global y-axis limits
    all_wm_data = {f: [] for f in factor_names}
    all_gm_data = {f: [] for f in factor_names}
    
    # Collect data from all three groups
    datasets = [
        (all_factor_z_scores, all_gm_factor_z_scores),
        (left_factor_z_scores if left_factor_z_scores is not None else all_factor_z_scores,
         left_gm_factor_z_scores if left_gm_factor_z_scores is not None else all_gm_factor_z_scores),
        (right_factor_z_scores if right_factor_z_scores is not None else all_factor_z_scores,
         right_gm_factor_z_scores if right_gm_factor_z_scores is not None else all_gm_factor_z_scores),
    ]
    
    for wm_scores_dict, gm_scores_dict in datasets:
        for factor_name in factor_names:
            gm_scores = gm_scores_dict.get(factor_name, {})
            for region, z in gm_scores.items():
                if not pd.isna(z):
                    all_gm_data[factor_name].append(z)
            
            wm_scores = wm_scores_dict.get(factor_name, {})
            for roi_key, z in wm_scores.items():
                if not pd.isna(z):
                    all_wm_data[factor_name].append(z)
    
    # Compute global y-axis limits (including error bars)
    all_values = []
    for f in factor_names:
        if all_gm_data[f]:
            gm_mean = np.mean(all_gm_data[f])
            gm_sem = np.std(all_gm_data[f], ddof=1) / np.sqrt(len(all_gm_data[f]))
            all_values.extend([gm_mean + gm_sem, gm_mean - gm_sem])
            all_values.extend(all_gm_data[f])  # Include all data points
        if all_wm_data[f]:
            wm_mean = np.mean(all_wm_data[f])
            wm_sem = np.std(all_wm_data[f], ddof=1) / np.sqrt(len(all_wm_data[f]))
            all_values.extend([wm_mean + wm_sem, wm_mean - wm_sem])
            all_values.extend(all_wm_data[f])  # Include all data points
    
    if all_values:
        global_ymin = min(all_values)
        global_ymax = max(all_values)
        # Add padding
        y_range = global_ymax - global_ymin
        global_ymin -= y_range * 0.1
        global_ymax += y_range * 0.1
    else:
        global_ymin = -1
        global_ymax = 1
    
    # Helper function to create a single subplot
    def create_subplot(ax, wm_scores_dict, gm_scores_dict, title):
        # Collect all individual z-scores per factor for GM and WM
        gm_data = {f: [] for f in factor_names}
        wm_data = {f: [] for f in factor_names}
        
        for factor_name in factor_names:
            # Collect individual z-scores for GM regions
            gm_scores = gm_scores_dict.get(factor_name, {})
            for region, z in gm_scores.items():
                if not pd.isna(z):
                    gm_data[factor_name].append(z)
            
            # Collect individual z-scores for WM tract segments
            wm_scores = wm_scores_dict.get(factor_name, {})
            for roi_key, z in wm_scores.items():
                if not pd.isna(z):
                    wm_data[factor_name].append(z)
        
        # Compute means and standard errors
        gm_means = [np.mean(gm_data[f]) if gm_data[f] else 0 for f in factor_names]
        wm_means = [np.mean(wm_data[f]) if wm_data[f] else 0 for f in factor_names]
        
        gm_sems = []
        wm_sems = []
        for f in factor_names:
            if gm_data[f]:
                gm_sem = np.std(gm_data[f], ddof=1) / np.sqrt(len(gm_data[f]))
            else:
                gm_sem = 0
            gm_sems.append(gm_sem)
            
            if wm_data[f]:
                wm_sem = np.std(wm_data[f], ddof=1) / np.sqrt(len(wm_data[f]))
            else:
                wm_sem = 0
            wm_sems.append(wm_sem)
        
        x = np.arange(len(factor_names))
        width = 0.35
        
        # Set random seed for reproducibility (different for each subplot)
        np.random.seed(42)
        
        # Plot bars with error bars - increased error bar visibility
        # GM: grey bar fill
        bars1 = ax.bar(x - width/2, gm_means, width, yerr=gm_sems, 
                       label='Gray Matter', color='gray', alpha=0.7, 
                       capsize=8, error_kw={'elinewidth': 3, 'capthick': 3})
        # WM: white bar fill with thin black border
        bars2 = ax.bar(x + width/2, wm_means, width, yerr=wm_sems,
                       label='White Matter', color='white', edgecolor='black', linewidth=1,
                       capsize=8, error_kw={'elinewidth': 3, 'capthick': 3})
        
        # Add jitter points with reduced jitter
        jitter_scale = width/8
        for i, factor_name in enumerate(factor_names):
            # GM jitter points - decreased point size
            if gm_data[factor_name]:
                jitter_gm = np.random.normal(0, jitter_scale, len(gm_data[factor_name]))
                gm_scores = np.array(gm_data[factor_name])
                ax.scatter(x[i] - width/2 + jitter_gm, gm_scores, 
                          alpha=0.6, s=8, color='gray', zorder=3)
            
            # WM jitter points - decreased point size
            if wm_data[factor_name]:
                jitter_wm = np.random.normal(0, jitter_scale, len(wm_data[factor_name]))
                wm_scores = np.array(wm_data[factor_name])
                ax.scatter(x[i] + width/2 + jitter_wm, wm_scores,
                          alpha=0.8, s=8, color='white', edgecolors='black', linewidths=0.5, zorder=3)
        
        ax.set_xlabel('Factor', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Z-Score', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(factor_names)
        
        # Set consistent y-axis limits
        ax.set_ylim(global_ymin, global_ymax)
        
        # Create custom legend (only for first subplot)
        if title == 'All Temporal':
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Gray Matter'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='black', 
                       markersize=8, label='White Matter'),
            ]
            ax.legend(handles=legend_elements, fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # All temporal
    create_subplot(axes[0], all_factor_z_scores, all_gm_factor_z_scores, 'All Temporal')
    
    # Left temporal (use provided or fallback to all)
    left_wm = left_factor_z_scores if left_factor_z_scores is not None else all_factor_z_scores
    left_gm = left_gm_factor_z_scores if left_gm_factor_z_scores is not None else all_gm_factor_z_scores
    create_subplot(axes[1], left_wm, left_gm, 'Left Temporal')
    
    # Right temporal (use provided or fallback to all)
    right_wm = right_factor_z_scores if right_factor_z_scores is not None else all_factor_z_scores
    right_gm = right_gm_factor_z_scores if right_gm_factor_z_scores is not None else all_gm_factor_z_scores
    create_subplot(axes[2], right_wm, right_gm, 'Right Temporal')
    
    # Add overall title
    fig.suptitle('Factor z-scores by tissue type', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Factor total z-score barplot with jitter saved: {output_path}")


def create_across_factor_abs_z_barplot_with_jitter(
    all_factor_z_scores: Dict[str, Dict[str, float]],
    all_gm_factor_z_scores: Dict[str, Dict[str, float]],
    factor_names: List[str],
    output_path: str,
    left_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
    left_gm_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
    right_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
    right_gm_factor_z_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """
    Create a grouped barplot with error bars and jitter points showing absolute z-scores per factor, with separate bars for GM and WM.
    Creates 3 subplots: All temporal, Left temporal, Right temporal.
    Shows overall magnitude of abnormalities (using absolute values).
    
    Args:
        all_factor_z_scores: Dict mapping factor_name to dict of {roi: z_score} for WM tract segments (all temporal)
        all_gm_factor_z_scores: Dict mapping factor_name to dict of {region: z_score} for GM regions (all temporal)
        factor_names: List of factor names (e.g., ["F1", "F2", "F3"])
        output_path: Path to save the plot PNG
        left_factor_z_scores: Optional dict for left temporal WM (if None, uses all)
        left_gm_factor_z_scores: Optional dict for left temporal GM (if None, uses all)
        right_factor_z_scores: Optional dict for right temporal WM (if None, uses all)
        right_gm_factor_z_scores: Optional dict for right temporal GM (if None, uses all)
    """
    # First pass: collect all data to compute global y-axis limits
    all_wm_data = {f: [] for f in factor_names}
    all_gm_data = {f: [] for f in factor_names}
    
    # Collect data from all three groups
    datasets = [
        (all_factor_z_scores, all_gm_factor_z_scores),
        (left_factor_z_scores if left_factor_z_scores is not None else all_factor_z_scores,
         left_gm_factor_z_scores if left_gm_factor_z_scores is not None else all_gm_factor_z_scores),
        (right_factor_z_scores if right_factor_z_scores is not None else all_factor_z_scores,
         right_gm_factor_z_scores if right_gm_factor_z_scores is not None else all_gm_factor_z_scores),
    ]
    
    for wm_scores_dict, gm_scores_dict in datasets:
        for factor_name in factor_names:
            gm_scores = gm_scores_dict.get(factor_name, {})
            for region, z in gm_scores.items():
                if not pd.isna(z):
                    all_gm_data[factor_name].append(abs(z))
            
            wm_scores = wm_scores_dict.get(factor_name, {})
            for roi_key, z in wm_scores.items():
                if not pd.isna(z):
                    all_wm_data[factor_name].append(abs(z))
    
    # Compute global y-axis limits (including error bars)
    all_values = []
    for f in factor_names:
        if all_gm_data[f]:
            gm_mean = np.mean(all_gm_data[f])
            gm_sem = np.std(all_gm_data[f], ddof=1) / np.sqrt(len(all_gm_data[f]))
            all_values.extend([gm_mean + gm_sem, gm_mean - gm_sem])
            all_values.extend(all_gm_data[f])  # Include all data points
        if all_wm_data[f]:
            wm_mean = np.mean(all_wm_data[f])
            wm_sem = np.std(all_wm_data[f], ddof=1) / np.sqrt(len(all_wm_data[f]))
            all_values.extend([wm_mean + wm_sem, wm_mean - wm_sem])
            all_values.extend(all_wm_data[f])  # Include all data points
    
    if all_values:
        global_ymin = 0  # Absolute values start at 0
        global_ymax = max(all_values)
        # Add padding
        global_ymax += global_ymax * 0.1
    else:
        global_ymin = 0
        global_ymax = 1
    
    # Helper function to create a single subplot
    def create_subplot(ax, wm_scores_dict, gm_scores_dict, title):
        # Collect all individual absolute z-scores per factor for GM and WM
        gm_data = {f: [] for f in factor_names}
        wm_data = {f: [] for f in factor_names}
        
        for factor_name in factor_names:
            # Collect individual absolute z-scores for GM regions
            gm_scores = gm_scores_dict.get(factor_name, {})
            for region, z in gm_scores.items():
                if not pd.isna(z):
                    gm_data[factor_name].append(abs(z))
            
            # Collect individual absolute z-scores for WM tract segments
            wm_scores = wm_scores_dict.get(factor_name, {})
            for roi_key, z in wm_scores.items():
                if not pd.isna(z):
                    wm_data[factor_name].append(abs(z))
        
        # Compute means and standard errors
        gm_means = [np.mean(gm_data[f]) if gm_data[f] else 0 for f in factor_names]
        wm_means = [np.mean(wm_data[f]) if wm_data[f] else 0 for f in factor_names]
        
        gm_sems = []
        wm_sems = []
        for f in factor_names:
            if gm_data[f]:
                gm_sem = np.std(gm_data[f], ddof=1) / np.sqrt(len(gm_data[f]))
            else:
                gm_sem = 0
            gm_sems.append(gm_sem)
            
            if wm_data[f]:
                wm_sem = np.std(wm_data[f], ddof=1) / np.sqrt(len(wm_data[f]))
            else:
                wm_sem = 0
            wm_sems.append(wm_sem)
        
        x = np.arange(len(factor_names))
        width = 0.35
        
        # Set random seed for reproducibility (different for each subplot)
        np.random.seed(42)
        
        # Plot bars with error bars - increased error bar visibility
        # GM: grey bar fill
        bars1 = ax.bar(x - width/2, gm_means, width, yerr=gm_sems, 
                       label='Gray Matter', color='gray', alpha=0.7, 
                       capsize=8, error_kw={'elinewidth': 3, 'capthick': 3})
        # WM: white bar fill with thin black border
        bars2 = ax.bar(x + width/2, wm_means, width, yerr=wm_sems,
                       label='White Matter', color='white', edgecolor='black', linewidth=1,
                       capsize=8, error_kw={'elinewidth': 3, 'capthick': 3})
        
        # Add jitter points with reduced jitter - decreased point size
        jitter_scale = width/8
        for i, factor_name in enumerate(factor_names):
            # GM jitter points
            if gm_data[factor_name]:
                jitter_gm = np.random.normal(0, jitter_scale, len(gm_data[factor_name]))
                gm_scores = np.array(gm_data[factor_name])
                ax.scatter(x[i] - width/2 + jitter_gm, gm_scores, 
                          alpha=0.6, s=8, color='gray', zorder=3)
            
            # WM jitter points
            if wm_data[factor_name]:
                jitter_wm = np.random.normal(0, jitter_scale, len(wm_data[factor_name]))
                wm_scores = np.array(wm_data[factor_name])
                ax.scatter(x[i] + width/2 + jitter_wm, wm_scores,
                          alpha=0.8, s=8, color='white', edgecolors='black', linewidths=0.5, zorder=3)
        
        ax.set_xlabel('Factor', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean |Z-Score|', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(factor_names)
        
        # Set consistent y-axis limits
        ax.set_ylim(global_ymin, global_ymax)
        
        # Create custom legend (only for first subplot)
        if title == 'All Temporal':
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Gray Matter'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='black', 
                       markersize=8, label='White Matter'),
            ]
            ax.legend(handles=legend_elements, fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # All temporal
    create_subplot(axes[0], all_factor_z_scores, all_gm_factor_z_scores, 'All Temporal')
    
    # Left temporal (use provided or fallback to all)
    left_wm = left_factor_z_scores if left_factor_z_scores is not None else all_factor_z_scores
    left_gm = left_gm_factor_z_scores if left_gm_factor_z_scores is not None else all_gm_factor_z_scores
    create_subplot(axes[1], left_wm, left_gm, 'Left Temporal')
    
    # Right temporal (use provided or fallback to all)
    right_wm = right_factor_z_scores if right_factor_z_scores is not None else all_factor_z_scores
    right_gm = right_gm_factor_z_scores if right_gm_factor_z_scores is not None else all_gm_factor_z_scores
    create_subplot(axes[2], right_wm, right_gm, 'Right Temporal')
    
    # Add overall title
    fig.suptitle('Factor |z-scores|', fontsize=20, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Across factor Mahalanobis distance barplot with jitter saved: {output_path}")


def compute_average_factor_scores_controls(
    factor_scores_dir: str,
    roi_names: Sequence[str],
    control_subjects: Sequence[str],
    *,
    cohort: str = "controls",
) -> pd.DataFrame:
    """
    Compute average factor scores per ROI for controls.
    
    Returns:
        controls_avg: DataFrame with ROIs as rows, factors as columns (all controls)
    """
    control_scores = {}
    
    for roi in roi_names:
        factor_scores = load_factor_scores_from_csv(factor_scores_dir, roi, cohort=cohort)
        if factor_scores.empty:
            continue
        
        # Get available factors
        factors = factor_scores.columns.tolist()
        
        # Debug: Check subject matching
        available_subjects = set(factor_scores.index)
        control_subjects_set = set(control_subjects)
        matching_subjects = available_subjects.intersection(control_subjects_set)
        
        if len(matching_subjects) == 0:
            # Try to match with different format (e.g., "sub-XXXX" vs "XXXX")
            # Check if control_subjects have "sub-" prefix but factor_scores don't, or vice versa
            if control_subjects and len(control_subjects) > 0:
                # Try removing "sub-" prefix from control_subjects
                control_subjects_no_prefix = [s.replace("sub-", "") if s.startswith("sub-") else s for s in control_subjects]
                available_subjects_no_prefix = {s.replace("sub-", "") if s.startswith("sub-") else s for s in available_subjects}
                matching_subjects_no_prefix = set(control_subjects_no_prefix).intersection(available_subjects_no_prefix)
                if matching_subjects_no_prefix:
                    # Use the original format from factor_scores - find subjects that match after removing prefix
                    matching_subjects = {s for s in available_subjects 
                                       if (s.replace("sub-", "") if s.startswith("sub-") else s) in matching_subjects_no_prefix}
        
        if len(matching_subjects) == 0:
            if len(control_scores) == 0:  # Only print once
                print(f"  Warning: No matching control subjects found for {roi}")
                print(f"    Available subjects in CSV: {sorted(list(available_subjects))[:5]}...")
                print(f"    Control subjects provided: {sorted(list(control_subjects_set))[:5]}...")
            continue
        
        # Compute averages for controls using matching subjects
        control_avg_scores = factor_scores.loc[
            factor_scores.index.isin(matching_subjects)
        ].mean(axis=0)
        
        control_scores[roi] = control_avg_scores
    
    controls_df = pd.DataFrame(control_scores).T
    
    
    return controls_df


def compute_segment_factor_scores_controls(
    factor_scores_dir: str,
    tract_names: Sequence[str],
    control_subjects: Sequence[str],
    *,
    cohort: str = "controls",
) -> pd.DataFrame:
    """
    Load and compute average factor scores per tract segment for controls.
    Loads pre-computed segment-specific factor scores from CSV files using end labels from metadata.
    
    Returns:
        controls_avg: DataFrame with (tract, segment) as MultiIndex, factors as columns
    """
    control_scores = []
    
    tract_scores_dir = _wm_factor_scores_dir(factor_scores_dir)
    
    # Load tract metadata to get end labels
    tract_metadata_df = load_tract_metadata_full()
    tract_to_end1_label = {}
    tract_to_end2_label = {}
    if not tract_metadata_df.empty and "label" in tract_metadata_df.columns:
        if "end1" in tract_metadata_df.columns:
            tract_to_end1_label = dict(zip(tract_metadata_df["label"], tract_metadata_df["end1"]))
        if "end2" in tract_metadata_df.columns:
            tract_to_end2_label = dict(zip(tract_metadata_df["label"], tract_metadata_df["end2"]))
    
    for tract in tract_names:
        # Load factor scores for each segment
        for segment in ['end1', 'core', 'end2']:
            roi_key = _tract_segment_to_roi_key(
                tract, segment, tract_to_end1_label, tract_to_end2_label,
            )
            factor_scores = load_factor_scores_from_csv(tract_scores_dir, roi_key, cohort=cohort)
            
            if factor_scores.empty:
                continue
            
            # Get available factors
            factors = factor_scores.columns.tolist()
            
            # Compute averages for controls
            control_avg_scores = factor_scores.loc[
                factor_scores.index.isin(control_subjects)
            ].mean(axis=0)
            
            control_scores.append((tract, segment, control_avg_scores))
    
    # Convert to DataFrame with MultiIndex
    if control_scores:
        control_data = {(tract, segment): scores for tract, segment, scores in control_scores}
        controls_df = pd.DataFrame(control_data).T
    else:
        controls_df = pd.DataFrame()
    
    return controls_df


def compute_segment_factor_scores_by_group(
    factor_scores_dir: str,
    tract_names: Sequence[str],
    all_patients: Sequence[str],
    left_lateralized: Sequence[str],
    right_lateralized: Sequence[str],
    *,
    cohort: str = "epilepsy",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and compute average factor scores per tract segment for different patient groups.
    Loads pre-computed segment-specific factor scores from CSV files using end labels from metadata.
    
    Returns:
        all_patients_avg: DataFrame with (tract, segment) as MultiIndex, factors as columns
        left_lateralized_avg: DataFrame with (tract, segment) as MultiIndex, factors as columns
        right_lateralized_avg: DataFrame with (tract, segment) as MultiIndex, factors as columns
    """
    all_scores = []
    left_scores = []
    right_scores = []
    
    tract_scores_dir = _wm_factor_scores_dir(factor_scores_dir)
    
    # Load tract metadata to get end labels
    tract_metadata_df = load_tract_metadata_full()
    tract_to_end1_label = {}
    tract_to_end2_label = {}
    if not tract_metadata_df.empty and "label" in tract_metadata_df.columns:
        if "end1" in tract_metadata_df.columns:
            tract_to_end1_label = dict(zip(tract_metadata_df["label"], tract_metadata_df["end1"]))
        if "end2" in tract_metadata_df.columns:
            tract_to_end2_label = dict(zip(tract_metadata_df["label"], tract_metadata_df["end2"]))
    
    for tract in tract_names:
        # Load factor scores for each segment
        for segment in ['end1', 'core', 'end2']:
            roi_key = _tract_segment_to_roi_key(
                tract, segment, tract_to_end1_label, tract_to_end2_label,
            )
            factor_scores = load_factor_scores_from_csv(tract_scores_dir, roi_key, cohort=cohort)
            
            if factor_scores.empty:
                continue
            
            # Get available factors
            factors = factor_scores.columns.tolist()
            
            # Compute averages for each group
            all_patient_scores = factor_scores.loc[
                factor_scores.index.isin(all_patients)
            ].mean(axis=0)
            
            left_patient_scores = factor_scores.loc[
                factor_scores.index.isin(left_lateralized)
            ].mean(axis=0)
            
            right_patient_scores = factor_scores.loc[
                factor_scores.index.isin(right_lateralized)
            ].mean(axis=0)
            
            all_scores.append((tract, segment, all_patient_scores))
            left_scores.append((tract, segment, left_patient_scores))
            right_scores.append((tract, segment, right_patient_scores))
    
    # Convert to DataFrames with MultiIndex
    if all_scores:
        all_data = {(tract, segment): scores for tract, segment, scores in all_scores}
        all_patients_df = pd.DataFrame(all_data).T
    else:
        all_patients_df = pd.DataFrame()
    
    if left_scores:
        left_data = {(tract, segment): scores for tract, segment, scores in left_scores}
        left_lateralized_df = pd.DataFrame(left_data).T
    else:
        left_lateralized_df = pd.DataFrame()
    
    if right_scores:
        right_data = {(tract, segment): scores for tract, segment, scores in right_scores}
        right_lateralized_df = pd.DataFrame(right_data).T
    else:
        right_lateralized_df = pd.DataFrame()
    
    return all_patients_df, left_lateralized_df, right_lateralized_df


def get_control_group_demographics(control_subjects: Sequence[str]) -> Dict[str, Dict]:
    """
    Get demographics (age, sex) for control subjects by group.
    
    Args:
        control_subjects: List of control subject IDs
        
    Returns:
        Dictionary mapping group names to demographics info:
        {
            "penn_controls": {"subjects": [...], "n": int, "age_mean": float, "age_min": float, "age_max": float, "n_female": int},
            "hcpya": {...},
            "hcpaging": {...}
        }
    """
    demographics = {
        "penn_controls": {"subjects": [], "n": 0, "age_mean": None, "age_min": None, "age_max": None, "n_female": 0},
        "hcpya": {"subjects": [], "n": 0, "age_mean": None, "age_min": None, "age_max": None, "n_female": 0},
        "hcpaging": {"subjects": [], "n": 0, "age_mean": None, "age_min": None, "age_max": None, "n_female": 0}
    }
    
    # Normalize control_subjects to handle both formats
    control_subjects_set = set(control_subjects)
    control_subjects_normalized = {}
    for sub in control_subjects:
        control_subjects_normalized[sub] = sub
        if sub.startswith("sub-"):
            control_subjects_normalized[sub[4:]] = sub
        else:
            control_subjects_normalized[f"sub-{sub}"] = sub
    
    # Helper function to determine group from subject ID
    def get_group_from_subject_id(sub):
        """Determine control group from subject ID pattern."""
        sub_clean = sub.replace("sub-", "") if sub.startswith("sub-") else sub
        if sub_clean.startswith("RID"):
            return "penn_controls"
        elif sub_clean.startswith("HCA"):
            return "hcpaging"
        elif sub_clean.isdigit() or (sub_clean.startswith("1") and len(sub_clean) == 6):
            # HCP-YA subjects are typically 6-digit numbers starting with 1
            return "hcpya"
        return None
    
    # Load Penn Controls demographics
    penn_demo_path = ospj(METADATA_DIR, "penn_basic_demo.csv")
    if os.path.exists(penn_demo_path):
        try:
            penn_demo = pd.read_csv(penn_demo_path)
            if 'sub' in penn_demo.columns:
                for _, row in penn_demo.iterrows():
                    sub = row['sub']
                    # Check if this subject is in our control subjects list and belongs to Penn Controls
                    if (sub in control_subjects_set or sub in control_subjects_normalized) and get_group_from_subject_id(sub) == "penn_controls":
                        demographics["penn_controls"]["subjects"].append(sub)
                        if 'age' in row and pd.notna(row['age']):
                            age = float(row['age'])
                            if demographics["penn_controls"]["age_min"] is None or age < demographics["penn_controls"]["age_min"]:
                                demographics["penn_controls"]["age_min"] = age
                            if demographics["penn_controls"]["age_max"] is None or age > demographics["penn_controls"]["age_max"]:
                                demographics["penn_controls"]["age_max"] = age
                        if 'sex' in row and pd.notna(row['sex']):
                            if str(row['sex']).upper() in ['F', 'FEMALE']:
                                demographics["penn_controls"]["n_female"] += 1
        except Exception as e:
            print(f"Warning: Could not load Penn Controls demographics: {e}")
    
    # Load HCP-YA demographics
    hcpya_demo_path = ospj(METADATA_DIR, "hcpya_basic_demo.csv")
    if os.path.exists(hcpya_demo_path):
        try:
            hcpya_demo = pd.read_csv(hcpya_demo_path)
            if 'sub' in hcpya_demo.columns:
                for _, row in hcpya_demo.iterrows():
                    sub = row['sub']
                    if (sub in control_subjects_set or sub in control_subjects_normalized) and get_group_from_subject_id(sub) == "hcpya":
                        demographics["hcpya"]["subjects"].append(sub)
                        if 'age' in row and pd.notna(row['age']):
                            age = float(row['age'])
                            if demographics["hcpya"]["age_min"] is None or age < demographics["hcpya"]["age_min"]:
                                demographics["hcpya"]["age_min"] = age
                            if demographics["hcpya"]["age_max"] is None or age > demographics["hcpya"]["age_max"]:
                                demographics["hcpya"]["age_max"] = age
                        if 'sex' in row and pd.notna(row['sex']):
                            if str(row['sex']).upper() in ['F', 'FEMALE']:
                                demographics["hcpya"]["n_female"] += 1
        except Exception as e:
            print(f"Warning: Could not load HCP-YA demographics: {e}")
    
    # Load HCP-Aging demographics
    hcpaging_demo_path = ospj(METADATA_DIR, "hcpaging", "imagingcollection01.txt")
    if os.path.exists(hcpaging_demo_path):
        try:
            # HCP-Aging file is tab-separated
            hcpaging_demo = pd.read_csv(hcpaging_demo_path, sep='\t')
            if 'src_subject_id' in hcpaging_demo.columns:
                # Get unique subjects (there may be multiple rows per subject)
                unique_subjects = hcpaging_demo.drop_duplicates(subset=['src_subject_id'])
                for _, row in unique_subjects.iterrows():
                    sub_id = row['src_subject_id']
                    # HCP-Aging subjects may have format like "HCA8271980" - need to add "sub-" prefix
                    sub_with_prefix = f"sub-{sub_id}" if not sub_id.startswith("sub-") else sub_id
                    if ((sub_with_prefix in control_subjects_set or sub_id in control_subjects_set or 
                         sub_with_prefix in control_subjects_normalized or sub_id in control_subjects_normalized) and
                        get_group_from_subject_id(sub_with_prefix) == "hcpaging"):
                        demographics["hcpaging"]["subjects"].append(sub_with_prefix)
                        if 'interview_age' in row and pd.notna(row['interview_age']):
                            # Age is in months, convert to years
                            age_months = float(row['interview_age'])
                            age_years = age_months / 12.0
                            if demographics["hcpaging"]["age_min"] is None or age_years < demographics["hcpaging"]["age_min"]:
                                demographics["hcpaging"]["age_min"] = age_years
                            if demographics["hcpaging"]["age_max"] is None or age_years > demographics["hcpaging"]["age_max"]:
                                demographics["hcpaging"]["age_max"] = age_years
                        if 'sex' in row and pd.notna(row['sex']):
                            if str(row['sex']).upper() in ['F', 'FEMALE']:
                                demographics["hcpaging"]["n_female"] += 1
        except Exception as e:
            print(f"Warning: Could not load HCP-Aging demographics: {e}")
    
    # Calculate means and finalize counts
    for group in demographics:
        subjects = demographics[group]["subjects"]
        demographics[group]["n"] = len(subjects)
        
        # Calculate age mean from all subjects (need to reload to get all ages)
        ages = []
        if group == "penn_controls" and os.path.exists(penn_demo_path):
            try:
                penn_demo = pd.read_csv(penn_demo_path)
                for sub in subjects:
                    sub_row = penn_demo[penn_demo['sub'] == sub]
                    if not sub_row.empty and 'age' in sub_row.columns and pd.notna(sub_row.iloc[0]['age']):
                        ages.append(float(sub_row.iloc[0]['age']))
            except Exception:
                pass
        elif group == "hcpya" and os.path.exists(hcpya_demo_path):
            try:
                hcpya_demo = pd.read_csv(hcpya_demo_path)
                for sub in subjects:
                    sub_row = hcpya_demo[hcpya_demo['sub'] == sub]
                    if not sub_row.empty and 'age' in sub_row.columns and pd.notna(sub_row.iloc[0]['age']):
                        ages.append(float(sub_row.iloc[0]['age']))
            except Exception:
                pass
        elif group == "hcpaging" and os.path.exists(hcpaging_demo_path):
            try:
                hcpaging_demo = pd.read_csv(hcpaging_demo_path, sep='\t')
                unique_subjects = hcpaging_demo.drop_duplicates(subset=['src_subject_id'])
                for sub in subjects:
                    sub_id = sub.replace("sub-", "") if sub.startswith("sub-") else sub
                    sub_row = unique_subjects[unique_subjects['src_subject_id'] == sub_id]
                    if not sub_row.empty and 'interview_age' in sub_row.columns and pd.notna(sub_row.iloc[0]['interview_age']):
                        ages.append(float(sub_row.iloc[0]['interview_age']) / 12.0)
            except Exception:
                pass
        
        if ages:
            demographics[group]["age_mean"] = np.mean(ages)
    
    return demographics


def get_subject_counts_by_group(
    regions: Sequence[str],
    tracts: Sequence[str],
    scalar_labels: Sequence[str],
    control_groups: Sequence[str],
    patient_groups: Sequence[str],
) -> Dict[str, int]:
    """
    Get subject counts for each group by loading GAM data directly (all subjects in
    each normative / epilepsy group in the sample files, not FA-only lists).
    
    Returns:
        Dictionary mapping group names to subject counts
    """
    group_counts = {
        "epilepsy": 0,
        "penn_controls": 0,
        "hcpya": 0,
        "hcpaging": 0
    }
    
    # Collect subjects by group from GAM files
    subjects_by_group = {
        "epilepsy": set(),
        "penn_controls": set(),
        "hcpya": set(),
        "hcpaging": set()
    }
    
    # Map GAM file group names to display names
    # GAM files use "penn_epilepsy" but we want to count it as "epilepsy"
    group_name_mapping = {
        "penn_epilepsy": "epilepsy",
        "penn_controls": "penn_controls",  # Already correct
        "hcpya": "hcpya",  # Already correct
        "hcpaging": "hcpaging",  # Already correct
    }
    
    # Check GM regions - load one sample to get group info
    if regions and scalar_labels:
        sample_region = regions[0]
        sample_scalar = scalar_labels[0]
        gm_base = get_mni_micro_gm_profile_dir_for_region(sample_region)
        gam_path = ospj(gm_base, sample_region, f"{sample_region}_{sample_scalar}_stat-mean_gam.csv")
        if not os.path.exists(gam_path):
            gam_path = ospj(gm_base, sample_region, f"{sample_region}_{sample_scalar}_gam.csv")
        if os.path.exists(gam_path):
            try:
                gam_data = pd.read_csv(gam_path)
                if 'group' in gam_data.columns and 'sub' in gam_data.columns:
                    for group in gam_data['group'].unique():
                        # Map group name from GAM file to display name
                        display_group = group_name_mapping.get(group, group)
                        if display_group in subjects_by_group:
                            group_subjects = set(gam_data[gam_data['group'] == group]['sub'].unique())
                            subjects_by_group[display_group].update(group_subjects)
            except Exception as e:
                print(f"Warning: Could not load group info from {gam_path}: {e}")
    
    # Check WM tracts - load one sample to get group info
    if tracts and scalar_labels:
        sample_tract = tracts[0]
        sample_scalar = scalar_labels[0]
        gam_path = ospj(WM_PROFILE_DIR_PYAFQ, sample_tract, f"{sample_tract}_{sample_scalar}_stat-mean_gam.csv")
        if not os.path.exists(gam_path):
            gam_path = ospj(WM_PROFILE_DIR_PYAFQ, sample_tract, f"{sample_tract}_{sample_scalar}_gam.csv")
        if os.path.exists(gam_path):
            try:
                gam_data = pd.read_csv(gam_path)
                if 'group' in gam_data.columns and 'sub' in gam_data.columns:
                    for group in gam_data['group'].unique():
                        # Map group name from GAM file to display name
                        display_group = group_name_mapping.get(group, group)
                        if display_group in subjects_by_group:
                            group_subjects = set(gam_data[gam_data['group'] == group]['sub'].unique())
                            subjects_by_group[display_group].update(group_subjects)
            except Exception as e:
                print(f"Warning: Could not load group info from {gam_path}: {e}")
    
    # Count unique subjects per group
    for group in group_counts:
        group_counts[group] = len(subjects_by_group[group])
    
    return group_counts


def expand_gm_region_abbreviation(region_name):
    """
    Expand GM region abbreviations to full names.
    
    Args:
        region_name: Region name after hemisphere prefix removal (e.g., "Ca", "Pu", "SNc_PBP_VTA")
    
    Returns:
        Expanded region name (e.g., "Caudate nucleus", "Putamen", "Substantia nigra, pars campacta")
    """
    abbreviation_expansions = {
        "Ca": "Caudate nucleus",
        "Pu": "Putamen",
        "NAC": "Nucleus accumbens",
        "EXA": "Extended amygdala",
        "GPe": "Globus pallidus external",
        "GPi": "Globus pallidus internal",
        "SNc_PBP_VTA": "Substantia nigra, pars campacta",
        "STH": "Subthalamic nucleus",
        "RN": "Red nucleus",
        "SNr": "Substantia nigra, pars reticulata",
        "HTH": "Hypothalamus",
        "HN": "Habenular nuclei",
        "VeP": "Ventral pallidum",
        "MN": "Mammillary nucleus"
    }
    return abbreviation_expansions.get(region_name, region_name)


def generate_abnormality_summary_html(scores_dict, top_n=10, is_tract=False, tract_label_to_name=None, tract_to_end1=None, tract_to_end2=None):
    """
    Generate HTML for abnormality summary bar charts.
    
    Args:
        scores_dict: Dictionary mapping ROI names to scores
        top_n: Number of top abnormalities to show
        is_tract: If True, format as tract segments; if False, format as regions
        tract_label_to_name: Dictionary mapping tract labels to names
        tract_to_end1: Dictionary mapping tract labels to end1 labels
        tract_to_end2: Dictionary mapping tract labels to end2 labels
    
    Returns:
        HTML string for bar chart
    """
    if not scores_dict:
        return '<div class="bar-item">No abnormalities found</div>'
    
    # Sort by absolute value and get top N
    sorted_items = sorted(scores_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_items = sorted_items[:top_n]
    
    if not top_items:
        return '<div class="bar-item">No abnormalities found</div>'
    
    max_count = max([abs(count) for _, count in top_items])
    
    html_items = []
    for roi_name, count in top_items:
        if is_tract:
            # Check if roi_name is already formatted (contains parentheses)
            if " (" in roi_name and roi_name.endswith(")"):
                # Already formatted, use as-is
                display_name = roi_name
            else:
                # Parse tract segment name: format is {tract}_{segment} or {tract}_{hemi}_{segment}
                parts = roi_name.rsplit("_", 2)
                if len(parts) == 3:
                    tract_base, hemi, segment_label = parts
                    tract_label = f"{tract_base}_{hemi}"
                elif len(parts) == 2:
                    tract_base, segment_label = parts
                    tract_label = tract_base
                    hemi = ""
                else:
                    tract_label = roi_name
                    segment_label = ""
                    hemi = ""
                
                # Get human-readable tract name
                if tract_label_to_name and tract_label in tract_label_to_name:
                    human_name = tract_label_to_name[tract_label]
                    if human_name.endswith("_L") or human_name.endswith("_R"):
                        human_name = human_name[:-2]
                    human_name = human_name.replace("_", " ")
                else:
                    human_name = tract_label.replace("_", " ")
                
                # Expand segment label - handle both "end-A" format and single letter format
                # First, remove "end-" prefix if present
                segment_clean = segment_label.replace("end-", "") if segment_label.startswith("end-") else segment_label
                
                segment_expansions = {
                    "core": "Core",
                    "A": "Anterior",
                    "P": "Posterior",
                    "I": "Inferior",
                    "S": "Superior",
                    "M": "Medial",
                    "L": "Lateral"
                }
                expanded_segment = segment_expansions.get(segment_clean, segment_clean)
                
                # Format display name - always put segment label in parentheses if it exists
                if expanded_segment and expanded_segment != "":
                    if hemi:
                        display_name = f"{hemi} {human_name} ({expanded_segment})"
                    else:
                        display_name = f"{human_name} ({expanded_segment})"
                else:
                    # If no segment, don't add parentheses
                    if hemi:
                        display_name = f"{hemi} {human_name}"
                    else:
                        display_name = human_name
        else:
            # Format region name
            clean_region = roi_name
            hemisphere = ""
            if clean_region.startswith("LH-") or clean_region.startswith("LH_"):
                clean_region = clean_region[3:]
                hemisphere = "L"
            elif clean_region.startswith("RH-") or clean_region.startswith("RH_"):
                clean_region = clean_region[3:]
                hemisphere = "R"
            
            # Expand abbreviations
            expanded_region = expand_gm_region_abbreviation(clean_region)
            # If abbreviation was expanded, use it directly; otherwise format normally
            if expanded_region != clean_region:
                human_region = expanded_region
            else:
                human_region = clean_region.replace("_", " ").title()
            
            if hemisphere:
                display_name = f"{hemisphere} {human_region}"
            else:
                display_name = human_region
        
        percentage = (abs(count) / max_count) * 100 if max_count > 0 else 0
        html_items.append(f'''
                            <div class="bar-item">
                                <div class="bar-label">{display_name}</div>
                                <div class="bar-container">
                                    <div class="bar-fill" style="width: {percentage:.1f}%"></div>
                                </div>
                                <div class="bar-count">{count:.2f}</div>
                            </div>''')
    
    return ''.join(html_items)


def create_master_report(
    output_path: str,
    factor_scores_dir: str,
    all_regions: Sequence[str],
    all_tracts: Sequence[str],
    all_patients: Sequence[str],
    left_lateralized: Sequence[str],
    right_lateralized: Sequence[str],
    tract_label_to_name: Dict[str, str],
    subject_counts: Dict[str, int] = None,
    control_subjects: Optional[Sequence[str]] = None,
) -> None:
    """
    Create a master HTML report with abnormality summaries using signed z-scores.
    
    Args:
        output_path: Path to save the HTML report
        factor_scores_dir: Directory containing factor scores
        all_regions: List of all GM regions
        all_tracts: List of all WM tracts
        all_patients: List of all patient subjects
        left_lateralized: List of left-lateralized patients
        right_lateralized: List of right-lateralized patients
        tract_label_to_name: Dictionary mapping tract labels to names
        subject_counts: Dictionary mapping group names to subject counts
        control_subjects: List of control subjects (for computing z-scores)
    """
    # Only generate signed z-scores report (not absolute)
    report_type = "Signed Z-Scores"
    
    # Create directory for brain map images
    brain_maps_dir = ospj(os.path.dirname(output_path), "brain_maps")
    os.makedirs(brain_maps_dir, exist_ok=True)
    
    # Load full tract metadata for segment labels
    tract_metadata_df = load_tract_metadata_full()
    tract_to_end1 = {}
    tract_to_end2 = {}
    if not tract_metadata_df.empty and "label" in tract_metadata_df.columns:
        if "end1" in tract_metadata_df.columns:
            tract_to_end1 = dict(zip(tract_metadata_df["label"], tract_metadata_df["end1"]))
        if "end2" in tract_metadata_df.columns:
            tract_to_end2 = dict(zip(tract_metadata_df["label"], tract_metadata_df["end2"]))
    
    # Load factor scores for regions (from epilepsy directory)
    region_scores_dir = ospj(factor_scores_dir, "epilepsy", "gm_regions")
    # Fallback to old structure if new structure doesn't exist
    if not os.path.exists(region_scores_dir):
        region_scores_dir = ospj(factor_scores_dir, "gm_regions")
    # Consolidated layout: wide CSVs at factor_scores root (epilepsy_Fk_scores.csv)
    if not os.path.exists(region_scores_dir):
        region_scores_dir = factor_scores_dir
    region_all, region_left, region_right = compute_average_factor_scores_by_group(
        region_scores_dir, all_regions,
        all_patients, left_lateralized, right_lateralized,
        cohort="epilepsy",
    )
    
    # Load segment-specific factor scores for tracts (from epilepsy directory)
    # Use epilepsy subdirectory for factor scores
    epilepsy_factor_scores_dir = ospj(factor_scores_dir, "epilepsy")
    # Fallback to old structure if new structure doesn't exist
    if not os.path.exists(epilepsy_factor_scores_dir):
        epilepsy_factor_scores_dir = factor_scores_dir
    tract_segment_all, tract_segment_left, tract_segment_right = compute_segment_factor_scores_by_group(
        epilepsy_factor_scores_dir, all_tracts,
        all_patients, left_lateralized, right_lateralized,
        cohort="epilepsy",
    )
    
    # Compute control factor scores using new method if control_subjects provided
    region_controls = pd.DataFrame()
    tract_segment_controls = pd.DataFrame()
    roi_means_rescaled_gm = pd.DataFrame()
    roi_means_rescaled_wm = pd.DataFrame()
    if control_subjects:
        # Load factor loadings
        factor_loadings = load_factor_loadings()
        if not factor_loadings.empty:
            # Load scalar labels
            scalar_labels = load_scalar_labels()
            
            # Compute roi_means_rescaled for controls
            print("Computing control roi_means_rescaled and roi_factor_scores...")
            roi_means_rescaled_gm, roi_means_rescaled_wm = compute_roi_means_rescaled_controls(
                all_regions, all_tracts, scalar_labels, CONTROL_GROUPS
            )
            
            # Save roi_means_rescaled to CSV files
            roi_means_output_dir = ospj(OUTPUT_PROJECT_ROOT, "roi_means_rescaled")
            os.makedirs(roi_means_output_dir, exist_ok=True)
            
            if not roi_means_rescaled_gm.empty:
                gm_output_path = ospj(roi_means_output_dir, "gm_regions_rescaled.csv")
                roi_means_rescaled_gm.to_csv(gm_output_path)
                print(f"  Saved GM roi_means_rescaled to {gm_output_path}")
            
            if not roi_means_rescaled_wm.empty:
                wm_output_path = ospj(roi_means_output_dir, "wm_tracts_rescaled.csv")
                roi_means_rescaled_wm.to_csv(wm_output_path)
                print(f"  Saved WM roi_means_rescaled to {wm_output_path}")
            
            # Compute roi_factor_scores from rescaled means and absolute loadings
            region_controls, tract_segment_controls = compute_roi_factor_scores_from_rescaled(
                roi_means_rescaled_gm, roi_means_rescaled_wm, factor_loadings, scalar_labels
            )
            
            # Save roi_factor_scores to CSV files
            if not region_controls.empty:
                factor_scores_output_dir = ospj(OUTPUT_PROJECT_ROOT, "roi_factor_scores")
                os.makedirs(factor_scores_output_dir, exist_ok=True)
                gm_scores_path = ospj(factor_scores_output_dir, "gm_regions_factor_scores.csv")
                region_controls.to_csv(gm_scores_path)
                print(f"  Saved GM roi_factor_scores to {gm_scores_path}")
            
            if not tract_segment_controls.empty:
                wm_scores_path = ospj(factor_scores_output_dir, "wm_tracts_factor_scores.csv")
                tract_segment_controls.to_csv(wm_scores_path)
                print(f"  Saved WM roi_factor_scores to {wm_scores_path}")
        else:
            print(f"  Warning: Could not load factor loadings. Skipping control brain maps.")
    
    if region_all.empty and tract_segment_all.empty:
        print("Warning: No factor scores found. Skipping master report.")
        return
    
    # Get factors
    factors = []
    if not region_all.empty:
        factors = region_all.columns.tolist()
    elif not tract_segment_all.empty:
        factors = tract_segment_all.columns.tolist()
    
    # Compute factor z-scores from control statistics (if control_subjects provided)
    # This converts average factor scores to z-scores relative to control mean/std
    all_factor_z_scores = {}  # {factor_name: {roi: z_score}} for WM
    all_gm_factor_z_scores = {}  # {factor_name: {region: z_score}} for GM
    left_factor_z_scores = None  # {factor_name: {roi: z_score}} for WM (left temporal)
    left_gm_factor_z_scores = None  # {factor_name: {region: z_score}} for GM (left temporal)
    right_factor_z_scores = None  # {factor_name: {roi: z_score}} for WM (right temporal)
    right_gm_factor_z_scores = None  # {factor_name: {region: z_score}} for GM (right temporal)
    controls_scores_subdir = ospj(factor_scores_dir, "controls")
    has_control_factor_scores = os.path.exists(controls_scores_subdir) or bool(
        glob.glob(ospj(factor_scores_dir, "controls_*_scores.csv"))
    )
    controls_factor_scores_dir = (
        controls_scores_subdir if os.path.exists(controls_scores_subdir) else factor_scores_dir
    )

    if control_subjects and has_control_factor_scores:
        # Check cache for z-scores
        z_scores_cache_path = ospj(os.path.dirname(output_path), "z_scores_cache.pkl")
        cached_z_scores = None
        # if os.path.exists(z_scores_cache_path):
        #     print(f"Loading cached z-scores from {z_scores_cache_path}")
        #     try:
        #         with open(z_scores_cache_path, 'rb') as f:
        #             cached_z_scores = pickle.load(f)
        #         print("Using cached z-scores - skipping computation")
        #         all_gm_factor_z_scores = cached_z_scores.get('all_gm_factor_z_scores', {})
        #         all_factor_z_scores = cached_z_scores.get('all_factor_z_scores', {})
        #         left_gm_factor_z_scores = cached_z_scores.get('left_gm_factor_z_scores', {})
        #         left_factor_z_scores = cached_z_scores.get('left_factor_z_scores', {})
        #         right_gm_factor_z_scores = cached_z_scores.get('right_gm_factor_z_scores', {})
        #         right_factor_z_scores = cached_z_scores.get('right_factor_z_scores', {})
        #     except Exception as e:
        #         print(f"Warning: Failed to load z-scores cache: {e}. Will recompute.")
        #         cached_z_scores = None
        
        # Define directories for epilepsy factor scores (needed for both cached and non-cached paths)
        epilepsy_region_scores_dir = ospj(epilepsy_factor_scores_dir, "gm_regions")
        if not os.path.exists(epilepsy_region_scores_dir):
            epilepsy_region_scores_dir = ospj(factor_scores_dir, "gm_regions")
        if not os.path.exists(epilepsy_region_scores_dir):
            epilepsy_region_scores_dir = factor_scores_dir

        epilepsy_tract_scores_dir = _wm_factor_scores_dir(epilepsy_factor_scores_dir)
        if not os.path.isdir(ospj(epilepsy_factor_scores_dir, "wm_tracts")):
            epilepsy_tract_scores_dir = _wm_factor_scores_dir(factor_scores_dir)
        
        if cached_z_scores is None:
            n_wm = len(tract_segment_all) if not tract_segment_all.empty else 0
            n_gm = len(region_all) if not region_all.empty else 0
            print(
                f"Computing factor z-scores from control statistics "
                f"({n_gm} GM + {n_wm} WM ROIs)..."
            )
            # Preload consolidated tables once (avoids re-reading multi-MB CSVs per ROI)
            _load_consolidated_cohort_tables(
                _consolidated_scores_root(controls_factor_scores_dir), "controls"
            )
            _load_consolidated_cohort_tables(
                _consolidated_scores_root(epilepsy_region_scores_dir), "epilepsy"
            )

        for factor in factors:
            all_gm_factor_z_scores[factor] = {}
            if not region_all.empty and factor in region_all.columns:
                for region in region_all.index:
                    # Compute z-scores for each patient, then average
                    z_score = compute_averaged_z_scores_from_individual_patients(
                        all_patients,
                        epilepsy_region_scores_dir,
                        region,
                        factor,
                        controls_factor_scores_dir,
                        cohort="epilepsy",
                    )
                    if not pd.isna(z_score):
                        all_gm_factor_z_scores[factor][region] = z_score
        
        # Compute z-scores for WM tract segments
        # Use per-patient z-scores, then average (correct approach)
        for factor in factors:
            all_factor_z_scores[factor] = {}
            if not tract_segment_all.empty and factor in tract_segment_all.columns:
                for (tract, segment), row in tract_segment_all.iterrows():
                    # Get segment label for file naming
                    end1_label = tract_to_end1.get(tract, "end1")
                    end2_label = tract_to_end2.get(tract, "end2")
                    segment_to_label = {
                        'end1': f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
                        'core': 'core',
                        'end2': f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
                    }
                    segment_label = segment_to_label.get(segment, segment)
                    roi_name = f"{tract}_{segment_label}"
                    
                    # Compute z-scores for each patient, then average
                    z_score = compute_averaged_z_scores_from_individual_patients(
                        all_patients,
                        epilepsy_tract_scores_dir,
                        roi_name,
                        factor,
                        controls_factor_scores_dir,
                        cohort="epilepsy",
                    )
                    if not pd.isna(z_score):
                        # Store with tract segment identifier format: "{tract_base}_{hemi}_{seg_label}"
                        # Need to determine hemi from tract name
                        if tract.endswith("_L"):
                            tract_base = tract[:-2]
                            hemi = "L"
                        elif tract.endswith("_R"):
                            tract_base = tract[:-2]
                            hemi = "R"
                        else:
                            tract_base = tract
                            hemi = ""
                        
                        if hemi:
                            roi_key = f"{tract_base}_{hemi}_{segment_label}"
                        else:
                            roi_key = f"{tract_base}_{segment_label}"
                        all_factor_z_scores[factor][roi_key] = z_score
        
        print(f"  Computed z-scores for {len(factors)} factors")
        
        # Also compute z-scores for left and right lateralized groups separately
        left_factor_z_scores = {}  # {factor_name: {roi: z_score}} for WM
        left_gm_factor_z_scores = {}  # {factor_name: {region: z_score}} for GM
        right_factor_z_scores = {}  # {factor_name: {roi: z_score}} for WM
        right_gm_factor_z_scores = {}  # {factor_name: {region: z_score}} for GM
        
        # Compute z-scores for left lateralized GM regions
        # Use per-patient z-scores, then average (correct approach)
        for factor in factors:
            left_gm_factor_z_scores[factor] = {}
            if not region_left.empty and factor in region_left.columns:
                for region in region_left.index:
                    # Compute z-scores for each patient, then average
                    z_score = compute_averaged_z_scores_from_individual_patients(
                        left_lateralized,
                        epilepsy_region_scores_dir,
                        region,
                        factor,
                        controls_factor_scores_dir,
                        cohort="epilepsy",
                    )
                    if not pd.isna(z_score):
                        left_gm_factor_z_scores[factor][region] = z_score
        
        # Compute z-scores for left lateralized WM tract segments
        # Use per-patient z-scores, then average (correct approach)
        for factor in factors:
            left_factor_z_scores[factor] = {}
            if not tract_segment_left.empty and factor in tract_segment_left.columns:
                for (tract, segment), row in tract_segment_left.iterrows():
                    end1_label = tract_to_end1.get(tract, "end1")
                    end2_label = tract_to_end2.get(tract, "end2")
                    segment_to_label = {
                        'end1': f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
                        'core': 'core',
                        'end2': f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
                    }
                    segment_label = segment_to_label.get(segment, segment)
                    roi_name = f"{tract}_{segment_label}"
                    
                    # Compute z-scores for each patient, then average
                    z_score = compute_averaged_z_scores_from_individual_patients(
                        left_lateralized,
                        epilepsy_tract_scores_dir,
                        roi_name,
                        factor,
                        controls_factor_scores_dir,
                        cohort="epilepsy",
                    )
                    if not pd.isna(z_score):
                        if tract.endswith("_L"):
                            tract_base = tract[:-2]
                            hemi = "L"
                        elif tract.endswith("_R"):
                            tract_base = tract[:-2]
                            hemi = "R"
                        else:
                            tract_base = tract
                            hemi = ""
                        if hemi:
                            roi_key = f"{tract_base}_{hemi}_{segment_label}"
                        else:
                            roi_key = f"{tract_base}_{segment_label}"
                        left_factor_z_scores[factor][roi_key] = z_score
        
        # Compute z-scores for right lateralized GM regions
        # Use per-patient z-scores, then average (correct approach)
        for factor in factors:
            right_gm_factor_z_scores[factor] = {}
            if not region_right.empty and factor in region_right.columns:
                for region in region_right.index:
                    # Compute z-scores for each patient, then average
                    z_score = compute_averaged_z_scores_from_individual_patients(
                        right_lateralized,
                        epilepsy_region_scores_dir,
                        region,
                        factor,
                        controls_factor_scores_dir,
                        cohort="epilepsy",
                    )
                    if not pd.isna(z_score):
                        right_gm_factor_z_scores[factor][region] = z_score
        
        # Compute z-scores for right lateralized WM tract segments
        # Use per-patient z-scores, then average (correct approach)
        for factor in factors:
            right_factor_z_scores[factor] = {}
            if not tract_segment_right.empty and factor in tract_segment_right.columns:
                for (tract, segment), row in tract_segment_right.iterrows():
                    end1_label = tract_to_end1.get(tract, "end1")
                    end2_label = tract_to_end2.get(tract, "end2")
                    segment_to_label = {
                        'end1': f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
                        'core': 'core',
                        'end2': f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
                    }
                    segment_label = segment_to_label.get(segment, segment)
                    roi_name = f"{tract}_{segment_label}"
                    
                    # Compute z-scores for each patient, then average
                    z_score = compute_averaged_z_scores_from_individual_patients(
                        right_lateralized,
                        epilepsy_tract_scores_dir,
                        roi_name,
                        factor,
                        controls_factor_scores_dir,
                        cohort="epilepsy",
                    )
                    if not pd.isna(z_score):
                        if tract.endswith("_L"):
                            tract_base = tract[:-2]
                            hemi = "L"
                        elif tract.endswith("_R"):
                            tract_base = tract[:-2]
                            hemi = "R"
                        else:
                            tract_base = tract
                            hemi = ""
                        if hemi:
                            roi_key = f"{tract_base}_{hemi}_{segment_label}"
                        else:
                            roi_key = f"{tract_base}_{segment_label}"
                        right_factor_z_scores[factor][roi_key] = z_score

        # Save z-scores to cache
        print(f"Saving z-scores to cache: {z_scores_cache_path}")
        cache_data = {
            'all_gm_factor_z_scores': all_gm_factor_z_scores,
            'all_factor_z_scores': all_factor_z_scores,
            'left_gm_factor_z_scores': left_gm_factor_z_scores,
            'left_factor_z_scores': left_factor_z_scores,
            'right_gm_factor_z_scores': right_gm_factor_z_scores,
            'right_factor_z_scores': right_factor_z_scores
        }
        with open(z_scores_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    else:
        print("Warning: Control subjects or control factor scores directory not found. Cannot compute z-scores.")
        left_factor_z_scores = None
        left_gm_factor_z_scores = None
        right_factor_z_scores = None
        right_gm_factor_z_scores = None
        # Use raw factor scores as fallback (will be treated as z-scores)
        for factor in factors:
            all_gm_factor_z_scores[factor] = region_all[factor].to_dict() if not region_all.empty and factor in region_all.columns else {}
            if not tract_segment_all.empty and factor in tract_segment_all.columns:
                all_factor_z_scores[factor] = {}
                for (tract, segment), row in tract_segment_all.iterrows():
                    score = row[factor]
                    if not pd.isna(score):
                        end1_label = tract_to_end1.get(tract, "end1")
                        end2_label = tract_to_end2.get(tract, "end2")
                        segment_to_label = {
                            'end1': f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
                            'core': 'core',
                            'end2': f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
                        }
                        segment_label = segment_to_label.get(segment, segment)
                        if tract.endswith("_L"):
                            tract_base = tract[:-2]
                            hemi = "L"
                        elif tract.endswith("_R"):
                            tract_base = tract[:-2]
                            hemi = "R"
                        else:
                            tract_base = tract
                            hemi = ""
                        if hemi:
                            roi_key = f"{tract_base}_{hemi}_{segment_label}"
                        else:
                            roi_key = f"{tract_base}_{segment_label}"
                        all_factor_z_scores[factor][roi_key] = score
    
    # Save factor z-scores to CSV files
    factor_z_scores_output_dir = ospj(OUTPUT_PROJECT_ROOT, "factor_z_scores")
    os.makedirs(factor_z_scores_output_dir, exist_ok=True)
    
    # Save all-patient z-scores
    if all_gm_factor_z_scores:
        for factor in factors:
            if factor in all_gm_factor_z_scores and all_gm_factor_z_scores[factor]:
                gm_df = pd.DataFrame.from_dict(all_gm_factor_z_scores[factor], orient='index', columns=[factor])
                gm_df.index.name = 'region'
                gm_output_path = ospj(factor_z_scores_output_dir, f"all_patients_gm_{factor}_z_scores.csv")
                gm_df.to_csv(gm_output_path)
                print(f"  Saved all-patient GM {factor} z-scores to {gm_output_path}")
    
    if all_factor_z_scores:
        for factor in factors:
            if factor in all_factor_z_scores and all_factor_z_scores[factor]:
                wm_df = pd.DataFrame.from_dict(all_factor_z_scores[factor], orient='index', columns=[factor])
                wm_df.index.name = 'roi'
                wm_output_path = ospj(factor_z_scores_output_dir, f"all_patients_wm_{factor}_z_scores.csv")
                wm_df.to_csv(wm_output_path)
                print(f"  Saved all-patient WM {factor} z-scores to {wm_output_path}")
    
    # Save left TLE z-scores
    if left_gm_factor_z_scores:
        for factor in factors:
            if factor in left_gm_factor_z_scores and left_gm_factor_z_scores[factor]:
                gm_df = pd.DataFrame.from_dict(left_gm_factor_z_scores[factor], orient='index', columns=[factor])
                gm_df.index.name = 'region'
                gm_output_path = ospj(factor_z_scores_output_dir, f"left_tle_gm_{factor}_z_scores.csv")
                gm_df.to_csv(gm_output_path)
                print(f"  Saved left TLE GM {factor} z-scores to {gm_output_path}")
    
    if left_factor_z_scores:
        for factor in factors:
            if factor in left_factor_z_scores and left_factor_z_scores[factor]:
                wm_df = pd.DataFrame.from_dict(left_factor_z_scores[factor], orient='index', columns=[factor])
                wm_df.index.name = 'roi'
                wm_output_path = ospj(factor_z_scores_output_dir, f"left_tle_wm_{factor}_z_scores.csv")
                wm_df.to_csv(wm_output_path)
                print(f"  Saved left TLE WM {factor} z-scores to {wm_output_path}")
    
    # Save right TLE z-scores
    if right_gm_factor_z_scores:
        for factor in factors:
            if factor in right_gm_factor_z_scores and right_gm_factor_z_scores[factor]:
                gm_df = pd.DataFrame.from_dict(right_gm_factor_z_scores[factor], orient='index', columns=[factor])
                gm_df.index.name = 'region'
                gm_output_path = ospj(factor_z_scores_output_dir, f"right_tle_gm_{factor}_z_scores.csv")
                gm_df.to_csv(gm_output_path)
                print(f"  Saved right TLE GM {factor} z-scores to {gm_output_path}")
    
    if right_factor_z_scores:
        for factor in factors:
            if factor in right_factor_z_scores and right_factor_z_scores[factor]:
                wm_df = pd.DataFrame.from_dict(right_factor_z_scores[factor], orient='index', columns=[factor])
                wm_df.index.name = 'roi'
                wm_output_path = ospj(factor_z_scores_output_dir, f"right_tle_wm_{factor}_z_scores.csv")
                wm_df.to_csv(wm_output_path)
                print(f"  Saved right TLE WM {factor} z-scores to {wm_output_path}")
    
    # Save epilepsy patient-specific factor z-scores (rows=subjects, columns=GM regions + WM segments)
    if control_subjects and has_control_factor_scores:
        from scipy.linalg import LinAlgError as ScipyLinAlgError
        n_wm = len(tract_segment_all) if not tract_segment_all.empty else 0
        n_gm = len(region_all) if not region_all.empty else 0
        print(
            f"Building epilepsy patient-specific factor z-scores "
            f"({n_gm} GM + {n_wm} WM ROIs; first run may take several minutes on network storage)..."
        )
        patients_index = list(all_patients)
        all_epilepsy_dfs = {}  # factor -> DataFrame (subjects x ROIs) for Mahalanobis later
        for factor in factors:
            columns_list = []  # collect (name, Series) then concat once to avoid fragmentation
            # GM regions
            if not region_all.empty and factor in region_all.columns:
                for region in region_all.index:
                    per_patient = compute_per_patient_z_scores(
                        all_patients,
                        epilepsy_region_scores_dir,
                        region,
                        factor,
                        controls_factor_scores_dir,
                        cohort="epilepsy",
                    )
                    columns_list.append((region, pd.Series(per_patient)))
            # WM tract segments
            if not tract_segment_all.empty and factor in tract_segment_all.columns:
                for (tract, segment), row in tract_segment_all.iterrows():
                    end1_label = tract_to_end1.get(tract, "end1")
                    end2_label = tract_to_end2.get(tract, "end2")
                    segment_to_label = {
                        'end1': f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
                        'core': 'core',
                        'end2': f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
                    }
                    segment_label = segment_to_label.get(segment, segment)
                    roi_name = f"{tract}_{segment_label}"
                    if tract.endswith("_L"):
                        tract_base = tract[:-2]
                        hemi = "L"
                    elif tract.endswith("_R"):
                        tract_base = tract[:-2]
                        hemi = "R"
                    else:
                        tract_base = tract
                        hemi = ""
                    roi_key = f"{tract_base}_{hemi}_{segment_label}" if hemi else f"{tract_base}_{segment_label}"
                    per_patient = compute_per_patient_z_scores(
                        all_patients,
                        epilepsy_tract_scores_dir,
                        roi_key,
                        factor,
                        controls_factor_scores_dir,
                        cohort="epilepsy",
                    )
                    columns_list.append((roi_key, pd.Series(per_patient)))
            if columns_list:
                epilepsy_df = pd.concat([s.reindex(patients_index) for _, s in columns_list], axis=1, copy=False)
                epilepsy_df.columns = [name for name, _ in columns_list]
                epilepsy_df.index.name = 'subject'
                all_epilepsy_dfs[factor] = epilepsy_df
                out_path = ospj(factor_z_scores_output_dir, f"epilepsy_{factor}_z_scores.csv")
                epilepsy_df.to_csv(out_path)
                print(f"  Saved epilepsy patient-specific {factor} z-scores to {out_path}")
        # Build and save epilepsy_mahalanobis.csv (per-patient Mahalanobis distance per ROI)
        if all_epilepsy_dfs and factors:
            roi_names = list(all_epilepsy_dfs[factors[0]].columns)
            mahal_columns_list = []
            for roi_name in roi_names:
                control_z_array = load_control_z_scores_for_roi(
                    controls_factor_scores_dir, roi_name, factors, control_subjects
                )
                if control_z_array is None or len(control_z_array) == 0:
                    continue
                n_factors = control_z_array.shape[1]
                control_mean = np.zeros(n_factors)
                try:
                    if len(control_z_array) > n_factors:
                        cov_matrix = np.cov(control_z_array.T, ddof=0)
                        if np.linalg.cond(cov_matrix) < 1e12:
                            cov_inv = np.linalg.inv(cov_matrix)
                            use_mahal = True
                        else:
                            use_mahal = False
                    else:
                        use_mahal = False
                except (ScipyLinAlgError, np.linalg.LinAlgError):
                    use_mahal = False
                per_patient_mahal = {}
                for patient in patients_index:
                    z_vec = np.array([
                        all_epilepsy_dfs[f].loc[patient, roi_name]
                        for f in factors
                        if f in all_epilepsy_dfs and roi_name in all_epilepsy_dfs[f].columns
                    ], dtype=float)
                    if len(z_vec) != n_factors or np.any(np.isnan(z_vec)):
                        per_patient_mahal[patient] = np.nan
                        continue
                    diff = z_vec - control_mean
                    if use_mahal:
                        try:
                            mahal_dist = np.sqrt(diff @ cov_inv @ diff)
                        except (ScipyLinAlgError, np.linalg.LinAlgError):
                            mahal_dist = np.linalg.norm(diff)
                    else:
                        mahal_dist = np.linalg.norm(diff)
                    per_patient_mahal[patient] = float(mahal_dist)
                mahal_columns_list.append((roi_name, pd.Series(per_patient_mahal)))
            if mahal_columns_list:
                mahal_df = pd.concat([s.reindex(patients_index) for _, s in mahal_columns_list], axis=1, copy=False)
                mahal_df.columns = [name for name, _ in mahal_columns_list]
                mahal_df.index.name = 'subject'
                mahal_path = ospj(factor_z_scores_output_dir, "epilepsy_mahalanobis.csv")
                mahal_df.to_csv(mahal_path)
                print(f"  Saved epilepsy patient-specific Mahalanobis distances to {mahal_path}")

        # Save controls factor z-scores (rows=control subjects, columns=subject, group, then ROIs)
        print("Building controls factor z-scores (with group column)...")
        controls_index = list(control_subjects)

        def _load_control_z_scores_for_roi_factor(controls_dir, roi_name, factor, subjects):
            """Load control factor scores for one ROI and one factor; return Series of z-scores (subject -> z)."""
            cons = _resolve_consolidated_cohort_factor_path(controls_dir, "controls", factor)
            if cons is not None:
                try:
                    df = pd.read_csv(cons, index_col=0)
                    if roi_name not in df.columns:
                        return None
                    available = set(df.index.astype(str))
                    subject_set = {str(s) for s in subjects}
                    matching = available.intersection(subject_set)
                    if not matching:
                        no_prefix = [
                            str(s).replace("sub-", "") if str(s).startswith("sub-") else str(s)
                            for s in subjects
                        ]
                        avail_no_prefix = {
                            str(s).replace("sub-", "") if str(s).startswith("sub-") else str(s)
                            for s in available
                        }
                        match_no = set(no_prefix).intersection(avail_no_prefix)
                        if match_no:
                            matching = {
                                s
                                for s in available
                                if (
                                    str(s).replace("sub-", "")
                                    if str(s).startswith("sub-")
                                    else str(s)
                                )
                                in match_no
                            }
                    if not matching:
                        return None
                    df = df.loc[df.index.astype(str).isin(matching)]
                    mean_val = df[roi_name].mean()
                    std_val = df[roi_name].std(ddof=0)
                    if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
                        return None
                    z = (df[roi_name] - mean_val) / std_val
                    return z.reindex(subjects)
                except Exception:
                    pass

            region_scores_dir = ospj(controls_dir, "gm_regions")
            csv_path = ospj(region_scores_dir, f"{roi_name}_factor_scores.csv")
            if not os.path.exists(csv_path):
                tract_scores_dir = ospj(controls_dir, "wm_tracts")
                csv_path = ospj(tract_scores_dir, f"{roi_name}_factor_scores.csv")
                if not os.path.exists(csv_path):
                    return None
            try:
                df = pd.read_csv(csv_path, index_col=0)
                factor_col = factor if factor in df.columns else None
                if factor_col is None and factor.startswith("F") and len(factor) > 1:
                    alt = f"Factor{factor[1:]}"
                    if alt in df.columns:
                        factor_col = alt
                if factor_col is None:
                    return None
                # Filter to requested subjects (match with/without sub- prefix)
                available = set(df.index)
                subject_set = set(subjects)
                matching = available.intersection(subject_set)
                if not matching:
                    no_prefix = [s.replace("sub-", "") if s.startswith("sub-") else s for s in subjects]
                    avail_no_prefix = {s.replace("sub-", "") if s.startswith("sub-") else s for s in available}
                    match_no = set(no_prefix).intersection(avail_no_prefix)
                    if match_no:
                        matching = {s for s in available
                                    if (s.replace("sub-", "") if s.startswith("sub-") else s) in match_no}
                if not matching:
                    return None
                df = df.loc[df.index.isin(matching)]
                mean_val = df[factor_col].mean()
                std_val = df[factor_col].std(ddof=0)
                if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
                    return None
                z = (df[factor_col] - mean_val) / std_val
                return z.reindex(subjects)
            except Exception:
                return None

        for factor in factors:
            columns_list = []
            if not region_all.empty and factor in region_all.columns:
                for region in region_all.index:
                    z_series = _load_control_z_scores_for_roi_factor(
                        controls_factor_scores_dir, region, factor, controls_index
                    )
                    if z_series is not None:
                        columns_list.append((region, z_series))
            if not tract_segment_all.empty and factor in tract_segment_all.columns:
                for (tract, segment), row in tract_segment_all.iterrows():
                    end1_label = tract_to_end1.get(tract, "end1")
                    end2_label = tract_to_end2.get(tract, "end2")
                    segment_to_label = {
                        'end1': f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
                        'core': 'core',
                        'end2': f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
                    }
                    segment_label = segment_to_label.get(segment, segment)
                    roi_name = f"{tract}_{segment_label}"
                    if tract.endswith("_L"):
                        tract_base = tract[:-2]
                        hemi = "L"
                    elif tract.endswith("_R"):
                        tract_base = tract[:-2]
                        hemi = "R"
                    else:
                        tract_base = tract
                        hemi = ""
                    roi_key = f"{tract_base}_{hemi}_{segment_label}" if hemi else f"{tract_base}_{segment_label}"
                    z_series = _load_control_z_scores_for_roi_factor(
                        controls_factor_scores_dir, roi_key, factor, controls_index
                    )
                    if z_series is not None:
                        columns_list.append((roi_key, z_series))
            if columns_list:
                controls_df = pd.concat([s.reindex(controls_index) for _, s in columns_list], axis=1, copy=False)
                controls_df.columns = [name for name, _ in columns_list]
                controls_df.index.name = 'subject'
                # Insert group column next to subject (penn_controls, hcpya, hcpaging)
                group_series = pd.Series(
                    [get_group_from_subject_id(s) for s in controls_df.index],
                    index=controls_df.index,
                    name='group',
                )
                controls_df = pd.concat([group_series, controls_df], axis=1)
                out_path = ospj(factor_z_scores_output_dir, f"controls_{factor}_z_scores.csv")
                controls_df.to_csv(out_path)
                print(f"  Saved controls {factor} z-scores (with group) to {out_path}")

    # Function to save colorbar as separate PNG
    def save_colorbar_png(vmin, vmax, output_path, use_absolute=False, is_controls=False, label=None):
        """Save a horizontal colorbar as a separate PNG file."""
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        fig = plt.figure(figsize=(8, 0.5))
        fig.patch.set_facecolor('white')
        
        # Determine colorbar range
        if vmin is not None and vmax is not None:
            if use_absolute or is_controls:
                # For absolute values or controls, always start at 0
                vmin_symmetric = 0
                vmax_symmetric = abs(vmax)  # vmax is already the max absolute value when use_absolute=True
            else:
                abs_max = max(abs(vmin), abs(vmax))
                vmin_symmetric = -abs_max
                vmax_symmetric = abs_max
        else:
            if use_absolute or is_controls:
                vmin_symmetric = 0
                vmax_symmetric = 1
            else:
                vmin_symmetric = -1
                vmax_symmetric = 1
        
        # Create ScalarMappable for colorbar
        # Use Reds colormap for absolute values or controls (lower limit 0), RdBu_r for raw values
        cmap_to_use = 'Reds' if (use_absolute or is_controls) else 'RdBu_r'
        norm = Normalize(vmin=vmin_symmetric, vmax=vmax_symmetric)
        sm = ScalarMappable(cmap=cmap_to_use, norm=norm)
        sm.set_array([])
        
        # Create colorbar axis
        cbar_ax = fig.add_axes([0.1, 0.3, 0.8, 0.4])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        # Set label - use provided label, or fall back to default based on is_controls
        if label is not None:
            cbar_label = label
        elif is_controls:
            cbar_label = "Factor scores"
        else:
            cbar_label = "Factor z-score"
        cbar.set_label(cbar_label, fontsize=14, fontweight='bold', fontfamily='serif')
        cbar.ax.tick_params(labelsize=12)
        # Set tick labels font family
        for tick_label in cbar.ax.get_xticklabels():
            tick_label.set_fontfamily('serif')
        cbar.ax.xaxis.label.set_fontfamily('serif')
        
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor='white')
        plt.close(fig)
    
    # Function to save colorbar with 0-1 range and "min"/"max" labels
    def save_colorbar_01_png(output_path, label_text="Rescaled values"):
        """Save a horizontal colorbar with 0-1 range and 'min'/'max' tick labels."""
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        fig = plt.figure(figsize=(8, 0.5))
        fig.patch.set_facecolor('white')
        
        # Create ScalarMappable for colorbar with 0-1 range
        cmap_to_use = 'Reds'
        norm = Normalize(vmin=0.0, vmax=1.0)
        sm = ScalarMappable(cmap=cmap_to_use, norm=norm)
        sm.set_array([])
        
        # Create colorbar axis
        cbar_ax = fig.add_axes([0.1, 0.3, 0.8, 0.4])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(label_text, fontsize=14, fontweight='bold', fontfamily='serif')
        
        # Set custom tick labels: "min" at 0, "max" at 1
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(['min', 'max'])
        cbar.ax.tick_params(labelsize=12)
        
        # Set tick labels font family
        for label in cbar.ax.get_xticklabels():
            label.set_fontfamily('serif')
        cbar.ax.xaxis.label.set_fontfamily('serif')
        
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor='white')
        plt.close(fig)
    
    # Generate brain maps for GM regions and WM tracts by lateralization
    # Use signed z-scores (not absolute) for all factor-specific brain maps
    # Check cache for brain maps
    brain_maps_cache_path = ospj(brain_maps_dir, "brain_maps_cache.pkl")
    cached_brain_maps = None
    if os.path.exists(brain_maps_cache_path):
        print(f"Loading cached brain maps from {brain_maps_cache_path}")
        try:
            with open(brain_maps_cache_path, 'rb') as f:
                cached_brain_maps = pickle.load(f)
            print("Using cached brain maps - skipping generation")
            brain_map_paths = cached_brain_maps.get('brain_map_paths', {})
            colorbar_paths = cached_brain_maps.get('colorbar_paths', {})
        except Exception as e:
            print(f"Warning: Failed to load brain maps cache: {e}. Will regenerate.")
            cached_brain_maps = None
    
    if cached_brain_maps is None:
        brain_map_paths = {}  # (factor, lateralization, map_type) -> path
        colorbar_paths = {}  # (factor, lateralization) -> path
        # map_type: "ctx", "sctx", "assoc", "proj"
        for factor in tqdm(factors, desc="Brain maps"):
            # Use z-scores from left/right specific dictionaries if available, otherwise fall back to all
            # Left lateralized - compute min/max across left-specific data (GM + WM) for consistent colorbars
            left_all_scores = []
            left_gm_scores_dict = left_gm_factor_z_scores if (left_gm_factor_z_scores and factor in left_gm_factor_z_scores) else (all_gm_factor_z_scores if factor in all_gm_factor_z_scores else {})
            left_wm_scores_dict = left_factor_z_scores if (left_factor_z_scores and factor in left_factor_z_scores) else (all_factor_z_scores if factor in all_factor_z_scores else {})
            
            if left_gm_scores_dict and factor in left_gm_scores_dict:
                left_scores = list(left_gm_scores_dict[factor].values())
                valid_scores = [s for s in left_scores if not pd.isna(s)]
                left_all_scores.extend(valid_scores)
            
            if left_wm_scores_dict and factor in left_wm_scores_dict:
                wm_scores = list(left_wm_scores_dict[factor].values())
                valid_wm_scores = [s for s in wm_scores if not pd.isna(s)]
                left_all_scores.extend(valid_wm_scores)
            
            # Compute colorbar range for left lateralized (signed z-scores, symmetric)
            if left_all_scores:
                abs_max = max([abs(s) for s in left_all_scores])
                left_vmin = -abs_max
                left_vmax = abs_max
            else:
                left_vmin = None
                left_vmax = None

            # Right lateralized - compute min/max across right-specific data (GM + WM) for consistent colorbars
            right_all_scores = []
            right_gm_scores_dict = right_gm_factor_z_scores if (right_gm_factor_z_scores and factor in right_gm_factor_z_scores) else (all_gm_factor_z_scores if factor in all_gm_factor_z_scores else {})
            right_wm_scores_dict = right_factor_z_scores if (right_factor_z_scores and factor in right_factor_z_scores) else (all_factor_z_scores if factor in all_factor_z_scores else {})

            if right_gm_scores_dict and factor in right_gm_scores_dict:
                right_scores = list(right_gm_scores_dict[factor].values())
                valid_scores = [s for s in right_scores if not pd.isna(s)]
                right_all_scores.extend(valid_scores)

            if right_wm_scores_dict and factor in right_wm_scores_dict:
                wm_scores = list(right_wm_scores_dict[factor].values())
                valid_wm_scores = [s for s in wm_scores if not pd.isna(s)]
                right_all_scores.extend(valid_wm_scores)

            # Compute colorbar range for right lateralized (signed z-scores, symmetric)
            if right_all_scores:
                abs_max = max([abs(s) for s in right_all_scores])
                right_vmin = -abs_max
                right_vmax = abs_max
            else:
                right_vmin = None
                right_vmax = None

            # Save colorbar for left lateralized (signed z-scores)
            if left_vmin is not None and left_vmax is not None:
                left_colorbar_path = ospj(brain_maps_dir, f"{factor}_left_lateralized_colorbar.png")
                save_colorbar_png(left_vmin, left_vmax, left_colorbar_path, use_absolute=False)
                colorbar_paths[(factor, "left")] = left_colorbar_path

            # Save colorbar for right lateralized (signed z-scores)
            if right_vmin is not None and right_vmax is not None:
                right_colorbar_path = ospj(brain_maps_dir, f"{factor}_right_lateralized_colorbar.png")
                save_colorbar_png(right_vmin, right_vmax, right_colorbar_path, use_absolute=False)
                colorbar_paths[(factor, "right")] = right_colorbar_path

            # Left lateralized - GM regions (use signed z-scores)
            # Use left-specific z-scores if available, otherwise fall back to all
            if left_gm_factor_z_scores and factor in left_gm_factor_z_scores and left_gm_factor_z_scores[factor]:
                region_scores_left = left_gm_factor_z_scores[factor]
                print(f"  Left TLE {factor}: Using left_gm_factor_z_scores with {len(region_scores_left)} regions")
            elif factor in all_gm_factor_z_scores and all_gm_factor_z_scores[factor]:
                region_scores_left = all_gm_factor_z_scores[factor]
                print(f"  Left TLE {factor}: Using all_gm_factor_z_scores with {len(region_scores_left)} regions (fallback)")
            else:
                region_scores_left = {}
                print(f"  Left TLE {factor}: WARNING - No GM z-scores found!")
        
            # Debug: Check z-score ranges
            if region_scores_left:
                left_scores_list = [s for s in region_scores_left.values() if not pd.isna(s)]
                if left_scores_list:
                    print(f"    Left TLE {factor} GM z-scores: min={min(left_scores_list):.3f}, max={max(left_scores_list):.3f}, has_negative={any(s < 0 for s in left_scores_list)}, count={len(left_scores_list)}")
        
            if region_scores_left:
                brain_map_path = ospj(brain_maps_dir, f"{factor}_left_lateralized_brain_map.png")
                # Always regenerate (no caching)
                create_brain_map(region_scores_left, factor, "left", brain_map_path, 
                               vmin=left_vmin, vmax=left_vmax, use_absolute=False)
                # Load the separate cortex and subcortex maps with y and lr views
                ctx_y_path = brain_map_path.replace('.png', '_ctx_y.png')
                ctx_lr_path = brain_map_path.replace('.png', '_ctx_lr.png')
                sctx_y_path = brain_map_path.replace('.png', '_sctx_y.png')
                sctx_lr_path = brain_map_path.replace('.png', '_sctx_lr.png')
                if os.path.exists(ctx_y_path):
                    brain_map_paths[(factor, "left", "ctx_y")] = ctx_y_path
                if os.path.exists(ctx_lr_path):
                    brain_map_paths[(factor, "left", "ctx_lr")] = ctx_lr_path
                if os.path.exists(sctx_y_path):
                    brain_map_paths[(factor, "left", "sctx_y")] = sctx_y_path
                if os.path.exists(sctx_lr_path):
                    brain_map_paths[(factor, "left", "sctx_lr")] = sctx_lr_path
        
            # Left lateralized - WM tracts (use signed z-scores)
            # Use left-specific z-scores if available, otherwise fall back to all
            left_wm_z_scores_dict = left_factor_z_scores if (left_factor_z_scores and factor in left_factor_z_scores) else (all_factor_z_scores if factor in all_factor_z_scores else {})
            if left_wm_z_scores_dict and factor in left_wm_z_scores_dict:
                # Convert roi_key format to (tract, segment) format for brain map
                tract_scores_dict = {}
                for roi_key, z_score in left_wm_z_scores_dict[factor].items():
                    if not pd.isna(z_score):
                        # Parse roi_key: "{tract_base}_{hemi}_{seg_label}" or "{tract_base}_{seg_label}"
                        parts = roi_key.rsplit("_", 2)
                        if len(parts) == 3:
                            tract_base, hemi, seg_label = parts
                            tract_name = f"{tract_base}_{hemi}"
                            # Map segment label back to internal segment name
                            end1_label = tract_to_end1.get(tract_name, "end1")
                            end2_label = tract_to_end2.get(tract_name, "end2")
                            segment_to_label = {
                                'end1': f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
                                'core': 'core',
                                'end2': f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
                            }
                            # Reverse lookup
                            segment = None
                            for seg, seg_lab in segment_to_label.items():
                                if seg_lab == seg_label:
                                    segment = seg
                                    break
                            if segment:
                                tract_scores_dict[(tract_name, segment)] = z_score
            
                if tract_scores_dict:
                    assoc_path = ospj(brain_maps_dir, f"{factor}_left_lateralized_wm_association.png")
                    proj_path = ospj(brain_maps_dir, f"{factor}_left_lateralized_wm_projection.png")
                    # Always regenerate (no caching)
                    create_wm_tract_brain_map(
                        tract_scores_dict, factor, "left",
                        assoc_path, proj_path,
                        tract_metadata_df, use_absolute=False,
                        vmin=left_vmin, vmax=left_vmax
                    )
                    # Load the separate y and lr views for association and projection tracts
                    assoc_y_path = assoc_path.replace('.png', '_y.png')
                    assoc_lr_path = assoc_path.replace('.png', '_lr.png')
                    proj_y_path = proj_path.replace('.png', '_y.png')
                    proj_lr_path = proj_path.replace('.png', '_lr.png')
                    if os.path.exists(assoc_y_path):
                        brain_map_paths[(factor, "left", "assoc_y")] = assoc_y_path
                    if os.path.exists(assoc_lr_path):
                        brain_map_paths[(factor, "left", "assoc_lr")] = assoc_lr_path
                    if os.path.exists(proj_y_path):
                        brain_map_paths[(factor, "left", "proj_y")] = proj_y_path
                    if os.path.exists(proj_lr_path):
                        brain_map_paths[(factor, "left", "proj_lr")] = proj_lr_path
        
            # Right lateralized - GM regions (use signed z-scores)
            # Use right-specific z-scores if available, otherwise fall back to all
            if right_gm_factor_z_scores and factor in right_gm_factor_z_scores and right_gm_factor_z_scores[factor]:
                region_scores_right = right_gm_factor_z_scores[factor]
                print(f"  Right TLE {factor}: Using right_gm_factor_z_scores with {len(region_scores_right)} regions")
            elif factor in all_gm_factor_z_scores and all_gm_factor_z_scores[factor]:
                region_scores_right = all_gm_factor_z_scores[factor]
                print(f"  Right TLE {factor}: Using all_gm_factor_z_scores with {len(region_scores_right)} regions (fallback)")
            else:
                region_scores_right = {}
                print(f"  Right TLE {factor}: WARNING - No GM z-scores found!")
        
            # Debug: Check z-score ranges
            if region_scores_right:
                right_scores_list = [s for s in region_scores_right.values() if not pd.isna(s)]
                if right_scores_list:
                    print(f"    Right TLE {factor} GM z-scores: min={min(right_scores_list):.3f}, max={max(right_scores_list):.3f}, has_negative={any(s < 0 for s in right_scores_list)}, count={len(right_scores_list)}")
        
            if region_scores_right:
                brain_map_path = ospj(brain_maps_dir, f"{factor}_right_lateralized_brain_map.png")
                # Always regenerate (no caching)
                create_brain_map(region_scores_right, factor, "right", brain_map_path,
                               vmin=right_vmin, vmax=right_vmax, use_absolute=False)
                # Load the separate cortex and subcortex maps with y and lr views
                ctx_y_path = brain_map_path.replace('.png', '_ctx_y.png')
                ctx_lr_path = brain_map_path.replace('.png', '_ctx_lr.png')
                sctx_y_path = brain_map_path.replace('.png', '_sctx_y.png')
                sctx_lr_path = brain_map_path.replace('.png', '_sctx_lr.png')
                if os.path.exists(ctx_y_path):
                    brain_map_paths[(factor, "right", "ctx_y")] = ctx_y_path
                if os.path.exists(ctx_lr_path):
                    brain_map_paths[(factor, "right", "ctx_lr")] = ctx_lr_path
                if os.path.exists(sctx_y_path):
                    brain_map_paths[(factor, "right", "sctx_y")] = sctx_y_path
                if os.path.exists(sctx_lr_path):
                    brain_map_paths[(factor, "right", "sctx_lr")] = sctx_lr_path
        
            # Right lateralized - WM tracts (use signed z-scores)
            # Use right-specific z-scores if available, otherwise fall back to all
            right_wm_z_scores_dict = right_factor_z_scores if (right_factor_z_scores and factor in right_factor_z_scores) else (all_factor_z_scores if factor in all_factor_z_scores else {})
            if right_wm_z_scores_dict and factor in right_wm_z_scores_dict:
                # Convert roi_key format to (tract, segment) format for brain map
                tract_scores_dict = {}
                for roi_key, z_score in right_wm_z_scores_dict[factor].items():
                    if not pd.isna(z_score):
                        # Parse roi_key: "{tract_base}_{hemi}_{seg_label}" or "{tract_base}_{seg_label}"
                        parts = roi_key.rsplit("_", 2)
                        if len(parts) == 3:
                            tract_base, hemi, seg_label = parts
                            tract_name = f"{tract_base}_{hemi}"
                            # Map segment label back to internal segment name
                            end1_label = tract_to_end1.get(tract_name, "end1")
                            end2_label = tract_to_end2.get(tract_name, "end2")
                            segment_to_label = {
                                'end1': f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
                                'core': 'core',
                                'end2': f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
                            }
                            # Reverse lookup
                            segment = None
                            for seg, seg_lab in segment_to_label.items():
                                if seg_lab == seg_label:
                                    segment = seg
                                    break
                            if segment:
                                tract_scores_dict[(tract_name, segment)] = z_score
            
                if tract_scores_dict:
                    assoc_path = ospj(brain_maps_dir, f"{factor}_right_lateralized_wm_association.png")
                    proj_path = ospj(brain_maps_dir, f"{factor}_right_lateralized_wm_projection.png")
                    # Always regenerate (no caching)
                    create_wm_tract_brain_map(
                        tract_scores_dict, factor, "right",
                        assoc_path, proj_path,
                        tract_metadata_df, use_absolute=False,
                        vmin=right_vmin, vmax=right_vmax
                    )
                    # Load the separate y and lr views for association and projection tracts
                    assoc_y_path = assoc_path.replace('.png', '_y.png')
                    assoc_lr_path = assoc_path.replace('.png', '_lr.png')
                    proj_y_path = proj_path.replace('.png', '_y.png')
                    proj_lr_path = proj_path.replace('.png', '_lr.png')
                    if os.path.exists(assoc_y_path):
                        brain_map_paths[(factor, "right", "assoc_y")] = assoc_y_path
                    if os.path.exists(assoc_lr_path):
                        brain_map_paths[(factor, "right", "assoc_lr")] = assoc_lr_path
                    if os.path.exists(proj_y_path):
                        brain_map_paths[(factor, "right", "proj_y")] = proj_y_path
                    if os.path.exists(proj_lr_path):
                        brain_map_paths[(factor, "right", "proj_lr")] = proj_lr_path
        
            # Controls - generate for both abs and raw reports, using roi_factor_scores
            if control_subjects and (not region_controls.empty or not tract_segment_controls.empty):
                # Compute colorbar range for controls (from roi_factor_scores - can be signed)
                control_all_scores = []
                if not region_controls.empty and factor in region_controls.columns:
                    control_scores = region_controls[factor].values
                    valid_scores = [s for s in control_scores if not pd.isna(s)]
                    control_all_scores.extend(valid_scores)
            
                if not tract_segment_controls.empty and factor in tract_segment_controls.columns:
                    wm_scores = tract_segment_controls[factor].values
                    valid_wm_scores = [s for s in wm_scores if not pd.isna(s)]
                    control_all_scores.extend(valid_wm_scores)
            
                # Rescale control factor scores to 0-1 range for display
                # Compute min and max for rescaling
                if control_all_scores:
                    control_min = min(control_all_scores)
                    control_max = max(control_all_scores)
                    control_range = control_max - control_min
                else:
                    control_min = None
                    control_max = None
                    control_range = None
            
                # Save colorbar for controls - rescale to 0-1 with "min"/"max" labels
                if control_min is not None and control_max is not None:
                    suffix = "_raw"  # Only using signed z-scores, not absolute
                    control_colorbar_path = ospj(brain_maps_dir, f"{factor}_controls_colorbar{suffix}.png")
                    # Use the 0-1 colorbar function with "min"/"max" labels
                    save_colorbar_01_png(control_colorbar_path, label_text="Factor scores")
                    colorbar_paths[(factor, "controls")] = control_colorbar_path
            
                # Controls - GM regions (using roi_factor_scores, rescaled to 0-1)
                if not region_controls.empty and factor in region_controls.columns:
                    region_scores_controls = region_controls[factor].to_dict()
                
                    # Rescale to 0-1 if we have valid range
                    if control_range is not None and control_range > 0:
                        region_scores_controls_rescaled = {}
                        for region, score in region_scores_controls.items():
                            if not pd.isna(score):
                                rescaled_score = (score - control_min) / control_range
                                region_scores_controls_rescaled[region] = rescaled_score
                            else:
                                region_scores_controls_rescaled[region] = np.nan
                        region_scores_controls = region_scores_controls_rescaled
                
                    brain_map_path = ospj(brain_maps_dir, f"{factor}_controls_brain_map.png")
                    # Always regenerate (no caching)
                    # Use 0-1 range for rescaled data
                    create_brain_map(region_scores_controls, factor, "controls", brain_map_path,
                                   vmin=0.0, vmax=1.0, use_absolute=True)
                    # Load the separate cortex and subcortex maps with y and lr views
                    ctx_y_path = brain_map_path.replace('.png', '_ctx_y.png')
                    ctx_lr_path = brain_map_path.replace('.png', '_ctx_lr.png')
                    sctx_y_path = brain_map_path.replace('.png', '_sctx_y.png')
                    sctx_lr_path = brain_map_path.replace('.png', '_sctx_lr.png')
                    if os.path.exists(ctx_y_path):
                        brain_map_paths[(factor, "controls", "ctx_y")] = ctx_y_path
                    if os.path.exists(ctx_lr_path):
                        brain_map_paths[(factor, "controls", "ctx_lr")] = ctx_lr_path
                    if os.path.exists(sctx_y_path):
                        brain_map_paths[(factor, "controls", "sctx_y")] = sctx_y_path
                    if os.path.exists(sctx_lr_path):
                        brain_map_paths[(factor, "controls", "sctx_lr")] = sctx_lr_path
            
                # Controls - WM tracts (using roi_factor_scores, rescaled to 0-1)
                if not tract_segment_controls.empty and factor in tract_segment_controls.columns:
                    # Convert DataFrame to dict of (tract, segment) -> score
                    tract_scores_dict = {}
                    for (tract, segment), row in tract_segment_controls.iterrows():
                        score = row[factor]
                        if not pd.isna(score):
                            tract_scores_dict[(tract, segment)] = score
                
                    # Rescale to 0-1 if we have valid range
                    if control_range is not None and control_range > 0:
                        tract_scores_dict_rescaled = {}
                        for (tract, segment), score in tract_scores_dict.items():
                            if not pd.isna(score):
                                rescaled_score = (score - control_min) / control_range
                                tract_scores_dict_rescaled[(tract, segment)] = rescaled_score
                            else:
                                tract_scores_dict_rescaled[(tract, segment)] = np.nan
                        tract_scores_dict = tract_scores_dict_rescaled
                
                    assoc_path = ospj(brain_maps_dir, f"{factor}_controls_wm_association.png")
                    proj_path = ospj(brain_maps_dir, f"{factor}_controls_wm_projection.png")
                    # Always regenerate (no caching)
                    # Use 0-1 range for rescaled data
                    create_wm_tract_brain_map(
                            tract_scores_dict, factor, "controls",
                            assoc_path, proj_path,
                            tract_metadata_df, use_absolute=True,
                            vmin=0.0, vmax=1.0
                        )
                    # Load the separate y and lr views for association and projection tracts
                    assoc_y_path = assoc_path.replace('.png', '_y.png')
                    assoc_lr_path = assoc_path.replace('.png', '_lr.png')
                    proj_y_path = proj_path.replace('.png', '_y.png')
                    proj_lr_path = proj_path.replace('.png', '_lr.png')
                    if os.path.exists(assoc_y_path):
                        brain_map_paths[(factor, "controls", "assoc_y")] = assoc_y_path
                    if os.path.exists(assoc_lr_path):
                        brain_map_paths[(factor, "controls", "assoc_lr")] = assoc_lr_path
                    if os.path.exists(proj_y_path):
                        brain_map_paths[(factor, "controls", "proj_y")] = proj_y_path
                    if os.path.exists(proj_lr_path):
                        brain_map_paths[(factor, "controls", "proj_lr")] = proj_lr_path
        
        # Save brain maps to cache (after all factors are processed)
        print(f"Saving brain maps to cache: {brain_maps_cache_path}")
        cache_data = {
            'brain_map_paths': brain_map_paths,
            'colorbar_paths': colorbar_paths
        }
        with open(brain_maps_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    else:
        # Use cached brain maps
        brain_map_paths = cached_brain_maps.get('brain_map_paths', {})
        colorbar_paths = cached_brain_maps.get('colorbar_paths', {})
    
    # Compute across-factor |z| scores and generate plots
    across_factor_wm_scores = {}
    across_factor_gm_scores = {}
    across_factor_brain_map_paths = {}
    across_factor_colorbar_path = None
    factor_total_z_barplot_path = None
    across_factor_abs_z_barplot_path = None  # Initialize early
    across_factor_raincloud_path = None
    
    if all_factor_z_scores and all_gm_factor_z_scores:
        print("Computing across-factor Mahalanobis distances...")
        across_factor_wm_scores, across_factor_gm_scores = compute_across_factor_abs_z_scores(
            all_factor_z_scores, all_gm_factor_z_scores, factors, controls_factor_scores_dir, control_subjects
        )
        
        # Compute across-factor Mahalanobis distances for left and right separately
        left_across_wm_scores = {}
        left_across_gm_scores = {}
        right_across_wm_scores = {}
        right_across_gm_scores = {}
        
        if left_factor_z_scores and left_gm_factor_z_scores:
            left_across_wm_scores, left_across_gm_scores = compute_across_factor_abs_z_scores(
                left_factor_z_scores, left_gm_factor_z_scores, factors, controls_factor_scores_dir, control_subjects
            )
        
        if right_factor_z_scores and right_gm_factor_z_scores:
            right_across_wm_scores, right_across_gm_scores = compute_across_factor_abs_z_scores(
                right_factor_z_scores, right_gm_factor_z_scores, factors, controls_factor_scores_dir, control_subjects
            )
        
        # Generate raincloud plot for signed z-scores per factor (with GM on left, WM on right flipped)
        plots_dir = ospj(os.path.dirname(output_path), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        factor_total_z_barplot_path = ospj(plots_dir, "factor_z_raincloud.png")
        # Always regenerate (no caching)
        create_factor_z_raincloud_plot(
            all_factor_z_scores, all_gm_factor_z_scores, factors, factor_total_z_barplot_path,
            left_factor_z_scores=left_factor_z_scores,
            left_gm_factor_z_scores=left_gm_factor_z_scores,
            right_factor_z_scores=right_factor_z_scores,
            right_gm_factor_z_scores=right_gm_factor_z_scores,
        )
        
        # Generate raincloud plot for absolute z-scores per factor (across factor |z|)
        across_factor_abs_z_barplot_path = ospj(plots_dir, "across_factor_abs_z_raincloud.png")
        # Always regenerate (no caching)
        create_across_factor_abs_z_raincloud_plot(
            all_factor_z_scores, all_gm_factor_z_scores, factors, across_factor_abs_z_barplot_path,
            left_factor_z_scores=left_factor_z_scores,
            left_gm_factor_z_scores=left_gm_factor_z_scores,
            right_factor_z_scores=right_factor_z_scores,
            right_gm_factor_z_scores=right_gm_factor_z_scores,
        )
        
        # Generate raincloud plot for across-factor |z|
        across_factor_raincloud_path = ospj(plots_dir, "across_factor_raincloud.png")
        # Always regenerate (no caching)
        create_across_factor_raincloud(
            across_factor_wm_scores, across_factor_gm_scores, across_factor_raincloud_path,
            left_wm_scores=left_across_wm_scores if left_across_wm_scores else None,
            left_gm_scores=left_across_gm_scores if left_across_gm_scores else None,
            right_wm_scores=right_across_wm_scores if right_across_wm_scores else None,
            right_gm_scores=right_across_gm_scores if right_across_gm_scores else None,
        )
        
        # Generate Factor 1 top 10 loadings plot (single figure for All TLE only)
        factor1_top10_path = ospj(plots_dir, "factor1_top10_loadings.png")
        create_factor1_top10_loadings_plot(
            all_regions, all_tracts, all_patients, left_lateralized, right_lateralized,
            factor1_top10_path
        )
        
        # Generate Factor 1 top 10 loadings plot (sorted by median absolute value)
        # DISABLED: No longer computing/plotting/embedding this plot
        # factor1_top10_sorted_path = ospj(plots_dir, "factor1_top10_loadings_sorted.png")
        # create_factor1_top10_loadings_plot_sorted(
        #     all_regions, all_tracts, all_patients, left_lateralized, right_lateralized,
        #     factor1_top10_sorted_path
        # )
        
        # Generate hemisphere mahalanobis plot (all patients)
        hemisphere_mahalanobis_path = ospj(plots_dir, "hemisphere_mahalanobis.png")
        # Use empty intervention dicts if not available (plot will still work, just without colored intervention points)
        gm_intervention_dict = {}
        tract_intervention_dict = {}
        create_hemisphere_mahalanobis_plot(
            across_factor_wm_scores, across_factor_gm_scores,
            gm_intervention_dict, tract_intervention_dict,
            hemisphere_mahalanobis_path
        )
        
        # Generate hemisphere mahalanobis plots for left and right TLE patients separately
        left_hemisphere_mahalanobis_path = None
        right_hemisphere_mahalanobis_path = None
        
        if left_across_wm_scores or left_across_gm_scores:
            left_hemisphere_mahalanobis_path = ospj(plots_dir, "hemisphere_mahalanobis_left.png")
            create_hemisphere_mahalanobis_plot(
                left_across_wm_scores if left_across_wm_scores else {},
                left_across_gm_scores if left_across_gm_scores else {},
                gm_intervention_dict, tract_intervention_dict,
                left_hemisphere_mahalanobis_path
            )
        
        if right_across_wm_scores or right_across_gm_scores:
            right_hemisphere_mahalanobis_path = ospj(plots_dir, "hemisphere_mahalanobis_right.png")
            create_hemisphere_mahalanobis_plot(
                right_across_wm_scores if right_across_wm_scores else {},
                right_across_gm_scores if right_across_gm_scores else {},
                gm_intervention_dict, tract_intervention_dict,
                right_hemisphere_mahalanobis_path
        )
        
        # Generate brain maps for across-factor |z|
        if across_factor_gm_scores or across_factor_wm_scores:
            # Compute colorbar range
            all_across_scores = []
            for score_dict in [across_factor_gm_scores, across_factor_wm_scores]:
                for score in score_dict.values():
                    if not pd.isna(score):
                        all_across_scores.append(score)
            
            if all_across_scores:
                vmin_across = 0
                vmax_across = max(all_across_scores)
                
                # Save colorbar
                across_factor_colorbar_path = ospj(brain_maps_dir, "across_factor_colorbar.png")
                save_colorbar_png(vmin_across, vmax_across, across_factor_colorbar_path, 
                                 use_absolute=True, label="Mahalanobis Distance Across Factors")
                
                # Generate GM brain map
                if across_factor_gm_scores:
                    gm_brain_map_path = ospj(brain_maps_dir, "across_factor_gm_brain_map.png")
                    # Always regenerate (no caching)
                    create_brain_map(
                        across_factor_gm_scores,
                        "Across Factor Mahalanobis Distance",
                        "all",
                        gm_brain_map_path,
                        vmin=vmin_across,
                        vmax=vmax_across,
                        use_absolute=True
                    )
                    across_factor_brain_map_paths['ctx_y'] = gm_brain_map_path.replace('.png', '_ctx_y.png')
                    across_factor_brain_map_paths['ctx_lr'] = gm_brain_map_path.replace('.png', '_ctx_lr.png')
                    across_factor_brain_map_paths['sctx_y'] = gm_brain_map_path.replace('.png', '_sctx_y.png')
                    across_factor_brain_map_paths['sctx_lr'] = gm_brain_map_path.replace('.png', '_sctx_lr.png')
                
                # Generate WM brain maps
                if across_factor_wm_scores:
                    # Convert across_factor_wm_scores to (tract_name, segment) format
                    tract_scores_dict = {}
                    for roi_key, score in across_factor_wm_scores.items():
                        if not pd.isna(score):
                            parts = roi_key.rsplit("_", 2)
                            if len(parts) == 3:
                                tract_base, hemi, seg_label = parts
                                tract_name = f"{tract_base}_{hemi}"
                                end1_label = tract_to_end1.get(tract_name, "end1")
                                end2_label = tract_to_end2.get(tract_name, "end2")
                                segment_to_label = {
                                    'end1': f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
                                    'core': 'core',
                                    'end2': f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
                                }
                                segment = None
                                for seg, seg_lab in segment_to_label.items():
                                    if seg_lab == seg_label:
                                        segment = seg
                                        break
                                if segment:
                                    tract_scores_dict[(tract_name, segment)] = score
                    
                    if tract_scores_dict:
                        assoc_path = ospj(brain_maps_dir, "across_factor_wm_association.png")
                        proj_path = ospj(brain_maps_dir, "across_factor_wm_projection.png")
                        # Always regenerate (no caching)
                        create_wm_tract_brain_map(
                            tract_scores_dict,
                            "Across Factor Mahalanobis Distance",
                            "all",  # All patients combined (not separated by lateralization)
                            assoc_path,
                            proj_path,
                            tract_metadata_df,
                            use_absolute=True,
                            vmin=vmin_across,
                            vmax=vmax_across
                        )
                        across_factor_brain_map_paths['assoc_y'] = assoc_path.replace('.png', '_y.png')
                        across_factor_brain_map_paths['assoc_lr'] = assoc_path.replace('.png', '_lr.png')
                        across_factor_brain_map_paths['proj_y'] = proj_path.replace('.png', '_y.png')
                        across_factor_brain_map_paths['proj_lr'] = proj_path.replace('.png', '_lr.png')
    
    # Get tract segment label mappings for HTML generation
    tract_to_end1 = {}
    tract_to_end2 = {}
    if not tract_metadata_df.empty and "label" in tract_metadata_df.columns:
        if "end1" in tract_metadata_df.columns:
            tract_to_end1 = dict(zip(tract_metadata_df["label"], tract_metadata_df["end1"]))
        if "end2" in tract_metadata_df.columns:
            tract_to_end2 = dict(zip(tract_metadata_df["label"], tract_metadata_df["end2"]))
    
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format subject counts with demographics breakdown
    subject_counts_html = ""
    if subject_counts:
        # Get control group demographics if control_subjects are provided
        control_demographics = {}
        if control_subjects:
            control_demographics = get_control_group_demographics(control_subjects)
        
        subject_counts_html = f"""
        <p><strong>Subject Counts:</strong></p>
        <ul>
            <li>Epilepsy: {subject_counts.get('epilepsy', 0)}</li>
        </ul>
        <p><strong>Control Groups Breakdown:</strong></p>
        <table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
            <thead>
                <tr style="background-color: #4C72B0; color: white;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Group</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">N</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Age Mean (years)</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Age Range (years)</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">N Female</th>
                </tr>
            </thead>
            <tbody>"""
        
        # Add rows for each control group
        for group_name, display_name in [("penn_controls", "Penn Controls"), ("hcpya", "HCP-YA"), ("hcpaging", "HCP-Aging")]:
            n = subject_counts.get(group_name, 0)
            demo = control_demographics.get(group_name, {})
            
            age_mean_str = f"{demo.get('age_mean', 0):.1f}" if demo.get('age_mean') is not None else "N/A"
            age_min = demo.get('age_min')
            age_max = demo.get('age_max')
            if age_min is not None and age_max is not None:
                age_range_str = f"{age_min:.1f} - {age_max:.1f}"
            elif age_min is not None:
                age_range_str = f"{age_min:.1f} - N/A"
            elif age_max is not None:
                age_range_str = f"N/A - {age_max:.1f}"
            else:
                age_range_str = "N/A"
            
            n_female = demo.get('n_female', 0)
            
            subject_counts_html += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>{display_name}</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{n}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{age_mean_str}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{age_range_str}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{n_female}</td>
                </tr>"""
        
        subject_counts_html += """
            </tbody>
        </table>
        <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
            <em>Note: Control counts reflect all subjects with GAM rows in normative groups (hcpya, hcpaging, penn_controls).</em>
        </p>
"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Master Factor Score Report - {report_type}</title>
    <style>
        body {{
            font-family: Georgia, serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
            width: 100%;
            box-sizing: border-box;
        }}
        .section {{
            width: 100%;
            box-sizing: border-box;
            overflow-x: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
            width: 100%;
            box-sizing: border-box;
            overflow-x: hidden;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        .section h3 {{
            color: #667eea;
            font-size: 1.4em;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
        }}
        .summary-card h3 {{
            color: #667eea;
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .bar-chart {{
            margin: 0;
            padding: 0;
        }}
        .bar-item {{
            display: flex;
            align-items: center;
            margin-bottom: 2px;
            padding: 1px 0;
        }}
        .bar-label {{
            width: 300px;
            max-width: 300px;
            font-size: 0.9em;
            color: #333;
            margin-right: 10px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .bar-container {{
            flex: 1;
            height: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            position: relative;
            margin-right: 10px;
        }}
        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
        .bar-count {{
            font-weight: 600;
            color: #667eea;
            font-size: 0.9em;
            min-width: 30px;
            text-align: right;
        }}
        .brain-map-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .brain-map-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .brain-map-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .brain-map-column {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        .brain-map-column h4 {{
            text-align: center;
            margin: 10px 0;
            color: #333;
            font-size: 1.1em;
            font-weight: bold;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .brain-map-item {{
            text-align: center;
        }}
        .brain-map-item p {{
            font-size: 0.9em;
            margin: 3px 0;
            color: #333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: normal;
        }}
        .brain-map-item img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }}
        .factor-section {{
            margin: 30px 0;
        }}
        .footer {{
            background-color: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e9ecef;
        }}
        .timestamp {{
            font-size: 0.9em;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
    <h1>Master Factor Score Report</h1>
            <p>{report_type} Values</p>
        </div>
        
        <div class="content">
    <div class="section">
        <h2>Table of Contents</h2>
        <ul style="list-style-type: none; padding-left: 0;">
            <li style="margin: 10px 0;"><a href="#analysis-summary" style="color: #667eea; text-decoration: none; font-weight: 500;">Analysis Summary</a></li>
            <li style="margin: 10px 0;"><a href="#across-factor-summary" style="color: #667eea; text-decoration: none; font-weight: 500;">Microstructural Factor Abnormality Summary</a></li>
            <li style="margin: 10px 0;"><a href="#left-tle" style="color: #667eea; text-decoration: none; font-weight: 500;">Left TLE</a></li>
            <li style="margin: 10px 0;"><a href="#right-tle" style="color: #667eea; text-decoration: none; font-weight: 500;">Right TLE</a></li>
            <li style="margin: 10px 0;"><a href="#controls" style="color: #667eea; text-decoration: none; font-weight: 500;">Controls</a></li>
    </div>
    
    <div class="section" id="analysis-summary">
                <h2>Analysis Summary</h2>
        <p><strong>Report generated:</strong> {timestamp_str}</p>
        <p><strong>Total GM Regions:</strong> {len(all_regions)}</p>
        <p><strong>Total WM Tracts:</strong> {len(all_tracts)}</p>
                <p><strong>Left Lateralized Patients:</strong> {len(left_lateralized)}</p>
                <p><strong>Right Lateralized Patients:</strong> {len(right_lateralized)}</p>
                {subject_counts_html}
    </div>
"""
    
    # Helper function to convert image to base64
    def image_to_base64(image_path: str) -> str | None:
        if not image_path or not os.path.exists(image_path):
            return None
        with open(image_path, "rb") as img_f:
            img_data = img_f.read()
        img_base64 = b64encode(img_data).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".png":
            mime = "image/png"
        elif ext in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        else:
            mime = "image/png"
        return f"data:{mime};base64,{img_base64}"
    
    # Quality check: Generate glass brain plots for 3 randomly selected statistics
    # (Functionality kept but not included in HTML - DISABLED: not running generation)
    # if not roi_means_rescaled_gm.empty and not roi_means_rescaled_wm.empty:
    #     print("Generating quality check glass brain plots for roi_means_rescaled...")
    #     # Get available scalars
    #     available_scalars = list(set(roi_means_rescaled_gm.columns) & set(roi_means_rescaled_wm.columns))
    #     print(f"  Found {len(available_scalars)} available scalars")
    #     # Randomly select 3 scalars
    #     if len(available_scalars) >= 3:
    #         selected_scalars = random.sample(available_scalars, 3)
    #     else:
    #         selected_scalars = available_scalars
    #     print(f"  Selected {len(selected_scalars)} scalars for quality check: {selected_scalars}")
    #     
    #     # Generate quality check plots but don't include in HTML
    #     if selected_scalars:
    #         
    #         for scalar in selected_scalars:
    #             print(f"    Generating glass brain plots for {scalar}...")
    #             # Create region scores dict from roi_means_rescaled_gm for this scalar
    #             region_scores_dict = {}
    #             for region in roi_means_rescaled_gm.index:
    #                 if scalar in roi_means_rescaled_gm.columns:
    #                     value = roi_means_rescaled_gm.loc[region, scalar]
    #                     if not pd.isna(value):
    #                         region_scores_dict[region] = float(value)
    #             
    #             if region_scores_dict:
    #                 # Generate GM glass brain plot
    #                 gm_lr_path = ospj(brain_maps_dir, f"quality_check_gm_{scalar}_lr.png")
    #                 gm_y_path = ospj(brain_maps_dir, f"quality_check_gm_{scalar}_y.png")
    #                 # Use create_brain_map to generate glass brain plots
    #                 # roi_means_rescaled is 0-1, so vmin=0, vmax=1, use_absolute=True for positive-only
    #                 create_brain_map(region_scores_dict, scalar, "controls", gm_lr_path,
    #                                vmin=0.0, vmax=1.0, use_absolute=True)
    #                 # Also create y view
    #                 create_brain_map(region_scores_dict, scalar, "controls", gm_y_path,
    #                                vmin=0.0, vmax=1.0, use_absolute=True)
    #             
    #             # Create tract segment scores dict from roi_means_rescaled_wm for this scalar
    #             tract_segment_scores_dict = {}
    #             for (tract, segment) in roi_means_rescaled_wm.index:
    #                 if scalar in roi_means_rescaled_wm.columns:
    #                     value = roi_means_rescaled_wm.loc[(tract, segment), scalar]
    #                     if not pd.isna(value):
    #                         tract_segment_scores_dict[(tract, segment)] = float(value)
    #             
    #             if tract_segment_scores_dict:
    #                 # Generate WM glass brain plots
    #                 wm_assoc_path = ospj(brain_maps_dir, f"quality_check_wm_{scalar}_assoc.png")
    #                 wm_proj_path = ospj(brain_maps_dir, f"quality_check_wm_{scalar}_proj.png")
    #                 create_wm_tract_brain_map(
    #                     tract_segment_scores_dict, scalar, "controls",
    #                     wm_assoc_path, wm_proj_path,
    #                     tract_metadata_df, use_absolute=True,
    #                     vmin=0.0, vmax=1.0
    #                 )
    #             
    #             # Quality check plots generated but not included in HTML
    #             # (Functionality kept for potential future use)
    #         
    #         print("  Completed quality check glass brain plots")
    #     else:
    #         print(f"  Warning: Not enough available scalars for quality check (need at least 1, found {len(available_scalars)})")
    
    # Add Across Factor Summary section
    across_factor_section = ""
    # Note: across_factor_abs_z_barplot_path is already set earlier if plots were generated
    # Don't reset it here - it will be None if plots weren't generated
    if across_factor_gm_scores or across_factor_wm_scores:
        # Helper function to convert image to base64
        def image_to_base64(image_path: str) -> str | None:
            if not image_path or not os.path.exists(image_path):
                return None
            with open(image_path, "rb") as img_f:
                img_data = img_f.read()
            img_base64 = b64encode(img_data).decode("utf-8")
            ext = os.path.splitext(image_path)[1].lower()
            if ext == ".png":
                mime = "image/png"
            elif ext in [".jpg", ".jpeg"]:
                mime = "image/jpeg"
            else:
                mime = "image/png"
            return f"data:{mime};base64,{img_base64}"
        
        # Get brain map images
        ctx_y = image_to_base64(across_factor_brain_map_paths.get('ctx_y')) if 'ctx_y' in across_factor_brain_map_paths else None
        sctx_y = image_to_base64(across_factor_brain_map_paths.get('sctx_y')) if 'sctx_y' in across_factor_brain_map_paths else None
        ctx_lr = image_to_base64(across_factor_brain_map_paths.get('ctx_lr')) if 'ctx_lr' in across_factor_brain_map_paths else None
        sctx_lr = image_to_base64(across_factor_brain_map_paths.get('sctx_lr')) if 'sctx_lr' in across_factor_brain_map_paths else None
        assoc_y = image_to_base64(across_factor_brain_map_paths.get('assoc_y')) if 'assoc_y' in across_factor_brain_map_paths else None
        proj_y = image_to_base64(across_factor_brain_map_paths.get('proj_y')) if 'proj_y' in across_factor_brain_map_paths else None
        assoc_lr = image_to_base64(across_factor_brain_map_paths.get('assoc_lr')) if 'assoc_lr' in across_factor_brain_map_paths else None
        proj_lr = image_to_base64(across_factor_brain_map_paths.get('proj_lr')) if 'proj_lr' in across_factor_brain_map_paths else None
        
        # Build brain map HTML
        brain_map_html = ""
        if ctx_y or sctx_y or ctx_lr or sctx_lr or assoc_y or proj_y or assoc_lr or proj_lr:
            brain_map_html = f"""
                <!-- Brain maps side-by-side: Coronal (Y) on left, Lateral (LR) on right -->
                <div style="display: flex; gap: 30px; margin: 30px 0; width: 100%; box-sizing: border-box;">
                    <!-- Coronal View (Y): 2x2 grid -->
                    <div style="flex: 1; min-width: 0; box-sizing: border-box;">
                        <h4 style="text-align: center; margin: 0;">Coronal View</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0; margin: 0;">
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Cortex</p>
                                {f'<img src="{ctx_y}" alt="Cortex Y" style="max-width: 100%; border: 1px solid #ddd;">' if ctx_y else '<p>No data</p>'}
                            </div>
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Association</p>
                                {f'<img src="{assoc_y}" alt="Association Y" style="max-width: 100%; border: 1px solid #ddd;">' if assoc_y else '<p>No data</p>'}
                            </div>
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Subcortex</p>
                                {f'<img src="{sctx_y}" alt="Subcortex Y" style="max-width: 100%; border: 1px solid #ddd;">' if sctx_y else '<p>No data</p>'}
                            </div>
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Projection</p>
                                {f'<img src="{proj_y}" alt="Projection Y" style="max-width: 100%; border: 1px solid #ddd;">' if proj_y else '<p>No data</p>'}
                            </div>
                        </div>
                    </div>
                    
                    <!-- Lateral View (LR): 2x2 grid -->
                    <div style="flex: 1; min-width: 0; box-sizing: border-box;">
                        <h4 style="text-align: center; margin: 0;">Lateral Views</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0; margin: 0;">
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Cortex</p>
                                {f'<img src="{ctx_lr}" alt="Cortex LR" style="max-width: 100%; border: 1px solid #ddd;">' if ctx_lr else '<p>No data</p>'}
                            </div>
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Association</p>
                                {f'<img src="{assoc_lr}" alt="Association LR" style="max-width: 100%; border: 1px solid #ddd;">' if assoc_lr else '<p>No data</p>'}
                            </div>
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Subcortex</p>
                                {f'<img src="{sctx_lr}" alt="Subcortex LR" style="max-width: 100%; border: 1px solid #ddd;">' if sctx_lr else '<p>No data</p>'}
                            </div>
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Projection</p>
                                {f'<img src="{proj_lr}" alt="Projection LR" style="max-width: 100%; border: 1px solid #ddd;">' if proj_lr else '<p>No data</p>'}
                            </div>
                        </div>
                    </div>
                </div>
"""
        
        # Generate summaries for across-factor |z|
        # Note: generate_abnormality_summary_html expects (tract_scores_dict, gm_scores_dict) but we have separate dicts
        # We need to call it correctly - it returns a single HTML string, not a tuple
        # Let's create separate calls for tract and GM
        across_factor_tract_summary_html = generate_abnormality_summary_html(
            across_factor_wm_scores, top_n=10, is_tract=True,
            tract_label_to_name=tract_label_to_name,
            tract_to_end1=tract_to_end1, tract_to_end2=tract_to_end2
        )
        across_factor_gm_summary_html = generate_abnormality_summary_html(
            across_factor_gm_scores, top_n=10, is_tract=False
        )
        
        # Get plot images
        factor_total_z_barplot_img = image_to_base64(factor_total_z_barplot_path) if factor_total_z_barplot_path and os.path.exists(factor_total_z_barplot_path) else None
        across_factor_abs_z_barplot_img = image_to_base64(across_factor_abs_z_barplot_path) if across_factor_abs_z_barplot_path and os.path.exists(across_factor_abs_z_barplot_path) else None
        across_factor_raincloud_img = image_to_base64(across_factor_raincloud_path) if across_factor_raincloud_path and os.path.exists(across_factor_raincloud_path) else None
        hemisphere_mahalanobis_img = image_to_base64(hemisphere_mahalanobis_path) if hemisphere_mahalanobis_path and os.path.exists(hemisphere_mahalanobis_path) else None
        left_hemisphere_mahalanobis_img = image_to_base64(left_hemisphere_mahalanobis_path) if left_hemisphere_mahalanobis_path and os.path.exists(left_hemisphere_mahalanobis_path) else None
        right_hemisphere_mahalanobis_img = image_to_base64(right_hemisphere_mahalanobis_path) if right_hemisphere_mahalanobis_path and os.path.exists(right_hemisphere_mahalanobis_path) else None
        factor1_top10_img = image_to_base64(factor1_top10_path) if factor1_top10_path and os.path.exists(factor1_top10_path) else None
        across_factor_colorbar_img = image_to_base64(across_factor_colorbar_path) if across_factor_colorbar_path and os.path.exists(across_factor_colorbar_path) else None
        
        across_factor_section = f"""
        <div class="section" id="across-factor-summary">
            <h2>Microstructural Factor Abnormality Summary</h2>
            <p>This section shows factor-level summaries. The raincloud plots show the distribution of z-scores per factor, separated by tissue type (gray vs white matter). The brain maps show the Mahalanobis distance of factor z-scores from the control mean for each ROI, providing an overall measure of abnormality magnitude across all factors.</p>
            
            <div class="heatmap-container" style="margin: 30px 0;" id="across-factor-abs-z">
                <h3>Factor |z-scores|</h3>
                <p style="font-size: 0.9em; color: #666;">Distribution of absolute z-scores across all ROIs (gray and white matter combined) for each factor, shown as raincloud plots (density on left, mean indicated, points jittered to right). Shows overall magnitude of abnormalities regardless of direction.</p>
                {f'<img src="{across_factor_abs_z_barplot_img}" alt="Across Factor Mahalanobis Distance Raincloud" style="max-width: 100%; border: 1px solid #ddd;">' if across_factor_abs_z_barplot_img else '<p>Raincloud plot not available</p>'}
            </div>
            
            <div class="heatmap-container" style="margin: 30px 0;" id="distribution-z-scores">
                <h3>Factor z-scores by tissue type</h3>
                <p style="font-size: 0.9em; color: #666;">Distribution of signed z-scores across all ROIs for each factor, shown as raincloud plots with gray matter on the left and white matter on the right (flipped so density bases face each other). Positive values indicate positive deviations from controls, negative values indicate negative deviations.</p>
                {f'<img src="{factor_total_z_barplot_img}" alt="Factor Z-Score Raincloud" style="max-width: 100%; border: 1px solid #ddd;">' if factor_total_z_barplot_img else '<p>Raincloud plot not available</p>'}
            </div>
            
            <div class="heatmap-container" style="margin: 30px 0;" id="hemisphere-mahalanobis">
                <h3>Across Factor Mahalanobis Distances by Hemisphere</h3>
                <p style="font-size: 0.9em; color: #666;">Comparison of across-factor Mahalanobis distances between left and right hemispheres, shown separately for all regions, gray matter regions, and white matter tract segments.</p>
                {f'<img src="{hemisphere_mahalanobis_img}" alt="Hemisphere Mahalanobis Comparison" style="max-width: 100%; border: 1px solid #ddd;">' if hemisphere_mahalanobis_img else '<p>Hemisphere comparison plot not available</p>'}
            </div>
            
            <div class="heatmap-container" style="margin: 30px 0;" id="across-factor-raincloud">
                <h3>Across Factor Mahalanobis Distance: Gray vs White Matter</h3>
                <p style="font-size: 0.9em; color: #666;">Comparison of across-factor Mahalanobis distances grouped by gray and white matter.</p>
                {f'<img src="{across_factor_raincloud_img}" alt="Across Factor Raincloud" style="max-width: 100%; border: 1px solid #ddd;">' if across_factor_raincloud_img else '<p>Raincloud plot not available</p>'}
            </div>
            
            <div class="heatmap-container" style="margin: 30px 0;" id="factor1-top10-loadings">
                <h3>Abnormalities of statistics with high (>=0.8) "Overall diffusivity" loadings</h3>
                <p style="font-size: 0.9em; color: #666;">Region-level mean z-scores (signed, not absolute) for scalars with Factor 1 loading >= 0.8, shown for all temporal patients. Two subplots show Grey Matter and White Matter side-by-side. Statistics are sorted by |z-score| within each subplot, and median (signed) z-scores are labeled on the x-axis. X-axis labels are colored by reconstruction model.</p>
                {f'<img src="{factor1_top10_img}" alt="Factor 1 Top 10 Loadings" style="max-width: 100%; border: 1px solid #ddd;">' if factor1_top10_img else '<p>Factor 1 top 10 loadings plot not available</p>'}
            </div>
            
            
            <div style="margin: 30px 0;" id="across-factor-brain-maps">
                <h3>Across Factor Mahalanobis Distance Brain Maps</h3>
                <p style="font-size: 0.9em; color: #666;">Mahalanobis distance of factor z-scores from control mean for each ROI.</p>
                {brain_map_html}
                {f'<div style="text-align: center; margin: 20px 0;"><img src="{across_factor_colorbar_img}" alt="Across Factor Mahalanobis Distance Colorbar" style="max-width: 600px;"></div>' if across_factor_colorbar_img else ''}
            </div>
            
            <div class="summary-grid" style="margin: 30px 0;" id="most-abnormal-summaries">
                <div class="summary-card">
                    <h3>Most Abnormal Gray Matter Regions</h3>
                    <div class="bar-chart">
                        {across_factor_gm_summary_html}
            </div>
        </div>
                
                <div class="summary-card">
                    <h3>Most Abnormal White Matter Tracts</h3>
                    <div class="bar-chart">
                        {across_factor_tract_summary_html}
                    </div>
                </div>
            </div>
    </div>
"""
    
    html += across_factor_section
    
    # Restructure HTML: Group by Left TLE/Right TLE/Controls, then show F1-F3 within each
    # Define groups to process
    groups_to_process = []
    
    # Add Left and Right lateralized groups first
    groups_to_process.append(("Left TLE", "left", region_left, tract_segment_left, "Deviation"))
    groups_to_process.append(("Right TLE", "right", region_right, tract_segment_right, "Deviation"))
    
    # Add Controls group last
    if control_subjects and (not region_controls.empty or not tract_segment_controls.empty):
        groups_to_process.append(("Controls", "controls", region_controls, tract_segment_controls, "Expression"))
    
    # Process each group
    for group_name, group_key, region_scores_df, tract_segment_scores_df, map_type_label in groups_to_process:
        # Create section ID for TOC links
        section_id = group_key.replace(" ", "-").lower()
        if section_id == "left":
            section_id = "left-tle"
        elif section_id == "right":
            section_id = "right-tle"
        
        html += f"""
    <div class="section group-section" id="{section_id}">
        <h2>{group_name}</h2>
"""
        
        # Add "Across Factor Mahalanobis Distance Brain Maps" section for left/right TLE
        if group_key in ["left", "right"]:
            # Get across-factor scores for this group
            if group_key == "left" and (left_across_gm_scores or left_across_wm_scores):
                across_gm_scores = left_across_gm_scores if left_across_gm_scores else {}
                across_wm_scores = left_across_wm_scores if left_across_wm_scores else {}
                across_group_label = "left"
            elif group_key == "right" and (right_across_gm_scores or right_across_wm_scores):
                across_gm_scores = right_across_gm_scores if right_across_gm_scores else {}
                across_wm_scores = right_across_wm_scores if right_across_wm_scores else {}
                across_group_label = "right"
            else:
                across_gm_scores = {}
                across_wm_scores = {}
                across_group_label = group_key
            
            # Generate brain maps for across-factor |z| if we have data
            if across_gm_scores or across_wm_scores:
                # Compute colorbar range
                all_across_scores = []
                for score_dict in [across_gm_scores, across_wm_scores]:
                    for score in score_dict.values():
                        if not pd.isna(score):
                            all_across_scores.append(score)
                
                if all_across_scores:
                    vmin_across_group = 0
                    vmax_across_group = max(all_across_scores)
                    
                    # Save colorbar
                    across_group_colorbar_path = ospj(brain_maps_dir, f"across_factor_{across_group_label}_colorbar.png")
                    save_colorbar_png(vmin_across_group, vmax_across_group, across_group_colorbar_path, 
                                     use_absolute=True, label="Mahalanobis Distance Across Factors")
                    
                    # Generate GM brain map
                    if across_gm_scores:
                        gm_brain_map_path = ospj(brain_maps_dir, f"across_factor_{across_group_label}_gm_brain_map.png")
                        create_brain_map(
                            across_gm_scores,
                            "Across Factor Mahalanobis Distance",
                            across_group_label,
                            gm_brain_map_path,
                            vmin=vmin_across_group,
                            vmax=vmax_across_group,
                            use_absolute=True
                        )
                        brain_map_paths[("across_factor", across_group_label, "ctx_y")] = gm_brain_map_path.replace('.png', '_ctx_y.png')
                        brain_map_paths[("across_factor", across_group_label, "ctx_lr")] = gm_brain_map_path.replace('.png', '_ctx_lr.png')
                        brain_map_paths[("across_factor", across_group_label, "sctx_y")] = gm_brain_map_path.replace('.png', '_sctx_y.png')
                        brain_map_paths[("across_factor", across_group_label, "sctx_lr")] = gm_brain_map_path.replace('.png', '_sctx_lr.png')
                    
                    # Generate WM brain maps
                    if across_wm_scores:
                        tract_scores_dict = {}
                        for roi_key, score in across_wm_scores.items():
                            if not pd.isna(score):
                                parts = roi_key.rsplit("_", 2)
                                if len(parts) == 3:
                                    tract_base, hemi, seg_label = parts
                                    tract_name = f"{tract_base}_{hemi}"
                                    end1_label = tract_to_end1.get(tract_name, "end1")
                                    end2_label = tract_to_end2.get(tract_name, "end2")
                                    segment_to_label = {
                                        'end1': f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
                                        'core': 'core',
                                        'end2': f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
                                    }
                                    segment = None
                                    for seg, seg_lab in segment_to_label.items():
                                        if seg_lab == seg_label:
                                            segment = seg
                                            break
                                    if segment:
                                        tract_scores_dict[(tract_name, segment)] = score
                        
                        if tract_scores_dict:
                            assoc_path = ospj(brain_maps_dir, f"across_factor_{across_group_label}_wm_association.png")
                            proj_path = ospj(brain_maps_dir, f"across_factor_{across_group_label}_wm_projection.png")
                            create_wm_tract_brain_map(
                                tract_scores_dict,
                                "Across Factor Mahalanobis Distance",
                                across_group_label,
                                assoc_path,
                                proj_path,
                                tract_metadata_df,
                                use_absolute=True,
                                vmin=vmin_across_group,
                                vmax=vmax_across_group
                            )
                            brain_map_paths[("across_factor", across_group_label, "assoc_y")] = assoc_path.replace('.png', '_y.png')
                            brain_map_paths[("across_factor", across_group_label, "assoc_lr")] = assoc_path.replace('.png', '_lr.png')
                            brain_map_paths[("across_factor", across_group_label, "proj_y")] = proj_path.replace('.png', '_y.png')
                            brain_map_paths[("across_factor", across_group_label, "proj_lr")] = proj_path.replace('.png', '_lr.png')
                    
                    # Get brain map images
                    across_ctx_y_img = image_to_base64(brain_map_paths.get(("across_factor", across_group_label, "ctx_y")))
                    across_ctx_lr_img = image_to_base64(brain_map_paths.get(("across_factor", across_group_label, "ctx_lr")))
                    across_sctx_y_img = image_to_base64(brain_map_paths.get(("across_factor", across_group_label, "sctx_y")))
                    across_sctx_lr_img = image_to_base64(brain_map_paths.get(("across_factor", across_group_label, "sctx_lr")))
                    across_assoc_y_img = image_to_base64(brain_map_paths.get(("across_factor", across_group_label, "assoc_y")))
                    across_assoc_lr_img = image_to_base64(brain_map_paths.get(("across_factor", across_group_label, "assoc_lr")))
                    across_proj_y_img = image_to_base64(brain_map_paths.get(("across_factor", across_group_label, "proj_y")))
                    across_proj_lr_img = image_to_base64(brain_map_paths.get(("across_factor", across_group_label, "proj_lr")))
                    across_colorbar_img = image_to_base64(across_group_colorbar_path) if os.path.exists(across_group_colorbar_path) else None
                    
                    # Generate summaries
                    across_tract_summary_dict = {}
                    across_gm_summary_dict = {}
                    
                    # Format WM scores for summary
                    for roi_key, score in across_wm_scores.items():
                        if not pd.isna(score):
                            parts = roi_key.rsplit("_", 2)
                            if len(parts) == 3:
                                tract_base, hemi, seg_label = parts
                                tract_label = f"{tract_base}_{hemi}"
                                tract_name = tract_label_to_name.get(tract_label, tract_label)
                                if tract_name.endswith("_L") or tract_name.endswith("_R"):
                                    tract_name = tract_name[:-2]
                                tract_name = tract_name.replace("_", " ")
                                segment_clean = seg_label.replace("end-", "") if seg_label.startswith("end-") else seg_label
                                segment_expansions = {
                                    "core": "Core",
                                    "A": "Anterior",
                                    "P": "Posterior",
                                    "I": "Inferior",
                                    "S": "Superior",
                                    "M": "Medial",
                                    "L": "Lateral"
                                }
                                expanded_segment = segment_expansions.get(segment_clean, segment_clean)
                                tract_segment_id = f"{hemi} {tract_name} ({expanded_segment})"
                                across_tract_summary_dict[tract_segment_id] = score
                    
                    # Format GM scores for summary
                    for region, score in across_gm_scores.items():
                        if not pd.isna(score):
                            clean_region = region
                            hemisphere = ""
                            if clean_region.startswith("LH-") or clean_region.startswith("LH_"):
                                clean_region = clean_region[3:]
                                hemisphere = "L"
                            elif clean_region.startswith("RH-") or clean_region.startswith("RH_"):
                                clean_region = clean_region[3:]
                                hemisphere = "R"
                            expanded_region = expand_gm_region_abbreviation(clean_region)
                            if expanded_region != clean_region:
                                human_region = expanded_region
                            else:
                                human_region = clean_region.replace("_", " ").title()
                            if hemisphere:
                                human_region = f"{hemisphere} {human_region}"
                            across_gm_summary_dict[human_region] = score
                    
                    across_tract_summary_html = generate_abnormality_summary_html(
                        across_tract_summary_dict, top_n=10, is_tract=True,
                        tract_label_to_name=tract_label_to_name,
                        tract_to_end1=tract_to_end1, tract_to_end2=tract_to_end2
                    )
                    across_gm_summary_html = generate_abnormality_summary_html(
                        across_gm_summary_dict, top_n=10, is_tract=False
                    )
                    
                    # Get hemisphere plot image for this group
                    hemisphere_plot_path = ospj(os.path.dirname(output_path), "plots", f"hemisphere_mahalanobis_{across_group_label}.png")
                    hemisphere_plot_img = None
                    if os.path.exists(hemisphere_plot_path):
                        # Helper function to convert image to base64 (local to this scope)
                        def image_to_base64_local(image_path: str) -> str | None:
                            if not image_path or not os.path.exists(image_path):
                                return None
                            with open(image_path, "rb") as img_f:
                                img_data = img_f.read()
                            img_base64 = b64encode(img_data).decode("utf-8")
                            ext = os.path.splitext(image_path)[1].lower()
                            if ext == ".png":
                                mime = "image/png"
                            elif ext in [".jpg", ".jpeg"]:
                                mime = "image/jpeg"
                            else:
                                mime = "image/png"
                            return f"data:{mime};base64,{img_base64}"
                        hemisphere_plot_img = image_to_base64_local(hemisphere_plot_path)
                    
                    # Add HTML section
                    html += f"""
        <h3>Across Factor Mahalanobis Distance Brain Maps</h3>
        <p style="font-size: 0.9em; color: #666;">Sum of absolute z-scores across all factors for each ROI.</p>
        <!-- Brain maps side-by-side: Coronal (Y) on left, Lateral (LR) on right -->
        <div style="display: flex; gap: 30px; margin: 30px 0; width: 100%; box-sizing: border-box;">
            <!-- Coronal View (Y): 2x2 grid -->
            <div style="flex: 1; min-width: 0; box-sizing: border-box;">
                <h4 style="text-align: center; margin: 0;">Coronal View</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0; margin: 0;">
                    <div style="text-align: center;">
                        <p style="font-size: 0.9em; margin: 3px 0;">Cortex</p>
                        {f'<img src="{across_ctx_y_img}" alt="across_factor_{across_group_label}_cortex_y" style="max-width: 100%; border: 1px solid #ddd;">' if across_ctx_y_img else '<p>No data</p>'}
                </div>
                    <div style="text-align: center;">
                        <p style="font-size: 0.9em; margin: 3px 0;">Association</p>
                        {f'<img src="{across_assoc_y_img}" alt="across_factor_{across_group_label}_association_y" style="max-width: 100%; border: 1px solid #ddd;">' if across_assoc_y_img else '<p>No data</p>'}
                </div>
                    <div style="text-align: center;">
                        <p style="font-size: 0.9em; margin: 3px 0;">Subcortex</p>
                        {f'<img src="{across_sctx_y_img}" alt="across_factor_{across_group_label}_subcortex_y" style="max-width: 100%; border: 1px solid #ddd;">' if across_sctx_y_img else '<p>No data</p>'}
                </div>
                    <div style="text-align: center;">
                        <p style="font-size: 0.9em; margin: 3px 0;">Projection</p>
                        {f'<img src="{across_proj_y_img}" alt="across_factor_{across_group_label}_projection_y" style="max-width: 100%; border: 1px solid #ddd;">' if across_proj_y_img else '<p>No data</p>'}
                </div>
            </div>
                </div>
            
            <!-- Lateral View (LR): 2x2 grid -->
            <div style="flex: 1; min-width: 0; box-sizing: border-box;">
                <h4 style="text-align: center; margin: 0;">Lateral Views</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0; margin: 0;">
                    <div style="text-align: center;">
                        <p style="font-size: 0.9em; margin: 3px 0;">Cortex</p>
                        {f'<img src="{across_ctx_lr_img}" alt="across_factor_{across_group_label}_cortex_lr" style="max-width: 100%; border: 1px solid #ddd;">' if across_ctx_lr_img else '<p>No data</p>'}
                </div>
                    <div style="text-align: center;">
                        <p style="font-size: 0.9em; margin: 3px 0;">Association</p>
                        {f'<img src="{across_assoc_lr_img}" alt="across_factor_{across_group_label}_association_lr" style="max-width: 100%; border: 1px solid #ddd;">' if across_assoc_lr_img else '<p>No data</p>'}
                </div>
                    <div style="text-align: center;">
                        <p style="font-size: 0.9em; margin: 3px 0;">Subcortex</p>
                        {f'<img src="{across_sctx_lr_img}" alt="across_factor_{across_group_label}_subcortex_lr" style="max-width: 100%; border: 1px solid #ddd;">' if across_sctx_lr_img else '<p>No data</p>'}
                    </div>
                    <div style="text-align: center;">
                        <p style="font-size: 0.9em; margin: 3px 0;">Projection</p>
                        {f'<img src="{across_proj_lr_img}" alt="across_factor_{across_group_label}_projection_lr" style="max-width: 100%; border: 1px solid #ddd;">' if across_proj_lr_img else '<p>No data</p>'}
                    </div>
                </div>
            </div>
        </div>
        {f'<div style="text-align: center; margin: 20px 0;"><img src="{across_colorbar_img}" alt="across_factor_{across_group_label}_colorbar" style="max-width: 600px;"></div>' if across_colorbar_img else ''}
        
        <div class="summary-grid" style="margin: 30px 0;">
            <div class="summary-card">
                <h3>Most Abnormal Gray Matter Regions</h3>
                <div class="bar-chart">
                    {across_gm_summary_html}
                </div>
            </div>
            
            <div class="summary-card">
                <h3>Most Abnormal White Matter Tracts</h3>
                <div class="bar-chart">
                    {across_tract_summary_html}
                </div>
            </div>
        </div>
        
        <div class="heatmap-container" style="margin: 30px 0;" id="hemisphere-mahalanobis-{across_group_label}">
            <h3>Across Factor Mahalanobis Distances by Hemisphere</h3>
            <p style="font-size: 0.9em; color: #666;">Comparison of across-factor Mahalanobis distances between left and right hemispheres, shown separately for all regions, gray matter regions, and white matter tract segments.</p>
            {f'<img src="{hemisphere_plot_img}" alt="Hemisphere Mahalanobis Comparison ({across_group_label})" style="max-width: 100%; border: 1px solid #ddd;">' if hemisphere_plot_img else '<p>Hemisphere comparison plot not available</p>'}
        </div>
"""
        
        # For each factor within this group
        for factor in factors:
            factor_label = get_factor_label(factor)
            html += f"""
        <h3>{factor_label} {map_type_label}</h3>
"""
            
            # Brain maps - get y and lr views for GM and WM
            ctx_y_key = (factor, group_key, "ctx_y")
            ctx_lr_key = (factor, group_key, "ctx_lr")
            sctx_y_key = (factor, group_key, "sctx_y")
            sctx_lr_key = (factor, group_key, "sctx_lr")
            assoc_y_key = (factor, group_key, "assoc_y")
            assoc_lr_key = (factor, group_key, "assoc_lr")
            proj_y_key = (factor, group_key, "proj_y")
            proj_lr_key = (factor, group_key, "proj_lr")
            
            # Get all brain map images
            ctx_y_img = image_to_base64(brain_map_paths.get(ctx_y_key)) if ctx_y_key in brain_map_paths else None
            ctx_lr_img = image_to_base64(brain_map_paths.get(ctx_lr_key)) if ctx_lr_key in brain_map_paths else None
            sctx_y_img = image_to_base64(brain_map_paths.get(sctx_y_key)) if sctx_y_key in brain_map_paths else None
            sctx_lr_img = image_to_base64(brain_map_paths.get(sctx_lr_key)) if sctx_lr_key in brain_map_paths else None
            assoc_y_img = image_to_base64(brain_map_paths.get(assoc_y_key)) if assoc_y_key in brain_map_paths else None
            assoc_lr_img = image_to_base64(brain_map_paths.get(assoc_lr_key)) if assoc_lr_key in brain_map_paths else None
            proj_y_img = image_to_base64(brain_map_paths.get(proj_y_key)) if proj_y_key in brain_map_paths else None
            proj_lr_img = image_to_base64(brain_map_paths.get(proj_lr_key)) if proj_lr_key in brain_map_paths else None
            
            # Get colorbar image
            colorbar_key = (factor, group_key)
            colorbar_img = image_to_base64(colorbar_paths.get(colorbar_key)) if colorbar_key in colorbar_paths else None
            
            if ctx_y_img or ctx_lr_img or sctx_y_img or sctx_lr_img or assoc_y_img or assoc_lr_img or proj_y_img or proj_lr_img:
                # Layout: Coronal view (2x2) on left, Lateral view (2x2) on right
                html += f"""
                <div style="display: flex; gap: 30px; margin: 30px 0; width: 100%; box-sizing: border-box;">
                    <!-- Coronal View (Y): 2x2 grid -->
                    <div style="flex: 1; min-width: 0; box-sizing: border-box;">
                        <h4 style="text-align: center; margin: 0;">Coronal View</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0; margin: 0;">
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Cortex</p>
                                {f'<img src="{ctx_y_img}" alt="{factor}_{group_key}_cortex_y" style="max-width: 100%; border: 1px solid #ddd;">' if ctx_y_img else '<p>No data</p>'}
                        </div>
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Association</p>
                                {f'<img src="{assoc_y_img}" alt="{factor}_{group_key}_association_y" style="max-width: 100%; border: 1px solid #ddd;">' if assoc_y_img else '<p>No data</p>'}
                        </div>
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Subcortex</p>
                                {f'<img src="{sctx_y_img}" alt="{factor}_{group_key}_subcortex_y" style="max-width: 100%; border: 1px solid #ddd;">' if sctx_y_img else '<p>No data</p>'}
                        </div>
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Projection</p>
                                {f'<img src="{proj_y_img}" alt="{factor}_{group_key}_projection_y" style="max-width: 100%; border: 1px solid #ddd;">' if proj_y_img else '<p>No data</p>'}
                        </div>
                    </div>
                        </div>
                    
                    <!-- Lateral View (LR): 2x2 grid -->
                    <div style="flex: 1; min-width: 0; box-sizing: border-box;">
                        <h4 style="text-align: center; margin: 0;">Lateral Views</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0; margin: 0;">
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Cortex</p>
                                {f'<img src="{ctx_lr_img}" alt="{factor}_{group_key}_cortex_lr" style="max-width: 100%; border: 1px solid #ddd;">' if ctx_lr_img else '<p>No data</p>'}
                        </div>
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Association</p>
                                {f'<img src="{assoc_lr_img}" alt="{factor}_{group_key}_association_lr" style="max-width: 100%; border: 1px solid #ddd;">' if assoc_lr_img else '<p>No data</p>'}
                        </div>
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Subcortex</p>
                                {f'<img src="{sctx_lr_img}" alt="{factor}_{group_key}_subcortex_lr" style="max-width: 100%; border: 1px solid #ddd;">' if sctx_lr_img else '<p>No data</p>'}
                            </div>
                            <div style="text-align: center;">
                                <p style="font-size: 0.9em; margin: 3px 0;">Projection</p>
                                {f'<img src="{proj_lr_img}" alt="{factor}_{group_key}_projection_lr" style="max-width: 100%; border: 1px solid #ddd;">' if proj_lr_img else '<p>No data</p>'}
                            </div>
                        </div>
                    </div>
                </div>
"""
            
            # Add colorbar if available
            if colorbar_img:
                html += f"""
                <div style="text-align: center; margin: 20px 0;">
                    <img src="{colorbar_img}" alt="{factor}_{group_key}_colorbar" style="max-width: 600px;">
                </div>
"""
            
            # Collect scores for bar charts
            tract_scores_dict = {}
            gm_scores_dict = {}
            
            if group_key == "controls":
                # For controls, use factor scores from DataFrames (rescaled 0-1)
                if not tract_segment_scores_df.empty and factor in tract_segment_scores_df.columns:
                    for (tract, segment), row in tract_segment_scores_df.iterrows():
                        score = row[factor]
                        if not pd.isna(score):
                            # Get segment label for display
                            end1_label = tract_to_end1.get(tract, "end1")
                            end2_label = tract_to_end2.get(tract, "end2")
                            segment_to_label = {
                                'end1': f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
                                'core': 'core',
                                'end2': f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
                            }
                            seg_label = segment_to_label.get(segment, segment)
                            
                            # Get tract name
                            tract_name = tract_label_to_name.get(tract, tract)
                            if tract_name.endswith("_L") or tract_name.endswith("_R"):
                                tract_name = tract_name[:-2]
                            tract_name = tract_name.replace("_", " ")
                            
                            # Expand segment label
                            segment_clean = seg_label.replace("end-", "") if seg_label.startswith("end-") else seg_label
                            segment_expansions = {
                                "core": "Core",
                                "A": "Anterior",
                                "P": "Posterior",
                                "I": "Inferior",
                                "S": "Superior",
                                "M": "Medial",
                                "L": "Lateral"
                            }
                            expanded_segment = segment_expansions.get(segment_clean, segment_clean)
                            tract_segment_id = f"{tract_name} ({expanded_segment})"
                            tract_scores_dict[tract_segment_id] = score
                
                if not region_scores_df.empty and factor in region_scores_df.columns:
                    for region, row in region_scores_df.iterrows():
                        score = row[factor]
                        if not pd.isna(score):
                            # Format region name
                            clean_region = region
                            hemisphere = ""
                            if clean_region.startswith("LH-") or clean_region.startswith("LH_"):
                                clean_region = clean_region[3:]
                                hemisphere = "L"
                            elif clean_region.startswith("RH-") or clean_region.startswith("RH_"):
                                clean_region = clean_region[3:]
                                hemisphere = "R"
                            
                            # Expand abbreviations
                            expanded_region = expand_gm_region_abbreviation(clean_region)
                            if expanded_region != clean_region:
                                human_region = expanded_region
                            else:
                                human_region = clean_region.replace("_", " ").title()
                            
                            if hemisphere:
                                human_region = f"{hemisphere} {human_region}"
                            
                            gm_scores_dict[human_region] = score
            else:
                # For patient groups, use signed z-scores (not absolute)
                # Use left/right specific z-scores if available, otherwise fall back to all
                if group_key == "left":
                    wm_z_scores_dict = left_factor_z_scores if (left_factor_z_scores and factor in left_factor_z_scores) else (all_factor_z_scores if factor in all_factor_z_scores else {})
                    gm_z_scores_dict = left_gm_factor_z_scores if (left_gm_factor_z_scores and factor in left_gm_factor_z_scores) else (all_gm_factor_z_scores if factor in all_gm_factor_z_scores else {})
                elif group_key == "right":
                    wm_z_scores_dict = right_factor_z_scores if (right_factor_z_scores and factor in right_factor_z_scores) else (all_factor_z_scores if factor in all_factor_z_scores else {})
                    gm_z_scores_dict = right_gm_factor_z_scores if (right_gm_factor_z_scores and factor in right_gm_factor_z_scores) else (all_gm_factor_z_scores if factor in all_gm_factor_z_scores else {})
                else:
                    # Fallback to all (shouldn't happen for left/right, but handle it)
                    wm_z_scores_dict = all_factor_z_scores if factor in all_factor_z_scores else {}
                    gm_z_scores_dict = all_gm_factor_z_scores if factor in all_gm_factor_z_scores else {}
                
                # Process tract segments using z-scores
                if wm_z_scores_dict and factor in wm_z_scores_dict:
                    for roi_key, z_score in wm_z_scores_dict[factor].items():
                        if not pd.isna(z_score):
                            # Parse roi_key: "{tract_base}_{hemi}_{seg_label}" or "{tract_base}_{seg_label}"
                            parts = roi_key.rsplit("_", 2)
                            if len(parts) == 3:
                                tract_base, hemi, seg_label = parts
                                tract_label = f"{tract_base}_{hemi}"
                                tract_name = tract_label_to_name.get(tract_label, tract_label)
                                if tract_name.endswith("_L") or tract_name.endswith("_R"):
                                    tract_name = tract_name[:-2]
                                tract_name = tract_name.replace("_", " ")
                                # Expand segment label
                                segment_clean = seg_label.replace("end-", "") if seg_label.startswith("end-") else seg_label
                                segment_expansions = {
                                    "core": "Core",
                                    "A": "Anterior",
                                    "P": "Posterior",
                                    "I": "Inferior",
                                    "S": "Superior",
                                    "M": "Medial",
                                    "L": "Lateral"
                                }
                                expanded_segment = segment_expansions.get(segment_clean, segment_clean)
                                # Include hemisphere prefix in tract name
                                tract_segment_id = f"{hemi} {tract_name} ({expanded_segment})"
                                tract_scores_dict[tract_segment_id] = z_score
                
                # Process GM regions using z-scores
                if gm_z_scores_dict and factor in gm_z_scores_dict:
                    for region, z_score in gm_z_scores_dict[factor].items():
                        if not pd.isna(z_score):
                            # Format region name
                            clean_region = region
                            hemisphere = ""
                            if clean_region.startswith("LH-") or clean_region.startswith("LH_"):
                                clean_region = clean_region[3:]
                                hemisphere = "L"
                            elif clean_region.startswith("RH-") or clean_region.startswith("RH_"):
                                clean_region = clean_region[3:]
                                hemisphere = "R"
                            
                            # Expand abbreviations
                            expanded_region = expand_gm_region_abbreviation(clean_region)
                            if expanded_region != clean_region:
                                human_region = expanded_region
                            else:
                                human_region = clean_region.replace("_", " ").title()
                            
                            if hemisphere:
                                human_region = f"{hemisphere} {human_region}"
                            
                            gm_scores_dict[human_region] = z_score
                
                # Generate bar chart summaries
            if tract_scores_dict or gm_scores_dict:
                tract_summary_html = generate_abnormality_summary_html(
                    tract_scores_dict, top_n=10, is_tract=True,
                    tract_label_to_name=tract_label_to_name,
                    tract_to_end1=tract_to_end1, tract_to_end2=tract_to_end2
                )
                gm_summary_html = generate_abnormality_summary_html(
                    gm_scores_dict, top_n=10, is_tract=False
                )
                
                # Use different titles for controls vs patients
                if group_key == "controls":
                    summary_title_gm = "Top Gray Matter Regions"
                    summary_title_wm = "Top White Matter Tracts"
                else:
                    summary_title_gm = "Most Abnormal Gray Matter Regions"
                    summary_title_wm = "Most Abnormal White Matter Tracts"
                
                html += f"""
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>{summary_title_gm}</h3>
                        <div class="bar-chart">
                            {gm_summary_html}
                        </div>
                    </div>
                    
                    <div class="summary-card">
                        <h3>{summary_title_wm}</h3>
                        <div class="bar-chart">
                            {tract_summary_html}
                        </div>
                    </div>
                </div>
"""
        
        # Close the group-section div
        html += """
        </div>
"""
    
    # Add footer and close content/container divs (outside the group loop)
    html += """
        <div class="footer">
            <p>Generated on {timestamp_str}</p>
            <p class="timestamp">Structural Tractometry Analysis Pipeline - Factor Scores</p>
        </div>
        </div>
    </div>
</body>
</html>""".format(timestamp_str=timestamp_str)
    
    # Save HTML report (single report, no suffix)
    with open(output_path, "w") as f:
        f.write(html)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    print("factor_z-scores: starting (discovery + load can take a few minutes on network storage)...")
    # Discover all regions and tracts
    all_regions = discover_all_gm_regions()
    all_tracts = discover_all_wm_tracts()
    print(f"  Discovered {len(all_regions)} GM regions, {len(all_tracts)} WM tracts")
    # Filter out excluded tracts
    all_tracts = [t for t in all_tracts if t not in TRACTS_TO_REMOVE]
    
    if not all_regions or not all_tracts:
        print("No regions or tracts found; aborting.")
        return
    
    # Check if factor scores already exist
    factor_scores_output_dir = ospj(OUTPUT_PROJECT_ROOT, "factor_scores")
    # Check new directory structure (epilepsy/controls subdirectories)
    epilepsy_region_dir = ospj(factor_scores_output_dir, "epilepsy", "gm_regions")
    epilepsy_tract_dir = ospj(factor_scores_output_dir, "epilepsy", "wm_tracts")
    # Also check old structure for backward compatibility
    old_region_scores_dir = ospj(factor_scores_output_dir, "gm_regions")
    old_tract_scores_dir = ospj(factor_scores_output_dir, "wm_tracts")
    
    factor_scores_exist = False
    # Consolidated wide tables: epilepsy_F1_scores.csv, … at factor_scores root
    if glob.glob(ospj(factor_scores_output_dir, "epilepsy_*_scores.csv")):
        factor_scores_exist = True
    # Per-ROI layout under epilepsy/gm_regions and epilepsy/wm_tracts
    elif os.path.exists(epilepsy_region_dir) and os.path.exists(epilepsy_tract_dir):
        existing_regions = [f.replace("_factor_scores.csv", "")
                          for f in os.listdir(epilepsy_region_dir)
                          if f.endswith("_factor_scores.csv")]
        existing_tracts = [f.replace("_factor_scores.csv", "")
                         for f in os.listdir(epilepsy_tract_dir)
                         if f.endswith("_factor_scores.csv")]
        if len(existing_regions) > 0 or len(existing_tracts) > 0:
            factor_scores_exist = True
    # Legacy gm_regions / wm_tracts at root
    elif os.path.exists(old_region_scores_dir) and os.path.exists(old_tract_scores_dir):
        existing_regions = [f.replace("_factor_scores.csv", "")
                          for f in os.listdir(old_region_scores_dir)
                          if f.endswith("_factor_scores.csv")]
        existing_tracts = [f.replace("_factor_scores.csv", "")
                         for f in os.listdir(old_tract_scores_dir)
                         if f.endswith("_factor_scores.csv")]
        if len(existing_regions) > 0 or len(existing_tracts) > 0:
            factor_scores_exist = True
    
    # Only load data if we need to compute factor scores or generate plots
    # Check if plots already exist
    plots_dir = ospj(OUTPUT_PROJECT_ROOT, "plots")
    plots_exist = os.path.exists(plots_dir) and len([f for f in os.listdir(plots_dir) if f.endswith(".png")]) > 0
    
    if factor_scores_exist and plots_exist:
        # Cohort lists for master report (same logic as full run: inclusion patients + GAM union controls)
        scalar_labels = load_scalar_labels()
        patient_subjects = load_temporal_patient_subjects_ordered()
        if not patient_subjects and all_regions and scalar_labels:
            sample_region = all_regions[0]
            sample_scalar = scalar_labels[0]
            sample_data = load_gm_region_scalar_data(sample_region, sample_scalar, PATIENT_GROUPS)
            included_temporal = load_included_temporal_subjects()
            if sample_data is not None:
                patient_subjects = sorted(
                    str(s) for s in sample_data.index if str(s) in included_temporal
                )
        left_lateralized_subjects, right_lateralized_subjects = get_lateralization_groups(
            patient_subjects
        )
        control_subjects = (
            collect_control_subjects_union_from_gam(
                all_regions, all_tracts, scalar_labels, CONTROL_GROUPS
            )
            if scalar_labels
            else []
        )
        
        gm_data = {}
        wm_tract_data = {}
        common_subjects = []
    elif factor_scores_exist:
        # Factor scores exist but plots don't - still need to load data for plots
        scalar_labels = load_scalar_labels()
        
        # Load all data (needed for plots)
        gm_data, wm_tract_data, common_subjects = load_all_data(
            all_regions,
            all_tracts,
            scalar_labels,
            CONTROL_GROUPS,
            PATIENT_GROUPS,
        )
        
        if not common_subjects:
            print("No subjects found in any loaded table; aborting.")
            return
        
        # Get subject groups
        control_subjects, patient_subjects = get_subject_groups(
            gm_data, wm_tract_data,
            all_regions, all_tracts,
            scalar_labels,
            CONTROL_GROUPS, PATIENT_GROUPS,
        )
        print(f"Controls: {len(control_subjects)}, Patients: {len(patient_subjects)}")
        
        # Get lateralization groups (needed for factor score analysis)
        left_lateralized_subjects, right_lateralized_subjects = get_lateralization_groups(patient_subjects)
    else:
        # Load scalar labels
        scalar_labels = load_scalar_labels()
        
        # Load all data
        gm_data, wm_tract_data, common_subjects = load_all_data(
            all_regions,
            all_tracts,
            scalar_labels,
            CONTROL_GROUPS,
            PATIENT_GROUPS,
        )
        
        if not common_subjects:
            print("No subjects found in any loaded table; aborting.")
            return
        
        # Get subject groups
        control_subjects, patient_subjects = get_subject_groups(
            gm_data, wm_tract_data,
            all_regions, all_tracts,
            scalar_labels,
            CONTROL_GROUPS, PATIENT_GROUPS,
        )
        print(f"Controls: {len(control_subjects)}, Patients: {len(patient_subjects)}")
        
        # Get lateralization groups (needed for factor score analysis)
        left_lateralized_subjects, right_lateralized_subjects = get_lateralization_groups(patient_subjects)
    
    # Create output directories
    plots_dir = ospj(OUTPUT_PROJECT_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract region bases (e.g., "Hippocampus" from "LH_Hippocampus" and "RH_Hippocampus")
    # Handle both "LH_/RH_" and "LH-/RH-" prefixes
    region_bases = set()
    for region in all_regions:
        if region.startswith("LH_") or region.startswith("LH-"):
            # Remove "LH_" or "LH-" prefix
            region_bases.add(region[3:])
        elif region.startswith("RH_") or region.startswith("RH-"):
            # Remove "RH_" or "RH-" prefix
            region_bases.add(region[3:])
        else:
            region_bases.add(region)
    
    # Extract tract bases (e.g., "F" from "F_L" and "F_R")
    tract_bases = set()
    for tract in all_tracts:
        if tract.endswith("_L"):
            tract_bases.add(tract[:-2])  # Remove "_L" suffix
        elif tract.endswith("_R"):
            tract_bases.add(tract[:-2])  # Remove "_R" suffix
        else:
            tract_bases.add(tract)
    
    # Load tract metadata for human-readable names
    tract_label_to_name = load_tract_metadata()
    
    # Load factor loadings
    factor_loadings = load_factor_loadings()
    
    if factor_loadings.empty:
        print("Warning: Could not load factor loadings. Skipping factor score computation.")
    else:
        # Compute and save factor scores for ALL regions and tracts
        # Check if factor scores already exist
        factor_scores_output_dir = ospj(OUTPUT_PROJECT_ROOT, "factor_scores")
        if not factor_scores_exist:
            compute_and_save_all_factor_scores(
                scalar_labels, PATIENT_GROUPS, CONTROL_GROUPS,
                factor_loadings,
                factor_scores_output_dir,
            )
        
        # Use the same groups already computed above
        
        # Load and plot region factor scores (3-row layout)
        # Load pre-computed factor scores from CSV files instead of recomputing
        region_factor_plot_paths: Dict[str, str] = {}
        if gm_data:
            # Load from epilepsy directory (for patient plots)
            region_scores_dir = ospj(factor_scores_output_dir, "epilepsy", "gm_regions")
            # Fallback to old structure if new structure doesn't exist
            if not os.path.exists(region_scores_dir):
                region_scores_dir = ospj(factor_scores_output_dir, "gm_regions")
            if not os.path.exists(region_scores_dir):
                region_scores_dir = factor_scores_output_dir
            for region_base in tqdm(sorted(region_bases), desc="Region factor scores"):
                # Handle both "LH_/RH_" and "LH-/RH-" prefixes
                left_region = None
                right_region = None
                if f"LH_{region_base}" in gm_data:
                    left_region = f"LH_{region_base}"
                elif f"LH-{region_base}" in gm_data:
                    left_region = f"LH-{region_base}"
                
                if f"RH_{region_base}" in gm_data:
                    right_region = f"RH_{region_base}"
                elif f"RH-{region_base}" in gm_data:
                    right_region = f"RH-{region_base}"
                
                factor_scores_left = pd.DataFrame()
                factor_scores_right = pd.DataFrame()
                
                # Try to load pre-computed factor scores (per-ROI CSV or consolidated wide tables)
                if left_region:
                    factor_scores_left = load_factor_scores_from_csv(
                        region_scores_dir, left_region, cohort="epilepsy",
                    )
                    if factor_scores_left.empty:
                        left_roi_data = {scalar: gm_data[left_region][scalar]
                                        for scalar in scalar_labels
                                        if scalar in gm_data[left_region]}
                        factor_scores_left = compute_factor_scores(
                            left_roi_data, left_region, scalar_labels,
                            patient_subjects, factor_loadings,
                        )

                if right_region:
                    factor_scores_right = load_factor_scores_from_csv(
                        region_scores_dir, right_region, cohort="epilepsy",
                    )
                    if factor_scores_right.empty:
                        right_roi_data = {scalar: gm_data[right_region][scalar]
                                         for scalar in scalar_labels
                                         if scalar in gm_data[right_region]}
                        factor_scores_right = compute_factor_scores(
                            right_roi_data, right_region, scalar_labels,
                            patient_subjects, factor_loadings,
                        )
                
                if not factor_scores_left.empty or not factor_scores_right.empty:
                    # Factor score plots with 3-row layout
                    plot_path = ospj(plots_dir, f"{region_base}_factor_scores.png")
                    # Check if plot already exists
                    if os.path.exists(plot_path):
                        region_factor_plot_paths[region_base] = plot_path
                    else:
                        plot_region_factor_scores_by_groups(
                            factor_scores_left, factor_scores_right, region_base,
                            patient_subjects,
                            left_lateralized_subjects, right_lateralized_subjects,
                            plot_path,
                        )
                        if os.path.exists(plot_path):
                            region_factor_plot_paths[region_base] = plot_path
        else:
            # Load existing plot paths
            if os.path.exists(plots_dir):
                for region_base in sorted(region_bases):
                    plot_path = ospj(plots_dir, f"{region_base}_factor_scores.png")
                    if os.path.exists(plot_path):
                        region_factor_plot_paths[region_base] = plot_path
        
        # Load and plot tract factor scores (3-row layout)
        # Load pre-computed segment-specific factor scores and average them for tract-level scores
        tract_factor_plot_paths: Dict[str, str] = {}
        if wm_tract_data:
            # Load from epilepsy directory (for patient plots)
            tract_scores_dir = ospj(factor_scores_output_dir, "epilepsy", "wm_tracts")
            # Fallback to old structure if new structure doesn't exist
            if not os.path.exists(tract_scores_dir):
                tract_scores_dir = ospj(factor_scores_output_dir, "wm_tracts")
            if not os.path.exists(tract_scores_dir):
                tract_scores_dir = factor_scores_output_dir
            # Load tract metadata to get end labels for file names
            tract_metadata_df = load_tract_metadata_full()
            tract_to_end1_label = {}
            tract_to_end2_label = {}
            if not tract_metadata_df.empty and "label" in tract_metadata_df.columns:
                if "end1" in tract_metadata_df.columns:
                    tract_to_end1_label = dict(zip(tract_metadata_df["label"], tract_metadata_df["end1"]))
                if "end2" in tract_metadata_df.columns:
                    tract_to_end2_label = dict(zip(tract_metadata_df["label"], tract_metadata_df["end2"]))
            
            # Helper function to load and average segment-specific factor scores
            def load_averaged_tract_scores(tract_name: str) -> pd.DataFrame:
                """Load segment-specific factor scores and average them across segments."""
                end1_label = tract_to_end1_label.get(tract_name, "end1")
                end2_label = tract_to_end2_label.get(tract_name, "end2")
                segment_to_label = {
                    'end1': f"end-{end1_label}" if end1_label != "NA" and pd.notna(end1_label) else "end1",
                    'core': 'core',
                    'end2': f"end-{end2_label}" if end2_label != "NA" and pd.notna(end2_label) else "end2",
                }
                segment_scores = []
                for segment in ['end1', 'core', 'end2']:
                    roi_key = _tract_segment_to_roi_key(
                        tract_name, segment, tract_to_end1_label, tract_to_end2_label,
                    )
                    seg_scores = load_factor_scores_from_csv(
                        tract_scores_dir, roi_key, cohort="epilepsy",
                    )
                    if not seg_scores.empty:
                        segment_scores.append(seg_scores)
                        continue
                    segment_label = segment_to_label[segment]
                    csv_path = ospj(
                        tract_scores_dir, f"{tract_name}_{segment_label}_factor_scores.csv",
                    )
                    if os.path.exists(csv_path):
                        try:
                            segment_scores.append(pd.read_csv(csv_path, index_col=0))
                        except Exception as e:
                            print(f"  Warning: Could not load {csv_path}: {e}")

                if segment_scores:
                    # Average across segments for each subject and factor
                    all_subjects = set()
                    all_factors = set()
                    for seg_df in segment_scores:
                        all_subjects.update(seg_df.index)
                        all_factors.update(seg_df.columns)

                    averaged = pd.DataFrame(index=sorted(all_subjects), columns=sorted(all_factors))
                    for subject in averaged.index:
                        for factor in averaged.columns:
                            values = []
                            for seg_df in segment_scores:
                                if subject in seg_df.index and factor in seg_df.columns:
                                    val = seg_df.loc[subject, factor]
                                    if not pd.isna(val):
                                        values.append(val)
                            if values:
                                averaged.loc[subject, factor] = np.mean(values)
                            else:
                                averaged.loc[subject, factor] = np.nan
                    return averaged
                return pd.DataFrame()
            
            for tract_base in tqdm(sorted(tract_bases), desc="Tract factor scores"):
                left_tract = f"{tract_base}_L" if f"{tract_base}_L" in wm_tract_data else None
                right_tract = f"{tract_base}_R" if f"{tract_base}_R" in wm_tract_data else None
                
                factor_scores_left = pd.DataFrame()
                factor_scores_right = pd.DataFrame()
                
                # Try to load pre-computed factor scores from CSV files
                if left_tract:
                    factor_scores_left = load_averaged_tract_scores(left_tract)
                    if factor_scores_left.empty:
                        # Fallback to computation if CSV files don't exist
                        left_roi_data = {scalar: wm_tract_data[left_tract][scalar] 
                                        for scalar in scalar_labels 
                                        if scalar in wm_tract_data[left_tract]}
                        factor_scores_left = compute_factor_scores(
                            left_roi_data, left_tract, scalar_labels,
                            patient_subjects, factor_loadings,
                        )
                
                if right_tract:
                    factor_scores_right = load_averaged_tract_scores(right_tract)
                    if factor_scores_right.empty:
                        # Fallback to computation if CSV files don't exist
                        right_roi_data = {scalar: wm_tract_data[right_tract][scalar] 
                                         for scalar in scalar_labels 
                                         if scalar in wm_tract_data[right_tract]}
                        factor_scores_right = compute_factor_scores(
                            right_roi_data, right_tract, scalar_labels,
                            patient_subjects, factor_loadings,
                        )
                
                if not factor_scores_left.empty or not factor_scores_right.empty:
                    # Factor score plots with 3-row layout
                    plot_path = ospj(plots_dir, f"{tract_base}_factor_scores.png")
                    # Check if plot already exists
                    if os.path.exists(plot_path):
                        tract_factor_plot_paths[tract_base] = plot_path
                    else:
                        plot_tract_factor_scores_by_groups(
                            factor_scores_left, factor_scores_right, tract_base,
                            tract_label_to_name,
                            patient_subjects,
                            left_lateralized_subjects, right_lateralized_subjects,
                            plot_path,
                        )
                        if os.path.exists(plot_path):
                            tract_factor_plot_paths[tract_base] = plot_path
        else:
            # Load existing plot paths
            if os.path.exists(plots_dir):
                for tract_base in sorted(tract_bases):
                    plot_path = ospj(plots_dir, f"{tract_base}_factor_scores.png")
                    if os.path.exists(plot_path):
                        tract_factor_plot_paths[tract_base] = plot_path
    
    # Generate master report sorting regions/tracts by factor scores
    if not factor_loadings.empty:
        factor_scores_output_dir = ospj(OUTPUT_PROJECT_ROOT, "factor_scores")
        master_report_path = ospj(OUTPUT_PROJECT_ROOT, "master_factor_score_report.html")
        
        # Get subject counts
        subject_counts = get_subject_counts_by_group(
            all_regions, all_tracts,
            scalar_labels if 'scalar_labels' in locals() else load_scalar_labels(),
            CONTROL_GROUPS, PATIENT_GROUPS
        )
        
        # Generate single report with signed z-scores
        print("factor_z-scores: building master report and z-score CSV exports...")
        create_master_report(
            master_report_path,
            factor_scores_output_dir,
            all_regions,
            all_tracts,
            patient_subjects,
            left_lateralized_subjects,
            right_lateralized_subjects,
            tract_label_to_name,
                subject_counts=subject_counts,
                control_subjects=control_subjects,
            )

    # Per-scalar GAM z-scores (subjects x GM + WM segments); does not require factor loadings
    tmeta = load_tract_metadata_full()
    te1: Dict[str, str] = {}
    te2: Dict[str, str] = {}
    if not tmeta.empty and "label" in tmeta.columns:
        if "end1" in tmeta.columns:
            te1 = dict(zip(tmeta["label"], tmeta["end1"]))
        if "end2" in tmeta.columns:
            te2 = dict(zip(tmeta["label"], tmeta["end2"]))
    save_scalar_z_scores(
        all_regions,
        all_tracts,
        patient_subjects,
        list(control_subjects) if control_subjects else [],
        PATIENT_GROUPS,
        CONTROL_GROUPS,
        te1,
        te2,
    )


if __name__ == "__main__":
    main()

