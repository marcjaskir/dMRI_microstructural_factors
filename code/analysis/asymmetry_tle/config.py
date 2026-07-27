"""Paths and constants for tract asymmetry pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

# Node ranges (from factor_analysis)
END1_NODES = list(range(1, 35))   # nodes 1-34
CORE_NODES = list(range(35, 67))  # nodes 35-66
END2_NODES = list(range(67, 101))  # nodes 67-100

EXCLUDED_SCALARS = [
    "map_li", "map_am", "dti_txx", "dti_txy", "dti_txz",
    "dti_tyy", "dti_tyz", "dti_tzz", "dti_ha", "rdi_rd1", "rdi_rd2",
]

FACTOR_ORDER = ["F1", "F2", "F3", "F4", "Mahalanobis"]
MICROSTRUCTURAL_FACTORS = {"F1", "F2", "F3", "F4", "Mahalanobis"}

SCALAR_MODEL_LABEL_MAP = {
    "dti": "DTI",
    "dki": "DKI",
    "noddi": "NODDI",
    "gqi": "GQI",
    "map": "MAP-MRI",
}

# Cortex-by-community strip plots: omit asymmetry values outside [Q1 - k*IQR, Q3 + k*IQR]
STRIP_PLOT_OUTLIER_IQR_MULTIPLIER = 3.0

def get_paths(base_dir: Path) -> Dict[str, Path]:
    """Return path dict relative to base_dir. Default output: asymmetry_tle."""
    base = Path(base_dir)
    return {
        "mni_micro_gam_dir": base / "derivatives" / "gam" / "mni_micro",
        "pyafq_gam_dir": base / "derivatives" / "gam" / "pyafq" / "HCP1065",
        "output_dir": base / "derivatives" / "analysis" / "asymmetry_tle",
        "clinical_path": base / "derivatives" / "metadata" / "clinical_penn_epilepsy_qsirecon.csv",
        "atlas_nifti_4s": base / "data" / "atlases" / "4S" / "tpl-MNI152NLin2009cAsym_atlas-4S156Parcels_res-01_dseg.nii.gz",
        "atlas_tsv_4s": base / "data" / "atlases" / "4S" / "atlas-4S156Parcels_dseg.tsv",
        "atlas_nifti_glasser": base / "data" / "atlases" / "Glasser" / "atlas-Glasser_space-MNI152NLin2009cAsym_res-1_dseg.nii.gz",
        "atlas_tsv_glasser": base / "data" / "atlases" / "Glasser" / "atlas-Glasser_dseg.tsv",
        "tract_metadata_path": base / "data" / "atlases" / "HCP1065" / "HCP1065_tract_metadata.csv",
        "endpoint_nii_dir": base / "data" / "atlases" / "HCP1065" / "endpoint_nii_bin",
        "scalar_labels_to_filenames": base / "data" / "metadata" / "scalar_labels_to_filenames.json",
        "scalar_to_human": base / "data" / "metadata" / "scalar_labels_to_human.json",
        "scalar_to_color": base / "data" / "metadata" / "scalar_labels_to_colors.json",
    }


def load_clinical(clinical_path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Load clinical CSV if path exists and has sub column. Otherwise None."""
    if not clinical_path or not Path(clinical_path).exists():
        return None
    try:
        df = pd.read_csv(clinical_path)
        if "sub" not in df.columns:
            return None
        return df
    except Exception:
        return None


def get_model_to_color(scalar_to_color_map: Dict[str, str]) -> Dict[str, str]:
    """Build model -> color from scalar colors (first scalar per model wins)."""
    model_to_color: Dict[str, str] = {}
    for scalar, color in scalar_to_color_map.items():
        model = scalar.split("_")[0] if "_" in scalar else scalar
        if model not in model_to_color:
            model_to_color[model] = color
    return model_to_color



def get_intervention_hemisphere_from_clinical(clinical_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    """Map subject -> 'L' or 'R' from intervention_laterality (left -> L, right -> R)."""
    out: Dict[str, str] = {}
    if clinical_df is None or "intervention_laterality" not in clinical_df.columns:
        return out
    for _, row in clinical_df.iterrows():
        sub = row.get("sub")
        if pd.isna(sub):
            continue
        lat = row.get("intervention_laterality")
        if pd.isna(lat):
            continue
        lat = str(lat).strip().lower()
        if lat == "left":
            out[str(sub)] = "L"
        elif lat == "right":
            out[str(sub)] = "R"
    return out


def get_left_right_tle_subjects(
    clinical_df: Optional[pd.DataFrame],
    restrict_to_subjects: Optional[set] = None,
) -> tuple:
    """Return (left_tle_subs, right_tle_subs) from seizure_lateralization."""
    left_tle_subs = []
    right_tle_subs = []
    if clinical_df is None or "seizure_lateralization" not in clinical_df.columns:
        return left_tle_subs, right_tle_subs
    for _, row in clinical_df.iterrows():
        sub = row.get("sub")
        if pd.isna(sub):
            continue
        sub = str(sub)
        if restrict_to_subjects is not None and sub not in restrict_to_subjects:
            continue
        lat = str(row.get("seizure_lateralization", "")).strip().lower() if pd.notna(row.get("seizure_lateralization")) else ""
        if lat in ("left", "left > right"):
            left_tle_subs.append(sub)
        elif lat in ("right", "right > left"):
            right_tle_subs.append(sub)
    return list(dict.fromkeys(left_tle_subs)), list(dict.fromkeys(right_tle_subs))


def get_4s_subcortical_regions(atlas_tsv_4s: Optional[Path]) -> List[str]:
    """Return 4S region labels where network_label == 'n/a' (subcortex) from atlas_tsv_4s."""
    if not atlas_tsv_4s or not Path(atlas_tsv_4s).exists():
        return []
    try:
        df = pd.read_csv(atlas_tsv_4s, sep="\t")
        if "label" not in df.columns or "network_label" not in df.columns:
            return []
        net = df["network_label"].fillna("n/a").astype(str).str.strip().str.lower()
        subcortex = df.loc[net.isin(("n/a", "nan", "")), "label"].tolist()
        return subcortex
    except Exception:
        return []


def get_scalar_labels(paths: Dict[str, Path]) -> List[str]:
    """Return scalar labels from scalar_labels_to_filenames.json keys, excluding EXCLUDED_SCALARS."""
    p = paths.get("scalar_labels_to_filenames")
    if not p or not Path(p).exists():
        return []
    try:
        with open(p, "r") as f:
            labels = list(json.load(f).keys())
        return sorted(s for s in labels if s not in EXCLUDED_SCALARS)
    except Exception:
        return []


def load_scalar_metadata(paths: Dict[str, Path]) -> Dict[str, Any]:
    """Load scalar_to_human, scalar_to_color, and scalar_labels_to_filenames. Missing files -> empty dict."""
    out: Dict[str, Any] = {
        "scalar_to_human": {},
        "scalar_to_color": {},
        "scalar_labels_to_filenames": {},
    }
    for key, path_key in [
        ("scalar_to_human", "scalar_to_human"),
        ("scalar_to_color", "scalar_to_color"),
        ("scalar_labels_to_filenames", "scalar_labels_to_filenames"),
    ]:
        p = paths.get(path_key)
        if p and Path(p).exists():
            try:
                with open(p, "r") as f:
                    out[key] = json.load(f)
            except Exception:
                pass
    return out
