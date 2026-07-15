import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
Microstructural asymmetry report: separate bar-plot and strip-plot figures (whole-brain and 2×2 by quadrant) for |Cohen's d| by scalar;
whole-brain map of mean |Cohen's d| per region (nilearn glass brain).
Uses tract_asymmetry (WM) and region_asymmetry_tle (GM: Glasser, 4S cortex, 4S subcortex; WM: HCP1065).
Atlases: 4S subcortex, 4S cortex (Schaefer100), Glasser, HCP1065 whole-tracts, HCP1065 along-tract thirds.
"""
from __future__ import annotations

import html as html_module
import json
import pickle
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths and config
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path("{project_root()}")
TRACT_ASYM_DIR = PROJECT_ROOT / "derivatives" / "analysis" / "tract_asymmetry"
REGION_ASYM_DIR = PROJECT_ROOT / "derivatives" / "analysis" / "region_asymmetry_tle"
ATLAS_TSV_4S = PROJECT_ROOT / "data" / "atlases" / "4S" / "atlas-4S156Parcels_dseg.tsv"
ATLAS_NII_4S = PROJECT_ROOT / "data" / "atlases" / "4S" / "tpl-MNI152NLin2009cAsym_atlas-4S156Parcels_res-01_dseg.nii.gz"
ATLAS_NII_4S_FALLBACK = PROJECT_ROOT / "data" / "atlases" / "4S" / "tpl-MNI152NLin2009cAsym_atlas-4S156Parcels_res-01_dseg.nii"
ATLAS_NII_GLASSER = PROJECT_ROOT / "data" / "atlases" / "Glasser" / "atlas-Glasser_space-MNI152NLin2009cAsym_res-1_dseg.nii.gz"
ATLAS_TSV_GLASSER = PROJECT_ROOT / "data" / "atlases" / "Glasser" / "atlas-Glasser_dseg.tsv"
GLASSER_ADDITIONAL_METADATA_PATH = (
    PROJECT_ROOT / "data" / "atlases" / "Glasser" / "glasser_additional_metadata.csv"
)
TRACT_METADATA_PATH = PROJECT_ROOT / "data" / "atlases" / "HCP1065" / "HCP1065_tract_metadata.csv"
ENDPOINT_NII_DIR = PROJECT_ROOT / "data" / "atlases" / "HCP1065" / "endpoint_nii_bin"
SCALAR_LABELS_PATH = PROJECT_ROOT / "data" / "metadata" / "scalar_labels_to_human.json"
SCALAR_COLORS_PATH = PROJECT_ROOT / "data" / "metadata" / "scalar_labels_to_colors.json"
OUTPUT_DIR = PROJECT_ROOT / "derivatives" / "analysis" / "microstructural_asymmetries"
FACTOR_Z_SCORES_DIR = PROJECT_ROOT / "derivatives" / "analysis" / "factor_z-scores" / "factor_z_scores"
FACTOR_SCORES_DIR = PROJECT_ROOT / "derivatives" / "analysis" / "factor_z-scores" / "factor_scores"

# Factor score asymmetry (F1–F3); F4 deprecated (isotropic).
FACTOR_DISPLAY_LABELS: Dict[int, str] = {
    1: "Overall",
    2: "Non-Gaussian",
    3: "Anisotropic",
}
DEPRECATED_FACTOR_INDICES = frozenset({4})
DEFAULT_FACTOR_INDICES: List[int] = [1, 2, 3]

# Wide factor Cohen's d columns (F1–F3) for summary tables / cohend_top{N} LaTeX.
FACTOR_COHENS_D_COLS: Dict[int, str] = {
    1: "factor_cohens_d_1",
    2: "factor_cohens_d_2",
    3: "factor_cohens_d_3",
}
COHEND_D_TEX_COL_KEYS: List[str] = ["mahal_cohens_d"] + [FACTOR_COHENS_D_COLS[k] for k in (1, 2, 3)]
COHEND_D_TEX_SUBHEADERS: List[str] = ["Mahalanobis", "Overall", "Non-Gaussian", "Anisotropic"]

# Fixed longtable widths (identical across all summary_*_cohend_top{N}.tex fragments).
# COHEND_TEX_TABLE_WIDTH_FRAC: overall table width as a fraction of \\textwidth (e.g. 0.90 = 10% narrower).
# Label / Cohen's d shares partition the table (must sum to 1.0); each Cohen's d subcolumn gets an equal share.
SUMMARY_COHEND_TEX_USEPACKAGE = (
    "% Requires \\usepackage{array,booktabs,longtable,xcolor} in the main document.\n"
)
COHEND_TEX_TABLE_WIDTH_FRAC = 0.98
COHEND_TEX_LABEL_SHARE = 0.40
COHEND_TEX_COHENS_BLOCK_SHARE = 0.60
COHEND_TEX_N_VALUE_COLS = 4


def _cohend_p_col_spec(align: str, width_frac: float) -> str:
    w = f"{width_frac:g}\\textwidth"
    return rf">{{\{align}\arraybackslash}}p{{{w}}}"


def _cohend_col_width_fracs() -> Tuple[float, float]:
    """Return (label_width, value_col_width) as fractions of \\textwidth."""
    table_w = COHEND_TEX_TABLE_WIDTH_FRAC
    label_w = table_w * COHEND_TEX_LABEL_SHARE
    value_w = table_w * COHEND_TEX_COHENS_BLOCK_SHARE / COHEND_TEX_N_VALUE_COLS
    return label_w, value_w


def _cohend_longtable_col_spec() -> str:
    label_w, value_w = _cohend_col_width_fracs()
    return (
        "@{}"
        + _cohend_p_col_spec("raggedright", label_w)
        + _cohend_p_col_spec("centering", value_w) * COHEND_TEX_N_VALUE_COLS
        + "@{}"
    )


COHEND_LONGTABLE_COL_SPEC = _cohend_longtable_col_spec()


def _cohend_subheader_tex(label: str) -> str:
    """Subcolumn title: centered, single line (no wrap)."""
    return rf"\mbox{{{_latex_escape(label)}}}"


def _cohend_region_label_tex(label: str) -> str:
    """Region / tract label: single line (may extend slightly over Cohen's d columns)."""
    return rf"\mbox{{{_latex_escape(label)}}}"


def _configure_matplotlib_georgia() -> None:
    """Prefer Georgia for all matplotlib text in this report (serif stack with system fallbacks)."""
    import matplotlib

    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = [
        "Georgia",
        "DejaVu Serif",
        "Liberation Serif",
        "Times New Roman",
        "Times",
        "Nimbus Roman",
    ]
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"


_configure_matplotlib_georgia()

# Excluded from analyses (mathematical dependencies, missing data, or lack of interpretability)
EXCLUDED_SCALARS = [
    "map_li", "map_am", "dti_txx", "dti_txy", "dti_txz",
    "dti_tyy", "dti_tyz", "dti_tzz", "dti_ha", "rdi_rd1", "rdi_rd2", "gqi_iso",
]


def _filter_excluded_scalars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "scalar" not in df.columns:
        return df
    return df[~df["scalar"].isin(EXCLUDED_SCALARS)].copy()

# Skip these tracts from tract-level volumetric asymmetry analyses (WM).
# Provided as tract labels with hemisphere suffix; we also derive tract bases without suffix.
EXCLUDED_VOLUMETRIC_ASYMMETRY_TRACTS = [
    "AF_L",
    "AF_R",
    "FAT_L",
    "FAT_R",
    "SLF3_L",
    "SLF3_R",
]
EXCLUDED_VOLUMETRIC_ASYMMETRY_TRACT_BASES = {
    t[:-2] for t in EXCLUDED_VOLUMETRIC_ASYMMETRY_TRACTS if t.endswith(("_L", "_R"))
}  # e.g. "AF_L" -> "AF"

MODEL_FALLBACK_COLORS = {
    "dki": "#7A297F",
    "dti": "#C43031",
    "gqi": "#FAA51A",
    "noddi": "#38489E",
    "map": "#289144",
    "rdi": "#C43031",
}

# NODDI → MAPMRI → DKI → DTI → GQI (``map_`` prefix = MAPMRI).
RECONSTRUCTION_MODEL_ORDER: Tuple[str, ...] = ("noddi", "map", "dki", "dti", "gqi")
FACTOR_LOADINGS_ORDERED_CSV = (
    PROJECT_ROOT
    / "derivatives"
    / "analysis"
    / "factor_analysis"
    / "All4_Combined"
    / "controls_All4_Combined_scalar_factor_loadings_ordered.csv"
)

# CIT168-style 4S subcortex parcel abbreviations -> prose labels (``summary_4s_subcortex_scalars.tex`` only).
FOUR_S_SUBCORTEX_TEX_ABBREV_LABELS: Dict[str, str] = {
    "Ca": "Caudate nucleus",
    "Pu": "Putamen",
    "GPe": "Globus pallidus external",
    "GPi": "Globus pallidus internal",
    "STH": "Subthalamic nucleus",
    "SNc_PBP_VTA": "Midbrain dopaminergic nuclei",
    "SNr": "Substantia nigra pars reticulata",
    "HTH": "Hypothalamus",
    "HN": "Habenular nucleus",
    "VeP": "Ventral pallidum",
    "NAC": "Nucleus accumbens",
    "RN": "Red nucleus",
    "MN": "Mammillary bodies",
    "EXA": "Extended amygdala",
}

# ThalamusHCP nuclei (atlas ``label`` base after LH-/RH- strip) -> table abbreviations.
THALAMUS_NUCLEUS_TEX_ABBREV: Dict[str, str] = {
    "Pulvinar": "Pu",
    "Anterior": "A",
    "Medio_Dorsal": "MD",
    "Ventral_Latero_Dorsal": "VLD",
    "Central_Lateral-Lateral_Posterior-Medial_Pulvinar": "CL-LP-PuM",
    "Ventral_Anterior": "VA",
    "Ventral_Latero_Ventral": "VLV",
}


def _format_thalamus_nucleus_tex_label(nucleus_base: str) -> str:
    """``Thalamus - {abbrev}`` for HCP thalamic nuclei in subcortex summary tables."""
    lab = str(nucleus_base).strip()
    abbrev = THALAMUS_NUCLEUS_TEX_ABBREV.get(lab, lab.replace("_", " "))
    return f"Thalamus - {abbrev}"


def _cohens_d_paired(ipsi_vals: List[float], contra_vals: List[float]) -> float:
    """Paired Cohen's d: mean(ipsi - contra) / std(ipsi - contra). NaN if n<2 or std=0."""
    ipsi = np.asarray(ipsi_vals, dtype=float)
    contra = np.asarray(contra_vals, dtype=float)
    valid = np.isfinite(ipsi) & np.isfinite(contra)
    diff = ipsi[valid] - contra[valid]
    n = len(diff)
    if n < 2:
        return float("nan")
    std_diff = float(np.std(diff, ddof=1))
    if std_diff <= 0:
        return float("nan")
    return float(np.mean(diff)) / std_diff


def _get_4s_subcortical_bases() -> set:
    """Set of region base names (suffix after LH_/RH_ or LH-/RH-) for 4S156 subcortical."""
    if not ATLAS_TSV_4S.exists():
        return set()
    try:
        df = pd.read_csv(ATLAS_TSV_4S, sep="\t")
        if "label" not in df.columns or "network_label" not in df.columns:
            return set()
        net = df["network_label"].fillna("n/a").astype(str).str.strip().str.lower()
        subcort = df.loc[net.isin(("n/a", "nan", "")), "label"].tolist()
        bases = set()
        for label in subcort:
            if label.startswith("LH_"):
                bases.add(label[3:])
            elif label.startswith("LH-"):
                bases.add(label[3:])
            elif label.startswith("RH_"):
                bases.add(label[3:])
            elif label.startswith("RH-"):
                bases.add(label[3:])
        return bases
    except Exception:
        return set()


def _get_4s_cortical_bases() -> set:
    """Set of region base names (suffix after LH_/RH_) for 4S156 cortical (Schaefer100)."""
    if not ATLAS_TSV_4S.exists():
        return set()
    try:
        df = pd.read_csv(ATLAS_TSV_4S, sep="\t")
        if "label" not in df.columns or "network_label" not in df.columns:
            return set()
        net = df["network_label"].fillna("n/a").astype(str).str.strip().str.lower()
        cort = df.loc[~net.isin(("n/a", "nan", "")), "label"].tolist()
        bases = set()
        for label in cort:
            if label.startswith("LH_"):
                bases.add(label[3:])
            elif label.startswith("LH-"):
                bases.add(label[3:])
            elif label.startswith("RH_"):
                bases.add(label[3:])
            elif label.startswith("RH-"):
                bases.add(label[3:])
        return bases
    except Exception:
        return set()


def _get_glasser_bases() -> set:
    """Set of region base names for Glasser (label suffix after Left_/Right_, to match region_asymmetry_tle CSV)."""
    if not ATLAS_TSV_GLASSER.exists():
        return set()
    try:
        df = pd.read_csv(ATLAS_TSV_GLASSER, sep="\t")
        if "label" not in df.columns:
            return set()
        bases = set()
        for label in df["label"].dropna().astype(str):
            label = label.strip()
            if label.startswith("Left_"):
                bases.add(label[5:].strip())
            elif label.startswith("Right_"):
                bases.add(label[6:].strip())
        return bases
    except Exception:
        return set()


def load_tract_asymmetry() -> pd.DataFrame:
    """Load all tract_asymmetry asym_scalars CSVs into one DataFrame. Adds roi_id=(tract,segment), roi_type='wm'."""
    rows: List[dict] = []
    for sub_dir in TRACT_ASYM_DIR.iterdir():
        if not sub_dir.is_dir():
            continue
        csv_path = sub_dir / f"{sub_dir.name}_asym_scalars.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty or "tract" not in df.columns or "segment" not in df.columns:
            continue
        df = df.copy()
        df["roi_id"] = df["tract"] + "_" + df["segment"].astype(str)
        df["roi_type"] = "wm"
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_region_asymmetry(
    subcortical_bases: set,
    cortical_bases_4s: set,
    glasser_bases: set,
) -> pd.DataFrame:
    """Load all region_asymmetry_tle asym_regions CSVs. Use stat=mean only. roi_id=region, roi_type=cortical_gm or subcortical_gm, atlas=4s_subcortex|4s_cortex|glasser."""
    rows: List[dict] = []
    for sub_dir in REGION_ASYM_DIR.iterdir():
        if not sub_dir.is_dir():
            continue
        csv_path = sub_dir / f"{sub_dir.name}_asym_regions.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty or "region" not in df.columns:
            continue
        if "stat" in df.columns:
            df = df[df["stat"] == "mean"].copy()
        df = df.copy()
        df["roi_id"] = df["region"].astype(str)

        def _atlas(r: str) -> str:
            if r in subcortical_bases:
                return "4s_subcortex"
            if r in cortical_bases_4s:
                return "4s_cortex"
            if r in glasser_bases:
                return "glasser"
            return ""

        def _roi_type(r: str) -> str:
            if r in subcortical_bases:
                return "subcortical_gm"
            return "cortical_gm"

        df["atlas"] = df["region"].astype(str).apply(_atlas)
        df["roi_type"] = df["region"].astype(str).apply(_roi_type)
        # Drop rows with no atlas (e.g. HCP1065 regions from region_asymmetry_tle are in tract_df)
        df = df[df["atlas"] != ""].copy()
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _load_tract_metadata() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load HCP1065 tract metadata. Returns (metadata_df, tract_base_to_type) for WM association/projection split."""
    tract_base_to_type: Dict[str, str] = {}
    if not TRACT_METADATA_PATH.exists():
        return pd.DataFrame(), tract_base_to_type
    try:
        meta = pd.read_csv(TRACT_METADATA_PATH)
        if "label" not in meta.columns or "type" not in meta.columns:
            return meta, tract_base_to_type
        for _, row in meta.iterrows():
            label = str(row["label"]).strip()
            ttype = str(row["type"]).strip().lower()
            if label.endswith("_L") or label.endswith("_R"):
                base = label[:-2]
                if ttype in ("association", "projection") and base not in tract_base_to_type:
                    tract_base_to_type[base] = ttype
        return meta, tract_base_to_type
    except Exception:
        return pd.DataFrame(), tract_base_to_type


def _wm_roi_to_tract_segment(roi_id: str) -> Tuple[str, str]:
    """Parse WM roi_id (e.g. 'AF_core', 'C_FP_end1') into (tract_base, segment)."""
    if "_" not in roi_id:
        return roi_id, ""
    return roi_id.rsplit("_", 1)[0], roi_id.rsplit("_", 1)[1]


def _wm_roi_tract_base_key(roi_id: str) -> str:
    """Hemisphere-agnostic HCP1065 tract key (``AF_L_core`` / ``AF_core`` -> ``AF``)."""
    tract_with_hemi, _ = _wm_roi_to_tract_segment(str(roi_id).strip())
    t = str(tract_with_hemi).strip()
    if t.endswith("_L") or t.endswith("_R"):
        return t[:-2]
    return t


def _is_excluded_volumetric_asymmetry_wm_roi(roi_id: str) -> bool:
    """True for WM roi_id / label belonging to EXCLUDED_VOLUMETRIC_ASYMMETRY_TRACTS."""
    return _wm_roi_tract_base_key(roi_id) in EXCLUDED_VOLUMETRIC_ASYMMETRY_TRACT_BASES


def _exclude_volumetric_asymmetry_tracts(df: pd.DataFrame) -> pd.DataFrame:
    """Drop WM rows whose tract base is in EXCLUDED_VOLUMETRIC_ASYMMETRY_TRACT_BASES."""
    if df.empty:
        return df
    if "roi_type" not in df.columns or "roi_id" not in df.columns:
        return df
    mask_wm = df["roi_type"] == "wm"
    if not mask_wm.any():
        return df
    excluded = df["roi_id"].map(
        lambda rid: _is_excluded_volumetric_asymmetry_wm_roi(str(rid)) if pd.notna(rid) else False
    )
    return df[~(mask_wm & excluded)].copy()


def get_combined_long(
    tract_df: pd.DataFrame,
    region_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build long table (sub, roi_id, roi_type, scalar, atlas, ipsi_mean_z, contra_mean_z) for summary stats."""
    combined = []
    if not tract_df.empty:
        part = tract_df[["sub", "roi_id", "roi_type", "scalar", "ipsi_mean_z", "contra_mean_z"]].copy()
        part["atlas"] = ""
        combined.append(part)
    if not region_df.empty:
        cols = ["sub", "roi_id", "roi_type", "scalar", "ipsi_mean_z", "contra_mean_z"]
        if "atlas" in region_df.columns:
            part = region_df[cols + ["atlas"]].copy()
        else:
            part = region_df[cols].copy()
            part["atlas"] = ""
        combined.append(part)
    if not combined:
        return pd.DataFrame()
    full = pd.concat(combined, ignore_index=True)
    if "atlas" not in full.columns:
        full["atlas"] = ""
    return full


def compute_cohens_d_per_roi_scalar(
    tract_df: pd.DataFrame,
    region_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute paired Cohen's d per (roi_id, roi_type, scalar). Returns long table with columns roi_id, roi_type, scalar, cohens_d; region rows have atlas (4s_subcortex|4s_cortex|glasser)."""
    full = get_combined_long(tract_df, region_df)
    if full.empty:
        return pd.DataFrame(columns=["roi_id", "roi_type", "scalar", "cohens_d", "atlas"])
    return _filter_excluded_scalars(_compute_cohens_d_from_combined(full))


def _compute_cohens_d_from_combined(full: pd.DataFrame) -> pd.DataFrame:
    """Compute Cohen's d per (roi_id, roi_type, scalar); preserve atlas if present."""
    results = []
    group_cols = ["roi_id", "roi_type", "scalar"]
    if "atlas" in full.columns:
        group_cols = group_cols + ["atlas"]
    for key, grp in full.groupby(group_cols):
        if isinstance(key, tuple):
            roi_id, roi_type, scalar = key[0], key[1], key[2]
            atlas = key[3] if len(key) > 3 else ""
        else:
            roi_id, roi_type, scalar = key
            atlas = ""
        ipsi = grp["ipsi_mean_z"].dropna().tolist()
        contra = grp["contra_mean_z"].dropna().tolist()
        if len(ipsi) != len(contra) or len(ipsi) < 2:
            continue
        d = _cohens_d_paired(ipsi, contra)
        if np.isfinite(d):
            row = {"roi_id": roi_id, "roi_type": roi_type, "scalar": scalar, "cohens_d": d}
            if "atlas" in full.columns:
                row["atlas"] = atlas
            results.append(row)
    return pd.DataFrame(results)


def add_quadrant_column(cohens_df: pd.DataFrame, tract_base_to_type: Dict[str, str]) -> pd.DataFrame:
    """Add column 'quadrant': cortex, subcortex, association_wm, projection_wm (or None for unclassified WM)."""
    if cohens_df.empty:
        return cohens_df
    df = cohens_df.copy()
    def _quadrant(row):
        if row["roi_type"] == "cortical_gm":
            return "cortex"
        if row["roi_type"] == "subcortical_gm":
            return "subcortex"
        if row["roi_type"] == "wm":
            tract_base, _ = _wm_roi_to_tract_segment(row["roi_id"])
            ttype = tract_base_to_type.get(tract_base, "")
            if ttype == "association":
                return "association_wm"
            if ttype == "projection":
                return "projection_wm"
        return None
    df["quadrant"] = df.apply(_quadrant, axis=1)
    return df


def get_quadrant_data(
    cohens_df: pd.DataFrame,
    tract_base_to_type: Dict[str, str],
) -> Dict[str, List]:
    """
    Split cohens_df into 2x2 quadrants: cortex, subcortex, association, projection.
    Returns dict with keys cortex, subcortex, association, projection.
    - cortex/subcortex: list of (region_label, mean_abs_d) sorted by value desc.
    - association/projection: list of (tract, segment, mean_abs_d) sorted by value desc.
    """
    out: Dict[str, List] = {"cortex": [], "subcortex": [], "association": [], "projection": []}
    if cohens_df.empty:
        return out
    by_roi = cohens_df.groupby("roi_id")["cohens_d"].apply(lambda s: np.abs(s).mean())
    for roi_id, mean_abs_d in by_roi.items():
        if pd.isna(mean_abs_d):
            continue
        row = cohens_df[cohens_df["roi_id"] == roi_id].iloc[0]
        roi_type = row["roi_type"]
        if roi_type == "cortical_gm":
            out["cortex"].append((str(roi_id), float(mean_abs_d)))
        elif roi_type == "subcortical_gm":
            out["subcortex"].append((str(roi_id), float(mean_abs_d)))
        elif roi_type == "wm":
            tract_base, segment = _wm_roi_to_tract_segment(roi_id)
            ttype = tract_base_to_type.get(tract_base, "")
            if ttype == "association":
                out["association"].append((tract_base, segment, float(mean_abs_d)))
            elif ttype == "projection":
                out["projection"].append((tract_base, segment, float(mean_abs_d)))
    for k in ("cortex", "subcortex"):
        out[k] = sorted(out[k], key=lambda x: x[1], reverse=True)
    for k in ("association", "projection"):
        out[k] = sorted(out[k], key=lambda x: x[2], reverse=True)
    return out


def _label_lh_rh_variants(label: str) -> List[str]:
    """Atlas label may use LH- vs LH_; add both for lookup."""
    s = str(label).strip()
    out: List[str] = [s]
    if s.startswith("LH-"):
        out.append("LH_" + s[3:])
    elif s.startswith("LH_"):
        out.append("LH-" + s[3:])
    if s.startswith("RH-"):
        out.append("RH_" + s[3:])
    elif s.startswith("RH_"):
        out.append("RH-" + s[3:])
    return list(dict.fromkeys(out))


def _load_4s_subcortex_labels_full() -> Set[str]:
    """All 4S156 subcortex parcel names (with LH-/LH_ variants) for factor z-score column matching."""
    found: Set[str] = set()
    if not ATLAS_TSV_4S.exists():
        return found
    try:
        df = pd.read_csv(ATLAS_TSV_4S, sep="\t")
        if "label" not in df.columns or "network_label" not in df.columns:
            return found
        net = df["network_label"].fillna("n/a").astype(str).str.strip().str.lower()
        sub_mask = net.isin(("n/a", "nan", ""))
        for lab in df.loc[sub_mask, "label"].dropna().astype(str):
            for v in _label_lh_rh_variants(lab):
                found.add(v)
    except Exception:
        pass
    return found


def _load_4s_cortex_labels_full() -> Set[str]:
    """All 4S156 cortical (Schaefer) parcel names with hemisphere variants."""
    found: Set[str] = set()
    if not ATLAS_TSV_4S.exists():
        return found
    try:
        df = pd.read_csv(ATLAS_TSV_4S, sep="\t")
        if "label" not in df.columns or "network_label" not in df.columns:
            return found
        net = df["network_label"].fillna("n/a").astype(str).str.strip().str.lower()
        ctx_mask = ~net.isin(("n/a", "nan", ""))
        for lab in df.loc[ctx_mask, "label"].dropna().astype(str):
            for v in _label_lh_rh_variants(lab):
                found.add(v)
    except Exception:
        pass
    return found


def _load_glasser_labels_full() -> Set[str]:
    """Glasser parcel labels (as in atlas-Glasser_dseg.tsv)."""
    found: Set[str] = set()
    if not ATLAS_TSV_GLASSER.exists():
        return found
    try:
        df = pd.read_csv(ATLAS_TSV_GLASSER, sep="\t")
        if "label" not in df.columns:
            return found
        for lab in df["label"].dropna().astype(str):
            found.add(lab.strip())
    except Exception:
        pass
    return found


def _classify_factor_z_column(
    col: str,
    labels_4s_sub: Set[str],
    labels_4s_ctx: Set[str],
    labels_glasser: Set[str],
    tract_base_to_type: Dict[str, str],
) -> Optional[str]:
    """
    Map one column name from epilepsy_F*_z_scores.csv to quadrant tissue class.
    Returns cortex_gm, subcortex_gm, association_wm, projection_wm, or None.
    """
    c = str(col).strip()
    if c == "subject":
        return None

    # WM tract × segment (HCP1065 thirds): TR_S_R_end-S, UF_L_core, ...
    if c.endswith("_core") or "_end-" in c:
        if c.endswith("_core"):
            tract_full = c[: -len("_core")]
        else:
            tract_full = c.rsplit("_", 1)[0]
        if tract_full.endswith("_L") or tract_full.endswith("_R"):
            tb = tract_full[:-2]
            ttype = tract_base_to_type.get(tb, "")
            if ttype == "association":
                return "association_wm"
            if ttype == "projection":
                return "projection_wm"
        return None

    for key in _label_lh_rh_variants(c):
        if key in labels_4s_sub:
            return "subcortex_gm"
    for key in _label_lh_rh_variants(c):
        if key in labels_4s_ctx or key in labels_glasser:
            return "cortex_gm"
    return None


def build_epilepsy_factor_z_mean_by_tissue(
    tract_base_to_type: Dict[str, str],
) -> Optional[pd.DataFrame]:
    """
    Load epilepsy_F1..F4 z-score CSVs; per subject, factor, and tissue class, mean z across ROIs in that class.

    Returns long DataFrame with columns: subject, factor, tissue, mean_z; or None if no data.
    """
    if not FACTOR_Z_SCORES_DIR.is_dir():
        return None
    labels_sub = _load_4s_subcortex_labels_full()
    labels_ctx = _load_4s_cortex_labels_full()
    labels_gl = _load_glasser_labels_full()
    parts: List[pd.DataFrame] = []
    for fac in ("F1", "F2", "F3", "F4"):
        path = FACTOR_Z_SCORES_DIR / f"epilepsy_{fac}_z_scores.csv"
        if not path.exists():
            continue
        try:
            wide = pd.read_csv(path)
        except Exception:
            continue
        if wide.empty or "subject" not in wide.columns:
            continue
        long = wide.melt(id_vars=["subject"], var_name="roi", value_name="z")
        long["tissue"] = long["roi"].astype(str).apply(
            lambda r: _classify_factor_z_column(
                r, labels_sub, labels_ctx, labels_gl, tract_base_to_type
            )
        )
        long = long.dropna(subset=["tissue", "z"])
        if long.empty:
            continue
        g = long.groupby(["subject", "tissue"], as_index=False)["z"].mean()
        g["factor"] = fac
        g = g.rename(columns={"z": "mean_z"})
        parts.append(g)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


_BROAD_TISSUE_MAP = {
    "cortex_gm": "grey_matter",
    "subcortex_gm": "grey_matter",
    "association_wm": "white_matter",
    "projection_wm": "white_matter",
}


def build_epilepsy_factor_z_mean_by_gm_wm(
    tract_base_to_type: Dict[str, str],
) -> Optional[pd.DataFrame]:
    """
    Same sources as build_epilepsy_factor_z_mean_by_tissue, but cortex + subcortex GM are pooled
    into grey_matter and association + projection WM into white_matter (mean z across all ROIs
    in that broad class per subject and factor).
    """
    if not FACTOR_Z_SCORES_DIR.is_dir():
        return None
    labels_sub = _load_4s_subcortex_labels_full()
    labels_ctx = _load_4s_cortex_labels_full()
    labels_gl = _load_glasser_labels_full()
    parts: List[pd.DataFrame] = []
    for fac in ("F1", "F2", "F3", "F4"):
        path = FACTOR_Z_SCORES_DIR / f"epilepsy_{fac}_z_scores.csv"
        if not path.exists():
            continue
        try:
            wide = pd.read_csv(path)
        except Exception:
            continue
        if wide.empty or "subject" not in wide.columns:
            continue
        long = wide.melt(id_vars=["subject"], var_name="roi", value_name="z")
        fine = long["roi"].astype(str).apply(
            lambda r: _classify_factor_z_column(
                r, labels_sub, labels_ctx, labels_gl, tract_base_to_type
            )
        )
        long["tissue"] = fine.map(_BROAD_TISSUE_MAP)
        long = long.dropna(subset=["tissue", "z"])
        if long.empty:
            continue
        g = long.groupby(["subject", "tissue"], as_index=False)["z"].mean()
        g["factor"] = fac
        g = g.rename(columns={"z": "mean_z"})
        parts.append(g)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def _mean_sem_1d(values: np.ndarray) -> Tuple[float, float]:
    """Return (mean, SEM) for finite values; SEM 0 if n < 2; (nan, nan) if empty."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n == 0:
        return float("nan"), float("nan")
    m = float(np.mean(a))
    if n < 2:
        return m, 0.0
    sem = float(np.std(a, ddof=1) / np.sqrt(n))
    return m, sem


def _factor_z_subplot_black_frame(ax) -> None:
    """Draw a full black rectangle around the axes (all four spines visible)."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)


def plot_factor_z_mean_sem_bars_faceted(
    df: pd.DataFrame,
    out_path: Path,
    tissue_order: Sequence[str],
    tissue_labels: Sequence[str],
    suptitle: str,
) -> None:
    """
    One row of subplots (one per tissue category): neutral bars, mean factor z ± SEM across subjects;
    x = Factor 1–4. Per-subject values in df are mean z across ROIs in that tissue bucket.
    """
    import matplotlib.pyplot as plt

    order_f = ["F1", "F2", "F3", "F4"]
    if len(tissue_order) != len(tissue_labels):
        return

    plot_df = df[df["factor"].isin(order_f) & df["tissue"].isin(tissue_order)].copy()
    if plot_df.empty:
        return

    n_p = len(tissue_order)
    fig_w = max(3.5 * n_p, 5.0)
    fig, axes = plt.subplots(1, n_p, figsize=(fig_w, 4.2), sharey=True)
    if n_p == 1:
        axes = np.asarray([axes])
    x = np.arange(len(order_f))
    xtick_labels = [f"Factor {i}" for i in range(1, len(order_f) + 1)]
    bar_face = "#7a7a7a"
    bar_edge = "0.2"

    for ax, tcode, tlab in zip(axes.flat, tissue_order, tissue_labels):
        means: List[float] = []
        sems: List[float] = []
        for fac in order_f:
            vals = plot_df.loc[
                (plot_df["factor"] == fac) & (plot_df["tissue"] == tcode), "mean_z"
            ].to_numpy(dtype=float)
            m, s = _mean_sem_1d(vals)
            means.append(m)
            sems.append(s)
        ax.bar(
            x,
            means,
            yerr=sems,
            color=bar_face,
            capsize=4,
            width=0.65,
            edgecolor=bar_edge,
            linewidth=0.6,
            error_kw={"linewidth": 1.0, "ecolor": "0.35"},
            zorder=2,
        )
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=14, rotation=45, ha="right")
        ax.set_title(tlab, fontsize=14)
        _factor_z_subplot_black_frame(ax)

    axes.flat[0].set_ylabel("Factor z-score", fontsize=14)
    fig.suptitle(suptitle, fontsize=12, y=1.06)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_factor_z_mean_sem_bars_by_tissue(df: pd.DataFrame, out_path: Path) -> None:
    """1×4 panels: four fine tissue classes (neutral bars)."""
    plot_factor_z_mean_sem_bars_faceted(
        df,
        out_path,
        tissue_order=["cortex_gm", "subcortex_gm", "association_wm", "projection_wm"],
        tissue_labels=["Cortex GM", "Subcortex GM", "Association WM", "Projection WM"],
        suptitle=(
            "Epilepsy: mean factor z-scores by factor and tissue "
            "(mean ± SEM across subjects; per subject, mean across ROIs)"
        ),
    )


def plot_factor_z_mean_sem_bars_by_gm_wm(df: pd.DataFrame, out_path: Path) -> None:
    """
    1×3 panels matching ``plot_tissue_pc1_correlations_with_wholebrain`` (GM/WM + diff) styling:
    Grey Matter and White Matter mean factor z ± SEM; third panel: |z_GM| − |z_WM| per subject,
    then mean ± SEM (symmetric y-axis around 0).
    """
    import matplotlib.pyplot as plt

    order_f = ["F1", "F2", "F3", "F4"]
    # Align with factor_analysis.plot_tissue_pc1_correlations_with_wholebrain (combined + diff row).
    title_fs = 16
    axis_fs = 14
    tick_fs = 10
    ylabel_fs = 14
    ncols = 3
    fig_width = max(7.0, 3.0 * ncols)
    fig_height = 5.5

    plot_df = df[df["factor"].isin(order_f) & df["tissue"].isin(["grey_matter", "white_matter"])].copy()
    if plot_df.empty:
        return

    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, fig_height))
    axes = np.atleast_1d(axes).ravel()
    axes[1].sharey(axes[0])

    x = np.arange(len(order_f))
    bar_kw = {"color": "#9E9E9E", "edgecolor": "black", "linewidth": 0.3, "alpha": 0.9}
    err_kw = {"elinewidth": 1.0, "ecolor": "0.25"}

    for i, (tcode, tlab) in enumerate(
        [("grey_matter", "Grey Matter"), ("white_matter", "White Matter")]
    ):
        ax = axes[i]
        means: List[float] = []
        sems: List[float] = []
        for fac in order_f:
            vals = plot_df.loc[
                (plot_df["factor"] == fac) & (plot_df["tissue"] == tcode), "mean_z"
            ].to_numpy(dtype=float)
            m, s = _mean_sem_1d(vals)
            means.append(m)
            sems.append(s)
        ax.bar(
            x,
            means,
            yerr=sems,
            capsize=4,
            width=0.65,
            error_kw=err_kw,
            **bar_kw,
        )
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(order_f, fontsize=tick_fs)
        ax.set_title(tlab, fontsize=title_fs)
        ax.set_xlabel("Factor", fontsize=axis_fs)
        if i == 0:
            ax.set_ylabel("Factor z-score", fontsize=ylabel_fs)
        else:
            ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=tick_fs)
        _factor_z_subplot_black_frame(ax)

    axd = axes[2]
    grey = plot_df[plot_df["tissue"] == "grey_matter"][["subject", "factor", "mean_z"]]
    white = plot_df[plot_df["tissue"] == "white_matter"][["subject", "factor", "mean_z"]]
    merged = grey.merge(white, on=["subject", "factor"], how="inner", suffixes=("_g", "_w"))
    if not merged.empty:
        merged["abs_diff"] = np.abs(merged["mean_z_g"].astype(float)) - np.abs(
            merged["mean_z_w"].astype(float)
        )
    d_means: List[float] = []
    d_sems: List[float] = []
    if not merged.empty:
        for fac in order_f:
            vals = merged.loc[merged["factor"] == fac, "abs_diff"].to_numpy(dtype=float)
            m, s = _mean_sem_1d(vals)
            d_means.append(m)
            d_sems.append(s)

    if merged.empty or not np.any(np.isfinite(np.array(d_means, dtype=float))):
        axd.axis("off")
    else:
        marr = np.asarray(d_means, dtype=float)
        sarr = np.asarray(d_sems, dtype=float)
        axd.bar(
            x,
            d_means,
            yerr=d_sems,
            capsize=4,
            width=0.65,
            error_kw=err_kw,
            **bar_kw,
        )
        axd.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        axd.set_title("Grey-white matter differences", fontsize=title_fs)
        axd.set_xlabel("Factor", fontsize=axis_fs)
        axd.set_ylabel("|Factor z-score| difference", fontsize=ylabel_fs)
        axd.set_xticks(x)
        axd.set_xticklabels(order_f, fontsize=tick_fs)
        axd.tick_params(axis="x", labelsize=tick_fs)
        axd.grid(True, axis="y", alpha=0.3)
        _factor_z_subplot_black_frame(axd)
        span = np.concatenate(
            [
                marr + sarr,
                marr - sarr,
                marr,
                [0.0],
            ]
        )
        span = span[np.isfinite(span)]
        ymax = float(np.nanmax(np.abs(span))) if span.size else 0.1
        ymax = max(0.1, ymax * 1.15)
        axd.set_ylim(-ymax, ymax)

    fig.suptitle(
        "Epilepsy: mean factor z-scores — grey vs white matter "
        "(mean ± SEM across subjects; per subject, mean across pooled ROIs)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Top N for atlas summary tables (always top 20, never top 5 and bottom 5)
ATLAS_TOP_N = 20
# Top N regions in truncated LaTeX summary fragments (*_top25.tex, etc.).
SUMMARY_TEX_TOP_N = 25


def get_atlas_data(
    cohens_df: pd.DataFrame,
    tract_base_to_type: Dict[str, str],
) -> Dict[str, List]:
    """
    Per-atlas data for summary tables and pkl. Keys: 4s_subcortex, 4s_cortex, glasser, hcp1065_whole, hcp1065_thirds.
    - GM atlases: list of (region_label, mean_abs_d) sorted desc.
    - hcp1065_whole: list of (tract, mean_abs_d) sorted desc (mean across segments per tract).
    - hcp1065_thirds: list of (tract, segment, mean_abs_d) sorted desc.
    """
    out: Dict[str, List] = {
        "4s_subcortex": [],
        "4s_cortex": [],
        "glasser": [],
        "hcp1065_whole": [],
        "hcp1065_thirds": [],
    }
    if cohens_df.empty:
        return out
    if "atlas" in cohens_df.columns:
        for atlas in ("4s_subcortex", "4s_cortex", "glasser"):
            sub = cohens_df[cohens_df["atlas"] == atlas]
            if sub.empty:
                continue
            by_roi = sub.groupby("roi_id")["cohens_d"].apply(lambda s: np.abs(s).mean())
            for roi_id, mean_abs_d in by_roi.items():
                if pd.isna(mean_abs_d):
                    continue
                out[atlas].append((str(roi_id), float(mean_abs_d)))
            out[atlas] = sorted(out[atlas], key=lambda x: x[1], reverse=True)

    wm = cohens_df[cohens_df["roi_type"] == "wm"] if not cohens_df.empty else pd.DataFrame()
    if not wm.empty:
        by_roi = wm.groupby("roi_id")["cohens_d"].apply(lambda s: np.abs(s).mean())
        # Along-tract thirds: (tract, segment, mean_abs_d)
        for roi_id, mean_abs_d in by_roi.items():
            if pd.isna(mean_abs_d):
                continue
            if _is_excluded_volumetric_asymmetry_wm_roi(str(roi_id)):
                continue
            tract_base, segment = _wm_roi_to_tract_segment(roi_id)
            out["hcp1065_thirds"].append((tract_base, segment, float(mean_abs_d)))
        out["hcp1065_thirds"] = sorted(out["hcp1065_thirds"], key=lambda x: x[2], reverse=True)
        # Whole-tract: aggregate by tract (mean of mean_abs_d across segments)
        by_tract = wm.copy()
        by_tract["tract_base"] = by_tract["roi_id"].apply(lambda x: _wm_roi_to_tract_segment(x)[0])
        tract_means = by_tract.groupby("tract_base")["cohens_d"].apply(lambda s: np.abs(s).mean())
        for tract_base, mean_abs_d in tract_means.items():
            if pd.isna(mean_abs_d):
                continue
            if _is_excluded_volumetric_asymmetry_wm_roi(str(tract_base)):
                continue
            out["hcp1065_whole"].append((str(tract_base), float(mean_abs_d)))
        out["hcp1065_whole"] = sorted(out["hcp1065_whole"], key=lambda x: x[1], reverse=True)
    return out


def _roi_stats_for_summary(grp: pd.DataFrame) -> dict:
    ipsi = grp["ipsi_mean_z"].dropna()
    contra = grp["contra_mean_z"].dropna()
    diff = grp["ipsi_mean_z"] - grp["contra_mean_z"]
    return {
        "mean_ipsi": float(ipsi.mean()) if len(ipsi) else float("nan"),
        "mean_contra": float(contra.mean()) if len(contra) else float("nan"),
        "mean_asymmetry": float(diff.mean()) if len(diff) and diff.notna().any() else float("nan"),
    }


def _cohens_d_summary_by_label(
    cohens_sub: pd.DataFrame,
    *,
    label_col: str,
) -> pd.DataFrame:
    """Per ``label_col``: mean signed d, mean |d|, signed d at peak |d|, peak |d|, peak scalar."""
    cohens_sub = _filter_excluded_scalars(cohens_sub)
    empty_cols = [
        "label",
        "mean_cohens_d",
        "mean_abs_cohens_d",
        "max_cohens_d",
        "max_abs_cohens_d",
        "max_abs_cohens_scalar",
    ]
    if cohens_sub.empty or label_col not in cohens_sub.columns:
        return pd.DataFrame(columns=empty_cols)
    rows: List[dict] = []
    for label, grp in cohens_sub.groupby(label_col):
        g = grp.dropna(subset=["cohens_d"])
        if g.empty:
            rows.append(
                {
                    "label": label,
                    "mean_cohens_d": float("nan"),
                    "mean_abs_cohens_d": float("nan"),
                    "max_cohens_d": float("nan"),
                    "max_abs_cohens_d": float("nan"),
                    "max_abs_cohens_scalar": None,
                }
            )
            continue
        signed_d = g["cohens_d"]
        abs_d = signed_d.abs()
        idx_max = abs_d.idxmax()
        rows.append(
            {
                "label": label,
                "mean_cohens_d": float(signed_d.mean()),
                "mean_abs_cohens_d": float(abs_d.mean()),
                "max_cohens_d": float(signed_d.loc[idx_max]),
                "max_abs_cohens_d": float(abs_d.loc[idx_max]),
                "max_abs_cohens_scalar": str(g.loc[idx_max, "scalar"]),
            }
        )
    return pd.DataFrame(rows)


def _factor_z_segment_to_asym_segment(segment: str) -> str:
    """Factor z segment suffix -> tract-asymmetry ``segment`` (``end-A`` -> ``A``; ``core`` unchanged)."""
    s = str(segment).strip()
    if s.startswith("end-"):
        return s[4:]
    return s


def _factor_z_wm_column_to_roi_label(col: str) -> str:
    """``ILF_L_core`` / ``ILF_L_end-A`` -> ``ILF_core`` / ``ILF_A`` (tract asymmetry ``roi_id``)."""
    c = str(col).strip()
    if "_L_" in c:
        tract_base, segment = c.split("_L_", 1)
        return f"{tract_base}_{_factor_z_segment_to_asym_segment(segment)}"
    if c.endswith("_core"):
        tract_hemi = c[: -len("_core")]
        if tract_hemi.endswith("_L"):
            return f"{tract_hemi[:-2]}_core"
    return c


def _factor_z_pair_to_summary_label(pair: Dict[str, object]) -> str:
    """Map a factor-z ROI pair to the summary table ``label`` (roi_id / tract_base)."""
    q = str(pair["quadrant"])
    lc = str(pair["left_col"])
    if q == "glasser_cortex":
        return lc[5:] if lc.startswith("Left_") else lc
    if q in ("wm_association", "wm_projection"):
        return _factor_z_wm_column_to_roi_label(lc)
    for pref in ("LH_", "LH-"):
        if lc.startswith(pref):
            return lc[len(pref) :]
    return lc


def _factor_z_pair_to_atlas_roi_type(pair: Dict[str, object]) -> Tuple[str, str]:
    q = str(pair["quadrant"])
    if q == "glasser_cortex":
        return "glasser", "cortical_gm"
    if q == "4s_subcortex":
        return "4s_subcortex", "subcortical_gm"
    if q in ("wm_association", "wm_projection"):
        return "", "wm"
    return "", "cortical_gm"


def compute_factor_z_cohens_df(
    tract_base_to_type: Dict[str, str],
    factor_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Paired Cohen's d per summary label and factor from control-z-scored factor scores."""
    from microstructural_asymmetry_report_factor_z import (
        NON_ROI_COLUMNS,
        build_roi_pairs,
        load_laterality_map,
        normalize_subject_id,
        zscore_epilepsy_vs_controls,
    )
    from microstructural_asymmetry_report_mahalanobis import (
        _get_4s_subcortical_base_to_labels,
        _get_glasser_bases,
    )

    if factor_indices is None:
        factor_indices = [k for k in DEFAULT_FACTOR_INDICES if k not in DEPRECATED_FACTOR_INDICES]

    probe = FACTOR_SCORES_DIR / "epilepsy_F1_scores.csv"
    if not probe.exists():
        return pd.DataFrame(
            columns=["label", "factor_index", "factor_name", "cohens_d", "atlas", "roi_type"]
        )

    glasser_bases = _get_glasser_bases()
    subcortical_base_to_labels = _get_4s_subcortical_base_to_labels()
    all_cols = set(pd.read_csv(probe, nrows=0).columns) - NON_ROI_COLUMNS
    pairs, _used = build_roi_pairs(
        all_cols, glasser_bases, subcortical_base_to_labels, tract_base_to_type
    )
    lat_map = load_laterality_map()
    rows: List[dict] = []

    for fk in factor_indices:
        cpath = FACTOR_SCORES_DIR / f"controls_F{fk}_scores.csv"
        epath = FACTOR_SCORES_DIR / f"epilepsy_F{fk}_scores.csv"
        if not cpath.exists() or not epath.exists():
            continue
        ctrl = pd.read_csv(cpath)
        epi = pd.read_csv(epath)
        roi_cols = sorted((set(ctrl.columns) & set(epi.columns)) - NON_ROI_COLUMNS)
        z_df = zscore_epilepsy_vs_controls(ctrl, epi, roi_cols)
        z_df = z_df.drop_duplicates(subset=["subject"], keep="first")
        subj_list = sorted(s for s in z_df["subject"].unique() if s in lat_map)
        if len(subj_list) < 2:
            continue
        rowby = z_df.set_index("subject")
        factor_name = FACTOR_DISPLAY_LABELS.get(fk, f"F{fk}")

        for p in pairs:
            lc = str(p["left_col"])
            rc = str(p["right_col"])
            if lc not in z_df.columns or rc not in z_df.columns:
                continue
            ipsi_all: List[float] = []
            contra_all: List[float] = []
            for s in subj_list:
                laterality = lat_map.get(s)
                if laterality not in ("left", "right"):
                    continue
                ipsi_col = lc if laterality == "left" else rc
                contra_col = rc if laterality == "left" else lc
                i = rowby.at[s, ipsi_col]
                c = rowby.at[s, contra_col]
                if np.isfinite(i) and np.isfinite(c):
                    ipsi_all.append(float(i))
                    contra_all.append(float(c))
            d = _cohens_d_paired(ipsi_all, contra_all)
            if not np.isfinite(d):
                continue
            atlas, roi_type = _factor_z_pair_to_atlas_roi_type(p)
            rows.append(
                {
                    "label": _factor_z_pair_to_summary_label(p),
                    "factor_index": fk,
                    "factor_name": factor_name,
                    "cohens_d": float(d),
                    "atlas": atlas,
                    "roi_type": roi_type,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["label", "factor_index", "factor_name", "cohens_d", "atlas", "roi_type"]
        )
    out = pd.DataFrame(rows)
    wm_mask = out["roi_type"] == "wm"
    if wm_mask.any():
        out.loc[wm_mask, "roi_id"] = out.loc[wm_mask, "label"]
    return _exclude_volumetric_asymmetry_tracts(out)


def _factor_cohens_wide_by_label(
    cohens_sub: pd.DataFrame,
    *,
    label_col: str,
) -> pd.DataFrame:
    """Per label: one column per factor (F1 Overall, F2 Non-Gaussian, F3 Anisotropic)."""
    wide_cols = ["label"] + [FACTOR_COHENS_D_COLS[k] for k in (1, 2, 3)]
    if cohens_sub.empty:
        return pd.DataFrame(columns=wide_cols)
    df = cohens_sub.dropna(subset=["cohens_d"]).copy()
    if label_col == "tract_base":
        df["label"] = df["label"].apply(lambda x: _wm_roi_to_tract_segment(x)[0])
    agg = df.groupby(["label", "factor_index"], as_index=False)["cohens_d"].mean()
    wide = agg.pivot(index="label", columns="factor_index", values="cohens_d").reset_index()
    rename: Dict[object, str] = {}
    for c in wide.columns:
        if c == "label":
            continue
        try:
            fk = int(c)
        except (TypeError, ValueError):
            continue
        if fk in FACTOR_COHENS_D_COLS:
            rename[c] = FACTOR_COHENS_D_COLS[fk]
    wide = wide.rename(columns=rename)
    for fk in (1, 2, 3):
        col = FACTOR_COHENS_D_COLS[fk]
        if col not in wide.columns:
            wide[col] = float("nan")
    return wide[wide_cols]


def _attach_max_factor_from_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``max_factor_cohens_d`` / ``max_factor_name`` from wide F1–F3 columns (peak |d|)."""
    out = df.copy()
    rows_max_d: List[float] = []
    rows_max_name: List[Optional[str]] = []
    idx_to_name = {fk: FACTOR_DISPLAY_LABELS[fk] for fk in (1, 2, 3)}
    for _, row in out.iterrows():
        best_abs = -1.0
        best_d = float("nan")
        best_name: Optional[str] = None
        for fk in (1, 2, 3):
            col = FACTOR_COHENS_D_COLS[fk]
            v = row.get(col)
            if v is None or pd.isna(v) or not np.isfinite(float(v)):
                continue
            fv = float(v)
            if abs(fv) > best_abs:
                best_abs = abs(fv)
                best_d = fv
                best_name = idx_to_name[fk]
        rows_max_d.append(best_d)
        rows_max_name.append(best_name)
    out["max_factor_cohens_d"] = rows_max_d
    out["max_factor_name"] = rows_max_name
    return out


def _factor_cohens_summary_by_label(
    cohens_sub: pd.DataFrame,
    *,
    label_col: str,
) -> pd.DataFrame:
    """Per label: signed Cohen's d and factor name at peak |d| across F1–F3."""
    empty_cols = ["label", "max_factor_cohens_d", "max_factor_name"]
    if cohens_sub.empty:
        return pd.DataFrame(columns=empty_cols)
    df = cohens_sub.dropna(subset=["cohens_d"]).copy()
    if label_col == "tract_base":
        df["tract_base"] = df["label"].apply(lambda x: _wm_roi_to_tract_segment(x)[0])
        group_key = "tract_base"
    else:
        group_key = label_col if label_col in df.columns else "label"
    rows: List[dict] = []
    for label, grp in df.groupby(group_key):
        g = grp.dropna(subset=["cohens_d"])
        if g.empty:
            rows.append(
                {
                    "label": label,
                    "max_factor_cohens_d": float("nan"),
                    "max_factor_name": None,
                }
            )
            continue
        signed_d = g["cohens_d"]
        abs_d = signed_d.abs()
        idx_max = abs_d.idxmax()
        rows.append(
            {
                "label": label,
                "max_factor_cohens_d": float(signed_d.loc[idx_max]),
                "max_factor_name": str(g.loc[idx_max, "factor_name"]),
            }
        )
    out = pd.DataFrame(rows)
    if label_col != "label" and "label" not in out.columns:
        out = out.rename(columns={group_key: "label"})
    return out


def load_mahalanobis_cohens_df(
    subcortical_bases: Set[str],
    cortical_bases_4s: Set[str],
    glasser_bases: Set[str],
    variant: str = "raw",
) -> pd.DataFrame:
    """Paired Cohen's d per ROI from Mahalanobis asymmetry (one value per region/segment, not per scalar)."""
    from microstructural_asymmetry_report_mahalanobis import (
        compute_cohens_d_mahal,
        load_region_mahal,
        load_tract_mahal,
    )

    tract_df = load_tract_mahal()
    region_df = load_region_mahal(subcortical_bases, cortical_bases_4s, glasser_bases)
    cohens_df, _ = compute_cohens_d_mahal(tract_df, region_df, variant)
    if cohens_df.empty:
        return cohens_df
    return _exclude_volumetric_asymmetry_tracts(cohens_df)


def _mahal_cohens_summary_by_label(
    cohens_df_mahal: pd.DataFrame,
    *,
    label_col: str,
) -> pd.DataFrame:
    """Mahalanobis Cohen's d per summary label (mean across segments when ``label_col`` is tract_base)."""
    if cohens_df_mahal.empty:
        return pd.DataFrame(columns=["label", "mahal_cohens_d"])
    df = cohens_df_mahal.dropna(subset=["cohens_d"]).copy()
    if label_col == "tract_base":
        df["tract_base"] = df["roi_id"].apply(lambda x: _wm_roi_to_tract_segment(x)[0])
        agg = df.groupby("tract_base", as_index=False)["cohens_d"].mean()
        return agg.rename(columns={"tract_base": "label", "cohens_d": "mahal_cohens_d"})
    agg = df.groupby(label_col, as_index=False)["cohens_d"].mean()
    if label_col != "label":
        agg = agg.rename(columns={label_col: "label"})
    return agg.rename(columns={"cohens_d": "mahal_cohens_d"})


def build_summary_table_dataframes(
    full_long: pd.DataFrame,
    cohens_df: pd.DataFrame,
    cohens_df_mahal: Optional[pd.DataFrame] = None,
    cohens_df_factor: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Build per-atlas summary tables: Mahalanobis d, largest factor d, largest scalar d."""
    out: Dict[str, pd.DataFrame] = {}
    cohens_df = _filter_excluded_scalars(cohens_df)
    if full_long.empty or cohens_df.empty:
        return out

    factor_wide_cols = [FACTOR_COHENS_D_COLS[k] for k in (1, 2, 3)]
    cols = [
        "label",
        "mean_ipsi",
        "mean_contra",
        "mean_asymmetry",
        "mahal_cohens_d",
        *factor_wide_cols,
        "max_factor_cohens_d",
        "max_factor_name",
        "max_cohens_d",
        "max_abs_cohens_scalar",
    ]

    def _merge_summary_and_sort(
        roi_stats: pd.DataFrame,
        mahal_agg: pd.DataFrame,
        factor_wide: pd.DataFrame,
        scalar_agg: pd.DataFrame,
    ) -> pd.DataFrame:
        merged = roi_stats.merge(mahal_agg, on="label", how="left")
        fw_cols = ["label"] + factor_wide_cols
        if not factor_wide.empty:
            merged = merged.merge(factor_wide[fw_cols], on="label", how="left")
        scalar_cols = ["label", "max_cohens_d", "max_abs_cohens_scalar"]
        if not scalar_agg.empty:
            merged = merged.merge(scalar_agg[scalar_cols], on="label", how="left")
        for c in ("mahal_cohens_d", *factor_wide_cols, "max_cohens_d"):
            if c not in merged.columns:
                merged[c] = float("nan")
        if "max_abs_cohens_scalar" not in merged.columns:
            merged["max_abs_cohens_scalar"] = None
        merged = _attach_max_factor_from_wide(merged)
        return merged.sort_values("mahal_cohens_d", ascending=False, na_position="last")[cols]

    empty_mahal = pd.DataFrame(columns=["label", "mahal_cohens_d"])
    empty_factor = pd.DataFrame(
        columns=["label"] + factor_wide_cols + ["max_factor_cohens_d", "max_factor_name"]
    )
    cohens_mahal = cohens_df_mahal if cohens_df_mahal is not None else pd.DataFrame()
    cohens_factor = cohens_df_factor if cohens_df_factor is not None else pd.DataFrame()

    for atlas in ("4s_subcortex", "4s_cortex", "glasser"):
        sub = full_long[full_long["atlas"] == atlas]
        if sub.empty:
            continue
        roi_stats = sub.groupby("roi_id").apply(lambda g: pd.Series(_roi_stats_for_summary(g)))
        if roi_stats.empty:
            continue
        roi_stats = roi_stats.reset_index()
        roi_stats = roi_stats.rename(columns={"roi_id": "label"})
        sub_cohens = cohens_df[cohens_df["atlas"] == atlas]
        mahal_sub = cohens_mahal[cohens_mahal["atlas"] == atlas] if not cohens_mahal.empty else empty_mahal
        factor_sub = cohens_factor[cohens_factor["atlas"] == atlas] if not cohens_factor.empty else empty_factor
        mahal_agg = _mahal_cohens_summary_by_label(mahal_sub, label_col="roi_id")
        factor_wide = _factor_cohens_wide_by_label(factor_sub, label_col="label")
        if not sub_cohens.empty:
            scalar_agg = _cohens_d_summary_by_label(sub_cohens, label_col="roi_id")
            out[atlas] = _merge_summary_and_sort(roi_stats, mahal_agg, factor_wide, scalar_agg)
        else:
            assign = dict(
                mahal_cohens_d=float("nan"),
                max_factor_cohens_d=float("nan"),
                max_factor_name=None,
                max_cohens_d=float("nan"),
                max_abs_cohens_scalar=None,
            )
            for fc in factor_wide_cols:
                assign[fc] = float("nan")
            out[atlas] = roi_stats.assign(**assign)[cols]

    wm_long = full_long[full_long["roi_type"] == "wm"]
    wm_cohens = cohens_df[cohens_df["roi_type"] == "wm"]
    wm_mahal = cohens_mahal[cohens_mahal["roi_type"] == "wm"] if not cohens_mahal.empty else empty_mahal
    wm_factor = cohens_factor[cohens_factor["roi_type"] == "wm"] if not cohens_factor.empty else empty_factor

    def _drop_excluded_wm(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "roi_id" not in df.columns:
            return df
        return df[
            ~df["roi_id"].map(lambda rid: _is_excluded_volumetric_asymmetry_wm_roi(str(rid)))
        ].copy()

    wm_long = _drop_excluded_wm(wm_long)
    wm_cohens = _drop_excluded_wm(wm_cohens)
    wm_mahal = _drop_excluded_wm(wm_mahal)
    wm_factor = _drop_excluded_wm(wm_factor)

    if not wm_long.empty and not wm_cohens.empty:
        roi_stats = wm_long.groupby("roi_id").apply(lambda g: pd.Series(_roi_stats_for_summary(g)))
        roi_stats = roi_stats.reset_index()
        roi_stats = roi_stats.rename(columns={"roi_id": "label"})
        scalar_agg = _cohens_d_summary_by_label(wm_cohens, label_col="roi_id")
        mahal_agg = _mahal_cohens_summary_by_label(wm_mahal, label_col="roi_id")
        factor_wide = _factor_cohens_wide_by_label(wm_factor, label_col="label")
        out["hcp1065_thirds"] = _merge_summary_and_sort(roi_stats, mahal_agg, factor_wide, scalar_agg)

        wm_long = wm_long.copy()
        wm_long["tract_base"] = wm_long["roi_id"].apply(lambda x: _wm_roi_to_tract_segment(x)[0])
        tract_stats = wm_long.groupby("tract_base").apply(lambda g: pd.Series(_roi_stats_for_summary(g)))
        tract_stats = tract_stats.reset_index()
        tract_stats = tract_stats.rename(columns={"tract_base": "label"})
        wm_cohens = wm_cohens.copy()
        wm_cohens["tract_base"] = wm_cohens["roi_id"].apply(lambda x: _wm_roi_to_tract_segment(x)[0])
        scalar_tract = _cohens_d_summary_by_label(wm_cohens, label_col="tract_base")
        mahal_tract = _mahal_cohens_summary_by_label(wm_mahal, label_col="tract_base")
        factor_tract = _factor_cohens_wide_by_label(wm_factor, label_col="tract_base")
        out["hcp1065_whole"] = _merge_summary_and_sort(tract_stats, mahal_tract, factor_tract, scalar_tract)

    return out


def _summary_table_sorted_for_variant(df: pd.DataFrame, sort_col: str) -> pd.DataFrame:
    if df.empty or sort_col not in df.columns:
        return df
    return df.sort_values(sort_col, ascending=False, na_position="last").copy()


def save_summary_tables_per_atlas(
    full_long: pd.DataFrame,
    cohens_df: pd.DataFrame,
    tract_base_to_type: Dict[str, str],
    report_dir: Path,
    cohens_df_mahal: Optional[pd.DataFrame] = None,
    cohens_df_factor: Optional[pd.DataFrame] = None,
) -> None:
    """Save per-atlas summary CSVs sorted by Mahalanobis Cohen's d (descending)."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    tables = build_summary_table_dataframes(
        full_long, cohens_df, cohens_df_mahal, cohens_df_factor
    )
    csv_stems = {
        "4s_subcortex": "summary_4s_subcortex_scalars",
        "4s_cortex": "summary_4s_cortex_scalars",
        "glasser": "summary_glasser_scalars",
        "hcp1065_thirds": "summary_hcp1065_thirds_scalars",
        "hcp1065_whole": "summary_hcp1065_whole_scalars",
    }
    for key, df in tables.items():
        if key not in csv_stems:
            continue
        stem = csv_stems[key]
        sorted_df = _summary_table_sorted_for_variant(df, "mahal_cohens_d")
        sorted_df.to_csv(report_dir / f"{stem}_mean-abscohend.csv", index=False)
        sorted_df.to_csv(report_dir / f"{stem}.csv", index=False)


def _latex_escape(s: str) -> str:
    """Escape special characters for LaTeX text mode."""
    if not isinstance(s, str):
        s = str(s)
    out: List[str] = []
    for ch in s:
        if ch == "\\":
            out.append("\\textbackslash{}")
        elif ch == "&":
            out.append("\\&")
        elif ch == "%":
            out.append("\\%")
        elif ch == "$":
            out.append("\\$")
        elif ch == "#":
            out.append("\\#")
        elif ch == "^":
            out.append("\\textasciicircum{}")
        elif ch == "_":
            out.append("\\_")
        elif ch == "{":
            out.append("\\{")
        elif ch == "}":
            out.append("\\}")
        elif ch == "~":
            out.append("\\textasciitilde{}")
        else:
            out.append(ch)
    return "".join(out)


def _load_4s_thalamus_roi_bases() -> Set[str]:
    """Region name bases (e.g. Pulvinar, Anterior) that belong to thalamus nuclei in 4S156."""
    if not ATLAS_TSV_4S.exists():
        return set()
    try:
        df = pd.read_csv(ATLAS_TSV_4S, sep="\t")
    except Exception:
        return set()
    if "label" not in df.columns or "atlas_name" not in df.columns:
        return set()
    th = df[df["atlas_name"].astype(str).str.strip() == "ThalamusHCP"]
    bases: Set[str] = set()
    for label in th["label"].dropna().astype(str):
        for pref in ("LH_", "LH-", "RH_", "RH-"):
            if label.startswith(pref):
                bases.add(label[len(pref) :])
                break
    return bases


def _wm_tract_base_key(tract_with_hemi: str) -> str:
    """ILF_L -> ILF for lookup in tract_base_to_type."""
    t = str(tract_with_hemi).strip()
    if t.endswith("_L") or t.endswith("_R"):
        return t[:-2]
    return t


# HCP1065 along-tract segment codes -> human labels (end1/end2/core; A/P/I/S from tract profiling).
HCP1065_SEGMENT_LABELS: Dict[str, str] = {
    "A": "Anterior",
    "P": "Posterior",
    "I": "Inferior",
    "S": "Superior",
    "M": "Medial",
    "L": "Lateral",
    "end-M": "Medial",
    "end-L": "Lateral",
    "core": "Core",
    "end1": "End 1",
    "end2": "End 2",
}


def _hcp1065_segment_human(segment: str) -> str:
    s = str(segment).strip()
    if s in HCP1065_SEGMENT_LABELS:
        return HCP1065_SEGMENT_LABELS[s]
    if s.startswith("end"):
        return f"End {s[3:]}" if len(s) > 3 else s
    return s.replace("_", " ").title()


def _hcp1065_metadata_name_without_hemi_suffix(raw_name: str) -> str:
    """Metadata ``name`` often ends with ``_L``/``_R``; strip for ipsilateral ``tract`` base keys (e.g. AF)."""
    n = str(raw_name).strip()
    if n.endswith("_L") or n.endswith("_R"):
        n = n[:-2]
    return n.replace("_", " ")


def _load_tract_label_to_pretty_name() -> Dict[str, str]:
    """HCP1065 tract label -> long name from metadata ``name`` column.

    Includes atlas-style labels (``AF_L``), and tract bases without hemisphere (``AF``) as in
    tract asymmetry CSVs, mapped to the same prose with side suffix removed from ``name``.
    """
    if not TRACT_METADATA_PATH.exists():
        return {}
    try:
        meta = pd.read_csv(TRACT_METADATA_PATH)
    except Exception:
        return {}
    if "label" not in meta.columns or "name" not in meta.columns:
        return {}
    out: Dict[str, str] = {}
    for _, row in meta.iterrows():
        lab = str(row["label"]).strip()
        raw_name = str(row["name"]).strip()
        out[lab] = raw_name.replace("_", " ")
        if lab.endswith("_L") or lab.endswith("_R"):
            base = lab[:-2]
            if base not in out:
                out[base] = _hcp1065_metadata_name_without_hemi_suffix(raw_name)
    return out


def _format_tex_region_label(
    raw_label: str,
    atlas: str,
    thalamus_bases: Set[str],
) -> str:
    """Pretty region column for GM tables; thalamus nuclei get Thalamus- prefix for 4s_subcortex."""
    lab = str(raw_label).strip()
    if atlas == "4s_subcortex" and lab in thalamus_bases:
        return _format_thalamus_nucleus_tex_label(lab)
    return lab.replace("_", " ")


def _format_4s_subcortex_tex_label(raw_label: str, thalamus_bases: Set[str]) -> str:
    """4S subcortex .tex Region column: thalamus prefix, then CIT168 abbreviation map, else underscores to spaces."""
    lab = str(raw_label).strip()
    if lab in thalamus_bases:
        return _format_thalamus_nucleus_tex_label(lab)
    if lab in FOUR_S_SUBCORTEX_TEX_ABBREV_LABELS:
        return FOUR_S_SUBCORTEX_TEX_ABBREV_LABELS[lab]
    return lab.replace("_", " ")


def _format_tex_wm_thirds_label(roi_id: str, tract_names: Dict[str, str]) -> str:
    """WM roi_id like ILF_L_A -> Inferior Longitudinal Fasciculus L — Anterior."""
    tract_hemi, seg = _wm_roi_to_tract_segment(roi_id)
    tract_pretty = tract_names.get(tract_hemi, tract_hemi.replace("_", " "))
    seg_pretty = _hcp1065_segment_human(seg)
    return f"{tract_pretty} — {seg_pretty}"


def _hex_to_latex_html_color(hex_color: str) -> str:
    return str(hex_color).lstrip("#").upper()


def _latex_textcolor(content: str, hex_color: str) -> str:
    return rf"\textcolor[HTML]{{{_hex_to_latex_html_color(hex_color)}}}{{{content}}}"


def _format_tex_cohens_d_value(v: object) -> str:
    """Cohen's d: nearest thousandth, centered in column; ``\\phantom{-}`` aligns digits."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return "---"
    fv = round(float(v), 3)
    s = f"{fv:.3f}"
    if fv >= 0:
        s = rf"\phantom{{-}}{s}"
    return rf"\makebox[\linewidth][c]{{{s}}}"


def _format_tex_cohens_d_value_bold(v: object, bold: bool) -> str:
    s = _format_tex_cohens_d_value(v)
    if s == "---" or not bold:
        return s
    return rf"\textbf{{{s}}}"


def _tex_cohens_d_supercolumn_cells(row: pd.Series) -> List[str]:
    """Four Cohen's d cells; bold largest |d| among F1–F3 only (Mahalanobis never bolded)."""
    factor_keys = [FACTOR_COHENS_D_COLS[k] for k in (1, 2, 3)]
    parsed: List[Tuple[str, Optional[float]]] = []
    for k in COHEND_D_TEX_COL_KEYS:
        v = row.get(k)
        if v is None or pd.isna(v) or not np.isfinite(float(v)):
            parsed.append((k, None))
        else:
            parsed.append((k, float(v)))
    factor_abs = [abs(v) for k, v in parsed if k in factor_keys and v is not None]
    max_factor_abs = max(factor_abs) if factor_abs else None
    out: List[str] = []
    for k, v in parsed:
        if v is None:
            out.append("---")
        elif k in factor_keys and max_factor_abs is not None and abs(v) >= max_factor_abs - 1e-12:
            out.append(_format_tex_cohens_d_value_bold(v, True))
        else:
            out.append(_format_tex_cohens_d_value_bold(v, False))
    return out


def _format_tex_scalar_abbrev_colored(
    scalar: object,
    scalar_colors: Dict[str, str],
) -> Optional[str]:
    if scalar is None or (isinstance(scalar, float) and np.isnan(scalar)) or pd.isna(scalar):
        return None
    s = str(scalar).strip()
    if not s:
        return None
    abbrev = _scalar_abbrev(s)
    color = _scalar_color(s, scalar_colors, MODEL_FALLBACK_COLORS)
    return _latex_textcolor(abbrev, color)


def _format_tex_max_with_parenthetical(
    max_val: object,
    parenthetical: Optional[str],
) -> str:
    """LaTeX cell: ``0.8534 (Anisotropic)`` or ``0.8534 (\\textcolor{...}{FA})``."""
    max_str = _format_tex_cohens_d_value(max_val)
    if max_str == "---" and not parenthetical:
        return "---"
    if not parenthetical:
        return max_str
    if max_str == "---":
        return f"({parenthetical})"
    return f"{max_str} ({parenthetical})"


def _format_tex_max_with_statistic(
    max_val: object,
    scalar: object,
    scalar_colors: Dict[str, str],
) -> str:
    """LaTeX cell: ``0.8534 (\\textcolor{...}{ICVF})``."""
    stat_tex = _format_tex_scalar_abbrev_colored(scalar, scalar_colors)
    return _format_tex_max_with_parenthetical(max_val, stat_tex)


def _format_tex_max_with_factor(
    max_val: object,
    factor_name: object,
) -> str:
    """LaTeX cell: ``0.4408 (Anisotropic)``."""
    if factor_name is None or (isinstance(factor_name, float) and np.isnan(factor_name)) or pd.isna(
        factor_name
    ):
        parent = None
    else:
        parent = _latex_escape(str(factor_name).strip())
    return _format_tex_max_with_parenthetical(max_val, parent)


def _load_glasser_additional_metadata() -> pd.DataFrame:
    """Glasser parcel metadata indexed by unilateral ``region`` id (matches summary ``label``)."""
    if not GLASSER_ADDITIONAL_METADATA_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(GLASSER_ADDITIONAL_METADATA_PATH)
    except Exception:
        return pd.DataFrame()
    need = {"region", "regionLongName"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    df = df.dropna(subset=["region"]).copy()
    df["region"] = df["region"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["region"], keep="first")
    return df.set_index("region", drop=True)


def _save_summary_table_tex(
    df: pd.DataFrame,
    out_path: Path,
    *,
    region_col_formatter: Optional[Callable[[str], str]] = None,
    extra_column_headers: Optional[List[str]] = None,
    extra_cells_for_raw_label: Optional[Callable[[str], List[str]]] = None,
    first_column_header: str = "Region",
    top_n: Optional[int] = None,
    scalar_colors: Optional[Dict[str, str]] = None,
    mahal_col: str = "mahal_cohens_d",
    factor_col: str = "max_factor_cohens_d",
    factor_name_col: str = "max_factor_name",
    max_col: str = "max_cohens_d",
    include_statistic_column: bool = False,
) -> None:
    """Write a ``longtable``: Mahalanobis d; optional factor/scalar largest-d columns."""
    if df.empty:
        return
    if top_n is not None:
        df = df.head(int(top_n)).copy()
    if df.empty:
        return
    colors = scalar_colors if scalar_colors is not None else _load_scalar_colors()
    n_extra = len(extra_column_headers) if extra_column_headers else 0
    if n_extra:
        if extra_cells_for_raw_label is None:
            raise ValueError("extra_cells_for_raw_label is required when extra_column_headers is set")
    else:
        extra_cells_for_raw_label = None

    col1 = _latex_escape(first_column_header)
    n_leading = 1 + n_extra
    n_cohend = 4
    if include_statistic_column:
        mahal_header = r"Mahalanobis $\mathrm{Cohen's\ }d$"
        largest_header = r"Largest Cohen's d"
        col_spec = (
            "@{}"
            + _COHEND_TEX_LABEL_COL
            + r">{\raggedleft\arraybackslash}p{0.10\textwidth}"
            + r">{\raggedleft\arraybackslash}p{0.15\textwidth}"
            + r">{\raggedleft\arraybackslash}p{0.15\textwidth}"
            + "@{}"
        )
        cmid_start = n_leading + 2
        cmid_end = n_leading + 3
        leading_cells = [col1] + ([_latex_escape(h) for h in extra_column_headers] if n_extra else [])
        row1 = " & ".join(
            leading_cells + [mahal_header, rf"\multicolumn{{2}}{{c}}{{{largest_header}}}"]
        )
        row2 = " & ".join([""] * (n_leading + 1) + ["Factor", "Statistic"])
        header_block = [
            r"\toprule",
            row1 + r" \\",
            rf"\cmidrule(lr){{{cmid_start}-{cmid_end}}}",
            row2 + r" \\",
            r"\midrule",
        ]
    else:
        col_spec = _cohend_longtable_col_spec()
        cohend_header = r"Cohen's d"
        cmid_start = n_leading + 1
        cmid_end = n_leading + n_cohend
        row1 = " & ".join(
            [""] * n_leading + [rf"\multicolumn{{{n_cohend}}}{{c}}{{{cohend_header}}}"]
        )
        row2_cells: List[str] = []
        if n_extra:
            row2_cells.extend(_latex_escape(h) for h in extra_column_headers)
        row2_cells.append(_cohend_subheader_tex(first_column_header))
        row2_cells.extend(_cohend_subheader_tex(h) for h in COHEND_D_TEX_SUBHEADERS)
        row2 = " & ".join(row2_cells)
        header_block = [
            r"\toprule",
            row1 + r" \\",
            rf"\cmidrule(lr){{{cmid_start}-{cmid_end}}}",
            row2 + r" \\",
            r"\midrule",
        ]
    lines = [
        SUMMARY_COHEND_TEX_USEPACKAGE,
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{longtable}{" + col_spec + "}",
        *header_block,
        r"\endfirsthead",
        *header_block,
        r"\endhead",
        r"\midrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, row in df.iterrows():
        raw_lab = str(row["label"]).strip()
        lab = raw_lab
        if region_col_formatter is not None:
            lab = region_col_formatter(raw_lab)
        lab_tex = _cohend_region_label_tex(str(lab))
        extra_tex: List[str] = []
        if n_extra and extra_cells_for_raw_label is not None:
            cells = list(extra_cells_for_raw_label(raw_lab))
            while len(cells) < n_extra:
                cells.append("")
            cells = cells[:n_extra]
            extra_tex = [_latex_escape(str(c)) for c in cells]
        if include_statistic_column:
            mahal_tex = _format_tex_cohens_d_value(row.get(mahal_col))
            factor_tex = _format_tex_max_with_factor(
                row.get(factor_col),
                row.get(factor_name_col),
            )
            stat_tex = _format_tex_max_with_statistic(
                row.get(max_col),
                row.get("max_abs_cohens_scalar"),
                colors,
            )
            value_tex = [mahal_tex, factor_tex, stat_tex]
        else:
            value_tex = _tex_cohens_d_supercolumn_cells(row)
        lines.append(" & ".join([lab_tex] + extra_tex + value_tex) + r" \\")
    lines.extend([r"\end{longtable}", r"\endgroup"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_summary_cohend_top_tex(
    df: pd.DataFrame,
    stem: str,
    report_dir: Path,
    *,
    region_col_formatter: Optional[Callable[[str], str]] = None,
    first_column_header: str = "Region",
    top_n: int = SUMMARY_TEX_TOP_N,
) -> None:
    """Write ``{stem}_cohend_top{N}.tex``: Cohen's d supercolumn (Mahalanobis + factors), sorted by Mahalanobis."""
    sorted_df = _summary_table_sorted_for_variant(df, "mahal_cohens_d")
    _save_summary_table_tex(
        sorted_df,
        report_dir / f"{stem}_cohend_top{top_n}.tex",
        region_col_formatter=region_col_formatter,
        first_column_header=first_column_header,
        top_n=top_n,
        include_statistic_column=False,
    )


def _save_summary_table_tex_variants(
    df: pd.DataFrame,
    stem: str,
    report_dir: Path,
    *,
    region_col_formatter: Optional[Callable[[str], str]] = None,
    first_column_header: str = "Region",
    top_n: Optional[int] = None,
    scalar_colors: Optional[Dict[str, str]] = None,
    include_statistic_column: bool = False,
) -> None:
    """Write legacy scalar summary ``.tex`` variants sorted by Mahalanobis Cohen's d."""
    sorted_df = _summary_table_sorted_for_variant(df, "mahal_cohens_d")
    tex_kw = dict(
        region_col_formatter=region_col_formatter,
        first_column_header=first_column_header,
        scalar_colors=scalar_colors,
        include_statistic_column=include_statistic_column,
    )
    _save_summary_table_tex(
        sorted_df,
        report_dir / f"{stem}_mean-abscohend.tex",
        **tex_kw,
    )
    _save_summary_table_tex(
        sorted_df,
        report_dir / f"{stem}.tex",
        **tex_kw,
    )
    if top_n is not None:
        _save_summary_table_tex(
            sorted_df,
            report_dir / f"{stem}_mean-abscohend_top{top_n}.tex",
            top_n=top_n,
            **tex_kw,
        )
        _save_summary_table_tex(
            sorted_df,
            report_dir / f"{stem}_top{top_n}.tex",
            top_n=top_n,
            **tex_kw,
        )


def save_summary_tables_tex_per_atlas(
    full_long: pd.DataFrame,
    cohens_df: pd.DataFrame,
    tract_base_to_type: Dict[str, str],
    report_dir: Path,
    cohens_df_mahal: Optional[pd.DataFrame] = None,
    cohens_df_factor: Optional[pd.DataFrame] = None,
) -> None:
    """Write atlas-specific LaTeX ``longtable`` files (page breaks). HCP1065 thirds split association vs projection."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    tables = build_summary_table_dataframes(
        full_long, cohens_df, cohens_df_mahal, cohens_df_factor
    )
    if not tables:
        return

    thalamus_bases = _load_4s_thalamus_roi_bases()
    tract_pretty = _load_tract_label_to_pretty_name()
    glasser_add = _load_glasser_additional_metadata()

    for atlas in ("4s_subcortex", "4s_cortex", "glasser"):
        df = tables.get(atlas)
        if df is None or df.empty:
            continue

        def _fmt(lab: str) -> str:
            if atlas == "glasser" and not glasser_add.empty:
                rid = str(lab).strip()
                if rid in glasser_add.index:
                    ser = glasser_add.loc[rid]
                    if isinstance(ser, pd.DataFrame):
                        ser = ser.iloc[0]
                    try:
                        v = ser["regionLongName"]
                    except (KeyError, TypeError, IndexError):
                        v = None
                    if v is not None and pd.notna(v) and str(v).strip():
                        name = str(v).strip()
                        if name == "Hippocampus":
                            return "Parahippocampal"
                        return name
            if atlas == "4s_subcortex":
                return _format_4s_subcortex_tex_label(lab, thalamus_bases)
            return _format_tex_region_label(lab, atlas, thalamus_bases)

        _save_summary_cohend_top_tex(
            df,
            f"summary_{atlas}",
            report_dir,
            region_col_formatter=_fmt,
        )

    df_whole = tables.get("hcp1065_whole")
    if df_whole is not None and not df_whole.empty:

        def _fmt_wm_whole(lab: str) -> str:
            return tract_pretty.get(str(lab), str(lab).replace("_", " "))

        _save_summary_cohend_top_tex(
            df_whole,
            "summary_hcp1065_whole",
            report_dir,
            region_col_formatter=_fmt_wm_whole,
            first_column_header="Tract segment",
        )

    df_thirds = tables.get("hcp1065_thirds")
    if df_thirds is not None and not df_thirds.empty:

        def _wm_row_tract_type(rid: str) -> Optional[str]:
            tract_hemi, _s = _wm_roi_to_tract_segment(str(rid))
            return tract_base_to_type.get(_wm_tract_base_key(tract_hemi))

        mask_a = df_thirds["label"].map(lambda rid: _wm_row_tract_type(rid) == "association")
        mask_p = df_thirds["label"].map(lambda rid: _wm_row_tract_type(rid) == "projection")
        fmt_wm = lambda lab: _format_tex_wm_thirds_label(str(lab), tract_pretty)
        if mask_a.any():
            _save_summary_cohend_top_tex(
                df_thirds.loc[mask_a].copy(),
                "summary_hcp1065_thirds_association",
                report_dir,
                region_col_formatter=fmt_wm,
                first_column_header="Tract segment",
            )
        if mask_p.any():
            _save_summary_cohend_top_tex(
                df_thirds.loc[mask_p].copy(),
                "summary_hcp1065_thirds_projection",
                report_dir,
                region_col_formatter=fmt_wm,
                first_column_header="Tract segment",
            )


def _load_scalar_colors() -> Dict[str, str]:
    """Load scalar -> hex color from metadata; fallback by model prefix."""
    if SCALAR_COLORS_PATH.exists():
        try:
            return json.loads(SCALAR_COLORS_PATH.read_text())
        except Exception:
            pass
    return {}


def _scalar_color(
    scalar: str,
    scalar_colors: Dict[str, str],
    model_fallback: Optional[Dict[str, str]] = None,
) -> str:
    if model_fallback is None:
        model_fallback = MODEL_FALLBACK_COLORS
    if scalar in scalar_colors:
        return scalar_colors[scalar]
    for prefix, color in model_fallback.items():
        if scalar.startswith(prefix):
            return color
    return "#333333"


def _scalar_abbrev(scalar: str) -> str:
    """Capitalized abbreviation for radar labels (e.g. dti_fa -> FA)."""
    # Mathtext for ∥/⊥ so symbols come from the math font (Georgia lacks these codepoints).
    if scalar == "map_ngpar":
        return r"$\mathrm{NG}\parallel$"
    if scalar == "map_ngperp":
        return r"$\mathrm{NG}\perp$"
    if "_" in scalar:
        return scalar.split("_", 1)[-1].upper()
    return scalar.upper()


def _canonical_scalars_for_plots(
    scalar_labels: Dict[str, str],
    plot_df: pd.DataFrame,
) -> List[str]:
    """X-axis order for bar/strip plots: sorted ``scalar_labels`` keys (minus EXCLUDED), then any data-only scalars.

    Matches radar-style fixed ordering so scalars without ROI-level Cohen's d rows still appear as empty categories.
    """
    if scalar_labels:
        keys = [s for s in sorted(scalar_labels.keys()) if s not in EXCLUDED_SCALARS]
        in_data = set(plot_df["scalar"].dropna().unique())
        extra = sorted(s for s in in_data if s not in EXCLUDED_SCALARS and s not in keys)
        return keys + extra
    avail = sorted(pd.unique(plot_df["scalar"].dropna()))
    return [s for s in avail if s not in EXCLUDED_SCALARS]


def scalar_reconstruction_model_prefix(scalar: str) -> Optional[str]:
    """Model prefix for a scalar column (e.g. ``noddi_isovf`` → ``noddi``)."""
    if "_" not in scalar:
        return None
    return scalar.split("_", 1)[0]


def load_factor_loading_scalar_order(
    csv_path: Optional[Path] = None,
) -> List[str]:
    """Column order from ``*_scalar_factor_loadings_ordered.csv`` (excludes ``factor``)."""
    path = csv_path or FACTOR_LOADINGS_ORDERED_CSV
    if not path.exists():
        return []
    cols = list(pd.read_csv(path, nrows=0).columns)
    return [c for c in cols if c != "factor"]


def sort_scalars_by_reconstruction_model(
    present: Sequence[str],
    reference_order: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Order scalars by reconstruction model (NODDI, MAPMRI, DKI, DTI, GQI), then within
    each model by ``reference_order`` (factor loadings CSV columns by default).
    """
    if reference_order is None:
        reference_order = load_factor_loading_scalar_order()
    model_rank = {m: i for i, m in enumerate(RECONSTRUCTION_MODEL_ORDER)}
    ref_rank = {s: i for i, s in enumerate(reference_order)}

    def _sort_key(s: str) -> Tuple[int, int, str]:
        model = scalar_reconstruction_model_prefix(s) or ""
        mr = model_rank.get(model, len(RECONSTRUCTION_MODEL_ORDER))
        sr = ref_rank.get(s, len(reference_order) + 1)
        return (mr, sr, s)

    return sorted(present, key=_sort_key)


def sort_scalars_in_quadrant_by_reconstruction_model(
    sub: pd.DataFrame,
    reference_order: Optional[Sequence[str]] = None,
) -> List[str]:
    """Scalars with Cohen's d rows in ``sub``, ordered by reconstruction model."""
    if sub.empty or "scalar" not in sub.columns:
        return []
    present: List[str] = []
    seen: Set[str] = set()
    for s in sub["scalar"].dropna().unique():
        t = str(s)
        if t in seen:
            continue
        v = sub.loc[sub["scalar"] == t, "cohens_d"].dropna().astype(float)
        if len(v) == 0:
            continue
        seen.add(t)
        present.append(t)
    return sort_scalars_by_reconstruction_model(present, reference_order)


def _sort_scalars_by_mean_abs_desc(scalars: List[str], plot_df: pd.DataFrame) -> List[str]:
    """Order scalars by mean |Cohen's d| in ``plot_df`` (descending); scalars with no rows last (alphabetically)."""
    if not scalars:
        return []
    scored: List[Tuple[int, float, str]] = []
    for s in scalars:
        v = plot_df.loc[plot_df["scalar"] == s, "abs_cohens_d"].dropna().astype(float)
        n = int(v.size)
        if n == 0:
            scored.append((1, 0.0, s))
        else:
            scored.append((0, -float(np.mean(v)), s))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[2] for t in scored]


def _mean_sem_abs_per_scalar_values(
    sub: pd.DataFrame, scalars_ordered: List[str], value_col: str = "abs_cohens_d"
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and SEM of ``value_col`` per scalar; 0 / 0 when no rows for that scalar."""
    means = np.zeros(len(scalars_ordered), dtype=float)
    sems = np.zeros(len(scalars_ordered), dtype=float)
    for i, s in enumerate(scalars_ordered):
        v = sub.loc[sub["scalar"] == s, value_col].dropna().astype(float).to_numpy()
        n = int(v.size)
        if n == 0:
            continue
        if n == 1:
            means[i] = float(v[0])
        else:
            means[i] = float(np.mean(v))
            sems[i] = float(np.std(v, ddof=1) / np.sqrt(n))
    return means, sems


def _table_effect_size_all_scalars(
    by_scalar: pd.Series,
    scalar_labels: Dict[str, str],
    scalar_colors: Dict[str, str],
) -> Tuple[List[str], List[str], List[str]]:
    """Return (names, effect_sizes, colors) for table rows, sorted by effect size desc."""
    model_fallback = {"dki": "#7A297F", "dti": "#C43031", "gqi": "#FAA51A", "noddi": "#38489E", "map": "#289144", "rdi": "#C43031"}
    names = [scalar_labels.get(s, s) for s in by_scalar.index.tolist()]
    effect_sizes = [f"{v:.4f}" for v in by_scalar.values]
    colors = [_scalar_color(s, scalar_colors, model_fallback) for s in by_scalar.index.tolist()]
    return names, effect_sizes, colors


def _plot1_whole_brain_prep(
    df: pd.DataFrame,
    scalar_labels: Dict[str, str],
    scalar_colors: Dict[str, str],
) -> Optional[Tuple[pd.DataFrame, List[str], List[str], pd.Series, List[str]]]:
    """Return (plot_df, scalars_sorted, bar_palette, by_scalar_mean, tick_labels) or None if no data.

    ``tick_labels`` are short abbreviations (e.g. MD); same label may repeat—model is shown by bar/strip color.
    """
    if df.empty:
        return None
    plot_df = df[~df["scalar"].isin(EXCLUDED_SCALARS)].copy()
    plot_df["abs_cohens_d"] = np.abs(plot_df["cohens_d"])
    plot_df = plot_df.dropna(subset=["abs_cohens_d"])
    if plot_df.empty:
        return None
    scalars_full = _canonical_scalars_for_plots(scalar_labels, plot_df)
    if not scalars_full:
        return None
    scalars_sorted = _sort_scalars_by_mean_abs_desc(scalars_full, plot_df)
    plot_df = plot_df[plot_df["scalar"].isin(scalars_sorted)].copy()
    model_fallback = {"dki": "#7A297F", "dti": "#C43031", "gqi": "#FAA51A", "noddi": "#38489E", "map": "#289144"}
    bar_palette = [_scalar_color(s, scalar_colors, model_fallback) for s in scalars_sorted]
    by_scalar = plot_df.groupby("scalar")["abs_cohens_d"].mean()
    tick_labels = [_scalar_abbrev(s) for s in scalars_sorted]
    return plot_df, scalars_sorted, bar_palette, by_scalar, tick_labels


def plot1_whole_brain_bars(
    df: pd.DataFrame,
    out_path: Path,
    scalar_labels: Dict[str, str],
    scalar_colors: Dict[str, str],
) -> None:
    """Whole-brain bar plot: mean ± SEM of |Cohen's d| across ROIs per scalar."""
    import matplotlib.pyplot as plt

    prep = _plot1_whole_brain_prep(df, scalar_labels, scalar_colors)
    if prep is None:
        plt.figure(figsize=(7, 4.25))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    plot_df, scalars_sorted, bar_palette, _, tick_labels = prep
    n = len(scalars_sorted)
    means, sems = _mean_sem_abs_per_scalar_values(plot_df, scalars_sorted)
    # Y-limits from mean ± SEM only (not raw ROI max), so they can differ from plot 1b (strips).
    y_top_bar = float(np.nanmax(means + sems)) * 1.05 if n else 0.1
    y_max = max(y_top_bar, 0.1)
    y_min = 0.05
    fig_w = max(7.0, 0.32 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 4.25))
    x = np.arange(n, dtype=float)
    ax.bar(
        x,
        means,
        yerr=sems,
        width=min(0.74, 20.0 / max(n, 1)),
        capsize=2.5,
        color=bar_palette,
        edgecolor="0.25",
        linewidth=0.8,
        error_kw={"ecolor": "black", "linewidth": 1.2},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=13.5)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("|Cohen's d|", fontsize=15.5, fontweight="normal")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=13.5)
    ax.tick_params(axis="y", labelsize=13.5)
    ax.set_title(
        r"$\bf{Mean\,absolute\,asymmetries}$\nWhole-brain — mean ± SEM across ROIs".replace(r"\n", "\n"),
        fontsize=14,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot1_whole_brain_strips(
    df: pd.DataFrame,
    out_path: Path,
    scalar_labels: Dict[str, str],
    scalar_colors: Dict[str, str],
) -> None:
    """Whole-brain strip plot: |Cohen's d| per ROI by scalar with mean lines."""
    import matplotlib.pyplot as plt

    prep = _plot1_whole_brain_prep(df, scalar_labels, scalar_colors)
    if prep is None:
        plt.figure(figsize=(7, 5))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    plot_df, scalars_sorted, bar_palette, by_scalar, tick_labels = prep
    n = len(scalars_sorted)
    y_min = 0.05
    y_max = max(float(plot_df["abs_cohens_d"].max()) * 1.05, y_min + 0.05)
    fig_w = max(7.0, 0.32 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    rng = np.random.default_rng(42)
    for i, s in enumerate(scalars_sorted):
        pts = plot_df.loc[plot_df["scalar"] == s, "abs_cohens_d"].dropna().astype(float).to_numpy()
        if pts.size == 0:
            continue
        xj = rng.normal(float(i), 0.08, size=pts.size)
        ax.scatter(
            xj,
            pts,
            c=bar_palette[i],
            s=16,
            alpha=0.85,
            edgecolors="none",
            zorder=3,
        )
    for i, s in enumerate(scalars_sorted):
        if s in by_scalar.index and pd.notna(by_scalar.loc[s]):
            ax.hlines(
                float(by_scalar.loc[s]),
                i - 0.45,
                i + 0.45,
                colors="black",
                linewidth=0.8,
                zorder=5,
            )
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("|Cohen's d|", fontsize=14, fontweight="normal")
    ax.set_xlabel("")
    ax.set_xticks(np.arange(n, dtype=float))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=12)
    ax.set_xlim(-0.5, n - 0.5)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title(
        r"$\bf{Mean\,absolute\,asymmetries}$\nWhole-brain — |Cohen's d| per ROI".replace(r"\n", "\n"),
        fontsize=14,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _draw_radar_ax(
    ax,
    scalars_ordered: List[str],
    scalar_to_d: Dict[str, float],
    scalar_colors: Dict[str, str],
    title: str,
    model_fallback: Optional[Dict[str, str]] = None,
    max_abs_override: Optional[float] = None,
) -> None:
    """Draw a single radar plot on polar axes (|Cohen's d| or mean |d| per scalar). Same style as asymmetry_tle_region.
    If max_abs_override is set, use it for the radial scale so multiple plots share the same axis."""
    import matplotlib.pyplot as plt
    if model_fallback is None:
        model_fallback = {"dki": "#7A297F", "dti": "#C43031", "gqi": "#FAA51A", "noddi": "#38489E", "map": "#289144"}
    n = len(scalars_ordered)
    if n == 0:
        return
    d_vals = np.array([scalar_to_d.get(s, float("nan")) for s in scalars_ordered], dtype=float)
    abs_d = np.where(np.isfinite(d_vals), np.abs(d_vals), 0.0)
    if max_abs_override is not None and max_abs_override > 0:
        max_abs = float(max_abs_override)
    else:
        max_abs = float(np.max(abs_d)) if np.any(abs_d > 0) else 0.01
        if max_abs <= 0:
            max_abs = 0.01
    # Angular positions: reverse of uniform spacing so traversal around the circle matches the opposite
    # direction vs. ``theta = i * 2π/n`` (θ=0 at N via set_theta_zero_location below).
    delta = 2 * np.pi / n
    theta_base = delta * np.arange(n - 1, -1, -1, dtype=float)
    idx_max = int(np.nanargmax(abs_d)) if np.any(np.isfinite(abs_d)) else -1
    if idx_max >= 0:
        # Rotate so the largest |d| sits at 12 o'clock (θ=0) instead of one step clockwise from it.
        theta = np.mod(theta_base - theta_base[idx_max] + 2 * np.pi, 2 * np.pi)
    else:
        theta = theta_base
    R = 1.0
    r_vals = R * (abs_d / max_abs)
    theta_closed = np.append(theta, theta[0])
    r_closed = np.append(r_vals, r_vals[0])
    default_color = "#888888"
    colors = [_scalar_color(s, scalar_colors, model_fallback) for s in scalars_ordered]
    ax.fill(theta_closed, r_closed, alpha=0.15, color="gray", linewidth=0)
    for i in range(n):
        ax.plot([0, theta[i]], [0, r_vals[i]], color=colors[i], linewidth=2.5, solid_capstyle="round")
    ax.scatter(theta, r_vals, color="black", s=28, zorder=5)
    ax.plot(theta_closed, np.zeros_like(theta_closed), color="gray", linewidth=0.5, alpha=0.5)
    radial_limit = R * 1.32
    ax.set_ylim(0, radial_limit)
    ax.set_yticks([0, R / 4, R / 2, 3 * R / 4, R])
    ax.set_yticklabels(
        [
            "0",
            f"{max_abs / 4:.2f}",
            f"{max_abs / 2:.2f}",
            f"{3 * max_abs / 4:.2f}",
            f"{max_abs:.2f}",
        ],
        fontsize=13,
    )
    ax.set_theta_zero_location("N")
    # Hide circular outline (spine) so it doesn't occlude scalar labels; keep radial grid rings
    if "polar" in ax.spines:
        ax.spines["polar"].set_visible(False)
    label_r = R * 1.26
    for i, scalar in enumerate(scalars_ordered):
        abbrev = _scalar_abbrev(scalar)
        weight = "bold" if i == idx_max else "normal"
        ax.text(theta[i], label_r, abbrev, ha="center", va="center", fontsize=14.5,
                fontweight=weight, color=colors[i])
    if idx_max >= 0:
        t_max = theta[idx_max]
        tip_r = radial_limit + 0.01
        base_r = radial_limit + 0.07
        half_w = 0.05
        ax.fill(
            np.array([t_max, t_max - half_w, t_max + half_w]),
            np.array([tip_r, base_r, base_r]),
            color="black",
            zorder=10,
        )
    ax.set_xticks(theta)
    ax.set_xticklabels([])
    # Title: bold tissue type; optional second line (e.g. "360 regions") unbold as part of main title (close to plot)
    if "\n" in title:
        line1, line2 = title.split("\n", 1)
        ax.text(0.5, 1.06, line1, transform=ax.transAxes, ha="center", fontsize=14, fontweight="bold")
        ax.text(0.5, 1.02, line2, transform=ax.transAxes, ha="center", fontsize=12, fontweight="normal")
    else:
        ax.set_title(title, pad=12, fontsize=14, fontweight="bold")


def plot_radar_mean_abs_cohend(
    scalars_ordered: List[str],
    scalar_to_d: Dict[str, float],
    scalar_colors: Dict[str, str],
    out_path: Path,
    title: str = "Mean absolute asymmetries\nWhole-brain",
) -> None:
    """Single radar plot: mean |Cohen's d| per scalar. Angular order follows ``scalars_ordered``. Saves to ``out_path``."""
    import matplotlib.pyplot as plt
    model_fallback = {"dki": "#7A297F", "dti": "#C43031", "gqi": "#FAA51A", "noddi": "#38489E", "map": "#289144"}
    if not scalars_ordered:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"))
        if "\n" in title:
            line1, line2 = title.split("\n", 1)
            ax.text(0.5, 1.06, line1, transform=ax.transAxes, ha="center", fontsize=14, fontweight="bold")
            ax.text(0.5, 1.02, line2, transform=ax.transAxes, ha="center", fontsize=12, fontweight="normal")
        else:
            ax.set_title(title, fontsize=14)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"))
    _draw_radar_ax(ax, scalars_ordered, scalar_to_d, scalar_colors, title, model_fallback)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot2_2x2_radar(
    df: pd.DataFrame,
    tract_base_to_type: Dict[str, str],
    out_path: Path,
    scalar_labels: Dict[str, str],
    scalar_colors: Dict[str, str],
    *,
    sort_by_mean: bool = False,
) -> None:
    """2×2 polar radar plots: same quadrants as plot2_2x2_bars / plot2_2x2_strips.

    If ``sort_by_mean`` is False, angular order matches metadata (sorted scalar keys).
    If True, each quadrant orders scalars by mean |Cohen's d| in that quadrant (descending).
    """
    import matplotlib.pyplot as plt
    model_fallback = {"dki": "#7A297F", "dti": "#C43031", "gqi": "#FAA51A", "noddi": "#38489E", "map": "#289144"}
    df_quad = add_quadrant_column(df, tract_base_to_type)
    quadrants = [
        ("cortex", "Cortex GM"),
        ("association_wm", "Association WM"),
        ("subcortex", "Subcortex GM"),
        ("projection_wm", "Projection WM"),
    ]
    scalars_ordered = sorted(scalar_labels.keys()) if scalar_labels else []
    if not scalars_ordered and not df_quad.empty:
        scalars_ordered = sorted(df_quad["scalar"].unique().tolist())
    scalars_ordered = [s for s in scalars_ordered if s not in EXCLUDED_SCALARS]
    # Global max |Cohen's d| across quadrants for identical radial axes
    global_max_abs = 0.01
    for (quad_key, _) in quadrants:
        sub = df_quad[df_quad["quadrant"] == quad_key] if not df_quad.empty else pd.DataFrame()
        if quad_key == "cortex" and not sub.empty and "atlas" in sub.columns:
            sub = sub[sub["atlas"] == "glasser"].copy()
        if sub.empty or not scalars_ordered:
            continue
        by_scalar = sub.groupby("scalar")["cohens_d"].apply(lambda s: np.abs(s).mean())
        for s in scalars_ordered:
            v = by_scalar.get(s, float("nan"))
            if np.isfinite(v) and v > global_max_abs:
                global_max_abs = float(v)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(projection="polar"))
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for ax, (quad_key, title) in zip(axes_flat, quadrants):
        sub = df_quad[df_quad["quadrant"] == quad_key] if not df_quad.empty else pd.DataFrame()
        # Cortex GM: Glasser only (exclude Schaefer/4s_cortex from radar)
        if quad_key == "cortex" and not sub.empty and "atlas" in sub.columns:
            sub = sub[sub["atlas"] == "glasser"].copy()
        if sub.empty or not scalars_ordered:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=14)
            R = 1.0
            ax.set_ylim(0, R * 1.32)
            ax.set_yticks([0, R / 4, R / 2, 3 * R / 4, R])
            ax.set_yticklabels(
                [
                    "0",
                    f"{global_max_abs / 4:.2f}",
                    f"{global_max_abs / 2:.2f}",
                    f"{3 * global_max_abs / 4:.2f}",
                    f"{global_max_abs:.2f}",
                ],
                fontsize=13,
            )
            continue
        n_units = sub["roi_id"].nunique()
        roi_ids = sorted(sub["roi_id"].unique().tolist())
        if quad_key in ("cortex", "subcortex"):
            count_label = f"{n_units} regions"
            # print(f"Radar {title}: {n_units} regions", file=sys.stdout)
        else:
            count_label = f"{n_units} tract segments"
            # print(f"Radar {title}: {n_units} tract segments", file=sys.stdout)
        # for roi_id in roi_ids:
        #     print(f"  {roi_id}", file=sys.stdout)
        title_with_count = title + "\n" + count_label
        by_scalar = sub.groupby("scalar")["cohens_d"].apply(lambda s: np.abs(s).mean())
        sub_plot = sub.copy()
        sub_plot["abs_cohens_d"] = np.abs(sub_plot["cohens_d"])
        scalars_for_ax = (
            _sort_scalars_by_mean_abs_desc(scalars_ordered, sub_plot)
            if sort_by_mean
            else list(scalars_ordered)
        )
        scalar_to_d = {s: by_scalar.get(s, float("nan")) for s in scalars_for_ax}
        _draw_radar_ax(ax, scalars_for_ax, scalar_to_d, scalar_colors, title_with_count, model_fallback, max_abs_override=global_max_abs)
    order_note = (
        "Scalars sorted by mean |d| within quadrant"
        if sort_by_mean
        else "Fixed scalar order (metadata)"
    )
    fig.suptitle(
        r"$\bf{Mean\ absolute\ asymmetries}$"
        + "\nIpsilateral-Contralateral |Cohen's d|"
        + f"\n({order_note})",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout(h_pad=2.0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _summary_table_html(
    rows: List[Tuple[str, float]],
    title: str,
    value_header: str = "Effect size",
    roi_header: str = "Region",
    top_n: int = ATLAS_TOP_N,
) -> str:
    """GM table: top N rows by value (default top 20). Red styling for high values."""
    if not rows:
        return f'<div class="summary-table-wrap"><p><strong>{title}</strong></p><p>No data</p></div>'
    rows_to_show = sorted(rows, key=lambda x: x[1], reverse=True)[:top_n]
    lines = [
        f'<div class="summary-table-wrap"><p><strong>{title}</strong></p>',
        f'<table class="summary-table"><thead><tr><th>{html_module.escape(roi_header)}</th><th>{html_module.escape(value_header)}</th></tr></thead><tbody>',
    ]
    for (label, val) in rows_to_show:
        safe_label = html_module.escape(str(label))
        lines.append(f'<tr><td>{safe_label}</td><td>{val:.4f}</td></tr>')
    lines.append("</tbody></table></div>")
    return "\n".join(lines)


def _summary_table_wm_html(
    rows: List[Tuple[str, str, float]],
    title: str,
    value_header: str,
    top_n: int = ATLAS_TOP_N,
) -> str:
    """WM table: top N rows (default top 20); columns Tract, Segment, value."""
    if not rows:
        return f'<div class="summary-table-wrap"><p><strong>{title}</strong></p><p>No data</p></div>'
    rows_to_show = sorted(rows, key=lambda x: x[2], reverse=True)[:top_n]
    lines = [
        f'<div class="summary-table-wrap"><p><strong>{title}</strong></p>',
        f'<table class="summary-table"><thead><tr><th>Tract</th><th>Segment</th><th>{html_module.escape(value_header)}</th></tr></thead><tbody>',
    ]
    for (tract, segment, val) in rows_to_show:
        lines.append(f'<tr><td>{html_module.escape(str(tract))}</td><td>{html_module.escape(str(segment))}</td><td>{val:.4f}</td></tr>')
    lines.append("</tbody></table></div>")
    return "\n".join(lines)


def tables_2x2_html(quadrant_data: Dict[str, List], value_header: str = "Mean Ipsilateral-Contralateral |Cohen's d|") -> str:
    """Build 2x2 grid HTML: Row 1 = Cortex GM, Association WM; Row 2 = Subcortex GM, Projection WM (top 20 each)."""
    cortex = _summary_table_html(quadrant_data.get("cortex", []), "Cortex GM", value_header=value_header, roi_header="Region")
    assoc = _summary_table_wm_html(quadrant_data.get("association", []), "Association WM", value_header)
    subcortex = _summary_table_html(quadrant_data.get("subcortex", []), "Subcortex GM", value_header=value_header, roi_header="Region")
    proj = _summary_table_wm_html(quadrant_data.get("projection", []), "Projection WM", value_header)
    return f'<div class="grid-tables-2x2">{cortex}{assoc}{subcortex}{proj}</div>'


def atlas_tables_html(atlas_data: Dict[str, List], value_header: str = "Mean Ipsilateral-Contralateral |Cohen's d|") -> str:
    """Build HTML summary tables for all 5 atlases (top 20 each): 4s_subcortex, 4s_cortex, glasser, hcp1065_whole, hcp1065_thirds."""
    titles = {
        "4s_subcortex": "4S subcortex (GM subcortex)",
        "4s_cortex": "4S cortex / Schaefer100 (GM cortex)",
        "glasser": "Glasser (GM cortex)",
        "hcp1065_whole": "HCP1065 whole-tracts (WM)",
        "hcp1065_thirds": "HCP1065 along-tract thirds (WM)",
    }
    parts = []
    for key in ("4s_subcortex", "4s_cortex", "glasser", "hcp1065_whole", "hcp1065_thirds"):
        data = atlas_data.get(key, [])
        title = titles.get(key, key)
        if key == "hcp1065_thirds" and data and len(data[0]) == 3:
            parts.append(_summary_table_wm_html(data, title, value_header))
        elif key == "hcp1065_whole" and data and len(data[0]) == 2:
            parts.append(_summary_table_html(data, title, value_header=value_header, roi_header="Tract"))
        elif data and len(data[0]) == 2:
            parts.append(_summary_table_html(data, title, value_header=value_header, roi_header="Region"))
        else:
            parts.append(f'<div class="summary-table-wrap"><p><strong>{html_module.escape(title)}</strong></p><p>No data</p></div>')
    return '<div class="grid-tables-atlas">' + "".join(parts) + "</div>"


def _plot2_quadrants_prep(
    df: pd.DataFrame,
    tract_base_to_type: Dict[str, str],
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Build quadrant-labeled dataframe and fixed quadrant list for 2×2 plots."""
    df_quad = add_quadrant_column(df[~df["scalar"].isin(EXCLUDED_SCALARS)], tract_base_to_type)
    df_quad = df_quad.copy()
    df_quad["abs_cohens_d"] = np.abs(df_quad["cohens_d"])
    df_quad = df_quad.dropna(subset=["abs_cohens_d"])
    quadrants: List[Tuple[str, str]] = [
        ("cortex", "Cortex GM"),
        ("association_wm", "Association WM"),
        ("subcortex", "Subcortex GM"),
        ("projection_wm", "Projection WM"),
    ]
    return df_quad, quadrants


def _plot2_shared_y_range_bars(
    df_quad: pd.DataFrame,
    quadrants: List[Tuple[str, str]],
    scalars_full: List[str],
) -> Tuple[float, float, np.ndarray]:
    """One y-axis range for the bar 2×2 figure: max over quadrants of bar top (mean + SEM)."""
    y_min = 0.05
    peak = y_min
    for quad_key, _ in quadrants:
        sub = df_quad[df_quad["quadrant"] == quad_key] if not df_quad.empty else pd.DataFrame()
        if quad_key == "cortex" and not sub.empty and "atlas" in sub.columns:
            sub = sub[sub["atlas"] == "glasser"].copy()
        if sub.empty:
            continue
        scalars_ord = _sort_scalars_by_mean_abs_desc(scalars_full, sub)
        means, sems = _mean_sem_abs_per_scalar_values(sub, scalars_ord)
        if means.size:
            peak = max(peak, float(np.nanmax(means + sems)), float(np.nanmax(means)))
    y_max = peak * 1.05 if peak > y_min else 0.10
    y_ticks = np.linspace(y_min, y_max, 5)
    return y_min, y_max, y_ticks


def _plot2_shared_y_range_strips(df_quad: pd.DataFrame) -> Tuple[float, float, np.ndarray]:
    """One y-axis range for the strip 2×2 figure: from raw |Cohen's d| points (all quadrants)."""
    y_min = 0.05
    if df_quad.empty:
        y_max = 0.10
    else:
        y_max = max(float(df_quad["abs_cohens_d"].max()) * 1.05, y_min + 0.05)
    y_ticks = np.linspace(y_min, y_max, 5)
    return y_min, y_max, y_ticks


def plot2_2x2_bars(
    df: pd.DataFrame,
    tract_base_to_type: Dict[str, str],
    out_path: Path,
    scalar_labels: Dict[str, str],
    scalar_colors: Dict[str, str],
) -> None:
    """2×2 bar plots (mean ± SEM): Cortex GM | Association WM; Subcortex GM | Projection WM."""
    import matplotlib.pyplot as plt

    model_fallback = {"dki": "#7A297F", "dti": "#C43031", "gqi": "#FAA51A", "noddi": "#38489E", "map": "#289144"}
    df_quad, quadrants = _plot2_quadrants_prep(df, tract_base_to_type)
    scalars_full = _canonical_scalars_for_plots(scalar_labels, df_quad)
    y_min, y_max, y_ticks = _plot2_shared_y_range_bars(df_quad, quadrants, scalars_full)
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.25))
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for ax, (quad_key, title) in zip(axes_flat, quadrants):
        sub = df_quad[df_quad["quadrant"] == quad_key] if not df_quad.empty else pd.DataFrame()
        if quad_key == "cortex" and not sub.empty and "atlas" in sub.columns:
            sub = sub[sub["atlas"] == "glasser"].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=14)
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f"{t:.2f}" for t in y_ticks], fontsize=13.5)
            ax.set_ylabel("|Cohen's d|", fontsize=15.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            continue
        sub = sub.copy()
        scalars_ord = _sort_scalars_by_mean_abs_desc(scalars_full, sub)
        n_sc = len(scalars_ord)
        tick_labels = [_scalar_abbrev(s) for s in scalars_ord]
        means, sems = _mean_sem_abs_per_scalar_values(sub, scalars_ord)
        bar_palette = [_scalar_color(s, scalar_colors, model_fallback) for s in scalars_ord]
        x = np.arange(n_sc, dtype=float)
        ax.bar(
            x,
            means,
            yerr=sems,
            width=min(0.64, 18.5 / max(n_sc, 1)),
            capsize=2.0,
            color=bar_palette,
            edgecolor="0.25",
            linewidth=0.8,
            error_kw={"ecolor": "black", "linewidth": 1.0},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=13.5)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{t:.2f}" for t in y_ticks], fontsize=13.5)
        ax.set_xlabel("")
        ax.set_ylabel("|Cohen's d|", fontsize=15.5)
        ax.set_title(title, fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Mean absolute asymmetries — mean ± SEM across ROIs", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot2_2x2_strips(
    df: pd.DataFrame,
    tract_base_to_type: Dict[str, str],
    out_path: Path,
    scalar_labels: Dict[str, str],
    scalar_colors: Dict[str, str],
) -> None:
    """2×2 strip plots with mean lines: Cortex GM | Association WM; Subcortex GM | Projection WM."""
    import matplotlib.pyplot as plt

    model_fallback = {"dki": "#7A297F", "dti": "#C43031", "gqi": "#FAA51A", "noddi": "#38489E", "map": "#289144"}
    df_quad, quadrants = _plot2_quadrants_prep(df, tract_base_to_type)
    scalars_full = _canonical_scalars_for_plots(scalar_labels, df_quad)
    y_min, y_max, y_ticks = _plot2_shared_y_range_strips(df_quad)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for panel_i, (ax, (quad_key, title)) in enumerate(zip(axes_flat, quadrants)):
        sub = df_quad[df_quad["quadrant"] == quad_key] if not df_quad.empty else pd.DataFrame()
        if quad_key == "cortex" and not sub.empty and "atlas" in sub.columns:
            sub = sub[sub["atlas"] == "glasser"].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=14)
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f"{t:.2f}" for t in y_ticks], fontsize=12)
            ax.set_ylabel("|Cohen's d|", fontsize=14)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            continue
        sub = sub.copy()
        scalars_ord = _sort_scalars_by_mean_abs_desc(scalars_full, sub)
        n_sc = len(scalars_ord)
        tick_labels = [_scalar_abbrev(s) for s in scalars_ord]
        bar_palette = [_scalar_color(s, scalar_colors, model_fallback) for s in scalars_ord]
        by_scalar = sub.groupby("scalar")["abs_cohens_d"].mean()
        rng = np.random.default_rng(42 + panel_i)
        for i, s in enumerate(scalars_ord):
            pts = sub.loc[sub["scalar"] == s, "abs_cohens_d"].dropna().astype(float).to_numpy()
            if pts.size == 0:
                continue
            xj = rng.normal(float(i), 0.08, size=pts.size)
            ax.scatter(
                xj,
                pts,
                c=bar_palette[i],
                s=16,
                alpha=0.85,
                edgecolors="none",
                zorder=3,
            )
        for i, s in enumerate(scalars_ord):
            if s in by_scalar.index and pd.notna(by_scalar.loc[s]):
                ax.hlines(
                    float(by_scalar.loc[s]),
                    i - 0.45,
                    i + 0.45,
                    colors="black",
                    linewidth=0.8,
                    zorder=5,
                )
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{t:.2f}" for t in y_ticks], fontsize=12)
        ax.set_xticks(np.arange(n_sc, dtype=float))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
        ax.set_xlim(-0.5, n_sc - 0.5)
        ax.set_xlabel("")
        ax.set_ylabel("|Cohen's d|", fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Mean absolute asymmetries — |Cohen's d| per ROI (strip + mean line)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_cortex_region_scores_for_brain_map(cohens_df: pd.DataFrame) -> Dict[str, float]:
    """Build Glasser label -> mean |Cohen's d| for cortical GM (Left_X, Right_X). Uses atlas=glasser only."""
    if "atlas" in cohens_df.columns:
        cortex = cohens_df[(cohens_df["roi_type"] == "cortical_gm") & (cohens_df["atlas"] == "glasser")]
    else:
        cortex = cohens_df[cohens_df["roi_type"] == "cortical_gm"]
    if cortex.empty:
        return {}
    by_roi = cortex.groupby("roi_id")["cohens_d"].apply(lambda s: np.abs(s).mean())
    out: Dict[str, float] = {}
    for roi_base, val in by_roi.items():
        if pd.isna(val):
            continue
        out["Left_" + str(roi_base)] = float(val)
        out["Right_" + str(roi_base)] = float(val)
    return out


def build_4s_cortex_region_scores(cohens_df: pd.DataFrame, cortical_bases_4s: set) -> Dict[str, float]:
    """Build 4S cortical (Schaefer100) label -> mean |Cohen's d| for pkl (LH_X, RH_X)."""
    if "atlas" not in cohens_df.columns:
        return {}
    sub = cohens_df[cohens_df["atlas"] == "4s_cortex"]
    if sub.empty:
        return {}
    label_to_index, _ = _load_4s_label_to_index()
    by_roi = sub.groupby("roi_id")["cohens_d"].apply(lambda s: np.abs(s).mean()).to_dict()
    out: Dict[str, float] = {}
    for roi_id, val in by_roi.items():
        if roi_id not in cortical_bases_4s or pd.isna(val):
            continue
        for prefix in ("LH_", "RH_", "LH-", "RH-"):
            key = prefix + roi_id
            if key in label_to_index:
                out[key] = float(val)
    return out


def build_hcp1065_whole_scores(cohens_df: pd.DataFrame) -> Dict[str, float]:
    """Build tract name -> mean |Cohen's d| for HCP1065 whole-tract (mean across segments)."""
    wm = cohens_df[cohens_df["roi_type"] == "wm"]
    if wm.empty:
        return {}
    wm = wm.copy()
    wm["tract_base"] = wm["roi_id"].apply(lambda x: _wm_roi_to_tract_segment(x)[0])
    by_tract = wm.groupby("tract_base")["cohens_d"].apply(lambda s: np.abs(s).mean())
    return {str(k): float(v) for k, v in by_tract.items() if np.isfinite(v)}


def build_hcp1065_thirds_scores(cohens_df: pd.DataFrame) -> Dict[str, float]:
    """Build 'tract_L_segment' / 'tract_R_segment' -> mean |Cohen's d| for HCP1065 along-tract thirds (pkl)."""
    wm = cohens_df[cohens_df["roi_type"] == "wm"]
    if wm.empty:
        return {}
    by_roi = wm.groupby("roi_id")["cohens_d"].apply(lambda s: np.abs(s).mean())
    out: Dict[str, float] = {}
    for roi_id, val in by_roi.items():
        if pd.isna(val):
            continue
        tract_base, segment = _wm_roi_to_tract_segment(roi_id)
        out[f"{tract_base}_L_{segment}"] = float(val)
        out[f"{tract_base}_R_{segment}"] = float(val)
    return out


def build_subcortex_region_scores_for_brain_map(
    cohens_df: pd.DataFrame,
    subcortical_bases: set,
) -> Dict[str, float]:
    """Build 4S156 label -> mean |Cohen's d| for subcortical GM (LH_X, RH_X)."""
    label_to_index, _ = _load_4s_label_to_index()
    roi_mean_abs_d = cohens_df[cohens_df["roi_type"] == "subcortical_gm"].groupby("roi_id")["cohens_d"].apply(lambda s: np.abs(s).mean()).to_dict()
    label_to_val: Dict[str, float] = {}
    for roi_id, val in roi_mean_abs_d.items():
        if roi_id not in subcortical_bases or pd.isna(val):
            continue
        for prefix in ("LH_", "RH_", "LH-", "RH-"):
            key = prefix + roi_id
            if key in label_to_index:
                label_to_val[key] = float(val)
    return label_to_val


def build_wm_tract_segment_scores_for_brain_map(
    cohens_df: pd.DataFrame,
    tract_base_to_type: Dict[str, str],
) -> Dict[Tuple[str, str], float]:
    """Build (tract_label, segment) -> mean |Cohen's d| for WM; tract_label = AF_L, AF_R etc."""
    wm = cohens_df[cohens_df["roi_type"] == "wm"]
    if wm.empty:
        return {}
    by_roi = wm.groupby("roi_id")["cohens_d"].apply(lambda s: np.abs(s).mean())
    out: Dict[Tuple[str, str], float] = {}
    for roi_id, val in by_roi.items():
        if pd.isna(val):
            continue
        tract_base, segment = _wm_roi_to_tract_segment(roi_id)
        if tract_base_to_type.get(tract_base) not in ("association", "projection"):
            continue
        out[(tract_base + "_L", segment)] = float(val)
        out[(tract_base + "_R", segment)] = float(val)
    return out


def plot3_2x2_brain_maps(
    cohens_df: pd.DataFrame,
    quadrant_data: Dict[str, List],
    subcortical_bases: set,
    cortical_bases_4s: set,
    tract_base_to_type: Dict[str, str],
    tract_metadata_df: pd.DataFrame,
    report_dir: Path,
) -> Optional[Path]:
    """
    Create 2x2 brain maps (one medial view per tissue category) using nilearn glass_brain: Cortex GM | Association WM; Subcortex GM | Projection WM.
    Saves one medial plot per category and a 2x2 composite. Saves 5 pkl files: glasser, 4s_subcortex, 4s_cortex, hcp1065_whole, hcp1065_thirds.
    Returns path to composite PNG or None.
    """
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "code" / "analysis"))
        from asymmetry_tle import brain_maps as bm
    except Exception as e:
        import traceback
        print("Brain maps skipped (install nilearn for glass brain figures):", e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None
    vmax = 0.0
    for data in quadrant_data.values():
        for item in data:
            v = item[-1] if isinstance(item, (list, tuple)) and len(item) >= 2 else (item[1] if len(item) == 2 else item[2])
            if isinstance(v, (int, float)):
                vmax = max(vmax, abs(v))
    if vmax <= 0:
        vmax = 1.0
    vmin, vmax = 0.0, vmax
    use_absolute = True

    # Cortex GM (Glasser) — pkl
    cortex_scores = build_cortex_region_scores_for_brain_map(cohens_df)
    with open(report_dir / "glasser_mean_abs_cohend.pkl", "wb") as f:
        pickle.dump(cortex_scores, f)
    if cortex_scores and ATLAS_NII_GLASSER.exists() and ATLAS_TSV_GLASSER.exists():
        bm.create_gm_brain_map(
            cortex_scores,
            "Cortex GM",
            str(report_dir / "plot3_cortex"),
            ATLAS_NII_GLASSER,
            ATLAS_TSV_GLASSER,
            vmin=vmin,
            vmax=vmax,
            use_absolute=use_absolute,
            display_views=("x",),  # medial only, one plot per tissue category
        )
    # Subcortex GM (4S) — pkl
    subcortex_scores = build_subcortex_region_scores_for_brain_map(cohens_df, subcortical_bases)
    with open(report_dir / "4s_subcortical_mean_abs_cohend.pkl", "wb") as f:
        pickle.dump(subcortex_scores, f)

    # 4S cortex (Schaefer100) — pkl
    s4_cortex_scores = build_4s_cortex_region_scores(cohens_df, cortical_bases_4s)
    with open(report_dir / "4s_cortical_mean_abs_cohend.pkl", "wb") as f:
        pickle.dump(s4_cortex_scores, f)

    # HCP1065 whole-tract and along-tract thirds — pkl
    hcp_whole_scores = build_hcp1065_whole_scores(cohens_df)
    with open(report_dir / "hcp1065_whole_mean_abs_cohend.pkl", "wb") as f:
        pickle.dump(hcp_whole_scores, f)
    hcp_thirds_scores = build_hcp1065_thirds_scores(cohens_df)
    with open(report_dir / "hcp1065_thirds_mean_abs_cohend.pkl", "wb") as f:
        pickle.dump(hcp_thirds_scores, f)
    atlas_4s = ATLAS_NII_4S if ATLAS_NII_4S.exists() else (ATLAS_NII_4S_FALLBACK if ATLAS_NII_4S_FALLBACK.exists() else None)
    if subcortex_scores and atlas_4s:
        bm.create_gm_brain_map(
            subcortex_scores,
            "Subcortex GM",
            str(report_dir / "plot3_subcortex"),
            Path(atlas_4s),
            ATLAS_TSV_4S,
            vmin=vmin,
            vmax=vmax,
            use_absolute=use_absolute,
            display_views=("x",),
        )
    # WM association + projection
    wm_scores = build_wm_tract_segment_scores_for_brain_map(cohens_df, tract_base_to_type)
    if wm_scores and ENDPOINT_NII_DIR.exists() and not tract_metadata_df.empty:
        bm.create_wm_brain_map(
            wm_scores,
            "WM",
            str(report_dir / "plot3_association.png"),
            str(report_dir / "plot3_projection.png"),
            tract_metadata_df,
            ENDPOINT_NII_DIR,
            vmin=vmin,
            vmax=vmax,
            use_absolute=use_absolute,
            display_views=("x",),
        )

    # Composite 2x2: one medial brain plot per tissue category (Cortex GM | Association WM; Subcortex GM | Projection WM)
    cell_paths = [
        report_dir / "plot3_cortex_ctx_x.png",
        report_dir / "plot3_association_x.png",
        report_dir / "plot3_subcortex_sctx_x.png",
        report_dir / "plot3_projection_x.png",
    ]
    existing = [p for p in cell_paths if p.exists()]
    if not existing:
        return None
    try:
        from PIL import Image
    except Exception:
        return existing[0] if existing else None
    sub_w, cell_h = 600, 300  # one medial view per cell; 2x2 composite
    composite = Image.new("RGB", (sub_w * 2, cell_h * 2), (248, 248, 248))
    for idx, img_path in enumerate(cell_paths):
        if not img_path.exists():
            continue
        ri, ci = idx // 2, idx % 2
        x0 = ci * sub_w
        y0 = ri * cell_h
        im = Image.open(img_path).convert("RGB")
        composite.paste(im.resize((sub_w, cell_h)), (x0, y0))
    out_path = report_dir / "plot3_2x2_brain_maps.png"
    composite.save(str(out_path))
    return out_path


def plot3_brain_map(region_scores: Dict[str, float], out_path: Path, title: str = "Mean Ipsilateral-Contralateral |Cohen's d| per region") -> None:
    """
    Whole-brain map using nilearn glass brain with 4S156 NIfTI and jet colormap.
    """
    if not region_scores:
        return
    atlas_path = ATLAS_NII_4S if ATLAS_NII_4S.exists() else (ATLAS_NII_4S_FALLBACK if ATLAS_NII_4S_FALLBACK.exists() else None)
    if not atlas_path:
        return
    try:
        import nibabel as nib
        from nilearn import plotting as nilearn_plotting

        label_to_index, _ = _load_4s_label_to_index()
        if not label_to_index:
            return
        img = nib.load(str(atlas_path))
        data = np.asarray(img.get_fdata(), dtype=float)
        out = np.zeros_like(data)
        for label, score in region_scores.items():
            idx = label_to_index.get(label)
            if idx is not None and np.isfinite(score):
                out[data == idx] = abs(score)
        stat_img = nib.Nifti1Image(out, img.affine)
        nilearn_plotting.plot_glass_brain(
            stat_img,
            title=title,
            display_mode="lyrz",
            colorbar=True,
            plot_abs=True,
            cmap="jet",
            output_file=str(out_path),
        )
    except Exception:
        pass


def _load_4s_label_to_index() -> Tuple[Dict[str, int], Dict[str, str]]:
    """Load 4S TSV -> label_to_index, label_to_network."""
    label_to_index: Dict[str, int] = {}
    label_to_network: Dict[str, str] = {}
    if not ATLAS_TSV_4S.exists():
        return label_to_index, label_to_network
    try:
        df = pd.read_csv(ATLAS_TSV_4S, sep="\t")
        if "label" in df.columns and "index" in df.columns:
            label_to_index = dict(zip(df["label"], df["index"].astype(int)))
        if "label" in df.columns and "network_label" in df.columns:
            label_to_network = dict(zip(df["label"], df["network_label"].fillna("n/a").astype(str)))
    except Exception:
        pass
    return label_to_index, label_to_network


def create_report_html(
    plot1_bars_path: Path,
    plot1_strips_path: Path,
    plot2_bars_path: Path,
    plot2_strips_path: Path,
    plot3_path: Path,
    out_html: Path,
    figures_dir: Path,
    quadrant_data: Optional[Dict[str, List]] = None,
    atlas_data: Optional[Dict[str, List]] = None,
    plot1_radar_path: Optional[Path] = None,
    plot1_radar_sorted_path: Optional[Path] = None,
    plot2_radar_path: Optional[Path] = None,
    plot2_radar_sorted_path: Optional[Path] = None,
    factor_z_plot_path: Optional[Path] = None,
    factor_z_gm_wm_plot_path: Optional[Path] = None,
) -> None:
    """Write HTML report: whole-brain bars/strips, 2×2 quadrant bars/strips, radars, brain maps, factor z bar figures, tables."""
    p1b = plot1_bars_path.name if plot1_bars_path and plot1_bars_path.exists() else None
    p1s = plot1_strips_path.name if plot1_strips_path and plot1_strips_path.exists() else None
    p2b = plot2_bars_path.name if plot2_bars_path and plot2_bars_path.exists() else None
    p2s = plot2_strips_path.name if plot2_strips_path and plot2_strips_path.exists() else None
    p3 = plot3_path.name if plot3_path and plot3_path.exists() else None
    p1_radar = plot1_radar_path.name if plot1_radar_path and plot1_radar_path.exists() else None
    p1_radar_s = plot1_radar_sorted_path.name if plot1_radar_sorted_path and plot1_radar_sorted_path.exists() else None
    p2_radar = plot2_radar_path.name if plot2_radar_path and plot2_radar_path.exists() else None
    p2_radar_s = plot2_radar_sorted_path.name if plot2_radar_sorted_path and plot2_radar_sorted_path.exists() else None
    p_fz = factor_z_plot_path.name if factor_z_plot_path and factor_z_plot_path.exists() else None
    p_fz_gmwm = factor_z_gm_wm_plot_path.name if factor_z_gm_wm_plot_path and factor_z_gm_wm_plot_path.exists() else None
    # Embed 2x2 summary tables HTML for Figure 2 and Figure 3 (top 20)
    tables_2x2 = tables_2x2_html(quadrant_data, "Mean Ipsilateral-Contralateral |Cohen's d|") if quadrant_data else ""
    # Per-atlas summary tables (top 20 each)
    atlas_tables = atlas_tables_html(atlas_data, "Mean Ipsilateral-Contralateral |Cohen's d|") if atlas_data else ""
    # Relative path for img src: report is in OUTPUT_DIR, figures in OUTPUT_DIR/figures
    fig_prefix = "figures/" if (out_html.parent / "figures").exists() else ""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Microstructural asymmetry report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1600px; margin: 2em auto; padding: 0 2em; }}
    h1 {{ font-size: 1.5rem; }}
    h2 {{ font-size: 1.2rem; margin-top: 1.5em; }}
    img {{ max-width: 100%; height: auto; }}
    .caption {{ color: #555; font-size: 0.9rem; margin-top: 0.5em; }}
    .figure-brain {{ margin: 2em 0; padding: 1.5em 0; max-width: 100%; }}
    .figure-brain img {{ display: block; max-width: 100%; width: auto; height: auto; margin: 1em auto; }}
    .grid-tables-2x2 {{ display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto auto; gap: 16px; margin-top: 16px; margin-bottom: 24px; width: 100%; }}
    .grid-tables-atlas {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; margin-top: 16px; margin-bottom: 24px; width: 100%; }}
    .summary-table-wrap {{ border: 1px solid #ddd; padding: 10px; background: #fafafa; }}
    .summary-table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
    .summary-table th, .summary-table td {{ padding: 4px 8px; text-align: left; border: 1px solid #ddd; }}
    .summary-table th {{ background: #eee; }}
  </style>
</head>
<body>
  <h1>Microstructural asymmetry report</h1>
  <p>Mean of absolute paired Cohen's d (ipsi − contra) across ROI asymmetries. Atlases: 4S subcortex, 4S cortex (Schaefer100), Glasser (GM); HCP1065 whole-tracts and along-tract thirds (WM). Data: tract_asymmetry + region_asymmetry_tle.</p>

  <h2>Figure 1: Mean Ipsilateral-Contralateral |Cohen's d| by scalar (all ROIs)</h2>
  <h3>1a — Bar plot (mean ± SEM across ROIs)</h3>
  """ + (
    (f'<p><img src="{fig_prefix + p1b}" alt="Whole-brain bar plot"></p><p class="caption">Mean ± SEM of |Cohen\'s d| per scalar (across ROIs); colored by model.</p>' if p1b else "<p>No bar plot data.</p>")
  ) + """
  <h3>1b — Strip plot (|Cohen's d| per ROI)</h3>
  """ + (
    (f'<p><img src="{fig_prefix + p1s}" alt="Whole-brain strip plot"></p><p class="caption">Each point is one ROI; horizontal lines: mean |d| per scalar; colored by model.</p>' if p1s else "<p>No strip plot data.</p>")
  ) + """
  <h3>1c — Radar (fixed scalar order)</h3>
  """ + (
    (f'<p><img src="{fig_prefix + p1_radar}" alt="Radar all ROIs fixed order" style="max-width: 420px;"></p><p class="caption">Mean |Cohen\'s d| by scalar; angular order matches metadata (sorted keys).</p>' if p1_radar else "<p>No radar.</p>")
  ) + """
  <h3>1d — Radar (sorted by mean |d|)</h3>
  """ + (
    (f'<p><img src="{fig_prefix + p1_radar_s}" alt="Radar all ROIs sorted" style="max-width: 420px;"></p><p class="caption">Same data; scalars ordered by whole-brain mean |Cohen\'s d| (largest first).</p>' if p1_radar_s else "<p>No radar.</p>")
  ) + """

  <h2>Figure 2: By quadrant (2×2: Cortex GM | Association WM; Subcortex GM | Projection WM)</h2>
  <h3>2a — Bar plots (mean ± SEM)</h3>
  """ + (
    (f'<p><img src="{fig_prefix + p2b}" alt="2x2 bar plots"></p><p class="caption">Column 1: Cortex GM (top), Subcortex GM (bottom). Column 2: Association WM (top), Projection WM (bottom). Mean ± SEM across ROIs per scalar.</p>' if p2b else "<p>No bar plot data.</p>")
  ) + """
  <h3>2b — Strip plots</h3>
  """ + (
    (f'<p><img src="{fig_prefix + p2s}" alt="2x2 strip plots"></p><p class="caption">Same layout as 2a; points = ROIs, black lines = mean |d| per scalar.</p>' if p2s else "<p>No strip plot data.</p>")
  ) + """
  <h3>2c — Radar (fixed scalar order)</h3>
  """ + (
    (f'<p><img src="{fig_prefix + p2_radar}" alt="2x2 radar fixed order" style="max-width: 100%;"></p><p class="caption">2×2 radar; angular order matches metadata (sorted keys) in each quadrant.</p>' if p2_radar else "<p>No radar.</p>")
  ) + """
  <h3>2d — Radar (sorted by mean |d| per quadrant)</h3>
  """ + (
    (f'<p><img src="{fig_prefix + p2_radar_s}" alt="2x2 radar sorted" style="max-width: 100%;"></p><p class="caption">Same data; within each quadrant, scalars ordered by mean |Cohen\'s d| (largest first).</p>' if p2_radar_s else "<p>No radar.</p>")
  ) + """

  <h2>Figure 3: Brain maps (2×2: Cortex GM | Association WM; Subcortex GM | Projection WM)</h2>
  """ + (
    (f'<div class="figure-brain"><img src="{fig_prefix + p3}" alt="2x2 brain maps"></div><p class="caption">One medial view per tissue category (nilearn glass brain; left-hemisphere data only; mean |Cohen\'s d| per region/tract).</p>' if p3 else '<p>No brain maps: ensure <code>nilearn</code> and <code>asymmetry_tle</code> brain_maps are available, and atlas/endpoint paths exist.</p>')
  ) + """

  <h2>Figure 4: Epilepsy mean factor z-scores by tissue class</h2>
  <h3>4a — Four tissue categories</h3>
  """ + (
    (f'<p><img src="{fig_prefix + p_fz}" alt="Factor z-scores by four tissue classes (mean ± SEM)" style="max-width: 100%;"></p>'
     f'<p class="caption">1×4 panels (cortex GM, subcortex GM, association WM, projection WM). Neutral bars: mean factor z-score ± SEM across subjects; '
     f'each subject’s value is the mean z across ROIs in that tissue. Factors F1–F4 from epilepsy cohort CSVs. '
     f'Tissue rules: cortex GM = Glasser + 4S cortex; subcortex GM = 4S subcortex; association / projection WM from HCP1065 metadata. '
     f'Data: <code>{html_module.escape(str(FACTOR_Z_SCORES_DIR))}</code> (<code>epilepsy_F1_z_scores.csv</code> … <code>epilepsy_F4_z_scores.csv</code>).</p>')
    if p_fz
    else '<p>No four-panel factor z-score figure: run the factor_z-scores pipeline and ensure epilepsy_F1..F4 z-score CSVs exist under <code>derivatives/analysis/factor_z-scores/factor_z_scores/</code>.</p>'
  ) + """
  <h3>4b — Grey matter vs white matter (pooled ROIs)</h3>
  """ + (
    (f'<p><img src="{fig_prefix + p_fz_gmwm}" alt="Factor z-scores grey vs white matter and GM–WM |z| difference (mean ± SEM)" style="max-width: 100%;"></p>'
     f'<p class="caption">1×3 panels (layout/styling aligned with tissue-specific PC1 vs whole-brain factor plots). '
     f'<strong>Grey Matter</strong> / <strong>White Matter</strong>: pooled ROIs as in 4a; bars = mean factor z ± SEM across subjects. '
     f'Third panel: per subject and factor, <code>|z<sub>GM</sub>| − |z<sub>WM</sub>|</code>, then mean ± SEM; y-axis = |Factor z-score| difference. '
     f'Same epilepsy CSVs as 4a.</p>')
    if p_fz_gmwm
    else '<p>No grey/white matter factor z-score figure (missing data or failed plot).</p>'
  ) + """

  <h2>Summary tables by atlas (top 20 per atlas)</h2>
  <p class="caption">4S subcortex (GM subcortex), 4S cortex / Schaefer100 (GM cortex), Glasser (GM cortex), HCP1065 whole-tracts (WM), HCP1065 along-tract thirds (WM).</p>
  """ + atlas_tables + """

</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    subcortical_bases = _get_4s_subcortical_bases()
    cortical_bases_4s = _get_4s_cortical_bases()
    glasser_bases = _get_glasser_bases()
    tract_metadata_df, tract_base_to_type = _load_tract_metadata()

    factor_z_plot_path: Optional[Path] = None
    factor_z_gm_wm_plot_path: Optional[Path] = None
    fz_df = build_epilepsy_factor_z_mean_by_tissue(tract_base_to_type)
    if fz_df is not None and not fz_df.empty:
        _fz_out = figures_dir / "plot4_factor_z_by_tissue_bars.png"
        plot_factor_z_mean_sem_bars_by_tissue(fz_df, _fz_out)
        if _fz_out.exists():
            factor_z_plot_path = _fz_out
    fz_gmwm_df = build_epilepsy_factor_z_mean_by_gm_wm(tract_base_to_type)
    if fz_gmwm_df is not None and not fz_gmwm_df.empty:
        _fz_gmwm_out = figures_dir / "plot4_factor_z_by_gm_wm_bars.png"
        plot_factor_z_mean_sem_bars_by_gm_wm(fz_gmwm_df, _fz_gmwm_out)
        if _fz_gmwm_out.exists():
            factor_z_gm_wm_plot_path = _fz_gmwm_out

    tract_df = load_tract_asymmetry()
    region_df = load_region_asymmetry(subcortical_bases, cortical_bases_4s, glasser_bases)
    cohens_df = compute_cohens_d_per_roi_scalar(tract_df, region_df)
    if cohens_df.empty:
        print("No Cohen's d data (missing tract or region asymmetry CSVs).", file=sys.stderr)
        create_report_html(
            Path(""),
            Path(""),
            Path(""),
            Path(""),
            Path(""),
            OUTPUT_DIR / "microstructural_asymmetry_report.html",
            figures_dir,
            quadrant_data=None,
            atlas_data=None,
            plot1_radar_path=None,
            plot1_radar_sorted_path=None,
            plot2_radar_path=None,
            plot2_radar_sorted_path=None,
            factor_z_plot_path=factor_z_plot_path,
            factor_z_gm_wm_plot_path=factor_z_gm_wm_plot_path,
        )
        return 1

    # Exclude requested tracts from tract-level WM analyses.
    cohens_df = _exclude_volumetric_asymmetry_tracts(cohens_df)

    scalar_labels: Dict[str, str] = {}
    if SCALAR_LABELS_PATH.exists():
        try:
            scalar_labels = json.loads(SCALAR_LABELS_PATH.read_text())
        except Exception:
            pass
    scalar_colors: Dict[str, str] = _load_scalar_colors()
    quadrant_data = get_quadrant_data(cohens_df, tract_base_to_type)
    atlas_data = get_atlas_data(cohens_df, tract_base_to_type)

    full_long = get_combined_long(tract_df, region_df)
    full_long = _exclude_volumetric_asymmetry_tracts(full_long)
    cohens_df_mahal = load_mahalanobis_cohens_df(
        subcortical_bases, cortical_bases_4s, glasser_bases
    )
    cohens_df_factor = compute_factor_z_cohens_df(tract_base_to_type)
    save_summary_tables_per_atlas(
        full_long,
        cohens_df,
        tract_base_to_type,
        OUTPUT_DIR,
        cohens_df_mahal,
        cohens_df_factor,
    )
    save_summary_tables_tex_per_atlas(
        full_long,
        cohens_df,
        tract_base_to_type,
        OUTPUT_DIR,
        cohens_df_mahal,
        cohens_df_factor,
    )

    plot1_bars_path = figures_dir / "plot1_mean_abs_cohend_by_scalar_bars.png"
    plot1_strips_path = figures_dir / "plot1_mean_abs_cohend_by_scalar_strips.png"
    plot2_bars_path = figures_dir / "plot2_2x2_mean_abs_cohend_bars.png"
    plot2_strips_path = figures_dir / "plot2_2x2_mean_abs_cohend_strips.png"
    plot1_radar_path = figures_dir / "plot1_radar_mean_abs_cohend.png"
    plot1_radar_sorted_path = figures_dir / "plot1_radar_mean_abs_cohend_sorted.png"
    plot2_radar_path = figures_dir / "plot2_2x2_radar_mean_abs_cohend.png"
    plot2_radar_sorted_path = figures_dir / "plot2_2x2_radar_mean_abs_cohend_sorted.png"

    plot1_whole_brain_bars(cohens_df, plot1_bars_path, scalar_labels, scalar_colors)
    plot1_whole_brain_strips(cohens_df, plot1_strips_path, scalar_labels, scalar_colors)
    by_scalar1 = cohens_df.groupby("scalar")["cohens_d"].apply(lambda s: np.abs(s).mean())
    scalars_ordered = sorted(scalar_labels.keys()) if scalar_labels else sorted(cohens_df["scalar"].unique().tolist())
    scalars_ordered = [s for s in scalars_ordered if s not in EXCLUDED_SCALARS]
    scalar_to_d1 = {s: by_scalar1.get(s, float("nan")) for s in scalars_ordered}
    plot_radar_mean_abs_cohend(
        scalars_ordered,
        scalar_to_d1,
        scalar_colors,
        plot1_radar_path,
        title="Mean absolute asymmetries\nWhole-brain (fixed scalar order)",
    )
    cdf_wb = cohens_df[~cohens_df["scalar"].isin(EXCLUDED_SCALARS)].copy()
    cdf_wb["abs_cohens_d"] = np.abs(cdf_wb["cohens_d"])
    scalars_wb_sorted = _sort_scalars_by_mean_abs_desc(scalars_ordered, cdf_wb)
    scalar_to_d1_sorted = {s: scalar_to_d1.get(s, float("nan")) for s in scalars_wb_sorted}
    plot_radar_mean_abs_cohend(
        scalars_wb_sorted,
        scalar_to_d1_sorted,
        scalar_colors,
        plot1_radar_sorted_path,
        title="Mean absolute asymmetries\nWhole-brain (sorted by mean |d|)",
    )
    plot2_2x2_bars(cohens_df, tract_base_to_type, plot2_bars_path, scalar_labels, scalar_colors)
    plot2_2x2_strips(cohens_df, tract_base_to_type, plot2_strips_path, scalar_labels, scalar_colors)
    plot2_2x2_radar(cohens_df, tract_base_to_type, plot2_radar_path, scalar_labels, scalar_colors, sort_by_mean=False)
    plot2_2x2_radar(cohens_df, tract_base_to_type, plot2_radar_sorted_path, scalar_labels, scalar_colors, sort_by_mean=True)

    plot3_path = plot3_2x2_brain_maps(
        cohens_df,
        quadrant_data,
        subcortical_bases,
        cortical_bases_4s,
        tract_base_to_type,
        tract_metadata_df,
        figures_dir,
    )
    if plot3_path is None:
        plot3_path = Path("")

    create_report_html(
        plot1_bars_path,
        plot1_strips_path,
        plot2_bars_path,
        plot2_strips_path,
        plot3_path,
        OUTPUT_DIR / "microstructural_asymmetry_report.html",
        figures_dir,
        quadrant_data=quadrant_data,
        atlas_data=atlas_data,
        plot1_radar_path=plot1_radar_path,
        plot1_radar_sorted_path=plot1_radar_sorted_path,
        plot2_radar_path=plot2_radar_path,
        plot2_radar_sorted_path=plot2_radar_sorted_path,
        factor_z_plot_path=factor_z_plot_path,
        factor_z_gm_wm_plot_path=factor_z_gm_wm_plot_path,
    )
    print(f"Report written to {OUTPUT_DIR / 'microstructural_asymmetry_report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
