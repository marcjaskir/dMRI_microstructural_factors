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
Microstructural asymmetry report (Mahalanobis): 2x2 brain maps and summary tables by atlas.
Uses segment-level tract Mahalanobis (*_asym_mahal_segment.csv) and region Mahalanobis (*_asym_mahal_regions.csv).
Mahalanobis distance was computed using region-specific scalar covariance estimated in controls.
Paired Cohen's d from (ipsi_mahal - contra_mahal) per ROI; segment-level WM and region-level GM.
"""
from __future__ import annotations

import html as html_module
import pickle
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths and config
# -----------------------------------------------------------------------------
PROJECT_ROOT = project_root()
TRACT_ASYM_DIR = analysis_dir() / "tract_asymmetry"
REGION_ASYM_DIR = analysis_dir() / "region_asymmetry_tle"
ATLAS_TSV_4S = PROJECT_ROOT / "data" / "atlases" / "4S" / "atlas-4S156Parcels_dseg.tsv"
ATLAS_NII_4S = PROJECT_ROOT / "data" / "atlases" / "4S" / "tpl-MNI152NLin2009cAsym_atlas-4S156Parcels_res-01_dseg.nii.gz"
ATLAS_NII_4S_FALLBACK = PROJECT_ROOT / "data" / "atlases" / "4S" / "tpl-MNI152NLin2009cAsym_atlas-4S156Parcels_res-01_dseg.nii"
ATLAS_NII_GLASSER = PROJECT_ROOT / "data" / "atlases" / "Glasser" / "atlas-Glasser_space-MNI152NLin2009cAsym_res-1_dseg.nii.gz"
ATLAS_TSV_GLASSER = PROJECT_ROOT / "data" / "atlases" / "Glasser" / "atlas-Glasser_dseg.tsv"
GLASSER_PARC_PATH = PROJECT_ROOT / "data" / "atlases" / "Glasser" / "glasser_parc.csv"
GLASSER_ADDITIONAL_METADATA_PATH = (
    PROJECT_ROOT / "data" / "atlases" / "Glasser" / "glasser_additional_metadata.csv"
)
SA_RANKS_PATH = PROJECT_ROOT / "data" / "atlases" / "S-A_ArchetypalAxis" / "Glasser360_MMP" / "Sensorimotor_Association_Axis_AverageRanks.csv"
TRACT_METADATA_PATH = PROJECT_ROOT / "data" / "atlases" / "HCP1065" / "HCP1065_tract_metadata.csv"
ENDPOINT_NII_DIR = PROJECT_ROOT / "data" / "atlases" / "HCP1065" / "endpoint_nii_bin"
OUTPUT_DIR = analysis_dir() / "microstructural_asymmetries"
INCLUSION_PATH = PROJECT_ROOT / "results" / "inclusion" / "penn_epilepsy_included_basic_metadata.csv"

ATLAS_TOP_N = 20
# Top N regions in truncated LaTeX summary fragments (*_top25.tex, etc.).
SUMMARY_TEX_TOP_N = 25

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

# CIT168-style 4S subcortex parcel abbreviations -> prose labels for LaTeX tables.
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


def _load_subject_group() -> Dict[str, str]:
    """Load TLE inclusion CSV; return sub -> 'left_TLE' or 'right_TLE' from laterality column."""
    out: Dict[str, str] = {}
    if not INCLUSION_PATH.exists():
        return out
    try:
        df = pd.read_csv(INCLUSION_PATH)
        if "sub" not in df.columns or "laterality" not in df.columns:
            return out
        if "lobe" in df.columns:
            df = df[df["lobe"].astype(str).str.strip().str.lower() == "temporal"]
        for _, row in df.iterrows():
            sub = row.get("sub")
            if pd.isna(sub):
                continue
            sub = str(sub)
            lat = str(row.get("laterality", "")).strip().lower()
            if lat == "left":
                out[sub] = "left_TLE"
            elif lat == "right":
                out[sub] = "right_TLE"
    except Exception:
        pass
    return out


def _load_sub_to_mts() -> Dict[str, object]:
    """Map subject id -> MTS indicator from inclusion CSV (``mts`` or ``lesion_mts`` column)."""
    out: Dict[str, object] = {}
    if not INCLUSION_PATH.exists():
        return out
    try:
        df = pd.read_csv(INCLUSION_PATH)
        if "sub" not in df.columns:
            return out
        col = None
        for c in ("mts", "lesion_mts"):
            if c in df.columns:
                col = c
                break
        if col is None:
            return out
        for _, row in df.iterrows():
            sub = row.get("sub")
            if pd.isna(sub):
                continue
            out[str(sub)] = row[col]
    except Exception:
        pass
    return out


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


def _get_4s_subcortical_base_to_labels() -> Dict[str, Tuple[str, str]]:
    """Map region_base -> (left_label, right_label) for 4S subcortex. Handles LH_/RH_ and LH-/RH- formats."""
    out: Dict[str, Tuple[str, str]] = {}
    if not ATLAS_TSV_4S.exists():
        return out
    try:
        df = pd.read_csv(ATLAS_TSV_4S, sep="\t")
        if "label" not in df.columns or "network_label" not in df.columns:
            return out
        net = df["network_label"].fillna("n/a").astype(str).str.strip().str.lower()
        subcort = df.loc[net.isin(("n/a", "nan", "")), "label"].tolist()
        for label in subcort:
            if label.startswith("LH_"):
                base, right = label[3:], "RH_" + label[3:]
                if right in subcort:
                    out[base] = (label, right)
            elif label.startswith("LH-"):
                base, right = label[3:], "RH-" + label[3:]
                if right in subcort:
                    out[base] = (label, right)
    except Exception:
        pass
    return out


def _get_4s_cortical_bases() -> set:
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


def _load_tract_metadata() -> Tuple[pd.DataFrame, Dict[str, str]]:
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
    if "_" not in roi_id:
        return roi_id, ""
    return roi_id.rsplit("_", 1)[0], roi_id.rsplit("_", 1)[1]


def _wm_roi_tract_base_key(roi_id: str) -> str:
    """Hemisphere-agnostic HCP1065 tract key (``AF_L_core`` / ``AF_core`` -> ``AF``)."""
    tract_with_hemi, _ = _wm_roi_to_tract_segment(str(roi_id).strip())
    return _wm_tract_base_key(tract_with_hemi)


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


def _load_4s_label_to_index() -> Tuple[Dict[str, int], Dict[str, str]]:
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


def load_tract_mahal() -> pd.DataFrame:
    """Load all tract_asymmetry *_asym_mahal_segment.csv into one DataFrame. roi_id=tract_segment, roi_type=wm."""
    rows: List[dict] = []
    for sub_dir in TRACT_ASYM_DIR.iterdir():
        if not sub_dir.is_dir():
            continue
        csv_path = sub_dir / f"{sub_dir.name}_asym_mahal_segment.csv"
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


def load_region_mahal(
    subcortical_bases: set,
    cortical_bases_4s: set,
    glasser_bases: set,
) -> pd.DataFrame:
    """Load all region_asymmetry_tle *_asym_mahal_regions.csv. roi_id=region, roi_type, atlas."""
    rows: List[dict] = []
    for sub_dir in REGION_ASYM_DIR.iterdir():
        if not sub_dir.is_dir():
            continue
        csv_path = sub_dir / f"{sub_dir.name}_asym_mahal_regions.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty or "region" not in df.columns:
            continue
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
        df = df[df["atlas"] != ""].copy()
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compute_cohens_d_mahal(
    tract_df: pd.DataFrame,
    region_df: pd.DataFrame,
    variant: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute paired Cohen's d per (roi_id, roi_type) from ipsi_mahal_{variant} - contra_mahal_{variant}.
    variant selects the column suffix in the input data (e.g. 'raw'). Returns (cohens_df, full_long)."""
    ipsi_col = f"ipsi_mahal_{variant}"
    contra_col = f"contra_mahal_{variant}"
    combined = []
    if not tract_df.empty:
        part = tract_df[["sub", "roi_id", "roi_type", ipsi_col, contra_col]].copy()
        part = part.rename(columns={ipsi_col: "ipsi", contra_col: "contra"})
        part["atlas"] = ""
        combined.append(part)
    if not region_df.empty:
        need = ["sub", "roi_id", "roi_type", ipsi_col, contra_col]
        if "atlas" in region_df.columns:
            part = region_df[need + ["atlas"]].copy()
        else:
            part = region_df[need].copy()
            part["atlas"] = ""
        part = part.rename(columns={ipsi_col: "ipsi", contra_col: "contra"})
        combined.append(part)
    if not combined:
        return pd.DataFrame(columns=["roi_id", "roi_type", "cohens_d", "atlas"]), pd.DataFrame()
    full = pd.concat(combined, ignore_index=True)
    if "atlas" not in full.columns:
        full["atlas"] = ""
    results = []
    group_cols = ["roi_id", "roi_type"]
    if "atlas" in full.columns:
        group_cols = group_cols + ["atlas"]
    for key, grp in full.groupby(group_cols):
        if isinstance(key, tuple):
            roi_id, roi_type = key[0], key[1]
            atlas = key[2] if len(key) > 2 else ""
        else:
            roi_id, roi_type = key
            atlas = ""
        ipsi = grp["ipsi"].dropna().tolist()
        contra = grp["contra"].dropna().tolist()
        if len(ipsi) != len(contra) or len(ipsi) < 2:
            continue
        d = _cohens_d_paired(ipsi, contra)
        if np.isfinite(d):
            row = {"roi_id": roi_id, "roi_type": roi_type, "cohens_d": d}
            if "atlas" in full.columns:
                row["atlas"] = atlas
            results.append(row)
    return pd.DataFrame(results), full


def add_quadrant_column(cohens_df: pd.DataFrame, tract_base_to_type: Dict[str, str]) -> pd.DataFrame:
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
    """Values are signed mean Cohen's d; lists are sorted by Cohen's d (descending)."""
    out: Dict[str, List] = {"cortex": [], "subcortex": [], "association": [], "projection": []}
    if cohens_df.empty:
        return out
    by_roi_mean = cohens_df.groupby("roi_id")["cohens_d"].mean()
    for roi_id, mean_d in by_roi_mean.items():
        if pd.isna(mean_d):
            continue
        row = cohens_df[cohens_df["roi_id"] == roi_id].iloc[0]
        roi_type = row["roi_type"]
        if roi_type == "cortical_gm":
            out["cortex"].append((str(roi_id), float(mean_d)))
        elif roi_type == "subcortical_gm":
            out["subcortex"].append((str(roi_id), float(mean_d)))
        elif roi_type == "wm":
            tract_base, segment = _wm_roi_to_tract_segment(roi_id)
            ttype = tract_base_to_type.get(tract_base, "")
            if ttype == "association":
                out["association"].append((tract_base, segment, float(mean_d)))
            elif ttype == "projection":
                out["projection"].append((tract_base, segment, float(mean_d)))
    for k in ("cortex", "subcortex"):
        out[k] = sorted(out[k], key=lambda x: x[1], reverse=True)
    for k in ("association", "projection"):
        out[k] = sorted(out[k], key=lambda x: x[2], reverse=True)
    return out


def get_atlas_data(
    cohens_df: pd.DataFrame,
    tract_base_to_type: Dict[str, str],
) -> Dict[str, List]:
    """Values are signed mean Cohen's d; lists are sorted by Cohen's d (descending)."""
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
            by_roi = sub.groupby("roi_id")["cohens_d"].mean()
            for roi_id, mean_d in by_roi.items():
                if pd.isna(mean_d):
                    continue
                out[atlas].append((str(roi_id), float(mean_d)))
            out[atlas] = sorted(out[atlas], key=lambda x: x[1], reverse=True)

    wm = cohens_df[cohens_df["roi_type"] == "wm"] if not cohens_df.empty else pd.DataFrame()
    if not wm.empty:
        by_roi = wm.groupby("roi_id")["cohens_d"].mean()
        for roi_id, mean_d in by_roi.items():
            if pd.isna(mean_d):
                continue
            if _is_excluded_volumetric_asymmetry_wm_roi(str(roi_id)):
                continue
            tract_base, segment = _wm_roi_to_tract_segment(roi_id)
            out["hcp1065_thirds"].append((tract_base, segment, float(mean_d)))
        out["hcp1065_thirds"] = sorted(out["hcp1065_thirds"], key=lambda x: x[2], reverse=True)
        by_tract = wm.copy()
        by_tract["tract_base"] = by_tract["roi_id"].apply(lambda x: _wm_roi_to_tract_segment(x)[0])
        tract_means = by_tract.groupby("tract_base")["cohens_d"].mean()
        for tract_base, mean_d in tract_means.items():
            if pd.isna(mean_d):
                continue
            if _is_excluded_volumetric_asymmetry_wm_roi(str(tract_base)):
                continue
            out["hcp1065_whole"].append((str(tract_base), float(mean_d)))
        out["hcp1065_whole"] = sorted(out["hcp1065_whole"], key=lambda x: x[1], reverse=True)
    return out


# Glasser community: Yeo only
GLASSER_COMMUNITY_TITLES = {"yeo": "Yeo functional network asymmetries"}


def _load_sa_ranks() -> Optional[pd.DataFrame]:
    """Load Sensorimotor_Association_Axis_AverageRanks.csv; return DataFrame with region, final.rank."""
    if not SA_RANKS_PATH.exists():
        return None
    try:
        df = pd.read_csv(SA_RANKS_PATH)
        if "region" not in df.columns or "final.rank" not in df.columns:
            return None
        return df[["region", "final.rank"]].copy()
    except Exception:
        return None


def _load_glasser_parc() -> Optional[pd.DataFrame]:
    """Load glasser_parc.csv; return DataFrame with region, economo, mesulam, sa, yeo; add base (region without _L/_R) for merging with roi_id."""
    if not GLASSER_PARC_PATH.exists():
        return None
    try:
        df = pd.read_csv(GLASSER_PARC_PATH)
        required = ["region", "economo", "mesulam", "sa", "yeo"]
        if not all(c in df.columns for c in required):
            return None
        out = df[list(required)].copy()
        out["base"] = out["region"].astype(str).str.rsplit("_", n=1).str[0]
        out = out.drop_duplicates(subset=["base"], keep="first")
        return out
    except Exception:
        return None


def plot_cortex_community_yeo(
    cohens_df: pd.DataFrame,
    glasser_parc: Optional[pd.DataFrame],
    out_dir: Path,
    suffix: str = "",
) -> List[Path]:
    """Yeo functional network swarm plot: signed Cohen's d (mean±SEM), x-axis order by mean Cohen's d. Square figure; y-axis '|Cohen's d|'. Returns single-element list with path."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = Path(out_dir)
    out_paths: List[Path] = []
    font_family = "Georgia"
    title_fontsize = 28
    label_fontsize = 24
    tick_fontsize = 20
    col = "yeo"

    if not cohens_df.empty and "atlas" in cohens_df.columns:
        cortex = cohens_df[(cohens_df["roi_type"] == "cortical_gm") & (cohens_df["atlas"] == "glasser")].copy()
    else:
        cortex = cohens_df[cohens_df["roi_type"] == "cortical_gm"] if not cohens_df.empty else pd.DataFrame()
    if cortex.empty or glasser_parc is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontfamily=font_family)
        p = out_dir / f"plot2_cortex_community_{col}{suffix}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(p)
        return out_paths

    def _base(s: str) -> str:
        s = str(s).strip()
        if s.startswith("Left_"):
            return s[5:].strip()
        if s.startswith("Right_"):
            return s[6:].strip()
        return s

    cortex = cortex.copy()
    cortex["roi_base"] = cortex["roi_id"].astype(str)
    cortex["base"] = cortex["roi_base"].apply(_base)
    merged = cortex.merge(
        glasser_parc[["base", "yeo"]],
        on="base",
        how="inner",
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.rcParams["font.family"] = font_family
    if merged.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No matching regions", ha="center", va="center", transform=ax.transAxes, fontfamily=font_family)
    else:
        order = merged.groupby(col)["cohens_d"].mean().sort_values(ascending=False).index.tolist()
        plot_df = merged[merged[col].isin(order)].copy()
        if plot_df.empty:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontfamily=font_family)
        else:
            sns.swarmplot(data=plot_df, x=col, y="cohens_d", order=order, ax=ax, color="steelblue", size=4, alpha=0.8)
            means = plot_df.groupby(col)["cohens_d"].mean().reindex(order)
            sems = plot_df.groupby(col)["cohens_d"].agg(lambda s: float(s.std() / np.sqrt(len(s))) if len(s) > 1 else 0.0).reindex(order)
            x_pos = np.arange(len(order))
            ax.errorbar(
                x_pos,
                means.values,
                yerr=sems.values,
                fmt="o",
                color="black",
                markersize=9,
                capsize=4,
                capthick=1.5,
                elinewidth=1.5,
            )
            ax.set_ylabel("|Cohen's d|", fontfamily=font_family, fontsize=label_fontsize)
            ax.set_xlabel("")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(l).title() for l in order], rotation=45, ha="right", fontfamily=font_family, fontsize=tick_fontsize)
            ax.tick_params(axis="y", labelsize=tick_fontsize)
            for label in ax.get_yticklabels():
                label.set_fontfamily(font_family)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(GLASSER_COMMUNITY_TITLES[col], fontfamily=font_family, fontsize=title_fontsize)
    plt.tight_layout()
    p = out_dir / f"plot2_cortex_community_{col}{suffix}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(p)
    return out_paths


def save_summary_tables_per_atlas_mahal(
    full_long: pd.DataFrame,
    cohens_df: pd.DataFrame,
    tract_base_to_type: Dict[str, str],
    report_dir: Path,
    suffix: str,
) -> None:
    """Save per-atlas summary CSVs: label, mean_ipsi, mean_contra, mean_asymmetry, mean_cohen_d, sorted by mean_cohen_d desc."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    tables = build_summary_table_dataframes_mahal(full_long, cohens_df)
    if not tables:
        return
    csv_name = {
        "4s_subcortex": f"summary_4s_subcortex{suffix}.csv",
        "4s_cortex": f"summary_4s_cortex{suffix}.csv",
        "glasser": f"summary_glasser{suffix}.csv",
        "hcp1065_thirds": f"summary_hcp1065_thirds{suffix}.csv",
        "hcp1065_whole": f"summary_hcp1065_whole{suffix}.csv",
    }
    for key, df in tables.items():
        if key in csv_name:
            df.to_csv(report_dir / csv_name[key], index=False)


def _roi_stats_for_summary_mahal(grp: pd.DataFrame) -> dict:
    ipsi = grp["ipsi"].dropna()
    contra = grp["contra"].dropna()
    diff = grp["ipsi"] - grp["contra"]
    return {
        "mean_ipsi": float(ipsi.mean()) if len(ipsi) else float("nan"),
        "mean_contra": float(contra.mean()) if len(contra) else float("nan"),
        "mean_asymmetry": float(diff.mean()) if len(diff) and diff.notna().any() else float("nan"),
    }


def build_summary_table_dataframes_mahal(
    full_long: pd.DataFrame,
    cohens_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Build per-atlas Mahalanobis summary tables for shared CSV and LaTeX export."""
    out: Dict[str, pd.DataFrame] = {}
    if full_long.empty or cohens_df.empty:
        return out

    cols = ["label", "mean_ipsi", "mean_contra", "mean_asymmetry", "mean_cohen_d"]

    for atlas in ("4s_subcortex", "4s_cortex", "glasser"):
        sub = full_long[full_long["atlas"] == atlas]
        if sub.empty:
            continue
        roi_stats = sub.groupby("roi_id").apply(
            lambda g: pd.Series(_roi_stats_for_summary_mahal(g)),
            include_groups=False,
        )
        if roi_stats.empty:
            continue
        roi_stats = roi_stats.reset_index()
        roi_stats = roi_stats.rename(columns={"roi_id": "label"})
        sub_cohens = cohens_df[cohens_df["atlas"] == atlas]
        if not sub_cohens.empty:
            d_agg = sub_cohens.groupby("roi_id").agg(
                mean_cohen_d=("cohens_d", lambda s: s.mean()),
            ).reset_index()
            roi_stats = roi_stats.merge(
                d_agg.rename(columns={"roi_id": "label"}),
                on="label",
                how="left",
            )
        else:
            roi_stats["mean_cohen_d"] = float("nan")
        roi_stats = roi_stats.sort_values("mean_cohen_d", ascending=False, na_position="last")
        out[atlas] = roi_stats[cols]

    wm_long = full_long[full_long["roi_type"] == "wm"]
    wm_cohens = cohens_df[cohens_df["roi_type"] == "wm"]
    if not wm_long.empty:
        wm_long = wm_long[
            ~wm_long["roi_id"].map(
                lambda rid: _is_excluded_volumetric_asymmetry_wm_roi(str(rid))
            )
        ].copy()
    if not wm_cohens.empty:
        wm_cohens = wm_cohens[
            ~wm_cohens["roi_id"].map(
                lambda rid: _is_excluded_volumetric_asymmetry_wm_roi(str(rid))
            )
        ].copy()
    if not wm_long.empty and not wm_cohens.empty:
        roi_stats = wm_long.groupby("roi_id").apply(
            lambda g: pd.Series(_roi_stats_for_summary_mahal(g)),
            include_groups=False,
        )
        roi_stats = roi_stats.reset_index()
        roi_stats = roi_stats.rename(columns={"roi_id": "label"})
        d_agg = wm_cohens.groupby("roi_id").agg(
            mean_cohen_d=("cohens_d", lambda s: s.mean()),
        ).reset_index()
        roi_stats = roi_stats.merge(d_agg.rename(columns={"roi_id": "label"}), on="label", how="left")
        roi_stats = roi_stats.sort_values("mean_cohen_d", ascending=False, na_position="last")
        out["hcp1065_thirds"] = roi_stats[cols]

        wm_long = wm_long.copy()
        wm_long["tract_base"] = wm_long["roi_id"].apply(lambda x: _wm_roi_to_tract_segment(x)[0])
        tract_stats = wm_long.groupby("tract_base").apply(
            lambda g: pd.Series(_roi_stats_for_summary_mahal(g)),
            include_groups=False,
        )
        tract_stats = tract_stats.reset_index()
        tract_stats = tract_stats.rename(columns={"tract_base": "label"})
        wm_cohens = wm_cohens.copy()
        wm_cohens["tract_base"] = wm_cohens["roi_id"].apply(lambda x: _wm_roi_to_tract_segment(x)[0])
        d_tract = wm_cohens.groupby("tract_base").agg(
            mean_cohen_d=("cohens_d", lambda s: s.mean()),
        ).reset_index()
        tract_stats = tract_stats.merge(d_tract.rename(columns={"tract_base": "label"}), on="label", how="left")
        tract_stats = tract_stats.sort_values("mean_cohen_d", ascending=False, na_position="last")
        out["hcp1065_whole"] = tract_stats[cols]

    return out


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
    """Region name bases that belong to thalamus nuclei in 4S156."""
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
    """Strip side suffix from HCP1065 metadata names for base tract labels."""
    n = str(raw_name).strip()
    if n.endswith("_L") or n.endswith("_R"):
        n = n[:-2]
    return n.replace("_", " ")


def _load_tract_label_to_pretty_name() -> Dict[str, str]:
    """HCP1065 tract label -> long name from metadata ``name`` column."""
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


def _format_tex_region_label(raw_label: str, atlas: str, thalamus_bases: Set[str]) -> str:
    """Pretty region label for GM tables."""
    lab = str(raw_label).strip()
    if atlas == "4s_subcortex" and lab in thalamus_bases:
        return f"Thalamus-{lab.replace('_', ' ')}"
    return lab.replace("_", " ")


def _format_4s_subcortex_tex_label(raw_label: str, thalamus_bases: Set[str]) -> str:
    """4S subcortex LaTeX labels with CIT168-friendly names."""
    lab = str(raw_label).strip()
    if lab in thalamus_bases:
        return f"Thalamus-{lab.replace('_', ' ')}"
    if lab in FOUR_S_SUBCORTEX_TEX_ABBREV_LABELS:
        return FOUR_S_SUBCORTEX_TEX_ABBREV_LABELS[lab]
    return lab.replace("_", " ")


def _format_tex_wm_thirds_label(roi_id: str, tract_names: Dict[str, str]) -> str:
    """WM roi_id like ILF_L_A -> Inferior Longitudinal Fasciculus L - Anterior."""
    tract_hemi, seg = _wm_roi_to_tract_segment(roi_id)
    tract_pretty = tract_names.get(tract_hemi, tract_hemi.replace("_", " "))
    seg_pretty = _hcp1065_segment_human(seg)
    return f"{tract_pretty} - {seg_pretty}"


def _load_glasser_additional_metadata() -> pd.DataFrame:
    """Glasser parcel metadata indexed by unilateral ``region`` id."""
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


def _save_summary_table_tex_mahal(
    df: pd.DataFrame,
    out_path: Path,
    *,
    region_col_formatter: Optional[Callable[[str], str]] = None,
    first_column_header: str = "Region",
    top_n: Optional[int] = None,
) -> None:
    """Write a ``longtable`` fragment with signed Mahalanobis Cohen's d."""
    if df.empty:
        return
    if top_n is not None:
        df = df.head(int(top_n)).copy()
    if df.empty:
        return
    cohen_header = r"$\mathrm{Cohen's\ }d$"
    header_parts = [_latex_escape(first_column_header), cohen_header]
    header_line = " & ".join(header_parts) + r" \\"
    header_block = [
        r"\toprule",
        header_line,
        r"\midrule",
    ]
    lines = [
        "% Requires \\usepackage{booktabs,longtable} in the main document.",
        r"\begin{longtable}{@{}lr@{}}",
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
        lab = region_col_formatter(raw_lab) if region_col_formatter is not None else raw_lab
        lab_tex = _latex_escape(str(lab))
        v = row["mean_cohen_d"]
        val_tex = "---" if pd.isna(v) else f"{float(v):.4f}"
        lines.append(" & ".join([lab_tex, val_tex]) + r" \\")
    lines.append(r"\end{longtable}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary_tables_tex_per_atlas_mahal(
    full_long: pd.DataFrame,
    cohens_df: pd.DataFrame,
    tract_base_to_type: Dict[str, str],
    report_dir: Path,
) -> None:
    """Write atlas-specific Mahalanobis LaTeX ``longtable`` files."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    tables = build_summary_table_dataframes_mahal(full_long, cohens_df)
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

        _save_summary_table_tex_mahal(
            df,
            report_dir / f"summary_{atlas}_mahalanobis.tex",
            region_col_formatter=_fmt,
        )
        _save_summary_table_tex_mahal(
            df,
            report_dir / f"summary_{atlas}_mahalanobis_top{SUMMARY_TEX_TOP_N}.tex",
            region_col_formatter=_fmt,
            top_n=SUMMARY_TEX_TOP_N,
        )

    df_whole = tables.get("hcp1065_whole")
    if df_whole is not None and not df_whole.empty:

        def _fmt_wm_whole(lab: str) -> str:
            return tract_pretty.get(str(lab), str(lab).replace("_", " "))

        _save_summary_table_tex_mahal(
            df_whole,
            report_dir / "summary_hcp1065_whole_mahalanobis.tex",
            region_col_formatter=_fmt_wm_whole,
            first_column_header="Tract segment",
        )
        _save_summary_table_tex_mahal(
            df_whole,
            report_dir / f"summary_hcp1065_whole_mahalanobis_top{SUMMARY_TEX_TOP_N}.tex",
            region_col_formatter=_fmt_wm_whole,
            first_column_header="Tract segment",
            top_n=SUMMARY_TEX_TOP_N,
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
            _save_summary_table_tex_mahal(
                df_thirds.loc[mask_a].copy(),
                report_dir / "summary_hcp1065_thirds_association_mahalanobis.tex",
                region_col_formatter=fmt_wm,
                first_column_header="Tract segment",
            )
            _save_summary_table_tex_mahal(
                df_thirds.loc[mask_a].copy(),
                report_dir / f"summary_hcp1065_thirds_association_mahalanobis_top{SUMMARY_TEX_TOP_N}.tex",
                region_col_formatter=fmt_wm,
                first_column_header="Tract segment",
                top_n=SUMMARY_TEX_TOP_N,
            )
        if mask_p.any():
            _save_summary_table_tex_mahal(
                df_thirds.loc[mask_p].copy(),
                report_dir / "summary_hcp1065_thirds_projection_mahalanobis.tex",
                region_col_formatter=fmt_wm,
                first_column_header="Tract segment",
            )
            _save_summary_table_tex_mahal(
                df_thirds.loc[mask_p].copy(),
                report_dir / f"summary_hcp1065_thirds_projection_mahalanobis_top{SUMMARY_TEX_TOP_N}.tex",
                region_col_formatter=fmt_wm,
                first_column_header="Tract segment",
                top_n=SUMMARY_TEX_TOP_N,
            )


_SUMMARY_BILATERAL_COLS = ["sub", "group", "mts", "label", "mahalanobis"]


def save_summary_bilateral(
    full_long: pd.DataFrame,
    subject_group: Dict[str, str],
    subcortical_base_to_labels: Dict[str, Tuple[str, str]],
    report_dir: Path,
) -> None:
    """Save subject-level bilateral ROI Mahalanobis: sub, group, mts, label, mahalanobis.

    ``mts`` is taken per subject from ``penn_epilepsy_included_basic_metadata.csv``
    (``mts`` or ``lesion_mts`` column). label = hemisphere-specific ROIs (e.g. LH_Vis_1, AF_L);
    group = left_TLE or right_TLE.
    Outputs: summary_*_mahalanobis_bilateral.csv for 4s_cortex, 4s_subcortex, glasser,
    hcp1065_whole, hcp1065_thirds."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    sub_to_mts = _load_sub_to_mts()

    def _expand_bilateral(rows: List[dict], sub_df: pd.DataFrame, left_label_fn, right_label_fn) -> None:
        """Expand bilateral (base, ipsi, contra) to two rows per ROI (label, mahalanobis)."""
        for _, r in sub_df.iterrows():
            sub = str(r["sub"])
            group = subject_group.get(sub)
            if group not in ("left_TLE", "right_TLE"):
                continue
            base = str(r["roi_id"])
            ipsi = float(r["ipsi"])
            contra = float(r["contra"])
            left_lbl = left_label_fn(base)
            right_lbl = right_label_fn(base)
            if left_lbl is None or right_lbl is None:
                continue
            mts_val = sub_to_mts.get(sub, pd.NA)
            if group == "left_TLE":
                rows.append({"sub": sub, "group": group, "mts": mts_val, "label": left_lbl, "mahalanobis": ipsi})
                rows.append({"sub": sub, "group": group, "mts": mts_val, "label": right_lbl, "mahalanobis": contra})
            else:
                rows.append({"sub": sub, "group": group, "mts": mts_val, "label": left_lbl, "mahalanobis": contra})
                rows.append({"sub": sub, "group": group, "mts": mts_val, "label": right_lbl, "mahalanobis": ipsi})

    # 4S cortex: LH_{base}, RH_{base}
    cortex = full_long[(full_long["atlas"] == "4s_cortex") & full_long["ipsi"].notna() & full_long["contra"].notna()]
    if not cortex.empty:
        rows_4s: List[dict] = []
        _expand_bilateral(rows_4s, cortex, lambda b: f"LH_{b}", lambda b: f"RH_{b}")
        if rows_4s:
            pd.DataFrame(rows_4s)[_SUMMARY_BILATERAL_COLS].to_csv(
                report_dir / "summary_4s_cortex_mahalanobis_bilateral.csv", index=False
            )

    # 4S subcortex: use atlas label format (LH_/RH_ or LH-/RH-)
    subcort = full_long[(full_long["atlas"] == "4s_subcortex") & full_long["ipsi"].notna() & full_long["contra"].notna()]
    if not subcort.empty:
        rows_sub: List[dict] = []
        def _subcort_left(b): return subcortical_base_to_labels.get(b, (None, None))[0]
        def _subcort_right(b): return subcortical_base_to_labels.get(b, (None, None))[1]
        _expand_bilateral(rows_sub, subcort, _subcort_left, _subcort_right)
        if rows_sub:
            pd.DataFrame(rows_sub)[_SUMMARY_BILATERAL_COLS].to_csv(
                report_dir / "summary_4s_subcortex_mahalanobis_bilateral.csv", index=False
            )

    # Glasser: Left_{base}, Right_{base}
    glasser = full_long[(full_long["atlas"] == "glasser") & full_long["ipsi"].notna() & full_long["contra"].notna()]
    if not glasser.empty:
        rows_gl: List[dict] = []
        _expand_bilateral(rows_gl, glasser, lambda b: f"Left_{b}", lambda b: f"Right_{b}")
        if rows_gl:
            pd.DataFrame(rows_gl)[_SUMMARY_BILATERAL_COLS].to_csv(
                report_dir / "summary_glasser_mahalanobis_bilateral.csv", index=False
            )

    # HCP1065 thirds: roi_id = tract_base_segment (e.g. AF_1) -> AF_L_1, AF_R_1
    wm = full_long[(full_long["roi_type"] == "wm") & full_long["ipsi"].notna() & full_long["contra"].notna()]
    if not wm.empty:
        rows_thirds: List[dict] = []
        for _, r in wm.iterrows():
            sub = str(r["sub"])
            group = subject_group.get(sub)
            if group not in ("left_TLE", "right_TLE"):
                continue
            mts_val = sub_to_mts.get(sub, pd.NA)
            tract_base, segment = _wm_roi_to_tract_segment(str(r["roi_id"]))
            ipsi, contra = float(r["ipsi"]), float(r["contra"])
            if group == "left_TLE":
                rows_thirds.append(
                    {"sub": sub, "group": group, "mts": mts_val, "label": f"{tract_base}_L_{segment}", "mahalanobis": ipsi}
                )
                rows_thirds.append(
                    {"sub": sub, "group": group, "mts": mts_val, "label": f"{tract_base}_R_{segment}", "mahalanobis": contra}
                )
            else:
                rows_thirds.append(
                    {"sub": sub, "group": group, "mts": mts_val, "label": f"{tract_base}_L_{segment}", "mahalanobis": contra}
                )
                rows_thirds.append(
                    {"sub": sub, "group": group, "mts": mts_val, "label": f"{tract_base}_R_{segment}", "mahalanobis": ipsi}
                )
        if rows_thirds:
            pd.DataFrame(rows_thirds)[_SUMMARY_BILATERAL_COLS].to_csv(
                report_dir / "summary_hcp1065_thirds_mahalanobis_bilateral.csv", index=False
            )

        # HCP1065 whole: aggregate by tract_base, then expand to tract_L, tract_R
        wm_copy = wm.copy()
        wm_copy["tract_base"] = wm_copy["roi_id"].apply(lambda x: _wm_roi_to_tract_segment(str(x))[0])
        agg = wm_copy.groupby(["sub", "tract_base"]).agg(ipsi=("ipsi", "mean"), contra=("contra", "mean")).reset_index()
        rows_whole: List[dict] = []
        for _, r in agg.iterrows():
            sub = str(r["sub"])
            group = subject_group.get(sub)
            if group not in ("left_TLE", "right_TLE"):
                continue
            mts_val = sub_to_mts.get(sub, pd.NA)
            tb = str(r["tract_base"])
            ipsi, contra = float(r["ipsi"]), float(r["contra"])
            if group == "left_TLE":
                rows_whole.append({"sub": sub, "group": group, "mts": mts_val, "label": f"{tb}_L", "mahalanobis": ipsi})
                rows_whole.append({"sub": sub, "group": group, "mts": mts_val, "label": f"{tb}_R", "mahalanobis": contra})
            else:
                rows_whole.append({"sub": sub, "group": group, "mts": mts_val, "label": f"{tb}_L", "mahalanobis": contra})
                rows_whole.append({"sub": sub, "group": group, "mts": mts_val, "label": f"{tb}_R", "mahalanobis": ipsi})
        if rows_whole:
            pd.DataFrame(rows_whole)[_SUMMARY_BILATERAL_COLS].to_csv(
                report_dir / "summary_hcp1065_whole_mahalanobis_bilateral.csv", index=False
            )


def build_cortex_region_scores_for_brain_map(cohens_df: pd.DataFrame) -> Dict[str, float]:
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
    wm = cohens_df[cohens_df["roi_type"] == "wm"]
    if wm.empty:
        return {}
    wm = wm.copy()
    wm["tract_base"] = wm["roi_id"].apply(lambda x: _wm_roi_to_tract_segment(x)[0])
    by_tract = wm.groupby("tract_base")["cohens_d"].apply(lambda s: np.abs(s).mean())
    return {str(k): float(v) for k, v in by_tract.items() if np.isfinite(v)}


def build_hcp1065_thirds_scores(cohens_df: pd.DataFrame) -> Dict[str, float]:
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
    label_to_index, _ = _load_4s_label_to_index()
    roi_mean_abs_d = (
        cohens_df[cohens_df["roi_type"] == "subcortical_gm"]
        .groupby("roi_id")["cohens_d"]
        .apply(lambda s: np.abs(s).mean())
        .to_dict()
    )
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


def _summary_table_html(
    rows: List[Tuple[str, float]],
    title: str,
    value_header: str = "Effect size",
    roi_header: str = "Region",
    top_n: int = ATLAS_TOP_N,
) -> str:
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
    if not rows:
        return f'<div class="summary-table-wrap"><p><strong>{title}</strong></p><p>No data</p></div>'
    rows_to_show = sorted(rows, key=lambda x: x[2], reverse=True)[:top_n]
    lines = [
        f'<div class="summary-table-wrap"><p><strong>{title}</strong></p>',
        f'<table class="summary-table"><thead><tr><th>Tract</th><th>Segment</th><th>{html_module.escape(value_header)}</th></tr></thead><tbody>',
    ]
    for (tract, segment, val) in rows_to_show:
        lines.append(
            f'<tr><td>{html_module.escape(str(tract))}</td><td>{html_module.escape(str(segment))}</td><td>{val:.4f}</td></tr>'
        )
    lines.append("</tbody></table></div>")
    return "\n".join(lines)


def atlas_tables_html(
    atlas_data: Dict[str, List],
    value_header: str = "Mean Cohen's d",
    top_n: int = ATLAS_TOP_N,
) -> str:
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
            parts.append(_summary_table_wm_html(data, title, value_header, top_n=top_n))
        elif key == "hcp1065_whole" and data and len(data[0]) == 2:
            parts.append(
                _summary_table_html(data, title, value_header=value_header, roi_header="Tract", top_n=top_n)
            )
        elif data and len(data[0]) == 2:
            parts.append(
                _summary_table_html(data, title, value_header=value_header, roi_header="Region", top_n=top_n)
            )
        else:
            parts.append(
                f'<div class="summary-table-wrap"><p><strong>{html_module.escape(title)}</strong></p><p>No data</p></div>'
            )
    return '<div class="grid-tables-atlas">' + "".join(parts) + "</div>"


def plot3_2x2_brain_maps(
    cohens_df: pd.DataFrame,
    quadrant_data: Dict[str, List],
    subcortical_bases: set,
    cortical_bases_4s: set,
    tract_base_to_type: Dict[str, str],
    tract_metadata_df: pd.DataFrame,
    report_dir: Path,
    suffix: str,
) -> Optional[Path]:
    """Create 2x2 brain maps with output filenames including optional suffix. Returns path to composite PNG or None."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "code" / "analysis"))
        from microstructural_asymmetries import brain_maps as bm
    except Exception as e:
        import traceback
        print("Brain maps skipped (install nilearn for glass brain figures):", e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None
    vmax = 0.0
    for data in quadrant_data.values():
        for item in data:
            v = (
                item[-1]
                if isinstance(item, (list, tuple)) and len(item) >= 2
                else (item[1] if len(item) == 2 else item[2])
            )
            if isinstance(v, (int, float)):
                vmax = max(vmax, abs(v))
    if vmax <= 0:
        vmax = 1.0
    vmin, vmax = 0.0, vmax
    use_absolute = True

    cortex_scores = build_cortex_region_scores_for_brain_map(cohens_df)
    with open(report_dir / f"glasser_mean_abs_cohend{suffix}.pkl", "wb") as f:
        pickle.dump(cortex_scores, f)
    if cortex_scores and ATLAS_NII_GLASSER.exists() and ATLAS_TSV_GLASSER.exists():
        bm.create_gm_brain_map(
            cortex_scores,
            "Cortex GM",
            str(report_dir / f"plot3_cortex{suffix}"),
            ATLAS_NII_GLASSER,
            ATLAS_TSV_GLASSER,
            vmin=vmin,
            vmax=vmax,
            use_absolute=use_absolute,
            display_views=("x",),
        )
    subcortex_scores = build_subcortex_region_scores_for_brain_map(cohens_df, subcortical_bases)
    with open(report_dir / f"4s_subcortical_mean_abs_cohend{suffix}.pkl", "wb") as f:
        pickle.dump(subcortex_scores, f)
    s4_cortex_scores = build_4s_cortex_region_scores(cohens_df, cortical_bases_4s)
    with open(report_dir / f"4s_cortical_mean_abs_cohend{suffix}.pkl", "wb") as f:
        pickle.dump(s4_cortex_scores, f)
    hcp_whole_scores = build_hcp1065_whole_scores(cohens_df)
    with open(report_dir / f"hcp1065_whole_mean_abs_cohend{suffix}.pkl", "wb") as f:
        pickle.dump(hcp_whole_scores, f)
    hcp_thirds_scores = build_hcp1065_thirds_scores(cohens_df)
    with open(report_dir / f"hcp1065_thirds_mean_abs_cohend{suffix}.pkl", "wb") as f:
        pickle.dump(hcp_thirds_scores, f)
    atlas_4s = ATLAS_NII_4S if ATLAS_NII_4S.exists() else (ATLAS_NII_4S_FALLBACK if ATLAS_NII_4S_FALLBACK.exists() else None)
    if subcortex_scores and atlas_4s:
        bm.create_gm_brain_map(
            subcortex_scores,
            "Subcortex GM",
            str(report_dir / f"plot3_subcortex{suffix}"),
            Path(atlas_4s),
            ATLAS_TSV_4S,
            vmin=vmin,
            vmax=vmax,
            use_absolute=use_absolute,
            display_views=("x",),
        )
    wm_scores = build_wm_tract_segment_scores_for_brain_map(cohens_df, tract_base_to_type)
    if wm_scores and ENDPOINT_NII_DIR.exists() and not tract_metadata_df.empty:
        bm.create_wm_brain_map(
            wm_scores,
            "WM",
            str(report_dir / f"plot3_association{suffix}.png"),
            str(report_dir / f"plot3_projection{suffix}.png"),
            tract_metadata_df,
            ENDPOINT_NII_DIR,
            vmin=vmin,
            vmax=vmax,
            use_absolute=use_absolute,
            display_views=("x",),
        )

    cell_paths = [
        report_dir / f"plot3_cortex{suffix}_ctx_x.png",
        report_dir / f"plot3_association{suffix}_x.png",
        report_dir / f"plot3_subcortex{suffix}_sctx_x.png",
        report_dir / f"plot3_projection{suffix}_x.png",
    ]
    existing = [p for p in cell_paths if p.exists()]
    if not existing:
        return None
    try:
        from PIL import Image
    except Exception:
        return existing[0] if existing else None
    sub_w, cell_h = 600, 300
    composite = Image.new("RGB", (sub_w * 2, cell_h * 2), (248, 248, 248))
    for idx, img_path in enumerate(cell_paths):
        if not img_path.exists():
            continue
        ri, ci = idx // 2, idx % 2
        x0 = ci * sub_w
        y0 = ri * cell_h
        im = Image.open(img_path).convert("RGB")
        composite.paste(im.resize((sub_w, cell_h)), (x0, y0))
    out_path = report_dir / f"plot3_2x2_brain_maps{suffix}.png"
    composite.save(str(out_path))
    return out_path


def create_report_html_mahalanobis(
    out_html: Path,
    figures_dir: Path,
    plot3_path: Optional[Path],
    plot_community_paths: Optional[List[Path]] = None,
    atlas_tables: str = "",
) -> None:
    """Write HTML report: 2x2 brain map + Yeo functional network plot + atlas tables (top 20)."""
    p_brain = plot3_path.name if plot3_path and plot3_path.exists() else None
    p_comms = [p.name for p in (plot_community_paths or []) if p and p.exists()]
    fig_prefix = "figures/" if (out_html.parent / "figures").exists() else ""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Microstructural asymmetry report (Mahalanobis)</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1600px; margin: 2em auto; padding: 0 2em; }}
    h1 {{ font-size: 1.5rem; }}
    h2 {{ font-size: 1.2rem; margin-top: 1.5em; }}
    img {{ max-width: 100%; height: auto; }}
    .caption {{ color: #555; font-size: 0.9rem; margin-top: 0.5em; }}
    .figure-brain {{ margin: 2em 0; padding: 1.5em 0; max-width: 100%; }}
    .figure-brain img {{ display: block; max-width: 100%; width: auto; height: auto; margin: 1em auto; }}
    .grid-tables-atlas {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; margin-top: 16px; margin-bottom: 24px; width: 100%; }}
    .summary-table-wrap {{ border: 1px solid #ddd; padding: 10px; background: #fafafa; }}
    .summary-table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
    .summary-table th, .summary-table td {{ padding: 4px 8px; text-align: left; border: 1px solid #ddd; }}
    .summary-table th {{ background: #eee; }}
  </style>
</head>
<body>
  <h1>Microstructural asymmetry report (Mahalanobis)</h1>
  <p>Mahalanobis distance was computed using region-specific scalar covariance estimated in controls. Paired Cohen's d from (ipsi_mahal − contra_mahal) per ROI across subjects; segment-level WM and region-level GM.</p>

  <h2>2×2 brain maps, TLE Cohen's d by region group / Yeo / Mesulam, summary tables</h2>
  <p class="caption">Signed paired Cohen's d (ipsi − contra Mahalanobis) across temporal-lobe epilepsy subjects. Group bar charts show mean Cohen's d ± SEM across ROIs in each group (sorted descending). White bars = WM tract families; grey = GM (lobes and subcortex).</p>
  """ + (
    (f'<div class="figure-brain"><img src="{fig_prefix + p_brain}" alt="2x2 brain maps Mahalanobis"></div><p class="caption">One medial view per tissue category (color = magnitude of Cohen\'s d).</p>' if p_brain else '<p>No brain maps.</p>')
  ) + """
  """ + (
    "".join(
      f'<p><img src="{fig_prefix + p}" alt="TLE Cohen\'s d group summaries"></p>'
      for p in p_comms
    )
    + (
      '<p class="caption">Left: anatomical region groups (WM + GM). Middle: Glasser cortex by Yeo network. Right: Glasser cortex by Mesulam cytoarchitecture. CSVs: <code>summary_cohens_d_by_region_group_mahalanobis.csv</code>, <code>summary_cohens_d_by_yeo_mahalanobis.csv</code>, <code>summary_cohens_d_by_mesulam_mahalanobis.csv</code>.</p>'
      if p_comms
      else ""
    )
  ) + """
  """ + atlas_tables + """
  <h2>Subgroups</h2>
  <p><a href="subgroup_correlation_matrices.html">Left and right TLE correlation matrices</a> by tissue group (Glasser cortex, 4S subcortex, 4S cortex, association WM, projection WM). Subject × subject correlation of asymmetry profiles with hierarchical clustering; anonymized subject keys on axes.</p>

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

    tract_df = load_tract_mahal()
    region_df = load_region_mahal(subcortical_bases, cortical_bases_4s, glasser_bases)
    if tract_df.empty and region_df.empty:
        print("No Mahalanobis data (missing asym_mahal_segment or asym_mahal_regions CSVs).", file=sys.stderr)
        create_report_html_mahalanobis(
            OUTPUT_DIR / "microstructural_asymmetry_report_mahalanobis.html",
            figures_dir,
            None,
            None,
            atlas_tables_html({}, top_n=ATLAS_TOP_N),
        )
        return 1

    cohens_df, full_long = compute_cohens_d_mahal(tract_df, region_df, "raw")

    # Exclude requested tracts from tract-level WM analyses.
    cohens_df = _exclude_volumetric_asymmetry_tracts(cohens_df)
    full_long = _exclude_volumetric_asymmetry_tracts(full_long)

    save_summary_tables_per_atlas_mahal(full_long, cohens_df, tract_base_to_type, OUTPUT_DIR, "_mahalanobis")
    save_summary_tables_tex_per_atlas_mahal(full_long, cohens_df, tract_base_to_type, OUTPUT_DIR)
    subject_group = _load_subject_group()
    subcortical_base_to_labels = _get_4s_subcortical_base_to_labels()
    save_summary_bilateral(full_long, subject_group, subcortical_base_to_labels, OUTPUT_DIR)

    from region_score_summary_tables import save_region_score_tex_per_tle_group

    save_region_score_tex_per_tle_group(full_long, tract_base_to_type, OUTPUT_DIR)

    # Subgroup correlation matrices (left/right TLE by tissue group)
    import subprocess
    analysis_dir = Path(__file__).parent.parent

    if subgroup_script.exists():
        try:
            subprocess.run(
                [sys.executable, str(subgroup_script)],
                cwd=str(analysis_dir / "3_subgroups"),
                check=False,
                capture_output=True,
            )
        except Exception as e:
            print(f"Subgroup correlation matrices skipped: {e}", file=sys.stderr)

    quadrant_data = get_quadrant_data(cohens_df, tract_base_to_type)
    atlas_data = get_atlas_data(cohens_df, tract_base_to_type)

    plot3_path = plot3_2x2_brain_maps(
        cohens_df,
        quadrant_data,
        subcortical_bases,
        cortical_bases_4s,
        tract_base_to_type,
        tract_metadata_df,
        figures_dir,
        suffix="",
    )

    from mahalanobis_group_bars import (
        plot_tle_cohens_d_region_yeo_mesulam_bars,
        save_region_group_mapping_tex,
    )

    plot_community_paths: List[Path] = []
    grouped_bars = plot_tle_cohens_d_region_yeo_mesulam_bars(
        cohens_df, figures_dir, PROJECT_ROOT, suffix="_mahalanobis"
    )
    if grouped_bars is not None:
        plot_community_paths.append(grouped_bars)

    save_region_group_mapping_tex(
        cohens_df,
        OUTPUT_DIR / "region_to_region_group_mahalanobis.tex",
        PROJECT_ROOT,
        csv_path=OUTPUT_DIR / "region_to_region_group_mahalanobis.csv",
    )

    atlas_tables_html_str = atlas_tables_html(
        atlas_data,
        value_header="Mean Cohen's d (Mahalanobis)",
        top_n=ATLAS_TOP_N,
    )

    create_report_html_mahalanobis(
        OUTPUT_DIR / "microstructural_asymmetry_report_mahalanobis.html",
        figures_dir,
        plot3_path,
        plot_community_paths,
        atlas_tables_html_str,
    )
    print(f"Report written to {OUTPUT_DIR / 'microstructural_asymmetry_report_mahalanobis.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
