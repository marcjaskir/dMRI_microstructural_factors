"""Paths and constants for asymmetry TLE region pipeline."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

EXCLUDED_SCALARS = [
    "map_li", "map_am", "dti_txx", "dti_txy", "dti_txz",
    "dti_tyy", "dti_tyz", "dti_tzz", "dti_ha", "rdi_rd1", "rdi_rd2",
    "gqi_iso",
]

# Default bilateral region pair (4S atlas); used when --region is not provided
DEFAULT_REGION_LEFT = "LH_Hippocampus"
DEFAULT_REGION_RIGHT = "RH_Hippocampus"

# WM tract segment node ranges (HCP1065 / pyAFQ); same as asymmetry_tle
END1_NODES = list(range(1, 35))   # nodes 1-34
CORE_NODES = list(range(35, 67))  # nodes 35-66
END2_NODES = list(range(67, 101))  # nodes 67-100
SEGMENT_NODES: Dict[str, List[int]] = {
    "end1": END1_NODES,
    "core": CORE_NODES,
    "end2": END2_NODES,
}

def load_subject_outcomes(path: Optional[Path] = None) -> Dict[str, Optional[str]]:
    """Load subject surgical outcomes from an external CSV (sub, outcome columns).

    No subject identifiers are stored in this repository. Provide a local CSV via
    config.yaml ``subject_outcome_csv`` or pass ``path`` explicitly.
    """
    if path is None:
        import sys

        repo = Path(__file__).resolve()
    while repo != repo.parent and not (repo / "lib" / "paths.py").exists():
        repo = repo.parent
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from lib.paths import subject_outcome_csv

        path = subject_outcome_csv()
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "sub" not in df.columns or "outcome" not in df.columns:
        return {}
    return {
        str(row["sub"]): (None if pd.isna(row["outcome"]) else str(row["outcome"]))
        for _, row in df.iterrows()
    }


def get_atlas_labels(paths: Dict[str, Path], atlas: str) -> List[str]:
    """Load atlas parcel labels from TSV. 4S156 -> atlas_tsv_4s, Glasser -> atlas_tsv_glasser. Returns empty list if missing."""
    if atlas == "4S156":
        p = paths.get("atlas_tsv_4s")
    elif atlas == "Glasser":
        p = paths.get("atlas_tsv_glasser")
    else:
        p = None
    if not p or not Path(p).exists():
        return []
    try:
        df = pd.read_csv(p, sep="\t")
        col = "label" if "label" in df.columns else df.columns[1]
        return df[col].astype(str).tolist()
    except Exception:
        return []


def get_bilateral_pairs_from_atlas(paths: Dict[str, Path], atlas: str) -> List[Tuple[str, str]]:
    """Return [(left_label, right_label), ...] for all bilateral pairs.
    4S156: LH_*/RH_* or LH-*/RH-* with same suffix (underscore vs hyphen both supported). Glasser: Left_*/Right_* with same suffix."""
    labels = get_atlas_labels(paths, atlas)
    left_suffixes: Dict[str, str] = {}
    right_suffixes: Dict[str, str] = {}
    if atlas == "4S156":
        for lab in labels:
            lab = lab.strip()
            if lab.startswith("LH_"):
                left_suffixes[lab[3:]] = lab
            elif lab.startswith("LH-"):
                left_suffixes[lab[3:]] = lab
            elif lab.startswith("RH_"):
                right_suffixes[lab[3:]] = lab
            elif lab.startswith("RH-"):
                right_suffixes[lab[3:]] = lab
    elif atlas == "Glasser":
        for lab in labels:
            lab = lab.strip()
            if lab.startswith("Left_"):
                left_suffixes[lab[5:]] = lab
            elif lab.startswith("Right_"):
                right_suffixes[lab[6:]] = lab
    else:
        return []
    pairs: List[Tuple[str, str]] = []
    for suffix in left_suffixes:
        if suffix in right_suffixes:
            pairs.append((left_suffixes[suffix], right_suffixes[suffix]))
    return pairs


def resolve_region_to_pair(region_spec: Optional[str], paths: Dict[str, Path], atlas: str) -> Optional[Tuple[str, str]]:
    """Resolve a single --region string to (region_left, region_right). 4S156/Glasser only. Returns None if not found or if region_spec is None/empty."""
    if not region_spec or not str(region_spec).strip():
        return None
    region_spec = str(region_spec).strip().replace(" ", "_")
    pairs = get_bilateral_pairs_from_atlas(paths, atlas)
    if not pairs:
        return None
    labels = get_atlas_labels(paths, atlas)
    label_set = set(labels)
    # Suffix length: 4S156 LH_/RH_ = 3 chars; Glasser Left_/Right_ = 5 and 6 chars (suffix after Left_ = 5)
    left_prefix_len = 3 if atlas == "4S156" else 5
    lh_prefix, rh_prefix = ("LH_", "RH_") if atlas == "4S156" else ("Left_", "Right_")
    # Exact suffix match first
    for lh, rh in pairs:
        suffix = lh[left_prefix_len:]
        if suffix == region_spec:
            return (lh, rh)
    # Substring match: region_spec in suffix (case-insensitive)
    for lh, rh in pairs:
        suffix = lh[left_prefix_len:]
        if region_spec.lower() in suffix.lower():
            return (lh, rh)
    # Try constructed names (4S: both LH_/RH_ and LH-/RH- match factor-score / atlas conventions)
    if atlas == "4S156":
        for lh_p, rh_p in (("LH_", "RH_"), ("LH-", "RH-")):
            lh_candidate = f"{lh_p}{region_spec}"
            rh_candidate = f"{rh_p}{region_spec}"
            if lh_candidate in label_set and rh_candidate in label_set:
                return (lh_candidate, rh_candidate)
    else:
        lh_candidate = f"{lh_prefix}{region_spec}"
        rh_candidate = f"{rh_prefix}{region_spec}"
        if lh_candidate in label_set and rh_candidate in label_set:
            return (lh_candidate, rh_candidate)
    return None


def discover_wm_tracts(pyafq_gam_dir: Optional[Path]) -> List[str]:
    """List tract directory names under pyAFQ GAM HCP1065 dir."""
    if not pyafq_gam_dir or not Path(pyafq_gam_dir).exists():
        return []
    return [d.name for d in Path(pyafq_gam_dir).iterdir() if d.is_dir()]


def load_tract_metadata(tract_metadata_path: Optional[Path]) -> pd.DataFrame:
    """Load HCP1065 tract metadata; filter to left/right profilable tracts. Empty DataFrame if missing."""
    if not tract_metadata_path or not Path(tract_metadata_path).exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(tract_metadata_path)
        df = df[df["hemi"].isin(["left", "right"])]
        df = df[df["profilable"].astype(str).str.upper() == "TRUE"]
        return df
    except Exception:
        return pd.DataFrame()


def get_bilateral_tract_pairs(meta: pd.DataFrame, gam_tracts: List[str]) -> Dict[str, str]:
    """Left tract -> right tract for bilateral pairs present in GAM dir."""
    if meta.empty or "label" not in meta.columns or "hemi" not in meta.columns:
        return {}
    left_tracts = meta[meta["hemi"] == "left"]["label"].tolist()
    right_set = set(meta[meta["hemi"] == "right"]["label"].tolist())
    gam_set = set(gam_tracts)
    pairs: Dict[str, str] = {}
    for lt in left_tracts:
        rt = lt.replace("_L", "_R")
        if rt in right_set and lt in gam_set and rt in gam_set:
            pairs[lt] = rt
    return pairs


def segment_to_anatomical(tract: str, segment: str, tract_meta: pd.DataFrame) -> str:
    """Convert segment label (end1, end2, core) to anatomical description using tract metadata end1/end2.
    E.g. AF_L + end1 with end1='A' -> 'end-A'; core -> 'core'."""
    if segment == "core":
        return "core"
    if tract_meta.empty or "label" not in tract_meta.columns:
        return segment
    row = tract_meta[tract_meta["label"] == tract]
    if row.empty:
        return segment
    r = row.iloc[0]
    if segment == "end1" and "end1" in r.index:
        val = r["end1"]
        if pd.notna(val) and str(val).strip().upper() not in ("NA", ""):
            return f"end-{str(val).strip()}"
    if segment == "end2" and "end2" in r.index:
        val = r["end2"]
        if pd.notna(val) and str(val).strip().upper() not in ("NA", ""):
            return f"end-{str(val).strip()}"
    return segment


def resolve_wm_tract_segment(
    region_spec: Optional[str],
    segment_spec: Optional[str],
    paths: Dict[str, Path],
) -> Optional[Tuple[str, str, str, List[int]]]:
    """Resolve --region (tract base e.g. C_PH) and --segment (anatomical label e.g. A, P, I, S, or 'core')
    to (tract_left, tract_right, segment_name, segment_nodes). segment_name is end1/end2/core for node lookup."""
    if not region_spec or not str(region_spec).strip():
        return None
    segment_spec = (segment_spec or "core").strip()
    meta = load_tract_metadata(paths.get("tract_metadata_path"))
    pyafq = paths.get("pyafq_gam_dir")
    gam_tracts = discover_wm_tracts(pyafq)
    bilateral = get_bilateral_tract_pairs(meta, gam_tracts)
    base = str(region_spec).strip()
    tract_left = f"{base}_L"
    tract_right = f"{base}_R"
    if tract_left not in bilateral or bilateral[tract_left] != tract_right:
        for lt, rt in bilateral.items():
            if base.upper() in lt.upper():
                tract_left, tract_right = lt, rt
                break
        else:
            return None
    # Resolve segment: "core" or anatomical label (e.g. A, P, I, S) -> end1/end2/core + nodes
    seg_lower = segment_spec.lower()
    if seg_lower == "core":
        return (tract_left, tract_right, "core", SEGMENT_NODES["core"])
    if meta.empty or "label" not in meta.columns:
        return None
    row = meta[meta["label"] == tract_left]
    if row.empty:
        return None
    r = row.iloc[0]
    end1_val = r.get("end1")
    end2_val = r.get("end2")
    if pd.notna(end1_val) and str(end1_val).strip().upper() not in ("NA", ""):
        if seg_lower == str(end1_val).strip().lower():
            return (tract_left, tract_right, "end1", SEGMENT_NODES["end1"])
    if pd.notna(end2_val) and str(end2_val).strip().upper() not in ("NA", ""):
        if seg_lower == str(end2_val).strip().lower():
            return (tract_left, tract_right, "end2", SEGMENT_NODES["end2"])
    return None


def wm_region_slug(tract_left: str, segment_name: str, tract_meta: pd.DataFrame) -> str:
    """Filesystem-safe slug for WM: {tract}_core or {tract}_end-{anatomical_label} (hyphen preserved)."""
    base = tract_left.replace("_L", "") if tract_left.endswith("_L") else tract_left
    base_safe = _sanitize_slug(base)
    anatomical = segment_to_anatomical(tract_left, segment_name, tract_meta)
    # Preserve hyphen in end-A, end-P etc.; do not sanitize the whole slug
    return f"{base_safe}_{anatomical}"


def _inclusion_metadata_path(base: Path) -> Path:
    import sys

    repo = Path(__file__).resolve()
    while repo != repo.parent and not (repo / "lib" / "paths.py").exists():
        repo = repo.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from lib.paths import inclusion_csv

    return inclusion_csv("penn_epilepsy_included_basic_metadata.csv")


def get_paths(base_dir: Path) -> Dict[str, Path]:
    """Return path dict relative to base_dir. Output: asymmetry_tle_region (atlas-specific subdirs under output_dir)."""
    base = Path(base_dir)
    return {
        "mni_micro_gam_dir": base / "derivatives" / "gam" / "mni_micro",
        "pyafq_gam_dir": base / "derivatives" / "gam" / "pyafq" / "HCP1065",
        "output_dir": base / "derivatives" / "analysis" / "asymmetry_tle_region",
        "clinical_path": base / "derivatives" / "metadata" / "clinical_penn_epilepsy_qsirecon.csv",
        "inclusion_metadata": _inclusion_metadata_path(base),
        "region_asymmetry_dir": base / "derivatives" / "analysis" / "region_asymmetry_tle",
        "tract_asymmetry_dir": base / "derivatives" / "analysis" / "tract_asymmetry",
        "factor_scores_dir": base / "derivatives" / "analysis" / "factor_z-scores" / "factor_scores",
        "atlas_tsv_4s": base / "data" / "atlases" / "4S" / "atlas-4S156Parcels_dseg.tsv",
        "atlas_tsv_glasser": base / "data" / "atlases" / "Glasser" / "atlas-Glasser_dseg.tsv",
        "tract_metadata_path": base / "data" / "atlases" / "HCP1065" / "HCP1065_tract_metadata.csv",
        "scalar_labels_to_filenames": base / "data" / "metadata" / "scalar_labels_to_filenames.json",
        "scalar_to_human": base / "data" / "metadata" / "scalar_labels_to_human.json",
        "scalar_labels_to_colors": base / "data" / "metadata" / "scalar_labels_to_colors.json",
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
    """Load scalar_to_human, scalar_to_color, etc. Missing files -> empty dict."""
    out: Dict[str, Any] = {"scalar_to_human": {}, "scalar_to_color": {}}
    p = paths.get("scalar_to_human")
    if p and Path(p).exists():
        try:
            with open(p, "r") as f:
                out["scalar_to_human"] = json.load(f)
        except Exception:
            pass
    p = paths.get("scalar_labels_to_colors")
    if p and Path(p).exists():
        try:
            with open(p, "r") as f:
                out["scalar_to_color"] = json.load(f)
        except Exception:
            pass
    return out


def region_slug(region_left: str, region_right: str) -> str:
    """Derive a short, filesystem-safe slug from the region pair.
    4S: LH_Hippocampus / RH_Hippocampus -> Hippocampus. Glasser: Left_V1 / Right_V1 -> V1.
    """
    left = region_left.strip()
    right = region_right.strip()
    if left.startswith("LH_") and right.startswith("RH_"):
        suffix_left, suffix_right = left[3:], right[3:]
        if suffix_left == suffix_right:
            return _sanitize_slug(suffix_left)
    if left.startswith("LH-") and right.startswith("RH-"):
        suffix_left, suffix_right = left[3:], right[3:]
        if suffix_left == suffix_right:
            return _sanitize_slug(suffix_left)
    if left.startswith("Left_") and right.startswith("Right_"):
        suffix_left, suffix_right = left[5:], right[6:]
        if suffix_left == suffix_right:
            return _sanitize_slug(suffix_left)
    return _sanitize_slug(f"{left}_{right}")


def _sanitize_slug(s: str) -> str:
    """Replace non-alphanumeric with underscore, collapse underscores."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    return s.strip("_") or "region"
