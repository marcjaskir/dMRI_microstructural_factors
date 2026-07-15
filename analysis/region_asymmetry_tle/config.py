"""Paths and constants for region asymmetry (TLE). GAM mni_micro: Glasser, 4S156 (cortex + subcortex), HCP1065."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

EXCLUDED_SCALARS = [
    "map_li", "map_am", "dti_txx", "dti_txy", "dti_txz",
    "dti_tyy", "dti_tyz", "dti_tzz", "dti_ha", "rdi_rd1", "rdi_rd2",
]


def get_paths(base_dir: Path) -> Dict[str, Path]:
    """Return path dict. GAM mni_micro for Glasser, 4S156, HCP1065; 4S TSV for cortex/subcortex split."""
    import sys

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from lib.paths import inclusion_csv

    base = Path(base_dir)
    return {
        "gam_glasser": base / "derivatives" / "gam" / "mni_micro" / "Glasser",
        "gam_4s156": base / "derivatives" / "gam" / "mni_micro" / "4S156",
        "gam_hcp1065": base / "derivatives" / "gam" / "mni_micro" / "HCP1065",
        "atlas_tsv_4s": base / "data" / "atlases" / "4S" / "atlas-4S156Parcels_dseg.tsv",
        "inclusion_path": inclusion_csv("penn_epilepsy_included_basic_metadata.csv"),
        "output_dir": base / "derivatives" / "analysis" / "region_asymmetry_tle",
        "normative_dir": base / "derivatives" / "analysis" / "region_asymmetry_tle_normative",
    }


def load_tle_inclusion(inclusion_path: Path) -> tuple:
    """
    Load TLE inclusion CSV. Returns (subjects: List[str], subject_ipsi_hemi: Dict[str, str]).
    Lateralization from 'laterality': left -> L, right -> R.
    If 'lobe' column exists, only include lobe == 'temporal'.
    """
    subjects: List[str] = []
    subject_ipsi_hemi: Dict[str, str] = {}
    if not Path(inclusion_path).exists():
        return subjects, subject_ipsi_hemi
    try:
        df = pd.read_csv(inclusion_path)
        if "sub" not in df.columns or "laterality" not in df.columns:
            return subjects, subject_ipsi_hemi
        if "lobe" in df.columns:
            df = df[df["lobe"].astype(str).str.strip().str.lower() == "temporal"]
        for _, row in df.iterrows():
            sub = row.get("sub")
            if pd.isna(sub):
                continue
            sub = str(sub)
            lat = str(row.get("laterality", "")).strip().lower()
            if lat == "left":
                subjects.append(sub)
                subject_ipsi_hemi[sub] = "L"
            elif lat == "right":
                subjects.append(sub)
                subject_ipsi_hemi[sub] = "R"
        subjects = list(dict.fromkeys(subjects))
    except Exception:
        pass
    return subjects, subject_ipsi_hemi


def get_4s_subcortical_regions(atlas_tsv_4s: Path) -> List[str]:
    """Return 4S region labels where network_label == 'n/a' (subcortex) from atlas TSV."""
    if not Path(atlas_tsv_4s).exists():
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


def get_4s_cortical_regions(atlas_tsv_4s: Path) -> List[str]:
    """Return 4S region labels where network_label != 'n/a' (cortex) from atlas TSV."""
    if not Path(atlas_tsv_4s).exists():
        return []
    try:
        df = pd.read_csv(atlas_tsv_4s, sep="\t")
        if "label" not in df.columns or "network_label" not in df.columns:
            return []
        net = df["network_label"].fillna("n/a").astype(str).str.strip().str.lower()
        cortex = df.loc[~net.isin(("n/a", "nan", "")), "label"].tolist()
        return cortex
    except Exception:
        return []
