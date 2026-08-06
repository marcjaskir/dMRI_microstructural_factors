"""Paths and constants for tract asymmetry pipeline. Output: asym_scalars.csv only."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

# Node ranges (from factor_analysis)
END1_NODES = list(range(1, 35))   # nodes 1-34
CORE_NODES = list(range(35, 67))  # nodes 35-66
END2_NODES = list(range(67, 101))  # nodes 67-100


def get_paths(base_dir: Path) -> Dict[str, Path]:
    """Return path dict relative to base_dir. Output: tract_asymmetry."""
    import sys

    repo = Path(__file__).resolve()
    while repo != repo.parent and not (repo / "lib" / "paths.py").exists():
        repo = repo.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from lib.paths import inclusion_csv

    base = Path(base_dir)
    return {
        "gam_dir": base / "derivatives" / "gam" / "pyafq" / "HCP1065",
        "metadata_path": base / "data" / "atlases" / "HCP1065" / "HCP1065_tract_metadata.csv",
        "inclusion_path": inclusion_csv("penn_epilepsy_included_basic_metadata.csv"),
        "output_dir": base / "derivatives" / "analysis" / "tract_asymmetry",
        "normative_dir": base / "derivatives" / "analysis" / "tract_asymmetry_normative",
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
