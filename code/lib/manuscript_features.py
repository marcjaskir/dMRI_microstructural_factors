"""Manuscript feature allowlists for open packaging / export.

Analysis code should not filter with exclusion lists; it consumes whatever
scalars and tracts are present. These allowlists are used only when building
``data/open/`` and the OSF HDF5 so shipped products contain the analyzed set.
"""
from __future__ import annotations

import re
from pathlib import Path

# All4_Combined factor analysis scalars (n=26), loadings column order.
MANUSCRIPT_SCALARS: tuple[str, ...] = (
    "dki_ad",
    "dki_ak",
    "dki_kfa",
    "dki_md",
    "dki_mk",
    "dki_mkt",
    "dki_rd",
    "dki_rk",
    "dki_fa",
    "gqi_gfa",
    "gqi_qa",
    "dti_ad",
    "dti_fa",
    "dti_md",
    "dti_rd",
    "noddi_icvf",
    "noddi_isovf",
    "noddi_od",
    "map_ng",
    "map_ngpar",
    "map_ngperp",
    "map_pa",
    "map_path",
    "map_rtap",
    "map_rtop",
    "map_rtpp",
)
MANUSCRIPT_SCALAR_SET: frozenset[str] = frozenset(MANUSCRIPT_SCALARS)

# HCP1065 WM tracts retained in factor / asymmetry analyses (n=48).
MANUSCRIPT_WM_TRACTS: tuple[str, ...] = (
    "CPT_F_L",
    "CPT_F_R",
    "CPT_O_L",
    "CPT_O_R",
    "CPT_P_L",
    "CPT_P_R",
    "CST_L",
    "CST_R",
    "CS_A_L",
    "CS_A_R",
    "CS_P_L",
    "CS_P_R",
    "CS_S_L",
    "CS_S_R",
    "C_FPH_L",
    "C_FPH_R",
    "C_FP_L",
    "C_FP_R",
    "C_PH_L",
    "C_PH_R",
    "C_PO_L",
    "C_PO_R",
    "F_L",
    "F_R",
    "IFOF_L",
    "IFOF_R",
    "ILF_L",
    "ILF_R",
    "ML_L",
    "ML_R",
    "MdLF_L",
    "MdLF_R",
    "OR_L",
    "OR_R",
    "PAT_L",
    "PAT_R",
    "SLF1_L",
    "SLF1_R",
    "TR_A_L",
    "TR_A_R",
    "TR_P_L",
    "TR_P_R",
    "TR_S_L",
    "TR_S_R",
    "UF_L",
    "UF_R",
    "VOF_L",
    "VOF_R",
)
MANUSCRIPT_WM_TRACT_SET: frozenset[str] = frozenset(MANUSCRIPT_WM_TRACTS)

# Base labels used in asymmetry tables (no hemisphere suffix), e.g. ILF, CST.
MANUSCRIPT_WM_TRACT_BASES: frozenset[str] = frozenset(
    t[:-2] for t in MANUSCRIPT_WM_TRACTS if t.endswith(("_L", "_R"))
)

_GAM_STAT_SUFFIXES = (
    "_stat-mean_gam",
    "_stat-standard_deviation_gam",
    "_gam",
)


def is_manuscript_scalar(name: str) -> bool:
    return name in MANUSCRIPT_SCALAR_SET


def is_manuscript_wm_tract(name: str) -> bool:
    return name in MANUSCRIPT_WM_TRACT_SET


def filter_metadata_dict(data: dict) -> dict:
    """Keep only manuscript scalar keys, in manuscript order then leftovers."""
    out = {k: data[k] for k in MANUSCRIPT_SCALARS if k in data}
    return out


def _scalar_from_gam_stem(stem: str, region: str) -> str | None:
    prefix = f"{region}_"
    if not stem.startswith(prefix):
        return None
    rest = stem[len(prefix) :]
    for suf in _GAM_STAT_SUFFIXES:
        if rest.endswith(suf):
            return rest[: -len(suf)]
    return None


def gam_relpath_is_manuscript(relpath: str) -> bool:
    """Return True if a path under ``gam/`` belongs in the open/HDF5 bundle."""
    parts = Path(relpath).parts
    name = parts[-1] if parts else ""
    stem = name[: -len(".csv")] if name.endswith(".csv") else name

    if "pyafq" in parts and "HCP1065" in parts:
        try:
            tract = parts[parts.index("HCP1065") + 1]
        except (ValueError, IndexError):
            return False
        if not is_manuscript_wm_tract(tract):
            return False
        scalar = _scalar_from_gam_stem(stem, tract)
        return scalar is None or is_manuscript_scalar(scalar)

    if "mni_micro" in parts and "HCP1065" in parts:
        try:
            region = parts[parts.index("HCP1065") + 1]
        except (ValueError, IndexError):
            return True
        # Drop whole-tract dirs that are not manuscript WM tracts when the
        # directory name itself is a hemispheric tract label.
        if re.fullmatch(r"[A-Za-z0-9]+_[LR]", region) and not is_manuscript_wm_tract(region):
            return False
        scalar = _scalar_from_gam_stem(stem, region)
        return scalar is None or is_manuscript_scalar(scalar)

    # Glasser / 4S156 / other: keep only manuscript scalar files
    if name.endswith(".csv") and len(parts) >= 2:
        region = parts[-2]
        scalar = _scalar_from_gam_stem(stem, region)
        if scalar is not None:
            return is_manuscript_scalar(scalar)
    return True


def open_relpath_is_manuscript(relpath: str) -> bool:
    """Gatekeeper for packing any path under ``data/open/``."""
    rel = relpath.replace("\\", "/")
    if rel.startswith("gam/") or "/gam/" in rel:
        # strip leading gam/
        idx = rel.find("gam/")
        return gam_relpath_is_manuscript(rel[idx + len("gam/") :])
    if rel.startswith("metadata/") and rel.endswith(".json"):
        return True  # contents filtered separately when writing
    if "HCP1065_tract_metadata.csv" in rel:
        return True  # contents filtered when writing / pruning
    # Skip analysis products for non-manuscript WM tract directories
    if "/tract_asymmetry/" in rel or "/tract_asymmetry_normative/" in rel:
        parts = Path(rel).parts
        for p in parts:
            if re.fullmatch(r"[A-Za-z0-9]+_[LR]", p) and not is_manuscript_wm_tract(p):
                return False
    if "/region_asymmetry_tle/" in rel or "/region_asymmetry_tle_normative/" in rel:
        parts = Path(rel).parts
        for p in parts:
            if p in MANUSCRIPT_WM_TRACT_SET:
                continue
            if re.fullmatch(r"[A-Za-z0-9]+_[LR]", p) and p.split("_")[0] in {
                "AF",
                "FAT",
                "SLF2",
                "SLF3",
                "CBT",
                "RST",
                "DRTT",
                "EMC",
                "C_PHP",
            }:
                # hemispheric labels not in manuscript set
                if not is_manuscript_wm_tract(p):
                    return False
    return True
