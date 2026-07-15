"""Region-group definitions for G1/G2 bar summaries.

Coarse anatomical buckets combining HCP1065 white-matter tract families,
4S subcortical nuclei groups (Basal ganglia / Thalamus / Midbrain / Limbic subcortex /
Cerebellum), and Glasser cortical lobes. Mirrors ``factor_gradients`` /
``gradients`` conventions but rebased on ``gradients_group-controls``'s
``io.py`` (``parse_wm_tract_end_column_name`` / ``region_is_white_matter_column``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import (
    parse_wm_tract_end_column_name,
    region_is_white_matter_column,
)

GLASSER_LOBE_DISPLAY: dict[str, str] = {
    "Fr": "Frontal",
    "Par": "Parietal",
    "Temp": "Temporal",
    "Occ": "Occipital",
    "Ins": "Insula",
}

# 4S short name (after ``LH-``/``RH-`` strip) -> anatomic group.
SUBCORTEX_ROI_TO_GROUP: dict[str, str] = {
    "Pu": "Basal ganglia",
    "Ca": "Basal ganglia",
    "NAC": "Basal ganglia",
    "VeP": "Basal ganglia",
    "GPe": "Basal ganglia",
    "GPi": "Basal ganglia",
    "EXA": "Basal ganglia",
    "STN": "Basal ganglia",
    "STH": "Basal ganglia",
    "SNc_PBP_VTA": "Midbrain",
    "SNr": "Midbrain",
    "RN": "Midbrain",
    "HN": "Midbrain",
    "HTH": "Limbic subcortex",
    "MN": "Limbic subcortex",
    "Pulvinar": "Thalamus",
    "Anterior": "Thalamus",
    "Medio_Dorsal": "Thalamus",
    "Ventral_Latero_Dorsal": "Thalamus",
    "Central_Lateral-Lateral_Posterior-Medial_Pulvinar": "Thalamus",
    "Ventral_Anterior": "Thalamus",
    "Ventral_Latero_Ventral": "Thalamus",
    "Hippocampus": "Limbic subcortex",
    "Amygdala": "Limbic subcortex",
    "Cerebellar_Region1": "Cerebellum",
    "Cerebellar_Region2": "Cerebellum",
    "Cerebellar_Region3": "Cerebellum",
    "Cerebellar_Region4": "Cerebellum",
    "Cerebellar_Region5": "Cerebellum",
    "Cerebellar_Region6": "Cerebellum",
    "Cerebellar_Region7": "Cerebellum",
    "Cerebellar_Region8": "Cerebellum",
    "Cerebellar_Region9": "Cerebellum",
    "Cerebellar_Region10": "Cerebellum",
}
GREY_MATTER_NUCLEI_PANEL_GROUP_ORDER: tuple[str, ...] = (
    "Basal ganglia",
    "Thalamus",
    "Midbrain",
    "Limbic subcortex",
    "Cerebellum",
)

# HCP1065 tract label -> high-level name (``_L``/``_R`` paired).
TRACT_SEGMENT_GROUPINGS: dict[str, tuple[str, ...]] = {
    "Mesial limbic": (
        "C_FP_L",
        "C_FPH_L",
        "C_PH_L",
        "C_PO_L",
        "UF_L",
        "F_L",
    ),
    "Occipital association": ("IFOF_L", "ILF_L", "VOF_L"),
    "Parietal association": ("PAT_L", "SLF1_L", "MdLF_L"),
    "Thalamic radiations": ("TR_A_L", "TR_P_L", "TR_S_L", "OR_L"),
    "Corticostriatal": ("CS_A_L", "CS_P_L", "CS_S_L"),
    "Ascending tracts": ("CPT_F_L", "CPT_O_L", "CPT_P_L", "CST_L", "ML_L"),
}


def _cortical_lobe_region_group_label(lobe_code: str) -> str | None:
    code = str(lobe_code).strip()
    d = GLASSER_LOBE_DISPLAY.get(code)
    if d is None:
        return None
    if code == "Ins":
        return "Insula"
    return f"{d} lobe"


def tract_base_to_functional_group_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for gname, tracts in TRACT_SEGMENT_GROUPINGS.items():
        for t in tracts:
            t = str(t)
            m[t] = gname
            if t.endswith("_L"):
                m[t[:-2] + "_R"] = gname
    return m


def load_cortical_lobe_region_group_by_roi(tractometry_root: Path) -> dict[str, str]:
    """Glasser ROI -> ``"<Lobe> lobe"`` mapping (or empty if metadata missing)."""
    p = tractometry_root / "data/atlases/Glasser/glasser_additional_metadata.csv"
    if not p.is_file():
        return {}
    df = pd.read_csv(p)
    if "Lobe" not in df.columns or "region" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        region = str(row["region"]).strip()
        if not region:
            continue
        name = _cortical_lobe_region_group_label(row["Lobe"])
        if name is None:
            continue
        out[region] = name
        out[f"Left_{region}"] = name
        out[f"Right_{region}"] = name
    return out


def _glasser_suffix(n: str) -> str:
    t = str(n)
    for p in ("Left_", "Right_"):
        if t.startswith(p):
            return t[len(p):]
    return t


def _strip_4s_hemi_prefixes(label: str) -> str:
    s = str(label)
    for pref in ("LH-", "RH-", "LH_", "RH_"):
        if s.startswith(pref):
            s = s[len(pref):]
    return s


def white_matter_tract_base_name(column_name: str) -> str | None:
    s = str(column_name)
    if s.endswith("_core") and "_end-" not in s:
        return s[: -len("_core")]
    parsed = parse_wm_tract_end_column_name(s)
    if parsed is not None:
        return parsed[0]
    if "_end-" in s:
        before = s.split("_end-")[0]
        if before.endswith("_core"):
            return before[: -len("_core")]
    return None


# Tract / functional family names = white matter; 4S nuclei + cerebellum + lobes = grey matter.
WM_REGION_GROUP_NAMES: frozenset[str] = frozenset(TRACT_SEGMENT_GROUPINGS.keys())
GM_REGION_GROUP_NAMES: frozenset[str] = frozenset(
    (
        *GREY_MATTER_NUCLEI_PANEL_GROUP_ORDER,
        "Frontal lobe",
        "Parietal lobe",
        "Temporal lobe",
        "Occipital lobe",
        "Insula",
    )
)
REGION_GROUP_BAR_EDGE_WIDTH = 2.0
REGION_GROUP_WM_FACE = "#ffffff"
REGION_GROUP_GM_FACE = "#9e9e9e"


def region_group_bar_facecolor(label: str) -> str:
    """White for tract families (WM), grey for subcortex / cortical lobe (GM)."""
    s = str(label)
    if s in WM_REGION_GROUP_NAMES:
        return REGION_GROUP_WM_FACE
    if s in GM_REGION_GROUP_NAMES:
        return REGION_GROUP_GM_FACE
    return REGION_GROUP_GM_FACE


REGION_GROUP_DISPLAY_ORDER: tuple[str, ...] = (
    *tuple(TRACT_SEGMENT_GROUPINGS.keys()),
    *GREY_MATTER_NUCLEI_PANEL_GROUP_ORDER,
    "Frontal lobe",
    "Parietal lobe",
    "Temporal lobe",
    "Occipital lobe",
    "Insula",
)


def region_group_for_column(
    name: str,
    *,
    tract_to_group: dict[str, str],
    cortical_by_roi: dict[str, str],
    subcortical_matched_columns: frozenset[str],
) -> str | None:
    s = str(name)
    if region_is_white_matter_column(s):
        b = white_matter_tract_base_name(s)
        if b and b in tract_to_group:
            return tract_to_group[b]
        return None
    bare = _strip_4s_hemi_prefixes(s)
    if s in subcortical_matched_columns or bare in subcortical_matched_columns:
        g = SUBCORTEX_ROI_TO_GROUP.get(bare) or SUBCORTEX_ROI_TO_GROUP.get(s)
        return g
    gs = _glasser_suffix(s)
    for k in (s, bare, gs, f"Left_{gs}", f"Right_{gs}"):
        g = cortical_by_roi.get(k)
        if g is not None:
            return g
    return None


def build_roi_to_region_group(
    g: pd.Series,
    *,
    tract_to_group: dict[str, str],
    cortical_by_roi: dict[str, str],
    subcortical_matched_columns: frozenset[str],
) -> dict[str, str]:
    """``roi_name`` -> region-group label; ROIs without a class are omitted."""
    out: dict[str, str] = {}
    for roi in g.index:
        rg = region_group_for_column(
            str(roi),
            tract_to_group=tract_to_group,
            cortical_by_roi=cortical_by_roi,
            subcortical_matched_columns=subcortical_matched_columns,
        )
        if rg is not None:
            out[str(roi)] = rg
    return out
