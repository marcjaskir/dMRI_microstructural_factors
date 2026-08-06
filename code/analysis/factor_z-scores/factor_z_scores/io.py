"""Discover ROIs and load GAM residual z, loadings, and cohort metadata."""
from __future__ import annotations

import json
import os
from os.path import join as ospj
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from .config import (
    CONTROL_GROUPS,
    END1_NODES,
    END2_NODES,
    CORE_NODES,
    FACTOR_LOADINGS_PATH,
    FOUR_S156_DSEG_PATH,
    GM_4S156_PROFILE_DIR,
    GM_GLASSER_PROFILE_DIR,
    HCP1065_TRACT_METADATA_PATH,
    INCLUSION_METADATA_PATH,
    METADATA_DIR,
    N_NODES,
    SCALAR_PREFIX_ORDER,
    WM_PROFILE_DIR_PYAFQ,
)


def subject_id_column(df: pd.DataFrame) -> str:
    """Prefer ``sub`` (controlled); fall back to ``anon_id`` (open exports)."""
    if "sub" in df.columns:
        return "sub"
    if "anon_id" in df.columns:
        return "anon_id"
    raise KeyError("GAM/metadata table needs a 'sub' or 'anon_id' column")


def inclusion_id_column(df: pd.DataFrame) -> str:
    return subject_id_column(df)


def get_group_from_subject_id(sub: str) -> Optional[str]:
    """Map control subject ID pattern → cohort (penn_controls / hcpya / hcpaging)."""
    sub_clean = sub.replace("sub-", "") if sub.startswith("sub-") else sub
    if sub_clean.startswith("RID"):
        return "penn_controls"
    if sub_clean.startswith("HCA"):
        return "hcpaging"
    if sub_clean.isdigit() or (sub_clean.startswith("1") and len(sub_clean) == 6):
        return "hcpya"
    return None


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


def load_scalar_labels() -> List[str]:
    path = ospj(METADATA_DIR, "scalar_labels_to_filenames.json")
    with open(path) as f:
        return list(json.load(f).keys())


def order_scalars_by_prefix(scalars: Sequence[str]) -> List[str]:
    prefix_rank = {p: i for i, p in enumerate(SCALAR_PREFIX_ORDER)}

    def _get_prefix(name: str) -> str:
        return name.split("_", 1)[0] if "_" in name else name

    return sorted(
        list(scalars),
        key=lambda name: (prefix_rank.get(_get_prefix(name), len(prefix_rank)), name),
    )


def load_temporal_patient_subjects_ordered() -> List[str]:
    """Temporal lobe subjects from inclusion metadata, stable CSV order."""
    if not os.path.exists(INCLUSION_METADATA_PATH):
        return []
    try:
        df = pd.read_csv(INCLUSION_METADATA_PATH)
        id_col = inclusion_id_column(df)
        mask = df["lobe"].astype(str).str.strip().str.lower() == "temporal"
        subs = df.loc[mask, id_col].astype(str).tolist()
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


def load_tract_metadata_full() -> pd.DataFrame:
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
    return _list_subdirs(GM_GLASSER_PROFILE_DIR)


def get_subcortex_4s156_regions() -> List[str]:
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
    if os.path.isdir(ospj(GM_GLASSER_PROFILE_DIR, region)):
        return GM_GLASSER_PROFILE_DIR
    return GM_4S156_PROFILE_DIR


def get_tracts_by_type(tract_type: str) -> List[str]:
    meta = load_tract_metadata_full()
    if meta.empty or "label" not in meta.columns or "type" not in meta.columns:
        return []
    tracts = meta.loc[meta["type"].astype(str) == tract_type, "label"].astype(str).tolist()
    available_bases: Set[str] = set(_list_subdirs(WM_PROFILE_DIR_PYAFQ))
    return sorted([t for t in tracts if t in available_bases])


def discover_all_gm_regions() -> List[str]:
    all_gm = list(dict.fromkeys(get_glasser_regions() + get_subcortex_4s156_regions()))
    return sorted(all_gm)


def discover_all_wm_tracts() -> List[str]:
    assoc = get_tracts_by_type("association")
    proj = get_tracts_by_type("projection")
    return sorted(list(dict.fromkeys(assoc + proj)))


def load_factor_loadings(scalar_labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
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


def load_gm_region_scalar_data(
    region: str,
    scalar: str,
    groups: Sequence[str],
) -> pd.DataFrame | None:
    """
    Load z-score data for a GM region and scalar.

    Returns DataFrame indexed by subject id with column ``{scalar}_z``,
    plus a ``_group`` column when the GAM table provides ``group``.
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
        id_col = subject_id_column(group_data)
        cols = [id_col, z_col]
        out = group_data[cols].set_index(id_col)
        if "group" in group_data.columns:
            out["_group"] = group_data.set_index(id_col)["group"]
        return out
    except Exception as e:  # noqa: BLE001
        print(f"Error loading {region}_{scalar}: {e}")
        return None


def load_wm_tract_scalar_data(
    tract: str,
    scalar: str,
    groups: Sequence[str],
) -> pd.DataFrame | None:
    """Load node-level WM GAM z; index = subject; columns node1_z..node100_z (+ optional _group)."""
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
        id_col = subject_id_column(group_data)
        out = group_data[[id_col] + z_cols].set_index(id_col)
        if "group" in group_data.columns:
            out["_group"] = group_data.set_index(id_col)["group"]
        return out
    except Exception as e:  # noqa: BLE001
        print(f"Error loading {tract}_{scalar}: {e}")
        return None


def get_segment_mean_z(z_scores: np.ndarray, segment_nodes: List[int]) -> float:
    segment_indices = [node - 1 for node in segment_nodes]
    segment_values = z_scores[segment_indices]
    return float(np.nanmean(segment_values))


def compute_tract_segment_z_scores(tract_node_data: pd.DataFrame) -> pd.DataFrame:
    z_cols = [f"node{i}_z" for i in range(1, N_NODES + 1)]
    results = {}
    for subject in tract_node_data.index:
        z_scores = tract_node_data.loc[subject, z_cols].values
        results[subject] = {
            "end1": get_segment_mean_z(z_scores, END1_NODES),
            "core": get_segment_mean_z(z_scores, CORE_NODES),
            "end2": get_segment_mean_z(z_scores, END2_NODES),
        }
    return pd.DataFrame(results).T


def wm_tract_segment_roi_key(
    tract: str,
    segment: str,
    tract_to_end1: Dict[str, str],
    tract_to_end2: Dict[str, str],
) -> str:
    """WM column label (e.g. AF_L → AF_L_end-A)."""
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


def collect_control_subjects_union_from_gam(
    all_regions: Sequence[str],
    all_tracts: Sequence[str],
    scalar_labels: Sequence[str],
    control_groups: Sequence[str] = CONTROL_GROUPS,
) -> List[str]:
    """Controls with any GAM row in at least one region×scalar or tract×scalar."""
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


def collect_control_group_map(
    all_regions: Sequence[str],
    all_tracts: Sequence[str],
    scalar_labels: Sequence[str],
    control_groups: Sequence[str] = CONTROL_GROUPS,
) -> Dict[str, str]:
    """Subject → GAM ``group`` (needed when IDs are anonymized)."""
    group_map: Dict[str, str] = {}
    if not scalar_labels:
        return group_map
    # One scalar across GM+WM is enough: each subject appears with a stable group label.
    scalar = scalar_labels[0]
    for region in all_regions:
        data = load_gm_region_scalar_data(region, scalar, control_groups)
        if data is None or data.empty or "_group" not in data.columns:
            continue
        for sid, g in data["_group"].items():
            group_map.setdefault(str(sid), str(g))
    for tract in all_tracts:
        data = load_wm_tract_scalar_data(tract, scalar, control_groups)
        if data is None or data.empty or "_group" not in data.columns:
            continue
        for sid, g in data["_group"].items():
            group_map.setdefault(str(sid), str(g))
    return group_map
