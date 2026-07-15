#!/usr/bin/env python3
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
Compute factor scores (weighted-sum and regression/ML) for all subjects (controls + patients)
using loadings, uniquenesses, and scalar means from factor_analysis. Writes .h5 files per atlas
and method under derivatives/analysis/factor_analysis/{atlas}/factor_scores/{method}/.

Same parcel list and imputation rules for all subjects. WM parcel labels use HCP1065 tract
metadata (end1/end2 -> end-A, end-B, etc.). Invoked by default from factor_analysis.py.
"""

from __future__ import annotations

import json
import os
from os.path import join as ospj
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Config (overridable when called from factor_analysis)
# =============================================================================

METADATA_DIR = ospj(PROJECT_ROOT, "data", "metadata")
GM_PROFILE_DIR = ospj(PROJECT_ROOT, "derivatives", "gam", "mni_micro", "4S156")
GM_GLASSER_PROFILE_DIR = ospj(PROJECT_ROOT, "derivatives", "gam", "mni_micro", "Glasser")
WM_PROFILE_DIR = ospj(PROJECT_ROOT, "derivatives", "gam", "pyafq", "HCP1065")
HCP1065_TRACT_METADATA_PATH = ospj(PROJECT_ROOT, "data", "atlases", "HCP1065", "HCP1065_tract_metadata.csv")

EXCLUDED_SCALARS = [
    "map_li", "map_am", "dti_txx", "dti_txy", "dti_txz", "dti_tyy", "dti_tyz", "dti_tzz",
    "dti_ha", "rdi_rd1", "rdi_rd2",
]
N_NODES = 100
END1_NODES = list(range(1, 35))
CORE_NODES = list(range(35, 67))
END2_NODES = list(range(67, 101))

TRACTS_TO_REMOVE = [
    "CBT_L", "CBT_R", "RST_L", "RST_R", "DRTT_L", "DRTT_R",
    "EMC_L", "EMC_R", "C_PHP_L", "C_PHP_R",
]

# Controls + patients: same parcel list and imputation for all
GROUPS_ALL = ["penn_controls", "hcpya", "hcpaging", "penn_epilepsy"]

# Group label for display/storage (GAM files use "penn_epilepsy", we may keep as-is)
def get_group_for_subject(sub: str, group_from_gam: str | None) -> str:
    """Return group string for a subject (from GAM or heuristics)."""
    if group_from_gam:
        return group_from_gam
    sub_clean = sub.replace("sub-", "") if sub.startswith("sub-") else sub
    if sub_clean.startswith("RID"):
        return "penn_controls"
    if sub_clean.startswith("HCA"):
        return "hcpaging"
    if sub_clean.isdigit() or (len(sub_clean) == 6 and sub_clean.startswith("1")):
        return "hcpya"
    return "unknown"


# =============================================================================
# Helpers
# =============================================================================

def load_scalar_labels() -> List[str]:
    path = ospj(METADATA_DIR, "scalar_labels_to_filenames.json")
    with open(path) as f:
        all_labels = list(json.load(f).keys())
    return [label for label in all_labels if label not in EXCLUDED_SCALARS]


def get_mni_micro_gm_profile_dir_for_region(region_name: str, default_4s156_dir: str) -> str:
    """
    Route GM parcel to its corresponding `mni_micro` base directory.

    - Glasser parcels live under `.../mni_micro/Glasser/<parcel>/...`
    - 4S156 parcels live under `.../mni_micro/4S156/<parcel>/...`
    """
    if os.path.isdir(ospj(GM_GLASSER_PROFILE_DIR, region_name)):
        return GM_GLASSER_PROFILE_DIR
    return default_4s156_dir


def get_regions() -> List[str]:
    if not os.path.isdir(GM_PROFILE_DIR):
        return []
    regions = [d for d in os.listdir(GM_PROFILE_DIR) if os.path.isdir(ospj(GM_PROFILE_DIR, d))]
    return sorted(regions)


def get_tracts() -> List[str]:
    # For default WM runs, return base tract labels (not segment directories).
    meta = load_hcp1065_tract_metadata()
    if meta.empty or "label" not in meta.columns:
        return []
    if "label" in meta.columns:
        tracts = meta["label"].astype(str).tolist()
    else:
        tracts = []
    return sorted([t for t in tracts if t not in TRACTS_TO_REMOVE])


def load_region_scalar_data(
    region: str,
    scalar: str,
    groups: Sequence[str],
    gm_profile_dir: str,
) -> pd.DataFrame | None:
    gm_profile_dir = get_mni_micro_gm_profile_dir_for_region(region, gm_profile_dir)
    gam_path = ospj(gm_profile_dir, region, f"{region}_{scalar}_stat-mean_gam.csv")
    if not os.path.exists(gam_path):
        # Legacy naming fallback (rare for mni_micro runs)
        gam_path = ospj(gm_profile_dir, region, f"{region}_{scalar}_gam.csv")
    if not os.path.exists(gam_path):
        return None
    try:
        gam_data = pd.read_csv(gam_path)
        group_data = gam_data[gam_data["group"].isin(groups)].copy()
        if group_data.empty:
            return None
        z_col = f"{scalar}_z"
        if z_col not in group_data.columns:
            return None
        return group_data[["sub", "group", z_col]].set_index("sub")
    except Exception:
        return None


def load_tract_scalar_data(
    tract: str,
    scalar: str,
    groups: Sequence[str],
    wm_profile_dir: str,
) -> pd.DataFrame | None:
    try:
        gam_path = ospj(wm_profile_dir, tract, f"{tract}_{scalar}_stat-mean_gam.csv")
        if not os.path.exists(gam_path):
            gam_path_legacy = ospj(wm_profile_dir, tract, f"{tract}_{scalar}_gam.csv")
            if not os.path.exists(gam_path_legacy):
                return None
            gam_path = gam_path_legacy

        gam_data = pd.read_csv(gam_path)
        group_data = gam_data[gam_data["group"].isin(groups)].copy()
        if group_data.empty:
            return None

        z_cols = [f"node{i}_z" for i in range(1, N_NODES + 1)]
        missing = [c for c in z_cols if c not in group_data.columns]
        if missing:
            return None

        # node columns are returned in the same order as z_cols.
        return group_data[["sub"] + z_cols].set_index("sub")
    except Exception:
        return None


def get_segment_mean_z(z_scores: np.ndarray, segment_nodes: List[int]) -> float:
    segment_indices = [n - 1 for n in segment_nodes]
    return float(np.nanmean(z_scores[segment_indices]))


def load_hcp1065_tract_metadata() -> pd.DataFrame:
    """Load HCP1065 tract metadata for end1/end2 parcel labels."""
    if not os.path.exists(HCP1065_TRACT_METADATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(HCP1065_TRACT_METADATA_PATH)


def parcel_labels_wm(tracts: Sequence[str], metadata_df: pd.DataFrame) -> List[str]:
    """Return parcel labels for WM tracts using metadata end1/end2 (e.g. AF_L_end-A, AF_L_core, AF_L_end-P)."""
    labels = []
    meta = metadata_df.set_index("label") if "label" in metadata_df.columns else pd.DataFrame()
    for tract in sorted(tracts):
        end1 = "end1"
        end2 = "end2"
        if not meta.empty and tract in meta.index:
            e1 = meta.loc[tract, "end1"] if "end1" in meta.columns else "end1"
            e2 = meta.loc[tract, "end2"] if "end2" in meta.columns else "end2"
            if pd.notna(e1) and str(e1).upper() != "NA":
                end1 = str(e1)
            if pd.notna(e2) and str(e2).upper() != "NA":
                end2 = str(e2)
        labels.append(f"{tract}_end-{end1}")
        labels.append(f"{tract}_core")
        labels.append(f"{tract}_end-{end2}")
    return labels


def build_parcel_list(
    regions: Sequence[str],
    tracts: Sequence[str],
    metadata_df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Return (parcel_names, parcel_kind) where parcel_kind is "gm" or "wm".
    GM parcels = region names; WM parcels = tract_end-A, tract_core, tract_end-B from metadata.
    """
    parcel_names: List[str] = []
    parcel_kind: List[str] = []
    for r in sorted(regions):
        parcel_names.append(r)
        parcel_kind.append("gm")
    wm_labels = parcel_labels_wm(tracts, metadata_df)
    parcel_names.extend(wm_labels)
    parcel_kind.extend(["wm"] * len(wm_labels))
    return parcel_names, parcel_kind


def build_observation_matrix(
    subjects: List[str],
    parcel_names: List[str],
    parcel_kind: List[str],
    regions: Sequence[str],
    tracts: Sequence[str],
    scalar_order: List[str],
    groups: Sequence[str],
    gm_profile_dir: str,
    wm_profile_dir: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build (n_subjects * n_parcels, n_scalars) matrix and subject order (repeated per parcel).
    Columns of Y follow scalar_order (same as loadings). Parcel order: sorted(regions) then for each tract: end1, core, end2.
    """
    n_parcels = len(parcel_names)
    n_subjects = len(subjects)
    n_scalars = len(scalar_order)
    subject_order: List[str] = []

    # Load all region data: region -> scalar -> DataFrame (sub, group, scalar_z)
    all_region_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for region in tqdm(sorted(regions), desc="Loading regions"):
        all_region_data[region] = {}
        for scalar in scalar_order:
            df = load_region_scalar_data(region, scalar, groups, gm_profile_dir)
            if df is not None:
                all_region_data[region][scalar] = df

    all_tract_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for tract in tqdm(sorted(tracts), desc="Loading tracts"):
        all_tract_data[tract] = {}
        for scalar in scalar_order:
            df = load_tract_scalar_data(tract, scalar, groups, wm_profile_dir)
            if df is not None:
                all_tract_data[tract][scalar] = df

    # Parcel index -> (region or tract, segment)
    n_regions = len(regions)
    parcel_spec: List[Tuple[str, str | None]] = []
    for r in sorted(regions):
        parcel_spec.append((r, None))
    for tract in sorted(tracts):
        parcel_spec.append((tract, "end1"))
        parcel_spec.append((tract, "core"))
        parcel_spec.append((tract, "end2"))

    Y = np.full((n_subjects * n_parcels, n_scalars), np.nan, dtype=np.float64)
    for s_idx, subj in enumerate(subjects):
        for p_idx in range(n_parcels):
            row = s_idx * n_parcels + p_idx
            subject_order.append(subj)
            is_gm = p_idx < n_regions
            if is_gm:
                region = parcel_spec[p_idx][0]
                for sc_idx, scalar in enumerate(scalar_order):
                    if region in all_region_data and scalar in all_region_data[region]:
                        df = all_region_data[region][scalar]
                        if subj in df.index:
                            z_val = df.loc[subj, f"{scalar}_z"]
                            if np.isfinite(z_val):
                                Y[row, sc_idx] = float(z_val)
            else:
                tract, seg = parcel_spec[p_idx][0], parcel_spec[p_idx][1]  # WM segment
                if tract not in all_tract_data:
                    continue
                for sc_idx, scalar in enumerate(scalar_order):
                    if scalar not in all_tract_data[tract]:
                        continue
                    df = all_tract_data[tract][scalar]
                    if subj not in df.index:
                        continue
                    node_cols = [f"node{i}_z" for i in range(1, N_NODES + 1)]
                    z_all = df.loc[subj, node_cols].values.astype(float)
                    if seg == "end1":
                        z_val = get_segment_mean_z(z_all, END1_NODES)
                    elif seg == "core":
                        z_val = get_segment_mean_z(z_all, CORE_NODES)
                    else:
                        z_val = get_segment_mean_z(z_all, END2_NODES)
                    if np.isfinite(z_val):
                        Y[row, sc_idx] = float(z_val)

    return Y, subject_order


def get_subjects_with_complete_data(
    regions: Sequence[str],
    tracts: Sequence[str],
    scalar_labels: List[str],
    groups: Sequence[str],
    gm_profile_dir: str,
    wm_profile_dir: str,
) -> Tuple[List[str], List[str]]:
    """Return (subject_list, group_per_subject) for subjects with data in all regions and all tracts (or only the relevant parcel set)."""
    example_scalar = scalar_labels[0] if scalar_labels else None
    if not example_scalar:
        return [], []

    subs_all_regions = set()
    if regions:
        regions_with_subs: Dict[str, set] = {}
        for region in sorted(regions):
            df = load_region_scalar_data(region, example_scalar, groups, gm_profile_dir)
            if df is not None and not df.empty:
                regions_with_subs[region] = set(df.index)
            else:
                regions_with_subs[region] = set()
        subs_all_regions = set.intersection(*regions_with_subs.values()) if regions_with_subs else set()

    subs_all_tracts = set()
    if tracts:
        tract_with_subs: Dict[str, set] = {}
        for tract in sorted(tracts):
            df = load_tract_scalar_data(tract, example_scalar, groups, wm_profile_dir)
            if df is not None and not df.empty:
                tract_with_subs[tract] = set(df.index)
            else:
                tract_with_subs[tract] = set()
        subs_all_tracts = set.intersection(*tract_with_subs.values()) if tract_with_subs else set()

    if regions and tracts:
        subjects_complete = sorted(subs_all_regions & subs_all_tracts)
    elif regions:
        subjects_complete = sorted(subs_all_regions)
    else:
        subjects_complete = sorted(subs_all_tracts)
    if not subjects_complete:
        return [], []

    # Get group per subject from first available GAM (region or tract)
    group_per_sub: Dict[str, str] = {}
    for region in sorted(regions):
        df = load_region_scalar_data(region, example_scalar, groups, gm_profile_dir)
        if df is not None and "group" in df.columns:
            for sub in subjects_complete:
                if sub in df.index and sub not in group_per_sub:
                    group_per_sub[sub] = str(df.loc[sub, "group"])
    for tract in sorted(tracts):
        df = load_tract_scalar_data(tract, example_scalar, groups, wm_profile_dir)
        if df is not None and "group" in df.columns:
            for sub in subjects_complete:
                if sub in df.index and sub not in group_per_sub:
                    group_per_sub[sub] = str(df.loc[sub, "group"])
    group_list = [get_group_for_subject(s, group_per_sub.get(s)) for s in subjects_complete]
    return subjects_complete, group_list


# =============================================================================
# Factor score methods
# =============================================================================

def compute_weighted_sum_scores(
    Y: np.ndarray,
    loadings_df: pd.DataFrame,
    scalar_order: List[str],
) -> np.ndarray:
    """Y: (n_obs, n_scalars). loadings_df: factors x scalars. Returns (n_obs, n_factors)."""
    L = loadings_df[scalar_order].values.T  # (n_scalars, n_factors)
    return Y @ L  # (n_obs, n_factors)


def compute_regression_scores(
    Y: np.ndarray,
    loadings_df: pd.DataFrame,
    uniquenesses: np.ndarray,
    scalar_means: np.ndarray,
    scalar_order: List[str],
    uniqueness_floor: float = 1e-6,
) -> np.ndarray:
    """Regression (ML) factor scores: f = W @ (y - y_bar), W = L' (LL' + Psi)^{-1}."""
    L = loadings_df[scalar_order].values.T  # (n_scalars, n_factors)
    u = np.asarray(uniquenesses, dtype=np.float64)
    u = np.where(np.isfinite(u) & (u > 0), u, uniqueness_floor)
    Psi = np.diag(u)
    y_bar = np.asarray(scalar_means, dtype=np.float64).ravel()
    if y_bar.size != L.shape[0]:
        y_bar = np.broadcast_to(np.nanmean(Y, axis=0), L.shape[0])
    LLp = L @ L.T
    M = LLp + Psi
    W = np.linalg.solve(M.T, L).T  # (n_factors, n_scalars)
    Y_centered = Y - y_bar
    return (W @ Y_centered.T).T  # (n_obs, n_factors)


# =============================================================================
# H5 write
# =============================================================================

def write_factor_scores_h5(
    out_path: str,
    atlas_name: str,
    factor_names: List[str],
    scores_per_factor: Dict[str, np.ndarray],
    subject_ids: List[str],
    group_per_subject: List[str],
    parcel_columns: List[str],
) -> None:
    """Write one .h5 file (one method) with /{atlas}/FactorK/scores, sub, group, parcel_columns."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_subjects = len(subject_ids)
    with h5py.File(out_path, "w") as f:
        for fac in factor_names:
            grp = f.create_group(f"/{atlas_name}/{fac}")
            grp.create_dataset("scores", data=scores_per_factor[fac], dtype=np.float64)
            grp.create_dataset("sub", data=np.array(subject_ids, dtype="S200"))
            grp.create_dataset("group", data=np.array(group_per_subject, dtype="S100"))
            grp.create_dataset("parcel_columns", data=np.array(parcel_columns, dtype="S200"))


# =============================================================================
# Main entry
# =============================================================================

def run_factor_scores_h5(
    base_dir: str | None = None,
    output_base_dir: str | None = None,
    atlas_runs: List[Dict] | None = None,
) -> None:
    """
    Load FA outputs per atlas, build observation matrix (controls + patients),
    impute with scalar means, compute weighted-sum and regression scores, write .h5 files.

    atlas_runs: list of dicts with keys "name", "regions", "tracts" (e.g. from factor_analysis).
    """
    global PROJECT_ROOT, METADATA_DIR, GM_PROFILE_DIR, GM_GLASSER_PROFILE_DIR, WM_PROFILE_DIR, HCP1065_TRACT_METADATA_PATH
    if base_dir:
        PROJECT_ROOT = base_dir
        METADATA_DIR = ospj(PROJECT_ROOT, "data", "metadata")
        GM_PROFILE_DIR = ospj(PROJECT_ROOT, "derivatives", "gam", "mni_micro", "4S156")
        GM_GLASSER_PROFILE_DIR = ospj(PROJECT_ROOT, "derivatives", "gam", "mni_micro", "Glasser")
        WM_PROFILE_DIR = ospj(PROJECT_ROOT, "derivatives", "gam", "pyafq", "HCP1065")
        HCP1065_TRACT_METADATA_PATH = ospj(PROJECT_ROOT, "data", "atlases", "HCP1065", "HCP1065_tract_metadata.csv")
    out_dir = output_base_dir or ospj(PROJECT_ROOT, "derivatives", "analysis", "factor_analysis")

    # Align subject list with factor_analysis.py (global master subjects).
    master_subjects_path = ospj(out_dir, "subjects_included.csv")
    master_subjects: List[str] = []
    if os.path.exists(master_subjects_path):
        try:
            master_df = pd.read_csv(master_subjects_path)
            if "subject" in master_df.columns:
                master_subjects = master_df["subject"].astype(str).tolist()
        except Exception:
            master_subjects = []

    runs = atlas_runs or [
        {"name": "4S156", "regions": get_regions(), "tracts": []},
        {"name": "HCP1065", "regions": [], "tracts": get_tracts()},
        {"name": "HCP1065-4S156", "regions": get_regions(), "tracts": get_tracts()},
    ]

    scalar_labels = load_scalar_labels()
    metadata_df = load_hcp1065_tract_metadata()

    for run in runs:
        atlas_name = run["name"]
        regions = run["regions"]
        tracts = run["tracts"]
        run_output_dir = ospj(out_dir, atlas_name)
        file_prefix = f"controls_{atlas_name}"

        loadings_path = ospj(run_output_dir, f"{file_prefix}_scalar_factor_loadings.csv")
        uniquenesses_path = ospj(run_output_dir, f"{file_prefix}_scalar_uniquenesses.csv")
        means_path = ospj(run_output_dir, f"{file_prefix}_scalar_means.csv")

        if not os.path.exists(loadings_path):
            print(f"Skipping {atlas_name}: loadings not found at {loadings_path}")
            continue
        if not os.path.exists(uniquenesses_path):
            print(f"Skipping {atlas_name}: uniquenesses not found at {uniquenesses_path}")
            continue
        if not os.path.exists(means_path):
            print(f"Skipping {atlas_name}: scalar means not found at {means_path}")
            continue

        loadings_df = pd.read_csv(loadings_path, index_col=0)
        uniquenesses_df = pd.read_csv(uniquenesses_path, index_col=0)
        scalar_means_df = pd.read_csv(means_path, index_col=0)

        scalar_order = [c for c in loadings_df.columns if c in scalar_labels]
        if len(scalar_order) != loadings_df.shape[1]:
            scalar_order = loadings_df.columns.tolist()
        uniquenesses = uniquenesses_df.reindex(scalar_order)[uniquenesses_df.columns[0]].values
        scalar_means = scalar_means_df.reindex(scalar_order)[scalar_means_df.columns[0]].values
        u_flat = np.asarray(uniquenesses, dtype=np.float64).ravel()
        uniquenesses = np.where(np.isfinite(u_flat) & (u_flat > 0), u_flat, 1e-6)
        sm_flat = np.asarray(scalar_means, dtype=np.float64).ravel()
        scalar_means = np.where(np.isfinite(sm_flat), sm_flat, 0.0)

        parcel_names, _ = build_parcel_list(regions, tracts, metadata_df)
        subjects_complete, group_list_complete = get_subjects_with_complete_data(
            regions, tracts, scalar_order, GROUPS_ALL, GM_PROFILE_DIR, WM_PROFILE_DIR
        )
        if not subjects_complete:
            print(f"Skipping {atlas_name}: no subjects with complete data")
            continue

        if master_subjects:
            master_subjects_set = set(master_subjects)
            subjects = [s for s in master_subjects if s in set(subjects_complete)]
            group_map = {subjects_complete[i]: group_list_complete[i] for i in range(len(subjects_complete))}
            group_list = [group_map[s] for s in subjects if s in group_map]
        else:
            subjects = subjects_complete
            group_list = group_list_complete

        Y, subject_order_flat = build_observation_matrix(
            subjects,
            parcel_names,
            [""] * len(parcel_names),
            regions,
            tracts,
            scalar_order,
            GROUPS_ALL,
            GM_PROFILE_DIR,
            WM_PROFILE_DIR,
        )
        n_parcels = len(parcel_names)
        n_subjects = len(subjects)
        # subject_order_flat is repeated per parcel; take first occurrence per subject
        group_per_subject = group_list

        # Impute missing with scalar means (same as factor_analysis)
        for j in range(Y.shape[1]):
            mask = np.isnan(Y[:, j])
            if np.any(mask) and j < len(scalar_means):
                Y[mask, j] = scalar_means[j]

        factor_names = loadings_df.index.tolist()

        for method in ["weighted_sum", "regression"]:
            if method == "weighted_sum":
                F = compute_weighted_sum_scores(Y, loadings_df, scalar_order)
            else:
                F = compute_regression_scores(
                    Y, loadings_df, uniquenesses, scalar_means, scalar_order
                )
            # F: (n_subjects * n_parcels, n_factors) -> per factor (n_subjects, n_parcels)
            scores_per_factor = {}
            for k, fac in enumerate(factor_names):
                scores_flat = F[:, k]
                scores_per_factor[fac] = scores_flat.reshape(n_subjects, n_parcels)

            method_dir = ospj(run_output_dir, "factor_scores", method)
            out_path = ospj(method_dir, f"{atlas_name}_factor_scores_method-{method}.h5")
            write_factor_scores_h5(
                out_path,
                atlas_name,
                factor_names,
                scores_per_factor,
                subjects,
                group_per_subject,
                parcel_names,
            )
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    run_factor_scores_h5()
