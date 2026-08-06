"""Per-scalar GAM residual z wide CSVs under ``scalar_z-scores/``."""
from __future__ import annotations

import os
from os.path import join as ospj
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from .config import (
    CONTROL_GROUPS,
    CORE_NODES,
    END1_NODES,
    END2_NODES,
    PATIENT_GROUPS,
    SCALAR_Z_SCORES_OUTPUT_DIR,
    WM_PROFILE_DIR_PYAFQ,
)
from .io import (
    get_group_from_subject_id,
    get_mni_micro_gm_profile_dir_for_region,
    load_scalar_labels,
    order_scalars_by_prefix,
    subject_id_column,
    wm_tract_segment_roi_key,
)


def _load_gm_scalar_z_series(
    region: str,
    scalar: str,
    groups: Sequence[str],
    subjects: Sequence[str],
) -> Optional[pd.Series]:
    gm_base = get_mni_micro_gm_profile_dir_for_region(region)
    for fname in (f"{region}_{scalar}_stat-mean_gam.csv", f"{region}_{scalar}_gam.csv"):
        path = ospj(gm_base, region, fname)
        if not os.path.exists(path):
            continue
        try:
            gam = pd.read_csv(path)
            g = gam[gam["group"].isin(groups)]
            if g.empty:
                return None
            zc = f"{scalar}_z"
            if zc not in g.columns:
                return None
            id_col = subject_id_column(g)
            s = g.set_index(id_col)[zc]
            return s.reindex(list(subjects))
        except Exception:
            return None
    return None


def _load_wm_segment_z_series(
    tract: str,
    scalar: str,
    groups: Sequence[str],
    segment: str,
    subjects: Sequence[str],
) -> Optional[pd.Series]:
    for fname in (f"{tract}_{scalar}_stat-mean_gam.csv", f"{tract}_{scalar}_gam.csv"):
        path = ospj(WM_PROFILE_DIR_PYAFQ, tract, fname)
        if not os.path.exists(path):
            continue
        try:
            gam = pd.read_csv(path)
            g = gam[gam["group"].isin(groups)]
            if g.empty:
                return None
            seg_nodes = {
                "end1": END1_NODES,
                "core": CORE_NODES,
                "end2": END2_NODES,
            }[segment]
            seg_cols = [f"node{i}_z" for i in seg_nodes]
            if not seg_cols or any(c not in g.columns for c in seg_cols):
                return None
            id_col = subject_id_column(g)
            idxd = g.set_index(id_col)
            means = idxd[seg_cols].mean(axis=1)
            return means.reindex(list(subjects))
        except Exception:
            return None
    return None


def save_scalar_z_scores(
    all_regions: Sequence[str],
    all_tracts: Sequence[str],
    patient_subjects: Sequence[str],
    control_subjects: Sequence[str],
    patient_groups: Sequence[str],
    control_groups: Sequence[str],
    tract_to_end1: Dict[str, str],
    tract_to_end2: Dict[str, str],
    *,
    output_dir: str = SCALAR_Z_SCORES_OUTPUT_DIR,
    control_group_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write ``epilepsy_{scalar}_z_scores.csv`` and ``controls_{scalar}_z_scores.csv``.
    """
    os.makedirs(output_dir, exist_ok=True)
    scalar_labels = order_scalars_by_prefix(load_scalar_labels())
    tracts_f = list(all_tracts)
    group_map = control_group_map or {}

    def _one_table(
        scalar: str,
        groups: Sequence[str],
        subjects: Sequence[str],
        include_group: bool,
    ) -> Optional[pd.DataFrame]:
        if not groups or not subjects:
            return None
        subs = list(subjects)
        parts: List[Tuple[str, pd.Series]] = []
        for region in all_regions:
            s = _load_gm_scalar_z_series(region, scalar, groups, subs)
            if s is not None:
                parts.append((region, s))
        for tract in tracts_f:
            for seg in ("end1", "core", "end2"):
                roi_key = wm_tract_segment_roi_key(tract, seg, tract_to_end1, tract_to_end2)
                s = _load_wm_segment_z_series(tract, scalar, groups, seg, subs)
                if s is not None:
                    parts.append((roi_key, s))
        if not parts:
            return None
        out = pd.concat([s.reindex(subs) for _, s in parts], axis=1, copy=False)
        out.columns = [n for n, _ in parts]
        out.index.name = "subject"
        if include_group:
            group_series = pd.Series(
                [
                    group_map.get(str(s)) or get_group_from_subject_id(str(s))
                    for s in out.index
                ],
                index=out.index,
                name="group",
            )
            out = pd.concat([group_series, out], axis=1)
        return out

    for scalar in tqdm(scalar_labels, desc="scalar z-score CSVs"):
        ep_df = _one_table(scalar, list(patient_groups), list(patient_subjects), False)
        if ep_df is not None and not ep_df.empty:
            p = ospj(output_dir, f"epilepsy_{scalar}_z_scores.csv")
            ep_df.to_csv(p)
            print(f"  Saved scalar z-scores (epilepsy) to {p}")
        ct_df = _one_table(scalar, list(control_groups), list(control_subjects), True)
        if ct_df is not None and not ct_df.empty:
            p = ospj(output_dir, f"controls_{scalar}_z_scores.csv")
            ct_df.to_csv(p)
            print(f"  Saved scalar z-scores (controls) to {p}")


def run_scalar_z(
    all_regions: Sequence[str],
    all_tracts: Sequence[str],
    patient_subjects: Sequence[str],
    control_subjects: Sequence[str],
    tract_to_end1: Dict[str, str],
    tract_to_end2: Dict[str, str],
    **kwargs,
) -> None:
    save_scalar_z_scores(
        all_regions,
        all_tracts,
        patient_subjects,
        control_subjects,
        PATIENT_GROUPS,
        CONTROL_GROUPS,
        tract_to_end1,
        tract_to_end2,
        **kwargs,
    )
