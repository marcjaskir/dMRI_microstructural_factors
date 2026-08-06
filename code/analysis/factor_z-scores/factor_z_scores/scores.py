"""Apply loadings → wide ``factor_scores/{cohort}_F*_scores.csv``."""
from __future__ import annotations

import os
from os.path import join as ospj
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CONTROL_GROUPS, PATIENT_GROUPS
from .io import (
    collect_control_group_map,
    collect_control_subjects_union_from_gam,
    compute_tract_segment_z_scores,
    discover_all_gm_regions,
    discover_all_wm_tracts,
    get_group_from_subject_id,
    load_gm_region_scalar_data,
    load_temporal_patient_subjects_ordered,
    load_tract_metadata_full,
    load_wm_tract_scalar_data,
    resolve_subject_key,
    wm_tract_segment_roi_key,
)


def compute_factor_scores(
    roi_data: Dict[str, pd.DataFrame],
    roi_name: str,
    scalar_labels: Sequence[str],
    subjects: Sequence[str],
    factor_loadings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Factor score = Σ (scalar_z × loading) across available scalars.

    GM: column ``{scalar}_z``. WM multi-segment tables average end1/core/end2 scores;
    the wide CSV writer typically passes one segment at a time as ``{scalar}_z``.
    """
    del roi_name  # retained for call-site compatibility / messages
    if factor_loadings.empty:
        return pd.DataFrame()

    factor_scores: Dict[str, Dict] = {}
    available_scalars = set(factor_loadings.columns) & set(scalar_labels)

    is_wm_tract = False
    if available_scalars:
        sample_scalar = list(available_scalars)[0]
        if sample_scalar in roi_data:
            sample_data = roi_data[sample_scalar]
            if not sample_data.empty:
                if f"{sample_scalar}_z_end1" in sample_data.columns:
                    is_wm_tract = True

    for subject in subjects:
        subject_scores = {}
        for factor in factor_loadings.index:
            score = 0.0
            n_scalars = 0

            if is_wm_tract:
                segment_scores = []
                for segment in ["end1", "core", "end2"]:
                    segment_score = 0.0
                    segment_n_scalars = 0
                    for scalar in available_scalars:
                        if scalar not in roi_data:
                            continue
                        data = roi_data[scalar]
                        z_col = f"{scalar}_z_{segment}"
                        row_key = resolve_subject_key(subject, data.index)
                        if row_key is not None and z_col in data.columns:
                            z_score = data.loc[row_key, z_col]
                            if not np.isnan(z_score):
                                loading = factor_loadings.loc[factor, scalar]
                                if not np.isnan(loading):
                                    segment_score += z_score * loading
                                    segment_n_scalars += 1
                    if segment_n_scalars > 0:
                        segment_scores.append(segment_score)
                if segment_scores:
                    subject_scores[factor] = float(np.mean(segment_scores))
                else:
                    subject_scores[factor] = np.nan
            else:
                for scalar in available_scalars:
                    if scalar not in roi_data:
                        continue
                    data = roi_data[scalar]
                    z_col = f"{scalar}_z"
                    row_key = resolve_subject_key(subject, data.index)
                    if row_key is not None and z_col in data.columns:
                        z_score = data.loc[row_key, z_col]
                        if not np.isnan(z_score):
                            loading = factor_loadings.loc[factor, scalar]
                            if not np.isnan(loading):
                                score += z_score * loading
                                n_scalars += 1
                subject_scores[factor] = score if n_scalars > 0 else np.nan

        factor_scores[subject] = subject_scores

    return pd.DataFrame(factor_scores).T


def _control_group_for_subject(sid: str, group_map: Dict[str, str]) -> Optional[str]:
    if sid in group_map:
        return group_map[sid]
    return get_group_from_subject_id(sid)


def compute_and_save_all_factor_scores(
    scalar_labels: Sequence[str],
    patient_groups: Sequence[str],
    control_groups: Sequence[str],
    factor_loadings: pd.DataFrame,
    output_dir: str,
    *,
    all_regions: Optional[Sequence[str]] = None,
    all_tracts: Optional[Sequence[str]] = None,
) -> None:
    """
    Write ``epilepsy_F*_scores.csv`` and ``controls_F*_scores.csv`` (controls with ``group``).
    """
    all_regions = list(all_regions) if all_regions is not None else discover_all_gm_regions()
    all_tracts = list(all_tracts) if all_tracts is not None else discover_all_wm_tracts()
    os.makedirs(output_dir, exist_ok=True)

    patient_subjects = load_temporal_patient_subjects_ordered()
    if not patient_subjects and all_regions:
        sample_region = all_regions[0]
        sample_scalar = scalar_labels[0] if scalar_labels else None
        if sample_scalar:
            sample_data = load_gm_region_scalar_data(sample_region, sample_scalar, patient_groups)
            if sample_data is not None:
                patient_subjects = sorted(list(sample_data.index.astype(str)))

    control_subjects = collect_control_subjects_union_from_gam(
        all_regions, all_tracts, scalar_labels, control_groups
    )
    control_group_map = collect_control_group_map(
        all_regions, all_tracts, scalar_labels, control_groups
    )

    if not patient_subjects and not control_subjects:
        print("Warning: No subjects found. Skipping factor score computation.")
        return

    if patient_subjects:
        print(f"Found {len(patient_subjects)} patient subjects")
    if control_subjects:
        print(f"Found {len(control_subjects)} control subjects")

    tract_metadata_df = load_tract_metadata_full()
    tract_to_end1_label: Dict[str, str] = {}
    tract_to_end2_label: Dict[str, str] = {}
    if not tract_metadata_df.empty and "label" in tract_metadata_df.columns:
        if "end1" in tract_metadata_df.columns:
            tract_to_end1_label = dict(zip(tract_metadata_df["label"], tract_metadata_df["end1"]))
        if "end2" in tract_metadata_df.columns:
            tract_to_end2_label = dict(zip(tract_metadata_df["label"], tract_metadata_df["end2"]))

    for group_name, groups, subjects in [
        ("epilepsy", patient_groups, patient_subjects),
        ("controls", control_groups, control_subjects),
    ]:
        if not subjects:
            print(f"\nSkipping {group_name} group - no subjects found.")
            continue

        wide_by_factor: Dict[str, Dict[str, pd.Series]] = {
            str(f): {} for f in factor_loadings.index
        }

        for region in tqdm(all_regions, desc=f"GM regions ({group_name})"):
            roi_data: Dict[str, pd.DataFrame] = {}
            for scalar in scalar_labels:
                data = load_gm_region_scalar_data(region, scalar, groups)
                if data is not None:
                    # Drop helper column before scoring
                    roi_data[scalar] = data.drop(columns=["_group"], errors="ignore")

            if not roi_data:
                continue

            factor_scores = compute_factor_scores(
                roi_data, region, scalar_labels, list(subjects), factor_loadings
            )
            if factor_scores.empty:
                continue
            for fac in factor_scores.columns:
                wide_by_factor[str(fac)][region] = factor_scores[fac]

        for tract in tqdm(all_tracts, desc=f"WM tracts ({group_name})"):
            roi_data = {}
            for scalar in scalar_labels:
                node_data = load_wm_tract_scalar_data(tract, scalar, groups)
                if node_data is not None:
                    node_data = node_data.drop(columns=["_group"], errors="ignore")
                    segment_data = compute_tract_segment_z_scores(node_data)
                    roi_data[scalar] = pd.DataFrame(
                        {
                            f"{scalar}_z_end1": segment_data["end1"],
                            f"{scalar}_z_core": segment_data["core"],
                            f"{scalar}_z_end2": segment_data["end2"],
                        }
                    )

            if not roi_data:
                continue

            for segment in ["end1", "core", "end2"]:
                segment_roi_data: Dict[str, pd.DataFrame] = {}
                for scalar in scalar_labels:
                    if scalar in roi_data:
                        data = roi_data[scalar]
                        z_col = f"{scalar}_z_{segment}"
                        if z_col in data.columns:
                            segment_roi_data[scalar] = pd.DataFrame(
                                {f"{scalar}_z": data[z_col]}
                            )

                if not segment_roi_data:
                    continue

                segment_factor_scores = compute_factor_scores(
                    segment_roi_data,
                    f"{tract}_{segment}",
                    scalar_labels,
                    list(subjects),
                    factor_loadings,
                )
                if segment_factor_scores.empty:
                    continue
                roi_key = wm_tract_segment_roi_key(
                    tract, segment, tract_to_end1_label, tract_to_end2_label
                )
                for fac in segment_factor_scores.columns:
                    wide_by_factor[str(fac)][roi_key] = segment_factor_scores[fac]

        for fac, col_map in wide_by_factor.items():
            if not col_map:
                continue
            df = pd.DataFrame(col_map)
            df.index.name = "subject"
            if group_name == "controls":
                group_series = pd.Series(
                    [_control_group_for_subject(str(s), control_group_map) for s in df.index],
                    index=df.index,
                    name="group",
                )
                df = pd.concat([group_series, df], axis=1)
            csv_path = ospj(output_dir, f"{group_name}_{fac}_scores.csv")
            df.to_csv(csv_path)
            print(f"  Saved {group_name} {fac} factor scores to {csv_path}")


def run_scores(
    scalar_labels: Sequence[str],
    factor_loadings: pd.DataFrame,
    output_dir: str,
    *,
    all_regions: Optional[Sequence[str]] = None,
    all_tracts: Optional[Sequence[str]] = None,
) -> None:
    compute_and_save_all_factor_scores(
        scalar_labels,
        PATIENT_GROUPS,
        CONTROL_GROUPS,
        factor_loadings,
        output_dir,
        all_regions=all_regions,
        all_tracts=all_tracts,
    )
