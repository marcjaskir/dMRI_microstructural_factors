"""Control-normative factor z from wide ``factor_scores`` tables."""
from __future__ import annotations

import glob
import os
from os.path import join as ospj
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import FACTOR_SCORES_DIR, FACTOR_Z_SCORES_DIR, MIN_CONTROLS_FOR_ROI_Z


def _parse_factor_from_scores_filename(path: str, cohort: str) -> Optional[str]:
    base = os.path.basename(path)
    prefix = f"{cohort}_"
    suffix = "_scores.csv"
    if not base.startswith(prefix) or not base.endswith(suffix):
        return None
    return base[len(prefix) : -len(suffix)]


def list_factors(scores_dir: str, cohort: str = "controls") -> List[str]:
    paths = sorted(glob.glob(ospj(scores_dir, f"{cohort}_*_scores.csv")))
    factors = []
    for p in paths:
        fac = _parse_factor_from_scores_filename(p, cohort)
        if fac:
            factors.append(fac)
    return factors


def _roi_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "group"]


def control_mean_sd(
    control_scores: pd.DataFrame,
    roi: str,
    *,
    min_controls: int = MIN_CONTROLS_FOR_ROI_Z,
) -> Tuple[float, float]:
    """Population mean/SD (ddof=0) of raw control factor scores at one ROI."""
    if roi not in control_scores.columns:
        return (np.nan, np.nan)
    vals = control_scores[roi].dropna()
    if len(vals) < min_controls:
        return (np.nan, np.nan)
    mean_val = float(vals.mean())
    std_val = float(vals.std(ddof=0))
    if std_val == 0.0 or np.isnan(std_val):
        return (np.nan, np.nan)
    return (mean_val, std_val)


def zscore_wide_table(
    scores: pd.DataFrame,
    control_scores: pd.DataFrame,
    *,
    min_controls: int = MIN_CONTROLS_FOR_ROI_Z,
) -> pd.DataFrame:
    """
    ``(x - control_mean) / control_sd`` per ROI column.

    ``control_scores`` must be numeric ROI columns only (no ``group``).
    Column order follows ``scores`` ROI columns.
    """
    rois = _roi_columns(scores)
    means = []
    stds = []
    for roi in rois:
        mean_val, std_val = control_mean_sd(
            control_scores, roi, min_controls=min_controls
        )
        means.append(mean_val)
        stds.append(std_val)
    mean_s = pd.Series(means, index=rois, dtype=float)
    std_s = pd.Series(stds, index=rois, dtype=float)
    valid = mean_s.notna() & std_s.notna() & (std_s != 0)
    out = (scores[rois] - mean_s) / std_s
    out.loc[:, ~valid] = np.nan
    return out


def write_factor_z_scores(
    scores_dir: str = FACTOR_SCORES_DIR,
    output_dir: str = FACTOR_Z_SCORES_DIR,
    factors: Optional[Sequence[str]] = None,
) -> None:
    """
    Write ``controls_F*_z_scores.csv`` and ``epilepsy_F*_z_scores.csv`` only.

    Uses all control rows in each ``controls_{factor}_scores.csv`` for mean/SD
    (same as the historical consolidated path with default ``compute_control_stats``).
    """
    os.makedirs(output_dir, exist_ok=True)
    if factors is None:
        factors = list_factors(scores_dir, "controls")
        if not factors:
            factors = list_factors(scores_dir, "epilepsy")

    for factor in factors:
        ct_path = ospj(scores_dir, f"controls_{factor}_scores.csv")
        ep_path = ospj(scores_dir, f"epilepsy_{factor}_scores.csv")
        if not os.path.exists(ct_path):
            print(f"Warning: missing {ct_path}; skip {factor}")
            continue

        ct = pd.read_csv(ct_path, index_col=0)
        group = None
        if "group" in ct.columns:
            group = ct["group"].copy()
            ct_num = ct.drop(columns=["group"])
        else:
            ct_num = ct

        ct_z = zscore_wide_table(ct_num, ct_num)
        ct_z.index.name = ct.index.name or "subject"
        if group is not None:
            ct_z = pd.concat([group.rename("group"), ct_z], axis=1)
        out_ct = ospj(output_dir, f"controls_{factor}_z_scores.csv")
        ct_z.to_csv(out_ct)
        print(f"  Saved controls {factor} z-scores to {out_ct}")

        if os.path.exists(ep_path):
            ep = pd.read_csv(ep_path, index_col=0)
            ep_num = ep.drop(columns=["group"], errors="ignore")
            # Preserve column order of epilepsy scores (same as historical export)
            ep_z = zscore_wide_table(ep_num, ct_num)
            ep_z.index.name = ep.index.name or "subject"
            out_ep = ospj(output_dir, f"epilepsy_{factor}_z_scores.csv")
            ep_z.to_csv(out_ep)
            print(f"  Saved epilepsy {factor} z-scores to {out_ep}")
