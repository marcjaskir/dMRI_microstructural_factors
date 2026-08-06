"""
Region-level Mahalanobis distance and factor z-score summary tables (not asymmetry Cohen's d).

Per left_TLE / right_TLE group: top-N regions with Region labeled Ipsilateral or Contralateral
(relative to seizure focus), plus mean Mahalanobis and mean factor z (Overall, Non-Gaussian, Anisotropic).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from microstructural_asymmetry_report_mahalanobis import (
    _format_4s_subcortex_tex_label,
    _format_tex_wm_thirds_label,
    _load_glasser_additional_metadata,
    _load_subject_group,
    _load_tract_label_to_pretty_name,
    _wm_roi_to_tract_segment,
    _wm_tract_base_key,
)
from microstructural_asymmetry_report_scalars import (
    COHEND_D_TEX_SUBHEADERS,
    COHEND_LONGTABLE_COL_SPEC,
    FACTOR_Z_SCORES_DIR,
    SUMMARY_COHEND_TEX_USEPACKAGE,
    SUMMARY_TEX_TOP_N,
    _classify_factor_z_column,
    _cohend_region_label_tex,
    _cohend_subheader_tex,
    _factor_z_segment_to_asym_segment,
    _factor_z_wm_column_to_roi_label,
    _format_tex_cohens_d_value,
    _format_tex_cohens_d_value_bold,
    _label_lh_rh_variants,
    _latex_escape,
    _load_4s_cortex_labels_full,
    _load_4s_subcortex_labels_full,
    _load_4s_thalamus_roi_bases,
    _load_glasser_labels_full,
)

# Atlases in combined left/right TLE region-score tables (no whole-tract WM aggregate).
# Mahalanobis: derivatives/analysis/region_asymmetry_tle (GM), tract_asymmetry segments (WM).
# Factor z: derivatives/analysis/factor_z-scores/factor_z_scores/epilepsy_F{1,2,3}_z_scores.csv
INCLUDED_ATLASES = frozenset({"4s_subcortex", "glasser", "hcp1065_thirds"})
HCP1065_WM_TRACT_TYPES = frozenset({"association", "projection"})
FACTOR_Z_TEX_COLS = ["mean_factor_1", "mean_factor_2", "mean_factor_3"]

SIDE_IPSI = "Ipsi."
SIDE_CONTRA = "Contra."

FACTOR_Z_INDICES: Dict[int, str] = {1: "F1", 2: "F2", 3: "F3"}
REGION_SCORE_VALUE_COLS = [
    "mean_mahalanobis",
    "mean_factor_1",
    "mean_factor_2",
    "mean_factor_3",
]


def _column_is_left_hemisphere(col: str) -> Optional[bool]:
    """Return True/False for left/right hemisphere column names; None if unknown."""
    c = str(col).strip()
    if c.startswith("Left_"):
        return True
    if c.startswith("Right_"):
        return False
    if c.startswith("LH_") or c.startswith("LH-"):
        return True
    if c.startswith("RH_") or c.startswith("RH-"):
        return False
    if "_L_" in c:
        return True
    if "_R_" in c:
        return False
    if c.endswith("_L"):
        return True
    if c.endswith("_R"):
        return False
    return None


def _side_for_tle_group(is_left: bool, tle_group: str) -> str:
    if tle_group == "left_TLE":
        return SIDE_IPSI if is_left else SIDE_CONTRA
    if tle_group == "right_TLE":
        return SIDE_CONTRA if is_left else SIDE_IPSI
    raise ValueError(f"Unknown TLE group: {tle_group}")


def _atlas_for_factor_column(
    col: str,
    labels_sub: Set[str],
    labels_ctx: Set[str],
    labels_gl: Set[str],
    tract_base_to_type: Dict[str, str],
) -> Optional[str]:
    """Map factor z column to summary atlas key (matches per-atlas .tex stems)."""
    tissue = _classify_factor_z_column(
        col, labels_sub, labels_ctx, labels_gl, tract_base_to_type
    )
    if tissue is None:
        return None
    if tissue == "subcortex_gm":
        return "4s_subcortex"
    if tissue == "cortex_gm":
        for key in _label_lh_rh_variants(col):
            if key in labels_gl:
                return "glasser"
        # Schaefer / 4S156 cortical parcels are excluded from these tables.
        return None
    if tissue in ("association_wm", "projection_wm"):
        return "hcp1065_thirds"
    return None


def _hcp1065_wm_tract_allowed(
    tract_with_hemi: str, tract_base_to_type: Dict[str, str]
) -> bool:
    """Only HCP1065 association / projection thirds (not commissural or other WM)."""
    tb = _wm_tract_base_key(str(tract_with_hemi).strip())
    return tract_base_to_type.get(tb) in HCP1065_WM_TRACT_TYPES


def _wm_tract_type_from_column(col: str, tract_base_to_type: Dict[str, str]) -> Optional[str]:
    """association or projection for HCP1065 thirds split."""
    c = str(col).strip()
    if c.endswith("_core"):
        tract_full = c[: -len("_core")]
    elif "_end-" in c:
        tract_full = c.rsplit("_", 1)[0]
    elif "_L_" in c:
        tract_full = c.split("_L_", 1)[0] + "_L"
    elif "_R_" in c:
        tract_full = c.split("_R_", 1)[0] + "_R"
    else:
        return None
    if tract_full.endswith("_L") or tract_full.endswith("_R"):
        tb = tract_full[:-2]
        return tract_base_to_type.get(tb)
    return None


def _roi_key_from_factor_column(col: str, atlas: str) -> Optional[str]:
    """Pairing key aligned with Mahalanobis ``roi_id`` / tract keys."""
    c = str(col).strip()
    if atlas == "glasser":
        if c.startswith("Left_"):
            return c[5:]
        if c.startswith("Right_"):
            return c[6:]
        return None
    if atlas == "4s_subcortex":
        for pref in ("LH_", "LH-", "RH_", "RH-"):
            if c.startswith(pref):
                return c[len(pref) :]
        return None
    if atlas == "hcp1065_thirds":
        return _wm_factor_column_to_roi_key(c)
    return None


def _wm_factor_column_to_roi_key(col: str) -> Optional[str]:
    """
    Factor-z WM column -> tract-asymmetry ``roi_id`` (e.g. ``CPT_F_L_end-I`` -> ``CPT_F_I``).
    Matches ``_factor_z_wm_column_to_roi_label`` in the scalars report; adds ``_R_`` handling.
    """
    c = str(col).strip()
    mapped = _factor_z_wm_column_to_roi_label(c)
    if mapped != c:
        return mapped
    for hemi in ("_L_", "_R_"):
        if hemi in c:
            tract_base, segment = c.split(hemi, 1)
            return f"{tract_base}_{_factor_z_segment_to_asym_segment(segment)}"
    for hemi_suffix in ("_L", "_R"):
        if c.endswith(f"{hemi_suffix}_core"):
            tract_hemi = c[: -len("_core")]
            if tract_hemi.endswith(hemi_suffix):
                return f"{tract_hemi[: -len(hemi_suffix)]}_core"
    return None


def _is_wm_factor_z_column(col: str) -> bool:
    c = str(col).strip()
    return bool(
        c.endswith("_core")
        or "_end-" in c
        or "_L_" in c
        or "_R_" in c
        or (c.endswith("_L") or c.endswith("_R"))
    )


def build_mahalanobis_ipsi_contra_long(
    full_long: pd.DataFrame,
    subject_group: Dict[str, str],
    tract_base_to_type: Dict[str, str],
) -> pd.DataFrame:
    """
    Subject-level long table: Mahalanobis distance per (atlas, roi_key, Ipsilateral|Contralateral).
    """
    rows: List[dict] = []

    def _append_pair(
        sub: str,
        group: str,
        atlas: str,
        roi_key: str,
        ipsi_val: float,
        contra_val: float,
        *,
        wm_tract_type: Optional[str] = None,
    ) -> None:
        if group not in ("left_TLE", "right_TLE"):
            return
        for side, val in ((SIDE_IPSI, ipsi_val), (SIDE_CONTRA, contra_val)):
            row: dict = {
                "sub": sub,
                "group": group,
                "atlas": atlas,
                "roi_key": roi_key,
                "side": side,
                "mahalanobis": float(val),
            }
            if wm_tract_type is not None:
                row["wm_tract_type"] = wm_tract_type
            rows.append(row)

    for atlas in ("4s_subcortex", "glasser"):
        sub = full_long[(full_long["atlas"] == atlas) & full_long["ipsi"].notna() & full_long["contra"].notna()]
        for _, r in sub.iterrows():
            subj = str(r["sub"])
            group = subject_group.get(subj)
            if group not in ("left_TLE", "right_TLE"):
                continue
            base = str(r["roi_id"])
            _append_pair(subj, group, atlas, base, float(r["ipsi"]), float(r["contra"]))

    wm = full_long[(full_long["roi_type"] == "wm") & full_long["ipsi"].notna() & full_long["contra"].notna()]
    if not wm.empty:
        for _, r in wm.iterrows():
            subj = str(r["sub"])
            group = subject_group.get(subj)
            if group not in ("left_TLE", "right_TLE"):
                continue
            tract_base, segment = _wm_roi_to_tract_segment(str(r["roi_id"]))
            if not _hcp1065_wm_tract_allowed(tract_base, tract_base_to_type):
                continue
            roi_key = f"{tract_base}_{segment}"
            tb_key = _wm_tract_base_key(tract_base)
            wm_type = tract_base_to_type.get(tb_key) if tract_base_to_type else None
            _append_pair(
                subj,
                group,
                "hcp1065_thirds",
                roi_key,
                float(r["ipsi"]),
                float(r["contra"]),
                wm_tract_type=wm_type,
            )

    if not rows:
        return pd.DataFrame(
            columns=["sub", "group", "atlas", "roi_key", "side", "mahalanobis"]
        )
    return pd.DataFrame(rows)


def build_factor_z_ipsi_contra_long(
    tract_base_to_type: Dict[str, str],
    subject_group: Dict[str, str],
) -> pd.DataFrame:
    """Subject × region × factor long table with Ipsilateral/Contralateral side labels."""
    if not FACTOR_Z_SCORES_DIR.is_dir():
        return pd.DataFrame(
            columns=["sub", "group", "atlas", "roi_key", "side", "factor_index", "z"]
        )
    labels_sub = _load_4s_subcortex_labels_full()
    labels_ctx = _load_4s_cortex_labels_full()
    labels_gl = _load_glasser_labels_full()
    rows: List[dict] = []

    for fk, fac in FACTOR_Z_INDICES.items():
        path = FACTOR_Z_SCORES_DIR / f"epilepsy_{fac}_z_scores.csv"
        if not path.exists():
            continue
        try:
            wide = pd.read_csv(path)
        except Exception:
            continue
        if wide.empty or "subject" not in wide.columns:
            continue
        value_cols = [c for c in wide.columns if c != "subject"]
        for _, srow in wide.iterrows():
            subj = str(srow["subject"]).strip()
            group = subject_group.get(subj)
            if group not in ("left_TLE", "right_TLE"):
                continue
            for col in value_cols:
                z = srow[col]
                if pd.isna(z) or not np.isfinite(float(z)):
                    continue
                is_left = _column_is_left_hemisphere(col)
                if is_left is None:
                    continue
                side = _side_for_tle_group(is_left, group)
                if _is_wm_factor_z_column(col):
                    roi_key = _wm_factor_column_to_roi_key(col)
                    if roi_key is None:
                        continue
                    tract_base, _seg = _wm_roi_to_tract_segment(roi_key)
                    if not _hcp1065_wm_tract_allowed(tract_base, tract_base_to_type):
                        continue
                    atlas = "hcp1065_thirds"
                else:
                    atlas = _atlas_for_factor_column(
                        col, labels_sub, labels_ctx, labels_gl, tract_base_to_type
                    )
                    if atlas is None or atlas not in INCLUDED_ATLASES:
                        continue
                    roi_key = _roi_key_from_factor_column(col, atlas)
                    if roi_key is None:
                        continue
                row = {
                    "sub": subj,
                    "group": group,
                    "atlas": atlas,
                    "roi_key": roi_key,
                    "side": side,
                    "factor_index": fk,
                    "z": float(z),
                }
                if atlas == "hcp1065_thirds":
                    row["wm_tract_type"] = _wm_tract_type_from_column(col, tract_base_to_type)
                rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["sub", "group", "atlas", "roi_key", "side", "factor_index", "z"]
        )
    return pd.DataFrame(rows)


def _aggregate_mahalanobis(long_df: pd.DataFrame, tle_group: str) -> pd.DataFrame:
    sub = long_df[long_df["group"] == tle_group]
    if sub.empty:
        return pd.DataFrame(columns=["atlas", "roi_key", "side", "mean_mahalanobis"])
    agg = (
        sub.groupby(["atlas", "roi_key", "side"], as_index=False)["mahalanobis"]
        .mean()
        .rename(columns={"mahalanobis": "mean_mahalanobis"})
    )
    return agg


def _aggregate_factor_z(long_df: pd.DataFrame, tle_group: str) -> pd.DataFrame:
    sub = long_df[long_df["group"] == tle_group]
    if sub.empty:
        return pd.DataFrame(
            columns=["atlas", "roi_key", "side", "mean_factor_1", "mean_factor_2", "mean_factor_3"]
        )
    agg = sub.groupby(["atlas", "roi_key", "side", "factor_index"], as_index=False)["z"].mean()
    wide = agg.pivot_table(
        index=["atlas", "roi_key", "side"],
        columns="factor_index",
        values="z",
        aggfunc="first",
    ).reset_index()
    rename = {1: "mean_factor_1", 2: "mean_factor_2", 3: "mean_factor_3"}
    wide = wide.rename(columns=rename)
    for c in ("mean_factor_1", "mean_factor_2", "mean_factor_3"):
        if c not in wide.columns:
            wide[c] = float("nan")
    return wide[["atlas", "roi_key", "side", "mean_factor_1", "mean_factor_2", "mean_factor_3"]]


def _merge_region_scores(mahal_agg: pd.DataFrame, factor_agg: pd.DataFrame) -> pd.DataFrame:
    if mahal_agg.empty and factor_agg.empty:
        return pd.DataFrame()
    if mahal_agg.empty:
        merged = factor_agg.copy()
        merged["mean_mahalanobis"] = float("nan")
    elif factor_agg.empty:
        merged = mahal_agg.copy()
        for c in ("mean_factor_1", "mean_factor_2", "mean_factor_3"):
            merged[c] = float("nan")
    else:
        merged = mahal_agg.merge(
            factor_agg, on=["atlas", "roi_key", "side"], how="outer"
        )
    for c in REGION_SCORE_VALUE_COLS:
        if c not in merged.columns:
            merged[c] = float("nan")
    return merged


def _display_region_label(
    roi_key: str,
    side: str,
    atlas: str,
    *,
    thalamus_bases: Set[str],
    tract_pretty: Dict[str, str],
    glasser_add: pd.DataFrame,
) -> str:
    """``Ipsilateral`` / ``Contralateral`` prefix + anatomical name (not Left/Right)."""
    base = str(roi_key).strip()
    if atlas == "glasser" and not glasser_add.empty and base in glasser_add.index:
        ser = glasser_add.loc[base]
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[0]
        try:
            v = ser["regionLongName"]
        except (KeyError, TypeError, IndexError):
            v = None
        if v is not None and pd.notna(v) and str(v).strip():
            name = str(v).strip()
            if name == "Hippocampus":
                name = "Parahippocampal"
        else:
            name = base.replace("_", " ")
    elif atlas == "4s_subcortex":
        name = _format_4s_subcortex_tex_label(base, thalamus_bases)
    elif atlas == "hcp1065_thirds":
        name = _format_tex_wm_thirds_label(base, tract_pretty)
    else:
        name = base.replace("_", " ")
    return f"{side} {name}"


def _tex_region_score_cells(row: pd.Series) -> List[str]:
    """Mahalanobis + three factor z columns; bold largest |z| among factors (not Mahalanobis)."""
    mahal_tex = _format_tex_cohens_d_value(row.get("mean_mahalanobis"))
    factor_vals: List[Optional[float]] = []
    for c in FACTOR_Z_TEX_COLS:
        v = row.get(c)
        if v is None or pd.isna(v) or not np.isfinite(float(v)):
            factor_vals.append(None)
        else:
            factor_vals.append(float(v))
    factor_abs = [abs(v) for v in factor_vals if v is not None]
    max_abs = max(factor_abs) if factor_abs else None
    factor_tex: List[str] = []
    for v in factor_vals:
        if v is None:
            factor_tex.append("---")
        elif max_abs is not None and abs(v) >= max_abs - 1e-12:
            factor_tex.append(_format_tex_cohens_d_value_bold(v, True))
        else:
            factor_tex.append(_format_tex_cohens_d_value_bold(v, False))
    return [mahal_tex] + factor_tex


def _save_region_score_table_tex(
    df: pd.DataFrame,
    out_path: Path,
    *,
    region_col_formatter: Callable[[str, str, str], str],
    first_column_header: str = "Region",
    top_n: int = SUMMARY_TEX_TOP_N,
) -> None:
    """Write longtable: Region | Mahalanobis | Overall | Non-Gaussian | Anisotropic."""
    if df.empty:
        return
    work = df.sort_values("mean_mahalanobis", ascending=False, na_position="last").head(int(top_n))
    if work.empty:
        return

    col_spec = COHEND_LONGTABLE_COL_SPEC
    header_cells = [
        _cohend_subheader_tex(first_column_header),
        *[_cohend_subheader_tex(h) for h in COHEND_D_TEX_SUBHEADERS],
    ]
    header_line = " & ".join(header_cells) + r" \\"
    header_block = [
        r"\toprule",
        header_line,
        r"\midrule",
    ]
    lines = [
        SUMMARY_COHEND_TEX_USEPACKAGE,
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{longtable}{" + col_spec + "}",
        *header_block,
        r"\endfirsthead",
        *header_block,
        r"\endhead",
        r"\midrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, row in work.iterrows():
        disp = region_col_formatter(str(row["roi_key"]), str(row["side"]), str(row["atlas"]))
        lab_tex = _cohend_region_label_tex(disp)
        vals = _tex_region_score_cells(row)
        lines.append(" & ".join([lab_tex] + vals) + r" \\")
    lines.extend([r"\end{longtable}", r"\endgroup"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_combined_table_for_tle_group(
    mahal_long: pd.DataFrame,
    factor_long: pd.DataFrame,
    tle_group: str,
) -> pd.DataFrame:
    """One table per TLE group: all atlases (GM + WM), sorted by mean Mahalanobis."""
    mahal_agg = _aggregate_mahalanobis(mahal_long, tle_group)
    factor_agg = _aggregate_factor_z(factor_long, tle_group)
    merged = _merge_region_scores(mahal_agg, factor_agg)
    if merged.empty:
        return merged
    merged = merged[merged["atlas"].isin(INCLUDED_ATLASES)].copy()
    return merged.sort_values("mean_mahalanobis", ascending=False, na_position="last")


def save_region_score_tex_per_tle_group(
    full_long: pd.DataFrame,
    tract_base_to_type: Dict[str, str],
    report_dir: Path,
    *,
    top_n: int = SUMMARY_TEX_TOP_N,
) -> None:
    """Write ``summary_{left_TLE|right_TLE}_region_scores_top{N}.tex`` (all atlases combined)."""
    report_dir = Path(report_dir)
    subject_group = _load_subject_group()
    if not subject_group:
        print(
            "region_score_summary_tables: no TLE groups in inclusion metadata; skipping.",
            file=sys.stderr,
        )
        return

    mahal_long = build_mahalanobis_ipsi_contra_long(
        full_long, subject_group, tract_base_to_type
    )
    factor_long = build_factor_z_ipsi_contra_long(tract_base_to_type, subject_group)

    thalamus_bases = _load_4s_thalamus_roi_bases()
    tract_pretty = _load_tract_label_to_pretty_name()
    glasser_add = _load_glasser_additional_metadata()

    def _fmt(roi_key: str, side: str, atlas: str) -> str:
        return _display_region_label(
            roi_key,
            side,
            atlas,
            thalamus_bases=thalamus_bases,
            tract_pretty=tract_pretty,
            glasser_add=glasser_add,
        )

    for tle_group in ("left_TLE", "right_TLE"):
        df = build_combined_table_for_tle_group(mahal_long, factor_long, tle_group)
        stem = f"summary_{tle_group}_region_scores"
        out_path = report_dir / f"{stem}_top{top_n}.tex"
        _save_region_score_table_tex(
            df,
            out_path,
            region_col_formatter=_fmt,
            first_column_header="Region",
            top_n=top_n,
        )
        if out_path.exists():
            print(f"Wrote {out_path}")
