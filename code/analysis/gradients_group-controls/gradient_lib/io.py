"""Controls-only wide-table I/O: parsers, region column collection, matrix load, tissue helpers."""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Literal

import pandas as pd

from .config import (
    DEFAULT_TRACTOMETRY_ROOT,
    FOUR_S_SUBCORTICAL_ATLAS_NAMES,
    HCP1065_TRACT_METADATA_REL,
    NON_REGION_COLS,
)


def parse_factor_files(
    factor_dir: Path,
    factors_filter: set[int] | None,
) -> list[tuple[str, Path]]:
    """``controls_F{n}_scores.csv`` only (never ``*_z_scores.csv``)."""
    by_factor: dict[int, Path] = {}
    for p in sorted(glob.glob(str(factor_dir / "controls_F*_scores.csv"))):
        name = Path(p).name
        m = re.match(r"controls_F(\d+)_scores\.csv$", name, re.I)
        if not m:
            continue
        n = int(m.group(1))
        if factors_filter is not None and n not in factors_filter:
            continue
        by_factor[n] = Path(p)
    return [(f"F{n}", by_factor[n]) for n in sorted(by_factor)]


def load_factor_matrix_from_df(df: pd.DataFrame) -> pd.DataFrame:
    region_cols = [c for c in df.columns if c not in NON_REGION_COLS]
    out = df[region_cols].apply(pd.to_numeric, errors="coerce")
    out.columns = out.columns.astype(str)
    return out


def load_region_means_from_df(df: pd.DataFrame) -> pd.Series:
    region_cols = [c for c in df.columns if c not in NON_REGION_COLS]
    return df[region_cols].mean(axis=0, skipna=True)


def region_is_white_matter_column(name: str) -> bool:
    s = str(name)
    return "_core" in s or "_end-" in s


def region_is_wm_end_segment(name: str) -> bool:
    return "_end-" in str(name)


def region_is_wm_core_segment(name: str) -> bool:
    s = str(name)
    if "_end-" in s:
        return False
    return "_core" in s


def parse_wm_tract_end_column_name(name: str) -> tuple[str, str] | None:
    s = str(name)
    if "_end-" not in s:
        return None
    tract_label, end_code = s.rsplit("_end-", 1)
    if not tract_label or not end_code:
        return None
    return (tract_label, end_code)


def load_hcp1065_tract_metadata(tractometry_root: Path | None = None) -> pd.DataFrame:
    """HCP1065 tract table with ``end1_loc`` / ``end2_loc`` columns, or empty DF if missing."""
    root = (
        Path(tractometry_root)
        if tractometry_root is not None
        else DEFAULT_TRACTOMETRY_ROOT
    )
    p = root / HCP1065_TRACT_METADATA_REL
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p)


def _wm_endpoint_code_norm(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    t = str(x).strip()
    if t.upper() in ("", "NA", "N/A", "NAN"):
        return ""
    return t


def wm_end_loc_class_from_metadata(
    column_name: str, meta: pd.DataFrame
) -> Literal["cortex", "subcortex", "other"] | None:
    """Classify a WM end column as ``cortex`` / ``subcortex`` / ``other`` via HCP1065 metadata."""
    parsed = parse_wm_tract_end_column_name(column_name)
    if parsed is None:
        return None
    tract_label, end_code = parsed
    if meta is None or meta.empty or "label" not in meta.columns:
        return "other"
    m = meta.loc[meta["label"].astype(str) == str(tract_label)]
    if m.empty:
        return "other"
    row = m.iloc[0]
    e1 = _wm_endpoint_code_norm(row.get("end1", ""))
    e2 = _wm_endpoint_code_norm(row.get("end2", ""))
    seg = str(end_code).strip()
    if e1 and seg.upper() == e1.upper():
        loc_raw = str(row.get("end1_loc", "") or "")
    elif e2 and seg.upper() == e2.upper():
        loc_raw = str(row.get("end2_loc", "") or "")
    else:
        return "other"
    t = str(loc_raw).strip().lower()
    if t in ("", "na", "n/a", "nan"):
        return "other"
    if t == "cortex":
        return "cortex"
    if t == "subcortex":
        return "subcortex"
    return "other"


def subcortical_grey_matter_column_names(tractometry_root: Path) -> frozenset[str]:
    s4_tsv = tractometry_root / "data/atlases/4S/atlas-4S156Parcels_dseg.tsv"
    df_4s = pd.read_csv(s4_tsv, sep="\t")
    four_s_subcortical_mask = df_4s["atlas_name"].astype(str).isin(
        FOUR_S_SUBCORTICAL_ATLAS_NAMES
    )
    names: set[str] = set()
    for _, row in df_4s.loc[four_s_subcortical_mask].iterrows():
        full = str(row["label"])
        suffix = (
            full.replace("LH-", "")
            .replace("RH-", "")
            .replace("LH_", "")
            .replace("RH_", "")
        )
        names.add(full)
        names.add(suffix)
    return frozenset(names)


def region_hemisphere_side(region: object) -> Literal["left", "right"] | None:
    s = str(region)
    if s.startswith("Left_") or s.startswith("LH-") or s.startswith("LH_"):
        return "left"
    if s.startswith("Right_") or s.startswith("RH-") or s.startswith("RH_"):
        return "right"
    if "_L_" in s:
        return "left"
    if "_R_" in s:
        return "right"
    return None


def collect_plotted_region_columns(
    X: pd.DataFrame,
    glasser_tsv: pd.DataFrame,
    df_4s: pd.DataFrame,
    four_s_subcortical_mask: pd.Series,
) -> list[str]:
    """
    Select ROI columns from the wide table that match atlas ROI names. Mirrors the
    legacy factor_gradients pipeline: Glasser labels (with ``Left_`` / ``Right_`` or the
    stripped suffix), 4S subcortical labels, and WM tract columns (``_core``, ``_end-``).
    """
    cols = set(X.columns.astype(str))
    needed: set[str] = set()
    for _, row in glasser_tsv.iterrows():
        full = str(row["label"])
        suffix = full.replace("Left_", "").replace("Right_", "")
        if full in cols:
            needed.add(full)
        elif suffix in cols:
            needed.add(suffix)
    for _, row in df_4s.loc[four_s_subcortical_mask].iterrows():
        full = str(row["label"])
        suffix = (
            full.replace("LH-", "")
            .replace("RH-", "")
            .replace("LH_", "")
            .replace("RH_", "")
        )
        if full in cols:
            needed.add(full)
        elif suffix in cols:
            needed.add(suffix)
    for k in cols:
        if "_core" in k or "_end-" in k:
            needed.add(k)
    return sorted(needed)


def load_glasser_atlas_tsv(tractometry_root: Path) -> pd.DataFrame:
    return pd.read_csv(
        tractometry_root / "data/atlases/Glasser/atlas-Glasser_dseg.tsv",
        sep="\t",
    )


def glasser_parcel_name_set(tractometry_root: Path) -> frozenset[str]:
    """All Glasser parcel name variants that can appear in wide tables (``Left_*`` / ``Right_*`` and suffix)."""
    df = load_glasser_atlas_tsv(tractometry_root)
    names: set[str] = set()
    for _, row in df.iterrows():
        full = str(row["label"])
        suffix = full.replace("Left_", "").replace("Right_", "")
        names.add(full)
        names.add(suffix)
    return frozenset(names)


def load_4s_atlas_tsv(tractometry_root: Path) -> tuple[pd.DataFrame, pd.Series]:
    df_4s = pd.read_csv(
        tractometry_root / "data/atlases/4S/atlas-4S156Parcels_dseg.tsv",
        sep="\t",
    )
    mask = df_4s["atlas_name"].astype(str).isin(FOUR_S_SUBCORTICAL_ATLAS_NAMES)
    return df_4s, mask
