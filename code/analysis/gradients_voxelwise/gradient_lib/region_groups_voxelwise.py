"""Region-group mapping for parcel-level voxelwise gradients (WM by HCP1065 type)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_TRACTOMETRY_ROOT,
    GM_REGION_GROUP_NAMES,
    HCP1065_TRACT_METADATA_CSV,
    REGION_GROUP_GM_FACE,
    REGION_GROUP_WM_FACE,
    WM_REGION_GROUP_NAMES,
    WM_TYPE_DISPLAY,
)
from .gc_imports import gc_region_groups

_rg = gc_region_groups()
SUBCORTEX_ROI_TO_GROUP = _rg.SUBCORTEX_ROI_TO_GROUP
load_cortical_lobe_region_group_by_roi = _rg.load_cortical_lobe_region_group_by_roi
_glasser_suffix = _rg._glasser_suffix
_strip_4s_hemi_prefixes = _rg._strip_4s_hemi_prefixes

WM_TYPE_DISPLAY_ORDER: tuple[str, ...] = (
    "Association",
    "Projection",
    "Commissural",
    "Cerebellar",
    "Cranial nerves",
)


def load_tract_label_to_type_group() -> dict[str, str]:
    """Map HCP1065 tract label -> display type group."""
    df = pd.read_csv(HCP1065_TRACT_METADATA_CSV)
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        label = str(row["label"])
        raw_type = str(row["type"]).strip().lower()
        display = WM_TYPE_DISPLAY.get(raw_type)
        if display is not None:
            out[label] = display
    return out


def region_group_bar_facecolor(label: str) -> str:
    s = str(label)
    if s in WM_REGION_GROUP_NAMES:
        return REGION_GROUP_WM_FACE
    if s in GM_REGION_GROUP_NAMES:
        return REGION_GROUP_GM_FACE
    return REGION_GROUP_GM_FACE


def region_group_for_parcel(
    name: str,
    *,
    tract_to_type: dict[str, str],
    cortical_by_roi: dict[str, str],
) -> str | None:
    s = str(name)
    if s in tract_to_type:
        return tract_to_type[s]
    bare = _strip_4s_hemi_prefixes(s)
    g = SUBCORTEX_ROI_TO_GROUP.get(bare) or SUBCORTEX_ROI_TO_GROUP.get(s)
    if g is not None:
        return g
    gs = _glasser_suffix(s)
    for k in (s, bare, gs, f"Left_{gs}", f"Right_{gs}"):
        g = cortical_by_roi.get(k)
        if g is not None:
            return g
    return None


def build_parcel_to_region_group(
    g: pd.Series,
    *,
    tract_to_type: dict[str, str] | None = None,
    cortical_by_roi: dict[str, str] | None = None,
    tractometry_root: Path | None = None,
) -> dict[str, str]:
    """Parcel label -> region-group label."""
    root = tractometry_root or DEFAULT_TRACTOMETRY_ROOT
    if tract_to_type is None:
        tract_to_type = load_tract_label_to_type_group()
    if cortical_by_roi is None:
        cortical_by_roi = load_cortical_lobe_region_group_by_roi(root)
    out: dict[str, str] = {}
    for parcel in g.index:
        rg = region_group_for_parcel(
            str(parcel),
            tract_to_type=tract_to_type,
            cortical_by_roi=cortical_by_roi,
        )
        if rg is not None:
            out[str(parcel)] = rg
    return out


def subcortical_parcel_labels(tractometry_root: Path | None = None) -> frozenset[str]:
    """All 4S subcortex label variants from SUBCORTEX_ROI_TO_GROUP keys."""
    _ = tractometry_root
    labels: set[str] = set()
    for short in SUBCORTEX_ROI_TO_GROUP:
        labels.add(short)
        labels.add(f"LH-{short}")
        labels.add(f"RH-{short}")
        labels.add(f"LH_{short}")
        labels.add(f"RH_{short}")
    return frozenset(labels)
