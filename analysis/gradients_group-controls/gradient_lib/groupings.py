"""Yeo network and Mesulam laminar-type loaders + color palettes.

All loaders return mappings keyed by **both** the full Glasser ROI name (``Left_*`` /
``Right_*``) and the stripped suffix (``V1``, etc.) so they can be joined against wide
factor-score tables regardless of whether the wide columns keep the hemisphere prefix.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

NEUROAXIS_COLS: tuple[str, ...] = ("rank_ap", "rank_dv", "rank_ml")
NEUROAXIS_LABELS: dict[str, str] = {
    "rank_ap": "A-P",
    "rank_dv": "D-V",
    "rank_ml": "M-L",
}


def _glasser_suffix(name: str) -> str:
    s = str(name)
    for pref in ("Left_", "Right_"):
        if s.startswith(pref):
            return s[len(pref) :]
    return s


def _add_both_keys(out: dict, full_name: str, value: object) -> None:
    out[full_name] = value
    suf = _glasser_suffix(full_name)
    if suf and suf not in out:
        out[suf] = value


def load_yeo_labels(tractometry_root: Path) -> dict[str, str]:
    """ROI -> Yeo functional network label (``community_yeo`` in ``atlas-Glasser_dseg.tsv``)."""
    p = tractometry_root / "data/atlases/Glasser/atlas-Glasser_dseg.tsv"
    df = pd.read_csv(p, sep="\t")
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        label = str(row["label"])
        yeo = str(row["community_yeo"]).strip()
        if not yeo or yeo.lower() in ("nan", "n/a", "none"):
            continue
        _add_both_keys(out, label, yeo)
    return out


def _glasser_parc_region_to_roi_keys(region: str) -> list[str]:
    """``glasser_parc.csv`` uses ``V1_L`` / ``V1_R``; factor tables often use ``Left_V1`` / ``V1``."""
    s = str(region).strip()
    keys = [s]
    if s.endswith("_L") and len(s) > 2:
        base = s[:-2]
        keys.extend([f"Left_{base}", base])
    elif s.endswith("_R") and len(s) > 2:
        base = s[:-2]
        keys.extend([f"Right_{base}", base])
    return keys


def load_mesulam_labels(tractometry_root: Path) -> dict[str, str]:
    """ROI -> Mesulam laminar type from ``data/atlases/Glasser/glasser_parc.csv`` (``mesulam``)."""
    p = tractometry_root / "data/atlases/Glasser/glasser_parc.csv"
    df = pd.read_csv(p)
    if "region" not in df.columns or "mesulam" not in df.columns:
        warnings.warn(
            f"{p}: expected columns 'region' and 'mesulam'; got {list(df.columns)}",
            stacklevel=2,
        )
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        m = str(row["mesulam"]).strip()
        if not m or m.lower() in ("nan", "n/a", "none"):
            continue
        region = str(row["region"]).strip()
        for k in _glasser_parc_region_to_roi_keys(region):
            out[k] = m
    return out


def yeo_network_color(network: str) -> str:
    """Standard Yeo 7-network palette."""
    key = str(network).strip().lower()
    table = {
        "visual": "#781286",
        "somatomotor": "#4682B4",
        "somatosensory": "#4682B4",
        "dorsal attention": "#00760E",
        "ventral attention": "#C43AFA",
        "limbic": "#DCF8A4",
        "frontoparietal": "#E69422",
        "default": "#CD3E4E",
        "default mode": "#CD3E4E",
    }
    return table.get(key, "#808080")


def load_neuroaxis_ranks(
    tractometry_root: Path,
    *,
    roi_labels: set[str] | frozenset[str] | None = None,
) -> dict[str, dict[str, float]]:
    """ROI -> ``{"A-P": float, "D-V": float, "M-L": float}`` ranks over the whole brain.

    Reads ``derivatives/atlas_centroids/wholebrain_centroids.csv`` (label, atlas,
    x, y, z) and computes ranks across the union of Glasser cortex, 4S156
    subcortex, and HCP1065 tract thirds. When ``roi_labels`` is set, ranks are
    computed only among those labels (typically the gradient embedding ROIs:
    360 Glasser cortex + 56 4S156 subcortex + 144 bilateral HCP1065 tract-thirds).

    * ``A-P``: ``1`` = most anterior (largest ``y``), increasing posteriorly.
    * ``D-V``: ``1`` = most dorsal (largest ``z``), increasing ventrally.
    * ``M-L``: ``1`` = most mesial (smallest ``|x|``), increasing laterally.

    Output keys carry both the full label and, for ``Left_*`` / ``Right_*`` rows,
    the bare suffix (via ``_add_both_keys``).
    """
    p = tractometry_root / "derivatives/atlas_centroids/wholebrain_centroids.csv"
    if not p.is_file():
        warnings.warn(f"Missing whole-brain centroids file: {p}", stacklevel=2)
        return {}
    df = pd.read_csv(p)
    if roi_labels is not None:
        labels_keep = {str(x) for x in roi_labels}
        df = df[df["label"].astype(str).isin(labels_keep)].copy()
        if df.empty:
            warnings.warn(
                "load_neuroaxis_ranks: no centroid rows match roi_labels filter.",
                stacklevel=2,
            )
            return {}
    required = {"label", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        warnings.warn(
            f"Whole-brain centroids file missing columns {sorted(missing)}: {p}",
            stacklevel=2,
        )
        return {}
    rk_ap = df["y"].rank(method="min", ascending=False).to_numpy()
    rk_dv = df["z"].rank(method="min", ascending=False).to_numpy()
    rk_ml = df["x"].abs().rank(method="min", ascending=True).to_numpy()
    out: dict[str, dict[str, float]] = {}
    labels = df["label"].astype(str).tolist()
    for label, ap, dv, ml in zip(labels, rk_ap, rk_dv, rk_ml):
        rec = {"A-P": float(ap), "D-V": float(dv), "M-L": float(ml)}
        _add_both_keys(out, label, rec)
        # Also key 4S subcortex labels (LH-/RH-/LH_/RH_) by their bare suffix
        # so lookups match either the full or the suffix form used downstream.
        for pref in ("LH-", "RH-", "LH_", "RH_"):
            if label.startswith(pref):
                bare = label[len(pref):]
                if bare and bare not in out:
                    out[bare] = rec
                break
    return out


def mesulam_type_color(mesulam_type: str) -> str:
    """Stable colors for Mesulam classes from ``glasser_parc.csv``."""
    key = str(mesulam_type).strip().lower()
    table = {
        "heteromodal": "#6a1b9a",
        "idiotypic": "#1565c0",
        "paralimbic": "#c62828",
        "unimodal": "#2e7d32",
    }
    return table.get(key, "#757575")
