#!/usr/bin/env python3
"""Compute normative (scalar × scalar) covariance from control tract GAM z-scores.

Uses the same pyAFQ paths/config as tract asymmetry; controls = group != penn_epilepsy.
Levels: segment (end1/core/end2) and node (100 nodes). Writes cov/invcov (+ optional MinCovDet).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root  # noqa: E402

PROJECT_ROOT = project_root()

import numpy as np
import pandas as pd

try:
    from sklearn.covariance import MinCovDet
except ImportError:
    MinCovDet = None  # type: ignore

# Package root is code/analysis (sibling of this package dir)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tract_asymmetry import config as cfg
from tract_asymmetry.core import TractAsymmetry, _parse_segment

CONTROL_GROUP_FILTER = "penn_epilepsy"
DEFAULT_STAT = "mean"
OUTPUT_COV_FNAME = "cov.csv"
OUTPUT_INVCOV_FNAME = "invcov.csv"
OUTPUT_COV_MINCOVDET_FNAME = "cov_mincovdet.csv"
OUTPUT_INVCOV_MINCOVDET_FNAME = "invcov_mincovdet.csv"
SAMPLE_TRACT = "ILF_L"


def get_paths(base_dir: Path) -> Dict[str, Path]:
    """Same paths as tract_asymmetry; output_dir points to tract_asymmetry_normative."""
    base = Path(base_dir)
    paths = cfg.get_paths(base)
    paths["output_dir"] = base / "derivatives" / "analysis" / "tract_asymmetry_normative"
    return paths


def get_scalars(gam_dir: Path, stat: str) -> List[str]:
    """Discover scalar names from GAM dir (sample tract)."""
    prefix = f"{SAMPLE_TRACT}_"
    suffix = f"_stat-{stat}_gam"
    all_scalars: List[str] = []
    sample_dir = gam_dir / SAMPLE_TRACT
    if not sample_dir.is_dir():
        return []
    for p in sample_dir.iterdir():
        if not p.is_file() or p.suffix != ".csv":
            continue
        stem = p.stem
        if not stem.startswith(prefix) or not stem.endswith(suffix):
            continue
        scalar = stem[len(prefix) : -len(suffix)]
        all_scalars.append(scalar)
    return sorted(set(all_scalars))


def get_all_tract_segment_pairs(ta: TractAsymmetry, gam_dir: Path) -> List[Tuple[str, str]]:
    """Return (tract, segment) for every tract in gam_dir and each of its segments (end1, core, end2)."""
    if ta._meta is None:
        ta.load_metadata()
    meta = ta._meta
    out: List[Tuple[str, str]] = []
    tract_dirs = sorted(d.name for d in gam_dir.iterdir() if d.is_dir())
    for tract in tract_dirs:
        row = meta[meta["label"] == tract]
        if row.empty:
            continue
        row = row.iloc[0]
        segments = ["core"]
        e1 = _parse_segment(row.get("end1", ""))
        e2 = _parse_segment(row.get("end2", ""))
        if e1:
            segments.append(e1)
        if e2 and e2 not in segments:
            segments.append(e2)
        for seg in segments:
            if ta.segment_to_nodes(tract, seg) is not None:
                out.append((tract, seg))
    return out


def _load_gam_controls(gam_dir: Path, tract: str, scalar: str, stat: str) -> Optional[pd.DataFrame]:
    """Load GAM CSV for (tract, scalar), filter to controls; return DataFrame with sub and node*_z columns."""
    path = gam_dir / tract / f"{tract}_{scalar}_stat-{stat}_gam.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "group" not in df.columns:
        return None
    df = df.loc[df["group"].astype(str).str.strip() != CONTROL_GROUP_FILTER].copy()
    return df


def load_control_z_segment(
    gam_dir: Path,
    tract: str,
    segment: str,
    node_list: List[int],
    scalars: List[str],
    stat: str,
) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """Build (controls x scalars) matrix: for each scalar, mean of node_k_z over node_list. Returns (df, scalars) or None."""
    z_cols = [f"node{k}_z" for k in node_list]
    merged: Optional[pd.DataFrame] = None
    for scalar in scalars:
        df = _load_gam_controls(gam_dir, tract, scalar, stat)
        if df is None:
            return None
        missing = [c for c in z_cols if c not in df.columns]
        if missing:
            return None
        mean_z = df[z_cols].mean(axis=1)
        sub_df = df[["sub"]].copy()
        sub_df[f"{scalar}_z"] = mean_z
        if merged is None:
            merged = sub_df
        else:
            merged = merged.merge(sub_df[["sub", f"{scalar}_z"]], on="sub", how="inner")
    if merged is None or merged.empty:
        return None
    out_z = [f"{s}_z" for s in scalars]
    merged = merged.dropna(subset=out_z)
    if len(merged) < 2:
        return None
    return merged, scalars


def load_control_z_node(
    gam_dir: Path,
    tract: str,
    node: int,
    scalars: List[str],
    stat: str,
) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """Build (controls x scalars) matrix: for each scalar, node_k_z. Returns (df, scalars) or None."""
    col = f"node{node}_z"
    merged: Optional[pd.DataFrame] = None
    for scalar in scalars:
        df = _load_gam_controls(gam_dir, tract, scalar, stat)
        if df is None or col not in df.columns:
            return None
        sub_df = df[["sub", col]].copy()
        sub_df = sub_df.rename(columns={col: f"{scalar}_z"})
        if merged is None:
            merged = sub_df
        else:
            merged = merged.merge(sub_df, on="sub", how="inner")
    if merged is None or merged.empty:
        return None
    out_z = [f"{s}_z" for s in scalars]
    merged = merged.dropna(subset=out_z)
    if len(merged) < 2:
        return None
    return merged, scalars


def _save_cov_invcov(
    cov: np.ndarray,
    scalars: List[str],
    output_dir: Path,
    cov_fname: str,
    invcov_fname: str,
) -> None:
    """Save covariance and inverse covariance DataFrames (scalar x scalar)."""
    cov_df = pd.DataFrame(cov, index=scalars, columns=scalars)
    cov_df.to_csv(output_dir / cov_fname)
    try:
        invcov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        invcov = np.full_like(cov, np.nan)
    invcov_df = pd.DataFrame(invcov, index=scalars, columns=scalars)
    invcov_df.to_csv(output_dir / invcov_fname)


def _compute_and_save_covariance(
    df: pd.DataFrame,
    scalars: List[str],
    output_dir: Path,
) -> bool:
    """From (controls x scalars) df, compute sample cov + invcov and optionally MinCovDet; save CSVs. Returns True."""
    z_cols = [f"{s}_z" for s in scalars]
    X = df[z_cols].values
    n, p = X.shape
    if n < 2 or p == 0:
        return False
    output_dir.mkdir(parents=True, exist_ok=True)

    cov = np.cov(X, rowvar=False)
    if cov.shape != (p, p):
        return False
    _save_cov_invcov(
        cov, scalars, output_dir,
        OUTPUT_COV_FNAME, OUTPUT_INVCOV_FNAME,
    )

    if MinCovDet is not None:
        try:
            mcd = MinCovDet(store_precision=True, random_state=0).fit(X)
            cov_mcd = mcd.covariance_
            invcov_mcd = mcd.precision_
            if cov_mcd.shape == (p, p) and invcov_mcd.shape == (p, p):
                pd.DataFrame(cov_mcd, index=scalars, columns=scalars).to_csv(
                    output_dir / OUTPUT_COV_MINCOVDET_FNAME
                )
                pd.DataFrame(invcov_mcd, index=scalars, columns=scalars).to_csv(
                    output_dir / OUTPUT_INVCOV_MINCOVDET_FNAME
                )
            else:
                nan_cov = np.full((p, p), np.nan)
                _save_cov_invcov(
                    nan_cov, scalars, output_dir,
                    OUTPUT_COV_MINCOVDET_FNAME, OUTPUT_INVCOV_MINCOVDET_FNAME,
                )
        except Exception:
            nan_cov = np.full((p, p), np.nan)
            _save_cov_invcov(
                nan_cov, scalars, output_dir,
                OUTPUT_COV_MINCOVDET_FNAME, OUTPUT_INVCOV_MINCOVDET_FNAME,
            )
    else:
        nan_cov = np.full((p, p), np.nan)
        _save_cov_invcov(
            nan_cov, scalars, output_dir,
            OUTPUT_COV_MINCOVDET_FNAME, OUTPUT_INVCOV_MINCOVDET_FNAME,
        )

    return True


def run_segment_level(
    gam_dir: Path,
    output_dir: Path,
    ta: TractAsymmetry,
    stat: str,
    tracts_filter: Optional[List[str]],
) -> int:
    """Compute and save segment-level covariances. Returns count of (tract, segment) units saved."""
    scalars = get_scalars(gam_dir, stat)
    if not scalars:
        return 0
    pairs = get_all_tract_segment_pairs(ta, gam_dir)
    if tracts_filter is not None:
        pairs = [(t, s) for t, s in pairs if t in tracts_filter]
    saved = 0
    for tract, segment in pairs:
        node_list = ta.segment_to_nodes(tract, segment)
        if node_list is None:
            continue
        result = load_control_z_segment(gam_dir, tract, segment, node_list, scalars, stat)
        if result is None:
            continue
        df, _ = result
        out_path = output_dir / "segment_level" / tract / segment
        if _compute_and_save_covariance(df, scalars, out_path):
            saved += 1
            print(f"  segment_level/{tract}/{segment}", file=sys.stderr)
    return saved


def run_node_level(
    gam_dir: Path,
    output_dir: Path,
    stat: str,
    tracts_filter: Optional[List[str]],
) -> int:
    """Compute and save node-level covariances (node_001 .. node_100). Returns count of (tract, node) units saved."""
    scalars = get_scalars(gam_dir, stat)
    if not scalars:
        return 0
    tract_dirs = sorted(d.name for d in gam_dir.iterdir() if d.is_dir())
    if tracts_filter is not None:
        tract_dirs = [t for t in tract_dirs if t in tracts_filter]
    saved = 0
    for tract in tract_dirs:
        tract_saved = 0
        for node in range(1, 101):
            result = load_control_z_node(gam_dir, tract, node, scalars, stat)
            if result is None:
                continue
            df, _ = result
            node_label = f"node_{node:03d}"
            out_path = output_dir / "node_level" / tract / node_label
            if _compute_and_save_covariance(df, scalars, out_path):
                saved += 1
                tract_saved += 1
        if tract_saved > 0:
            print(f"  node_level/{tract} ({tract_saved} nodes)", file=sys.stderr)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute tract-level normative (scalar x scalar) covariance at segment and/or node level (controls = group != penn_epilepsy)."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=project_root(),
        help="Project base directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root; default: <base-dir>/derivatives/analysis/tract_asymmetry_normative.",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["segment", "node", "both"],
        default="both",
        help="Compute segment-level, node-level, or both (default: both).",
    )
    parser.add_argument(
        "--stat",
        type=str,
        default=DEFAULT_STAT,
        help="GAM stat: mean or standard_deviation (default: mean).",
    )
    parser.add_argument(
        "--tracts",
        type=str,
        nargs="*",
        default=None,
        metavar="TRACT",
        help="Restrict to these tract labels (e.g. ILF_L ILF_R); default: all tracts in GAM dir.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    paths = get_paths(base_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    gam_dir = paths["gam_dir"]
    if not gam_dir.is_dir():
        print(f"GAM dir missing: {gam_dir}", file=sys.stderr)
        return 1

    ta = TractAsymmetry(
        base_dir=base_dir,
        metadata_path=paths["metadata_path"],
        gam_dir=gam_dir,
        gam_stat=args.stat,
    )
    ta.load_metadata()

    total_saved = 0
    if args.level in ("segment", "both"):
        total_saved += run_segment_level(
            gam_dir, output_dir, ta, args.stat, args.tracts
        )
    if args.level in ("node", "both"):
        total_saved += run_node_level(
            gam_dir, output_dir, args.stat, args.tracts
        )

    print(f"Saved {total_saved} covariance sets under {output_dir}", file=sys.stderr)
    if MinCovDet is None:
        print("MinCovDet skipped (sklearn.covariance not available); *_mincovdet.csv contain NaN.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
