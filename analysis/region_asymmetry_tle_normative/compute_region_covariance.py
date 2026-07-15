#!/usr/bin/env python3
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
Compute and save region-wise (scalar x scalar) covariance matrices from healthy control subjects.
Uses same GAM mni_micro paths and config as region_asymmetry_tle; controls = group != "penn_epilepsy".
Outputs under derivatives/analysis/region_asymmetry_tle_normative/{atlas}/{region}/:
  - cov.csv, invcov.csv (sample covariance and inverse)
  - cov_mincovdet.csv, invcov_mincovdet.csv (MinCovDet robust covariance and inverse)
for downstream Mahalanobis distance in patients.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet

# Reuse config and helpers from region_asymmetry_tle
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_asymmetry_tle import config as cfg
from region_asymmetry_tle.core import get_scalars_for_atlas

CONTROL_GROUP_FILTER = "penn_epilepsy"  # exclude this group; all others are controls
DEFAULT_STAT = "mean"
OUTPUT_COV_FNAME = "cov.csv"
OUTPUT_INVCOV_FNAME = "invcov.csv"
OUTPUT_COV_MINCOVDET_FNAME = "cov_mincovdet.csv"
OUTPUT_INVCOV_MINCOVDET_FNAME = "invcov_mincovdet.csv"


def get_paths(base_dir: Path) -> Dict[str, Path]:
    """Same paths as region_asymmetry_tle; output_dir points to region_asymmetry_tle_normative."""
    base = Path(base_dir)
    paths = cfg.get_paths(base)
    paths["output_dir"] = base / "derivatives" / "analysis" / "region_asymmetry_tle_normative"
    return paths


def get_regions_per_atlas(paths: Dict[str, Path]) -> Dict[str, List[str]]:
    """Return {atlas: [region, ...]} for Glasser, 4S156, HCP1065 (region = directory name in GAM dir)."""
    out: Dict[str, List[str]] = {}
    if paths["gam_glasser"].is_dir():
        out["Glasser"] = sorted(d.name for d in paths["gam_glasser"].iterdir() if d.is_dir())
    else:
        out["Glasser"] = []
    if paths["gam_4s156"].is_dir():
        out["4S156"] = sorted(d.name for d in paths["gam_4s156"].iterdir() if d.is_dir())
    else:
        out["4S156"] = []
    if paths.get("gam_hcp1065") and paths["gam_hcp1065"].is_dir():
        out["HCP1065"] = sorted(d.name for d in paths["gam_hcp1065"].iterdir() if d.is_dir())
    else:
        out["HCP1065"] = []
    return out


def get_gam_dir(paths: Dict[str, Path], atlas: str) -> Optional[Path]:
    if atlas == "Glasser":
        return paths["gam_glasser"] if paths["gam_glasser"].is_dir() else None
    if atlas == "4S156":
        return paths["gam_4s156"] if paths["gam_4s156"].is_dir() else None
    if atlas == "HCP1065":
        return paths.get("gam_hcp1065") if paths.get("gam_hcp1065") and paths["gam_hcp1065"].is_dir() else None
    return None


def load_control_z_matrix(
    gam_dir: Path,
    region: str,
    stat: str,
) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """
    Load GAM z-scores for all scalars in region, filter to controls (group != penn_epilepsy).
    Returns (dataframe with columns sub, scalar1_z, scalar2_z, ..., scalar_list) or None.
    Rows with any NaN in scalar columns are dropped.
    """
    scalars = get_scalars_for_atlas(gam_dir, region, stat)
    if not scalars:
        return None

    z_cols = [f"{s}_z" for s in scalars]
    merged: Optional[pd.DataFrame] = None

    for scalar in scalars:
        path = gam_dir / region / f"{region}_{scalar}_stat-{stat}_gam.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "group" not in df.columns:
            return None
        # Controls = not epilepsy
        df = df.loc[df["group"].astype(str).str.strip() != CONTROL_GROUP_FILTER].copy()
        df = df[["sub", f"{scalar}_z"]].copy()
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="sub", how="inner")

    if merged is None or merged.empty:
        return None
    merged = merged.dropna(subset=z_cols)
    if len(merged) < 2:
        return None
    return merged, scalars


def _save_cov_invcov(
    cov: np.ndarray,
    scalars: List[str],
    output_region_dir: Path,
    cov_fname: str,
    invcov_fname: str,
) -> None:
    """Save covariance and inverse covariance DataFrames (scalar x scalar)."""
    cov_df = pd.DataFrame(cov, index=scalars, columns=scalars)
    cov_df.to_csv(output_region_dir / cov_fname)
    try:
        invcov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        invcov = np.full_like(cov, np.nan)
    invcov_df = pd.DataFrame(invcov, index=scalars, columns=scalars)
    invcov_df.to_csv(output_region_dir / invcov_fname)


def compute_and_save_region_covariance(
    gam_dir: Path,
    region: str,
    output_region_dir: Path,
    stat: str = DEFAULT_STAT,
) -> bool:
    """Build (scalar x scalar) cov and invcov (sample + MinCovDet); save to CSV. Returns True if saved."""
    result = load_control_z_matrix(gam_dir, region, stat)
    if result is None:
        return False
    df, scalars = result
    z_cols = [f"{s}_z" for s in scalars]
    X = df[z_cols].values
    n, p = X.shape
    if n < 2 or p == 0:
        return False

    output_region_dir.mkdir(parents=True, exist_ok=True)

    # Sample (empirical) covariance and inverse
    cov = np.cov(X, rowvar=False)
    if cov.shape != (p, p):
        return False
    _save_cov_invcov(
        cov, scalars, output_region_dir,
        OUTPUT_COV_FNAME, OUTPUT_INVCOV_FNAME,
    )

    # MinCovDet robust covariance and inverse (labeled separately)
    try:
        mcd = MinCovDet(store_precision=True, random_state=0).fit(X)
        cov_mcd = mcd.covariance_
        invcov_mcd = mcd.precision_
        if cov_mcd.shape == (p, p) and invcov_mcd.shape == (p, p):
            pd.DataFrame(cov_mcd, index=scalars, columns=scalars).to_csv(
                output_region_dir / OUTPUT_COV_MINCOVDET_FNAME
            )
            pd.DataFrame(invcov_mcd, index=scalars, columns=scalars).to_csv(
                output_region_dir / OUTPUT_INVCOV_MINCOVDET_FNAME
            )
        else:
            nan_cov = np.full((p, p), np.nan)
            _save_cov_invcov(
                nan_cov, scalars, output_region_dir,
                OUTPUT_COV_MINCOVDET_FNAME, OUTPUT_INVCOV_MINCOVDET_FNAME,
            )
    except Exception:
        # MinCovDet can fail if n < p or degenerate data; write NaN matrices
        nan_cov = np.full((p, p), np.nan)
        _save_cov_invcov(
            nan_cov, scalars, output_region_dir,
            OUTPUT_COV_MINCOVDET_FNAME, OUTPUT_INVCOV_MINCOVDET_FNAME,
        )

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute region-wise (scalar x scalar) covariance matrices from control GAM z-scores (group != penn_epilepsy)."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("{project_root()}"),
        help="Project base directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root; default: <base-dir>/derivatives/analysis/region_asymmetry_tle_normative.",
    )
    parser.add_argument(
        "--atlas",
        type=str,
        choices=["Glasser", "4S156", "HCP1065"],
        default=None,
        help="Run only this atlas; default: all three.",
    )
    parser.add_argument(
        "--stat",
        type=str,
        default=DEFAULT_STAT,
        help="GAM stat (mean or standard_deviation); default: mean.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    paths = get_paths(base_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    atlases = [args.atlas] if args.atlas else ["Glasser", "4S156", "HCP1065"]
    regions_per_atlas = get_regions_per_atlas(paths)

    total_saved = 0
    for atlas in atlases:
        gam_dir = get_gam_dir(paths, atlas)
        if gam_dir is None:
            print(f"Skipping {atlas}: GAM dir missing.", file=sys.stderr)
            continue
        regions = regions_per_atlas.get(atlas, [])
        atlas_out = output_dir / atlas
        for region in regions:
            region_out = atlas_out / region
            if compute_and_save_region_covariance(gam_dir, region, region_out, stat=args.stat):
                total_saved += 1
                print(f"  {atlas}/{region}: cov.csv, invcov.csv, cov_mincovdet.csv, invcov_mincovdet.csv", file=sys.stderr)
    print(f"Saved {total_saved} region covariance sets under {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
