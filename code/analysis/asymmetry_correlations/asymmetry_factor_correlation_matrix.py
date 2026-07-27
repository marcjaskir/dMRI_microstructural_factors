import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
"""
Per-factor correlation matrices of normalized factor-score asymmetry across TLE regions.

For each factor Fk (default F1–F4), loads wide factor score CSVs, z-scores epilepsy vs controls
per ROI, then per subject and region computes
``(ipsi_z − contra_z) / (|ipsi_z| + |contra_z|)`` with ipsi/contra from inclusion laterality
(same cohort rules as factor_z asymmetry). Region list matches asymmetry_tle_region discovery
order (same as Mahalanobis correlation script).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

_ATLR_DIR = _DIR.parent / "asymmetry_tle_region"
if str(_ATLR_DIR) not in sys.path:
    sys.path.insert(0, str(_ATLR_DIR))

_FZ_DIR = _DIR.parent / "microstructural_asymmetries"
if str(_FZ_DIR) not in sys.path:
    sys.path.insert(0, str(_FZ_DIR))

import asymmetry_mahalanobis_correlation_matrix as maha  # noqa: E402
import asymmetry_tle_region as atlr  # noqa: E402
import config as cfg  # noqa: E402
import microstructural_asymmetry_report_factor_z as fz  # noqa: E402

DEFAULT_PROJECT_ROOT = project_root()
def _set_matplotlib_georgia() -> None:
    """Use Georgia for all figures produced by this script (heatmap, scatter, colorbar)."""
    try:
        import matplotlib

        matplotlib.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Georgia", "DejaVu Serif"],
            }
        )
    except ImportError:
        pass


def load_laterality_map_normalized(inclusion_path: Path) -> Dict[str, str]:
    """sub (normalized) -> 'left' | 'right' for temporal lobe rows only."""
    out: Dict[str, str] = {}
    if not inclusion_path.is_file():
        return out
    try:
        df = pd.read_csv(inclusion_path)
        if "sub" not in df.columns or "laterality" not in df.columns:
            return out
        if "lobe" in df.columns:
            df = df[df["lobe"].astype(str).str.strip().str.lower() == "temporal"]
        for _, row in df.iterrows():
            sub = fz.normalize_subject_id(row["sub"])
            lat = str(row.get("laterality", "")).strip().lower()
            if lat in ("left", "right"):
                out[str(sub)] = lat
    except Exception:
        pass
    return out


def per_subject_factor_normalized_asymmetry(
    zdf: pd.DataFrame,
    l_col: str,
    r_col: str,
    lat_map: Dict[str, str],
    temporal_set: Set[str],
) -> pd.Series:
    """
    Per subject: (ipsi_z − contra_z) / (|ipsi_z| + |contra_z|); NaN if denominator 0
    or missing laterality / outside temporal cohort.
    """
    if zdf.empty or l_col not in zdf.columns or r_col not in zdf.columns:
        return pd.Series(dtype=float)
    subs = zdf["subject"].map(fz.normalize_subject_id)
    v_l = pd.to_numeric(zdf[l_col], errors="coerce").to_numpy(dtype=float)
    v_r = pd.to_numeric(zdf[r_col], errors="coerce").to_numpy(dtype=float)
    lat = subs.map(lambda s: lat_map.get(s, ""))
    ipsi = np.full(len(zdf), np.nan, dtype=float)
    contra = np.full(len(zdf), np.nan, dtype=float)
    left = (lat == "left").to_numpy()
    right = (lat == "right").to_numpy()
    ipsi[left] = v_l[left]
    contra[left] = v_r[left]
    ipsi[right] = v_r[right]
    contra[right] = v_l[right]
    den = np.abs(ipsi) + np.abs(contra)
    num = ipsi - contra
    out = num / den
    out = np.where(np.isfinite(den) & (den > 0), out, np.nan)
    temporal_ok = subs.isin(temporal_set).to_numpy()
    out = np.where(temporal_ok, out, np.nan)
    valid_lat = left | right
    out = np.where(valid_lat, out, np.nan)
    s = pd.Series(out, index=subs.astype(str))
    s = s[~s.index.duplicated(keep="first")]
    return s


def factor_scatter_output_stem(
    factor_id: int,
    x_atlas: str,
    x_slug: str,
    y_atlas: str,
    y_slug: str,
    method: str,
    output_suffix: str = "",
) -> str:
    def safe(part: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in part)

    return (
        f"factor_F{factor_id}_asymmetry_scatter_{safe(x_atlas)}-{safe(x_slug)}_vs_"
        f"{safe(y_atlas)}-{safe(y_slug)}_{method}{output_suffix}"
    )


def build_wide_factor_table(
    paths: Dict[str, Path],
    region_list: List[Tuple[str, str]],
    factor_id: int,
    temporal_set: Set[str],
    lat_map: Dict[str, str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Columns = internal atlas:slug; rows = sub; values = normalized factor asymmetry."""
    factor_dir = paths.get("factor_scores_dir")
    skipped: List[str] = []
    if factor_dir is None or not Path(factor_dir).is_dir():
        skipped.append("factor_scores_dir missing")
        for msg in skipped:
            print(f"Skip: {msg}", file=sys.stderr)
        return pd.DataFrame(), skipped

    ctrl_p = Path(factor_dir) / f"controls_F{factor_id}_scores.csv"
    epi_p = Path(factor_dir) / f"epilepsy_F{factor_id}_scores.csv"
    if not ctrl_p.is_file() or not epi_p.is_file():
        skipped.append(f"F{factor_id}: missing controls or epilepsy CSV")
        for msg in skipped:
            print(f"Skip: {msg}", file=sys.stderr)
        return pd.DataFrame(), skipped

    try:
        ctrl = pd.read_csv(ctrl_p)
        epi = pd.read_csv(epi_p)
    except Exception as exc:
        skipped.append(f"F{factor_id}: could not read CSVs ({exc})")
        for msg in skipped:
            print(f"Skip: {msg}", file=sys.stderr)
        return pd.DataFrame(), skipped

    roi_union: List[str] = []
    seen: Set[str] = set()
    resolved_regions: List[Tuple[str, str, str, str]] = []
    for atlas, slug in region_list:
        key = maha.col_key(atlas, slug)
        lr = maha.resolve_labels_from_slug(atlas, slug, paths)
        if lr is None:
            skipped.append(f"{key} (could not resolve L/R columns)")
            continue
        l_col, r_col = lr
        if l_col not in epi.columns or r_col not in epi.columns:
            skipped.append(f"{key} (columns missing in epilepsy_F{factor_id})")
            continue
        if l_col not in ctrl.columns or r_col not in ctrl.columns:
            skipped.append(f"{key} (columns missing in controls_F{factor_id})")
            continue
        resolved_regions.append((atlas, slug, l_col, r_col))
        for c in (l_col, r_col):
            if c not in seen:
                seen.add(c)
                roi_union.append(c)

    if not roi_union:
        skipped.append(f"F{factor_id}: no regions with valid factor columns")
        for msg in skipped:
            print(f"Skip: {msg}", file=sys.stderr)
        return pd.DataFrame(), skipped

    try:
        zdf = fz.zscore_epilepsy_vs_controls(ctrl, epi, roi_union)
    except Exception as exc:
        skipped.append(f"F{factor_id}: z-score failed ({exc})")
        for msg in skipped:
            print(f"Skip: {msg}", file=sys.stderr)
        return pd.DataFrame(), skipped

    cols: Dict[str, pd.Series] = {}
    for atlas, slug, l_col, r_col in resolved_regions:
        key = maha.col_key(atlas, slug)
        series = per_subject_factor_normalized_asymmetry(
            zdf, l_col, r_col, lat_map, temporal_set
        )
        if series.empty or series.notna().sum() == 0:
            skipped.append(f"{key} (no overlapping asymmetry values)")
            continue
        cols[key] = series

    if not cols:
        for msg in skipped:
            print(f"Skip: {msg}", file=sys.stderr)
        return pd.DataFrame(), skipped

    for msg in skipped:
        print(f"Skip: {msg}", file=sys.stderr)

    wide = pd.DataFrame(cols)
    order_keys = [maha.col_key(a, s) for a, s in region_list if maha.col_key(a, s) in wide.columns]
    wide = wide[[c for c in order_keys if c in wide.columns]]
    wide.sort_index(inplace=True)
    return wide, skipped


def run_one_factor(
    factor_id: int,
    wide_display: pd.DataFrame,
    method: str,
    min_periods: int,
    output_dir: Path,
    output_suffix: str,
    scatter_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]],
    save_diff: bool,
    dpi: int,
) -> bool:
    kw: Dict[str, object] = {"min_periods": min_periods}
    if method == "pearson":
        corr = wide_display.corr(method="pearson", **kw)
    elif method == "spearman":
        corr = wide_display.corr(method="spearman", **kw)
    else:
        print(f"Unknown method: {method}", file=sys.stderr)
        return False

    stem = f"factor_F{factor_id}_asymmetry_corr_{method}{output_suffix}"
    corr_path = output_dir / f"{stem}.csv"
    corr.to_csv(corr_path, index_label="region")
    print(f"Wrote {corr_path}")

    png_path = output_dir / f"{stem}.png"
    maha.plot_correlation_heatmap(corr, png_path, dpi=dpi)
    if png_path.is_file():
        print(f"Wrote {png_path}")

    for (xa, xs), (ya, ys) in scatter_pairs:
        scatter_stem = factor_scatter_output_stem(
            factor_id, xa, xs, ya, ys, method, output_suffix
        )
        scatter_path = output_dir / f"{scatter_stem}.png"
        maha.plot_region_pair_scatter(
            wide_display, xa, xs, ya, ys, scatter_path, method, dpi=dpi
        )
        if scatter_path.is_file():
            print(f"Wrote {scatter_path}")

    if save_diff:
        diff_path = (
            output_dir
            / f"factor_F{factor_id}_asymmetry_per_subject_wide{output_suffix}.csv"
        )
        wide_display.to_csv(diff_path, index_label="sub")
        print(f"Wrote {diff_path}")

    return True


def run(
    base_dir: Path,
    output_dir: Path,
    method: str,
    min_periods: int,
    save_diff: bool,
    dpi: int,
    scatter_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]],
    mts_only: bool,
    factor_ids: List[int],
) -> int:
    _set_matplotlib_georgia()
    base_dir = Path(base_dir).resolve()
    output_dir = Path(output_dir).resolve()
    paths = cfg.get_paths(base_dir)
    asym_root = paths["output_dir"]

    discovered = maha.discover_region_slugs(asym_root)
    if not discovered:
        print(f"No region folders found under {asym_root}", file=sys.stderr)
        return 1

    region_list = maha.order_region_pairs(discovered)

    inclusion_path = paths["inclusion_metadata"]
    temporal_list = atlr.load_temporal_subjects(inclusion_path)
    temporal_set = {fz.normalize_subject_id(s) for s in temporal_list}
    lat_map = load_laterality_map_normalized(inclusion_path)

    output_suffix = "_mts" if mts_only else ""
    output_dir.mkdir(parents=True, exist_ok=True)

    mts_subs: Optional[Set[str]] = None
    if mts_only:
        mts_set, err = maha.load_mts_positive_subjects(paths["inclusion_metadata"])
        if err:
            print(err, file=sys.stderr)
            return 1
        if not mts_set:
            print(
                "MTS filter: no subjects flagged in mts/lesion_mts.",
                file=sys.stderr,
            )
            return 1
        mts_subs = mts_set
        print(
            f"MTS-only: {len(mts_subs)} MTS+ subjects in inclusion metadata.",
            file=sys.stderr,
        )

    any_ok = False
    cbar_written = False

    for fk in factor_ids:
        wide, _sk = build_wide_factor_table(
            paths, region_list, fk, temporal_set, lat_map
        )
        if wide.empty:
            print(f"F{fk}: empty wide table; skipping.", file=sys.stderr)
            continue

        if mts_subs is not None:
            n_before = len(wide)
            wide = wide.loc[wide.index.isin(mts_subs)]
            if wide.empty:
                print(
                    f"F{fk}: MTS filter left no subjects; skipping.",
                    file=sys.stderr,
                )
                continue
            print(
                f"F{fk}: {len(wide)} subjects after MTS filter "
                f"({n_before - len(wide)} excluded).",
                file=sys.stderr,
            )

        wide_display = maha.wide_columns_to_display(wide)
        if run_one_factor(
            fk,
            wide_display,
            method,
            min_periods,
            output_dir,
            output_suffix,
            scatter_pairs,
            save_diff,
            dpi,
        ):
            any_ok = True

        if not cbar_written:
            cbar_path = output_dir / f"factor_asymmetry_corr_colorbar{output_suffix}.png"
            maha.plot_standalone_correlation_colorbar(cbar_path, dpi=dpi)
            if cbar_path.is_file():
                print(f"Wrote {cbar_path}")
                cbar_written = True

    if not any_ok:
        print("No factor correlation outputs written.", file=sys.stderr)
        return 1
    return 0


def parse_factors_arg(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            k = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid factor id {part!r}") from exc
        if k < 1:
            raise argparse.ArgumentTypeError(f"Factor id must be >= 1, got {k}")
        out.append(k)
    if not out:
        raise argparse.ArgumentTypeError("At least one factor id required")
    return out


def _default_output_dir(base_dir: Path) -> Path:
    return base_dir / "derivatives" / "analysis" / "asymmetry_correlations"


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Per-factor correlation matrices of normalized factor z asymmetry "
            "(ipsi−contra)/(|ipsi|+|contra|) across TLE regions."
        ),
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Project base (structural_tractometry root)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: derivatives/analysis/asymmetry_correlations)",
    )
    p.add_argument(
        "--method",
        choices=("pearson", "spearman"),
        default="pearson",
        help="Correlation method (default: pearson)",
    )
    p.add_argument(
        "--min-periods",
        type=int,
        default=3,
        help="Minimum pairwise observations for correlation (pandas corr min_periods)",
    )
    p.add_argument(
        "--save-diff-csv",
        action="store_true",
        help="Also save wide per-subject asymmetry table per factor",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for heatmap, scatter, and colorbar PNGs (default: 300)",
    )
    p.add_argument(
        "--factors",
        type=parse_factors_arg,
        default=parse_factors_arg("1,2,3,4"),
        help="Comma-separated factor indices (default: 1,2,3,4)",
    )
    p.add_argument(
        "--scatter-pair",
        action="append",
        default=None,
        metavar="X,Y",
        help=(
            "Scatter asymmetry: two atlas:slug specs separated by a comma "
            "(e.g. 4S156:Anterior,HCP1065:F_core). Repeat per pair. "
            "Default when omitted: that pair. Use --no-scatter to skip."
        ),
    )
    p.add_argument(
        "--no-scatter",
        action="store_true",
        help="Do not write region-pair scatter PNGs",
    )
    p.add_argument(
        "--mts-only",
        action="store_true",
        help=(
            "Restrict to MTS+ subjects (mts / lesion_mts in inclusion CSV). "
            "Filenames get _mts suffix."
        ),
    )
    args = p.parse_args()
    out = args.output_dir
    if out is None:
        out = _default_output_dir(Path(args.base_dir))

    try:
        if args.no_scatter:
            scatter_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
        elif args.scatter_pair:
            scatter_pairs = [maha.parse_scatter_pair_arg(s) for s in args.scatter_pair]
        else:
            scatter_pairs = [maha.parse_scatter_pair_arg("4S156:Anterior,HCP1065:F_core")]
    except argparse.ArgumentTypeError as e:
        p.error(str(e))

    return run(
        Path(args.base_dir),
        Path(out),
        args.method,
        args.min_periods,
        args.save_diff_csv,
        args.dpi,
        scatter_pairs,
        args.mts_only,
        args.factors,
    )


if __name__ == "__main__":
    sys.exit(main())
