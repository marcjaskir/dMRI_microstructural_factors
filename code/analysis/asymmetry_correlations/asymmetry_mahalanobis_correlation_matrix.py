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
Correlation matrix of per-subject Mahalanobis asymmetry across TLE regions.

Discovers (atlas, region slug) pairs from derivatives/analysis/asymmetry_tle_region,
loads bilateral Mahalanobis summary CSVs, and builds per-subject asymmetry as
``(ipsi − contra) / (|ipsi| + |contra|)`` (undefined when both sides are 0).

Saves Pearson/Spearman correlation matrix + heatmap, plus optional per-subject
asymmetry scatter plots (``--scatter-pair``; same axis font scale as the heatmap).

Pairwise correlations use pandas default (pairwise-complete observations per pair);
``--min-periods`` sets the minimum count required for each pair.

``--mts-only`` keeps subjects with MTS (``mts`` or ``lesion_mts`` in
``results/inclusion/penn_epilepsy_included_basic_metadata.csv``) and tags outputs
with ``_mts``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

_ATLR_DIR = Path(__file__).resolve().parent.parent / "asymmetry_tle_region"
if str(_ATLR_DIR) not in sys.path:
    sys.path.insert(0, str(_ATLR_DIR))

import config as cfg  # noqa: E402
import asymmetry_tle_region as atlr  # noqa: E402

DEFAULT_PROJECT_ROOT = project_root()
# Human-readable labels (slug → name). Ordering: PRIORITY_REGIONS; other regions follow sorted (atlas, slug).
SLUG_TO_DISPLAY_LABEL: Dict[str, str] = {
    "Hippocampus": "Hippocampus",
    "Amygdala": "Amygdala",
    "HTH": "Hypothalamus",
    "MN": "Mammillary bodies",
    "Anterior": "Anterior thalamus",
    "EC": "Entorhinal cortex",
    "OFC": "Orbitofrontal cortex",
    "F_core": "Fornix (core)",
    "C_PH_core": "Parahippocampal cingulum (core)",
    "UF_core": "Uncinate fasciculus (core)",
}

# Shared with heatmap axis ticks (and scatter axes) for consistent styling.
_AXIS_TICK_FONTSIZE = 8 * 1.66

PRIORITY_REGIONS: List[Tuple[str, str]] = [
    ("4S156", "Hippocampus"),
    ("4S156", "Amygdala"),
    ("4S156", "HTH"),
    ("4S156", "MN"),
    ("4S156", "Anterior"),
    ("Glasser", "EC"),
    ("Glasser", "OFC"),
    ("HCP1065", "F_core"),
    ("HCP1065", "C_PH_core"),
    ("HCP1065", "UF_core"),
]


def order_region_pairs(discovered: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Priority order first, then remaining (atlas, slug) lexicographically."""
    disc_set = set(discovered)
    ordered: List[Tuple[str, str]] = [p for p in PRIORITY_REGIONS if p in disc_set]
    rest = sorted(p for p in discovered if p not in ordered)
    ordered.extend(rest)
    return ordered


def display_label_for(atlas: str, slug: str) -> str:
    """Plot/CSV label without atlas prefix; unknown slugs use atlas:slug."""
    if slug in SLUG_TO_DISPLAY_LABEL:
        return SLUG_TO_DISPLAY_LABEL[slug]
    return f"{atlas}:{slug}"


def col_key(atlas: str, slug: str) -> str:
    """Internal wide-table column id."""
    return f"{atlas}:{slug}"


def parse_region_spec(spec: str) -> Tuple[str, str]:
    """Parse ``atlas:slug`` (first colon separates atlas from slug)."""
    s = spec.strip()
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"Region spec must be atlas:slug, got {spec!r}"
        )
    atlas, slug = s.split(":", 1)
    atlas, slug = atlas.strip(), slug.strip()
    if not atlas or not slug:
        raise argparse.ArgumentTypeError(f"Empty atlas or slug in {spec!r}")
    return atlas, slug


def parse_scatter_pair_arg(s: str) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """Parse ``x_atlas:x_slug,y_atlas:y_slug`` (x = horizontal axis, y = vertical)."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Scatter pair needs two comma-separated atlas:slug specs, got {s!r}"
        )
    return parse_region_spec(parts[0]), parse_region_spec(parts[1])


def scatter_output_stem(
    x_atlas: str,
    x_slug: str,
    y_atlas: str,
    y_slug: str,
    method: str,
    output_suffix: str = "",
) -> str:
    """Filename stem for a region-pair scatter (filesystem-safe)."""

    def safe(part: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in part)

    return (
        f"mahalanobis_asymmetry_scatter_{safe(x_atlas)}-{safe(x_slug)}_vs_"
        f"{safe(y_atlas)}-{safe(y_slug)}_{method}{output_suffix}"
    )


def _truthy_mts_indicator(val: object) -> bool:
    """True if inclusion-table value counts as MTS-positive."""
    if val is None:
        return False
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val) != 0
    if isinstance(val, float):
        if np.isnan(val):
            return False
        return float(val) != 0.0
    if isinstance(val, str):
        s = val.strip().lower()
        return s in ("1", "true", "yes", "y")
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "y")


def load_mts_positive_subjects(inclusion_csv: Path) -> Tuple[Optional[Set[str]], Optional[str]]:
    """
    Subject IDs with MTS from inclusion metadata (``mts`` or ``lesion_mts``, same
    as microstructural_asymmetry_report_mahalanobis).
    """
    p = Path(inclusion_csv)
    if not p.is_file():
        return None, f"MTS filter: inclusion CSV not found: {p}"
    try:
        df = pd.read_csv(p)
    except Exception as exc:
        return None, f"MTS filter: could not read {p}: {exc}"
    if "sub" not in df.columns:
        return None, "MTS filter: inclusion CSV has no 'sub' column"
    col = None
    for c in ("mts", "lesion_mts"):
        if c in df.columns:
            col = c
            break
    if col is None:
        return None, "MTS filter: inclusion CSV has neither 'mts' nor 'lesion_mts'"
    subs: Set[str] = set()
    for _, row in df.iterrows():
        if not _truthy_mts_indicator(row[col]):
            continue
        sub = row["sub"]
        if pd.isna(sub):
            continue
        subs.add(str(sub))
    return subs, None


def wide_columns_to_display(wide: pd.DataFrame) -> pd.DataFrame:
    """Rename internal atlas:slug columns to human-readable labels."""
    mapping: Dict[str, str] = {}
    for c in wide.columns:
        parts = str(c).split(":", 1)
        if len(parts) != 2:
            mapping[c] = str(c)
            continue
        atlas, slug = parts[0], parts[1]
        mapping[c] = display_label_for(atlas, slug)
    return wide.rename(columns=mapping)


# Bilateral long CSVs (per plan)
BILATERAL_CSV_BY_ATLAS = {
    "4S156": "summary_4s_subcortex_mahalanobis_bilateral.csv",
    "Glasser": "summary_glasser_mahalanobis_bilateral.csv",
    "HCP1065": "summary_hcp1065_thirds_mahalanobis_bilateral.csv",
}


def _maha_report_dir(base_dir: Path) -> Path:
    return base_dir / "derivatives" / "analysis" / "microstructural_asymmetries"


def _default_output_dir(base_dir: Path) -> Path:
    return base_dir / "derivatives" / "analysis" / "asymmetry_correlations"


def discover_region_slugs(asymmetry_root: Path) -> List[Tuple[str, str]]:
    """(atlas, slug) for each subdirectory under asymmetry_tle_region that has figures/."""
    out: List[Tuple[str, str]] = []
    for atlas in ("4S156", "Glasser", "HCP1065"):
        atlas_dir = asymmetry_root / atlas
        if not atlas_dir.is_dir():
            continue
        for p in sorted(atlas_dir.iterdir()):
            if not p.is_dir() or p.name.startswith("."):
                continue
            if (p / "figures").is_dir():
                out.append((atlas, p.name))
    return out


def resolve_labels_from_slug(
    atlas: str,
    slug: str,
    paths: Dict[str, Path],
) -> Optional[Tuple[str, str]]:
    """Map asymmetry_tle_region folder slug to (left_label, right_label) in bilateral CSVs."""
    if atlas in ("4S156", "Glasser"):
        return atlr.resolve_factor_score_columns(atlas, slug, None, paths)
    if atlas == "HCP1065":
        base, _, tail = slug.rpartition("_")
        if not base:
            return None
        if tail == "core":
            tract, seg_spec = base, "core"
        elif tail.startswith("end_"):
            seg_spec = tail[4:].lower()
            tract = base
        else:
            tract, seg_spec = base, tail.lower()
        return atlr.resolve_factor_score_columns("HCP1065", tract, seg_spec, paths)
    return None


def per_subject_mahalanobis_asymmetry(
    bilateral_df: pd.DataFrame,
    l_label: str,
    r_label: str,
) -> pd.Series:
    """Per subject: (ipsi − contra) / (|ipsi| + |contra|) on Mahalanobis; NaN if denominator is 0."""
    need = bilateral_df[bilateral_df["label"].isin([l_label, r_label])].copy()
    if need.empty:
        return pd.Series(dtype=float)
    pv = need.pivot_table(
        index="sub",
        columns="label",
        values="mahalanobis",
        aggfunc="first",
    )
    if l_label not in pv.columns or r_label not in pv.columns:
        return pd.Series(dtype=float)
    grp = need.drop_duplicates(subset=["sub"]).set_index("sub")["group"]
    v_l = pv[l_label].astype(float)
    v_r = pv[r_label].astype(float)
    g = grp.reindex(pv.index)
    ipsi = pd.Series(np.nan, index=pv.index, dtype=float)
    contra = pd.Series(np.nan, index=pv.index, dtype=float)
    left_tle = g == "left_TLE"
    right_tle = g == "right_TLE"
    ipsi.loc[left_tle] = v_l.loc[left_tle]
    contra.loc[left_tle] = v_r.loc[left_tle]
    ipsi.loc[right_tle] = v_r.loc[right_tle]
    contra.loc[right_tle] = v_l.loc[right_tle]
    den = ipsi.abs() + contra.abs()
    num = ipsi - contra
    out = num / den
    out = out.where(den > 0, np.nan)
    return out


def load_bilateral_tables(
    base_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Load each atlas bilateral CSV once."""
    maha_dir = _maha_report_dir(base_dir)
    tables: Dict[str, pd.DataFrame] = {}
    for atlas, fname in BILATERAL_CSV_BY_ATLAS.items():
        p = maha_dir / fname
        if not p.is_file():
            print(f"Warning: missing {p}", file=sys.stderr)
            continue
        tables[atlas] = pd.read_csv(p)
    return tables


def build_wide_diff_table(
    base_dir: Path,
    paths: Dict[str, Path],
    region_list: List[Tuple[str, str]],
    bilateral_tables: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, List[str]]:
    """Columns = internal 'atlas:slug' keys; rows = sub; values = asymmetry formula on Mahalanobis."""
    cols: Dict[str, pd.Series] = {}
    skipped: List[str] = []
    for atlas, slug in region_list:
        lr = resolve_labels_from_slug(atlas, slug, paths)
        key = col_key(atlas, slug)
        if lr is None:
            skipped.append(f"{key} (could not resolve labels)")
            continue
        l_label, r_label = lr
        df = bilateral_tables.get(atlas)
        if df is None or df.empty:
            skipped.append(f"{key} (no bilateral table)")
            continue
        series = per_subject_mahalanobis_asymmetry(df, l_label, r_label)
        if series.empty or series.notna().sum() == 0:
            skipped.append(f"{key} (no overlapping data for {l_label} / {r_label})")
            continue
        cols[key] = series
    if skipped:
        for msg in skipped:
            print(f"Skip: {msg}", file=sys.stderr)
    if not cols:
        return pd.DataFrame(), skipped
    wide = pd.DataFrame(cols)
    # Column order follows region_list (caller passes ordered pairs)
    wide = wide[[c for c in [col_key(a, s) for a, s in region_list] if c in wide.columns]]
    wide.sort_index(inplace=True)
    return wide, skipped


def _annotate_lower_triangle(ax, corr_values: np.ndarray, n: int) -> None:
    """Text in strict lower triangle (j < i); diagonal omitted. imshow origin top."""
    for i in range(n):
        for j in range(n):
            if j >= i:
                continue
            val = corr_values[i, j]
            if not np.isfinite(val):
                continue
            t = f"{val:.2f}"
            # Contrast: dark text on light mid, light text on dark reds/blues
            if abs(val) < 0.35:
                color = "0.1"
            else:
                color = "0.98"
            ax.text(j, i, t, ha="center", va="center", fontsize=12, color=color)


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Heatmap only (no colorbar); vmin/vmax = -1, 1; strict lower-triangle r labels (no diagonal)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure", file=sys.stderr)
        return
    n = len(corr.columns)
    fig_w = max(8.0, 0.38 * n + 2)
    fig_h = max(7.0, 0.38 * n + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    arr = corr.values.astype(float)
    ax.imshow(arr, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
    _annotate_lower_triangle(ax, arr, n)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    labels = [str(c) for c in corr.columns]
    ax.set_xticklabels(
        labels, rotation=45, ha="right", fontsize=_AXIS_TICK_FONTSIZE
    )
    ax.set_yticklabels(labels, fontsize=_AXIS_TICK_FONTSIZE)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# Standalone correlation colorbar: tick label size (points).
_COLORBAR_TICK_FONTSIZE = 22.0


def plot_standalone_correlation_colorbar(
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Standalone vertical colorbar for r in [-1, 1]; large ticks, no title/label."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        print("matplotlib not available; skipping colorbar figure", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(2.0, 9.0))
    sm = cm.ScalarMappable(norm=plt.Normalize(-1.0, 1.0), cmap="RdBu_r")
    sm.set_array([])
    fig.colorbar(sm, cax=ax, orientation="vertical", fraction=1.0)
    ax.tick_params(
        axis="y",
        labelsize=_COLORBAR_TICK_FONTSIZE,
        length=12,
        width=1.5,
    )
    ax.tick_params(axis="x", which="both", length=0, labelsize=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_region_pair_scatter(
    wide_display: pd.DataFrame,
    x_atlas: str,
    x_slug: str,
    y_atlas: str,
    y_slug: str,
    output_path: Path,
    method: str,
    dpi: int = 300,
) -> None:
    """Scatter of Mahalanobis asymmetry (ipsi−contra)/(|ipsi|+|contra|) for two regions."""
    x_lab = display_label_for(x_atlas, x_slug)
    y_lab = display_label_for(y_atlas, y_slug)
    if x_lab not in wide_display.columns or y_lab not in wide_display.columns:
        print(
            f"Skip scatter {x_atlas}:{x_slug} vs {y_atlas}:{y_slug}: "
            f"need columns {x_lab!r} and {y_lab!r} in wide table",
            file=sys.stderr,
        )
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping scatter figure", file=sys.stderr)
        return
    pair = wide_display[[x_lab, y_lab]].dropna()
    if pair.empty:
        print("Skip scatter: no paired observations.", file=sys.stderr)
        return
    x = pair[x_lab].to_numpy(dtype=float)
    y = pair[y_lab].to_numpy(dtype=float)
    if method == "pearson":
        r = float(pd.Series(x).corr(pd.Series(y), method="pearson"))
    else:
        r = float(pd.Series(x).corr(pd.Series(y), method="spearman"))

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(x, y, s=28, alpha=0.75, edgecolors="0.35", linewidths=0.4)
    ax.set_xlabel(f"{x_lab}", fontsize=_AXIS_TICK_FONTSIZE)
    ax.set_ylabel(f"{y_lab}", fontsize=_AXIS_TICK_FONTSIZE)
    ax.tick_params(axis="both", labelsize=_AXIS_TICK_FONTSIZE)
    ax.text(
        0.04,
        0.96,
        f"$r_{{{method[0]}}}$ = {r:.2f}\n$n$ = {len(pair)}",
        transform=ax.transAxes,
        fontsize=_AXIS_TICK_FONTSIZE,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.92),
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run(
    base_dir: Path,
    output_dir: Path,
    method: str,
    min_periods: int,
    save_diff: bool,
    dpi: int,
    scatter_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]],
    mts_only: bool,
) -> int:
    base_dir = Path(base_dir).resolve()
    output_dir = Path(output_dir).resolve()
    paths = cfg.get_paths(base_dir)
    asym_root = paths["output_dir"]  # asymmetry_tle_region root

    discovered = discover_region_slugs(asym_root)
    if not discovered:
        print(f"No region folders found under {asym_root}", file=sys.stderr)
        return 1

    region_list = order_region_pairs(discovered)

    bilateral_tables = load_bilateral_tables(base_dir)
    wide, _skipped = build_wide_diff_table(base_dir, paths, region_list, bilateral_tables)
    if wide.empty:
        print("No columns in wide diff table; nothing to correlate.", file=sys.stderr)
        return 1

    output_suffix = "_mts" if mts_only else ""
    if mts_only:
        mts_subs, err = load_mts_positive_subjects(paths["inclusion_metadata"])
        if err:
            print(err, file=sys.stderr)
            return 1
        if not mts_subs:
            print(
                "MTS filter: no subjects flagged in mts/lesion_mts.",
                file=sys.stderr,
            )
            return 1
        n_before = len(wide)
        wide = wide.loc[wide.index.isin(mts_subs)]
        if wide.empty:
            print(
                "MTS filter: no overlap between MTS+ subjects and asymmetry table rows.",
                file=sys.stderr,
            )
            return 1
        print(
            f"MTS-only: {len(wide)} subjects in analysis "
            f"({n_before - len(wide)} excluded; {len(mts_subs)} MTS+ in inclusion).",
            file=sys.stderr,
        )

    wide_display = wide_columns_to_display(wide)

    kw: Dict[str, object] = {"min_periods": min_periods}
    if method == "pearson":
        corr = wide_display.corr(method="pearson", **kw)
    elif method == "spearman":
        corr = wide_display.corr(method="spearman", **kw)
    else:
        print(f"Unknown method: {method}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    corr_stem = f"mahalanobis_asymmetry_corr_{method}{output_suffix}"
    corr_path = output_dir / f"{corr_stem}.csv"
    corr.to_csv(corr_path, index_label="region")
    print(f"Wrote {corr_path}")

    png_path = output_dir / f"{corr_stem}.png"
    plot_correlation_heatmap(corr, png_path, dpi=dpi)
    if png_path.is_file():
        print(f"Wrote {png_path}")

    for (xa, xs), (ya, ys) in scatter_pairs:
        scatter_stem = scatter_output_stem(
            xa, xs, ya, ys, method, output_suffix
        )
        scatter_path = output_dir / f"{scatter_stem}.png"
        plot_region_pair_scatter(
            wide_display, xa, xs, ya, ys, scatter_path, method, dpi=dpi
        )
        if scatter_path.is_file():
            print(f"Wrote {scatter_path}")

    cbar_path = output_dir / f"mahalanobis_asymmetry_corr_colorbar{output_suffix}.png"
    plot_standalone_correlation_colorbar(cbar_path, dpi=dpi)
    if cbar_path.is_file():
        print(f"Wrote {cbar_path}")

    if save_diff:
        diff_path = (
            output_dir
            / f"mahalanobis_asymmetry_per_subject_diff_wide{output_suffix}.csv"
        )
        wide_display.to_csv(diff_path, index_label="sub")
        print(f"Wrote {diff_path}")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Correlation matrix of relative Mahalanobis asymmetry "
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
        help="Output directory for CSV/PNG (default: derivatives/analysis/asymmetry_correlations)",
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
        help="Also save wide per-subject asymmetry table (ipsi−contra)/(|ipsi|+|contra|)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for heatmap, scatter, and colorbar PNGs (default: 300)",
    )
    p.add_argument(
        "--scatter-pair",
        action="append",
        default=None,
        metavar="X,Y",
        help=(
            "Scatter relative Mahalanobis asymmetry: two atlas:slug specs separated by a comma; "
            "first is x-axis, second is y-axis (e.g. 4S156:Anterior,HCP1065:F_core). "
            "Repeat for multiple figures. Default when omitted: that pair. "
            "Use --no-scatter to skip scatter plots."
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
            "Restrict to subjects with MTS (non-zero mts or lesion_mts in "
            "results/inclusion/penn_epilepsy_included_basic_metadata.csv). "
            "Output filenames get a _mts suffix."
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
            scatter_pairs = [parse_scatter_pair_arg(s) for s in args.scatter_pair]
        else:
            scatter_pairs = [parse_scatter_pair_arg("4S156:Anterior,HCP1065:F_core")]
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
    )


if __name__ == "__main__":
    sys.exit(main())
