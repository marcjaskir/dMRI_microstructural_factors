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
Within-patient scalar asymmetry: bar plots of the asymmetry index per diffusion scalar for a given
(subject, region), plus a subject-level HTML gallery.

The asymmetry index is (I − C) / (|I| + |C|) with I, C = ipsi/contra mean z-scores. Scalars are ordered
by index value from highest to lowest.

Data sources match microstructural_asymmetry_report_scalars.py:
  - GM: region_asymmetry_tle/{sub}/{sub}_asym_regions.csv (stat == mean)
  - WM (HCP1065 thirds): tract_asymmetry/{sub}/{sub}_asym_scalars.csv
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import shared helpers from the microstructural asymmetry report (sibling package path).
_ANALYSIS_DIR = Path(__file__).resolve().parent.parent
_MRS_DIR = _ANALYSIS_DIR / "microstructural_asymmetries"
if str(_MRS_DIR) not in sys.path:
    sys.path.insert(0, str(_MRS_DIR))
import microstructural_asymmetry_report_scalars as mrs  # noqa: E402

# -----------------------------------------------------------------------------
# Paths (default base = structural_tractometry root)
# -----------------------------------------------------------------------------
def _configure_matplotlib() -> None:
    mrs._configure_matplotlib_georgia()


def _normalized_asymmetry(ipsi: float, contra: float) -> float:
    """(I - C) / (|I| + |C|); NaN if denominator is 0."""
    i, c = float(ipsi), float(contra)
    den = abs(i) + abs(c)
    if den <= 0 or not (np.isfinite(i) and np.isfinite(c)):
        return float("nan")
    return (i - c) / den


def _sanitize_filename_token(s: str) -> str:
    return (
        str(s)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _normalize_gm_region_token(raw: str) -> str:
    """Glasser-style Left_X / Right_X -> base X; otherwise strip and return."""
    s = str(raw).strip()
    if s.startswith("Left_"):
        return s[5:].strip()
    if s.startswith("Right_"):
        return s[6:].strip()
    return s


def load_subject_region_csv(base_dir: Path, sub: str) -> pd.DataFrame:
    path = base_dir / "derivatives" / "analysis" / "region_asymmetry_tle" / sub / f"{sub}_asym_regions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "sub" not in df.columns and len(df):
        df = df.copy()
        df["sub"] = sub
    return df


def load_subject_tract_csv(base_dir: Path, sub: str) -> pd.DataFrame:
    path = base_dir / "derivatives" / "analysis" / "tract_asymmetry" / sub / f"{sub}_asym_scalars.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "sub" not in df.columns and len(df):
        df = df.copy()
        df["sub"] = sub
    return df


def _tract_has_roi(tract_df: pd.DataFrame, sub: str, tract: str, segment: str) -> bool:
    if tract_df.empty:
        return False
    m = (tract_df["sub"].astype(str) == sub) & (tract_df["tract"].astype(str) == tract) & (
        tract_df["segment"].astype(str) == str(segment)
    )
    return bool(m.any())


def resolve_region_kind(
    region_token: str,
    sub: str,
    tract_df: pd.DataFrame,
    glasser_bases: set,
    subcortex_bases: set,
    explicit: Optional[str],
) -> Tuple[str, str]:
    """
    Returns (kind, lookup_key) where kind is 'wm' | 'glasser' | '4s_subcortex'.
    For WM, lookup_key is roi_id 'tract_segment' for filenames; GM uses region base string.
    """
    raw = str(region_token).strip()
    if explicit:
        e = explicit.strip().lower()
        if e == "hcp1065_thirds":
            t, s = mrs._wm_roi_to_tract_segment(raw)
            return "wm", f"{t}_{s}"
        if e == "glasser":
            return "glasser", _normalize_gm_region_token(raw)
        if e in ("4s_subcortex", "4s-subcortex", "subcortex"):
            return "4s_subcortex", _normalize_gm_region_token(raw)
        raise ValueError(f"Unknown --atlas {explicit!r}; use glasser, 4s_subcortex, hcp1065_thirds")

    # 1) WM: parse and verify against tract CSV
    if "_" in raw:
        t, s = mrs._wm_roi_to_tract_segment(raw)
        if t and _tract_has_roi(tract_df, sub, t, s):
            return "wm", f"{t}_{s}"

    base = _normalize_gm_region_token(raw)
    # 2) Glasser
    if base in glasser_bases:
        return "glasser", base
    # 3) 4S subcortex
    if base in subcortex_bases:
        return "4s_subcortex", base

    # 4) Retry WM without requiring underscore (unlikely)
    if _tract_has_roi(tract_df, sub, raw, "core"):
        return "wm", f"{raw}_core"

    raise ValueError(
        f"Could not resolve region {region_token!r} for {sub}: not found as WM (tract_asymmetry), "
        f"Glasser base, or 4S subcortex base. Try --atlas glasser|4s_subcortex|hcp1065_thirds."
    )


def extract_scalar_asymmetries_region(
    df: pd.DataFrame,
    sub: str,
    region_base: str,
) -> pd.DataFrame:
    """Rows: scalar, asymmetry, ipsi_mean_z, contra_mean_z (stat == mean)."""
    if df.empty:
        return pd.DataFrame(columns=["scalar", "asymmetry", "ipsi_mean_z", "contra_mean_z"])
    d = df[(df["sub"].astype(str) == sub) & (df["region"].astype(str) == str(region_base))].copy()
    if "stat" in d.columns:
        d = d[d["stat"].astype(str) == "mean"]
    need = ["scalar", "asymmetry", "ipsi_mean_z", "contra_mean_z"]
    for c in need:
        if c not in d.columns:
            return pd.DataFrame(columns=need)
    out = d[need].drop_duplicates(subset=["scalar"])
    return out


def extract_scalar_asymmetries_tract(
    df: pd.DataFrame,
    sub: str,
    tract: str,
    segment: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["scalar", "asymmetry", "ipsi_mean_z", "contra_mean_z"])
    d = df[
        (df["sub"].astype(str) == sub)
        & (df["tract"].astype(str) == str(tract))
        & (df["segment"].astype(str) == str(segment))
    ].copy()
    need = ["scalar", "asymmetry", "ipsi_mean_z", "contra_mean_z"]
    for c in need:
        if c not in d.columns:
            return pd.DataFrame(columns=need)
    out = d[need].drop_duplicates(subset=["scalar"])
    return out


def verify_asymmetry_column(rows: pd.DataFrame, tol: float = 1e-5) -> None:
    """Log warning if stored asymmetry differs from recomputed (I-C)/(|I|+|C|)."""
    if rows.empty:
        return
    for _, r in rows.iterrows():
        a = float(r["asymmetry"]) if pd.notna(r["asymmetry"]) else float("nan")
        rec = _normalized_asymmetry(r["ipsi_mean_z"], r["contra_mean_z"])
        if np.isfinite(a) and np.isfinite(rec) and abs(a - rec) > tol:
            print(
                f"Warning: asymmetry mismatch for scalar {r['scalar']}: "
                f"file={a:.6g} recomputed={rec:.6g}",
                file=sys.stderr,
            )


def load_scalar_labels_and_colors(base_dir: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    labels_path = base_dir / "data" / "metadata" / "scalar_labels_to_human.json"
    colors_path = base_dir / "data" / "metadata" / "scalar_labels_to_colors.json"
    labels: Dict[str, str] = {}
    colors: Dict[str, str] = {}
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    if colors_path.exists():
        with open(colors_path, "r", encoding="utf-8") as f:
            colors = json.load(f)
    return labels, colors


def _human_title_region(
    kind: str,
    lookup_key: str,
    base_dir: Path,
) -> str:
    if kind == "glasser":
        meta_path = base_dir / "data" / "atlases" / "Glasser" / "glasser_additional_metadata.csv"
        if meta_path.exists():
            try:
                gm = pd.read_csv(meta_path)
                if "region" in gm.columns and "regionLongName" in gm.columns:
                    row = gm[gm["region"].astype(str) == str(lookup_key)]
                    if not row.empty:
                        return str(row["regionLongName"].iloc[0])
            except Exception:
                pass
        return str(lookup_key)
    if kind == "4s_subcortex":
        th = mrs._load_4s_thalamus_roi_bases()
        return mrs._format_4s_subcortex_tex_label(str(lookup_key), th)
    # WM
    tract_names = mrs._load_tract_label_to_pretty_name()
    tract_hemi, seg = mrs._wm_roi_to_tract_segment(lookup_key)
    tract_pretty = tract_names.get(tract_hemi, tract_hemi.replace("_", " "))
    seg_pretty = mrs._hcp1065_segment_human(seg)
    return f"{tract_pretty} — {seg_pretty}"


def plot_asymmetry_bars(
    rows: pd.DataFrame,
    out_png: Path,
    title: str,
    scalar_labels: Dict[str, str],
    scalar_colors: Dict[str, str],
) -> None:
    import matplotlib.pyplot as plt

    if rows.empty:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        return

    plot_df = rows[~rows["scalar"].isin(mrs.EXCLUDED_SCALARS)].copy()
    plot_df["asymmetry"] = pd.to_numeric(plot_df["asymmetry"], errors="coerce")
    plot_df = plot_df.dropna(subset=["asymmetry"])
    if plot_df.empty:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "No data after exclusions", ha="center", va="center")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        return

    plot_df = plot_df.copy()
    # Sort by asymmetry index descending (highest first); one row per scalar
    plot_df = plot_df.sort_values("asymmetry", ascending=False).drop_duplicates(
        subset=["scalar"], keep="first"
    )
    scalars_sorted = plot_df["scalar"].tolist()
    if not scalars_sorted:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "No scalars to plot", ha="center", va="center")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        return

    model_fallback = {"dki": "#7A297F", "dti": "#C43031", "gqi": "#FAA51A", "noddi": "#38489E", "map": "#289144", "rdi": "#C43031"}
    bar_palette = [mrs._scalar_color(s, scalar_colors, model_fallback) for s in scalars_sorted]
    vals_arr = plot_df["asymmetry"].to_numpy(dtype=float)
    n = len(scalars_sorted)
    tick_labels = [mrs._scalar_abbrev(s) for s in scalars_sorted]

    fig_w = max(7.0, 0.32 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    x = np.arange(n, dtype=float)
    ax.bar(
        x,
        vals_arr,
        width=min(0.74, 20.0 / max(n, 1)),
        color=bar_palette,
        edgecolor="0.25",
        linewidth=0.8,
    )
    ax.axhline(0.0, color="0.35", linewidth=1.0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=13.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(r"$\frac{I - C}{|I| + |C|}$", fontsize=15)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def write_subject_html(
    base_dir: Path,
    sub: str,
    out_dir: Path,
    html_path: Path,
) -> None:
    """Embed all *asymmetry_by_scalar.png under out_dir in a single HTML page."""
    pattern = "*_asymmetry_by_scalar.png"
    pngs = sorted(out_dir.glob(pattern))
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>Within-patient scalar asymmetry — {html.escape(sub)}</title>",
        "<style>body{font-family:Georgia,serif;max-width:1200px;margin:24px auto;padding:0 16px;}"
        "h2{border-bottom:1px solid #ccc;padding-bottom:6px;}img{max-width:100%;height:auto;border:1px solid #ddd;}"
        ".equation{font-size:1.05em;margin:0.5em 0 1em 1.5em;font-family:Georgia,'Times New Roman',serif;}"
        ".note{color:#444;font-size:0.95em;}</style>",
        "</head><body>",
        f"<h1>Within-patient scalar asymmetry — {html.escape(sub)}</h1>",
        "<p><strong>Asymmetry index</strong> (per diffusion scalar), with "
        "<em>I</em> = ipsilateral mean z-score and <em>C</em> = contralateral mean z-score:</p>",
        '<p class="equation">asymmetry index = (I − C) / (|I| + |C|)</p>',
        '<p class="note">Figures plot this <strong>asymmetry index</strong> (signed). Scalars are ordered from '
        "<strong>highest</strong> to <strong>lowest</strong> index value.</p>",
    ]
    rel_root = html_path.parent
    for p in pngs:
        rel = p.relative_to(rel_root).as_posix()
        stem = p.stem.replace("_asymmetry_by_scalar", "")
        lines.append(f'<h2>{html.escape(stem)}</h2>')
        lines.append(f'<p><img src="{html.escape(rel)}" alt="{html.escape(stem)}"></p>')
    if not pngs:
        lines.append("<p><em>No figures found matching *_asymmetry_by_scalar.png in this folder.</em></p>")
    lines.append("</body></html>")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_region(
    sub: str,
    region_token: str,
    base_dir: Path,
    explicit_atlas: Optional[str],
    scalar_labels: Dict[str, str],
    scalar_colors: Dict[str, str],
) -> Path:
    """Generate one PNG; return path to PNG."""
    _configure_matplotlib()
    tract_df = load_subject_tract_csv(base_dir, sub)
    region_df = load_subject_region_csv(base_dir, sub)
    glasser_bases = mrs._get_glasser_bases()
    subcortex_bases = mrs._get_4s_subcortical_bases()

    kind, lookup_key = resolve_region_kind(
        region_token, sub, tract_df, glasser_bases, subcortex_bases, explicit_atlas
    )

    if kind == "wm":
        tract, segment = mrs._wm_roi_to_tract_segment(lookup_key)
        rows = extract_scalar_asymmetries_tract(tract_df, sub, tract, segment)
        src = base_dir / "derivatives" / "analysis" / "tract_asymmetry" / sub / f"{sub}_asym_scalars.csv"
    else:
        rows = extract_scalar_asymmetries_region(region_df, sub, lookup_key)
        src = base_dir / "derivatives" / "analysis" / "region_asymmetry_tle" / sub / f"{sub}_asym_regions.csv"

    if rows.empty:
        raise RuntimeError(f"No rows for {sub} region {region_token!r} (resolved: {kind} {lookup_key!r}). Source: {src}")

    verify_asymmetry_column(rows)

    out_dir = base_dir / "derivatives" / "analysis" / "within_patient_scalar_asymmetries" / sub
    safe = _sanitize_filename_token(lookup_key if kind != "wm" else lookup_key)
    png_name = f"{sub}_{safe}_asymmetry_by_scalar.png"
    out_png = out_dir / png_name

    region_pretty = _human_title_region(kind, lookup_key if kind != "wm" else lookup_key, base_dir)
    title = f"{sub}\n{region_pretty}"

    plot_asymmetry_bars(rows, out_png, title, scalar_labels, scalar_colors)
    return out_png


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Within-patient asymmetry index bar plots (sorted by index desc) and subject HTML gallery."
    )
    p.add_argument("--subject", required=True, help="Subject id (BIDS format, e.g. sub-XXXX)")
    p.add_argument(
        "--regions",
        nargs="+",
        required=True,
        help="Region tokens: Glasser base (V1), 4S subcortex (Ca), or WM roi (AF_core, C_FP_L_A)",
    )
    p.add_argument(
        "--atlas",
        action="append",
        default=None,
        help="Optional atlas for each region in order: glasser | 4s_subcortex | hcp1065_thirds "
        "(repeat --atlas per region if needed, or omit for auto-detect)",
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help=f"Project root (default: {DEFAULT_PROJECT_ROOT})",
    )
    args = p.parse_args(argv)

    sub = args.subject.strip()
    base_dir = args.base_dir.resolve()
    scalar_labels, scalar_colors = load_scalar_labels_and_colors(base_dir)

    atlas_list: Optional[List[Optional[str]]] = None
    if args.atlas:
        if len(args.atlas) not in (1, len(args.regions)):
            p.error("Provide either one --atlas (applies to all regions) or one --atlas per region.")
        if len(args.atlas) == 1:
            atlas_list = [args.atlas[0]] * len(args.regions)
        else:
            atlas_list = args.atlas

    out_dir = base_dir / "derivatives" / "analysis" / "within_patient_scalar_asymmetries" / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, region_token in enumerate(args.regions):
        explicit = atlas_list[i] if atlas_list else None
        try:
            run_one_region(sub, region_token, base_dir, explicit, scalar_labels, scalar_colors)
            print(f"Wrote figure for region {region_token!r}", file=sys.stderr)
        except Exception as e:
            print(f"Error for region {region_token!r}: {e}", file=sys.stderr)
            return 1

    html_path = base_dir / "derivatives" / "analysis" / "within_patient_scalar_asymmetries" / f"{sub}.html"
    write_subject_html(base_dir, sub, out_dir, html_path)
    print(f"Wrote {html_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
