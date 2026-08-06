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
White-matter tract profiles (Mahalanobis): node-level paired Cohen's d.

This script computes signed paired Cohen's d (ipsi - contra) across subjects,
*only* for white matter tracts at the node-level (100 along-tract nodes).

Input data:
  {project_root()}/derivatives/analysis/tract_asymmetry/
    sub-*/sub-* _asym_mahal_node.csv

Each node-level CSV contains columns like:
  sub, tract, node, hemi_ipsi, ipsi_mahal_raw, contra_mahal_raw, asymmetry_mahal_raw

Outputs:
  - summary_hcp1065_nodes_mahalanobis.csv (one row per tract_base x node)
  - summary_hcp1065_nodes_mahalanobis_unilateral.csv (subject-level: sub, group, label, node, mahalanobis; label = e.g. AF_L)
  - hcp1065_nodes_mahalanobis_profiles.png (contact-sheet: 1 subplot per tract)
  - microstructural_asymmetry_report_mahalanobis_profiles.html
"""

from __future__ import annotations

import html as html_module
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = project_root()
TRACT_ASYM_DIR = analysis_dir() / "tract_asymmetry"
TRACT_METADATA_PATH = PROJECT_ROOT / "data" / "atlases" / "HCP1065" / "HCP1065_tract_metadata.csv"
OUTPUT_DIR = analysis_dir() / "microstructural_asymmetries"
INCLUSION_PATH = PROJECT_ROOT / "results" / "inclusion" / "penn_epilepsy_included_basic_metadata.csv"


def _load_subject_group() -> Dict[str, str]:
    """Load TLE inclusion CSV; return sub -> 'left_TLE' or 'right_TLE' from laterality column."""
    out: Dict[str, str] = {}
    if not INCLUSION_PATH.exists():
        return out
    try:
        df = pd.read_csv(INCLUSION_PATH)
        if "sub" not in df.columns or "laterality" not in df.columns:
            return out
        if "lobe" in df.columns:
            df = df[df["lobe"].astype(str).str.strip().str.lower() == "temporal"]
        for _, row in df.iterrows():
            sub = row.get("sub")
            if pd.isna(sub):
                continue
            sub = str(sub)
            lat = str(row.get("laterality", "")).strip().lower()
            if lat == "left":
                out[sub] = "left_TLE"
            elif lat == "right":
                out[sub] = "right_TLE"
    except Exception:
        pass
    return out


def _cohens_d_paired(ipsi_vals: List[float], contra_vals: List[float]) -> float:
    """Paired Cohen's d = mean(ipsi - contra) / std(ipsi - contra). NaN if n<2 or std=0."""
    ipsi = np.asarray(ipsi_vals, dtype=float)
    contra = np.asarray(contra_vals, dtype=float)
    valid = np.isfinite(ipsi) & np.isfinite(contra)
    diff = ipsi[valid] - contra[valid]
    if len(diff) < 2:
        return float("nan")
    std_diff = float(np.std(diff, ddof=1))
    if std_diff <= 0:
        return float("nan")
    return float(np.mean(diff)) / std_diff


def _load_tract_metadata() -> Dict[str, str]:
    """Return tract_base -> type (association/projection). tract_base matches node CSV 'tract' values."""
    tract_base_to_type: Dict[str, str] = {}
    if not TRACT_METADATA_PATH.exists():
        return tract_base_to_type
    try:
        meta = pd.read_csv(TRACT_METADATA_PATH)
        if "label" not in meta.columns or "type" not in meta.columns:
            return tract_base_to_type
        for _, row in meta.iterrows():
            label = str(row["label"]).strip()
            ttype = str(row["type"]).strip().lower()
            if not (label.endswith("_L") or label.endswith("_R")):
                continue
            base = label[:-2]
            if ttype in ("association", "projection") and base not in tract_base_to_type:
                tract_base_to_type[base] = ttype
    except Exception:
        return {}
    return tract_base_to_type


def _load_tract_end_locs() -> Dict[str, Tuple[str, str]]:
    """
    Return tract_base -> (end1_word, end2_word) derived from HCP1065 metadata.

    Important: per user request we do NOT use `end1_loc/end2_loc`. We map the `end1/end2`
    endpoint codes (e.g. A/P/I/S) to anatomical words (anterior/posterior/inf/sup).
    """
    if not TRACT_METADATA_PATH.exists():
        return {}

    code_to_word = {
        "A": "anterior",
        "P": "posterior",
        "I": "inferior",
        "S": "superior",
        "M": "medial",
        "L": "lateral",
        "NA": "",
        "": "",
        "N/A": "",
    }

    base_to_locs: Dict[str, Tuple[str, str]] = {}
    try:
        meta = pd.read_csv(TRACT_METADATA_PATH)
        needed = {"label", "end1", "end2", "type"}
        if not needed.issubset(set(meta.columns)):
            return {}
        for _, row in meta.iterrows():
            label = str(row["label"]).strip()
            if not (label.endswith("_L") or label.endswith("_R")):
                continue
            base = label[:-2]
            end1_code = str(row.get("end1", "")).strip()
            end2_code = str(row.get("end2", "")).strip()
            end1_word = code_to_word.get(end1_code, end1_code.lower() if end1_code else "")
            end2_word = code_to_word.get(end2_code, end2_code.lower() if end2_code else "")
            # Prefer first seen mapping (should be consistent across hemispheres for the base)
            if base not in base_to_locs and (end1_word or end2_word):
                base_to_locs[base] = (end1_word, end2_word)
    except Exception:
        return {}

    return base_to_locs


def load_tract_mahal_node() -> pd.DataFrame:
    """Load all *_asym_mahal_node.csv files into one DataFrame (WM nodes only)."""
    rows: List[dict] = []
    for sub_dir in TRACT_ASYM_DIR.iterdir():
        if not sub_dir.is_dir():
            continue
        csv_path = sub_dir / f"{sub_dir.name}_asym_mahal_node.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        required = {"sub", "tract", "node", "ipsi_mahal_raw", "contra_mahal_raw"}
        if not required.issubset(set(df.columns)):
            # Skip unexpected schema variants
            continue
        # Ensure node numeric and keep only the needed columns
        df = df.copy()
        df["node"] = pd.to_numeric(df["node"], errors="coerce")
        df = df[df["node"].notna()].copy()
        if "asymmetry_mahal_raw" not in df.columns:
            df["asymmetry_mahal_raw"] = df["ipsi_mahal_raw"] - df["contra_mahal_raw"]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compute_cohens_d_per_tract_node(df_nodes: pd.DataFrame) -> pd.DataFrame:
    """Compute signed paired Cohen's d per (tract, node) across subjects."""
    if df_nodes.empty:
        return pd.DataFrame()

    required = {"sub", "tract", "node", "ipsi_mahal_raw", "contra_mahal_raw", "asymmetry_mahal_raw"}
    missing = required.difference(set(df_nodes.columns))
    if missing:
        raise ValueError(f"Missing columns in node dataframe: {sorted(missing)}")

    results: List[dict] = []
    for (tract, node), grp in df_nodes.groupby(["tract", "node"]):
        ipsi = grp["ipsi_mahal_raw"].astype(float).tolist()
        contra = grp["contra_mahal_raw"].astype(float).tolist()
        mean_ipsi = float(np.nanmean(ipsi)) if len(ipsi) else float("nan")
        mean_contra = float(np.nanmean(contra)) if len(contra) else float("nan")
        mean_asym = float(np.nanmean(grp["asymmetry_mahal_raw"].astype(float).values)) if len(grp) else float("nan")
        d = _cohens_d_paired(ipsi, contra)
        results.append(
            {
                "label": str(tract),
                "node": int(node),
                "mean_ipsi": mean_ipsi,
                "mean_contra": mean_contra,
                "mean_asymmetry": mean_asym,
                "mean_cohen_d": float(d) if np.isfinite(d) else float("nan"),
            }
        )

    out = pd.DataFrame(results)
    if out.empty:
        return out
    out = out.sort_values(["label", "node"]).reset_index(drop=True)
    return out


def save_summary_hcp1065_nodes_unilateral(
    node_df: pd.DataFrame,
    subject_group: Dict[str, str],
    report_dir: Path,
) -> None:
    """Save subject-level unilateral node Mahalanobis: sub, group, label, node, mahalanobis.
    label = unilateral tract (e.g. AF_L, AF_R); node = along-tract node index; group = left_TLE or right_TLE.
    Reorganizes ipsi/contra into hemisphere-specific values per subject at each node."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    valid = node_df["ipsi_mahal_raw"].notna() & node_df["contra_mahal_raw"].notna()
    node_df = node_df[valid].copy()
    if node_df.empty:
        return
    rows: List[dict] = []
    for _, r in node_df.iterrows():
        sub = str(r["sub"])
        group = subject_group.get(sub)
        if group not in ("left_TLE", "right_TLE"):
            continue
        tract_base = str(r["tract"])
        node = int(r["node"])
        ipsi = float(r["ipsi_mahal_raw"])
        contra = float(r["contra_mahal_raw"])
        if group == "left_TLE":
            rows.append({"sub": sub, "group": group, "label": f"{tract_base}_L", "node": node, "mahalanobis": ipsi})
            rows.append({"sub": sub, "group": group, "label": f"{tract_base}_R", "node": node, "mahalanobis": contra})
        else:
            rows.append({"sub": sub, "group": group, "label": f"{tract_base}_L", "node": node, "mahalanobis": contra})
            rows.append({"sub": sub, "group": group, "label": f"{tract_base}_R", "node": node, "mahalanobis": ipsi})
    if rows:
        pd.DataFrame(rows).to_csv(report_dir / "summary_hcp1065_nodes_mahalanobis_unilateral.csv", index=False)


def plot_tract_node_profiles(
    summary_df: pd.DataFrame,
    tract_base_to_type: Dict[str, str],
    tract_base_to_end_locs: Dict[str, Tuple[str, str]],
    out_png: Path,
    tract_type: str,
    ncols: int = 4,
) -> Optional[Path]:
    """Create a contact-sheet with 1 subplot per tract (x=node, y=mean_cohen_d)."""
    if summary_df.empty:
        return None

    import matplotlib.pyplot as plt

    all_tracts = summary_df["label"].unique().tolist()
    tracts = sorted([t for t in all_tracts if tract_base_to_type.get(str(t), "") == tract_type])
    if not tracts:
        return None

    n = len(tracts)
    nrows = int(math.ceil(n / ncols))

    global_abs = float(
        np.nanmax(np.abs(summary_df.loc[summary_df["label"].isin(tracts), "mean_cohen_d"].values))
    ) if not summary_df.empty else 0.0
    if not np.isfinite(global_abs) or global_abs <= 0:
        global_abs = 0.1
    ylim = (-(global_abs * 1.05), global_abs * 1.05)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.0), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for i, tract in enumerate(tracts):
        ax = axes_flat[i]
        sub = summary_df[summary_df["label"] == tract].sort_values("node")
        ax.plot(sub["node"].values, sub["mean_cohen_d"].values, color="#1f77b4", linewidth=1.0)
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.7)
        ax.set_title(str(tract), fontsize=9)
        ax.set_ylim(*ylim)
        ax.set_xlim(float(sub["node"].min()), float(sub["node"].max()))
        ax.grid(True, alpha=0.15, linewidth=0.5)
        # Endpoint anatomical labels should be shown as x-axis ticks (node min/max).
        node_min = int(sub["node"].min())
        node_max = int(sub["node"].max())
        end1_word, end2_word = tract_base_to_end_locs.get(str(tract), ("", ""))
        # Sparse ticks: show node min + middle (if present) + node max.
        candidate_ticks = [node_min]
        if node_min <= 50 <= node_max:
            candidate_ticks.append(50)
        if node_max != node_min:
            candidate_ticks.append(node_max)
        candidate_ticks = sorted(set(candidate_ticks))
        tick_labels: List[str] = []
        for t in candidate_ticks:
            if t == node_min and end1_word:
                tick_labels.append(end1_word)
            elif t == node_max and end2_word:
                tick_labels.append(end2_word)
            elif t == 50 and (node_min <= 50 <= node_max):
                tick_labels.append("50")
            else:
                tick_labels.append(str(t))
        ax.set_xticks(candidate_ticks)
        ax.set_xticklabels(tick_labels)
        if i % ncols == 0:
            ax.set_ylabel("Mean Cohen's d")
        if i >= (n - ncols):
            ax.set_xlabel("Node")

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"HCP1065 node-level tract profiles ({tract_type} tracts; paired Cohen's d: ipsi - contra)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def create_report_html(
    out_html: Path,
    association_profile_png: Optional[Path],
    projection_profile_png: Optional[Path],
    summary_csv: Path,
) -> None:
    """Create a small HTML report embedding the tract profile contact-sheets."""
    out_html.parent.mkdir(parents=True, exist_ok=True)

    summary_csv_rel = summary_csv.name
    assoc_tag = "<p>No association tract profiles figure.</p>"
    if association_profile_png and association_profile_png.exists():
        try:
            rel = association_profile_png.relative_to(out_html.parent)
            assoc_png_rel = rel.as_posix()
        except ValueError:
            assoc_png_rel = f"figures/{association_profile_png.name}"
        assoc_tag = (
            f'<img src="{html_module.escape(assoc_png_rel)}" alt="Association tract profiles contact sheet" style="max-width: 100%; height: auto;"/>'
        )

    proj_tag = "<p>No projection tract profiles figure.</p>"
    if projection_profile_png and projection_profile_png.exists():
        try:
            rel = projection_profile_png.relative_to(out_html.parent)
            proj_png_rel = rel.as_posix()
        except ValueError:
            proj_png_rel = f"figures/{projection_profile_png.name}"
        proj_tag = (
            f'<img src="{html_module.escape(proj_png_rel)}" alt="Projection tract profiles contact sheet" style="max-width: 100%; height: auto;"/>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Mahalanobis tract node profiles report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1600px; margin: 2em auto; padding: 0 2em; }}
    h1 {{ font-size: 1.5rem; }}
    h2 {{ font-size: 1.2rem; margin-top: 1.5em; }}
    img {{ max-width: 100%; height: auto; }}
    code {{ background: #f2f2f2; padding: 0 6px; border-radius: 4px; }}
    .caption {{ color: #555; font-size: 0.9rem; margin-top: 0.5em; }}
  </style>
</head>
<body>
  <h1>Mahalanobis tract asymmetry profiles (node-level)</h1>
  <p>
    Signed paired Cohen's d computed from <code>ipsi_mahal_raw</code> and <code>contra_mahal_raw</code>
    at the node-level: <code>d = mean(ipsi - contra) / std(ipsi - contra)</code>, grouped per
    tract base and node across subjects.
  </p>

  <h2>Tract profiles (node-level)</h2>
  <h3>Association tracts</h3>
  {assoc_tag}
  <p class="caption">X-axis = node (1..100). Y-axis = mean paired Cohen's d across subjects. Endpoint labels come from HCP1065 metadata.</p>

  <h3>Projection tracts</h3>
  {proj_tag}
  <p class="caption">X-axis = node (1..100). Y-axis = mean paired Cohen's d across subjects. Endpoint labels come from HCP1065 metadata.</p>

  <h2>Summary CSV</h2>
  <p>
    Saved to <code>{html_module.escape(summary_csv_rel)}</code>.
    Columns include <code>label</code> (tract base), <code>node</code>,
    <code>mean_ipsi</code>, <code>mean_contra</code>, <code>mean_asymmetry</code>, and <code>mean_cohen_d</code>.
  </p>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    tract_base_to_type = _load_tract_metadata()
    tract_base_to_end_locs = _load_tract_end_locs()
    node_df = load_tract_mahal_node()
    if node_df.empty:
        print("No *_asym_mahal_node.csv files found or schema mismatch; nothing to compute.", file=sys.stderr)
        out_html = OUTPUT_DIR / "microstructural_asymmetry_report_mahalanobis_profiles.html"
        create_report_html(out_html, None, None, OUTPUT_DIR / "summary_hcp1065_nodes_mahalanobis.csv")
        return 1

    summary_df = compute_cohens_d_per_tract_node(node_df)
    if summary_df.empty:
        print("Node-level summary computation produced no rows.", file=sys.stderr)
        out_html = OUTPUT_DIR / "microstructural_asymmetry_report_mahalanobis_profiles.html"
        create_report_html(out_html, None, None, OUTPUT_DIR / "summary_hcp1065_nodes_mahalanobis.csv")
        return 1

    summary_csv = OUTPUT_DIR / "summary_hcp1065_nodes_mahalanobis.csv"
    summary_df.to_csv(summary_csv, index=False)

    subject_group = _load_subject_group()
    save_summary_hcp1065_nodes_unilateral(node_df, subject_group, OUTPUT_DIR)

    assoc_png = figures_dir / "hcp1065_nodes_mahalanobis_profiles_association.png"
    proj_png = figures_dir / "hcp1065_nodes_mahalanobis_profiles_projection.png"
    plot_tract_node_profiles(
        summary_df,
        tract_base_to_type,
        tract_base_to_end_locs,
        assoc_png,
        tract_type="association",
        ncols=4,
    )
    plot_tract_node_profiles(
        summary_df,
        tract_base_to_type,
        tract_base_to_end_locs,
        proj_png,
        tract_type="projection",
        ncols=4,
    )

    out_html = OUTPUT_DIR / "microstructural_asymmetry_report_mahalanobis_profiles.html"
    create_report_html(out_html, assoc_png, proj_png, summary_csv)

    print(f"WM tract node profile report written to: {out_html}")
    print(f"Summary CSV written to: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

