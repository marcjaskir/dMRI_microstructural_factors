import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
# GOAL: Compute and plot GAM-based z-scores by:
# * 1) white matter tract segments (end1, core, end2), 2) cortical grey matter regions (Glasser), 3) subcortical grey matter regions (4S156)
# * 1) left vs. right TLE using get_left_right_tle_subjects(), and 2) individual subjects
# * Plot using nilearn, similar to factor_z-scores: Cortex top left, Subcortex bottom left, Association tracts top right, Projection tracts bottom right
# * Embed outputs in a summary .html file
# * Compute mean effect size (mean across scalars of paired Cohen's d) for bilateral ipsi-contra pairs

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import os
import pickle
import sys
from base64 import b64encode
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

try:
    from . import config as cfg
    from . import brain_maps
except ImportError:
    # Allow running as script: python asymmetry_tle.py (from asymmetry_tle dir)
    _pkg_dir = Path(__file__).resolve().parents[0]
    sys.path.insert(0, str(_pkg_dir))
    import config as cfg
    import brain_maps

# Default base directory (project root)
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Cache for GM/WM z-scores so brain maps can be regenerated without re-reading CSVs.
# Backend: joblib (default, fast) or pickle. Alternatives you could add:
# - Parquet: two DataFrames (gm: atlas,region,subject,scalar,z; wm: tract,segment,subject,scalar,z);
#   load with pd.read_parquet(), rebuild dict with zip(df.columns).
# - HDF5: pandas store["gm_z"] = df_gm; same column layout as Parquet.
CACHE_DIR_NAME = "cache"
CACHE_FILENAME = "z_scores.joblib"

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore


def _z_cache_key(
    mni_micro: Path,
    pyafq: Path,
    scalars: List[str],
    subjects: List[str],
    segment_list: List[Tuple[str, List[int]]],
) -> str:
    """Stable hash so cache is invalid when inputs change."""
    blob = pickle.dumps(
        (
            str(mni_micro.resolve()),
            str(pyafq.resolve()),
            tuple(sorted(scalars)),
            tuple(sorted(subjects)),
            tuple((seg, tuple(nodes)) for seg, nodes in segment_list),
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return hashlib.sha256(blob).hexdigest()[:24]


def _load_z_cache(
    cache_path: Path, current_key: str
) -> Optional[Tuple[Dict[Tuple[str, str, str, str], float], Dict[Tuple[str, str, str, str], float]]]:
    """Load (gm_z, wm_z) from cache if valid and key matches. Returns None if missing or invalid."""
    if not cache_path.exists():
        return None
    try:
        data = joblib.load(cache_path) if joblib else pickle.loads(cache_path.read_bytes())
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("cache_key") != current_key or "gm_z" not in data or "wm_z" not in data:
        return None
    return (data["gm_z"], data["wm_z"])


def _save_z_cache(
    cache_path: Path,
    cache_key: str,
    gm_z: Dict[Tuple[str, str, str, str], float],
    wm_z: Dict[Tuple[str, str, str, str], float],
) -> None:
    """Write gm_z and wm_z to cache with key for validation."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cache_key": cache_key, "gm_z": gm_z, "wm_z": wm_z}
    if joblib:
        joblib.dump(payload, cache_path, compress=3)
    else:
        cache_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def _parse_segment(s) -> str:
    if pd.isna(s) or str(s).strip().upper() in ("NA", ""):
        return ""
    return str(s).strip()


def _tract_base(tract_label: str) -> str:
    if tract_label.endswith("_L"):
        return tract_label[:-2]
    if tract_label.endswith("_R"):
        return tract_label[:-2]
    return tract_label


def discover_gm_regions(mni_micro_gam_dir: Path, atlas: str) -> List[str]:
    """List region directory names for an atlas."""
    atlas_dir = mni_micro_gam_dir / atlas
    if not atlas_dir.exists() or not atlas_dir.is_dir():
        return []
    return [d.name for d in atlas_dir.iterdir() if d.is_dir()]


def discover_wm_tracts(pyafq_gam_dir: Path) -> List[str]:
    """List tract directory names."""
    if not pyafq_gam_dir.exists():
        return []
    return [d.name for d in pyafq_gam_dir.iterdir() if d.is_dir()]


def load_gm_z(
    mni_micro_gam_dir: Path,
    atlas: str,
    region: str,
    scalar: str,
    subjects: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Load {scalar}_z per subject for one GM region. Returns dict sub -> z."""
    path = mni_micro_gam_dir / atlas / region / f"{region}_{scalar}_stat-mean_gam.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if f"{scalar}_z" not in df.columns or "sub" not in df.columns:
            return {}
        df["sub"] = df["sub"].astype(str)
        out = dict(zip(df["sub"], df[f"{scalar}_z"]))
        if subjects is not None:
            out = {s: out[s] for s in subjects if s in out}
        return out
    except Exception:
        return {}


def load_wm_z(
    pyafq_gam_dir: Path,
    tract: str,
    scalar: str,
    segment_nodes: List[int],
    subjects: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Load mean z over segment nodes per subject for one tract/scalar. Returns dict sub -> mean_z."""
    path = pyafq_gam_dir / tract / f"{tract}_{scalar}_gam.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        df["sub"] = df["sub"].astype(str)
        z_cols = [f"node{k}_z" for k in segment_nodes if f"node{k}_z" in df.columns]
        if not z_cols:
            return {}
        df["_mean_z"] = df[z_cols].mean(axis=1)
        out = dict(zip(df["sub"], df["_mean_z"]))
        if subjects is not None:
            out = {s: out[s] for s in subjects if s in out}
        return out
    except Exception:
        return {}


def cohens_d_paired(ipsi_vals: List[float], contra_vals: List[float]) -> float:
    """Paired Cohen's d: mean(ipsi - contra) / std(ipsi - contra). Returns NaN if n<2 or std=0."""
    ipsi = np.asarray(ipsi_vals, dtype=float)
    contra = np.asarray(contra_vals, dtype=float)
    valid = np.isfinite(ipsi) & np.isfinite(contra)
    diff = ipsi[valid] - contra[valid]
    n = len(diff)
    if n < 2:
        return float("nan")
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    if std_diff <= 0:
        return float("nan")
    return mean_diff / std_diff


def get_bilateral_tract_pairs(meta: pd.DataFrame, gam_tracts: Set[str]) -> Dict[str, str]:
    """Left tract -> right tract for bilateral pairs present in GAM."""
    left_tracts = meta[meta["hemi"] == "left"]["label"].tolist()
    right_set = set(meta[meta["hemi"] == "right"]["label"].tolist())
    pairs = {}
    for lt in left_tracts:
        rt = lt.replace("_L", "_R")
        if rt in right_set and lt in gam_tracts and rt in gam_tracts:
            pairs[lt] = rt
    return pairs


def get_gm_bilateral_pairs(regions: List[str]) -> List[Tuple[str, str]]:
    """Return (left_region, right_region) for bilateral pairs.
    - 4S atlas: LH_/RH_ or LH-/RH- prefix (e.g. LH_Vis_1 <-> RH_Vis_1).
    - Glasser atlas: Left_/Right_ prefix (e.g. Left_V1 <-> Right_V1).
    """
    region_set = set(regions)
    pairs = []
    # 4S: LH_* <-> RH_*, or LH-* <-> RH-*
    for r in region_set:
        if r.startswith("LH_"):
            right_r = "RH_" + r[3:]
            if right_r in region_set:
                pairs.append((r, right_r))
        elif r.startswith("LH-"):
            right_r = "RH-" + r[3:]
            if right_r in region_set:
                pairs.append((r, right_r))
    # Glasser: Left_* <-> Right_*
    for r in region_set:
        if r.startswith("Left_"):
            right_r = "Right_" + r[5:]
            if right_r in region_set and (r, right_r) not in pairs:
                pairs.append((r, right_r))
    return pairs


def _segment_to_anatomical(tract: str, segment: str, tract_meta: pd.DataFrame) -> str:
    """Convert segment label (end1, end2, core) to anatomical description using tract metadata end1/end2 columns.
    E.g. AF_L + end1 with end1='A' -> 'end-A'; core -> 'core'."""
    if segment == "core":
        return "core"
    if tract_meta.empty or "label" not in tract_meta.columns:
        return segment
    row = tract_meta[tract_meta["label"] == tract]
    if row.empty:
        return segment
    r = row.iloc[0]
    if segment == "end1" and "end1" in r.index:
        val = r["end1"]
        if pd.notna(val) and str(val).strip().upper() not in ("NA", ""):
            return f"end-{str(val).strip()}"
    if segment == "end2" and "end2" in r.index:
        val = r["end2"]
        if pd.notna(val) and str(val).strip().upper() not in ("NA", ""):
            return f"end-{str(val).strip()}"
    return segment


def _summary_table_html(
    rows: List[Tuple[str, float]],
    title: str,
    value_header: str = "Value",
    roi_header: str = "Region",
) -> str:
    """Build GM table with bottom 5 and top 5 rows (by value), displayed in decreasing order; columns roi_header and value_header."""
    if not rows:
        return f'<div class="summary-table-wrap"><p><strong>{title}</strong></p><p>No data</p></div>'
    n = len(rows)
    bottom_5_indices = set(range(min(5, n)))
    top_5_indices = set(range(max(0, n - 5), n))
    indices_to_show = sorted(bottom_5_indices | top_5_indices)
    rows_to_show = [(rows[i], i in top_5_indices, i in bottom_5_indices) for i in indices_to_show]
    rows_to_show.sort(key=lambda x: x[0][1], reverse=True)
    lines = [
        f'<div class="summary-table-wrap"><p><strong>{title}</strong></p>',
        f'<table class="summary-table"><thead><tr><th>{html_module.escape(roi_header)}</th><th>{html_module.escape(value_header)}</th></tr></thead><tbody>',
    ]
    for (label, val), is_top, is_bot in rows_to_show:
        if is_top and is_bot:
            bg = "background: rgba(128,0,128,0.2);"
        elif is_top:
            bg = "background: rgba(255,0,0,0.25);"
        else:
            bg = "background: rgba(0,0,255,0.25);"
        safe_label = html_module.escape(str(label))
        lines.append(f'<tr style="{bg}"><td>{safe_label}</td><td>{val:.4f}</td></tr>')
    lines.append("</tbody></table></div>")
    return "\n".join(lines)


def _summary_table_wm_html(
    rows: List[Tuple[str, str, float]],
    title: str,
    value_header: str,
) -> str:
    """Build WM table with bottom 5 and top 5 rows, displayed in decreasing order; columns Tract, Segment, value_header."""
    if not rows:
        return f'<div class="summary-table-wrap"><p><strong>{title}</strong></p><p>No data</p></div>'
    n = len(rows)
    bottom_5_indices = set(range(min(5, n)))
    top_5_indices = set(range(max(0, n - 5), n))
    indices_to_show = sorted(bottom_5_indices | top_5_indices)
    rows_to_show = [(rows[i], i in top_5_indices, i in bottom_5_indices) for i in indices_to_show]
    rows_to_show.sort(key=lambda x: x[0][2], reverse=True)
    lines = [
        f'<div class="summary-table-wrap"><p><strong>{title}</strong></p>',
        f'<table class="summary-table"><thead><tr><th>Tract</th><th>Segment</th><th>{html_module.escape(value_header)}</th></tr></thead><tbody>',
    ]
    for (tract, segment, val), is_top, is_bot in rows_to_show:
        if is_top and is_bot:
            bg = "background: rgba(128,0,128,0.2);"
        elif is_top:
            bg = "background: rgba(255,0,0,0.25);"
        else:
            bg = "background: rgba(0,0,255,0.25);"
        safe_tract = html_module.escape(str(tract))
        safe_segment = html_module.escape(str(segment))
        lines.append(f'<tr style="{bg}"><td>{safe_tract}</td><td>{safe_segment}</td><td>{val:.4f}</td></tr>')
    lines.append("</tbody></table></div>")
    return "\n".join(lines)


VALUE_HEADERS = {"mean_z": "Mean z", "effect_size": "Paired Cohen's d"}


def _tables_2x2_html(data: Dict[str, List], value_header: str) -> str:
    """Build 2x2 grid: Row 1 = Cortex GM, Association WM; Row 2 = Subcortex GM, Projection WM.
    GM tables use (Region, value_header); WM tables use (Tract, Segment, value_header)."""
    cortex = _summary_table_html(data.get("cortex", []), "Cortex GM", value_header=value_header, roi_header="Region")
    subcortex = _summary_table_html(data.get("subcortex", []), "Subcortex GM", value_header=value_header, roi_header="Region")
    assoc = _summary_table_wm_html(data.get("association", []), "Association WM", value_header)
    proj = _summary_table_wm_html(data.get("projection", []), "Projection WM", value_header)
    return f'<div class="grid-tables-2x2">{cortex}{assoc}{subcortex}{proj}</div>'


COMMUNITY_COLUMNS = ("community_yeo", "community_mesulam", "community_economo")

# Report structure: (section title, metric key, use effect-size grid styling)
REPORT_SECTIONS = [
    ("Mean z", "mean_z", False),
    ("Mean paired Cohen's d (ipsilateral hemisphere only)", "effect_size", True),
]
STRIP_SUFFIXES = ("yeo", "mesulam", "economo")

# Group-first order for report: group id, group label, then metrics
REPORT_GROUPS = [
    ("left_tle", "Left TLE"),
    ("right_tle", "Right TLE"),
]


def _grid_image_paths(output_dir: Path, group: str, metric: str) -> List[Path]:
    """Return 8 paths for 4x2 grid in order: ctx_y, ctx_lr, assoc_y, assoc_lr, sctx_y, sctx_lr, proj_y, proj_lr."""
    b_ctx = output_dir / f"group_{group}_gm_cortex_{metric}"
    b_sctx = output_dir / f"group_{group}_gm_subcortex_{metric}"
    b_assoc = output_dir / f"group_{group}_wm_association_{metric}"
    b_proj = output_dir / f"group_{group}_wm_projection_{metric}"
    return [
        Path(f"{b_ctx}_ctx_y.png"), Path(f"{b_ctx}_ctx_lr.png"),
        Path(f"{b_assoc}_y.png"), Path(f"{b_assoc}_lr.png"),
        Path(f"{b_sctx}_sctx_y.png"), Path(f"{b_sctx}_sctx_lr.png"),
        Path(f"{b_proj}_y.png"), Path(f"{b_proj}_lr.png"),
    ]


def _load_glasser_communities(atlas_tsv_glasser: Path) -> Optional[pd.DataFrame]:
    """Load Glasser TSV and return DataFrame with label and community_yeo, community_mesulam, community_economo."""
    path = Path(atlas_tsv_glasser)
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    required = ["label"] + list(COMMUNITY_COLUMNS)
    if not all(c in df.columns for c in required):
        return None
    out = df[required].copy()
    for c in COMMUNITY_COLUMNS:
        out[c] = out[c].fillna("n/a").astype(str).str.strip()
    return out


def _plot_cortex_community_strips(
    cortex_list: List[Tuple[str, float]],
    glasser_df: pd.DataFrame,
    output_dir: Path,
    group_name: str,
    metric_key: str,
    y_label: str,
) -> None:
    """Create 1x3 strip plots (Yeo, Mesulam, Economo) with labeled mean indicators; save as PNGs."""
    if not cortex_list or glasser_df is None or plt is None:
        return
    label_set = set(glasser_df["label"])
    rows = [{"region": label, "value": val} for label, val in cortex_list if label in label_set]
    if not rows:
        return
    base_name = f"group_{group_name}_cortex_by_{metric_key}"
    for col in COMMUNITY_COLUMNS:
        df = pd.DataFrame(rows).copy()
        df["community"] = df["region"].map(glasser_df.set_index("label")[col])
        df = df.dropna(subset=["community"])
        if df.empty:
            continue
        if metric_key == "effect_size":
            q1, q3 = df["value"].quantile(0.25), df["value"].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                k = cfg.STRIP_PLOT_OUTLIER_IQR_MULTIPLIER
                low, high = q1 - k * iqr, q3 + k * iqr
                df = df[(df["value"] >= low) & (df["value"] <= high)]
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        means = df.groupby("community", sort=False)["value"].mean()
        uniq = means.sort_values(ascending=False).index.tolist()
        x_map = {u: i for i, u in enumerate(uniq)}
        x = df["community"].map(x_map)
        if sns is not None:
            try:
                sns.stripplot(data=df, x="community", y="value", color="0.3", size=3, ax=ax, order=uniq, alpha=0.5)
                ax.set_xticklabels(uniq, rotation=45, ha="right")
            except Exception:
                jitter = np.random.uniform(-0.15, 0.15, size=len(x))
                ax.scatter(x + jitter, df["value"], alpha=0.5, s=20, color="0.3")
                ax.set_xticks(range(len(uniq)))
                ax.set_xticklabels(uniq, rotation=45, ha="right")
        else:
            jitter = np.random.uniform(-0.15, 0.15, size=len(x))
            ax.scatter(x + jitter, df["value"], alpha=0.5, s=20, color="0.3")
            ax.set_xticks(range(len(uniq)))
            ax.set_xticklabels(uniq, rotation=45, ha="right")
        for i, cat in enumerate(uniq):
            mu = means.get(cat, np.nan)
            if pd.notna(mu):
                mu = float(mu)
                ax.scatter(i, mu, marker='D', color='red', s=35, zorder=4, label="Mean" if i == 0 else None)
                ax.text(i, mu, f"μ={mu:.2f}", fontsize=8, color="red", va="center", zorder=5)
        ax.set_ylabel(y_label)
        ax.set_title(col.replace("community_", "").replace("_", " ").title())
        plt.tight_layout()
        suffix = col.replace("community_", "")
        fig.savefig(output_dir / f"{base_name}_{suffix}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def run(
    base_dir: Optional[Path] = None,
    subjects: Optional[List[str]] = None,
    refresh_cache: bool = False,
    no_scalar_reports: bool = False,
    scalars_filter: Optional[List[str]] = None,
) -> None:

    base_dir = Path(base_dir or DEFAULT_PROJECT_ROOT)
    paths = cfg.get_paths(base_dir)
    paths["base_dir"] = base_dir
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    clinical_df = cfg.load_clinical(paths.get("clinical_path"))
    left_tle_subs, right_tle_subs = cfg.get_left_right_tle_subjects(clinical_df, restrict_to_subjects=set(subjects) if subjects else None)
    all_tle = left_tle_subs + right_tle_subs
    if not all_tle:
        print("No left/right TLE subjects found. Exiting.")
        return

    # Here
    mni_micro = paths["mni_micro_gam_dir"]
    pyafq = paths["pyafq_gam_dir"]
    scalars = cfg.get_scalar_labels(paths)
    if not scalars:
        print("No scalar labels from config (scalar_labels_to_filenames.json). Exiting.")
        return

    # GM: cortex = Glasser (all ROIs), subcortex = 4S156 (regions where network_label == "n/a" in atlas_tsv_4s)
    atlas_cortex = "Glasser"
    atlas_subcortex = "4S156"
    cortex_regions = discover_gm_regions(mni_micro, atlas_cortex)
    all_4s_regions = discover_gm_regions(mni_micro, atlas_subcortex)
    subcortex_regions = [
        r for r in cfg.get_4s_subcortical_regions(paths["atlas_tsv_4s"])
        if r in all_4s_regions
    ]

    # WM: load metadata and bilateral pairs
    meta_path = paths["tract_metadata_path"]
    if not meta_path.exists():
        print("Tract metadata not found. Skipping WM.")
        tract_meta = pd.DataFrame()
        bilateral_tracts = {}
        wm_tracts = []
        segment_list: List[Tuple[str, List[int]]] = []
    else:
        tract_meta = pd.read_csv(meta_path)
        tract_meta = tract_meta[tract_meta["hemi"].isin(["left", "right"])]
        tract_meta = tract_meta[tract_meta["profilable"].astype(str).str.upper() == "TRUE"]
        wm_tracts = discover_wm_tracts(pyafq)
        bilateral_tracts = get_bilateral_tract_pairs(tract_meta, set(wm_tracts))
        segment_list = [
            ("end1", cfg.END1_NODES),
            ("core", cfg.CORE_NODES),
            ("end2", cfg.END2_NODES),
        ]
        # Resolve segment name: metadata has end1/end2 as "A","P" etc; we use end1/core/end2 and node ranges
        pass

    tract_to_type: Dict[str, str] = {}
    if not tract_meta.empty and "label" in tract_meta.columns and "type" in tract_meta.columns:
        tract_to_type = dict(zip(tract_meta["label"], tract_meta["type"]))

    def _create_gm_map(scores: Dict, group: str, metric: str, domain: str, title: str, vmin: float, vmax: float, use_abs: bool, **kw) -> None:
        if not scores:
            return
        if domain == "cortex" and not paths["atlas_nifti_glasser"].exists():
            return
        path = report_dir / f"group_{group}_gm_{domain}_{metric}.png"
        nii = paths["atlas_nifti_glasser"] if domain == "cortex" else paths["atlas_nifti_4s"]
        tsv = paths["atlas_tsv_glasser"] if domain == "cortex" else paths["atlas_tsv_4s"]
        brain_maps.create_gm_brain_map(scores, title=title, output_path=str(path), atlas_nifti_path=nii, atlas_tsv_path=tsv, vmin=vmin, vmax=vmax, use_absolute=use_abs, **kw)

    def _create_wm_map(scores: Dict, group: str, metric: str, title: str, vmin: float, vmax: float, use_abs: bool, **kw) -> None:
        if not scores:
            return
        brain_maps.create_wm_brain_map(scores, title=title, output_path_association=str(report_dir / f"group_{group}_wm_association_{metric}.png"), output_path_projection=str(report_dir / f"group_{group}_wm_projection_{metric}.png"), tract_metadata_df=tract_meta, endpoint_nii_dir=paths["endpoint_nii_dir"], vmin=vmin, vmax=vmax, use_absolute=use_abs, **kw)

    # Subject -> ipsi hemisphere (L or R) for TLE
    sub_to_ipsi: Dict[str, str] = {}
    for s in left_tle_subs:
        sub_to_ipsi[s] = "L"
    for s in right_tle_subs:
        sub_to_ipsi[s] = "R"

    cache_dir = output_dir / CACHE_DIR_NAME
    cache_path = cache_dir / CACHE_FILENAME
    cache_key = _z_cache_key(mni_micro, pyafq, scalars, all_tle, segment_list)

    # Load GM/WM z from cache or from CSVs
    gm_z: Dict[Tuple[str, str, str, str], float] = {}
    wm_z: Dict[Tuple[str, str, str, str], float] = {}
    if not refresh_cache:
        cached = _load_z_cache(cache_path, cache_key)
        if cached is not None:
            gm_z, wm_z = cached
            print(f"Loaded z-scores from cache ({len(gm_z)} GM, {len(wm_z)} WM entries)")

    if not gm_z and not wm_z:
        # Load all GM z: (atlas, region, subject, scalar) -> z
        gm_tasks = [(atlas_cortex, r, s) for r in cortex_regions for s in scalars]
        gm_tasks += [(atlas_subcortex, r, s) for r in subcortex_regions for s in scalars]
        for atlas, region, scalar in tqdm(gm_tasks, desc="Loading GM z"):
            for sub, z in load_gm_z(mni_micro, atlas, region, scalar, all_tle).items():
                if pd.notna(z):
                    gm_z[(atlas, region, sub, scalar)] = float(z)

        # Load all WM z: (tract, segment, subject, scalar) -> z
        wm_tasks: List[Tuple[str, str, List[int], str]] = []
        seen_wm_keys: Set[Tuple[str, str, str]] = set()
        for (left_tract, right_tract) in bilateral_tracts.items():
            for seg_name, nodes in segment_list:
                for scalar in scalars:
                    for tract in (left_tract, right_tract):
                        key3 = (tract, seg_name, scalar)
                        if key3 not in seen_wm_keys:
                            seen_wm_keys.add(key3)
                            wm_tasks.append((tract, seg_name, nodes, scalar))
        for tract, seg_name, nodes, scalar in tqdm(wm_tasks, desc="Loading WM z"):
            for sub, z in load_wm_z(pyafq, tract, scalar, nodes, all_tle).items():
                if pd.notna(z):
                    wm_z[(tract, seg_name, sub, scalar)] = float(z)

        if gm_z or wm_z:
            _save_z_cache(cache_path, cache_key, gm_z, wm_z)
            print(f"Saved z-scores to cache: {cache_path}")

    full_gm_z = gm_z
    full_wm_z = wm_z
    scalar_metadata = cfg.load_scalar_metadata(paths)
    all_scalars = scalars

    def _gm_vals(atlas: str, region: str, sub_list: List[str]) -> List[float]:
        vals = [gm_z.get((atlas, region, s, sc), float("nan")) for s in sub_list for sc in scalars]
        return [v for v in vals if pd.notna(v)]

    def _wm_vals(tract: str, segment: str, sub_list: List[str]) -> List[float]:
        vals = [wm_z.get((tract, segment, s, sc), float("nan")) for s in sub_list for sc in scalars]
        return [v for v in vals if pd.notna(v)]

    def gm_region_mean_z(atlas: str, region: str, sub_list: List[str]) -> float:
        v = _gm_vals(atlas, region, sub_list)
        return float(np.nanmean(v)) if v else float("nan")

    def wm_tract_segment_mean_z(tract: str, segment: str, sub_list: List[str]) -> float:
        v = _wm_vals(tract, segment, sub_list)
        return float(np.nanmean(v)) if v else float("nan")

    # Bilateral pairs for GM effect size (cortex + subcortex)
    cortex_pairs = get_gm_bilateral_pairs(cortex_regions)
    subcortex_pairs = get_gm_bilateral_pairs(subcortex_regions)

    def gm_region_mean_paired_cohens_d(atlas: str, ipsi_region: str, contra_region: str, sub_list: List[str]) -> float:
        """Mean (across scalars) of paired Cohen's d for ipsi vs contra. Per scalar: d = mean(ipsi-contra)/std(ipsi-contra)."""
        d_per_scalar = []
        for sc in scalars:
            ipsi_zs = []
            contra_zs = []
            for s in sub_list:
                z_ipsi = gm_z.get((atlas, ipsi_region, s, sc), float("nan"))
                z_contra = gm_z.get((atlas, contra_region, s, sc), float("nan"))
                if pd.notna(z_ipsi) and pd.notna(z_contra):
                    ipsi_zs.append(float(z_ipsi))
                    contra_zs.append(float(z_contra))
            d = cohens_d_paired(ipsi_zs, contra_zs)
            if pd.notna(d):
                d_per_scalar.append(d)
        return float(np.nanmean(d_per_scalar)) if d_per_scalar else float("nan")

    def wm_tract_segment_mean_paired_cohens_d(
        ipsi_tract: str, contra_tract: str, segment: str, sub_list: List[str]
    ) -> float:
        """Mean (across scalars) of paired Cohen's d for ipsi vs contra tract/segment."""
        d_per_scalar = []
        for sc in scalars:
            ipsi_zs = []
            contra_zs = []
            for s in sub_list:
                z_ipsi = wm_z.get((ipsi_tract, segment, s, sc), float("nan"))
                z_contra = wm_z.get((contra_tract, segment, s, sc), float("nan"))
                if pd.notna(z_ipsi) and pd.notna(z_contra):
                    ipsi_zs.append(float(z_ipsi))
                    contra_zs.append(float(z_contra))
            d = cohens_d_paired(ipsi_zs, contra_zs)
            if pd.notna(d):
                d_per_scalar.append(d)
        return float(np.nanmean(d_per_scalar)) if d_per_scalar else float("nan")

    def _build_one_report() -> None:
        """Build one report using report_dir, html_path, report_title, gm_z, wm_z, scalars, csv_dir from closure."""
        groups = [("left_tle", left_tle_subs, "L"), ("right_tle", right_tle_subs, "R")]
        table_data: Dict[str, Dict[str, Dict[str, List[Tuple[str, float]]]]] = {}
        for group_name, sub_list, ipsi_hemi in tqdm(groups, desc="Brain maps"):
            if not sub_list:
                continue
            hemisphere_only = "left" if ipsi_hemi == "L" else "right"

            # ---- Mean z maps (full brain) ----
            cortex_scores_z = {r: gm_region_mean_z(atlas_cortex, r, sub_list) for r in cortex_regions}
            cortex_scores_z = {k: v for k, v in cortex_scores_z.items() if pd.notna(v)}
            subcortex_scores_z = {r: gm_region_mean_z(atlas_subcortex, r, sub_list) for r in subcortex_regions}
            subcortex_scores_z = {k: v for k, v in subcortex_scores_z.items() if pd.notna(v)}
            tract_segment_z: Dict[Tuple[str, str], float] = {}
            if bilateral_tracts:
                for (left_tract, right_tract) in bilateral_tracts.items():
                    for seg_name, _ in segment_list:
                        for tract in (left_tract, right_tract):
                            m = wm_tract_segment_mean_z(tract, seg_name, sub_list)
                            if pd.notna(m):
                                tract_segment_z[(tract, seg_name)] = m
            all_z_vals = list(cortex_scores_z.values()) + list(subcortex_scores_z.values()) + list(tract_segment_z.values())
            if all_z_vals:
                abs_max_z = float(np.max(np.abs(all_z_vals)))
                vmin_z, vmax_z = -abs_max_z, abs_max_z
            else:
                vmin_z, vmax_z = -1.0, 1.0

            _create_gm_map(cortex_scores_z, group_name, "mean_z", "cortex", f"Mean z cortex ({group_name})", vmin_z, vmax_z, False)
            _create_gm_map(subcortex_scores_z, group_name, "mean_z", "subcortex", f"Mean z subcortex ({group_name})", vmin_z, vmax_z, False)

            def _wm_split(ts_dict: Dict[Tuple[str, str], float]) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
                assoc = sorted(
                    [(t, _segment_to_anatomical(t, s, tract_meta), v) for (t, s), v in ts_dict.items() if tract_to_type.get(t) == "association"],
                    key=lambda x: x[2],
                )
                proj = sorted(
                    [(t, _segment_to_anatomical(t, s, tract_meta), v) for (t, s), v in ts_dict.items() if tract_to_type.get(t) == "projection"],
                    key=lambda x: x[2],
                )
                return assoc, proj

            table_data[group_name] = {}
            cortex_list = sorted(cortex_scores_z.items(), key=lambda x: x[1])
            subcortex_list = sorted(subcortex_scores_z.items(), key=lambda x: x[1])
            assoc_z, proj_z = _wm_split(tract_segment_z)
            table_data[group_name]["mean_z"] = {"cortex": cortex_list, "subcortex": subcortex_list, "association": assoc_z, "projection": proj_z}

            # ---- Mean paired Cohen's d (effect size, ipsilateral hemisphere only) ----
            cortex_effect: Dict[str, float] = {}
            for (left_r, right_r) in cortex_pairs:
                ipsi_r = left_r if ipsi_hemi == "L" else right_r
                contra_r = right_r if ipsi_hemi == "L" else left_r
                m = gm_region_mean_paired_cohens_d(atlas_cortex, ipsi_r, contra_r, sub_list)
                if pd.notna(m):
                    cortex_effect[ipsi_r] = m
            subcortex_effect: Dict[str, float] = {}
            for (left_r, right_r) in subcortex_pairs:
                ipsi_r = left_r if ipsi_hemi == "L" else right_r
                contra_r = right_r if ipsi_hemi == "L" else left_r
                m = gm_region_mean_paired_cohens_d(atlas_subcortex, ipsi_r, contra_r, sub_list)
                if pd.notna(m):
                    subcortex_effect[ipsi_r] = m
            tract_segment_effect: Dict[Tuple[str, str], float] = {}
            if bilateral_tracts:
                for (left_tract, right_tract) in bilateral_tracts.items():
                    ipsi_t = left_tract if ipsi_hemi == "L" else right_tract
                    contra_t = right_tract if ipsi_hemi == "L" else left_tract
                    for seg_name, _ in segment_list:
                        m = wm_tract_segment_mean_paired_cohens_d(ipsi_t, contra_t, seg_name, sub_list)
                        if pd.notna(m):
                            tract_segment_effect[(ipsi_t, seg_name)] = m
            all_effect_vals = list(cortex_effect.values()) + list(subcortex_effect.values()) + list(tract_segment_effect.values())
            if all_effect_vals:
                abs_max_effect = float(np.max(np.abs(all_effect_vals)))
                vmin_effect, vmax_effect = -abs_max_effect, abs_max_effect
            else:
                vmin_effect, vmax_effect = -1.0, 1.0

            _create_gm_map(cortex_effect, group_name, "effect_size", "cortex", f"Mean paired Cohen's d cortex ({group_name}, ipsi only)", vmin_effect, vmax_effect, False, hemisphere_only=hemisphere_only, ipsilateral_hemisphere=hemisphere_only)
            _create_gm_map(subcortex_effect, group_name, "effect_size", "subcortex", f"Mean paired Cohen's d subcortex ({group_name}, ipsi only)", vmin_effect, vmax_effect, False, hemisphere_only=hemisphere_only, ipsilateral_hemisphere=hemisphere_only)

            _create_wm_map(tract_segment_z, group_name, "mean_z", f"Mean z ({group_name})", vmin_z, vmax_z, False)
            _create_wm_map(tract_segment_effect, group_name, "effect_size", f"Mean paired Cohen's d ({group_name}, ipsi only)", vmin_effect, vmax_effect, False, ipsilateral_hemisphere=hemisphere_only)

            cortex_effect_list = sorted(cortex_effect.items(), key=lambda x: x[1])
            subcortex_effect_list = sorted(subcortex_effect.items(), key=lambda x: x[1])
            assoc_effect, proj_effect = _wm_split(tract_segment_effect)
            table_data[group_name]["effect_size"] = {"cortex": cortex_effect_list, "subcortex": subcortex_effect_list, "association": assoc_effect, "projection": proj_effect}

            # ---- Labeled colorbars for this group ----
            brain_maps.save_colorbar(
                vmin_z,
                vmax_z,
                str(report_dir / f"group_{group_name}_mean_z_colorbar.png"),
                cmap="RdBu_r",
                label="Mean z",
            )
            brain_maps.save_colorbar(
                vmin_effect,
                vmax_effect,
                str(report_dir / f"group_{group_name}_effect_size_colorbar.png"),
                cmap="RdBu_r",
                label="Mean paired Cohen's d (ipsi−contra)",
            )

        # Cortex-by-community strip plots (1x3: Yeo, Mesulam, Economo) below each summary table
        glasser_df = _load_glasser_communities(paths["atlas_tsv_glasser"])
        metric_specs = [
            ("mean_z", "Mean z"),
            ("effect_size", "Mean paired Cohen's d"),
        ]
        for group_name in ("left_tle", "right_tle"):
            for metric_key, y_label in metric_specs:
                data = table_data.get(group_name, {}).get(metric_key, {})
                cortex_list = data.get("cortex", [])
                _plot_cortex_community_strips(
                    cortex_list,
                    glasser_df,
                    report_dir,
                    group_name,
                    metric_key,
                    y_label,
                )

        def image_to_base64(path: Optional[Path]) -> Optional[str]:
            if path is None or not Path(path).exists():
                return None
            try:
                with open(path, "rb") as f:
                    return "data:image/png;base64," + b64encode(f.read()).decode()
            except Exception:
                return None

        grid_imgs: Dict[Tuple[str, str], List[Optional[str]]] = {}
        for group in ("left_tle", "right_tle"):
            for metric in ("mean_z", "effect_size"):
                grid_imgs[(group, metric)] = [image_to_base64(p) for p in _grid_image_paths(report_dir, group, metric)]
        cbar: Dict[Tuple[str, str], Optional[str]] = {}
        for group in ("left_tle", "right_tle"):
            for metric in ("mean_z", "effect_size"):
                cbar[(group, metric)] = image_to_base64(report_dir / f"group_{group}_{metric}_colorbar.png")
        strip_imgs: Dict[Tuple[str, str], Tuple[Optional[str], ...]] = {}
        for group in ("left_tle", "right_tle"):
            for metric in ("mean_z", "effect_size"):
                base = report_dir / f"group_{group}_cortex_by_{metric}"
                strip_imgs[(group, metric)] = tuple(image_to_base64(Path(f"{base}_{s}.png")) for s in STRIP_SUFFIXES)

        def _strip_1x3_html(yeo_b64: Optional[str], mesulam_b64: Optional[str], economo_b64: Optional[str]) -> str:
            a = lambda src, alt: f'<img src="{src}" alt="{alt}" />' if src else "<p>No data</p>"
            return f'<div class="grid-1x3"><div><p><strong>Yeo</strong></p>{a(yeo_b64, "Yeo")}</div><div><p><strong>Mesulam</strong></p>{a(mesulam_b64, "Mesulam")}</div><div><p><strong>Economo</strong></p>{a(economo_b64, "Economo")}</div></div>'

        labels = ["Cortex GM", "Cortex GM", "Association WM", "Association WM", "Subcortex GM", "Subcortex GM", "Projection WM", "Projection WM"]
        sublabels = ["cor.", "lat."] * 4

        def _toc_html() -> str:
            toc_lines = ['<nav class="toc" aria-label="Table of contents"><h2>Contents</h2><ul>']
            for group_id, group_label in REPORT_GROUPS:
                toc_lines.append(f'<li><a href="#{group_id}">{group_label}</a><ul>')
                for title, key, is_asym in REPORT_SECTIONS:
                    anchor = f"{group_id}_{key}"
                    toc_lines.append(f'<li><a href="#{anchor}">{title}</a></li>')
                toc_lines.append("</ul></li>")
            toc_lines.append("</ul></nav>")
            return "\n".join(toc_lines)

        def _render_group_block(group_id: str, group_label: str) -> str:
            lines = [f'<h2 id="{group_id}" class="group-heading">{group_label}</h2>']
            grid_class = "grid-4x2"
            for title, metric_key, is_asymmetry in REPORT_SECTIONS:
                anchor = f"{group_id}_{metric_key}"
                lines.append(f'<h3 id="{anchor}">{title}</h3>')
                imgs = grid_imgs.get((group_id, metric_key), [None] * 8)
                cells = []
                for i, (src, label, sub) in enumerate(zip(imgs, labels, sublabels)):
                    img = f'<img src="{src}" alt="{label} {sub}" />' if src else "<p>No data</p>"
                    cells.append(f'<div><p><strong>{label}</strong> ({sub})</p>{img}</div>')
                lines.append(f'<div class="{grid_class}">{"".join(cells)}</div>')
                cbar_src = cbar.get((group_id, metric_key))
                cbar_img = f'<img src="{cbar_src}" alt="{title} colorbar ({group_label})" class="cbar" />' if cbar_src else ""
                lines.append(f"<p class=\"colorbar\">{cbar_img}</p>")
                lines.append(_tables_2x2_html(table_data.get(group_id, {}).get(metric_key, {}), VALUE_HEADERS.get(metric_key, "Value")))
                yeo, mesulam, economo = strip_imgs.get((group_id, metric_key), (None, None, None))
                lines.append(_strip_1x3_html(yeo, mesulam, economo))
            return "\n".join(lines)

        toc_html = _toc_html()
        sections_html = "\n".join(_render_group_block(gid, glabel) for gid, glabel in REPORT_GROUPS)
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{html_module.escape(report_title)}</title>
<style>
body {{ font-family: sans-serif; margin: 20px; }}
h1 {{ font-size: 1.5em; }}
h2 {{ font-size: 1.2em; margin-top: 24px; }}
h2.group-heading {{ margin-top: 32px; padding-top: 8px; border-top: 1px solid #ccc; }}
h2.group-heading:first-of-type {{ margin-top: 24px; border-top: none; padding-top: 0; }}
h3 {{ font-size: 1em; margin-top: 16px; }}
nav.toc {{ margin: 20px 0 32px 0; padding: 16px; background: #f8f8f8; border: 1px solid #ddd; max-width: 400px; }}
nav.toc h2 {{ margin: 0 0 10px 0; font-size: 1.1em; }}
nav.toc ul {{ list-style: none; padding-left: 0; margin: 0; }}
nav.toc ul ul {{ padding-left: 1.2em; margin: 4px 0; }}
nav.toc a {{ color: #0066cc; text-decoration: none; }}
nav.toc a:hover {{ text-decoration: underline; }}
.grid-4x2 {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; grid-template-rows: auto auto; gap: 12px; max-width: 1600px; }}
.grid-4x2 > div {{ text-align: center; border: 1px solid #ddd; padding: 8px; }}
.grid-4x2 img {{ max-width: 100%; height: auto; }}
.colorbar {{ margin: 8px 0 16px 0; }}
.colorbar .cbar {{ max-width: 400px; height: auto; display: block; }}
.grid-tables-2x2 {{ display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto auto; gap: 16px; margin-top: 16px; margin-bottom: 24px; width: 100%; }}
.grid-1x3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 12px; margin-bottom: 24px; width: 100%; }}
.grid-1x3 > div {{ text-align: center; border: 1px solid #ddd; padding: 8px; }}
.grid-1x3 img {{ max-width: 100%; height: auto; }}
.summary-table-wrap {{ border: 1px solid #ddd; padding: 10px; background: #fafafa; }}
.summary-table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
.summary-table th, .summary-table td {{ padding: 4px 8px; text-align: left; border: 1px solid #ddd; }}
.summary-table th {{ background: #eee; }}
</style>
</head>
<body>
<h1>{html_module.escape(report_title)}</h1>
<p>Left TLE n={len(left_tle_subs)}, Right TLE n={len(right_tle_subs)}. Mean z (across subjects and scalars). Effect size: mean paired Cohen's d (mean across scalars of within-subject ipsi−contra d), ipsilateral hemisphere only.</p>

{toc_html}

{sections_html}

</body>
</html>
"""
        with open(html_path, "w") as f:
            f.write(html)
        print(f"Wrote {html_path}")

        # Save group summaries as CSV (GM: mean_z, effect_size; WM: same)
        def _write_csv(rows: list, path: Path) -> None:
            if rows:
                pd.DataFrame(rows).to_csv(path, index=False)

        for group_name, sub_list, ipsi_hemi in groups:
            if not sub_list:
                continue
            _write_csv(
                [{"atlas": atlas_cortex, "region": r, "mean_z": gm_region_mean_z(atlas_cortex, r, sub_list)} for r in cortex_regions]
                + [{"atlas": atlas_subcortex, "region": r, "mean_z": gm_region_mean_z(atlas_subcortex, r, sub_list)} for r in subcortex_regions],
                csv_dir / f"group_{group_name}_gm_mean_z.csv",
            )
            effect_rows = []
            for (left_r, right_r) in cortex_pairs + subcortex_pairs:
                ipsi_r = left_r if ipsi_hemi == "L" else right_r
                contra_r = right_r if ipsi_hemi == "L" else left_r
                atlas = atlas_cortex if (left_r, right_r) in cortex_pairs else atlas_subcortex
                m = gm_region_mean_paired_cohens_d(atlas, ipsi_r, contra_r, sub_list)
                effect_rows.append({"atlas": atlas, "region_ipsi": ipsi_r, "region_contra": contra_r, "mean_paired_cohens_d": m})
            _write_csv(effect_rows, csv_dir / f"group_{group_name}_gm_mean_effect_size.csv")
            if bilateral_tracts:
                wm_z_rows = [{"tract": t, "segment": s, "mean_z": wm_tract_segment_mean_z(t, s, sub_list)} for (lt, rt) in bilateral_tracts.items() for s, _ in segment_list for t in (lt, rt)]
                wm_effect_rows = [{"tract_ipsi": lt if ipsi_hemi == "L" else rt, "tract_contra": rt if ipsi_hemi == "L" else lt, "segment": s, "mean_paired_cohens_d": wm_tract_segment_mean_paired_cohens_d(lt if ipsi_hemi == "L" else rt, rt if ipsi_hemi == "L" else lt, s, sub_list)} for (lt, rt) in bilateral_tracts.items() for s, _ in segment_list]
                _write_csv(wm_z_rows, csv_dir / f"group_{group_name}_wm_mean_z.csv")
                _write_csv(wm_effect_rows, csv_dir / f"group_{group_name}_wm_mean_effect_size.csv")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_dir = figures_dir
    html_path = output_dir / "summary.html"
    report_title = "Asymmetry TLE: GAM z-scores and brain maps"
    csv_dir = output_dir
    _build_one_report()

    if not no_scalar_reports:
        scalars_for_reports = [s for s in (scalars_filter or all_scalars) if s in all_scalars]
        if scalars_filter:
            for s in scalars_filter:
                if s not in all_scalars:
                    print(f"Warning: scalar '{s}' not in config, skipping.")
        for scalar in scalars_for_reports:
            gm_z = {k: v for k, v in full_gm_z.items() if k[3] == scalar}
            wm_z = {k: v for k, v in full_wm_z.items() if k[3] == scalar}
            scalars = [scalar]
            scalars_dir = output_dir / "scalars" / scalar
            figures_dir = scalars_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            report_dir = figures_dir
            html_path = scalars_dir / f"TLE_asymmetry_{scalar}.html"
            report_title = scalar_metadata["scalar_to_human"].get(scalar, scalar)
            csv_dir = scalars_dir
            _build_one_report()


def main() -> None:
    parser = argparse.ArgumentParser(description="Asymmetry TLE: GAM z-scores and brain maps for left vs right TLE.")
    parser.add_argument("--base-dir", type=Path, default=None, help="Project base directory")
    parser.add_argument("--subjects", nargs="*", default=None, help="Restrict to these subject IDs")
    parser.add_argument("--refresh-cache", action="store_true", help="Recompute z-scores from CSVs and overwrite cache")
    parser.add_argument("--no-scalar-reports", action="store_true", help="Skip scalar-specific reports; only generate main (averaged) report")
    parser.add_argument("--scalars", nargs="*", default=None, metavar="SCALAR", help="Restrict scalar reports to these scalars (subset of config); default: all included scalars")
    args = parser.parse_args()
    run(
        base_dir=args.base_dir,
        subjects=args.subjects,
        refresh_cache=args.refresh_cache,
        no_scalar_reports=args.no_scalar_reports,
        scalars_filter=args.scalars,
    )


if __name__ == "__main__":
    main()
