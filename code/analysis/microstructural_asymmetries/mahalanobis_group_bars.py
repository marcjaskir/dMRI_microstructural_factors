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
TLE Mahalanobis asymmetry: Cohen's d summarized by region group, Yeo, and Mesulam.

Bar charts (mean Cohen's d ± SEM across ROIs in each group), sorted descending by mean.
Region-group styling matches ``gradients_group-controls`` (white = WM tract families, grey = GM).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd

from microstructural_asymmetry_report_mahalanobis import (
    PROJECT_ROOT,
    _get_4s_subcortical_bases,
    _is_excluded_volumetric_asymmetry_wm_roi,
    _load_glasser_parc,
    _wm_roi_to_tract_segment,
)

ATLAS_DISPLAY: Dict[str, str] = {
    "glasser": "Glasser",
    "4s_subcortex": "4S",
    "hcp1065_thirds": "HCP1065",
}

# Short labels for the region→region-group mapping table only.
REGION_LABEL_TEX_OVERRIDES: Dict[str, str] = {
    "Thalamus-Central Lateral-Lateral Posterior-Medial Pulvinar": (
        "Thalamus-CL-LP-Medial Pulvinar"
    ),
}


def _configure_georgia_font() -> None:
    """Prefer Georgia for matplotlib figures; register TTF on Linux if needed."""
    try:
        fm.findfont("Georgia", fallback_to_default=False)
        return
    except Exception:
        pass
    _seen: set[str] = set()
    _candidates: List[Path] = []
    for _p in (
        PROJECT_ROOT / "data" / "fonts" / "georgia.ttf",
        Path("/usr/share/fonts/truetype/georgia.ttf"),
        Path("/usr/local/share/fonts/truetype/georgia.ttf"),
    ):
        if _p.is_file() and str(_p) not in _seen:
            _seen.add(str(_p))
            _candidates.append(_p)
    for _d in (
        PROJECT_ROOT / "data" / "fonts",
        Path("/usr/share/fonts/truetype"),
        Path("/usr/share/fonts/truetype/msttcorefonts"),
        Path("/usr/share/fonts/truetype/microsoft"),
        Path("/usr/local/share/fonts/truetype/msttcorefonts"),
    ):
        if not _d.is_dir():
            continue
        for _f in sorted(_d.iterdir()):
            if _f.name.lower() != "georgia.ttf" or not _f.is_file():
                continue
            if str(_f) in _seen:
                continue
            _seen.add(str(_f))
            _candidates.append(_f)
    for _f in _candidates:
        if _f.is_file():
            try:
                fm.fontManager.addfont(str(_f))
                break
            except Exception:
                continue
    matplotlib.rcParams["font.family"] = ["Georgia", "DejaVu Serif", "serif"]
    matplotlib.rcParams["font.serif"] = [
        "Georgia",
        "DejaVu Serif",
        "Liberation Serif",
        "Times New Roman",
        "Times",
        "Nimbus Roman",
    ]
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
    matplotlib.rcParams["axes.unicode_minus"] = False


def _display_atlas_label(atlas_key: str) -> str:
    key = str(atlas_key).strip()
    if not key:
        return ""
    return ATLAS_DISPLAY.get(key, key.replace("_", " "))


def _hcp1065_roi_abbreviation(roi_id: str) -> str:
    """HCP1065 tract label for mapping tables (e.g. ``C_FPH_A`` -> ``C_FPH``)."""
    tract, _seg = _wm_roi_to_tract_segment(str(roi_id).strip())
    return tract


def _mapping_table_abbreviation(roi_id: str, roi_type: str, atlas_key: str) -> str:
    """Abbreviation column: HCP1065 WM only; empty for Glasser and 4S."""
    if str(roi_type).strip() == "wm" or atlas_key == "hcp1065_thirds":
        return _hcp1065_roi_abbreviation(roi_id)
    return ""

# --- Region groups (aligned with gradient_lib/region_groups.py) ---
GLASSER_LOBE_DISPLAY: Dict[str, str] = {
    "Fr": "Frontal",
    "Par": "Parietal",
    "Temp": "Temporal",
    "Occ": "Occipital",
    "Ins": "Insula",
}

SUBCORTEX_ROI_TO_GROUP: Dict[str, str] = {
    "Pu": "Basal ganglia",
    "Ca": "Basal ganglia",
    "NAC": "Basal ganglia",
    "VeP": "Basal ganglia",
    "GPe": "Basal ganglia",
    "GPi": "Basal ganglia",
    "EXA": "Basal ganglia",
    "STN": "Basal ganglia",
    "STH": "Basal ganglia",
    "SNc_PBP_VTA": "Midbrain",
    "SNr": "Midbrain",
    "RN": "Midbrain",
    "HN": "Midbrain",
    "HTH": "Limbic subcortex",
    "MN": "Limbic subcortex",
    "Pulvinar": "Thalamus",
    "Anterior": "Thalamus",
    "Medio_Dorsal": "Thalamus",
    "Ventral_Latero_Dorsal": "Thalamus",
    "Central_Lateral-Lateral_Posterior-Medial_Pulvinar": "Thalamus",
    "Ventral_Anterior": "Thalamus",
    "Ventral_Latero_Ventral": "Thalamus",
    "Hippocampus": "Limbic subcortex",
    "Amygdala": "Limbic subcortex",
    "Cerebellar_Region1": "Cerebellum",
    "Cerebellar_Region2": "Cerebellum",
    "Cerebellar_Region3": "Cerebellum",
    "Cerebellar_Region4": "Cerebellum",
    "Cerebellar_Region5": "Cerebellum",
    "Cerebellar_Region6": "Cerebellum",
    "Cerebellar_Region7": "Cerebellum",
    "Cerebellar_Region8": "Cerebellum",
    "Cerebellar_Region9": "Cerebellum",
    "Cerebellar_Region10": "Cerebellum",
}

TRACT_SEGMENT_GROUPINGS: Dict[str, Tuple[str, ...]] = {
    "Mesial limbic": ("C_FP_L", "C_FPH_L", "C_PH_L", "C_PO_L", "UF_L", "F_L"),
    "Occipital association": ("IFOF_L", "ILF_L", "VOF_L"),
    "Parietal association": ("PAT_L", "SLF1_L", "MdLF_L"),
    "Thalamic radiations": ("TR_A_L", "TR_P_L", "TR_S_L", "OR_L"),
    "Corticostriatal": ("CS_A_L", "CS_P_L", "CS_S_L"),
    "Ascending tracts": ("CPT_F_L", "CPT_O_L", "CPT_P_L", "CST_L", "ML_L"),
}

REGION_GROUP_SORT_ORDER: Tuple[str, ...] = (
    *tuple(TRACT_SEGMENT_GROUPINGS.keys()),
    "Basal ganglia",
    "Thalamus",
    "Midbrain",
    "Limbic subcortex",
    "Cerebellum",
    "Frontal lobe",
    "Parietal lobe",
    "Temporal lobe",
    "Occipital lobe",
    "Insula",
)

WM_REGION_GROUP_NAMES = frozenset(TRACT_SEGMENT_GROUPINGS.keys())
GM_REGION_GROUP_NAMES = frozenset(
    {
        "Basal ganglia",
        "Thalamus",
        "Midbrain",
        "Limbic subcortex",
        "Cerebellum",
        "Frontal lobe",
        "Parietal lobe",
        "Temporal lobe",
        "Occipital lobe",
        "Insula",
    }
)

REGION_GROUP_WM_FACE = "#ffffff"
REGION_GROUP_GM_FACE = "#9e9e9e"
REGION_GROUP_BAR_EDGE_WIDTH = 2.0
GROUP_BAR_TICK_FONTSIZE = 13
GROUP_BAR_AXIS_LABEL_FONTSIZE = 15
GROUP_BAR_TITLE_FONTSIZE = 17

YEO_DISPLAY: Dict[str, str] = {
    "visual": "Visual",
    "somatosensory": "Somatosensory",
    "somatomotor": "Somatosensory",
    "dorsal attention": "Dorsal Attention",
    "ventral attention": "Ventral Attention",
    "limbic": "Limbic",
    "frontoparietal": "Frontoparietal",
    "default": "Default Mode",
    "default mode": "Default Mode",
}

MESULAM_DISPLAY: Dict[str, str] = {
    "granular": "Granular",
    "agranular": "Agranular",
    "parietal": "Parietal",
    "unimodal": "Unimodal",
    "heteromodal": "Heteromodal",
    "paralimbic": "Paralimbic",
    "idiotypic": "Idiotypic",
}


def _cortical_lobe_region_group_label(lobe_code: str) -> Optional[str]:
    code = str(lobe_code).strip()
    d = GLASSER_LOBE_DISPLAY.get(code)
    if d is None:
        return None
    if code == "Ins":
        return "Insula"
    return f"{d} lobe"


def tract_base_to_functional_group_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for gname, tracts in TRACT_SEGMENT_GROUPINGS.items():
        for t in tracts:
            out[str(t)] = gname
            if t.endswith("_L"):
                out[t[:-2] + "_R"] = gname
    return out


def load_cortical_lobe_region_group_by_roi(tractometry_root: Path) -> Dict[str, str]:
    p = tractometry_root / "data/atlases/Glasser/glasser_additional_metadata.csv"
    if not p.is_file():
        return {}
    df = pd.read_csv(p)
    if "Lobe" not in df.columns or "region" not in df.columns:
        return {}
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        region = str(row["region"]).strip()
        if not region:
            continue
        name = _cortical_lobe_region_group_label(row["Lobe"])
        if name is None:
            continue
        out[region] = name
        out[f"Left_{region}"] = name
        out[f"Right_{region}"] = name
    return out


def _strip_4s_hemi_prefixes(label: str) -> str:
    s = str(label)
    for pref in ("LH-", "RH-", "LH_", "RH_"):
        if s.startswith(pref):
            return s[len(pref) :]
    return s


def _glasser_suffix(n: str) -> str:
    t = str(n)
    for p in ("Left_", "Right_"):
        if t.startswith(p):
            return t[len(p) :]
    return t


def _wm_tract_hemi_label_for_group(roi_id: str) -> Optional[str]:
    """``AF_core`` / ``TR_A_P`` -> ``AF_L`` / ``TR_A_L`` for tract-family lookup."""
    tract_with_hemi, _seg = _wm_roi_to_tract_segment(str(roi_id))
    t = str(tract_with_hemi).strip()
    if t.endswith("_L") or t.endswith("_R"):
        return t
    return f"{t}_L"


def region_group_for_cohens_row(
    roi_id: str,
    roi_type: str,
    atlas: str,
    *,
    tract_to_group: Dict[str, str],
    cortical_by_roi: Dict[str, str],
    subcortical_bases: set,
) -> Optional[str]:
    rid = str(roi_id).strip()
    if roi_type == "wm":
        if _is_excluded_volumetric_asymmetry_wm_roi(rid):
            return None
        tract_lab = _wm_tract_hemi_label_for_group(rid)
        if tract_lab and tract_lab in tract_to_group:
            return tract_to_group[tract_lab]
        return None
    if roi_type == "subcortical_gm" and atlas == "4s_subcortex":
        bare = _strip_4s_hemi_prefixes(rid)
        return SUBCORTEX_ROI_TO_GROUP.get(bare) or SUBCORTEX_ROI_TO_GROUP.get(rid)
    if roi_type == "cortical_gm" and atlas == "glasser":
        gs = _glasser_suffix(rid)
        for k in (rid, gs, f"Left_{gs}", f"Right_{gs}"):
            g = cortical_by_roi.get(k)
            if g is not None:
                return g
    return None


def region_group_bar_facecolor(label: str) -> str:
    s = str(label)
    if s in WM_REGION_GROUP_NAMES:
        return REGION_GROUP_WM_FACE
    return REGION_GROUP_GM_FACE


def _glasser_base_from_roi_id(roi_id: str) -> str:
    s = str(roi_id).strip()
    if s.startswith("Left_"):
        return s[5:].strip()
    if s.startswith("Right_"):
        return s[6:].strip()
    return s


def _community_for_cortex_row(
    roi_id: str,
    roi_type: str,
    atlas: str,
    glasser_parc: pd.DataFrame,
    col: str,
) -> Optional[str]:
    if roi_type != "cortical_gm" or atlas != "glasser" or glasser_parc is None or glasser_parc.empty:
        return None
    base = _glasser_base_from_roi_id(roi_id)
    if "base" not in glasser_parc.columns or col not in glasser_parc.columns:
        return None
    hit = glasser_parc.loc[glasser_parc["base"] == base, col]
    if hit.empty:
        return None
    val = str(hit.iloc[0]).strip()
    if not val or val.lower() in ("nan", "n/a", "none"):
        return None
    return val


def _prepare_group_bars(
    cohens_df: pd.DataFrame,
    group_fn,
) -> Tuple[List[str], np.ndarray, np.ndarray, List[int]]:
    """Mean ± SEM of Cohen's d per group label; sort descending by mean."""
    grouped: Dict[str, List[float]] = {}
    for _, row in cohens_df.iterrows():
        lab = group_fn(row)
        if lab is None:
            continue
        d = row.get("cohens_d")
        if d is None or not np.isfinite(float(d)):
            continue
        grouped.setdefault(str(lab), []).append(float(d))
    if not grouped:
        return [], np.asarray([]), np.asarray([]), []
    labels_all = list(grouped.keys())
    means_all = np.array([float(np.mean(grouped[lab])) for lab in labels_all], dtype=np.float64)
    sems_all = np.array(
        [
            float(np.std(grouped[lab], ddof=1) / np.sqrt(len(grouped[lab])))
            if len(grouped[lab]) > 1
            else 0.0
            for lab in labels_all
        ],
        dtype=np.float64,
    )
    counts_all = [len(grouped[lab]) for lab in labels_all]
    order = np.argsort(-means_all)
    labels = [labels_all[i] for i in order]
    means = means_all[order]
    sems = sems_all[order]
    counts = [counts_all[i] for i in order]
    return labels, means, sems, counts


def _display_yeo(label: str) -> str:
    key = str(label).strip().lower()
    return YEO_DISPLAY.get(key, str(label).replace("_", " ").title())


def _display_mesulam(label: str) -> str:
    key = str(label).strip().lower()
    return MESULAM_DISPLAY.get(key, str(label).replace("_", " ").title())


def _plot_bars_ax(
    ax: plt.Axes,
    labels: List[str],
    means: np.ndarray,
    sems: np.ndarray,
    *,
    title: str,
    facecolors: Optional[List[str]] = None,
    show_ylabel: bool,
) -> None:
    font_family = "Georgia"
    if not labels:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontfamily=font_family,
            fontsize=GROUP_BAR_TITLE_FONTSIZE,
        )
        ax.set_title(title, fontfamily=font_family, fontsize=GROUP_BAR_TITLE_FONTSIZE)
        return
    x = np.arange(len(labels))
    if facecolors is None:
        facecolors = [REGION_GROUP_GM_FACE] * len(labels)
    ax.bar(
        x,
        means,
        yerr=sems,
        color=facecolors,
        edgecolor="k",
        linewidth=REGION_GROUP_BAR_EDGE_WIDTH,
        capsize=3,
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        fontfamily=font_family,
        fontsize=GROUP_BAR_TICK_FONTSIZE,
    )
    if show_ylabel:
        ax.set_ylabel(
            "Cohen's d",
            fontfamily=font_family,
            fontsize=GROUP_BAR_AXIS_LABEL_FONTSIZE,
        )
    ax.set_title(title, fontfamily=font_family, fontsize=GROUP_BAR_TITLE_FONTSIZE)
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=GROUP_BAR_TICK_FONTSIZE)
    for tick in ax.get_yticklabels() + ax.get_xticklabels():
        tick.set_fontfamily(font_family)


def _latex_escape(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    repl = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "^": "\\textasciicircum{}",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _display_region_label(
    roi_id: str,
    roi_type: str,
    atlas: str,
    tractometry_root: Path,
) -> str:
    """Human-readable region label for mapping tables."""
    from microstructural_asymmetry_report_mahalanobis import (
        _format_4s_subcortex_tex_label,
        _format_tex_wm_thirds_label,
        _hcp1065_segment_human,
        _load_4s_thalamus_roi_bases,
        _load_glasser_additional_metadata,
        _load_tract_label_to_pretty_name,
        _wm_roi_to_tract_segment as _wm_seg,
    )

    rid = str(roi_id).strip()
    if roi_type == "wm":
        return _format_tex_wm_thirds_label(rid, _load_tract_label_to_pretty_name())
    if roi_type == "subcortical_gm" and atlas == "4s_subcortex":
        bare = _strip_4s_hemi_prefixes(rid)
        return _format_4s_subcortex_tex_label(bare, _load_4s_thalamus_roi_bases())
    if roi_type == "cortical_gm" and atlas == "glasser":
        glasser_add = _load_glasser_additional_metadata()
        base = _glasser_base_from_roi_id(rid)
        if not glasser_add.empty and base in glasser_add.index:
            ser = glasser_add.loc[base]
            if isinstance(ser, pd.DataFrame):
                ser = ser.iloc[0]
            try:
                v = ser["regionLongName"]
            except (KeyError, TypeError, IndexError):
                v = None
            if v is not None and pd.notna(v) and str(v).strip():
                name = str(v).strip()
                if name == "Hippocampus":
                    return "Parahippocampal"
                return name
        return base.replace("_", " ")
    tract_base, segment = _wm_seg(rid)
    if segment:
        return f"{tract_base.replace('_', ' ')} — {_hcp1065_segment_human(segment)}"
    return rid.replace("_", " ")


def build_region_to_region_group_table(
    cohens_df: pd.DataFrame,
    tractometry_root: Path,
) -> pd.DataFrame:
    """One row per ROI with a region-group assignment (Mahalanobis Cohen's d cohort)."""
    if cohens_df.empty:
        return pd.DataFrame(columns=["region_group", "region", "roi_id", "tissue", "atlas"])

    tract_to_group = tract_base_to_functional_group_map()
    cortical_by_roi = load_cortical_lobe_region_group_by_roi(tractometry_root)
    subcortical_bases = _get_4s_subcortical_bases()
    rows: List[dict] = []
    for _, row in cohens_df.iterrows():
        roi_id = str(row["roi_id"])
        roi_type = str(row["roi_type"])
        atlas = str(row.get("atlas", ""))
        rg = region_group_for_cohens_row(
            roi_id,
            roi_type,
            atlas,
            tract_to_group=tract_to_group,
            cortical_by_roi=cortical_by_roi,
            subcortical_bases=subcortical_bases,
        )
        if rg is None:
            continue
        tissue = "WM" if roi_type == "wm" else "GM"
        atlas_key = atlas if atlas else ("hcp1065_thirds" if roi_type == "wm" else "")
        disp = _display_region_label(roi_id, roi_type, atlas_key, tractometry_root)
        disp = REGION_LABEL_TEX_OVERRIDES.get(disp, disp)
        rows.append(
            {
                "region_group": rg,
                "region": disp,
                "abbreviation": _mapping_table_abbreviation(roi_id, roi_type, atlas_key),
                "roi_id": roi_id,
                "tissue": tissue,
                "atlas": atlas_key,
                "atlas_display": _display_atlas_label(atlas_key),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["region_group", "region", "roi_id", "tissue", "atlas"])
    out = pd.DataFrame(rows)
    rg_rank = {g: i for i, g in enumerate(REGION_GROUP_SORT_ORDER)}
    out["_rg_ord"] = out["region_group"].map(lambda g: rg_rank.get(g, 999))
    out["_reg_sort"] = out["region"].astype(str).str.strip().str.lower()
    out = out.sort_values(["_rg_ord", "_reg_sort"], kind="mergesort").drop(
        columns=["_rg_ord", "_reg_sort"]
    )
    return out.reset_index(drop=True)


def save_region_group_mapping_tex(
    cohens_df: pd.DataFrame,
    out_path: Path,
    tractometry_root: Path,
    *,
    csv_path: Optional[Path] = None,
) -> Optional[Path]:
    """Write longtable: Region group | Region | Abbreviation | Atlas (sorted by group, then region A–Z)."""
    df = build_region_to_region_group_table(cohens_df, tractometry_root)
    out_path = Path(out_path)
    if csv_path is None:
        csv_path = out_path.with_suffix(".csv")
    if df.empty:
        return None
    df.to_csv(csv_path, index=False)

    col_spec = (
        r"@{}>{\raggedright\arraybackslash}p{0.22\linewidth}"
        r">{\raggedright\arraybackslash}p{0.50\linewidth}"
        r">{\raggedright\arraybackslash}p{0.12\linewidth}"
        r">{\raggedright\arraybackslash}p{0.14\linewidth}@{}"
    )
    header_line = (
        r"\mbox{Region group} & \mbox{Region} & \mbox{Abbreviation} & \mbox{Atlas} \\"
    )
    header_block = [r"\toprule", header_line, r"\midrule"]
    lines = [
        "% Requires \\usepackage{array,booktabs,longtable} in the main document.",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{longtable}{" + col_spec + "}",
        *header_block,
        r"\endfirsthead",
        *header_block,
        r"\endhead",
        r"\midrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, row in df.iterrows():
        rg_tex = _latex_escape(str(row["region_group"]))
        # Allow line breaks in the Region column (avoid \mbox overlap into Abbreviation).
        reg_tex = _latex_escape(str(row["region"]))
        abbrev = str(row.get("abbreviation", "")).strip()
        abbrev_tex = _latex_escape(abbrev) if abbrev else ""
        atlas_tex = rf"\mbox{{{_latex_escape(str(row.get('atlas_display', row['atlas'])))}}}"
        lines.append(f"{rg_tex} & {reg_tex} & {abbrev_tex} & {atlas_tex} \\\\")
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")
    return out_path


def build_group_summary_tables(
    cohens_df: pd.DataFrame,
    tractometry_root: Path,
) -> Dict[str, pd.DataFrame]:
    """Return summary DataFrames keyed by ``region_group``, ``yeo``, ``mesulam``."""
    tract_to_group = tract_base_to_functional_group_map()
    cortical_by_roi = load_cortical_lobe_region_group_by_roi(tractometry_root)
    subcortical_bases = _get_4s_subcortical_bases()
    glasser_parc = _load_glasser_parc()

    def _rg(row: pd.Series) -> Optional[str]:
        return region_group_for_cohens_row(
            row["roi_id"],
            row["roi_type"],
            str(row.get("atlas", "")),
            tract_to_group=tract_to_group,
            cortical_by_roi=cortical_by_roi,
            subcortical_bases=subcortical_bases,
        )

    def _yeo(row: pd.Series) -> Optional[str]:
        return _community_for_cortex_row(
            row["roi_id"], row["roi_type"], str(row.get("atlas", "")), glasser_parc, "yeo"
        )

    def _mes(row: pd.Series) -> Optional[str]:
        return _community_for_cortex_row(
            row["roi_id"], row["roi_type"], str(row.get("atlas", "")), glasser_parc, "mesulam"
        )

    out: Dict[str, pd.DataFrame] = {}
    for key, fn, disp in (
        ("region_group", _rg, lambda s: s),
        ("yeo", _yeo, _display_yeo),
        ("mesulam", _mes, _display_mesulam),
    ):
        labels, means, sems, counts = _prepare_group_bars(cohens_df, fn)
        if not labels:
            out[key] = pd.DataFrame(columns=["group", "mean_cohens_d", "sem", "n_rois"])
            continue
        out[key] = pd.DataFrame(
            {
                "group": [disp(l) for l in labels],
                "mean_cohens_d": means,
                "sem": sems,
                "n_rois": counts,
            }
        )
    return out


def plot_tle_cohens_d_region_yeo_mesulam_bars(
    cohens_df: pd.DataFrame,
    out_dir: Path,
    tractometry_root: Path,
    *,
    suffix: str = "",
) -> Optional[Path]:
    """
    1×3 bar figure: region groups | Yeo (cortex) | Mesulam (cortex).

    Writes ``plot_tle_cohens_d_region_yeo_mesulam{suffix}.png`` and CSV summaries.
    """
    if cohens_df.empty:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = build_group_summary_tables(cohens_df, tractometry_root)
    for stem, df in tables.items():
        csv_path = out_dir.parent / f"summary_cohens_d_by_{stem}{suffix}.csv"
        df.to_csv(csv_path, index=False)

    tract_to_group = tract_base_to_functional_group_map()
    cortical_by_roi = load_cortical_lobe_region_group_by_roi(tractometry_root)
    subcortical_bases = _get_4s_subcortical_bases()
    glasser_parc = _load_glasser_parc()

    def _rg(row: pd.Series) -> Optional[str]:
        return region_group_for_cohens_row(
            row["roi_id"],
            row["roi_type"],
            str(row.get("atlas", "")),
            tract_to_group=tract_to_group,
            cortical_by_roi=cortical_by_roi,
            subcortical_bases=subcortical_bases,
        )

    def _yeo(row: pd.Series) -> Optional[str]:
        v = _community_for_cortex_row(
            row["roi_id"], row["roi_type"], str(row.get("atlas", "")), glasser_parc, "yeo"
        )
        return _display_yeo(v) if v is not None else None

    def _mes(row: pd.Series) -> Optional[str]:
        v = _community_for_cortex_row(
            row["roi_id"], row["roi_type"], str(row.get("atlas", "")), glasser_parc, "mesulam"
        )
        return _display_mesulam(v) if v is not None else None

    rg_labels, rg_means, rg_sems, _ = _prepare_group_bars(cohens_df, _rg)
    yeo_labels, yeo_means, yeo_sems, _ = _prepare_group_bars(cohens_df, _yeo)
    mes_labels, mes_means, mes_sems, _ = _prepare_group_bars(cohens_df, _mes)

    _configure_georgia_font()

    # Panel widths 50% : 30% : 20% so bar thickness is more comparable across panels.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5),
        gridspec_kw={"width_ratios": [5, 3, 2]},
    )
    rg_colors = [region_group_bar_facecolor(l) for l in rg_labels] if rg_labels else None
    _plot_bars_ax(
        axes[0],
        rg_labels,
        rg_means,
        rg_sems,
        title="Region groups",
        facecolors=rg_colors,
        show_ylabel=True,
    )
    _plot_bars_ax(
        axes[1],
        yeo_labels,
        yeo_means,
        yeo_sems,
        title="Yeo functional network (cortex)",
        show_ylabel=False,
    )
    _plot_bars_ax(
        axes[2],
        mes_labels,
        mes_means,
        mes_sems,
        title="Mesulam cytoarchitecture community (cortex)",
        show_ylabel=False,
    )
    fig.tight_layout()
    out_path = out_dir / f"plot_tle_cohens_d_region_yeo_mesulam{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path
