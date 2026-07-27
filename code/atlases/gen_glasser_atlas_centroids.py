#!/usr/bin/env python3
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
Compute MNI-space centroid (mm) per Glasser parcel from the discrete segmentation NIfTI.

Joins parcel names from atlas-Glasser_dseg.tsv (``index`` -> ``label``).
Adds axis ranks (A-P, D-V, M-L, S-A) and writes CSV: label, x, y, z, rank_ap, rank_dv,
rank_ml, rank_sa.

Optionally saves nilearn glass-brain PNGs per axis to verify encoding.
Optionally saves additional glass-brain figures coloring parcels by Economo,
Mesulam, and Yeo columns from glasser_parc.csv (with legends).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import nibabel as nib
    from nibabel.affines import apply_affine
except ImportError as e:
    print("This script requires nibabel.", file=sys.stderr)
    raise SystemExit(1) from e

DEFAULT_PROJECT_ROOT = project_root()
DEFAULT_ATLAS_REL = Path("data/atlases/Glasser/atlas-Glasser_space-MNI152NLin2009cAsym_res-1_dseg.nii.gz")
DEFAULT_TSV_REL = Path("data/atlases/Glasser/atlas-Glasser_dseg.tsv")
DEFAULT_OUT_REL = Path("data/atlases/Glasser/glasser_centroids.csv")
DEFAULT_PARC_CSV_REL = Path("data/atlases/Glasser/glasser_parc.csv")
DEFAULT_SA_RANKS_REL = Path(
    "data/atlases/S-A_ArchetypalAxis/Glasser360_MMP/"
    "Sensorimotor_Association_Axis_AverageRanks.csv"
)


def glasser_label_to_region(label: str) -> str:
    """Strip Left_/Right_ prefix for joining to S-A table ``region`` column."""
    s = str(label).strip()
    if s.startswith("Left_"):
        return s[5:]
    if s.startswith("Right_"):
        return s[6:]
    return s


def glasser_label_to_parc_region(label: str) -> str:
    """Map Left_7Pm / Right_V1 to 7Pm_L / V1_R for glasser_parc ``region`` column."""
    s = str(label).strip()
    if s.startswith("Left_"):
        return s[5:] + "_L"
    if s.startswith("Right_"):
        return s[6:] + "_R"
    return s


def load_index_to_label(tsv_path: Path) -> Dict[int, str]:
    df = pd.read_csv(tsv_path, sep="\t")
    if "index" not in df.columns or "label" not in df.columns:
        raise ValueError(f"TSV must have 'index' and 'label' columns: {tsv_path}")
    out: Dict[int, str] = {}
    for _, row in df.iterrows():
        idx = int(row["index"])
        lab = str(row["label"]).strip()
        out[idx] = lab
    return out


def load_sa_region_to_final_rank(sa_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(sa_csv)
    if "region" not in df.columns or "final.rank" not in df.columns:
        raise ValueError(
            f"S-A CSV must have 'region' and 'final.rank' columns: {sa_csv}"
        )
    return dict(zip(df["region"].astype(str).str.strip(), df["final.rank"].astype(float)))


def add_axis_ranks(df: pd.DataFrame, sa_csv: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    rank_ap: 1 = most anterior (largest MNI y); higher rank = more posterior.
    rank_dv: 1 = most dorsal (largest MNI z); higher rank = more ventral.
    rank_ml: 1 = most mesial (smallest |x|); higher rank = more lateral.
    rank_sa: Glasser360 S-A ``final.rank`` (same for L/R); float, may include ties as .5.
    """
    warnings: List[str] = []
    out = df.copy()
    out["rank_ap"] = out["y"].rank(method="min", ascending=False).astype(int)
    out["rank_dv"] = out["z"].rank(method="min", ascending=False).astype(int)
    out["rank_ml"] = out["x"].abs().rank(method="min", ascending=True).astype(int)

    region_to_sa = load_sa_region_to_final_rank(sa_csv)
    warned_region: set = set()

    def lookup_sa(lab: str) -> float:
        reg = glasser_label_to_region(lab)
        if reg not in region_to_sa:
            if reg not in warned_region:
                warnings.append(f"No S-A final.rank for region {reg!r}")
                warned_region.add(reg)
            return float("nan")
        return float(region_to_sa[reg])

    out["rank_sa"] = out["label"].map(lambda lb: lookup_sa(str(lb)))
    return out, warnings


def centroids_from_dseg(
    data: np.ndarray,
    affine: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Map parcel_id -> (3,) world mm centroid; excludes label 0."""
    lab = np.rint(data).astype(np.int32)
    ids = np.unique(lab)
    ids = ids[ids > 0]
    result: Dict[int, np.ndarray] = {}
    for pid in ids:
        pid = int(pid)
        ijk = np.argwhere(lab == pid).astype(np.float64)
        if ijk.size == 0:
            continue
        ijk += 0.5
        xyz = apply_affine(affine, ijk)
        result[pid] = np.nanmean(xyz, axis=0)
    return result


def build_rows(
    id_to_centroid: Dict[int, np.ndarray],
    index_to_label: Dict[int, str],
) -> Tuple[List[Tuple[int, str, float, float, float]], List[str]]:
    """Returns sorted rows (index, label, x, y, z) and warning lines."""
    warnings: List[str] = []
    rows: List[Tuple[int, str, float, float, float]] = []

    for pid, xyz in id_to_centroid.items():
        name = index_to_label.get(pid)
        if name is None:
            warnings.append(
                f"Parcel id {pid} present in NIfTI but missing from TSV; skipping."
            )
            continue
        rows.append(
            (pid, name, float(xyz[0]), float(xyz[1]), float(xyz[2]))
        )

    seen = set(id_to_centroid.keys())
    for idx in sorted(index_to_label.keys()):
        if idx <= 0:
            continue
        if idx not in seen:
            warnings.append(
                f"TSV index {idx} ({index_to_label[idx]!r}) has no voxels in NIfTI; omitting."
            )

    rows.sort(key=lambda r: r[0])
    return rows, warnings


def stat_map_from_label_ranks(
    lab_data: np.ndarray,
    affine: np.ndarray,
    index_to_label: Dict[int, str],
    label_to_value: Dict[str, float],
) -> nib.Nifti1Image:
    """Paint each parcel with a scalar from label_to_value (skips missing / NaN)."""
    lab = np.rint(lab_data).astype(np.int32)
    out = np.zeros(lab_data.shape, dtype=np.float32)
    for pid in np.unique(lab):
        if pid <= 0:
            continue
        name = index_to_label.get(int(pid))
        if name is None:
            continue
        v = label_to_value.get(name)
        if v is None or not np.isfinite(v):
            continue
        out[lab == pid] = np.float32(v)
    return nib.Nifti1Image(out, affine)


def save_rank_axis_plots(
    lab_data: np.ndarray,
    affine: np.ndarray,
    index_to_label: Dict[int, str],
    df: pd.DataFrame,
    plot_dir: Path,
) -> List[str]:
    """Glass-brain PNG per axis; returns list of written paths or notes."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_msgs: List[str] = []
    try:
        from nilearn import plotting as niplot
    except ImportError:
        return ["nilearn not installed; skipping rank-axis brain plots"]

    axes = [
        ("rank_ap", "glasser_rank_axis_ap.png", "A–P rank (1=anterior, high y)"),
        ("rank_dv", "glasser_rank_axis_dv.png", "D–V rank (1=dorsal, high z)"),
        ("rank_ml", "glasser_rank_axis_ml.png", "M–L rank (1=mesial, low |x|)"),
        ("rank_sa", "glasser_rank_axis_sa.png", "S–A final.rank (Glasser360 table)"),
    ]
    lut = df.set_index("label")
    for col, fname, title in axes:
        label_to_value: Dict[str, float] = {}
        for lab in df["label"].astype(str):
            v = lut.loc[lab, col]
            if pd.isna(v):
                continue
            label_to_value[str(lab)] = float(v)
        stat_img = stat_map_from_label_ranks(
            lab_data, affine, index_to_label, label_to_value
        )
        data = np.asanyarray(stat_img.dataobj)
        positive = data[data > 0]
        if positive.size == 0:
            out_msgs.append(f"skip plot {fname}: no positive values")
            continue
        vmin = float(np.min(positive))
        vmax = float(np.max(positive))
        out_path = plot_dir / fname
        niplot.plot_glass_brain(
            stat_img,
            title=title,
            display_mode="lyrz",
            colorbar=True,
            plot_abs=False,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            threshold=1e-6,
            output_file=str(out_path),
        )
        out_msgs.append(str(out_path))
    return out_msgs


def _categorical_label_values(
    labels: pd.Series,
    parc_df: pd.DataFrame,
    col: str,
) -> Tuple[Dict[str, float], List[Tuple[str, int]], List[str]]:
    """
    Map each ``label`` to 1..k for distinct ``col`` values present.
    Returns label_to_value, (category, code) sorted list for the legend, and
    per-label join warnings.
    """
    if "region" not in parc_df.columns or col not in parc_df.columns:
        return {}, [], [f"parc table missing 'region' or {col!r}"]

    pr = parc_df.set_index("region", drop=False)
    if not pr.index.is_unique:
        return {}, [], [f"parc 'region' column is not unique"]
    # Casefold maps e.g. 7PL_L to the parc table row 7Pl_L
    reg_casefold: Dict[str, str] = {}
    for r in pr.index:
        cfd = str(r).casefold()
        if cfd in reg_casefold and reg_casefold[cfd] != r:
            return (
                {},
                [],
                [
                    f"parc: two 'region' values differ only by case: {reg_casefold[cfd]!r} and {r!r}"
                ],
            )
        reg_casefold[cfd] = r

    used_cats: List[Optional[str]] = []
    n_no_parc = 0
    for lab in labels.astype(str):
        rkey = glasser_label_to_parc_region(lab)
        rkey = reg_casefold.get(str(rkey).casefold(), rkey)
        if rkey not in pr.index:
            n_no_parc += 1
            used_cats.append(None)
            continue
        c = pr.loc[rkey, col]
        s = str(c).strip() if c is not None and pd.notna(c) else None
        used_cats.append(s if s not in ("", "nan", "None") else None)

    distinct = sorted(
        {c for c in used_cats if c is not None},
        key=str.lower,
    )
    if not distinct:
        return (
            {},
            [],
            [f"no {col!r} categories to plot for present parcels"],
        )

    cat_to_code: Dict[str, int] = {c: i + 1 for i, c in enumerate(distinct)}
    order_for_legend: List[Tuple[str, int]] = [(c, cat_to_code[c]) for c in distinct]

    label_to_value: Dict[str, float] = {}
    for lab, c in zip(labels.astype(str), used_cats):
        if c is not None and c in cat_to_code:
            label_to_value[str(lab)] = float(cat_to_code[c])

    warnings: List[str] = []
    if n_no_parc:
        warnings.append(
            f"parc: {n_no_parc} label(s) had no matching 'region' row; those parcels are omitted from {col} map"
        )
    return label_to_value, order_for_legend, warnings


# economo: fixed colors (matches ``glasser_parc.csv`` category labels)
ECONOMO_PARC_COLORS: Dict[str, str] = {
    "agranular": "#6a0dad",  # purple
    "frontal": "#1f77b4",  # blue
    "parietal": "#2ca02c",  # green
    "polar": "#ff7f0e",  # orange
    "granular": "#e6c200",  # yellow
}


def _economo_listed_cmap(order_legend: List[Tuple[str, int]]):
    from matplotlib.colors import ListedColormap

    out: List[str] = []
    for cat, _code in order_legend:
        c = ECONOMO_PARC_COLORS.get(str(cat).casefold().strip(), "#808080")
        out.append(c)
    if not out:
        return ListedColormap(["#808080"], name="economo_parc")
    return ListedColormap(out, name="economo_parc")


def _qualitative_colormap_n(n: int):
    from matplotlib import colormaps
    from matplotlib.colors import ListedColormap

    if n <= 0:
        n = 1
    if n <= 20:
        return colormaps["tab20"].resampled(n)
    tab = [colormaps["tab20"](i / 19) for i in range(20)]
    tab2 = [colormaps["tab20b"](i / 19) for i in range(20)]
    if n <= 40:
        return ListedColormap((tab + tab2)[:n], name="parcats")
    return ListedColormap(
        [colormaps["hsv"].resampled(n)(i / n) for i in range(n)], name="parcats"
    )


def save_parc_category_plots(
    lab_data: np.ndarray,
    affine: np.ndarray,
    index_to_label: Dict[int, str],
    df: pd.DataFrame,
    parc_df: pd.DataFrame,
    plot_dir: Path,
) -> List[str]:
    """
    Glass-brain PNGs for Economo, Mesulam, and Yeo (categorical) with a legend
    centered below the axis grid.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_msgs: List[str] = []
    try:
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch
        from nilearn import plotting as niplot
    except ImportError as e:
        return [f"matplotlib or nilearn not available for parc plots: {e}"]

    # Legend box titles (figure titles use the long form above)
    legend_titles = {
        "economo": "Economo",
        "mesulam": "Mesulam",
        "yeo": "Yeo (communities)",
    }
    col_specs = [
        ("economo", "glasser_parc_economo.png", "Glasser: Economo"),
        ("mesulam", "glasser_parc_mesulam.png", "Glasser: Mesulam"),
        ("yeo", "glasser_parc_yeo.png", "Glasser: Yeo (communities)"),
    ]

    for col, fname, title in col_specs:
        if col not in parc_df.columns:
            out_msgs.append(f"skip {fname}: no column {col!r} in parc table")
            continue
        label_to_value, order_legend, warns = _categorical_label_values(
            df["label"], parc_df, col
        )
        for w in warns:
            if w:
                out_msgs.append(f"note ({col}): {w}")

        if not label_to_value or not order_legend:
            out_msgs.append(
                f"skip plot {fname}: " + (warns[-1] if warns else "no values")
            )
            continue

        n_cat = len(order_legend)
        stat_img = stat_map_from_label_ranks(
            lab_data, affine, index_to_label, label_to_value
        )
        data = np.asanyarray(stat_img.dataobj)
        positive = data[data > 0]
        if positive.size == 0:
            out_msgs.append(f"skip plot {fname}: no positive voxels after mapping")
            continue

        if col == "economo":
            cmap = _economo_listed_cmap(order_legend)
            colors = [
                ECONOMO_PARC_COLORS.get(str(c).casefold().strip(), "#808080")
                for c, _ in order_legend
            ]
        else:
            cmap = _qualitative_colormap_n(n_cat)
            colors = [
                cmap((i + 0.5) / max(1, n_cat))
                for i, _ in enumerate(order_legend)
            ]
        if n_cat == 1:
            v_min, v_max, thr = 0.5, 1.5, 0.1
        else:
            v_min, v_max, thr = 1.0, float(n_cat), 0.4
        out_path = plot_dir / fname
        try:
            niplot.plot_glass_brain(
                stat_img,
                title=title,
                display_mode="lyrz",
                colorbar=False,
                plot_abs=False,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                threshold=thr,
                resampling_interpolation="nearest",
                output_file=None,
            )
        except TypeError:
            niplot.plot_glass_brain(
                stat_img,
                title=title,
                display_mode="lyrz",
                colorbar=False,
                plot_abs=False,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                threshold=thr,
                output_file=None,
            )
        fig = plt.gcf()
        handles = [
            Patch(
                facecolor=colors[i],
                edgecolor="0.2",
                linewidth=0.4,
                label=cat,
            )
            for i, (cat, _code) in enumerate(order_legend)
        ]
        n_items = len(handles)
        ncol = min(max(n_items, 1), 8)
        leg = fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=ncol,
            frameon=True,
            title=legend_titles.get(col, col),
            borderaxespad=0.0,
        )
        fig.savefig(
            str(out_path),
            dpi=200,
            bbox_inches="tight",
            bbox_extra_artists=(leg,),
        )
        plt.close(fig)
        out_msgs.append(str(out_path))
    return out_msgs


def run(
    atlas_nii: Path,
    dseg_tsv: Path,
    output_csv: Path,
    sa_ranks_csv: Path,
    parc_csv: Path,
    plot_dir: Optional[Path],
    no_plots: bool,
) -> int:
    atlas_nii = Path(atlas_nii).resolve()
    dseg_tsv = Path(dseg_tsv).resolve()
    output_csv = Path(output_csv).resolve()
    sa_ranks_csv = Path(sa_ranks_csv).resolve()

    if not atlas_nii.is_file():
        print(f"Atlas NIfTI not found: {atlas_nii}", file=sys.stderr)
        return 1
    if not dseg_tsv.is_file():
        print(f"DSEG TSV not found: {dseg_tsv}", file=sys.stderr)
        return 1
    if not sa_ranks_csv.is_file():
        print(f"S-A ranks CSV not found: {sa_ranks_csv}", file=sys.stderr)
        return 1

    try:
        index_to_label = load_index_to_label(dseg_tsv)
    except Exception as exc:
        print(f"Failed to read TSV: {exc}", file=sys.stderr)
        return 1

    try:
        img = nib.load(str(atlas_nii))
        data = np.asanyarray(img.dataobj)
        affine = np.asarray(img.affine)
    except Exception as exc:
        print(f"Failed to load NIfTI: {exc}", file=sys.stderr)
        return 1

    id_to_centroid = centroids_from_dseg(data, affine)
    rows, warns = build_rows(id_to_centroid, index_to_label)
    for w in warns:
        print(f"Warning: {w}", file=sys.stderr)

    if not rows:
        print("No centroids to write.", file=sys.stderr)
        return 1

    base_df = pd.DataFrame(
        [{"label": r[1], "x": r[2], "y": r[3], "z": r[4]} for r in rows]
    )
    try:
        out_df, rank_warns = add_axis_ranks(base_df, sa_ranks_csv)
    except Exception as exc:
        print(f"Failed to add axis ranks: {exc}", file=sys.stderr)
        return 1
    for w in rank_warns:
        print(f"Warning: {w}", file=sys.stderr)

    cols = ["label", "x", "y", "z", "rank_ap", "rank_dv", "rank_ml", "rank_sa"]
    out_df = out_df[cols]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {output_csv}")

    if not no_plots:
        pdir = (
            plot_dir
            if plot_dir is not None
            else (output_csv.parent / "rank_axis_plots")
        )
        plot_msgs = save_rank_axis_plots(data, affine, index_to_label, out_df, pdir)
        for msg in plot_msgs:
            if msg.endswith(".png"):
                print(f"Wrote {msg}")
            else:
                print(f"Note: {msg}", file=sys.stderr)

        parc_path = Path(parc_csv).resolve()
        if parc_path.is_file():
            try:
                parc_df = pd.read_csv(parc_path)
            except Exception as exc:
                print(
                    f"Note: could not read parc table {parc_path}: {exc}; "
                    "skipping economo / mesulam / yeo brain plots",
                    file=sys.stderr,
                )
            else:
                parc_msgs = save_parc_category_plots(
                    data, affine, index_to_label, out_df, parc_df, pdir
                )
                for msg in parc_msgs:
                    if msg.endswith(".png"):
                        print(f"Wrote {msg}")
                    else:
                        print(f"Note: {msg}", file=sys.stderr)
        else:
            print(
                f"Note: {parc_path} not found; skipping economo / mesulam / yeo parc plots",
                file=sys.stderr,
            )

    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Glasser parcel centroids (MNI mm) + axis ranks from discrete segmentation.",
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Structural tractometry project root",
    )
    p.add_argument(
        "--atlas-nii",
        type=Path,
        default=None,
        help="Glasser dseg NIfTI (default: under --base-dir)",
    )
    p.add_argument(
        "--dseg-tsv",
        type=Path,
        default=None,
        help="atlas-Glasser_dseg.tsv path (default: under --base-dir)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/atlases/Glasser/glasser_centroids.csv)",
    )
    p.add_argument(
        "--sa-ranks-csv",
        type=Path,
        default=None,
        help="Glasser360 S-A ranks CSV (default: Sensorimotor_Association_Axis_AverageRanks.csv)",
    )
    p.add_argument(
        "--parc-csv",
        type=Path,
        default=None,
        help="glasser_parc.csv (region, economo, mesulam, yeo, ...; default: under --base-dir)",
    )
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for rank-axis glass-brain PNGs (default: <output_dir>/rank_axis_plots)",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not write nilearn glass-brain figures",
    )
    args = p.parse_args()
    base = Path(args.base_dir).resolve()
    atlas = (
        Path(args.atlas_nii).resolve()
        if args.atlas_nii is not None
        else (base / DEFAULT_ATLAS_REL)
    )
    tsv = (
        Path(args.dseg_tsv).resolve()
        if args.dseg_tsv is not None
        else (base / DEFAULT_TSV_REL)
    )
    out = (
        Path(args.output).resolve()
        if args.output is not None
        else (base / DEFAULT_OUT_REL)
    )
    sa_csv = (
        Path(args.sa_ranks_csv).resolve()
        if args.sa_ranks_csv is not None
        else (base / DEFAULT_SA_RANKS_REL)
    )
    parc_path = (
        Path(args.parc_csv).resolve()
        if args.parc_csv is not None
        else (base / DEFAULT_PARC_CSV_REL)
    )
    plot_dir = Path(args.plot_dir).resolve() if args.plot_dir is not None else None
    return run(atlas, tsv, out, sa_csv, parc_path, plot_dir, args.no_plots)


if __name__ == "__main__":
    sys.exit(main())
