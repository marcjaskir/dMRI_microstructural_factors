import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
Compare parcel volume distributions between GM and WM random atlases at each resolution.

Loads random atlases at resolutions 156, 256, 356, 456, 556 (5 permutations each per tissue type),
computes parcel volumes (mm³), and compares GM vs WM with Mann-Whitney U tests.
"""
import os
import glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------------------------------------------------------
# Config / paths (same layout as gen_random_atlases.py)
# -----------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_RANDOM_DIR = os.path.join(_PROJECT_ROOT, "data", "atlases", "random")

PARCEL_COUNTS = [156, 256, 356, 456, 556]
VOLUME_ADJUSTED_PARCEL_COUNTS = [110, 180, 250, 321, 391]  # WM counts matching GM resolution levels
N_PERMUTATIONS = 5


def get_atlas_paths(tissue: str, n_parcels: int):
    """Return list of atlas paths for tissue (gm/wm) at given parcel count.
    Uses glob to find files regardless of label suffix (gm, gm_probseg-25, etc.) or version format (v01, v0001).
    """
    subdir = "gm" if tissue == "gm" else "wm"
    base = os.path.join(_RANDOM_DIR, subdir, f"parcels-{n_parcels}")
    pattern = os.path.join(base, f"tpl-MNI152NLin2009cAsym_label-{tissue}*_random_parcels-{n_parcels}_v*.nii.gz")
    paths = sorted(glob.glob(pattern))
    return paths[:N_PERMUTATIONS] if paths else []


def parcel_volumes_mm3(atlas_path: str) -> np.ndarray:
    """Load atlas and return parcel volumes in mm³ (one per parcel)."""
    img = nib.load(atlas_path)
    data = img.get_fdata()
    affine = img.affine
    vol_per_voxel = np.abs(np.linalg.det(affine[:3, :3]))

    labels = np.unique(data[data > 0]).astype(int)
    vols = np.array([np.sum(data == i) * vol_per_voxel for i in labels])
    return vols


def collect_volumes_at_resolution(n_parcels: int):
    """Collect all parcel volumes from GM and WM atlases at given resolution."""
    gm_paths = get_atlas_paths("gm", n_parcels)
    wm_paths = get_atlas_paths("wm", n_parcels)

    gm_vols = []
    for p in gm_paths:
        gm_vols.extend(parcel_volumes_mm3(p))
    gm_vols = np.array(gm_vols) if gm_vols else np.array([])

    wm_vols = []
    for p in wm_paths:
        wm_vols.extend(parcel_volumes_mm3(p))
    wm_vols = np.array(wm_vols) if wm_vols else np.array([])

    return gm_vols, wm_vols


def collect_volumes_single_tissue(tissue: str, n_parcels: int):
    """Collect all parcel volumes from one tissue type at given parcel count."""
    paths = get_atlas_paths(tissue, n_parcels)
    vols = []
    for p in paths:
        vols.extend(parcel_volumes_mm3(p))
    return np.array(vols) if vols else np.array([])


def main():
    if not os.path.isdir(_RANDOM_DIR):
        raise FileNotFoundError(f"Random atlas directory not found: {_RANDOM_DIR}")

    results = {}
    for n in PARCEL_COUNTS:
        gm_vols, wm_vols = collect_volumes_at_resolution(n)
        if gm_vols.size == 0 or wm_vols.size == 0:
            print(f"Warning: missing data at n={n}")
            results[n] = {"gm": gm_vols, "wm": wm_vols, "stat": None, "p": np.nan}
            continue

        # Mann-Whitney U test (non-parametric, compares distributions)
        stat, p = stats.mannwhitneyu(gm_vols, wm_vols, alternative="two-sided")
        results[n] = {"gm": gm_vols, "wm": wm_vols, "stat": stat, "p": p}
        print(f"n={n}: GM n={len(gm_vols)} parcels, WM n={len(wm_vols)} parcels | "
              f"Mann-Whitney U={stat:.0f}, p={p:.2e}")

    # Plot
    n_res = len(PARCEL_COUNTS)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, n in enumerate(PARCEL_COUNTS):
        ax = axes[idx]
        r = results[n]
        gm_vols, wm_vols = r["gm"], r["wm"]
        p_val = r["p"]

        if gm_vols.size > 0 and wm_vols.size > 0:
            bp = ax.boxplot(
                [gm_vols, wm_vols],
                labels=["GM", "WM"],
                patch_artist=True,
                widths=0.6,
            )
            bp["boxes"][0].set_facecolor("#2e7d32")
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][1].set_facecolor("#1565c0")
            bp["boxes"][1].set_alpha(0.7)

            p_str = f"p = {p_val:.2e}" if p_val < 0.001 else f"p = {p_val:.3f}"
            ax.text(0.5, 0.95, p_str, transform=ax.transAxes, ha="center", va="top", fontsize=10)

        ax.set_title(f"{n} parcels")
        ax.set_ylabel("Parcel volume (mm³)")
        ax.grid(True, alpha=0.3, axis="y")

    # Hide unused subplot
    axes[-1].axis("off")

    # Summary table in last subplot
    ax_tab = axes[-1]
    ax_tab.axis("on")
    ax_tab.axis("off")
    table_data = [["Resolution", "GM mean±std", "WM mean±std", "p-value"]]
    for n in PARCEL_COUNTS:
        r = results[n]
        if r["gm"].size > 0 and r["wm"].size > 0:
            gm_str = f"{r['gm'].mean():.0f} ± {r['gm'].std():.0f}"
            wm_str = f"{r['wm'].mean():.0f} ± {r['wm'].std():.0f}"
            p_str = f"{r['p']:.2e}" if r["p"] < 0.001 else f"{r['p']:.3f}"
        else:
            gm_str = wm_str = p_str = "N/A"
        table_data.append([str(n), gm_str, wm_str, p_str])

    tab = ax_tab.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.2, 0.25, 0.25, 0.2],
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.2, 2)
    ax_tab.set_title("Summary (Mann-Whitney U)")

    plt.suptitle("GM vs WM random atlas parcel volume distributions", fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(_SCRIPT_DIR, "random_atlas_wm_gm_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")

    # -------------------------------------------------------------------------
    # Volume-adjusted comparison: GM at N parcels vs WM at volume-adjusted count
    # -------------------------------------------------------------------------
    results_va = {}
    for idx, (gm_n, wm_n) in enumerate(zip(PARCEL_COUNTS, VOLUME_ADJUSTED_PARCEL_COUNTS)):
        gm_vols = collect_volumes_single_tissue("gm", gm_n)
        wm_vols = collect_volumes_single_tissue("wm", wm_n)
        if gm_vols.size == 0 or wm_vols.size == 0:
            print(f"Warning: missing data for volume-adjusted GM {gm_n} / WM {wm_n}")
            results_va[idx] = {"gm": gm_vols, "wm": wm_vols, "gm_n": gm_n, "wm_n": wm_n, "p": np.nan}
            continue
        stat, p = stats.mannwhitneyu(gm_vols, wm_vols, alternative="two-sided")
        results_va[idx] = {"gm": gm_vols, "wm": wm_vols, "gm_n": gm_n, "wm_n": wm_n, "stat": stat, "p": p}
        print(f"Volume-adj: GM {gm_n} vs WM {wm_n} | Mann-Whitney U={stat:.0f}, p={p:.2e}")

    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))
    axes2 = axes2.flatten()

    for idx in range(len(PARCEL_COUNTS)):
        ax = axes2[idx]
        r = results_va[idx]
        gm_vols, wm_vols = r["gm"], r["wm"]
        gm_n, wm_n = r["gm_n"], r["wm_n"]
        p_val = r["p"]

        if gm_vols.size > 0 and wm_vols.size > 0:
            bp = ax.boxplot(
                [gm_vols, wm_vols],
                labels=[f"GM ({gm_n})", f"WM ({wm_n})"],
                patch_artist=True,
                widths=0.6,
            )
            bp["boxes"][0].set_facecolor("#2e7d32")
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][1].set_facecolor("#1565c0")
            bp["boxes"][1].set_alpha(0.7)

            p_str = f"p = {p_val:.2e}" if p_val < 0.001 else f"p = {p_val:.3f}"
            ax.text(0.5, 0.95, p_str, transform=ax.transAxes, ha="center", va="top", fontsize=10)

        ax.set_title(f"GM {gm_n} vs WM {wm_n} (vol.-adj.)")
        ax.set_ylabel("Parcel volume (mm³)")
        ax.grid(True, alpha=0.3, axis="y")

    axes2[-1].axis("off")
    ax_tab2 = axes2[-1]
    ax_tab2.axis("on")
    ax_tab2.axis("off")
    table_data2 = [["GM parcels", "WM parcels", "GM mean±std", "WM mean±std", "p-value"]]
    for idx in range(len(PARCEL_COUNTS)):
        r = results_va[idx]
        gm_n, wm_n = r["gm_n"], r["wm_n"]
        if r["gm"].size > 0 and r["wm"].size > 0:
            gm_str = f"{r['gm'].mean():.0f} ± {r['gm'].std():.0f}"
            wm_str = f"{r['wm'].mean():.0f} ± {r['wm'].std():.0f}"
            p_str = f"{r['p']:.2e}" if r["p"] < 0.001 else f"{r['p']:.3f}"
        else:
            gm_str = wm_str = p_str = "N/A"
        table_data2.append([str(gm_n), str(wm_n), gm_str, wm_str, p_str])

    tab2 = ax_tab2.table(
        cellText=table_data2,
        loc="center",
        cellLoc="center",
        colWidths=[0.12, 0.12, 0.22, 0.22, 0.15],
    )
    tab2.auto_set_font_size(False)
    tab2.set_fontsize(9)
    tab2.scale(1.2, 2)
    ax_tab2.set_title("Summary (volume-adjusted, Mann-Whitney U)")

    plt.suptitle("GM vs WM volume-adjusted parcel volume distributions", fontsize=12)
    plt.tight_layout()

    out_path_va = os.path.join(_SCRIPT_DIR, "random_atlas_wm_gm_comparison_volume_adjusted.png")
    plt.savefig(out_path_va, dpi=150)
    print(f"Saved: {out_path_va}")


if __name__ == "__main__":
    main()
