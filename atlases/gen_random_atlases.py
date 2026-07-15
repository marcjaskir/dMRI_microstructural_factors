import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
Generate random parcellations of grey matter (GM) and white matter (WM) in MNI space.

Uses thresholded tissue probability images and a grassfire expansion from random seeds
(adapted from Revell Lab randomAtlasGeneration.py) to produce balanced random atlases.
"""

import os
import numpy as np
import nibabel as nib

# -----------------------------------------------------------------------------
# Config / paths
# -----------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "atlases")
_MNI_DIR = os.path.join(_DATA_DIR, "MNI")
_OUT_DIR = os.path.join(_DATA_DIR, "random")

GM_PROB_PATH = os.path.join(
    _MNI_DIR, "tpl-MNI152NLin2009cAsym_res-1mm_label-gm_probseg.nii.gz"
)
WM_PROB_PATH = os.path.join(
    _MNI_DIR, "tpl-MNI152NLin2009cAsym_res-1mm_label-wm_probseg.nii.gz"
)
GM_THRESHOLD = 0.51
WM_THRESHOLD = 0.51
PARCEL_COUNTS = list(range(156, 557, 100))
N_PERMUTATIONS = 5
BASE_SEED = 20250213


def grassfire_algorithm(edge_points, atlas, vols):
    """
    One step of grassfire expansion: grow each region into adjacent unlabeled voxels.

    Adapted from Revell Lab randomAtlasGeneration.py. Processes 6-neighbors of each
    edge voxel; unlabeled (0) neighbors get assigned the same region_id. New frontier
    is sorted by current region volume (ascending) so smaller regions expand first.

    Parameters
    ----------
    edge_points : np.ndarray
        N x 4 or N x 5, columns (i, j, k, region_id) with optional 5th for sort key.
    atlas : np.ndarray
        3D int array; 0 = unlabeled, -1 = outside mask, 1..n = region labels.
    vols : np.ndarray
        1D array of current voxel counts per region (index 0 = region 1).

    Returns
    -------
    new_edge_points : np.ndarray
        New frontier (N x 5, sorted by volume).
    atlas : np.ndarray
        Updated in place; also returned.
    vols : np.ndarray
        Updated in place; also returned.
    """
    dims = atlas.shape
    n_edge = edge_points.shape[0]
    max_new = min(n_edge * 7, dims[0] * dims[1] * dims[2])
    new_edge_points = np.zeros((max_new, 4))
    counter = 0
    for i in range(n_edge):
        point = edge_points[i, :4].astype(int)
        region_id = point[3]
        # +x
        if point[0] + 1 < dims[0]:
            if atlas[point[0] + 1, point[1], point[2]] == 0:
                new_edge_points[counter, :] = [
                    point[0] + 1,
                    point[1],
                    point[2],
                    region_id,
                ]
                atlas[point[0] + 1, point[1], point[2]] = region_id
                counter += 1
                vols[region_id - 1] += 1
        # -x
        if point[0] - 1 >= 0:
            if atlas[point[0] - 1, point[1], point[2]] == 0:
                new_edge_points[counter, :] = [
                    point[0] - 1,
                    point[1],
                    point[2],
                    region_id,
                ]
                atlas[point[0] - 1, point[1], point[2]] = region_id
                counter += 1
                vols[region_id - 1] += 1
        # +y
        if point[1] + 1 < dims[1]:
            if atlas[point[0], point[1] + 1, point[2]] == 0:
                new_edge_points[counter, :] = [
                    point[0],
                    point[1] + 1,
                    point[2],
                    region_id,
                ]
                atlas[point[0], point[1] + 1, point[2]] = region_id
                counter += 1
                vols[region_id - 1] += 1
        # -y
        if point[1] - 1 >= 0:
            if atlas[point[0], point[1] - 1, point[2]] == 0:
                new_edge_points[counter, :] = [
                    point[0],
                    point[1] - 1,
                    point[2],
                    region_id,
                ]
                atlas[point[0], point[1] - 1, point[2]] = region_id
                counter += 1
                vols[region_id - 1] += 1
        # +z
        if point[2] + 1 < dims[2]:
            if atlas[point[0], point[1], point[2] + 1] == 0:
                new_edge_points[counter, :] = [
                    point[0],
                    point[1],
                    point[2] + 1,
                    region_id,
                ]
                atlas[point[0], point[1], point[2] + 1] = region_id
                counter += 1
                vols[region_id - 1] += 1
        # -z
        if point[2] - 1 >= 0:
            if atlas[point[0], point[1], point[2] - 1] == 0:
                new_edge_points[counter, :] = [
                    point[0],
                    point[1],
                    point[2] - 1,
                    region_id,
                ]
                atlas[point[0], point[1], point[2] - 1] = region_id
                counter += 1
                vols[region_id - 1] += 1
    new_edge_points = new_edge_points[:counter]
    if new_edge_points.size == 0:
        return new_edge_points, atlas, vols
    # Add volume column and sort by volume (smaller regions first for balanced growth)
    vol_col = np.array([vols[int(r) - 1] for r in new_edge_points[:, 3]])
    new_edge_points = np.column_stack([new_edge_points, vol_col])
    new_edge_points = new_edge_points[new_edge_points[:, 4].argsort()]
    return new_edge_points, atlas, vols


def mask_from_prob(prob_path, threshold):
    """
    Load a tissue probability NIfTI, threshold and binarize.

    Parameters
    ----------
    prob_path : str
        Path to probability image.
    threshold : float
        Voxels with value >= threshold become 1, others 0.

    Returns
    -------
    mask : np.ndarray
        3D binary (1 inside tissue, 0 outside).
    affine : np.ndarray
        Affine from the NIfTI (for saving outputs in same space).
    """
    img = nib.load(prob_path)
    data = img.get_fdata()
    mask = (data >= threshold).astype(np.float64)
    return mask, img.affine.copy()


def generate_random_atlas_tissue(mask, affine, n_regions, rng):
    """
    Generate one random parcellation of a tissue mask with n_regions parcels.

    Parameters
    ----------
    mask : np.ndarray
        3D binary mask (1 = tissue, 0 = outside).
    affine : np.ndarray
        Affine for the output NIfTI.
    n_regions : int
        Number of parcels.
    rng : np.random.Generator
        Random number generator (for reproducibility).

    Returns
    -------
    atlas : np.ndarray
        3D int array; 0 = outside mask, 1..n_regions = parcel labels.
    """
    dims = mask.shape
    # Brain voxel coordinates: (N, 3) array of (i, j, k)
    brain_ijk = np.column_stack(np.where(mask > 0))
    n_brain = brain_ijk.shape[0]
    if n_regions > n_brain:
        raise ValueError(
            f"n_regions ({n_regions}) cannot exceed number of mask voxels ({n_brain})"
        )
    # Atlas: -1 outside mask, 0 unlabeled inside, 1..n_regions assigned
    atlas = np.where(mask > 0, 0, -1).astype(np.int64)
    # Random seeds without replacement
    idx = rng.choice(n_brain, size=n_regions, replace=False)
    seeds_ijk = brain_ijk[idx]
    # Initialize seeds in atlas and build edge list (i, j, k, region_id)
    start_points = np.zeros((n_regions, 4))
    for i in range(n_regions):
        ii, jj, kk = seeds_ijk[i, 0], seeds_ijk[i, 1], seeds_ijk[i, 2]
        region_id = i + 1
        atlas[ii, jj, kk] = region_id
        start_points[i, :] = [ii, jj, kk, region_id]
    vols = np.ones(n_regions, dtype=np.int64)
    # Add sort column for first iteration (all 1s)
    start_points = np.column_stack([start_points, np.ones(n_regions)])
    # Grassfire until no frontier
    while start_points.size > 0:
        start_points, atlas, vols = grassfire_algorithm(
            start_points[:, :4], atlas, vols
        )
        if start_points.size == 0:
            break
    atlas[atlas == -1] = 0
    return atlas


def _run():
    os.makedirs(_OUT_DIR, exist_ok=True)

    # Confirm required probability images exist
    if not os.path.exists(GM_PROB_PATH) or not os.path.exists(WM_PROB_PATH):
        raise FileNotFoundError("GM and WM probability images required.")

    # Load masks for both tissues for volume ratio calculation
    print(f"Loading GM mask from {GM_PROB_PATH} (threshold={GM_THRESHOLD})")
    gm_mask, gm_affine = mask_from_prob(GM_PROB_PATH, GM_THRESHOLD)
    gm_voxels = int((gm_mask > 0).sum())
    print(f"  GM mask voxels: {gm_voxels}")

    print(f"Loading WM mask from {WM_PROB_PATH} (threshold={WM_THRESHOLD})")
    wm_mask, wm_affine = mask_from_prob(WM_PROB_PATH, WM_THRESHOLD)
    wm_voxels = int((wm_mask > 0).sum())
    print(f"  WM mask voxels: {wm_voxels}")

    if gm_voxels == 0:
        raise ValueError("GM mask has zero voxels after thresholding.")
    if wm_voxels == 0:
        raise ValueError("WM mask has zero voxels after thresholding.")

    wm_gm_ratio = wm_voxels / gm_voxels
    print(f"WM:GM volume ratio: {wm_gm_ratio:.3f}")

    # Set up tissues and related info
    tissues = [
        ("gm", gm_mask, gm_affine, "gm", PARCEL_COUNTS),
        ("wm", wm_mask, wm_affine, "wm", PARCEL_COUNTS),
    ]

    for tissue_name, mask, affine, label_suffix, parcel_counts in tissues:
        n_mask = int((mask > 0).sum())
        print(f"Processing {tissue_name.upper()} - Mask voxels: {n_mask}")

        # For WM only, also compute volume-adjusted parcel counts (rounded)
        voladj_parcel_counts = []
        if tissue_name == "wm":
            voladj_set = set()
            for n in PARCEL_COUNTS:
                na = int(round(n * wm_gm_ratio))
                # Avoid duplicates, handle only meaningful, unique, in-range values
                if na > 0 and na <= n_mask and na not in parcel_counts:
                    voladj_set.add(na)
            if voladj_set:
                voladj_parcel_counts = sorted(voladj_set)
            print(f"  Volume-adjusted parcel counts for WM: {voladj_parcel_counts}")

        # Combine standard counts with volume-adjusted ones for WM; GM stays standard
        if tissue_name == "wm":
            counts_to_generate = sorted(set(parcel_counts) | set(voladj_parcel_counts))
        else:
            counts_to_generate = sorted(set(parcel_counts))

        for n in counts_to_generate:
            if n > n_mask:
                print(f"  Skipping n_parcels={n} (exceeds mask voxels)")
                continue
            parcel_dir = os.path.join(_OUT_DIR, tissue_name, f"parcels-{n}")
            os.makedirs(parcel_dir, exist_ok=True)
            for perm in range(1, N_PERMUTATIONS + 1):
                out_name = (
                    f"tpl-MNI152NLin2009cAsym_label-{label_suffix}_random_parcels-{n}_v{perm:02d}.nii.gz"
                )
                out_path = os.path.join(parcel_dir, out_name)
                if os.path.exists(out_path):
                    print(f"  Exists: {out_name}")
                    continue
                seed = (
                    BASE_SEED
                    + (0 if tissue_name == "gm" else 100000)
                    + n * 100
                    + perm
                )
                rng = np.random.default_rng(seed)
                print(f"  Generating {out_name}")
                atlas = generate_random_atlas_tissue(mask, affine, n, rng)
                nib.save(nib.Nifti1Image(atlas.astype(np.int32), affine), out_path)
    print("Done.")


if __name__ == "__main__":
    _run()
