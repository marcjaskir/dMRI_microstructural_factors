import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
Compute region-wise mean diffusion MRI scalars for multiple atlases in MNI space.

Supports:
- 4S multiscale atlases (156, 256, ..., 1056 parcels)
- Glasser atlas
- HCP1065 WM tract segments (pyAFQ endpoint/core masks; overlapping, per-mask means)
- Random GM/WM atlases (parcels-50..500 and parcels-156..1056, multiple versions)

Writes one HDF5 file per atlas under derivatives/mni_micro/:
- 4S156_mni_micro.h5, 4S256_mni_micro.h5, ... (one per 4S resolution)
- Glasser_mni_micro.h5
- HCP1065_mni_micro.h5
- Random110_mni_micro.h5, Random156_mni_micro.h5, ... (one per Random resolution; versions as subgroups)

Within each file, layout is unchanged:
- 4S: /4S{resolution}/{scalar}/mean and /4S{resolution}/{scalar}/standard_deviation (and sub, group, parcel_columns).
- Glasser: /Glasser/{scalar}/mean and standard_deviation — same layout.
- HCP1065: /HCP1065/{scalar}/mean and standard_deviation — same layout.
- Random: /random{parcels_n}/{gm|wm}/{version}/{scalar}/mean and standard_deviation — same layout; versions kept as separate subgroups.
Separate datasets avoid HDF5 "object header too large" with many parcels (e.g. Glasser 360).
"""

import json
import glob
import traceback
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import resample_to_img
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_QSIRECON_DIR = _PROJECT_ROOT / "derivatives" / "qsirecon"
_ATLAS_4S_DIR = _PROJECT_ROOT / "data" / "atlases" / "4S"
_ATLAS_GLASSER_DIR = _PROJECT_ROOT / "data" / "atlases" / "Glasser"
_ATLAS_RANDOM_DIR = _PROJECT_ROOT / "data" / "atlases" / "random"
_ATLAS_HCP1065_MASKS_DIR = _PROJECT_ROOT / "data" / "atlases" / "HCP1065" / "mni_micro"
_ATLAS_HCP1065_RESAMPLED_DIR = _PROJECT_ROOT / "data" / "atlases" / "HCP1065" / "mni_micro_resampled"
_OUT_DIR = _PROJECT_ROOT / "derivatives" / "mni_micro"
_METADATA_DIR = _PROJECT_ROOT / "data" / "metadata"
_SCALAR_LABELS_TO_DIRS = _METADATA_DIR / "scalar_labels_to_directories.json"
_SCALAR_LABELS_TO_FILES = _METADATA_DIR / "scalar_labels_to_filenames.json"

NO_SESSION_GROUP = "hcpaging"
MNI_SUFFIX = "space-MNI152NLin2009cAsym"

# 4S multiscale resolutions (parcel counts)
FOUR_S_RESOLUTIONS = [156]
# FOUR_S_RESOLUTIONS = (156, 256, 356, 456, 556)

# Random atlas resolutions to compute scalar means for (parcel counts, e.g. 156 → parcels-156)
# RANDOM_RESOLUTIONS = [110, 156, 180, 250, 256, 321, 356, 391, 456, 556]
RANDOM_RESOLUTIONS = [110,156]

# Subject loop parallelism (always parallel; must be >= 2)
SUBJECT_WORKERS = 32

# Scalars to skip when computing region means
EXCLUDED_SCALARS = [
    "map_li", "map_am", "dti_txx", "dti_txy", "dti_txz", "dti_tyy", "dti_tyz", "dti_tzz",
    "dti_ha", "rdi_rd1", "rdi_rd2",
]

# Atlases to compute, in run order: "4S", "Glasser", "HCP1065", "random" (each runs all its resolutions/versions)
ATLAS_INCLUSION = ["4S", "Glasser", "HCP1065", "random"]
# ATLAS_INCLUSION = ["Glasser", "random"]

def _load_metadata():
    with open(_SCALAR_LABELS_TO_DIRS) as f:
        scalar_to_dir = json.load(f)
    with open(_SCALAR_LABELS_TO_FILES) as f:
        scalar_to_file = json.load(f)
    return scalar_to_dir, scalar_to_file


def _discover_groups():
    if not _QSIRECON_DIR.exists():
        return []
    groups = []
    for p in _QSIRECON_DIR.iterdir():
        if p.is_dir() and not p.name.startswith("sub-") and p.name not in ("atlases", "logs", "log"):
            groups.append(p.name)
    return sorted(groups)


def _discover_subject_sessions(group):
    """Yield (sub, ses) for each subject-session that has a dwi dir. Tries derivatives first, then group/sub."""
    group_dir = _QSIRECON_DIR / group
    if not group_dir.exists():
        return
    scalar_to_dir, _ = _load_metadata()
    scalar_dirs = list(scalar_to_dir.values())
    base = None
    if scalar_dirs:
        base = group_dir / "derivatives" / scalar_dirs[0]
    if base is not None and base.exists():
        sub_iter = sorted(base.iterdir())
    else:
        sub_iter = sorted(d for d in group_dir.iterdir() if d.is_dir() and d.name.startswith("sub-"))
    if group == NO_SESSION_GROUP:
        for sub_dir in sub_iter:
            if sub_dir.is_dir() and sub_dir.name.startswith("sub-"):
                if (sub_dir / "dwi").is_dir():
                    yield sub_dir.name, None
    else:
        for sub_dir in sub_iter:
            if sub_dir.is_dir() and sub_dir.name.startswith("sub-"):
                for ses_dir in sorted(sub_dir.iterdir()):
                    if ses_dir.is_dir() and ses_dir.name.startswith("ses-"):
                        if (ses_dir / "dwi").is_dir():
                            yield sub_dir.name, ses_dir.name


def _log_skip(sub, group, scalar_label, e, scalar_path=None, extra=None):
    """Print detailed skip message for debugging."""
    print("  Skip %s %s %s: %s" % (sub, group, scalar_label, e))
    if scalar_path is not None:
        print("    scalar_path: %s" % scalar_path)
    if extra:
        for k, v in extra.items():
            print("    %s: %s" % (k, v))
    print("    traceback:\n%s" % "".join(traceback.format_list(traceback.extract_tb(e.__traceback__)[-6:])))


def _abs_path(p):
    """Return absolute path as string for use in worker processes (stable across cwd)."""
    if p is None:
        return None
    return str(Path(p).resolve())


def _scalar_path(group, sub, ses, scalar_directory, scalar_filename):
    """Return path to MNI-space scalar NIfTI; try derivatives subdir first, then group/sub/ses/dwi."""
    candidates = []
    bases = []
    if group == NO_SESSION_GROUP:
        bases = [
            _QSIRECON_DIR / group / "derivatives" / scalar_directory / sub / "dwi",
            _QSIRECON_DIR / group / sub / "dwi",
        ]
    else:
        bases = [
            _QSIRECON_DIR / group / "derivatives" / scalar_directory / sub / ses / "dwi",
            _QSIRECON_DIR / group / sub / ses / "dwi",
        ]
    for base in bases:
        if not base.exists():
            continue
        for pattern in [
            "*" + MNI_SUFFIX + "_" + scalar_filename + ".nii.gz",
            "*" + MNI_SUFFIX + "_" + scalar_filename + "_dwimap.nii.gz",
            "*" + MNI_SUFFIX + "*" + scalar_filename + "*.nii.gz",
        ]:
            candidates.extend(glob.glob(str(base / pattern)))
    return _abs_path(candidates[0]) if candidates else None


def _region_stats_for_atlas(scalar_nii_path, atlas_nii_path_or_img, n_parcels=None, strategy="mean"):
    """Return 1D array of scalar stat (mean or standard_deviation) per parcel in order 1..N (index 0 = parcel 1).
    atlas_nii_path_or_img: path (str/Path) or pre-loaded nibabel image.
    n_parcels: expected number of parcels; if None, use max label from masker.
    strategy: 'mean' or 'standard_deviation' for NiftiLabelsMasker."""
    scalar_path = str(Path(scalar_nii_path).resolve()) if scalar_nii_path else None
    if not scalar_path:
        return np.full(n_parcels or 0, np.nan, dtype=np.float64)
    scalar_img = nib.load(scalar_path)
    if hasattr(atlas_nii_path_or_img, "get_fdata"):
        atlas_img = atlas_nii_path_or_img
    else:
        atlas_img = nib.load(str(Path(atlas_nii_path_or_img).resolve()))
    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False, strategy=strategy)
    out = np.asarray(masker.fit_transform(scalar_img), dtype=np.float64)
    if out.ndim == 2:
        raw = out[0]
    else:
        raw = out
    raw = np.atleast_1d(raw)
    masker_labels = np.asarray(masker.labels_)
    result = np.full(n_parcels, np.nan, dtype=np.float64)
    for j in range(0, len(masker_labels)-1):
        result[j] = float(raw[j])
    return result


def _region_means_for_atlas(scalar_nii_path, atlas_nii_path_or_img, n_parcels=None):
    """Return 1D array of mean scalar value per parcel. See _region_stats_for_atlas."""
    return _region_stats_for_atlas(scalar_nii_path, atlas_nii_path_or_img, n_parcels, strategy="mean")


def _region_std_for_atlas(scalar_nii_path, atlas_nii_path_or_img, n_parcels=None):
    """Return 1D array of standard deviation per parcel. See _region_stats_for_atlas."""
    return _region_stats_for_atlas(scalar_nii_path, atlas_nii_path_or_img, n_parcels, strategy="standard_deviation")


def _region_mean_std_for_atlas(scalar_nii_path, atlas_nii_path_or_img, n_parcels=None):
    """Load scalar once, compute mean and standard_deviation per parcel. Returns (means, stds) as 1D arrays."""
    means = _region_means_for_atlas(scalar_nii_path, atlas_nii_path_or_img, n_parcels)
    stds = _region_std_for_atlas(scalar_nii_path, atlas_nii_path_or_img, n_parcels)
    return means, stds


def _sanitize_field_name(s):
    """Make string valid as numpy/HDF5 field name: replace spaces/dashes, ensure not leading digit."""
    s = str(s).replace(" ", "_").replace("-", "_")
    if s and s[0].isdigit():
        s = "p" + s
    return s or "unnamed"


def _discover_hcp1065_masks():
    """List all binary mask NIfTIs in endpoint_nii_bin. Return (parcel_labels, mask_paths)."""
    if not _ATLAS_HCP1065_MASKS_DIR.exists():
        return [], []
    paths = sorted(
        p for p in _ATLAS_HCP1065_MASKS_DIR.glob("*.nii.gz")
        if not p.name.startswith(".")
    )
    labels = [p.stem.replace(".nii", "") for p in paths]
    return labels, [str(p) for p in paths]


def _build_hcp1065_resampled_masks(reference_scalar_path, mask_paths):
    """Load and resample all HCP1065 masks to reference scalar grid once. Saves resampled NIfTIs under data/atlases/HCP1065/mni_micro_resampled. Returns list of 3D int arrays (no thresholding)."""
    ref_img = nib.load(reference_scalar_path)
    out_dir = _ATLAS_HCP1065_RESAMPLED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    resampled = []
    for mask_path in mask_paths:
        mask_img = nib.load(mask_path)
        mask_resampled = resample_to_img(mask_img, ref_img, interpolation="nearest")
        mask_data = np.asarray(mask_resampled.get_fdata(), dtype=np.int32)
        resampled.append(mask_data)
        out_path = out_dir / Path(mask_path).name
        nib.save(nib.Nifti1Image(mask_data, mask_resampled.affine), str(out_path))
    return resampled


def _region_means_hcp1065_overlapping(scalar_nii_path, pre_resampled_masks):
    """Compute mean scalar value per mask (masks can overlap). pre_resampled_masks: list of 3D int arrays (from _build_hcp1065_resampled_masks). Returns 1D array of length len(pre_resampled_masks)."""
    scalar_img = nib.load(scalar_nii_path)
    scalar_data = np.asarray(scalar_img.get_fdata(), dtype=np.float64)
    means = np.full(len(pre_resampled_masks), np.nan, dtype=np.float64)
    for i, mask_data in enumerate(pre_resampled_masks):
        in_mask = mask_data != 0
        if not np.any(in_mask):
            continue
        vals = scalar_data[in_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            means[i] = np.mean(vals)
    return means


def _region_std_hcp1065_overlapping(scalar_nii_path, pre_resampled_masks):
    """Compute standard deviation per mask (masks can overlap). Same signature as _region_means_hcp1065_overlapping. Returns 1D array of length len(pre_resampled_masks)."""
    scalar_img = nib.load(scalar_nii_path)
    scalar_data = np.asarray(scalar_img.get_fdata(), dtype=np.float64)
    stds = np.full(len(pre_resampled_masks), np.nan, dtype=np.float64)
    for i, mask_data in enumerate(pre_resampled_masks):
        in_mask = mask_data != 0
        if not np.any(in_mask):
            continue
        vals = scalar_data[in_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size > 1:
            stds[i] = np.std(vals, ddof=1)
        elif vals.size == 1:
            stds[i] = 0.0
    return stds


def _iter_4s_atlases():
    """Yield (atlas_key, res_key, nii_path, tsv_path) for each 4S resolution."""
    for res in FOUR_S_RESOLUTIONS:
        nii_name = f"tpl-MNI152NLin2009cAsym_atlas-4S{res}Parcels_res-01_dseg.nii.gz"
        nii_path = _ATLAS_4S_DIR / nii_name
        if not nii_path.exists():
            nii_path = _ATLAS_4S_DIR / nii_name.replace(".nii.gz", ".nii")
        if not nii_path.exists():
            continue
        tsv_path = _ATLAS_4S_DIR / f"atlas-4S{res}Parcels_dseg.tsv"
        if not tsv_path.exists():
            continue
        yield "4S", str(res), str(nii_path), str(tsv_path)


def _iter_glasser_atlases():
    """Yield (atlas_key, res_key, nii_path, tsv_path) for Glasser."""
    nii_path = _ATLAS_GLASSER_DIR / "atlas-Glasser_space-MNI152NLin2009cAsym_res-1_dseg.nii.gz"
    if not nii_path.exists():
        nii_path = _ATLAS_GLASSER_DIR / "atlas-Glasser_space-MNI152NLin2009cAsym_res-1_dseg.nii"
    if not nii_path.exists():
        return
    tsv_path = _ATLAS_GLASSER_DIR / "atlas-Glasser_dseg.tsv"
    if not tsv_path.exists():
        return
    yield "Glasser", "res-1", str(nii_path), str(tsv_path)


def _iter_random_atlases():
    """Yield (tissue, parcels_n, version, atlas_path) for random atlases in RANDOM_RESOLUTIONS."""
    for tissue_dir in ("gm", "wm"):
        tissue_path = _ATLAS_RANDOM_DIR / tissue_dir
        if not tissue_path.exists():
            continue
        for parcel_dir in sorted(tissue_path.iterdir()):
            if not parcel_dir.is_dir() or not parcel_dir.name.startswith("parcels-"):
                continue
            parcels_n = parcel_dir.name
            try:
                n_parcels = int(parcels_n.split("-")[1])
            except (IndexError, ValueError):
                continue
            if n_parcels not in RANDOM_RESOLUTIONS:
                continue
            for nii in sorted(parcel_dir.glob("*.nii.gz")):
                stem = nii.stem.replace(".nii", "")
                version = "v" + stem.split("_v")[-1] if "_v" in stem else "v0001"
                yield tissue_dir, parcels_n, version, str(nii)


def _parcel_labels_from_tsv(tsv_path):
    """Load parcel labels (column 'label') from TSV for use as mean table column names."""
    if not tsv_path or not Path(tsv_path).exists():
        return None
    df = pd.read_csv(tsv_path, sep="\t")
    if "label" not in df.columns:
        return None
    return df["label"].astype(str).tolist()


def _h5_path_4s(res_key):
    """Path for 4S atlas file: e.g. 4S156_mni_micro.h5."""
    return _OUT_DIR / f"4S{res_key}_mni_micro.h5"


def _h5_path_glasser():
    return _OUT_DIR / "Glasser_mni_micro.h5"


def _h5_path_hcp1065():
    return _OUT_DIR / "HCP1065_mni_micro.h5"


def _h5_path_random(parcels_n):
    """Path for Random atlas file by resolution: e.g. parcels-156 -> Random156_mni_micro.h5."""
    n = parcels_n.replace("parcels-", "")
    return _OUT_DIR / f"Random{n}_mni_micro.h5"


# ---- Iterative HDF5 write (separate datasets to avoid "object header too large") ----
# Store mean and standard_deviation as 2D float (n_rows x n_parcels), sub/group as 1D strings. No compound dtype = no header limit.


def _write_mean_row(ds_mean, ds_sub, ds_group, i, sub, group, means):
    """Write one row to separate mean/sub/group datasets."""
    ds_sub[i] = str(sub).encode("utf-8")[:200].ljust(200, b"\x00")[:200]
    ds_group[i] = str(group).encode("utf-8")[:100].ljust(100, b"\x00")[:100]
    ds_mean[i, :] = np.asarray(means, dtype=np.float64)


def _write_mean_std_row(ds_mean, ds_std, ds_sub, ds_group, i, sub, group, means, stds):
    """Write one row to separate mean, standard_deviation, sub, group datasets."""
    ds_sub[i] = str(sub).encode("utf-8")[:200].ljust(200, b"\x00")[:200]
    ds_group[i] = str(group).encode("utf-8")[:100].ljust(100, b"\x00")[:100]
    ds_mean[i, :] = np.asarray(means, dtype=np.float64)
    ds_std[i, :] = np.asarray(stds, dtype=np.float64)


def _create_mean_dataset_iterative(grp, n_rows, n_parcels, parcel_col_names):
    """Create mean (2D float), sub (1D), group (1D), parcel_columns. Returns (ds_mean, ds_sub, ds_group)."""
    for name in ("mean", "sub", "group", "parcel_columns"):
        if name in grp:
            del grp[name]
    ds_mean = grp.create_dataset("mean", shape=(n_rows, n_parcels), dtype=np.float64)
    ds_sub = grp.create_dataset("sub", shape=(n_rows,), dtype="S200")
    ds_group = grp.create_dataset("group", shape=(n_rows,), dtype="S100")
    grp.create_dataset("parcel_columns", data=np.array(parcel_col_names, dtype="S200"))
    return ds_mean, ds_sub, ds_group


def _create_mean_std_datasets_iterative(grp, n_rows, n_parcels, parcel_col_names):
    """Create mean, standard_deviation (2D float), sub (1D), group (1D), parcel_columns. Returns (ds_mean, ds_std, ds_sub, ds_group)."""
    for name in ("mean", "standard_deviation", "sub", "group", "parcel_columns"):
        if name in grp:
            del grp[name]
    ds_mean = grp.create_dataset("mean", shape=(n_rows, n_parcels), dtype=np.float64)
    ds_std = grp.create_dataset("standard_deviation", shape=(n_rows, n_parcels), dtype=np.float64)
    ds_sub = grp.create_dataset("sub", shape=(n_rows,), dtype="S200")
    ds_group = grp.create_dataset("group", shape=(n_rows,), dtype="S100")
    grp.create_dataset("parcel_columns", data=np.array(parcel_col_names, dtype="S200"))
    return ds_mean, ds_std, ds_sub, ds_group


def _ensure_means_length(means, n_parcels):
    means = np.asarray(means, dtype=np.float64)
    if len(means) > n_parcels:
        return means[:n_parcels]
    if len(means) < n_parcels:
        return np.pad(means, (0, n_parcels - len(means)), constant_values=np.nan)
    return means


def _ensure_stds_length(stds, n_parcels):
    """Same as _ensure_means_length for standard_deviation arrays."""
    return _ensure_means_length(stds, n_parcels)


def _subject_means_worker(args):
    """Worker for parallel subject loop: (r, scalar_path, atlas_path, n_parcels) -> (sub, group, means, stds) or None."""
    r, scalar_path, atlas_path, n_parcels = args
    if scalar_path is None:
        return None
    try:
        means, stds = _region_mean_std_for_atlas(scalar_path, atlas_path, n_parcels)
        means = _ensure_means_length(means, n_parcels)
        stds = _ensure_stds_length(stds, n_parcels)
        return (r["sub"], r["group"], means, stds)
    except Exception:
        return None


def _run():
    if SUBJECT_WORKERS < 2:
        raise ValueError("SUBJECT_WORKERS must be >= 2 (parallel subject loop only).")
    scalar_to_dir, scalar_to_file = _load_metadata()
    scalar_labels = [
        s for s in scalar_to_file.keys()
        if s not in EXCLUDED_SCALARS
    ]
    groups = _discover_groups()
    if not groups:
        print("No groups found under", _QSIRECON_DIR)
        return

    rows_meta = []
    for group in groups:
        for sub, ses in _discover_subject_sessions(group):
            rows_meta.append({"sub": sub, "group": group, "ses": ses})

    if not rows_meta:
        print("No subject-sessions found. Check:", _QSIRECON_DIR)
        return

    atlases_4s = list(_iter_4s_atlases())
    atlases_glasser = list(_iter_glasser_atlases())
    atlases_random = list(_iter_random_atlases())
    hcp1065_labels, hcp1065_mask_paths = _discover_hcp1065_masks()

    print("Discovery: %d subject-sessions, 4S=%d, Glasser=%d, random=%d, HCP1065=%d"
          % (len(rows_meta), len(atlases_4s), len(atlases_glasser), len(atlases_random), len(hcp1065_labels)))

    if not atlases_4s and not atlases_glasser and not atlases_random and not hcp1065_labels:
        print("No atlases found. Check atlas paths.")
        return

    r0 = rows_meta[0]
    sample_path = _scalar_path(r0["group"], r0["sub"], r0["ses"],
                               scalar_to_dir[scalar_labels[0]], scalar_to_file[scalar_labels[0]])
    if sample_path is None:
        print("Warning: no MNI scalar file for first subject. Expect *space-MNI152NLin2009cAsym_* in .../dwi/")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_rows = len(rows_meta)
    written_files = []

    # Run atlases in ATLAS_INCLUSION order
    for atlas_name in ATLAS_INCLUSION:
        if atlas_name == "Glasser":
            for atlas_key, res_key, nii_path, tsv_path in tqdm(atlases_glasser, desc="Glasser", unit="atlas"):
                atlas_img = nib.load(nii_path)
                labels = _parcel_labels_from_tsv(tsv_path)
                # Use TSV to define n_parcels so all parcels (e.g. 1–1180) are present in the H5
                # even if the atlas image has no voxels for some indices (e.g. 181–1000)
                if labels is not None and len(labels) > 0:
                    n_parcels = len(labels)
                else:
                    # Get all unique, sorted, nonzero labels from the atlas image data
                    labels_data = np.unique(atlas_img.get_fdata())
                    labels_data = labels_data[labels_data != 0]  # Exclude background/zero if present
                    n_parcels = len(labels_data)
                    labels = [f"Parcel_{int(lab)}" for lab in labels_data]
                if labels is not None and len(labels) != n_parcels:
                    # Try to reconstruct unique labels again in case of mismatch
                    labels = [f"Parcel_{int(lab)}" for lab in labels_data]

                h5_path = _h5_path_glasser()
                with h5py.File(h5_path, "w") as f:
                    for scalar_label in tqdm(scalar_labels, desc="Scalars", leave=False, unit="scalar"):
                        scalar_dir = scalar_to_dir[scalar_label]
                        scalar_filename = scalar_to_file[scalar_label]
                        tasks = [
                            (r, _scalar_path(r["group"], r["sub"], r["ses"], scalar_dir, scalar_filename), _abs_path(nii_path), n_parcels)
                            for r in rows_meta
                        ]
                        grp = f.require_group(f"Glasser/{scalar_label}")
                        ds_mean, ds_std, ds_sub, ds_group = _create_mean_std_datasets_iterative(grp, n_rows, n_parcels, labels)
                        with ProcessPoolExecutor(max_workers=SUBJECT_WORKERS) as ex:
                            futures = [ex.submit(_subject_means_worker, t) for t in tasks]
                            for i, fut in enumerate(tqdm(futures, desc="Subjects", leave=False, unit="sub")):
                                res = fut.result()
                                r = rows_meta[i]
                                if res is not None:
                                    sub, group, means, stds = res
                                    _write_mean_std_row(ds_mean, ds_std, ds_sub, ds_group, i, sub, group, means, stds)
                                else:
                                    _write_mean_std_row(ds_mean, ds_std, ds_sub, ds_group, i, r["sub"], r["group"], np.full(n_parcels, np.nan), np.full(n_parcels, np.nan))
                written_files.append(h5_path)

        elif atlas_name == "HCP1065" and hcp1065_labels:
            n_parcels = len(hcp1065_labels)
            reference_scalar_path = None
            for r in rows_meta:
                for scalar_label in scalar_labels:
                    p = _scalar_path(r["group"], r["sub"], r["ses"], scalar_to_dir[scalar_label], scalar_to_file[scalar_label])
                    if p is not None:
                        reference_scalar_path = p
                        break
                if reference_scalar_path is not None:
                    break
            if reference_scalar_path is not None:
                print("Pre-resampling HCP1065 masks to scalar grid (once)...")
                hcp1065_resampled_masks = _build_hcp1065_resampled_masks(reference_scalar_path, hcp1065_mask_paths)
            else:
                hcp1065_resampled_masks = None

            h5_path = _h5_path_hcp1065()
            with h5py.File(h5_path, "w") as f:
                for scalar_label in tqdm(scalar_labels, desc="HCP1065 scalars", leave=False, unit="scalar"):
                    scalar_dir = scalar_to_dir[scalar_label]
                    scalar_filename = scalar_to_file[scalar_label]
                    grp = f.require_group(f"HCP1065/{scalar_label}")
                    ds_mean, ds_std, ds_sub, ds_group = _create_mean_std_datasets_iterative(grp, n_rows, n_parcels, hcp1065_labels)
                    if hcp1065_resampled_masks is None:
                        for i, r in enumerate(tqdm(rows_meta, desc="Subjects", leave=False, unit="sub")):
                            _write_mean_std_row(ds_mean, ds_std, ds_sub, ds_group, i, r["sub"], r["group"], np.full(n_parcels, np.nan), np.full(n_parcels, np.nan))
                    else:
                        for i, r in enumerate(tqdm(rows_meta, desc="Subjects", leave=False, unit="sub")):
                            sub, group, ses = r["sub"], r["group"], r["ses"]
                            scalar_path = _scalar_path(group, sub, ses, scalar_dir, scalar_filename)
                            if scalar_path is None:
                                _write_mean_std_row(ds_mean, ds_std, ds_sub, ds_group, i, sub, group, np.full(n_parcels, np.nan), np.full(n_parcels, np.nan))
                                continue
                            try:
                                means = _region_means_hcp1065_overlapping(scalar_path, hcp1065_resampled_masks)
                                stds = _region_std_hcp1065_overlapping(scalar_path, hcp1065_resampled_masks)
                            except Exception as e:
                                _log_skip(sub, group, scalar_label, e, scalar_path=scalar_path,
                                          extra={"n_masks": len(hcp1065_mask_paths)})
                                _write_mean_std_row(ds_mean, ds_std, ds_sub, ds_group, i, sub, group, np.full(n_parcels, np.nan), np.full(n_parcels, np.nan))
                                continue
                            _write_mean_std_row(ds_mean, ds_std, ds_sub, ds_group, i, sub, group, means, stds)
            written_files.append(h5_path)

        elif atlas_name == "4S":
            for atlas_key, res_key, nii_path, tsv_path in tqdm(atlases_4s, desc="4S atlases", unit="atlas"):
                atlas_img = nib.load(nii_path)
                n_parcels = int(np.max(atlas_img.get_fdata()))
                labels = _parcel_labels_from_tsv(tsv_path)
                if labels is None or len(labels) != n_parcels:
                    labels = [f"Parcel_{i+1}" for i in range(n_parcels)]

                h5_path = _h5_path_4s(res_key)
                with h5py.File(h5_path, "w") as f:
                    for scalar_label in tqdm(scalar_labels, desc="Scalars", leave=False, unit="scalar"):
                        scalar_dir = scalar_to_dir[scalar_label]
                        scalar_filename = scalar_to_file[scalar_label]
                        tasks = [
                            (r, _scalar_path(r["group"], r["sub"], r["ses"], scalar_dir, scalar_filename), _abs_path(nii_path), n_parcels)
                            for r in rows_meta
                        ]
                        grp = f.require_group(f"4S{res_key}/{scalar_label}")
                        ds_mean, ds_std, ds_sub, ds_group = _create_mean_std_datasets_iterative(grp, n_rows, n_parcels, labels)
                        with ProcessPoolExecutor(max_workers=SUBJECT_WORKERS) as ex:
                            futures = [ex.submit(_subject_means_worker, t) for t in tasks]
                            for i, fut in enumerate(tqdm(futures, desc="Subjects", leave=False, unit="sub")):
                                res = fut.result()
                                r = rows_meta[i]
                                if res is not None:
                                    sub, group, means, stds = res
                                    _write_mean_std_row(ds_mean, ds_std, ds_sub, ds_group, i, sub, group, means, stds)
                                else:
                                    _write_mean_std_row(ds_mean, ds_std, ds_sub, ds_group, i, r["sub"], r["group"], np.full(n_parcels, np.nan), np.full(n_parcels, np.nan))
                written_files.append(h5_path)

        elif atlas_name == "random":
            atlases_random_sorted = sorted(atlases_random, key=lambda x: x[1])
            for parcels_n, version_iter in groupby(atlases_random_sorted, key=lambda x: x[1]):
                atlases_this_res = list(version_iter)
                h5_path = _h5_path_random(parcels_n)
                with h5py.File(h5_path, "w") as f:
                    for tissue, pn, version, atlas_path in tqdm(atlases_this_res, desc=f"Random {parcels_n}", unit="atlas"):
                        atlas_img = nib.load(atlas_path)
                        n_parcels = int(np.max(atlas_img.get_fdata()))
                        parcel_cols = [f"Parcel_{i+1}" for i in range(n_parcels)]

                        for scalar_label in tqdm(scalar_labels, desc="Scalars", leave=False, unit="scalar"):
                            scalar_dir = scalar_to_dir[scalar_label]
                            scalar_filename = scalar_to_file[scalar_label]
                            tasks = [
                                (r, _scalar_path(r["group"], r["sub"], r["ses"], scalar_dir, scalar_filename), _abs_path(atlas_path), n_parcels)
                                for r in rows_meta
                            ]
                            grp = f.require_group(f"random{pn}/{tissue}/{version}/{scalar_label}")
                            ds_mean, ds_std, ds_sub, ds_group = _create_mean_std_datasets_iterative(grp, n_rows, n_parcels, parcel_cols)
                            with ProcessPoolExecutor(max_workers=SUBJECT_WORKERS) as ex:
                                futures = [ex.submit(_subject_means_worker, t) for t in tasks]
                                for i, fut in enumerate(tqdm(futures, desc="Subjects", leave=False, unit="sub")):
                                    res = fut.result()
                                    r = rows_meta[i]
                                    if res is not None:
                                        sub, group, means, stds = res
                                        _write_mean_std_row(ds_mean, ds_std, ds_sub, ds_group, i, sub, group, means, stds)
                                    else:
                                        _write_mean_std_row(ds_mean, ds_std, ds_sub, ds_group, i, r["sub"], r["group"], np.full(n_parcels, np.nan), np.full(n_parcels, np.nan))
                written_files.append(h5_path)

    print("Done. Wrote %d file(s):" % len(written_files))
    for p in written_files:
        print("  ", p)


if __name__ == "__main__":
    _run()
