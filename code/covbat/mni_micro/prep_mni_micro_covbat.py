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
Prep CovBat inputs for mni_micro: read .h5 files from derivatives/mni_micro,
output bat/covar/mean_data/standard_deviation_data per (atlas, parcel, scalar)
to derivatives/covbat/inputs/mni_micro/{atlas}/{parcel}/.
"""
import os
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Config (match gm_region_micro style) ---
base_dir = project_root()
mni_micro_dir = base_dir / "derivatives" / "mni_micro"
covbat_inputs_base = base_dir / "derivatives" / "covbat" / "inputs" / "mni_micro"
# h5_basenames = ["4S156_mni_micro.h5", "Glasser_mni_micro.h5", "HCP1065_mni_micro.h5"]
h5_basenames = ["HCP1065_mni_micro.h5"]

# Inclusion: only subjects in these CSVs (under results/inclusion/)
inclusion_dir = base_dir / "results" / "inclusion"
inclusion_files = [
    (inclusion_dir / "penn_epilepsy_included.csv", "penn_epilepsy"),
    (inclusion_dir / "hcpaging_included.csv", "hcpaging"),
    (inclusion_dir / "hcpya_included.csv", "hcpya"),
    (inclusion_dir / "penn_controls_included.csv", "penn_controls"),
]

# Demographic files
hcpya_demo_path = f"{project_root()}/derivatives/metadata/demo_hcpya.csv"
hcpaging_demo_path = f"{project_root()}/derivatives/metadata/demo_hcpaging.csv"
penn_controls_demo_path = f"{project_root()}/derivatives/metadata/demo_penn_controls.csv"
penn_epilepsy_demo_path = f"{project_root()}/derivatives/metadata/demo_penn_epilepsy.csv"

# Scanner files
hcpya_scanner_path = f"{project_root()}/derivatives/metadata/scanner_ids_hcpya.csv"
hcpaging_scanner_path = f"{project_root()}/derivatives/metadata/scanner_ids_hcpaging.csv"
penn_scanner_path = f"{project_root()}/derivatives/metadata/scanner_ids_penn.csv"

# Build allowed (sub, group) from inclusion CSVs
def load_included_subjects():
    """Return set of (sub, group) for subjects in the four inclusion CSVs."""
    allowed = set()
    for path, group in inclusion_files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "sub" not in df.columns:
            continue
        for sub in df["sub"].astype(str).dropna().unique():
            sub = sub.strip()
            if sub:
                allowed.add((sub, group))
    return allowed


included_subjects = load_included_subjects()

# Load demo and scanner data
hcpya_demo = pd.read_csv(hcpya_demo_path)
hcpaging_demo = pd.read_csv(hcpaging_demo_path)
penn_controls_demo = pd.read_csv(penn_controls_demo_path)
penn_epilepsy_demo = pd.read_csv(penn_epilepsy_demo_path)
hcpya_scanner = pd.read_csv(hcpya_scanner_path)
hcpaging_scanner = pd.read_csv(hcpaging_scanner_path)
penn_scanner = pd.read_csv(penn_scanner_path)


# --- Helpers (same as gm_region_micro) ---
def get_demo_df(group):
    if group == "penn_epilepsy":
        return penn_epilepsy_demo
    elif group == "penn_controls":
        return penn_controls_demo
    elif group == "hcpya":
        return hcpya_demo
    elif group == "hcpaging":
        return hcpaging_demo
    else:
        raise ValueError(f"Unknown group: {group}")


def get_scanner_df(group):
    if group in ("penn_epilepsy", "penn_controls"):
        return penn_scanner
    elif group == "hcpya":
        return hcpya_scanner
    elif group == "hcpaging":
        return hcpaging_scanner
    else:
        raise ValueError(f"Unknown group: {group}")


def get_subject_demo(sub, group):
    demo_df = get_demo_df(group)
    row = demo_df.loc[demo_df["sub"] == sub]
    if row.empty:
        return None
    row = row.iloc[0]
    return {"sub": sub, "age": row["age"], "sex": row["sex"], "group": group}


def get_subject_scanner(sub, group):
    scanner_df = get_scanner_df(group)
    row = scanner_df.loc[scanner_df["sub"] == sub]
    if row.empty:
        return None
    return row.iloc[0]["scanner_id"]


def _decode_h5_strings(ds):
    """Decode HDF5 string dataset (S200/S100 bytes) to list of str."""
    arr = ds[:]
    if arr.dtype.kind == "S" or arr.dtype.kind == "O":
        try:
            return [x.decode("utf-8").rstrip("\x00").strip() if isinstance(x, bytes) else str(x).strip() for x in arr]
        except Exception:
            return [str(x).strip() for x in arr]
    return list(arr)


def _sanitize_parcel(parcel):
    """Replace path-unsafe characters in parcel label."""
    s = str(parcel).strip()
    s = re.sub(r'[/\\:*?"<>|]', "_", s)
    return s or "unknown"


def _atlas_key_from_h5_path(h5_path):
    """e.g. 4S156_mni_micro.h5 -> 4S156, Glasser_mni_micro.h5 -> Glasser."""
    stem = Path(h5_path).stem
    if stem.endswith("_mni_micro"):
        return stem.replace("_mni_micro", "")
    return stem


def prep_mni_micro_covbat(mni_micro_dir_path, covbat_base_path, h5_list):
    for h5_basename in h5_list:
        h5_path = Path(mni_micro_dir_path) / h5_basename
        if not h5_path.exists():
            print(f"Skipping missing file: {h5_path}")
            continue
        atlas_key = _atlas_key_from_h5_path(h5_path)
        print(f"Processing {h5_basename} -> atlas {atlas_key}")

        with h5py.File(h5_path, "r") as f:
            if atlas_key not in f:
                print(f"  Atlas key {atlas_key} not in file; top-level keys: {list(f.keys())}")
                continue
            atlas_grp = f[atlas_key]
            scalars = [k for k in atlas_grp.keys() if isinstance(atlas_grp[k], h5py.Group)]

            # Subjects missing map_rtop (no MAP-MRI qsirecon) are excluded from all covbat inputs for this atlas
            excluded_subjects = set()
            if "map_rtop" in atlas_grp:
                rtop_grp = atlas_grp["map_rtop"]
                if "mean" in rtop_grp:
                    rtop_mean = np.asarray(rtop_grp["mean"])
                    rtop_subs = _decode_h5_strings(rtop_grp["sub"])
                    rtop_groups = _decode_h5_strings(rtop_grp["group"])
                    for i in range(rtop_mean.shape[0]):
                        if np.all(np.isnan(rtop_mean[i, :])):
                            excluded_subjects.add((rtop_subs[i], rtop_groups[i]))
                    if excluded_subjects:
                        print(f"  Excluding {len(excluded_subjects)} subjects missing map_rtop for atlas {atlas_key}")

            for scalar in tqdm(scalars, desc=f"Scalars ({atlas_key})", unit="scalar"):
                grp = atlas_grp[scalar]
                if "mean" not in grp or "standard_deviation" not in grp:
                    continue
                mean_arr = np.asarray(grp["mean"])
                std_arr = np.asarray(grp["standard_deviation"])
                subs = _decode_h5_strings(grp["sub"])
                groups = _decode_h5_strings(grp["group"])
                parcel_columns = _decode_h5_strings(grp["parcel_columns"])

                n_subj, n_parcels = mean_arr.shape
                if std_arr.shape != (n_subj, n_parcels) or len(subs) != n_subj or len(groups) != n_subj:
                    print(f"  Shape mismatch in {atlas_key}/{scalar}, skipping")
                    continue
                if len(parcel_columns) != n_parcels:
                    parcel_columns = [f"Parcel_{i+1}" for i in range(n_parcels)]

                for col_idx, parcel_label in enumerate(parcel_columns):
                    parcel_safe = _sanitize_parcel(parcel_label)
                    output_dir = Path(covbat_base_path) / atlas_key / parcel_safe
                    output_dir.mkdir(parents=True, exist_ok=True)

                    data_dict_mean = {}
                    data_dict_std = {}
                    bat_dict = {}
                    covar_dict = {}

                    for i in range(n_subj):
                        sub = subs[i]
                        group = groups[i]
                        if (sub, group) in excluded_subjects:
                            continue
                        if (sub, group) not in included_subjects:
                            continue
                        demo = get_subject_demo(sub, group)
                        if demo is None:
                            continue
                        scanner_id = get_subject_scanner(sub, group)
                        if pd.isnull(scanner_id):
                            continue
                        mean_val = mean_arr[i, col_idx]
                        std_val = std_arr[i, col_idx]
                        if np.isnan(mean_val) and np.isnan(std_val):
                            continue
                        data_dict_mean[sub] = mean_val
                        data_dict_std[sub] = std_val
                        bat_dict[sub] = {"bat": scanner_id}
                        covar_dict[sub] = {"sub": demo["sub"], "age": demo["age"], "sex": demo["sex"], "group": demo["group"]}

                    if not data_dict_mean:
                        continue

                    # Shared bat and covar (same for mean and standard_deviation)
                    bat_df = pd.DataFrame.from_dict(bat_dict, orient="index")
                    bat_df.index.name = "sub"
                    bat_df.reset_index(inplace=True)
                    covar_df = pd.DataFrame.from_dict(covar_dict, orient="index")
                    covar_df.index.name = "sub"
                    covar_df.reset_index(drop=True, inplace=True)

                    mean_df = pd.DataFrame.from_dict(data_dict_mean, orient="index")
                    mean_df.index.name = "sub"
                    mean_df.columns = [scalar]
                    mean_df.reset_index(inplace=True)
                    std_df = pd.DataFrame.from_dict(data_dict_std, orient="index")
                    std_df.index.name = "sub"
                    std_df.columns = [scalar]
                    std_df.reset_index(inplace=True)

                    bat_fname = f"{parcel_safe}_{scalar}_bat.csv"
                    covar_fname = f"{parcel_safe}_{scalar}_covar.csv"
                    mean_data_fname = f"{parcel_safe}_{scalar}_mean_data.csv"
                    std_data_fname = f"{parcel_safe}_{scalar}_standard_deviation_data.csv"

                    bat_df.to_csv(output_dir / bat_fname, index=False)
                    covar_df.to_csv(output_dir / covar_fname, index=False)
                    mean_df.to_csv(output_dir / mean_data_fname, index=False)
                    std_df.to_csv(output_dir / std_data_fname, index=False)


if __name__ == "__main__":
    print(f"Inclusion: {len(included_subjects)} (sub, group) pairs from penn_epilepsy, hcpaging, hcpya, penn_controls")
    os.makedirs(covbat_inputs_base, exist_ok=True)
    prep_mni_micro_covbat(mni_micro_dir, covbat_inputs_base, h5_basenames)
