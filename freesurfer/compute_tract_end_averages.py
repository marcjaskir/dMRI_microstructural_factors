import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
import os
import sys
import glob
import numpy as np
import nibabel as nib
import pandas as pd

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Compute tract end averages for surface measures.")
    parser.add_argument(
        "--group",
        required=True,
        choices=["hcpaging", "hcpya", "penn_controls", "penn_epilepsy"],
        help="Subject group"
    )
    return parser.parse_args()

def get_tract_and_end(mask_fname):
    # Example: C_FP_R_end-P.rh.shape.gii
    base = os.path.basename(mask_fname)
    if "_end-" not in base:
        raise ValueError(f"Mask filename {base} does not contain '_end-'")
    tract = base.split("_end-")[0]
    end = base.split("_end-")[1].split(".")[0]  # e.g., P.rh
    # end may have .lh or .rh in it, so split further
    if "." in end:
        end, hemi = end.split(".", 1)
    else:
        hemi = None
    return tract, end, hemi

def get_hemi_from_tract(tract):
    # Tracts end with _L or _R
    if tract.endswith("_L"):
        return "lh"
    elif tract.endswith("_R"):
        return "rh"
    else:
        raise ValueError(f"Tract {tract} does not end with _L or _R")

def get_mask_files(mask_dir):
    # All .gii files in mask_dir
    return sorted(glob.glob(os.path.join(mask_dir, "*.shape.gii")))

def get_subject_dirs(surf_dir):
    # All sub-* directories
    return sorted(glob.glob(os.path.join(surf_dir, "sub-*")))

def get_surface_file(sub_surf_dir, hemi, measure):
    # Example: lh.sulc.fsaverage.fwhm10.shape.gii
    pattern = f"{hemi}.{measure}.fsaverage.fwhm10.shape.gii"
    fpath = os.path.join(sub_surf_dir, pattern)
    if os.path.exists(fpath):
        return fpath
    else:
        return None

def main():
    args = parse_args()
    group = args.group

    mask_dir = "{project_root()}/data/atlases/HCP1065/endpoint_gii"
    surf_dir = f"{project_root()}/derivatives/freesurfer/{group}"
    output_dir = f"{project_root()}/derivatives/freesurfer_post/{group}"
    os.makedirs(output_dir, exist_ok=True)

    surface_measures = ["area", "curv", "jacobian_white", "sulc", "thickness", "volume"]

    mask_files = get_mask_files(mask_dir)
    subject_dirs = get_subject_dirs(surf_dir)

    # For each mask (tract end)
    for mask_file in mask_files:
        tract, end, mask_hemi = get_tract_and_end(mask_file)
        hemi = get_hemi_from_tract(tract)  # 'lh' or 'rh'

        # Only process if mask file matches hemisphere
        if mask_hemi is not None and not mask_file.endswith(f".{hemi}.shape.gii"):
            continue

        # Load mask data
        mask_gii = nib.load(mask_file)
        mask_data = mask_gii.darrays[0].data
        # Binarize mask (in case it's not exactly 0/1)
        mask_bin = (mask_data > 0).astype(float)

        for measure in surface_measures:
            results = []
            for sub_dir in subject_dirs:
                sub = os.path.basename(sub_dir)
                sub_surf_dir = os.path.join(sub_dir, "surf")
                surf_file = get_surface_file(sub_surf_dir, hemi, measure)
                if surf_file is None or not os.path.exists(surf_file):
                    continue
                try:
                    surf_gii = nib.load(surf_file)
                    surf_data = surf_gii.darrays[0].data
                except Exception as e:
                    print(f"Error loading {surf_file}: {e}", file=sys.stderr)
                    continue

                # Multiply mask and surface
                masked_values = surf_data * mask_bin
                # Only consider nonzero mask locations
                masked_values = masked_values[mask_bin > 0]
                if masked_values.size == 0:
                    mean_val = np.nan
                    median_val = np.nan
                else:
                    mean_val = np.nanmean(masked_values)
                    median_val = np.nanmedian(masked_values)
                results.append({
                    "sub": sub,
                    "group": group,
                    f"{measure}_mean": mean_val,
                    f"{measure}_median": median_val
                })

            # Save results to CSV
            out_fname = f"{tract}_end-{end}_{measure}.csv"
            out_fpath = os.path.join(output_dir, out_fname)
            df = pd.DataFrame(results)
            # Ensure columns order
            df = df[["sub", "group", f"{measure}_mean", f"{measure}_median"]]
            df.to_csv(out_fpath, index=False)
            print(f"Saved: {out_fpath}")

if __name__ == "__main__":
    main()

