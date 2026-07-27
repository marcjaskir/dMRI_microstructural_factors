import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
import os
from os.path import join as ospj
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# Specify wm_atlas
wm_atlas = "HCP1065"

# Specify pyafq (input) directory
pyafq_dir = f"{project_root()}/derivatives/pyafq"

# Specify covbat (output) directory
covbat_base_dir = f"{project_root()}/derivatives/covbat/inputs/pyafq"
if not os.path.exists(covbat_base_dir):
    os.makedirs(covbat_base_dir)

# Demographic files
hcpya_demo_path = f"{project_root()}/derivatives/metadata/demo_hcpya.csv"
hcpaging_demo_path = f"{project_root()}/derivatives/metadata/demo_hcpaging.csv"
penn_controls_demo_path = f"{project_root()}/derivatives/metadata/demo_penn_controls.csv"
penn_epilepsy_demo_path = f"{project_root()}/derivatives/metadata/demo_penn_epilepsy.csv"

# Load in demo data
hcpya_demo = pd.read_csv(hcpya_demo_path)
hcpaging_demo = pd.read_csv(hcpaging_demo_path)
penn_controls_demo = pd.read_csv(penn_controls_demo_path)
penn_epilepsy_demo = pd.read_csv(penn_epilepsy_demo_path)

# Scanner files
hcpya_scanner_path = f"{project_root()}/derivatives/metadata/scanner_ids_hcpya.csv"
hcpaging_scanner_path = f"{project_root()}/derivatives/metadata/scanner_ids_hcpaging.csv"
penn_scanner_path = f"{project_root()}/derivatives/metadata/scanner_ids_penn.csv"

# Load in scanner data
hcpya_scanner = pd.read_csv(hcpya_scanner_path)
hcpaging_scanner = pd.read_csv(hcpaging_scanner_path)
penn_scanner = pd.read_csv(penn_scanner_path)

# Measures files
measures_json_path = f"{project_root()}/data/metadata/scalar_labels_to_filenames.json"

# Load in bundleseg json
bundleseg_json_path = f"{project_root()}/code/bundleseg/config/config_HCP1065_association_projection.json"

# Specify output base directory
output_base_dir = ff"{project_root()}/derivatives/covbat/inputs/pyafq/{wm_atlas}"


# --- HELPERS ---

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
    if group == "penn_epilepsy" or group == "penn_controls":
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
    return {
        "sub": sub,
        "age": row["age"],
        "sex": row["sex"],
        "group": group
    }

def get_subject_scanner(sub, group):
    scanner_df = get_scanner_df(group)
    row = scanner_df.loc[scanner_df["sub"] == sub]
    if row.empty:
        return None
    row = row.iloc[0]
    return row["scanner_id"]

def prep_covbat_pyafq(pyafq_dir):

    # Read measures json
    measures = list(json.load(open(measures_json_path)).keys())
    scalars_to_omit = ["map_li", "map_am", "dti_txx", "dti_txy", "dti_txz", "dti_tyy", "dti_tyz", "dti_tzz", "dti_ha", "rdi_rd1", "rdi_rd2"]
    measures = [measure for measure in measures if measure not in scalars_to_omit]

    # Read bundleseg json to infer which tracts were segmented
    tracts = json.load(open(bundleseg_json_path)).keys()
    tracts = [tract.replace(".trk", "") for tract in tracts]

    # Omit the following tracts: CBT_L, CBT_R, C_PHP_L, C_PHP_R, DRTT_L, DRTT_R, EMC_L, EMC_R, RST_L, RST_R, SLF2_L, SLF2_R
    tracts_to_omit = ["CBT_L", "CBT_R", "C_PHP_L", "C_PHP_R", "DRTT_L", "DRTT_R", "EMC_L", "EMC_R", "RST_L", "RST_R", "SLF2_L", "SLF2_R"]
    tracts = [tract for tract in tracts if tract not in tracts_to_omit]

    for tract in tracts:

        print(f"Preparing CovBat inputs for {tract}")

        # Define output directory
        output_dir = ospj(output_base_dir, tract)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for measure in tqdm(measures):

            data_mean_dict = {}
            data_sd_dict = {}
            bat_dict = {}
            covar_dict = {}

            profile_mean_suffix = f"{measure}_profile-pyafq_stat-mean.csv"
            profile_sd_suffix = f"{measure}_profile-pyafq_stat-sd.csv"
            bat_fname = f"{tract}_{measure}_bat.csv"
            covar_fname = f"{tract}_{measure}_covar.csv"

            for group_dir in os.listdir(pyafq_dir):
                group_dir_path = ospj(pyafq_dir, group_dir)

                group = os.path.basename(group_dir_path)

                # Skip hidden files
                if group_dir.startswith("."):
                    continue

                subs = [f for f in os.listdir(group_dir_path) if f.startswith("sub-")]
                for i, sub in enumerate(subs):

                    # Get demographics for this subject as a dictionary, e.g., {'sub': 'sub-XXXXXX', 'age': 34, 'sex': 'F', 'group': 'hcpya'}
                    demo = get_subject_demo(sub, group)
                    if demo is None:
                        print(f"Demo not found for {sub} in {group}")
                        continue

                    # Get scanner ID
                    scanner_id = get_subject_scanner(sub, group)
                    if pd.isnull(scanner_id):
                        print(f"Scanner ID not found for {sub} in {group}")
                        continue

                    profile_dir = ospj(group_dir_path, sub, wm_atlas, tract, "profile")
                    profile_mean_path = ospj(profile_dir, profile_mean_suffix)
                    profile_sd_path = ospj(profile_dir, profile_sd_suffix)

                    if not os.path.exists(profile_mean_path):
                        continue
                    if not os.path.exists(profile_sd_path):
                        continue

                    profile_mean = pd.read_csv(profile_mean_path, header=None)
                    profile_mean = profile_mean.iloc[:, 0].values
                    profile_sd = pd.read_csv(profile_sd_path, header=None)
                    profile_sd = profile_sd.iloc[:, 0].values

                    data_mean_dict[sub] = profile_mean
                    data_sd_dict[sub] = profile_sd
                    bat_dict[sub] = {"bat": scanner_id}
                    covar_dict[sub] = {"sub": demo["sub"], "age": demo["age"], "sex": demo["sex"], "group": demo["group"]}

            if not data_mean_dict:
                continue

            n_nodes = len(next(iter(data_mean_dict.values())))
            node_cols = [f"{measure}_node{i+1}" for i in range(n_nodes)]

            # Mean data
            data_mean_df = pd.DataFrame.from_dict(data_mean_dict, orient="index")
            data_mean_df.index.name = "sub"
            data_mean_df.columns = node_cols
            data_mean_df.reset_index(inplace=True)
            data_mean_df.to_csv(ospj(output_dir, f"{tract}_{measure}_mean_data.csv"), index=False)

            # Standard deviation data
            data_sd_df = pd.DataFrame.from_dict(data_sd_dict, orient="index")
            data_sd_df.index.name = "sub"
            data_sd_df.columns = node_cols
            data_sd_df.reset_index(inplace=True)
            data_sd_df.to_csv(ospj(output_dir, f"{tract}_{measure}_standard_deviation_data.csv"), index=False)

            # Bat and covar (same for both stats)
            bat_df = pd.DataFrame.from_dict(bat_dict, orient="index")
            bat_df.index.name = "sub"
            bat_df.reset_index(inplace=True)
            bat_df.to_csv(ospj(output_dir, bat_fname), index=False)

            covar_df = pd.DataFrame.from_dict(covar_dict, orient="index")
            covar_df.index.name = "sub"
            covar_df.reset_index(drop=True, inplace=True)
            covar_df.to_csv(ospj(output_dir, covar_fname), index=False)


# --- MAIN ---
prep_covbat_pyafq(pyafq_dir)
