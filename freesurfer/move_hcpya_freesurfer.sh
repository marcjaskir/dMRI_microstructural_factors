#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"

hcpya_data_dir=$BASE/data/hcpya/hcp1200/HCP1200
hcpya_freesurfer_dir=$BASE/derivatives/freesurfer/hcpya; mkdir -p ${hcpya_freesurfer_dir}

for hcpya_sub_dir in ${hcpya_data_dir}/*; do

    hcpya_sub=$(basename ${hcpya_sub_dir})
    echo ${hcpya_sub}

    hcpya_data_fs_dir=${hcpya_sub_dir}/T1w/${hcpya_sub}
    ls ${hcpya_data_fs_dir}

done