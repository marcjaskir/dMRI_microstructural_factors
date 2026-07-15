#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"

import_dir=$BASE/data/import/hcpaging/fmriresults01
output_dir=$BASE/derivatives/freesurfer/hcpaging; mkdir -p ${output_dir}

for sub_dir in ${import_dir}/HCA*; do
    
    sub_hcp=$(basename ${sub_dir})

    # Remove everything past first underscore and add sub- prefix
    sub_id=${sub_hcp/_*}
    sub=sub-${sub_id}

    # Create sub-dir in output_dir
    mkdir -p ${output_dir}/${sub}

    import_sub_dir=${import_dir}/${sub_hcp}/T1w/${sub_hcp}

    for fs_dir in ${import_sub_dir}/*; do

        if [ -d ${fs_dir} ]; then

            if [ ! -d ${output_dir}/${sub} ]; then

                fs_dir_name=$(basename ${fs_dir})
                mv ${fs_dir} ${output_dir}/${sub}

            fi

        fi

    done

done
