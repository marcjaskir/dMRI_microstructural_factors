#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"

# Define input directories
pyafq_dir=$BASE/derivatives/pyafq/penn_epilepsy
centroid_coordinates_dir=$BASE/derivatives/centroid_coordinates
logs_dir=$BASE/code/centroid_coordinates/logs; mkdir -p ${logs_dir}

# Specify number of subjects to run TE2I on
n_subs=120

# Iterate over subjects with bundleseg outputs
sub_counter=0
for sub_dir in ${pyafq_dir}/sub-*; do
    sub=$(basename ${sub_dir})

    # Check if centroid coordinates outputs exist for this subject
    if [[ -d ${centroid_coordinates_dir}/${sub} ]]; then
        continue
    fi

    echo "Computing GM and WM tract centroid coordinates (in ACPC space) for ${sub}"
    log_output="${logs_dir}/centroid_coordinates_${sub}-%j.o"
    log_error="${logs_dir}/centroid_coordinates_${sub}-%j.e"
    sbatch --output=${log_output} --error=${log_error} centroid_coordinates.sh ${sub}
    #./centroid_coordinates.sh ${sub}

    sub_counter=$((sub_counter + 1))
    if [ ${sub_counter} -ge ${n_subs} ]; then
        break
    fi

done
