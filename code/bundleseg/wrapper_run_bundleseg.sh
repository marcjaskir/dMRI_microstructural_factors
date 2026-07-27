#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
while [ ! -f "$CODE_ROOT/lib/paths.py" ] && [ "$CODE_ROOT" != "/" ]; do
  CODE_ROOT="$(dirname "$CODE_ROOT")"
done
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import project_root; print(project_root())")}"

group="hcpaging" # Completed for rerun: hcpya, penn_controls, penn_epilepsy

source activate ${CONDA_PREFIX:-}

trekker_dir="$BASE/derivatives/trekker/${group}"
xfm_dir="$BASE/derivatives/acpc_mni_xfm/${group}"
logs_dir="$CODE_ROOT/bundleseg/logs" && mkdir -p ${logs_dir}

# Set the maximum number of subjects to submit
n_subs=100

sub_counter=0

# If penn_epilepsy, filter subjects using the included CSV
if [[ "${group}" == "penn_epilepsy" ]]; then
    included_csv="$BASE/results/inclusion/penn_epilepsy_included.csv"
    # Read included subject IDs (skip header)
    included_subs=($(tail -n +2 "${included_csv}"))
    # Convert to associative array for quick lookup
    declare -A included_subs_lookup
    for inc_sub in "${included_subs[@]}"; do
        included_subs_lookup["${inc_sub}"]=1
    done
fi

for sub_dir in ${trekker_dir}/sub-*; do
    sub=$(basename ${sub_dir})

    # If penn_epilepsy, skip if not in included list
    if [[ "${group}" == "penn_epilepsy" ]]; then
        if [[ -z "${included_subs_lookup[${sub}]}" ]]; then
            continue
        fi
    fi

    echo ${sub}

    # Check for .trk file
    trk_file=${sub_dir}/${sub}_space-ACPC_desc-preproc_trekker.trk
    if [ ! -f ${trk_file} ]; then
        continue
    fi

    # Check for affine transform file
    if [[ ${group} == "penn_epilepsy" || ${group} == "penn_controls" || ${group} == "hcpaging" ]]; then
        xfm_file="${xfm_dir}/${sub}/${sub}_from-ACPC_to-MNI152NLin2009cAsym_AffineTransform.mat"
    elif [[ ${group} == "hcpya" ]]; then
        xfm_file="${xfm_dir}/${sub}/${sub}_from-T1w_to-MNI152NLin2009cAsym_AffineTransform.mat"
    fi
    if [ ! -f ${xfm_file} ]; then
        continue
    fi
    
    # Check if outputs directory already exists
    outputs_dir=$BASE/derivatives/bundleseg/${group}/${sub}
    if [ -d ${outputs_dir} ]; then
        continue
    fi

    sbatch --job-name=bundleseg_${sub}_${group} \
        --output=${logs_dir}/bundleseg_${group}_${sub}.o \
        --error=${logs_dir}/bundleseg_${group}_${sub}.e \
        --cpus-per-task=8 \
        --mem=48GB \
        ./run_bundleseg.sh ${group} ${sub}

    # Increment subject counter
    sub_counter=$((sub_counter + 1))
    if [[ ${sub_counter} -ge ${n_subs} ]]; then
        break
    fi

done
