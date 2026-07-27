#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
while [ ! -f "$CODE_ROOT/lib/paths.py" ] && [ "$CODE_ROOT" != "/" ]; do
  CODE_ROOT="$(dirname "$CODE_ROOT")"
done
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import project_root; print(project_root())")}"

# trekker_dir=$BASE/derivatives/trekker/hcpya
hcp_raw_dir=$BASE/data/hcpya/hcp1200/HCP1200
outdir=$BASE/derivatives/acpc_mni_xfm/hcpya
logs_dir=$CODE_ROOT/acpc_mni_xfm/logs
mkdir -p ${logs_dir}

sublist=$BASE/derivatives/metadata/hcp_dwi_trekker_batch-1_hcpya_subjects.csv

n_subs=1

# NOTE: Keep in mind that if not using sublist, fix so that input to ingress2qsirecon_hcpya.sh includes sub- prefix

sub_counter=0
for sub_dir in ${hcp_raw_dir}/100206*; do
# while read -r sub_id; do
    sub_id=$(basename ${sub_dir})
    sub=sub-${sub_id}
    
    # sub_dir=${hcp_raw_dir}/${sub_id}


    # First check if already run
    # if [[ -f ${outdir}/sub-${sub_id}/sub-${sub_id}_from-T1w_to-MNI152NLin2009cAsym_AffineTransform.mat ]]; then
    #     echo "-- Already run for ${sub_id}"
    #     continue
    # fi

    if [[ -f ${outdir}/${sub}/${sub}_from-T1w_to-MNI152NLin2009cAsym_AffineTransform.mat ]]; then
        echo "-- Already run for ${sub_id}"
        continue
    fi

    logs_output=${logs_dir}/ingress2qsirecon_hcpya_${sub}.o
    logs_error=${logs_dir}/ingress2qsirecon_hcpya_${sub}.e

    sbatch --output=${logs_output} --error=${logs_error} --mem=4GB ./ingress2qsirecon_hcpya.sh ${sub}

    sub_counter=$((sub_counter + 1))
    if [[ $sub_counter -ge $n_subs ]]; then
        break
    fi

done
# done < ${sublist}