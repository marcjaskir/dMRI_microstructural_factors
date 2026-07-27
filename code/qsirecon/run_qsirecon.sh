#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
while [ ! -f "$CODE_ROOT/lib/paths.py" ] && [ "$CODE_ROOT" != "/" ]; do
  CODE_ROOT="$(dirname "$CODE_ROOT")"
done
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import project_root; print(project_root())")}"

group=hcpaging

qsiprep_dir=$BASE/derivatives/qsiprep/${group}
freesurfer_dir=$BASE/derivatives/freesurfer/${group}
qsirecon_dir=$BASE/derivatives/qsirecon/${group}; mkdir -p ${qsirecon_dir}
logs_dir=$CODE_ROOT/qsirecon/logs; mkdir -p ${logs_dir}

# sublist=$BASE/derivatives/metadata/hcp_dwi_trekker_batch-1_hcpaging_subjects.csv
# for sub in $(cat ${sublist}); do
    # sub_dir=${qsiprep_dir}/${sub}
for sub_dir in ${qsiprep_dir}/sub-*; do

    if [[ -d ${sub_dir} ]]; then
        sub=$(basename ${sub_dir})
        
        # Check if qsirecon is finished
        if [[ -f ${qsirecon_dir}/derivatives/qsirecon-DIPYDKI/${sub}.html ]]; then
            if [[ -f ${qsirecon_dir}/derivatives/qsirecon-DSIStudio/${sub}.html ]]; then
                if [[ -f ${qsirecon_dir}/derivatives/qsirecon-MRtrix3_act-HSVS/${sub}.html ]]; then
                    if [[ -f ${qsirecon_dir}/derivatives/qsirecon-NODDI/${sub}.html ]]; then
                        if [[ -f ${qsirecon_dir}/derivatives/qsirecon-TORTOISE_model-MAPMRI/${sub}.html ]]; then
                            if [[ -f ${qsirecon_dir}/derivatives/qsirecon-TORTOISE_model-tensor/${sub}.html ]]; then
                                echo "${sub} already completed qsirecon"
                                continue 
                            fi
                        fi
                    fi
                fi
            fi
        fi

        # Check if qsiprep is finished
        if [[ ! -f ${qsiprep_dir}/${sub}.html ]]; then
            echo "${sub} has not finished qsiprep"
            continue
        fi

        # Check if freesurfer is finished
        if [[ ! -d ${freesurfer_dir}/${sub} ]]; then
            echo "${sub} has not finished freesurfer"
            continue
        elif [[ -f ${freesurfer_dir}/${sub}/scripts/IsRunning.lh+rh ]]; then
            echo "${sub} has not finished freesurfer"
            continue
        fi

        echo "Processing subject ${sub}"

        logs_output=${logs_dir}/${sub}_qsirecon_job-%j.o
        logs_error=${logs_dir}/${sub}_qsirecon_job-%j.e
        sbatch --output=${logs_output} --error=${logs_error} ./qsirecon.sh $sub $group

    fi

done
