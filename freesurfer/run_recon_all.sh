#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"

group=hcpaging

qsiprep_dir=$BASE/derivatives/qsiprep/${group}
freesurfer_dir=$BASE/derivatives/freesurfer/${group}
logs_dir=$REPO_ROOT/freesurfer/logs

# Make parent freesurfer directory if it doesn't exist
if [[ ! -d ${freesurfer_dir} ]]; then
    mkdir -p ${freesurfer_dir}
fi

# Iterate over subdirectories in qsiprep_dir
for sub_dir in ${qsiprep_dir}/sub-*; do
    sub=$(basename ${sub_dir})

    if [[ -d ${sub_dir} ]]; then

        # Check if qsiprep finished based on .html file
        if [[ -f ${qsiprep_dir}/${sub}.html ]]; then

            # Check if subject exists in freesurfer directory
            if [[ ! -d ${freesurfer_dir}/${sub} ]]; then

                echo "Running recon-all for ${sub}"
                logs_output=${logs_dir}/${sub}_recon-all_job-%j.o
                logs_error=${logs_dir}/${sub}_recon-all_job-%j.e
                sbatch --output=${logs_output} --error=${logs_error} ./recon_all.sh ${sub} ${group}


            fi

        fi

    fi

done
