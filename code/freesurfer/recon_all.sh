#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
while [ ! -f "$CODE_ROOT/lib/paths.py" ] && [ "$CODE_ROOT" != "/" ]; do
  CODE_ROOT="$(dirname "$CODE_ROOT")"
done
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import project_root; print(project_root())")}"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --job-name=recon_all

sub=${1}
group=${2}

qsiprep_dir=$BASE/derivatives/qsiprep/${group}
freesurfer_dir=$BASE/derivatives/freesurfer/${group}

apptainer exec -B "$BASE":"$BASE" --writable-tmpfs -e "${FREESURFER_IMAGE}" /bin/bash -c "
export SUBJECTS_DIR=${freesurfer_dir}
recon-all -i ${qsiprep_dir}/${sub}/anat/${sub}_space-ACPC_desc-preproc_T1w.nii.gz -s ${sub} -all
"