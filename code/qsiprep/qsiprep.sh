#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
while [ ! -f "$CODE_ROOT/lib/paths.py" ] && [ "$CODE_ROOT" != "/" ]; do
  CODE_ROOT="$(dirname "$CODE_ROOT")"
done
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import project_root; print(project_root())")}"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --job-name=qsiprep

SUBJ=${1}
GROUP=${2}
SES=${3}

# Specify temp directory for apptainer
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-/tmp}

# Define key directories
DATA=${PENN_BIDS_DIR}
QSIPREP_IMG=${CODE_ROOT}/qsiprep/qsiprep-1.0.0.sif #  path to qsiprep image
FSLIC=${CODE_ROOT}/qsiprep/license.txt
# WORK=${CODE_ROOT}/qsiprep/work && mkdir -p ${WORK}
# OUTDIR=${BASE}/derivatives/qsiprep/${SITE}

# Run qsiprep

# If a session is provided, use it
if [[ ${GROUP} == "penn_controls" ]] || [[ ${GROUP} == "penn_epilepsy" ]]; then

    cmd="apptainer run \
    -B $DATA:/mnt/penn_neurobridge_epilepsy \
    -B $BASE:/mnt/structural_tractometry \
    -B $FSLIC:/mnt/structural_tractometry/code/qsiprep/license.txt \
    $QSIPREP_IMG \
    /mnt/penn_neurobridge_epilepsy/ \
    /mnt/structural_tractometry/derivatives/qsiprep/${GROUP} \
    participant \
    --participant-label $SUBJ \
    --session-id $SES \
    --work-dir /mnt/structural_tractometry/code/qsiprep/work \
    --skip-bids-validation \
    --output-resolution 1.25 \
    --unringing-method rpg \
    --denoise-method dwidenoise \
    --fs-license-file /mnt/structural_tractometry/code/qsiprep/license.txt"

elif [[ ${GROUP} == "hcpaging" ]]; then

    cmd="apptainer run \
    -B $BASE:/mnt/structural_tractometry \
    -B $FSLIC:/mnt/structural_tractometry/code/qsiprep/license.txt \
    $QSIPREP_IMG \
    /mnt/structural_tractometry/data/${GROUP} \
    /mnt/structural_tractometry/derivatives/qsiprep/${GROUP} \
    participant \
    --participant-label $SUBJ \
    --work-dir /mnt/structural_tractometry/code/qsiprep/work \
    --skip-bids-validation \
    --output-resolution 1.25 \
    --unringing-method rpg \
    --denoise-method dwidenoise \
    --fs-license-file /mnt/structural_tractometry/code/qsiprep/license.txt"
fi



echo $cmd
$cmd
