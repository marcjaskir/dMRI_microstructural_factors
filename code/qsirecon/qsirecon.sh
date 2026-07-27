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
#SBATCH --mem=48GB
#SBATCH --job-name=qsirecon

SUBJ=${1}
GROUP=${2}
# SUBJ=sub-EXAMPLE

# Specify temp directory for apptainer
export APPTAINER_TMPDIR=/home/mjaskir/software/apptainer/apptainer_tmp

# Define key directories
QSIRECON_IMG=${CODE_ROOT}/qsirecon/qsirecon-1.0.0rc2.sif #  path to qsiprep image
FSLIC=${CODE_ROOT}/qsirecon/license.txt
# CHANGE THIS BACK LATER
WORK=${CODE_ROOT}/qsirecon/work_hcpaging && mkdir -p ${WORK}
OUTDIR=${BASE}/derivatives/qsirecon/${GROUP}
GLASSER_ATLAS=${CODE_ROOT}/qsirecon/glasser_atlas

# CHANGE THIS BACK LATER
if [[ ${GROUP} == "penn_pnes" || ${GROUP} == "penn_epilepsy" ]]; then
    RECON_SPEC=recon_spec_custom_penn.yaml
elif [[ ${GROUP} == "hcpaging" ]]; then
    RECON_SPEC=recon_spec_custom_hcpaging_mapmri.yaml
fi

# CHANGE WORKING DIRECTORY BACK LATER
cmd="apptainer run \
-B $BASE:/mnt/structural_tractometry \
-B $FSLIC:/mnt/structural_tractometry/code/qsirecon/license.txt \
-B $GLASSER_ATLAS:/glasser_atlas \
$QSIRECON_IMG \
/mnt/structural_tractometry/derivatives/qsiprep/${GROUP} \
/mnt/structural_tractometry/derivatives/qsirecon/${GROUP} \
participant \
--participant-label $SUBJ \
--work-dir /mnt/structural_tractometry/code/qsirecon/work_hcpaging \
--nthreads 4 \
--omp-nthreads 4 \
--atlases 4S456Parcels Glasser \
--datasets /glasser_atlas \
--fs-subjects-dir /mnt/structural_tractometry/derivatives/freesurfer/${GROUP} \
--fs-license-file /mnt/structural_tractometry/code/qsirecon/license.txt \
--recon-spec /mnt/structural_tractometry/code/qsirecon/${RECON_SPEC}"

echo $cmd
$cmd
