#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"

sub=${1}

PROJ_ROOT=$BASE
QSIRECON_IMG=${PROJ_ROOT}/code/qsirecon/qsirecon-1.0.0rc2.sif
HCP1200=${PROJ_ROOT}/data/hcpya/hcp1200/HCP1200
WORK=${PROJ_ROOT}/code/acpc_mni_xfm/work/${sub}; mkdir -p $WORK
OUTDIR=${PROJ_ROOT}/derivatives/acpc_mni_xfm/hcpya; mkdir -p $OUTDIR

apptainer exec \
    --containall \
    --writable-tmpfs \
    -B $PROJ_ROOT:$PROJ_ROOT \
    $QSIRECON_IMG \
    /opt/conda/envs/qsiprep/bin/ingress2qsirecon \
    --participant-label $sub \
    --work_dir $WORK \
    $HCP1200 \
    $OUTDIR \
    hcpya

# Remove the output dwi directory
if [[ -d ${OUTDIR}/${sub}/dwi ]]; then
    rm -rf ${OUTDIR}/${sub}/dwi
fi

# Disassemble xfm file to get the affine matrix
xfm_file=${OUTDIR}/${sub}/anat/${sub}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5
CompositeTransformUtil --disassemble ${xfm_file} ${sub}_from-T1w_to-MNI152NLin2009cAsym

# Move transforms to output directory
mv ${PROJ_ROOT}/code/acpc_mni_xfm/00_${sub}_from-T1w_to-MNI152NLin2009cAsym_AffineTransform.mat ${OUTDIR}/${sub}/${sub}_from-T1w_to-MNI152NLin2009cAsym_AffineTransform.mat
mv ${PROJ_ROOT}/code/acpc_mni_xfm/01_${sub}_from-T1w_to-MNI152NLin2009cAsym_DisplacementFieldTransform.nii.gz ${OUTDIR}/${sub}/${sub}_from-T1w_to-MNI152NLin2009cAsym_DisplacementFieldTransform.nii.gz

# Remove the T1w image/mask
# rm ${OUTDIR}/${sub}/anat/${sub}_desc-preproc_T1w.nii.gz
# rm ${OUTDIR}/${sub}/anat/${sub}_desc-brain_mask.nii.gz

# Move original h5 files to parent directory
mv ${OUTDIR}/${sub}/anat/* ${OUTDIR}/${sub}

# Remove the anat subdirectory, which should be empty
rmdir ${OUTDIR}/${sub}/anat

# Remove work directory
if [[ -d $WORK ]]; then
    rm -rf $WORK
fi