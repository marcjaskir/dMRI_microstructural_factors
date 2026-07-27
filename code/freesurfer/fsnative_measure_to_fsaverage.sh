#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
while [ ! -f "$CODE_ROOT/lib/paths.py" ] && [ "$CODE_ROOT" != "/" ]; do
  CODE_ROOT="$(dirname "$CODE_ROOT")"
done
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import project_root; print(project_root())")}"

group=${1}

code_dir=$CODE_ROOT/freesurfer
freesurfer_dir=$BASE/derivatives/freesurfer/${group}
fsaverage_dir=${FREESURFER_SUBJECTS_DIR}/fsaverage

cd ${freesurfer_dir}
export SUBJECTS_DIR=${freesurfer_dir}
export FS_LICENSE=$CODE_ROOT/freesurfer/license.txt

# Check if fsaverage directory exists in freesurfer directory - if not, copy it
if [ ! -d ${freesurfer_dir}/fsaverage ]; then
    cp -r ${fsaverage_dir} ${freesurfer_dir}/fsaverage
fi

# Check if fsaverage directory is empty
if [ -z "$(ls -A ${freesurfer_dir}/fsaverage)" ]; then
    echo "fsaverage directory is empty"
    exit 1
fi

# Specify measures to convert (thickness, area, curv, volume)
measures=(thickness area curv volume sulc jacobian_white)
# measures=(thickness)

# Loop through measures
for measure in ${measures[@]}; do

    for subject in ${freesurfer_dir}/sub-*; do
        subject_id=$(basename ${subject})
        echo ${subject_id}

        echo $SUBJECTS_DIR
        
        # Left hemisphere
        mris_preproc --s ${subject_id} \
        --target fsaverage \
        --hemi lh \
        --meas ${measure} \
        --fwhm 10 \
        --out ${freesurfer_dir}/${subject_id}/surf/lh.${measure}.fsaverage.fwhm10.mgz

        # Remove log
        rm ${freesurfer_dir}/${subject_id}/surf/lh.${measure}.fsaverage.fwhm10.mris_preproc.log

        # Convert to mgz to .shape.gii
        mris_convert -c ${freesurfer_dir}/${subject_id}/surf/lh.${measure}.fsaverage.fwhm10.mgz \
        ${SUBJECTS_DIR}/fsaverage/surf/lh.white \
        ${freesurfer_dir}/${subject_id}/surf/lh.${measure}.fsaverage.fwhm10.shape.gii

        wb_command -set-structure ${freesurfer_dir}/${subject_id}/surf/lh.${measure}.fsaverage.fwhm10.shape.gii CORTEX_LEFT

        # Right hemisphere
        mris_preproc --s ${subject_id} \
        --target fsaverage \
        --hemi rh \
        --meas ${measure} \
        --fwhm 10 \
        --out ${freesurfer_dir}/${subject_id}/surf/rh.${measure}.fsaverage.fwhm10.mgz

        # Remove log
        rm ${freesurfer_dir}/${subject_id}/surf/rh.${measure}.fsaverage.fwhm10.mris_preproc.log

        # Convert to mgz to .shape.gii
        mris_convert -c ${freesurfer_dir}/${subject_id}/surf/rh.${measure}.fsaverage.fwhm10.mgz \
        ${SUBJECTS_DIR}/fsaverage/surf/rh.white \
        ${freesurfer_dir}/${subject_id}/surf/rh.${measure}.fsaverage.fwhm10.shape.gii

        wb_command -set-structure ${freesurfer_dir}/${subject_id}/surf/rh.${measure}.fsaverage.fwhm10.shape.gii CORTEX_RIGHT

    done

done

cd ${code_dir}