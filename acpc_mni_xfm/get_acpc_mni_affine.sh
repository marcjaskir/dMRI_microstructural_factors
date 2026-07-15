#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"

group=hcpaging

code_dir=$BASE/code/acpc_mni_xfm
qsiprep_group_dir=$BASE/derivatives/qsiprep/${group}
acpc_mni_xfm_group_dir=$BASE/derivatives/acpc_mni_xfm/${group}

for sub_dir in ${qsiprep_group_dir}/sub-*; do
    sub=$(basename ${sub_dir})

    xfm_file=${qsiprep_group_dir}/${sub}/anat/${sub}_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5
    if [[ -f ${xfm_file} ]]; then

        outdir=${acpc_mni_xfm_group_dir}/${sub}

	if [[ ! -f ${outdir}/${sub}_from-ACPC_to-MNI152NLin2009cAsym_AffineTransform.mat ]]; then
                echo ${sub}
                mkdir -p ${outdir}
        	CompositeTransformUtil --disassemble ${xfm_file} ${sub}_from-ACPC_to-MNI152NLin2009cAsym

        	# Move transforms to output directory
        	mv ${code_dir}/00_${sub}_from-ACPC_to-MNI152NLin2009cAsym_AffineTransform.mat ${outdir}/${sub}_from-ACPC_to-MNI152NLin2009cAsym_AffineTransform.mat
        	mv ${code_dir}/01_${sub}_from-ACPC_to-MNI152NLin2009cAsym_DisplacementFieldTransform.nii.gz ${outdir}/${sub}_from-ACPC_to-MNI152NLin2009cAsym_DisplacementFieldTransform.nii.gz
        fi
    fi
done
