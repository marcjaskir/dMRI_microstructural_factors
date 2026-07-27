#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
while [ ! -f "$CODE_ROOT/lib/paths.py" ] && [ "$CODE_ROOT" != "/" ]; do
  CODE_ROOT="$(dirname "$CODE_ROOT")"
done
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import project_root; print(project_root())")}"

hcp_raw_dir="$BASE/data/hcpya/hcp1200/HCP1200"

# Activate datalad environment with conda
source activate ${DATALAD_CONDA_ENV}

for sub_dir in ${hcp_raw_dir}/*; do
    sub=$(basename ${sub_dir})
    echo ${sub}
    if [[ -f ${sub_dir}/T1w/${sub}/surf/lh.pial ]]; then
    
        datalad drop ${sub_dir}/T1w/${sub}/surf/lh.pial
        datalad drop ${sub_dir}/T1w/${sub}/surf/rh.pial
        datalad drop ${sub_dir}/T1w/${sub}/surf/lh.white
        datalad drop ${sub_dir}/T1w/${sub}/surf/rh.white
        datalad drop ${sub_dir}/T1w/${sub}/surf/lh.sphere.reg
        datalad drop ${sub_dir}/T1w/${sub}/surf/rh.sphere.reg
        datalad drop ${sub_dir}/T1w/${sub}/surf/lh.sphere
        datalad drop ${sub_dir}/T1w/${sub}/surf/rh.sphere


        # Surface measures
        datalad drop ${sub_dir}/T1w/${sub}/surf/lh.thickness
        datalad drop ${sub_dir}/T1w/${sub}/surf/rh.thickness
        datalad drop ${sub_dir}/T1w/${sub}/surf/lh.area
        datalad drop ${sub_dir}/T1w/${sub}/surf/rh.area
        datalad drop ${sub_dir}/T1w/${sub}/surf/lh.curv
        datalad drop ${sub_dir}/T1w/${sub}/surf/rh.curv
        datalad drop ${sub_dir}/T1w/${sub}/surf/lh.volume
        datalad drop ${sub_dir}/T1w/${sub}/surf/rh.volume
        datalad drop ${sub_dir}/T1w/${sub}/surf/lh.jacobian_white
        datalad drop ${sub_dir}/T1w/${sub}/surf/rh.jacobian_white
        datalad drop ${sub_dir}/T1w/${sub}/surf/lh.sulc
        datalad drop ${sub_dir}/T1w/${sub}/surf/rh.sulc
    fi
    echo "--------------------------------"
done