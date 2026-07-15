#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"

group="${1:-penn_controls}"
sub="${2:?Usage: $0 <group> <subject_id>}"

freesurfer_dir=$BASE/derivatives/freesurfer/${group}
output_dir=$BASE/derivatives/freesurfer/${group}/${sub}/surf
mkdir -p ${output_dir}

apptainer exec -B "$BASE":"$BASE" --writable-tmpfs -e "${FREESURFER_IMAGE}" /bin/bash -c "
export SUBJECTS_DIR=${freesurfer_dir}
mri_surf2surf --srcsubject ${sub} \
--srchemi lh \
--sval thickness \
--srcsurfreg sphere.reg \
--sfmt curv \
--trgsubject fsaverage \
--trghemi lh \
--trgsurfreg sphere.reg \
--tval ${output_dir}/fsaverage.lh.thickness.mgh \
--noreshape \
--cortex
"

# Convert .mgh to .gii
apptainer exec -B "$BASE":"$BASE" --writable-tmpfs -e "${FREESURFER_IMAGE}" /bin/bash -c "
mri_convert ${output_dir}/fsaverage.lh.thickness.mgh \
${output_dir}/fsaverage.lh.thickness.shape.gii \
--in_type mgh \
--out_type gii
"

# Remove .mgh file
rm ${output_dir}/fsaverage.lh.thickness.mgh

# mri_surf2surf --srcsubject ${sub} \
# --srchemi lh \
# --srcsurfreg ${sub}/surf/lh.sphere.reg \
# --trgsubject fsaverage \
# --trghemi lh \
# --trgsurfreg ${sub}/surf/fsaverage.lh.sphere.reg \
# --tval ${sub}/surf/fsaverage.lh.thickness \
# --sval ${sub}/surf/lh.thickness \
# --sfmt curv \
# --noreshape \
# --cortex