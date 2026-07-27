#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
while [ ! -f "$CODE_ROOT/lib/paths.py" ] && [ "$CODE_ROOT" != "/" ]; do
  CODE_ROOT="$(dirname "$CODE_ROOT")"
done
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import project_root; print(project_root())")}"

glasser_dir=$BASE/data/atlases/Glasser
glasser_img_orig=${glasser_dir}/glasser_MNI152NLin2009cAsym_labels_p20.nii.gz
glasser_img_new=${glasser_dir}/atlas-Glasser_space-MNI152NLin2009cAsym_res-1_dseg.nii.gz

# Create tmp directory
tmpdir=$CODE_ROOT/atlases/tmp
mkdir -p ${tmpdir}

# Copy original image to tmp directory
cp ${glasser_img_orig} ${tmpdir}/glasser_MNI152NLin2009cAsym_labels_p20.nii.gz

fslmaths ${tmpdir}/glasser_MNI152NLin2009cAsym_labels_p20.nii.gz -uthr 180 ${tmpdir}/glasser_MNI152NLin2009cAsym_labels_p20_1-180.nii.gz
fslmaths ${tmpdir}/glasser_MNI152NLin2009cAsym_labels_p20.nii.gz -thr 181 ${tmpdir}/glasser_MNI152NLin2009cAsym_labels_p20_181-360.nii.gz
fslmaths ${tmpdir}/glasser_MNI152NLin2009cAsym_labels_p20_181-360.nii.gz -add 820 -thr 1001 ${tmpdir}/glasser_MNI152NLin2009cAsym_labels_p20_1001-1180.nii.gz
fslmaths ${tmpdir}/glasser_MNI152NLin2009cAsym_labels_p20_1-180.nii.gz -add ${tmpdir}/glasser_MNI152NLin2009cAsym_labels_p20_1001-1180.nii.gz ${glasser_img_new}

# Delete tmp directory
rm -rf ${tmpdir}