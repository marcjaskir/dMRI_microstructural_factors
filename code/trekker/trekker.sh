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
#SBATCH --job-name=trekker

sub=${1}
ses=${2}
group=${3}
space=${4}

# Specify temp directory for apptainer
export APPTAINER_TMPDIR=/home/mjaskir/software/apptainer/apptainer_tmp

# Define key directories
TREKKER_IMG=${CODE_ROOT}/trekker/trekker.sif

# HCP-Aging doesn't have ses- directories
if [[ ${group} != "hcpaging" ]]; then

    cmd="apptainer run \
    -B $BASE:/mnt/structural_tractometry \
    $TREKKER_IMG \
    track \
    /mnt/structural_tractometry/derivatives/qsirecon/${group}/derivatives/qsirecon-MRtrix3_act-HSVS/${sub}/${ses}/dwi/${sub}_${ses}_space-${space}_model-msmtcsd_param-fod_label-WM_dwimap.nii.gz \
    --seed /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_bin_wm.nii.gz \
    --seed_count 5000000 \
    --pathway discard_if_ends_inside_A /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_thr-90pct_bin_csf.nii.gz \
    --pathway discard_if_ends_inside_B /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_thr-90pct_bin_csf.nii.gz \
    --pathway stop_after_entry_A /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_ctx-50pct_sctx-90pct_bin_gm.nii.gz \
    --pathway stop_after_entry_B /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_ctx-50pct_sctx-90pct_bin_gm.nii.gz \
    --minlength 20 \
    --maxlength 300 \
    --numberOfThreads 8 \
    --force \
    --output /mnt/structural_tractometry/derivatives/trekker/${group}/${sub}/${sub}_space-ACPC_desc-preproc_trekker.tck"

else

    cmd="apptainer run \
    -B $BASE:/mnt/structural_tractometry \
    $TREKKER_IMG \
    track \
    /mnt/structural_tractometry/derivatives/qsirecon/${group}/derivatives/qsirecon-MRtrix3_act-HSVS/${sub}/dwi/${sub}_space-${space}_model-msmtcsd_param-fod_label-WM_dwimap.nii.gz \
    --seed /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_bin_wm.nii.gz \
    --seed_count 5000000 \
    --pathway discard_if_ends_inside_A /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_thr-90pct_bin_csf.nii.gz \
    --pathway discard_if_ends_inside_B /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_thr-90pct_bin_csf.nii.gz \
    --pathway stop_after_entry_A /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_ctx-50pct_sctx-90pct_bin_gm.nii.gz \
    --pathway stop_after_entry_B /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_ctx-50pct_sctx-90pct_bin_gm.nii.gz \
    --minlength 20 \
    --maxlength 300 \
    --numberOfThreads 8 \
    --force \
    --output /mnt/structural_tractometry/derivatives/trekker/${group}/${sub}/${sub}_space-ACPC_desc-preproc_trekker.tck"

fi

echo $cmd
echo ""
$cmd
echo ""

echo "Converting tck to trk..."

# Activate structural_tractometry environment
source activate ${CONDA_PREFIX:-}

# Define qsiprep root directory
qsiprep_root_dir="$BASE/derivatives/qsiprep"
hcp_raw_dir="$BASE/data/hcpya/hcp1200/HCP1200"

# Convert tck to trk, keeping trk only
tck_file=$BASE/derivatives/trekker/${group}/${sub}/${sub}_space-ACPC_desc-preproc_trekker.tck
trk_file=$BASE/derivatives/trekker/${group}/${sub}/${sub}_space-ACPC_desc-preproc_trekker.trk

# Get T1w as anatomical reference
if [[ "$group" == "penn_epilepsy" || "$group" == "penn_controls" || "$group" == "hcpaging" ]]; then
    ref_file="${qsiprep_root_dir}/${group}/${sub}/anat/${sub}_space-ACPC_desc-preproc_T1w.nii.gz"
elif [[ "$group" == "hcpya" ]]; then
    hcp_sub="${sub#sub-}"
    ref_file="${hcp_raw_dir}/${hcp_sub}/T1w/T1w_acpc_dc_restore.nii.gz"
fi

python scil_tractogram_convert.py ${tck_file} ${trk_file} --reference ${ref_file}

# Wait a few seconds
sleep 3

# Remove tck file after double checking the trk file exists
if [[ -f ${trk_file} ]]; then
    rm ${tck_file}
fi

echo "Done!"

# --seed /mnt/structural_tractometry/derivatives/freesurfer/${sub}/surf/white.native_acpc.surf.gii \
# --seed_surf_useSurfNorm \

# cmd="apptainer run \
# -B $BASE:/mnt/structural_tractometry \
# $TREKKER_IMG \
# track \
# /mnt/structural_tractometry/derivatives/qsirecon/${group}/derivatives/qsirecon-MRtrix3_act-HSVS/${sub}/${ses}/dwi/${sub}_${ses}_space-ACPC_model-msmtcsd_param-fod_label-WM_dwimap.nii.gz \
# --seed /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_bin_wm.nii.gz \
# --seed_count 5000000 \
# --pathway discard_if_ends_inside_A /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_thr-50pct_bin_csf.nii.gz \
# --pathway discard_if_ends_inside_B /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_thr-50pct_bin_csf.nii.gz \
# --minlength 20 \
# --maxlength 300 \
# --numberOfThreads 8 \
# --force \
# --output /mnt/structural_tractometry/derivatives/trekker/${group}/${sub}/${sub}_space-ACPC_desc-preproc_trekker_termination-CSF.tck"

# echo $cmd
# echo ""
# $cmd




# Archived settings (changed on 9/23)
# Attempt with surface-based GM termination
# Got error: NIBRARY::ERROR: stop_after_entry option is not allowed for 2D-interpreted surfaces [/opt/staging/src/nibrary/src/dMRI/tractography/pathway/pathwayAdd.cpp line:595]

# HCP-Aging doesn't have ses- directories
# if [[ ${group} != "hcpaging" ]]; then

#     cmd="apptainer run \
#     -B $BASE:/mnt/structural_tractometry \
#     $TREKKER_IMG \
#     track \
#     /mnt/structural_tractometry/derivatives/qsirecon/${group}/derivatives/qsirecon-MRtrix3_act-HSVS/${sub}/${ses}/dwi/${sub}_${ses}_space-${space}_model-msmtcsd_param-fod_label-WM_dwimap.nii.gz \
#     --seed /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_bin_wm.nii.gz \
#     --seed_count 5000000 \
#     --pathway discard_if_ends_inside_A /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_thr-90pct_bin_csf.nii.gz \
#     --pathway discard_if_ends_inside_B /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_thr-90pct_bin_csf.nii.gz \
#     --pathway stop_after_entry_A /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_subcortex_thr-90pct_bin_gm.nii.gz \
#     --pathway stop_after_entry_B /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_subcortex_thr-90pct_bin_gm.nii.gz \
#     --pathway stop_after_entry_A /mnt/structural_tractometry/derivatives/surfaces/${group}/${sub}/${sub}.lh.white.native_acpc.no_medial_wall.surf.gii \
#     --pathway stop_after_entry_B /mnt/structural_tractometry/derivatives/surfaces/${group}/${sub}/${sub}.lh.white.native_acpc.no_medial_wall.surf.gii \
#     --pathway stop_after_entry_A /mnt/structural_tractometry/derivatives/surfaces/${group}/${sub}/${sub}.rh.white.native_acpc.no_medial_wall.surf.gii \
#     --pathway stop_after_entry_B /mnt/structural_tractometry/derivatives/surfaces/${group}/${sub}/${sub}.rh.white.native_acpc.no_medial_wall.surf.gii \
#     --minlength 20 \
#     --maxlength 300 \
#     --numberOfThreads 8 \
#     --force \
#     --output /mnt/structural_tractometry/derivatives/trekker/${group}/${sub}/${sub}_space-ACPC_desc-preproc_trekker.tck"

# else

#     cmd="apptainer run \
#     -B $BASE:/mnt/structural_tractometry \
#     $TREKKER_IMG \
#     track \
#     /mnt/structural_tractometry/derivatives/qsirecon/${group}/derivatives/qsirecon-MRtrix3_act-HSVS/${sub}/dwi/${sub}_space-${space}_model-msmtcsd_param-fod_label-WM_dwimap.nii.gz \
#     --seed /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_bin_wm.nii.gz \
#     --seed_count 5000000 \
#     --pathway discard_if_ends_inside_A /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_thr-90pct_bin_csf.nii.gz \
#     --pathway discard_if_ends_inside_B /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_thr-90pct_bin_csf.nii.gz \
#     --pathway stop_after_entry_A /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_subcortex_thr-90pct_bin_gm.nii.gz \
#     --pathway stop_after_entry_B /mnt/structural_tractometry/derivatives/qsirecon/${group}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_subcortex_thr-90pct_bin_gm.nii.gz \
#     --pathway stop_after_entry_A /mnt/structural_tractometry/derivatives/surfaces/${group}/${sub}/${sub}.lh.white.native_acpc.no_medial_wall.surf.gii \
#     --pathway stop_after_entry_B /mnt/structural_tractometry/derivatives/surfaces/${group}/${sub}/${sub}.lh.white.native_acpc.no_medial_wall.surf.gii \
#     --pathway stop_after_entry_A /mnt/structural_tractometry/derivatives/surfaces/${group}/${sub}/${sub}.rh.white.native_acpc.no_medial_wall.surf.gii \
#     --pathway stop_after_entry_B /mnt/structural_tractometry/derivatives/surfaces/${group}/${sub}/${sub}.rh.white.native_acpc.no_medial_wall.surf.gii \
#     --minlength 20 \
#     --maxlength 300 \
#     --numberOfThreads 8 \
#     --force \
#     --output /mnt/structural_tractometry/derivatives/trekker/${group}/${sub}/${sub}_space-ACPC_desc-preproc_trekker.tck"

# fi