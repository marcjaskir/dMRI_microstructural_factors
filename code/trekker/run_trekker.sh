#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
while [ ! -f "$CODE_ROOT/lib/paths.py" ] && [ "$CODE_ROOT" != "/" ]; do
  CODE_ROOT="$(dirname "$CODE_ROOT")"
done
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import project_root; print(project_root())")}"

group=penn_epilepsy

hcp_raw_dir="$BASE/data/hcpya/hcp1200/HCP1200"
qsiprep_dir=$BASE/derivatives/qsiprep/${group}
qsirecon_dir=$BASE/derivatives/qsirecon/${group}
surfaces_dir=$BASE/derivatives/surfaces/${group}
trekker_dir=$BASE/derivatives/trekker/${group} && mkdir -p ${trekker_dir}
log_dir=$CODE_ROOT/trekker/logs && mkdir -p ${log_dir}
# sub_list=$BASE/derivatives/covbat/splits/hcpaging_covbat_training_subs.csv
if [[ ${group} == "hcpya" ]]; then
	sub_list=$BASE/derivatives/metadata/hcp_dwi_trekker_batch-1_hcpya_subjects.csv
elif [[ ${group} == "hcpaging" ]]; then
	sub_list=$BASE/derivatives/metadata/hcp_dwi_trekker_batch-1_hcpaging_subjects.csv
fi

c3d_cmd=${ITKSNAP_C3D:-c3d}

# Set the maximum number of subjects to submit
n_subs=100   # Change this value as needed

if [[ ${group} == "penn_epilepsy" || ${group} == "penn_controls" || ${group} == "hcpaging" ]]; then
	space=ACPC
	qsiprep_dir=$BASE/derivatives/qsiprep/${group}
elif [[ ${group} == "hcpya" ]]; then
	space=T1w
fi

# Activate structural_tractometry environment with conda
source activate ${CONDA_PREFIX:-}

sub_counter=0
for sub_dir in ${qsirecon_dir}/derivatives/qsirecon-MRtrix3_act-HSVS/sub-*; do
#for sub in $(cat ${sub_list}); do

	# Stop if we've reached the limit
	if [[ ${sub_counter} -ge ${n_subs} ]]; then
		break
	fi

	#sub_dir=${qsirecon_dir}/derivatives/qsirecon-MRtrix3_act-HSVS/${sub}

	# Make sure it's a directory and not the report
	if [[ ! -d ${sub_dir} ]]; then
		continue
	fi

	# Parse subject and session
	sub=$(basename ${sub_dir})
	if [[ ${group} != "hcpaging" ]]; then
		ses_dir=$(ls -d ${sub_dir}/ses-* | head -n 1)
		ses=$(basename ${ses_dir})
	fi

	# TODO: Remove this later!
	# Check if trekker directory exists for this subject
	#if [[ -d ${trekker_dir}/${sub} ]]; then
	#	continue
	#fi

	# Check if trekker has already been run for this subject
	if [[ -f ${trekker_dir}/${sub}/${sub}_space-ACPC_desc-preproc_trekker.trk ]]; then
		echo "Trekker already run for ${sub}"
		continue
	fi

	# Check if WM FOD .nii.gz exists
	if [[ ${group} != "hcpaging" ]]; then
		fod_nii=${ses_dir}/dwi/${sub}_${ses}_space-${space}_model-msmtcsd_param-fod_label-WM_dwimap.nii.gz
	else
		fod_nii=${sub_dir}/dwi/${sub}_space-${space}_model-msmtcsd_param-fod_label-WM_dwimap.nii.gz
	fi

	if [[ ! -f ${fod_nii} ]]; then

		# Check if WM FOD .mif.gz exists
		if [[ ${group} == "penn_epilepsy" || ${group} == "penn_controls" ]]; then
			fod_mgz=${ses_dir}/dwi/${sub}_${ses}_space-${space}_model-msmtcsd_param-fod_label-WM_dwimap.mif.gz
		elif [[ ${group} == "hcpya" ]]; then
			fod_mgz=${ses_dir}/dwi/${sub}_space-${space}_model-msmtcsd_param-fod_label-WM_dwimap.mif.gz
		elif [[ ${group} == "hcpaging" ]]; then
			fod_mgz=${sub_dir}/dwi/${sub}_space-${space}_model-msmtcsd_param-fod_label-WM_dwimap.mif.gz
		fi

		# Convert WM FOD .mif.gz to .nii.gz if available
		if [[ ! -f ${fod_mgz} ]]; then
			echo "FOD file does not exist for ${sub}"
			continue
		else
			mrconvert ${fod_mgz} ${fod_nii}
		fi
	fi

	# Check for white surface without medial wall
	# white_lh_file=${surfaces_dir}/${sub}/${sub}.lh.white.native_acpc.no_medial_wall.surf.gii
	# white_rh_file=${surfaces_dir}/${sub}/${sub}.rh.white.native_acpc.no_medial_wall.surf.gii
	# if [[ ! -f ${white_lh_file} || ! -f ${white_rh_file} ]]; then
	# 	echo "White surface without medial wall does not exist for ${sub}"
	# 	continue
	# fi

	# Check for HSVS segmentation file
	seg_file=${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_probseg.nii.gz
	if [[ ! -f ${seg_file} ]]; then
		echo "Segmentation file does not exist for ${sub}"
		continue
	fi

	# Split HSVS segmentation file into individual tissue probability maps
	seg_base=${seg_file%.nii.gz}
	seg_output_example=${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_ctx-50pct_sctx-90pct_bin_gm.nii.gz
	if [[ ! -f ${seg_output_example} ]]; then

		echo "Splitting and binarizing HSVS segmentations for ${sub}"
		$FSLDIR/bin/fslsplit ${seg_file} ${seg_base}_ -t

		# Rename tissue segmentations (1=gm_cortex, 2=gm_subcortex, 3=wm, 4=csf, 5=brainstem)
		mv ${seg_base}_0000.nii.gz ${seg_base}_gm_cortex.nii.gz
		mv ${seg_base}_0001.nii.gz ${seg_base}_gm_subcortex.nii.gz
		mv ${seg_base}_0002.nii.gz ${seg_base}_wm.nii.gz
		mv ${seg_base}_0003.nii.gz ${seg_base}_csf.nii.gz
		mv ${seg_base}_0004.nii.gz ${seg_base}_brainstem.nii.gz

		# Get T1w as anatomical reference
		if [[ "$group" == "penn_epilepsy" || "$group" == "penn_controls" || "$group" == "hcpaging" ]]; then
			ref_file="${qsiprep_dir}/${sub}/anat/${sub}_space-ACPC_desc-preproc_T1w.nii.gz"
		elif [[ "$group" == "hcpya" ]]; then
			hcp_sub="${sub#sub-}"
			ref_file="${hcp_raw_dir}/${hcp_sub}/T1w/T1w_acpc_dc_restore.nii.gz"
		fi

		# Reslice tissue segmentations to T1w with c3d
		${c3d_cmd} ${ref_file} ${seg_base}_gm_cortex.nii.gz -reslice-identity -o ${seg_base}_gm_cortex.nii.gz
		${c3d_cmd} ${ref_file} ${seg_base}_gm_subcortex.nii.gz -reslice-identity -o ${seg_base}_gm_subcortex.nii.gz
		${c3d_cmd} ${ref_file} ${seg_base}_wm.nii.gz -reslice-identity -o ${seg_base}_wm.nii.gz
		${c3d_cmd} ${ref_file} ${seg_base}_csf.nii.gz -reslice-identity -o ${seg_base}_csf.nii.gz
		${c3d_cmd} ${ref_file} ${seg_base}_brainstem.nii.gz -reslice-identity -o ${seg_base}_brainstem.nii.gz

		# Create WM mask for streamline seeding (binarized tissue segmentations)
		fslmaths ${seg_base}_wm.nii.gz -bin ${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_bin_wm.nii.gz

		# Create CSF termination masks
		fslmaths ${seg_base}_csf.nii.gz -thr 0.9 -bin ${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_thr-90pct_bin_csf.nii.gz

		# Create GM cortex mask
		fslmaths ${seg_base}_gm_cortex.nii.gz -thr 0.5 -bin ${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_cortex_thr-50pct_bin_gm.nii.gz

		# Create subcortical GM mask
		fslmaths ${seg_base}_gm_subcortex.nii.gz -thr 0.9 -bin ${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_subcortex_thr-90pct_bin_gm.nii.gz

		# Combine GM cortex and subcortical GM masks
		fslmaths ${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_cortex_thr-50pct_bin_gm.nii.gz -add ${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_subcortex_thr-90pct_bin_gm.nii.gz -bin ${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_ctx-50pct_sctx-90pct_bin_gm.nii.gz

		# Remove unused files
		rm ${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_probseg_brainstem.nii.gz
		rm ${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_probseg_csf.nii.gz
		rm ${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_probseg_gm_cortex.nii.gz
		rm ${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_probseg_gm_subcortex.nii.gz
		rm ${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_probseg_wm.nii.gz
		rm ${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_cortex_thr-50pct_bin_gm.nii.gz
		rm ${qsirecon_dir}/${sub}/anat/${sub}_space-ACPC_seg-hsvs_subcortex_thr-90pct_bin_gm.nii.gz

	fi

	# Make output directory
	if [[ ! -d ${trekker_dir}/${sub} ]]; then
		mkdir -p ${trekker_dir}/${sub}
	fi

	echo "Running trekker for ${sub}"

	# Include a dummy ses variable for hcpaging
	if [[ ${group} == "hcpaging" ]]; then
		ses="ses"
	fi

	# Run trekker
	log_output=$CODE_ROOT/trekker/logs/${sub}_trekker_job-%j.o
	log_error=$CODE_ROOT/trekker/logs/${sub}_trekker_job-%j.e
	sbatch --output=${log_output} --error=${log_error} trekker.sh ${sub} ${ses} ${group} ${space}

	# Increment the subject counter
	sub_counter=$((sub_counter+1))

done

