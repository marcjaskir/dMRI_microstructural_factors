#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"

# NOTE: This iterates over trekker outputs, so it will only process a subset of HCP-YA subjects for now.

scripts_dir=$BASE/code/hcpya
hcpya_bids_dir=$BASE/data/hcpya/hcp1200/HCP1200
qsirecon_dir=$BASE/derivatives/qsirecon/hcpya
freesurfer_dir=$BASE/derivatives/freesurfer/hcpya; mkdir -p ${freesurfer_dir}
sublist=$BASE/derivatives/metadata/hcp_dwi_trekker_batch-1_hcpya_subjects.csv

# Activate datalad environment with conda
source activate ${DATALAD_CONDA_ENV}

cd ${hcpya_bids_dir}

# for sub_dir in ${qsirecon_dir}/sub-*; do

while read -r sub; do
	# sub=$(basename ${sub_dir})
	sub_id=${sub#sub-}

	# If sub_id is less than 107220
	if [[ ${sub_id} -lt 107220 ]]; then
		continue
	fi


	# Pull T1w image
	datalad get -n ${sub_id}
	datalad get -n ${sub_id}/T1w
	datalad get ${sub_id}/T1w/T1w_acpc_dc_restore.nii.gz
	datalad get ${sub_id}/T1w/T1w_acpc_dc_restore_brain.nii.gz
	datalad get ${sub_id}/T1w/brainmask_fs.nii.gz

	# Pull xfm-related files
	datalad get ${sub_id}/MNINonLinear/xfms/acpc_dc2standard.nii.gz
	datalad get ${sub_id}/MNINonLinear/xfms/standard2acpc_dc.nii.gz
	datalad get ${sub_id}/MNINonLinear/T1w_restore_brain.nii.gz


	# Pull Freesurfer images

	# Volumes
	datalad get -n ${sub_id}/T1w/${sub_id}/mri
	datalad get ${sub_id}/T1w/${sub_id}/mri/nu.mgz

	# Surfaces
	datalad get -n ${sub_id}/T1w/${sub_id}/surf
	datalad get ${sub_id}/T1w/${sub_id}/surf/lh.pial
	datalad get ${sub_id}/T1w/${sub_id}/surf/rh.pial
	datalad get ${sub_id}/T1w/${sub_id}/surf/lh.white
	datalad get ${sub_id}/T1w/${sub_id}/surf/rh.white
	datalad get ${sub_id}/T1w/${sub_id}/surf/lh.sphere.reg
	datalad get ${sub_id}/T1w/${sub_id}/surf/rh.sphere.reg
	datalad get ${sub_id}/T1w/${sub_id}/surf/lh.sphere
	datalad get ${sub_id}/T1w/${sub_id}/surf/rh.sphere


	# Surface measures
	datalad get ${sub_id}/T1w/${sub_id}/surf/lh.thickness
	datalad get ${sub_id}/T1w/${sub_id}/surf/rh.thickness
	datalad get ${sub_id}/T1w/${sub_id}/surf/lh.area
	datalad get ${sub_id}/T1w/${sub_id}/surf/rh.area
	datalad get ${sub_id}/T1w/${sub_id}/surf/lh.curv
	datalad get ${sub_id}/T1w/${sub_id}/surf/rh.curv
	datalad get ${sub_id}/T1w/${sub_id}/surf/lh.volume
	datalad get ${sub_id}/T1w/${sub_id}/surf/rh.volume
	datalad get ${sub_id}/T1w/${sub_id}/surf/lh.jacobian_white
	datalad get ${sub_id}/T1w/${sub_id}/surf/rh.jacobian_white
	datalad get ${sub_id}/T1w/${sub_id}/surf/lh.sulc
	datalad get ${sub_id}/T1w/${sub_id}/surf/rh.sulc

	# Labels
	datalad get ${sub_id}/T1w/${sub_id}/label/lh.cortex.label
	datalad get ${sub_id}/T1w/${sub_id}/label/rh.cortex.label

	# Pull all the eddy logs
	datalad get -n ${sub_id}/T1w/Diffusion
	datalad get -n ${sub_id}/T1w/Diffusion/eddylogs
	datalad get ${sub_id}/T1w/Diffusion/eddylogs/eddy_unwarped_images.eddy_movement_rms

	# Pull DWI data
	# NOTE: Once run_ingress2qsirecon_hcpya.sh is run on everyone, we can clear the original DWI data off the servers
	datalad get -n ${sub_id}/T1w/Diffusion
	datalad get ${sub_id}/T1w/Diffusion/data.nii.gz
	datalad get ${sub_id}/T1w/Diffusion/nodif_brain_mask.nii.gz
	datalad get ${sub_id}/T1w/Diffusion/bvals
	datalad get ${sub_id}/T1w/Diffusion/bvecs

	# Copy Freesurfer data into derivatives

	# Make directories
	mkdir -p ${freesurfer_dir}/${sub}/mri
	mkdir -p ${freesurfer_dir}/${sub}/surf
	mkdir -p ${freesurfer_dir}/${sub}/label

	# Copy Volumes
	if [[ ! -f ${freesurfer_dir}/${sub}/mri/nu.mgz ]]; then
		cp ${sub_id}/T1w/${sub_id}/mri/nu.mgz ${freesurfer_dir}/${sub}/mri/nu.mgz
	fi

	# Copy Surfaces
	if [[ ! -f ${freesurfer_dir}/${sub}/surf/lh.pial ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/lh.pial ${freesurfer_dir}/${sub}/surf/lh.pial
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/rh.pial ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/rh.pial ${freesurfer_dir}/${sub}/surf/rh.pial
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/lh.white ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/lh.white ${freesurfer_dir}/${sub}/surf/lh.white
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/rh.white ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/rh.white ${freesurfer_dir}/${sub}/surf/rh.white
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/lh.sphere.reg ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/lh.sphere.reg ${freesurfer_dir}/${sub}/surf/lh.sphere.reg
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/rh.sphere.reg ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/rh.sphere.reg ${freesurfer_dir}/${sub}/surf/rh.sphere.reg
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/lh.sphere ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/lh.sphere ${freesurfer_dir}/${sub}/surf/lh.sphere
	fi
	
	if [[ ! -f ${freesurfer_dir}/${sub}/surf/rh.sphere ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/rh.sphere ${freesurfer_dir}/${sub}/surf/rh.sphere
	fi

	# Copy Surface measures
	if [[ ! -f ${freesurfer_dir}/${sub}/surf/lh.thickness ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/lh.thickness ${freesurfer_dir}/${sub}/surf/lh.thickness
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/rh.thickness ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/rh.thickness ${freesurfer_dir}/${sub}/surf/rh.thickness
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/lh.area ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/lh.area ${freesurfer_dir}/${sub}/surf/lh.area
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/rh.area ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/rh.area ${freesurfer_dir}/${sub}/surf/rh.area
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/lh.curv ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/lh.curv ${freesurfer_dir}/${sub}/surf/lh.curv
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/rh.curv ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/rh.curv ${freesurfer_dir}/${sub}/surf/rh.curv
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/lh.volume ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/lh.volume ${freesurfer_dir}/${sub}/surf/lh.volume
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/rh.volume ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/rh.volume ${freesurfer_dir}/${sub}/surf/rh.volume
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/lh.jacobian_white ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/lh.jacobian_white ${freesurfer_dir}/${sub}/surf/lh.jacobian_white
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/rh.jacobian_white ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/rh.jacobian_white ${freesurfer_dir}/${sub}/surf/rh.jacobian_white
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/lh.sulc ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/lh.sulc ${freesurfer_dir}/${sub}/surf/lh.sulc
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/surf/rh.sulc ]]; then
		cp ${sub_id}/T1w/${sub_id}/surf/rh.sulc ${freesurfer_dir}/${sub}/surf/rh.sulc
	fi

	# Copy Labels
	if [[ ! -f ${freesurfer_dir}/${sub}/label/lh.cortex.label ]]; then
		cp ${sub_id}/T1w/${sub_id}/label/lh.cortex.label ${freesurfer_dir}/${sub}/label/lh.cortex.label
	fi

	if [[ ! -f ${freesurfer_dir}/${sub}/label/rh.cortex.label ]]; then
		cp ${sub_id}/T1w/${sub_id}/label/rh.cortex.label ${freesurfer_dir}/${sub}/label/rh.cortex.label
	fi

done < ${sublist}
