#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"

qsiprep_dir=$BASE/derivatives/qsiprep
report_dir=${qsiprep_dir}/reports
script_dir=$(pwd)

for sub_dir in $(ls -d ${qsiprep_dir}/sub-*/); do

	sub=$(basename ${sub_dir})
	sub_report_dir=${report_dir}/${sub}

	# check that this subject has completed qsiprep
	if [ ! -f ${qsiprep_dir}/${sub}.html ]; then
		echo "${sub} has not completed qsiprep"
		continue
	fi
	
	# check that report hasn't already been prepared
	if [ -d ${sub_report_dir} ]; then
                echo "${sub} already has a prepared report"
                continue
        else
                mkdir -p ${sub_report_dir}
        fi
	
	# make report
	cd ${sub_report_dir}
	ln -s ../../${sub}.html . # take html file
	mkdir ${sub}
	cd ${sub}
	ln -s ../../../${sub}/figures/ . # take figures in top level directory
	for ses in ../../../${sub}/ses-*; do # loop over sessions, get figures
		sesname=$(basename ${ses})
		mkdir ${sesname}
		cd ${sesname}
		ln -s ../${ses}/figures/ .
		cd -
	done
        
	# return to scriptdir
	cd ${scriptdir}

done
