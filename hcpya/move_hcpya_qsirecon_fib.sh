#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"

qsirecon_fib_dir=/cbica/home/jaskirm/comp_space/hcpya/derivatives/qsirecon_fib
dsistudio_dir=/Users/mjaskir/cnt/data/borel/sauce/littlab/users/mjaskir/structural_tractometry/derivatives/qsirecon/hcpya/derivatives/qsirecon-DSIStudio

for sub_dir in ${dsistudio_dir}/sub-*; do

	sub=$(basename ${sub_dir})
	echo ${sub}

	trg_dir=${sub_dir}/ses-01/dwi

	rsync -au jaskirm@cubic-login:${qsirecon_fib_dir}/${sub}_space-T1w_dwimap.fib.gz ${trg_dir}/

done
