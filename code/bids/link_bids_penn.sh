#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
while [ ! -f "$CODE_ROOT/lib/paths.py" ] && [ "$CODE_ROOT" != "/" ]; do
  CODE_ROOT="$(dirname "$CODE_ROOT")"
done
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import project_root; print(project_root())")}"

bids_dir="${PENN_BIDS_DIR:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import config_value; v=config_value('penn_bids_dir'); print(v or '')")}"
if [ -z "$bids_dir" ]; then
  echo "Set penn_bids_dir in config.yaml or export PENN_BIDS_DIR" >&2
  exit 1
fi
target_dir=$BASE/data/penn

# Create target directory if it doesn't exist
mkdir -p $target_dir

# Link the BIDS data
for sub_dir in ${bids_dir}/sub-*; do
    sub=$(basename $sub_dir)

    # Check if there's a subdirectory beginning with ses-research3T
    if ls -d ${sub_dir}/ses-research3T* 1> /dev/null 2>&1; then

        # Check that they have TOP-UP scans under ${sub_dir}/ses-research3T*/fmap/*epi.nii.gz
        if ls ${sub_dir}/ses-research3T*/fmap/*epi.nii.gz 1> /dev/null 2>&1; then
            echo "Linking ${sub}"
            ln -s $sub_dir $target_dir/${sub}
        fi
    fi

done


