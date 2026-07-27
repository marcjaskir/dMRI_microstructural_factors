#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
while [ ! -f "$CODE_ROOT/lib/paths.py" ] && [ "$CODE_ROOT" != "/" ]; do
  CODE_ROOT="$(dirname "$CODE_ROOT")"
done
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import project_root; print(project_root())")}"

hcpya_qsirecon_dir=$BASE/derivatives/qsirecon/hcpya

for model_dir in ${hcpya_qsirecon_dir}/derivatives/*; do

    model=$(basename ${model_dir})

    for tar_file in ${model_dir}/*.tar.gz; do

        sub=$(basename ${tar_file} .tar.gz)
        
        # Check if sub is 100206
        # if [[ ${sub} != "100206" ]]; then
        #     continue
        # fi

        # Uncompress and remove .tar.gz file
        mkdir -p ${model_dir}/sub-${sub}
        tar -xzvf ${tar_file} -C ${model_dir}/sub-${sub}
        rm ${tar_file}
        

    done

done


