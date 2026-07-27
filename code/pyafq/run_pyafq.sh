#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
while [ ! -f "$CODE_ROOT/lib/paths.py" ] && [ "$CODE_ROOT" != "/" ]; do
  CODE_ROOT="$(dirname "$CODE_ROOT")"
done
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import project_root; print(project_root())")}"

group="hcpaging"

bundleseg_dir=$BASE/derivatives/bundleseg/${group}
pyafq_dir=$BASE/derivatives/pyafq
logs_dir=$CODE_ROOT/pyafq/logs; mkdir -p ${logs_dir}

n_subs=1

sub_counter=0
for sub_dir in ${bundleseg_dir}/sub-*; do

    sub=$(basename ${sub_dir})

    outputs_dir=${pyafq_dir}/${group}/${sub}
    if [[ -d ${outputs_dir} ]]; then
        echo "Outputs directory already exists for ${sub}. Skipping."
        continue
    fi
    mkdir -p ${outputs_dir}

    echo "Running pyAFQ for ${sub}"

    log_output="${logs_dir}/pyafq_${sub}-%j.o"
    log_error="${logs_dir}/pyafq_${sub}-%j.e"
    sbatch --output=${log_output} --error=${log_error} pyafq.sh ${sub} ${group}
    # ./pyafq.sh ${sub} ${group}

    sub_counter=$((sub_counter + 1))
    if [ ${sub_counter} -ge ${n_subs} ]; then
        break
    fi

done
