#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
while [ ! -f "$CODE_ROOT/lib/paths.py" ] && [ "$CODE_ROOT" != "/" ]; do
  CODE_ROOT="$(dirname "$CODE_ROOT")"
done
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import project_root; print(project_root())")}"
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --job-name=pyafq

sub=${1}
group=${2}

source activate ${CONDA_PREFIX:-}

echo "Starting pyafq at $(date)"
python pyafq.py ${sub} ${group}
echo "Finished pyafq at $(date)"