#!/usr/bin/env bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --job-name=tract_asym

set -e
sub="${1:?Usage: tract_asymmetry_job.sh SUBJECT [BASE_DIR]}"
base_dir="${2:-$BASE}"

# Run from code/analysis so "tract_asymmetry" is found as a package
cd "${base_dir}/code/analysis"
source activate ${CONDA_PREFIX:-}

python -m tract_asymmetry.cli --subject "$sub" --base-dir "$base_dir"
