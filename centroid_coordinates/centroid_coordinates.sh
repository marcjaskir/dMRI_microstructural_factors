#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --job-name=centroid_coordinates

sub=${1}

source activate ${CONDA_PREFIX:-}

python centroid_coordinates.py ${sub}