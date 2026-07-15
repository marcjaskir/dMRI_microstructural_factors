#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"

logs_dir="$BASE/code/trekker/logs"; mkdir -p ${logs_dir}

sbatch --job-name=convert_tck_to_trk \
    --output=${logs_dir}/convert_tck_to_trk.o \
    --error=${logs_dir}/convert_tck_to_trk.e \
    --cpus-per-task=4 \
    --mem=48GB \
    ./convert_tck_to_trk.sh