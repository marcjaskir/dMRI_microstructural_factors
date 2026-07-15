#!/bin/bash
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"

sub=sub-HCA9938309

qsiprep_dir=$BASE/derivatives/qsiprep/hcpaging
qsirecon_dir=$BASE/derivatives/qsirecon/hcpaging
trekker_dir=$BASE/derivatives/trekker/hcpaging

# Check qsiprep subject directory
if [[ -d $qsiprep_dir/${sub} ]]; then
    rm -rf $qsiprep_dir/${sub}
    rm $qsiprep_dir/${sub}.html
fi

# Check qsirecon subject directory
if [[ -d $qsirecon_dir/${sub} ]]; then
    rm -rf $qsirecon_dir/${sub}
fi

# Check qsirecon derivatives directories/html files
if [[ -d $qsirecon_dir/derivatives/qsirecon-DIPYDKI/${sub} ]]; then
    rm -rf $qsirecon_dir/derivatives/qsirecon-DIPYDKI/${sub}
    rm $qsirecon_dir/derivatives/qsirecon-DIPYDKI/${sub}.html
fi

if [[ -d $qsirecon_dir/derivatives/qsirecon-DSIStudio/${sub} ]]; then
    rm -rf $qsirecon_dir/derivatives/qsirecon-DSIStudio/${sub}
    rm $qsirecon_dir/derivatives/qsirecon-DSIStudio/${sub}.html
fi

if [[ -d $qsirecon_dir/derivatives/qsirecon-MRtrix3_act-HSVS/${sub} ]]; then
    rm -rf $qsirecon_dir/derivatives/qsirecon-MRtrix3_act-HSVS/${sub}
    rm $qsirecon_dir/derivatives/qsirecon-MRtrix3_act-HSVS/${sub}.html
fi

if [[ -d $qsirecon_dir/derivatives/qsirecon-NODDI/${sub} ]]; then
    rm -rf $qsirecon_dir/derivatives/qsirecon-NODDI/${sub}
    rm $qsirecon_dir/derivatives/qsirecon-NODDI/${sub}.html
fi

if [[ -d $qsirecon_dir/derivatives/qsirecon-TORTOISE_model-MAPMRI/${sub} ]]; then
    rm -rf $qsirecon_dir/derivatives/qsirecon-TORTOISE_model-MAPMRI/${sub}
    rm $qsirecon_dir/derivatives/qsirecon-TORTOISE_model-MAPMRI/${sub}.html
fi

if [[ -d $qsirecon_dir/derivatives/qsirecon-TORTOISE_model-tensor/${sub} ]]; then
    rm -rf $qsirecon_dir/derivatives/qsirecon-TORTOISE_model-tensor/${sub}
    rm $qsirecon_dir/derivatives/qsirecon-TORTOISE_model-tensor/${sub}.html
fi

# Check trekker subject directory
if [[ -d $trekker_dir/${sub} ]]; then
    rm -rf $trekker_dir/${sub}
fi