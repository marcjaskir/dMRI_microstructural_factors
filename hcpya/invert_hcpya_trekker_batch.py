import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
import os

# Path to the directory containing all HCP-YA subjects
subjects_dir = "{project_root()}/derivatives/qsirecon/hcpya"

# Path to the CSV listing already included subjects (to be excluded)
exclude_csv = "{project_root()}/derivatives/metadata/hcp_dwi_trekker_batch-1_hcpya_subjects.csv"

# Path to write the new inverse list
output_csv = "{project_root()}/derivatives/metadata/hcp_dwi_trekker_batch-1_hcpya_subjects-inv.csv"

# Get a set of subject folder names (beginning with sub- and isdir)
all_subjects = {d for d in os.listdir(subjects_dir)
                if d.startswith("sub-") and os.path.isdir(os.path.join(subjects_dir, d))}

# Read subjects to exclude
with open(exclude_csv, "r") as f:
    exclude_subjects = {line.strip() for line in f if line.strip()}

# Get the subjects present in all_subjects but not in exclude_subjects
inverse_subjects = sorted(all_subjects - exclude_subjects)

# Write the result to the output CSV, one per line
with open(output_csv, "w") as f:
    for subj in inverse_subjects:
        f.write(f"{subj}\n")
