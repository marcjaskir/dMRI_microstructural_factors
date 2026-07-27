import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
import csv
import os

UNCLEANED_PATH = f"{project_root()}/data/metadata/rid_group_demo.csv"
DERIVATIVES_DIR = f"{project_root()}/derivatives/metadata"
OUT_EPILEPSY = os.path.join(DERIVATIVES_DIR, "demo_penn_epilepsy.csv")
OUT_CONTROLS = os.path.join(DERIVATIVES_DIR, "demo_penn_controls.csv")
SUBJECTS_EPILEPSY = os.path.join(DERIVATIVES_DIR, "subjects_penn_epilepsy.csv")
SUBJECTS_CONTROLS = os.path.join(DERIVATIVES_DIR, "subjects_penn_controls.csv")

SEX_MAP = {"1": "M", "2": "F"}


def _load_age_overrides() -> dict:
    """Load optional age overrides from controlled metadata (never hardcode ages in source).

    Expected CSV columns: record_id, age
    Path: controlled_metadata_dir/penn_age_overrides.csv
    """
    override_path = Path(str(controlled_metadata_dir())) / "penn_age_overrides.csv"
    if not override_path.exists():
        # Legacy layout under project_root
        override_path = Path(f"{project_root()}/derivatives/metadata/penn_age_overrides.csv")
    if not override_path.exists():
        return {}
    out: dict = {}
    with open(override_path, newline="") as fh:
        for row in csv.DictReader(fh):
            rid = str(row.get("record_id", "")).strip()
            age = str(row.get("age", "")).strip()
            if rid and age:
                out[rid] = age
    return out


def main():
    # Ensure output directory exists
    os.makedirs(DERIVATIVES_DIR, exist_ok=True)

    # Initialize data structures to hold subjects by group
    epilepsy_data = []
    control_data = []

    # Read and process uncleaned metadata
    with open(UNCLEANED_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip if no t3_subject_id
            subj_id = row.get('t3_subject_id', '').strip()
            if not subj_id:
                continue

            # Skip if missing required fields
            if not row['record_id'] or not row['sex']:
                continue

            # Format subject ID
            try:
                sub = f"sub-RID{int(row['record_id']):04d}"
            except ValueError:
                continue

            # Get age from source CSV; optional controlled overrides for missing ages
            age = row.get('t3_ageatscan', '').strip()
            if not age:
                age_overrides = _load_age_overrides()
                age = age_overrides.get(str(row['record_id']), '')

            # Get sex
            sex = SEX_MAP.get(row['sex'].strip())
            if not sex:
                continue

            # Determine group from t3_subject_id and save data
            if subj_id.startswith('3T_P'):
                epilepsy_data.append({'sub': sub, 'age': age, 'sex': sex})
            elif subj_id.startswith('3T_C'):
                control_data.append({'sub': sub, 'age': age, 'sex': sex})

    # Write demographic data files
    for data, outfile in [(epilepsy_data, OUT_EPILEPSY), (control_data, OUT_CONTROLS)]:
        with open(outfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sub', 'age', 'sex'])
            writer.writeheader()
            writer.writerows(data)

    # Write subject list files (no headers)
    with open(SUBJECTS_EPILEPSY, 'w', newline='') as f:
        for row in epilepsy_data:
            f.write(f"{row['sub']}\n")

    with open(SUBJECTS_CONTROLS, 'w', newline='') as f:
        for row in control_data:
            f.write(f"{row['sub']}\n")

if __name__ == "__main__":
    main()
