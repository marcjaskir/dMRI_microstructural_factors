import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
import csv
import os

UNCLEANED_PATH = "{project_root()}/data/metadata/rid_group_demo.csv"
DERIVATIVES_DIR = "{project_root()}/derivatives/metadata"
OUT_EPILEPSY = os.path.join(DERIVATIVES_DIR, "demo_penn_epilepsy.csv")
OUT_CONTROLS = os.path.join(DERIVATIVES_DIR, "demo_penn_controls.csv")
SUBJECTS_EPILEPSY = os.path.join(DERIVATIVES_DIR, "subjects_penn_epilepsy.csv")
SUBJECTS_CONTROLS = os.path.join(DERIVATIVES_DIR, "subjects_penn_controls.csv")

SEX_MAP = {"1": "M", "2": "F"}

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

            # Get age
            age = row.get('t3_ageatscan', '').strip()

            # For missing age cases, fill in here after looking at REDCap, referencing date of 3T consent
            if row['record_id'] == "661":
                age = 35.6
            elif row['record_id'] == "662":
                age = 16.6
            elif row['record_id'] == "663":
                age = 52.8
            elif row['record_id'] == "664":
                age = 46.9
            elif row['record_id'] == "665":
                age = 49.7
            elif row['record_id'] == "666":
                age = 22.3
            elif row['record_id'] == "667":
                age = 40.0
            elif row['record_id'] == "668":
                age = 31.4
            elif row['record_id'] == "669":
                age = 60.2
            elif row['record_id'] == "670":
                age = 32.2
            elif row['record_id'] == "671":
                age = 24.8
            elif row['record_id'] == "672":
                age = 28.9
            elif row['record_id'] == "825":
                age = 34.4

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
