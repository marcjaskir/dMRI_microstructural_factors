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

INPUT_PATH = f"{project_root()}/data/metadata/hcpaging/imagingcollection01.txt"
OUTPUT_PATH = f"{project_root()}/derivatives/metadata/demo_hcpaging.csv"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

results = []
seen_subjects = set()

with open(INPUT_PATH, newline='') as f:
    reader = csv.DictReader(f, delimiter='\t')
    lines = list(reader)
    # Skip the second line (column descriptions)
    if len(lines) > 0 and lines[0][reader.fieldnames[0]].startswith('Description'):
        lines = lines[1:]
    for row in lines:
        sub = row.get('src_subject_id', '').strip()
        age_months = row.get('interview_age', '').strip()
        sex = row.get('sex', '').strip()
        if not sub or not age_months or not sex:
            continue
        try:
            age_years = round(float(age_months) / 12, 1)
        except ValueError:
            continue
            
        subject_id = f'sub-{sub}'
        if subject_id in seen_subjects:
            continue
            
        seen_subjects.add(subject_id)
        results.append({
            'sub': subject_id,
            'age': age_years,
            'sex': sex
        })

# Sort results by subject ID
results.sort(key=lambda x: x['sub'])

with open(OUTPUT_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['sub', 'age', 'sex'])
    writer.writeheader()
    writer.writerows(results)