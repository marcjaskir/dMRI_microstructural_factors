import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
import os
from os.path import join as ospj
import pandas as pd

# Load tract profiles data (prepped for GAM)
gam_dir = f"{project_root()}/derivatives/gam"

groups=['penn_controls', 'hcpya']

for group in groups:
    profiles_csv = ospj(gam_dir, f'{group}_profiles.csv') # columns: subjectID, group
    profiles_df = pd.read_csv(profiles_csv)
    profiles_subs = profiles_df['subjectID'].unique()

    # If group is penn_controls or penn_epilepsy, load Penn metadata
    if group in ['penn_controls', 'penn_epilepsy']:
        metadata_csv = f"{project_root()}/data/metadata/dwi_metadata.csv"
        metadata_df = pd.read_csv(metadata_csv) # columns: sub, age_t3scan, sex
        metadata_df = metadata_df.rename(columns={'age_t3scan': 'age'})
        metadata_df = metadata_df[['sub', 'age', 'sex']]

    elif group == 'hcpya':
        metadata_csv = f"{project_root()}/data/metadata/hcpya_basic_demo.csv"
        metadata_df = pd.read_csv(metadata_csv) # columns: sub, age, sex

    # Merge metadata_df with profiles_df
    merged_df = pd.merge(profiles_df, metadata_df, left_on="subjectID", right_on='sub', how='inner')

    # Remove the sub column
    merged_df = merged_df.drop(columns=['sub'])

    # Save merged_df
    merged_df.to_csv(ospj(gam_dir, f'{group}_profiles_with_demo.csv'), index=False)

    # Set the permissions of new file to 400
    os.chmod(ospj(gam_dir, f'{group}_profiles_with_demo.csv'), 0o400)





