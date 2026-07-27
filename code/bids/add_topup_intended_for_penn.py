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
import glob
import pandas as pd
import json

bids_dir = f"{project_root()}/data/penn"

def add_field_to_json(json_file, additional_field):

    # Step 1: Read the JSON file and parse its contents
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Step 2: Modify the data structure by adding the desired field
    field_name, field_value = additional_field
    data[field_name] = field_value

    # Step 3: Write the updated data structure back to the JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

for sub in os.listdir(bids_dir):
    if sub.startswith('sub-'):
        sub_dir = os.path.join(bids_dir, sub)
        
        # Check for directories beginning with research3T
        for ses in os.listdir(sub_dir):
            if ses.startswith('ses-research3T'):
                ses_dir = os.path.join(sub_dir, ses)
                
                # Check for topup directory
                fmap_dir = os.path.join(ses_dir, 'fmap')
                if os.path.exists(fmap_dir):

                    # Check for file ending with epi.json
                    epi_json = glob.glob(os.path.join(fmap_dir, '*epi.json'))
                    if len(epi_json) > 0:
                        epi_json = epi_json[0]
                        print(epi_json)
                        add_field_to_json(epi_json, ('IntendedFor', ses + '/' + 'dwi/' + sub + '_' + ses + '_dwi.nii.gz'))

                    
