import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
# dipy.stats.qc.neighboring_dwi_correlation(dwi_data, gtab, *, mask=None)

import dipy
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.stats.qc import neighboring_dwi_correlation
import os
import numpy as np
import pandas as pd

qc_dir = f'{project_root()}/data/hcpya/qc'
qsirecon_dir = f'{project_root()}/derivatives/qsirecon/hcpya'

os.makedirs(qc_dir, exist_ok=True)

for sub in sorted(os.listdir(qsirecon_dir)):

    if not sub.startswith('sub-'):
        continue

    sub_id = sub.replace('sub-', '')

    # if sub_id != '100206':
    #     continue

    print(sub_id)

    # Check if the qc file already exists
    if os.path.exists(os.path.join(qc_dir, f'{sub_id}_qc.csv')):
        continue
    
    dwi_file = f'{project_root()}/data/hcpya/hcp1200/HCP1200/{sub_id}/T1w/Diffusion/data.nii.gz'
    dwi_mask_file = f'{project_root()}/data/hcpya/hcp1200/HCP1200/{sub_id}/T1w/Diffusion/nodif_brain_mask.nii.gz'
    bvals_file = f'{project_root()}/data/hcpya/hcp1200/HCP1200/{sub_id}/T1w/Diffusion/bvals'
    bvecs_file = f'{project_root()}/data/hcpya/hcp1200/HCP1200/{sub_id}/T1w/Diffusion/bvecs'
    eddy_file = f'{project_root()}/data/hcpya/hcp1200/HCP1200/{sub_id}/T1w/Diffusion/eddylogs/eddy_unwarped_images.eddy_movement_rms'
    
    dwi_data, dwi_affine, dwi_img = load_nifti(dwi_file, return_img=True)
    dwi_mask_data, dwi_mask_affine, dwi_mask_img = load_nifti(dwi_mask_file, return_img=True)
    bvals, bvecs = read_bvals_bvecs(bvals_file, bvecs_file)
    gtab = gradient_table(bvals, bvecs)

    # Load the eddy file, which is a .txt file with the movement parameters
    eddy_data = np.loadtxt(eddy_file)

    # Skipping the first two rows (zeros), compute the mean of columns 1 and 2, saving to meanRMS_rel-vol1 and meanRMS_rel-lastvol
    meanRMS_rel_vol1 = np.mean(eddy_data[2:, 0])
    meanRMS_rel_lastvol = np.mean(eddy_data[2:, 1])
    
    raw_dwi_neighbor_corr = neighboring_dwi_correlation(dwi_data, gtab)
    masked_dwi_neighbor_corr = neighboring_dwi_correlation(dwi_data, gtab, mask=dwi_mask_data)

    # Save a .csv file with the headers sub, meanRMS_rel-vol1, meanRMS_rel-lastvol, raw_dwi_neighbor_corr, masked_dwi_neighbor_corr
    with open(os.path.join(qc_dir, f'{sub_id}_qc.csv'), 'w') as f:
        f.write(f'sub,meanRMS_rel-vol1,meanRMS_rel-lastvol,raw_dwi_neighbor_corr,masked_dwi_neighbor_corr\n')
        f.write(f'{sub_id},{meanRMS_rel_vol1},{meanRMS_rel_lastvol},{raw_dwi_neighbor_corr},{masked_dwi_neighbor_corr}\n')


# Read in all the csv files and combine into a single csv file
qc_files = [f for f in os.listdir(qc_dir) if f.endswith('.csv')]
qc_data = pd.concat([pd.read_csv(os.path.join(qc_dir, f)) for f in qc_files])

# Sort by sub
qc_data = qc_data.sort_values(by='sub')

# Save
qc_data.to_csv(os.path.join(qc_dir, 'hcpya_dwi_qc.csv'), index=False)

# Delete all the csv files
for f in qc_files:
    os.remove(os.path.join(qc_dir, f))
