import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
import sys
import os
from scil_tractogram_segment_with_bundleseg import main as bundleseg_main
from datetime import datetime
from dipy.tracking.utils import density_map
from dipy.io.streamline import load_trk, save_trk
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.utils import create_nifti_header, get_reference_info
import nibabel as nib
import numpy as np
from os.path import join as ospj

import warnings
warnings.filterwarnings("ignore")

group = sys.argv[1]
sub = sys.argv[2]

# Load in inputs for bundleseg
in_tractograms = f"{project_root()}/derivatives/trekker/{group}/{sub}/{sub}_space-ACPC_desc-preproc_trekker.trk"
in_config_file = "{project_root()}/code/bundleseg/config/config_HCP1065_association_projection.json"
# in_config_file = "{project_root()}/code/bundleseg/config/config_C_PO_L.json"
in_directory = "{project_root()}/data/atlases/HCP1065/association_projection_bundleseg"

if group in ["penn_epilepsy", "penn_controls", "hcpaging"]:
    in_transfo = f"{project_root()}/derivatives/acpc_mni_xfm/{group}/{sub}/{sub}_from-ACPC_to-MNI152NLin2009cAsym_AffineTransform.mat"
elif group == "hcpya":
    in_transfo = f"{project_root()}/derivatives/acpc_mni_xfm/{group}/{sub}/{sub}_from-T1w_to-MNI152NLin2009cAsym_AffineTransform.mat"

out_dir = f"{project_root()}/derivatives/bundleseg/{group}/{sub}"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Define directories for anatomical reference
qsiprep_dir = f"{project_root()}/derivatives/qsiprep/{group}"
hcp_raw_dir = "{project_root()}/data/hcpya/hcp1200/HCP1200"

# The bundleseg script expects: script_name, tractograms, config_file, directory, transfo, [optional args]
sys.argv = [
    "scil_tractogram_segment_with_bundleseg.py",
    in_tractograms,
    in_config_file,
    in_directory,
    in_transfo,
    "--out_dir", out_dir,
    "-f",
    "--processes", "8"
    ]

print(f"Bundleseg started at {group} {sub} at {datetime.now()}")
bundleseg_main()
print(f"Bundleseg finished at {datetime.now()}\n")

# Iterate over .trk files in out_dir
# print("Saving .trk files as .nii.gz files in ACPC space")
# for trk_file in os.listdir(out_dir):
#     if trk_file.endswith(".trk"):

#         # Get T1w as anatomical reference
#         if group in ["penn_epilepsy", "penn_controls", "hcpaging"]:
#             t1w = nib.load(ospj(qsiprep_dir, sub, "anat", f"{sub}_space-ACPC_desc-preproc_T1w.nii.gz"))
#         elif group == "hcpya":
#             hcp_sub = sub.replace("sub-", "")
#             t1w = nib.load(ospj(hcp_raw_dir, hcp_sub, "T1w", "T1w_acpc_dc_restore.nii.gz"))
#         acpc_affine, acpc_dimensions, acpc_voxel_sizes, acpc_voxel_order = get_reference_info(t1w)
#         acpc_nifti_header = create_nifti_header(acpc_affine, acpc_dimensions, acpc_voxel_sizes)
    
#         # Load in .trk file
#         trk = load_trk(ospj(out_dir, trk_file), reference=t1w)
        
#         # Save segmented streamlines as .nii.gz files in ACPC space
#         trk.to_vox()
#         trk.to_corner()
#         segmented_acpc_density = density_map(trk.streamlines, np.eye(4), acpc_dimensions)
#         nib.save(nib.Nifti1Image(segmented_acpc_density, acpc_affine, acpc_nifti_header), ospj(out_dir, trk_file.replace(".trk", ".nii.gz")))

#         print(f"-- Saved {trk_file.replace('.trk', '.nii.gz')}")