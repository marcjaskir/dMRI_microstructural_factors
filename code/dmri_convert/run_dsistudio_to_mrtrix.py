import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
from dsistudio_to_mrtrix import dsistudio_to_mrtrix
import os
from os.path import join as ospj
import subprocess

group = "hcpya"
sub = "sub-100206"

base_dir = str(project_root())

hcp_raw_dir = ospj(base_dir, "data/hcpya/hcp1200/HCP1200")
qsiprep_dir = ospj(base_dir, f"derivatives/qsiprep/{group}")
qsirecon_dir = ospj(base_dir, f"derivatives/qsirecon/{group}")


if group == "hcpya":
    sub_id = sub.split("-")[1]
    ses = "ses-01"
    space = "T1w"

    t1w = ospj(hcp_raw_dir, sub_id, space, "T1w_acpc_dc_restore.nii.gz")

else:
    ses = [d for d in os.listdir(ospj(qsiprep_dir, "derivatives", "qsiprep", sub)) if "ses-" in d][0]
    space = "ACPC"

    t1w = ospj(qsiprep_dir, "derivatives", "qsiprep", sub, "anat", f"{sub}_space-{space}_desc-preproc_T1w.nii.gz")

fib = ospj(qsirecon_dir, "derivatives", "qsirecon-DSIStudio", sub, ses, "dwi", f"{sub}_space-{space}_dwimap.fib.gz")
mif = ospj(qsirecon_dir, "derivatives", "qsirecon-DSIStudio", sub, ses, "dwi", f"{sub}_space-{space}_dwimap.mif")
nii = ospj(qsirecon_dir, "derivatives", "qsirecon-DSIStudio", sub, ses, "dwi", f"{sub}_space-{space}_dwimap.nii.gz")

if not os.path.exists(mif):
    dsistudio_to_mrtrix(fib, t1w, mif)

if not os.path.exists(nii):
    subprocess.run(["mrconvert", mif, nii])

# Remove mif
if os.path.exists(mif):
    os.remove(mif)
