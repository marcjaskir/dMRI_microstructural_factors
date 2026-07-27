import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
"""
Optimize tissue mask thresholding: plot GM only, WM only, overlap, and "Unassigned"
voxel counts as a function of probability threshold (0.1 -> 1).
Also compute the probability threshold where the GM only, WM only, and Unassigned voxel counts are closest.

"Unassigned" is defined as: the number of voxels where at least one of the probability
images is nonzero, but neither meets the current threshold.

Additionally, report the threshold at which overlap is 0 for the first time.
"""
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

GM_PATH = f"{project_root()}/data/atlases/MNI/tpl-MNI152NLin2009cAsym_res-1mm_label-gm_probseg.nii.gz"
WM_PATH = f"{project_root()}/data/atlases/MNI/tpl-MNI152NLin2009cAsym_res-1mm_label-wm_probseg.nii.gz"

# Load tissue probability maps
gm_data = nib.load(GM_PATH).get_fdata()
wm_data = nib.load(WM_PATH).get_fdata()

# Thresholds from 0 to 1 (step 0.01)
thresholds = np.arange(0, 1.01, 0.01)

# Mask with at least one nonzero probability, to define "foreground" voxels
tissue_mask = (gm_data > 0) | (wm_data > 0)

# Count voxels above each threshold: GM only, WM only, overlap, Unassigned
gm_only = np.array([np.sum(((gm_data >= t) & (wm_data < t))) for t in thresholds])
wm_only = np.array([np.sum(((wm_data >= t) & (gm_data < t))) for t in thresholds])
overlap = np.array([np.sum(((gm_data >= t) & (wm_data >= t))) for t in thresholds])

# Report the threshold at which overlap is 0 for the first time
overlap_zero_indices = np.where(overlap == 0)[0]
if len(overlap_zero_indices) > 0:
    first_overlap_zero_idx = overlap_zero_indices[0]
    threshold_overlap_zero = thresholds[first_overlap_zero_idx]
    print(f"Threshold at which overlap is zero for the first time: {threshold_overlap_zero:.3f}")
else:
    print("Overlap is never zero within the threshold range.")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, gm_only, color="#2e7d32", linewidth=2, label="GM only")
ax.plot(thresholds, wm_only, color="#1565c0", linewidth=2, label="WM only")
ax.plot(thresholds, overlap, color="#e65100", linewidth=2, label="Overlap")

# If desired, plot a vertical line at threshold_overlap_zero for reference
if len(overlap_zero_indices) > 0:
    ax.axvline(threshold_overlap_zero, color="red", linestyle="--", alpha=0.7, label=f"Overlap=0 @ {threshold_overlap_zero:.3f}")

# Commented out because not defined, but left as a reminder
# ax.axvline(optimal_threshold, color="k", linestyle="--", alpha=0.7, label=f"Optimal @ {optimal_threshold:.3f}")

ax.set_xlabel("Probability threshold")
ax.set_ylabel("Number of voxels above threshold")
ax.set_xlim(0.1, 1)
ax.set_ylim(0, None)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{project_root()}/code/atlases/tissue_mask_threshold_plot.png", dpi=150)
plt.show()
