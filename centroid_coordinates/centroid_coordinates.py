import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()

# GAM directory : {project_root()}/derivatives/gam/
#     Can get node-wise and region-wise z-scores under pyafq and gm_region_micro subdirectories

# Need to get node-wise and region-wise coordinates
#     GM regions: Compute centroid of region
#     WM tract segments: Get streamline centroid

# Standard library imports
import os
from os.path import join as ospj
import sys
import json

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.cm import tab20
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass

# DIPY imports
import dipy.data as dpd
import dipy.stats.analysis as dsa
import dipy.tracking.streamline as dts
from dipy.data.fetcher import get_two_hcp842_bundles
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.image import load_nifti
from dipy.io.streamline import load_trk, save_tractogram
from dipy.io.utils import create_nifti_header, get_reference_info
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.tracking.utils import density_map
from dipy.tracking.streamline import set_number_of_points, transform_streamlines
from dipy.viz import window, actor, colormap

sub = sys.argv[1]
# sub = "sub-RID1112"

base_dir = f"{project_root()}"
# base_dir = f"/Users/mjaskir/cnt/data/borel/sauce/littlab/users/mjaskir/structural_tractometry"

# WM atlas label
wm_atlas_label = "HCP1065"
gm_atlas_label = "4S156"

wm_atlas_dir = f"{base_dir}/data/atlases/{wm_atlas_label}"
model_dir = f"{wm_atlas_dir}/all_trk"
centroids_dir = f"{wm_atlas_dir}/centroids"
metadata_dir = f"{base_dir}/data/metadata"
hcpya_raw_dir = f"{base_dir}/data/hcpya/hcp1200/HCP1200"
qsiprep_group_dir = f"{base_dir}/derivatives/qsiprep/penn_epilepsy"
qsirecon_group_dir = f"{base_dir}/derivatives/qsirecon/penn_epilepsy"
bundleseg_group_dir = f"{base_dir}/derivatives/bundleseg/penn_epilepsy"
bundleseg_config_dir = f"{base_dir}/code/bundleseg/config"

# Get T1w image
ref_t1w_nii = ospj(qsiprep_group_dir, f"{sub}/anat/{sub}_space-ACPC_desc-preproc_T1w.nii.gz")
ses = [d for d in os.listdir(f"{qsiprep_group_dir}/{sub}") if d.startswith("ses-research3T")][0]
ref_t1w = nib.load(ref_t1w_nii)
t1w = ref_t1w.get_fdata()
acpc_affine, acpc_dimensions, acpc_voxel_sizes, acpc_voxel_order = get_reference_info(ref_t1w)
acpc_nifti_header = create_nifti_header(acpc_affine, acpc_dimensions, acpc_voxel_sizes)

# Load in tract metadata
tract_metadata = pd.read_csv(ospj(wm_atlas_dir, f"{wm_atlas_label}_tract_metadata.csv"))

# Get atlas metadata, including index-label mapping
gm_atlas_dir = f"{base_dir}/data/atlases/4S"
gm_atlas_metadata = pd.read_csv(f"{gm_atlas_dir}/atlas-{gm_atlas_label}Parcels_dseg.tsv", sep="\t")
gm_atlas_labels = gm_atlas_metadata['label'].tolist()
gm_atlas_indices = gm_atlas_metadata['index'].tolist()
gm_atlas_index_to_label = dict(zip(gm_atlas_metadata['index'], gm_atlas_metadata['label']))

# Load in list of diffusion MRI scalars from .json file (they are the keys of the json)
labels_to_filenames = json.load(open(ospj(metadata_dir, "scalar_labels_to_filenames.json")))
labels_to_directories = json.load(open(ospj(metadata_dir, "scalar_labels_to_directories.json")))

# Load in tract labels from bundleseg config
tract_labels = json.load(open(ospj(bundleseg_config_dir, f"config_{wm_atlas_label}_association_projection.json")))
tract_labels = list(tract_labels.keys())
tract_labels = [tract_label.replace('.trk', '') for tract_label in tract_labels]

if gm_atlas_label == "4S156":
    gm_atlas_seg_nii = nib.load(ospj(qsirecon_group_dir, f"{sub}/{ses}/dwi/{sub}_{ses}_space-ACPC_seg-4S156Parcels_dseg.nii.gz"))
    gm_atlas_seg_data = gm_atlas_seg_nii.get_fdata()


print("Computing GM centroids")
gm_centroids = []
gm_atlas_affine = gm_atlas_seg_nii.affine
for gm_atlas_index in gm_atlas_indices:
    gm_parcel_label = gm_atlas_index_to_label[gm_atlas_index]
    print(f"-- {gm_parcel_label}")

    # Mask data by GM atlas index, compute center of mass (voxel coords)
    gm_atlas_seg_data_masked = np.where(gm_atlas_seg_data == gm_atlas_index, 1, 0)
    centroid_vox = center_of_mass(gm_atlas_seg_data_masked)
    # Convert voxel to world (mm) using the segmentation affine
    voxel_h = np.array([centroid_vox[0], centroid_vox[1], centroid_vox[2], 1.0])
    centroid_world = (gm_atlas_affine @ voxel_h)[:3]
    gm_centroids.append({
        "region": gm_parcel_label,
        "centroid_x": centroid_world[0],
        "centroid_y": centroid_world[1],
        "centroid_z": centroid_world[2]
    })

# Make directory for outputs (only once)
outputs_centroid_coordinates_gm_dir = f"{base_dir}/derivatives/centroid_coordinates/{sub}/{gm_atlas_label}"
if not os.path.exists(outputs_centroid_coordinates_gm_dir):
    os.makedirs(outputs_centroid_coordinates_gm_dir)

# Save all centroids into a single CSV file
outputs_centroid_coordinates_gm_path = ospj(outputs_centroid_coordinates_gm_dir, f"{sub}_{gm_atlas_label}_centroid_coordinates.csv")
gm_centroids_df = pd.DataFrame(gm_centroids)
gm_centroids_df.to_csv(outputs_centroid_coordinates_gm_path, index=False)

print(f"Computing WM tract centoids")
for tract_label in tract_labels:

    print(f"-- {tract_label}")

    # Skip C_PO_L, C_PO_R, SLF2_L, SLF2_R
    if tract_label in ["C_PO_L", "C_PO_R", "SLF2_L", "SLF2_R"]:
        print(f"---- Skipping {tract_label} because it is a Cingulum-PO or SLF2 are unsuitable for profiling")
        continue
    
    # Make directory for outputs
    outputs_centroid_coordinates_pyafq_dir = f"{base_dir}/derivatives/centroid_coordinates/{sub}/{wm_atlas_label}"
    if not os.path.exists(outputs_centroid_coordinates_pyafq_dir):
        os.makedirs(outputs_centroid_coordinates_pyafq_dir)

    # Define ouputs and check for existence
    outputs_centroid_coordinates_pyafq_path = ospj(outputs_centroid_coordinates_pyafq_dir, f"{sub}_{tract_label}_centroid_coordinates.csv")
    if os.path.exists(outputs_centroid_coordinates_pyafq_path):
        print(f"---- Skipping {tract_label} because centroid coordinates already exist")
        continue

    # Get .trk file path
    trk_path = ospj(bundleseg_group_dir, f"{sub}/{tract_label}.trk")

    # Check that .trk file exists
    if not os.path.exists(trk_path):
        print(f"---- Skipping {tract_label} because .trk file does not exist")
        continue

    # Load .trk file
    trk = load_trk(trk_path, reference="same", bbox_valid_check=False)
    
    # Check that the .trk file contains at least 5 streamlines
    if len(trk.streamlines) < 5:
        print(f"---- Skipping {tract_label} because .trk file contains less than 5 streamlines")
        continue

    # Get model centroids
    centroids_model = np.load(ospj(centroids_dir, f"{tract_label}_model_centroids.npy"))

    # Reorient streamlines
    trk_streamlines_reoriented = dts.orient_by_streamline(trk.streamlines, centroids_model)
    trk_streamlines_reoriented_weights = dsa.gaussian_weights(trk_streamlines_reoriented)

    # Get streamline segment centroids
    feature = ResampleFeature(nb_points=100)
    metric = AveragePointwiseEuclideanMetric(feature)
    qb = QuickBundles(np.inf, metric=metric)
    qb_cluster = qb.cluster(trk_streamlines_reoriented)
    centroids_trk = qb_cluster.centroids[0]
    
    # Save centroids_trk
    np.savetxt(outputs_centroid_coordinates_pyafq_path, centroids_trk, delimiter=',', fmt='%.6f')

    