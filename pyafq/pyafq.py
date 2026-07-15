import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
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
group = sys.argv[2]

# Render indicator (Only use if running locally)
render = False
if render:
    # FURY imports
    from fury import actor, window
    from fury.colormap import create_colormap

base_dir = f"{project_root()}"
# base_dir = f"/Users/mjaskir/cnt/data/borel/sauce/littlab/users/mjaskir/structural_tractometry"

# WM atlas label
atlas_label = "HCP1065"

atlas_dir = f"{base_dir}/data/atlases/{atlas_label}"
model_dir = f"{atlas_dir}/all_trk"
centroids_dir = f"{atlas_dir}/centroids"
metadata_dir = f"{base_dir}/data/metadata"
hcpya_raw_dir = f"{base_dir}/data/hcpya/hcp1200/HCP1200"
qsiprep_group_dir = f"{base_dir}/derivatives/qsiprep/{group}"
qsirecon_group_dir = f"{base_dir}/derivatives/qsirecon/{group}"
bundleseg_group_dir = f"{base_dir}/derivatives/bundleseg/{group}"
bundleseg_config_dir = f"{base_dir}/code/bundleseg/config"

# Get T1w image
if group == "penn_controls" or group == "penn_epilepsy":
    ref_t1w_nii = ospj(qsiprep_group_dir, f"{sub}/anat/{sub}_space-ACPC_desc-preproc_T1w.nii.gz")
    ses = [d for d in os.listdir(f"{qsiprep_group_dir}/{sub}") if d.startswith("ses-research3T")][0]
elif group == "hcpaging":
    ref_t1w_nii = ospj(qsiprep_group_dir, f"{sub}/anat/{sub}_space-ACPC_desc-preproc_T1w.nii.gz")
elif group == "hcpya":
    hcp_id = sub.split("-")[1]
    ref_t1w_nii = ospj(hcpya_raw_dir, f"{hcp_id}/T1w/T1w_acpc_dc_restore.nii.gz")
ref_t1w = nib.load(ref_t1w_nii)
t1w = ref_t1w.get_fdata()
acpc_affine, acpc_dimensions, acpc_voxel_sizes, acpc_voxel_order = get_reference_info(ref_t1w)
acpc_nifti_header = create_nifti_header(acpc_affine, acpc_dimensions, acpc_voxel_sizes)

# Load in tract metadata
tract_metadata = pd.read_csv(ospj(atlas_dir, f"{atlas_label}_tract_metadata.csv"))

# Load in list of diffusion MRI scalars from .json file (they are the keys of the json)
labels_to_filenames = json.load(open(ospj(metadata_dir, "scalar_labels_to_filenames.json")))
labels_to_directories = json.load(open(ospj(metadata_dir, "scalar_labels_to_directories.json")))

# Load in tract labels from bundleseg config
tract_labels = json.load(open(ospj(bundleseg_config_dir, f"config_{atlas_label}_association_projection.json")))
tract_labels = list(tract_labels.keys())
tract_labels = [tract_label.replace('.trk', '') for tract_label in tract_labels]

# ---- Utility functions ----
def get_streamline_segment(sl, which_end, proportion=1/3):
    """
    Extracts a proportion of points from the start or end of a streamline.
    Args:
        sl: streamline (Nx3 array)
        which_end: 'end1', 'core', 'end2'
        proportion: float in (0,1], proportion of points to extract
    Returns:
        Subset of streamline points
    """
    n_points = len(sl)
    n_extract = max(1, int(np.round(proportion * n_points)))
    if which_end == 'end1':
        return sl[:n_extract]
    elif which_end == 'end2':
        return sl[-n_extract:]
    elif which_end == 'core':
        return sl[n_extract:-n_extract]
    else:
        raise ValueError("which_end must be 'end1', 'end2', or 'core'")

for tract_label in tract_labels:

    print(f"{tract_label}")

    # Skip Tracts unsuitable for profiling
    if tract_label in ["SLF2_L", "SLF2_R"]:
        print(f"---- Skipping {tract_label} because SLF2 is unsuitable for profiling")
        continue

    # Define output directories
    outputs_profile_dir = f"{base_dir}/derivatives/pyafq/{group}/{sub}/{atlas_label}/{tract_label}/profile"
    outputs_weights_dir = f"{base_dir}/derivatives/pyafq/{group}/{sub}/{atlas_label}/{tract_label}/weights"
    outputs_segmentation_dir = f"{base_dir}/derivatives/pyafq/{group}/{sub}/{atlas_label}/{tract_label}/segmentation"
    if not os.path.exists(outputs_profile_dir):
        os.makedirs(outputs_profile_dir)
    if not os.path.exists(outputs_weights_dir):
        os.makedirs(outputs_weights_dir)
    if not os.path.exists(outputs_segmentation_dir):
        os.makedirs(outputs_segmentation_dir)

    if render:
        outputs_render = f"{base_dir}/derivatives/pyafq/{group}/{sub}/{atlas_label}/{tract_label}/render"
        if not os.path.exists(outputs_render):
            os.makedirs(outputs_render)

    # Save segmentation (by thirds) if not already saved
    end1_label = tract_metadata.loc[tract_metadata['label'] == tract_label, 'end1'].values[0]
    end2_label = tract_metadata.loc[tract_metadata['label'] == tract_label, 'end2'].values[0]

    # Define output nii and trk paths
    end1_trk_path = ospj(outputs_segmentation_dir, f"{tract_label}_end-{end1_label}.trk")
    end2_trk_path = ospj(outputs_segmentation_dir, f"{tract_label}_end-{end2_label}.trk")
    core_trk_path = ospj(outputs_segmentation_dir, f"{tract_label}_core.trk")
    end1_nii_path = ospj(outputs_segmentation_dir, f"{tract_label}_end-{end1_label}.nii.gz")
    end2_nii_path = ospj(outputs_segmentation_dir, f"{tract_label}_end-{end2_label}.nii.gz")
    core_nii_path = ospj(outputs_segmentation_dir, f"{tract_label}_core.nii.gz")
    
    # If trk paths exist, skip
    if os.path.exists(end1_trk_path) and os.path.exists(end2_trk_path) and os.path.exists(core_trk_path):
        print(f"---- Skipping {tract_label} because endpoint trk files already exist")
        continue

    # Get .trk file path
    trk_path = ospj(bundleseg_group_dir, f"{sub}/{tract_label}.trk")

    # Check that .trk file exists
    if not os.path.exists(trk_path):
        print(f"---- Skipping {tract_label} because .trk file does not exist")
        continue

    # Load .trk file
    trk = load_trk(trk_path, reference="same", bbox_valid_check=False)

    # Check that .trk file contains at least 5 streamlines
    if len(trk.streamlines) < 5:
        print(f"---- Skipping {tract_label} because .trk file contains less than 5 streamlines")
        continue

    # Make model centroids file if necessary
    if os.path.exists(ospj(centroids_dir, f"{tract_label}_model_centroids.npy")):
        centroids_model = np.load(ospj(centroids_dir, f"{tract_label}_model_centroids.npy"))

    else:

        # Create model centroids directory if it doesn't exist
        os.makedirs(centroids_dir, exist_ok=True)

        # Get model .trk file
        model_trk_path = ospj(model_dir, f"{tract_label}.trk")
        model_trk = load_trk(model_trk_path, "same", bbox_valid_check=False)
        model_trk_streamlines = model_trk.streamlines

        # Puts all streamlines into one cluster, using the centroid streamline as the standard to orient all streamlines
        feature = ResampleFeature(nb_points=100)
        metric = AveragePointwiseEuclideanMetric(feature)
        qb = QuickBundles(np.inf, metric=metric)
        clusters_model = qb.cluster(model_trk_streamlines)
        centroids_model = clusters_model.centroids[0]

        # Enforce centroids of model trk to have node order proceed comparably between left and right hemispheres
        # C_FPH_L is the only one that currently needs to be flipped after visually inspecting the profiles with the plotting code below, but should double check
        if tract_label == "C_FPH_L":
            centroids_model = centroids_model[::-1]
        
        # Save as .npy file
        np.save(ospj(centroids_dir, f"{tract_label}_model_centroids.npy"), centroids_model)

    # Reorient streamlines
    trk_streamlines_reoriented = dts.orient_by_streamline(trk.streamlines, centroids_model)
    trk_streamlines_reoriented_weights = dsa.gaussian_weights(trk_streamlines_reoriented)

    # Save Gaussian weights if not already saved
    if not os.path.exists(ospj(outputs_weights_dir, f"{tract_label}_gaussian_weights.csv")):

        # Compute mean weights across each streamline (quantifies streamline distance from centroid streamline)
        streamline_mean_weights = trk_streamlines_reoriented_weights.mean(axis=1)
        np.savetxt(ospj(outputs_weights_dir, f"{tract_label}_gaussian_weights.csv"), streamline_mean_weights, delimiter=',', fmt='%.6f')

    # Get streamlines
    end1_streamlines = [get_streamline_segment(sl, 'end1') for sl in trk_streamlines_reoriented]
    end2_streamlines = [get_streamline_segment(sl, 'end2') for sl in trk_streamlines_reoriented]
    core_streamlines = [get_streamline_segment(sl, 'core') for sl in trk_streamlines_reoriented]

    end1_tractogram = StatefulTractogram(end1_streamlines, reference=ref_t1w_nii, space=trk.space)
    end2_tractogram = StatefulTractogram(end2_streamlines, reference=ref_t1w_nii, space=trk.space)
    core_tractogram = StatefulTractogram(core_streamlines, reference=ref_t1w_nii, space=trk.space)
    
    # Save .trk files
    save_tractogram(end1_tractogram, end1_trk_path, bbox_valid_check=False)
    save_tractogram(end2_tractogram, end2_trk_path, bbox_valid_check=False)
    save_tractogram(core_tractogram, core_trk_path, bbox_valid_check=False)

    # Save .nii.gz files
    end1_tractogram.to_vox()
    end1_tractogram.to_corner()
    end1_acpc_density = density_map(end1_tractogram.streamlines, np.eye(4), acpc_dimensions)
    nib.save(nib.Nifti1Image(end1_acpc_density, acpc_affine, acpc_nifti_header), end1_nii_path)

    end2_tractogram.to_vox()
    end2_tractogram.to_corner()
    end2_acpc_density = density_map(end2_tractogram.streamlines, np.eye(4), acpc_dimensions)
    nib.save(nib.Nifti1Image(end2_acpc_density, acpc_affine, acpc_nifti_header), end2_nii_path)
    
    core_tractogram.to_vox()
    core_tractogram.to_corner()
    core_acpc_density = density_map(core_tractogram.streamlines, np.eye(4), acpc_dimensions)
    nib.save(nib.Nifti1Image(core_acpc_density, acpc_affine, acpc_nifti_header), core_nii_path)
    
    for measure in labels_to_filenames.keys():

        # Skip measures
        excluded_measures = ["map_li", "map_am", "dti_txx", "dti_txy", "dti_txz", "dti_tyy", "dti_tyz", "dti_tzz", "dti_ha", "rdi_rd1", "rdi_rd2"]
        if measure in excluded_measures:
            continue

        print(f"Running pyAFQ for {tract_label} - {measure}")

        filename = labels_to_filenames[measure]
        directory = labels_to_directories[measure]

        # Get scalar file
        if group == "penn_controls" or group == "penn_epilepsy":
            scalar_nii = ospj(qsirecon_group_dir, f"derivatives/{directory}/{sub}/{ses}/dwi/{sub}_{ses}_space-ACPC_{filename}_dwimap.nii.gz")
        elif group == "hcpaging":
            scalar_nii = ospj(qsirecon_group_dir, f"derivatives/{directory}/{sub}/dwi/{sub}_space-ACPC_{filename}_dwimap.nii.gz")
        elif group == "hcpya":
            scalar_nii = ospj(qsirecon_group_dir, f"derivatives/{directory}/{sub}/ses-01/dwi/{sub}_space-T1w_{filename}_dwimap.nii.gz")
        scalar, scalar_affine = load_nifti(scalar_nii)

        # Use the weights to calculate the tract profile for each bundle
        profile_trk_mean = dsa.afq_profile(scalar, 
                                            trk_streamlines_reoriented, 
                                            scalar_affine, 
                                            weights=trk_streamlines_reoriented_weights)
        profile_trk_sd = dsa.afq_profile(scalar, 
                                            trk_streamlines_reoriented, 
                                            scalar_affine,
                                            profile_stat=np.std)
        
        # Save numpy array profile as a .csv file
        np.savetxt(ospj(outputs_profile_dir, f"{measure}_profile-pyafq_stat-mean.csv"), profile_trk_mean, delimiter=',', fmt='%.6f')
        np.savetxt(ospj(outputs_profile_dir, f"{measure}_profile-pyafq_stat-sd.csv"), profile_trk_sd, delimiter=',', fmt='%.6f')

        ### PLOT ###

        # BELOW is the plotting code for visualizing/QC-ing orientation of along-streamline segments. Unfortunately, only able to get this working if run locally, but will test on GPU when resources are available.

        if render:
            if not os.path.exists(ospj(outputs_render, f'{sub}_{tract_label}_streamline_orientation.png')):

                # Get inverse affine matrix
                inv_affine = np.linalg.inv(ref_t1w.affine)

                # Transform streamlines to T1w space
                trk_streamlines_reoriented_in_t1w = transform_streamlines(trk_streamlines_reoriented, inv_affine)

                def lines_as_tubes(sl, line_width, **kwargs):
                    line_actor = actor.line(sl, **kwargs)
                    line_actor.GetProperty().SetRenderLinesAsTubes(1)
                    line_actor.GetProperty().SetLineWidth(line_width)
                    return line_actor

                def slice_volume(data, x=None, y=None, z=None):
                    slicer_actors = []
                    slicer_actor_z = actor.slicer(data)
                    if z is not None:
                        slicer_actor_z.display_extent(
                            0, data.shape[0] - 1,
                            0, data.shape[1] - 1,
                            z, z)
                        slicer_actors.append(slicer_actor_z)
                    if y is not None:
                        slicer_actor_y = slicer_actor_z.copy()
                        slicer_actor_y.display_extent(
                            0, data.shape[0] - 1,
                            y, y,
                            0, data.shape[2] - 1)
                        slicer_actors.append(slicer_actor_y)
                    if x is not None:
                        slicer_actor_x = slicer_actor_z.copy()
                        slicer_actor_x.display_extent(
                            x, x,
                            0, data.shape[1] - 1,
                            0, data.shape[2] - 1)
                        slicer_actors.append(slicer_actor_x)

                    return slicer_actors

                slicers = slice_volume(t1w, x=t1w.shape[0] // 2, z=t1w.shape[-1] // 4)

                # Create scene
                scene = window.Scene()
                scene.clear()

                # Make background white
                scene.background((1, 1, 1))

                # Add streamlines
                for sl in trk_streamlines_reoriented_in_t1w:
                    trk_actor = actor.line([sl], colors=create_colormap(np.linspace(0,100, len(sl)), 'jet_r'))
                    scene.add(trk_actor)

                for slicer in slicers:
                    scene.add(slicer)

                # If tract label ends with L, set camera to left
                if tract_label.endswith('L'):
                    scene.set_camera(
                        position=(25, 0, 0),         
                        focal_point=(0, 0, 0),
                        view_up=(0, 0, 1)
                    )
                elif tract_label.endswith('R'):
                    scene.set_camera(
                        position=(-25, 0, 0),         
                        focal_point=(0, 0, 0),
                        view_up=(0, 0, 1)
                    )
                else:
                    scene.set_camera(
                        position=(0, 0, 25),         
                        focal_point=(0, 0, 0),
                        view_up=(0, 0, 1)
                    )

                window.record(
                    scene,
                    out_path=ospj(outputs_render, f'{sub}_{tract_label}_streamline_orientation.png'),
                    size=(2400, 2400))

                window.show(scene)


                