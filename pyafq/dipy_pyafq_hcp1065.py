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

# DIPY imports
import dipy.data as dpd
import dipy.stats.analysis as dsa
import dipy.tracking.streamline as dts
from dipy.data.fetcher import get_two_hcp842_bundles
from dipy.io.image import load_nifti
from dipy.io.streamline import load_trk
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import set_number_of_points, transform_streamlines

# FURY imports
from fury import actor, window
from fury.colormap import create_colormap

# Specify atlas directory
atlas_dir = "{project_root()}/data/atlases/hcp1065/all_trk"

# Specify and create centroids directory as subdirectory of atlas_dir
centroids_dir = ospj(atlas_dir, "centroids")
os.makedirs(centroids_dir, exist_ok=True)

# Iterate over .trk files in atlas_dir
for trk_file in os.listdir(atlas_dir):
    if trk_file.endswith(".trk"):

        # Ignore hidden files
        if trk_file.startswith("."):
            continue

        # Get tract name
        tract_label = trk_file.split(".")[0]

        print(tract_label)

        # Load trk file
        trk_path = ospj(atlas_dir, trk_file)
        trk = load_trk(trk_path, "same", bbox_valid_check=False)
        trk_streamlines = trk.streamlines

        # Puts all streamlines into one cluster, using the centroid streamline as the standard to orient all streamlines
        feature = ResampleFeature(nb_points=100)
        metric = AveragePointwiseEuclideanMetric(feature)
        qb = QuickBundles(np.inf, metric=metric)
        clusters_model = qb.cluster(trk_streamlines)
        centroids_model = clusters_model.centroids[0]

        # Save as .npy file
        np.save(ospj(centroids_dir, f"{tract_label}_model_centroids.npy"), centroids_model)

        # Reorient streamlines based on centroids
        trk_streamlines_reoriented = dts.orient_by_streamline(trk.streamlines, centroids_model)
        trk_streamlines_reoriented_weights = dsa.gaussian_weights(trk_streamlines_reoriented)

        # Visualize the orientation of along-streamline segments for the tract

        # Get inverse affine matrix (identity, since atlas is in its own space)
        inv_affine = np.eye(4)

        # Transform streamlines to "T1w" space (no-op here, but for consistency)
        trk_streamlines_reoriented_in_t1w = transform_streamlines(trk_streamlines_reoriented, inv_affine)

        # Helper to make lines as tubes
        def lines_as_tubes(sl, line_width, **kwargs):
            line_actor = actor.line(sl, **kwargs)
            line_actor.GetProperty().SetRenderLinesAsTubes(1)
            line_actor.GetProperty().SetLineWidth(line_width)
            return line_actor

        # Create scene
        scene = window.Scene()
        scene.clear()

        # Add streamlines, colored by node index along the streamline
        for sl in trk_streamlines_reoriented_in_t1w:
            n_points = len(sl)
            colors = create_colormap(np.linspace(0, 100, n_points), 'jet')
            trk_actor = actor.line([sl], colors=colors)
            scene.add(trk_actor)

        # Set camera based on tract label
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

        # Save the image
        out_img = ospj(centroids_dir, f"{tract_label}_streamline_orientation.png")
        window.record(
            scene,
            out_path=out_img,
            size=(1200, 1200)
        )



