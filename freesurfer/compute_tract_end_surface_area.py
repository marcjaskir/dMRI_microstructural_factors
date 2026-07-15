import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
import os
import numpy as np
import nibabel as nib
import csv

# Directory containing the .gii mask files
gii_dir = "{project_root()}/data/atlases/HCP1065/endpoint_gii"
output_csv = "{project_root()}/derivatives/freesurfer_post/HCP1065_tract_end_surface_area.csv"
output_csv_ipsi = "{project_root()}/derivatives/freesurfer_post/HCP1065_tract_end_surface_area_ipsi.csv"

# Reference surface files (fsaverage, 164k)
lh_surf = "{project_root()}/code/tract_to_region/tpl-fsaverage_den-164k_hemi-L_white.surf.gii"
rh_surf = "{project_root()}/code/tract_to_region/tpl-fsaverage_den-164k_hemi-R_white.surf.gii"

# Load reference surfaces
def load_surface(surf_path):
    surf_gii = nib.load(surf_path)
    vertices = surf_gii.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
    faces = surf_gii.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
    return np.array(vertices), np.array(faces, dtype=int)

lh_vertices, lh_faces = load_surface(lh_surf)
rh_vertices, rh_faces = load_surface(rh_surf)

def compute_triangle_area(v0, v1, v2):
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

results = []

for fname in sorted(os.listdir(gii_dir)):

    if not fname.endswith(".gii"):
        continue

    # Skip hidden files
    if fname.startswith("."):
        continue

    fpath = os.path.join(gii_dir, fname)
    if not os.path.exists(fpath):
        continue

    # Parse tract and end from filename
    # Example: C_FP_R_end-P.rh.shape.gii
    base = os.path.basename(fname)
    if "_end" not in base:
        continue
    tract = base.split("_end")[0]
    after_end = base.split("_end", 1)[1]
    if after_end.startswith("-"):
        after_end = after_end[1:]
    end = after_end.split(".", 1)[0]

    print(tract, end)

    # Determine hemisphere from filename (".lh." or ".rh.")
    if ".lh." in fname:
        hemi = "lh"
        vertices = lh_vertices
        faces = lh_faces
    elif ".rh." in fname:
        hemi = "rh"
        vertices = rh_vertices
        faces = rh_faces
    else:
        # If hemisphere cannot be determined, skip
        continue

    # Load mask data from gifti file (assume first darray is the mask)
    gii = nib.load(fpath)
    mask_data = gii.darrays[0].data
    mask_data = np.array(mask_data)

    # Find which vertices are in the mask (nonzero)
    mask_indices = np.where(mask_data != 0)[0]
    mask_set = set(mask_indices)

    # Find faces where all three vertices are in the mask
    mask_faces = []
    for face in faces:
        if all(idx in mask_set for idx in face):
            mask_faces.append(face)
    mask_faces = np.array(mask_faces, dtype=int)

    # Compute surface area for the masked region
    area = 0.0
    for face in mask_faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        area += compute_triangle_area(v0, v1, v2)

    results.append({
        "tract": tract,
        "end": end,
        "hemi": hemi,
        "surface_area": area
    })

# Write results to CSV (all hemispheres)
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["tract", "end", "hemi", "surface_area"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# Write ipsilateral-only CSV (omit hemi column)
# Determine ipsilateral hemisphere from tract name: tracts ending with _L are lh, _R are rh
def get_ipsi_hemi(tract):
    if tract.endswith("_L"):
        return "lh"
    elif tract.endswith("_R"):
        return "rh"
    else:
        return None

ipsi_rows = []
for row in results:
    ipsi_hemi = get_ipsi_hemi(row["tract"])
    if ipsi_hemi is not None and row["hemi"] == ipsi_hemi:
        ipsi_rows.append({
            "tract": row["tract"],
            "end": row["end"],
            "surface_area": row["surface_area"]
        })

with open(output_csv_ipsi, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["tract", "end", "surface_area"])
    writer.writeheader()
    for row in ipsi_rows:
        writer.writerow(row)
