"""
Glasser atlas visualization with Yeo-network filtering (ipyniivue).

This creates binary mask volumes for each `community_yeo` value in:
  - `data/atlases/Glasser/atlas-Glasser_dseg.tsv`

and lets you select which Yeo network to display via a dropdown.

Additionally, when a limbic network is selected (Yeo name contains "Limbic"),
the Glasser atlas parcels are colored randomly (background).

Intended usage (in a Jupyter notebook):
    from visualize_glasser_yeo_networks_ipyniivue import display_glasser_yeo_networks
    display_glasser_yeo_networks()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import ipywidgets as widgets
from IPython.display import display
from ipyniivue import NiiVue, ShowRender, SliceType
import random


def _safe_name(s: str) -> str:
    """Filesystem-safe name for cached mask files."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


def _build_yeo_masks(
    atlas_nifti_path: Path,
    atlas_tsv_path: Path,
    cache_dir: Path,
) -> Tuple[List[str], Dict[str, Path], Dict[str, List[int]], pd.DataFrame, np.ndarray]:
    """
    Create cached binary mask volumes per Yeo network.

    Returns:
      - yeo_options: ["All", ...unique community_yeo values...]
      - yeo_to_mask_path
      - yeo_to_indices (atlas label indices)
      - atlas dataframe (for labels/colors)
      - atlas data (for label lookup)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(atlas_tsv_path, sep="\t")
    if "index" not in df.columns or "community_yeo" not in df.columns:
        raise ValueError("Expected columns `index` and `community_yeo` in atlas TSV.")

    df["community_yeo"] = df["community_yeo"].astype(str).str.strip()
    df = df.dropna(subset=["community_yeo"])

    # Some atlases store indices as ints; still coerce defensively.
    idx_series = pd.to_numeric(df["index"], errors="coerce").dropna().astype(int)
    if idx_series.empty:
        raise ValueError("No valid `index` values found in atlas TSV.")

    all_indices = sorted(idx_series.tolist())
    unique_yeo = sorted(df["community_yeo"].unique().tolist())
    yeo_options = ["All"] + unique_yeo

    yeo_to_indices: Dict[str, List[int]] = {"All": all_indices}
    for yeo in unique_yeo:
        idxs = (
            df.loc[df["community_yeo"] == yeo, "index"]
            .pipe(lambda s: pd.to_numeric(s, errors="coerce"))
            .dropna()
            .astype(int)
            .tolist()
        )
        yeo_to_indices[yeo] = sorted(idxs)

    # Load atlas once and round labels to integer ids.
    img = nib.load(str(atlas_nifti_path))
    atlas_data = np.asarray(img.get_fdata())
    atlas_int = np.rint(atlas_data).astype(np.int32)

    yeo_to_mask_path: Dict[str, Path] = {}
    for yeo in yeo_options:
        out_path = cache_dir / f"glasser_yeo_{_safe_name(yeo)}.nii.gz"
        yeo_to_mask_path[yeo] = out_path

        if out_path.exists():
            continue

        idxs = yeo_to_indices.get(yeo, [])
        if not idxs:
            mask = np.zeros_like(atlas_int, dtype=np.uint8)
        else:
            mask = np.isin(atlas_int, idxs).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(out_path))

    # Return also the atlas dataframe and atlas_int for label processing
    return yeo_options, yeo_to_mask_path, yeo_to_indices, df, atlas_int


def _create_random_colormap_for_labels(df: pd.DataFrame) -> dict:
    """
    Generate a JSON-style colormap with random RGB colors for each label in the Glasser atlas.
    """
    # We expect "index" and "name" columns
    indices = df["index"].astype(int).tolist()
    labels = df["name"].fillna(df["index"].astype(str)).tolist()
    n_labels = len(indices)
    random.seed(0)  # Deterministic colors within session for testing
    R = [0]  # First index is background, always black
    G = [0]
    B = [0]
    A = [0]
    label_names = ["bg"]

    for i in range(n_labels):
        color = [random.randint(60,255), random.randint(60,255), random.randint(60,255)]
        R.append(color[0])
        G.append(color[1])
        B.append(color[2])
        A.append(255)
        label_names.append(labels[i])

    return {
        "R": R,
        "G": G,
        "B": B,
        "A": A,
        "labels": label_names
    }


def display_glasser_yeo_networks(
    atlas_nifti_path: Path | None = None,
    atlas_tsv_path: Path | None = None,
    cache_dir: Path | None = None,
    initial_yeo: str = "All",
) -> None:
    """
    Render an ipyniivue viewer with a Yeo-network dropdown for the Glasser atlas.
    When a limbic Yeo network is selected (case-insensitive substring "limbic"),
    the Glasser parcels background will be colored randomly.
    """
    # Resolve default paths relative to this file.
    this_file = Path(__file__).resolve()
    project_dir = this_file.parents[2]  # .../structural_tractometry
    if atlas_nifti_path is None:
        atlas_nifti_path = (
            project_dir
            / "data"
            / "atlases"
            / "Glasser"
            / "atlas-Glasser_space-MNI152NLin2009cAsym_res-1_dseg.nii.gz"
        )
    if atlas_tsv_path is None:
        atlas_tsv_path = project_dir / "data" / "atlases" / "Glasser" / "atlas-Glasser_dseg.tsv"
    if cache_dir is None:
        cache_dir = project_dir / "derivatives" / "atlas_viz_cache" / "glasser_yeo_masks"

    if not atlas_nifti_path.exists():
        raise FileNotFoundError(f"Missing atlas NIfTI: {atlas_nifti_path}")
    if not atlas_tsv_path.exists():
        raise FileNotFoundError(f"Missing atlas TSV: {atlas_tsv_path}")

    # GET dataframe and atlas_int for colormap construction
    yeo_options, yeo_to_mask_path, _, atlas_df, _ = _build_yeo_masks(
        atlas_nifti_path=atlas_nifti_path,
        atlas_tsv_path=atlas_tsv_path,
        cache_dir=cache_dir,
    )
    if initial_yeo not in yeo_options:
        initial_yeo = "All"

    # ----------------------------
    # Niivue viewer
    # ----------------------------
    nv = NiiVue(
        back_color=(0.3, 0.3, 0.3, 1),
        show_3d_crosshair=True,
    )
    nv.set_radiological_convention(False)

    # Volume 0: atlas labels (background option)
    # Volume 1..N: per-network binary masks (dropdown selection)
    volumes = [
        {"path": str(atlas_nifti_path), "opacity": 1.0},  # base atlas volume (background)
    ]
    base_vol_idx = 0
    mask_start_idx = 1
    for yeo in yeo_options:
        # Use a built-in colormap so the viewer doesn't depend on set_colormap_label() succeeding.
        # (Mask volumes are binary {0,1}, so any label colormap is mainly aesthetic.)
        volumes.append({"path": str(yeo_to_mask_path[yeo]), "opacity": 0.0, "colormap": "jet"})

    nv.load_volumes(volumes)
    nv.opts.multiplanar_show_render = ShowRender.ALWAYS
    nv.set_slice_type(SliceType.MULTIPLANAR)
    nv.graph.auto_size_multiplanar = True

    # ----------------------------
    # Widgets
    # ----------------------------
    background_checkbox = widgets.Checkbox(value=True, description="Background (atlas)")
    mask_checkbox = widgets.Checkbox(value=True, description="Show selected network")
    smooth_checkbox = widgets.Checkbox(value=True, description="Smooth")

    opacity_slider = widgets.IntSlider(
        value=170,
        min=1,
        max=255,
        description="Mask opacity",
        continuous_update=True,
        style={"description_width": "140px"},
        layout=widgets.Layout(width="420px"),
    )

    yeo_dropdown = widgets.Dropdown(
        options=yeo_options,
        value=initial_yeo,
        description="Yeo network",
    )

    # Map yeo -> volume index in nv.volumes
    yeo_to_vol_idx = {yeo: mask_start_idx + i for i, yeo in enumerate(yeo_options)}

    def _current_mask_opacity() -> float:
        return opacity_slider.value / 255.0 if mask_checkbox.value else 0.0

    # --------- Create a random colormap for atlas "labels" volume (background) ----------
    random_colormap_for_labels = _create_random_colormap_for_labels(atlas_df)  # preserve for event reuse

    # --------- Default (gray) Glasser colormap (ipyniivue expects per-label-class JSON colormap) ----
    # We'll use a simple gray map as the default fallback
    # (N.B.: assumes label 0 is background, rest gray)
    N_LABELS = 1 + atlas_df.shape[0]  # 0=bg, rest per tsv row
    gray_colormap_for_labels = {
        "R": [0] + [192]*(N_LABELS-1),
        "G": [0] + [192]*(N_LABELS-1),
        "B": [0] + [192]*(N_LABELS-1),
        "A": [0] + [255]*(N_LABELS-1),
        "labels": ["bg"] + atlas_df["name"].fillna(atlas_df["index"].astype(str)).tolist()
    }

    # State for currently applied colormap to background
    colormap_used_for_atlas = "default"

    def _apply_atlas_colormap_json(colormap_dict: dict):
        try:
            nv.volumes[base_vol_idx].set_colormap_label(colormap_dict)
        except Exception as e:
            print(f"Could not set background colormap: {e}")

    # Initially set to gray
    _apply_atlas_colormap_json(gray_colormap_for_labels)

    def _apply_mask_visibility(selected_yeo: str) -> None:
        sel_idx = yeo_to_vol_idx[selected_yeo]
        op = _current_mask_opacity()
        for yeo, vix in yeo_to_vol_idx.items():
            nv.set_opacity(vix, op if vix == sel_idx else 0.0)

        # NEW: If selected Yeo network is limbic, set random coloring for parcels, else reset to gray
        nonlocal colormap_used_for_atlas
        if selected_yeo.lower().find("limbic") >= 0:
            if colormap_used_for_atlas != "limbic":
                _apply_atlas_colormap_json(random_colormap_for_labels)
                colormap_used_for_atlas = "limbic"
        else:
            if colormap_used_for_atlas != "default":
                _apply_atlas_colormap_json(gray_colormap_for_labels)
                colormap_used_for_atlas = "default"

    def on_background_checkbox_change(change) -> None:
        nv.set_opacity(base_vol_idx, 1.0 if change["new"] else 0.0)

    def on_mask_checkbox_change(change) -> None:
        # Matches the template's intent: also toggle alpha clip behavior.
        nv.opts.is_alpha_clip_dark = change["new"]
        _apply_mask_visibility(yeo_dropdown.value)

    def on_smooth_checkbox_change(change) -> None:
        nv.set_interpolation(not change["new"])

    def on_opacity_change(change) -> None:
        _apply_mask_visibility(yeo_dropdown.value)

    def on_yeo_change(change) -> None:
        _apply_mask_visibility(change["new"])

    background_checkbox.observe(on_background_checkbox_change, names="value")
    mask_checkbox.observe(on_mask_checkbox_change, names="value")
    smooth_checkbox.observe(on_smooth_checkbox_change, names="value")
    opacity_slider.observe(on_opacity_change, names="value")
    yeo_dropdown.observe(on_yeo_change, names="value")

    # Apply initial visibility states
    on_background_checkbox_change({"new": background_checkbox.value})
    on_mask_checkbox_change({"new": mask_checkbox.value})
    on_smooth_checkbox_change({"new": smooth_checkbox.value})

    # ----------------------------
    # Custom colormap (JSON) for masks
    # ----------------------------
    colormap_textarea = widgets.Textarea(
        value="""{
  "R": [0,  120],
  "G": [0,   90],
  "B": [0,  175],
  "A": [0,  255],
  "labels": ["bg", "mask"]
}""",
        description="Mask colormap (JSON)",
        style={"description_width": "200px"},
        layout=widgets.Layout(width="100%", height="150px"),
    )

    apply_button = widgets.Button(description="Apply Colormap", button_style="primary")

    def on_apply_button_click(_b) -> None:
        try:
            cmap = json.loads(colormap_textarea.value)
        except json.JSONDecodeError as e:
            print("Invalid JSON format:", e)
            return
        for yeo in yeo_options:
            vix = yeo_to_vol_idx[yeo]
            nv.volumes[vix].set_colormap_label(cmap)

    apply_button.on_click(on_apply_button_click)
    # Apply once, but don't hard-fail the notebook if ipyniivue's colormap validator rejects the JSON.
    try:
        on_apply_button_click(None)
    except Exception as e:
        print("Warning: could not apply custom mask colormap label:", e)

    # ----------------------------
    # Hover info
    # ----------------------------
    location_label = widgets.HTML("&nbsp;")

    def handle_location_change(data) -> None:
        # `data['string']` matches ipyniivue template usage.
        location_label.value = "&nbsp;&nbsp;" + data["string"]

    nv.on_location_change(handle_location_change)

    # ----------------------------
    # Display
    # ----------------------------
    controls = widgets.HBox(
        [background_checkbox, mask_checkbox, smooth_checkbox, opacity_slider, yeo_dropdown]
    )
    display(widgets.VBox([controls, nv, location_label, widgets.VBox([colormap_textarea, apply_button])]))

