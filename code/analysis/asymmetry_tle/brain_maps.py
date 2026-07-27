"""Brain map creation for GM (cortex/subcortex) and WM (association/projection). Adapted from factor_z-scores."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting


def load_atlas_metadata(atlas_tsv_path: Path) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Load atlas TSV; return label_to_index and label_to_network.
    Handles 4S atlas (network_label) and Glasser (community_yeo as network_label if present).
    """
    label_to_index: Dict[str, int] = {}
    label_to_network: Dict[str, str] = {}
    path = Path(atlas_tsv_path)
    if not path.exists():
        return label_to_index, label_to_network
    try:
        atlas_df = pd.read_csv(path, sep="\t")
        # Always get label_to_index if possible
        if "label" in atlas_df.columns and "index" in atlas_df.columns:
            label_to_index = dict(zip(atlas_df["label"], atlas_df["index"]))
        # Prefer "network_label" if present (4S), otherwise try "community_yeo" if present (Glasser)
        network_col = None
        if "network_label" in atlas_df.columns:
            network_col = "network_label"
        elif "community_yeo" in atlas_df.columns:
            network_col = "community_yeo"
        if network_col is not None:
            network_labels = atlas_df[network_col].fillna("n/a").astype(str)
            network_labels = network_labels.replace("nan", "n/a")
            label_to_network = dict(zip(atlas_df["label"], network_labels))
    except Exception:
        pass
    return label_to_index, label_to_network


def _norm_network(s) -> str:
    if s is None or (isinstance(s, str) and s.lower() in ("nan", "")):
        return "n/a"
    return str(s).strip()


def _region_label_to_atlas(region_label: str, label_to_index: Dict, label_to_network: Dict):
    """Resolve region label to (atlas_idx, network_label_str) or (None, None).
    Handles 4S: LH_/RH_, LH-/RH-. Handles Glasser: Left_/Right_.
    """
    if region_label in label_to_index:
        return label_to_index[region_label], _norm_network(label_to_network.get(region_label, "n/a"))
    if region_label.startswith("LH_") or region_label.startswith("RH_"):
        alt = region_label.replace("_", "-", 1)
        if alt in label_to_index:
            return label_to_index[alt], _norm_network(label_to_network.get(alt, "n/a"))
    if region_label.startswith("LH-") or region_label.startswith("RH-"):
        alt = region_label.replace("-", "_", 1)
        if alt in label_to_index:
            return label_to_index[alt], _norm_network(label_to_network.get(alt, "n/a"))
    if region_label.startswith("Left_") or region_label.startswith("Right_"):
        if region_label in label_to_index:
            return label_to_index[region_label], _norm_network(label_to_network.get(region_label, "n/a"))
    return None, None


def _mask_hemisphere(stat_map_data: np.ndarray, affine: np.ndarray, keep_hemisphere: str) -> None:
    """Zero out the contralateral hemisphere in-place. keep_hemisphere is 'left' or 'right'.
    RAS+ convention: negative x = left, positive x = right."""
    shape = stat_map_data.shape
    i = np.arange(shape[0])
    j = np.arange(shape[1])
    k = np.arange(shape[2])
    world_x = (
        affine[0, 0] * i[:, None, None]
        + affine[0, 1] * j[None, :, None]
        + affine[0, 2] * k[None, None, :]
        + affine[0, 3]
    )
    if keep_hemisphere == "left":
        stat_map_data[world_x > 0] = 0.0
    else:
        stat_map_data[world_x < 0] = 0.0

def save_colorbar(
    vmin: float,
    vmax: float,
    output_path: str,
    cmap: str = "jet",
    label: str = "",
) -> None:
    """Save a horizontal labeled colorbar as a small PNG for embedding in reports."""
    if vmin == vmax:
        vmax = vmin + 1.0
    fig, ax = plt.subplots(figsize=(4, 0.5))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal", pad=0.02, shrink=0.8)
    if label:
        cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_gm_brain_map(
    region_scores: Dict[str, float],
    title: str,
    output_path: str,
    atlas_nifti_path: Path,
    atlas_tsv_path: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_absolute: bool = False,
    hemisphere_only: Optional[str] = None,
    ipsilateral_hemisphere: Optional[str] = None,
    display_views: Optional[Tuple[str, str]] = None,
    cmap: Optional[Any] = None,
) -> None:
    """Create cortex and subcortex glass brain maps from region_scores.
    By default saves _ctx_y, _ctx_lr, _sctx_y, _sctx_lr. If display_views=('x',), saves only medial (sagittal at midline): _ctx_x, _sctx_x. If display_views=('l','x'), saves Left lateral + Medial: _ctx_l, _ctx_x, _sctx_l, _sctx_x.
    By default hemisphere_only is 'left' so only left-hemisphere data is plotted (medial view then shows true medial left, not the opposite hemisphere).
    If hemisphere_only is 'left' or 'right', only that hemisphere is shown (other voxels zeroed).
    If ipsilateral_hemisphere is 'left' or 'right', lateral view uses display_mode 'l' or 'r' (contralateral not shown)."""
    if hemisphere_only is None:
        hemisphere_only = "left"
    path_nifti = Path(atlas_nifti_path)
    if not path_nifti.exists():
        alt = path_nifti.parent / (path_nifti.stem + "_resliced-hcp1065" + path_nifti.suffix)
        if alt.exists():
            path_nifti = alt
        else:
            return
    label_to_index, label_to_network = load_atlas_metadata(atlas_tsv_path)
    if not label_to_index:
        return
    try:
        atlas_img = nib.load(str(path_nifti))
        atlas_data = atlas_img.get_fdata()
    except Exception:
        return

    stat_map_data_ctx = np.zeros_like(atlas_data, dtype=np.float64)
    stat_map_data_sctx = np.zeros_like(atlas_data, dtype=np.float64)
    background_mask = atlas_data == 0

    sorted_regions = sorted(
        region_scores.items(),
        key=lambda x: abs(x[1]) if x[1] is not None and not (isinstance(x[1], float) and np.isnan(x[1])) else 0,
    )
    for region_label, score in sorted_regions:
        if score is None or (isinstance(score, float) and np.isnan(score)):
            continue
        signed = score
        score = abs(signed) if use_absolute else signed
        atlas_idx, network_label_str = _region_label_to_atlas(region_label, label_to_index, label_to_network)
        if atlas_idx is None:
            continue
        if network_label_str is None or str(network_label_str).lower() in ("nan", ""):
            network_label_str = "n/a"
        mask = (atlas_data == atlas_idx) & (~background_mask)
        if network_label_str == "n/a":
            stat_map_data_sctx[mask] = score
        else:
            stat_map_data_ctx[mask] = score

    stat_map_data_ctx[background_mask] = 0.0
    stat_map_data_sctx[background_mask] = 0.0
    threshold = 1e-9
    stat_map_data_ctx[np.abs(stat_map_data_ctx) < threshold] = 0.0
    stat_map_data_sctx[np.abs(stat_map_data_sctx) < threshold] = 0.0

    if hemisphere_only in ("left", "right"):
        _mask_hemisphere(stat_map_data_ctx, atlas_img.affine, hemisphere_only)
        _mask_hemisphere(stat_map_data_sctx, atlas_img.affine, hemisphere_only)

    header_ctx = atlas_img.header.copy()
    header_sctx = atlas_img.header.copy()
    header_ctx.set_data_dtype(stat_map_data_ctx.dtype)
    header_sctx.set_data_dtype(stat_map_data_sctx.dtype)
    header_ctx["scl_slope"] = 1.0
    header_ctx["scl_inter"] = 0.0
    header_sctx["scl_slope"] = 1.0
    header_sctx["scl_inter"] = 0.0

    stat_map_img_ctx = nib.Nifti1Image(stat_map_data_ctx.copy(), atlas_img.affine, header_ctx)
    stat_map_img_sctx = nib.Nifti1Image(stat_map_data_sctx.copy(), atlas_img.affine, header_sctx)

    ctx_values = stat_map_data_ctx[stat_map_data_ctx != 0]
    sctx_values = stat_map_data_sctx[stat_map_data_sctx != 0]
    all_values = np.concatenate([ctx_values.flatten(), sctx_values.flatten()]) if (len(ctx_values) + len(sctx_values)) > 0 else np.array([])
    if len(all_values) > 0:
        if use_absolute:
            vmin_sym, vmax_sym = 0.0, float(np.max(np.abs(all_values)))
        else:
            abs_max = float(np.max(np.abs(all_values)))
            vmin_sym, vmax_sym = -abs_max, abs_max
    else:
        vmin_sym = vmin
        vmax_sym = vmax
    if vmin is not None and vmax is not None:
        if use_absolute:
            vmin_sym, vmax_sym = 0.0, abs(vmax)
        else:
            abs_max = max(abs(vmin), abs(vmax))
            vmin_sym, vmax_sym = -abs_max, abs_max

    cmap = cmap if cmap is not None else "jet"
    use_symmetric = not use_absolute and vmin_sym is not None and vmin_sym < 0

    lr_display = "lr"
    if ipsilateral_hemisphere == "left":
        lr_display = "l"
    elif ipsilateral_hemisphere == "right":
        lr_display = "r"

    is_asymmetry = hemisphere_only is not None or ipsilateral_hemisphere is not None

    def _save_map(stat_map_img, display_mode: str, out_file: str, cut_coords=None) -> None:
        figsize = (11, 5.5)
        fig = plt.figure(figsize=figsize)
        plot_kw = dict(
            colorbar=False,
            cmap=cmap,
            symmetric_cbar=use_symmetric,
            title="",
            figure=fig,
            vmin=vmin_sym,
            vmax=vmax_sym,
            plot_abs=False,
            black_bg=False,
        )
        if cut_coords is not None:
            plot_kw["cut_coords"] = cut_coords
        plotting.plot_glass_brain(stat_map_img, display_mode=display_mode, **plot_kw)
        if vmin_sym is not None and vmax_sym is not None:
            for ax in fig.axes:
                for im in ax.get_images():
                    im.set_clim(vmin_sym, vmax_sym)
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

    base = output_path.replace(".png", "")
    if display_views == ("x",):
        # Medial only: one sagittal view at midline per tissue
        _save_map(stat_map_img_ctx, "x", f"{base}_ctx_x.png", cut_coords=[0])
        _save_map(stat_map_img_sctx, "x", f"{base}_sctx_x.png", cut_coords=[0])
    elif display_views == ("l", "x"):
        # Left lateral + Medial (sagittal at midline)
        _save_map(stat_map_img_ctx, "l", f"{base}_ctx_l.png")
        _save_map(stat_map_img_ctx, "x", f"{base}_ctx_x.png", cut_coords=[0])
        _save_map(stat_map_img_sctx, "l", f"{base}_sctx_l.png")
        _save_map(stat_map_img_sctx, "x", f"{base}_sctx_x.png", cut_coords=[0])
    else:
        _save_map(stat_map_img_ctx, "y", f"{base}_ctx_y.png")
        _save_map(stat_map_img_ctx, lr_display, f"{base}_ctx_lr.png")
        _save_map(stat_map_img_sctx, "y", f"{base}_sctx_y.png")
        _save_map(stat_map_img_sctx, lr_display, f"{base}_sctx_lr.png")


def create_wm_brain_map(
    tract_segment_scores: Dict[Tuple[str, str], float],
    title: str,
    output_path_association: str,
    output_path_projection: str,
    tract_metadata_df: pd.DataFrame,
    endpoint_nii_dir: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_absolute: bool = False,
    ipsilateral_hemisphere: Optional[str] = None,
    display_views: Optional[Tuple[str, str]] = None,
    cmap: Optional[Any] = None,
) -> None:
    """Create association and projection tract glass brain maps. Default: only left-hemisphere tracts (_L) are included so medial view shows true medial left.
    If display_views=('x',), save only medial: _x.png. If display_views=('l','x'), save Left and Medial: _l.png, _x.png."""
    endpoint_nii_dir = Path(endpoint_nii_dir)
    _cmap = cmap if cmap is not None else "jet"
    # Default to left hemisphere so medial view is not the opposite hemisphere
    hem = "left" if ipsilateral_hemisphere is None else ipsilateral_hemisphere
    lr_display = "l" if hem == "left" else ("r" if hem == "right" else "lr")
    if not endpoint_nii_dir.exists():
        return
    tract_to_type = {}
    if not tract_metadata_df.empty and "label" in tract_metadata_df.columns and "type" in tract_metadata_df.columns:
        tract_to_type = dict(zip(tract_metadata_df["label"], tract_metadata_df["type"]))

    association_scores: Dict[Tuple[str, str], Tuple[float, float]] = {}
    projection_scores: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for (tract, segment), score in tract_segment_scores.items():
        if score is None or (isinstance(score, float) and np.isnan(score)):
            continue
        # By default include only left-hemisphere tracts so medial view shows true medial left
        if hem == "left" and not str(tract).endswith("_L"):
            continue
        if hem == "right" and not str(tract).endswith("_R"):
            continue
        signed = score
        abs_s = abs(signed)
        ttype = tract_to_type.get(tract, "unknown")
        if ttype == "association":
            association_scores[(tract, segment)] = (signed, abs_s)
        elif ttype == "projection":
            projection_scores[(tract, segment)] = (signed, abs_s)

    tract_to_end1 = {}
    tract_to_end2 = {}
    if "label" in tract_metadata_df.columns:
        if "end1" in tract_metadata_df.columns:
            tract_to_end1 = dict(zip(tract_metadata_df["label"], tract_metadata_df["end1"]))
        if "end2" in tract_metadata_df.columns:
            tract_to_end2 = dict(zip(tract_metadata_df["label"], tract_metadata_df["end2"]))

    def segment_to_file_label(segment: str, tract: str) -> str:
        e1 = tract_to_end1.get(tract, "end1")
        e2 = tract_to_end2.get(tract, "end2")
        if segment == "end1":
            return f"end-{e1}" if e1 not in ("NA", "") and pd.notna(e1) else "end1"
        if segment == "end2":
            return f"end-{e2}" if e2 not in ("NA", "") and pd.notna(e2) else "end2"
        return "core"

    def build_tract_map(scores_dict: Dict, output_path: str) -> None:
        if not scores_dict:
            return
        sorted_items = sorted(scores_dict.items(), key=lambda x: x[1][1])
        first_tract, first_seg = sorted_items[0][0]
        first_label = segment_to_file_label(first_seg, first_tract)
        first_mask_path = endpoint_nii_dir / f"{first_tract}_{first_label}.nii.gz"
        if not first_mask_path.exists():
            return
        try:
            ref_img = nib.load(str(first_mask_path))
            ref_data = ref_img.get_fdata()
            ref_affine = ref_img.affine
            ref_header = ref_img.header.copy()
        except Exception:
            return
        output_map = np.zeros_like(ref_data, dtype=np.float32)
        for (tract, segment), (signed_score, _) in sorted_items:
            score = abs(signed_score) if use_absolute else signed_score
            seg_label = segment_to_file_label(segment, tract)
            mask_path = endpoint_nii_dir / f"{tract}_{seg_label}.nii.gz"
            if not mask_path.exists():
                continue
            try:
                mask_img = nib.load(str(mask_path))
                mask_data = mask_img.get_fdata()
                if mask_data.shape != ref_data.shape:
                    continue
                output_map[mask_data > 0] = score
            except Exception:
                continue
        out_img = nib.Nifti1Image(output_map, ref_affine, ref_header)
        out_img.header.set_data_dtype(output_map.dtype)
        out_img.header["scl_slope"] = 1.0
        out_img.header["scl_inter"] = 0.0
        nii_path = output_path.replace(".png", ".nii.gz")
        nib.save(out_img, nii_path)

        non_zero = output_map[output_map != 0]
        if vmin is not None and vmax is not None:
            if use_absolute:
                vmin_sym, vmax_sym = 0.0, abs(vmax)
            else:
                ab = max(abs(vmin), abs(vmax))
                vmin_sym, vmax_sym = -ab, ab
        elif len(non_zero) > 0:
            if use_absolute:
                vmin_sym, vmax_sym = 0.0, float(np.max(np.abs(non_zero)))
            else:
                ab = float(np.max(np.abs(non_zero)))
                vmin_sym, vmax_sym = -ab, ab
        else:
            vmin_sym, vmax_sym = None, None

        is_asymmetry = hem in ("left", "right")

        def _save_map(display_mode: str, out_file: str, cut_coords=None) -> None:
            figsize = (11, 5.5)
            fig = plt.figure(figsize=figsize)
            plot_kw = dict(
                colorbar=False,
                cmap=_cmap,
                symmetric_cbar=(vmin_sym is None),
                title="",
                figure=fig,
                vmin=vmin_sym,
                vmax=vmax_sym,
                plot_abs=False,
                black_bg=False,
            )
            if cut_coords is not None:
                plot_kw["cut_coords"] = cut_coords
            plotting.plot_glass_brain(out_img, display_mode=display_mode, **plot_kw)
            if display_mode == "y" and is_asymmetry:
                plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
            if vmin_sym is not None and vmax_sym is not None:
                for ax in fig.axes:
                    for im in ax.get_images():
                        im.set_clim(vmin_sym, vmax_sym)
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close(fig)

        base = output_path.replace(".png", "")
        if display_views == ("x",):
            _save_map("x", f"{base}_x.png", cut_coords=[0])  # medial only
        elif display_views == ("l", "x"):
            _save_map("l", f"{base}_l.png")
            _save_map("x", f"{base}_x.png", cut_coords=[0])
        else:
            _save_map("y", f"{base}_y.png")
            _save_map(lr_display, f"{base}_lr.png")

    if association_scores:
        build_tract_map(association_scores, output_path_association)
    if projection_scores:
        build_tract_map(projection_scores, output_path_projection)
