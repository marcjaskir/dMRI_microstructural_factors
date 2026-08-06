#!/usr/bin/env python3
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
Voxelwise microstructural factor analysis from raw MNI-space qsirecon maps.

This keeps the factor-analysis/PCA output style of the region-wise pipeline, but
builds the scalar correlation matrix by streaming masked voxelwise data from raw
qsirecon scalar images. The full subject-by-voxel-by-scalar matrix is never kept
in memory.

Per subject, voxel values outside [Q1 - k*IQR, Q3 + k*IQR] (computed across
in-mask voxels for each scalar separately) are omitted before correlations are
accumulated. Use --outlier-iqr-multiplier 0 to disable this filter.

FA can be run on two voxel sets via --voxel-mask: brain (with CSF voxels) and
brain_no_csf (brain mask minus CSF probseg voxels with prob > 0.5), saved under nested
with_csf/ and no_csf/ subdirectories per analysis group.

Factor analysis is always fit with fixed 3- and 4-factor solutions, written to
factors-3/ and factors-4/ under each mask-specific output directory.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

FA_DIR = Path(__file__).resolve().parents[1] / "factor_analysis"

PROJECT_ROOT = project_root()
QSIRECON_DIR = PROJECT_ROOT / "derivatives" / "qsirecon"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
SCALAR_FILES_JSON = METADATA_DIR / "scalar_labels_to_filenames.json"
SCALAR_DIRS_JSON = METADATA_DIR / "scalar_labels_to_directories.json"
MASK_PATH = PROJECT_ROOT / "data/atlases/MNI/tpl-MNI152NLin2009cAsym_res-1mm_desc-brain_mask.nii.gz"
CSF_PROBSEG_PATH = PROJECT_ROOT / "data/atlases/MNI/tpl-MNI152NLin2009cAsym_res-1mm_label-csf_probseg.nii.gz"
CSF_PROBSEG_THRESHOLD = 0.5
T1W_PATH = PROJECT_ROOT / "data/atlases/MNI/tpl-MNI152NLin2009cAsym_res-1mm_T1w.nii.gz"
OUTPUT_PROJECT_ROOT = PROJECT_ROOT / "derivatives/analysis/factor_analysis_voxelwise"
REDUCED_SUBJECTS_PATH = PROJECT_ROOT / "derivatives/analysis/factor_z-scores/factor_scores/controls_F1_scores.csv"

VOXEL_MASK_CHOICES = ("brain", "brain_no_csf", "both")
VOXEL_MASK_SUBDIRS = {"brain": "with_csf", "brain_no_csf": "no_csf"}
VOXEL_MASK_DESCRIPTIONS = {
    "with_csf": "with_csf (MNI brain mask only)",
    "no_csf": f"no_csf (brain mask minus CSF probseg voxels with prob > {CSF_PROBSEG_THRESHOLD})",
}

GROUPS = ["penn_controls", "hcpya", "hcpaging"]
ANALYSIS_GROUP_CHOICES = ("all", *GROUPS)
GROUP_LABEL = "controls"
GROUP_MODE = "controls"
RUN_NAMES = {
    "all": "Voxelwise_AllControls",
    "reduced": "Voxelwise_ReducedControls",
}
MNI_SUFFIX = "space-MNI152NLin2009cAsym"
NO_SESSION_GROUP = "hcpaging"
N_PCA_COMPONENTS_FULL = 26

DEFAULT_OUTLIER_IQR_MULTIPLIER = 1.5
MIN_FINITE_VOXELS_FOR_IQR = 4
FIXED_FACTOR_COUNTS = (3, 4)


def import_factor_analysis_helpers() -> None:
    """Import heavy FA/plotting dependencies only when modeling is requested."""
    sys.path.insert(0, str(FA_DIR))
    from factor_analysis import (  # noqa: PLC0415
        COMBINED_HEATMAP_DTI_DKI_GQI_PREFIXES as _COMBINED_PREFIXES,
        CORR_FACTOR_PCA_FACTOR_ORDERED_DPI as _ORDERED_DPI,
        create_html_factor_report as _create_html_factor_report,
        order_scalars_by_max_abs_factor_loading as _order_scalars_by_max_abs_factor_loading,
        plot_corr_and_ica_combined as _plot_corr_and_ica_combined,
        plot_corr_and_loadings_combined as _plot_corr_and_loadings_combined,
        plot_corr_and_loadings_combined_bottom as _plot_corr_and_loadings_combined_bottom,
        plot_corr_factor_loadings_and_pca_components_combined as _plot_corr_factor_loadings_and_pca_components_combined,
        plot_corr_matrix_minimal as _plot_corr_matrix_minimal,
        plot_factor_loadings_standalone as _plot_factor_loadings_standalone,
        plot_factor_pca_combined_summary as _plot_factor_pca_combined_summary,
        plot_pca_factor_correlation as _plot_pca_factor_correlation,
    )

    globals().update(
        {
            "COMBINED_HEATMAP_DTI_DKI_GQI_PREFIXES": _COMBINED_PREFIXES,
            "CORR_FACTOR_PCA_FACTOR_ORDERED_DPI": _ORDERED_DPI,
            "create_html_factor_report": _create_html_factor_report,
            "order_scalars_by_max_abs_factor_loading": _order_scalars_by_max_abs_factor_loading,
            "plot_corr_and_ica_combined": _plot_corr_and_ica_combined,
            "plot_corr_and_loadings_combined": _plot_corr_and_loadings_combined,
            "plot_corr_and_loadings_combined_bottom": _plot_corr_and_loadings_combined_bottom,
            "plot_corr_factor_loadings_and_pca_components_combined": _plot_corr_factor_loadings_and_pca_components_combined,
            "plot_corr_matrix_minimal": _plot_corr_matrix_minimal,
            "plot_factor_loadings_standalone": _plot_factor_loadings_standalone,
            "plot_factor_pca_combined_summary": _plot_factor_pca_combined_summary,
            "plot_pca_factor_correlation": _plot_pca_factor_correlation,
        }
    )


@dataclass(frozen=True)
class SubjectSession:
    group: str
    sub: str
    ses: str | None

    @property
    def subject_id(self) -> str:
        return f"{self.sub}_{self.ses}" if self.ses else self.sub


@dataclass
class StreamingStats:
    scalar_counts: np.ndarray
    scalar_sums: np.ndarray
    scalar_sum_squares: np.ndarray
    voxel_sums: np.ndarray
    voxel_counts: np.ndarray
    pair_counts: np.ndarray
    pair_sum_x: np.ndarray
    pair_sum_x2: np.ndarray
    pair_cross: np.ndarray
    n_subjects_accumulated: int = 0
    n_rows_accumulated: int = 0


def load_scalar_metadata() -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    with open(SCALAR_FILES_JSON) as f:
        scalar_to_file = json.load(f)
    with open(SCALAR_DIRS_JSON) as f:
        scalar_to_dir = json.load(f)
    scalar_labels = list(scalar_to_file.keys())
    missing = [s for s in scalar_labels if s not in scalar_to_dir]
    if missing:
        raise ValueError(f"Scalars missing qsirecon directory metadata: {missing}")
    return scalar_labels, scalar_to_file, scalar_to_dir


def discover_subject_sessions(groups: Sequence[str], scalar_to_dir: Dict[str, str]) -> List[SubjectSession]:
    subjects: List[SubjectSession] = []
    scalar_dirs = list(dict.fromkeys(scalar_to_dir.values()))
    for group in tqdm(groups, desc="Discovering control groups"):
        group_dir = QSIRECON_DIR / group
        if not group_dir.exists():
            continue
        base = None
        for scalar_dir in scalar_dirs:
            candidate = group_dir / "derivatives" / scalar_dir
            if candidate.exists():
                base = candidate
                break
        if base is None:
            base = group_dir
        for sub_dir in sorted(base.iterdir()):
            if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
                continue
            if group == NO_SESSION_GROUP:
                if (sub_dir / "dwi").is_dir():
                    subjects.append(SubjectSession(group, sub_dir.name, None))
            else:
                for ses_dir in sorted(sub_dir.iterdir()):
                    if ses_dir.is_dir() and ses_dir.name.startswith("ses-") and (ses_dir / "dwi").is_dir():
                        subjects.append(SubjectSession(group, sub_dir.name, ses_dir.name))
    return subjects


def load_reduced_subject_ids(path: Path) -> set[str]:
    """Load the reduced control subject set from existing factor-score outputs."""
    df = pd.read_csv(path, usecols=["subject"])
    subjects = set(df["subject"].astype(str))
    if not subjects:
        raise ValueError(f"No subjects found in reduced subject file: {path}")
    return subjects


def scalar_path(subject: SubjectSession, scalar_directory: str, scalar_filename: str) -> str | None:
    group_dir = QSIRECON_DIR / subject.group
    if subject.group == NO_SESSION_GROUP:
        bases = [
            group_dir / "derivatives" / scalar_directory / subject.sub / "dwi",
            group_dir / subject.sub / "dwi",
        ]
    else:
        if subject.ses is None:
            return None
        bases = [
            group_dir / "derivatives" / scalar_directory / subject.sub / subject.ses / "dwi",
            group_dir / subject.sub / subject.ses / "dwi",
        ]

    candidates: List[str] = []
    for base in bases:
        if not base.exists():
            continue
        for pattern in (
            f"*{MNI_SUFFIX}_{scalar_filename}.nii.gz",
            f"*{MNI_SUFFIX}_{scalar_filename}_dwimap.nii.gz",
            f"*{MNI_SUFFIX}*{scalar_filename}*.nii.gz",
        ):
            candidates.extend(glob.glob(str(base / pattern)))
    return str(Path(sorted(candidates)[0]).resolve()) if candidates else None


def build_manifest(
    subjects: Sequence[SubjectSession],
    scalar_labels: Sequence[str],
    scalar_to_file: Dict[str, str],
    scalar_to_dir: Dict[str, str],
    max_subjects: int | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    manifest_columns = ["subject", "sub", "ses", "group", "scalar", "path"]
    missing_columns = ["subject", "sub", "ses", "group", "missing_scalars", "n_missing"]
    rows: List[Dict[str, str]] = []
    missing_rows: List[Dict[str, str]] = []
    n_complete = 0
    for subject in tqdm(subjects, desc="Validating scalar image manifest"):
        paths = {
            scalar: scalar_path(subject, scalar_to_dir[scalar], scalar_to_file[scalar])
            for scalar in scalar_labels
        }
        missing = [scalar for scalar, path in paths.items() if path is None]
        if missing:
            missing_rows.append(
                {
                    "subject": subject.subject_id,
                    "sub": subject.sub,
                    "ses": subject.ses or "",
                    "group": subject.group,
                    "missing_scalars": ",".join(missing),
                    "n_missing": str(len(missing)),
                }
            )
            continue
        for scalar, path in paths.items():
            rows.append(
                {
                    "subject": subject.subject_id,
                    "sub": subject.sub,
                    "ses": subject.ses or "",
                    "group": subject.group,
                    "scalar": scalar,
                    "path": str(path),
                }
            )
        n_complete += 1
        if max_subjects is not None and n_complete >= max_subjects:
            break
    return pd.DataFrame(rows, columns=manifest_columns), pd.DataFrame(missing_rows, columns=missing_columns)


def load_reference_image(manifest_df: pd.DataFrame, scalar_labels: Sequence[str]) -> nib.Nifti1Image:
    """Use the first included subject/scalar image as the analysis grid reference."""
    first_subject = manifest_df["subject"].iloc[0]
    first_scalar = scalar_labels[0]
    row = manifest_df[(manifest_df["subject"] == first_subject) & (manifest_df["scalar"] == first_scalar)]
    if row.empty:
        row = manifest_df.iloc[[0]]
    ref_path = str(row["path"].iloc[0])
    ref_img = nib.load(ref_path)
    print(f"Using analysis grid reference: {ref_path}")
    print(f"Reference image shape: {ref_img.shape[:3]}")
    return ref_img


def resample_atlas_to_reference(
    atlas_path: Path,
    reference_img: nib.Nifti1Image,
    *,
    order: int,
) -> nib.Nifti1Image:
    """Resample an atlas image once onto the scalar analysis grid."""
    from nibabel.processing import resample_from_to  # noqa: PLC0415

    source_img = nib.load(str(atlas_path))
    target = (reference_img.shape[:3], reference_img.affine)
    return resample_from_to(source_img, target, order=order)


def apply_max_voxels_limit(mask: np.ndarray, max_voxels: int | None) -> np.ndarray:
    if max_voxels is None:
        return mask
    keep = np.flatnonzero(mask.ravel())[:max_voxels]
    limited = np.zeros(mask.size, dtype=bool)
    limited[keep] = True
    return limited.reshape(mask.shape)


def load_mask(mask_path: Path, reference_img: nib.Nifti1Image, max_voxels: int | None) -> Tuple[nib.Nifti1Image, np.ndarray]:
    """Resample the T1w brain mask once onto the scalar image grid."""
    mask_img = resample_atlas_to_reference(mask_path, reference_img, order=0)
    mask = apply_max_voxels_limit(np.asarray(mask_img.get_fdata() > 0), max_voxels)
    return mask_img, mask


def build_voxel_masks(
    reference_img: nib.Nifti1Image,
    max_voxels: int | None,
) -> Dict[str, Tuple[nib.Nifti1Image, np.ndarray]]:
    """Build with_csf and no_csf boolean masks on the scalar grid."""
    mask_img, brain_mask = load_mask(MASK_PATH, reference_img, max_voxels)
    csf_img = resample_atlas_to_reference(CSF_PROBSEG_PATH, reference_img, order=0)
    csf_mask = np.asarray(csf_img.get_fdata() > CSF_PROBSEG_THRESHOLD)
    no_csf_mask = brain_mask & ~csf_mask
    n_brain = int(brain_mask.sum())
    n_no_csf = int(no_csf_mask.sum())
    n_csf_excluded = n_brain - n_no_csf
    print(
        f"Voxel masks on analysis grid: with_csf={n_brain:,}; "
        f"no_csf={n_no_csf:,} (excluded {n_csf_excluded:,} CSF voxels)"
    )
    return {
        "with_csf": (mask_img, brain_mask),
        "no_csf": (mask_img, no_csf_mask),
    }


def resolve_voxel_mask_modes(voxel_mask: str) -> List[str]:
    if voxel_mask == "both":
        return ["brain", "brain_no_csf"]
    return [voxel_mask]


def save_masked_t1w_preview(mask_img: nib.Nifti1Image, mask: np.ndarray, output_path: Path) -> None:
    """Save T1w resampled to the analysis grid, zeroed outside the no-CSF mask."""
    t1w_img = resample_atlas_to_reference(T1W_PATH, mask_img, order=1)
    preview = np.where(mask, np.asarray(t1w_img.get_fdata(dtype=np.float32), dtype=np.float32), 0.0)
    header = t1w_img.header.copy()
    header.set_data_dtype(np.float32)
    out_img = nib.Nifti1Image(preview.astype(np.float32), mask_img.affine, header)
    nib.save(out_img, str(output_path))
    print(f"Saved masked T1w preview to: {output_path}")


def save_analysis_mask(mask_img: nib.Nifti1Image, mask: np.ndarray, output_path: Path) -> None:
    """Save the exact boolean mask used to select voxel rows."""
    header = mask_img.header.copy()
    header.set_data_dtype(np.uint8)
    out_img = nib.Nifti1Image(mask.astype(np.uint8), mask_img.affine, header)
    nib.save(out_img, str(output_path))
    print(f"Saved analysis mask to: {output_path}")


def load_masked_scalar(path: str, mask_img: nib.Nifti1Image, mask: np.ndarray) -> np.ndarray:
    img = nib.load(path)
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim > 3:
        data = np.squeeze(data)
    if data.shape != mask.shape:
        raise ValueError(f"Image shape {data.shape} does not match mask shape {mask.shape}: {path}")
    if not np.allclose(img.affine, mask_img.affine, atol=1e-3):
        raise ValueError(f"Image affine does not match analysis mask affine: {path}")
    return data[mask].astype(np.float32, copy=False)


def initialize_stats(n_scalars: int, n_voxels: int) -> StreamingStats:
    return StreamingStats(
        scalar_counts=np.zeros(n_scalars, dtype=np.int64),
        scalar_sums=np.zeros(n_scalars, dtype=np.float64),
        scalar_sum_squares=np.zeros(n_scalars, dtype=np.float64),
        voxel_sums=np.zeros((n_scalars, n_voxels), dtype=np.float64),
        voxel_counts=np.zeros((n_scalars, n_voxels), dtype=np.uint16),
        pair_counts=np.zeros((n_scalars, n_scalars), dtype=np.int64),
        pair_sum_x=np.zeros((n_scalars, n_scalars), dtype=np.float64),
        pair_sum_x2=np.zeros((n_scalars, n_scalars), dtype=np.float64),
        pair_cross=np.zeros((n_scalars, n_scalars), dtype=np.float64),
    )


def mask_subject_scalar_iqr_outliers(
    subject_matrix: np.ndarray,
    multiplier: float,
    min_finite_voxels: int = MIN_FINITE_VOXELS_FOR_IQR,
) -> np.ndarray:
    """Set per-subject, per-scalar IQR outliers to NaN (fences across in-mask voxels)."""
    out = np.array(subject_matrix, dtype=np.float32, copy=True)
    finite = np.isfinite(out)
    finite_counts = finite.sum(axis=0)
    if int(finite_counts.max()) < min_finite_voxels:
        return out

    with np.errstate(invalid="ignore"):
        q1, q3 = np.nanpercentile(out, [25, 75], axis=0)
    iqr = q3 - q1
    active = (iqr > 0) & (finite_counts >= min_finite_voxels)
    low = q1 - multiplier * iqr
    high = q3 + multiplier * iqr
    outlier = finite & active & ((out < low) | (out > high))
    out[outlier] = np.nan
    return out


def update_stats(stats: StreamingStats, subject_matrix: np.ndarray) -> None:
    finite = np.isfinite(subject_matrix)
    zeroed = np.where(finite, subject_matrix, 0.0).astype(np.float64, copy=False)
    finite_f = finite.astype(np.float64, copy=False)

    stats.scalar_counts += finite.sum(axis=0)
    stats.scalar_sums += zeroed.sum(axis=0)
    stats.scalar_sum_squares += (zeroed * zeroed).sum(axis=0)
    stats.voxel_sums += zeroed.T
    stats.voxel_counts += finite.T.astype(np.uint16, copy=False)
    stats.pair_counts += finite.T.astype(np.int64) @ finite.astype(np.int64)
    stats.pair_sum_x += zeroed.T @ finite_f
    stats.pair_sum_x2 += (zeroed * zeroed).T @ finite_f
    stats.pair_cross += zeroed.T @ zeroed
    stats.n_subjects_accumulated += 1
    stats.n_rows_accumulated += subject_matrix.shape[0]


def accumulate_voxelwise_stats(
    manifest_df: pd.DataFrame,
    scalar_labels: Sequence[str],
    mask_img: nib.Nifti1Image,
    mask: np.ndarray,
    outlier_iqr_multiplier: float | None = DEFAULT_OUTLIER_IQR_MULTIPLIER,
) -> StreamingStats:
    stats = initialize_stats(len(scalar_labels), int(mask.sum()))
    for subject_id, subject_df in tqdm(
        manifest_df.groupby("subject", sort=False),
        total=manifest_df["subject"].nunique(),
        desc="Accumulating voxelwise correlations",
    ):
        paths = subject_df.set_index("scalar")["path"].to_dict()
        columns = []
        for scalar in tqdm(scalar_labels, desc=f"Loading {subject_id}", leave=False):
            columns.append(load_masked_scalar(paths[scalar], mask_img, mask))
        subject_matrix = np.column_stack(columns)
        if outlier_iqr_multiplier is not None and outlier_iqr_multiplier > 0:
            subject_matrix = mask_subject_scalar_iqr_outliers(subject_matrix, outlier_iqr_multiplier)
        update_stats(stats, subject_matrix)
    return stats


def finalize_statistics(stats: StreamingStats, scalar_labels: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    n = stats.pair_counts.astype(np.float64)
    sum_x = stats.pair_sum_x
    sum_y = sum_x.T
    sum_x2 = stats.pair_sum_x2
    sum_y2 = sum_x2.T
    cross = stats.pair_cross

    with np.errstate(invalid="ignore", divide="ignore"):
        cov = (cross - (sum_x * sum_y / n)) / (n - 1.0)
        var_x = (sum_x2 - (sum_x * sum_x / n)) / (n - 1.0)
        var_y = (sum_y2 - (sum_y * sum_y / n)) / (n - 1.0)
        corr = cov / np.sqrt(var_x * var_y)
    corr[(n < 3) | ~np.isfinite(corr)] = np.nan
    np.fill_diagonal(corr, 1.0)
    corr_df = pd.DataFrame(corr, index=scalar_labels, columns=scalar_labels)

    means = np.divide(
        stats.scalar_sums,
        stats.scalar_counts,
        out=np.full_like(stats.scalar_sums, np.nan, dtype=np.float64),
        where=stats.scalar_counts > 0,
    )
    means_df = pd.DataFrame({"mean": means, "count": stats.scalar_counts}, index=scalar_labels)
    return corr_df, means_df, n


def voxel_scalar_means(stats: StreamingStats) -> np.ndarray:
    """Return scalar x voxel group-mean matrix, using finite observations only."""
    return np.divide(
        stats.voxel_sums,
        stats.voxel_counts,
        out=np.full_like(stats.voxel_sums, np.nan, dtype=np.float64),
        where=stats.voxel_counts > 0,
    )


def save_masked_vector_nii(
    values: np.ndarray,
    mask_img: nib.Nifti1Image,
    mask: np.ndarray,
    output_path: Path,
    fill_value: float = 0.0,
) -> None:
    """Save a masked vector back into the mask image grid."""
    data = np.full(mask.shape, fill_value, dtype=np.float32)
    data[mask] = values.astype(np.float32, copy=False)
    header = mask_img.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data, mask_img.affine, header), str(output_path))


def save_scalar_mean_images(
    scalar_mean_matrix: np.ndarray,
    scalar_labels: Sequence[str],
    mask_img: nib.Nifti1Image,
    mask: np.ndarray,
    output_dir: Path,
    file_prefix: str,
) -> None:
    """Write across-control mean images for each scalar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for scalar_idx, scalar in enumerate(tqdm(scalar_labels, desc="Saving scalar mean NIfTIs")):
        out_path = output_dir / f"{file_prefix}_scalar-{scalar}_mean.nii.gz"
        save_masked_vector_nii(scalar_mean_matrix[scalar_idx], mask_img, mask, out_path)


def save_factor_score_images(
    scalar_mean_matrix: np.ndarray,
    loadings_df: pd.DataFrame,
    scalar_labels: Sequence[str],
    mask_img: nib.Nifti1Image,
    mask: np.ndarray,
    output_dir: Path,
    file_prefix: str,
) -> None:
    """Write voxelwise factor score images from mean scalar vectors and factor loadings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    finite_means = np.where(np.isfinite(scalar_mean_matrix), scalar_mean_matrix, 0.0)
    for factor in tqdm(loadings_df.index, desc="Saving factor score NIfTIs"):
        weights = loadings_df.loc[factor, list(scalar_labels)].to_numpy(dtype=np.float64)
        scores = weights @ finite_means
        out_path = output_dir / f"{file_prefix}_{factor}_factor-score.nii.gz"
        save_masked_vector_nii(scores, mask_img, mask, out_path)


def regularize_correlation(corr_df: pd.DataFrame) -> Tuple[np.ndarray, bool]:
    corr = corr_df.fillna(0.0).to_numpy(dtype=np.float64)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    changed = bool(np.min(eigvals) < 1e-8)
    if changed:
        eigvals = np.clip(eigvals, 1e-8, None)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
    return corr, changed


def pca_from_correlation(corr: np.ndarray, scalar_labels: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    for j in range(eigvecs.shape[1]):
        idx = int(np.argmax(np.abs(eigvecs[:, j])))
        if eigvecs[idx, j] < 0:
            eigvecs[:, j] *= -1.0
    n_comp = min(N_PCA_COMPONENTS_FULL, len(scalar_labels))
    pc_labels = [f"PC{i + 1}" for i in range(n_comp)]
    loadings_df = pd.DataFrame(eigvecs[:, :n_comp].T, index=pc_labels, columns=scalar_labels)
    ev_ratio = eigvals[:n_comp] / np.sum(eigvals) if np.sum(eigvals) > 0 else np.zeros(n_comp)
    ev_df = pd.DataFrame(
        {
            "component": pc_labels,
            "variance_ratio": ev_ratio,
            "variance_percent": ev_ratio * 100.0,
        }
    )
    return loadings_df, ev_df, eigvals


def fit_factor_analysis(corr: np.ndarray, scalar_labels: Sequence[str], n_factors: int) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    from factor_analyzer import FactorAnalyzer  # noqa: PLC0415

    rotation = "promax"
    try:
        fa = FactorAnalyzer(
            n_factors=n_factors,
            method="minres",
            rotation=rotation,
            svd_method="lapack",
            is_corr_matrix=True,
        )
        fa.fit(corr)
    except Exception as exc:
        print(f"Warning: promax rotation failed ({exc}). Retrying without rotation.")
        fa = FactorAnalyzer(
            n_factors=n_factors,
            method="minres",
            rotation=None,
            svd_method="lapack",
            is_corr_matrix=True,
        )
        fa.fit(corr)
        rotation = "none (unrotated)"
    factor_labels = [f"F{i + 1}" for i in range(n_factors)]
    loadings_df = pd.DataFrame(fa.loadings_.T, index=factor_labels, columns=scalar_labels)
    loadings_df.index.name = "factor"
    uniquenesses_df = pd.DataFrame({"uniqueness": fa.get_uniquenesses()}, index=scalar_labels)
    return loadings_df, uniquenesses_df, rotation


def write_report_text(
    output_path: Path,
    subject_set: str,
    analysis_group: str,
    n_subjects: int,
    n_voxels: int,
    regularized: bool,
    scalar_labels: Sequence[str],
    outlier_iqr_multiplier: float | None,
    voxel_mask_label: str,
) -> None:
    if outlier_iqr_multiplier is not None and outlier_iqr_multiplier > 0:
        outlier_filter_line = (
            f"Subject-level IQR outlier filter: enabled "
            f"(omit values outside [Q1 - {outlier_iqr_multiplier}*IQR, Q3 + {outlier_iqr_multiplier}*IQR] "
            "per scalar across in-mask voxels)"
        )
    else:
        outlier_filter_line = "Subject-level IQR outlier filter: disabled"

    output_path.write_text(
        "\n".join(
            [
                "Voxelwise factor analysis notes",
                "================================",
                "",
                f"Subject set: {subject_set}",
                f"Analysis group: {analysis_group}",
                f"Voxel mask: {VOXEL_MASK_DESCRIPTIONS[voxel_mask_label]}",
                f"Subjects included: {n_subjects}",
                f"Mask voxels used: {n_voxels}",
                f"Scalars included: {len(scalar_labels)}",
                f"Correlation matrix regularized for modeling: {regularized}",
                outlier_filter_line,
                "",
                "The primary correlation matrix is computed by streaming raw MNI-space qsirecon maps",
                "inside the analysis voxel mask. Pairwise finite observations are used for each",
                "scalar-scalar Pearson correlation.",
                "",
                "Voxels outside the saved analysis mask are never extracted from the images and",
                "therefore do not contribute zeros, counts, means, or cross-products.",
                "Saved scalar and factor NIfTI files use 0 outside the analysis mask.",
                "",
                f"Fixed-factor FA outputs: factors-{FIXED_FACTOR_COUNTS[0]}/ and factors-{FIXED_FACTOR_COUNTS[1]}/ "
                f"(retain {FIXED_FACTOR_COUNTS[0]} and {FIXED_FACTOR_COUNTS[1]} factors respectively).",
            ]
        )
        + "\n"
    )


def split_hcpya_manifest(
    hcpya_manifest_df: pd.DataFrame,
    seed: int,
) -> List[Tuple[str, pd.DataFrame]]:
    """Randomly split HCP-YA subjects in half for separate FA runs."""
    subjects = sorted(hcpya_manifest_df["subject"].unique())
    if len(subjects) < 2:
        raise RuntimeError(
            f"--hcpya-split requires at least 2 complete hcpya subject-sessions; found {len(subjects)}."
        )
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(subjects))
    mid = len(perm) // 2
    splits = [
        ("hcpya_split-1", set(perm[:mid])),
        ("hcpya_split-2", set(perm[mid:])),
    ]
    print(
        f"HCPYA random split (seed={seed}): "
        f"{len(perm[:mid])} + {len(perm[mid:])} subject-sessions -> hcpya_split-1, hcpya_split-2"
    )
    return [
        (label, hcpya_manifest_df.loc[hcpya_manifest_df["subject"].isin(subject_ids)].copy())
        for label, subject_ids in splits
    ]


def build_analysis_subsets(
    manifest_df: pd.DataFrame,
    requested_groups: Sequence[str] | None,
    *,
    hcpya_split: bool = False,
    hcpya_split_seed: int = 42,
) -> List[Tuple[str, pd.DataFrame]]:
    """Return (subset_label, manifest) pairs to run. Default: all + each cohort."""
    labels = list(requested_groups) if requested_groups else list(ANALYSIS_GROUP_CHOICES)
    invalid = [label for label in labels if label not in ANALYSIS_GROUP_CHOICES]
    if invalid:
        raise ValueError(
            f"Invalid analysis group(s): {invalid}. "
            f"Choose from: {', '.join(ANALYSIS_GROUP_CHOICES)}"
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate analysis group labels requested: {labels}")

    subsets: List[Tuple[str, pd.DataFrame]] = []
    for label in labels:
        if label == "all":
            subset_df = manifest_df.copy()
        else:
            subset_df = manifest_df.loc[manifest_df["group"] == label].copy()
        if subset_df.empty:
            print(f"Skipping analysis group {label}: no complete subjects.")
            continue
        if label == "hcpya" and hcpya_split:
            subsets.extend(split_hcpya_manifest(subset_df, hcpya_split_seed))
        else:
            subsets.append((label, subset_df))
    if not subsets:
        raise RuntimeError("No analysis groups had complete subjects to run.")
    return subsets


def compute_pca_factor_correlation(
    pca_loadings_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    n_factors: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pca_plot_df, pca_corr_df) using the first n_factors PCs."""
    pca_plot_df = pca_loadings_df.iloc[:n_factors]
    pca_corr = np.full((n_factors, loadings_df.shape[0]), np.nan, dtype=float)
    for i in range(n_factors):
        pc_vec = pca_plot_df.iloc[i].to_numpy(dtype=float)
        for j, fac in enumerate(loadings_df.index):
            f_vec = loadings_df.loc[fac].to_numpy(dtype=float)
            if np.std(pc_vec) > 0 and np.std(f_vec) > 0:
                pca_corr[i, j] = np.corrcoef(pc_vec, f_vec)[0, 1]
    return pca_plot_df, pd.DataFrame(pca_corr, index=pca_plot_df.index, columns=loadings_df.index)


def run_fixed_factor_outputs(
    *,
    n_factors: int,
    factor_output_dir: Path,
    file_prefix: str,
    corr_df: pd.DataFrame,
    corr_for_model: np.ndarray,
    eigenvalues: np.ndarray,
    scalar_mean_matrix: np.ndarray,
    scalar_labels: Sequence[str],
    pca_loadings_df: pd.DataFrame,
    pca_ev_df: pd.DataFrame,
    mask_img: nib.Nifti1Image,
    mask: np.ndarray,
    skip_plots: bool,
    run_name: str,
    analysis_group_label: str,
    voxel_mask_label: str,
    n_subjects: int,
    n_voxels: int,
) -> None:
    factor_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fitting {n_factors} factors -> {factor_output_dir}")

    loadings_df, _, rotation_used = fit_factor_analysis(corr_for_model, scalar_labels, n_factors)
    loadings_path = factor_output_dir / f"{file_prefix}_scalar_factor_loadings.csv"
    loadings_df.to_csv(loadings_path)
    save_factor_score_images(
        scalar_mean_matrix,
        loadings_df,
        scalar_labels,
        mask_img,
        mask,
        factor_output_dir / "factor_nii",
        file_prefix,
    )
    loadings_df[order_scalars_by_max_abs_factor_loading(loadings_df)].to_csv(
        factor_output_dir / f"{file_prefix}_scalar_factor_loadings_ordered.csv"
    )

    pca_cum_pct = np.cumsum(pca_ev_df["variance_percent"].to_numpy(dtype=float))
    pca_plot_df, pca_corr_df = compute_pca_factor_correlation(pca_loadings_df, loadings_df, n_factors)

    if skip_plots:
        print(f"Done. Factor outputs (no plots): {factor_output_dir}")
        return

    corr_minimal_path = factor_output_dir / f"{file_prefix}_corr_matrix_minimal.png"
    corr_loadings_path = factor_output_dir / f"{file_prefix}_scalar_corr_and_factor_loadings.png"
    corr_loadings_ordered_path = factor_output_dir / f"{file_prefix}_scalar_corr_and_factor_loadings_factor_loading_ordered.png"
    corr_loadings_bottom_path = factor_output_dir / f"{file_prefix}_scalar_corr_and_factor_loadings_factor_loading_ordered_bottom.png"
    loadings_standalone_path = factor_output_dir / f"{file_prefix}_scalar_factor_loadings_standalone.png"
    pca_corr_path = factor_output_dir / f"{file_prefix}_pca_factor_correlation.png"
    pca_corr_loadings_path = factor_output_dir / f"{file_prefix}_scalar_corr_and_pca_loadings.png"
    combined_ordered_path = factor_output_dir / f"{file_prefix}_scalar_corr_factor_loadings_pca_components_combined_factor_loading_ordered.png"
    summary_path = factor_output_dir / f"{file_prefix}_factor_pca_combined_scree_and_correlations.png"

    plot_corr_matrix_minimal(corr_df, str(corr_minimal_path))
    plot_corr_and_loadings_combined(corr_df, loadings_df, str(corr_loadings_path))
    plot_corr_and_loadings_combined(
        corr_df,
        loadings_df,
        str(corr_loadings_ordered_path),
        row_order="max_factor_loading",
        dpi=CORR_FACTOR_PCA_FACTOR_ORDERED_DPI,
    )
    plot_corr_and_loadings_combined_bottom(
        corr_df,
        loadings_df,
        str(corr_loadings_bottom_path),
        row_order="max_factor_loading",
        dpi=CORR_FACTOR_PCA_FACTOR_ORDERED_DPI,
    )
    plot_factor_loadings_standalone(loadings_df, str(loadings_standalone_path))
    plot_pca_factor_correlation(pca_corr_df, str(pca_corr_path))
    plot_corr_and_ica_combined(corr_df, pca_plot_df, str(pca_corr_loadings_path))
    plot_corr_factor_loadings_and_pca_components_combined(
        corr_df,
        loadings_df,
        pca_plot_df,
        str(combined_ordered_path),
        row_order="max_factor_loading",
        dpi=CORR_FACTOR_PCA_FACTOR_ORDERED_DPI,
    )
    plot_factor_pca_combined_summary(
        eigenvalues,
        n_factors,
        pca_cum_pct,
        n_factors,
        pca_corr_df,
        str(summary_path),
    )
    create_html_factor_report(
        output_path=str(factor_output_dir / f"{file_prefix}_scalar_factor_analysis_report.html"),
        group_label=GROUP_LABEL,
        group_mode=GROUP_MODE,
        n_factors=n_factors,
        eigenvalues=eigenvalues,
        optimal_n_factors=n_factors,
        rotation_used=rotation_used,
        n_subjects=n_subjects,
        n_regions=n_voxels,
        n_tracts=0,
        n_scalars=len(scalar_labels),
        loadings_csv_path=str(loadings_path),
        variance_plot_path=str(summary_path),
        corr_heatmap_path=str(corr_loadings_path),
        pca_corr_and_loadings_plot_path=str(pca_corr_loadings_path),
        pca_corr_plot_path=str(pca_corr_path),
        pca_variance_plot_path=None,
        corr_factor_pca_components_plot_path=None,
        corr_factor_pca_components_plot_path_factor_ordered=str(combined_ordered_path),
        corr_and_loadings_factor_loading_ordered_path=str(corr_loadings_ordered_path),
        corr_factor_pca_components_plot_path_factor_ordered_dti_dki_gqi=None,
        factor_pca_combined_summary_plot_path=str(summary_path),
        atlas_set_label=f"{run_name}_{analysis_group_label}_{voxel_mask_label}_F{n_factors}",
    )
    print(f"Done. Factor outputs: {factor_output_dir}")


def run_analysis_subset(
    *,
    args: argparse.Namespace,
    run_name: str,
    analysis_group_label: str,
    subset_path_label: str,
    voxel_mask_label: str,
    subset_manifest_df: pd.DataFrame,
    scalar_labels: Sequence[str],
    mask_img: nib.Nifti1Image,
    mask: np.ndarray,
    n_voxels: int,
) -> None:
    subset_output_dir = OUTPUT_PROJECT_ROOT / run_name / subset_path_label
    subset_output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"{GROUP_LABEL}_{run_name}_{analysis_group_label}_{voxel_mask_label}"

    subset_manifest_df.to_csv(subset_output_dir / f"{file_prefix}_scalar_image_manifest.csv", index=False)
    subject_df = subset_manifest_df[["subject", "sub", "ses", "group"]].drop_duplicates()
    subject_df_out = subject_df.rename(columns={"subject": "subject_session"})
    subject_df_out.to_csv(subset_output_dir / "subjects_included.csv", index=False)
    save_analysis_mask(mask_img, mask, subset_output_dir / f"{file_prefix}_mni_t1w_mask_used.nii.gz")
    if voxel_mask_label == "no_csf":
        save_masked_t1w_preview(
            mask_img,
            mask,
            subset_output_dir / f"{file_prefix}_mni_t1w_brain_no_csf_preview.nii.gz",
        )

    print("\n" + "=" * 80)
    print(
        f"Voxelwise factor analysis: {subset_path_label} "
        f"({VOXEL_MASK_DESCRIPTIONS[voxel_mask_label]}, "
        f"n_subjects={subset_manifest_df['subject'].nunique()}, "
        f"n_voxels={n_voxels}, n_scalars={len(scalar_labels)})"
    )
    print("=" * 80)

    if args.dry_run:
        for n_factors in FIXED_FACTOR_COUNTS:
            (subset_output_dir / f"factors-{n_factors}").mkdir(parents=True, exist_ok=True)
        print(f"Dry run complete for {subset_path_label}; no scalar images were loaded.")
        return

    stats = accumulate_voxelwise_stats(
        subset_manifest_df,
        scalar_labels,
        mask_img,
        mask,
        outlier_iqr_multiplier=args.outlier_iqr_multiplier,
    )
    corr_df, _, _ = finalize_statistics(stats, scalar_labels)
    corr_for_model, regularized = regularize_correlation(corr_df)
    scalar_mean_matrix = voxel_scalar_means(stats)
    save_scalar_mean_images(
        scalar_mean_matrix,
        scalar_labels,
        mask_img,
        mask,
        subset_output_dir / "scalar_nii",
        file_prefix,
    )

    corr_df.to_csv(subset_output_dir / f"{file_prefix}_scalar_correlations.csv")
    if args.stop_after_stats:
        write_report_text(
            subset_output_dir / f"{file_prefix}_voxelwise_notes.md",
            args.subject_set,
            analysis_group_label,
            subset_manifest_df["subject"].nunique(),
            n_voxels,
            regularized,
            scalar_labels,
            args.outlier_iqr_multiplier,
            voxel_mask_label,
        )
        print(f"Stopped after streaming statistics for {subset_path_label} as requested.")
        print(f"Done. Outputs: {subset_output_dir}")
        return

    import_factor_analysis_helpers()

    eigenvalues = np.linalg.eigvalsh(corr_for_model)[::-1]
    cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else np.zeros_like(eigenvalues)
    variance_df = pd.DataFrame(
        {
            "Factor": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "variance_fraction": eigenvalues / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else np.zeros_like(eigenvalues),
            "variance_percent": eigenvalues / np.sum(eigenvalues) * 100.0 if np.sum(eigenvalues) > 0 else np.zeros_like(eigenvalues),
            "cumulative_fraction": cumulative,
            "cumulative_percent": cumulative * 100.0,
        }
    )
    variance_df.to_csv(subset_output_dir / f"{file_prefix}_scalar_factor_eigenvalues.csv", index=False)

    pca_loadings_df, pca_ev_df, _ = pca_from_correlation(corr_for_model, scalar_labels)
    pca_loadings_df.to_csv(subset_output_dir / f"{file_prefix}_pca_component_loadings.csv")
    pca_ev_df.to_csv(subset_output_dir / f"{file_prefix}_pca_explained_variance_ratio.csv", index=False)

    write_report_text(
        subset_output_dir / f"{file_prefix}_voxelwise_notes.md",
        args.subject_set,
        analysis_group_label,
        subset_manifest_df["subject"].nunique(),
        n_voxels,
        regularized,
        scalar_labels,
        args.outlier_iqr_multiplier,
        voxel_mask_label,
    )

    for n_factors in FIXED_FACTOR_COUNTS:
        if n_factors >= len(scalar_labels):
            raise RuntimeError(
                f"Cannot fit {n_factors} factors with only {len(scalar_labels)} scalars."
            )
        run_fixed_factor_outputs(
            n_factors=n_factors,
            factor_output_dir=subset_output_dir / f"factors-{n_factors}",
            file_prefix=file_prefix,
            corr_df=corr_df,
            corr_for_model=corr_for_model,
            eigenvalues=eigenvalues,
            scalar_mean_matrix=scalar_mean_matrix,
            scalar_labels=scalar_labels,
            pca_loadings_df=pca_loadings_df,
            pca_ev_df=pca_ev_df,
            mask_img=mask_img,
            mask=mask,
            skip_plots=args.skip_plots,
            run_name=run_name,
            analysis_group_label=analysis_group_label,
            voxel_mask_label=voxel_mask_label,
            n_subjects=subset_manifest_df["subject"].nunique(),
            n_voxels=n_voxels,
        )

    print(f"Done. Outputs: {subset_output_dir}")


def run_pipeline(args: argparse.Namespace) -> None:
    scalar_labels, scalar_to_file, scalar_to_dir = load_scalar_metadata()
    subjects = discover_subject_sessions(GROUPS, scalar_to_dir)
    if args.subject_set == "reduced":
        reduced_subject_ids = load_reduced_subject_ids(REDUCED_SUBJECTS_PATH)
        discovered_subject_ids = {subject.sub for subject in subjects}
        unmatched_subject_ids = sorted(reduced_subject_ids - discovered_subject_ids)
        subjects = [subject for subject in subjects if subject.sub in reduced_subject_ids]
        print(f"Reduced subject set requested: {len(reduced_subject_ids)} subjects listed; {len(subjects)} discovered subject-sessions matched.")
        if unmatched_subject_ids:
            print(f"Warning: {len(unmatched_subject_ids)} reduced subjects were not discovered in qsirecon outputs.")
    run_name = RUN_NAMES[args.subject_set]
    run_output_dir = OUTPUT_PROJECT_ROOT / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    if args.subject_set == "reduced":
        pd.DataFrame({"subject": sorted(reduced_subject_ids)}).to_csv(
            run_output_dir / f"{GROUP_LABEL}_{run_name}_reduced_subjects_requested.csv",
            index=False,
        )
        if unmatched_subject_ids:
            pd.DataFrame({"subject": unmatched_subject_ids}).to_csv(
                run_output_dir / f"{GROUP_LABEL}_{run_name}_reduced_subjects_not_discovered.csv",
                index=False,
            )

    manifest_df, missing_df = build_manifest(
        subjects,
        scalar_labels,
        scalar_to_file,
        scalar_to_dir,
        args.max_subjects,
    )
    file_prefix = f"{GROUP_LABEL}_{run_name}"
    manifest_df.to_csv(run_output_dir / f"{file_prefix}_scalar_image_manifest.csv", index=False)
    missing_df.to_csv(run_output_dir / f"{file_prefix}_missing_scalar_images.csv", index=False)
    if manifest_df.empty:
        raise RuntimeError("No complete control subjects found with all required scalar images.")
    subject_df = manifest_df[["subject", "sub", "ses", "group"]].drop_duplicates()
    subject_df_out = subject_df.rename(columns={"subject": "subject_session"})
    subject_df_out.to_csv(run_output_dir / "subjects_included.csv", index=False)
    subject_df_out.to_csv(OUTPUT_PROJECT_ROOT / f"subjects_included_{args.subject_set}.csv", index=False)

    print(f"Complete subjects: {manifest_df['subject'].nunique()} / discovered subject-sessions: {len(subjects)}")
    print(f"Scalars: {len(scalar_labels)}")
    reference_img = load_reference_image(manifest_df, scalar_labels)
    voxel_masks = build_voxel_masks(reference_img, args.max_voxels)
    with_csf_mask_img, with_csf_mask = voxel_masks["with_csf"]
    save_analysis_mask(
        with_csf_mask_img,
        with_csf_mask,
        run_output_dir / f"{file_prefix}_mni_t1w_mask_used.nii.gz",
    )
    analysis_subsets = build_analysis_subsets(
        manifest_df,
        args.analysis_group,
        hcpya_split=args.hcpya_split,
        hcpya_split_seed=args.hcpya_split_seed,
    )
    voxel_mask_modes = resolve_voxel_mask_modes(args.voxel_mask)
    print(
        "Analysis groups to run: "
        + ", ".join(f"{label} (n={df['subject'].nunique()})" for label, df in analysis_subsets)
    )
    print(f"Voxel mask modes: {', '.join(voxel_mask_modes)}")

    for analysis_group_label, subset_manifest_df in analysis_subsets:
        for mask_mode in voxel_mask_modes:
            voxel_mask_label = VOXEL_MASK_SUBDIRS[mask_mode]
            mask_img, mask = voxel_masks[voxel_mask_label]
            subset_path_label = f"{analysis_group_label}/{voxel_mask_label}"
            run_analysis_subset(
                args=args,
                run_name=run_name,
                analysis_group_label=analysis_group_label,
                subset_path_label=subset_path_label,
                voxel_mask_label=voxel_mask_label,
                subset_manifest_df=subset_manifest_df,
                scalar_labels=scalar_labels,
                mask_img=mask_img,
                mask=mask,
                n_voxels=int(mask.sum()),
            )

    print(f"Done. Group-specific outputs: {run_output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject-set",
        choices=("all", "reduced"),
        default="all",
        help=(
            "Subject set to use: 'all' for every complete control subject-session with scalar images, "
            "or 'reduced' for subjects listed in controls_F1_scores.csv."
        ),
    )
    parser.add_argument(
        "--analysis-group",
        choices=ANALYSIS_GROUP_CHOICES,
        nargs="+",
        default=None,
        metavar="GROUP",
        help=(
            "Run FA on one or more cohort subsets under the run output directory. "
            "Choices: all, penn_controls, hcpya, hcpaging. "
            "Default: run all four subsets."
        ),
    )
    parser.add_argument(
        "--hcpya-split",
        action="store_true",
        help=(
            "When hcpya is among the analysis groups to run, randomly split its subjects in half "
            "and run separate FA in hcpya_split-1/ and hcpya_split-2/ instead of a single hcpya/ run."
        ),
    )
    parser.add_argument(
        "--hcpya-split-seed",
        type=int,
        default=42,
        help="Random seed for --hcpya-split subject assignment (default: 42).",
    )
    parser.add_argument(
        "--voxel-mask",
        choices=VOXEL_MASK_CHOICES,
        default="both",
        help=(
            "Voxel set for FA: brain (with CSF, saved under with_csf/), "
            "brain_no_csf (exclude CSF voxels with prob > 0.5, saved under no_csf/), or both."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Build manifests only; do not load images.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Limit complete subject-sessions for smoke tests.")
    parser.add_argument("--max-voxels", type=int, default=None, help="Use only the first N mask voxels for smoke tests.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip PNG/HTML report generation.")
    parser.add_argument("--stop-after-stats", action="store_true", help="Stop after streamed correlations/means are written.")
    parser.add_argument(
        "--outlier-iqr-multiplier",
        type=float,
        default=DEFAULT_OUTLIER_IQR_MULTIPLIER,
        help=(
            "Per subject, omit voxel values outside [Q1 - k*IQR, Q3 + k*IQR] for each scalar "
            f"(IQR across in-mask voxels). Default: {DEFAULT_OUTLIER_IQR_MULTIPLIER}. "
            "Set to 0 to disable outlier filtering."
        ),
    )
    args = parser.parse_args()
    if args.hcpya_split:
        groups = list(args.analysis_group) if args.analysis_group else list(ANALYSIS_GROUP_CHOICES)
        if "hcpya" not in groups:
            parser.error("--hcpya-split requires hcpya among the analysis groups to run")
    return args


if __name__ == "__main__":
    run_pipeline(parse_args())
