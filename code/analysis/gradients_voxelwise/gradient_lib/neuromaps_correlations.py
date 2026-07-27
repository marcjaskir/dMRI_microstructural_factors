"""Screen factor gradient NIfTIs against neuromaps reference annotations."""

from __future__ import annotations

import ast
import logging
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from tqdm import tqdm

from .config import (
    COHORT_TAG,
    DEFAULT_MASK_NII,
    NEUROMAPS_ANNOTATION_INFO_CSV,
    NEUROMAPS_DEFAULT_N_PERM_FSLR,
    NEUROMAPS_DEFAULT_N_PERM_MNI,
    NEUROMAPS_FSLR_DENSITY,
    NEUROMAPS_LABEL_OVERRIDES,
    NEUROMAPS_MNI_DENSITY,
    NEUROMAPS_NULL_SEED,
    NEUROMAPS_SPACE_FSLR,
    NEUROMAPS_SPACE_MNI,
)
from .io_voxelwise import gradient_nii_path

logger = logging.getLogger(__name__)

AnnotationTuple = tuple[str, str, str, str]


@dataclass(frozen=True)
class AnnotationEntry:
    source: str
    desc: str
    space: str
    den_or_res: str
    pool: str  # "MNI152" or "fsLR"
    origin: str  # "native" or "mni_to_fslr"

    @property
    def key(self) -> str:
        return f"{self.source}:{self.desc}"

    @property
    def annotation_key(self) -> str:
        if self.origin == "mni_to_fslr":
            return f"{self.key} (MNI→fsLR)"
        return self.key


def neuromaps_data_dir(output_dir: Path) -> Path:
    return output_dir / "_cache" / "neuromaps" / "data"


def neuromaps_nulls_dir(output_dir: Path) -> Path:
    return output_dir / "_cache" / "neuromaps" / "nulls"


def neuromaps_correlation_csv_path(
    output_dir: Path,
    factor_tag: str,
    gradient_index: int,
    *,
    cohort_tag: str = COHORT_TAG,
) -> Path:
    return (
        output_dir
        / "csv"
        / f"{factor_tag}_gradient{gradient_index}_neuromaps_correlations_cohort-{cohort_tag}.csv"
    )


def setup_neuromaps_data_dir(output_dir: Path) -> Path:
    """Point neuromaps downloads at ``output_dir/_cache/neuromaps/data``."""
    data_dir = neuromaps_data_dir(output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NEUROMAPS_DATA", str(data_dir))
    return data_dir


@dataclass
class AnnotationMetadataIndex:
    """Parsed neuromaps annotation metadata for labels and deduplication."""

    description_brief: dict[tuple[str, str], str] = field(default_factory=dict)
    sample_size: dict[tuple[str, str], int] = field(default_factory=dict)
    allowed_keys: set[tuple[str, str]] = field(default_factory=set)
    metadata_keys: set[tuple[str, str]] = field(default_factory=set)


def parse_sample_size_n(value: object) -> int:
    """Parse total N from ``N (males)`` column (e.g. ``36 (24)`` → 36)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    s = str(value).strip()
    if not s:
        return 0
    m = re.match(r"^\s*(\d+)", s)
    if m:
        return int(m.group(1))
    return 0


def load_annotation_metadata(
    metadata_csv: Path | None = None,
) -> AnnotationMetadataIndex:
    """Load brief labels, sample sizes, and deduped allowed (source, desc) keys."""
    path = metadata_csv or NEUROMAPS_ANNOTATION_INFO_CSV
    idx = AnnotationMetadataIndex()
    if not path.is_file():
        return idx

    per_key_brief: dict[tuple[str, str], str] = {}
    per_key_n: dict[tuple[str, str], int] = {}

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        raw = row.get("annotation")
        if pd.isna(raw):
            continue
        try:
            ann = ast.literal_eval(str(raw))
            source, desc = str(ann[0]), str(ann[1])
        except (SyntaxError, ValueError, IndexError):
            continue
        key = (source, desc)
        idx.metadata_keys.add(key)

        brief = row.get("description_brief")
        if pd.isna(brief) or not str(brief).strip():
            brief = row.get("description")
        if not pd.isna(brief) and str(brief).strip():
            per_key_brief[key] = str(brief).strip()

        n_col = row.get("N (males)")
        if n_col is None:
            n_col = row.get("N")
        n_val = parse_sample_size_n(n_col)
        per_key_n[key] = max(per_key_n.get(key, 0), n_val)

    idx.description_brief = per_key_brief
    idx.sample_size = per_key_n
    for key, label in NEUROMAPS_LABEL_OVERRIDES.items():
        idx.description_brief[key] = label

    by_brief: dict[str, list[tuple[str, str]]] = {}
    for key, brief in per_key_brief.items():
        by_brief.setdefault(brief, []).append(key)

    allowed: set[tuple[str, str]] = set()
    for brief, keys in by_brief.items():
        best = max(keys, key=lambda k: (per_key_n.get(k, 0), k[0], k[1]))
        allowed.add(best)
    idx.allowed_keys = allowed
    return idx


def _filter_entries_by_metadata(
    entries: list[AnnotationEntry],
    metadata: AnnotationMetadataIndex,
) -> list[AnnotationEntry]:
    """Keep only the largest-N map per ``description_brief`` when deduplicating."""
    if not metadata.metadata_keys:
        return entries
    out: list[AnnotationEntry] = []
    for entry in entries:
        key = (entry.source, entry.desc)
        if key not in metadata.metadata_keys:
            out.append(entry)
        elif key in metadata.allowed_keys:
            out.append(entry)
    return out


def _annotation_entries_mni(
    metadata: AnnotationMetadataIndex | None = None,
) -> list[AnnotationEntry]:
    from neuromaps.datasets import available_annotations

    entries: list[AnnotationEntry] = []
    for source, desc, space, res in available_annotations(
        format="volume", return_restricted=False
    ):
        entries.append(
            AnnotationEntry(source, desc, space, res, pool=NEUROMAPS_SPACE_MNI, origin="native")
        )
    if metadata is not None:
        entries = _filter_entries_by_metadata(entries, metadata)
    return entries


def _annotation_entries_fslr(
    metadata: AnnotationMetadataIndex | None = None,
) -> list[AnnotationEntry]:
    from neuromaps.datasets import available_annotations

    native_keys: set[tuple[str, str]] = set()
    entries: list[AnnotationEntry] = []

    for source, desc, space, den in available_annotations(
        space=NEUROMAPS_SPACE_FSLR,
        den=NEUROMAPS_FSLR_DENSITY,
        return_restricted=False,
    ):
        native_keys.add((source, desc))
        entries.append(
            AnnotationEntry(
                source, desc, space, den, pool=NEUROMAPS_SPACE_FSLR, origin="native"
            )
        )

    for source, desc, space, res in available_annotations(
        format="volume", return_restricted=False
    ):
        if (source, desc) in native_keys:
            continue
        entries.append(
            AnnotationEntry(
                source,
                desc,
                space,
                res,
                pool=NEUROMAPS_SPACE_FSLR,
                origin="mni_to_fslr",
            )
        )
    if metadata is not None:
        entries = _filter_entries_by_metadata(entries, metadata)
    return entries


def fetch_annotation_path(entry: AnnotationEntry) -> str | list:
    from neuromaps.datasets import fetch_annotation

    kwargs: dict[str, object] = {
        "source": entry.source,
        "desc": entry.desc,
        "return_single": True,
        "verbose": 0,
    }
    if entry.origin == "native" and entry.pool == NEUROMAPS_SPACE_FSLR:
        kwargs.update(space=entry.space, den=entry.den_or_res)
    else:
        kwargs.update(space=entry.space, res=entry.den_or_res)
    return fetch_annotation(**kwargs)


def _mask_nifti(img: nib.Nifti1Image, mask: np.ndarray) -> nib.Nifti1Image:
    data = np.asarray(img.get_fdata(), dtype=np.float64).copy()
    data[~mask] = 0.0
    return nib.Nifti1Image(data, img.affine, img.header)


def prepare_gradient_mni(
    gradient_path: Path,
    *,
    analysis_mask_path: Path = DEFAULT_MASK_NII,
    density: str = NEUROMAPS_MNI_DENSITY,
) -> tuple[nib.Nifti1Image, np.ndarray]:
    from neuromaps import transforms

    grad = transforms.mni152_to_mni152(str(gradient_path), density)
    mask_img = nib.load(str(analysis_mask_path))
    mask_resampled = resample_from_to(mask_img, (grad.shape, grad.affine), order=0)
    mask = np.asarray(mask_resampled.get_fdata()) > 0
    return _mask_nifti(grad, mask), mask


def prepare_gradient_fslr(
    gradient_path: Path,
    *,
    density: str = NEUROMAPS_FSLR_DENSITY,
) -> tuple[nib.GiftiImage, nib.GiftiImage]:
    from neuromaps import transforms

    lh, rh = transforms.mni152_to_fslr(str(gradient_path), fslr_density=density, method="linear")
    return lh, rh


def _null_cache_path(
    output_dir: Path,
    factor_tag: str,
    gradient_index: int,
    space: str,
    *,
    n_perm: int,
) -> Path:
    model = "burt2020" if space == NEUROMAPS_SPACE_MNI else "alexander_bloch"
    return (
        neuromaps_nulls_dir(output_dir)
        / f"{factor_tag}_G{gradient_index}_{space}_{model}_n{n_perm}.npz"
    )


def load_or_compute_nulls_mni(
    grad_mni: nib.Nifti1Image,
    *,
    output_dir: Path,
    factor_tag: str,
    gradient_index: int,
    n_perm: int,
    seed: int = NEUROMAPS_NULL_SEED,
    skip_nulls: bool = False,
) -> np.ndarray | None:
    if skip_nulls or n_perm <= 0:
        return None

    cache_path = _null_cache_path(
        output_dir, factor_tag, gradient_index, NEUROMAPS_SPACE_MNI, n_perm=n_perm
    )
    if cache_path.is_file():
        logger.info("Loading cached MNI nulls: %s", cache_path)
        return np.load(cache_path)["nulls"]

    from neuromaps import nulls

    t0 = time.time()
    logger.info(
        "Computing burt2020 MNI nulls for %s G%d (n_perm=%d; may be slow)...",
        factor_tag,
        gradient_index,
        n_perm,
    )
    null_arr = nulls.burt2020(
        grad_mni,
        atlas=NEUROMAPS_SPACE_MNI,
        density=NEUROMAPS_MNI_DENSITY,
        n_perm=n_perm,
        seed=seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, nulls=null_arr)
    logger.info(
        "Cached MNI nulls (%s) in %.1fs",
        null_arr.shape,
        time.time() - t0,
    )
    return null_arr


def load_or_compute_nulls_fslr(
    grad_fslr: tuple[nib.GiftiImage, nib.GiftiImage],
    *,
    output_dir: Path,
    factor_tag: str,
    gradient_index: int,
    n_perm: int,
    seed: int = NEUROMAPS_NULL_SEED,
    skip_nulls: bool = False,
) -> np.ndarray | None:
    if skip_nulls or n_perm <= 0:
        return None

    cache_path = _null_cache_path(
        output_dir, factor_tag, gradient_index, NEUROMAPS_SPACE_FSLR, n_perm=n_perm
    )
    if cache_path.is_file():
        logger.info("Loading cached fsLR nulls: %s", cache_path)
        return np.load(cache_path)["nulls"]

    from neuromaps import nulls

    t0 = time.time()
    logger.info(
        "Computing alexander_bloch fsLR nulls for %s G%d (n_perm=%d)...",
        factor_tag,
        gradient_index,
        n_perm,
    )
    null_arr = nulls.alexander_bloch(
        grad_fslr,
        atlas=NEUROMAPS_SPACE_FSLR,
        density=NEUROMAPS_FSLR_DENSITY,
        n_perm=n_perm,
        seed=seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, nulls=null_arr)
    logger.info(
        "Cached fsLR nulls (%s) in %.1fs",
        null_arr.shape,
        time.time() - t0,
    )
    return null_arr


def _compare(
    src,
    trg,
    *,
    nulls: np.ndarray | None,
) -> tuple[float, float]:
    from neuromaps import stats

    if nulls is None:
        r = float(stats.compare_images(src, trg, metric="pearsonr", ignore_zero=True, nan_policy="omit"))
        return r, float("nan")
    r, p = stats.compare_images(
        src, trg, metric="pearsonr", ignore_zero=True, nulls=nulls, nan_policy="omit"
    )
    return float(r), float(p)


def _prepare_annotation_mni(
    entry: AnnotationEntry,
    grad_mni: nib.Nifti1Image,
    mask: np.ndarray,
) -> nib.Nifti1Image | None:
    from neuromaps import resampling, transforms

    try:
        path = fetch_annotation_path(entry)
    except Exception as exc:
        logger.warning("Skip fetch %s: %s", entry.annotation_key, exc)
        return None

    try:
        if entry.origin == "mni_to_fslr":
            return None
        ann_res = transforms.mni152_to_mni152(path, NEUROMAPS_MNI_DENSITY)
        ann_aligned, grad_aligned = resampling.resample_images(
            ann_res, grad_mni, NEUROMAPS_SPACE_MNI, NEUROMAPS_SPACE_MNI
        )
        ann_masked = _mask_nifti(ann_aligned, mask)
        _ = grad_aligned  # gradient already masked; resample keeps grid
        return ann_masked
    except Exception as exc:
        logger.warning("Skip MNI prep %s: %s", entry.annotation_key, exc)
        return None


def _prepare_annotation_fslr(
    entry: AnnotationEntry,
    grad_fslr: tuple[nib.GiftiImage, nib.GiftiImage],
):
    from neuromaps import resampling, transforms

    try:
        path = fetch_annotation_path(entry)
    except Exception as exc:
        logger.warning("Skip fetch %s: %s", entry.annotation_key, exc)
        return None

    try:
        if entry.origin == "mni_to_fslr":
            ann_fslr = transforms.mni152_to_fslr(
                path, fslr_density=NEUROMAPS_FSLR_DENSITY, method="linear"
            )
        else:
            ann_fslr = path

        ann_aligned, grad_aligned = resampling.resample_images(
            ann_fslr, grad_fslr, NEUROMAPS_SPACE_FSLR, NEUROMAPS_SPACE_FSLR
        )
        return ann_aligned, grad_aligned
    except Exception as exc:
        logger.warning("Skip fsLR prep %s: %s", entry.annotation_key, exc)
        return None


def correlate_annotation_mni(
    entry: AnnotationEntry,
    grad_mni: nib.Nifti1Image,
    mask: np.ndarray,
    *,
    nulls: np.ndarray | None,
) -> tuple[float, float] | None:
    ann = _prepare_annotation_mni(entry, grad_mni, mask)
    if ann is None:
        return None
    try:
        return _compare(grad_mni, ann, nulls=nulls)
    except Exception as exc:
        logger.warning("Skip MNI compare %s: %s", entry.annotation_key, exc)
        return None


def correlate_annotation_fslr(
    entry: AnnotationEntry,
    grad_fslr: tuple[nib.GiftiImage, nib.GiftiImage],
    *,
    nulls: np.ndarray | None,
) -> tuple[float, float] | None:
    prepared = _prepare_annotation_fslr(entry, grad_fslr)
    if prepared is None:
        return None
    ann, grad = prepared
    try:
        return _compare(grad, ann, nulls=nulls)
    except Exception as exc:
        logger.warning("Skip fsLR compare %s: %s", entry.annotation_key, exc)
        return None


def screen_factor_gradient(
    output_dir: Path,
    factor_tag: str,
    gradient_index: int,
    *,
    n_perm_mni: int = NEUROMAPS_DEFAULT_N_PERM_MNI,
    n_perm_fslr: int = NEUROMAPS_DEFAULT_N_PERM_FSLR,
    skip_nulls: bool = False,
    max_annotations: int | None = None,
    cohort_tag: str = COHORT_TAG,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute correlations for one factor gradient map against neuromaps pools."""
    setup_neuromaps_data_dir(output_dir)
    gradient_path = gradient_nii_path(output_dir, factor_tag, gradient_index)
    if not gradient_path.is_file():
        raise FileNotFoundError(f"Missing gradient map: {gradient_path}")

    grad_mni, mask = prepare_gradient_mni(gradient_path)
    grad_fslr = prepare_gradient_fslr(gradient_path)

    nulls_mni = load_or_compute_nulls_mni(
        grad_mni,
        output_dir=output_dir,
        factor_tag=factor_tag,
        gradient_index=gradient_index,
        n_perm=n_perm_mni,
        skip_nulls=skip_nulls,
    )
    nulls_fslr = load_or_compute_nulls_fslr(
        grad_fslr,
        output_dir=output_dir,
        factor_tag=factor_tag,
        gradient_index=gradient_index,
        n_perm=n_perm_fslr,
        skip_nulls=skip_nulls,
    )

    rows: list[dict[str, object]] = []

    metadata = load_annotation_metadata()
    mni_entries = _annotation_entries_mni(metadata)
    fslr_entries = _annotation_entries_fslr(metadata)
    if max_annotations is not None:
        mni_entries = mni_entries[:max_annotations]
        fslr_entries = fslr_entries[:max_annotations]

    mni_iter = tqdm(
        mni_entries,
        desc=f"{factor_tag} G{gradient_index} MNI152",
        leave=False,
        disable=not show_progress,
    )
    for entry in mni_iter:
        result = correlate_annotation_mni(entry, grad_mni, mask, nulls=nulls_mni)
        if result is None:
            continue
        r, p = result
        rows.append(
            {
                "factor": factor_tag,
                "gradient": gradient_index,
                "space": NEUROMAPS_SPACE_MNI,
                "source": entry.source,
                "desc": entry.desc,
                "annotation_key": entry.annotation_key,
                "origin": entry.origin,
                "pearson_r": r,
                "abs_r": abs(r),
                "p_null": p,
                "cohort_tag": cohort_tag,
            }
        )

    fslr_iter = tqdm(
        fslr_entries,
        desc=f"{factor_tag} G{gradient_index} fsLR",
        leave=False,
        disable=not show_progress,
    )
    for entry in fslr_iter:
        result = correlate_annotation_fslr(entry, grad_fslr, nulls=nulls_fslr)
        if result is None:
            continue
        r, p = result
        rows.append(
            {
                "factor": factor_tag,
                "gradient": gradient_index,
                "space": NEUROMAPS_SPACE_FSLR,
                "source": entry.source,
                "desc": entry.desc,
                "annotation_key": entry.annotation_key,
                "origin": entry.origin,
                "pearson_r": r,
                "abs_r": abs(r),
                "p_null": p,
                "cohort_tag": cohort_tag,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["rank_abs_r"] = (
        df.groupby(["factor", "gradient", "space"], sort=False)["abs_r"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return df.sort_values(["space", "rank_abs_r"]).reset_index(drop=True)


def save_correlation_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def run_all_correlations(
    output_dir: Path,
    *,
    factors: list[str],
    gradient_indices: list[int],
    n_perm_mni: int = NEUROMAPS_DEFAULT_N_PERM_MNI,
    n_perm_fslr: int = NEUROMAPS_DEFAULT_N_PERM_FSLR,
    skip_nulls: bool = False,
    max_annotations: int | None = None,
    cohort_tag: str = COHORT_TAG,
    show_progress: bool = True,
) -> list[Path]:
    paths: list[Path] = []
    jobs = [(factor, gi) for factor in factors for gi in gradient_indices]
    job_iter = tqdm(jobs, desc="Neuromaps screening", disable=not show_progress)
    for factor, gi in job_iter:
        if show_progress and hasattr(job_iter, "set_postfix_str"):
            job_iter.set_postfix_str(f"{factor} G{gi}", refresh=False)
        logger.info("Screening neuromaps annotations: %s gradient %d", factor, gi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = screen_factor_gradient(
                output_dir,
                factor,
                gi,
                n_perm_mni=n_perm_mni,
                n_perm_fslr=n_perm_fslr,
                skip_nulls=skip_nulls,
                max_annotations=max_annotations,
                cohort_tag=cohort_tag,
                show_progress=show_progress,
            )
        out = neuromaps_correlation_csv_path(
            output_dir, factor, gi, cohort_tag=cohort_tag
        )
        save_correlation_csv(df, out)
        paths.append(out)
        logger.info("  saved %s (%d rows)", out, len(df))
    return paths


def load_annotation_description_lookup(
    metadata_csv: Path | None = None,
) -> dict[tuple[str, str], str]:
    """Map (source, desc) → brief plot label from neuromaps metadata."""
    return load_annotation_metadata(metadata_csv).description_brief


def load_annotation_sample_size_lookup(
    metadata_csv: Path | None = None,
) -> dict[tuple[str, str], int]:
    """Map (source, desc) → total sample size N."""
    return load_annotation_metadata(metadata_csv).sample_size


def dedupe_correlations_by_description_brief(
    df: pd.DataFrame,
    metadata: AnnotationMetadataIndex | None = None,
) -> pd.DataFrame:
    """Keep one row per ``description_brief`` (largest N; tie-break by |r|)."""
    if df.empty:
        return df
    meta = metadata or load_annotation_metadata()
    if not meta.description_brief:
        return df

    work = df.copy()
    work["_brief"] = work.apply(
        lambda r: annotation_display_label(
            str(r["source"]),
            str(r["desc"]),
            meta.description_brief,
            fallback=str(r.get("annotation_key", f"{r['source']}:{r['desc']}")),
        ),
        axis=1,
    )
    work["_n"] = work.apply(
        lambda r: meta.sample_size.get((str(r["source"]), str(r["desc"])), 0),
        axis=1,
    )
    work = work.sort_values(["_n", "abs_r"], ascending=[False, False])
    work = work.drop_duplicates(subset=["_brief"], keep="first")
    return work.drop(columns=["_brief", "_n"])


def annotation_display_label(
    source: str,
    desc: str,
    lookup: dict[tuple[str, str], str],
    *,
    fallback: str | None = None,
) -> str:
    """Resolve plot label for an annotation row."""
    key = (str(source), str(desc))
    if key in NEUROMAPS_LABEL_OVERRIDES:
        return NEUROMAPS_LABEL_OVERRIDES[key]
    label = lookup.get(key)
    if label:
        return label
    if fallback:
        return fallback
    return f"{source}:{desc}"
