"""Screen group-mean factor score NIfTIs against neuromaps annotations."""

from __future__ import annotations

import logging
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    DEFAULT_MAP_VARIANT,
    DEFAULT_MASK_NII,
    FILE_PREFIX,
    parse_neuromaps_spaces,
)

# Reuse annotation pools, transforms, and comparison helpers from vendored gradient_lib.
from gradient_lib.config import (  # noqa: E402
    COHORT_TAG,
    NEUROMAPS_DEFAULT_N_PERM_FSLR,
    NEUROMAPS_DEFAULT_N_PERM_MNI,
    NEUROMAPS_FSLR_DENSITY,
    NEUROMAPS_MNI_DENSITY,
    NEUROMAPS_NULL_SEED,
    NEUROMAPS_SPACE_FSLR,
    NEUROMAPS_SPACE_MNI,
)
from gradient_lib.neuromaps_correlations import (  # noqa: E402
    _annotation_entries_fslr,
    _annotation_entries_mni,
    _compare,
    _prepare_annotation_fslr,
    _prepare_annotation_mni,
    load_annotation_metadata,
    neuromaps_nulls_dir,
    save_correlation_csv,
    setup_neuromaps_data_dir,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _progress_write(message: str, *, show_progress: bool) -> None:
    if show_progress:
        tqdm.write(message)


def _truncate_label(text: str, max_len: int = 48) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _run_with_elapsed_postfix(
    pbar: tqdm,
    label: str,
    fn: Callable[[], T],
) -> T:
    """Keep the progress bar postfix ticking with elapsed seconds during long work."""
    stop = threading.Event()
    t0 = time.time()

    def _tick() -> None:
        while not stop.wait(1.0):
            elapsed = int(time.time() - t0)
            pbar.set_postfix_str(f"{label} ({elapsed}s)", refresh=True)

    thread = threading.Thread(target=_tick, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        stop.set()
        thread.join(timeout=0.2)


def factor_score_nii_path(
    input_dir: Path,
    factor_tag: str,
    *,
    file_prefix: str = FILE_PREFIX,
    map_variant: str = DEFAULT_MAP_VARIANT,
) -> Path:
    return input_dir / f"{file_prefix}_{factor_tag}_factor-score_{map_variant}.nii.gz"


def neuromaps_correlation_csv_path(
    output_dir: Path,
    factor_tag: str,
    *,
    map_variant: str = DEFAULT_MAP_VARIANT,
    cohort_tag: str = COHORT_TAG,
) -> Path:
    return (
        output_dir
        / "csv"
        / f"{factor_tag}_factor-score_{map_variant}_neuromaps_correlations_cohort-{cohort_tag}.csv"
    )


def _null_cache_path(
    output_dir: Path,
    factor_tag: str,
    space: str,
    *,
    map_variant: str,
    n_perm: int,
) -> Path:
    model = "burt2020" if space == NEUROMAPS_SPACE_MNI else "alexander_bloch"
    return (
        neuromaps_nulls_dir(output_dir)
        / f"{factor_tag}_factor-score_{map_variant}_{space}_{model}_n{n_perm}.npz"
    )


def prepare_factor_score_mni(
    map_path: Path,
    *,
    analysis_mask_path: Path = DEFAULT_MASK_NII,
    density: str = NEUROMAPS_MNI_DENSITY,
) -> tuple[nib.Nifti1Image, np.ndarray]:
    from neuromaps import transforms

    from gradient_lib.neuromaps_correlations import _mask_nifti

    grad = transforms.mni152_to_mni152(str(map_path), density)
    mask_img = nib.load(str(analysis_mask_path))
    from nibabel.processing import resample_from_to

    mask_resampled = resample_from_to(mask_img, (grad.shape, grad.affine), order=0)
    mask = np.asarray(mask_resampled.get_fdata()) > 0
    return _mask_nifti(grad, mask), mask


def prepare_factor_score_fslr(
    map_path: Path,
    *,
    density: str = NEUROMAPS_FSLR_DENSITY,
) -> tuple[nib.GiftiImage, nib.GiftiImage]:
    from neuromaps import transforms

    return transforms.mni152_to_fslr(str(map_path), fslr_density=density, method="linear")


def load_or_compute_nulls_mni(
    map_mni: nib.Nifti1Image,
    *,
    output_dir: Path,
    factor_tag: str,
    map_variant: str,
    n_perm: int,
    seed: int = NEUROMAPS_NULL_SEED,
    skip_nulls: bool = False,
    pbar: tqdm | None = None,
    show_progress: bool = True,
) -> np.ndarray | None:
    if skip_nulls or n_perm <= 0:
        return None

    cache_path = _null_cache_path(
        output_dir, factor_tag, NEUROMAPS_SPACE_MNI, map_variant=map_variant, n_perm=n_perm
    )
    if cache_path.is_file():
        msg = f"[{factor_tag}] MNI burt2020 nulls: loading cache (n_perm={n_perm})"
        _progress_write(msg, show_progress=show_progress)
        logger.info("Loading cached MNI nulls: %s", cache_path)
        if pbar is not None:
            pbar.set_postfix_str(f"MNI nulls cached n_perm={n_perm}", refresh=True)
        return np.load(cache_path)["nulls"]

    from neuromaps import nulls

    label = f"MNI burt2020 nulls n_perm={n_perm}"
    _progress_write(
        f"[{factor_tag}] Computing {label} (this is often the slowest step)...",
        show_progress=show_progress,
    )
    if pbar is not None:
        pbar.set_postfix_str(label, refresh=True)

    def _compute() -> np.ndarray:
        t0 = time.time()
        null_arr = nulls.burt2020(
            map_mni,
            atlas=NEUROMAPS_SPACE_MNI,
            density=NEUROMAPS_MNI_DENSITY,
            n_perm=n_perm,
            seed=seed,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, nulls=null_arr)
        _progress_write(
            f"[{factor_tag}] {label} finished in {time.time() - t0:.0f}s",
            show_progress=show_progress,
        )
        return null_arr

    if pbar is not None:
        return _run_with_elapsed_postfix(pbar, label, _compute)

    t0 = time.time()
    null_arr = _compute()
    logger.info("Cached MNI nulls (%s) in %.1fs", null_arr.shape, time.time() - t0)
    return null_arr


def load_or_compute_nulls_fslr(
    map_fslr: tuple[nib.GiftiImage, nib.GiftiImage],
    *,
    output_dir: Path,
    factor_tag: str,
    map_variant: str,
    n_perm: int,
    seed: int = NEUROMAPS_NULL_SEED,
    skip_nulls: bool = False,
    pbar: tqdm | None = None,
    show_progress: bool = True,
) -> np.ndarray | None:
    if skip_nulls or n_perm <= 0:
        return None

    cache_path = _null_cache_path(
        output_dir, factor_tag, NEUROMAPS_SPACE_FSLR, map_variant=map_variant, n_perm=n_perm
    )
    if cache_path.is_file():
        msg = f"[{factor_tag}] fsLR alexander_bloch nulls: loading cache (n_perm={n_perm})"
        _progress_write(msg, show_progress=show_progress)
        logger.info("Loading cached fsLR nulls: %s", cache_path)
        if pbar is not None:
            pbar.set_postfix_str(f"fsLR nulls cached n_perm={n_perm}", refresh=True)
        return np.load(cache_path)["nulls"]

    from neuromaps import nulls

    label = f"fsLR alexander_bloch nulls n_perm={n_perm}"
    _progress_write(
        f"[{factor_tag}] Computing {label} (spin permutations; can take several minutes)...",
        show_progress=show_progress,
    )
    if pbar is not None:
        pbar.set_postfix_str(label, refresh=True)

    def _compute() -> np.ndarray:
        t0 = time.time()
        null_arr = nulls.alexander_bloch(
            map_fslr,
            atlas=NEUROMAPS_SPACE_FSLR,
            density=NEUROMAPS_FSLR_DENSITY,
            n_perm=n_perm,
            seed=seed,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, nulls=null_arr)
        _progress_write(
            f"[{factor_tag}] {label} finished in {time.time() - t0:.0f}s",
            show_progress=show_progress,
        )
        return null_arr

    if pbar is not None:
        return _run_with_elapsed_postfix(pbar, label, _compute)

    t0 = time.time()
    null_arr = _compute()
    logger.info("Cached fsLR nulls (%s) in %.1fs", null_arr.shape, time.time() - t0)
    return null_arr


def screen_factor_score(
    input_dir: Path,
    output_dir: Path,
    factor_tag: str,
    *,
    file_prefix: str = FILE_PREFIX,
    map_variant: str = DEFAULT_MAP_VARIANT,
    mask_nii: Path = DEFAULT_MASK_NII,
    spaces: tuple[str, ...] | str = (NEUROMAPS_SPACE_MNI, NEUROMAPS_SPACE_FSLR),
    n_perm_mni: int = NEUROMAPS_DEFAULT_N_PERM_MNI,
    n_perm_fslr: int = NEUROMAPS_DEFAULT_N_PERM_FSLR,
    skip_nulls: bool = False,
    max_annotations: int | None = None,
    cohort_tag: str = COHORT_TAG,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Correlate one factor score map against neuromaps MNI and/or fsLR annotation pools."""
    if isinstance(spaces, str):
        spaces = parse_neuromaps_spaces(spaces)
    run_mni = NEUROMAPS_SPACE_MNI in spaces
    run_fslr = NEUROMAPS_SPACE_FSLR in spaces
    if not run_mni and not run_fslr:
        raise ValueError(f"No neuromaps spaces selected: {spaces!r}")

    setup_neuromaps_data_dir(output_dir)
    map_path = factor_score_nii_path(
        input_dir, factor_tag, file_prefix=file_prefix, map_variant=map_variant
    )
    if not map_path.is_file():
        raise FileNotFoundError(f"Missing factor score map: {map_path}")

    metadata = load_annotation_metadata()
    mni_entries = _annotation_entries_mni(metadata) if run_mni else []
    fslr_entries = _annotation_entries_fslr(metadata) if run_fslr else []
    if max_annotations is not None:
        mni_entries = mni_entries[:max_annotations]
        fslr_entries = fslr_entries[:max_annotations]

    null_steps = 0 if skip_nulls else int(run_mni) + int(run_fslr)
    total_steps = 1 + null_steps + len(mni_entries) + len(fslr_entries)
    desc = f"{factor_tag} ({map_variant})"

    _progress_write(f"[{factor_tag}] Map: {map_path.name}", show_progress=show_progress)
    pbar = tqdm(total=total_steps, desc=desc, unit="step", disable=not show_progress, leave=True)

    map_mni = mask = None
    map_fslr = None
    if run_mni and run_fslr:
        pbar.set_postfix_str("resample map to MNI/fsLR", refresh=True)
    elif run_mni:
        pbar.set_postfix_str("resample map to MNI", refresh=True)
    else:
        pbar.set_postfix_str("resample map to fsLR", refresh=True)

    if run_mni:
        map_mni, mask = prepare_factor_score_mni(map_path, analysis_mask_path=mask_nii)
    if run_fslr:
        map_fslr = prepare_factor_score_fslr(map_path)
    pbar.update(1)

    nulls_mni = None
    if run_mni:
        nulls_mni = load_or_compute_nulls_mni(
            map_mni,
            output_dir=output_dir,
            factor_tag=factor_tag,
            map_variant=map_variant,
            n_perm=n_perm_mni,
            skip_nulls=skip_nulls,
            pbar=pbar,
            show_progress=show_progress,
        )
        if not skip_nulls:
            pbar.update(1)

    nulls_fslr = None
    if run_fslr:
        nulls_fslr = load_or_compute_nulls_fslr(
            map_fslr,
            output_dir=output_dir,
            factor_tag=factor_tag,
            map_variant=map_variant,
            n_perm=n_perm_fslr,
            skip_nulls=skip_nulls,
            pbar=pbar,
            show_progress=show_progress,
        )
        if not skip_nulls:
            pbar.update(1)

    if skip_nulls and show_progress:
        _progress_write(f"[{factor_tag}] Spatial nulls skipped (--skip-nulls)", show_progress=True)

    rows: list[dict[str, object]] = []

    for entry in mni_entries:
        ann_label = _truncate_label(entry.annotation_key)
        pbar.set_postfix_str(f"MNI annotate: {ann_label}", refresh=True)
        ann = _prepare_annotation_mni(entry, map_mni, mask)
        if ann is None:
            pbar.update(1)
            continue
        try:
            r, p = _compare(map_mni, ann, nulls=nulls_mni)
        except Exception as exc:
            logger.warning("Skip MNI compare %s: %s", entry.annotation_key, exc)
            pbar.update(1)
            continue
        rows.append(
            {
                "factor": factor_tag,
                "map_variant": map_variant,
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
        pbar.update(1)

    for entry in fslr_entries:
        ann_label = _truncate_label(entry.annotation_key)
        pbar.set_postfix_str(f"fsLR annotate: {ann_label}", refresh=True)
        prepared = _prepare_annotation_fslr(entry, map_fslr)
        if prepared is None:
            pbar.update(1)
            continue
        ann, aligned_map = prepared
        try:
            r, p = _compare(aligned_map, ann, nulls=nulls_fslr)
        except Exception as exc:
            logger.warning("Skip fsLR compare %s: %s", entry.annotation_key, exc)
            pbar.update(1)
            continue
        rows.append(
            {
                "factor": factor_tag,
                "map_variant": map_variant,
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
        pbar.update(1)

    pbar.set_postfix_str("done", refresh=True)
    pbar.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["rank_abs_r"] = (
        df.groupby(["factor", "space"], sort=False)["abs_r"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return df.sort_values(["space", "rank_abs_r"]).reset_index(drop=True)


def run_all_correlations(
    input_dir: Path,
    output_dir: Path,
    *,
    factors: list[str],
    file_prefix: str = FILE_PREFIX,
    map_variant: str = DEFAULT_MAP_VARIANT,
    mask_nii: Path = DEFAULT_MASK_NII,
    spaces: tuple[str, ...] | str = (NEUROMAPS_SPACE_MNI, NEUROMAPS_SPACE_FSLR),
    n_perm_mni: int = NEUROMAPS_DEFAULT_N_PERM_MNI,
    n_perm_fslr: int = NEUROMAPS_DEFAULT_N_PERM_FSLR,
    skip_nulls: bool = False,
    max_annotations: int | None = None,
    cohort_tag: str = COHORT_TAG,
    show_progress: bool = True,
) -> list[Path]:
    if isinstance(spaces, str):
        spaces = parse_neuromaps_spaces(spaces)
    run_mni = NEUROMAPS_SPACE_MNI in spaces
    run_fslr = NEUROMAPS_SPACE_FSLR in spaces

    paths: list[Path] = []
    n_factors = len(factors)
    plan_parts = [f"{n_factors} factor map(s)"]
    if run_mni:
        plan_parts.append(f"MNI burt2020 n_perm={n_perm_mni}")
    if run_fslr:
        plan_parts.append(f"fsLR alexander_bloch n_perm={n_perm_fslr}")
    plan_parts.append(f"skip_nulls={skip_nulls}")
    _progress_write(
        "Neuromaps screening plan: " + "; ".join(plan_parts),
        show_progress=show_progress,
    )

    for index, factor in enumerate(factors, start=1):
        _progress_write(
            f"\n=== Factor {index}/{n_factors}: {factor} ===",
            show_progress=show_progress,
        )
        logger.info("Screening neuromaps annotations: %s (%s)", factor, map_variant)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = screen_factor_score(
                input_dir,
                output_dir,
                factor,
                file_prefix=file_prefix,
                map_variant=map_variant,
                mask_nii=mask_nii,
                spaces=spaces,
                n_perm_mni=n_perm_mni,
                n_perm_fslr=n_perm_fslr,
                skip_nulls=skip_nulls,
                max_annotations=max_annotations,
                cohort_tag=cohort_tag,
                show_progress=show_progress,
            )
        out = neuromaps_correlation_csv_path(
            output_dir, factor, map_variant=map_variant, cohort_tag=cohort_tag
        )
        save_correlation_csv(df, out)
        paths.append(out)
        _progress_write(f"[{factor}] saved {out.name} ({len(df)} rows)", show_progress=show_progress)
        logger.info("  saved %s (%d rows)", out, len(df))
    return paths
