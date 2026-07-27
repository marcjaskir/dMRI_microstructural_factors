#!/usr/bin/env python3
"""Compute per-subject voxelwise factor score maps from FA loadings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gradient_lib.epilepsy_manifest import build_epilepsy_manifest
from gradient_lib.factor_loading_source import (
    FactorLoadingContext,
    add_factor_loading_arguments,
    contexts_from_args,
    write_loadings_source_metadata,
)
from gradient_lib.factor_scores import (
    apply_iqr_if_requested,
    compute_factor_scores_from_z,
    zscore_scalars_per_subject,
)
from gradient_lib.io_voxelwise import (
    check_duplicate_subs,
    factor_score_nii_path,
    load_analysis_mask,
    load_factor_loadings,
    load_manifest,
    load_masked_scalar,
    save_masked_vector_nii,
    scalar_labels_from_loadings,
)


def _parse_factors(arg: str | None, loadings_index: list[str]) -> list[str]:
    if not arg or not arg.strip():
        return [str(x) for x in loadings_index]
    out: list[str] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        tag = part if part.upper().startswith("F") else f"F{part}"
        if tag not in loadings_index:
            raise SystemExit(f"Unknown factor {tag!r}; available: {loadings_index}")
        if tag not in out:
            out.append(tag)
    return out


def _parse_subjects(arg: str | None) -> set[str] | None:
    if not arg or not arg.strip():
        return None
    return {s.strip() for s in arg.split(",") if s.strip()}


def _run_manifest(
    manifest,
    *,
    label: str,
    loadings,
    scalar_labels: list[str],
    factors: list[str],
    subject_filter: set[str] | None,
    use_iqr: bool,
    skip_existing: bool,
    mask_img,
    mask,
    output_dir: Path,
) -> None:
    check_duplicate_subs(manifest)
    subjects = manifest.groupby("subject", sort=False)
    subject_ids = [s for s in subjects.groups if subject_filter is None or s in subject_filter]

    for subject_id in tqdm(subject_ids, desc=f"Subjects ({label})"):
        subject_df = manifest[manifest["subject"] == subject_id]
        row0 = subject_df.iloc[0]
        group = str(row0["group"])
        sub = str(row0["sub"])

        if skip_existing and all(
            factor_score_nii_path(output_dir, group, sub, f).is_file() for f in factors
        ):
            continue

        paths = subject_df.set_index("scalar")["path"].to_dict()
        columns = []
        for scalar in scalar_labels:
            columns.append(load_masked_scalar(paths[scalar], mask_img, mask))
        subject_matrix = np.column_stack(columns)
        subject_matrix = apply_iqr_if_requested(subject_matrix, use_iqr=use_iqr)
        z_matrix = zscore_scalars_per_subject(subject_matrix)
        scores = compute_factor_scores_from_z(z_matrix, loadings, scalar_labels, factors)

        for factor_tag, values in scores.items():
            out_path = factor_score_nii_path(output_dir, group, sub, factor_tag)
            if skip_existing and out_path.is_file():
                continue
            save_masked_vector_nii(values, mask_img, mask, out_path)


def _run_for_context(
    ctx: FactorLoadingContext,
    *,
    factors: list[str],
    subject_filter: set[str] | None,
    use_iqr: bool,
    skip_existing: bool,
) -> None:
    label = ctx.run_label
    print(f"\n=== Factor scores ({label}) ===")
    meta_path = write_loadings_source_metadata(ctx)
    print(f"  loadings source: {ctx.source} ({ctx.loadings_csv})")
    print(f"  saved {meta_path}")

    loadings = load_factor_loadings(ctx.loadings_csv)
    scalar_labels = scalar_labels_from_loadings(loadings)
    manifest = load_manifest(ctx.manifest_csv)

    mask_img, mask = load_analysis_mask(ctx.mask_nii)
    output_dir = ctx.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _run_manifest(
        manifest,
        label=label,
        loadings=loadings,
        scalar_labels=scalar_labels,
        factors=factors,
        subject_filter=subject_filter,
        use_iqr=use_iqr,
        skip_existing=skip_existing,
        mask_img=mask_img,
        mask=mask,
        output_dir=output_dir,
    )

    print("Building penn_epilepsy manifest for factor score projection...")
    epilepsy_manifest = build_epilepsy_manifest(scalar_labels)
    if epilepsy_manifest.empty:
        print("  No complete penn_epilepsy subjects found; skipping.")
    else:
        n_epi = epilepsy_manifest["subject"].nunique()
        print(f"  {n_epi} penn_epilepsy subject(s) with complete scalars")
        _run_manifest(
            epilepsy_manifest,
            label=f"{label}, penn_epilepsy",
            loadings=loadings,
            scalar_labels=scalar_labels,
            factors=factors,
            subject_filter=subject_filter,
            use_iqr=use_iqr,
            skip_existing=skip_existing,
            mask_img=mask_img,
            mask=mask,
            output_dir=output_dir,
        )

    print(f"Done ({label}). Factor score NIfTIs under {output_dir / 'factor_score_nii'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Voxelwise per-subject factor score maps from qsirecon scalars."
    )
    add_factor_loading_arguments(parser)
    parser.add_argument("--factors", type=str, default=None, help="e.g. 1,2,3,4")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs")
    parser.add_argument(
        "--iqr-outliers",
        action="store_true",
        help="Apply 1.5×IQR outlier filter per scalar (default: off).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subject/factor outputs that already exist.",
    )
    args = parser.parse_args()

    contexts = contexts_from_args(args)
    subject_filter = _parse_subjects(args.subjects)

    for ctx in contexts:
        loadings = load_factor_loadings(ctx.loadings_csv)
        factors = _parse_factors(args.factors, list(loadings.index))
        _run_for_context(
            ctx,
            factors=factors,
            subject_filter=subject_filter,
            use_iqr=args.iqr_outliers,
            skip_existing=args.skip_existing,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
