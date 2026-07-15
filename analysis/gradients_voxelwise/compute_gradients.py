#!/usr/bin/env python3
"""Voxelwise Laplacian G1/G2 gradients from per-subject factor score maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Georgia", "DejaVu Serif", "serif"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.titlesize": 16,
    }
)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gradient_lib.config import COHORT_TAG, GRADIENT_GROUPS
from gradient_lib.embed_run import (
    EmbedRunParams,
    add_embed_run_arguments,
    embed_run_params_from_args,
    write_embed_run_metadata,
)
from gradient_lib.emit_figures import emit_all_figures
from gradient_lib.factor_loading_source import (
    FactorLoadingContext,
    add_factor_loading_arguments,
    contexts_from_args,
    write_loadings_source_metadata,
)
from gradient_lib.io_voxelwise import load_analysis_mask, load_factor_loadings, load_manifest
from gradient_lib.neuroaxis_voxelwise import save_neuroaxis_correlations_csv
from gradient_lib.run_laplacian_voxelwise import (
    compute_laplacian_voxel_row,
    save_voxel_laplacian_outputs,
)
from gradient_lib.types import VoxelGradientRunRow


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


def _parse_groups(arg: str | None) -> tuple[str, ...] | None:
    if not arg or not arg.strip():
        return None
    return tuple(s.strip() for s in arg.split(",") if s.strip())


def _run_for_context(
    ctx: FactorLoadingContext,
    *,
    embed_params: EmbedRunParams,
    factors: list[str],
    subject_filter: set[str] | None,
    group_filter: tuple[str, ...] | None,
    emit_figures: bool,
) -> None:
    label = ctx.run_label
    gradient_dir = embed_params.gradient_run_dir(ctx.output_dir)
    print(f"\n=== Laplacian gradients ({label}) ===")
    meta_path = write_loadings_source_metadata(ctx)
    embed_meta_path = write_embed_run_metadata(
        gradient_dir, embed_params, base_output_dir=ctx.output_dir
    )
    print(f"  loadings source: {ctx.source} ({ctx.loadings_csv})")
    print(f"  saved {meta_path}")
    print(f"  embed run: {embed_params.subdir_name} -> {gradient_dir}")
    print(f"  saved {embed_meta_path}")
    if group_filter is not None:
        print(f"  gradient groups: {', '.join(group_filter)}")

    manifest = load_manifest(ctx.manifest_csv)
    mask_img, mask = load_analysis_mask(ctx.mask_nii)
    rows: list[VoxelGradientRunRow] = []

    for factor_tag in factors:
        print(f"Computing Laplacian gradients for {factor_tag} ({label})...")
        row = compute_laplacian_voxel_row(
            factor_tag,
            manifest,
            gradient_dir,
            embed_stride=embed_params.embed_stride,
            embed_top_k=embed_params.embed_top_k,
            max_embed_voxels=embed_params.max_embed_voxels,
            subject_filter=subject_filter,
            group_filter=group_filter,
            factor_score_dir=ctx.output_dir,
            mask_nii=ctx.mask_nii,
        )
        paths, mean_path = save_voxel_laplacian_outputs(
            row, gradient_dir, mask_img, mask, cohort_tag=COHORT_TAG
        )
        for p in paths:
            print(f"  saved {p}")
        print(f"  saved {mean_path}")
        rows.append(row)

    csv_dir = gradient_dir / "csv"
    nax_path = save_neuroaxis_correlations_csv(
        rows,
        csv_dir / f"neuroaxis_correlations_cohort-{COHORT_TAG}.csv",
        cohort_tag=COHORT_TAG,
        n_gradients=2,
    )
    print(f"  saved {nax_path}")

    if emit_figures:
        emit_all_figures(
            rows,
            gradient_dir / "figures",
            gradient_dir,
            mask_nii=ctx.mask_nii,
            csf_mode=ctx.csf_mode,
        )

    print(f"Done ({label}). Outputs under {gradient_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Voxelwise Laplacian G1/G2 from subject factor score NIfTIs."
    )
    add_factor_loading_arguments(parser)
    add_embed_run_arguments(parser, top_k_alias=True)
    parser.add_argument("--factors", type=str, default=None)
    parser.add_argument("--figures", dest="figures", action="store_true", default=True)
    parser.add_argument("--no-figures", dest="figures", action="store_false")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs")
    parser.add_argument(
        "--gradient-groups",
        type=str,
        default=",".join(GRADIENT_GROUPS),
        help=(
            "Comma-separated manifest groups used for Laplacian gradient fitting "
            f"(default: {','.join(GRADIENT_GROUPS)}). "
            "penn_epilepsy factor score NIfTIs are saved separately and excluded by default."
        ),
    )
    args = parser.parse_args()

    embed_params = embed_run_params_from_args(args)
    contexts = contexts_from_args(args)
    subject_filter = None
    if args.subjects and args.subjects.strip():
        subject_filter = {s.strip() for s in args.subjects.split(",") if s.strip()}
    group_filter = _parse_groups(args.gradient_groups)

    for ctx in contexts:
        loadings = load_factor_loadings(ctx.loadings_csv)
        factors = _parse_factors(args.factors, list(loadings.index))
        _run_for_context(
            ctx,
            embed_params=embed_params,
            factors=factors,
            subject_filter=subject_filter,
            group_filter=group_filter,
            emit_figures=args.figures,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
