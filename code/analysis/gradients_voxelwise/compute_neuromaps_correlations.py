#!/usr/bin/env python3
"""Screen factor gradient maps against neuromaps reference annotations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gradient_lib.config import (
    COHORT_TAG,
    NEUROMAPS_DEFAULT_N_PERM_FSLR,
    NEUROMAPS_DEFAULT_N_PERM_MNI,
    NEUROMAPS_TOP_K,
)
from gradient_lib.embed_run import (
    EmbedRunParams,
    add_embed_run_arguments,
    embed_run_params_from_args,
)
from gradient_lib.factor_loading_source import (
    FactorLoadingContext,
    add_factor_loading_arguments,
    contexts_from_args,
    write_loadings_source_metadata,
)
from gradient_lib.io_voxelwise import load_factor_loadings
from gradient_lib.load_outputs import load_voxel_rows_from_output
from gradient_lib.neuromaps_correlations import run_all_correlations, setup_neuromaps_data_dir
from gradient_lib.plots_neuromaps_lollipop import plot_neuromaps_figures_from_output
from gradient_lib.plots_summary_with_neuromaps import plot_summary_with_neuromaps_figures


def _parse_list(arg: str | None, *, prefix: str = "") -> list[str] | None:
    if not arg or not arg.strip():
        return None
    out: list[str] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if prefix and not part.upper().startswith(prefix.upper()):
            part = f"{prefix}{part}"
        if part not in out:
            out.append(part)
    return out


def _parse_int_list(arg: str | None) -> list[int] | None:
    if not arg or not arg.strip():
        return None
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def _run_for_context(
    ctx: FactorLoadingContext,
    *,
    embed_params: EmbedRunParams,
    factors: list[str],
    gradients: list[int],
    figures: bool,
    figures_only: bool,
    n_perm_mni: int,
    n_perm_fslr: int,
    skip_nulls: bool,
    max_annotations: int | None,
    show_progress: bool,
) -> None:
    label = ctx.run_label
    gradient_dir = embed_params.gradient_run_dir(ctx.output_dir)
    print(f"\n=== Neuromaps ({label}) ===")
    if not figures_only:
        meta_path = write_loadings_source_metadata(ctx)
        print(f"  loadings source: {ctx.source} ({ctx.loadings_csv})")
        print(f"  saved {meta_path}")
    print(f"  embed run: {embed_params.subdir_name} -> {gradient_dir}")

    setup_neuromaps_data_dir(gradient_dir)

    if not figures_only:
        run_all_correlations(
            gradient_dir,
            factors=factors,
            gradient_indices=gradients,
            n_perm_mni=n_perm_mni,
            n_perm_fslr=n_perm_fslr,
            skip_nulls=skip_nulls,
            max_annotations=max_annotations,
            cohort_tag=COHORT_TAG,
            show_progress=show_progress,
        )

    if figures or figures_only:
        paths = plot_neuromaps_figures_from_output(
            gradient_dir,
            factors=factors,
            gradient_indices=gradients,
            cohort_tag=COHORT_TAG,
        )
        for p in paths:
            print(f"  saved {p}")

        rows = load_voxel_rows_from_output(
            gradient_dir, cohort_tag=COHORT_TAG, factors=factors, mask_nii=ctx.mask_nii
        )
        if rows:
            summary_paths = plot_summary_with_neuromaps_figures(
                rows,
                gradient_dir,
                gradient_indices=gradients,
                cohort_tag=COHORT_TAG,
            )
            for p in summary_paths:
                print(f"  saved {p}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute neuromaps spatial correlations for voxelwise factor gradient NIfTIs. "
            "MNI nulls: burt2020; fsLR nulls: alexander_bloch. "
            "Set NEUROMAPS_DATA (defaults to output_dir/_cache/neuromaps/data). "
            "Requires: pip install neuromaps brainsmash"
        )
    )
    add_factor_loading_arguments(parser)
    add_embed_run_arguments(parser)
    parser.add_argument("--factors", type=str, default=None)
    parser.add_argument("--gradients", type=str, default="1,2")
    parser.add_argument("--top-k", type=int, default=NEUROMAPS_TOP_K)
    parser.add_argument("--n-perm-mni", type=int, default=NEUROMAPS_DEFAULT_N_PERM_MNI)
    parser.add_argument("--n-perm-fslr", type=int, default=NEUROMAPS_DEFAULT_N_PERM_FSLR)
    parser.add_argument("--skip-nulls", action="store_true")
    parser.add_argument("--max-annotations", type=int, default=None)
    parser.add_argument("--figures", action="store_true", default=True)
    parser.add_argument("--no-figures", dest="figures", action="store_false")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    embed_params = embed_run_params_from_args(args)
    contexts = contexts_from_args(args)

    for ctx in contexts:
        loadings = load_factor_loadings(ctx.loadings_csv)
        factors = _parse_list(args.factors, prefix="F") or [str(x) for x in loadings.index]
        gradients = _parse_int_list(args.gradients) or [1, 2]
        _run_for_context(
            ctx,
            embed_params=embed_params,
            factors=factors,
            gradients=gradients,
            figures=args.figures,
            figures_only=args.figures_only,
            n_perm_mni=args.n_perm_mni,
            n_perm_fslr=args.n_perm_fslr,
            skip_nulls=args.skip_nulls,
            max_annotations=args.max_annotations,
            show_progress=not args.no_progress,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
