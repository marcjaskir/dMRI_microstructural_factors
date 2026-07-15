#!/usr/bin/env python3
"""Rebuild atlas-encoded gradient figures from saved voxelwise outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gradient_lib.config import COHORT_TAG
from gradient_lib.embed_run import (
    EmbedRunParams,
    add_embed_run_arguments,
    embed_run_params_from_args,
)
from gradient_lib.emit_figures import emit_all_figures, emit_neuromaps_figures_if_available
from gradient_lib.factor_loading_source import (
    FactorLoadingContext,
    add_factor_loading_arguments,
    contexts_from_args,
)
from gradient_lib.load_outputs import load_voxel_rows_from_output
from gradient_lib.neuroaxis_voxelwise import save_neuroaxis_correlations_csv


def _parse_factors(arg: str | None) -> list[str] | None:
    if not arg or not arg.strip():
        return None
    factors: list[str] = []
    for part in arg.split(","):
        part = part.strip()
        if part:
            factors.append(part if part.upper().startswith("F") else f"F{part}")
    return factors


def _run_for_context(
    ctx: FactorLoadingContext,
    *,
    embed_params: EmbedRunParams,
    factors: list[str] | None,
    neuromaps_figures: bool,
) -> None:
    label = ctx.run_label
    gradient_dir = embed_params.gradient_run_dir(ctx.output_dir)
    print(f"\n=== Regenerate figures ({label}) ===")
    print(f"  embed run: {embed_params.subdir_name} -> {gradient_dir}")
    print(f"  loadings source: {ctx.source} ({ctx.loadings_csv})")

    rows = load_voxel_rows_from_output(
        gradient_dir, cohort_tag=COHORT_TAG, factors=factors, mask_nii=ctx.mask_nii
    )
    if not rows:
        raise SystemExit(f"No gradient rows found under {gradient_dir / 'csv'}")

    print(f"Loaded {len(rows)} factor row(s): {[r[0] for r in rows]}")
    csv_dir = gradient_dir / "csv"
    nax_path = save_neuroaxis_correlations_csv(
        rows,
        csv_dir / f"neuroaxis_correlations_cohort-{COHORT_TAG}.csv",
        cohort_tag=COHORT_TAG,
        n_gradients=2,
    )
    print(f"  saved {nax_path}")

    emit_all_figures(
        rows,
        gradient_dir / "figures",
        gradient_dir,
        mask_nii=ctx.mask_nii,
        csf_mode=ctx.csf_mode,
    )

    if neuromaps_figures:
        factor_tags = [r[0] for r in rows]
        emit_neuromaps_figures_if_available(
            gradient_dir,
            gradient_dir / "figures",
            factors=factor_tags,
            rows=rows,
            cache_dir=gradient_dir / "_cache",
            mask_nii=ctx.mask_nii,
            csf_mode=ctx.csf_mode,
        )

    print(f"Done ({label}). Figures under {gradient_dir / 'figures'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate atlas-based figures from saved gradients_voxelwise outputs."
    )
    add_factor_loading_arguments(parser)
    add_embed_run_arguments(parser)
    parser.add_argument("--factors", type=str, default=None)
    parser.add_argument(
        "--neuromaps-figures",
        action="store_true",
        help="Also replot neuromaps lollipop figures from saved CSVs (if present).",
    )
    args = parser.parse_args()

    embed_params = embed_run_params_from_args(args)
    contexts = contexts_from_args(args)
    factors = _parse_factors(args.factors)

    for ctx in contexts:
        _run_for_context(
            ctx,
            embed_params=embed_params,
            factors=factors,
            neuromaps_figures=args.neuromaps_figures,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
