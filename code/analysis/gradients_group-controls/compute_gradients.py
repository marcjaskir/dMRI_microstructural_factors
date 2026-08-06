#!/usr/bin/env python3
"""
Controls-only group-level BrainSpace gradient pipeline.

Manuscript default is **Laplacian eigenmaps** (LE). Diffusion-map (DM) remains available
via ``--method diffusion_embedding`` for exploratory comparisons but is not used in the paper.

For each method, the unthresholded Pearson affinity is built across the controls wide table
per factor, then BrainSpace gradients (``LaplacianEigenmaps`` for LE, or ``GradientMaps`` with
``approach='dm'`` for DM) are computed. Outputs are grouped under
``gradients-{K}/`` subdirectories where ``K`` is the gradient subspace size:

* ``gradients-2/`` — CSVs for G1, G2 + 2D scatter figures and ``legend-*.png`` files (G1 colorbar;
  tissue, Yeo, Mesulam encodings). When K ≥ 2, also ``gradient2_summary_cohort-controls.png``.
* ``gradients-3/`` — CSVs for G1, G2, G3 + 3D scatter figures and the same standalone legends
  (with 3D-consistent G1 scaling where applicable).

Default ``--output-dir`` is ``$GRADIENTS_CONTROLS_OUTPUT_DIR`` or
``$STRUCTURAL_TRACTOMETRY_ROOT/derivatives/analysis/gradients_group-controls``.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Literal

from gradient_lib.figure_style import apply_figure_font_rcparams

from gradient_lib.config import (
    ALPHA_DEFAULT,
    DEFAULT_FACTOR_SCORES_DIR,
    DEFAULT_GRADIENTS_DIR,
    DEFAULT_TRACTOMETRY_ROOT,
    GRADIENT_SUBSPACE_CHOICES,
    diffusion_embedding_dirs,
    laplacian_eigenmodes_dirs,
)
from gradient_lib.io import parse_factor_files
from gradient_lib.plots_bars import plot_gradient_by_groups_axes_bars
from gradient_lib.plots_scatter import (
    plot_gradient1_summary,
    plot_gradient_summary,
    plot_gradient_by_tissue_scatter,
    plot_gradient_by_yeo_mesulam_scatter,
    plot_gradients_by_gradient1_scatter,
    save_standalone_legend_gradient1,
    save_standalone_legend_mesulam,
    save_standalone_legend_tissue,
    save_standalone_legend_yeo,
)
from gradient_lib.run_diffusion import (
    compute_diffusion_embedding_row,
    save_diffusion_gradient_outputs,
)
from gradient_lib.neuroaxis_correlations import save_neuroaxis_correlations_csv
from gradient_lib.run_laplacian import (
    compute_laplacian_row,
    save_laplacian_gradient_outputs,
)
from gradient_lib.types import GradientRunRow

MethodName = Literal["diffusion_embedding", "laplacian_eigenmodes", "both"]


def _parse_gradient_k_choices(arg: str) -> tuple[int, ...]:
    """Parse ``--gradients-k`` (comma-separated K's) into a deduplicated tuple."""
    raw = (arg or "").strip()
    if not raw:
        return GRADIENT_SUBSPACE_CHOICES
    parts = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not parts:
        return GRADIENT_SUBSPACE_CHOICES
    ordered_unique: list[int] = []
    for k in parts:
        if k < 1:
            raise SystemExit(f"Invalid --gradients-k: {k} (need K >= 1).")
        if k not in ordered_unique:
            ordered_unique.append(k)
    return tuple(ordered_unique)


def _figure_outputs_for_k(
    rows: list[GradientRunRow],
    *,
    method_tag: str,
    figures_dir: Path,
    tractometry_root: Path,
    k: int,
) -> None:
    """Emit scatter figures plus standalone legend PNGs in the same ``gradients-{K}/`` folder."""
    dims: Literal[2, 3] = 2 if k == 2 else 3
    dim_suffix = "" if dims == 2 else "_3D"
    bt = figures_dir / f"gradients_by-gradient1_cohort-controls{dim_suffix}.png"
    plot_gradients_by_gradient1_scatter(
        rows,
        bt,
        dims=dims,
        tractometry_root=tractometry_root,
        method_tag=method_tag,
        cohort_tag="controls",
    )
    print(f"  saved {bt}")
    tissue_p = figures_dir / f"gradients_by-tissue_cohort-controls{dim_suffix}.png"
    plot_gradient_by_tissue_scatter(
        rows,
        tissue_p,
        dims=dims,
        tractometry_root=tractometry_root,
        method_tag=method_tag,
        cohort_tag="controls",
    )
    print(f"  saved {tissue_p}")
    ym_p = figures_dir / f"gradients_by-yeo-mesulam_cohort-controls{dim_suffix}.png"
    plot_gradient_by_yeo_mesulam_scatter(
        rows,
        ym_p,
        dims=dims,
        tractometry_root=tractometry_root,
        method_tag=method_tag,
        cohort_tag="controls",
    )
    print(f"  saved {ym_p}")
    g1_b = figures_dir / "gradient1_by-groups-axes_cohort-controls.png"
    plot_gradient_by_groups_axes_bars(
        rows,
        g1_b,
        gradient_index=0,
        tractometry_root=tractometry_root,
        method_tag=method_tag,
        cohort_tag="controls",
    )
    print(f"  saved {g1_b}")
    sum_p = figures_dir / f"gradient1_summary_cohort-controls{dim_suffix}.png"
    plot_gradient1_summary(
        rows,
        sum_p,
        dims=dims,
        tractometry_root=tractometry_root,
        method_tag=method_tag,
        cohort_tag="controls",
    )
    print(f"  saved {sum_p}")
    if k >= 2:
        g2_b = figures_dir / "gradient2_by-groups-axes_cohort-controls.png"
        plot_gradient_by_groups_axes_bars(
            rows,
            g2_b,
            gradient_index=1,
            tractometry_root=tractometry_root,
            method_tag=method_tag,
            cohort_tag="controls",
        )
        print(f"  saved {g2_b}")
        sum_g2_p = figures_dir / f"gradient2_summary_cohort-controls{dim_suffix}.png"
        plot_gradient_summary(
            rows,
            sum_g2_p,
            gradient_index=1,
            dims=dims,
            tractometry_root=tractometry_root,
            method_tag=method_tag,
            cohort_tag="controls",
        )
        print(f"  saved {sum_g2_p}")
    lg = figures_dir / "legend-gradient1.png"
    save_standalone_legend_gradient1(rows, lg, k_dims=int(dims))
    print(f"  saved {lg}")
    lt = figures_dir / "legend-tissue.png"
    save_standalone_legend_tissue(lt)
    print(f"  saved {lt}")
    ly = figures_dir / "legend-yeo.png"
    save_standalone_legend_yeo(rows, ly, tractometry_root=tractometry_root, k_dims=int(dims))
    print(f"  saved {ly}")
    lm = figures_dir / "legend-mesulam.png"
    save_standalone_legend_mesulam(rows, lm, tractometry_root=tractometry_root, k_dims=int(dims))
    print(f"  saved {lm}")


def run_diffusion_pipeline(
    *,
    tractometry_root: Path,
    factor_pairs: list[tuple[str, Path]],
    figures_dir: Path,
    csv_dir: Path,
    gradient_ks: tuple[int, ...],
) -> None:
    rows: list[GradientRunRow] = []
    for factor_tag, csv_path in factor_pairs:
        print(f"  [DM] factor={factor_tag} alpha={ALPHA_DEFAULT:.2f}")
        row = compute_diffusion_embedding_row(
            factor_tag,
            csv_path,
            tractometry_root=tractometry_root,
            alpha=ALPHA_DEFAULT,
            cohort_tag="controls",
        )
        rows.append(row)

    for k in gradient_ks:
        sub_csv = csv_dir / f"gradients-{k}"
        sub_fig = figures_dir / f"gradients-{k}"
        sub_csv.mkdir(parents=True, exist_ok=True)
        sub_fig.mkdir(parents=True, exist_ok=True)
        for row in rows:
            factor_tag, mean_per_roi, grads, _ = row
            save_diffusion_gradient_outputs(
                factor_tag,
                grads,
                mean_per_roi,
                cohort_tag="controls",
                alpha=ALPHA_DEFAULT,
                out_dir=sub_csv,
                n_gradients_to_save=k,
            )
        _figure_outputs_for_k(
            rows,
            method_tag="diffusion_embedding",
            figures_dir=sub_fig,
            tractometry_root=tractometry_root,
            k=k,
        )

    nax_path = save_neuroaxis_correlations_csv(
        rows,
        csv_dir / "neuroaxis_correlations_cohort-controls.csv",
        tractometry_root=tractometry_root,
        cohort_tag="controls",
        n_gradients=max(gradient_ks),
    )
    print(f"  saved {nax_path}")


def run_laplacian_pipeline(
    *,
    tractometry_root: Path,
    factor_pairs: list[tuple[str, Path]],
    figures_dir: Path,
    csv_dir: Path,
    gradient_ks: tuple[int, ...],
) -> None:
    rows: list[GradientRunRow] = []
    for factor_tag, csv_path in factor_pairs:
        print(f"  [LE] factor={factor_tag}")
        row = compute_laplacian_row(
            factor_tag,
            csv_path,
            tractometry_root=tractometry_root,
            cohort_tag="controls",
        )
        rows.append(row)

    for k in gradient_ks:
        sub_csv = csv_dir / f"gradients-{k}"
        sub_fig = figures_dir / f"gradients-{k}"
        sub_csv.mkdir(parents=True, exist_ok=True)
        sub_fig.mkdir(parents=True, exist_ok=True)
        for row in rows:
            factor_tag, mean_per_roi, grads, _ = row
            save_laplacian_gradient_outputs(
                factor_tag,
                grads,
                mean_per_roi,
                "controls",
                sub_csv,
                n_gradients_to_save=k,
            )
        _figure_outputs_for_k(
            rows,
            method_tag="laplacian_eigenmodes",
            figures_dir=sub_fig,
            tractometry_root=tractometry_root,
            k=k,
        )

    nax_path = save_neuroaxis_correlations_csv(
        rows,
        csv_dir / "neuroaxis_correlations_cohort-controls.csv",
        tractometry_root=tractometry_root,
        cohort_tag="controls",
        n_gradients=max(gradient_ks),
    )
    print(f"  saved {nax_path}")


def main() -> None:
    epilog = textwrap.dedent(
        """
        Examples:
          {out}/diffusion_embedding/csv/gradients-2/F{n}_principal_gradient{1,2}_scores_cohort-controls_alpha-0p50.csv
          {out}/diffusion_embedding/csv/gradients-3/F{n}_principal_gradient{1,2,3}_scores_cohort-controls_alpha-0p50.csv
          {out}/diffusion_embedding/figures/gradients-2/gradients_by-gradient1_cohort-controls.png
          {out}/diffusion_embedding/figures/gradients-3/gradients_by-gradient1_cohort-controls_3D.png
        Use --gradients-k 2 or --gradients-k 3 to skip the other subspace.
        """
    ).strip()
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument(
        "--method",
        choices=["diffusion_embedding", "laplacian_eigenmodes", "both"],
        default="laplacian_eigenmodes",
        help=(
            "Which pipeline(s) to run. Default: laplacian_eigenmodes (manuscript). "
            "Use diffusion_embedding or both only for exploratory DM comparisons."
        ),
    )
    p.add_argument("--factor-dir", type=Path, default=DEFAULT_FACTOR_SCORES_DIR)
    p.add_argument("--tractometry-root", type=Path, default=DEFAULT_TRACTOMETRY_ROOT)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GRADIENTS_DIR,
        help=(
            "Gradients output root (default: $GRADIENTS_CONTROLS_OUTPUT_DIR or "
            ".../derivatives/analysis/gradients_group-controls)."
        ),
    )
    p.add_argument(
        "--factors",
        type=str,
        default="",
        help="Comma-separated factor indices (e.g. 1,2). Default: all.",
    )
    p.add_argument(
        "--gradients-k",
        type=str,
        default="",
        dest="gradients_k",
        metavar="K",
        help=(
            "Comma-separated gradient subspace sizes (e.g. 2 or 2,3). "
            f"Default: {','.join(str(k) for k in GRADIENT_SUBSPACE_CHOICES)}. "
            "Outputs are written under gradients-<K>/."
        ),
    )
    args = p.parse_args()

    apply_figure_font_rcparams()

    gradient_ks = _parse_gradient_k_choices(args.gradients_k)
    print(f"Gradient K subspaces (output subdirs): {list(gradient_ks)}")

    factor_dir = args.factor_dir.expanduser()
    if not factor_dir.is_dir():
        raise SystemExit(f"Not a directory: {factor_dir}")
    tractometry_root = args.tractometry_root.expanduser()
    if not tractometry_root.is_dir():
        raise SystemExit(f"Not a directory: {tractometry_root}")
    out_root = args.output_dir.expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    factors_filter: set[int] | None = None
    if args.factors.strip():
        factors_filter = {int(x.strip()) for x in args.factors.split(",") if x.strip()}

    factor_pairs = parse_factor_files(factor_dir, factors_filter)
    if not factor_pairs:
        raise SystemExit(f"No controls_F*_scores.csv under {factor_dir}")

    method: MethodName = args.method  # type: ignore[assignment]
    if method in ("diffusion_embedding", "both"):
        csv_dir, fig_dir = diffusion_embedding_dirs(out_root)
        for d in (csv_dir, fig_dir):
            d.mkdir(parents=True, exist_ok=True)
        print(f"--- diffusion_embedding -> {csv_dir} {fig_dir}")
        run_diffusion_pipeline(
            tractometry_root=tractometry_root,
            factor_pairs=factor_pairs,
            figures_dir=fig_dir,
            csv_dir=csv_dir,
            gradient_ks=gradient_ks,
        )

    if method in ("laplacian_eigenmodes", "both"):
        csv_dir, fig_dir = laplacian_eigenmodes_dirs(out_root)
        for d in (csv_dir, fig_dir):
            d.mkdir(parents=True, exist_ok=True)
        print(f"--- laplacian_eigenmodes -> {csv_dir} {fig_dir}")
        run_laplacian_pipeline(
            tractometry_root=tractometry_root,
            factor_pairs=factor_pairs,
            figures_dir=fig_dir,
            csv_dir=csv_dir,
            gradient_ks=gradient_ks,
        )


if __name__ == "__main__":
    main()
