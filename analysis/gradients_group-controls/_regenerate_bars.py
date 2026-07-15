#!/usr/bin/env python3
"""One-off: rebuild neuroaxis-dependent gradient figures from saved CSVs.

Reconstructs ``GradientRunRow`` tuples from the per-factor CSVs already on disk
(``<method>/csv/gradients-<K>/F*_principal_gradient*_scores_*.csv`` +
``F*_factor_score_means_*.csv``) and re-emits:

* ``gradient{1,2}_by-groups-axes_cohort-controls.png``
* ``gradient{1,2}_summary_cohort-controls{,_3D}.png``

Useful after changes to bar/summary plotting or ``wholebrain_centroids.csv`` that do
not require recomputing gradients.

Run from anywhere::

    python analysis/gradients_group-controls/_regenerate_bars.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gradient_lib.config import (  # noqa: E402
    DEFAULT_GRADIENTS_DIR,
    DEFAULT_TRACTOMETRY_ROOT,
)
from gradient_lib.figure_style import apply_figure_font_rcparams  # noqa: E402
from gradient_lib.csv_outputs import write_sorted_copies_for_gradient_csv  # noqa: E402
from gradient_lib.neuroaxis_correlations import save_neuroaxis_correlations_csv  # noqa: E402
from gradient_lib.plots_bars import plot_gradient_by_groups_axes_bars  # noqa: E402
from gradient_lib.plots_scatter import (  # noqa: E402
    plot_gradient1_summary,
    plot_gradient_summary,
)
from gradient_lib.types import GradientRunRow  # noqa: E402


def _load_series(csv_path: Path, value_col: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    return pd.Series(df[value_col].to_numpy(), index=df["region"].astype(str))


def _gradient_scores_csv(csv_dir: Path, tag: str, j: int) -> Path | None:
    """LE: ``..._cohort-controls.csv``; DM: ``..._cohort-controls_alpha-0p50.csv``."""
    direct = csv_dir / f"{tag}_principal_gradient{j}_scores_cohort-controls.csv"
    if direct.is_file():
        return direct
    matches = sorted(
        csv_dir.glob(
            f"{tag}_principal_gradient{j}_scores_cohort-controls_alpha-*.csv"
        )
    )
    return matches[0] if matches else None


def _rows_from_csv_dir(csv_dir: Path) -> list[GradientRunRow]:
    factor_tags = sorted(
        {
            p.name.split("_principal_gradient")[0]
            for p in csv_dir.glob("F*_principal_gradient1_scores_cohort-controls*.csv")
        }
    )
    rows: list[GradientRunRow] = []
    for tag in factor_tags:
        means_path = csv_dir / f"{tag}_factor_score_means_cohort-controls.csv"
        if not means_path.is_file():
            print(f"  skip {tag}: missing {means_path.name}")
            continue
        means = _load_series(means_path, "mean_factor_score")
        grads: list[pd.Series] = []
        for j in range(1, 10):
            gp = _gradient_scores_csv(csv_dir, tag, j)
            if gp is None:
                break
            grads.append(_load_series(gp, f"principal_gradient{j}_score"))
        rows.append(
            (tag, means, grads, np.array([], dtype=np.float64))
        )
    return rows


def _write_sorted_score_csvs(csv_root: Path) -> None:
    for path in sorted(csv_root.rglob("*_principal_gradient*_scores_*.csv")):
        if path.name.endswith("_sorted.csv"):
            continue
        out = write_sorted_copies_for_gradient_csv(path)
        if out is not None:
            print(f"  sorted {out.name}")


def _max_gradient_k_from_csv_root(csv_root: Path) -> int:
    ks: list[int] = []
    for k_dir in csv_root.glob("gradients-*"):
        try:
            ks.append(int(k_dir.name.split("-", 1)[1]))
        except ValueError:
            continue
    return max(ks) if ks else 2


def _regenerate_for_method(
    method_root: Path,
    *,
    method_tag: str,
    tractometry_root: Path,
) -> None:
    csv_root = method_root / "csv"
    fig_root = method_root / "figures"
    print("  writing *_sorted.csv from existing gradient score tables")
    _write_sorted_score_csvs(csv_root)
    max_k = _max_gradient_k_from_csv_root(csv_root)
    all_rows: list = []
    for k_dir in sorted(csv_root.glob("gradients-*")):
        try:
            k = int(k_dir.name.split("-", 1)[1])
        except ValueError:
            continue
        rows = _rows_from_csv_dir(k_dir)
        if not rows:
            continue
        if not all_rows:
            all_rows = rows
        fig_dir = fig_root / k_dir.name
        fig_dir.mkdir(parents=True, exist_ok=True)
        g1_b = fig_dir / "gradient1_by-groups-axes_cohort-controls.png"
        plot_gradient_by_groups_axes_bars(
            rows,
            g1_b,
            gradient_index=0,
            tractometry_root=tractometry_root,
            method_tag=method_tag,
            cohort_tag="controls",
        )
        print(f"  wrote {g1_b}")
        if k >= 2:
            g2_b = fig_dir / "gradient2_by-groups-axes_cohort-controls.png"
            plot_gradient_by_groups_axes_bars(
                rows,
                g2_b,
                gradient_index=1,
                tractometry_root=tractometry_root,
                method_tag=method_tag,
                cohort_tag="controls",
            )
            print(f"  wrote {g2_b}")

        dims: int = 2 if k == 2 else 3
        dim_suffix = "" if dims == 2 else "_3D"
        sum_g1 = fig_dir / f"gradient1_summary_cohort-controls{dim_suffix}.png"
        plot_gradient1_summary(
            rows,
            sum_g1,
            dims=dims,  # type: ignore[arg-type]
            tractometry_root=tractometry_root,
            method_tag=method_tag,
            cohort_tag="controls",
        )
        print(f"  wrote {sum_g1}")
        if k >= 2:
            sum_g2 = fig_dir / f"gradient2_summary_cohort-controls{dim_suffix}.png"
            plot_gradient_summary(
                rows,
                sum_g2,
                gradient_index=1,
                dims=dims,  # type: ignore[arg-type]
                tractometry_root=tractometry_root,
                method_tag=method_tag,
                cohort_tag="controls",
            )
            print(f"  wrote {sum_g2}")

    if all_rows:
        nax_path = save_neuroaxis_correlations_csv(
            all_rows,
            csv_root / "neuroaxis_correlations_cohort-controls.csv",
            tractometry_root=tractometry_root,
            cohort_tag="controls",
            n_gradients=max_k,
        )
        print(f"  wrote {nax_path}")


def main() -> int:
    apply_figure_font_rcparams()
    gradients_root = DEFAULT_GRADIENTS_DIR
    tractometry_root = DEFAULT_TRACTOMETRY_ROOT
    print(f"gradients_root  = {gradients_root}")
    print(f"tractometry_root= {tractometry_root}")
    for method_tag in ("diffusion_embedding", "laplacian_eigenmodes"):
        method_root = gradients_root / method_tag
        if not method_root.is_dir():
            print(f"skip {method_tag}: {method_root} not found")
            continue
        print(f"--- {method_tag} ---")
        _regenerate_for_method(
            method_root,
            method_tag=method_tag,
            tractometry_root=tractometry_root,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
