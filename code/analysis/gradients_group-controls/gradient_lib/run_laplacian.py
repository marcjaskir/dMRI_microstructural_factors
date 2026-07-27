"""Laplacian eigenmap gradients (BrainSpace ``LaplacianEigenmaps``) and controls CSV writes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .affinity import laplacian_nontrivial_gradients
from .config import N_GRADIENTS_TO_COMPUTE
from .csv_outputs import write_principal_gradient_scores_csv
from .io import (
    collect_plotted_region_columns,
    load_4s_atlas_tsv,
    load_factor_matrix_from_df,
    load_glasser_atlas_tsv,
    load_region_means_from_df,
)
from .types import GradientRunRow


def save_laplacian_gradient_outputs(
    factor_tag: str,
    gradients: list[pd.Series],
    mean_per_roi: pd.Series,
    cohort_tag: str,
    out_dir: Path,
    *,
    n_gradients_to_save: int = 2,
) -> tuple[list[Path], Path]:
    """Write G1..G{n_gradients_to_save} gradient CSVs and one region-mean CSV (same column names as diffusion)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not gradients:
        raise ValueError("save_laplacian_gradient_outputs: empty gradient list")
    idx = gradients[0].index.astype(str)
    mean_aligned = mean_per_roi.reindex(idx)
    paths: list[Path] = []
    k_save = min(max(1, int(n_gradients_to_save)), len(gradients))
    for j in range(k_save):
        g = gradients[j].reindex(idx)
        path_g = out_dir / (
            f"{factor_tag}_principal_gradient{j + 1}_scores"
            f"_cohort-{cohort_tag}.csv"
        )
        grad_df = pd.DataFrame(
            {
                "region": idx,
                f"principal_gradient{j + 1}_score": g.to_numpy(dtype=np.float64),
            }
        )
        path_g, _ = write_principal_gradient_scores_csv(
            grad_df, path_g, gradient_index=j + 1
        )
        paths.append(path_g)
    path_mean = out_dir / f"{factor_tag}_factor_score_means_cohort-{cohort_tag}.csv"
    pd.DataFrame(
        {
            "region": idx,
            "mean_factor_score": mean_aligned.to_numpy(dtype=np.float64),
        }
    ).to_csv(path_mean, index=False)
    return paths, path_mean


def compute_laplacian_row(
    factor_tag: str,
    csv_path: Path,
    *,
    tractometry_root: Path,
    cohort_tag: str = "controls",
) -> GradientRunRow:
    """Run BrainSpace LaplacianEigenmaps on the controls wide table for one factor."""
    df = pd.read_csv(csv_path)
    X = load_factor_matrix_from_df(df)
    ref_mean = load_region_means_from_df(df)
    ref_mean.index = ref_mean.index.astype(str)

    glasser_tsv = load_glasser_atlas_tsv(tractometry_root)
    df_4s, four_s_subcortical_mask = load_4s_atlas_tsv(tractometry_root)

    needed = collect_plotted_region_columns(
        X, glasser_tsv, df_4s, four_s_subcortical_mask
    )
    regions = [c for c in needed if c in X.columns]
    n_max = max(0, len(regions) - 1)
    k_req = min(N_GRADIENTS_TO_COMPUTE, max(1, n_max))

    glist, lambdas = laplacian_nontrivial_gradients(X, needed, k_req)
    mean_aligned = ref_mean.reindex(glist[0].index)
    return factor_tag, mean_aligned, glist, lambdas
