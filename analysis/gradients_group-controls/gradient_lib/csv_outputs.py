"""CSV writers for gradient score tables (unsorted + sorted-by-score copies)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _sorted_gradient_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_sorted{path.suffix}")


def write_principal_gradient_scores_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    gradient_index: int,
) -> tuple[Path, Path]:
    """Write gradient scores CSV and a copy sorted descending by that gradient's score."""
    score_col = f"principal_gradient{gradient_index}_score"
    if score_col not in df.columns:
        raise ValueError(
            f"write_principal_gradient_scores_csv: missing column {score_col!r}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    sorted_path = _sorted_gradient_path(path)
    df.sort_values(score_col, ascending=False, kind="mergesort").to_csv(
        sorted_path, index=False
    )
    return path, sorted_path


def write_sorted_copies_for_gradient_csv(path: Path) -> Path | None:
    """If ``path`` is a principal-gradient scores CSV, write its ``_sorted`` sibling."""
    name = path.name
    if name.endswith("_sorted.csv") or not name.endswith(".csv"):
        return None
    if "_principal_gradient" not in name or "_scores_" not in name:
        return None
    header = list(pd.read_csv(path, nrows=0).columns)
    score_cols = [
        c
        for c in header
        if c.startswith("principal_gradient") and c.endswith("_score")
    ]
    if len(score_cols) != 1:
        return None
    col = score_cols[0]
    df = pd.read_csv(path)
    sorted_path = _sorted_gradient_path(path)
    df.sort_values(col, ascending=False, kind="mergesort").to_csv(
        sorted_path, index=False
    )
    return sorted_path
