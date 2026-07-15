"""Load controls gradient CSVs and aggregate epilepsy factor z-scores per ROI."""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import NON_REGION_COLS

# (factor_tag, g1, g2, z_color) — each Series indexed by ROI name.
TleZScatterRow = tuple[str, pd.Series, pd.Series, pd.Series]


def parse_epilepsy_z_files(
    z_dir: Path,
    factors_filter: set[int] | None,
) -> list[tuple[str, Path]]:
    """``epilepsy_F{n}_z_scores.csv`` only."""
    by_factor: dict[int, Path] = {}
    for p in sorted(glob.glob(str(z_dir / "epilepsy_F*_z_scores.csv"))):
        name = Path(p).name
        m = re.match(r"epilepsy_F(\d+)_z_scores\.csv$", name, re.I)
        if not m:
            continue
        n = int(m.group(1))
        if factors_filter is not None and n not in factors_filter:
            continue
        by_factor[n] = Path(p)
    return [(f"F{n}", by_factor[n]) for n in sorted(by_factor)]


def _region_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_REGION_COLS]


def aggregate_epilepsy_z_scores(
    z_path: Path,
    *,
    absolute: bool,
) -> pd.Series:
    """Per-ROI epilepsy group mean: signed mean(z) or mean(|z|) across subjects."""
    df = pd.read_csv(z_path)
    region_cols = _region_columns(df)
    numeric = df[region_cols].apply(pd.to_numeric, errors="coerce")
    if absolute:
        out = numeric.abs().mean(axis=0, skipna=True)
    else:
        out = numeric.mean(axis=0, skipna=True)
    out.index = out.index.astype(str)
    return out


def load_gradient_series(csv_path: Path, value_col: str) -> pd.Series:
    """Load one gradient column from a controls gradient CSV (index = region)."""
    df = pd.read_csv(csv_path)
    if "region" not in df.columns:
        raise ValueError(f"Expected 'region' column in {csv_path}")
    if value_col not in df.columns:
        raise ValueError(f"Expected '{value_col}' column in {csv_path}")
    s = pd.to_numeric(df[value_col], errors="coerce")
    s.index = df["region"].astype(str)
    return s


def controls_gradient_csv_paths(
    gradients_csv_dir: Path,
    factor_tag: str,
) -> tuple[Path, Path]:
    """Return (G1 csv, G2 csv) for a factor under ``gradients-{K}/``."""
    m = re.match(r"^F(\d+)$", factor_tag.strip(), re.I)
    if not m:
        raise ValueError(f"Unsupported factor tag: {factor_tag!r}")
    k = int(m.group(1))
    g1 = gradients_csv_dir / f"F{k}_principal_gradient1_scores_cohort-controls.csv"
    g2 = gradients_csv_dir / f"F{k}_principal_gradient2_scores_cohort-controls.csv"
    return g1, g2


def load_controls_gradients(
    gradients_csv_dir: Path,
    factor_tag: str,
) -> tuple[pd.Series, pd.Series]:
    g1_path, g2_path = controls_gradient_csv_paths(gradients_csv_dir, factor_tag)
    g1 = load_gradient_series(g1_path, "principal_gradient1_score")
    g2 = load_gradient_series(g2_path, "principal_gradient2_score")
    return g1, g2


def build_scatter_rows(
    *,
    gradients_csv_dir: Path,
    z_dir: Path,
    factor_z_paths: list[tuple[str, Path]],
    absolute: bool,
) -> list[TleZScatterRow]:
    rows: list[TleZScatterRow] = []
    for factor_tag, z_path in factor_z_paths:
        g1, g2 = load_controls_gradients(gradients_csv_dir, factor_tag)
        z_color = aggregate_epilepsy_z_scores(z_path, absolute=absolute)
        rows.append((factor_tag, g1, g2, z_color))
    return rows


def save_aggregated_z_csvs(
    z_dir: Path,
    factor_z_paths: list[tuple[str, Path]],
    csv_out_dir: Path,
) -> None:
    """Write per-factor epilepsy mean signed and mean |z| tables for reproducibility."""
    csv_out_dir.mkdir(parents=True, exist_ok=True)
    for factor_tag, z_path in factor_z_paths:
        signed = aggregate_epilepsy_z_scores(z_path, absolute=False)
        absv = aggregate_epilepsy_z_scores(z_path, absolute=True)
        out = pd.DataFrame(
            {
                "region": signed.index.astype(str),
                "mean_z": signed.to_numpy(dtype=np.float64),
                "mean_abs_z": absv.reindex(signed.index).to_numpy(dtype=np.float64),
            }
        )
        out.to_csv(
            csv_out_dir / f"epilepsy_{factor_tag}_mean_z_scores.csv",
            index=False,
        )
