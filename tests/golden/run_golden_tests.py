#!/usr/bin/env python3
"""Golden-output regression tests for cleaned dMRI_microstructural_factors code."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parent / "baseline"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
RTOL = 1e-6
ATOL = 1e-8


def _ensure_paths() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_csv(a: Path, b: Path, label: str) -> None:
    df_a = pd.read_csv(a)
    df_b = pd.read_csv(b)
    if list(df_a.columns) != list(df_b.columns):
        raise AssertionError(f"{label}: column mismatch\n{a.columns}\n{b.columns}")
    if len(df_a) != len(df_b):
        raise AssertionError(f"{label}: row count {len(df_a)} vs {len(df_b)}")
    for col in df_a.columns:
        if pd.api.types.is_numeric_dtype(df_a[col]):
            np.testing.assert_allclose(
                df_a[col].astype(float).to_numpy(),
                df_b[col].astype(float).to_numpy(),
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"{label}: numeric mismatch in {col}",
            )
        else:
            assert df_a[col].astype(str).tolist() == df_b[col].astype(str).tolist(), (
                f"{label}: string mismatch in {col}"
            )
    print(f"PASS {label}")


def export_profile_means(out_csv: Path) -> None:
    _ensure_paths()
    from analysis.profile_thirds_example.plot_normative_example_minimal import (
        N_NODES_PROFILE,
        SCALAR,
        TRACT,
        load_ilf_gam_mean,
    )

    df = load_ilf_gam_mean(SCALAR)
    node_cols = [f"node{i}" for i in range(1, N_NODES_PROFILE + 1)]
    profile = df[node_cols].astype(float).mean(axis=0)
    out = pd.DataFrame({"node": range(1, N_NODES_PROFILE + 1), "mean": profile})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


def export_asymmetry_summary(out_csv: Path) -> None:
    _ensure_paths()
    from lib.paths import derivatives_dir

    tract_dir = derivatives_dir() / "analysis" / "tract_asymmetry"
    rows = []
    for sub_dir in sorted(tract_dir.glob("sub-*")):
        csv_path = sub_dir / f"{sub_dir.name}_asym_scalars.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        rows.append(
            {
                "sub": sub_dir.name,
                "n_rows": len(df),
                "mean_abs_cohend": float(df["cohens_d"].abs().mean()) if "cohens_d" in df.columns else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def run_profile_thirds_script() -> Path:
    script = REPO / "analysis/profile_thirds_example/plot_normative_example_minimal.py"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(script)], check=True, cwd=str(REPO))
    png = REPO / "derivatives/analysis/profile_thirds_example/ILF_L_dti_md_mean_profile_nodes_harmonized_pyafq_gam.png"
    # script writes under project_root derivatives
    from lib.paths import derivatives_dir

    png = derivatives_dir() / "analysis/profile_thirds_example/ILF_L_dti_md_mean_profile_nodes_harmonized_pyafq_gam.png"
    return png


def capture_baseline() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    export_profile_means(BASELINE_DIR / "profile_means.csv")
    export_asymmetry_summary(BASELINE_DIR / "asymmetry_tract_summary.csv")
    meta = {
        "profile_means_rows": int(pd.read_csv(BASELINE_DIR / "profile_means.csv").shape[0]),
        "asymmetry_subjects": int(pd.read_csv(BASELINE_DIR / "asymmetry_tract_summary.csv").shape[0]),
    }
    (BASELINE_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Baseline captured:", meta)


def run_tests() -> None:
    if not (BASELINE_DIR / "profile_means.csv").exists():
        print("No baseline found; capturing baseline first...")
        capture_baseline()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_profile_means(OUTPUT_DIR / "profile_means.csv")
    export_asymmetry_summary(OUTPUT_DIR / "asymmetry_tract_summary.csv")

    compare_csv(BASELINE_DIR / "profile_means.csv", OUTPUT_DIR / "profile_means.csv", "profile_means")
    compare_csv(
        BASELINE_DIR / "asymmetry_tract_summary.csv",
        OUTPUT_DIR / "asymmetry_tract_summary.csv",
        "asymmetry_tract_summary",
    )

    # Optional: regenerate profile_thirds PNG and compare to existing derivative
    from lib.paths import derivatives_dir

    ref_png = derivatives_dir() / "analysis/profile_thirds_example/ILF_L_dti_md_mean_profile_nodes_harmonized_pyafq_gam.png"
    if ref_png.exists():
        before = md5_file(ref_png)
        run_profile_thirds_script()
        after = md5_file(ref_png)
        assert before == after, "profile_thirds PNG changed after cleanup"
        print("PASS profile_thirds_png_unchanged")

    print("All golden tests passed.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        capture_baseline()
    else:
        run_tests()
