#!/usr/bin/env python3
"""Golden-output regression tests for cleaned dMRI_microstructural_factors code.

Freeze digests for manuscript DAG products (loadings, factor z, LE neuroaxis,
group asymmetry summaries) plus the original profile / tract-asymmetry checks.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve()
while REPO != REPO.parent and not (REPO / "lib" / "paths.py").exists():
    REPO = REPO.parent
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_csv(a: Path, b: Path, label: str) -> None:
    df_a = pd.read_csv(a)
    df_b = pd.read_csv(b)
    if list(df_a.columns) != list(df_b.columns):
        raise AssertionError(f"{label}: column mismatch\n{df_a.columns}\n{df_b.columns}")
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
        load_ilf_gam_mean,
    )

    df = load_ilf_gam_mean(SCALAR)
    node_cols_z = [f"node{i}_z" for i in range(1, N_NODES_PROFILE + 1)]
    node_cols = [f"node{i}" for i in range(1, N_NODES_PROFILE + 1)]
    if all(c in df.columns for c in node_cols_z):
        cols = node_cols_z
    elif all(c in df.columns for c in node_cols):
        cols = node_cols
    else:
        raise KeyError("GAM table missing node* or node*_z profile columns")
    profile = df[cols].astype(float).mean(axis=0)
    out = pd.DataFrame({"node": range(1, N_NODES_PROFILE + 1), "mean": profile.to_numpy()})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


def export_asymmetry_summary(out_csv: Path) -> None:
    _ensure_paths()
    from lib.paths import analysis_dir

    tract_dir = analysis_dir() / "tract_asymmetry"
    rows = []
    subject_dirs = sorted(tract_dir.glob("anon_*")) or sorted(tract_dir.glob("sub-*"))
    for i, sub_dir in enumerate(subject_dirs, start=1):
        candidates = list(sub_dir.glob("*_asym_scalars.csv"))
        if not candidates:
            continue
        csv_path = candidates[0]
        df = pd.read_csv(csv_path)
        effect = None
        for col in ("cohens_d", "asymmetry"):
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                effect = float(df[col].abs().mean())
                break
        rows.append(
            {
                "anon_id": f"anon_{i:03d}",
                "n_rows": len(df),
                "mean_abs_cohend": effect if effect is not None else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _digest_frame(df: pd.DataFrame, source: str) -> dict:
    """Stable numeric digest for a table (order-sensitive)."""
    numeric = df.select_dtypes(include=[np.number])
    payload = {
        "source": source,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns.astype(str)),
    }
    if numeric.shape[1]:
        vals = numeric.to_numpy(dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        payload.update(
            {
                "n_finite": int(vals.size),
                "sum": float(vals.sum()) if vals.size else 0.0,
                "mean": float(vals.mean()) if vals.size else 0.0,
                "std": float(vals.std(ddof=0)) if vals.size else 0.0,
                "abs_sum": float(np.abs(vals).sum()) if vals.size else 0.0,
            }
        )
    # Content hash of CSV text for exact freeze
    text = df.to_csv(index=False).encode("utf-8")
    payload["sha256"] = hashlib.sha256(text).hexdigest()
    return payload


def export_manuscript_digests(out_json: Path) -> dict:
    """Freeze key manuscript DAG table digests under analysis_dir()."""
    _ensure_paths()
    from lib.paths import analysis_dir

    root = analysis_dir()
    from lib.paths import controls_le_csv_dir

    le_csv = controls_le_csv_dir()
    targets = [
        ("factor_loadings", root / "factor_analysis/controls_All4_Combined_scalar_factor_loadings.csv"),
        (
            "factor_loadings_ordered",
            root / "factor_analysis/controls_All4_Combined_scalar_factor_loadings_ordered.csv",
        ),
        ("controls_F1_z", root / "factor_z-scores/factor_z_scores/controls_F1_z_scores.csv"),
        ("controls_F2_z", root / "factor_z-scores/factor_z_scores/controls_F2_z_scores.csv"),
        ("controls_F3_z", root / "factor_z-scores/factor_z_scores/controls_F3_z_scores.csv"),
        (
            "le_neuroaxis",
            root / "gradients_group-controls/neuroaxis_correlations_cohort-controls.csv"
            if (root / "gradients_group-controls/neuroaxis_correlations_cohort-controls.csv").exists()
            else root
            / "gradients_group-controls/laplacian_eigenmodes/csv/neuroaxis_correlations_cohort-controls.csv",
        ),
        (
            "le_F1_G1",
            le_csv / "F1_principal_gradient1_scores_cohort-controls.csv",
        ),
        (
            "le_F1_G2",
            le_csv / "F1_principal_gradient2_scores_cohort-controls.csv",
        ),
        (
            "asym_thirds_mahalanobis",
            root / "microstructural_asymmetries/summary_hcp1065_thirds_mahalanobis.csv",
        ),
        (
            "asym_factor_z_summary",
            root / "microstructural_asymmetries/factor_score_z_ipsi_contra_cohens_d_summary.csv",
        ),
        (
            "asym_whole_scalars",
            root / "microstructural_asymmetries/summary_hcp1065_whole_scalars.csv",
        ),
    ]
    digests: dict[str, dict] = {}
    missing: list[str] = []
    for key, path in targets:
        if not path.exists():
            missing.append(f"{key}: {path}")
            continue
        digests[key] = _digest_frame(pd.read_csv(path), source=str(path.relative_to(root)))
    if missing:
        raise FileNotFoundError(
            "Missing manuscript digest sources:\n  " + "\n  ".join(missing)
        )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")
    return digests


def compare_digests(a: Path, b: Path, label: str) -> None:
    da = json.loads(a.read_text())
    db = json.loads(b.read_text())
    if set(da) != set(db):
        raise AssertionError(f"{label}: key mismatch {sorted(da)} vs {sorted(db)}")
    for key in sorted(da):
        if da[key].get("sha256") != db[key].get("sha256"):
            # Fall back to numeric rtol on sum/mean if schema drifted
            for metric in ("sum", "mean", "std", "abs_sum"):
                if metric in da[key] and metric in db[key]:
                    np.testing.assert_allclose(
                        da[key][metric],
                        db[key][metric],
                        rtol=RTOL,
                        atol=ATOL,
                        err_msg=f"{label}/{key}: {metric}",
                    )
            raise AssertionError(
                f"{label}/{key}: sha256 mismatch "
                f"{da[key].get('sha256')} vs {db[key].get('sha256')}"
            )
        for field in ("n_rows", "n_cols", "n_finite"):
            if field in da[key]:
                assert da[key][field] == db[key][field], f"{label}/{key}: {field}"
    print(f"PASS {label} ({len(da)} tables)")


def run_profile_thirds_script() -> Path:
    script = REPO / "analysis/profile_thirds_example/plot_normative_example_minimal.py"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(script)], check=True, cwd=str(REPO))
    from lib.paths import analysis_dir

    return analysis_dir() / "profile_thirds_example/ILF_L_dti_md_mean_profile_nodes_harmonized_pyafq_gam.png"


def capture_baseline() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    export_profile_means(BASELINE_DIR / "profile_means.csv")
    export_asymmetry_summary(BASELINE_DIR / "asymmetry_tract_summary.csv")
    digests = export_manuscript_digests(BASELINE_DIR / "manuscript_digests.json")
    meta = {
        "profile_means_rows": int(pd.read_csv(BASELINE_DIR / "profile_means.csv").shape[0]),
        "asymmetry_subjects": int(pd.read_csv(BASELINE_DIR / "asymmetry_tract_summary.csv").shape[0]),
        "manuscript_digest_keys": sorted(digests.keys()),
    }
    (BASELINE_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("Baseline captured:", meta)


def run_tests() -> None:
    if not (BASELINE_DIR / "profile_means.csv").exists():
        print("No baseline found; capturing baseline first...")
        capture_baseline()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_profile_means(OUTPUT_DIR / "profile_means.csv")
    export_asymmetry_summary(OUTPUT_DIR / "asymmetry_tract_summary.csv")
    export_manuscript_digests(OUTPUT_DIR / "manuscript_digests.json")

    compare_csv(BASELINE_DIR / "profile_means.csv", OUTPUT_DIR / "profile_means.csv", "profile_means")
    compare_csv(
        BASELINE_DIR / "asymmetry_tract_summary.csv",
        OUTPUT_DIR / "asymmetry_tract_summary.csv",
        "asymmetry_tract_summary",
    )
    if (BASELINE_DIR / "manuscript_digests.json").exists():
        compare_digests(
            BASELINE_DIR / "manuscript_digests.json",
            OUTPUT_DIR / "manuscript_digests.json",
            "manuscript_digests",
        )
    else:
        print("SKIP manuscript_digests (no baseline yet; run capture)")

    from lib.paths import analysis_dir, gam_dir

    gam_csv = gam_dir() / "pyafq/HCP1065/ILF_L/ILF_L_dti_md_stat-mean_gam.csv"
    ref_png = analysis_dir() / "profile_thirds_example/ILF_L_dti_md_mean_profile_nodes_harmonized_pyafq_gam.png"
    if ref_png.exists() and gam_csv.exists():
        cols = set(pd.read_csv(gam_csv, nrows=0).columns)
        has_raw_nodes = all(f"node{i}" in cols for i in range(1, 101))
        has_age_sex = {"age", "sex"}.issubset(cols)
        if has_raw_nodes and has_age_sex:
            before = md5_file(ref_png)
            run_profile_thirds_script()
            after = md5_file(ref_png)
            assert before == after, "profile_thirds PNG changed after cleanup"
            print("PASS profile_thirds_png_unchanged")
        else:
            print("SKIP profile_thirds_png (open GAM lacks age/sex or raw node columns)")

    print("All golden tests passed.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        capture_baseline()
    else:
        run_tests()
