#!/usr/bin/env python3
import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
"""Compute MNI-mm centroids for Glasser cortex, 4S156 subcortex, HCP1065 tract thirds.

Writes per-atlas xyz CSVs plus a combined ``wholebrain_centroids.csv``
(``label, atlas, x, y, z``) and a 3D-scatter QC PNG under
``derivatives/atlas_centroids/``.

Run from anywhere::

    python -m code.atlas_centroids.compute_atlas_centroids
    # or, with explicit roots:
    python code/atlas_centroids/compute_atlas_centroids.py --base-dir /path/to/repo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make ``lib.*`` importable when this script is invoked directly (no package install).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from lib.centroids_dseg import centroids_from_dseg  # noqa: E402
from lib.centroids_tract_thirds import tract_third_centroids  # noqa: E402
from lib.qc import render_3d_scatter  # noqa: E402

DEFAULT_PROJECT_ROOT = project_root()
# Defaults relative to base-dir.
GLASSER_NIFTI_REL = Path(
    "data/atlases/Glasser/atlas-Glasser_space-MNI152NLin2009cAsym_res-1_dseg.nii.gz"
)
GLASSER_TSV_REL = Path("data/atlases/Glasser/atlas-Glasser_dseg.tsv")
FOUR_S156_NIFTI_REL = Path(
    "data/atlases/4S/tpl-MNI152NLin2009cAsym_atlas-4S156Parcels_res-01_dseg.nii.gz"
)
FOUR_S156_TSV_REL = Path("data/atlases/4S/atlas-4S156Parcels_dseg.tsv")
HCP1065_CENTROIDS_REL = Path("data/atlases/HCP1065/centroids")
HCP1065_METADATA_REL = Path("data/atlases/HCP1065/HCP1065_tract_metadata.csv")

# 4S subcortical atlas names (matches gradients_group-controls/gradient_lib/config.py).
FOUR_S_SUBCORTICAL_ATLAS_NAMES: frozenset[str] = frozenset(
    {"ThalamusHCP", "SubcorticalHCP", "Cerebellum", "CIT168Subcortical"}
)

# Combined-CSV ``atlas`` column values (consumed by lib.qc).
ATLAS_TAG_GLASSER = "glasser_cortex"
ATLAS_TAG_FOUR_S = "four_s156_subcortex"
ATLAS_TAG_HCP1065 = "hcp1065_tract_third"


def _write_csv(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["label", "x", "y", "z"]].to_csv(out_path, index=False)
    return out_path


def run(
    base_dir: Path,
    output_dir: Path,
    *,
    no_plots: bool,
) -> int:
    base_dir = Path(base_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    glasser_nii = base_dir / GLASSER_NIFTI_REL
    glasser_tsv = base_dir / GLASSER_TSV_REL
    four_s_nii = base_dir / FOUR_S156_NIFTI_REL
    four_s_tsv = base_dir / FOUR_S156_TSV_REL
    hcp_centroids_dir = base_dir / HCP1065_CENTROIDS_REL
    hcp_metadata = base_dir / HCP1065_METADATA_REL

    print(f"[atlas_centroids] base_dir   = {base_dir}")
    print(f"[atlas_centroids] output_dir = {output_dir}")

    print("[atlas_centroids] Computing Glasser cortex centroids...")
    glasser_df = centroids_from_dseg(glasser_nii, glasser_tsv)
    glasser_csv = output_dir / "glasser_cortex_centroids.csv"
    _write_csv(glasser_df, glasser_csv)
    print(f"  wrote {glasser_csv} ({len(glasser_df)} rows)")

    print("[atlas_centroids] Computing 4S156 subcortex centroids...")
    four_s_df = centroids_from_dseg(
        four_s_nii,
        four_s_tsv,
        atlas_name_filter=FOUR_S_SUBCORTICAL_ATLAS_NAMES,
    )
    four_s_csv = output_dir / "four_s156_subcortex_centroids.csv"
    _write_csv(four_s_df, four_s_csv)
    print(f"  wrote {four_s_csv} ({len(four_s_df)} rows)")

    print("[atlas_centroids] Computing HCP1065 tract-third centroids...")
    hcp_df = tract_third_centroids(hcp_centroids_dir, hcp_metadata)
    hcp_csv = output_dir / "hcp1065_tract_third_centroids.csv"
    _write_csv(hcp_df, hcp_csv)
    print(f"  wrote {hcp_csv} ({len(hcp_df)} rows)")

    combined = pd.concat(
        [
            glasser_df.assign(atlas=ATLAS_TAG_GLASSER),
            four_s_df.assign(atlas=ATLAS_TAG_FOUR_S),
            hcp_df.assign(atlas=ATLAS_TAG_HCP1065),
        ],
        axis=0,
        ignore_index=True,
    )[["label", "atlas", "x", "y", "z"]]
    combined_csv = output_dir / "wholebrain_centroids.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_csv, index=False)
    print(
        f"  wrote {combined_csv} ({len(combined)} rows: "
        f"{(combined['atlas'] == ATLAS_TAG_GLASSER).sum()} cortex + "
        f"{(combined['atlas'] == ATLAS_TAG_FOUR_S).sum()} subcortex + "
        f"{(combined['atlas'] == ATLAS_TAG_HCP1065).sum()} tract thirds)"
    )

    if not no_plots:
        qc_png = output_dir / "wholebrain_centroids_3d.png"
        render_3d_scatter(combined, qc_png)
        print(f"  wrote {qc_png}")

    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute MNI-mm centroids for Glasser cortex, 4S156 subcortex, and "
            "HCP1065 tract thirds, plus a combined whole-brain table and 3D QC."
        ),
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Structural tractometry project root (default: repo root)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory (default: <base-dir>/derivatives/atlas_centroids)"
        ),
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the 3D QC scatter PNG",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    base_dir = Path(args.base_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (base_dir / "derivatives/atlas_centroids")
    )
    return run(base_dir, output_dir, no_plots=args.no_plots)


if __name__ == "__main__":
    sys.exit(main())
