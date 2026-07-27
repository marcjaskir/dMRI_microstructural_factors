import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
"""CLI: compute region asymmetry from GAM mni_micro (Glasser, 4S156 cortex+subcortex, HCP1065), write per-subject CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from . import config as cfg
from .core import RegionAsymmetryTLE


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Region asymmetry from GAM mni_micro: z-score by scalar ({sub}_asym_regions.csv) and Mahalanobis by region ({sub}_asym_mahal_regions.csv)."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=project_root(),
        help="Project base directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root; default: <base-dir>/derivatives/analysis/region_asymmetry_tle.",
    )
    parser.add_argument(
        "--subject",
        "--subjects",
        dest="subjects",
        action="append",
        type=str,
        default=None,
        metavar="SUB",
        help="Subject ID(s). If omitted, process all eligible (TLE inclusion + GAM).",
    )
    parser.add_argument(
        "--mean-only",
        action="store_true",
        help="Output only mean stat (default: mean + standard_deviation).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    paths = cfg.get_paths(base_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = RegionAsymmetryTLE(
        gam_glasser_dir=paths["gam_glasser"],
        gam_4s156_dir=paths["gam_4s156"],
        atlas_tsv_4s=paths["atlas_tsv_4s"],
        inclusion_path=paths["inclusion_path"],
        gam_hcp1065_dir=paths.get("gam_hcp1065"),
        normative_dir=paths.get("normative_dir"),
    )

    tle_subjects, subject_ipsi_hemi = cfg.load_tle_inclusion(paths["inclusion_path"])
    if args.subjects:
        eligible = set(runner.get_eligible_subjects())
        subjects = [s for s in list(dict.fromkeys(args.subjects)) if s in eligible]
        if len(subjects) < len(list(dict.fromkeys(args.subjects))):
            print("Warning: some requested subjects are not eligible (not in TLE inclusion or missing GAM).", file=sys.stderr)
        subject_ipsi_hemi = {s: subject_ipsi_hemi[s] for s in subjects if s in subject_ipsi_hemi}
    else:
        subjects = runner.get_eligible_subjects()
        subject_ipsi_hemi = {s: subject_ipsi_hemi[s] for s in subjects if s in subject_ipsi_hemi}

    if not subjects:
        print("No eligible subjects found (TLE inclusion + GAM mni_micro Glasser/4S156/HCP1065).", file=sys.stderr)
        return 1

    stats = ["mean"] if args.mean_only else ["mean", "standard_deviation"]
    print(f"Writing asym_regions.csv and asym_mahal_regions.csv for {len(subjects)} subject(s) (stats: {stats})")

    pbar = tqdm(
        subjects,
        desc="Subjects",
        unit="subj",
        leave=True,
        file=sys.stderr,
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for sub in pbar:
        pbar.set_postfix_str(sub, refresh=True)
        sub_out = output_dir / sub
        sub_out.mkdir(parents=True, exist_ok=True)

        # 1. Z-score asymmetry by scalar (existing)
        out_csv = sub_out / f"{sub}_asym_regions.csv"
        if out_csv.exists():
            pbar.set_postfix_str(f"{sub} (skip)", refresh=True)
            tqdm.write(f"  {sub}: asym_regions.csv exists, skipping.", file=sys.stderr)
        else:
            df = runner.compute_asymmetry(
                subjects=[sub],
                subject_ipsi_hemi=subject_ipsi_hemi,
                stats=stats,
                show_progress=True,
            )
            if df.empty:
                pbar.set_postfix_str(f"{sub} (0 rows)", refresh=True)
                tqdm.write(f"  {sub}: no asymmetry rows, skipping.", file=sys.stderr)
            else:
                df.to_csv(out_csv, index=False)
                pbar.set_postfix_str(f"{sub} ({len(df)} rows)", refresh=True)
                tqdm.write(f"  {sub}: {len(df)} rows -> {out_csv}", file=sys.stderr)

        # 2. Mahalanobis asymmetry across scalars (normative invcov)
        out_mahal_csv = sub_out / f"{sub}_asym_mahal_regions.csv"
        if out_mahal_csv.exists():
            tqdm.write(f"  {sub}: asym_mahal_regions.csv exists, skipping.", file=sys.stderr)
        else:
            mahal_df = runner.compute_mahal_asymmetry(
                subjects=[sub],
                subject_ipsi_hemi=subject_ipsi_hemi,
                stat="mean",
                show_progress=True,
            )
            if mahal_df.empty:
                tqdm.write(f"  {sub}: no Mahalanobis rows (normative invcov may be missing).", file=sys.stderr)
            else:
                mahal_df.to_csv(out_mahal_csv, index=False)
                tqdm.write(f"  {sub}: {len(mahal_df)} Mahalanobis rows -> {out_mahal_csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
