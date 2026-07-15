import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""CLI: per-subject segment/node z-score asymmetry and Mahalanobis asymmetry CSVs (uses tract_asymmetry_normative)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import config as cfg
from .core import TractAsymmetry, _tract_base


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute tract asymmetry: z-score (segment + node) and Mahalanobis (segment + node) CSVs per subject."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("{project_root()}"),
        help="Project base directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root; default: <base-dir>/derivatives/analysis/tract_asymmetry.",
    )
    parser.add_argument(
        "--subject",
        "--subjects",
        dest="subjects",
        action="append",
        type=str,
        default=None,
        metavar="SUB",
        help="Subject ID(s). If omitted, use all eligible (TLE inclusion + GAM).",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Submit one SLURM job per subject; logs go to analysis/tract_asymmetry/logs (tract_asymmetry_SUB-%%j.o/.e).",
    )
    args = parser.parse_args()
    base_dir = Path(args.base_dir).resolve()
    paths = cfg.get_paths(base_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    normative_dir = paths.get("normative_dir")
    runner = TractAsymmetry(
        base_dir=base_dir,
        metadata_path=paths["metadata_path"],
        gam_dir=paths["gam_dir"],
        excluded_scalars=cfg.EXCLUDED_SCALARS,
        inclusion_path=paths["inclusion_path"],
        normative_dir=normative_dir,
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
        print("No eligible subjects found (TLE inclusion list + GAM).", file=sys.stderr)
        return 1

    if args.slurm:
        log_dir = (base_dir / "code" / "analysis" / "tract_asymmetry" / "logs").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        job_script = Path(__file__).resolve().parent / "tract_asymmetry_job.sh"
        if not job_script.is_file():
            print(f"Job script not found: {job_script}", file=sys.stderr)
            return 1
        base_str = str(base_dir)
        for sub in subjects:
            log_o = log_dir / f"tract_asymmetry_{sub}-%j.o"
            log_e = log_dir / f"tract_asymmetry_{sub}-%j.e"
            log_o, log_e = str(log_o), str(log_e)
            cmd = [
                "sbatch",
                f"--job-name=tract_asym_{sub}",
                f"--output={log_o}",
                f"--error={log_e}",
                str(job_script),
                sub,
                base_str,
            ]
            try:
                out = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"  {sub}: {out.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"  {sub}: sbatch failed: {e.stderr or e}", file=sys.stderr)
                return 1
        print(f"Submitted {len(subjects)} SLURM job(s). Logs: {log_dir}/tract_asymmetry_<sub>-<jobid>.o/.e")
        return 0

    # Per-tract count of subjects that had at least one segment missing for that tract
    tract_failure_subs: dict[str, set[str]] = {}
    all_tract_bases = sorted(
        set(_tract_base(t) for (t, _) in runner.get_all_bilateral_tract_segment_pairs())
    )
    for tb in all_tract_bases:
        tract_failure_subs[tb] = set()

    print(f"Writing asymmetry outputs (segment + node z-score, segment + node Mahalanobis) for {len(subjects)} subject(s)")
    for sub in subjects:
        sub_out = output_dir / sub
        sub_out.mkdir(parents=True, exist_ok=True)

        # 1. Segment-level z-score asymmetry
        out_csv = sub_out / f"{sub}_asym_scalars.csv"
        if out_csv.exists():
            print(f"  {sub}: asym_scalars.csv exists, skipping.")
        else:
            asym_scalar, missing_info = runner.compute_asymmetry(
                [sub], subject_ipsi_hemi, return_missing=True
            )
            if asym_scalar.empty:
                print(f"  {sub}: no segment asymmetry rows, skipping scalar CSV.")
            else:
                info = missing_info.get(sub, {})
                missing_seg = info.get("missing_segments", set())
                missing_scal = info.get("missing_scalars", set())
                if missing_seg:
                    seg_list = sorted(missing_seg, key=lambda x: (x[0], x[1]))[:20]
                    seg_str = ", ".join(f"{t}/{s}" for (t, s) in seg_list)
                    extra = f" ... and {len(missing_seg) - 20} more" if len(missing_seg) > 20 else ""
                    print(f"  Warning [{sub}]: {len(missing_seg)} missing tract segment(s): {seg_str}{extra}", file=sys.stderr)
                if missing_scal:
                    scal_list = sorted(missing_scal)[:20]
                    extra = f" ... and {len(missing_scal) - 20} more" if len(missing_scal) > 20 else ""
                    print(f"  Warning [{sub}]: {len(missing_scal)} missing scalar(s): {scal_list}{extra}", file=sys.stderr)
                for (tract_base, _seg) in missing_seg:
                    tract_failure_subs.setdefault(tract_base, set()).add(sub)
                asym_scalar.to_csv(out_csv, index=False)
                print(f"  {sub}: {len(asym_scalar)} rows -> {out_csv}")

        # 2. Node-level z-score asymmetry
        out_node_csv = sub_out / f"{sub}_asym_scalars_node.csv"
        if out_node_csv.exists():
            print(f"  {sub}: asym_scalars_node.csv exists, skipping.")
        else:
            asym_node = runner.compute_asymmetry_node_level([sub], subject_ipsi_hemi)
            if not asym_node.empty:
                asym_node.to_csv(out_node_csv, index=False)
                print(f"  {sub}: {len(asym_node)} rows -> {out_node_csv}")
            else:
                print(f"  {sub}: no node-level asymmetry rows.", file=sys.stderr)

        # 3. Segment-level Mahalanobis asymmetry (normative invcov)
        out_mahal_seg = sub_out / f"{sub}_asym_mahal_segment.csv"
        if out_mahal_seg.exists():
            print(f"  {sub}: asym_mahal_segment.csv exists, skipping.")
        else:
            mahal_seg = runner.compute_mahal_asymmetry_segment([sub], subject_ipsi_hemi)
            if not mahal_seg.empty:
                mahal_seg.to_csv(out_mahal_seg, index=False)
                print(f"  {sub}: {len(mahal_seg)} rows -> {out_mahal_seg}")
            else:
                print(f"  {sub}: no segment Mahalanobis rows (normative invcov may be missing).", file=sys.stderr)

        # 4. Node-level Mahalanobis asymmetry
        out_mahal_node = sub_out / f"{sub}_asym_mahal_node.csv"
        if out_mahal_node.exists():
            print(f"  {sub}: asym_mahal_node.csv exists, skipping.")
        else:
            mahal_node = runner.compute_mahal_asymmetry_node([sub], subject_ipsi_hemi)
            if not mahal_node.empty:
                mahal_node.to_csv(out_mahal_node, index=False)
                print(f"  {sub}: {len(mahal_node)} rows -> {out_mahal_node}")
            else:
                print(f"  {sub}: no node Mahalanobis rows (normative invcov may be missing).", file=sys.stderr)

    failures_path = output_dir / "tract_segmentation_failures.csv"
    all_tracts_for_csv = sorted(set(all_tract_bases) | set(tract_failure_subs.keys()))
    failures_df = pd.DataFrame(
        [
            {"tract": tract, "n_subjects_failed": len(tract_failure_subs.get(tract, set()))}
            for tract in all_tracts_for_csv
        ]
    )
    failures_df.to_csv(failures_path, index=False)
    print(f"Wrote {failures_path} ({len(failures_df)} tracts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
