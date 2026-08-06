"""CLI: discover → factor scores → normative factor z → scalar z."""
from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

from .config import (
    CONTROL_GROUPS,
    FACTOR_SCORES_DIR,
    FACTOR_Z_SCORES_DIR,
    OUTPUT_PROJECT_ROOT,
    PATIENT_GROUPS,
    SCALAR_Z_SCORES_OUTPUT_DIR,
)
from .io import (
    collect_control_group_map,
    collect_control_subjects_union_from_gam,
    discover_all_gm_regions,
    discover_all_wm_tracts,
    load_factor_loadings,
    load_scalar_labels,
    load_temporal_patient_subjects_ordered,
    load_tract_metadata_full,
)
from .scores import compute_and_save_all_factor_scores
from .scalar_z import save_scalar_z_scores
from .zscores import write_factor_z_scores


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply combined GM+WM factor loadings to GAM residual z → wide factor scores, "
            "control-normative factor z, and per-scalar z CSVs."
        )
    )
    parser.add_argument(
        "--skip-scores",
        action="store_true",
        help="Reuse existing factor_scores/*.csv; only (re)write factor z CSVs.",
    )
    parser.add_argument(
        "--skip-scalar-z",
        action="store_true",
        help="Skip scalar_z-scores/*.csv writers.",
    )
    parser.add_argument(
        "--scores-dir",
        default=FACTOR_SCORES_DIR,
        help="Directory for {cohort}_F*_scores.csv",
    )
    parser.add_argument(
        "--z-dir",
        default=FACTOR_Z_SCORES_DIR,
        help="Directory for {cohort}_F*_z_scores.csv",
    )
    parser.add_argument(
        "--scalar-z-dir",
        default=SCALAR_Z_SCORES_OUTPUT_DIR,
        help="Directory for scalar z CSVs",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    os.makedirs(OUTPUT_PROJECT_ROOT, exist_ok=True)
    print("factor_z-scores: starting CSV pipeline...")

    all_regions = discover_all_gm_regions()
    all_tracts = discover_all_wm_tracts()
    print(f"  Discovered {len(all_regions)} GM regions, {len(all_tracts)} WM tracts")
    if not all_regions or not all_tracts:
        print("No regions or tracts found; aborting.")
        return

    scalar_labels = load_scalar_labels()
    factor_loadings = load_factor_loadings(scalar_labels)
    if factor_loadings.empty and not args.skip_scores:
        print(f"No factor loadings; aborting score computation.")
        return

    if not args.skip_scores:
        if factor_loadings.empty:
            print("No factor loadings; cannot compute scores.")
            return
        print("Computing wide factor scores...")
        compute_and_save_all_factor_scores(
            scalar_labels,
            PATIENT_GROUPS,
            CONTROL_GROUPS,
            factor_loadings,
            args.scores_dir,
            all_regions=all_regions,
            all_tracts=all_tracts,
        )
    else:
        print(f"Skipping score computation; using {args.scores_dir}")

    print("Writing control-normative factor z-scores...")
    write_factor_z_scores(args.scores_dir, args.z_dir)

    if not args.skip_scalar_z:
        patient_subjects = load_temporal_patient_subjects_ordered()
        control_subjects = collect_control_subjects_union_from_gam(
            all_regions, all_tracts, scalar_labels, CONTROL_GROUPS
        )
        control_group_map = collect_control_group_map(
            all_regions, all_tracts, scalar_labels, CONTROL_GROUPS
        )
        tmeta = load_tract_metadata_full()
        te1 = te2 = {}
        if not tmeta.empty and "label" in tmeta.columns:
            if "end1" in tmeta.columns:
                te1 = dict(zip(tmeta["label"], tmeta["end1"]))
            if "end2" in tmeta.columns:
                te2 = dict(zip(tmeta["label"], tmeta["end2"]))
        print("Writing scalar z-score CSVs...")
        save_scalar_z_scores(
            all_regions,
            all_tracts,
            patient_subjects,
            control_subjects,
            PATIENT_GROUPS,
            CONTROL_GROUPS,
            te1,
            te2,
            output_dir=args.scalar_z_dir,
            control_group_map=control_group_map,
        )

    print("factor_z-scores: done.")


if __name__ == "__main__":
    main()
