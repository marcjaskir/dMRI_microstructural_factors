#!/usr/bin/env python3
"""Screen group-mean normalized factor score maps against neuromaps annotations."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CODE_ANALYSIS_DIR = _HERE.parent
_GRADIENTS_DIR = _CODE_ANALYSIS_DIR / "gradients_voxelwise"
for path in (_HERE, _GRADIENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gradient_lib.config import (  # noqa: E402
    COHORT_TAG,
    NEUROMAPS_DEFAULT_N_PERM_FSLR,
    NEUROMAPS_DEFAULT_N_PERM_MNI,
)
from gradient_lib.neuromaps_correlations import setup_neuromaps_data_dir  # noqa: E402
from neuromaps_lib.config import (  # noqa: E402
    DEFAULT_INPUT_DIR,
    DEFAULT_MAP_VARIANT,
    DEFAULT_MASK_NII,
    DEFAULT_NEUROMAPS_SPACES,
    FILE_PREFIX,
    parse_neuromaps_spaces,
)
from neuromaps_lib.correlations import run_all_correlations  # noqa: E402
from neuromaps_lib.plots_lollipop import plot_neuromaps_figures_from_output  # noqa: E402


def _parse_factors(arg: str | None) -> list[str] | None:
    if not arg or not arg.strip():
        return None
    out: list[str] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        tag = part if part.upper().startswith("F") else f"F{part}"
        if tag not in out:
            out.append(tag)
    return out


def _write_run_metadata(
    output_dir: Path,
    *,
    input_dir: Path,
    mask_nii: Path,
    map_variant: str,
    factors: list[str],
    spaces: tuple[str, ...],
    n_perm_mni: int,
    n_perm_fslr: int,
    skip_nulls: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "neuromaps_run.json"
    payload = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "mask_nii": str(mask_nii.resolve()),
        "map_variant": map_variant,
        "factors": factors,
        "spaces": list(spaces),
        "n_perm_mni": n_perm_mni,
        "n_perm_fslr": n_perm_fslr,
        "skip_nulls": skip_nulls,
        "file_prefix": FILE_PREFIX,
        "cohort_tag": COHORT_TAG,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute neuromaps spatial correlations for group-mean factor score NIfTIs "
            f"(default: *_{DEFAULT_MAP_VARIANT}.nii.gz). "
            "MNI nulls: burt2020; fsLR nulls: alexander_bloch. "
            "Set NEUROMAPS_DATA (defaults to output_dir/_cache/neuromaps/data). "
            "Requires: pip install neuromaps brainsmash"
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mask-nii", type=Path, default=DEFAULT_MASK_NII)
    parser.add_argument("--file-prefix", type=str, default=FILE_PREFIX)
    parser.add_argument("--map-variant", type=str, default=DEFAULT_MAP_VARIANT)
    parser.add_argument("--factors", type=str, default="F1,F2,F3")
    parser.add_argument("--n-perm-mni", type=int, default=NEUROMAPS_DEFAULT_N_PERM_MNI)
    parser.add_argument("--n-perm-fslr", type=int, default=NEUROMAPS_DEFAULT_N_PERM_FSLR)
    parser.add_argument(
        "--spaces",
        type=str,
        default=DEFAULT_NEUROMAPS_SPACES,
        help=(
            "Annotation pools to screen: mni (volume/MNI152+burt2020), "
            "fslr (surface/fsLR+alexander_bloch), both (default), or comma-separated."
        ),
    )
    parser.add_argument("--skip-nulls", action="store_true")
    parser.add_argument("--max-annotations", type=int, default=None)
    parser.add_argument("--figures", action="store_true", default=True)
    parser.add_argument("--no-figures", dest="figures", action="store_false")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    output_dir = args.output_dir or args.input_dir
    factors = _parse_factors(args.factors) or ["F1", "F2", "F3"]
    spaces = parse_neuromaps_spaces(args.spaces)

    print(f"\n=== Neuromaps factor scores ({args.map_variant}) ===")
    print(f"  input:  {args.input_dir}")
    print(f"  output: {output_dir}")
    print(f"  spaces: {', '.join(spaces)}")

    setup_neuromaps_data_dir(output_dir)

    if not args.figures_only:
        meta_path = _write_run_metadata(
            output_dir,
            input_dir=args.input_dir,
            mask_nii=args.mask_nii,
            map_variant=args.map_variant,
            factors=factors,
            spaces=spaces,
            n_perm_mni=args.n_perm_mni,
            n_perm_fslr=args.n_perm_fslr,
            skip_nulls=args.skip_nulls,
        )
        print(f"  saved {meta_path}")
        run_all_correlations(
            args.input_dir,
            output_dir,
            factors=factors,
            file_prefix=args.file_prefix,
            map_variant=args.map_variant,
            mask_nii=args.mask_nii,
            spaces=spaces,
            n_perm_mni=args.n_perm_mni,
            n_perm_fslr=args.n_perm_fslr,
            skip_nulls=args.skip_nulls,
            max_annotations=args.max_annotations,
            cohort_tag=COHORT_TAG,
            show_progress=not args.no_progress,
        )

    if args.figures or args.figures_only:
        paths = plot_neuromaps_figures_from_output(
            output_dir,
            factors=factors,
            map_variant=args.map_variant,
            cohort_tag=COHORT_TAG,
            spaces=spaces,
        )
        for p in paths:
            print(f"  saved {p}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
