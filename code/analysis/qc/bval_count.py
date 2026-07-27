#!/usr/bin/env python3
"""Count volumes per b-value shell with +/-15 tolerance around target shells."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_SHELLS = (0, 300, 800, 2000)
SHELL_TOL = 15


def load_bvals(bval_path: Path) -> np.ndarray:
    """Load b-values from a .bval file (single row or column)."""
    bvals = np.loadtxt(bval_path, dtype=float)
    return np.atleast_1d(bvals).ravel()


def count_shells(
    bvals: Sequence[float],
    shells: Sequence[int] = DEFAULT_SHELLS,
    shell_tol: int = SHELL_TOL,
) -> dict[int, int]:
    """Return {shell: count} for volumes within shell +/- shell_tol."""
    counts: dict[int, int] = {}
    for shell in shells:
        lo, hi = shell - shell_tol, shell + shell_tol
        counts[shell] = sum(1 for b in bvals if lo <= b <= hi)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count DWI volumes per b-value shell (+/- tolerance)."
    )
    parser.add_argument(
        "bval",
        type=Path,
        help="Path to .bval file",
    )
    parser.add_argument(
        "--shells",
        type=int,
        nargs="+",
        default=list(DEFAULT_SHELLS),
        help=f"Target b-value shells (default: {list(DEFAULT_SHELLS)})",
    )
    parser.add_argument(
        "--tol",
        type=int,
        default=SHELL_TOL,
        help=f"Tolerance around each shell (default: {SHELL_TOL})",
    )
    args = parser.parse_args()

    bvals = load_bvals(args.bval)
    counts = count_shells(bvals, shells=args.shells, shell_tol=args.tol)

    print(f"bval: {args.bval}")
    print(f"tolerance: +/-{args.tol}")
    print("Shell counts:")
    for shell in args.shells:
        print(f"  b={shell:4d}: {counts[shell]:3d}")
    print(f"  total : {len(bvals):3d}")


if __name__ == "__main__":
    main()
