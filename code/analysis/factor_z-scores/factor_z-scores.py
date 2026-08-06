#!/usr/bin/env python3
"""Entrypoint for the factor score / normative z CSV pipeline.

Prefer: ``python -m factor_z_scores`` from this directory, or this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from factor_z_scores.cli import main

if __name__ == "__main__":
    main()
