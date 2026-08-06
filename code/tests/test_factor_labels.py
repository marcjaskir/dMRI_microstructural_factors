#!/usr/bin/env python3
"""Contract: factor label maps match the manuscript (F1/F2/F3 order)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_CODE = Path(__file__).resolve().parents[1]
if str(REPO_CODE) not in sys.path:
    sys.path.insert(0, str(REPO_CODE))

from lib.factor_labels import (  # noqa: E402
    FACTOR_DIFFUSIVITY_LABELS,
    FACTOR_IDS,
    FACTOR_SHORT_LABELS,
)

EXPECTED_FULL = {
    "F1": "Overall diffusivity",
    "F2": "Non-Gaussian diffusivity",
    "F3": "Anisotropic diffusivity",
}
EXPECTED_SHORT = {
    "F1": "Overall",
    "F2": "Non-Gaussian",
    "F3": "Anisotropic",
}


def test_canonical_labels() -> None:
    assert FACTOR_IDS == ("F1", "F2", "F3")
    assert FACTOR_DIFFUSIVITY_LABELS == EXPECTED_FULL
    assert FACTOR_SHORT_LABELS == EXPECTED_SHORT
    # Short labels must be prefixes of full names (sans " diffusivity")
    for fid in FACTOR_IDS:
        assert EXPECTED_FULL[fid].startswith(EXPECTED_SHORT[fid])


def test_no_swapped_inline_dicts_in_analysis() -> None:
    """Fail if analysis modules redefine F2/F3 with the known-swapped mapping."""
    swapped = {
        "F1": "Overall",
        "F2": "Anisotropic",
        "F3": "Non-Gaussian",
    }
    analysis = REPO_CODE / "analysis"
    offenders: list[str] = []
    for path in analysis.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Anisotropic" not in text or "Non-Gaussian" not in text:
            continue
        # Heuristic: look for Assign nodes named FACTOR_LABELS / similar
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id not in {
                    "FACTOR_LABELS",
                    "FACTOR_SHORT_LABELS",
                    "FACTOR_DIFFUSIVITY_LABELS",
                }:
                    continue
                if not isinstance(node.value, ast.Dict):
                    continue
                try:
                    d = ast.literal_eval(node.value)
                except Exception:
                    continue
                # Normalize to short labels for comparison
                short = {
                    k: (v.replace(" diffusivity", "") if isinstance(v, str) else v)
                    for k, v in d.items()
                    if k in ("F1", "F2", "F3")
                }
                if short.get("F2") == "Anisotropic" and short.get("F3") == "Non-Gaussian":
                    offenders.append(str(path.relative_to(REPO_CODE)))
    assert not offenders, (
        "Swapped F2/F3 labels (Anisotropic/Non-Gaussian) found in:\n  "
        + "\n  ".join(offenders)
        + "\nImport lib.factor_labels instead."
    )


if __name__ == "__main__":
    test_canonical_labels()
    test_no_swapped_inline_dicts_in_analysis()
    print("PASS factor label contract")
