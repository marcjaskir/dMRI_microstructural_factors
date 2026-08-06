"""Canonical microstructural factor labels (paper-aligned).

F1 = overall diffusivity
F2 = non-Gaussian diffusivity
F3 = anisotropic diffusivity

Import from this module instead of duplicating label maps.
"""
from __future__ import annotations

from typing import Dict

# Full paper names (figures, tables, reports)
FACTOR_DIFFUSIVITY_LABELS: Dict[str, str] = {
    "F1": "Overall diffusivity",
    "F2": "Non-Gaussian diffusivity",
    "F3": "Anisotropic diffusivity",
}

# Short axis / bar labels
FACTOR_SHORT_LABELS: Dict[str, str] = {
    "F1": "Overall",
    "F2": "Non-Gaussian",
    "F3": "Anisotropic",
}

# Manuscript factor order
FACTOR_IDS = ("F1", "F2", "F3")


def factor_name_to_diffusivity_label(factor_name: str) -> str:
    """Map factor id (e.g. F2) to full diffusivity label."""
    if factor_name in FACTOR_DIFFUSIVITY_LABELS:
        return FACTOR_DIFFUSIVITY_LABELS[factor_name]
    if factor_name.startswith("F") and factor_name[1:].isdigit():
        return f"Factor {factor_name[1:]}"
    return factor_name


def factor_name_to_short_label(factor_name: str) -> str:
    """Map factor id (e.g. F2) to short label for compact axes."""
    if factor_name in FACTOR_SHORT_LABELS:
        return FACTOR_SHORT_LABELS[factor_name]
    if factor_name.startswith("F") and factor_name[1:].isdigit():
        return f"F{factor_name[1:]}"
    return factor_name


# Back-compat alias used by factor_z-scores plotting helpers
FACTOR_LABELS = FACTOR_SHORT_LABELS


def get_factor_label(factor_name: str) -> str:
    """Descriptive short label for a factor name."""
    return FACTOR_SHORT_LABELS.get(factor_name, factor_name)
