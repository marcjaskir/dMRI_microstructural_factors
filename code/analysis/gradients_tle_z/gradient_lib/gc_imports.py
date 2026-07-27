"""Import helpers for gradients_group-controls modules without package name clash."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_GC_ROOT = Path(__file__).resolve().parents[2] / "gradients_group-controls"
_GC_LIB = _GC_ROOT / "gradient_lib"
_PKG = "gc_gradient_lib"


def _ensure_gc_package() -> ModuleType:
    if _PKG not in sys.modules:
        pkg = ModuleType(_PKG)
        pkg.__path__ = [str(_GC_LIB)]  # type: ignore[attr-defined]
        sys.modules[_PKG] = pkg
    return sys.modules[_PKG]


def _load_gc_module(name: str) -> ModuleType:
    full = f"{_PKG}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    path = _GC_LIB / f"{name}.py"
    if not path.is_file():
        raise ImportError(f"Missing group-controls module: {path}")
    _ensure_gc_package()
    spec = importlib.util.spec_from_file_location(full, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load group-controls module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def gc_plots_scatter():
    for dep in ("config", "types", "embedding", "io", "groupings", "region_groups"):
        _load_gc_module(dep)
    return _load_gc_module("plots_scatter")


def gc_groupings():
    return _load_gc_module("groupings")


def gc_plots_bars():
    for dep in (
        "config",
        "types",
        "embedding",
        "io",
        "groupings",
        "region_groups",
        "figure_style",
    ):
        _load_gc_module(dep)
    _load_gc_module("plots_scatter")  # axis label constants
    return _load_gc_module("plots_bars")
