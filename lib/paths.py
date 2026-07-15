"""Central path resolution for dMRI_microstructural_factors.

All scripts should import from this module instead of hardcoding filesystem roots.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


_REPO_ROOT = Path(__file__).resolve().parents[1]
_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def repo_root() -> Path:
    """Root of the dMRI_microstructural_factors code repository."""
    return _REPO_ROOT


def _expand_vars(value: str, context: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise KeyError(f"Unknown config variable: {key}")
        return str(context[key])

    prev = None
    current = value
    while prev != current:
        prev = current
        current = _VAR_PATTERN.sub(repl, current)
    return current


def _resolve_config_path() -> Path:
    env_path = os.environ.get("DMRI_MICRO_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    local = _REPO_ROOT / "config.yaml"
    if local.exists():
        return local
    return _REPO_ROOT / "config.example.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    config_path = _resolve_config_path()
    if yaml is None:
        raise ImportError("PyYAML is required to load config.yaml (pip install pyyaml)")

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    if os.environ.get("DMRI_MICRO_ROOT"):
        raw["project_root"] = os.environ["DMRI_MICRO_ROOT"]

    context: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, str) and "${" in value:
            context[key] = _expand_vars(value, {**raw, **context})
        else:
            context[key] = value

    for key, value in list(context.items()):
        if isinstance(value, str) and "${" in value:
            context[key] = _expand_vars(value, context)

    return context


def get_path(key: str) -> Path:
    """Return a configured path by key (e.g. project_root, data_dir)."""
    cfg = load_config()
    if key not in cfg:
        raise KeyError(f"Config key not found: {key}")
    return Path(cfg[key]).expanduser().resolve()


def project_root() -> Path:
    return get_path("project_root")


def data_dir() -> Path:
    return get_path("data_dir")


def derivatives_dir() -> Path:
    return get_path("derivatives_dir")


def results_dir() -> Path:
    return get_path("results_dir")


def analysis_derivatives(*parts: str) -> Path:
    """Build path under derivatives/analysis/<subdir>/..."""
    return derivatives_dir() / "analysis" / Path(*parts)


def code_dir(*parts: str) -> Path:
    """Build path under repo root (code lives at repo root, not nested)."""
    return repo_root().joinpath(*parts)


def singularity_tmpdir() -> Path | None:
    cfg = load_config()
    val = cfg.get("singularity_tmpdir")
    return Path(val).expanduser().resolve() if val else None


def conda_env() -> str | None:
    cfg = load_config()
    val = cfg.get("conda_env")
    return str(val) if val else None


def cohorts() -> list[str]:
    cfg = load_config()
    return list(cfg.get("cohorts") or [])


def subject_outcome_csv() -> Path:
    cfg = load_config()
    path = cfg.get("subject_outcome_csv")
    if path:
        return Path(_expand_vars(str(path), cfg)).expanduser().resolve()
    return results_dir() / "subject_outcomes.csv"


def inclusion_dir() -> Path:
    cfg = load_config()
    subdir = cfg.get("inclusion_subdir", "inclusion")
    return results_dir() / subdir


def inclusion_csv(name: str) -> Path:
    """Path to a named inclusion CSV under inclusion_dir()."""
    return inclusion_dir() / name


def analysis_output_dir(name: str) -> Path:
    """Path under derivatives/analysis/<name>."""
    return derivatives_dir() / "analysis" / name


def config_value(key: str, default: Any = None) -> Any:
    cfg = load_config()
    return cfg.get(key, default)
