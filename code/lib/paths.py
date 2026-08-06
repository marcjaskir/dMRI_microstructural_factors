"""Central path resolution for dMRI_microstructural_factors.

Scripts import from this module instead of hardcoding filesystem roots.
Open (`data/open/`) and controlled roots are configured in config.yaml.
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


# lib/paths.py -> code/ -> workspace root
_CODE_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def repo_root() -> Path:
    """Workspace root (git repository root containing code/ and data/)."""
    return _WORKSPACE_ROOT


def code_root() -> Path:
    """Root of the code/ directory."""
    return _CODE_ROOT


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
    for candidate in (
        _WORKSPACE_ROOT / "config.yaml",
        _CODE_ROOT / "config.yaml",
        _WORKSPACE_ROOT / "config.example.yaml",
    ):
        if candidate.exists():
            return candidate
    return _WORKSPACE_ROOT / "config.example.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    config_path = _resolve_config_path()
    if yaml is None:
        raise ImportError("PyYAML is required to load config.yaml (pip install pyyaml)")

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    if os.environ.get("DMRI_MICRO_ROOT"):
        raw["workspace_root"] = os.environ["DMRI_MICRO_ROOT"]
        raw.setdefault("project_root", os.environ["DMRI_MICRO_ROOT"])
    if os.environ.get("DMRI_MICRO_OPEN"):
        raw["data_open_dir"] = os.environ["DMRI_MICRO_OPEN"]
    if os.environ.get("DMRI_MICRO_CONTROLLED"):
        raw["data_controlled_dir"] = os.environ["DMRI_MICRO_CONTROLLED"]

    # Defaults if keys missing (for partial configs)
    ws = raw.get("workspace_root") or str(_WORKSPACE_ROOT)
    raw.setdefault("workspace_root", ws)
    raw.setdefault("code_dir", f"{ws}/code")
    raw.setdefault("data_open_dir", f"{ws}/data/open")
    raw.setdefault("data_controlled_dir", f"{ws}/data/controlled")
    raw.setdefault("gam_dir", "${data_open_dir}/gam")
    raw.setdefault("analysis_dir", "${data_open_dir}/analysis")
    raw.setdefault("atlas_dir", "${data_open_dir}/atlases")
    raw.setdefault("open_metadata_dir", "${data_open_dir}/metadata")
    raw.setdefault("inclusion_dir", "${data_open_dir}/inclusion")
    raw.setdefault("controlled_derivatives_dir", "${data_controlled_dir}/derivatives")
    raw.setdefault("controlled_metadata_dir", "${data_controlled_dir}/metadata")
    raw.setdefault("controlled_inclusion_dir", "${data_controlled_dir}/inclusion")
    raw.setdefault("subject_outcome_csv", "${data_controlled_dir}/subject_outcomes.csv")
    raw.setdefault("project_root", "${workspace_root}")
    raw.setdefault("data_dir", "${data_controlled_dir}")
    raw.setdefault("derivatives_dir", "${controlled_derivatives_dir}")
    raw.setdefault("results_dir", "${data_open_dir}")

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
    """Return a configured path by key."""
    cfg = load_config()
    if key not in cfg:
        raise KeyError(f"Config key not found: {key}")
    val = cfg[key]
    if val is None or val == "":
        raise KeyError(
            f"Config key {key!r} is unset (null). Required for this script, "
            "or set it in config.yaml if you have local controlled inputs."
        )
    return Path(str(val)).expanduser().resolve()


def workspace_root() -> Path:
    return get_path("workspace_root")


def project_root() -> Path:
    """Compatibility alias; typically the controlled/legacy data workspace."""
    return get_path("project_root")


def open_dir() -> Path:
    return get_path("data_open_dir")


def open_osf_url() -> str:
    """OSF project/storage URL for the open ``data/open/`` directory share."""
    cfg = load_config()
    env = __import__("os").environ.get("DMRI_MICRO_OSF_URL", "").strip()
    if env:
        return env
    return str(cfg.get("open_osf_url") or "").strip()


def controlled_dir() -> Path:
    return get_path("data_controlled_dir")


def gam_dir() -> Path:
    return get_path("gam_dir")


def analysis_dir() -> Path:
    return get_path("analysis_dir")


def controls_le_csv_dir() -> Path:
    """Directory with controls Laplacian G1/G2 score CSVs.

    Prefers the lab nested layout
    ``gradients_group-controls/laplacian_eigenmodes/csv/gradients-2/``,
    otherwise the flattened open layout ``gradients_group-controls/``.
    """
    root = analysis_dir() / "gradients_group-controls"
    nested = root / "laplacian_eigenmodes" / "csv" / "gradients-2"
    probe = "F1_principal_gradient1_scores_cohort-controls.csv"
    if (nested / probe).is_file():
        return nested
    return root


def atlas_dir() -> Path:
    return get_path("atlas_dir")


def open_metadata_dir() -> Path:
    return get_path("open_metadata_dir")


def controlled_metadata_dir() -> Path:
    return get_path("controlled_metadata_dir")


def controlled_derivatives_dir() -> Path:
    return get_path("controlled_derivatives_dir")


def data_dir() -> Path:
    return get_path("data_dir")


def derivatives_dir() -> Path:
    """Early-pipeline / controlled derivatives root."""
    return get_path("derivatives_dir")


def results_dir() -> Path:
    return get_path("results_dir")


def analysis_derivatives(*parts: str) -> Path:
    """Build path under open analysis products."""
    return analysis_dir().joinpath(*parts)


def code_dir(*parts: str) -> Path:
    """Build path under the code/ directory."""
    return get_path("code_dir").joinpath(*parts)


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
    return controlled_dir() / "subject_outcomes.csv"


def inclusion_dir() -> Path:
    """Open inclusion directory by default."""
    return get_path("inclusion_dir")


def controlled_inclusion_dir() -> Path:
    return get_path("controlled_inclusion_dir")


def inclusion_csv(name: str) -> Path:
    """Path to a named inclusion CSV under inclusion_dir()."""
    return inclusion_dir() / name


def analysis_output_dir(name: str) -> Path:
    """Path under analysis/<name>."""
    return analysis_dir() / name


def config_value(key: str, default: Any = None) -> Any:
    cfg = load_config()
    return cfg.get(key, default)
