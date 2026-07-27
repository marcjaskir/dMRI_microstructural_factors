"""Resolve factor loadings source and output directory for gradients_voxelwise."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .config import (
    CSF_MODES,
    DEFAULT_CSF_MODES,
    DEFAULT_FACTOR_LOADING_SOURCE,
    DEFAULT_GRADIENTS_VOXELWISE_DIR,
    DEFAULT_LOADINGS_BY_SOURCE,
    DEFAULT_MANIFEST_CSV,
    DEFAULT_MASK_NII,
    FACTOR_LOADING_SOURCES,
    GRADIENTS_RUN_DIR_VOXELWISE_CUSTOM,
    gradients_run_dir_name,
    voxelwise_loadings_csv,
    voxelwise_manifest_csv,
    voxelwise_mask_nii,
)
from .io_voxelwise import load_factor_loadings, scalar_labels_from_loadings


@dataclass(frozen=True)
class FactorLoadingContext:
    """Resolved loadings source, paths, and output directory."""

    source: str
    loadings_csv: Path
    output_dir: Path
    manifest_csv: Path = DEFAULT_MANIFEST_CSV
    mask_nii: Path = DEFAULT_MASK_NII
    csf_mode: str | None = None

    @property
    def run_label(self) -> str:
        """Human-readable label for logs."""
        if self.csf_mode:
            return f"voxelwise_{self.csf_mode}"
        return self.source


def parse_csf_modes(arg: str | None) -> tuple[str, ...]:
    """Parse ``--csf-modes`` values (comma-separated, ``both`` = with_csf + no_csf)."""
    if not arg or not str(arg).strip():
        return DEFAULT_CSF_MODES

    modes: list[str] = []
    for part in str(arg).split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token == "both":
            for mode in CSF_MODES:
                if mode not in modes:
                    modes.append(mode)
            continue
        if token not in CSF_MODES:
            raise ValueError(
                f"Unknown CSF mode {token!r}; expected one of {list(CSF_MODES)} or 'both'"
            )
        if token not in modes:
            modes.append(token)
    if not modes:
        raise ValueError("At least one CSF mode is required.")
    return tuple(modes)


def _resolve_output_dir(
    *,
    root: Path,
    source: str,
    csf_mode: str | None,
    output_dir: Path | None,
    exact: bool,
) -> Path:
    """Map CLI ``--output-dir`` to the run-specific output tree."""
    if exact:
        if output_dir is None:
            return root / GRADIENTS_RUN_DIR_VOXELWISE_CUSTOM
        return Path(output_dir)
    parent = Path(output_dir) if output_dir is not None else root
    return parent / gradients_run_dir_name(source, csf_mode)


def resolve_factor_loading_context(
    *,
    source: str = DEFAULT_FACTOR_LOADING_SOURCE,
    output_dir: Path | None = None,
    loadings_csv: Path | None = None,
    gradients_root: Path | None = None,
    csf_mode: str | None = None,
) -> FactorLoadingContext:
    """Resolve loadings CSV and output tree from source name."""
    src = str(source).strip().lower()
    if src not in FACTOR_LOADING_SOURCES:
        raise ValueError(
            f"Unknown factor-loading source {source!r}; "
            f"expected one of {list(FACTOR_LOADING_SOURCES)}"
        )

    root = gradients_root or DEFAULT_GRADIENTS_VOXELWISE_DIR

    if loadings_csv is not None:
        resolved_loadings = Path(loadings_csv)
        resolved_output = _resolve_output_dir(
            root=root,
            source=src,
            csf_mode=None,
            output_dir=output_dir,
            exact=True,
        )
        manifest_csv = DEFAULT_MANIFEST_CSV
        mask_nii = DEFAULT_MASK_NII
        resolved_csf_mode = None
    elif src == "voxelwise":
        if csf_mode is None:
            raise ValueError("csf_mode is required for voxelwise source without --loadings-csv")
        mode = str(csf_mode).strip().lower()
        resolved_loadings = voxelwise_loadings_csv(mode)
        manifest_csv = voxelwise_manifest_csv(mode)
        mask_nii = voxelwise_mask_nii(mode)
        resolved_output = _resolve_output_dir(
            root=root,
            source=src,
            csf_mode=mode,
            output_dir=output_dir,
            exact=False,
        )
        resolved_csf_mode = mode
    else:
        resolved_loadings = DEFAULT_LOADINGS_BY_SOURCE[src]
        resolved_output = _resolve_output_dir(
            root=root,
            source=src,
            csf_mode=None,
            output_dir=output_dir,
            exact=False,
        )
        manifest_csv = DEFAULT_MANIFEST_CSV
        mask_nii = DEFAULT_MASK_NII
        resolved_csf_mode = None

    if not resolved_loadings.is_file():
        raise FileNotFoundError(f"Missing factor loadings CSV: {resolved_loadings}")

    ctx = FactorLoadingContext(
        source=src,
        loadings_csv=resolved_loadings,
        output_dir=resolved_output,
        manifest_csv=manifest_csv,
        mask_nii=mask_nii,
        csf_mode=resolved_csf_mode,
    )
    validate_loadings_against_manifest(ctx)
    return ctx


def resolve_factor_loading_contexts(
    *,
    source: str = DEFAULT_FACTOR_LOADING_SOURCE,
    output_dir: Path | None = None,
    loadings_csv: Path | None = None,
    gradients_root: Path | None = None,
    csf_modes: tuple[str, ...] | None = None,
) -> list[FactorLoadingContext]:
    """Return one context per requested voxelwise CSF mode (or a single regionwise context)."""
    src = str(source).strip().lower()
    if loadings_csv is not None:
        return [
            resolve_factor_loading_context(
                source=src,
                output_dir=output_dir,
                loadings_csv=loadings_csv,
                gradients_root=gradients_root,
            )
        ]
    if src != "voxelwise":
        return [
            resolve_factor_loading_context(
                source=src,
                output_dir=output_dir,
                gradients_root=gradients_root,
            )
        ]

    modes = csf_modes or DEFAULT_CSF_MODES
    return [
        resolve_factor_loading_context(
            source=src,
            output_dir=output_dir,
            gradients_root=gradients_root,
            csf_mode=mode,
        )
        for mode in modes
    ]


def validate_loadings_against_manifest(ctx: FactorLoadingContext) -> None:
    """Ensure loadings scalar columns match the shared voxelwise manifest."""
    loadings = load_factor_loadings(ctx.loadings_csv)
    loading_scalars = set(scalar_labels_from_loadings(loadings))

    import pandas as pd

    manifest = pd.read_csv(ctx.manifest_csv)
    manifest_scalars = set(manifest["scalar"].astype(str).unique())
    missing = loading_scalars - manifest_scalars
    if missing:
        raise ValueError(
            f"Loadings scalars missing from manifest ({ctx.manifest_csv}): "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )


def write_loadings_source_metadata(ctx: FactorLoadingContext) -> Path:
    """Write reproducibility metadata under ``ctx.output_dir``."""
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.output_dir / "loadings_source.json"
    payload = {
        "source": ctx.source,
        "csf_mode": ctx.csf_mode,
        "run_label": ctx.run_label,
        "loadings_csv": str(ctx.loadings_csv.resolve()),
        "manifest_csv": str(ctx.manifest_csv.resolve()),
        "mask_nii": str(ctx.mask_nii.resolve()),
        "output_dir": str(ctx.output_dir.resolve()),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def add_factor_loading_arguments(parser: argparse.ArgumentParser) -> None:
    """Register shared CLI flags for factor loadings source resolution."""
    parser.add_argument(
        "--factor-loading-source",
        choices=list(FACTOR_LOADING_SOURCES),
        default=DEFAULT_FACTOR_LOADING_SOURCE,
        help=(
            "Factor loadings table: voxelwise FA (F1–F3) or regionwise All4_Combined (F1–F3). "
            "Voxelwise CSF modes write to gradients_voxelwise/voxelwise_{with_csf|no_csf}/; "
            "regionwise writes to gradients_voxelwise/regionwise/."
        ),
    )
    parser.add_argument(
        "--csf-modes",
        type=str,
        default=",".join(DEFAULT_CSF_MODES),
        help=(
            "Comma-separated voxelwise FA CSF modes: with_csf, no_csf, or both (default: both). "
            "Each mode uses matching FA loadings/manifest/mask and writes under "
            "{GRADIENTS_VOXELWISE_OUTPUT_DIR}/voxelwise_{mode}/. "
            "Ignored for regionwise or when --loadings-csv is set."
        ),
    )
    parser.add_argument(
        "--loadings-csv",
        type=Path,
        default=None,
        help="Override factor loadings CSV (must match voxelwise scalar manifest).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Parent directory for run-specific output trees. "
            "Voxelwise: {output-dir}/voxelwise_{with_csf|no_csf}/; "
            "regionwise: {output-dir}/regionwise/; "
            "gradients: .../stride{S}_topk{K}_max{M}/. "
            "Defaults to GRADIENTS_VOXELWISE_OUTPUT_DIR."
        ),
    )


def context_from_args(args: argparse.Namespace) -> FactorLoadingContext:
    """Build a single context (first CSF mode) from parsed CLI namespace."""
    contexts = contexts_from_args(args)
    return contexts[0]


def contexts_from_args(args: argparse.Namespace) -> list[FactorLoadingContext]:
    """Build one or more contexts from parsed CLI namespace."""
    csf_modes = parse_csf_modes(getattr(args, "csf_modes", None))
    return resolve_factor_loading_contexts(
        source=args.factor_loading_source,
        output_dir=args.output_dir,
        loadings_csv=args.loadings_csv,
        csf_modes=csf_modes,
    )
