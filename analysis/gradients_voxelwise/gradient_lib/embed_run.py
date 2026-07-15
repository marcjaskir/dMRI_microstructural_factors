"""Embed-run subdirectory naming for Laplacian gradient outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_EMBED_STRIDE, DEFAULT_MAX_EMBED_VOXELS, DEFAULT_TOP_K


def embed_run_subdir_name(
    *,
    embed_stride: int = DEFAULT_EMBED_STRIDE,
    embed_top_k: int = DEFAULT_TOP_K,
    max_embed_voxels: int = DEFAULT_MAX_EMBED_VOXELS,
) -> str:
    """Directory label encoding the three Laplacian subsampling parameters."""
    return f"stride{int(embed_stride)}_topk{int(embed_top_k)}_max{int(max_embed_voxels)}"


@dataclass(frozen=True)
class EmbedRunParams:
    embed_stride: int
    embed_top_k: int
    max_embed_voxels: int

    @property
    def subdir_name(self) -> str:
        return embed_run_subdir_name(
            embed_stride=self.embed_stride,
            embed_top_k=self.embed_top_k,
            max_embed_voxels=self.max_embed_voxels,
        )

    def gradient_run_dir(self, base_output_dir: Path) -> Path:
        return Path(base_output_dir) / self.subdir_name


def resolve_gradient_run_dir(
    base_output_dir: Path,
    *,
    embed_stride: int = DEFAULT_EMBED_STRIDE,
    embed_top_k: int = DEFAULT_TOP_K,
    max_embed_voxels: int = DEFAULT_MAX_EMBED_VOXELS,
) -> Path:
    """Return ``base_output_dir/stride{s}_topk{k}_max{m}``."""
    return EmbedRunParams(
        embed_stride=embed_stride,
        embed_top_k=embed_top_k,
        max_embed_voxels=max_embed_voxels,
    ).gradient_run_dir(base_output_dir)


def add_embed_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    top_k_alias: bool = False,
) -> None:
    """Register Laplacian embed subsampling flags (shared across gradient scripts)."""
    top_k_kwargs: dict = {
        "dest": "embed_top_k",
        "type": int,
        "default": DEFAULT_TOP_K,
        "help": "Top-k neighbors per voxel in the subject-correlation affinity graph.",
    }
    if top_k_alias:
        parser.add_argument("--embed-top-k", "--top-k", **top_k_kwargs)
    else:
        parser.add_argument("--embed-top-k", **top_k_kwargs)

    parser.add_argument(
        "--embed-stride",
        type=int,
        default=DEFAULT_EMBED_STRIDE,
        help="Grid stride for subsampling in-mask voxels before Laplacian embedding.",
    )
    parser.add_argument(
        "--max-embed-voxels",
        type=int,
        default=DEFAULT_MAX_EMBED_VOXELS,
        help="Maximum number of voxels in the Laplacian embed set after stride subsampling.",
    )


def embed_run_params_from_args(args: argparse.Namespace) -> EmbedRunParams:
    """Build embed params from parsed CLI namespace."""
    return EmbedRunParams(
        embed_stride=args.embed_stride,
        embed_top_k=args.embed_top_k,
        max_embed_voxels=args.max_embed_voxels,
    )


def write_embed_run_metadata(
    gradient_dir: Path,
    params: EmbedRunParams,
    *,
    base_output_dir: Path,
) -> Path:
    """Write reproducibility metadata for one embed-parameter run."""
    gradient_dir.mkdir(parents=True, exist_ok=True)
    path = gradient_dir / "embed_run.json"
    payload = {
        "embed_stride": params.embed_stride,
        "embed_top_k": params.embed_top_k,
        "max_embed_voxels": params.max_embed_voxels,
        "subdir_name": params.subdir_name,
        "base_output_dir": str(base_output_dir.resolve()),
        "gradient_run_dir": str(gradient_dir.resolve()),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
