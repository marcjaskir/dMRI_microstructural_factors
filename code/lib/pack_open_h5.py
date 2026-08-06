#!/usr/bin/env python3
"""Pack / unpack manuscript-reproduction open products into a single HDF5 file.

Default pack profile (``core``) includes tabular products needed for golden tests
and manuscript digests — not NIfTI, HTML/PNG, diffusion-map trees, or demographics.

Schema (attrs on root):
  schema_version, paper_citation, created_utc, profile, osf_url_placeholder

Groups:
  /files/<uuid>  dataset = gzip-compressed UTF-8 (or bytes) file payload
                 attrs: relpath (posix under data/open/), content_type, sha256

Usage:
  python -u code/lib/pack_open_h5.py pack [--profile core|full_csv]
  python -u code/lib/pack_open_h5.py unpack [--h5 PATH] [--dest data/open]
  python -u code/lib/pack_open_h5.py check [--h5 PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
OPEN = WORKSPACE / "data" / "open"
DEFAULT_H5_NAME = "dmri_microstructural_factors_open_v1.h5"
SCHEMA_VERSION = "1"
PAPER_CITATION = "Jaskir et al. 2026 (citation TBD)"
OSF_URL_PLACEHOLDER = "OSF_URL=https://osf.io/XXXXX/  # replace after upload"
FORBIDDEN_COL_TOKENS = {"age", "sex", "bat", "anon_id_map", "gender", "scanner"}

# Relative paths / globs under data/open/ for the manuscript core pack.
CORE_GLOBS: tuple[str, ...] = (
    "inclusion/*.csv",
    "metadata/*.json",
    "atlases/**/*.csv",
    "atlases/**/*.tsv",
    "atlases/**/*.json",
    "atlases/**/*.txt",
    "atlases/**/*.md",
    # Factor analysis / z / representation
    "analysis/factor_analysis/**/*.csv",
    "analysis/factor_z-scores/factor_z_scores/*.csv",
    "analysis/factor_z-scores/roi_factor_scores/*.csv",
    "analysis/factor_z-scores/roi_means_rescaled/*.csv",
    "analysis/factor_z-scores/plots/*statistics*.csv",
    "analysis/factor_representation/**/*.csv",
    # LE gradients only (not diffusion_embedding)
    "analysis/gradients_group-controls/laplacian_eigenmodes/csv/**/*.csv",
    "analysis/gradients_tle_z/**/*.csv",
    # Group asymmetry digests
    "analysis/microstructural_asymmetries/*.csv",
    "analysis/qc/**/*.csv",
    # Tract asymmetry per-subject scalars (golden digests)
    "analysis/tract_asymmetry/**/*_asym_scalars.csv",
    "analysis/region_asymmetry_tle/**/*.csv",
    # Minimal GAM table for profile golden / Methods schematic
    "gam/pyafq/HCP1065/ILF_L/ILF_L_dti_md_stat-mean_gam.csv",
)

# Broader CSV pack (still excludes NIfTI/HTML/PNG and diffusion_embedding).
FULL_CSV_GLOBS: tuple[str, ...] = (
    "inclusion/**/*.csv",
    "metadata/**/*",
    "atlases/**/*",
    "analysis/factor_analysis/**/*.csv",
    "analysis/factor_z-scores/**/*.csv",
    "analysis/factor_representation/**/*.csv",
    "analysis/factor_analysis_voxelwise/**/*.csv",
    "analysis/gradients_group-controls/laplacian_eigenmodes/**/*.csv",
    "analysis/gradients_tle_z/**/*.csv",
    "analysis/microstructural_asymmetries/**/*.csv",
    "analysis/qc/**/*.csv",
    "analysis/tract_asymmetry/**/*.csv",
    "analysis/tract_asymmetry_normative/**/*.csv",
    "analysis/region_asymmetry_tle/**/*.csv",
    "analysis/region_asymmetry_tle_normative/**/*.csv",
    "analysis/profile_thirds_example/**/*.csv",
    "analysis/covbat_example/**/*.csv",
    "gam/**/*.csv",
)

SKIP_NAME_PARTS = {".ipynb_checkpoints", "__pycache__", ".ipyniivue_cache", ".archive"}
SKIP_SUFFIXES = {
    ".nii",
    ".nii.gz",
    ".gii",
    ".mgz",
    ".trk",
    ".tck",
    ".sif",
    ".mat",
    ".mif",
    ".fib",
    ".png",
    ".html",
    ".pdf",
    ".svg",
}


def default_h5_path() -> Path:
    return OPEN / DEFAULT_H5_NAME


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _should_skip(path: Path) -> bool:
    if any(part in SKIP_NAME_PARTS for part in path.parts):
        return True
    lower = path.name.lower()
    for suf in SKIP_SUFFIXES:
        if lower.endswith(suf):
            return True
    # Never pack reversible ID maps or controlled trees
    if path.name == "anon_id_map.csv":
        return True
    if "diffusion_embedding" in path.parts:
        return True
    return False


def iter_pack_files(open_root: Path, globs: Iterable[str]) -> Iterator[Path]:
    seen: set[Path] = set()
    for pattern in globs:
        for path in sorted(open_root.glob(pattern)):
            if not path.is_file() or _should_skip(path):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def _csv_has_forbidden_header(path: Path) -> list[str]:
    if path.suffix.lower() != ".csv":
        return []
    try:
        header = path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    except OSError:
        return []
    if not header:
        return []
    cols = {c.strip().lower() for c in header[0].split(",")}
    return sorted(cols & FORBIDDEN_COL_TOKENS)


def pack_open_h5(
    *,
    open_root: Path,
    out_h5: Path,
    profile: str = "core",
    osf_url: str = OSF_URL_PLACEHOLDER,
) -> Path:
    globs = CORE_GLOBS if profile == "core" else FULL_CSV_GLOBS
    files = list(iter_pack_files(open_root, globs))
    if not files:
        raise FileNotFoundError(f"No files matched profile={profile!r} under {open_root}")

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    if out_h5.exists():
        out_h5.unlink()

    phi_hits: list[str] = []
    with h5py.File(out_h5, "w") as h5:
        h5.attrs["schema_version"] = SCHEMA_VERSION
        h5.attrs["paper_citation"] = PAPER_CITATION
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["profile"] = profile
        h5.attrs["osf_url"] = osf_url
        h5.attrs["n_files"] = 0
        files_grp = h5.create_group("files")

        for path in files:
            rel = path.relative_to(open_root).as_posix()
            hits = _csv_has_forbidden_header(path)
            if hits:
                phi_hits.append(f"{rel}: {hits}")
                continue
            data = path.read_bytes()
            key = uuid.uuid4().hex
            ds = files_grp.create_dataset(
                key,
                data=np.frombuffer(data, dtype=np.uint8),
                compression="gzip",
                compression_opts=6,
            )
            ds.attrs["relpath"] = rel
            ds.attrs["content_type"] = path.suffix.lower().lstrip(".") or "bin"
            ds.attrs["sha256"] = _sha256_bytes(data)
            ds.attrs["nbytes"] = len(data)

        if phi_hits:
            raise RuntimeError(
                "Refusing to pack CSVs with demographic/PHI columns:\n  "
                + "\n  ".join(phi_hits[:20])
            )
        h5.attrs["n_files"] = len(files_grp)

    # File-level checksum recorded beside the archive
    digest = _sha256_file(out_h5)
    sidecar = out_h5.with_suffix(out_h5.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {out_h5.name}\n")
    with h5py.File(out_h5, "a") as h5:
        h5.attrs["sha256"] = digest
    print(f"Packed {len(files)} files -> {out_h5} ({out_h5.stat().st_size / 1e6:.1f} MB)")
    print(f"SHA256: {digest}")
    return out_h5


def unpack_open_h5(*, h5_path: Path, dest_root: Path) -> int:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    dest_root.mkdir(parents=True, exist_ok=True)
    n = 0
    with h5py.File(h5_path, "r") as h5:
        files_grp = h5["files"]
        for key in files_grp.keys():
            ds = files_grp[key]
            rel = str(ds.attrs["relpath"])
            out = dest_root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = bytes(ds[()])
            expected = str(ds.attrs.get("sha256", ""))
            if expected and _sha256_bytes(payload) != expected:
                raise ValueError(f"Checksum mismatch for {rel}")
            out.write_bytes(payload)
            n += 1
    print(f"Unpacked {n} files -> {dest_root}")
    return n


def check_open_h5(h5_path: Path) -> None:
    """Validate schema + no forbidden column headers in packed CSV payloads."""
    with h5py.File(h5_path, "r") as h5:
        assert "schema_version" in h5.attrs, "missing schema_version"
        assert "files" in h5, "missing /files group"
        bad: list[str] = []
        for key in h5["files"].keys():
            ds = h5["files"][key]
            rel = str(ds.attrs["relpath"])
            if not rel.lower().endswith(".csv"):
                continue
            text = bytes(ds[()]).decode("utf-8", errors="replace")
            header = text.splitlines()[:1]
            if not header:
                continue
            cols = {c.strip().lower() for c in header[0].split(",")}
            hit = cols & FORBIDDEN_COL_TOKENS
            if hit:
                bad.append(f"{rel}: {sorted(hit)}")
        if bad:
            raise AssertionError("PHI columns in packed HDF5:\n  " + "\n  ".join(bad))
        print(
            f"PASS check_open_h5 schema={h5.attrs.get('schema_version')} "
            f"n_files={h5.attrs.get('n_files')} profile={h5.attrs.get('profile')}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pack = sub.add_parser("pack", help="Write manuscript HDF5 from data/open/")
    p_pack.add_argument("--open-root", type=Path, default=OPEN)
    p_pack.add_argument("--out", type=Path, default=None)
    p_pack.add_argument("--profile", choices=["core", "full_csv"], default="core")
    p_pack.add_argument("--osf-url", default=OSF_URL_PLACEHOLDER)

    p_unpack = sub.add_parser("unpack", help="Expand HDF5 into data/open/")
    p_unpack.add_argument("--h5", type=Path, default=None)
    p_unpack.add_argument("--dest", type=Path, default=OPEN)

    p_check = sub.add_parser("check", help="Validate packed HDF5 schema / PHI")
    p_check.add_argument("--h5", type=Path, default=None)

    args = p.parse_args(argv)
    if args.cmd == "pack":
        out = args.out or default_h5_path()
        pack_open_h5(
            open_root=args.open_root,
            out_h5=out,
            profile=args.profile,
            osf_url=args.osf_url,
        )
        check_open_h5(out)
        return 0
    if args.cmd == "unpack":
        h5_path = args.h5 or default_h5_path()
        unpack_open_h5(h5_path=h5_path, dest_root=args.dest)
        return 0
    if args.cmd == "check":
        check_open_h5(args.h5 or default_h5_path())
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
