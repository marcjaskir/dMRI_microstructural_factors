#!/usr/bin/env python3
"""Pack / unpack manuscript-reproduction open products into a single HDF5 file.

Schema version 2 — path-mirrored groups under ``/open/…`` (same layout as
``data/open/`` after unpack), plus a root ``/catalog`` inventory.

Leaf datasets are gzip-compressed file bytes with attrs:
  relpath, content_type, sha256, nbytes

Usage:
  python -u code/lib/pack_open_h5.py pack [--profile core|full_csv]
  python -u code/lib/pack_open_h5.py unpack [--h5 PATH] [--dest data/open]
  python -u code/lib/pack_open_h5.py check [--h5 PATH]
  python -u code/lib/pack_open_h5.py ls [--h5 PATH] [--tree]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

WORKSPACE = Path(__file__).resolve().parents[2]
CODE = WORKSPACE / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import h5py
import numpy as np

from lib.manuscript_features import open_relpath_is_manuscript

OPEN = WORKSPACE / "data" / "open"
DEFAULT_H5_NAME = "dmri_microstructural_factors_open_v1.h5"
SCHEMA_VERSION = "2"
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
    # Factor analysis / z / representation (flat factor_analysis/; no All4_Combined/)
    "analysis/factor_analysis/**/*.csv",
    "analysis/factor_z-scores/factor_scores/*.csv",
    "analysis/factor_z-scores/factor_z_scores/*.csv",
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

_H5_NAME_BAD = re.compile(r"[^\w.\-+=@]+", re.UNICODE)


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
    if path.name.startswith("._"):
        return True
    lower = path.name.lower()
    for suf in SKIP_SUFFIXES:
        if lower.endswith(suf):
            return True
    if path.name == "anon_id_map.csv":
        return True
    if "diffusion_embedding" in path.parts:
        return True
    return False


def normalize_open_relpath(rel: str) -> str:
    """Drop legacy ``factor_analysis/All4_Combined/`` directory (combined FA only)."""
    return rel.replace(
        "analysis/factor_analysis/All4_Combined/",
        "analysis/factor_analysis/",
    )


def iter_pack_files(open_root: Path, globs: Iterable[str]) -> Iterator[Path]:
    seen: set[Path] = set()
    for pattern in globs:
        for path in sorted(open_root.glob(pattern)):
            if not path.is_file() or _should_skip(path):
                continue
            rel = normalize_open_relpath(path.relative_to(open_root).as_posix())
            if not open_relpath_is_manuscript(rel):
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


def _sanitize_h5_name(name: str) -> str:
    cleaned = _H5_NAME_BAD.sub("_", name).strip("._")
    return cleaned or "file"


def _dataset_name_for_file(path: Path, parent_group: h5py.Group) -> str:
    """File stem as dataset name; on collision append _{suffix}."""
    stem = _sanitize_h5_name(path.stem)
    suffix = _sanitize_h5_name(path.suffix.lstrip(".") or "bin")
    if stem not in parent_group:
        return stem
    candidate = f"{stem}_{suffix}"
    if candidate not in parent_group:
        return candidate
    n = 2
    while f"{candidate}_{n}" in parent_group:
        n += 1
    return f"{candidate}_{n}"


def _ensure_group(root: h5py.Group, parts: list[str]) -> h5py.Group:
    grp = root
    for part in parts:
        name = _sanitize_h5_name(part)
        if name not in grp:
            grp = grp.create_group(name)
        else:
            obj = grp[name]
            if isinstance(obj, h5py.Dataset):
                raise RuntimeError(
                    f"Cannot create group {name!r}: a dataset already exists at this path"
                )
            grp = obj
    return grp


def _iter_open_leaf_datasets(grp: h5py.Group) -> Iterator[h5py.Dataset]:
    for key in grp.keys():
        obj = grp[key]
        if isinstance(obj, h5py.Dataset):
            yield obj
        elif isinstance(obj, h5py.Group):
            yield from _iter_open_leaf_datasets(obj)


def _catalog_dtype() -> np.dtype:
    return np.dtype(
        [
            ("relpath", h5py.string_dtype(encoding="utf-8")),
            ("h5_path", h5py.string_dtype(encoding="utf-8")),
            ("nbytes", np.int64),
            ("sha256", h5py.string_dtype(encoding="utf-8")),
            ("content_type", h5py.string_dtype(encoding="utf-8")),
        ]
    )


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
    catalog_rows: list[tuple[str, str, int, str, str]] = []

    with h5py.File(out_h5, "w") as h5:
        h5.attrs["schema_version"] = SCHEMA_VERSION
        h5.attrs["paper_citation"] = PAPER_CITATION
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["profile"] = profile
        h5.attrs["osf_url"] = osf_url
        h5.attrs["n_files"] = 0
        open_grp = h5.create_group("open")

        for path in files:
            rel = normalize_open_relpath(path.relative_to(open_root).as_posix())
            hits = _csv_has_forbidden_header(path)
            if hits:
                phi_hits.append(f"{rel}: {hits}")
                continue
            data = path.read_bytes()
            parts = Path(rel).parts
            parent_parts = list(parts[:-1])
            parent = _ensure_group(open_grp, parent_parts)
            ds_name = _dataset_name_for_file(path, parent)
            ds = parent.create_dataset(
                ds_name,
                data=np.frombuffer(data, dtype=np.uint8),
                compression="gzip",
                compression_opts=6,
            )
            content_type = path.suffix.lower().lstrip(".") or "bin"
            digest = _sha256_bytes(data)
            ds.attrs["relpath"] = rel
            ds.attrs["content_type"] = content_type
            ds.attrs["sha256"] = digest
            ds.attrs["nbytes"] = len(data)
            h5_path = "/open/" + "/".join(
                [_sanitize_h5_name(p) for p in parent_parts] + [ds_name]
            )
            catalog_rows.append((rel, h5_path, len(data), digest, content_type))

        if phi_hits:
            raise RuntimeError(
                "Refusing to pack CSVs with demographic/PHI columns:\n  "
                + "\n  ".join(phi_hits[:20])
            )

        catalog_rows.sort(key=lambda r: r[0])
        cat_arr = np.array(catalog_rows, dtype=_catalog_dtype())
        h5.create_dataset("catalog", data=cat_arr)
        h5.attrs["n_files"] = len(catalog_rows)

    digest = _sha256_file(out_h5)
    sidecar = out_h5.with_suffix(out_h5.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {out_h5.name}\n")
    with h5py.File(out_h5, "a") as h5:
        h5.attrs["sha256"] = digest
    print(f"Packed {len(catalog_rows)} files -> {out_h5} ({out_h5.stat().st_size / 1e6:.1f} MB)")
    print(f"SHA256: {digest}")
    return out_h5


def _unpack_v1_files_group(h5: h5py.File, dest_root: Path) -> int:
    files_grp = h5["files"]
    n = 0
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
    return n


def _unpack_v2_open_tree(h5: h5py.File, dest_root: Path) -> int:
    open_grp = h5["open"]
    n = 0
    for ds in _iter_open_leaf_datasets(open_grp):
        rel = str(ds.attrs["relpath"])
        out = dest_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = bytes(ds[()])
        expected = str(ds.attrs.get("sha256", ""))
        if expected and _sha256_bytes(payload) != expected:
            raise ValueError(f"Checksum mismatch for {rel}")
        out.write_bytes(payload)
        n += 1
    return n


def unpack_open_h5(*, h5_path: Path, dest_root: Path) -> int:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    dest_root.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as h5:
        version = str(h5.attrs.get("schema_version", "1"))
        if "open" in h5:
            n = _unpack_v2_open_tree(h5, dest_root)
        elif "files" in h5:
            n = _unpack_v1_files_group(h5, dest_root)
        else:
            raise ValueError(f"Unrecognized HDF5 layout in {h5_path} (schema={version})")
    print(f"Unpacked {n} files -> {dest_root}")
    return n


def _payload_datasets_for_check(h5: h5py.File) -> Iterator[h5py.Dataset]:
    if "open" in h5:
        yield from _iter_open_leaf_datasets(h5["open"])
    elif "files" in h5:
        for key in h5["files"].keys():
            yield h5["files"][key]
    else:
        raise AssertionError("missing /open or /files group")


def check_open_h5(h5_path: Path) -> None:
    """Validate schema + no forbidden column headers in packed CSV payloads."""
    with h5py.File(h5_path, "r") as h5:
        assert "schema_version" in h5.attrs, "missing schema_version"
        version = str(h5.attrs["schema_version"])
        if version == "2":
            assert "open" in h5, "schema 2 missing /open group"
            assert "catalog" in h5, "schema 2 missing /catalog"
            assert "files" not in h5, "schema 2 must not contain /files"
        else:
            assert "files" in h5, "schema 1 missing /files group"

        bad: list[str] = []
        n = 0
        for ds in _payload_datasets_for_check(h5):
            n += 1
            rel = str(ds.attrs["relpath"])
            if "All4_Combined/" in rel.replace("\\", "/"):
                bad.append(f"{rel}: legacy All4_Combined/ path not allowed in new packs")
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
            raise AssertionError("Packed HDF5 validation failed:\n  " + "\n  ".join(bad))
        print(
            f"PASS check_open_h5 schema={h5.attrs.get('schema_version')} "
            f"n_files={h5.attrs.get('n_files')} profile={h5.attrs.get('profile')} "
            f"leaf_datasets={n}"
        )


def ls_open_h5(h5_path: Path, *, tree: bool = False) -> None:
    """Print catalog inventory or a compact group tree."""
    with h5py.File(h5_path, "r") as h5:
        version = str(h5.attrs.get("schema_version", "?"))
        print(
            f"{h5_path.name}  schema={version}  profile={h5.attrs.get('profile')}  "
            f"n_files={h5.attrs.get('n_files')}"
        )
        if "catalog" in h5:
            cat = h5["catalog"][()]
            if tree and "open" in h5:

                def _walk(grp: h5py.Group, prefix: str) -> None:
                    for key in sorted(grp.keys()):
                        obj = grp[key]
                        path = f"{prefix}/{key}"
                        if isinstance(obj, h5py.Dataset):
                            nbytes = int(obj.attrs.get("nbytes", obj.size))
                            print(f"{path}  ({nbytes} bytes, {obj.attrs.get('content_type', '?')})")
                        else:
                            print(f"{path}/")
                            _walk(obj, path)

                print("/open/")
                _walk(h5["open"], "/open")
            else:
                print(f"{'relpath':<90} {'nbytes':>10}  content")
                for row in cat:
                    rel = row["relpath"]
                    if isinstance(rel, bytes):
                        rel = rel.decode()
                    ctype = row["content_type"]
                    if isinstance(ctype, bytes):
                        ctype = ctype.decode()
                    print(f"{rel:<90} {int(row['nbytes']):>10}  {ctype}")
        elif "files" in h5:
            print("(schema v1 — listing relpath attrs)")
            for key in sorted(h5["files"].keys(), key=lambda k: str(h5["files"][k].attrs["relpath"])):
                ds = h5["files"][key]
                print(f"{ds.attrs['relpath']}  ({ds.attrs.get('nbytes')} bytes)")
        else:
            raise SystemExit("No /catalog or /files in archive")


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

    p_ls = sub.add_parser("ls", help="List catalog / tree of packed files")
    p_ls.add_argument("--h5", type=Path, default=None)
    p_ls.add_argument("--tree", action="store_true", help="Print /open group tree")

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
    if args.cmd == "ls":
        ls_open_h5(args.h5 or default_h5_path(), tree=args.tree)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
