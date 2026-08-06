#!/usr/bin/env python3
"""Download the OSF-hosted manuscript-reproduction HDF5 and unpack into data/open/.

Default HDF5 location (put the downloaded file here if fetching manually)::

  <data_open_dir>/dmri_microstructural_factors_open_v1.h5
  # usually:  <repo>/data/open/dmri_microstructural_factors_open_v1.h5

Unpack expands ``gam/`` and ``analysis/`` CSVs beside that file (gitignored).
Those trees are not separate downloads and are not committed to git.

Set the download URL via (first match wins):
  1. CLI ``--url``
  2. env ``DMRI_MICRO_OSF_URL``
  3. ``config.yaml`` key ``open_h5_osf_url``
  4. placeholder (fails with a clear message until the OSF project exists)

Usage:
  python -u code/lib/fetch_open_data.py
  python -u code/lib/fetch_open_data.py --url https://osf.io/download/XXXXX/
  python -u code/lib/fetch_open_data.py --unpack-only
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
CODE = WORKSPACE / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from lib.pack_open_h5 import (  # noqa: E402
    DEFAULT_H5_NAME,
    OSF_URL_PLACEHOLDER,
    check_open_h5,
    unpack_open_h5,
)
from lib.paths import open_dir, open_h5_path  # noqa: E402


def resolve_osf_url(cli_url: str | None) -> str:
    if cli_url:
        return cli_url.strip()
    env = os.environ.get("DMRI_MICRO_OSF_URL", "").strip()
    if env:
        return env
    from lib.paths import load_config

    cfg = load_config()
    cfg_url = str(cfg.get("open_h5_osf_url") or "").strip()
    if cfg_url and "XXXXX" not in cfg_url:
        return cfg_url
    return OSF_URL_PLACEHOLDER


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> Path:
    if "XXXXX" in url or url.startswith("OSF_URL="):
        raise SystemExit(
            "OSF URL not configured. Set open_h5_osf_url in config.yaml, "
            "export DMRI_MICRO_OSF_URL, or pass --url after uploading the HDF5.\n"
            f"Expected file location: {dest}\n"
            f"Expected file name: {DEFAULT_H5_NAME}\n"
            f"Placeholder: {OSF_URL_PLACEHOLDER}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"  → {dest}")
    urllib.request.urlretrieve(url, dest)
    digest = _sha256_file(dest)
    print(f"Downloaded {dest.stat().st_size / 1e6:.1f} MB  SHA256={digest}")
    sidecar = dest.with_suffix(dest.suffix + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text().split()[0]
        if expected != digest:
            raise SystemExit(f"SHA256 mismatch: got {digest}, expected {expected}")
        print("PASS sha256 matches sidecar")
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", default=None, help="Direct OSF download URL")
    p.add_argument(
        "--h5",
        type=Path,
        default=None,
        help=f"Local HDF5 path (default: {DEFAULT_H5_NAME} under data/open/)",
    )
    p.add_argument("--dest", type=Path, default=None, help="Unpack root (default data/open)")
    p.add_argument("--unpack-only", action="store_true", help="Skip download; unpack existing HDF5")
    p.add_argument("--download-only", action="store_true", help="Download without unpacking")
    args = p.parse_args(argv)

    h5_path = (args.h5 or open_h5_path()).resolve()
    dest = (args.dest or open_dir()).resolve()

    print(f"HDF5 path: {h5_path}")
    print(f"Unpack root: {dest}  (creates gam/ and analysis/ here)")

    if not args.unpack_only:
        url = resolve_osf_url(args.url)
        download(url, h5_path)
    elif not h5_path.is_file():
        raise SystemExit(
            f"HDF5 not found at {h5_path}.\n"
            f"Download from OSF into that path, or pass --h5 / set open_h5_path.\n"
            f"Expected file name: {DEFAULT_H5_NAME}"
        )

    if args.download_only:
        print(
            "Download-only done. Unpack later with: "
            "python -u code/lib/fetch_open_data.py --unpack-only"
        )
        return 0

    check_open_h5(h5_path)
    unpack_open_h5(h5_path=h5_path, dest_root=dest)
    print(f"Unpacked into {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
