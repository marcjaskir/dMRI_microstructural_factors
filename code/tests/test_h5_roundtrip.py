#!/usr/bin/env python3
"""HDF5 pack → unpack round-trip digests match frozen open baselines."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_CODE = Path(__file__).resolve().parents[1]
REPO = REPO_CODE.parent
if str(REPO_CODE) not in sys.path:
    sys.path.insert(0, str(REPO_CODE))

from lib.pack_open_h5 import check_open_h5, pack_open_h5, unpack_open_h5  # noqa: E402


DIGEST_RELPATHS = [
    "analysis/factor_analysis/controls_All4_Combined_scalar_factor_loadings.csv",
    "analysis/gradients_group-controls/laplacian_eigenmodes/csv/neuroaxis_correlations_cohort-controls.csv",
    "analysis/microstructural_asymmetries/summary_hcp1065_thirds_mahalanobis.csv",
    "analysis/factor_z-scores/factor_z_scores/controls_F1_z_scores.csv",
]


def _sha256_text(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_h5_roundtrip_digests() -> None:
    open_root = REPO / "data" / "open"
    missing = [p for p in DIGEST_RELPATHS if not (open_root / p).exists()]
    if missing:
        print("SKIP HDF5 round-trip (missing open sources):")
        for m in missing:
            print(f"  {m}")
        return

    before = {rel: _sha256_text(open_root / rel) for rel in DIGEST_RELPATHS}
    with tempfile.TemporaryDirectory(prefix="dmri_open_h5_") as tmp:
        tmp_path = Path(tmp)
        h5_path = tmp_path / "open_core.h5"
        unpack_root = tmp_path / "unpacked"
        pack_open_h5(open_root=open_root, out_h5=h5_path, profile="core")
        check_open_h5(h5_path)
        import h5py

        with h5py.File(h5_path, "r") as h5:
            assert str(h5.attrs["schema_version"]) == "2"
            assert "open" in h5 and "catalog" in h5
            assert "files" not in h5
        unpack_open_h5(h5_path=h5_path, dest_root=unpack_root)
        after = {rel: _sha256_text(unpack_root / rel) for rel in DIGEST_RELPATHS}
    assert before == after, (
        "Round-trip digest mismatch:\n"
        + json.dumps({"before": before, "after": after}, indent=2)
    )
    print(f"PASS HDF5 round-trip ({len(DIGEST_RELPATHS)} tables, schema=2)")


if __name__ == "__main__":
    test_h5_roundtrip_digests()
