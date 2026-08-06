#!/usr/bin/env python3
"""Guard: tracked publishable paths must not contain PHI / reversible IDs."""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "code"

SUB_RE = re.compile(r"\bsub-\d+\b", re.IGNORECASE)
RID_RE = re.compile(r"\bRID\d{3,}\b")
FORBIDDEN_COLS = {"age", "sex", "bat", "anon_id_map"}
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".txt", ".csv", ".tsv", ".json", ".ipynb", ".sh"}


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "ls-files"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: walk code/ + data/open small trees
        paths: list[Path] = []
        for root in (CODE, REPO / "data" / "open"):
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if p.is_file():
                    paths.append(p)
        return paths
    return [REPO / line for line in out.splitlines() if line.strip()]


def _is_textish(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def test_no_anon_id_map_tracked() -> None:
    tracked = {p.relative_to(REPO).as_posix() for p in _tracked_files()}
    assert "data/controlled/anon_id_map.csv" not in tracked
    assert not any(p.endswith("anon_id_map.csv") for p in tracked)


def test_no_sub_or_rid_in_tracked_text() -> None:
    offenders: list[str] = []
    for path in _tracked_files():
        if not path.exists() or not _is_textish(path):
            continue
        # Allow docs that mention the pattern as forbidden examples
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith("code/tests/"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if SUB_RE.search(text) or RID_RE.search(text):
            # Narrow: real BIDS-like IDs with digits only after sub-
            for m in SUB_RE.finditer(text):
                offenders.append(f"{rel}: {m.group(0)}")
            for m in RID_RE.finditer(text):
                offenders.append(f"{rel}: {m.group(0)}")
    # Filter common false positives in comments about exclusion
    filtered = [
        o
        for o in offenders
        if "EXAMPLE" not in o.upper() and "placeholder" not in o.lower()
    ]
    assert not filtered, "Possible PHI identifiers in tracked files:\n  " + "\n  ".join(
        filtered[:50]
    )


def test_open_csvs_lack_demographics() -> None:
    open_dir = REPO / "data" / "open"
    if not open_dir.exists():
        return
    bad: list[str] = []
    for path in open_dir.rglob("*.csv"):
        # Skip gitignored large trees if somehow present — still check committed ones
        try:
            rel = path.relative_to(REPO).as_posix()
        except ValueError:
            continue
        with path.open(newline="", encoding="utf-8", errors="replace") as fh:
            try:
                reader = csv.reader(fh)
                header = next(reader, [])
            except Exception:
                continue
        cols = {c.strip().lower() for c in header}
        hit = cols & FORBIDDEN_COLS
        if hit:
            bad.append(f"{rel}: {sorted(hit)}")
    assert not bad, "Demographic columns in open CSVs:\n  " + "\n  ".join(bad)


def test_packed_h5_lacks_demographics() -> None:
    """If the manuscript HDF5 is present, run the packager schema/PHI check."""
    candidates = [
        REPO / "data" / "open" / "dmri_microstructural_factors_open_v1.h5",
        REPO / "data" / "open" / "dmri_microstructural_factors_open.h5",
    ]
    h5_path = next((p for p in candidates if p.exists()), None)
    if h5_path is None:
        print("SKIP packed HDF5 PHI check (archive not present)")
        return
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    from lib.pack_open_h5 import check_open_h5

    check_open_h5(h5_path)


if __name__ == "__main__":
    test_no_anon_id_map_tracked()
    test_no_sub_or_rid_in_tracked_text()
    test_open_csvs_lack_demographics()
    test_packed_h5_lacks_demographics()
    print("PASS PHI guard")
