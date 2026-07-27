#!/usr/bin/env python3
"""Export de-identified Tier 1 products into data/open/.

Reads from structural_tractometry (or config-controlled roots) and writes:
  data/open/{atlases,metadata,gam,analysis,inclusion}

- Maps real subject IDs -> anon_id consistently
- Keeps only residual z columns from GAM CSVs (drops age/sex/bat/unharm/…)
- Skips NIfTI / large voxelwise volumes
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_MAP_LOCK = threading.Lock()

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = Path("/mnt/sauce/littlab/users/mjaskir/structural_tractometry")
OPEN = WORKSPACE / "data" / "open"
MAP_PATH = WORKSPACE / "data" / "controlled" / "anon_id_map.csv"  # local only / gitignored

DROP_COLS = {
    "age", "sex", "bat", "split",
}
DROP_SUFFIXES = ("_pred", "_centile", "_unharm", "_unharm_pred")
DROP_EXACT_IF_NOT_Z = re.compile(r"^(?!.*_z$).+")  # unused; we keep explicitly

ANALYSIS_DIRS = [
    "factor_analysis",
    "factor_z-scores",
    "factor_representation",
    "gradients_group-controls",
    "gradients_tle_z",
    "tract_asymmetry",
    "tract_asymmetry_normative",
    "region_asymmetry_tle",
    "region_asymmetry_tle_normative",
    "asymmetry_tle",
    "asymmetry_tle_covbat_pyafq",
    "asymmetry_tle_region",
    "2_microstructural_asymmetries",
    "3_asymmetry_correlations",
    "5_within_patient_scalar_asymmetries",
    "profile_thirds_example",
    "qc",
    "covbat_example",
]

# Rename on copy into open analysis/
ANALYSIS_RENAME = {
    "2_microstructural_asymmetries": "microstructural_asymmetries",
    "3_asymmetry_correlations": "asymmetry_correlations",
    "5_within_patient_scalar_asymmetries": "within_patient_scalar_asymmetries",
}

SKIP_SUFFIXES = {".nii", ".nii.gz", ".gii", ".mgz", ".trk", ".tck", ".h5", ".sif", ".mat", ".mif", ".fib"}
SKIP_NAME_PARTS = {".ipyniivue_cache", "__pycache__", ".ipynb_checkpoints"}


def anon_id(sub: str, mapping: dict[str, str]) -> str:
    sub = str(sub).strip()
    if not sub:
        return sub
    with _MAP_LOCK:
        if sub not in mapping:
            # stable hash-based ID (not reversible without map)
            digest = hashlib.sha256(sub.encode()).hexdigest()[:10]
            mapping[sub] = f"anon_{digest}"
        return mapping[sub]


def load_or_init_map() -> dict[str, str]:
    if MAP_PATH.exists():
        df = pd.read_csv(MAP_PATH)
        return dict(zip(df["sub"].astype(str), df["anon_id"].astype(str)))
    return {}


def save_map(mapping: dict[str, str]) -> None:
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sub": list(mapping.keys()), "anon_id": list(mapping.values())}).sort_values(
        "sub"
    ).to_csv(MAP_PATH, index=False)


def is_skipped_file(path: Path) -> bool:
    name = path.name
    if any(part in path.parts for part in SKIP_NAME_PARTS):
        return True
    lower = name.lower()
    for suf in SKIP_SUFFIXES:
        if lower.endswith(suf):
            return True
    if lower.endswith(".png") or lower.endswith(".html") or lower.endswith(".pdf"):
        # keep small reports optionally; skip large html dumps if > 20MB later
        return False
    return False


def gam_keep_columns(columns: list[str]) -> list[str]:
    keep = []
    for c in columns:
        if c in ("sub", "group", "anon_id"):
            keep.append(c)
            continue
        if c in DROP_COLS:
            continue
        if any(c.endswith(suf) for suf in DROP_SUFFIXES):
            continue
        # keep residual z and raw feature only if *_z
        if c.endswith("_z") or re.fullmatch(r"node\d+_z", c):
            keep.append(c)
            continue
        # mni_micro sometimes has bare scalar name as observed — drop non-z
    return keep


def export_gam_csv(src: Path, dst: Path, mapping: dict[str, str]) -> None:
    df = pd.read_csv(src)
    cols = gam_keep_columns(list(df.columns))
    if "sub" not in df.columns:
        # copy as-is without PHI-looking cols
        df = df[[c for c in df.columns if c not in DROP_COLS and not any(c.endswith(s) for s in DROP_SUFFIXES)]]
    else:
        df = df[cols].copy()
        df["anon_id"] = df["sub"].map(lambda s: anon_id(str(s), mapping))
        df = df.drop(columns=["sub"])
        # anon_id first
        ordered = ["anon_id"] + [c for c in df.columns if c != "anon_id"]
        df = df[ordered]
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)


def export_gam_tree(src_root: Path, dst_root: Path, mapping: dict[str, str], workers: int = 16) -> int:
    files = list(src_root.rglob("*_gam.csv"))
    pending = []
    for src in files:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if dst.exists() and dst.stat().st_size > 0:
            continue
        pending.append((src, dst))

    def _one(pair: tuple[Path, Path]) -> None:
        export_gam_csv(pair[0], pair[1], mapping)

    n_done = len(files) - len(pending)
    if n_done:
        print(f"  skip {n_done} existing under {dst_root.name}")
    n = n_done
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, pair) for pair in pending]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"GAM {src_root.name}"):
            fut.result()
            n += 1
    return n


def rewrite_dataframe_ids(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        cl = col.lower()
        if cl in {"sub", "subject", "participant_id", "participant", "id"} or cl.endswith("_sub"):
            df[col] = df[col].astype(str).map(lambda s: anon_id(s, mapping) if s and s != "nan" else s)
            if cl == "sub" or cl == "subject":
                df = df.rename(columns={col: "anon_id"})
    # drop demographics if present
    for c in list(df.columns):
        if c.lower() in {"age", "sex", "gender", "bat", "scanner"}:
            df = df.drop(columns=[c])
    return df


def export_tabular(src: Path, dst: Path, mapping: dict[str, str]) -> None:
    if src.suffix.lower() == ".csv":
        df = pd.read_csv(src)
        df = rewrite_dataframe_ids(df, mapping)
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst, index=False)
    elif src.suffix.lower() in {".tsv"}:
        df = pd.read_csv(src, sep="\t")
        df = rewrite_dataframe_ids(df, mapping)
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst, sep="\t", index=False)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def export_analysis_dir(name: str, mapping: dict[str, str]) -> None:
    src = SRC / "derivatives" / "analysis" / name
    if not src.exists():
        print(f"skip missing {name}")
        return
    out_name = ANALYSIS_RENAME.get(name, name)
    dst_root = OPEN / "analysis" / out_name
    files = [p for p in src.rglob("*") if p.is_file() and not is_skipped_file(p)]
    for src_f in tqdm(files, desc=f"analysis/{name}"):
        rel = src_f.relative_to(src)
        # anonymize subject directory names
        parts = []
        for part in rel.parts:
            if part.startswith("sub-"):
                parts.append(anon_id(part, mapping))
            else:
                parts.append(part)
        dst = dst_root.joinpath(*parts)
        if src_f.suffix.lower() in {".csv", ".tsv"}:
            try:
                export_tabular(src_f, dst, mapping)
            except Exception as e:
                print(f"warn: {src_f}: {e}")
        elif src_f.suffix.lower() in {".json"}:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst)
        elif src_f.stat().st_size > 25_000_000:
            # skip huge html/png
            continue
        elif src_f.suffix.lower() in {".png", ".html", ".pdf", ".svg"}:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst)
        else:
            # other small text-like
            if src_f.stat().st_size < 5_000_000:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_f, dst)


def export_atlases() -> None:
    src = SRC / "data" / "atlases"
    dst = OPEN / "atlases"
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        if is_skipped_file(path):
            continue
        if path.suffix.lower() not in {".tsv", ".csv", ".json", ".txt", ".md"}:
            continue
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
    # centroids
    cent = SRC / "derivatives" / "atlas_centroids"
    if cent.exists():
        for path in cent.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".json"}:
                rel = path.relative_to(cent)
                out = OPEN / "atlases" / "centroids" / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)


def export_metadata() -> None:
    src = SRC / "data" / "metadata"
    dst = OPEN / "metadata"
    for name in [
        "scalar_labels_to_colors.json",
        "scalar_labels_to_filenames.json",
        "scalar_labels_to_human.json",
        "scalar_labels_to_directories.json",
    ]:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)


def export_inclusion(mapping: dict[str, str]) -> None:
    src = SRC / "results" / "1_inclusion"
    dst = OPEN / "inclusion"
    # cohort lists
    for name in [
        "penn_epilepsy_included.csv",
        "penn_controls_included.csv",
        "hcpya_included.csv",
        "hcpaging_included.csv",
    ]:
        p = src / name
        if not p.exists():
            continue
        raw = pd.read_csv(p, header=None)
        # single-column subject lists (with or without header-looking first row)
        col0 = raw.iloc[:, 0].astype(str)
        if col0.iloc[0].lower() in {"sub", "subject", "participant"}:
            col0 = col0.iloc[1:]
        out = pd.DataFrame({"anon_id": [anon_id(s, mapping) for s in col0.tolist() if s and s != "nan"]})
        out.to_csv(dst / name, index=False)

    basic = src / "penn_epilepsy_included_basic_metadata.csv"
    if basic.exists():
        df = pd.read_csv(basic)
        keep = [c for c in df.columns if c.lower() in {"sub", "laterality", "lobe", "laterality_strength", "group"}]
        # always keep laterality/lobe; drop age/sex/outcome/lesion
        prefer = []
        for c in df.columns:
            cl = c.lower()
            if cl in {"sub", "laterality", "lobe", "laterality_strength"}:
                prefer.append(c)
        df = df[prefer]
        df = rewrite_dataframe_ids(df, mapping)
        df.to_csv(dst / "penn_epilepsy_included_basic_metadata.csv", index=False)


def main() -> int:
    mapping = load_or_init_map()
    OPEN.mkdir(parents=True, exist_ok=True)
    for sub in ["atlases", "metadata", "gam", "analysis", "inclusion"]:
        (OPEN / sub).mkdir(parents=True, exist_ok=True)

    print("=== atlases ===")
    export_atlases()
    print("=== metadata ===")
    export_metadata()
    print("=== inclusion ===")
    export_inclusion(mapping)
    save_map(mapping)

    print("=== GAM pyafq ===")
    export_gam_tree(SRC / "derivatives" / "gam" / "pyafq", OPEN / "gam" / "pyafq", mapping)
    save_map(mapping)
    print("=== GAM mni_micro ===")
    export_gam_tree(SRC / "derivatives" / "gam" / "mni_micro", OPEN / "gam" / "mni_micro", mapping)
    save_map(mapping)

    print("=== analysis ===")
    for name in ANALYSIS_DIRS:
        export_analysis_dir(name, mapping)
        save_map(mapping)

    save_map(mapping)
    # summarize
    print("=== done ===")
    print(f"anon map: {MAP_PATH} ({len(mapping)} subjects)")
    for sub in ["atlases", "metadata", "gam", "analysis", "inclusion"]:
        p = OPEN / sub
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        print(f"  {sub}: {size/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(WORKSPACE / "code"))
    raise SystemExit(main())
