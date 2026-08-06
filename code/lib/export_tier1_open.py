#!/usr/bin/env python3
"""Export de-identified open products into data/open/.

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
    "age",
    "sex",
    "gender",
    "bat",
    "batch",
    "scanner",
    "scanner_id",
    "site",
    "site_id",
    "split",
}
DROP_SUFFIXES = ("_pred", "_centile", "_unharm", "_unharm_pred")
DROP_EXACT_IF_NOT_Z = re.compile(r"^(?!.*_z$).+")  # unused; we keep explicitly

# Manuscript-facing analysis products only (see code/docs/analysis.md).
ANALYSIS_DIRS = [
    "factor_analysis",
    "factor_z-scores",
    "factor_representation",
    "factor_analysis_voxelwise",
    "gradients_group-controls",
    "gradients_tle_z",
    "tract_asymmetry",
    "tract_asymmetry_normative",
    "region_asymmetry_tle",
    "region_asymmetry_tle_normative",
    "2_microstructural_asymmetries",
    "microstructural_asymmetries",
    "qc",
    "covbat_example",
]

# Core OSF share: enough for golden tests + manuscript digests (no full GAM dump).
CORE_ANALYSIS_DIRS = [
    "factor_analysis",
    "factor_z-scores",
    "factor_representation",
    "gradients_group-controls",
    "gradients_tle_z",
    "tract_asymmetry",
    "2_microstructural_asymmetries",
    "microstructural_asymmetries",
]

CORE_OPEN_GLOBS: tuple[str, ...] = (
    "inclusion/*.csv",
    "metadata/*.json",
    "atlases/**/*.csv",
    "atlases/**/*.tsv",
    "atlases/**/*.json",
    "atlases/**/*.txt",
    "atlases/**/*.md",
    # factor_analysis: exclude brainmaps glasser correlation tables
    "analysis/factor_analysis/controls_All4_Combined_scalar_*.csv",
    "analysis/factor_analysis/controls_All4_Combined_pca_*.csv",
    "analysis/factor_analysis/subjects_included.csv",
    # factor z: controls + epilepsy cohort digests only
    "analysis/factor_z-scores/factor_z_scores/controls_F*_z_scores.csv",
    "analysis/factor_z-scores/factor_z_scores/epilepsy_F*_z_scores.csv",
    "analysis/factor_z-scores/factor_z_scores/epilepsy_mahalanobis.csv",
    # factor representation: one subject-level table
    "analysis/factor_representation/factor_matched_subject_similarity.csv",
    # LE gradients flattened under module root (2D only)
    "analysis/gradients_group-controls/*.csv",
    "analysis/gradients_tle_z/*.csv",
    # group asymmetry digests used by golden tests (not full subject trees)
    "analysis/microstructural_asymmetries/summary_hcp1065_thirds_mahalanobis.csv",
    "analysis/microstructural_asymmetries/summary_hcp1065_whole_scalars.csv",
    "analysis/microstructural_asymmetries/factor_score_z_ipsi_contra_cohens_d_summary.csv",
    "analysis/tract_asymmetry/**/*_asym_scalars.csv",
    # Full manuscript-allowlisted GAM residual-z trees
    "gam/pyafq/**/*_gam.csv",
    "gam/mni_micro/**/*_gam.csv",
)

# Gradient CSVs kept after flattening (2D subspace; drop lambdas / sorted / 3D).
GRADIENT_FLAT_KEEP_SUFFIXES: tuple[str, ...] = (
    "_principal_gradient1_scores_cohort-controls.csv",
    "_principal_gradient2_scores_cohort-controls.csv",
    "_factor_score_means_cohort-controls.csv",
)
GRADIENT_FLAT_KEEP_NAMES: frozenset[str] = frozenset(
    {
        "neuroaxis_correlations_cohort-controls.csv",
        "factor_z_correlations_cohort-epilepsy.csv",
        "epilepsy_F1_mean_z_scores.csv",
        "epilepsy_F2_mean_z_scores.csv",
        "epilepsy_F3_mean_z_scores.csv",
    }
)

CORE_GAM_FILES: tuple[tuple[str, str], ...] = (
    (
        "derivatives/gam/pyafq/HCP1065/ILF_L/ILF_L_dti_md_stat-mean_gam.csv",
        "gam/pyafq/HCP1065/ILF_L/ILF_L_dti_md_stat-mean_gam.csv",
    ),
)

# Rename on copy into open analysis/
ANALYSIS_RENAME = {
    "2_microstructural_asymmetries": "microstructural_asymmetries",
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
    if name.startswith("._"):
        return True
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
        cl = c.lower()
        if c in ("sub", "group", "anon_id"):
            keep.append(c)
            continue
        if c in DROP_COLS or cl in DROP_COLS:
            continue
        if any(c.endswith(suf) for suf in DROP_SUFFIXES):
            continue
        # keep residual z only
        if c.endswith("_z") or re.fullmatch(r"node\d+_z", c):
            keep.append(c)
            continue
    return keep


def export_gam_csv(src: Path, dst: Path, mapping: dict[str, str]) -> None:
    df = pd.read_csv(src)
    cols = gam_keep_columns(list(df.columns))
    if "sub" not in df.columns:
        # copy as-is without PHI-looking cols
        drop = {c for c in df.columns if c in DROP_COLS or c.lower() in DROP_COLS}
        drop |= {c for c in df.columns if any(c.endswith(s) for s in DROP_SUFFIXES)}
        df = df[[c for c in df.columns if c not in drop]]
    else:
        df = df[cols].copy()
        df["anon_id"] = df["sub"].map(lambda s: anon_id(str(s), mapping))
        df = df.drop(columns=["sub"])
        # anon_id first
        ordered = ["anon_id"] + [c for c in df.columns if c != "anon_id"]
        df = df[ordered]
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)


def export_gam_tree(
    src_root: Path,
    dst_root: Path,
    mapping: dict[str, str],
    workers: int = 16,
    *,
    force: bool = False,
) -> int:
    from lib.manuscript_features import gam_relpath_is_manuscript

    files = [
        p
        for p in src_root.rglob("*_gam.csv")
        if gam_relpath_is_manuscript(p.relative_to(src_root).as_posix())
    ]
    pending = []
    for src in files:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if not force and dst.exists() and dst.stat().st_size > 0:
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


def anonymize_path_part(part: str, mapping: dict[str, str]) -> str:
    """Anonymize BIDS ``sub-*`` directory names and ``sub-*_...`` filenames."""
    if re.fullmatch(r"sub-[A-Za-z0-9]+", part):
        return anon_id(part, mapping)
    m = re.match(r"^(sub-[A-Za-z0-9]+)(_.+)$", part)
    if m:
        return anon_id(m.group(1), mapping) + m.group(2)
    return part


def export_analysis_dir(
    name: str,
    mapping: dict[str, str],
    *,
    tabular_only: bool = False,
) -> None:
    src = SRC / "derivatives" / "analysis" / name
    if not src.exists():
        print(f"skip missing {name}")
        return
    out_name = ANALYSIS_RENAME.get(name, name)
    dst_root = OPEN / "analysis" / out_name
    files = [p for p in src.rglob("*") if p.is_file() and not is_skipped_file(p)]
    for src_f in tqdm(files, desc=f"analysis/{name}"):
        rel = src_f.relative_to(src)
        parts = [anonymize_path_part(part, mapping) for part in rel.parts]
        dst = dst_root.joinpath(*parts)
        suf = src_f.suffix.lower()
        if suf in {".csv", ".tsv"}:
            try:
                export_tabular(src_f, dst, mapping)
            except Exception as e:
                print(f"warn: {src_f}: {e}")
        elif suf in {".json"}:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst)
        elif tabular_only:
            continue
        elif src_f.stat().st_size > 25_000_000:
            continue
        elif suf in {".png", ".html", ".pdf", ".svg"}:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst)
        else:
            if src_f.stat().st_size < 5_000_000:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_f, dst)


def flatten_factor_analysis_all4() -> None:
    """Move ``factor_analysis/All4_Combined/*`` up to ``factor_analysis/``."""
    nested = OPEN / "analysis" / "factor_analysis" / "All4_Combined"
    dest = OPEN / "analysis" / "factor_analysis"
    if not nested.is_dir():
        return
    for path in nested.iterdir():
        if path.is_file():
            target = dest / path.name
            if target.exists():
                target.unlink()
            shutil.move(str(path), str(target))
    shutil.rmtree(nested, ignore_errors=True)
    print("flattened factor_analysis/All4_Combined/ -> factor_analysis/")


def _should_keep_flat_gradient(name: str) -> bool:
    if name in GRADIENT_FLAT_KEEP_NAMES:
        return True
    if "sorted" in name or "embedding_lambdas" in name:
        return False
    return any(name.endswith(suf) for suf in GRADIENT_FLAT_KEEP_SUFFIXES)


def flatten_open_gradient_csvs(module: str, *, prefer_subdir: str = "gradients-2") -> None:
    """Lift 2D LE CSVs to ``analysis/<module>/`` and drop nested trees."""
    root = OPEN / "analysis" / module
    if not root.exists():
        return
    nested_csv = root / "laplacian_eigenmodes" / "csv"
    sources: list[Path] = []
    prefer = nested_csv / prefer_subdir
    if prefer.is_dir():
        sources.extend(p for p in prefer.glob("*.csv") if p.is_file())
    if nested_csv.is_dir():
        sources.extend(p for p in nested_csv.glob("*.csv") if p.is_file())
    # Also gather epilepsy mean z etc. already under nested trees
    for path in root.rglob("*.csv"):
        if path.is_file() and path not in sources:
            sources.append(path)

    kept = 0
    for src in sources:
        if not _should_keep_flat_gradient(src.name):
            continue
        dst = root / src.name
        if src.resolve() == dst.resolve():
            kept += 1
            continue
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
        kept += 1

    # Remove nested method trees
    for child in list(root.iterdir()):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
    print(f"flattened {module}/ LE CSVs -> analysis/{module}/ ({kept} files)")


def fix_unnamed_index_columns() -> int:
    """Rename pandas ``Unnamed: 0`` index columns to meaningful labels."""
    label_by_name = {
        "controls_All4_Combined_scalar_correlations.csv": "scalar",
        "controls_All4_Combined_scalar_means.csv": "scalar",
        "controls_All4_Combined_scalar_uniquenesses.csv": "scalar",
        "controls_All4_Combined_pca_component_loadings.csv": "component",
    }
    fixed = 0
    for path in OPEN.rglob("*.csv"):
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "Unnamed: 0" not in df.columns:
            continue
        label = label_by_name.get(path.name)
        if label is None:
            # Heuristic for similarity matrices / generic index dumps
            if path.name.startswith("similarity_matrix_"):
                label = "statistic"
            elif df.columns[0] == "Unnamed: 0":
                label = "index"
            else:
                continue
        df = df.rename(columns={"Unnamed: 0": label})
        df.to_csv(path, index=False)
        fixed += 1
    print(f"renamed Unnamed: 0 columns in {fixed} open CSVs")
    return fixed


def export_factor_matched_subject_similarity(mapping: dict[str, str]) -> None:
    """Build consolidated factor-representation subject table into data/open/."""
    # Point factor_representation path helpers at controlled derivatives for inputs,
    # then write anonymized open product.
    from analysis.factor_representation import factor_representation as fr

    # Temporarily override dirs used by the representation module.
    controlled_fz = SRC / "derivatives" / "analysis" / "factor_z-scores"
    controlled_fa = SRC / "derivatives" / "analysis" / "factor_analysis"
    fr.FACTOR_DIR = str(controlled_fz / "factor_scores")
    fr.SCALAR_DIR = str(controlled_fz / "scalar_z-scores")
    # Prefer flattened All4 loadings if present in open (already anonymized-free).
    open_loadings = (
        OPEN
        / "analysis"
        / "factor_analysis"
        / "controls_All4_Combined_scalar_factor_loadings_ordered.csv"
    )
    src_loadings = (
        controlled_fa
        / "All4_Combined"
        / "controls_All4_Combined_scalar_factor_loadings_ordered.csv"
    )
    if not src_loadings.exists():
        src_loadings = (
            controlled_fa / "controls_All4_Combined_scalar_factor_loadings_ordered.csv"
        )
    loadings_path = open_loadings if open_loadings.exists() else src_loadings
    fr.LOADINGS_PATH = str(loadings_path)

    sample = pd.read_csv(Path(fr.FACTOR_DIR) / f"{fr.GROUP}_{fr.FACTORS[0]}_scores.csv")
    real_ids = fr._subject_id_series(sample).tolist()
    anon_ids = [anon_id(str(s), mapping) for s in real_ids]
    loadings = pd.read_csv(loadings_path).set_index("factor").loc[fr.FACTORS]
    factors, ref_cols = fr.load_factor_gradients()
    df = fr.build_factor_matched_subject_similarity(
        loadings=loadings,
        factors=factors,
        subject_ids=anon_ids,
        ref_cols=ref_cols,
    )
    if "subject" in df.columns and "anon_id" not in df.columns:
        df = df.rename(columns={"subject": "anon_id"})
    out_dir = OPEN / "analysis" / "factor_representation"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Drop per-model intermediates from open share
    for path in list(out_dir.glob("*.csv")):
        path.unlink()
    out = out_dir / fr.FACTOR_MATCHED_SUBJECT_CSV
    df.to_csv(out, index=False)
    print(f"wrote {out.relative_to(OPEN)} ({len(df)} rows)")


def finalize_core_open_layout(mapping: dict[str, str]) -> None:
    """Apply open-share layout rules after tabular export."""
    # Drop excluded brainmaps / non-kept trees early
    fa = OPEN / "analysis" / "factor_analysis"
    if fa.exists():
        for path in fa.glob("*brainmaps_glasser_correlations*.csv"):
            path.unlink(missing_ok=True)
    for drop in ("profile_thirds_example", "qc"):
        p = OPEN / "analysis" / drop
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    # Keep only selected factor_z files
    fz = OPEN / "analysis" / "factor_z-scores"
    keep_fz = {
        "controls_F1_z_scores.csv",
        "controls_F2_z_scores.csv",
        "controls_F3_z_scores.csv",
        "epilepsy_F1_z_scores.csv",
        "epilepsy_F2_z_scores.csv",
        "epilepsy_F3_z_scores.csv",
        "epilepsy_mahalanobis.csv",
    }
    if fz.exists():
        for path in fz.rglob("*.csv"):
            if path.name not in keep_fz:
                path.unlink(missing_ok=True)
        # Remove emptied factor_scores dir etc.
        for path in sorted(fz.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()

    flatten_open_gradient_csvs("gradients_group-controls")
    flatten_open_gradient_csvs("gradients_tle_z")
    export_factor_matched_subject_similarity(mapping)
    fix_unnamed_index_columns()


def prune_open_to_core_globs() -> int:
    """Keep only paths matching manuscript core globs under data/open/."""
    import fnmatch

    kept: set[Path] = set()
    for pattern in CORE_OPEN_GLOBS:
        for path in OPEN.glob(pattern):
            if path.is_file():
                kept.add(path.resolve())
    # Always keep README and .gitkeep placeholders
    for path in OPEN.rglob("*"):
        if path.is_file() and (
            path.name == "README.md"
            or path.name == ".gitkeep"
            or path.suffix.lower() in {".md"}
            and "atlases" in path.parts
        ):
            kept.add(path.resolve())

    removed = 0
    for path in list(OPEN.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() in kept:
            continue
        rel = path.relative_to(OPEN).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in CORE_OPEN_GLOBS):
            continue
        path.unlink()
        removed += 1
    # Remove empty dirs (bottom-up), keep top-level placeholders
    for path in sorted(OPEN.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            if path.name in {"atlases", "metadata", "gam", "analysis", "inclusion"}:
                continue
            path.rmdir()
    print(f"pruned {removed} non-core files under data/open/")
    return removed



def export_atlases() -> None:
    from lib.manuscript_features import MANUSCRIPT_WM_TRACT_SET

    # Not needed for open manuscript reproduction / goldens.
    skip_top_level = {"S-A_ArchetypalAxis", "thalamus"}

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
        if rel.parts and rel.parts[0] in skip_top_level:
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "HCP1065_tract_metadata.csv":
            df = pd.read_csv(path)
            if "label" in df.columns:
                df = df[df["label"].astype(str).isin(MANUSCRIPT_WM_TRACT_SET)].copy()
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            continue
        shutil.copy2(path, out)
    # centroids
    cent = SRC / "derivatives" / "atlas_centroids"
    if cent.exists():
        for path in cent.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".json"}:
                rel = path.relative_to(cent)
                if rel.parts and rel.parts[0] in skip_top_level:
                    continue
                out = OPEN / "atlases" / "centroids" / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)



def export_metadata() -> None:
    from lib.manuscript_features import MANUSCRIPT_SCALARS, filter_metadata_dict

    src = SRC / "data" / "metadata"
    dst = OPEN / "metadata"
    for name in [
        "scalar_labels_to_colors.json",
        "scalar_labels_to_filenames.json",
        "scalar_labels_to_human.json",
        "scalar_labels_to_directories.json",
    ]:
        p = src / name
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as fh:
            data = json.load(fh)
        filtered = filter_metadata_dict(data)
        if len(filtered) != len(MANUSCRIPT_SCALARS):
            missing = [s for s in MANUSCRIPT_SCALARS if s not in filtered]
            print(f"warn: {name} missing manuscript scalars: {missing}")
        dst.mkdir(parents=True, exist_ok=True)
        with (dst / name).open("w", encoding="utf-8") as fh:
            json.dump(filtered, fh, indent=2)
            fh.write("\n")


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


def export_core_gam(mapping: dict[str, str]) -> None:
    for src_rel, dst_rel in CORE_GAM_FILES:
        src = SRC / src_rel
        dst = OPEN / dst_rel
        if not src.exists():
            print(f"warn: missing core GAM {src}")
            continue
        export_gam_csv(src, dst, mapping)
        print(f"  wrote {dst_rel}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--core",
        action="store_true",
        help=(
            "Manuscript core OSF share: trimmed analysis digests + full "
            "manuscript-allowlisted GAM residual-z trees (pyafq + mni_micro)."
        ),
    )
    p.add_argument(
        "--gam-only",
        action="store_true",
        help="Only export/update GAM trees under data/open/gam/ (skip atlases/analysis).",
    )
    p.add_argument(
        "--force-gam",
        action="store_true",
        help="Overwrite existing open GAM CSVs instead of skipping them.",
    )
    args = p.parse_args(argv)

    mapping = load_or_init_map()
    OPEN.mkdir(parents=True, exist_ok=True)
    for sub in ["atlases", "metadata", "gam", "analysis", "inclusion"]:
        (OPEN / sub).mkdir(parents=True, exist_ok=True)

    def _export_full_gam() -> None:
        print("=== GAM pyafq (manuscript residual-z) ===")
        export_gam_tree(
            SRC / "derivatives" / "gam" / "pyafq",
            OPEN / "gam" / "pyafq",
            mapping,
            force=args.force_gam,
        )
        save_map(mapping)
        print("=== GAM mni_micro (manuscript residual-z) ===")
        export_gam_tree(
            SRC / "derivatives" / "gam" / "mni_micro",
            OPEN / "gam" / "mni_micro",
            mapping,
            force=args.force_gam,
        )
        save_map(mapping)

    if args.gam_only:
        _export_full_gam()
    elif args.core:
        print("=== atlases ===")
        export_atlases()
        print("=== metadata ===")
        export_metadata()
        print("=== inclusion ===")
        export_inclusion(mapping)
        save_map(mapping)
        _export_full_gam()
        print("=== analysis (core, tabular only) ===")
        for name in CORE_ANALYSIS_DIRS:
            export_analysis_dir(name, mapping, tabular_only=True)
            save_map(mapping)
        flatten_factor_analysis_all4()
        finalize_core_open_layout(mapping)
        prune_open_to_core_globs()
    else:
        print("=== atlases ===")
        export_atlases()
        print("=== metadata ===")
        export_metadata()
        print("=== inclusion ===")
        export_inclusion(mapping)
        save_map(mapping)
        _export_full_gam()
        print("=== analysis ===")
        for name in ANALYSIS_DIRS:
            export_analysis_dir(name, mapping)
            save_map(mapping)
        flatten_factor_analysis_all4()
        fix_unnamed_index_columns()

    save_map(mapping)
    print("=== done ===")
    print(f"anon map: {MAP_PATH} ({len(mapping)} subjects)")
    for sub in ["atlases", "metadata", "gam", "analysis", "inclusion"]:
        pth = OPEN / sub
        size = sum(f.stat().st_size for f in pth.rglob("*") if f.is_file())
        print(f"  {sub}: {size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(WORKSPACE / "code"))
    raise SystemExit(main())
