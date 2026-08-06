"""Discover penn_epilepsy scalar image manifest for factor score projection."""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import (
    DEFAULT_PENN_EPILEPSY_INCLUSION_CSV,
    DEFAULT_TRACTOMETRY_ROOT,
    FACTOR_SCORE_EXTRA_GROUPS,
)

QSIRECON_DIR = DEFAULT_TRACTOMETRY_ROOT / "derivatives" / "qsirecon"
METADATA_DIR = DEFAULT_TRACTOMETRY_ROOT / "data" / "metadata"
SCALAR_FILES_JSON = METADATA_DIR / "scalar_labels_to_filenames.json"
SCALAR_DIRS_JSON = METADATA_DIR / "scalar_labels_to_directories.json"
MNI_SUFFIX = "space-MNI152NLin2009cAsym"


@dataclass(frozen=True)
class SubjectSession:
    group: str
    sub: str
    ses: str | None

    @property
    def subject_id(self) -> str:
        return f"{self.sub}_{self.ses}" if self.ses else self.sub


def load_scalar_metadata() -> tuple[list[str], dict[str, str], dict[str, str]]:
    with open(SCALAR_FILES_JSON, encoding="utf-8") as f:
        scalar_to_file = json.load(f)
    with open(SCALAR_DIRS_JSON, encoding="utf-8") as f:
        scalar_to_dir = json.load(f)
    scalar_labels = list(scalar_to_file.keys())
    missing = [s for s in scalar_labels if s not in scalar_to_dir]
    if missing:
        raise ValueError(f"Scalars missing qsirecon directory metadata: {missing}")
    return scalar_labels, scalar_to_file, scalar_to_dir


def load_included_epilepsy_subs(
    inclusion_csv: Path = DEFAULT_PENN_EPILEPSY_INCLUSION_CSV,
) -> set[str]:
    if not inclusion_csv.is_file():
        raise FileNotFoundError(f"Missing penn_epilepsy inclusion CSV: {inclusion_csv}")
    df = pd.read_csv(inclusion_csv)
    if "sub" not in df.columns:
        raise ValueError(f"Expected 'sub' column in {inclusion_csv}")
    return set(df["sub"].astype(str).unique())


def discover_subject_sessions(
    groups: tuple[str, ...],
    scalar_to_dir: dict[str, str],
    *,
    sub_filter: set[str] | None = None,
) -> list[SubjectSession]:
    subjects: list[SubjectSession] = []
    scalar_dirs = list(dict.fromkeys(scalar_to_dir.values()))
    for group in groups:
        group_dir = QSIRECON_DIR / group
        if not group_dir.exists():
            continue
        base = None
        for scalar_dir in scalar_dirs:
            candidate = group_dir / "derivatives" / scalar_dir
            if candidate.exists():
                base = candidate
                break
        if base is None:
            base = group_dir
        for sub_dir in sorted(base.iterdir()):
            if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
                continue
            if sub_filter is not None and sub_dir.name not in sub_filter:
                continue
            for ses_dir in sorted(sub_dir.iterdir()):
                if ses_dir.is_dir() and ses_dir.name.startswith("ses-") and (ses_dir / "dwi").is_dir():
                    subjects.append(SubjectSession(group, sub_dir.name, ses_dir.name))
    return subjects


def scalar_path(
    subject: SubjectSession,
    scalar_directory: str,
    scalar_filename: str,
) -> str | None:
    group_dir = QSIRECON_DIR / subject.group
    if subject.ses is None:
        return None
    bases = [
        group_dir / "derivatives" / scalar_directory / subject.sub / subject.ses / "dwi",
        group_dir / subject.sub / subject.ses / "dwi",
    ]
    candidates: list[str] = []
    for base in bases:
        if not base.exists():
            continue
        for pattern in (
            f"*{MNI_SUFFIX}_{scalar_filename}.nii.gz",
            f"*{MNI_SUFFIX}_{scalar_filename}_dwimap.nii.gz",
            f"*{MNI_SUFFIX}*{scalar_filename}*.nii.gz",
        ):
            candidates.extend(glob.glob(str(base / pattern)))
    return str(Path(sorted(candidates)[0]).resolve()) if candidates else None


def build_epilepsy_manifest(
    scalar_labels: list[str] | None = None,
    *,
    inclusion_csv: Path = DEFAULT_PENN_EPILEPSY_INCLUSION_CSV,
) -> pd.DataFrame:
    """Return long-form manifest for included penn_epilepsy subjects with complete scalars."""
    labels, scalar_to_file, scalar_to_dir = load_scalar_metadata()
    if scalar_labels is not None:
        missing = [s for s in scalar_labels if s not in labels]
        if missing:
            raise ValueError(f"Requested scalars not in metadata: {missing}")
        labels = list(scalar_labels)

    included_subs = load_included_epilepsy_subs(inclusion_csv)
    subjects = discover_subject_sessions(
        FACTOR_SCORE_EXTRA_GROUPS,
        scalar_to_dir,
        sub_filter=included_subs,
    )

    rows: list[dict[str, str]] = []
    for subject in tqdm(subjects, desc="Validating penn_epilepsy manifest"):
        paths = {
            scalar: scalar_path(subject, scalar_to_dir[scalar], scalar_to_file[scalar])
            for scalar in labels
        }
        if any(path is None for path in paths.values()):
            continue
        for scalar, path in paths.items():
            rows.append(
                {
                    "subject": subject.subject_id,
                    "sub": subject.sub,
                    "ses": subject.ses or "",
                    "group": subject.group,
                    "scalar": scalar,
                    "path": str(path),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["subject", "sub", "ses", "group", "scalar", "path"],
    )
