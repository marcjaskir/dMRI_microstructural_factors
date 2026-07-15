import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
Apply Penn epilepsy patient inclusion criteria and output included subject list.

Inclusion (per structural_tractometry_context.md):
1. Patient-specific qsirecon subdirectory exists under
   derivatives/qsirecon/penn_epilepsy
and
2a. "Aggregate lateralization: Strength" from primary_hypotheses_penn_epilepsy.csv
    is in ["Strongly lateralizing", "Moderately lateralizing"]
    (use attending (_a) when available, else fellow (_f)),
or
2b. Subject is missing from primary_hypotheses_penn_epilepsy.csv but received a
    surgical intervention based on clinical_penn_epilepsy.csv:
    at least one intervention-detail column is non-empty and none equal
    "no_intervention".
"""
from pathlib import Path

import pandas as pd

# Paths (project root = structural_tractometry)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRIMARY_HYPOTHESES_CSV = PROJECT_ROOT / "derivatives" / "metadata" / "primary_hypotheses_penn_epilepsy.csv"
CLINICAL_CSV = PROJECT_ROOT / "derivatives" / "metadata" / "clinical_penn_epilepsy.csv"
DEMO_CSV = PROJECT_ROOT / "derivatives" / "metadata" / "demo_penn_epilepsy.csv"
QSIRECON_PENN_DIR = PROJECT_ROOT / "derivatives" / "qsirecon" / "penn_epilepsy"
OUTPUT_DIR = PROJECT_ROOT / "results" / "inclusion"
OUTPUT_CSV = OUTPUT_DIR / "penn_epilepsy_included.csv"
OUTPUT_METADATA_CSV = OUTPUT_DIR / "penn_epilepsy_included_extended_metadata.csv"
OUTPUT_METADATA_BASIC_CSV = OUTPUT_DIR / "penn_epilepsy_included_basic_metadata.csv"
OUTPUT_METADATA_BASIC_TLE_CSV = OUTPUT_DIR / "penn_epilepsy_included_basic_metadata_tle.csv"
OUTPUT_EXCLUDED_CSV = OUTPUT_DIR / "penn_epilepsy_excluded_with_reason.csv"

# Values that map to "left" / "right" for laterality from primary_hypotheses
LAT_REG_LEFT = {"Left seizures only", "Bilateral independent seizures (Left > Right)"}
LAT_REG_RIGHT = {"Right seizures only", "Bilateral independent seizures (Right > Left)"}

ALLOWED_LAT_STRENGTH = {"Strongly lateralizing", "Moderately lateralizing"}
# Exclude from primary path if attending or fellow lateralization is this (bilateral equal)
EXCLUDED_LAT = "Bilateral independent seizures (Left = Right)"
INTERVENTION_COLS = [
    "intervention_laterality",
    "intervention_lobe",
    "intervention_type",
    "resection_histopathology",
    "resection_details",
    "ablation_target",
]


def _lat_strength_value(row: pd.Series) -> str | None:
    """Prefer attending (_a), fallback to fellow (_f). Return None if both missing/invalid."""
    a = row.get("aggr_lat_streng_post_a")
    f = row.get("aggr_lat_streng_post_f")
    for val in (a, f):
        if pd.notna(val) and str(val).strip():
            return str(val).strip()
    return None


def _has_unilateral_intervention(row: pd.Series) -> bool:
    """
    At least one non-empty intervention field, none equal 'no_intervention' (case-insensitive),
    and intervention_laterality must have a value (unilateral).
    """
    # Require intervention_laterality to be non-empty and not "no_intervention"
    lat = row.get("intervention_laterality")
    if pd.isna(lat) or not str(lat).strip() or str(lat).strip().lower() == "no_intervention":
        return False
    values = []
    for col in INTERVENTION_COLS:
        if col not in row:
            continue
        v = row.get(col)
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s:
            continue
        values.append(s)
    if not values:
        return False
    return all(s.lower() != "no_intervention" for s in values)


def _derive_laterality(row: pd.Series) -> str | None:
    """
    Derive laterality as "left" or "right" from available sources.
    Priority: aggr_lat_reg_post_a → aggr_lat_reg_post_f → intervention_laterality.
    Returns None if no usable value.
    """
    # Primary hypotheses: attending then fellow
    for col in ("aggr_lat_reg_post_a", "aggr_lat_reg_post_f"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            s = str(val).strip()
            if s in LAT_REG_LEFT:
                return "left"
            if s in LAT_REG_RIGHT:
                return "right"
    # Clinical: intervention_laterality
    val = row.get("intervention_laterality")
    if pd.notna(val) and str(val).strip():
        s = str(val).strip().lower()
        if s == "left":
            return "left"
        if s == "right":
            return "right"
    return None


def _derive_laterality_strength(row: pd.Series) -> str:
    """
    Laterality strength: prefer aggr_lat_streng_post_a, then aggr_lat_streng_post_f.
    If both missing, infer from seizure_lateralization: "left > right" / "right > left"
    -> moderately; "left" / "right" -> strongly.
    Always returns: "moderately", "strongly", or "unknown".
    """
    for col in ("aggr_lat_streng_post_a", "aggr_lat_streng_post_f"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            sval = str(val).strip().lower()
            if "moderate" in sval:
                return "moderately"
            if "strong" in sval:
                return "strongly"
    # Fallback: seizure_lateralization
    slat = row.get("seizure_lateralization")
    if pd.notna(slat) and str(slat).strip():
        s = str(slat).strip().lower()
        if s in ("left > right", "right > left"):
            return "moderately"
        if s in ("left", "right"):
            return "strongly"
    return "unknown"


def _derive_lobe(row: pd.Series) -> str:
    """
    Lobe: prefer aggr_loc_reg_post_a, then aggr_loc_reg_post_f, then intervention_lobe.
    Return "unknown" if none available.
    """
    for col in ("aggr_loc_reg_post_a", "aggr_loc_reg_post_f", "intervention_lobe"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            return str(val).strip().lower()
    return "unknown"


def _is_good_outcome_ilae(val) -> bool:
    """True if ILAE category indicates good outcome (1, 1a, or 2)."""
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s in ("1", "1a", "2")


def _is_good_outcome_engel(val) -> bool:
    """True if Engel class is I (e.g. IA, IB, IC, ID)."""
    if pd.isna(val):
        return False
    s = str(val).strip().upper()
    if not s:
        return False
    # Engel I: single letter I or I + A/B/C/D
    return s == "I" or (s.startswith("I") and len(s) >= 2 and s[1] in "ABCD")


def _derive_outcome(row: pd.Series) -> str:
    """
    Outcome: good vs bad. ILAE 1/2 or Engel I = good.
    Priority: ILAE over Engel; use latest followup with data.
    Order: ilae_category_3 → ilae_category_2 → ilae_category_pecclinical
           → engel_followup3 → engel_followup2 → engel_followup1.
    """
    # ILAE columns first (prioritize ILAE over Engel)
    ilae_cols = [
        "ilae_category_3_pecclinical",
        "ilae_category_2_pecclinical",
        "ilae_category_pecclinical",
    ]
    for col in ilae_cols:
        if col not in row:
            continue
        val = row[col]
        if pd.notna(val) and str(val).strip():
            return "good" if _is_good_outcome_ilae(val) else "bad"
    # Engel columns
    engel_cols = ["engel_followup3", "engel_followup2", "engel_followup1"]
    for col in engel_cols:
        if col not in row:
            continue
        val = row[col]
        if pd.notna(val) and str(val).strip():
            return "good" if _is_good_outcome_engel(val) else "bad"
    return "unknown"


def build_basic_metadata(meta_included: pd.DataFrame) -> pd.DataFrame:
    """
    Build a reduced metadata table with derived columns for included subjects.

    Output columns:
    - sub, sex, age
    - epilepsy_duration: age - age_onset when both exist, else NA
    - laterality: "left" or "right" from aggr_lat_reg (attending/fellow) or intervention_laterality
    - laterality_strength: from aggr_lat_streng (attending then fellow), else "unknown"
    - lobe: from aggr_loc_reg (attending then fellow) or intervention_lobe, else "unknown"
    - outcome: "good" (ILAE 1/2 or Engel I), "bad", or "unknown"; uses latest followup, ILAE over Engel
    - lesion_status: "lesional" if mri_result == "lesion_potential_epileptogenic" or any lesion_* == 1, else "nonlesional"
    - lesion_tbi, lesion_stroke, lesion_fcd, lesion_neoplasia, lesion_mts: pass-through from clinical (0/1); lesion_other excluded
    """
    basic = pd.DataFrame()
    basic["sub"] = meta_included["sub"]

    # --- Sex and age (pass through) ---
    basic["sex"] = meta_included["sex"] if "sex" in meta_included.columns else pd.NA
    basic["age"] = meta_included["age"] if "age" in meta_included.columns else pd.NA

    # --- Epilepsy duration: age - age_onset when both exist, else NA ---
    if "age" in meta_included.columns and "age_onset" in meta_included.columns:
        age_num = pd.to_numeric(meta_included["age"], errors="coerce")
        age_onset_num = pd.to_numeric(meta_included["age_onset"], errors="coerce")
        duration = age_num - age_onset_num
        basic["epilepsy_duration"] = duration.where(
            age_num.notna() & age_onset_num.notna()
        ).round(1)
    else:
        basic["epilepsy_duration"] = pd.NA

    # --- Laterality strength: attending then fellow aggr_lat_streng ---
    basic["laterality_strength"] = meta_included.apply(_derive_laterality_strength, axis=1)

    # --- Laterality: left/right from primary then intervention ---
    basic["laterality"] = meta_included.apply(_derive_laterality, axis=1)

    # --- Lobe: aggr_loc_reg (attending/fellow) then intervention_lobe ---
    basic["lobe"] = meta_included.apply(_derive_lobe, axis=1)

    # --- Outcome: good/bad/unknown from ILAE/Engel, latest followup, ILAE preferred ---
    basic["outcome"] = meta_included.apply(_derive_outcome, axis=1)

    # --- Lesion status: lesional if mri_result == "lesion_potential_epileptogenic" or any lesion type == 1 ---
    lesion_type_cols = ["lesion_tbi", "lesion_stroke", "lesion_fcd", "lesion_neoplasia", "lesion_mts", "lesion_other"]

    def _lesion_status(row: pd.Series) -> str:
        if pd.notna(row.get("mri_result")) and str(row.get("mri_result")).strip() == "lesion_potential_epileptogenic":
            return "lesional"
        for col in lesion_type_cols:
            if col not in row:
                continue
            val = row[col]
            if pd.notna(val) and (val == 1 or str(val).strip() == "1"):
                return "lesional"
        return "nonlesional"

    if "mri_result" in meta_included.columns or any(c in meta_included.columns for c in lesion_type_cols):
        basic["lesion_status"] = meta_included.apply(_lesion_status, axis=1)
    else:
        basic["lesion_status"] = pd.NA

    # --- Pass through lesion variables (0/1 from clinical); exclude lesion_other ---
    lesion_passthrough_cols = ["lesion_tbi", "lesion_stroke", "lesion_fcd", "lesion_neoplasia", "lesion_mts"]
    for col in lesion_passthrough_cols:
        if col in meta_included.columns:
            basic[col] = meta_included[col]

    return basic


def main() -> None:
    primary_df = pd.read_csv(PRIMARY_HYPOTHESES_CSV)

    # Get all subjects that appear in primary_hypotheses metadata
    primary_all_subs = set(
        primary_df["sub"].astype(str).str.strip()
        .replace("", pd.NA)
        .dropna()
    )

    # Require non-empty sub in primary_hypotheses metadata
    df = primary_df[primary_df["sub"].notna() & (primary_df["sub"].astype(str).str.strip() != "")].copy()

    # Exclude rows where attending or fellow lateralization is EXCLUDED_LAT (bilateral equal)
    def _has_excluded_lat(row: pd.Series) -> bool:
        for col in ("aggr_lat_reg_post_a", "aggr_lat_reg_post_f"):
            val = row.get(col)
            if pd.notna(val) and str(val).strip() == EXCLUDED_LAT:
                return True
        return False
    df = df[~df.apply(_has_excluded_lat, axis=1)].copy()

    # Lateralization strength: prefer _a, then _f; must be in allowed set
    lat_streng = df.apply(_lat_strength_value, axis=1)
    df = df[lat_streng.isin(ALLOWED_LAT_STRENGTH)].copy()

    # Primary path: require qsirecon subdirectory to exist
    primary_included: set[str] = set()
    for sub in df["sub"].astype(str).str.strip().unique():
        sub_dir = QSIRECON_PENN_DIR / sub
        if sub_dir.is_dir():
            primary_included.add(sub)

    # Clinical-only path: subjects missing from primary_hypotheses but with valid intervention
    clinical_df = pd.read_csv(CLINICAL_CSV)
    clinical_df = clinical_df[clinical_df["sub"].notna() & (clinical_df["sub"].astype(str).str.strip() != "")].copy()
    clinical_df["sub"] = clinical_df["sub"].astype(str).str.strip()

    clinical_missing = clinical_df[~clinical_df["sub"].isin(primary_all_subs)].copy()
    if not clinical_missing.empty:
        has_intervention = clinical_missing.apply(_has_unilateral_intervention, axis=1)
        clinical_missing = clinical_missing[has_intervention].copy()

    clinical_included: set[str] = set()
    for sub in clinical_missing["sub"].unique():
        sub_dir = QSIRECON_PENN_DIR / sub
        if sub_dir.is_dir():
            clinical_included.add(sub)

    all_included = sorted(primary_included.union(clinical_included))
    included_set = set(all_included)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write included subjects to penn_epilepsy_included.csv
    pd.DataFrame({"sub": all_included}).to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} ({len(all_included)} subjects)")

    # Merge clinical + primary_hypotheses metadata for included subjects
    primary_clean = primary_df[
        primary_df["sub"].notna() & (primary_df["sub"].astype(str).str.strip() != "")
    ].copy()
    primary_clean["sub"] = primary_clean["sub"].astype(str).str.strip()
    primary_clean = primary_clean.drop_duplicates(subset=["sub"], keep="first")
    clinical_dedup = clinical_df.drop_duplicates(subset=["sub"], keep="first")

    meta_included = pd.DataFrame({"sub": all_included})
    meta_included = meta_included.merge(clinical_dedup, on="sub", how="left")
    meta_included = meta_included.merge(
        primary_clean, on="sub", how="left", suffixes=("", "_ph")
    )
    # Drop duplicate columns from primary (e.g. if both had 'sub' we keep one)
    meta_included = meta_included.loc[:, ~meta_included.columns.duplicated()]

    # Fill missing age from demo_penn_epilepsy.csv
    if "age" in meta_included.columns and DEMO_CSV.exists():
        demo_df = pd.read_csv(DEMO_CSV)
        if "sub" in demo_df.columns and "age" in demo_df.columns:
            demo_df["sub"] = demo_df["sub"].astype(str).str.strip()
            demo_age = demo_df.set_index("sub")["age"].dropna()
            missing_age = meta_included["age"].isna()
            if missing_age.any():
                for idx in meta_included.index[missing_age]:
                    sub = meta_included.loc[idx, "sub"]
                    if sub in demo_age.index and pd.notna(demo_age.loc[sub]):
                        meta_included.loc[idx, "age"] = demo_age.loc[sub]

    meta_included.to_csv(OUTPUT_METADATA_CSV, index=False)
    print(f"Wrote {OUTPUT_METADATA_CSV} ({len(meta_included)} rows)")

    # penn_epilepsy_metadata_included_basic.csv: derived basic columns (sub, sex, age, epilepsy_duration, laterality, laterality_strength, lobe, outcome)
    basic_df = build_basic_metadata(meta_included)
    basic_df.to_csv(OUTPUT_METADATA_BASIC_CSV, index=False)
    print(f"Wrote {OUTPUT_METADATA_BASIC_CSV} ({len(basic_df)} rows)")

    # penn_epilepsy_included_basic_metadata_tle.csv: same as basic but only lobe == "temporal"
    if "lobe" in basic_df.columns:
        basic_tle = basic_df[
            basic_df["lobe"].astype(str).str.strip().str.lower() == "temporal"
        ].copy()
        basic_tle.to_csv(OUTPUT_METADATA_BASIC_TLE_CSV, index=False)
        print(f"Wrote {OUTPUT_METADATA_BASIC_TLE_CSV} ({len(basic_tle)} rows)")

    # penn_epilepsy_excluded_with_reason.csv: sub, exclusion_reason (detailed)
    all_considered = set(
        primary_clean["sub"].tolist() + clinical_df["sub"].astype(str).str.strip().tolist()
    )
    excluded_subs = sorted(all_considered - included_set)
    exclusion_reasons = []
    for sub in excluded_subs:
        has_qsirecon = (QSIRECON_PENN_DIR / sub).is_dir()
        in_primary = sub in primary_all_subs
        if not has_qsirecon:
            reason = (
                "No qsirecon derivatives"
            )
        elif in_primary:
            reason = (
                "Aggregate lateralization strength is not strongly or moderately lateralizing"
            )
        else:
            reason = (
                "No primary hypothesis available and clinical record does not meet unilateral intervention criteria"
            )
        exclusion_reasons.append(reason)
    excluded_df = pd.DataFrame({"sub": excluded_subs, "exclusion_reason": exclusion_reasons})
    excluded_df.to_csv(OUTPUT_EXCLUDED_CSV, index=False)
    print(f"Wrote {OUTPUT_EXCLUDED_CSV} ({len(excluded_df)} subjects)")


if __name__ == "__main__":
    main()
