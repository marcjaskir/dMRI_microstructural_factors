import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
"""
Process REDCap primary hypotheses export: rename/recode/remove columns and save
to derivatives for the Penn epilepsy cohort.
"""
from pathlib import Path

import pandas as pd

# Paths
INPUT_CSV = PROJECT_ROOT / "data" / "metadata" / "primary_hypotheses_redcap.csv"
OUTPUT_CSV = PROJECT_ROOT / "derivatives" / "metadata" / "primary_hypotheses_penn_epilepsy.csv"

# record_id: Remove this column

# cnt_rid: Change variable name to sub, add "sub-RIDXXXX" prefix (zero-padded to 4 digits)

# aggr_lat_reg_post_a / aggr_lat_reg_post_f:
AGGR_LAT_REG = {
    0: "Diffuse/non-lateralizing seizures",
    1: "Left seizures only",
    2: "Right seizures only",
    3: "Bilateral independent seizures (Left > Right)",
    4: "Bilateral independent seizures (Right > Left)",
    5: "Bilateral independent seizures (Left = Right)",
}

# aggr_lat_streng_post_a / aggr_lat_streng_post_f:
AGGR_LAT_STRENG = {
    0: "Non-lateralizing",
    1: "Weakly lateralizing",
    2: "Moderately lateralizing",
    3: "Strongly lateralizing",
}

# aggr_loc_reg_post_a / aggr_loc_reg_post_f:
AGGR_LOC_REG = {
    0: "Diffuse/multifocal",
    1: "Frontal",
    2: "Temporal",
    3: "Parietal",
    4: "Occipital",
    5: "Insular",
    6: "Cingulate",
    7: "Other (please specify)",
}

# aggr_loc_reg_spec_post_a / aggr_loc_reg_spec_post_f: Free text — no recoding

# aggr_loc_streng_post_a / aggr_loc_streng_post_f:
AGGR_LOC_STRENG = {
    0: "Non-localizing",
    1: "Weakly localizing",
    2: "Moderately localizing",
    3: "Strongly localizing",
}


def _recode(series: pd.Series, code_to_label: dict) -> pd.Series:
    """Map numeric codes to labels; leave NaN and unknown codes as-is (NaN)."""
    def map_val(val):
        if pd.isna(val) or str(val).strip() == "":
            return pd.NA
        try:
            k = int(float(val))
            return code_to_label.get(k, pd.NA)
        except (ValueError, TypeError):
            return pd.NA
    return series.apply(map_val)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    # If cnt_rid appears multiple times, keep only the row with the highest record_id
    if "cnt_rid" in df.columns and "record_id" in df.columns:
        # Coerce record_id to numeric so max works; rows that don't convert become NaN and sort last
        df["_record_id_num"] = pd.to_numeric(df["record_id"], errors="coerce")
        df = df.sort_values(["cnt_rid", "_record_id_num"], na_position="first")
        df = df.drop_duplicates(subset=["cnt_rid"], keep="last")
        df = df.drop(columns=["_record_id_num"])
        df = df.sort_index().reset_index(drop=True)

    # Remove record_id
    if "record_id" in df.columns:
        df = df.drop(columns=["record_id"])

    # cnt_rid -> sub: values as "sub-RIDXXXX" (zero-padded 4 digits)
    if "cnt_rid" in df.columns:
        def to_sub(val):
            if pd.isna(val) or str(val).strip() == "":
                return ""
            try:
                n = int(float(val))
                return f"sub-RID{n:04d}"
            except (ValueError, TypeError):
                return ""

        df["sub"] = df["cnt_rid"].apply(to_sub)
        df = df.drop(columns=["cnt_rid"])
        # Put sub first
        cols = ["sub"] + [c for c in df.columns if c != "sub"]
        df = df[cols]

    # Recode numeric variables to labels
    for col in ["aggr_lat_reg_post_a", "aggr_lat_reg_post_f"]:
        if col in df.columns:
            df[col] = _recode(df[col], AGGR_LAT_REG)
    for col in ["aggr_lat_streng_post_a", "aggr_lat_streng_post_f"]:
        if col in df.columns:
            df[col] = _recode(df[col], AGGR_LAT_STRENG)
    for col in ["aggr_loc_reg_post_a", "aggr_loc_reg_post_f"]:
        if col in df.columns:
            df[col] = _recode(df[col], AGGR_LOC_REG)
    for col in ["aggr_loc_streng_post_a", "aggr_loc_streng_post_f"]:
        if col in df.columns:
            df[col] = _recode(df[col], AGGR_LOC_STRENG)

    # Remove rows where sub is empty
    df = df[df["sub"].notna()]
    df = df[df["sub"] != ""]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
