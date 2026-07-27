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
Extract control-group subject lists from GAM CSV and write per-group included CSVs.

Reads sub, group, age, sex, bat from the GAM CSV; writes {group}_included.csv for each
control group (hcpya, hcpaging, penn_controls). Prints summary stats per group and,
for HCP-Aging, a frequency table for bat.
"""
from pathlib import Path

import pandas as pd

GAM_CSV = gam_dir() / "pyafq" / "HCP1065" / "AF_L" / "AF_L_dki_ad_gam.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "inclusion"

CONTROL_GROUPS = ["hcpya", "hcpaging", "penn_controls"]


def _to_int(x):
    if pd.isna(x):
        return None
    try:
        return int(round(float(x)))
    except (ValueError, TypeError):
        return None


def _n_pct(n: int, total: int) -> str:
    if total == 0:
        return "0 (0)"
    pct = round(100 * n / total)
    return f"{n} ({pct})"


def main() -> None:
    df = pd.read_csv(GAM_CSV, usecols=["sub", "group", "age", "sex", "bat"])
    df = df[df["group"].isin(CONTROL_GROUPS)].copy()
    # One row per subject (GAM CSV may have multiple rows per sub if multiple sessions/batches)
    df = df.drop_duplicates(subset=["sub", "group"], keep="first")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for group in CONTROL_GROUPS:
        g = df[df["group"] == group]
        subs = g["sub"].astype(str).str.strip().unique()
        n = len(subs)

        # Write {group}_included.csv
        out_path = OUTPUT_DIR / f"{group}_included.csv"
        pd.DataFrame({"sub": sorted(subs)}).to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({n} subjects)")

        # Summary stats (use first row per sub for age/sex; bat only for hcpaging)
        age = pd.to_numeric(g["age"], errors="coerce").dropna()
        if len(age) > 0:
            mean_age = age.mean()
            lo, hi = age.min(), age.max()
            age_str = f"{_to_int(mean_age)} ({_to_int(lo)}--{_to_int(hi)})"
        else:
            age_str = "—"
        female = (g["sex"].astype(str).str.strip().str.upper() == "F").sum()
        female_str = _n_pct(int(female), n)

        print(f"  {group}: N = {n}, Age (mean, range) years = {age_str}, Female = {female_str}")

        if group == "hcpaging" and "bat" in g.columns and not g["bat"].isna().all():
            print("  HCP-Aging bat frequency:")
            bat_counts = g["bat"].value_counts().sort_index()
            for bat_val, count in bat_counts.items():
                print(f"    bat {bat_val}: {count}")
        print()

    return None


if __name__ == "__main__":
    main()
