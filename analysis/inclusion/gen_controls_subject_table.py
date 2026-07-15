import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
Generate controls_subjects_table.tex with summary of control groups.

Reads control subject data from the GAM CSV (same source as controls_inclusion.py).
Reports per group: Number of subjects, Age in years mean (range), Female n (%).
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GAM_CSV = PROJECT_ROOT / "derivatives" / "gam" / "pyafq" / "HCP1065" / "AF_L" / "AF_L_dki_ad_gam.csv"
OUTPUT_TEX = PROJECT_ROOT / "results" / "inclusion" / "controls_subjects_table.tex"

CONTROL_GROUPS = ["hcpya", "hcpaging", "penn_controls"]
GROUP_LABELS = {"hcpya": "HCP-YA", "hcpaging": "HCP-Aging", "penn_controls": "Penn controls"}


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
    df = pd.read_csv(GAM_CSV, usecols=["sub", "group", "age", "sex"])
    df = df[df["group"].isin(CONTROL_GROUPS)].copy()
    df = df.drop_duplicates(subset=["sub", "group"], keep="first")

    def age_str(g: pd.DataFrame) -> str:
        age = pd.to_numeric(g["age"], errors="coerce").dropna()
        if len(age) == 0:
            return "---"
        m = age.mean()
        lo, hi = age.min(), age.max()
        return f"{_to_int(m)} ({_to_int(lo)}--{_to_int(hi)})"

    def female_str(g: pd.DataFrame) -> str:
        n = len(g)
        f = (g["sex"].astype(str).str.strip().str.upper() == "F").sum()
        return _n_pct(int(f), n)

    cols = [GROUP_LABELS[grp] for grp in CONTROL_GROUPS]
    n_cols = len(CONTROL_GROUPS)
    col_spec = "l" + "c" * n_cols

    header = " & ".join(["", *[f"\\textbf{{{c}}}" for c in cols]]) + " \\\\"
    n_rows = [str(len(df[df["group"] == grp])) for grp in CONTROL_GROUPS]
    age_rows = [age_str(df[df["group"] == grp]) for grp in CONTROL_GROUPS]
    female_rows = [female_str(df[df["group"] == grp]) for grp in CONTROL_GROUPS]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Control subject characteristics by group.}",
        "\\label{tab:controls-subjects}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
        "Number of subjects & " + " & ".join(n_rows) + " \\\\",
        "Age in years, mean (range) & " + " & ".join(age_rows) + " \\\\",
        "Female, $n$ (\\%) & " + " & ".join(female_rows) + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text("\n".join(lines))
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
