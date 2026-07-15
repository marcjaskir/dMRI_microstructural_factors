import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
Generate tle_subjects_table.tex from penn_epilepsy_included_basic_metadata_tle.csv.

Reports by Left TLE and Right TLE: N, Age (mean ± sd), Female, MRI Lesional
(with MTS, FCD, Stroke, TBI, Neoplasia), and surgical outcomes.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_CSV = PROJECT_ROOT / "results" / "inclusion" / "penn_epilepsy_included_basic_metadata_tle.csv"
OUTPUT_TEX = PROJECT_ROOT / "results" / "inclusion" / "tle_subjects_table.tex"

LESION_SUBS = [
    ("lesion_mts", "MTS"),
    ("lesion_fcd", "FCD")
]


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
    df = pd.read_csv(INPUT_CSV)
    df["laterality"] = df["laterality"].astype(str).str.strip().str.lower()
    df = df[df["laterality"].isin(("left", "right"))]

    left = df[df["laterality"] == "left"]
    right = df[df["laterality"] == "right"]
    n_left, n_right = len(left), len(right)

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

    def lesional_str(g: pd.DataFrame) -> str:
        n = len(g)
        if "lesion_status" not in g.columns:
            return "---"
        les = (g["lesion_status"].astype(str).str.strip().str.lower() == "lesional").sum()
        return _n_pct(int(les), n)

    def lesion_sub_str(g: pd.DataFrame, col: str) -> str:
        n = len(g)
        if col not in g.columns:
            return "---"
        v = pd.to_numeric(g[col], errors="coerce").fillna(0)
        return _n_pct(int((v >= 0.5).sum()), n)

    def outcome_str(g: pd.DataFrame, outcome_val: str) -> str:
        n = len(g)
        if "outcome" not in g.columns:
            return "---"
        o = g["outcome"].astype(str).str.strip().str.lower()
        return _n_pct(int((o == outcome_val.lower()).sum()), n)

    left_age = age_str(left)
    right_age = age_str(right)
    left_female = female_str(left)
    right_female = female_str(right)
    left_lesional = lesional_str(left)
    right_lesional = lesional_str(right)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{TLE subject characteristics by laterality.}",
        "\\label{tab:tle-subjects}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        " & \\textbf{Left TLE} & \\textbf{Right TLE} \\\\",
        "\\midrule",
        f"Number of subjects & {n_left} & {n_right} \\\\",
        f"Age in years, mean (range) & {left_age} & {right_age} \\\\",
        f"Female, $n$ (\\%) & {left_female} & {right_female} \\\\",
        "\\midrule",
        f"MRI Lesional, $n$ (\\%) & {left_lesional} & {right_lesional} \\\\",
    ]
    for col, label in LESION_SUBS:
        lines.append(f"\\quad {label}, $n$ (\\%) & {lesion_sub_str(left, col)} & {lesion_sub_str(right, col)} \\\\")
    lines.extend([
        "\\midrule",
        "Surgical outcome, $n$ (\\%) & & \\\\",
        f"\\quad Good & {outcome_str(left, 'good')} & {outcome_str(right, 'good')} \\\\",
        f"\\quad Bad & {outcome_str(left, 'bad')} & {outcome_str(right, 'bad')} \\\\",
        f"\\quad Unknown & {outcome_str(left, 'unknown')} & {outcome_str(right, 'unknown')} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text("\n".join(lines))
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
