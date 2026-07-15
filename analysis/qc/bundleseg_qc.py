#!/usr/bin/env python3
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
"""
BundleSeg QC

Quality control for tract segmentation outputs under
derivatives/bundleseg/{group}/{sub}/.

QC measures:
1. Tract segmentation failures — A segmentation is a failure if there is no .trk
   file for that subject–tract pair. Summarized (1) by group and (2) across groups.
2. Poor tract segmentation — A tract is considered poor if it has 5 or fewer
   streamlines (from trk.header['nb_streamlines']). Summarized (1) by group
   and (2) across groups.
"""

from collections import defaultdict
from pathlib import Path

import nibabel as nib
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, leave=True):
        return iterable

PROJECT_ROOT = Path("{project_root()}")
BUNDLESEG_DIR = PROJECT_ROOT / "derivatives" / "bundleseg"
OUTPUT_DIR = PROJECT_ROOT / "results" / "qc" / "bundleseg"
PENN_EPILEPSY_INCLUSION_CSV = PROJECT_ROOT / "results" / "inclusion" / "penn_epilepsy_included_basic_metadata.csv"
POOR_STREAMLINES_THRESHOLD = 5  # tract with <= this many streamlines is "poor"

# Publication table: column order and display names (dir name -> display)
TABLE_GROUP_ORDER = [
    "penn_epilepsy",
    "penn_controls",
    "hcpya",
    "hcpaging",
]
TABLE_GROUP_DISPLAY = {
    "penn_epilepsy": "Epilepsy (Penn)",
    "penn_controls": "Controls (Penn)",
    "hcpya": "HCP-YA",
    "hcpaging": "HCP-Aging",
}


def get_streamline_count(trk_path: Path) -> int | None:
    """Return nb_streamlines from .trk file header, or None if unreadable."""
    try:
        trk = nib.streamlines.load(str(trk_path))
        return int(trk.header["nb_streamlines"])
    except Exception:
        return None


def get_failures_by_group(
    bundleseg_dir: Path,
    groups: list[str],
    subjects_by_group: dict[str, list[str]],
    expected_tracts_by_group: dict[str, list[str]],
) -> dict[str, list[tuple[str, str]]]:
    """Return dict group -> list of (sub, tract) that are missing .trk.
    Uses per-group expected tracts.
    """
    failures_by_group = defaultdict(list)
    for g in tqdm(groups, desc="Groups"):
        for sub in tqdm(subjects_by_group[g], desc=f"Subjects ({g})", leave=False):
            sub_dir = bundleseg_dir / g / sub
            for tract in expected_tracts_by_group[g]:
                trk_path = sub_dir / f"{tract}.trk"
                if not trk_path.exists():
                    failures_by_group[g].append((sub, tract))
    return dict(failures_by_group)


def get_poor_by_group(
    bundleseg_dir: Path,
    groups: list[str],
    subjects_by_group: dict[str, list[str]],
    expected_tracts_by_group: dict[str, list[str]],
    threshold: int = 5,
) -> dict[str, list[tuple[str, str, int]]]:
    """Return dict group -> list of (sub, tract, n_streamlines) where n_streamlines <= threshold."""
    poor_by_group = defaultdict(list)
    for g in groups:
        for sub in tqdm(subjects_by_group[g], desc=f"Poor QC ({g})", leave=False):
            sub_dir = bundleseg_dir / g / sub
            for tract in expected_tracts_by_group[g]:
                trk_path = sub_dir / f"{tract}.trk"
                if not trk_path.exists():
                    continue
                n = get_streamline_count(trk_path)
                if n is not None and n <= threshold:
                    poor_by_group[g].append((sub, tract, n))
    return dict(poor_by_group)


def _latex_escape(s: str) -> str:
    """Escape special characters for LaTeX."""
    if not isinstance(s, str):
        s = str(s)
    repl = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "^": "\\textasciicircum{}",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _build_tract_group_table(
    tracts: list[str],
    table_groups: list[str],
    counts_by_group_tract: dict[str, dict[str, int]],
    n_subjects_by_group: dict[str, int],
    expected_tracts_by_group: dict[str, list[str]],
) -> list[list[str]]:
    """Build table rows: [Tract, col1, col2, ...] with cells as 'N (pct%)' or '—'."""
    rows = []
    for tract in tracts:
        row = [_latex_escape(tract)]
        for g in table_groups:
            if g not in n_subjects_by_group or tract not in expected_tracts_by_group.get(g, []):
                row.append("—")
                continue
            n_sub = n_subjects_by_group[g]
            count = counts_by_group_tract.get(g, {}).get(tract, 0)
            if n_sub == 0:
                row.append("—")
            else:
                pct = 100.0 * count / n_sub
                row.append(f"{count} ({pct:.1f}\\%)")
        rows.append(row)
    return rows


def _longtable_col_spec(n_cols: int, *, tract_col_fraction: float = 0.28) -> str:
    """Column spec matching microstructural-asymmetry mapping tables (full width, tight)."""
    n_data = max(n_cols - 1, 1)
    tract_w = tract_col_fraction
    data_w = (0.98 - tract_w) / n_data
    spec = rf"@{{}}>{{\raggedright\arraybackslash}}p{{{tract_w:.2f}\linewidth}}"
    spec += "".join(
        rf">{{\raggedleft\arraybackslash}}p{{{data_w:.2f}\linewidth}}" for _ in range(n_data)
    )
    return spec + r"@{}"

def _write_publication_tex(
    out_path: Path,
    rows: list[list[str]],
    col_headers: list[str],
    caption: str,
    label: str,
    table_groups: list[str] | None = None,
    n_by_group: dict[str, int] | None = None,
) -> None:
    """Write a compact booktabs longtable (full linewidth, footnotesize, wrapping tract column)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_cols = len(col_headers)
    col_spec = _longtable_col_spec(n_cols)

    header1_parts = [rf"\mbox{{{_latex_escape(h)}}}" for h in col_headers]
    header1 = " & ".join(header1_parts) + r" \\"
    header_block = [r"\toprule", header1]
    if table_groups is not None and n_by_group is not None:
        header2_parts = [""] + [
            rf"\mbox{{N={n_by_group.get(g, 0)}}}" for g in table_groups
        ]
        header_block.append(" & ".join(header2_parts) + r" \\")
    header_block.append(r"\midrule")

    body_lines: list[str] = []
    for row in rows:
        if not row:
            continue
        # Tract label: allow line breaks; numeric cells stay compact (raggedleft p columns).
        cells = [row[0]] + [rf"\mbox{{{c}}}" if c and c != "—" else c for c in row[1:]]
        body_lines.append(" & ".join(cells) + r" \\")

    lines = [
        "% Requires \\usepackage{array,booktabs,longtable} in the main document.",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        rf"\begin{{longtable}}{{{col_spec}}}",
        rf"\caption{{{caption}}} \label{{{label}}} \\",
        *header_block,
        r"\endfirsthead",
        *header_block,
        r"\endhead",
        r"\midrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
        *body_lines,
        r"\end{longtable}",
        r"\endgroup",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    # -------------------------------------------------------------------------
    # Discover groups, subjects, and expected tracts
    # -------------------------------------------------------------------------
    groups = sorted([d.name for d in BUNDLESEG_DIR.iterdir() if d.is_dir()])
    print("Groups:", groups)

    subjects_by_group = {
        g: sorted([d.name for d in (BUNDLESEG_DIR / g).iterdir() if d.is_dir()])
        for g in groups
    }

    # Restrict penn_epilepsy to inclusion list, lobe == temporal only
    if "penn_epilepsy" in groups and PENN_EPILEPSY_INCLUSION_CSV.exists():
        meta = pd.read_csv(PENN_EPILEPSY_INCLUSION_CSV)
        temporal_subs = set(
            meta.loc[meta["lobe"].fillna("").astype(str).str.strip().str.lower() == "temporal", "sub"].astype(str)
        )
        in_bundleseg = set(subjects_by_group["penn_epilepsy"])
        subjects_by_group["penn_epilepsy"] = sorted(temporal_subs & in_bundleseg)

    for g in groups:
        print(f"  {g}: {len(subjects_by_group[g])} subjects")

    expected_tracts_by_group = {}
    for g in groups:
        tracts_g = set()
        for sub in subjects_by_group[g]:
            sub_dir = BUNDLESEG_DIR / g / sub
            for f in sub_dir.glob("*.trk"):
                tracts_g.add(f.stem)
        expected_tracts_by_group[g] = sorted(tracts_g)
        print(f"  {g}: {len(expected_tracts_by_group[g])} tracts")

    all_tracts_union = sorted(set().union(*expected_tracts_by_group.values()))
    print(f"\nUnion of tracts across groups: {len(all_tracts_union)}")
    print(all_tracts_union[:15], "..." if len(all_tracts_union) > 15 else "")

    table_groups = [g for g in TABLE_GROUP_ORDER if g in groups]
    col_headers = ["Tract"] + [TABLE_GROUP_DISPLAY.get(g, g) for g in table_groups]

    # -------------------------------------------------------------------------
    # 1. Tract segmentation failures (missing .trk)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. Tract segmentation failures (missing .trk)")
    print("=" * 60)

    failures_by_group = get_failures_by_group(
        BUNDLESEG_DIR, groups, subjects_by_group, expected_tracts_by_group
    )

    print("\nSegmentation failures (missing .trk) by group:")
    for g in groups:
        n = len(failures_by_group[g])
        n_subs = len(subjects_by_group[g])
        n_pairs = n_subs * len(expected_tracts_by_group[g])
        pct = 100 * n / n_pairs if n_pairs else 0
        print(f"  {g}: {n} failures ({n_subs} subjects × {len(expected_tracts_by_group[g])} tracts → {pct:.1f}%)")

    failure_dfs_by_group = {}
    for g in groups:
        rows = [{"subject": sub, "tract": tract} for sub, tract in failures_by_group[g]]
        failure_dfs_by_group[g] = pd.DataFrame(rows)
        if not failure_dfs_by_group[g].empty:
            print(f"  ... ({len(failure_dfs_by_group[g])} rows total for {g})")

    # Across groups: aggregate failure counts
    all_failures = []
    for g in groups:
        for (sub, tract) in failures_by_group[g]:
            all_failures.append({"group": g, "subject": sub, "tract": tract})

    failures_df = pd.DataFrame(all_failures)
    print("\nSegmentation failures across all groups:")
    print(f"  Total failure count: {len(failures_df)}")

    if not failures_df.empty:
        tract_failure_counts = (
            failures_df.groupby("tract").size().sort_values(ascending=False)
        )
        print("\nTop tracts by number of failures (across groups):")
        print(tract_failure_counts.head(15).to_frame(name="n_failures").to_string())
        sub_failure_counts = (
            failures_df.groupby(["group", "subject"]).size().sort_values(ascending=False)
        )
        print("\nTop subject-group pairs by number of failed tracts:")
        print(sub_failure_counts.head(10).to_frame(name="n_failures").to_string())

    # Save failure table immediately after failures are computed
    failure_counts_gt: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for g in groups:
        for _sub, tract in failures_by_group[g]:
            failure_counts_gt[g][tract] += 1
    failure_table_rows = _build_tract_group_table(
        all_tracts_union,
        table_groups,
        failure_counts_gt,
        {g: len(subjects_by_group[g]) for g in groups},
        expected_tracts_by_group,
    )
    n_by_group = {g: len(subjects_by_group[g]) for g in groups}
    failure_tex = OUTPUT_DIR / "tract_segmentation_failure_table.tex"
    _write_publication_tex(
        failure_tex,
        failure_table_rows,
        col_headers,
        caption="Tract segmentation failures by tract and group. Values are count (\\%).",
        label="tab:tract_segmentation_failures",
        table_groups=table_groups,
        n_by_group=n_by_group,
    )
    print(f"\nWrote {failure_tex}")

    # -------------------------------------------------------------------------
    # 2. Poor tract segmentation (≤5 streamlines)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"2. Poor tract segmentation (≤{POOR_STREAMLINES_THRESHOLD} streamlines)")
    print("=" * 60)

    poor_by_group = get_poor_by_group(
        BUNDLESEG_DIR,
        groups,
        subjects_by_group,
        expected_tracts_by_group,
        threshold=POOR_STREAMLINES_THRESHOLD,
    )

    print(f"\nPoor tract segmentation (≤{POOR_STREAMLINES_THRESHOLD} streamlines) by group:")
    for g in groups:
        n = len(poor_by_group[g])
        print(f"  {g}: {n} subject–tract pairs")

    poor_dfs_by_group = {}
    for g in groups:
        rows = [
            {"subject": sub, "tract": tract, "nb_streamlines": n}
            for sub, tract, n in poor_by_group[g]
        ]
        poor_dfs_by_group[g] = pd.DataFrame(rows)
        if not poor_dfs_by_group[g].empty:
            print(f"  ... ({len(poor_dfs_by_group[g])} rows total for {g})")

    # Across groups: aggregate poor segmentation
    all_poor = []
    for g in groups:
        for (sub, tract, n_sl) in poor_by_group[g]:
            all_poor.append({"group": g, "subject": sub, "tract": tract, "nb_streamlines": n_sl})

    poor_df = pd.DataFrame(all_poor)
    print(f"\nPoor tract segmentation (≤{POOR_STREAMLINES_THRESHOLD} streamlines) across all groups:")
    print(f"  Total count: {len(poor_df)}")

    if not poor_df.empty:
        tract_poor_counts = poor_df.groupby("tract").size().sort_values(ascending=False)
        print("\nTop tracts by number of poor segmentations (across groups):")
        print(tract_poor_counts.head(15).to_frame(name="n_poor").to_string())
        sub_poor_counts = (
            poor_df.groupby(["group", "subject"]).size().sort_values(ascending=False)
        )
        print("\nTop subject-group pairs by number of poor tracts:")
        print(sub_poor_counts.head(10).to_frame(name="n_poor").to_string())

    # -------------------------------------------------------------------------
    # 3. Publication-style LaTeX table (poor segmentation, by tract × group)
    # -------------------------------------------------------------------------
    # Poor counts per (group, tract)
    poor_counts_gt: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for g in groups:
        for _sub, tract, _n in poor_by_group[g]:
            poor_counts_gt[g][tract] += 1

    poor_table_rows = _build_tract_group_table(
        all_tracts_union,
        table_groups,
        poor_counts_gt,
        {g: len(subjects_by_group[g]) for g in groups},
        expected_tracts_by_group,
    )
    poor_tex = OUTPUT_DIR / "tract_segmentation_poor_table.tex"
    _write_publication_tex(
        poor_tex,
        poor_table_rows,
        col_headers,
        caption=f"Poor tract segmentation (≤{POOR_STREAMLINES_THRESHOLD} streamlines) by tract and group. Values are count (\\%).",
        label="tab:bundleseg_poor",
        table_groups=table_groups,
        n_by_group=n_by_group,
    )
    print(f"Wrote {poor_tex}")


if __name__ == "__main__":
    main()
