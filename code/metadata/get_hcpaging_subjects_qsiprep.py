#!/usr/bin/env python3
import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p != _p.parent and not (_p / "lib" / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir, gam_dir, analysis_dir, atlas_dir, inclusion_dir, open_metadata_dir, controlled_metadata_dir, controlled_derivatives_dir
PROJECT_ROOT = project_root()
"""Filter HCP-Aging subject list to only those present in qsiprep (sub-HCAXXXXXXX)."""

from pathlib import Path

# Input: full subject list from get_hcpaging_subjects_balsa.py (IDs like HCAXXXXXXX_V1)
SUBJECTS_FILE = Path(
    f"{project_root()}/data/import/hcpaging/hcpaging_subjects.txt"
)
QSIPREP_DIR = Path(
    f"{project_root()}/derivatives/qsiprep/hcpaging"
)
OUTPUT_DIR = Path(
    f"{project_root()}/data/import/hcpaging"
)
OUTPUT_FILE = OUTPUT_DIR / "hcpaging_subjects_qsiprep.txt"
DELIMITER = ", "


def main():
    if not SUBJECTS_FILE.is_file():
        raise FileNotFoundError(f"Subject list not found: {SUBJECTS_FILE}")
    if not QSIPREP_DIR.is_dir():
        raise FileNotFoundError(f"QSIPrep directory not found: {QSIPREP_DIR}")

    all_subjects = [s.strip() for s in SUBJECTS_FILE.read_text(encoding="utf-8").split(DELIMITER)]
    # Map HCAXXXXXXX_V1 -> HCAXXXXXXX for qsiprep lookup (sub-HCAXXXXXXX)
    base_id = lambda sid: sid.split("_")[0] if "_" in sid else sid

    filtered = []
    for sid in all_subjects:
        sub_dir = QSIPREP_DIR / f"sub-{base_id(sid)}"
        if sub_dir.is_dir():
            filtered.append(sid)

    filtered.sort()

    # Full list
    output_text = DELIMITER.join(filtered)
    OUTPUT_FILE.write_text(output_text, encoding="utf-8")
    print(f"Wrote {len(filtered)} subject IDs (qsiprep only) to {OUTPUT_FILE}")

    # Halves: _batch1 and _batch2
    mid = len(filtered) // 2
    batch1, batch2 = filtered[:mid], filtered[mid:]
    stem, ext = OUTPUT_FILE.stem, OUTPUT_FILE.suffix
    batch1_file = OUTPUT_DIR / f"{stem}_batch1{ext}"
    batch2_file = OUTPUT_DIR / f"{stem}_batch2{ext}"
    batch1_file.write_text(DELIMITER.join(batch1), encoding="utf-8")
    batch2_file.write_text(DELIMITER.join(batch2), encoding="utf-8")
    print(f"Wrote {len(batch1)} subjects to {batch1_file}")
    print(f"Wrote {len(batch2)} subjects to {batch2_file}")


if __name__ == "__main__":
    main()
