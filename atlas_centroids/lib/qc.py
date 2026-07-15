"""QC plot for the combined atlas-centroids table."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (required for projection='3d')

_ATLAS_DISPLAY: dict[str, str] = {
    "glasser_cortex": "Glasser cortex",
    "four_s156_subcortex": "4S156 subcortex",
    "hcp1065_tract_third": "HCP1065 tract thirds",
}

_ATLAS_COLOR: dict[str, str] = {
    "glasser_cortex": "#1f77b4",
    "four_s156_subcortex": "#d62728",
    "hcp1065_tract_third": "#2ca02c",
}


def render_3d_scatter(combined: pd.DataFrame, out_path: Path) -> Path:
    """Render a single 3D scatter of all centroids, colored by ``atlas``.

    ``combined`` must have columns ``label, atlas, x, y, z``. Saves a 200-dpi PNG
    at ``out_path`` and returns the same path. Atlases absent from the table are
    skipped silently; legend reflects only the atlases actually present.
    """
    required = {"label", "atlas", "x", "y", "z"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"render_3d_scatter: missing columns {sorted(missing)}")

    fig = plt.figure(figsize=(9.0, 8.0))
    ax = fig.add_subplot(111, projection="3d")

    present = [a for a in _ATLAS_DISPLAY.keys() if (combined["atlas"] == a).any()]
    for atlas_key in present:
        sub = combined.loc[combined["atlas"] == atlas_key]
        ax.scatter(
            sub["x"].to_numpy(),
            sub["y"].to_numpy(),
            sub["z"].to_numpy(),
            s=14,
            color=_ATLAS_COLOR[atlas_key],
            edgecolor="black",
            linewidth=0.25,
            alpha=0.85,
            label=f"{_ATLAS_DISPLAY[atlas_key]} (n={len(sub)})",
        )
    # Mention skipped atlases in the title so a missing category is obvious.
    skipped = [
        _ATLAS_DISPLAY[a] for a in _ATLAS_DISPLAY.keys() if a not in present
    ]
    title = "Atlas centroids (MNI mm)"
    if skipped:
        title += f"  [missing: {', '.join(skipped)}]"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.view_init(elev=18, azim=-60)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
