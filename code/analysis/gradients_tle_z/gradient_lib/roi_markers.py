"""Fixed-color ROI centroid markers for TLE gradient scatters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import TleZScatterRow


@dataclass(frozen=True)
class RoiMarkerSpec:
    label: str
    left_region: str
    right_region: str
    rgb: tuple[float, float, float]

    @property
    def color(self) -> tuple[float, float, float]:
        return self.rgb


DEFAULT_ROI_MARKERS: tuple[RoiMarkerSpec, ...] = (
    RoiMarkerSpec(
        "Hippocampus",
        "LH_Hippocampus",
        "RH_Hippocampus",
        (219 / 255.0, 153 / 255.0, 154 / 255.0),
    ),
    RoiMarkerSpec(
        "Fornix - core",
        "F_L_core",
        "F_R_core",
        (254 / 255.0, 201 / 255.0, 140 / 255.0),
    ),
    RoiMarkerSpec(
        "Temporal pole",
        "Left_TGv",
        "Right_TGv",
        (166 / 255.0, 192 / 255.0, 201 / 255.0),
    ),
    RoiMarkerSpec(
        "Uncinate fasciculus - core",
        "UF_L_core",
        "UF_R_core",
        (185 / 255.0, 162 / 255.0, 205 / 255.0),
    ),
)

ROI_MARKER_SIZE_PT2 = 70.0
ROI_MARKER_EDGE_COLOR = "0.15"
ROI_MARKER_EDGE_WIDTH = 0.7
ROI_MARKER_BOX_EDGE_WIDTH = 1.2
ROI_MARKER_ZORDER = 6
ROI_MARKER_LEFT = "<"
ROI_MARKER_RIGHT = ">"


def region_g2_g1(
    g1: pd.Series,
    g2: pd.Series,
    region: str,
) -> tuple[float, float] | None:
    """Return finite (G2, G1) for one ROI, or None."""
    if region not in g1.index or region not in g2.index:
        return None
    x = float(g2[region])
    y = float(g1[region])
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y


def bilateral_centroid_g2_g1(
    g1: pd.Series,
    g2: pd.Series,
    left_region: str,
    right_region: str,
) -> tuple[float, float] | None:
    """Mean (G2, G1) across left/right ROIs when both coordinates are finite."""
    xs: list[float] = []
    ys: list[float] = []
    for region in (left_region, right_region):
        if region not in g1.index or region not in g2.index:
            continue
        x = float(g2[region])
        y = float(g1[region])
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def add_roi_marker_boxes(
    ax: plt.Axes,
    row: TleZScatterRow,
    *,
    markers: Sequence[RoiMarkerSpec] = DEFAULT_ROI_MARKERS,
) -> list[tuple[RoiMarkerSpec, tuple[float, float]]]:
    """Overlay small fixed-color squares at bilateral ROI centroids."""
    _, g1, g2, _ = row
    placed: list[tuple[RoiMarkerSpec, tuple[float, float]]] = []
    for spec in markers:
        centroid = bilateral_centroid_g2_g1(g1, g2, spec.left_region, spec.right_region)
        if centroid is None:
            continue
        x, y = centroid
        ax.scatter(
            [x],
            [y],
            marker="s",
            s=ROI_MARKER_SIZE_PT2,
            c=[spec.color],
            edgecolors=ROI_MARKER_EDGE_COLOR,
            linewidths=ROI_MARKER_BOX_EDGE_WIDTH,
            zorder=ROI_MARKER_ZORDER,
        )
        placed.append((spec, centroid))
    return placed


def add_roi_marker_lr_triangles(
    ax: plt.Axes,
    row: TleZScatterRow,
    *,
    markers: Sequence[RoiMarkerSpec] = DEFAULT_ROI_MARKERS,
) -> list[tuple[RoiMarkerSpec, str, tuple[float, float]]]:
    """Overlay left/right triangles at each hemisphere's ROI position."""
    _, g1, g2, _ = row
    placed: list[tuple[RoiMarkerSpec, str, tuple[float, float]]] = []
    for spec in markers:
        left = region_g2_g1(g1, g2, spec.left_region)
        if left is not None:
            x, y = left
            ax.scatter(
                [x],
                [y],
                marker=ROI_MARKER_LEFT,
                s=ROI_MARKER_SIZE_PT2,
                c=[spec.color],
                edgecolors=ROI_MARKER_EDGE_COLOR,
                linewidths=ROI_MARKER_EDGE_WIDTH,
                zorder=ROI_MARKER_ZORDER,
            )
            placed.append((spec, "left", left))
        right = region_g2_g1(g1, g2, spec.right_region)
        if right is not None:
            x, y = right
            ax.scatter(
                [x],
                [y],
                marker=ROI_MARKER_RIGHT,
                s=ROI_MARKER_SIZE_PT2,
                c=[spec.color],
                edgecolors=ROI_MARKER_EDGE_COLOR,
                linewidths=ROI_MARKER_EDGE_WIDTH,
                zorder=ROI_MARKER_ZORDER,
            )
            placed.append((spec, "right", right))
    return placed
