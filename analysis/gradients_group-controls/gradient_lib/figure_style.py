"""Matplotlib font setup for gradient figures (Georgia)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

FIGURE_FONT_RCPARAMS: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Georgia", "DejaVu Serif", "serif"],
}

_GEORGIA_CANDIDATE_PATHS: tuple[Path, ...] = (
    Path("/usr/share/fonts/truetype/georgia.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Georgia.ttf"),
    Path("/usr/share/fonts/TTF/Georgia.ttf"),
)
# Italic / bold companions (same directories as the regular face).
_GEORGIA_STYLE_FILENAMES: tuple[str, ...] = (
    "georgiai.ttf",  # italic
    "georgiab.ttf",  # bold
    "georgiaz.ttf",  # bold italic
    "Georgia Italic.ttf",
    "Georgia Bold.ttf",
    "Georgia Bold Italic.ttf",
)

_georgia_registered = False


def ensure_georgia_font() -> None:
    """Register system Georgia (roman + italic/bold faces) with matplotlib."""
    global _georgia_registered
    if _georgia_registered:
        return
    registered_dir: Path | None = None
    for path in _GEORGIA_CANDIDATE_PATHS:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            registered_dir = path.parent
            break
    if registered_dir is not None:
        for name in _GEORGIA_STYLE_FILENAMES:
            style_path = registered_dir / name
            if style_path.is_file():
                font_manager.fontManager.addfont(str(style_path))
    _georgia_registered = True


# Mathtext faces so ``$r$`` / ``$\mathit{r}$`` use Georgia italic, not Computer Modern.
GEORGIA_MATHTEXT_RCPARAMS: dict[str, object] = {
    "mathtext.fontset": "custom",
    "mathtext.rm": "Georgia",
    "mathtext.it": "Georgia:italic",
    "mathtext.bf": "Georgia:bold",
    "mathtext.cal": "Georgia",
    "mathtext.sf": "Georgia",
}


def apply_figure_font_rcparams() -> None:
    """Register Georgia (if available) and set the serif stack for all figures."""
    ensure_georgia_font()
    plt.rcParams.update(FIGURE_FONT_RCPARAMS)
    plt.rcParams.update(GEORGIA_MATHTEXT_RCPARAMS)


@contextmanager
def figure_font_context():
    """Context manager for one-off figures (legends, bar panels, etc.)."""
    ensure_georgia_font()
    with plt.rc_context({**FIGURE_FONT_RCPARAMS, **GEORGIA_MATHTEXT_RCPARAMS}):
        yield
