"""Shared matplotlib style for every decanter diagnostic figure.

Imported once at the start of stage 17 (and any Phase-2 modules that
plot). Centralized so a single rcParams change touches every figure.
"""

from __future__ import annotations

# Default style — keep small, override per-figure when needed.
RCPARAMS: dict[str, object] = {
    "figure.figsize": (8.5, 11.0),  # letter-size PDF pages
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "image.origin": "lower",
    "image.cmap": "viridis",
}


def apply() -> None:
    """Apply :data:`RCPARAMS` to the active matplotlib session."""
    import matplotlib

    matplotlib.rcParams.update(RCPARAMS)
