"""Multi-panel diagnostic figure builders.

Each function emits one of the consolidated PDFs described in
PLAN_FULL.md §Plots:

    * :func:`stage_summary` — raw / sky-subbed / CR-mask / flat-fielded
      grid, one page per frame.
    * :func:`order_extraction` — 2D strip + spatial profile + 1D
      extraction, one row per order.
    * :func:`wavelength_solution` — shift scatter + ThAr residuals.
    * :func:`spectrum_pages` — per-order overlaid frames, flux or
      normalized.
    * :func:`snr_summary` — SNR vs order for every cutrange.
    * :func:`slit_viewer` — start/end SV frames side by side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def stage_summary(out: Path, frames: list[dict[str, Any]]) -> None:
    """Render the per-frame 2x2 ZScale-stretched panel grid.

    Raises:
        NotImplementedError: not yet implemented.
    """
    raise NotImplementedError("plotting.diagnostics.stage_summary: not yet implemented")


def order_extraction(out: Path, orders: list[dict[str, Any]]) -> None:
    """Render the per-order 1x3 extraction panel.

    Raises:
        NotImplementedError: not yet implemented.
    """
    raise NotImplementedError("plotting.diagnostics.order_extraction: not yet implemented")


def spectrum_pages(out: Path, spectra: list[dict[str, Any]], normalized: bool) -> None:
    """Render one PDF page per order with all frames overlaid.

    Raises:
        NotImplementedError: not yet implemented.
    """
    raise NotImplementedError("plotting.diagnostics.spectrum_pages: not yet implemented")


def snr_summary(out: Path, snr_table: dict[float, dict[int, float]]) -> None:
    """Render the SNR-vs-order figure for every cutrange.

    Raises:
        NotImplementedError: not yet implemented.
    """
    raise NotImplementedError("plotting.diagnostics.snr_summary: not yet implemented")
