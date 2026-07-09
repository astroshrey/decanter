"""Median-filter wrapper with pinned boundary mode.

Why this module exists: ``scipy.ndimage.median_filter``'s default
boundary mode is ``'reflect'``; IRAF/WARP may differ. The parity diff
at the 5-pixel frame edge dominates pixel-level disagreement, so we
centralize the call and make the boundary policy explicit.

See PLAN_FULL.md §Cosmic-ray notes for context.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage

# Pin the boundary mode for every median filter decanter calls. If WARP
# turns out to use a different mode, change it here in one place.
BOUNDARY_MODE: str = "reflect"


def median_filter(
    data: np.ndarray,
    *,
    size: tuple[int, ...] | None = None,
    footprint: np.ndarray | None = None,
) -> np.ndarray:
    """Run ``scipy.ndimage.median_filter`` with pinned ``mode``.

    Exactly one of ``size`` or ``footprint`` must be provided.

    Args:
        data: input array.
        size: rectangular window size (passed to SciPy).
        footprint: arbitrary boolean footprint (e.g. diagonal kernel).

    Returns:
        Filtered array of the same shape as ``data``.

    Notes:
        Phase-2 swap path: a JAX-friendly equivalent will replace this
        call. JAX's ``medfilt2d`` supports rectangular footprints only,
        so the diagonal-footprint path needs ``lax.reduce_window``.
    """
    return scipy.ndimage.median_filter(data, size=size, footprint=footprint, mode=BOUNDARY_MODE)
