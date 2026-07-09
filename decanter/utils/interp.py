"""Cubic-spline resampling helpers.

Centralizing these calls lets us swap the interpolation kernel in one
place after Spike A (PLAN_FULL.md §Algorithmic spikes) determines
which SciPy primitive most closely matches IRAF's ``interptype=spline3``
(natural cubic spline-through-points).

The default below uses ``scipy.ndimage.map_coordinates(order=3)`` for
speed; this is B-spline, **not** spline-through-points. If Spike A
finds the difference matters, switch to ``RectBivariateSpline`` or a
custom natural-spline implementation here.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage

# B-spline = the SciPy default; spline_through_points uses
# RectBivariateSpline; pending Spike A.
DEFAULT_KERNEL: str = "bspline"


def resample_2d(
    data: np.ndarray,
    output_coords: np.ndarray,
    *,
    kernel: str = DEFAULT_KERNEL,
) -> np.ndarray:
    """Resample ``data`` at ``output_coords`` using the chosen kernel.

    Args:
        data: 2D input array.
        output_coords: shape ``(2, ...)`` array of ``(y, x)`` coords.
        kernel: ``"bspline"`` (fast, smooth) or ``"spline_through_points"``
            (RectBivariateSpline, IRAF-compatible).

    Raises:
        NotImplementedError: ``kernel == "spline_through_points"`` path
            pending Spike A.
    """
    if kernel == "bspline":
        return scipy.ndimage.map_coordinates(data, output_coords, order=3, mode="reflect")
    raise NotImplementedError(f"utils.interp.resample_2d: kernel={kernel!r} pending Spike A")
