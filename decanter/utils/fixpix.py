"""IRAF-fixpix-equivalent bad-pixel interpolation.

Replaces ``iraf.fixpix(image, mask, linterp="INDEF", cinterp="INDEF")``.

Algorithm
---------
For every masked pixel:

1. Measure the length of the contiguous bad-pixel run containing it
   along the row (``x``) and along the column (``y``).
2. Pick whichever axis has the **shorter** run — that's the axis along
   which we'll have the smallest gap to bridge.
3. Linearly interpolate from the nearest good pixels on that axis.
   ``np.interp`` performs constant-value extrapolation when a run
   touches the frame edge — degraded but defensible.

Ties go to the row direction (``x``), matching IRAF's stated
preference: "In the case of a square region, linear interpolation
across the row is used."

The function takes a single ``mask`` as input — callers are expected to
have already combined any static and per-frame masks (logical OR) before
invoking this.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _axis_run_lengths(mask: NDArray[np.bool_], axis: int) -> NDArray[np.int32]:
    """For each masked pixel, the length of its contiguous run along ``axis``.

    ``axis=1`` → run is along columns within a row (x-direction).
    ``axis=0`` → run is along rows within a column (y-direction).

    Unmasked pixels keep value 0.
    """
    if axis == 0:
        # Transpose so we can always work along axis 1, then transpose back.
        return _axis_run_lengths(mask.T, axis=1).T

    H, W = mask.shape
    out = np.zeros((H, W), dtype=np.int32)

    # Per-row Python loop; tolerable at 2048 rows.
    for r in range(H):
        row = mask[r]
        if not row.any():
            continue
        # Find run starts/ends via diff on a padded signed view.
        padded = np.concatenate(([False], row, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        for s, e in zip(starts, ends, strict=True):
            out[r, s:e] = e - s
    return out


def _axis_linear_interp(
    image: NDArray[np.floating],
    mask: NDArray[np.bool_],
    axis: int,
) -> NDArray[np.floating]:
    """Linearly interpolate masked pixels along ``axis`` using good neighbors.

    ``axis=1`` → interpolate along x (within each row).
    ``axis=0`` → interpolate along y (within each column).

    Returns a new array with masked pixels replaced. Unmasked pixels are
    unchanged.
    """
    out = image.copy()
    if axis == 0:
        # Recurse via transpose so we never branch the inner loop.
        return _axis_linear_interp(image.T, mask.T, axis=1).T

    H, W = image.shape
    x_full = np.arange(W)
    for r in range(H):
        m = mask[r]
        if not m.any():
            continue
        good = ~m
        if good.sum() < 2:
            # Whole row (or all-but-one) bad — np.interp would still
            # work with 1 point (constant), but we explicitly leave the
            # row alone here so the caller's cross-axis fallback can
            # cover it.
            continue
        out[r, m] = np.interp(x_full[m], x_full[good], image[r, good]).astype(image.dtype)
    return out


def fixpix(
    image: NDArray[np.floating],
    mask: NDArray[np.integer | np.bool_],
) -> NDArray[np.floating]:
    """Interpolate over masked pixels along the narrower run axis.

    Args:
        image: 2-D float image.
        mask: 2-D mask; any non-zero element is treated as a bad pixel.

    Returns:
        A new image with bad pixels filled by linear interpolation. The
        original array is not modified.

    Notes:
        For all-bad rows AND all-bad columns simultaneously, the
        pixel is left at its original value. This is consistent with
        IRAF's behavior of declining to fix unfixable runs.
    """
    if image.shape != mask.shape:
        raise ValueError(
            f"image shape {image.shape} does not match mask shape {mask.shape}"
        )
    bad = mask.astype(bool, copy=False)
    if not bad.any():
        return image.copy()

    row_runs = _axis_run_lengths(bad, axis=1)
    col_runs = _axis_run_lengths(bad, axis=0)

    row_interp = _axis_linear_interp(image, bad, axis=1)
    col_interp = _axis_linear_interp(image, bad, axis=0)

    # Choose per-pixel: row wins on tie (IRAF's "square region uses row").
    # `_axis_linear_interp` returns the original value when an axis is
    # un-interpolable (< 2 good pixels on that axis), so a masked pixel
    # whose row is un-interpolable will fall through to col_interp.
    use_col = bad & (col_runs < row_runs)
    use_row = bad & ~use_col

    out = image.copy()
    out[use_row] = row_interp[use_row]
    out[use_col] = col_interp[use_col]
    return out
