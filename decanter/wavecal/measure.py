"""Measuring a per-frame shift against a template.

One estimator serves both references: interpolate the template onto a grid of
trial shifts, correlate every frame against the whole bank in one matrix
product, and refine the peak with a parabola. The templates are physical
(HITRAN telluric transmission, OH emission at laboratory positions), so the
shifts are absolute.

Widths are given in resolution elements or km/s and converted to pixels per
order, since WINERED's sampling varies by a factor of five between modes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter, percentile_filter



def robust_scatter(values: NDArray | list) -> float:
    """1.4826 x MAD, the Gaussian-equivalent sigma."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(array - np.median(array))))


def continuum_normalize(flux: NDArray, width_pixels: int) -> NDArray:
    """Divide by a running high percentile: flattens the blaze, keeps lines."""
    flux = np.asarray(flux, dtype=float)
    width = max(5, int(width_pixels) | 1)
    continuum = percentile_filter(flux, percentile=90, size=width, mode="nearest")
    return np.divide(
        flux, continuum,
        out=np.full_like(flux, np.nan, dtype=float),
        where=np.isfinite(continuum) & (continuum > 0),
    )


def highpass(flux: NDArray, width_pixels: int) -> NDArray:
    """Subtract a running median: removes the sky continuum, keeps emission."""
    flux = np.asarray(flux, dtype=float)
    width = max(3, int(width_pixels) | 1)
    return flux - median_filter(flux, width, mode="nearest")


def pixels_per_resolution_element(dv_pix_kms: float, resolution_element_kms: float) -> float:
    return float(resolution_element_kms) / float(dv_pix_kms)


def shift_grid_pixels(
    search_kms: float, dv_pix_kms: float, *, step_kms: float = 0.05
) -> NDArray:
    """Trial shifts in pixels covering +/- ``search_kms``.

    The step is set in velocity, so a WIDE order (5 km/s per pixel) and a
    HIRES order (1 km/s per pixel) are searched at the same physical
    resolution rather than the same pixel resolution.
    """
    half = float(search_kms) / float(dv_pix_kms)
    step = max(float(step_kms) / float(dv_pix_kms), 1.0e-3)
    n = int(np.floor(half / step))
    return np.arange(-n, n + 1, dtype=float) * step


def parabolic_extremum(grid: NDArray, values: NDArray, index: int) -> float:
    """Sub-sample peak position from three points around ``index``."""
    if index <= 0 or index >= len(grid) - 1:
        return float(grid[index])
    left, centre, right = values[index - 1 : index + 2]
    curvature = left - 2.0 * centre + right
    if not np.isfinite(curvature) or curvature >= 0.0:
        return float(grid[index])
    fraction = 0.5 * (left - right) / curvature
    return float(grid[index] + fraction * (grid[1] - grid[0]))


def ccf_shifts(
    signal: NDArray,
    template: NDArray,
    support: NDArray,
    shift_grid: NDArray,
) -> tuple[NDArray, NDArray]:
    """Normalised Pearson CCF of many frames against one shifted template.

    Args:
        signal: ``(n_frames, n_pixels)`` feature signal -- absorption depth
            ``1 - normalised flux`` for tellurics, high-passed counts for OH.
        template: ``(n_pixels,)`` template in the same feature convention, at
            zero shift.
        support: ``(n_pixels,)`` bool, the pixels the CCF is evaluated over.
            Held fixed across frames so every frame sees the same lines.
        shift_grid: trial shifts in pixels.

    Returns:
        ``(shift_pixels, peak_correlation)``, each ``(n_frames,)``. A frame
        whose CCF peaks on the first or last trial shift returns NaN: its true
        shift is outside the search range, so the peak position would be
        pinned at the boundary rather than measured.
    """
    signal = np.atleast_2d(np.asarray(signal, dtype=float))
    template = np.asarray(template, dtype=float)
    support = np.asarray(support, dtype=bool)
    n_frames = signal.shape[0]

    if int(support.sum()) < 20 or not np.any(np.isfinite(template)):
        return np.full(n_frames, np.nan), np.full(n_frames, np.nan)

    pixel = np.arange(template.size, dtype=float)
    bank = np.asarray([
        np.interp(pixel - shift, pixel, template, left=np.nan, right=np.nan)[support]
        for shift in shift_grid
    ])
    bank = bank - np.nanmean(bank, axis=1, keepdims=True)
    bank = np.nan_to_num(bank, nan=0.0)
    bank_norm = np.sqrt(np.sum(bank**2, axis=1))

    data = signal[:, support]
    data = data - np.nanmean(data, axis=1, keepdims=True)
    data = np.nan_to_num(data, nan=0.0)
    data_norm = np.sqrt(np.sum(data**2, axis=1))

    correlation = np.divide(
        data @ bank.T,
        data_norm[:, None] * bank_norm[None, :],
        out=np.full((n_frames, shift_grid.size), np.nan),
        where=(data_norm[:, None] > 0) & (bank_norm[None, :] > 0),
    )

    shifts = np.full(n_frames, np.nan)
    peaks = np.full(n_frames, np.nan)
    for i in range(n_frames):
        row = correlation[i]
        if not np.any(np.isfinite(row)):
            continue
        index = int(np.nanargmax(row))
        if index == 0 or index == shift_grid.size - 1:
            continue
        shifts[i] = parabolic_extremum(shift_grid, row, index)
        peaks[i] = float(row[index])
    return shifts, peaks


def measure_series_shifts(
    signal_cube: NDArray,
    templates: NDArray,
    supports: NDArray,
    dv_pix_kms: NDArray,
    *,
    search_kms: float = 25.0,
    step_kms: float = 0.05,
    center_kms: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Run :func:`ccf_shifts` for every order of a series.

    Args:
        signal_cube: ``(n_frames, n_pixels, n_orders)``.
        templates: ``(n_pixels, n_orders)``.
        supports: ``(n_pixels, n_orders)`` bool.
        dv_pix_kms: ``(n_orders,)``.

    Returns:
        ``(velocity_kms, peak)``, each ``(n_frames, n_orders)``. The velocity
        is the shift converted with that order's own pixel scale.
    """
    n_frames, _, n_orders = signal_cube.shape
    centers = (np.zeros(n_orders, dtype=float) if center_kms is None
               else np.broadcast_to(np.asarray(center_kms, dtype=float), (n_orders,)))
    velocity = np.full((n_frames, n_orders), np.nan)
    peak = np.full((n_frames, n_orders), np.nan)
    for j in range(n_orders):
        grid = (shift_grid_pixels(search_kms, float(dv_pix_kms[j]), step_kms=step_kms)
                + centers[j] / float(dv_pix_kms[j]))
        shifts, peaks = ccf_shifts(
            signal_cube[:, :, j], templates[:, j], supports[:, j], grid
        )
        velocity[:, j] = shifts * float(dv_pix_kms[j])
        peak[:, j] = peaks
    return velocity, peak
