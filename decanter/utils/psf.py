"""PSF-fitting utilities for the rectified order strips (s07 helper).

NumPy port of ``warp/centersearch_fortrans.py:centersearch_fortrans`` —
fits a 1-D Gaussian to the *stacked* slit profile of a transformed
echelle order to recover ``(xshift, fwhm)`` for that order. WARP uses
these two numbers to set the box-extraction aperture limits in s08
(``-fwhm + xshift, +fwhm + xshift`` → ~2-σ window).

The "fortrans" in the WARP filename refers to the input being a
TRANS-formed FITS frame, not Fortran. The original implementation
shells out to a separate IRAF call; the algorithm itself is pure
Python + SciPy + NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares


@dataclass(frozen=True, slots=True)
class PsfFit:
    """Result of a per-order Gaussian fit on a stacked slit profile."""

    xshift: float          # offset of the Gaussian peak from the aperture center, in slit-x pixels
    fwhm: float            # FWHM of the Gaussian fit, in slit-x pixels
    peak: float            # fitted peak amplitude (in profile units, ~normalized to 1)
    offset: float          # fitted baseline offset (~0 for clean data)
    n_rows_used: int       # number of y-rows that survived clipping and contributed to the stack
    success: bool          # whether the Gaussian fit converged


def _gaussian(x: NDArray, peak: float, center: float, sigma: float, offset: float) -> NDArray:
    """Single-peak Gaussian used by both fits in WARP."""
    return abs(peak) * np.exp(-((x - center) ** 2) / (2.0 * sigma**2)) + offset


def stacked_slit_profile(
    image: NDArray[np.floating],
    trace_x: NDArray[np.floating],
    *,
    ap_low: float,
    ap_high: float,
    lowlim_y: int = 500,
    upplim_y: int | None = None,
    step_sampling: int = 5,
    n_bins: int = 100,
    iterate: int = 3,
    dist_thres: float = 8.0,
    abba: bool = False,
    o_position: tuple[float, float] = (-8.0, 8.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Build a stacked, low-noise slit profile from the rectified strip.

    Args:
        image: 2-D rectified image (rows along dispersion / wavelength,
            columns along slit / x).
        trace_x: 1-D float, length ``image.shape[0]``. Trace x (slit
            coordinate) as a function of y/row. From the trans-frame
            aperture database.
        ap_low, ap_high: slit-window bounds relative to the trace
            (typically negative for low, positive for high). 1-indexed
            IRAF convention.
        lowlim_y, upplim_y: y-pixel range to sample. ``upplim_y=None``
            defaults to ``image.shape[0] - lowlim_y``.
        step_sampling: stride along y.
        n_bins: number of bins for the median-stacked profile.
        iterate: sigma-clip iterations on max-flux per row.
        dist_thres: max distance (in slit-x pixels) of the per-row
            max from the median of all per-row maxes; rows beyond this
            are clipped out.
        abba: if True, mask out the ABBA "O" position in the slit
            (rejected pixels at ``[o_position[0], o_position[1]]``
            around the trace center).
        o_position: rejection window for ABBA mode.

    Returns:
        ``(med_x, med_y, n_rows_used)``. The stacked profile arrays
        have ``n_bins`` entries each.
    """
    H, W = image.shape
    if upplim_y is None:
        upplim_y = H - lowlim_y
    if upplim_y <= lowlim_y:
        # Image too small for the default y-window; fall back to the
        # full range.
        lowlim_y = max(0, H // 10)
        upplim_y = max(lowlim_y + 1, H - lowlim_y)

    ap_x_low_pix = (trace_x + ap_low - 1.0).astype(np.int32)
    ap_x_low_pix = np.clip(ap_x_low_pix, 0, W - 1)
    ap_x_upp_pix = (trace_x + ap_high - 1.0).astype(np.int32)
    ap_x_upp_pix = np.clip(ap_x_upp_pix, 0, W - 1)

    ysample = np.arange(lowlim_y, upplim_y, step_sampling)
    if ysample.size == 0:
        return np.array([]), np.array([]), 0

    # Per-row max position + max flux (excluding O-position when ABBA).
    max_x = np.full(ysample.size, np.nan)
    max_flux = np.full(ysample.size, np.nan)
    for k, y in enumerate(ysample):
        lo, hi = int(ap_x_low_pix[y]), int(ap_x_upp_pix[y])
        if hi <= lo:
            continue
        row = image[y, lo:hi]
        x_idx = np.arange(lo + 1, hi + 1, dtype=np.float64)  # 1-indexed
        if abba:
            keep = (x_idx < o_position[0] + trace_x[y] + 1) | (x_idx > o_position[1] + trace_x[y] + 1)
            row = row[keep]
            x_idx = x_idx[keep]
        if row.size == 0:
            continue
        i_max = int(np.argmax(row))
        max_x[k] = x_idx[i_max]
        max_flux[k] = row[i_max]

    valid = np.isfinite(max_x) & np.isfinite(max_flux)
    if not valid.any():
        return np.array([]), np.array([]), 0

    dist = max_x - trace_x[ysample]
    dist_med = float(np.median(dist[valid]))
    dist_clip = np.abs(dist - dist_med) < dist_thres

    flux_med = float(np.median(max_flux[valid]))
    flux_clip = (max_flux > 0.3 * flux_med) & (max_flux < 2.0 * flux_med)

    for _ in range(iterate):
        cur = flux_clip & valid
        if not cur.any():
            break
        flux_med = float(np.median(max_flux[cur]))
        flux_scat = float(np.std(max_flux[cur]))
        flux_clip = flux_clip & (max_flux - flux_med > -1.5 * flux_scat) & (
            max_flux - flux_med < 4.0 * flux_scat
        )

    combined = dist_clip & flux_clip & valid
    if not combined.any():
        combined = flux_clip & valid
    n_rows = int(combined.sum())
    if n_rows == 0:
        return np.array([]), np.array([]), 0

    pf_x_list, pf_y_list = [], []
    for k, y in enumerate(ysample):
        if not combined[k]:
            continue
        lo, hi = int(ap_x_low_pix[y]), int(ap_x_upp_pix[y])
        if hi <= lo:
            continue
        row = image[y, lo:hi].astype(np.float64)
        x_idx = np.arange(lo + 1, hi + 1, dtype=np.float64)
        pf_x_list.append(x_idx - trace_x[y])
        pf_y_list.append(row / max_flux[k])

    pf_x = np.concatenate(pf_x_list)
    pf_y = np.concatenate(pf_y_list)

    # Median-bin into n_bins bins along the slit coordinate.
    x_min, x_max = float(pf_x.min()), float(pf_x.max())
    bin_width = (x_max - x_min) / float(n_bins)
    bin_idx = np.clip(((pf_x - x_min) / bin_width).astype(np.int32), 0, n_bins - 1)
    med_x = np.array([np.median(pf_x[bin_idx == i]) if (bin_idx == i).any() else np.nan
                      for i in range(n_bins)])
    med_y = np.array([np.median(pf_y[bin_idx == i]) if (bin_idx == i).any() else np.nan
                      for i in range(n_bins)])
    return med_x, med_y, n_rows


def fit_slit_gaussian(
    med_x: NDArray[np.float64],
    med_y: NDArray[np.float64],
    *,
    abba: bool = False,
    o_position: tuple[float, float] = (-8.0, 8.0),
    trim_window_units: float = 8.0,
) -> PsfFit:
    """Fit a Gaussian to a stacked slit profile.

    Args:
        med_x, med_y: stacked profile (NaNs allowed; dropped).
        abba: if True, look for the peak outside the O-position window.
        o_position: ABBA reject window.
        trim_window_units: profile is trimmed to ``± trim_window_units``
            around the peak (in slit-x units) before fitting.

    Returns:
        :class:`PsfFit` with ``xshift`` = peak position (slit-x), ``fwhm``.
    """
    mask = np.isfinite(med_x) & np.isfinite(med_y)
    if mask.sum() < 5:
        return PsfFit(xshift=float("nan"), fwhm=float("nan"), peak=float("nan"),
                      offset=float("nan"), n_rows_used=0, success=False)
    mx, my = med_x[mask], med_y[mask]

    if abba:
        center = float(np.average(mx))
        opos_mask = (mx < o_position[0] + center) | (mx > o_position[1] + center)
        if opos_mask.any():
            idx_max = int(np.argmax(my[opos_mask]))
            xpeak = float(mx[opos_mask][idx_max])
        else:
            idx_max = int(np.argmax(my))
            xpeak = float(mx[idx_max])
    else:
        xpeak = float(mx[int(np.argmax(my))])

    bin_width = float(np.median(np.diff(mx))) if mx.size > 1 else 1.0
    trim_range = int(trim_window_units / max(abs(bin_width), 1e-6))
    peak_idx = int(np.argmin(np.abs(mx - xpeak)))
    lo = max(0, peak_idx - trim_range)
    hi = min(mx.size, peak_idx + trim_range)
    mx_t, my_t = mx[lo:hi], my[lo:hi]
    if mx_t.size < 4:
        return PsfFit(xshift=float("nan"), fwhm=float("nan"), peak=float("nan"),
                      offset=float("nan"), n_rows_used=0, success=False)

    # Fit y(x) = |peak| * exp(-(x - xshift)^2 / (2 σ^2)) + offset, with
    # initial guesses matching WARP's hand-picked seeds.
    def residual(p: NDArray) -> NDArray:
        peak, center, sigma, offset = p
        return my_t - _gaussian(mx_t, peak, center, sigma, offset)

    p0 = np.array([1.0, xpeak + 0.1, 5.0, 0.01])
    try:
        result = least_squares(residual, p0, method="lm")
        peak_fit, center_fit, sigma_fit, offset_fit = result.x
        sigma_fit = float(abs(sigma_fit))
        return PsfFit(
            xshift=float(center_fit),
            fwhm=sigma_fit * 2.3548,
            peak=float(peak_fit),
            offset=float(offset_fit),
            n_rows_used=0,
            success=bool(result.success),
        )
    except Exception:
        return PsfFit(xshift=float("nan"), fwhm=float("nan"), peak=float("nan"),
                      offset=float("nan"), n_rows_used=0, success=False)
