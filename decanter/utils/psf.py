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

import math
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


# ---------------------------------------------------------------------------
# WARP-exact center search (bit-identical port of centersearch_fortrans)
# ---------------------------------------------------------------------------

def warp_trace_x(entry, array_length: int) -> NDArray[np.float64]:
    """``warp/aperture.py:aperture.calculateTrace`` reproduced operation by operation.

    WARP evaluates the trace with hand-rolled float64 recurrences
    (``tracex += function(j) * coefficient[j]``, each ``function(j)`` rebuilding
    the recurrence from scratch). numpy's ``chebval`` uses Clenshaw's algorithm,
    which rounds differently; this reproduces WARP's order of operations.

    Args:
        entry: an :class:`~decanter.io.apdb.ApertureEntry` (values parsed with
            Python ``float()``, exactly as WARP's ``apertureSet`` does).
        array_length: number of rows (``NAXIS2`` of the trans strip).
    """
    ymin = entry.y_min
    ymax = entry.y_max
    tracey = np.arange(array_length) + 1.
    tracex = np.zeros(array_length) + entry.center_x
    norm = (2 * tracey - ymin - ymax) / (ymax - ymin)
    legendre_ = entry.function_type == 2
    if entry.function_type not in (1, 2):
        raise ValueError(f"unknown aperture function type {entry.function_type}")
    # p[n] of the recurrence; function(j) returns p[j] built from scratch, which
    # is the same value as extending the recurrence once (identical operations).
    p_prev2 = None
    p_prev = None
    for j in range(entry.poly_order):
        if j == 0:
            pj = 1.
        elif j == 1:
            pj = norm
        elif legendre_:
            pj = ((2 * j - 1) * norm * p_prev - (j - 1) * p_prev2) / j
        else:
            pj = 2 * norm * p_prev - p_prev2
        tracex += pj * entry.coefficients[j]
        p_prev2, p_prev = p_prev, pj
    return tracex


def default_warp_leastsq():
    """The ``leastsq`` WARP's center search is reproduced with.

    WARP calls ``scipy.optimize.leastsq``, so its fitted centers depend on the
    scipy build it ran under: scipy <= 1.14 ships Fortran MINPACK, scipy 1.15
    a C translation that stops ~1e-9 (relative) away. WARP runs in the same
    venv as decanter, so the default (``DECANTER_WARP_MINPACK`` unset or
    ``auto``, same as ``scipy``) is the installed ``scipy.optimize.leastsq``:
    the very function a WARP run made today calls, whatever its version (on
    this machine scipy 1.14.1 since 2026-10-19; TOI3486_decanterref and
    WASP69_decanterref were made with it). ``DECANTER_WARP_MINPACK=c1153``
    selects :func:`decanter.utils.minpack_exact.leastsq`, a bit-exact port of
    scipy 1.15.3's compiled C MINPACK, for WARP reductions made under 1.15.3
    (every archived one through 2026-10-19, e.g. TOI2109_decanterref) when
    another scipy is installed. With scipy 1.15.3 installed the two agree.
    The fitted floats differ between flavours, but the per-frame aperture
    DBs (``%.2f`` tokens) were identical on all three ``_decanterref``
    reductions, so the extracted spectra do not depend on the choice there.
    """
    import os

    flavour = os.environ.get("DECANTER_WARP_MINPACK", "auto").strip().lower()
    if flavour in ("auto", "scipy", ""):
        from scipy.optimize import leastsq
        return leastsq
    if flavour not in ("c1153", "exact"):
        raise ValueError(f"DECANTER_WARP_MINPACK={flavour!r}: use 'auto', 'scipy' or 'c1153'")
    from decanter.utils.minpack_exact import leastsq
    return leastsq


def warp_minpack_label() -> str:
    """What :func:`default_warp_leastsq` resolves to, e.g. ``"scipy 1.14.1"``
    or ``"c1153"`` (for provenance)."""
    import scipy

    from decanter.utils.minpack_exact import leastsq as _c1153
    return "c1153" if default_warp_leastsq() is _c1153 else f"scipy {scipy.__version__}"


def _gaussianfunc_cs(p, x, c):
    return math.sqrt(p[0] ** 2) * np.exp(-(x - p[1] - c) ** 2 / (2 * p[2] ** 2)) + p[3]


def _residue_cs(p, y, x, c):
    return y - _gaussianfunc_cs(p, x, c)


def _warp_bin_medians(pf_x, pf_y, min_x, width, step):
    """``np.median`` of ``pf_x`` / ``pf_y`` in each of WARP's ``step`` open bins.

    WARP evaluates, per bin ``i``, ``np.median(v[(min + w*i < x) & (x < min + w*(i+1))])``
    (open intervals; an empty bin gives NaN, a bin holding a NaN gives NaN).
    This computes the identical values in one pass: the same boundary
    expressions, membership by ``searchsorted``, and ``np.median`` on each bin's
    elements in their original order.
    """
    bounds = np.array([min_x + width * i for i in range(step + 1)])
    ok = ~np.isnan(pf_x)
    j = np.searchsorted(bounds, pf_x, side="left")
    member = ok & (j >= 1) & (j <= step)
    jj = np.where(member, j, 1)
    member &= pf_x < bounds[jj]
    member &= pf_x > bounds[jj - 1]
    # A stable sort by bin keeps each bin's elements in their original order,
    # i.e. exactly the array WARP's boolean mask selects; np.median on it is
    # then the very same computation (including NaN and signed-zero cases).
    bins = (jj - 1)[member]
    order = np.argsort(bins, kind="stable")
    sb = bins[order]
    counts = np.bincount(sb, minlength=step)
    ends = np.cumsum(counts)
    out = []
    for vals in (pf_x[member][order], pf_y[member][order]):
        med = np.full(step, np.nan)
        for i in range(step):
            if counts[i]:
                med[i] = np.median(vals[ends[i] - counts[i]:ends[i]])
        out.append(med)
    return out[0], out[1]


def warp_centersearch(
    data: NDArray,
    ref_entry,
    *,
    crval2: float,
    cdelt2: float,
    crpix2: float,
    abba: bool,
    leastsq=None,
    profile: dict | None = None,
) -> tuple[float, float]:
    """Bit-exact port of ``warp/centersearch_fortrans.py:centersearch_fortrans``.

    ``profile`` (optional dict) receives ``med_x``/``med_y``, the binned slit
    profile WARP writes to ``images/spatial_profile/<strip>.dat``, when WARP
    writes it (it is left empty when WARP gives up before that point).

    Every array operation, dtype promotion (the trans strip is float32; WARP's
    ``np.r_`` stacking upcasts to float64 after a float32 division) and median
    is reproduced in WARP's order, and the Gaussian is fitted with MINPACK
    ``lmdif`` through ``scipy.optimize.leastsq`` (the same call WARP makes).

    Args:
        data: the rectified ``_m###trans`` strip, shape ``(NAXIS2, NAXIS1)``;
            cast to float32 (the dtype WARP read from the FITS file).
        ref_entry: the multihole trans reference aperture
            (``ap{aptrans}_{m}trans``) as an :class:`~decanter.io.apdb.ApertureEntry`.
        crval2, cdelt2, crpix2: the strip's dispersion-axis WCS (FITS header
            values, as astropy parses them).
        abba: WARP's ``abbaflag`` (NODPOS has no ``O`` and is not ``NA``).
        leastsq: the least-squares driver. Default
            (:func:`default_warp_leastsq`): the installed
            ``scipy.optimize.leastsq``, which is what WARP calls in the same
            venv; ``DECANTER_WARP_MINPACK=c1153`` selects the bit-exact port of
            scipy 1.15.3's C MINPACK for reductions made under that build.

    Returns:
        ``(xshift, fwhm)`` exactly as WARP logs them (``nan, nan`` when WARP
        gives up on the order).
    """
    if leastsq is None:
        leastsq = default_warp_leastsq()
    trimg = np.asarray(data, dtype=np.float32)
    naxis2, naxis1 = trimg.shape

    lowlim = 500
    upplim = 2048 - lowlim
    step_sampling = 5
    iterate = 3
    distthres = 8.
    center_width = 50
    oposition = [-8, 8]

    lowlim_ypix = int((lowlim - crval2) / cdelt2 + crpix2 - 1) - 1
    upplim_ypix = int((upplim - crval2) / cdelt2 + crpix2 - 1)
    center_ypix = int(((lowlim + upplim) / 2. - crval2) / cdelt2 + crpix2 - 1)

    apx = warp_trace_x(ref_entry, naxis2)
    ap_low = ref_entry.low
    ap_high = ref_entry.high

    apxlow = apx + ap_low - 1
    apxlow_pix = apxlow.astype(np.int32)
    apxlow_pix[apxlow_pix < 0] = 0
    apxupp = apx + ap_high - 1
    apxupp_pix = apxupp.astype(np.int32)
    apxupp_pix[apxupp_pix > naxis1] = naxis1 - 1
    apcenter = apx + (ap_low + ap_high) / 2.

    # WARP builds the cut row for every one of the naxis2 rows; the rows are
    # independent, so only the ones that are read are materialised here.
    def cut_row(i):
        cx = np.arange(apxlow_pix[i] + 1, apxupp_pix[i] + 1)
        cy = trimg[i][apxlow_pix[i]:apxupp_pix[i]]
        for _ in range(len(cx) - len(cy)):
            cy = np.append(cy, 0.)
        if abba:
            ob = (cx < oposition[0] + apcenter[i] + 1) | (cx > oposition[1] + apcenter[i] + 1)
            cx = cx[ob]
            cy = cy[ob]
        return cx, cy

    centermax = [np.max(cut_row(center_ypix - center_width + i)[1])
                 for i in range(0, 2 * center_width, step_sampling)]
    if np.average(centermax) < 100.:
        step_sampling = 1

    ysample = np.arange(lowlim_ypix, upplim_ypix, step_sampling)
    cut_x: dict = {}
    cut_y: dict = {}
    for i in ysample:
        cut_x[i], cut_y[i] = cut_row(i)
    maxid = np.array([np.argmax(cut_y[i]) for i in ysample])
    maxpix = np.array([cut_x[ysample[i]][maxid[i]] for i in range(len(ysample))])
    maxflux = np.array([cut_y[ysample[i]][maxid[i]] for i in range(len(ysample))])
    maxdist = np.array([maxpix[i] - apx[ysample[i]] for i in range(len(ysample))])

    dist_med = np.median(maxdist)
    dist_clip = np.fabs(maxdist - dist_med) < distthres
    flux_clip = (maxflux > np.median(maxflux) * 0.3) & (maxflux < np.median(maxflux) * 2.0)
    for _ in range(iterate):
        flux_med = np.median(maxflux[flux_clip])
        flux_scat = np.std(maxflux[flux_clip])
        flux_clip = flux_clip & (maxflux - flux_med > -1.5 * flux_scat) & (maxflux - flux_med < 4. * flux_scat)

    if np.sum(np.logical_and(dist_clip, flux_clip)) > 0:
        clip_array = np.logical_and(dist_clip, flux_clip)
    elif np.sum(flux_clip) > 0:
        clip_array = flux_clip
    else:
        return math.nan, math.nan

    xs_parts = [np.array([])]
    ys_parts = [np.array([])]
    counter = 0
    for i in range(len(ysample)):
        if clip_array[i]:
            y = ysample[i]
            nocut_y = trimg[y][apxlow_pix[y]:apxupp_pix[y]]
            nocut_x = np.arange(apxlow_pix[y] + 1, apxupp_pix[y] + 1)
            xs_parts.append(nocut_x - apx[y])
            ys_parts.append(nocut_y / maxflux[i])
            counter += 1
    if counter == 0:
        return math.nan, math.nan
    # np.r_ over the pieces == one concatenate (float64 result, same values).
    pf_xdata = np.concatenate(xs_parts)
    pf_ydata = np.concatenate([np.asarray(a, dtype=np.float64) for a in ys_parts])

    min_xdata = np.amin(pf_xdata)
    max_xdata = np.amax(pf_xdata)
    step = 100
    pf_width = (max_xdata - min_xdata) / float(step)
    med_x, med_y = _warp_bin_medians(pf_xdata, pf_ydata, min_xdata, pf_width, step)
    if profile is not None:
        profile["med_x"] = med_x
        profile["med_y"] = med_y

    trimrange = int(8 / pf_width)
    if abba:
        medx_center = np.average(med_x)
        oposbool2 = (med_x < oposition[0] + medx_center) | (med_x > oposition[1] + medx_center)
        mid = np.argmax(med_y[oposbool2])
        mpix = med_x[oposbool2][mid]
        mid = np.where(med_x == mpix)[0][0]
    else:
        mid = np.argmax(med_y)
        mpix = med_x[mid]

    lowlim_trim = 0 if mid - trimrange < 0 else mid - trimrange
    upplim_trim = len(med_x) if mid + trimrange > len(med_x) else mid + trimrange
    med_xtrim = med_x[lowlim_trim:upplim_trim]
    med_ytrim = med_y[lowlim_trim:upplim_trim]

    p0 = [1., 0.1, 5., 0.01]
    out = leastsq(_residue_cs, p0, args=(med_ytrim, med_xtrim, mpix), full_output=True)
    centershift = out[0][1]
    gwidth = math.fabs(out[0][2])
    return centershift + mpix, gwidth * 2.3548


def warp_slit_profile(
    data: NDArray,
    ref_entry,
    *,
    crval2: float,
    cdelt2: float,
    crpix2: float,
) -> tuple[NDArray, NDArray]:
    """Port of ``warp/centersearch_fortrans.py:make_slit_profile``.

    WARP writes this profile to ``images/spatial_profile/<strip>.dat`` when
    the center search gives up on an order (``nan``) and for every order in
    manual-aperture mode: every 5th row of the reference window between
    500 and 1548 A, each normalised by its own maximum, binned into 100
    medians.  Returns ``(med_x, med_y)``.
    """
    trimg = np.asarray(data, dtype=np.float32)
    naxis2, naxis1 = trimg.shape
    lowlim = 500
    upplim = 2048 - lowlim
    step_sampling = 5
    lowlim_ypix = int((lowlim - crval2) / cdelt2 + crpix2 - 1) - 1
    upplim_ypix = int((upplim - crval2) / cdelt2 + crpix2 - 1)
    apx = warp_trace_x(ref_entry, naxis2)
    apxlow = apx + ref_entry.low - 1
    apxlow_pix = apxlow.astype(np.int32)
    apxlow_pix[apxlow_pix < 0] = 0
    apxupp = apx + ref_entry.high - 1
    apxupp_pix = apxupp.astype(np.int32)
    apxupp_pix[apxupp_pix > naxis1] = naxis1 - 1
    ysample = np.arange(lowlim_ypix, upplim_ypix, step_sampling)
    xs_parts = [np.array([])]
    ys_parts = [np.array([])]
    for y in ysample:
        nocut_y = trimg[y][apxlow_pix[y]:apxupp_pix[y]]
        nocut_x = np.arange(apxlow_pix[y] + 1, apxupp_pix[y] + 1)
        xs_parts.append(nocut_x - apx[y])
        ys_parts.append(nocut_y / np.max(nocut_y))
    pf_xdata = np.concatenate(xs_parts)
    pf_ydata = np.concatenate([np.asarray(a, dtype=np.float64) for a in ys_parts])
    min_xdata = np.amin(pf_xdata)
    max_xdata = np.amax(pf_xdata)
    step = 100
    pf_width = (max_xdata - min_xdata) / float(step)
    return _warp_bin_medians(pf_xdata, pf_ydata, min_xdata, pf_width, step)
