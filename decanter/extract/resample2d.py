"""WARP's ``resample2Dspec``: the 2-D spectrum resampled across the slit.

``Extract 2d spectrum: yes`` (``Warp_sci.py:576-582``) writes, next to IRAF's
``apall format=strip`` output, ``<strip>_resample.####``: for every
dispersion line of the rectified order a cubic ``scipy`` interpolant of the
line around the trace, averaged on a 0.01-pixel grid over unit pixels
centred on the trace, rescaled to the line's box sum
(``warp/Spec1Dtools.py:resample2Dspec``).  This is WARP's own Python, so the
port runs the same NumPy/SciPy calls on the same arrays (the rectified strip
as the big-endian float32 its FITS file holds, the aperture database as
WARP's ``apertureSet`` parses it) and gives the same bits on the same
NumPy/SciPy build.  One change, same values: the interpolant is evaluated
once on the union of the sub-pixel samples instead of once per output pixel
(a B-spline is evaluated point by point, so each sample's value is the same).
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from decanter.io.apdb import ApertureEntry


def warp_trace_x(entry: ApertureEntry, array_length: int) -> NDArray[np.float64]:
    """``warp/aperture.py:apertureSet.aperture.calculateTrace`` for a database
    entry: ``tracex = centerx + sum_j P_j(ynorm) * c_j`` in float64 with WARP's
    recurrences and operation order (``apertureSet.__init__`` parses the
    tokens with ``float``; the function type and order with ``int(float)``)."""
    toks = entry.raw_curve
    ftype = int(float(toks[0]))
    yorder = int(float(toks[1]))
    ymin = float(toks[2])
    ymax = float(toks[3])
    cm = [float(t) for t in toks[4:4 + yorder]]
    centerx = float(entry.raw_center[0])
    tracey = np.arange(array_length) + 1.
    tracex = np.zeros(array_length) + centerx
    ynorm = (2 * tracey - ymin - ymax) / (ymax - ymin)

    def cheb(n):
        p = [0. for _ in range(n + 1)]
        p[0] = 1.
        if n >= 1:
            p[1] = ynorm
        if n >= 2:
            for i in range(2, n + 1):
                p[i] = 2 * ynorm * p[i - 1] - p[i - 2]
        return p[n]

    def leg(n):
        p = [0. for _ in range(n + 1)]
        p[0] = 1.
        if n >= 1:
            p[1] = ynorm
        if n >= 2:
            for i in range(2, n + 1):
                p[i] = ((2 * i - 1) * ynorm * p[i - 1] - (i - 1) * p[i - 2]) / i
        return p[n]

    fn = cheb if ftype == 1 else leg
    for j in range(yorder):
        tracex += fn(j) * cm[j]
    return tracex


def resample_2d(trans: NDArray, entry: ApertureEntry, interpolation: str = "cubic",
                finepix: float = 0.01) -> NDArray[np.float32]:
    """``Spec1Dtools.resample2Dspec`` of one order: ``(xsize, nlines)`` float32.

    ``trans`` is the rectified strip (``nlines x ncols``); ``entry`` the
    frame's per-order aperture database (``ap<OBJ>_NO<i>_sscfm_m<m>trans``).
    Lines whose window leaves the strip stay 0.
    """
    from scipy import interpolate
    data_array = np.asarray(trans, dtype=np.float32).astype(">f4")
    naxis2, naxis1 = data_array.shape
    ap_low = float(entry.raw_low[0])
    ap_high = float(entry.raw_high[0])
    tracex = warp_trace_x(entry, naxis2)
    lowlim = math.floor(ap_low + 0.5)
    upplim = math.ceil(ap_high - 0.5)
    center = int((lowlim + upplim) / 2)
    lowdist = center - lowlim
    uppdist = upplim - center
    xnew = list(range(lowlim, upplim + 1))
    xsize = len(xnew)
    resampled = np.zeros((xsize, naxis2), dtype="float32")
    for y in range(naxis2):
        apcenter = tracex[y]
        center_i = int(apcenter + center)
        if (center_i - lowdist * 2 >= 1) and (center_i + uppdist * 2 + 1 <= naxis1):
            f = interpolate.interp1d(
                np.arange(center_i - lowdist * 2, center_i + uppdist * 2 + 1),
                data_array[y, center_i - lowdist * 2 - 1:center_i + uppdist * 2],
                kind=interpolation)
            xfine = np.arange(center_i - lowdist - 3, center_i + uppdist + 4, finepix)
            masks = [np.logical_and(xfine > apcenter + x - 0.5, xfine <= apcenter + x + 0.5)
                     for x in xnew]
            union = np.logical_or.reduce(masks)
            pos = np.cumsum(union) - 1
            fu = f(xfine[union])
            datanew = [np.average(fu[pos[mk]]) for mk in masks]
            resampled[:, y] += np.array(datanew) / np.sum(datanew) * np.sum(
                data_array[y, center_i - lowdist:center_i + uppdist + 1])
    return resampled
