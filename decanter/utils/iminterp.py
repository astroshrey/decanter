"""IRAF ``math/iminterp`` port: POLY5 fit + segment integration.

This module reproduces IRAF's 1-D image-interpolation routines used by
``noao.onedspec.dispcor`` (and indirectly by ``scombine``). The
algorithms here are direct ports of:

  - ``asifit`` POLY5 branch (``math/iminterp/asifit.x:136``) — pads the
    input with linear reflection at both ends.
  - ``ii_getpcoeff`` POLY5 branch (``math/iminterp/ii_1dinteg.x:215``)
    — Newton's-form conversion of the local 6-point window to
    polynomial coefficients in ``deltax = x - j`` for segment ``[j, j+1]``.
  - ``asigrl`` higher-order branch (``math/iminterp/asigrl.x:142``) —
    segment-by-segment integration of the local polynomial.

All arithmetic runs in ``float32`` (IRAF ``real``) so the float-precision
roundoff matches IRAF byte-for-byte.

Conventions
-----------
- ``asi`` coordinates are 1-based fractional positions in the array of
  length ``Nfit`` passed to ``asifit``. Valid range ``[1, Nfit]``.
- Segment ``j`` (1-based) covers ``[j, j+1]`` in asi coordinates and
  uses the local 6-point window ``coeff[j-1 .. j+4]`` (0-based numpy
  indexing into the padded coefficient array of length ``Nfit + 5``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from decanter.utils.fpemu import fma32, fma64, pow_ri

REAL = np.float32


def asifit_poly5(data: NDArray) -> NDArray[np.float32]:
    """Pad data per ``asifit`` POLY5 (2 left + 3 right slots, reflection).

    Mirrors ``math/iminterp/asifit.x:136``:

        coeff[c0ptr+1] = 2 * d[1] - d[3]
        coeff[c0ptr+2] = 2 * d[1] - d[2]
        coeff[cnptr+1] = 2 * d[N] - d[N-1]
        coeff[cnptr+2] = 2 * d[N] - d[N-2]
        coeff[cnptr+3] = 2 * d[N] - d[N-3]

    Returns a float32 array of length ``data.size + 5`` with the original
    data living at indices ``[2 .. 2 + N - 1]`` (asi ``OFFSET = 2``).
    """
    d = np.asarray(data, dtype=REAL)
    if d.size < 6:
        raise ValueError("asifit_poly5 requires at least 6 input points")
    n = d.size
    out = np.empty(n + 5, dtype=REAL)
    out[2 : 2 + n] = d
    out[0] = REAL(2) * d[0] - d[2]
    out[1] = REAL(2) * d[0] - d[1]
    out[n + 2] = REAL(2) * d[-1] - d[-2]
    out[n + 3] = REAL(2) * d[-1] - d[-3]
    out[n + 4] = REAL(2) * d[-1] - d[-4]
    return out


def per_segment_pcoeff_poly5(coeff: NDArray) -> NDArray[np.float32]:
    """Build the per-segment polynomial coefficients for POLY5.

    Bit-exact port of ``math/iminterp/ii_1dinteg.x:ii_getpcoeff`` (POLY5),
    vectorized over all valid segments. The Newton-to-power-basis shift
    ``diff[i] += diff[i-1] * (k - i - 3)`` is a fused multiply-add in the
    compiled IRAF (clang contracts it), so it is evaluated with
    :func:`~decanter.utils.fpemu.fma32`.

    Args:
        coeff: float32 padded coefficient array of length ``Nfit + 5``
            (output of :func:`asifit_poly5` on an asi array of length
            ``Nfit``).

    Returns:
        ``pcoeff`` of shape ``(6, Nfit - 1)``: for each segment ``j``
        (0-based ``j_idx = j - 1``, where IRAF ``j`` ∈ ``[1, Nfit - 1]``),
        ``pcoeff[i, j_idx]`` is the coefficient of ``deltax**i`` in the
        local poly5 expansion, where ``deltax = x - j`` for ``x`` in
        ``[j, j+1]`` (asi coords).
    """
    c = np.asarray(coeff, dtype=REAL)
    n_seg = c.size - 6
    if n_seg < 1:
        raise ValueError("coeff array too small")
    # IRAF: diff[i] = coeff[index - 3 + i], i = 1..6, index = OFFSET(2) + j
    # -> 0-based numpy coeff[j - 1 + r] for r = 0..5.
    diff = [c[r: r + n_seg].copy() for r in range(6)]
    for k in range(1, 6):
        for i in range(6 - k):
            diff[i] = ((diff[i + 1] - diff[i]) / REAL(k)).astype(REAL)
    for k in range(6, 1, -1):
        for i in range(2, k + 1):
            diff[i - 1] = fma32(diff[i - 2], REAL(k - i - 3), diff[i - 1])
    # pcoeff[i] (coefficient of deltax**(i+1-1)) = diff[nterms - i]
    return np.stack(diff[::-1]).astype(REAL)


_REC = [REAL(1) / REAL(i) for i in range(1, 7)]   # IRAF (1./i) in REAL


def _acc_d(accum: NDArray, t2: NDArray, d: NDArray) -> NDArray[np.float32]:
    """``accum += t2 * d`` where d is double: fused in double, stored REAL."""
    return fma64(t2.astype(np.float64), d, accum.astype(np.float64)).astype(REAL)


def asigrl_poly5(
    pcoeff: NDArray[np.float32], a: NDArray, b: NDArray
) -> NDArray[np.float32]:
    """Integrate the POLY5 interpolant from ``a`` to ``b`` (asi coords).

    Bit-exact vectorized port of ``math/iminterp/asigrl.x`` (higher-order
    branch) as compiled: ``(1./i)`` and ``pcoeff`` multiply in REAL, the
    ``deltax ** i`` powers are f2c ``pow_ri`` (double), and every
    ``accum = accum + ...`` is a fused multiply-add.

    Args:
        pcoeff: per-segment poly coefficients from
            :func:`per_segment_pcoeff_poly5`, shape ``(6, n_seg)``.
        a, b: float32 limits (asi coords). The integral is over
            ``[min(a,b), max(a,b)]``, negated when ``a > b`` (as IRAF).

    Returns:
        Float32 integrals, same shape as ``a``.
    """
    a = np.asarray(a, dtype=REAL)
    b = np.asarray(b, dtype=REAL)
    if a.shape != b.shape:
        raise ValueError("a and b must broadcast to the same shape")
    n_seg = pcoeff.shape[1]
    swap = a > b
    xa = np.where(swap, b, a)
    xb = np.where(swap, a, b)
    neara = xa.astype(np.int64)            # Fortran int() truncation
    nearb = xb.astype(np.int64)
    deltaa = (xa - neara.astype(REAL)).astype(REAL)
    deltab = (xb - nearb.astype(REAL)).astype(REAL)
    ia = np.clip(neara - 1, 0, n_seg - 1)
    ib = np.clip(nearb - 1, 0, n_seg - 1)
    single = neara == nearb
    accum = np.zeros(xa.shape, dtype=REAL)

    for i in range(1, 7):
        t2 = (_REC[i - 1] * pcoeff[i - 1, ia]).astype(REAL)
        pa = pow_ri(deltaa, i)
        d = np.where(single, pow_ri(deltab, i) - pa, 1.0 - pa)
        accum = _acc_d(accum, t2, d)

    nmid = np.where(single, 0, nearb - neara - 1)
    for step in range(int(nmid.max()) if nmid.size else 0):
        act = step < nmid
        j = np.clip(ia + 1 + step, 0, n_seg - 1)
        for i in range(1, 7):
            v = fma32(_REC[i - 1], pcoeff[i - 1, j], accum)
            accum = np.where(act, v, accum)

    multi = ~single
    for i in range(1, 7):
        t2 = (_REC[i - 1] * pcoeff[i - 1, ib]).astype(REAL)
        accum = np.where(multi, _acc_d(accum, t2, pow_ri(deltab, i)), accum)

    return np.where(a < b, accum, -accum).astype(REAL)


def _nint(x: NDArray) -> NDArray[np.int64]:
    """f2c ``i_nint`` (REAL arg promoted to double, round half away from 0)."""
    xd = np.asarray(x, np.float32).astype(np.float64)
    return np.where(xd >= 0, np.floor(xd + 0.5), -np.floor(0.5 - xd)).astype(np.int64)


def resample_poly5(
    input_data: NDArray,
    edges_x: NDArray[np.float64],
    *,
    flux: bool,
    xmin: float,
    xmax: float,
    dispcor: bool,
) -> NDArray[np.float32]:
    """Shared core of IRAF ``dispcor`` and ``shdr_linear`` (POLY5).

    ``edges_x`` are the ``n_out + 1`` output-pixel edges mapped into input
    logical pixels (double). Each edge becomes the REAL asi coordinate
    ``b = max(xmin, min(xmax, x)) + 1``; each output pixel is the POLY5
    integral over its edges divided by the width, except when both edges
    fall on input half-pixel boundaries, where IRAF sums ``asieval`` at the
    covered pixel centres instead (the exact-copy case).

    ``dispcor=True`` selects the ``dispcor.x`` details (the sum is not
    normalised by the pixel count, every output is divided by
    ``max(width, 1e-4)`` when ``flux`` is off, pixels whose two edges are
    both out of bounds are zero); ``False`` selects ``shdr.x:shdr_linear``
    (aligned sums divided by the number of pixel gaps, integrals by the
    width when positive).
    """
    in32 = np.asarray(input_data, dtype=REAL)
    n = in32.size
    temp = np.empty(n + 2, dtype=REAL)
    temp[0] = in32[0]
    temp[1: 1 + n] = in32
    temp[n + 1] = in32[-1]
    pcoeff = per_segment_pcoeff_poly5(asifit_poly5(temp))

    x = np.asarray(edges_x, dtype=np.float64)
    ofb = (x < xmin) | (x > xmax)
    e = (np.maximum(xmin, np.minimum(xmax, x)) + 1.0).astype(REAL)
    a = e[:-1]
    b = e[1:]
    fwd = a <= b
    lo = np.where(fwd, a, b)
    hi = np.where(fwd, b, a)
    lo5 = (lo + REAL(0.5)).astype(REAL)
    hi5 = (hi - REAL(0.5)).astype(REAL)
    jlo = _nint(lo5)
    jhi = _nint(hi5)
    aligned = (np.abs(lo5 - jlo.astype(REAL)) < REAL(1e-5)) & \
              (np.abs(hi5 - jhi.astype(REAL)) < REAL(1e-5))

    out = asigrl_poly5(pcoeff, lo, hi)
    idx = np.flatnonzero(aligned)
    if idx.size:
        # sum = sum + asieval(j) for j = jlo..jhi (asieval(j) == sample j),
        # each pixel's REAL sum accumulated left to right from 0; the pixels
        # are independent, so they advance together, one sample at a time
        jl = jlo[idx]
        cnt = jhi[idx] - jl + 1
        acc = np.zeros(idx.size, dtype=REAL)
        for step in range(int(cnt.max())):
            act = step < cnt
            j = np.where(act, jl - 1 + step, 0)
            acc = np.where(act, (acc + temp[j]).astype(REAL), acc)
        if not dispcor:
            gaps = cnt - 1
            div = gaps > 0
            acc = np.where(div, (acc / np.where(div, gaps, 1).astype(REAL)).astype(REAL), acc)
        out[idx] = acc
    width = (hi - lo).astype(REAL)
    if dispcor:
        if not flux:
            out = (out / np.maximum(width, REAL(1e-4))).astype(REAL)
        out = np.where(ofb[:-1] & ofb[1:], REAL(0), out)
    else:
        safe = np.where(width > 0, width, REAL(1))
        out = np.where(~aligned & (width > 0), (out / safe).astype(REAL), out)
    return out.astype(REAL)


def shdr_linear(
    input_data: NDArray,
    crval1_in: float,
    cdelt1_in: float,
    w0: float,
    w1: float,
    sn: int,
    *,
    crpix1_in: float = 1.0,
    mwcs: bool = False,
) -> NDArray[np.float32]:
    """IRAF ``smw/shdr.x:shdr_linear`` for a linear input WCS (POLY5).

    Resamples onto ``sn`` pixels spanning ``[w0, w1]``. ``w0``/``w1`` are
    REAL in IRAF, so the step is ``wp = REAL((w1 - w0) / (sn - 1))``; the
    edge wavelengths ``w = w0 + (x - 1) * wp`` are fused in double.

    ``mwcs=True`` maps the edge wavelengths to input pixels the way IRAF
    does (``shdr_wl`` through the compiled MWCS world->logical transform,
    ``sys/mwcs/mwsctran.x``: ``l = fma(1/CD, w, 1 - fma(1/CD, W, 0))``,
    for an input with ``CRPIX1 = 1`` and an identity Lterm) instead of
    ``(w - CRVAL1) / CDELT1 + CRPIX1``; the two differ by ~1e-10 pixel,
    which only rarely moves a REAL edge.
    """
    in32 = np.asarray(input_data, dtype=REAL)
    n = in32.size
    w0r = np.float32(w0)
    w1r = np.float32(w1)
    wp = float(np.float32(np.float32(w1r - w0r) / np.float32(sn - 1)))
    xs = np.arange(sn + 1, dtype=np.float64) + 0.5
    w = fma64(xs - 1.0, wp, float(w0r))
    if mwcs:
        if crpix1_in != 1.0:
            raise ValueError("mwcs=True mapping assumes CRPIX1 = 1")
        inv = 1.0 / cdelt1_in
        ltv = 0.0 + (1.0 - float(fma64(inv, crval1_in, 0.0)))
        x = (fma64(inv, w, ltv) - 1.0) + 1.0
    else:
        x = (w - crval1_in) / cdelt1_in + crpix1_in
    return resample_poly5(in32, x, flux=False, xmin=0.5, xmax=n + 0.5,
                          dispcor=False)


def dispcor_linear_poly5(
    input_data: NDArray,
    pixel_at_wavelength: callable,
    w_min: float,
    w_max: float,
    n_out: int,
    flux: bool = False,
) -> NDArray[np.float32]:
    """IRAF ``dispcor(flux=NO, linear=YES)`` for one row, given a λ→pixel map.

    ``pixel_at_wavelength`` maps the ``n_out + 1`` output-edge wavelengths
    (double) to fractional 1-based input pixels; the output grid is linear,
    ``w(x) = w_min + (x - 1) * dw``. See :func:`resample_poly5`. The exact
    WARP path (MWCS output grid + the multispec inverse) lives in
    :mod:`decanter.wavelength.dispcor`.
    """
    in32 = np.asarray(input_data, dtype=REAL)
    n = in32.size
    if n < 6:
        raise ValueError("dispcor_linear_poly5 requires at least 6 input pixels")
    if n_out < 1:
        raise ValueError("n_out must be positive")
    dw = (w_max - w_min) / (n_out - 1) if n_out > 1 else 1.0
    edge_x = np.arange(n_out + 1, dtype=np.float64) + 0.5
    edge_w = w_min + (edge_x - 1.0) * dw
    x = pixel_at_wavelength(edge_w)
    return resample_poly5(in32, x, flux=flux, xmin=0.5, xmax=n + 0.5,
                          dispcor=True)


def scombine_linear_poly5(
    input_data: NDArray,
    crval1_in: float,
    cdelt1_in: float,
    w1_out: float,
    dw_out: float,
    nw_out: int,
    flux: bool = False,
    *,
    w2_out: float | None = None,
    mwcs: bool = False,
) -> NDArray[np.float32]:
    """IRAF ``scopy``/``scombine`` resample of a linear-WCS spectrum.

    Both go through ``shdr.x:shdr_linear`` (see :func:`shdr_linear`), which
    short-circuits when the input WCS already equals the requested one.
    ``w2_out`` is the requested end wavelength (scopy's ``w2``); by default
    it is ``w1_out + (nw_out - 1) * dw_out``.
    """
    if cdelt1_in == 0:
        raise ValueError("cdelt1_in must be non-zero")
    if flux:
        raise NotImplementedError("flux-conserving resample not used by WARP")
    in_arr = np.asarray(input_data, dtype=REAL)
    nw_in = in_arr.size
    w_end = (w1_out + (nw_out - 1) * dw_out) if w2_out is None else w2_out
    # shdr_linear returns the input untouched when the input WCS already
    # matches (fp_equalr on the REAL endpoints, same length).
    if nw_in == int(nw_out):
        w0_in = np.float32(crval1_in + (1.0 - 1.0) * cdelt1_in)
        w1_in = np.float32(crval1_in + (nw_in - 1) * cdelt1_in)
        if _fp_equalr(w0_in, np.float32(w1_out)) and \
                _fp_equalr(w1_in, np.float32(w_end)):
            return in_arr.copy()
    return shdr_linear(in_arr, crval1_in, cdelt1_in, w1_out, w_end, int(nw_out),
                       mwcs=mwcs)


_R = np.float32


def _fp_normr(x: np.float32) -> tuple[np.float32, int]:
    """IRAF ``gio/fpnormr.x``: normalise to [1, 10) in REAL arithmetic."""
    tol = _R(_R(1.192e-7) * _R(10.0))
    absx = _R(abs(_R(x)))
    expon = 0
    if absx > 0:
        while absx < _R(_R(1.0) - tol):
            absx = _R(absx * _R(10.0))
            expon -= 1
            if absx == 0:
                return _R(0), 0
        while absx >= _R(_R(10.0) + tol):
            absx = _R(absx / _R(10.0))
            expon += 1
    return (_R(-absx) if x < 0 else absx), expon


def _fp_equalr(x, y) -> bool:
    """IRAF ``gio/fpequalr.x`` (REAL epsilon compare after normalisation)."""
    x = _R(x)
    y = _R(y)
    if x == y:
        return True
    if x == 0 or y == 0:
        return False
    nx, ex = _fp_normr(x)
    ny, ey = _fp_normr(y)
    if ex != ey:
        return False
    tol = _R(_R(1.192e-7) * _R(32.0))
    return _R(_R(1.0) + _R(abs(_R(nx - ny)))) <= _R(_R(1.0) + tol)
