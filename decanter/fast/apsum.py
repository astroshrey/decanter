"""IRAF ``apall`` box sum (``ap_sum``) of several images through one aperture.

:func:`decanter.extract.apsum_exact.apsum` (bit-exact against WARP) extracts
one image at a time and recomputes the aperture geometry each call. Fast mode
extracts the object and the sky strip of an order through the same aperture,
so this module evaluates the geometry once (cached per aperture: it only
depends on the database tokens and the strip size) and runs the data-dependent
part on a ``(K, lines, columns)`` stack. The arithmetic is the same REAL
(float32) sequence with the same fused multiply-adds; :func:`fma32x` is a
faster exact emulation of a float32 FMA (it agrees with
:func:`decanter.utils.fpemu.fma32` on every input), so the result is
bit-identical to ``apsum`` image by image.  The data are read as apall reads
them (FITS-kernel input mapping, :mod:`decanter.utils.iraf_imio`), except that
NaN stays NaN: in a band-evaluated strip it marks a pixel the rectification
did not compute, and must make the extraction non-finite.

IRAF equivalent: ``noao/twodspec/apextract/apextract.x`` (``ap_sum``,
``ap_edge``, ``ap_asifit``).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from decanter.extract.apsum_exact import (IrafRealBuffer, _ap_sum_line, _nint, _nint_v,
                                          extract_reals, irregular_lines)
from decanter.io.apdb import ApertureEntry
from decanter.utils.curfit_exact import ap_cveval
from decanter.utils.iraf_imio import imio_read_real

R = np.float32
_F64 = np.float64


def fma32x(a, b, c) -> NDArray[np.float32]:
    """``round32(a*b + c)`` with one rounding, for float32 operands.

    The product of two float32 values is exact in float64; ``s = p + c`` is
    rounded once to float64 with error ``e`` (``two_sum``). Rounding ``s`` to
    float32 is then correct unless ``s`` falls exactly on a float32 midpoint
    while ``e != 0`` (double rounding), which is detected from the low 29
    mantissa bits and fixed by moving one float32 ULP towards ``s + e``.
    """
    a = np.asarray(a, R).astype(_F64)
    b = np.asarray(b, R).astype(_F64)
    c = np.asarray(c, R).astype(_F64)
    p = a * b
    s = p + c
    bb = s - p
    e = (p - (s - bb)) + (c - bb)
    r = s.astype(R)
    mid = ((s.view(np.int64) & 0x1FFFFFFF) == 0x10000000) & (e != 0)
    if np.any(mid):
        rd = r.astype(_F64)
        r = np.where(mid & (e > 0) & (rd < s), np.nextafter(r, R(np.inf)), r)
        r = np.where(mid & (e < 0) & (rd > s), np.nextafter(r, R(-np.inf)), r)
    return r


def _grl_lin_v(cget, a: NDArray, b: NDArray) -> NDArray[np.float32]:
    """Linear ``asigrl`` (spans of at most two cells); ``a``/``b`` per line."""
    a = a.astype(R)
    b = b.astype(R)
    swap = a > b
    xa = np.where(swap, b, a)
    xb = np.where(swap, a, b)
    na = xa.astype(np.int64)
    nb = xb.astype(np.int64)
    if np.any(nb - na > 1):
        raise AssertionError("edge integral spans more than two cells")
    half = R(0.5)
    da = (xa - na.astype(R)).astype(R)
    db = (xb - nb.astype(R)).astype(R)
    ca0 = cget(na)
    ca1 = cget(na + 1)
    cb0 = cget(nb)
    cb1 = cget(nb + 1)
    diffa = ((ca1 - ca0).astype(R) * half).astype(R)
    single = fma32x(diffa, fma32x(db, db, -(da * da).astype(R)),
                    fma32x((db - da).astype(R), ca0, R(0)))
    first = fma32x(diffa, fma32x(-da, da, R(1.0)),
                   fma32x((R(1.0) - da).astype(R), ca0, R(0)))
    diffb = ((cb1 - cb0).astype(R) * half).astype(R)
    multi = fma32x((diffb * db).astype(R), db, fma32x(cb0, db, first))
    acc = np.where(na == nb, single, multi).astype(R)
    return np.where(a < b, acc, -acc).astype(R)


@lru_cache(maxsize=512)
def _geometry(tokens: tuple, nlines: int, ncols: int,
              buffer_cols: tuple[int, int] | None = None) -> dict:
    """Data-independent part of ``apsum`` for one aperture and strip size.

    ``buffer_cols`` is ``ap_extract``'s data-buffer column range when it is
    not the aperture's own (a background ICFIT widens it), as in
    :func:`decanter.extract.apsum_exact.apsum`.
    """
    entry = ApertureEntry(order=0, center_x=0.0, center_y=0.0, low=0.0, high=0.0,
                          function_type=1, poly_order=0, y_min=0.0, y_max=0.0,
                          coefficients=(), background_sample="INDEF",
                          raw_center=tokens[0], raw_low=tokens[1], raw_high=tokens[2],
                          raw_curve=tokens[3])
    (cen_a, cen_b), (low_a, low_b), (high_a, high_b), cv = extract_reals(entry)
    bmin = min(nlines, max(1, _nint(R(cen_b + low_b))))
    bmax = max(1, min(nlines, _nint(R(cen_b + high_b))))
    shifts = ap_cveval(cv, np.arange(bmin, bmax + 1, dtype=R))
    cmin = R(shifts.min())
    cmax = R(shifts.max())
    if buffer_cols is None:
        amin = min(ncols, max(1, _nint(R(R(cen_a + R(low_a - R(2))) + cmin))))
        amax = max(1, min(ncols, _nint(R(R(cen_a + R(high_a + R(2))) + cmax))))
    else:
        amin, amax = buffer_cols
    na = amax - amin + 1
    nb = bmax - bmin + 1
    astart = _nint_v((R(R(cen_a + low_a) - R(2)) + shifts).astype(R))
    rc1 = R(amin)
    low = R(cen_a + low_a)
    high = R(cen_a + high_a)
    step = R(R(high - low) / R(1))
    lines = np.arange(1, nb + 1, dtype=R)
    s = ((ap_cveval(cv, lines) - rc1).astype(R) + R(1)).astype(R)
    xsp = astart - amin + 1
    ax2 = np.minimum(R(R(na) + R(0.49)), (high + s).astype(R)).astype(R)
    n_fit = (_nint_v(ax2) + 1 - (xsp - 1)) + 1
    rxs = astart.astype(R)
    x1 = ((np.maximum(R(0.5), (low + s).astype(R)) + rc1).astype(R) - rxs).astype(R)
    x2 = ((np.minimum(R(R(na) + R(0.49)), ((low + step).astype(R) + s).astype(R))
           + rc1).astype(R) - rxs).astype(R)
    ix1 = _nint_v(x1)
    ix2 = _nint_v(x2)
    e1 = (x1 + R(1)).astype(R)
    e2 = (x2 + R(1)).astype(R)
    j1 = _nint_v(e1)
    j2 = _nint_v(e2)
    same = j1 == j2
    half = R(0.5)
    empty = x2 <= x1
    ax1 = np.maximum(R(0.5), (low + s).astype(R)).astype(R)
    # ap_sum's sky buffer width (ap_extract, W_NONE: nint(xmax) - nint(xmin) + 1)
    nx = _nint(R(R(cen_a + high_a) + R(2))) - _nint(R(R(cen_a + low_a) - R(2))) + 1
    bad = irregular_lines(empty=empty, ax1=ax1, ax2=ax2, n_fit=n_fit, xsp=xsp, ix1=ix1,
                          ix2=ix2, e1=e1, e2=e2, j1=j1, j2=j2, na=na)
    bad_sky = bad | irregular_lines(empty=empty, ax1=ax1, ax2=ax2, n_fit=n_fit, xsp=xsp,
                                    ix1=ix1, ix2=ix2, e1=e1, e2=e2, j1=j1, j2=j2, na=na,
                                    nx=nx)
    return dict(
        bmin=bmin, bmax=bmax, amin=amin, amax=amax, na=na, nb=nb, xsp=xsp,
        n_fit=n_fit, empty=empty, ix1=ix1, ix2=ix2, e1=e1, e2=e2, j1=j1, j2=j2,
        same=same, bad=bad, bad_sky=bad_sky, nx=nx, astart=astart, s=s, low=low,
        high=high, step=step, rc1=rc1,
        wt1=np.where(same, (e2 - e1).astype(R),
                     ((j1.astype(R) - e1).astype(R) + R(0.5)).astype(R)),
        wt2=np.where(same, R(0), ((e2 - j2.astype(R)).astype(R) + R(0.5)).astype(R)),
        j1lo=(j1.astype(R) - half).astype(R), j1hi=(j1.astype(R) + half).astype(R),
        j2lo=(j2.astype(R) - half).astype(R), j2hi=(j2.astype(R) + half).astype(R),
    )


def entry_tokens(entry: ApertureEntry) -> tuple:
    """Hashable raw database tokens of an aperture (the cache key)."""
    if entry.raw_center is None or entry.raw_curve is None:
        raise ValueError("aperture entry lacks raw database tokens")
    return (tuple(entry.raw_center), tuple(entry.raw_low), tuple(entry.raw_high),
            tuple(entry.raw_curve))


def apsum_multi(images, entry: ApertureEntry, *, buffer_cols: tuple[int, int] | None = None,
                sky: NDArray | None = None) -> NDArray[np.float32]:
    """``apsum`` of several same-shape strips through one aperture.

    Returns ``(K, n_lines)`` float32, row ``k`` bit-identical to
    ``decanter.extract.apsum_exact.apsum(images[k], entry)``.

    ``sky`` (``(K, nb)``, one value per aperture line, or None) is ``ap_sum``'s
    background subtraction (apextract.x:1576-1583): the sky over the same
    pixels with the same edge weights, ``fma(wt1, s, wt2 * s)`` plus ``s``
    per interior pixel, subtracted from the sum; ``buffer_cols`` is then the
    background-widened data buffer
    (:func:`decanter.extract.apbackground.extract_geometry`).  Rows match
    :func:`decanter.extract.apbackground.apsum_background`.
    """
    nlines, ncols = np.shape(images[0])
    K = len(images)
    g = _geometry(entry_tokens(entry), nlines, ncols,
                  None if buffer_cols is None else tuple(int(v) for v in buffer_cols))
    bmin, bmax, amin, amax = g["bmin"], g["bmax"], g["amin"], g["amax"]
    na, nb, xsp, n_fit = g["na"], g["nb"], g["xsp"], g["n_fit"]
    rows = np.arange(nb)
    # the data buffer as apall reads it through the FITS kernel (subnormals
    # and -0 -> +0, Inf -> 0; utils/iraf_imio).  NaN is kept: in a
    # band-evaluated strip it marks a pixel the rectification skipped, and an
    # extraction that reads one must come out non-finite (orders_to_redo).
    buf = np.stack([imio_read_real(im[bmin - 1: bmax, amin - 1: amax], keep_nan=True)
                    for im in images])                            # (K, nb, na)

    def pick(pos):
        return buf[:, rows, pos]

    def at(jj):
        return pick(np.clip(xsp - 3 + jj, 0, na - 1))

    last = at(n_fit)
    prev = at(n_fit - 1)
    pad = ((last * R(2.0)).astype(R) - prev).astype(R)

    def cget(j):
        return np.where(j == n_fit + 1, pad, at(np.minimum(j, n_fit))).astype(R)

    v1 = cget(g["j1"])
    b1 = _grl_lin_v(cget, g["j1lo"], g["j1hi"])
    a1 = _grl_lin_v(cget, g["e1"], np.where(g["same"], g["e2"], g["j1hi"]))
    ok1 = (v1 > 0) & (b1 > 0) & (a1 > 0) & (a1 < b1)
    wt1 = np.where(ok1, (a1 / np.where(b1 == 0, R(1), b1)).astype(R), g["wt1"])
    v2 = cget(g["j2"])
    b2 = _grl_lin_v(cget, g["j2lo"], g["j2hi"])
    a2 = _grl_lin_v(cget, g["j2lo"], g["e2"])
    ok2 = (~g["same"]) & (v2 > 0) & (b2 > 0) & (a2 > 0) & (a2 < b2)
    wt2 = np.where(ok2, (a2 / np.where(b2 == 0, R(1), b2)).astype(R), g["wt2"])

    def dat(k):
        return pick(np.clip(xsp - 2 + k, 0, na - 1))

    ix1, ix2 = g["ix1"], g["ix2"]
    sval = fma32x(wt1.astype(R), dat(ix1), (wt2 * dat(ix2)).astype(R))
    width = ix2 - ix1 - 1
    for t in range(int(width.max()) if width.size else 0):
        sval = np.where(t < width, (sval + dat(ix1 + 1 + t)).astype(R), sval)
    if sky is not None:
        sk = np.asarray(sky, dtype=R).reshape(K, nb)
        skyval = fma32x(wt1.astype(R), sk, (wt2 * sk).astype(R))
        for t in range(int(width.max()) if width.size else 0):
            skyval = np.where(t < width, (skyval + sk).astype(R), skyval)
        sval = (sval - skyval).astype(R)
    sval = np.where(g["empty"], R(0), sval).astype(R)
    # lines whose IRAF reads leave the line (bmin > 1 geometries only): the
    # scalar ap_sum line, IRAF's pointer arithmetic, or a refusal
    bad = g["bad"] if sky is None else g["bad_sky"]
    if bad.any():
        nx = g["nx"]
        for k in range(K):
            flat = buf[k].reshape(-1)
            sky_k = None if sky is None else IrafRealBuffer(np.repeat(sk[k], nx), "sky buffer")
            for i in np.nonzero(bad)[0]:
                sval[k, i] = _ap_sum_line(flat, int(i) * na, na, int(xsp[i]),
                                          int(g["astart"][i]), g["s"][i], g["low"], g["high"],
                                          g["step"], g["rc1"], sky=sky_k, sky0=int(i) * nx)
    out = np.zeros((K, nlines), dtype=R)
    out[:, bmin - 1: bmax] = sval
    return out
