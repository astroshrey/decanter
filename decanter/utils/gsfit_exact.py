"""Bit-exact port of IRAF's double-precision 2-D surface *fit* (``dgsfit``).

Complements :mod:`decanter.utils.gsurfit_exact` (the evaluators) with the
fitting half of ``math/gsurfit`` as ``ecidentify`` uses it:
``dgsinit`` (gsinitd.x) + ``dgsfit`` (gsfitd.x) = ``dgszero`` +
``dgsacpts`` (gsacptsd.x, WTS_USER) + ``dgssolve`` (gssolved.x,
``dgschofac`` / ``dgschoslv`` from gs_chomatd.x), for Chebyshev/Legendre
surfaces with full cross terms and no reference-point subtraction.

Compiled arithmetic (``objdump -d`` of ``libgsurfit.a(gsacptsd.o,
gs_chomatd.o)`` and ``libvops.a(adotd.o)`` on this machine, and checked
against those objects through the ctypes oracle):

- the basis vectors are ``dgs_bcheb`` / ``dgs_bleg`` (shared with the
  evaluators);
- ``byw = w*yb`` and ``bw = byw*xb`` are separate roundings (``amuld``);
- ``VECTOR += adotd(bw, z)``: ``adotd`` adds the products of the first
  ``8*floor(n/8)`` points one by one (each product rounded), then fuses the
  remaining ``n mod 8`` (``fmadd``);
- ``MATRIX += (bw*xb)*yb`` over the points in order, one ``fmadd`` each;
- Cholesky ``c -= a*b`` updates are fused.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from decanter.utils.fpemu import fma64_cr as fma64
from decanter.utils.gsurfit_exact import (
    GS_CHEBYSHEV, GS_LEGENDRE, GS_XFULL, GsSurface, _bcheb, _bleg,
)

D = np.float64
OK, SINGULAR, NO_DEG_FREEDOM = 0, 1, 2
_MAX_DOUBLE = 1.7976931348623157e308


def adotd(a: NDArray, b: NDArray) -> float:
    """vops ``adotd`` as compiled (8-wide unrolled body, fused tail)."""
    a = np.asarray(a, D)
    b = np.asarray(b, D)
    n = a.size
    m = 8 * (n // 8)
    s = 0.0
    prod = a[:m] * b[:m]
    for v in prod:
        s = s + float(v)
    for i in range(m, n):
        s = float(fma64(a[i], b[i], s))
    return s


def _chofac(matrix: NDArray) -> tuple[NDArray, int]:
    nb, nr = matrix.shape            # [band, row]
    ier = OK
    mf = matrix.copy()
    if nr == 1:
        if matrix[0, 0] > 0.:
            mf[0, 0] = 1. / matrix[0, 0]
        return mf, ier
    for n in range(nr):
        if ((mf[0, n] + matrix[0, n]) - matrix[0, n]) <= 1000 / _MAX_DOUBLE:
            mf[:, n] = 0.
            ier = SINGULAR
            continue
        mf[0, n] = 1. / mf[0, n]
        imax = min(nb - 1, nr - 1 - n)
        if imax < 1:
            continue
        jmax = imax
        for i in range(1, imax + 1):
            ratio = mf[i, n] * mf[0, n]
            for j in range(1, jmax + 1):
                mf[j - 1, n + i] = float(fma64(-mf[j - 1 + i, n], ratio, mf[j - 1, n + i]))
            jmax -= 1
            mf[i, n] = ratio
    return mf, ier


def _choslv(mf: NDArray, vector: NDArray) -> NDArray:
    nb, nr = mf.shape
    if nr == 1:
        return np.array([vector[0] * mf[0, 0]])
    c = [float(v) for v in vector]
    nbm1 = nb - 1
    for n in range(nr):
        for j in range(1, min(nbm1, nr - 1 - n) + 1):
            c[j + n] = float(fma64(-mf[j, n], c[n], c[j + n]))
    for n in range(nr - 1, -1, -1):
        c[n] = c[n] * mf[0, n]
        for j in range(1, min(nbm1, nr - 1 - n) + 1):
            c[n] = float(fma64(-mf[j, n], c[j + n], c[n]))
    return np.array(c, dtype=D)


def dgsfit(stype: int, xorder: int, yorder: int, xmin: float, xmax: float,
           ymin: float, ymax: float, x, y, z, w) -> tuple[GsSurface, int]:
    """``dgsinit`` (xterms = full) + ``dgsfit`` (WTS_USER).

    Returns the fitted surface and IRAF's ``ier``.
    """
    if stype not in (GS_CHEBYSHEV, GS_LEGENDRE):
        raise NotImplementedError(f"gsurfit type {stype}")
    x = np.asarray(x, D)
    y = np.asarray(y, D)
    z = np.asarray(z, D)
    w = np.asarray(w, D)
    xrange_ = 2. / (xmax - xmin)
    xmaxmin = -(xmax + xmin) / 2.
    yrange_ = 2. / (ymax - ymin)
    ymaxmin = -(ymax + ymin) / 2.
    basis = _bcheb if stype == GS_CHEBYSHEV else _bleg
    xb = basis(x, xorder, xmaxmin, xrange_)
    yb = basis(y, yorder, ymaxmin, yrange_)
    ncoeff = xorder * yorder
    vector = np.zeros(ncoeff, D)
    matrix = np.zeros((ncoeff, ncoeff), D)      # [band, row]
    rows, cols, bws = [], [], []
    for l in range(yorder):
        byw = w * yb[l]
        for k in range(xorder):
            row = l * xorder + k
            bw = byw * xb[k]
            vector[row] = vector[row] + adotd(bw, z)
            for j in range(row, ncoeff):
                rows.append(row)
                cols.append(j)
                bws.append(bw * xb[j % xorder])
    # MATRIX(mindex) = MATRIX(mindex) + (bw*xb)*yb, point by point (fmadd)
    prod = np.array(bws)                                   # (npairs, npts)
    ybs = np.array([yb[j // xorder] for j in cols])        # (npairs, npts)
    acc = np.zeros(len(rows), D)
    for p in range(x.size):
        acc = fma64(prod[:, p], ybs[:, p], acc)
    for r, c, v in zip(rows, cols, acc):
        matrix[c - r, r] = v
    npts = x.size
    sf = GsSurface(stype=stype, xorder=xorder, yorder=yorder, xterms=GS_XFULL,
                   xmin=float(xmin), xmax=float(xmax), ymin=float(ymin), ymax=float(ymax),
                   coeff=np.zeros(ncoeff, D), nxcoeff=xorder, nycoeff=yorder,
                   xrange=xrange_, xmaxmin=xmaxmin, yrange=yrange_, ymaxmin=ymaxmin)
    if npts - ncoeff < 0:
        return sf, NO_DEG_FREEDOM
    mf, ier = _chofac(matrix)
    sf.coeff = _choslv(mf, vector)
    return sf, ier


def save(sf: GsSurface) -> list[float]:
    """``dgssave``: ``[type, xorder, yorder, xterms, xmin, xmax, ymin, ymax, c...]``."""
    return [float(sf.stype), float(sf.xorder), float(sf.yorder), float(sf.xterms),
            sf.xmin, sf.xmax, sf.ymin, sf.ymax] + [float(c) for c in sf.coeff]
