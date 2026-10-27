"""Bit-exact port of IRAF ``center1d`` (``pkg/xtools/center1d.x``).

``center1d`` is what ``ecidentify`` (and ``identify``) use to centre a
feature: a triangle-weighted centroid computed from integrals of the
``spline3`` interpolant of the data (``math/iminterp``). Everything is REAL
(float32). The fused multiply-adds below were read off the arm64 object code
of this machine's IRAF (``libxtools.a(center1d.o)``,
``libiminterp.a(ii_spline.o, ii_1dinteg.o, asigrl.o)``) and the port is
checked against those compiled routines through the ctypes oracle
(``scripts/iraf_oracle``).

One piece of state is global, as in IRAF: ``c1d_center``'s ``dxcheck``
counter is a static local that is never initialised, so it carries over from
one call to the next within the IRAF process. :class:`Center1D` keeps it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from decanter.utils.fpemu import fma32, fma64, pow_ri

R = np.float32
INDEF = None
_EMISSION, _ABSORPTION = 1, 2
_REC = [R(R(1) / R(i)) for i in range(1, 5)]


def _f(x) -> np.float32:
    return R(x)


def _fma(a, b, c) -> np.float32:
    return R(fma32(R(a), R(b), R(c)))


def asifit_spline3(data: NDArray) -> NDArray[np.float32]:
    """``asifit`` (II_SPLINE3) + ``ii_spline``: the B-spline coefficients.

    Returns the coefficient array 0-based: element ``k`` is IRAF
    ``COEFF[k+1]`` (length ``npix + 3``, data offset 1).
    """
    d = np.asarray(data, dtype=R)
    n = d.size
    if n < 4:
        raise ValueError("ASIFIT: too few points for SPLINE3")
    b = np.zeros(n + 3, dtype=R)
    b[1:n + 1] = d
    # ii_spline on bcoeff[1..n+2] (0-based b[0..n+1]); diag 1-based.
    diag = np.zeros(n + 3, dtype=R)
    six = R(6)
    diag[1] = R(-2)
    b[0] = R(b[0] / six)
    diag[2] = R(0)
    b[1] = R(R(b[1] - b[0]) / six)
    for i in range(3, n + 2):
        diag[i] = R(R(1) / R(R(4) - diag[i - 1]))
        b[i - 1] = R(diag[i] * R(b[i - 1] - b[i - 2]))
    s2 = R(diag[n] + R(2))
    num = R(_fma(s2, b[n], -b[n - 1]) + R(b[n + 1] / six))
    den = _fma(diag[n + 1], s2, R(1))
    b[n + 1] = R(num / den)
    for i in range(n + 1, 2, -1):
        b[i - 1] = _fma(-diag[i], b[i], b[i - 1])
    b[0] = R(_fma(b[1], R(2), b[0]) - b[2])
    return b


def _pcoeff(c: NDArray[np.float32], index: int) -> list[np.float32]:
    """``ii_getpcoeff`` (II_SPLINE3) for 1-based ``index`` (ii_1dinteg.o)."""
    cm1, c0, c1, c2 = c[index - 2], c[index - 1], c[index], c[index + 1]
    three = R(3)
    p1 = R(_fma(c0, R(4), cm1) + c1)
    p2 = R(R(c1 - cm1) * three)
    p3 = R(R(_fma(c0, R(-2), cm1) + c1) * three)
    p4 = R(_fma(-c1, three, _fma(c0, three, -cm1)) + c2)
    return [p1, p2, p3, p4]


def _acc_d(accum, t2, d) -> np.float32:
    return R(fma64(float(t2), float(d), float(accum)))


def asigrl_spline3(c: NDArray[np.float32], a, b) -> np.float32:
    """``asigrl`` for an II_SPLINE3 interpolant (``asigrl.x``, default branch)."""
    a = R(a)
    b = R(b)
    xa, xb = (b, a) if a > b else (a, b)
    neara = int(xa)
    nearb = int(xb)
    accum = R(0)
    deltaa = R(xa - R(neara))
    if nearb == neara:
        deltab = R(xb - R(nearb))
        p = _pcoeff(c, 1 + neara)
        for i in range(1, 5):
            t2 = R(_REC[i - 1] * p[i - 1])
            accum = _acc_d(accum, t2, float(pow_ri(deltab, i)) - float(pow_ri(deltaa, i)))
    else:
        p = _pcoeff(c, 1 + neara)
        for i in range(1, 5):
            t2 = R(_REC[i - 1] * p[i - 1])
            accum = _acc_d(accum, t2, 1.0 - float(pow_ri(deltaa, i)))
        for j in range(neara + 1, nearb):
            p = _pcoeff(c, 1 + j)
            for i in range(1, 5):
                accum = _fma(_REC[i - 1], p[i - 1], accum)
        deltab = R(xb - R(nearb))
        p = _pcoeff(c, 1 + nearb)
        for i in range(1, 5):
            t2 = R(_REC[i - 1] * p[i - 1])
            accum = _acc_d(accum, t2, float(pow_ri(deltab, i)))
    return accum if a < b else R(-accum)


class Center1D:
    """``center1d`` with IRAF's process-wide state (``dxcheck``).

    Create one per IRAF process being emulated (e.g. one ``ecidentify``
    run) and call it for every feature in order.
    """

    EPSILON = R(0.001)
    EPSILON1 = R(0.005)

    def __init__(self) -> None:
        self.dxcheck = 0

    def __call__(self, x, data: NDArray, width, ftype: int, radius, threshold):
        """``center1d(x, data, npts, width, type, radius, threshold)``.

        Returns the REAL centre or ``None`` (IRAF INDEF).
        """
        data = np.asarray(data, dtype=R)
        npts = data.size
        x = R(x)
        width, radius, threshold = R(width), R(radius), R(threshold)
        if x < R(1) or x > R(npts):
            return None
        wid = max(width, R(3))
        rad = max(R(2), radius)
        half = R(wid / R(2))
        x1 = int(max(R(1), R(R(R(x - half) - rad) - wid)))
        x2 = int(min(R(npts), R(R(R(R(x + half) + rad) + wid) + R(1))))
        nx = x2 - x1 + 1
        seg = data[x1 - 1:x1 - 1 + nx]
        a = R(seg.min())
        b = R(seg.max())
        if R(b - a) < threshold:
            return None
        x1 = int(max(R(1), R(R(x - half) - rad)))
        x2 = int(min(R(npts), R(R(R(x + half) + rad) + R(1))))
        nx = x2 - x1 + 1
        seg = data[x1 - 1:x1 - 1 + nx]
        if ftype == _EMISSION:
            a = min(R(0), a)
            d1 = (seg - R(a + threshold)).astype(R)
            d1 = np.maximum(d1, R(0)).astype(R)
        elif ftype == _ABSORPTION:
            d1 = (-seg).astype(R)
            d1 = (d1 - R(threshold - b)).astype(R)
            d1 = np.maximum(d1, R(0)).astype(R)
        else:
            raise ValueError("Unknown feature type")
        xc = self._center(R(R(x - R(x1)) + R(1)), d1, width)
        if xc is not None:
            xc = R(R(xc + R(x1)) - R(1))
            if abs(R(x - xc)) > radius:
                xc = None
        return xc

    def _center(self, x, data, width):
        npts = data.size
        d = data   # 0-based; IRAF data[i] is d[i-1]
        i = int(R(x + R(0.5)))
        while i < npts and d[i - 1] <= d[i]:
            i += 1
        while i > 1 and d[i - 1] <= d[i - 2]:
            i -= 1
        j = int(R(x + R(0.5)))
        while j > 1 and d[j - 1] <= d[j - 2]:
            j -= 1
        while j < npts and d[j - 1] <= d[j]:
            j += 1
        if abs(R(R(i) - x)) < abs(R(x - R(j))):
            xc = R(i)
        else:
            xc = R(j)
        if width <= R(1):
            return xc
        wid = max(width, R(3))
        hwidth = R(wid / R(2))
        if R(xc - hwidth) < R(1) or R(xc + hwidth) > R(npts):
            return None
        c1 = asifit_spline3(d)
        xd = (d * np.arange(1, npts + 1, dtype=R)).astype(R)
        c2 = asifit_spline3(xd)
        dxlast = R(npts)
        h2 = R(hwidth * R(0.5))
        for _ in range(100):
            a = R(xc - hwidth)
            b = R(xc - h2)
            i1 = asigrl_spline3(c1, a, b)
            i2 = asigrl_spline3(c2, a, b)
            sum1 = _fma(R(xc - hwidth), i1, -i2)
            sum2 = R(-i1)
            a = b
            b = R(xc + h2)
            i1 = asigrl_spline3(c1, a, b)
            i2 = asigrl_spline3(c2, a, b)
            sum1 = R(_fma(-xc, i1, sum1) + i2)
            sum2 = R(sum2 + i1)
            a = b
            b = R(xc + hwidth)
            i1 = asigrl_spline3(c1, a, b)
            i2 = asigrl_spline3(c2, a, b)
            sum1 = R(_fma(R(xc + hwidth), i1, sum1) - i2)
            sum2 = R(sum2 - i1)
            if sum2 == R(0):
                return None
            dx = R(sum1 / abs(sum2))
            dxabs = abs(dx)
            step = max(R(-1), min(R(1), dx))
            xc = R(xc + step)
            if R(xc - hwidth) < R(1) or R(xc + hwidth) > R(npts):
                return None
            if dxabs < self.EPSILON:
                return xc
            if dxabs > R(dxlast + self.EPSILON1):
                self.dxcheck += 1
                if self.dxcheck > 3:
                    return None
            elif dxabs > R(dxlast - self.EPSILON1):
                xc = R(xc - R(step / R(2)))
                self.dxcheck = 0
            else:
                self.dxcheck = 0
                dxlast = dxabs
        return None
