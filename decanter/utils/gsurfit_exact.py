"""Bit-exact port of IRAF's double-precision 2-D surface fitting evaluators.

IRAF ``longslit.transform`` reads ``fitcoords`` surfaces into the double
precision ``math/gsurfit`` package (``dgsrestore``) through the REAL
interface of ``noao/twodspec/longslit/transform/igsfit/xgs.x``:

- ``xgseval(sf, x, y)`` -> ``real(dgseval(sf, double(x), double(y)))``
  (single-point evaluator, ``gsevald.x`` + ``gs_b1evald.x``),
- ``xgsder(sf, x, y, z, n, nxd, nyd)`` -> ``dgsder`` (``gsderd.x``), which
  for ``nxd = nyd = 0`` is ``dgsvector`` (``gs_fevald.x``) and otherwise
  builds the derivative surface and evaluates it (``gs_fderd.x`` +
  ``gs_devald.x``), then normalises.

Every arithmetic step below follows the f2c C and the arm64 object code of
this Mac's ``x_longslit.e`` (iraf-community 2.18.1, clang -O2):

- ``awsud(a, b, c, k1, k2)`` compiles to ``c = fma(a, k1, b*k2)`` with the
  ``b*k2`` product rounded first,
- ``gs_asumvpd`` (``d = c + a*b``) is a fused multiply-add,
- ``dgs_b1cheb``'s recurrence ``2*xnorm*b[i-1] - b[i-2]`` is fused
  (``fnmsub``); ``dgs_bcheb``'s vector recurrence is not (``amuld``,
  ``amulkd``, ``asubd`` are separate roundings),
- ``dgseval``'s inner ``accum = accum + coeff*xb`` is fused for
  ``xorder < 8`` (the scalar loop tail; the 8-wide vector body is not).

Only the cases WARP's calibration produces are implemented: Chebyshev
(type 1) and Legendre (type 2) surfaces with full cross terms (xterms 1).
Anything else raises :class:`NotImplementedError`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from decanter.utils.fpemu import fma64, iraf_ctod

f32 = np.float32
f64 = np.float64

GS_CHEBYSHEV = 1
GS_LEGENDRE = 2
GS_POLYNOMIAL = 3
GS_XNONE = 0
GS_XFULL = 1
GS_XHALF = 2


def _nint(x: float) -> int:
    """f2c ``i_dnnt``: round half away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(0.5 - x))


@dataclass
class GsSurface:
    """A restored ``dgsurfit`` surface (``dgsrestore``)."""

    stype: int
    xorder: int
    yorder: int
    xterms: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    coeff: NDArray[np.float64]
    nxcoeff: int = 0
    nycoeff: int = 0
    xrange: float = 0.0
    xmaxmin: float = 0.0
    yrange: float = 0.0
    ymaxmin: float = 0.0
    units: str | None = None
    axis: int = 2
    name: str = ""
    raw: tuple = field(default_factory=tuple)

    @classmethod
    def restore(cls, fit, **kw) -> "GsSurface":
        fit = [float(v) for v in fit]
        xorder = _nint(fit[1])
        yorder = _nint(fit[2])
        stype = _nint(fit[0])
        xterms = int(fit[3])  # SPP int = double assignment (truncation)
        xmin, xmax, ymin, ymax = fit[4], fit[5], fit[6], fit[7]
        if xorder < 1 or yorder < 1 or xmax <= xmin or ymax <= ymin:
            raise ValueError("GSRESTORE: illegal surface parameters")
        if stype not in (GS_CHEBYSHEV, GS_LEGENDRE):
            raise NotImplementedError(f"gsurfit type {stype} not implemented")
        if xterms != GS_XFULL:
            raise NotImplementedError(f"gsurfit xterms={xterms} not implemented")
        ncoeff = xorder * yorder
        coeff = np.asarray(fit[8:8 + ncoeff], dtype=f64)
        if coeff.size != ncoeff:
            raise ValueError("GSRESTORE: too few coefficients")
        return cls(
            stype=stype, xorder=xorder, yorder=yorder, xterms=xterms,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, coeff=coeff,
            nxcoeff=xorder, nycoeff=yorder,
            xrange=2.0 / (xmax - xmin), xmaxmin=-(xmax + xmin) / 2.0,
            yrange=2.0 / (ymax - ymin), ymaxmin=-(ymax + ymin) / 2.0,
            raw=tuple(fit), **kw,
        )

    @classmethod
    def from_fc_file(cls, path: Path | str) -> "GsSurface":
        """Read a ``fitcoords`` database entry the way ``lm_dbread`` does.

        Values are parsed with IRAF ``ctod`` (``dtgad``), not ``float()``.
        """
        lines = Path(path).read_text().splitlines()
        name, axis, units, n, vals = "", 2, None, None, []
        i = 0
        while i < len(lines):
            parts = lines[i].split()
            i += 1
            if not parts or parts[0].startswith("#"):
                continue
            key = parts[0]
            if key == "begin":
                name = " ".join(parts[1:])
            elif key == "axis":
                axis = int(parts[1])
            elif key == "units":
                units = " ".join(parts[1:])
            elif key == "surface":
                n = int(parts[1])
                while len(vals) < n and i < len(lines):
                    vals.extend(lines[i].split())
                    i += 1
                break
        if n is None or len(vals) < n:
            raise ValueError(f"{path}: no complete 'surface' record")
        fit = [iraf_ctod(v) for v in vals[:n]]
        return cls.restore(fit, units=units, axis=axis, name=name)

    # -- REAL interface getters (xgsgetr) -----------------------------------
    def getr(self, which: str) -> np.float32:
        return f32({"xmin": self.xmin, "xmax": self.xmax,
                    "ymin": self.ymin, "ymax": self.ymax}[which])


# ---------------------------------------------------------------------------
# Basis functions (vector forms, gs_bevald.x / gs_devald.x)
# ---------------------------------------------------------------------------

def _awsud(a, b, k1, k2):
    """vops ``awsud``: ``c = fma(a, k1, round(b*k2))``."""
    return fma64(a, k1, np.asarray(b, f64) * f64(k2))


def _bcheb(x, order, k1, k2):
    out = [np.ones_like(x)]
    if order >= 2:
        out.append((x + k1) * k2)  # altad
    for _k in range(3, order + 1):
        t = out[1] * out[-1]       # amuld
        t = t * 2.0                # amulkd
        out.append(t - out[-2])    # asubd
    return out[:order]


def _bleg(x, order, k1, k2):
    out = [np.ones_like(x)]
    if order >= 2:
        out.append((x + k1) * k2)
    for k in range(3, order + 1):
        ri = float(k)
        ri1 = (2.0 * ri - 3.0) / (ri - 1.0)
        ri2 = -(ri - 2.0) / (ri - 1.0)
        t = out[1] * out[-1]
        out.append(_awsud(t, out[-2], ri1, ri2))
    return out[:order]


def _dcheb(x, order, nder, k1, k2):
    if nder == 0:
        return _bcheb(x, order, k1, k2)
    n = order + nder
    xn = (x + k1) * k2
    fn = [np.ones_like(x)]
    if n >= 2:
        fn.append(xn.copy())
    for _k in range(3, n + 1):
        t = xn * fn[-1]
        t = t * 2.0
        fn.append(t - fn[-2])
    dfn = fn
    for i in range(1, nder + 1):
        new = []
        for k in range(1, n + 1):
            if k == 1:
                new.append(np.zeros_like(x))
            elif k == 2:
                new.append(np.ones_like(x) if i == 1 else np.zeros_like(x))
            else:
                t = xn * new[-1]
                t = t * 2.0
                t = t - new[-2]
                fac = 2.0 * float(i)
                new.append(_awsud(fn[k - 2], t, fac, 1.0))
        fn = new
        dfn = new
    return dfn[nder:nder + order]


def _dleg(x, order, nder, k1, k2):
    if nder == 0:
        return _bleg(x, order, k1, k2)
    n = order + nder
    xn = (x + k1) * k2
    fn = [np.ones_like(x)]
    if n >= 2:
        fn.append(xn.copy())
    for k in range(3, n + 1):
        ri = float(k)
        ri1 = (2.0 * ri - 3.0) / (ri - 1.0)
        ri2 = -(ri - 2.0) / (ri - 1.0)
        t = xn * fn[-1]
        fn.append(_awsud(t, fn[-2], ri1, ri2))
    dfn = fn
    for i in range(1, nder + 1):
        new = []
        for k in range(1, n + 1):
            if k == 1:
                new.append(np.zeros_like(x))
            elif k == 2:
                new.append(np.ones_like(x) if i == 1 else np.zeros_like(x))
            else:
                ri = float(k)
                ri1 = (2.0 * ri - 3.0) / (ri - 1.0)
                ri2 = -(ri - 2.0) / (ri - 1.0)
                t = xn * new[-1]
                t = _awsud(t, new[-2], ri1, ri2)
                fac = ri1 * float(i)
                new.append(_awsud(fn[k - 2], t, fac, 1.0))
        fn = new
        dfn = new
    return dfn[nder:nder + order]


def _basis(stype, x, order, nder, k1, k2):
    if stype == GS_CHEBYSHEV:
        return _dcheb(x, order, nder, k1, k2)
    return _dleg(x, order, nder, k1, k2)


def _accumulate(coeff, xb, yb, xorder, yorder):
    """``dgs_evcheb`` / ``dgs_dercheb`` full-cross-terms accumulation."""
    zfit = np.zeros_like(xb[0])
    cptr = 0
    for i in range(yorder):
        accum = np.zeros_like(xb[0])
        for k in range(xorder):
            accum = _awsud(accum, xb[k], 1.0, coeff[cptr + k])
        zfit = fma64(accum, yb[i], zfit)          # gs_asumvpd (fused)
        cptr += xorder
    return zfit


def gs_vector(sf: GsSurface, x, y) -> NDArray[np.float64]:
    """``dgsvector`` for a surface with ``xorder > 1`` and ``yorder > 1``."""
    x = np.asarray(x, f64)
    y = np.asarray(y, f64)
    if sf.xorder == 1 or sf.yorder == 1:
        raise NotImplementedError("1-D gsurfit evaluation not implemented")
    xb = _basis(sf.stype, x, sf.xorder, 0, sf.xmaxmin, sf.xrange)
    yb = _basis(sf.stype, y, sf.yorder, 0, sf.ymaxmin, sf.yrange)
    return _accumulate(sf.coeff, xb, yb, sf.xorder, sf.yorder)


def gs_der(sf: GsSurface, x, y, nxd: int, nyd: int) -> NDArray[np.float64]:
    """``dgsder`` (full cross terms)."""
    if nxd == 0 and nyd == 0:
        return gs_vector(sf, x, y)
    x = np.asarray(x, f64)
    y = np.asarray(y, f64)
    nxder = min(nxd, sf.nxcoeff)
    nyder = min(nyd, sf.nycoeff)
    if nxder >= sf.nxcoeff or nyder >= sf.nycoeff:
        # Derivative surface is identically zero (COEFF = 0); its value and
        # normalisation give exact zeros.
        return np.zeros(np.broadcast(x, y).shape, f64)
    nx2 = max(1, sf.nxcoeff - nxder)
    xo2 = max(1, sf.xorder - nxder)
    ny2 = max(1, sf.nycoeff - nyder)
    yo2 = max(1, sf.yorder - nyder)
    c1 = sf.coeff.reshape(sf.nycoeff, sf.nxcoeff)
    c2 = np.zeros((ny2, nx2), f64)
    # amovd(Memd[ptr1+nxder], COEFF(ptr2), NXCOEFF2) for rows NYCOEFF1..nyder+1
    for i1 in range(sf.nycoeff - 1, nyder - 1, -1):
        i2 = i1 - nyder
        c2[i2, :] = c1[i1, nxder:nxder + nx2]
    xb = _basis(sf.stype, x, xo2, nxder, sf.xmaxmin, sf.xrange)
    yb = _basis(sf.stype, y, yo2, nyder, sf.ymaxmin, sf.yrange)
    z = _accumulate(c2.ravel(), xb, yb, xo2, yo2)
    d1 = 2.0 / (sf.xmax - sf.xmin)
    d2 = 2.0 / (sf.ymax - sf.ymin)
    from decanter.utils.fpemu import pow_di
    norm = pow_di(d1, nxder) * pow_di(d2, nyder)
    return z * norm


def gs_eval(sf: GsSurface, x: float, y: float) -> float:
    """``dgseval``: single-point evaluator (``gs_b1cheb`` / ``gs_b1leg``)."""
    x = float(x)
    y = float(y)

    def b1(v, order, k1, k2):
        b = [1.0]
        if order == 1:
            return b
        xn = (v + k1) * k2
        b.append(xn)
        for i in range(3, order + 1):
            if sf.stype == GS_CHEBYSHEV:
                # fnmsub: round(2*xn*b[i-1] - b[i-2]) single rounding
                b.append(float(fma64(xn * 2.0, b[-1], -b[-2])))
            else:
                ri = float(i)
                t = (2.0 * ri - 3.0) * xn
                u = -((ri - 2.0) * b[-2])
                b.append(float(fma64(t, b[-1], u)) / (ri - 1.0))
        return b

    xb = b1(x, sf.nxcoeff, sf.xmaxmin, sf.xrange)
    yb = b1(y, sf.nycoeff, sf.ymaxmin, sf.yrange)
    total = 0.0
    ii = 0
    # The compiled k loop (gsevald.o) is unrolled by 8: the first
    # 8*floor(xorder/8) products are rounded and added in order, the
    # remaining ones are fused (fmadd).
    m = 8 * (sf.xorder // 8)
    for i in range(sf.yorder):
        accum = 0.0
        for k in range(sf.xorder):
            if k < m:
                accum = accum + float(sf.coeff[ii]) * xb[k]
            else:
                accum = float(fma64(sf.coeff[ii], xb[k], accum))
            ii += 1
        accum = accum * yb[i]
        total = total + accum
    return total


def xgseval(sf: GsSurface, x, y) -> np.float32:
    return f32(gs_eval(sf, float(f32(x)), float(f32(y))))


def xgsder(sf: GsSurface, x, y, nxd: int, nyd: int) -> NDArray[np.float32]:
    xd = np.asarray(x, f32).astype(f64)
    yd = np.asarray(y, f32).astype(f64)
    return gs_der(sf, xd, yd, nxd, nyd).astype(f32)
