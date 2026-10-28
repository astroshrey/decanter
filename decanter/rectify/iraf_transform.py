"""Bit-exact port of IRAF ``longslit.transform`` as WARP runs it.

WARP (``warp/cutransform.py:cutransform``) cuts each echelle order out of the
frame with ``imcopy`` and calls::

    iraf.transform(cut, trans, fitnames="<aptrans>_<m>", flux=<CUTRANSFORM flux>,
                   dy=dy, interptype="spline3")

leaving every other parameter at its default (``x1..ny = INDEF``,
``blank = INDEF``, no masks).  With ``flux = yes`` each output pixel is
multiplied by the Jacobian factor (``TransformPlan.flux_factor``: the
bilinear ``J(U,V)`` of ``tr_setup``'s ``zgrid`` times ``du`` times ``dv``,
``t_transform.x:577-581``).  IRAF source, all under
``noao/twodspec/longslit/transform/``:

* ``t_transform.x:t_transform`` -- task driver; reads the ``fc`` surface
  (``fcdbio.x:lm_dbread`` -> ``igsfit/xgs.x:xgsrestore``, double gsurfit).
* ``trsetup.x:tr_setup`` -- output grid (``t_transform.x:tr_setoutput``),
  then the *inverse* mapping ``X(U,V), Y(U,V)`` on a coarse grid
  (``step = 10``, line 182) by the stateful Newton iteration ``tr_invert``
  walked in a boustrophedon, fitted with ``msifit(II_BILINEAR)``.
* ``t_transform.x:tr_transform`` -- per output line: bilinear inverse
  (``msivector``), out-of-range clamping (edge extension), a sliding buffer
  of input lines (``NBUF = 16``, ``NEDGE = 2``) refitted with
  ``msifit(II_BISPLINE3)`` whenever the needed rows leave it, then
  ``msivector`` of the bicubic spline.

Arithmetic follows this Mac's ``x_longslit.e`` object code (iraf-community
2.18.1, clang -O2 on arm64): every fused multiply-add the compiler emitted is
reproduced with :func:`decanter.utils.fpemu.fma32` semantics, REAL stays
float32, the surfaces are double (:mod:`decanter.utils.gsurfit_exact`).

Everything up to the bicubic-spline evaluation depends only on the ``fc``
surface and ``dy`` -- not on the image -- so it is computed once per
(surface, dy) and cached (:func:`get_plan`).
"""
from __future__ import annotations

import math
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from decanter.utils.fpemu import iraf_ctod
from decanter.utils.gsurfit_exact import (
    GS_CHEBYSHEV, GsSurface, xgseval,
)
from decanter.utils.spline3 import (
    bilinear_fit_exact, bilinear_eval_exact, bispline3_eval_exact,
    spline2d_rows_exact, spline2d_cols_exact,
)

f32 = np.float32
f64 = np.float64

_STEP = 10            # trsetup.x:182
_MAX_ITERATE = 10     # trsetup.x: MAX_ITERATE
_ERROR = float(f32(0.05))
_NBUF = 16            # t_transform.x: NBUF
_NEDGE = 2            # t_transform.x: NEDGE

_FS = struct.Struct("<f")
_DS = struct.Struct("<d")
_QS = struct.Struct("<q")
_fpack, _funpack = _FS.pack, _FS.unpack
_dpack, _qunpack = _DS.pack, _QS.unpack


def _r32(x: float) -> float:
    """Round a Python float to float32 (round-to-nearest-even)."""
    return _funpack(_fpack(x))[0]


def _fma32(a: float, b: float, c: float) -> float:
    """Scalar float32 fused multiply-add on float32-valued Python floats."""
    s = a * b + c                      # a*b is exact in double
    r = _funpack(_fpack(s))[0]
    if r != s and (_qunpack(_dpack(s))[0] & 0x1FFFFFFF) == 0x10000000:
        p = a * b
        bb = s - p
        e = (p - (s - bb)) + (c - bb)
        if e > 0:
            r = _funpack(_fpack(math.nextafter(s, math.inf)))[0]
        elif e < 0:
            r = _funpack(_fpack(math.nextafter(s, -math.inf)))[0]
    return r


_SPLIT = 134217729.0


def _fma64(a: float, b: float, c: float) -> float:
    """Scalar float64 fused multiply-add (Dekker/Knuth error-free transforms)."""
    p = a * b
    t = _SPLIT * a
    ah = t - (t - a)
    al = a - ah
    t = _SPLIT * b
    bh = t - (t - b)
    bl = b - bh
    ep = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    s = p + c
    bb = s - p
    es = (p - (s - bb)) + (c - bb)
    return s + (ep + es)


# ---------------------------------------------------------------------------
# Scalar xgsder for the (value, d/dx, d/dy) triple of one surface
# ---------------------------------------------------------------------------

class _ScalarDer:
    """``xgsder(sf, x, y, .., 0/1, 0/1)`` triple for full-cross-term surfaces.

    Mirrors :func:`decanter.utils.gsurfit_exact.gs_der` operation for
    operation (verified against it and the IRAF oracle), specialised to
    Python floats for the serial Newton iteration of ``tr_invert``.
    """

    def __init__(self, sf: GsSurface):
        self.sf = sf
        self.cheb = sf.stype == GS_CHEBYSHEV
        self.xo, self.yo = sf.xorder, sf.yorder
        c = sf.coeff.reshape(sf.yorder, sf.xorder)
        self.c = [[float(v) for v in row] for row in c]
        self.k1x, self.k2x = sf.xmaxmin, sf.xrange
        self.k1y, self.k2y = sf.ymaxmin, sf.yrange
        self.nx = 2.0 / (sf.xmax - sf.xmin)
        self.ny = 2.0 / (sf.ymax - sf.ymin)
        # Legendre recurrence constants (dgs_bleg / dgs_dleg)
        self.leg = []
        for k in range(3, max(self.xo, self.yo) + 2):
            ri = float(k)
            self.leg.append(((2.0 * ri - 3.0) / (ri - 1.0), -(ri - 2.0) / (ri - 1.0)))

    def _basis(self, xn: float, n: int) -> list[float]:
        b = [1.0]
        if n >= 2:
            b.append(xn)
        if self.cheb:
            for _k in range(3, n + 1):
                b.append((xn * b[-1]) * 2.0 - b[-2])
        else:
            for k in range(3, n + 1):
                r1, r2 = self.leg[k - 3]
                b.append(_fma64(xn * b[-1], r1, b[-2] * r2))
        return b[:n]

    def _dbasis(self, xn: float, fn: list[float], order: int) -> list[float]:
        """First-derivative basis (nder = 1) from the order+1 basis ``fn``."""
        n = order + 1
        d = [0.0, 1.0]
        if self.cheb:
            for k in range(3, n + 1):
                t = (xn * d[-1]) * 2.0 - d[-2]
                d.append(_fma64(fn[k - 2], 2.0, t))
        else:
            for k in range(3, n + 1):
                r1, r2 = self.leg[k - 3]
                t = _fma64(xn * d[-1], r1, d[-2] * r2)
                d.append(_fma64(fn[k - 2], r1, t))
        return d[1:1 + order]

    def __call__(self, x: float, y: float) -> tuple[float, float, float]:
        xo, yo, c = self.xo, self.yo, self.c
        xn = (x + self.k1x) * self.k2x
        yn = (y + self.k1y) * self.k2y
        xb = self._basis(xn, xo)
        yb = self._basis(yn, yo)
        # value (dgs_evcheb): accum_i = awsud chain, zfit = fma(accum, yb, zfit)
        acc = []
        for i in range(yo):
            a = 0.0
            row = c[i]
            for k in range(xo):
                a = a + xb[k] * row[k]
            acc.append(a)
        z = 0.0
        for i in range(yo):
            z = _fma64(acc[i], yb[i], z)
        # d/dx: derivative surface rows c[i][1:], x basis = d/dx of order xo
        if xo >= 2:
            dxb = self._dbasis(xn, xb, xo - 1) if xo > 2 else [1.0]
            zx = 0.0
            for i in range(yo):
                a = 0.0
                row = c[i]
                for k in range(xo - 1):
                    a = a + dxb[k] * row[k + 1]
                zx = _fma64(a, yb[i], zx)
            zx = zx * self.nx
        else:
            zx = 0.0
        # d/dy: rows 1.., y basis = d/dy of order yo-1
        if yo >= 2:
            dyb = self._dbasis(yn, yb, yo - 1)
            zy = 0.0
            for i in range(yo - 1):
                zy = _fma64(acc[i + 1], dyb[i], zy)
            zy = zy * self.ny
        else:
            zy = 0.0
        return _r32(z), _r32(zx), _r32(zy)


# ---------------------------------------------------------------------------
# tr_setup
# ---------------------------------------------------------------------------

@dataclass
class TransformPlan:
    """Image-independent part of ``transform`` for one surface + ``dy``."""

    u1: np.float32
    u2: np.float32
    du: np.float32
    nu: int
    v1: np.float32
    v2: np.float32
    dv: np.float32
    nv: int
    nu1: int
    nv1: int
    du1: np.float32
    dv1: np.float32
    xgrid: NDArray[np.float32]   # (nv1, nu1) X(U,V) on the subsampled grid
    ygrid: NDArray[np.float32]   # (nv1, nu1) Y(U,V)
    xcoeff: NDArray[np.float32]  # msifit(II_BILINEAR) coefficients of xgrid
    ycoeff: NDArray[np.float32]
    uout: NDArray[np.float32]    # (nu,) output x in interpolation coords
    vout: NDArray[np.float32]    # (nv,) output y in interpolation coords
    units: str | None = None
    schedules: dict = field(default_factory=dict, repr=False)
    # J(U,V) = 1 / |dU/dX dV/dY - dU/dY dV/dX| on the subsampled grid and its
    # msifit(II_BILINEAR) coefficients (tr_setup's zgrid / jmsi, used only
    # with flux = yes)
    jgrid: NDArray[np.float32] | None = field(default=None, repr=False)
    jcoeff: NDArray[np.float32] | None = field(default=None, repr=False)

    def flux_factor(self, rows: slice | NDArray) -> NDArray[np.float32]:
        """``tr_transform``'s flux-conservation factor for output lines ``rows``.

        ``msivector(jmsi, xout, y)`` (bilinear J at the output grid), then
        ``amulr(dxout, jbuf)``, ``amulkr(jbuf, dyout[i])`` -- each a REAL
        product (t_transform.x:577-581; ``dxout = du``, ``dyout = dv`` for
        linear output axes).  The output pixel is then multiplied by it.
        """
        if self.jcoeff is None:
            raise ValueError("plan has no Jacobian surface")
        vr = np.atleast_1d(self.vout[rows])
        j = _bilinear_lines(self.jcoeff, self.uout, vr)
        j = (self.du * j).astype(f32)
        return (j * self.dv).astype(f32)


def _tr_setoutput(xmin, xmax, ymin, ymax, umin, umax, vmin, vmax, dv):
    """``t_transform.x:tr_setoutput`` with x1..ny INDEF and dy given (linear)."""
    u1, u2 = umin, umax
    v1, v2 = vmin, vmax
    nu = int(f32(f32(xmax - xmin) + f32(1.0)))      # C float -> integer (trunc)
    nv = int(f32(f32(ymax - ymin) + f32(1.0)))
    du = f32(f32(u2 - u1) / f32(nu - 1))
    r = f32(f32(f32(v2 - v1) / dv) + f32(1.0))
    rd = float(r)
    nv = int(math.floor(rd + 0.5)) if rd >= 0 else -int(math.floor(0.5 - rd))  # i_nint
    return u1, u2, du, nu, v1, v2, dv, nv


@dataclass
class _SetupHead:
    sf: GsSurface
    u1: np.float32
    u2: np.float32
    du: np.float32
    nu: int
    v1: np.float32
    v2: np.float32
    dv: np.float32
    nv: int
    nu1: int
    nv1: int
    du1: np.float32
    dv1: np.float32
    xmin: np.float32
    xmax: np.float32
    ymin: np.float32
    ymax: np.float32


def _setup_head(sf: GsSurface, dv: np.float32) -> _SetupHead:
    """First half of ``trsetup.x:tr_setup``: output grid and subsample grid."""
    dv = f32(dv)
    xmin, xmax = sf.getr("xmin"), sf.getr("xmax")
    ymin, ymax = sf.getr("ymin"), sf.getr("ymax")
    # V range from the four corners (tr_eval -> xgseval), float32 min/max.
    vmin = xgseval(sf, xmin, ymin)
    vmax = vmin
    for (xx, yy) in ((xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)):
        v = xgseval(sf, xx, yy)
        vmin = v if v <= vmin else vmin      # fcsel ..., ls  (min)
        vmax = v if v >= vmax else vmax      # fcsel ..., ge  (max)
    umin, umax = xmin, xmax                  # no U surface: U = X
    u1, u2, du, nu, v1, v2, dv, nv = _tr_setoutput(
        xmin, xmax, ymin, ymax, umin, umax, vmin, vmax, dv)
    nu1 = max(2, nu // _STEP)
    nv1 = max(2, nv // _STEP)
    du1 = f32(f32(u2 - u1) / f32(nu1 - 1))
    dv1 = f32(f32(v2 - v1) / f32(nv1 - 1))   # v2 is still vmax here
    return _SetupHead(sf, f32(u1), f32(u2), f32(du), nu, f32(v1), f32(v2), f32(dv), nv,
                      nu1, nv1, du1, dv1, xmin, xmax, ymin, ymax)


def _setup_tail(h: _SetupHead, xgrid, ygrid, zgrid=None) -> TransformPlan:
    """Second half of ``tr_setup``: bilinear fits and output coordinates.

    ``zgrid`` is the Jacobian grid ``1 / |der[4] der[8] - der[5] der[7]|``
    (trsetup.x:299, 319); with no U surface ``der[4] = 1`` and ``der[5] = 0``, so
    the determinant is ``der[8]`` exactly and ``zgrid = 1 / |dV/dy|`` (REAL),
    with ``der[8]`` as ``tr_invert`` left it (the derivative at the last
    point it evaluated, not at the converged one).
    """
    nu, nv = h.nu, h.nv
    # uout = u1 + i*du (fused), then interpolation coordinates
    # max(1, min(n, (uout - u1)/du1 + 1)) -- note the clamp is to nu/nv.
    iu = np.arange(nu, dtype=f32)
    uo = _fma32_vec(iu, np.full(nu, h.du, f32), np.full(nu, h.u1, f32))
    iv = np.arange(nv, dtype=f32)
    vo = _fma32_vec(iv, np.full(nv, h.dv, f32), np.full(nv, h.v1, f32))
    uout = (uo - h.u1) / h.du1 + f32(1.0)
    uout = np.where(uout >= f32(nu), f32(nu), uout)
    uout = np.where(uout <= f32(1.0), f32(1.0), uout).astype(f32)
    vout = (vo - h.v1) / h.dv1 + f32(1.0)
    vout = np.where(vout >= f32(nv), f32(nv), vout)
    vout = np.where(vout <= f32(1.0), f32(1.0), vout).astype(f32)
    return TransformPlan(
        u1=h.u1, u2=f32(uo[-1]), du=h.du, nu=nu,
        v1=h.v1, v2=f32(vo[-1]), dv=h.dv, nv=nv,
        nu1=h.nu1, nv1=h.nv1, du1=h.du1, dv1=h.dv1,
        xgrid=xgrid, ygrid=ygrid,
        xcoeff=bilinear_fit_exact(xgrid), ycoeff=bilinear_fit_exact(ygrid),
        uout=uout, vout=vout, units=h.sf.units,
        jgrid=zgrid, jcoeff=None if zgrid is None else bilinear_fit_exact(zgrid),
    )


def _jacobian(d8) -> NDArray[np.float32]:
    """``1. / abs(der[4]*der[8] - der[5]*der[7])`` with ``der[4] = 1``,
    ``der[5] = 0`` (no U surface): the products are exact, so the determinant
    is ``der[8]`` whether or not clang fuses them; one REAL division."""
    d8 = np.asarray(d8, f32)
    with np.errstate(divide="ignore"):
        return (f32(1.0) / np.abs(d8)).astype(f32)


def _invert_args(h: _SetupHead):
    return (h.sf, float(h.u1), float(h.v1), float(h.du1), float(h.dv1), h.nu1, h.nv1,
            float(h.xmin), float(h.xmax), float(h.ymin), float(h.ymax))


def tr_setup(sf: GsSurface, dv: np.float32, *, method: str = "auto") -> TransformPlan:
    """``trsetup.x:tr_setup`` for a single V (axis 2) surface, no U surface.

    ``method``: ``"exact"`` (straight scalar port), ``"fast"`` (inlined
    scalar), ``"lanes"`` (numpy, vectorised across surfaces) or ``"auto"``.
    All give identical results.
    """
    return tr_setup_many([sf], dv, method=method)[0]


def tr_setup_many(surfaces, dv, *, method: str = "auto") -> list[TransformPlan]:
    """:func:`tr_setup` for several surfaces at once.

    The Newton walk of each surface is inherently serial, but different
    surfaces (echelle orders) are independent, so the ``"lanes"`` method
    advances all of them together, one numpy lane per surface.
    """
    heads = [_setup_head(sf, dv) for sf in surfaces]
    grids: list = [None] * len(heads)
    if method in ("auto", "lanes"):
        groups: dict = {}
        for i, h in enumerate(heads):
            if _fast_ok(h.sf):
                groups.setdefault(h.sf.yorder, []).append(i)
        for idx in groups.values():
            if method == "auto" and len(idx) < 3:
                continue
            res = _invert_grid_lanes([heads[i] for i in idx])
            for i, g in zip(idx, res):
                grids[i] = g
    for i, h in enumerate(heads):
        if grids[i] is None:
            if method == "exact" or not _fast_ok(h.sf):
                grids[i] = _invert_grid(*_invert_args(h))
            else:
                grids[i] = _invert_grid_fast(*_invert_args(h))
    return [_setup_tail(h, *g) for h, g in zip(heads, grids)]


def _fma32_vec(a, b, c) -> NDArray[np.float32]:
    from decanter.utils.fpemu import fma32
    return fma32(a, b, c)


def _invert_grid(sf, u1, v1, du1, dv1, nu1, nv1, xmin, xmax, ymin, ymax):
    """The boustrophedon ``tr_invert`` walk of ``tr_setup`` (nusf=0, nvsf=1)."""
    der3 = _ScalarDer(sf)
    xg = np.empty((nv1, nu1), dtype=f32)
    yg = np.empty((nv1, nu1), dtype=f32)
    zg = np.empty((nv1, nu1), dtype=f32)
    # tr_init at (xmin, ymin)
    d1, d2 = xmin, ymin
    d3, d4, d5 = d1, 1.0, 0.0
    d6, d7, d8 = der3(d1, d2)
    err = _ERROR
    fudges = [_r32(1.0 + _r32(-0.5 / float(i))) for i in range(1, _MAX_ITERATE + 1)]
    r32, fma32 = _r32, _fma32

    def invert(u, v):
        nonlocal d1, d2, d3, d4, d5, d6, d7, d8
        x = y = 0.0
        for it in range(_MAX_ITERATE):
            du = r32(u - d3)
            dv = r32(v - d6)
            a = fma32(d8, du, -r32(d5 * dv))
            b = fma32(d8, d4, -r32(d5 * d7))
            if b != 0.0:
                dx = r32(a / b)
            else:
                dx = -2.0 if a < 0.0 else 2.0
            a = fma32(-d7, dx, dv)
            b = d8
            if b != 0.0:
                dy = r32(a / b)
            else:
                dy = -2.0 if a < 0.0 else 2.0
            fudge = fudges[it]
            x = fma32(fudge, dx, d1)
            y = fma32(fudge, dy, d2)
            t = xmax if xmax <= x else x
            d1 = xmin if xmin >= t else t
            t = ymax if ymax <= y else y
            d2 = ymin if ymin >= t else t
            if abs(dx) < err and abs(dy) < err:
                break
            d3 = d1
            d6, d7, d8 = der3(d1, d2)
        return x, y

    def jac():
        # trsetup.x:299  1. / abs(der[4]*der[8] - der[5]*der[7])
        det = fma32(d4, d8, -r32(d5 * d7))
        return r32(1.0 / abs(det)) if det != 0.0 else math.inf

    for i in range(1, nv1 + 1, 2):
        v = _fma32(float(i - 1), dv1, v1)
        for j in range(1, nu1 + 1):
            u = _fma32(float(j - 1), du1, u1)
            x, y = invert(u, v)
            xg[i - 1, j - 1] = x
            yg[i - 1, j - 1] = y
            zg[i - 1, j - 1] = jac()
        if i == nv1:
            break
        v = _fma32(float(i), dv1, v1)
        for j in range(nu1, 0, -1):
            u = _fma32(float(j - 1), du1, u1)
            x, y = invert(u, v)
            xg[i, j - 1] = x
            yg[i, j - 1] = y
            zg[i, j - 1] = jac()
    return xg, yg, zg


def _fast_ok(sf: GsSurface) -> bool:
    return sf.stype == GS_CHEBYSHEV and sf.xorder == 2 and sf.yorder >= 2


def _invert_grid_fast(sf, u1, v1, du1, dv1, nu1, nv1, xmin, xmax, ymin, ymax):
    """Same walk as :func:`_invert_grid`, specialised and inlined for speed.

    Chebyshev surface with ``xorder = 2`` (what WARP's fitcoords produce).
    REAL roundings use an ``array('f')`` slot; the three fused double
    ``gs_asumvpd`` chains of ``xgsder`` are evaluated unfused together with
    a rigorous bound on the fused-vs-unfused difference, and recomputed
    with exact fused multiply-adds only when that bound straddles a float32
    rounding boundary (so the float32 results are always those of the fused
    chain).  Verified bit-identical to :func:`_invert_grid` and the IRAF
    oracle.
    """
    from array import array
    fa = array("f", [0.0])
    mv = memoryview(bytearray(8))
    dslot = mv.cast("d")
    qslot = mv.cast("q")
    yo = sf.yorder
    cc = sf.coeff.reshape(yo, 2)
    c0 = [float(v) for v in cc[:, 0]]
    c1 = [float(v) for v in cc[:, 1]]
    k1x, k2x, k1y, k2y = sf.xmaxmin, sf.xrange, sf.ymaxmin, sf.yrange
    nrmx = 2.0 / (sf.xmax - sf.xmin)
    nrmy = 2.0 / (sf.ymax - sf.ymin)
    U = 2.0 ** -53
    kb = 4.0 * (yo + 1) * U
    fma64 = _fma64
    rng_y = range(2, yo)
    rng_yo = range(yo)
    rng_dy = range(1, yo)
    nfall = [0]

    def chain_exact(a, t, n0):
        z = 0.0
        for i in range(n0, len(t)):
            z = fma64(a[i], t[i], z)
        return z

    def der(x, y):
        xn = (x + k1x) * k2x
        yn = (y + k1y) * k2y
        T = [1.0, yn]
        dT = [0.0, 1.0]
        t1, t0 = yn, 1.0
        d1_, d0_ = 1.0, 0.0
        for _k in rng_y:
            tk = (yn * t1) * 2.0 - t0
            dk = 2.0 * t1 + ((yn * d1_) * 2.0 - d0_)
            T.append(tk)
            dT.append(dk)
            t0, t1 = t1, tk
            d0_, d1_ = d1_, dk
        acc = [c0[i] + xn * c1[i] for i in rng_yo]
        # value: zfit = fma(acc_i, T_i, zfit)
        z = 0.0
        sb = 0.0
        for i in rng_yo:
            pr = acc[i] * T[i]
            z = z + pr
            sb += (pr if pr >= 0 else -pr) + (z if z >= 0 else -z)
        b = kb * sb + 4.0 * U * (z if z >= 0 else -z)
        fa[0] = z - b
        lo = fa[0]
        fa[0] = z + b
        if lo != fa[0]:
            nfall[0] += 1
            z = chain_exact(acc, T, 0)
        fa[0] = z
        r6 = fa[0]
        # d/dx: zfit = fma(c1_i, T_i, zfit) * nrmx
        z = 0.0
        sb = 0.0
        for i in rng_yo:
            pr = c1[i] * T[i]
            z = z + pr
            sb += (pr if pr >= 0 else -pr) + (z if z >= 0 else -z)
        v = z * nrmx
        b = kb * sb * nrmx * (1.0 + 4.0 * U) + 6.0 * U * (v if v >= 0 else -v)
        fa[0] = v - b
        lo = fa[0]
        fa[0] = v + b
        if lo != fa[0]:
            nfall[0] += 1
            v = chain_exact(c1, T, 0) * nrmx
        fa[0] = v
        r7 = fa[0]
        # d/dy: zfit = fma(acc_{i}, dT_i, zfit) for i = 1..yo-1, * nrmy
        z = 0.0
        sb = 0.0
        for i in rng_dy:
            pr = acc[i] * dT[i]
            z = z + pr
            sb += (pr if pr >= 0 else -pr) + (z if z >= 0 else -z)
        v = z * nrmy
        b = kb * sb * nrmy * (1.0 + 4.0 * U) + 6.0 * U * (v if v >= 0 else -v)
        fa[0] = v - b
        lo = fa[0]
        fa[0] = v + b
        if lo != fa[0]:
            nfall[0] += 1
            v = chain_exact(acc, dT, 1) * nrmy
        fa[0] = v
        r8 = fa[0]
        return r6, r7, r8

    def fma32_fix(a, b, c, s, r):
        # s = fl64(a*b + c) sits on a float32 midpoint: resolve with the
        # exact error of the double addition.
        p = a * b
        bb = s - p
        e = (p - (s - bb)) + (c - bb)
        if e > 0:
            fa[0] = math.nextafter(s, math.inf)
            return fa[0]
        if e < 0:
            fa[0] = math.nextafter(s, -math.inf)
            return fa[0]
        return r

    MASK, HALF = 0x1FFFFFFF, 0x10000000
    xg = np.empty((nv1, nu1), dtype=f32)
    yg = np.empty((nv1, nu1), dtype=f32)
    zg = np.empty((nv1, nu1), dtype=f32)
    d1, d2 = xmin, ymin
    d3 = d1
    d6, d7, d8 = der(d1, d2)
    err = _ERROR
    fudges = [_r32(1.0 + _r32(-0.5 / float(i))) for i in range(1, _MAX_ITERATE + 1)]
    # d4 = 1, d5 = 0 throughout (no U surface): a = fma(d8, du, -(0*dv))
    # = round(d8*du) and b = fma(d8, 1, -(0*d7)) = d8.
    xrow = [0.0] * nu1
    yrow = [0.0] * nu1
    zrow = [0.0] * nu1   # d8 as tr_invert leaves it (Jacobian 1/|d8|)

    for i in range(1, nv1 + 1):
        if (i - 1) % 2 == 0:
            jr = range(1, nu1 + 1)
        else:
            jr = range(nu1, 0, -1)
        s = float(i - 1) * dv1 + v1
        fa[0] = s
        v = fa[0]
        if v != s:
            dslot[0] = s
            if (qslot[0] & MASK) == HALF:
                v = fma32_fix(float(i - 1), dv1, v1, s, v)
        for j in jr:
            s = float(j - 1) * du1 + u1
            fa[0] = s
            u = fa[0]
            if u != s:
                dslot[0] = s
                if (qslot[0] & MASK) == HALF:
                    u = fma32_fix(float(j - 1), du1, u1, s, u)
            x = y = 0.0
            for it in range(_MAX_ITERATE):
                fa[0] = u - d3
                du = fa[0]
                fa[0] = v - d6
                dv = fa[0]
                fa[0] = d8 * du
                a = fa[0]
                if d8 != 0.0:
                    fa[0] = a / d8
                    dx = fa[0]
                else:
                    dx = -2.0 if a < 0.0 else 2.0
                s = -d7 * dx + dv
                fa[0] = s
                a = fa[0]
                if a != s:
                    dslot[0] = s
                    if (qslot[0] & MASK) == HALF:
                        a = fma32_fix(-d7, dx, dv, s, a)
                if d8 != 0.0:
                    fa[0] = a / d8
                    dy = fa[0]
                else:
                    dy = -2.0 if a < 0.0 else 2.0
                fu = fudges[it]
                s = fu * dx + d1
                fa[0] = s
                x = fa[0]
                if x != s:
                    dslot[0] = s
                    if (qslot[0] & MASK) == HALF:
                        x = fma32_fix(fu, dx, d1, s, x)
                s = fu * dy + d2
                fa[0] = s
                y = fa[0]
                if y != s:
                    dslot[0] = s
                    if (qslot[0] & MASK) == HALF:
                        y = fma32_fix(fu, dy, d2, s, y)
                t = xmax if xmax <= x else x
                d1 = xmin if xmin >= t else t
                t = ymax if ymax <= y else y
                d2 = ymin if ymin >= t else t
                if (dx if dx >= 0 else -dx) < err and (dy if dy >= 0 else -dy) < err:
                    break
                d3 = d1
                d6, d7, d8 = der(d1, d2)
            xrow[j - 1] = x
            yrow[j - 1] = y
            zrow[j - 1] = d8
        xg[i - 1, :] = xrow
        yg[i - 1, :] = yrow
        zg[i - 1, :] = zrow
    _invert_grid_fast.last_fallbacks = nfall[0]
    return xg, yg, _jacobian(zg)


_MASK64 = np.int64(0x1FFFFFFF)
_HALF64 = np.int64(0x10000000)


def _fma32_lanes(a: NDArray, b: NDArray, c: NDArray) -> NDArray[np.float32]:
    """Exact float32 fma of float32 arrays (branch-free tie correction)."""
    a = a.astype(f64)
    b = b.astype(f64)
    c = c.astype(f64)
    pr = a * b
    s = pr + c
    r = s.astype(f32)
    tie = (s.view(np.int64) & _MASK64) == _HALF64
    if np.count_nonzero(tie):
        bb = s - pr
        e = (pr - (s - bb)) + (c - bb)
        fix = tie & (e != 0.0)
        if np.count_nonzero(fix):
            up = np.nextafter(s, np.where(e > 0, np.inf, -np.inf)).astype(f32)
            r = np.where(fix, up, r)
    return r


def _invert_grid_lanes(heads: list[_SetupHead]):
    """:func:`_invert_grid_fast` for many surfaces at once (one lane each).

    Each lane runs its own surface's serial boustrophedon Newton walk; all
    lanes advance one ``tr_invert`` iteration per numpy step.  Requires
    Chebyshev ``xorder = 2`` surfaces of equal ``yorder``.  The fused double
    chains of ``xgsder`` are evaluated unfused with a rigorous per-lane
    error bound (|T_k| <= 1, |T'_k| <= k^2 on the clamped domain); a lane
    whose bound straddles a float32 rounding boundary is recomputed exactly.
    """
    L = len(heads)
    n = heads[0].sf.yorder
    U = 2.0 ** -53
    lanes = np.arange(L)
    # --- per-lane point sequences (boustrophedon), exact float32 u, v ---
    npts = np.array([h.nu1 * h.nv1 for h in heads])
    K = int(npts.max()) + 1                 # +1 padding slot per lane
    Useq = np.zeros((L, K), f32)
    Vseq = np.zeros((L, K), f32)
    rowi = np.zeros((L, K), np.int64)
    colj = np.zeros((L, K), np.int64)
    for l, h in enumerate(heads):
        ii = np.repeat(np.arange(1, h.nv1 + 1), h.nu1)
        jj = np.tile(np.arange(1, h.nu1 + 1), h.nv1).reshape(h.nv1, h.nu1)
        jj[1::2] = jj[1::2, ::-1]
        jj = jj.ravel()
        m = ii.size
        Useq[l, :m] = _fma32_lanes((jj - 1).astype(f32), np.full(m, h.du1, f32),
                                   np.full(m, h.u1, f32))
        Vseq[l, :m] = _fma32_lanes((ii - 1).astype(f32), np.full(m, h.dv1, f32),
                                   np.full(m, h.v1, f32))
        Useq[l, m:] = Useq[l, m - 1]
        Vseq[l, m:] = Vseq[l, m - 1]
        rowi[l, :m] = ii - 1
        colj[l, :m] = jj - 1
    Uf, Vf = Useq.ravel(), Vseq.ravel()
    base = lanes * K
    kcap = npts.astype(np.int64)
    # --- per-lane surface constants ---
    C0 = np.empty((n, L), f64)
    C1 = np.empty((n, L), f64)
    for l, h in enumerate(heads):
        cc = h.sf.coeff.reshape(n, 2)
        C0[:, l] = cc[:, 0]
        C1[:, l] = cc[:, 1]
    k1x = np.array([h.sf.xmaxmin for h in heads])
    k2x = np.array([h.sf.xrange for h in heads])
    k1y = np.array([h.sf.ymaxmin for h in heads])
    k2y = np.array([h.sf.yrange for h in heads])
    NRM = np.vstack([np.ones(L), 2.0 / np.array([h.sf.xmax - h.sf.xmin for h in heads]),
                     2.0 / np.array([h.sf.ymax - h.sf.ymin for h in heads])])
    absc = np.abs(C0) + np.abs(C1)
    w2 = (np.arange(n, dtype=f64) ** 2)[:, None]
    kb = 8.0 * (2 * n + 1) * U
    BC = np.vstack([kb * absc.sum(0), kb * np.abs(C1).sum(0) * NRM[1],
                    kb * (absc[1:] * w2[1:]).sum(0) * NRM[2]]) * (1.0 + 1e-6)
    exact = [_ScalarDer(h.sf) for h in heads]
    xmin = np.array([h.xmin for h in heads], f32)
    xmax = np.array([h.xmax for h in heads], f32)
    ymin = np.array([h.ymin for h in heads], f32)
    ymax = np.array([h.ymax for h in heads], f32)
    # S[k, 0] = T_k, S[k, 1] = T'_k (dgs_bcheb / dgs_dcheb recurrences)
    S = np.zeros((n, 2, L), f64)
    S[0, 0] = 1.0
    S[1, 1] = 1.0
    tmp = np.empty(L, f64)
    P = np.empty((n, 3, L), f64)
    P[0, 2] = 0.0
    z = np.empty((3, L), f64)
    xn = np.empty(L, f64)
    yn = np.empty(L, f64)
    Sk = [S[kk] for kk in range(n)]
    S0 = [S[kk, 0] for kk in range(n)]
    S1 = [S[kk, 1] for kk in range(n)]
    Pr = [P[kk] for kk in range(n)]
    P0, P1, P2 = P[:, 0], P[:, 1], P[1:, 2]
    T_all, dT_all = S[:, 0], S[1:, 1]
    acc = np.empty((n, L), f64)
    Uq = 6.0 * U

    def der(xx, yy):
        np.add(xx, k1x, out=xn)
        np.multiply(xn, k2x, out=xn)
        np.add(yy, k1y, out=yn)
        np.multiply(yn, k2y, out=yn)
        S0[1][...] = yn
        for kk in range(2, n):
            w = Sk[kk]
            np.multiply(Sk[kk - 1], yn, out=w)
            w *= 2.0
            w -= Sk[kk - 2]
            np.multiply(S0[kk - 1], 2.0, out=tmp)
            np.add(tmp, S1[kk], out=S1[kk])
        np.multiply(C1, xn, out=acc)
        np.add(acc, C0, out=acc)
        np.multiply(acc, T_all, out=P0)
        np.multiply(C1, T_all, out=P1)
        np.multiply(acc[1:], dT_all, out=P2)
        z.fill(0.0)
        for kk in range(n):
            np.add(z, Pr[kk], out=z)
        v = z * NRM
        bnd = np.abs(v)
        bnd *= Uq
        bnd += BC
        r = v.astype(f32)
        lo = v - bnd
        bnd += v
        if np.count_nonzero(lo.astype(f32) != bnd.astype(f32)):
            bad = lo.astype(f32) != bnd.astype(f32)
            for l in np.nonzero(bad.any(axis=0))[0]:
                r[:, l] = exact[l](float(xx[l]), float(yy[l]))
        return r

    fudges = np.array([_r32(1.0 + _r32(-0.5 / float(i)))
                       for i in range(1, _MAX_ITERATE + 1)], f32)
    err = f32(_ERROR)
    # tr_init at (xmin, ymin)
    d1 = xmin.copy()
    d2 = ymin.copy()
    d3 = d1.copy()
    r = der(d1.astype(f64), d2.astype(f64))
    d6, d7, d8 = r[0].copy(), r[1].copy(), r[2].copy()
    it = np.zeros(L, np.int64)
    k = np.zeros(L, np.int64)
    Xf = np.zeros(L * K, f32)
    Yf = np.zeros(L * K, f32)
    Zf = np.ones(L * K, f32)      # der[8] when each point's tr_invert returns
    u = Uf[base]
    v = Vf[base]
    last = _MAX_ITERATE - 1
    step = 0
    lo2 = np.vstack([xmin, ymin])
    hi2 = np.vstack([xmax, ymax])
    D = np.empty((2, L), f32)
    Dd = np.empty((2, L), f32)
    while True:
        du = u - d3
        dv = v - d6
        dx = (d8 * du) / d8
        a = _fma32_lanes(-d7, dx, dv)
        dy = a / d8
        fu = fudges[it]
        D[0] = dx
        D[1] = dy
        Dd[0] = d1
        Dd[1] = d2
        xy = _fma32_lanes(fu, D, Dd)
        dd = np.maximum(lo2, np.minimum(hi2, xy))
        d1 = dd[0]
        d2 = dd[1]
        np.abs(D, out=D)
        conv = (D < err).all(axis=0)
        idx = base + k
        Xf[idx] = xy[0]
        Yf[idx] = xy[1]
        dd64 = dd.astype(f64)
        r = der(dd64[0], dd64[1])
        d3 = np.where(conv, d3, d1)
        d6 = np.where(conv, d6, r[0])
        d7 = np.where(conv, d7, r[1])
        d8 = np.where(conv, d8, r[2])
        Zf[idx] = d8
        fin = conv | (it == last)
        k = np.minimum(k + fin, kcap)
        it = np.where(fin, 0, it + 1)
        idx = base + k
        u = Uf[idx]
        v = Vf[idx]
        step += 1
        if (step & 63) == 0 and np.all(k >= kcap):
            break
    if not (np.isfinite(Xf).all() and np.isfinite(Yf).all()):
        raise FloatingPointError("non-finite tr_invert result (dV/dy == 0?)")
    out = []
    X2 = Xf.reshape(L, K)
    Y2 = Yf.reshape(L, K)
    Z2 = Zf.reshape(L, K)
    for l, h in enumerate(heads):
        m = int(npts[l])
        xg = np.empty((h.nv1, h.nu1), f32)
        yg = np.empty((h.nv1, h.nu1), f32)
        zg = np.empty((h.nv1, h.nu1), f32)
        xg[rowi[l, :m], colj[l, :m]] = X2[l, :m]
        yg[rowi[l, :m], colj[l, :m]] = Y2[l, :m]
        zg[rowi[l, :m], colj[l, :m]] = Z2[l, :m]
        out.append((xg, yg, _jacobian(zg)))
    _invert_grid_lanes.last_steps = step
    return out


# ---------------------------------------------------------------------------
# tr_transform (per image)
# ---------------------------------------------------------------------------

@dataclass
class LineSchedule:
    """Data-independent bookkeeping of ``tr_transform``'s input-line buffer.

    For output line ``i`` (0-based) the interpolant is the bicubic spline
    fitted to input lines ``line3[w]..line4[w]`` (1-based, inclusive) of
    window ``w = window[i]``.
    """

    nyin: int
    window: NDArray[np.int64]     # (nv,) window index per output line
    line3: NDArray[np.int64]      # (nwin,)
    line4: NDArray[np.int64]      # (nwin,)


def _bilinear_lines(coeff, uout, vrows):
    from decanter.utils.spline3 import bilinear_eval_exact
    x = np.broadcast_to(uout[None, :], (vrows.size, uout.size))
    y = np.broadcast_to(vrows[:, None], (vrows.size, uout.size))
    return bilinear_eval_exact(coeff, x, y)


def line_schedule(plan: TransformPlan, nyin: int, chunk: int = 512) -> LineSchedule:
    """Replay ``t_transform.x:tr_transform``'s buffer logic (lines 330-372).

    Uses only the Y(U,V) inverse, so it is shared by every image of the
    same size transformed with the same plan.
    """
    nv = plan.nv
    amin = np.empty(nv, f32)
    bmax = np.empty(nv, f32)
    for i0 in range(0, nv, chunk):
        yin = _bilinear_lines(plan.ycoeff, plan.uout, plan.vout[i0:i0 + chunk])
        amin[i0:i0 + chunk] = yin.min(axis=1)
        bmax[i0:i0 + chunk] = yin.max(axis=1)
    one = 1.0
    fnyin = float(nyin)
    window = np.empty(nv, np.int64)
    l3s: list[int] = []
    l4s: list[int] = []
    nlines = 0
    have = False
    line1 = line2 = 0
    for i in range(nv):
        a = float(amin[i])
        b = float(bmax[i])
        if a < one or b > fnyin:
            if a < one:
                a = one
                b = max(a, b)
            if b > fnyin:
                b = fnyin
                a = min(a, b)
        if (not have) or b > float(line2) or a < float(line1):
            nlines = max(nlines, int(_r32(_r32(b - a) + 18.0)))   # b - a + 2 + NBUF
            if not have:
                if a < float(nyin // 2):
                    line1 = max(1, int(a))
                    line2 = min(nyin, line1 + nlines - 1)
                else:
                    line2 = min(nyin, int(_r32(b + 1.0)))
                    line1 = max(1, line2 - nlines + 1)
            elif b > float(line2):
                line1 = max(1, int(a))
                line2 = min(nyin, line1 + nlines - 1)
                line1 = max(1, line2 - nlines + 1)
            else:
                line2 = min(nyin, int(_r32(b + 1.0)))
                line1 = max(1, line2 - nlines + 1)
                line2 = min(nyin, line1 + nlines - 1)
            have = True
            l3s.append(max(1, line1 - _NEDGE))
            l4s.append(min(nyin, line2 + _NEDGE))
        window[i] = len(l3s) - 1
    return LineSchedule(nyin=nyin, window=window, line3=np.array(l3s, np.int64),
                        line4=np.array(l4s, np.int64))


def tr_transform(cut: NDArray, plan: TransformPlan, sched: LineSchedule | None = None,
                 *, chunk: int = 256, flux: bool = False) -> NDArray[np.float32]:
    """``t_transform.x:tr_transform`` (interptype=spline3, blank=INDEF).

    ``cut`` is the ``imcopy``-cut order strip (REAL), shape ``(nyin, nxin)``.
    Returns the ``(nv, nu)`` transformed strip.  ``flux=True`` is WARP's
    ``CUTRANSFORM flux: yes`` (``transform flux=yes``): every output pixel is
    multiplied by :meth:`TransformPlan.flux_factor` (t_transform.x:577-581).
    """
    from decanter.utils.spline3 import (
        bispline3_eval_exact, spline2d_cols_exact, spline2d_rows_exact,
    )
    data = np.ascontiguousarray(cut, dtype=f32)
    nyin, nxin = data.shape
    if sched is None or sched.nyin != nyin:
        sched = line_schedule(plan, nyin)
    # msifit(II_BISPLINE3) of each buffer window: the x pass is per image
    # line, so do it once for the whole image; the y pass per window,
    # batched over windows of equal height.
    xc = spline2d_rows_exact(data)                       # (nyin, nxin + 2)
    nwin = sched.line3.size
    ncol = nxin + 3
    heights = sched.line4 - sched.line3 + 1
    sizes = (heights + 3) * ncol
    offs = np.zeros(nwin, np.int64)
    offs[1:] = np.cumsum(sizes)[:-1]
    flat = np.empty(int(sizes.sum()), f32)
    for h in np.unique(heights):
        ws = np.nonzero(heights == h)[0]
        stack = np.stack([xc[sched.line3[w] - 1: sched.line4[w]] for w in ws])
        coef = spline2d_cols_exact(stack)                # (len(ws), h + 3, ncol)
        for k, w in enumerate(ws):
            flat[offs[w]: offs[w] + sizes[w]] = coef[k].ravel()
    out = np.empty((plan.nv, plan.nu), f32)
    fnx, fny = f32(nxin), f32(nyin)
    one = f32(1.0)
    for i0 in range(0, plan.nv, chunk):
        rows = slice(i0, min(plan.nv, i0 + chunk))
        vr = plan.vout[rows]
        xin = _bilinear_lines(plan.xcoeff, plan.uout, vr)
        yin = _bilinear_lines(plan.ycoeff, plan.uout, vr)
        xin = np.where(xin < one, one, xin)              # arltr / argtr
        xin = np.where(xin > fnx, fnx, xin)
        yin = np.where(yin < one, one, yin)
        yin = np.where(yin > fny, fny, yin)
        w = sched.window[rows]
        yin = yin - (sched.line3[w] - 1).astype(f32)[:, None]   # asubkr
        base = np.broadcast_to(offs[w][:, None], yin.shape)
        out[rows] = bispline3_eval_exact(flat, base, ncol, xin, yin)
        if flux:
            out[rows] = (out[rows] * plan.flux_factor(rows)).astype(f32)
    return out


def _window_layout(sched: LineSchedule, ncol: int):
    heights = sched.line4 - sched.line3 + 1
    sizes = (heights + 3) * ncol
    offs = np.zeros(heights.size, np.int64)
    offs[1:] = np.cumsum(sizes)[:-1]
    return heights, sizes, offs


def _image_coeffs(cut: NDArray[np.float32], sched: LineSchedule) -> NDArray[np.float32]:
    """All buffer windows' ``msifit(II_BISPLINE3)`` coefficients, flattened.

    The x pass of ``msifit`` (``ii_spline2d`` over each buffer line) only
    depends on that image line, so it is done once for the whole image;
    the y pass is batched over windows of equal height.  Layout as in
    :func:`tr_transform` (window ``w`` at ``offs[w]``, row length
    ``nxin + 3``).
    """
    from decanter.utils.spline3 import spline2d_axis0

    nyin, nxin = cut.shape
    ncol = nxin + 3
    v = np.zeros((nxin + 2, nyin), f32)
    v[1:nxin + 1] = cut.T
    xc = spline2d_axis0(v).T                       # (nyin, nxin + 2)
    heights, sizes, offs = _window_layout(sched, ncol)
    flat = np.zeros(int(sizes.sum()), f32)
    for h in np.unique(heights):
        ws = np.nonzero(heights == h)[0]
        V = np.zeros((h + 2, ws.size, ncol), f32)
        for k, w in enumerate(ws):
            V[1:h + 1, k, :nxin + 2] = xc[sched.line3[w] - 1: sched.line4[w]]
        spline2d_axis0(V.reshape(h + 2, -1))
        for k, w in enumerate(ws):
            flat[offs[w]: offs[w] + (h + 2) * ncol] = V[:, k, :].ravel()
    return flat


def tr_transform_many(cuts, plan: TransformPlan, sched: LineSchedule | None = None,
                      *, chunk_pixels: int = 1 << 16,
                      flux: bool = False) -> list[NDArray[np.float32]]:
    """:func:`tr_transform` of several same-shape cuts with one plan.

    Same arithmetic as :func:`tr_transform` (the reference), restructured
    for speed: the inverse coordinates and B-spline weights are computed
    once for all cuts, buffer windows are fitted in batches, and the
    evaluation is chunked to stay in cache.  ``flux`` as in
    :func:`tr_transform`.
    """
    from decanter.utils.spline3 import bispline3_eval_exact, bispline3_eval_many

    cuts = [np.ascontiguousarray(c, dtype=f32) for c in cuts]
    nyin, nxin = cuts[0].shape
    if sched is None or sched.nyin != nyin:
        sched = schedule_for(plan, nyin)
    ncol = nxin + 3
    _, _, offs = _window_layout(sched, ncol)
    flats32 = [_image_coeffs(c, sched) for c in cuts]
    flats64 = [f.astype(f64) for f in flats32]
    outs = [np.empty((plan.nv, plan.nu), f32) for _ in cuts]
    fnx, fny = f32(nxin), f32(nyin)
    one = f32(1.0)
    redo: list = [[] for _ in cuts]
    nl = max(1, chunk_pixels // plan.nu)
    for i0 in range(0, plan.nv, nl):
        rows = slice(i0, min(plan.nv, i0 + nl))
        vr = plan.vout[rows]
        xin = _bilinear_lines(plan.xcoeff, plan.uout, vr)
        yin = _bilinear_lines(plan.ycoeff, plan.uout, vr)
        np.clip(xin, one, fnx, out=xin)              # arltr / argtr
        np.clip(yin, one, fny, out=yin)
        w = sched.window[rows]
        yin -= (sched.line3[w] - 1).astype(f32)[:, None]   # asubkr
        base = np.broadcast_to(offs[w][:, None], yin.shape)
        ties = bispline3_eval_many(flats64, base, ncol, xin, yin, [o[rows] for o in outs])
        for k, t in enumerate(ties):
            if t.size:
                redo[k].append((i0 * plan.nu + t, base.ravel()[t], xin.ravel()[t],
                                yin.ravel()[t]))
    # Pixels whose unchecked fast path hit a float32 midpoint: exact redo.
    for k, items in enumerate(redo):
        if items:
            flat_idx = np.concatenate([a[0] for a in items])
            vals = bispline3_eval_exact(flats32[k], np.concatenate([a[1] for a in items]),
                                        ncol, np.concatenate([a[2] for a in items]),
                                        np.concatenate([a[3] for a in items]))
            outs[k].reshape(-1)[flat_idx] = vals
    if flux:
        fac = plan.flux_factor(slice(0, plan.nv))
        outs = [(o * fac).astype(f32) for o in outs]
    return outs


# ---------------------------------------------------------------------------
# Plan cache
# ---------------------------------------------------------------------------

_PLAN_CACHE: dict = {}
_PLAN_LOCK = threading.Lock()


def _dv_of(dy) -> np.float32:
    """``dy`` as IRAF sees it: pyraf passes ``str(dy)``; ``clgetr`` -> ctod -> REAL."""
    return f32(iraf_ctod(str(float(dy))))


def plans_for_surfaces(surfaces, dy) -> list[TransformPlan]:
    """Cached :func:`tr_setup` for several surfaces (computed together)."""
    dv = _dv_of(dy)
    keys = [(sf.raw, float(dv)) for sf in surfaces]
    with _PLAN_LOCK:
        have = {k: _PLAN_CACHE[k] for k in keys if k in _PLAN_CACHE}
    todo = []
    for k, sf in zip(keys, surfaces):
        if k not in have and all(k != t[0] for t in todo):
            todo.append((k, sf))
    if todo:
        new = tr_setup_many([sf for _, sf in todo], dv)
        with _PLAN_LOCK:
            for (k, _), pl in zip(todo, new):
                _PLAN_CACHE[k] = pl
                have[k] = pl
    return [have[k] for k in keys]


def plan_for_surface(sf: GsSurface, dy) -> TransformPlan:
    return plans_for_surfaces([sf], dy)[0]


def get_plans(fc_paths, dy) -> list[TransformPlan]:
    """Cached plans for ``fc`` database files (``lm_dbread`` + ``tr_setup``).

    ``dy`` is what WARP passes to pyraf (comp ``CDELT1``).
    """
    return plans_for_surfaces([GsSurface.from_fc_file(p) for p in fc_paths], dy)


def get_plan(fc_path: Path | str, dy: float) -> TransformPlan:
    return get_plans([fc_path], dy)[0]


def schedule_for(plan: TransformPlan, nyin: int) -> LineSchedule:
    """Cached :func:`line_schedule` of a plan for an input of ``nyin`` lines."""
    sch = plan.schedules.get(nyin)
    if sch is None:
        sch = line_schedule(plan, nyin)
        plan.schedules[nyin] = sch
    return sch


def clear_plan_cache() -> None:
    with _PLAN_LOCK:
        _PLAN_CACHE.clear()
