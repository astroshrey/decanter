"""Fast dispcor pixel map: the MWCS multispec inverse without FMA emulation.

Warp mode's s12 (:mod:`decanter.wavelength.iraf_msp`) maps every dispcor
output-pixel edge back to an input pixel with IRAF's stateful Newton inverse
(``sys/mwcs/wfmspec.x:wf_msp_evali``: start point and derivative carried
from edge to edge, at most 10 steps, then a grid search, result quantised to
1e-4 pixel), evaluating the dispersion polynomial with emulated fused
multiply-adds (~20 NumPy operations per FMA). It depends only on the
calibration and the spectrum length and is cached, so it is a one-off cost of
a few seconds per calibration set.

Fast mode runs the same state machine (vectorised over orders) with plain
float64 multiply-adds. The two polynomial evaluations differ by about an ULP
of the wavelength, ~1e-13 pixel in the root, against the 1e-4 pixel
quantisation, so the edges agree unless a root falls within ~1e-13 pixel of a
quantisation boundary or a Newton step within ~1e-13 of the 1e-4 stopping
threshold. The output grid ``(w1, dw, nw)`` is warp mode's, computed exactly.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from decanter.utils.fpemu import fma64, iraf_ctod
from decanter.utils.iraf_fmt import dtoc3
from decanter.wavelength.iraf_msp import (
    _DX, _F32, MultispecFunctions, dispcor_grid, read_id_tokens,
)


class _PlainMultispec:
    """``wf_msp_eval`` / ``wf_msp_evali`` of :class:`MultispecFunctions`, no FMA."""

    def __init__(self, fn: MultispecFunctions) -> None:
        self.fn = fn
        self.K = fn.c.shape[0]
        self.rows = np.arange(self.K)
        # Every order of a WINERED id set shares one function type and order;
        # then the per-row np.where selection can be skipped (same values).
        self.uniform = (len(set(fn.ftype.tolist())) == 1
                        and len(set(fn.order.tolist())) == 1)
        self.cols = [np.ascontiguousarray(fn.c[:, i]) for i in range(fn.c.shape[1])]

    def eval_rows(self, rows: NDArray, xin: NDArray) -> NDArray:
        fn = self.fn
        xval = np.maximum(fn.xmin[rows], np.minimum(fn.xmax[rows], xin))
        x = (xval - fn.mid[rows]) * fn.scale[rows]
        c = fn.c[rows]
        order = fn.order[rows]
        leg = fn.ftype[rows] == 2
        y = np.zeros_like(x) + c[:, 0]
        y = np.where(order > 1, c[:, 1] * x + y, y)
        a = np.ones_like(x)
        b = x.copy()
        for i in range(3, int(order.max()) + 1):
            cheb = (x * 2) * b - a
            legv = (((2 * i - 3) * x) * b - ((i - 2) * a)) / (i - 1)
            cc = np.where(leg, legv, cheb)
            y = np.where(order >= i, c[:, i - 1] * cc + y, y)
            a = b
            b = cc
        return y

    def eval(self, xin: NDArray) -> NDArray:
        if not self.uniform:
            return self.eval_rows(self.rows, np.asarray(xin, float))
        fn = self.fn
        x = (np.maximum(fn.xmin, np.minimum(fn.xmax, xin)) - fn.mid) * fn.scale
        c = self.cols
        order = int(fn.order[0])
        y = c[0] + 0.0
        if order > 1:
            y = c[1] * x + y
        a = 1.0
        b = x
        if fn.ftype[0] == 2:
            for i in range(3, order + 1):
                cc = (((2 * i - 3) * x) * b - ((i - 2) * a)) / (i - 1)
                y = c[i - 1] * cc + y
                a, b = b, cc
        else:
            x2 = x * 2
            for i in range(3, order + 1):
                cc = x2 * b - a
                y = c[i - 1] * cc + y
                a, b = b, cc
        return y

    def _eval1(self, k: int, x: float) -> float:
        return float(self.eval_rows(np.array([k]), np.array([x]))[0])

    def _fallback(self, k: int, y: float, dydx: float) -> tuple[float, float]:
        fn = self.fn
        c2 = float(fn.xmin[k])
        c3 = float(fn.xmax[k])
        xval = (c2 + c3) / 2.0
        yval = abs(self._eval1(k, xval) - y)
        dx = (c3 - c2) / 18.0
        while dx > _DX:
            x = max(c2, -dx * 9.0 + xval)
            while x <= min(c3, dx * 9.0 + xval):
                dy = abs(self._eval1(k, x) - y)
                if dy < yval:
                    xval = x
                    yval = dy
                x = x + dx
            dx = dx / 10.0
        x = xval
        if x + 1 < c3:
            dy = self._eval1(k, x + 1.0) - self._eval1(k, x)
        else:
            dy = self._eval1(k, x) - self._eval1(k, x - 1.0)
        if dy != 0:
            dydx = dy
        return x, dydx

    def evali(self, y: NDArray, x: NDArray, dydx: NDArray):
        """One ``wf_msp_evali`` per order: same state machine as ``iraf_msp``."""
        fn = self.fn
        yval = np.maximum(fn.ymin, np.minimum(fn.ymax, y))
        x = x.copy()
        dydx = dydx.copy()
        dx = np.zeros_like(x)
        dy = np.zeros_like(x)
        done = np.zeros(x.shape, bool)
        for _ in range(10):
            y1 = self.eval(x)
            up = dx > 1.0
            dn = dx < -1.0
            if np.any(up | dn):
                ep = self.eval(x + 1.0)
                em = self.eval(x - 1.0)
                dy = np.where(up, np.where(x + 1 < fn.xmax, ep - y1, y1 - em),
                              np.where(dn, np.where(x - 1 > fn.xmin, y1 - em, ep - y1), dy))
            act = ~done
            dydx = np.where(act & (dy != 0), dy, dydx)
            step = (yval - y1) / dydx
            xn = np.maximum(fn.xmin, np.minimum(fn.xmax, x + step))
            x = np.where(act, xn, x)
            dx = np.where(act, step, dx)
            done |= act & (np.abs(step) < _DX)
            if done.all():
                break
        for k in np.nonzero(~done)[0]:
            x[k], dydx[k] = self._fallback(int(k), float(y[k]), float(dydx[k]))
        whole = np.trunc(x)
        d = (x - whole) / _DX
        n = np.where(d >= 0, np.floor(d + 0.5), -np.floor(0.5 - d))
        return whole + (n.astype(_F32) * _F32(1e-4)).astype(np.float64), dydx


def dispcor_edges_fast(id_paths: tuple[str, ...], npts: int):
    """Faster equivalent of :func:`decanter.wavelength.iraf_msp.dispcor_edges`.

    Returns ``(w1, dw, nw, edges)`` with the same meaning; ``(w1, dw, nw)``
    are computed exactly as warp mode does, ``edges`` by the same stateful
    inverse without FMA emulation (see the module docstring). Cached.
    """
    return _edges_cached(tuple(str(p) for p in id_paths), int(npts))


@lru_cache(maxsize=32)
def _edges_cached(id_paths: tuple[str, ...], npts: int):
    fn = MultispecFunctions([read_id_tokens(p) for p in id_paths], npts)
    w1, dw, nw = dispcor_grid(fn, npts)
    pm = _PlainMultispec(fn)
    # Output-WCS linear terms as dc_multispec stores them ("%.14g" text read
    # back with ctod), exactly as iraf_msp.dispcor_edges does.
    w1t = np.array([iraf_ctod(dtoc3(float(v), 14, "g")) for v in w1])
    dwt = np.array([iraf_ctod(dtoc3(float(v), 14, "g")) for v in dw])
    xs = np.arange(int(nw.max()) + 1) + 0.5
    x = fn.xmax - 0.5                                   # wf_msp_init(INVERSE)
    dydx = pm.eval(x) - pm.eval(x - 1)
    edges = np.empty((len(id_paths), xs.size))
    for i, xo in enumerate(xs):
        w = fma64(dwt, xo - 1.0, w1t)                   # as warp mode (exact)
        x, dydx = pm.evali(w, x, dydx)
        edges[:, i] = x
    return w1, dw, nw, edges
