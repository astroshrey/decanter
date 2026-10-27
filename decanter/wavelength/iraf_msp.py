"""Bit-exact IRAF ``dispcor`` pixel mapping (MWCS ``wfmspec`` + ``dc_defaults``).

``dispcor`` resamples each 1-D spectrum onto a linear wavelength grid. The
output-pixel edges are mapped back into input pixels through the inverse
of the ``identify`` dispersion function, which IRAF evaluates in
``sys/mwcs/wfmspec.x`` (``wf_msp_evali``). That inverse is unusual and has
to be reproduced literally to match WARP bit for bit:

- It is a Newton iteration whose start point and derivative are *state*
  carried from the previous call (dispcor walks the output edges in order),
  and the derivative is only refreshed when a step exceeds one pixel.
- After at most 10 steps it falls back to a coarse-to-fine grid search.
- The result is quantised: ``x = int(x) + REAL(nint(frac / 1e-4) * 1e-4)``.

Everything here runs in double with the fused multiply-adds clang emits
(``fpemu.fma64``), and the database coefficients are parsed with IRAF's
``ctod``. Orders are vectorised: one call maps every order at once, and the
result depends only on the calibration and the spectrum length, so it is
cached and shared by the object and sky paths of every frame.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from decanter.utils.fpemu import fma64, iraf_ctod

_F32 = np.float32
_DX = float(np.float32(1e-4))      # wfmspec DX, a REAL constant in the C


def _msp_coeff(token: str) -> float:
    """A database coefficient as ``wf_msp_coeff`` sees it.

    IRAF equivalent: ``noao/onedspec/dispcor/dcio.x:dc_gms`` (``dtgad`` ->
    ``fprintf(" %.15g")``) then ``sys/mwcs/wfmspec.x:wf_msp_coeff`` (``ctod``).
    """
    from decanter.utils.iraf_fmt import dtoc3
    return iraf_ctod(dtoc3(iraf_ctod(token), 15, "g"))


def read_id_tokens(id_path: Path | str) -> list[str]:
    """Coefficient tokens of the LAST ``begin`` record of an id database."""
    rec = Path(id_path).read_text().split("begin")[-1].splitlines()
    toks: list[str] = []
    count = None
    for line in rec:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "coefficients":
            count = int(parts[1])
            continue
        if count is not None and len(toks) < count:
            toks.extend(parts)
    if count is None:
        raise ValueError(f"{id_path}: no coefficients block")
    return toks


class MultispecFunctions:
    """``wfmspec`` nonlinear (Chebyshev/Legendre) dispersion functions for K orders."""

    def __init__(self, tokens: list[list[str]], npts: int) -> None:
        k = len(tokens)
        self.ftype = np.array([int(round(iraf_ctod(t[0]))) for t in tokens])
        self.order = np.array([int(round(iraf_ctod(t[1]))) for t in tokens])
        bad = set(self.ftype.tolist()) - {1, 2}
        if bad:
            raise NotImplementedError(f"dispersion function types {bad}")
        # dispcor never evaluates the database numbers directly:
        # ``dispcor/dcio.x:dc_gms`` re-encodes them (``" %.15g"``) into the
        # MULTISPEC attribute string that ``wfmspec.x:wf_msp_coeff`` parses
        # back with ``ctod``. The round trip matters: ``ctod("1.000000")`` is
        # 1.0000000000000007 but the re-encoded "1" parses to exactly 1.0,
        # which moves the function's centre/scale (and thus w1/dw) by an ULP.
        fxmin = np.array([_msp_coeff(t[2]) for t in tokens])
        fxmax = np.array([_msp_coeff(t[3]) for t in tokens])
        self.c = np.zeros((k, int(self.order.max())))
        for j, t in enumerate(tokens):
            self.c[j, : self.order[j]] = [_msp_coeff(v) for v in t[4: 4 + self.order[j]]]
        # wf_msp_coeff: range -> (centre, 2 / width)
        self.mid = (fxmax + fxmin) / 2
        self.scale = 2 / (fxmax - fxmin)
        self.xmin = np.full(k, 0.5)
        self.xmax = np.full(k, npts + 0.5)
        a = self.eval(self.xmin)
        b = self.eval(self.xmax)
        self.ymin = np.minimum(a, b)
        self.ymax = np.maximum(a, b)

    # -- wf_msp_eval ----------------------------------------------------
    def _eval_rows(self, rows: NDArray, xin: NDArray) -> NDArray[np.float64]:
        xval = np.maximum(self.xmin[rows], np.minimum(self.xmax[rows], xin))
        x = (xval - self.mid[rows]) * self.scale[rows]
        c = self.c[rows]
        order = self.order[rows]
        leg = self.ftype[rows] == 2
        y = np.zeros_like(x) + c[:, 0]                 # shift (0) + c1
        y = np.where(order > 1, fma64(c[:, 1], x, y), y)
        a = np.ones_like(x)
        b = x.copy()
        for i in range(3, int(order.max()) + 1):
            cheb = fma64(x * 2, b, -a)
            # Legendre: ((2i-3) x b - (i-2) a) / (i-1)
            legv = fma64((2 * i - 3) * x, b, -((i - 2) * a)) / (i - 1)
            cc = np.where(leg, legv, cheb)
            y = np.where(order >= i, fma64(c[:, i - 1], cc, y), y)
            a = b
            b = cc
        return fma64(1.0, y, 0.0) / 1.0               # ysum += w*y ; /wsum

    def eval(self, xin: NDArray) -> NDArray[np.float64]:
        return self._eval_rows(np.arange(self.c.shape[0]), np.asarray(xin, float))

    def _eval1(self, k: int, x: float) -> float:
        return float(self._eval_rows(np.array([k]), np.array([x]))[0])

    # -- wf_msp_evali ---------------------------------------------------
    def _fallback(self, k: int, y: float, dydx: float) -> tuple[float, float]:
        c2 = float(self.xmin[k])
        c3 = float(self.xmax[k])
        xval = (c2 + c3) / 2.0
        yval = abs(self._eval1(k, xval) - y)
        dx = (c3 - c2) / 18.0
        while dx > _DX:
            x = max(c2, float(fma64(-dx, 9.0, xval)))
            while x <= min(c3, float(fma64(dx, 9.0, xval))):
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

    def inverse_state(self) -> tuple[NDArray, NDArray]:
        """Initial (x, dydx) set by ``wf_msp_init(INVERSE)``."""
        x = self.xmax - 0.5
        return x.copy(), self.eval(x) - self.eval(x - 1)

    def evali(self, y: NDArray, x: NDArray, dydx: NDArray) -> tuple[NDArray, NDArray]:
        """One ``wf_msp_evali`` call per order (vectorised); returns new state."""
        yval = np.maximum(self.ymin, np.minimum(self.ymax, y))
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
                dy = np.where(up, np.where(x + 1 < self.xmax, ep - y1, y1 - em),
                              np.where(dn, np.where(x - 1 > self.xmin, y1 - em, ep - y1),
                                       dy))
            act = ~done
            dydx = np.where(act & (dy != 0), dy, dydx)
            step = (yval - y1) / dydx
            xn = np.maximum(self.xmin, np.minimum(self.xmax, x + step))
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
        x = whole + (n.astype(_F32) * _F32(1e-4)).astype(np.float64)
        return x, dydx


def dispcor_grid(fn: MultispecFunctions, npts: int):
    """``dc_defaults`` (case 4) output grid: returns (w1, dw, nw) per order."""
    a = fn.eval(np.ones(len(fn.order)))
    b = fn.eval(np.full(len(fn.order), float(npts)))
    w1 = np.minimum(a, b)
    w2 = np.maximum(a, b)
    dw = (b - a) / (npts - 1)
    nw = (np.abs((w2 - w1) / dw) + 1.5).astype(np.int64)
    dw = (w2 - w1) / (nw - 1)
    return w1, dw, nw


@lru_cache(maxsize=16)
def dispcor_edges(id_paths: tuple[str, ...], npts: int):
    """Output-edge -> input-pixel map for every order (cached per calibration).

    Returns ``(w1, dw, nw, edges)`` where ``edges[k]`` holds the ``nw[k]+1``
    input logical pixel positions of order k's output-pixel edges, exactly
    as ``dispcor.x`` computes them (``x = 0.5, 1.5, ...`` through the output
    linear WCS and the stateful multispec inverse).
    """
    from decanter.utils.iraf_fmt import dtoc3
    fn = MultispecFunctions([read_id_tokens(p) for p in id_paths], npts)
    w1, dw, nw = dispcor_grid(fn, npts)
    # The output edges' wavelengths come from the OUTPUT multispec WCS, whose
    # linear terms dc_multispec stores as "%.14g" attribute text
    # (smw/smwswattrs.x) and wf_msp_init parses back with ctod;
    # wf_msp_fwd then evaluates crval + cdelt * (x - 1) (one fmadd), / (1+z).
    # Using the unrounded w1/dw moved a Newton result across a 1e-4
    # quantisation boundary (LCO24b setting5 m178, output edge 3052).
    w1t = np.array([iraf_ctod(dtoc3(float(v), 14, "g")) for v in w1])
    dwt = np.array([iraf_ctod(dtoc3(float(v), 14, "g")) for v in dw])
    xs = np.arange(int(nw.max()) + 1) + 0.5
    x, dydx = fn.inverse_state()
    edges = np.empty((len(id_paths), xs.size))
    for i, xo in enumerate(xs):
        w = fma64(dwt, xo - 1.0, w1t) / 1.0
        x, dydx = fn.evali(w, x, dydx)
        edges[:, i] = x
    return w1, dw, nw, edges
