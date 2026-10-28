"""Bit-exact IRAF ``math/curfit`` evaluation in REAL (float32) precision.

Ports ``cvrestore`` (REAL) + ``cveval`` for the polynomial curve types, as
compiled: the Chebyshev/Legendre basis recurrences and the ``adot`` dot
product use fused multiply-adds (see :mod:`decanter.utils.fpemu`). Used for
the aperture traces (``ap_cveval``) that drive the apall extraction.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from decanter.utils.fpemu import fma32

R = np.float32
CHEBYSHEV, LEGENDRE, SPLINE3, SPLINE1 = 1, 2, 3, 4


class CurveR:
    """A ``cvrestore``d REAL curve: ``fit = [type, order, xmin, xmax, c1..]``."""

    def __init__(self, fit) -> None:
        fit = np.asarray(fit, dtype=R)
        self.ctype = int(np.floor(float(fit[0]) + 0.5))
        self.order = int(np.floor(float(fit[1]) + 0.5))
        self.xmin = R(fit[2])
        self.xmax = R(fit[3])
        if self.ctype not in (CHEBYSHEV, LEGENDRE):
            raise NotImplementedError(f"curfit type {self.ctype}")
        self.k2 = R(R(2.0) / R(self.xmax - self.xmin))
        self.k1 = R(-R(self.xmax + self.xmin) / R(2.0))
        self.coeff = fit[4: 4 + self.order].astype(R)

    def basis(self, x: NDArray) -> list[NDArray[np.float32]]:
        x = np.asarray(x, dtype=R)
        b = [np.ones_like(x)]
        if self.order == 1:
            return b
        xnorm = ((x + self.k1).astype(R) * self.k2).astype(R)
        b.append(xnorm)
        for i in range(3, self.order + 1):
            if self.ctype == CHEBYSHEV:
                b.append(fma32((xnorm * R(2.0)).astype(R), b[-1], -b[-2]))
            else:
                ri = R(i)
                t1 = (R(ri * R(2.0) - R(3.0)) * xnorm).astype(R)
                t2 = (R(ri - R(2.0)) * b[-2]).astype(R)
                b.append((fma32(t1, b[-1], -t2) / R(ri - R(1.0))).astype(R))
        return b

    def eval(self, x: NDArray) -> NDArray[np.float32]:
        """``cveval`` (no clamping; see :func:`ap_cveval`)."""
        b = self.basis(x)
        s = np.zeros_like(b[0])
        for i in range(self.order):
            s = fma32(b[i], self.coeff[i], s)
        return s


def ap_cveval(cv: CurveR, x: NDArray) -> NDArray[np.float32]:
    """``apextract/apcveval.x``: clamp to the curve range, then ``cveval``."""
    x = np.asarray(x, dtype=R)
    return cv.eval(np.minimum(np.maximum(x, cv.xmin), cv.xmax))
