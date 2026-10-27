"""Bit-exact, batched port of IRAF REAL ``curfit`` + ``icfit`` (Legendre).

This reproduces, operation by operation, what this Mac's IRAF build (f2c
``-KRw8``, ``clang -O2``, arm64) computes in

- ``math/curfit``: ``cvinit`` (cvinitr.x), ``cvfit`` = ``cvzero`` +
  ``cvacpts`` (cvacptsr.x, ``WTS_USER``) + ``cvsolve`` (cvsolver.x);
  ``rcv_bleg`` (cv_bevalr.x, via vops ``altar``/``amulr``/``awsur``);
  ``rcvchofac``/``rcvchoslv`` (cvchomatr.x); ``cvrject`` (cvrejectr.x,
  ``rcv_b1leg`` from cv_b1evalr.x); ``cvvector`` -> ``rcv_evleg``
  (cv_fevalr.x, via ``altmr``/``altar``/``amulr``/``awsur``/``amulkr``/
  ``aaddr``); ``cveval`` (cvevalr.x: ``rcv_b1leg`` + ``adotr``);
- ``pkg/xtools/icfit``: ``ic_rejectr`` (icrejectr.x) -> ``ic_deviantr``
  (icdeviantr.x) with grow = 0 and refit = YES, including the
  error-recovery branch of ``ic_rejectr``.

Fused multiply-adds, read off the disassembly of the matching ``.o`` in
``libcurfit.a`` / ``libvops.a`` / ``libxtools.a`` and confirmed against the
compiled kernels with the ctypes oracle (``scripts/iraf_oracle/
apscatter_oracle.py``):

==============================  ===========================================
SPP expression                  compiled as
==============================  ===========================================
``V += bw*y`` (cvacpts)         ``V = fma(bw, y, V)``, sequential in points
``M += bw*B`` (cvacpts)         ``M = fma(bw, B, M)``, sequential in points
``awsu: c = a*k1 + b*k2``       ``fma(a, k1, b*k2)``
``altm: b = a*k1 + k2``         ``fma(a, k1, k2)``
``ri2 = ri1*k1 + c1`` (evleg)   ``fma(ri1, k1, c1)``
``rcv_b1leg`` recurrence        ``fma(xn*(2i-3), b[i-1], -((i-2)*b[i-2]))/(i-1)``
``adot`` (n < 16)               ``s = fma(a[i], b[i], s)``, sequential
``c -= a*b`` (chofac, choslv)   ``fma(-a, b, c)``
``V -= bw*y``, ``M -= B*bw``    ``fma(-bw, y, V)``, ``fma(-B, bw, M)`` (cvrject)
``sigma += r**2``               ``fma(r, r, sigma)`` (ic_deviantr)
==============================  ===========================================

Everything is batched across independent fits.  Arrays are point-major,
``(npts, nfit)``: a sequential loop over the points of every fit is one
vectorized step per point, and each fit keeps its own IRAF operation order.
Fits may have different numbers of points; padding points carry a zero
product, which leaves an accumulator bit-unchanged.

Speed: a float32 FMA is emulated as ``RN32(RN64(a*b + c))`` (``a*b`` is
exact in float64).  That double rounding is wrong only when the float64 sum
lands exactly on a float32 rounding midpoint (or in the float32 subnormal
range); those cases are detected from the bit pattern and recomputed with
the error-free :func:`decanter.utils.fpemu.fma32`.  The result is exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from decanter.utils.fpemu import _round32, fma32, two_sum

R = np.float32
D = np.float64
CHEBYSHEV, LEGENDRE = 1, 2              # math/curfit.h
OK, SINGULAR, NO_DEG_FREEDOM = 0, 1, 2  # math/curfit.h
_SING_TOL = R(1.192e-6)                 # 10. * EPSILONR (compiled 0x359ffccf)
MAX_REAL = R(0.99e37)                   # hlib$mach.h
_TINY32 = 1.1754943508222875e-38        # smallest normal float32
_MID_MASK = np.int64(0x1FFFFFFF)        # float64 bits below float32 precision
_MID_BITS = np.int64(0x10000000)        # ... equal to half a float32 ulp


def _f32(a) -> NDArray[np.float32]:
    return np.asarray(a, dtype=R)


def _risky(prod: NDArray[np.float64], c: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Where ``RN32(RN64(prod + c))`` may differ from ``RN32(prod + c)``.

    The double rounding is wrong only if the float64 sum is inexact and lands
    exactly on a float32 rounding midpoint (or falls in the float32 subnormal
    range).  ``prod`` and ``c`` broadcast together.
    """
    prod, c = np.broadcast_arrays(np.asarray(prod, dtype=D), np.asarray(c, dtype=D))
    s = np.ascontiguousarray(prod + c)
    cand = (s.view(np.int64) & _MID_MASK) == _MID_BITS
    a = np.abs(s)
    tiny = (a < _TINY32) & (a > 0)
    if cand.any():
        ss, e = two_sum(prod[cand], c[cand])
        cand[cand] = e != 0
    return cand | tiny


def fma32f(a, b, c) -> NDArray[np.float32]:
    """Exact float32 ``fma(a, b, c)`` (fast path + exact fix-up)."""
    a = _f32(a)
    b = _f32(b)
    c = _f32(c)
    prod = np.atleast_1d(np.multiply(a, b, dtype=D))
    s = prod + c
    r = s.astype(R)
    bad = _risky(prod, c)
    if bad.any():
        aa, bb, cc = (np.atleast_1d(v) for v in np.broadcast_arrays(a, b, c))
        r[bad] = fma32(aa[bad], bb[bad], cc[bad])
    return r.reshape(np.broadcast_shapes(a.shape, b.shape, c.shape))


_CHAIN_BLOCK = 128


def fma_chain(a, b, valid=None) -> NDArray[np.float32]:
    """Sequential ``acc = fma(a[p], b[p], acc)`` from ``acc = 0`` over axis 0.

    ``a``, ``b`` (float32) and ``valid`` (bool, False = padding point that is
    skipped) broadcast together; axis 0 is the point index.  Works in blocks
    of points so the float64 products never exist for all points at once.
    Returns the final float32 accumulator (the broadcast shape minus axis 0).
    """
    a = _f32(a)
    b = _f32(b)
    shape = np.broadcast_shapes(a.shape, b.shape,
                                () if valid is None else valid.shape)
    n = shape[0]
    acc = np.zeros(shape[1:], dtype=R)
    start_acc = acc
    bad = np.zeros(shape[1:], dtype=bool)
    for p0 in range(0, n, _CHAIN_BLOCK):
        p1 = min(n, p0 + _CHAIN_BLOCK)
        prod = np.multiply(a[p0:p1] if a.shape[0] > 1 else a,
                           b[p0:p1] if b.shape[0] > 1 else b, dtype=D)
        if valid is not None:
            v = valid[p0:p1] if valid.shape[0] > 1 else valid
            prod = np.where(v, prod, 0.0)       # +0 leaves acc (never -0) as is
        prod = np.broadcast_to(prod, (p1 - p0,) + shape[1:])
        hist = np.empty((p1 - p0,) + shape[1:], dtype=R)
        start_acc = acc
        for p in range(p1 - p0):
            acc = (prod[p] + acc).astype(R)
            hist[p] = acc
        # The pre-rounding float64 sums of this block, checked for the rare
        # double-rounding cases.
        prev = np.empty(hist.shape, dtype=D)
        prev[0] = start_acc
        prev[1:] = hist[:-1]
        bad |= _risky(prod, prev).any(axis=0)
    if bad.any():
        # Redo the flagged accumulators with the error-free FMA.
        aa = np.broadcast_to(a, shape)[(slice(None),) + np.nonzero(bad)]
        bb = np.broadcast_to(b, shape)[(slice(None),) + np.nonzero(bad)]
        vv = (None if valid is None else
              np.broadcast_to(valid, shape)[(slice(None),) + np.nonzero(bad)])
        a64 = np.zeros(aa.shape[1:], dtype=D)
        for p in range(n):
            s, e = two_sum(np.multiply(aa[p], bb[p], dtype=D), a64)
            nxt = _round32(s, e).astype(D)
            a64 = nxt if vv is None else np.where(vv[p], nxt, a64)
        acc[bad] = a64.astype(R)
    return acc


def _ri_consts(i: int) -> tuple[np.float32, np.float32]:
    """``ri1 = (2ri-3)/(ri-1)``, ``ri2 = -(ri-2)/(ri-1)`` in REAL."""
    ri = R(i)
    ri1 = R(R(ri * R(2.0) - R(3.0)) / R(ri - R(1.0)))
    ri2 = R(R(-R(ri - R(2.0))) / R(ri - R(1.0)))
    return ri1, ri2


# --------------------------------------------------------------------------
# curfit kernels
# --------------------------------------------------------------------------

def cv_range(xmin, xmax):
    """``cvinit`` (cvinitr.x): ``k2 = 2/(xmax-xmin)``, ``k1 = (xmax+xmin)*-0.5``."""
    xmin = _f32(xmin)
    xmax = _f32(xmax)
    k2 = np.asarray(R(2.0) / (xmax - xmin), dtype=R)
    k1 = np.asarray((xmax + xmin) * R(-0.5), dtype=R)
    return k1, k2


def basis_vec(x, k1, k2, order: int) -> list[NDArray[np.float32]]:
    """``rcv_bleg`` (cv_bevalr.x, vector form): basis at every point of ``x``."""
    x = _f32(x)
    b = [np.ones_like(x)]
    if order >= 2:
        b.append(np.asarray((x + k1) * k2, dtype=R))          # altar
    for k in range(3, order + 1):
        ri1, ri2 = _ri_consts(k)
        t = np.asarray(b[1] * b[k - 2], dtype=R)              # amulr(B2, B[k-1])
        b.append(fma32f(t, ri1, np.asarray(b[k - 3] * ri2, dtype=R)))  # awsur
    return b


def basis_b1(x, k1, k2, order: int) -> list[NDArray[np.float32]]:
    """``rcv_b1leg`` (cv_b1evalr.x), evaluated elementwise."""
    x = _f32(x)
    b = [np.ones_like(x)]
    if order == 1:
        return b
    xn = np.asarray((x + k1) * k2, dtype=R)
    b.append(xn)
    for i in range(3, order + 1):
        ri = R(i)
        s7 = np.asarray(xn * R(ri * R(2.0) - R(3.0)), dtype=R)
        s16 = np.asarray(-(R(ri - R(2.0)) * b[i - 3]), dtype=R)
        b.append(np.asarray(fma32f(s7, b[i - 2], s16) / R(ri - R(1.0)), dtype=R))
    return b


def cveval(coeff, x, k1, k2) -> NDArray[np.float32]:
    """``cveval`` (cvevalr.x, Legendre): ``rcv_b1leg`` + ``adotr``.

    ``coeff`` is ``(order, ...)`` and broadcasts against ``x``.
    """
    order = coeff.shape[0]
    b = basis_b1(x, k1, k2, order)
    s = np.zeros(np.broadcast_shapes(b[0].shape, coeff[0].shape), dtype=R)
    for i in range(order):
        s = fma32f(b[i], coeff[i], s)
    return s


def cvvector(coeff, x, k1, k2) -> NDArray[np.float32]:
    """``cvvector`` -> ``rcv_evleg`` (cv_fevalr.x, Legendre)."""
    x = _f32(x)
    order = coeff.shape[0]
    if order == 1:
        return np.array(np.broadcast_to(coeff[0], np.broadcast_shapes(
            x.shape, coeff[0].shape)), dtype=R)
    ri1 = np.asarray(k2 * coeff[1], dtype=R)
    ri2 = fma32f(ri1, k1, coeff[0])
    yfit = fma32f(x, ri1, ri2)                         # altmr
    if order == 2:
        return yfit
    sx = np.asarray((x + k1) * k2, dtype=R)            # altar
    pnm2 = np.ones_like(sx)
    pnm1 = sx
    for i in range(3, order + 1):
        c1, c2 = _ri_consts(i)
        pn = np.asarray(sx * pnm1, dtype=R)            # amulr
        pn = fma32f(pn, c1, np.asarray(pnm2 * c2, dtype=R))   # awsur
        if i < order:
            pnm2 = pnm1
            pnm1 = pn
        pn = np.asarray(pn * coeff[i - 1], dtype=R)    # amulkr
        yfit = np.asarray(yfit + pn, dtype=R)          # aaddr
    return yfit


def accumulate_vector(basis, w, y, valid) -> NDArray[np.float32]:
    """``cvacpts`` right-hand side: ``V[k] = fma(w*B_k[p], y[p], V[k])`` over
    the points in order.  Returns ``(order, nfit)``."""
    bw = np.stack([np.asarray(w * bk, dtype=R) for bk in basis], axis=1)
    return fma_chain(bw, y[:, None, :], valid[:, None, :])


def accumulate_matrix(basis, w, valid) -> NDArray[np.float32]:
    """``cvacpts`` normal matrix, stored ``[band, row, fit]`` as IRAF does
    (``MATRIX[ii + (k-1)*order]`` = sum over points of
    ``fma(w*B_k, B_{k+ii}, .)``)."""
    order = len(basis)
    left, right, slots = [], [], []
    for k in range(order):
        bw = np.asarray(w * basis[k], dtype=R)
        for j in range(k, order):
            left.append(bw)
            right.append(basis[j])
            slots.append((j - k, k))
    acc = fma_chain(np.stack(np.broadcast_arrays(*left), axis=1),
                    np.stack(np.broadcast_arrays(*right), axis=1),
                    valid[:, None, :])
    matrix = np.zeros((order, order) + acc.shape[1:], dtype=R)
    for s, (band, row) in enumerate(slots):
        matrix[band, row] = acc[s]
    return matrix


def chofac(matrix: NDArray, matfac_prev: NDArray | None = None):
    """``rcvchofac`` (cvchomatr.x), nbands = nrows = order, batched."""
    nb, nr = matrix.shape[:2]
    ier = np.zeros(matrix.shape[2:], dtype=np.int64)
    if nr == 1:
        mf = np.zeros_like(matrix) if matfac_prev is None else matfac_prev.copy()
        pos = matrix[0, 0] > R(0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mf[0, 0] = np.where(pos, R(1.0) / matrix[0, 0], mf[0, 0])
        return mf, ier
    mf = matrix.copy()
    for n in range(nr):
        sing = ((mf[0, n] + matrix[0, n]) - matrix[0, n]) <= _SING_TOL
        ok = ~sing
        if sing.any():
            mf[:, n] = np.where(sing, R(0.0), mf[:, n])
            ier = np.where(sing, SINGULAR, ier)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.asarray(R(1.0) / mf[0, n], dtype=R)
        mf[0, n] = np.where(ok, inv, mf[0, n])
        imax = min(nb - 1, nr - 1 - n)
        if imax < 1:
            continue
        jmax = imax
        for i in range(1, imax + 1):
            ratio = np.asarray(mf[i, n] * mf[0, n], dtype=R)
            for j in range(1, jmax + 1):
                upd = fma32f(-mf[j - 1 + i, n], ratio, mf[j - 1, n + i])
                mf[j - 1, n + i] = np.where(ok, upd, mf[j - 1, n + i])
            jmax -= 1
            mf[i, n] = np.where(ok, ratio, mf[i, n])
    return mf, ier


def choslv(matfac: NDArray, vector: NDArray) -> NDArray[np.float32]:
    """``rcvchoslv`` (cvchomatr.x), batched."""
    nb, nr = matfac.shape[:2]
    if nr == 1:
        return np.asarray(vector[:1] * matfac[0, :1], dtype=R)
    c = [np.array(vector[i], dtype=R) for i in range(nr)]
    nbm1 = nb - 1
    for n in range(nr):
        for j in range(1, min(nbm1, nr - 1 - n) + 1):
            c[j + n] = fma32f(-matfac[j, n], c[n], c[j + n])
    for n in range(nr - 1, -1, -1):
        c[n] = np.asarray(c[n] * matfac[0, n], dtype=R)
        for j in range(1, min(nbm1, nr - 1 - n) + 1):
            c[n] = fma32f(-matfac[j, n], c[j + n], c[n])
    return np.stack(c)


@dataclass
class CurveBatch:
    """A batch of REAL Legendre ``cv`` descriptors, one per fit (last axis)."""

    order: int
    k1: NDArray[np.float32]        # CV_MAXMIN (nfit,)
    k2: NDArray[np.float32]        # CV_RANGE  (nfit,)
    xmin: NDArray[np.float32]      # CV_XMIN   (nfit,)
    xmax: NDArray[np.float32]      # CV_XMAX   (nfit,)
    matrix: NDArray[np.float32]    # (order, order, nfit)
    vector: NDArray[np.float32]    # (order, nfit)
    matfac: NDArray[np.float32]    # (order, order, nfit)
    coeff: NDArray[np.float32]     # (order, nfit)
    npts: NDArray[np.int64]        # CV_NPTS (nfit,)

    def take(self, idx) -> "CurveBatch":
        return CurveBatch(self.order, self.k1[idx], self.k2[idx],
                          self.xmin[idx], self.xmax[idx],
                          self.matrix[..., idx].copy(),
                          self.vector[..., idx].copy(),
                          self.matfac[..., idx].copy(),
                          self.coeff[..., idx].copy(), self.npts[idx].copy())

    def put(self, idx, other: "CurveBatch") -> None:
        self.matrix[..., idx] = other.matrix
        self.vector[..., idx] = other.vector
        self.matfac[..., idx] = other.matfac
        self.coeff[..., idx] = other.coeff
        self.npts[idx] = other.npts


def cvsolve(cv: CurveBatch) -> NDArray[np.int64]:
    """``cvsolve`` (cvsolver.x) for every fit in the batch; returns ``ier``."""
    ier = np.zeros(cv.npts.shape, dtype=np.int64)
    nodeg = (cv.npts - cv.order) < 0
    ier[nodeg] = NO_DEG_FREEDOM
    go = ~nodeg
    if go.any():
        idx = np.flatnonzero(go)
        mf, ier_f = chofac(cv.matrix[..., idx], cv.matfac[..., idx])
        cv.matfac[..., idx] = mf
        cv.coeff[..., idx] = choslv(mf, cv.vector[..., idx])
        ier[idx] = ier_f
    return ier


def cvfit(x, y, w, valid, order: int, xmin, xmax,
          shared_x: bool = False) -> tuple[CurveBatch, NDArray]:
    """``cvinit`` + ``cvfit`` (WTS_USER) for a batch of fits.

    ``x``, ``y``, ``w``, ``valid`` are ``(npts, nfit)``.  With ``shared_x``
    every fit has the same ``x``/``w``/``valid`` (given as ``(npts, 1)``)
    and the same ``xmin``/``xmax``, so the normal matrix is accumulated once
    and shared -- the arithmetic is identical per fit.
    """
    nfit = y.shape[1]
    xmin = np.broadcast_to(_f32(xmin), (nfit,)).copy()
    xmax = np.broadcast_to(_f32(xmax), (nfit,)).copy()
    k1, k2 = cv_range(xmin, xmax)
    if shared_x:
        basis = basis_vec(x, k1[:1][None, :], k2[:1][None, :], order)
        matrix = np.repeat(accumulate_matrix(basis, w, valid), nfit, axis=2)
        vector = accumulate_vector(basis, w, y, valid)
        npts = np.full(nfit, int(valid.sum()), dtype=np.int64)
    else:
        basis = basis_vec(x, k1[None, :], k2[None, :], order)
        matrix = accumulate_matrix(basis, w, valid)
        vector = accumulate_vector(basis, w, y, valid)
        npts = valid.sum(axis=0).astype(np.int64)
    cv = CurveBatch(order, k1, k2, xmin, xmax, matrix, vector,
                    np.zeros_like(matrix), np.zeros((order, nfit), dtype=R), npts)
    return cv, cvsolve(cv)


def cvrject_ranked(cv: CurveBatch, x, y, w, rej_new) -> None:
    """``cvrject`` (cvrejectr.x) for every newly rejected point, in order.

    ``rej_new`` is ``(npts, nfit)`` bool; ``x``/``w`` are ``(npts, nfit)``
    or ``(npts, 1)``.  Each fit's rejections are applied in increasing point
    index, as ``ic_deviantr`` does; fits run in parallel, rank by rank.
    """
    counts = rej_new.sum(axis=0)
    if counts.size == 0 or counts.max() == 0:
        return
    rank = np.cumsum(rej_new, axis=0) - 1
    pts, fits = np.nonzero(rej_new)
    rk = rank[pts, fits]
    xcol = 0 if x.shape[1] == 1 else None
    wcol = 0 if w.shape[1] == 1 else None
    order = cv.order
    for r in range(int(counts.max())):
        m = rk == r
        p, f = pts[m], fits[m]
        xf = x[p, f if xcol is None else 0]
        wf = w[p, f if wcol is None else 0]
        yf = y[p, f]
        xb = basis_b1(xf, cv.k1[f], cv.k2[f], order)
        for i in range(order):
            bw = np.asarray(xb[i] * wf, dtype=R)
            cv.vector[i, f] = fma32f(-bw, yf, cv.vector[i, f])
            for ii, j in enumerate(range(i, order)):
                cv.matrix[ii, i, f] = fma32f(-xb[j], bw, cv.matrix[ii, i, f])
        cv.npts[f] -= 1


def ic_reject(cv: CurveBatch, x, y, w, valid, low: float, high: float,
              niterate: int) -> NDArray[np.bool_]:
    """``ic_rejectr`` + ``ic_deviantr`` (grow = 0, refit = YES), batched.

    ``cv`` is updated in place; returns the final rejected-point mask.
    ``x``/``w``/``valid`` may be ``(npts, 1)`` (shared by every fit).
    """
    npts, nfit = y.shape
    rej = np.zeros((npts, nfit), dtype=bool)
    lowr, highr = R(low), R(high)
    if niterate <= 0 or (lowr == 0 and highr == 0):
        return rej
    shared = x.shape[1] == 1
    usable = valid & (w != R(0.0))
    active = np.ones(nfit, dtype=bool)
    for _ in range(int(niterate)):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        xs = x if shared else x[:, idx]
        ys = y[:, idx]
        us = (usable if shared else usable[:, idx]) & ~rej[:, idx]
        # ic_deviantr: residuals = y - rcvvector(cv, x)
        fit = cvvector(cv.coeff[:, idx][:, None, :], xs,
                       cv.k1[idx][None, :], cv.k2[idx][None, :])
        res = np.asarray(ys - fit, dtype=R)
        sig = fma_chain(res, res, us)
        j = us.sum(axis=0)
        few = j < 5
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = np.sqrt(np.asarray(sig / j.astype(R), dtype=R)).astype(R)
        low_cut = (np.asarray(-(lowr * sigma), dtype=R) if lowr > 0
                   else np.full_like(sigma, -MAX_REAL))
        high_cut = (np.asarray(highr * sigma, dtype=R) if highr > 0
                    else np.full_like(sigma, MAX_REAL))
        keep = (res < high_cut[None, :]) & (res > low_cut[None, :])
        new = us & ~keep & ~few[None, :]
        nnew = new.sum(axis=0)
        # j < 5: ic_deviantr returns without touching anything (repeating it
        # changes nothing); newreject == 0: ic_rejectr breaks.
        done = few | (nnew == 0)
        work = np.flatnonzero(~done)
        if work.size:
            wi = idx[work]
            sub = cv.take(wi)
            cvrject_ranked(sub, x if shared else x[:, wi], y[:, wi],
                           w if shared else w[:, wi], new[:, work])
            ier = cvsolve(sub)
            bad = ier != OK
            good = ~bad
            # ic_deviantr raises on SINGULAR / NO_DEG_FREEDOM; ic_rejectr
            # then refits from scratch and replays the completed iterations,
            # which (deterministically) restores the state before this one.
            cv.put(wi[good], sub.take(np.flatnonzero(good)))
            rej[:, wi[good]] |= new[:, work[good]]
            active[wi[bad]] = False
        active[idx[done]] = False
    return rej
