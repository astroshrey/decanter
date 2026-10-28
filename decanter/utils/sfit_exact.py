"""Bit-exact port of IRAF ``onedspec.continuum`` (``sfit``) as WARP runs it.

WARP normalises every FSR-cut spectrum with ``warp/PyContinuum.py``::

    iraf.continuum(input, output, interactive="no", low_rej=2., high_rej=3.,
                   grow=1., order=7 (15 for m < 100), sample="*",
                   function="spline3", niterate=2 (10 for m < 100))

i.e. ``noao/onedspec/t_sfit.x`` with ``type=ratio``: per spectrum,
``ic_fitr`` (``pkg/xtools/icfit/icfitr.x``) fits a REAL cubic spline
(``math/curfit``, SPLINE3) to the spectrum against its REAL wavelength
array, with ``ic_rejectr`` / ``ic_deviantr`` sigma clipping and region
growing (``grow`` is in wavelength units), and the output is
``advzr(y, cvvector(x))`` (``|fit| < 1e-20`` gives 0).

This module reproduces that arithmetic operation by operation, as this
Mac's IRAF build compiled it (f2c ``-KRw8``, ``clang -O2``, arm64):

==================================  =========================================
SPP expression                      compiled as (``objdump -d``)
==================================  =========================================
``rcv_bspline3`` / ``rcv_b1spline3``  ``sx = (x + k1) * k2`` (no fusion),
                                    ``left = min(int(sx), npieces)``,
                                    ``B1 = t*(t*t)``,
                                    ``B2 = fma(t, fma(t, fma(-t, 3, 3), 3), 1)``
                                    (same for ``B3`` with ``s``), ``B4 = s*(s*s)``
``rcv_evspline3``                   same basis (``apowkr(.,3)`` = ``a*(a*a)``);
                                    ``yfit = (((B1c + B2c) + B3c) + B4c)``
                                    with each ``Bkc = RN(Bk * coeff)``
``cvacpts`` (spline branch)         ``V = fma(w*Bk, y, V)``, ``M = fma(w*Bk, Bj, M)``,
                                    loops k, j, point
``cvrject``                         ``V = fma(-Bi*w, y, V)``, ``M = fma(-Bj, Bi*w, M)``
``rcvchofac`` / ``rcvchoslv``         :mod:`decanter.utils.icfit_exact`
``ic_deviantr``                     ``sigma = fma(r, r, sigma)``
==================================  =========================================

The ordered accumulations are computed exactly (every fused multiply-add
rounds once, :func:`decanter.utils.icfit_exact.fma32f`) and vectorised over
the independent accumulators: each accumulator receives its contributions
in IRAF's loop order.

Checked against the compiled IRAF objects with the ctypes oracle
(``scripts/iraf_oracle/sfit_oracle.py``) and against WARP's ``_VAC_norm``
products.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from decanter.utils.fpemu import fma32
from decanter.utils.icfit_exact import fma32f

R = np.float32
_THREE = R(3.0)
_ONE = R(1.0)
_ZERO = R(0.0)
SPLINE3_ORDER = 4
OK, SINGULAR, NO_DEG_FREEDOM = 0, 1, 2
MAX_REAL = R(0.99e37)


def _f32(a) -> NDArray[np.float32]:
    return np.asarray(a, dtype=R)


def _clip01(v: NDArray[np.float32]) -> NDArray[np.float32]:
    """``max(0., min(1., v))`` in REAL (no NaNs arise here)."""
    return np.maximum(_ZERO, np.minimum(_ONE, v)).astype(R)


def spline3_basis(x, npieces: int, k1, k2):
    """``rcv_bspline3`` (cv_bevalr.x): the 4 non-zero basis values and ``left``.

    ``x`` is REAL; ``k1 = -xmin``, ``k2 = spacing``.  Returns
    ``(B (4, n) float32, left (n,) int)``.
    """
    x = _f32(x)
    sx = _f32((x + R(k1)) * R(k2))                 # altar
    left = np.minimum(np.trunc(sx).astype(np.int64), npieces)   # achtri, aminki
    sx = _clip01(_f32(sx - left.astype(R)))
    tx = _clip01(_f32(_ONE - sx))
    b = np.empty((4,) + x.shape, dtype=R)
    b[0] = _f32(tx * _f32(tx * tx))
    b[1] = fma32f(tx, fma32f(tx, fma32f(-tx, _THREE, _THREE), _THREE), _ONE)
    b[2] = fma32f(sx, fma32f(sx, fma32f(-sx, _THREE, _THREE), _THREE), _ONE)
    b[3] = _f32(sx * _f32(sx * sx))
    return b, left


_MID_MASK = np.int64(0x1FFFFFFF)        # float64 bits below float32 precision
_MID_BITS = np.int64(0x10000000)        # ... at half a float32 ulp
_TINY32 = 1.1754943508222875e-38        # smallest normal float32


def _fma32_lean(a: NDArray[np.float32], b: NDArray[np.float32],
                c: NDArray[np.float32]) -> NDArray[np.float32]:
    """Exact float32 ``fma(a, b, c)`` for 1-D float32 arrays of one shape.

    ``a*b`` is exact in float64 and ``p + c`` is rounded once there; the
    float32 conversion of that sum is the single-rounded FMA except when the
    float64 sum is inexact and sits on a float32 rounding midpoint, or is
    subnormal in float32 -- those (rare) elements are redone with the
    error-free :func:`decanter.utils.fpemu.fma32`.
    """
    p = a.astype(np.float64) * b
    s = p + c
    r = s.astype(R)
    sv = s.view(np.int64)
    cand = ((sv & _MID_MASK) == _MID_BITS)
    aa = np.abs(s)
    cand |= (aa < _TINY32) & (aa > 0)
    if cand.any():
        idx = np.flatnonzero(cand)
        ss = s[idx]
        pp = p[idx]
        cc = c[idx].astype(np.float64)
        bb = ss - pp
        e = (pp - (ss - bb)) + (cc - bb)
        idx = idx[e != 0]
        if idx.size:
            r[idx] = fma32(a[idx], b[idx], c[idx])
    return r


_PACK = struct.Struct("<d")
_UNPACK = struct.Struct("<q")


def fma32_s(a: float, b: float, c: float) -> float:
    """Exact float32 ``fma(a, b, c)`` on Python floats holding float32 values."""
    p = a * b
    s = p + c
    r = float(np.float32(s))
    bb = s - p
    e = (p - (s - bb)) + (c - bb)
    if e != 0.0:
        bits = _UNPACK.unpack(_PACK.pack(s))[0]
        if (bits & 0x1FFFFFFF) == 0x10000000 or 0.0 < abs(s) < _TINY32:
            r = float(fma32(np.array([a], R), np.array([b], R), np.array([c], R))[0])
    return r


def fma_chain_1(a: NDArray[np.float32], b: NDArray[np.float32]) -> np.float32:
    """``acc = fma(a[k], b[k], acc)`` over k from ``acc = 0`` (one chain)."""
    av = _f32(a).astype(np.float64).tolist()
    bv = _f32(b).astype(np.float64).tolist()
    acc = 0.0
    for k in range(len(av)):
        acc = fma32_s(av[k], bv[k], acc)
    return R(acc)


def _ranks(target: NDArray, key: NDArray) -> NDArray[np.int64]:
    """Each contribution's 0-based position in its accumulator's sequence
    (increasing ``key`` within one ``target``)."""
    order = np.lexsort((key, target))
    t = target[order]
    start = np.ones(t.size, dtype=bool)
    start[1:] = t[1:] != t[:-1]
    first = np.flatnonzero(start)
    seg = np.cumsum(start) - 1
    rank = np.empty(t.size, dtype=np.int64)
    rank[order] = np.arange(t.size) - first[seg]
    return rank


def ordered_fma(acc: NDArray[np.float32], target: NDArray, a: NDArray, b: NDArray,
                key: NDArray | None = None, rank: NDArray | None = None) -> None:
    """``acc[target[p]] = fma(a[p], b[p], acc[target[p]])`` for p in order.

    Contributions are applied in increasing ``key`` (default: their position)
    within each accumulator -- or, when given, in increasing ``rank`` (each
    contribution's position in its accumulator's sequence); distinct
    accumulators are independent, so the work is vectorised across them, one
    rank at a time.  ``acc`` is a flat float32 array updated in place.
    Each step is :func:`_fma32_lean` with the exact float64 products formed
    once for all contributions.
    """
    if target.size == 0:
        return
    target = np.asarray(target, dtype=np.int64)
    a = _f32(a)
    b = _f32(b)
    if rank is None:
        rank = _ranks(target, np.arange(target.size) if key is None else np.asarray(key))
    rank = np.asarray(rank)
    nrank = int(rank.max()) + 1
    # rank after rank (a radix sort for small ranks; the order inside a rank
    # is free: every accumulator occurs once in it), each step on slices
    rk = rank.astype(np.int16 if nrank < 32767 else np.int64)
    by_rank = np.argsort(rk, kind="stable")
    bounds = np.zeros(nrank + 1, dtype=np.int64)
    bounds[1:] = np.cumsum(np.bincount(rk, minlength=nrank))
    tr = target[by_rank]
    ar = a[by_rank]
    br = b[by_rank]
    pr = ar.astype(np.float64) * br                 # exact in double
    for r in range(nrank):
        lo, hi = bounds[r], bounds[r + 1]
        tt = tr[lo:hi]
        c = acc[tt]
        sm = pr[lo:hi] + c                          # rounded once, in double
        res = sm.astype(R)
        sv = sm.view(np.int64)
        cand = ((sv & _MID_MASK) == _MID_BITS) | (np.abs(sm) < _TINY32)
        if cand.any():
            # a float32 midpoint or a float32 subnormal: exact unless the
            # double sum was itself inexact (zeros are exact)
            idx = np.flatnonzero(cand)
            ss = sm[idx]
            pp = pr[lo:hi][idx]
            cc = c[idx].astype(np.float64)
            bb = ss - pp
            e = (pp - (ss - bb)) + (cc - bb)
            idx = idx[e != 0]
            if idx.size:
                res[idx] = fma32(ar[lo:hi][idx], br[lo:hi][idx], c[idx])
        acc[tt] = res


@dataclass
class Spline3Curve:
    """A REAL SPLINE3 ``cv`` descriptor (``math/curfit/curfitdef.h``)."""

    npieces: int            # CV_NPIECES = order - 1
    ncoeff: int             # CV_NCOEFF = order + 3
    xmin: np.float32
    xmax: np.float32
    spacing: np.float32     # CV_SPACING = order / (xmax - xmin)
    matrix: NDArray[np.float32]   # (4, ncoeff): [diag ii, row]
    vector: NDArray[np.float32]   # (ncoeff,)
    matfac: NDArray[np.float32]   # (4, ncoeff)
    coeff: NDArray[np.float32]    # (ncoeff,)
    npts: int = 0

    @property
    def k1(self) -> np.float32:
        return R(-self.xmin)


def cvinit_spline3(order: int, xmin, xmax) -> Spline3Curve:
    """``cvinit`` (cvinitr.x) for SPLINE3 with ``order`` polynomial pieces."""
    xmin = R(xmin)
    xmax = R(xmax)
    if order < 1:
        raise ValueError("CVINIT: Illegal order.")
    if not xmax > xmin:
        raise ValueError("CVINIT: xmax <= xmin.")
    ncoeff = order + SPLINE3_ORDER - 1
    spacing = R(R(order) / R(xmax - xmin))
    z = np.zeros((SPLINE3_ORDER, ncoeff), dtype=R)
    return Spline3Curve(npieces=order - 1, ncoeff=ncoeff, xmin=xmin, xmax=xmax,
                        spacing=spacing, matrix=z, vector=np.zeros(ncoeff, R),
                        matfac=z.copy(), coeff=np.zeros(ncoeff, R))


def _cvacpts_terms(cv: Spline3Curve, x, y, w):
    """The FMA contributions of ``cvacpts`` (cvacptsr.x, SPLINE3 branch,
    ``WTS_USER``): ``(accumulator, a, b, order key)`` with the accumulators
    numbered ``[vector (ncoeff), matrix.T (ncoeff x 4)]`` and the key giving
    IRAF's loop order (k, then j, then the point) within each accumulator."""
    n = x.size
    basis, left = spline3_basis(x, cv.npieces, cv.k1, cv.spacing)
    order = SPLINE3_ORDER
    pts = np.arange(n)
    # vector: V[left + k] (0-based row left + k - 1 for k = 1..4), loops k, i
    tv, av, bv, kv = [], [], [], []
    tm, am, bm, km = [], [], [], []
    for k in range(order):                       # k = 0..3 (IRAF k = 1..4)
        bw = _f32(w * basis[k])                  # amulr
        tv.append(left + k)
        av.append(bw)
        bv.append(y)
        kv.append(k * n + pts)
        for j in range(k, order):
            ii = j - k
            tm.append((left + k) * order + ii)   # flat [row, ii] index
            am.append(bw)
            bm.append(basis[j])
            km.append(k * n + pts)               # k outer, then point
    nc = cv.ncoeff
    rank = None
    if n and np.all(left[1:] >= left[:-1]):
        # the sequence of every accumulator without sorting: accumulator
        # left + k (and each matrix entry of that row) gets the points of
        # left + k (k = 0), then left + k - 1 (k = 1), ... in point order, so
        # point p's contribution in block k comes after cnt[left_p + 1 ..
        # left_p + k] others plus its own position in its run of left_p.
        nl = cv.npieces + 1
        cum = np.zeros(nl + 1, dtype=np.int64)
        cum[1:] = np.cumsum(np.bincount(left, minlength=nl)[:nl])
        own = pts - cum[left]
        rk = [cum[np.minimum(left + k + 1, nl)] - cum[np.minimum(left + 1, nl)] + own
              for k in range(order)]
        rank = np.concatenate([rk[k] for k in range(order)]
                              + [rk[k] for k in range(order) for _j in range(k, order)])
    return (np.concatenate(tv + [t + nc for t in tm]), np.concatenate(av + am),
            np.concatenate(bv + bm), np.concatenate(kv + km), rank)


def _cvrject_terms(cv: Spline3Curve, x, y, w):
    """The FMA contributions of ``cvrject`` (cvrejectr.x), as
    :func:`_cvacpts_terms`."""
    n = x.size
    xb, left = spline3_basis(x, cv.npieces, cv.k1, cv.spacing)
    order = SPLINE3_ORDER
    seq = np.arange(n)
    tv, av, bv, kv = [], [], [], []
    tm, am, bm, km = [], [], [], []
    for i in range(order):
        bw = _f32(xb[i] * w)
        tv.append(left + i)
        av.append(-bw)
        bv.append(y)
        kv.append(seq)
        for j in range(i, order):
            tm.append((left + i) * order + (j - i))
            am.append(-xb[j])
            bm.append(bw)
            km.append(seq)
    nc = cv.ncoeff
    return (np.concatenate(tv + [t + nc for t in tm]), np.concatenate(av + am),
            np.concatenate(bv + bm), np.concatenate(kv + km), None)


def _apply_terms(cvs: list, terms: list) -> None:
    """Apply each curve's contributions to its ``vector`` / ``matrix``.

    The curves' accumulators are laid side by side and updated by one
    :func:`ordered_fma`: every accumulator still receives exactly its own
    contributions in its own order, so this is the same arithmetic as one
    curve at a time.
    """
    if not cvs:
        return
    width = 1 + SPLINE3_ORDER
    offs = []
    total = 0
    for cv in cvs:
        offs.append(total)
        total += cv.ncoeff * width
    acc = np.empty(total, dtype=R)
    for cv, o in zip(cvs, offs, strict=True):
        nc = cv.ncoeff
        acc[o:o + nc] = cv.vector
        acc[o + nc:o + nc * width] = np.ascontiguousarray(cv.matrix.T).ravel()
    ranks = [t[4] if t[4] is not None else _ranks(t[0], t[3]) for t in terms]
    ordered_fma(acc, np.concatenate([t[0] + o for t, o in zip(terms, offs, strict=True)]),
                np.concatenate([t[1] for t in terms]), np.concatenate([t[2] for t in terms]),
                rank=np.concatenate(ranks))
    for cv, o in zip(cvs, offs, strict=True):
        nc = cv.ncoeff
        cv.vector = acc[o:o + nc].copy()
        cv.matrix = acc[o + nc:o + nc * width].reshape(nc, SPLINE3_ORDER).T.copy()


def cvacpts(cv: Spline3Curve, x, y, w) -> None:
    """``cvacpts`` (cvacptsr.x, SPLINE3 branch, ``WTS_USER``)."""
    x = _f32(x)
    y = _f32(y)
    w = _f32(w)
    cv.npts += x.size
    _apply_terms([cv], [_cvacpts_terms(cv, x, y, w)])


def cvzero(cv: Spline3Curve) -> None:
    cv.matrix[...] = 0
    cv.vector[...] = 0
    cv.npts = 0


_SING_TOL = R(1.192e-6)                 # 10. * EPSILONR (cvchomatr.x)


def chofac_1(matrix: NDArray[np.float32]) -> tuple[NDArray[np.float32], int]:
    """``rcvchofac`` (cvchomatr.x) for one banded matrix ``[band, row]``;
    the same operations as :func:`decanter.utils.icfit_exact.chofac`."""
    nb, nr = matrix.shape
    m0 = [float(v) for v in matrix[0]]
    mf = [[float(v) for v in row] for row in matrix]
    ier = OK
    if nr == 1:
        out = np.zeros((nb, 1), dtype=R)
        if m0[0] > 0.0:
            out[0, 0] = R(R(1.0) / R(m0[0]))
        return out, ier
    f32 = np.float32
    for n in range(nr):
        sing = float(f32(f32(f32(mf[0][n]) + f32(m0[n])) - f32(m0[n]))) <= float(_SING_TOL)
        if sing:
            for k in range(nb):
                mf[k][n] = 0.0
            ier = SINGULAR
            continue
        mf[0][n] = float(f32(f32(1.0) / f32(mf[0][n])))
        imax = min(nb - 1, nr - 1 - n)
        if imax < 1:
            continue
        jmax = imax
        for i in range(1, imax + 1):
            ratio = float(f32(f32(mf[i][n]) * f32(mf[0][n])))
            for j in range(1, jmax + 1):
                mf[j - 1][n + i] = fma32_s(-mf[j - 1 + i][n], ratio, mf[j - 1][n + i])
            jmax -= 1
            mf[i][n] = ratio
    return np.array(mf, dtype=R), ier


def choslv_1(matfac: NDArray[np.float32], vector: NDArray[np.float32]) -> NDArray[np.float32]:
    """``rcvchoslv`` (cvchomatr.x) for one system; the same operations as
    :func:`decanter.utils.icfit_exact.choslv`."""
    nb, nr = matfac.shape
    mf = [[float(v) for v in row] for row in matfac]
    c = [float(v) for v in vector]
    if nr == 1:
        return np.array([float(R(R(c[0]) * R(mf[0][0])))], dtype=R)
    nbm1 = nb - 1
    for n in range(nr):
        for j in range(1, min(nbm1, nr - 1 - n) + 1):
            c[j + n] = fma32_s(-mf[j][n], c[n], c[j + n])
    for n in range(nr - 1, -1, -1):
        c[n] = float(R(R(c[n]) * R(mf[0][n])))
        for j in range(1, min(nbm1, nr - 1 - n) + 1):
            c[n] = fma32_s(-mf[j][n], c[j + n], c[n])
    return np.array(c, dtype=R)


def cvsolve(cv: Spline3Curve) -> int:
    """``cvsolve`` (cvsolver.x): ``rcvchofac`` + ``rcvchoslv``."""
    if cv.npts - cv.ncoeff < 0:
        return NO_DEG_FREEDOM
    mf, ier = chofac_1(cv.matrix)
    cv.matfac = mf
    cv.coeff = choslv_1(mf, cv.vector)
    return int(ier)


def cvfit(cv: Spline3Curve, x, y, w) -> int:
    """``cvfit`` (cvfitr.x): ``cvzero`` + ``cvacpts`` + ``cvsolve``."""
    cvzero(cv)
    cvacpts(cv, x, y, w)
    return cvsolve(cv)


def cvvector(cv: Spline3Curve, x) -> NDArray[np.float32]:
    """``cvvector`` -> ``rcv_evspline3`` (cv_fevalr.x)."""
    b, left = spline3_basis(x, cv.npieces, cv.k1, cv.spacing)
    c = cv.coeff
    y = np.zeros(b.shape[1:], dtype=R)                  # aclrr
    for k in range(4):
        y = _f32(y + _f32(b[k] * c[left + k]))
    return y


def cvrject(cv: Spline3Curve, x, y, w) -> None:
    """``cvrject`` (cvrejectr.x) for the points ``x, y, w`` in the given order."""
    x = _f32(np.atleast_1d(x))
    y = _f32(np.atleast_1d(y))
    w = _f32(np.atleast_1d(w))
    n = x.size
    if n == 0:
        return
    cv.npts -= n
    _apply_terms([cv], [_cvrject_terms(cv, x, y, w)])


class _FitError(Exception):
    pass


def _sum_sq(r: NDArray[np.float32]) -> np.float32:
    """``sigma = sigma + r**2`` over the points in order, compiled as an FMA."""
    if r.size == 0:
        return R(0.0)
    return fma_chain_1(r, r)


def _deviant_prep(cv: Spline3Curve, x, y, w, rejpts: NDArray[np.int64], low: float,
                  high: float, state: dict):
    """``ic_deviantr`` up to its ``sigma`` sum.  Returns ``("done", (nreject,
    newreject, None))`` where it returns before rejecting anything
    (``low = high = 0``, fewer than 5 points), else ``("sum", (resid, use,
    nreject, j))``."""
    lowr, highr = R(low), R(high)
    if lowr == 0 and highr == 0:
        return "done", (0, state.get("newreject", 0), None)
    resid = _f32(y - cvvector(cv, x))
    use = (w != 0) & (rejpts == 0)
    nreject = int(np.count_nonzero((~use) & (rejpts == 1)))
    j = int(np.count_nonzero(use))
    if j < 5:
        return "done", (nreject, state.get("newreject", 0), None)
    return "sum", (resid, use, nreject, j)


def _deviant_finish(x, w, rejpts: NDArray[np.int64], low: float, high: float, grow: float,
                    state: dict, resid, nreject: int, j: int, sigma):
    """``ic_deviantr`` from its ``sigma`` sum on (without the refit).
    Returns ``(nreject, newreject, seq)``, ``seq`` the newly rejected points
    in IRAF's order."""
    lowr, highr, growr = R(low), R(high), R(grow)
    npts = x.size
    sigma = R(np.sqrt(R(sigma / R(j))))
    low_cut = R(-R(lowr * sigma)) if lowr > 0 else -MAX_REAL
    high_cut = R(highr * sigma) if highr > 0 else MAX_REAL

    pixgrow = 0
    if growr > 0:
        # i == 1 sets pixgrow = grow/|dx|, later i take max(grow/|dx|,
        # real(pixgrow)); each assignment truncates to int, which for these
        # positive values equals truncating the overall maximum.
        dx = np.abs(_f32(x[1:] - x[:-1]))
        valid = dx >= R(0.0001)
        if valid.any():
            pixgrow = int(np.trunc(float(np.max(_f32(growr / dx[valid])))))

    newreject = 0
    deviant = ~((resid < high_cut) & (resid > low_cut))
    centers = np.flatnonzero(deviant & (w != 0) & (rejpts != 1))
    if centers.size:
        # Each deviant point rejects the not-yet-rejected points within
        # pixgrow pixels and grow (REAL |x_i - x_j|) of it.  For monotonic
        # x the windows are intervals moving right with i, so the points
        # IRAF rejects (in its loop order) are the sorted union.
        if np.all(np.diff(x) >= 0):
            offs = np.arange(-pixgrow, pixgrow + 1)
            jj = centers[:, None] + offs[None, :]
            inside = (jj >= 0) & (jj < npts)
            jc = np.clip(jj, 0, npts - 1)
            near = np.abs(_f32(x[centers][:, None] - x[jc])) <= growr
            ok = inside & near & (w[jc] != 0) & (rejpts[jc] == 0)
            seq = np.unique(jc[ok])
            order_seq = [int(v) for v in seq]
        else:
            order_seq = []
            for i in centers:
                i_min = max(0, i - pixgrow)
                i_max = min(npts - 1, i + pixgrow)
                for jj_ in range(i_min, i_max + 1):
                    if (abs(R(x[i] - x[jj_])) <= growr and w[jj_] != 0 and rejpts[jj_] == 0):
                        order_seq.append(jj_)
                        rejpts[jj_] = 2
        rejpts[np.asarray(order_seq, dtype=np.int64)] = 2
        newreject = len(order_seq)
    else:
        order_seq = []
    rejpts[rejpts != 0] = 1
    nreject += newreject
    state["newreject"] = newreject
    return nreject, newreject, np.asarray(order_seq, dtype=np.int64)


def _deviant_select(cv: Spline3Curve, x, y, w, rejpts: NDArray[np.int64], low: float,
                    high: float, grow: float, state: dict):
    """The rejection step of ``ic_deviantr`` (icdeviantr.x), without the refit.

    Returns ``(nreject, newreject, seq)``: ``seq`` holds the newly rejected
    points in IRAF's order, or is None where ``ic_deviantr`` returns before
    rejecting anything (``low = high = 0``, fewer than 5 points).
    """
    kind, v = _deviant_prep(cv, x, y, w, rejpts, low, high, state)
    if kind == "done":
        return v
    resid, use, nreject, j = v
    return _deviant_finish(x, w, rejpts, low, high, grow, state, resid, nreject, j,
                           _sum_sq(resid[use]))


def ic_deviant(cv: Spline3Curve, x, y, w, rejpts: NDArray[np.int64], low: float,
               high: float, grow: float, refit: bool, state: dict) -> tuple[int, int]:
    """``ic_deviantr`` (icdeviantr.x).  Returns ``(nreject, newreject)``.

    ``rejpts`` is updated in place (0 = NO, 1 = YES).  ``state['newreject']``
    carries the value of IRAF's static ``newreject`` across calls (it is not
    reset when fewer than 5 points remain).
    """
    nreject, newreject, sq = _deviant_select(cv, x, y, w, rejpts, low, high, grow, state)
    if sq is not None and refit and newreject > 0:
        cvrject(cv, x[sq], y[sq], w[sq])
        ier = cvsolve(cv)
        if ier in (SINGULAR, NO_DEG_FREEDOM):
            raise _FitError(ier)
    return nreject, newreject


def ic_reject(cv: Spline3Curve, x, y, w, low: float, high: float, niterate: int,
              grow: float, state: dict | None = None) -> NDArray[np.int64]:
    """``ic_rejectr`` (icrejectr.x).  Returns the rejected-point flags."""
    state = {} if state is None else state
    npts = x.size
    rejpts = np.zeros(npts, dtype=np.int64)
    if niterate <= 0:
        return rejpts
    nit = 0
    try:
        for _ in range(niterate):
            _, newrej = ic_deviant(cv, x, y, w, rejpts, low, high, grow, True, state)
            nit += 1
            if newrej == 0:
                break
    except _FitError:
        cvfit(cv, x, y, w)
        rejpts[:] = 0
        for _ in range(nit):
            ic_deviant(cv, x, y, w, rejpts, low, high, grow, True, state)
    return rejpts


def sfit_ratio(x, y, *, order: int, low: float, high: float, niterate: int,
               grow: float, state: dict | None = None):
    """``sfit`` (``continuum``, ``type=ratio``) of one spectrum.

    ``x`` (REAL wavelengths ``SX``) and ``y`` (``SY``) are the ``SN`` points
    ``shdr_open`` reads.  Returns ``(ratio (SN,) float32, curve)``.

    IRAF equivalent: ``noao/onedspec/t_sfit.x:sft_icfit`` -> ``ic_fitr``
    (``ic_dosetupr``: ``ord = min(order, nfit - 3)``, xmin/xmax from the
    data) -> ``advzr(SY, fit)``.
    """
    x = _f32(x)
    y = _f32(y)
    w = np.ones_like(x)
    n = x.size
    xmin = R(min(x[0], x[-1]))
    xmax = R(max(x[0], x[-1]))
    dmin, dmax = R(x.min()), R(x.max())
    xmin = R(min(xmin, dmin))
    xmax = R(max(xmax, dmax))
    ordv = min(order, n - 3)
    if ordv <= 0:
        raise NotImplementedError("sfit: fewer than 4 points (LEGENDRE fallback)")
    cv = cvinit_spline3(ordv, xmin, xmax)
    ier = cvfit(cv, x, y, w)
    if ier == NO_DEG_FREEDOM:
        raise NotImplementedError("sfit: no degrees of freedom")
    if low > 0 or high > 0:
        ic_reject(cv, x, y, w, low, high, niterate, grow, state)
    fit = cvvector(cv, x)
    small = (fit < R(1e-20)) & (fit > R(-1e-20))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(small, _ZERO, _f32(y / np.where(small, _ONE, fit))).astype(R)
    return ratio, cv


# ---------------------------------------------------------------------------
# Many independent fits at once (the same arithmetic, batched)
# ---------------------------------------------------------------------------

def _sum_sq_many(rs: list) -> list:
    """:func:`_sum_sq` of several residual vectors (bit for bit): with many,
    each chain is advanced one point at a time, all chains together (the
    shorter ones padded with zeros: ``fma(0, 0, acc) = acc`` for the
    non-negative ``acc`` of a sum of squares)."""
    if len(rs) < 8:
        return [_sum_sq(r) for r in rs]
    length = max(r.size for r in rs)
    m = np.zeros((length, len(rs)), dtype=np.float64)
    for k, r in enumerate(rs):
        m[:r.size, k] = r
    sq = m * m                                       # exact in double
    acc = np.zeros(len(rs), dtype=R)
    for i in range(length):
        sm = sq[i] + acc
        res = sm.astype(R)
        sv = sm.view(np.int64)
        cand = ((sv & _MID_MASK) == _MID_BITS) | (sm < _TINY32)
        if cand.any():
            idx = np.flatnonzero(cand)
            ss = sm[idx]
            pp = sq[i][idx]
            cc = acc[idx].astype(np.float64)
            bb = ss - pp
            e = (pp - (ss - bb)) + (cc - bb)
            idx = idx[e != 0]
            if idx.size:
                mi = m[i][idx].astype(R)
                res[idx] = fma32(mi, mi, acc[idx])
        acc = res
    return [R(v) for v in acc]


def ic_reject_many(cvs: list, xs: list, ys: list, ws: list, lows: list, highs: list,
                   niterates: list, grows: list, states: list) -> list:
    """:func:`ic_reject` of independent fits, run in lockstep.

    Each fit goes through exactly the steps :func:`ic_reject` takes for it
    (deviant selection, ``cvrject`` of the new rejections, ``cvsolve``, the
    same stopping rules and the same failure handling); the accumulations of
    the fits refitted in one iteration share one :func:`ordered_fma` and
    their ``sigma`` sums one chain loop.  Returns the rejection flags.
    """
    k_all = range(len(cvs))
    rej = [np.zeros(xs[k].size, dtype=np.int64) for k in k_all]
    nit = [0] * len(cvs)
    active = [k for k in k_all if niterates[k] > 0]
    failed = []
    while active:
        sel = {}
        summed = []
        for k in active:
            kind, sel[k] = _deviant_prep(cvs[k], xs[k], ys[k], ws[k], rej[k], lows[k],
                                         highs[k], states[k])
            if kind == "sum":
                summed.append(k)
        # the sigma sums of all of them as one chain loop
        sums = _sum_sq_many([sel[k][0][sel[k][1]] for k in summed])
        for k, sig in zip(summed, sums, strict=True):
            resid, _use, nreject, j = sel[k]
            sel[k] = _deviant_finish(xs[k], ws[k], rej[k], lows[k], highs[k], grows[k],
                                     states[k], resid, nreject, j, sig)
        refit = [k for k in active if sel[k][2] is not None and sel[k][1] > 0]
        terms = []
        for k in refit:
            sq = sel[k][2]
            cvs[k].npts -= sq.size
            terms.append(_cvrject_terms(cvs[k], _f32(xs[k][sq]), _f32(ys[k][sq]),
                                        _f32(ws[k][sq])))
        _apply_terms([cvs[k] for k in refit], terms)
        bad = set()
        for k in refit:
            if cvsolve(cvs[k]) in (SINGULAR, NO_DEG_FREEDOM):
                bad.add(k)
        nxt = []
        for k in active:
            if k in bad:
                failed.append(k)
                continue
            nit[k] += 1
            if sel[k][1] == 0 or nit[k] >= niterates[k]:
                continue
            nxt.append(k)
        active = nxt
    for k in failed:
        # ic_reject's error branch: refit from scratch, replay the iterations
        cvfit(cvs[k], xs[k], ys[k], ws[k])
        rej[k][:] = 0
        for _ in range(nit[k]):
            ic_deviant(cvs[k], xs[k], ys[k], ws[k], rej[k], lows[k], highs[k], grows[k], True,
                       states[k])
    return rej


def sfit_ratio_many(xs: list, ys: list, params: list) -> list:
    """:func:`sfit_ratio` of several independent spectra, bit for bit.

    ``params`` holds one ``dict(order=, low=, high=, niterate=, grow=)`` per
    spectrum.  The fits' accumulations (``cvacpts``, ``cvrject``) and
    ``sigma`` sums are batched across the spectra; every operation on a
    spectrum's values is the one :func:`sfit_ratio` performs, in the same
    order.  Returns ``[(ratio, curve)]``.
    """
    cvs, xx, yy, ww = [], [], [], []
    for x, y, prm in zip(xs, ys, params, strict=True):
        x = _f32(x)
        y = _f32(y)
        n = x.size
        xmin = R(min(x[0], x[-1]))
        xmax = R(max(x[0], x[-1]))
        dmin, dmax = R(x.min()), R(x.max())
        xmin = R(min(xmin, dmin))
        xmax = R(max(xmax, dmax))
        ordv = min(prm["order"], n - 3)
        if ordv <= 0:
            raise NotImplementedError("sfit: fewer than 4 points (LEGENDRE fallback)")
        cvs.append(cvinit_spline3(ordv, xmin, xmax))
        xx.append(x)
        yy.append(y)
        ww.append(np.ones_like(x))
    # cvfit: cvzero + cvacpts + cvsolve
    for cv, x in zip(cvs, xx, strict=True):
        cvzero(cv)
        cv.npts += x.size
    _apply_terms(cvs, [_cvacpts_terms(cv, x, y, w)
                       for cv, x, y, w in zip(cvs, xx, yy, ww, strict=True)])
    for cv in cvs:
        if cvsolve(cv) == NO_DEG_FREEDOM:
            raise NotImplementedError("sfit: no degrees of freedom")
    idx = [k for k, prm in enumerate(params) if prm["low"] > 0 or prm["high"] > 0]
    if idx:
        ic_reject_many([cvs[k] for k in idx], [xx[k] for k in idx], [yy[k] for k in idx],
                       [ww[k] for k in idx], [params[k]["low"] for k in idx],
                       [params[k]["high"] for k in idx], [params[k]["niterate"] for k in idx],
                       [params[k]["grow"] for k in idx], [{} for _ in idx])
    out = []
    for cv, x, y in zip(cvs, xx, yy, strict=True):
        fit = cvvector(cv, x)
        small = (fit < R(1e-20)) & (fit > R(-1e-20))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(small, _ZERO, _f32(y / np.where(small, _ONE, fit))).astype(R)
        out.append((ratio, cv))
    return out
