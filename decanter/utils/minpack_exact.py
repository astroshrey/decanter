"""Bit-exact port of the MINPACK ``lmdif`` that WARP's ``scipy.optimize.leastsq`` ran.

WARP fits the stacked slit profile with ``scipy.optimize.leastsq`` (forward-
difference Levenberg-Marquardt, MINPACK ``lmdif``). The WARP reductions on this
machine made through 2026-10-19 ran scipy **1.15.3** (later ones run the
venv's scipy 1.14.1, which decanter's default reproduces by calling it; this
port is selected with ``DECANTER_WARP_MINPACK=c1153``), whose ``_minpack`` extension is scipy's C
translation of MINPACK (``LMDIF``, ``qrfac``, ``lmpar`` with ``qrsolv``,
``enorm`` and ``fdjac2`` inlined), compiled with Apple clang for arm64. scipy
<= 1.14 ships the original Fortran MINPACK instead, which stops at a slightly
different point: ~1e-9 relative on WARP's slit-profile fits (up to ~1e-5 on
poorly conditioned ones) away from WARP's fitted centers and widths. To be
independent of the installed scipy, this module re-implements that compiled
C code operation by operation:

- every fused multiply-add the compiler emitted is an exact ``fma`` here, and
  every multiply/add it kept separate is rounded separately (read from the
  ``fmadd``/``fmsub``/``fnmadd``/``fmla``/``fmls`` instructions of
  ``scipy/optimize/_minpack.cpython-310-darwin.so`` in the scipy 1.15.3 wheel);
- where clang vectorised an in-order dot product (``qrfac``'s Householder
  column dot, ``lmdif``'s scaled-gradient dot), the vector body multiplies and
  adds separately for the first ``8*floor(count/8)`` terms and only the scalar
  remainder is fused -- reproduced exactly;
- the machine constants are the C ones (``DBL_EPSILON``, ``DBL_MIN``), not the
  Fortran ``dpmpar`` values (this is the main source of the 1.14 vs 1.15
  difference: the forward-difference step is ``sqrt(max(epsfcn, eps))``);
- the C translation's ``qrfac`` column-norm recomputation
  (``rdiag[k] = enorm(m - j, &a[j+1 + k*lda])``, an off-by-one that reads one
  element past the column) is reproduced. For the last column that read goes
  past the end of the Jacobian buffer: harmless at the second-to-last
  Householder step, otherwise heap garbage that makes the compiled result
  non-deterministic (counted in ``STATS["undefined"]``; ``STRICT`` raises).

Validated against the scipy 1.15.3 binary on random problems and on every
WARP center-search fit (see ``scripts/s07_aperture_parity.py``).
"""
from __future__ import annotations

import math

import numpy as np

try:  # Python >= 3.13
    from math import fma as _fma
except ImportError:  # pragma: no cover - depends on the interpreter
    import ctypes
    import ctypes.util

    _libm = ctypes.CDLL(ctypes.util.find_library("m") or None)
    _libm.fma.restype = ctypes.c_double
    _libm.fma.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
    _fma = _libm.fma


def _check_fma() -> None:
    a = 1.0 + 2.0 ** -52
    if _fma(a, a, -1.0) != 2.0 ** -51 + 2.0 ** -104:
        raise RuntimeError("libm fma is not a correctly rounded fused multiply-add")


_check_fma()

DBL_EPSILON = 2.220446049250313e-16      # 0x3cb0000000000000
DBL_MIN = 2.2250738585072014e-308        # 0x0010000000000000
_RDWARF = float.fromhex("0x1.6a09e667f3bccp-65")   # 3.834e-20
_RGIANT = float.fromhex("0x1.6a09e667f3bcdp+63")   # 1.304e19
_P1 = 0.1
_P05 = 0.05
_P001 = 0.001
_P0001 = 1e-4


# Diagnostics of the C translation's qrfac off-by-one (see _qrfac).
STATS = {"recompute": 0, "past_end": 0, "undefined": 0}
PAST_END_VALUE = 0.0   # stand-in for the out-of-bounds heap value
STRICT = False         # raise UndefinedBehaviour instead of using PAST_END_VALUE


class UndefinedBehaviour(RuntimeError):
    """The compiled MINPACK would read memory outside the Jacobian here."""


def _maxnm(a: float, b: float) -> float:
    """arm64 ``fmaxnm`` (IEEE maxNum: a NaN operand yields the other)."""
    if a != a:
        return b
    if b != b:
        return a
    if a > b:
        return a
    if b > a:
        return b
    return b if math.copysign(1.0, a) < 0 else a


def _minnm(a: float, b: float) -> float:
    """arm64 ``fminnm``."""
    if a != a:
        return b
    if b != b:
        return a
    if a < b:
        return a
    if b < a:
        return b
    return a if math.copysign(1.0, a) < 0 else b


def _enorm(v, start: int, count: int) -> float:
    """MINPACK ``enorm`` (inlined form): Euclidean norm of ``v[start:start+count]``.

    ``count`` is also the ``n`` used for ``agiant = rgiant / n``.
    """
    if count < 1:
        return 0.0 * math.sqrt(0.0)
    agiant = _RGIANT / float(count)
    s1 = s2 = s3 = 0.0
    x1max = x3max = 0.0
    rdwarf = _RDWARF
    for i in range(start, start + count):
        x = v[i]
        xabs = abs(x)
        if xabs > rdwarf and xabs < agiant:
            s2 = _fma(x, x, s2)
        elif xabs <= rdwarf:
            if xabs <= x3max:
                if x != 0.0:
                    q = xabs / x3max
                    s3 = s3 + q * q
            else:
                q = x3max / xabs
                s3 = _fma(s3, q * q, 1.0)
                x3max = xabs
        else:
            if xabs <= x1max:
                q = xabs / x1max
                s1 = s1 + q * q
            else:
                q = x1max / xabs
                s1 = _fma(s1, q * q, 1.0)
                x1max = xabs
    if s1 != 0.0:
        return x1max * math.sqrt(s1 + (s2 / x1max) / x1max)
    if s2 != 0.0:
        if s2 < x3max:
            return math.sqrt(x3max * _fma(x3max, s3, s2 / x3max))
        return math.sqrt(s2 * _fma(x3max / s2, s3 * x3max, 1.0))
    return x3max * math.sqrt(s3)


def _dot_vec8(a, ia: int, b, ib: int, count: int, *, bdiv: float | None = None) -> float:
    """In-order dot product as clang vectorised it (8-wide body unfused, fused tail).

    ``sum_{t<count} a[ia+t] * b[ib+t]`` (or ``* (b[ib+t] / bdiv)``).
    """
    s = 0.0
    nvec = (count // 8) * 8 if count >= 8 else 0
    if bdiv is None:
        for t in range(nvec):
            s = s + a[ia + t] * b[ib + t]
        for t in range(nvec, count):
            s = _fma(a[ia + t], b[ib + t], s)
    else:
        for t in range(nvec):
            s = s + a[ia + t] * (b[ib + t] / bdiv)
        for t in range(nvec, count):
            s = _fma(a[ia + t], b[ib + t] / bdiv, s)
    return s


def _qrfac(m: int, n: int, a: list, lda: int, ipvt: list, rdiag: list, acnorm: list, wa: list) -> None:
    """C-MINPACK ``qrfac`` with column pivoting (``pivot = 1``); ``a`` column-major."""
    for j in range(n):
        acnorm[j] = _enorm(a, j * lda, m)
        rdiag[j] = acnorm[j]
        wa[j] = rdiag[j]
        ipvt[j] = j
    minmn = m if m < n else n
    for j in range(minmn):
        kmax = j
        for k in range(j, n):
            if rdiag[k] > rdiag[kmax]:
                kmax = k
        if kmax != j:
            cj = j * lda
            ck = kmax * lda
            for i in range(m):
                a[cj + i], a[ck + i] = a[ck + i], a[cj + i]
            rdiag[kmax] = rdiag[j]
            wa[kmax] = wa[j]
            ipvt[j], ipvt[kmax] = ipvt[kmax], ipvt[j]
        cj = j * lda
        cnt = m - j
        ajnorm = _enorm(a, cj + j, cnt)
        if ajnorm == 0.0:
            rdiag[j] = -ajnorm
            continue
        if a[cj + j] < 0.0:
            ajnorm = -ajnorm
        for i in range(j, m):
            a[cj + i] = a[cj + i] / ajnorm
        a[cj + j] = a[cj + j] + 1.0
        for k in range(j + 1, n):
            ck = k * lda
            s = _dot_vec8(a, cj + j, a, ck + j, cnt)
            temp = (-s) / a[cj + j]
            for i in range(j, m):
                a[ck + i] = _fma(temp, a[cj + i], a[ck + i])
            if rdiag[k] != 0.0:
                temp = a[ck + j] / rdiag[k]
                rdiag[k] = rdiag[k] * math.sqrt(_maxnm(_fma(-temp, temp, 1.0), 0.0))
                q = rdiag[k] / wa[k]
                if (q * q) * _P05 <= DBL_EPSILON:
                    # C translation: enorm(m - j, &a[j+1 + k*lda]) -- one past the column.
                    STATS["recompute"] += 1
                    if ck + j + 1 + cnt > len(a):
                        # Only possible for the last column (k == n-1). The value
                        # read past the end of the Jacobian buffer is heap garbage.
                        # It is harmless when j == n-2 (rdiag[n-1] is never used
                        # again before being overwritten); otherwise it steers the
                        # later pivoting and the compiled result is not
                        # reproducible (it can differ between two scipy calls).
                        STATS["past_end"] += 1
                        if j != n - 2:
                            STATS["undefined"] += 1
                            if STRICT:
                                raise UndefinedBehaviour(
                                    "qrfac norm recompute reads past the Jacobian")
                        rdiag[k] = _enorm(a[ck + j + 1:] + [PAST_END_VALUE], 0, cnt)
                    else:
                        rdiag[k] = _enorm(a, ck + j + 1, cnt)
                    wa[k] = rdiag[k]
        rdiag[j] = -ajnorm


def _lmpar(n: int, r: list, ldr: int, ipvt: list, diag: list, qtb: list, delta: float,
           par: float, x: list, sdiag: list, wa1: list, wa2: list) -> float:
    """C-MINPACK ``lmpar`` with ``qrsolv`` inlined; returns the new ``par``."""
    nsing = n
    for j in range(n):
        wa1[j] = qtb[j]
        if r[j + j * ldr] == 0.0 and nsing == n:
            nsing = j
        if nsing < n:
            wa1[j] = 0.0
    for k in range(1, nsing + 1):
        j = nsing - k
        wa1[j] = wa1[j] / r[j + j * ldr]
        temp = wa1[j]
        cj = j * ldr
        for i in range(j):
            wa1[i] = _fma(-r[cj + i], temp, wa1[i])
    for j in range(n):
        x[ipvt[j]] = wa1[j]

    it = 0
    for j in range(n):
        wa2[j] = diag[j] * x[j]
    dxnorm = _enorm(wa2, 0, n)
    fp = dxnorm - delta
    p1delta = delta * _P1
    if fp <= p1delta:
        return 0.0

    parl = 0.0
    if nsing == n:
        for j in range(n):
            l = ipvt[j]
            wa1[j] = diag[l] * (wa2[l] / dxnorm)
        for j in range(n):
            s = 0.0
            cj = j * ldr
            for i in range(j):
                s = _fma(r[cj + i], wa1[i], s)
            wa1[j] = (wa1[j] - s) / r[j + j * ldr]
        temp = _enorm(wa1, 0, n)
        parl = ((fp / delta) / temp) / temp

    for j in range(n):
        s = 0.0
        cj = j * ldr
        for i in range(j + 1):
            s = _fma(r[cj + i], qtb[i], s)
        wa1[j] = s / diag[ipvt[j]]
    gnorm = _enorm(wa1, 0, n)
    paru = gnorm / delta
    if paru == 0.0:
        paru = DBL_MIN / _minnm(delta, _P1)

    par = _maxnm(par, parl)
    par = _minnm(par, paru)
    if par == 0.0:
        par = gnorm / dxnorm

    while True:
        it += 1
        if par == 0.0:
            par = _maxnm(paru * _P001, DBL_MIN)
        temp = math.sqrt(par)
        for j in range(n):
            wa1[j] = temp * diag[j]
        _qrsolv(n, r, ldr, ipvt, wa1, qtb, x, sdiag, wa2)
        for j in range(n):
            wa2[j] = diag[j] * x[j]
        dxnorm = _enorm(wa2, 0, n)
        temp = fp
        fp = dxnorm - delta
        if abs(fp) <= p1delta:
            return par
        if parl == 0.0:
            if fp <= temp and temp < 0.0:
                return par
            if it == 10:
                return par
        elif it == 10:
            return par
        for j in range(n):
            l = ipvt[j]
            wa1[j] = diag[l] * (wa2[l] / dxnorm)
        for j in range(n):
            wa1[j] = wa1[j] / sdiag[j]
            temp = wa1[j]
            cj = j * ldr
            for i in range(j + 1, n):
                wa1[i] = _fma(-r[cj + i], temp, wa1[i])
        temp = _enorm(wa1, 0, n)
        if not (fp <= 0.0):
            parl = _maxnm(parl, par)
        parc = ((fp / delta) / temp) / temp
        if fp < 0.0:
            paru = _minnm(paru, par)
        par = _maxnm(parl, par + parc)


def _qrsolv(n: int, r: list, ldr: int, ipvt: list, diag: list, qtb: list, x: list,
            sdiag: list, wa: list) -> None:
    """C-MINPACK ``qrsolv`` as inlined into ``lmpar``."""
    for j in range(n):
        for i in range(j, n):
            r[i + j * ldr] = r[j + i * ldr]
        x[j] = r[j + j * ldr]
        wa[j] = qtb[j]
    for j in range(n):
        l = ipvt[j]
        if diag[l] != 0.0:
            for k in range(j, n):
                sdiag[k] = 0.0
            sdiag[j] = diag[l]
            qtbpj = 0.0
            for k in range(j, n):
                sk = sdiag[k]
                if sk == 0.0:
                    continue
                ck = k * ldr
                rkk = r[ck + k]
                if not (abs(rkk) < abs(sk)):
                    tan = sk / rkk
                    cos = 0.5 / math.sqrt(_fma(tan * tan, 0.25, 0.25))
                    sin = tan * cos
                else:
                    cotan = rkk / sk
                    sin = 0.5 / math.sqrt(_fma(cotan * cotan, 0.25, 0.25))
                    cos = cotan * sin
                r[ck + k] = _fma(sin, sk, rkk * cos)
                wak = wa[k]
                temp = _fma(cos, wak, qtbpj * sin)
                qtbpj = _fma(-sin, wak, qtbpj * cos)
                wa[k] = temp
                for i in range(k + 1, n):
                    rik = r[ck + i]
                    si = sdiag[i]
                    temp = _fma(cos, rik, sin * si)
                    sdiag[i] = _fma(-sin, rik, cos * si)
                    r[ck + i] = temp
        sdiag[j] = r[j + j * ldr]
        r[j + j * ldr] = x[j]
    nsing = n
    for j in range(n):
        if sdiag[j] == 0.0 and nsing == n:
            nsing = j
        if nsing < n:
            wa[j] = 0.0
    for k in range(1, nsing + 1):
        j = nsing - k
        s = 0.0
        cj = j * ldr
        for i in range(j + 1, nsing):
            s = _fma(r[cj + i], wa[i], s)
        wa[j] = (wa[j] - s) / sdiag[j]
    for j in range(n):
        x[ipvt[j]] = wa[j]


def lmdif(fcn, x, m: int, *, ftol: float = 1.49012e-8, xtol: float = 1.49012e-8,
          gtol: float = 0.0, maxfev: int = 0, epsfcn: float = DBL_EPSILON,
          factor: float = 100.0):
    """C-MINPACK ``LMDIF`` (``mode = 1``, ``nprint = 0``), as scipy 1.15.3 calls it.

    Args:
        fcn: ``fcn(x_list) -> sequence of m floats``.
        x: starting point (list of floats; not modified).
        m: number of residuals.

    Returns:
        ``(x, fvec, info, nfev)``.
    """
    n = len(x)
    x = [float(v) for v in x]
    if maxfev == 0:
        maxfev = 200 * (n + 1)
    info = 0
    if n <= 0 or m < n or ftol < 0.0 or xtol < 0.0 or gtol < 0.0 or maxfev <= 0 or factor <= 0.0:
        return x, None, info, 0
    fvec = list(fcn(x))
    nfev = 1
    fnorm = _enorm(fvec, 0, m)
    par = 0.0
    it = 1
    eps = math.sqrt(_maxnm(epsfcn, DBL_EPSILON))
    lda = m
    fjac = [0.0] * (m * n)
    diag = [0.0] * n
    ipvt = [0] * n
    qtf = [0.0] * n
    wa1 = [0.0] * n
    wa2 = [0.0] * n
    wa3 = [0.0] * n
    wa4 = [0.0] * m
    xnorm = 0.0
    delta = 0.0
    while True:
        # fdjac2
        for j in range(n):
            temp = x[j]
            h = eps * abs(temp)
            if h == 0.0:
                h = eps
            x[j] = temp + h
            wa = list(fcn(x))
            x[j] = temp
            cj = j * lda
            for i in range(m):
                fjac[cj + i] = (wa[i] - fvec[i]) / h
        nfev += n
        _qrfac(m, n, fjac, lda, ipvt, wa1, wa2, wa3)
        if it == 1:
            for j in range(n):
                diag[j] = wa2[j]
                if wa2[j] == 0.0:
                    diag[j] = 1.0
            for j in range(n):
                wa3[j] = diag[j] * x[j]
            xnorm = _enorm(wa3, 0, n)
            delta = xnorm * factor
            if delta == 0.0:
                delta = factor
        for i in range(m):
            wa4[i] = fvec[i]
        for j in range(n):
            cj = j * lda
            fjj = fjac[cj + j]
            if fjj != 0.0 and j < m:
                s = 0.0
                for i in range(j, m):
                    s = _fma(fjac[cj + i], wa4[i], s)
                temp = (-s) / fjj
                for i in range(j, m):
                    wa4[i] = _fma(fjac[cj + i], temp, wa4[i])
            fjac[cj + j] = wa1[j]
            qtf[j] = wa4[j]
        gnorm = 0.0
        if fnorm != 0.0:
            for j in range(n):
                l = ipvt[j]
                if wa2[l] == 0.0:
                    continue
                s = _dot_vec8(fjac, j * lda, qtf, 0, j + 1, bdiv=fnorm)
                gnorm = _maxnm(gnorm, abs(s / wa2[l]))
        if gnorm <= gtol:
            info = 4
            return x, fvec, info, nfev
        for j in range(n):
            diag[j] = _maxnm(diag[j], wa2[j])
        while True:
            par = _lmpar(n, fjac, lda, ipvt, diag, qtf, delta, par, wa1, wa2, wa3, wa4)
            for j in range(n):
                w = wa1[j]
                wa1[j] = -w
                wa2[j] = x[j] - w
                wa3[j] = diag[j] * wa1[j]
            pnorm = _enorm(wa3, 0, n)
            if it == 1:
                delta = _minnm(delta, pnorm)
            wa4_new = list(fcn(wa2))
            for i in range(m):
                wa4[i] = wa4_new[i]
            nfev += 1
            fnorm1 = _enorm(wa4, 0, m)
            actred = -1.0
            if fnorm1 * _P1 < fnorm:
                q = fnorm1 / fnorm
                actred = 1.0 - q * q
            for j in range(n):
                wa3[j] = 0.0
                temp = wa1[ipvt[j]]
                cj = j * lda
                for i in range(j + 1):
                    wa3[i] = _fma(fjac[cj + i], temp, wa3[i])
            temp1 = _enorm(wa3, 0, n) / fnorm
            temp2 = (pnorm * math.sqrt(par)) / fnorm
            t2sq = temp2 * temp2
            prered = _fma(temp1, temp1, t2sq + t2sq)
            ratio = actred / prered if prered != 0.0 else 0.0
            if ratio <= 0.25:
                if not (actred < 0.0):
                    temp = 0.5
                else:
                    dirder = -_fma(temp1, temp1, t2sq)
                    temp = (_fma(temp1, temp1, t2sq) * -0.5) / _fma(actred, 0.5, dirder)
                if fnorm1 * _P1 >= fnorm or temp < _P1:
                    temp = _P1
                delta = _minnm(delta, pnorm / _P1) * temp
                par = par / temp
            elif not (par != 0.0 and ratio < 0.75):
                delta = pnorm + pnorm
                par = par * 0.5
            if ratio >= _P0001:
                for j in range(n):
                    x[j] = wa2[j]
                    wa2[j] = diag[j] * x[j]
                for i in range(m):
                    fvec[i] = wa4[i]
                xnorm = _enorm(wa2, 0, n)
                fnorm = fnorm1
                it += 1
            # Branch senses as compiled (a NaN fails the ``>`` tests).
            c1 = not (abs(actred) > ftol or prered > ftol or 0.5 * ratio > 1.0)
            c2 = delta <= xnorm * xtol
            if c1 or c2:
                info = 3 if (c1 and c2) else (1 if c1 else 2)
                return x, fvec, info, nfev
            c6 = not (0.5 * ratio > 1.0 or abs(actred) > DBL_EPSILON or prered > DBL_EPSILON)
            c7 = not (delta > xnorm * DBL_EPSILON)
            c8 = not (gnorm > DBL_EPSILON)
            if c8:
                return x, fvec, 8, nfev
            if c7:
                return x, fvec, 7, nfev
            if c6:
                return x, fvec, 6, nfev
            if nfev >= maxfev:
                return x, fvec, 5, nfev
            if not (ratio < _P0001):
                break


def leastsq(func, x0, args=(), full_output=False, **_ignored):
    """Drop-in for ``scipy.optimize.leastsq(func, x0, args, full_output)`` (no Dfun).

    Reproduces scipy 1.15.3's C MINPACK bit for bit (see module docstring).
    Only the solution vector (``out[0]``) and ``ier`` are computed; ``cov_x``
    is ``None``.
    """
    x0 = np.asarray(x0).flatten()
    n = len(x0)
    if not isinstance(args, tuple):
        args = (args,)
    res = np.atleast_1d(func(*((x0[:n],) + args)))
    m = res.shape[0]
    if n > m:
        raise TypeError(f"Improper input: func input vector length N={n} must"
                        f" not exceed func output vector length M={m}")

    def fcn(xl):
        out = np.atleast_1d(func(*((np.array(xl, dtype=np.float64),) + args)))
        return np.asarray(out, dtype=np.float64).ravel().tolist()

    x, fvec, info, nfev = lmdif(fcn, x0.astype(np.float64).tolist(), m,
                                epsfcn=float(np.finfo(res.dtype).eps)
                                if np.issubdtype(res.dtype, np.inexact) else DBL_EPSILON)
    xa = np.array(x, dtype=np.float64)
    if full_output:
        return xa, None, {"nfev": nfev, "fvec": np.array(fvec if fvec is not None else [])}, "", info
    return xa, info
