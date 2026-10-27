"""Bit-exact emulation of the floating-point arithmetic IRAF was compiled to.

IRAF's SPP code is translated to C by f2c (``-R``: REAL arithmetic stays
single precision) and compiled with ``clang -O2``. On arm64 clang contracts
``a*b + c`` inside one C expression into a fused multiply-add, which rounds
once instead of twice. numpy has no fma, so a straight numpy transcription
of an IRAF kernel lands a few float32 ULPs away from WARP. These helpers
reproduce the fused results exactly:

- :func:`fma32` -- ``round32(a*b + c)`` for float32 operands. The product of
  two float32 values is exact in float64, so only the final sum needs care
  (a tie-breaking fix-up handles the float64->float32 double rounding).
- :func:`fma64` -- ``round64(a*b + c)`` for float64 operands, via the
  Dekker/Knuth error-free transformations.
- :func:`pow_ri` -- f2c's ``pow_ri`` (REAL ** INTEGER), which IRAF evaluates
  in double precision by binary exponentiation.
- :func:`iraf_ctod` -- IRAF's ``ctod`` text->double conversion, which is NOT
  correctly rounded (digits * 10.0D0**e), unlike Python's ``float()``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

f32 = np.float32
f64 = np.float64
_SPLIT = f64(134217729.0)  # 2**27 + 1 (Veltkamp split constant)


def two_sum(a, b):
    """Error-free sum: ``s + e == a + b`` exactly, ``s = fl(a + b)``."""
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def two_prod(a, b):
    """Error-free product: ``p + e == a * b`` exactly, ``p = fl(a * b)``."""
    p = a * b
    t = _SPLIT * a
    ah = t - (t - a)
    al = a - ah
    t = _SPLIT * b
    bh = t - (t - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


_SCALARS = (float, int, np.floating, np.integer)


def _fma64_scalar(a: float, b: float, c: float) -> float:
    """:func:`fma64` on Python floats: the identical operation sequence
    (Python float arithmetic is IEEE double), without numpy's per-call cost."""
    p = a * b
    t = 134217729.0 * a
    ah = t - (t - a)
    al = a - ah
    t = 134217729.0 * b
    bh = t - (t - b)
    bl = b - bh
    ep = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    s = p + c
    bb = s - p
    es = (p - (s - bb)) + (c - bb)
    return s + (ep + es)


def fma64(a, b, c) -> NDArray[np.float64]:
    """``round64(a*b + c)`` with a single rounding (float64 fused multiply-add)."""
    if isinstance(a, _SCALARS) and isinstance(b, _SCALARS) and isinstance(c, _SCALARS):
        return f64(_fma64_scalar(float(a), float(b), float(c)))
    a = np.asarray(a, f64)
    b = np.asarray(b, f64)
    c = np.asarray(c, f64)
    p, ep = two_prod(a, b)
    s, es = two_sum(p, c)
    return s + (ep + es)


def _add_round_odd(x: NDArray, y: NDArray) -> NDArray[np.float64]:
    """``x + y`` rounded to odd (Boldo & Melquiond): exact sums are kept;
    an inexact one takes, of the two bracketing doubles, the one whose last
    mantissa bit is 1."""
    s, e = two_sum(x, y)
    bits = np.asarray(s, f64).view(np.int64)
    even = (bits & 1) == 0
    toward = np.where(e > 0, np.inf, -np.inf)
    alt = np.nextafter(s, toward)
    return np.where((e != 0) & even & np.isfinite(s), alt, s)


def fma64_cr(a, b, c) -> NDArray[np.float64]:
    """Correctly rounded float64 fused multiply-add, ``RN(a*b + c)``.

    :func:`fma64` (``s + (ep + es)``) can be one ULP off when the two error
    terms do not fit in one double (seen in ``dgsacpts`` sums). This is the
    Boldo-Melquiond emulation: with ``a*b = uh + ul`` and
    ``c + uh = th + tl`` exactly, ``RN(a*b + c) = RN(th + RO(tl + ul))``,
    where ``RO`` is round-to-odd. Valid away from overflow and underflow.
    """
    a = np.asarray(a, f64)
    b = np.asarray(b, f64)
    c = np.asarray(c, f64)
    uh, ul = two_prod(a, b)
    th, tl = two_sum(c, uh)
    return th + _add_round_odd(tl, ul)


def _round32(s: NDArray, e: NDArray) -> NDArray[np.float32]:
    """Correctly round the exact value ``s + e`` (``|e| <= ulp64(s)/2``) to float32."""
    r = s.astype(f32)
    rd = r.astype(f64)
    d = s - rd
    up = np.nextafter(r, f32(np.inf))
    dn = np.nextafter(r, f32(-np.inf))
    half_up = (up.astype(f64) - rd) / 2
    half_dn = (rd - dn.astype(f64)) / 2
    out = np.where((d == half_up) & (e > 0), up, r)
    out = np.where((d == -half_dn) & (e < 0), dn, out)
    return out.astype(f32)


def fma32(a, b, c) -> NDArray[np.float32]:
    """``round32(a*b + c)`` with a single rounding (float32 fused multiply-add)."""
    a = np.asarray(a, f32).astype(f64)
    b = np.asarray(b, f32).astype(f64)
    c = np.asarray(c, f32).astype(f64)
    s, e = two_sum(a * b, c)          # a*b is exact in float64
    return _round32(s, e)


def pow_ri(x, n: int) -> NDArray[np.float64]:
    """f2c ``pow_ri``: float32 ``x`` to a non-negative integer power, in double."""
    x = np.asarray(x, f32).astype(f64)
    p = np.ones_like(x)
    u = int(n)
    if u < 0:
        raise ValueError("pow_ri: negative exponents not needed here")
    while u:
        if u & 1:
            p = p * x
        u >>= 1
        if u:
            x = x * x
    return p


def pow_di(x: float, n: int) -> float:
    """f2c ``pow_di``: double ``x`` to an integer power (negative -> 1/x first)."""
    p = 1.0
    x = float(x)
    if n < 0:
        n = -n
        x = 1.0 / x
    u = n
    while u:
        if u & 1:
            p *= x
        u >>= 1
        if u:
            x *= x
    return p


_MAX_DIGITS = 25  # IRAF mach.h MAX_DIGITS


def iraf_ctod(s: str) -> float:
    """IRAF ``ctod``: parse a decimal string the way IRAF does.

    IRAF accumulates the (up to MAX_DIGITS) mantissa digits as an integer in
    double precision and scales by ``10.0D0 ** e`` (f2c ``pow_di``), then by
    ``10.0D0 ** expon`` for an explicit exponent. That is not correctly
    rounded, so the result can differ from ``float(s)`` by an ULP.
    Sexagesimal input is not supported (never occurs in the databases read).
    """
    t = s.strip()
    if t.startswith("INDEF"):
        return float("nan")
    neg = t.startswith("-")
    if t[:1] in "+-":
        t = t[1:]
    t = t.lstrip("0")
    i = 0
    dig = []
    while i < len(t) and t[i].isdigit():
        if len(dig) < _MAX_DIGITS:
            dig.append(t[i])
        i += 1
    e = 0
    # digits beyond MAX_DIGITS before the point are dropped but scale e
    nint = sum(1 for ch in t[:i] if ch.isdigit())
    e += max(0, nint - _MAX_DIGITS)
    if i < len(t) and t[i] == ".":
        i += 1
        if not dig:
            while i < len(t) and t[i] == "0":
                i += 1
                e -= 1
        while i < len(t) and t[i].isdigit():
            if len(dig) < _MAX_DIGITS:
                dig.append(t[i])
                e -= 1
            i += 1
    value = 0.0
    for ch in dig:
        # f2c: ``value = value * 10. + (dig[j-1] - 48)``, which clang fuses
        # (ctod.o has an fmadd here). Exact either way below 2**53, i.e. for
        # up to 15 digits; a 17-digit ``repr(float)`` needs the fused form.
        value = float(fma64(value, 10.0, float(ord(ch) - 48)))
    if e != 0:
        value = value * pow_di(10.0, e)
    expon = 0
    if i < len(t) and t[i] in "eEdD":
        rest = t[i + 1:]
        try:
            expon = int(rest.split()[0]) if rest.split() else 0
        except ValueError:
            expon = 0
    dval = 0.0 + value * 1.0
    if expon != 0:
        dval = dval * pow_di(10.0, expon)
    return -dval if neg else dval
