"""IRAF's number scanning and ``ranges`` sample parsing, as compiled.

``ctod`` / ``ctor`` / ``gctol`` (sys/fmtio) and the token parse of
``rg_xranges`` / ``rg_xadd`` (pkg/xtools/ranges/rgxranges.gx), which ICFIT's
``sample`` goes through (``ic_dosetup``: ``rg_xranges`` + ``rg_order`` +
``rg_merge``).  Ported from the f2c C of this Mac's iraf-community v2.18.1
build and checked against the compiled ``ctod_`` / ``ctor_`` /
``rg_xrangesr_``.

The part that matters for WARP: ``rg_xadd`` reads a token's second number
with ``rval2 = rval1; ctor (str, j, rval2)`` and ignores the return value,
but ``ctor`` stores ``dval`` whenever it is not out of REAL range, and
``ctod`` sets ``dval = 0`` before it finds no number.  So a token with one
number ``v`` is the range ``[min(v, 0), max(v, 0)]``, not ``[v, v]``; and
``INDEF`` (the ``sample`` WARP writes for every listfile line without
``bg=``) is ``[0, INDEFR]``, every x >= 0.
"""
from __future__ import annotations

import math

import numpy as np

from decanter.utils.fpemu import fma64, pow_di

R = np.float32

INDEFR = R(1.6e38)               # hlib$iraf.h INDEFR (f2c "(float)1.6e38")
INDEFD = 1.6e308                 # hlib$iraf.h INDEFD
_MAX_DIGITS = 25                 # hlib$mach.h MAX_DIGITS (ctod's dig[26])
_MAX_EXPONENTR = 38              # hlib$mach.h
_MAX_EXPONENTD = 308
_EPSILOND = 2.22e-16             # ctor.x EPSILOND as f2c wrote it (2.220d-16)
_LOG10E = 0.43429448190325182765  # libF77 d_lg10(x) = log10e * log(x)
_SEXAGESIMAL = "'\":dDhHmMsS"     # ctod.x: stridx(str[ip], "'\":dDhHmMsS")


def _digit(c: str) -> bool:
    return "0" <= c <= "9"


def _wrap64(v: int) -> int:
    """C ``long`` (64-bit two's complement) wrap-around."""
    return ((v + (1 << 63)) % (1 << 64)) - (1 << 63)


def _fcvtzs(x: float) -> int:
    """``(long) x`` as ctor.o compiles it (AArch64 ``fcvtzs``): truncation
    toward zero, saturating at the int64 limits, NaN -> 0."""
    if x != x:
        return 0
    if x >= 9.223372036854775807e18:
        return (1 << 63) - 1
    if x <= -9.223372036854775808e18:
        return -(1 << 63)
    return int(x)


def gctol(s: str, ip: int) -> tuple[int, int, int]:
    """IRAF ``gctol`` with radix 10 at 0-based ``ip`` (sys/fmtio/gctol.x).

    Returns ``(nchars, lval, q)``: ``nchars <= 0`` means no number, and ``q``
    is then back at the first non-blank character.  A ``B`` or ``X`` right
    after the digits is eaten whatever the radix (gctol.x:72-75 parse as
    ``(radix == OCTAL && ch == 'b') || ch == 'B'``).
    """
    n = len(s)
    while ip < n and s[ip] in " \t":
        ip += 1
    ip_save = ip
    neg = ip < n and s[ip] == "-"
    if neg or (ip < n and s[ip] == "+"):
        ip += 1
    first = ip
    lval = 0
    ch = ""
    while ip < n:
        ch = s[ip]
        if not _digit(ch):
            break
        lval = _wrap64(lval * 10 + (ord(ch) - 48))
        ip += 1
    if neg:
        lval = _wrap64(-lval)
    if ip == first:
        ip = ip_save
    elif ch in ("B", "X"):
        ip += 1
    return ip - first, lval, ip


def ctod(s: str, p: int) -> tuple[int, float, int]:
    """IRAF ``ctod`` at 0-based ``p`` (sys/fmtio/ctod.x, f2c + clang -O2).

    Returns ``(nchars, dval, q)``, ``q`` the first character not used.  No
    number gives ``(0, 0.0, q)`` (``dval = 0`` is set before the test; ``q``
    is past any leading blanks); ``INDEF`` gives :data:`INDEFD`; an exponent
    sum beyond 308 gives ``dval = 0`` with the characters counted.  The
    mantissa digits accumulate as ``fma(value, 10, digit)`` (ctod.o has the
    fmadd) and scale by ``pow_di(10, e)`` / ``pow_di(10, expon)``.

    Sexagesimal input (``ip > 1`` and the character after the integer digits
    one of ``'":dDhHmMsS``) raises :class:`NotImplementedError`: WARP stops on
    such a ``bg=`` token before IRAF sees it (``float()`` in
    ``write_apdatabase``).
    """
    n = len(s)
    ip = p
    while ip < n and s[ip] in " \t":
        ip += 1
    start = ip
    if s.startswith("INDEF", ip):
        ip += 5
        while ip < n and ("A" <= s[ip] <= "Z" or "a" <= s[ip] <= "z" or s[ip] == "_"):
            ip += 1
        return ip - start, INDEFD, ip
    neg = ip < n and s[ip] == "-"
    if neg or (ip < n and s[ip] == "+"):
        ip += 1
    while ip < n and s[ip] == "0":
        ip += 1
    dig: list[int] = []
    while len(dig) < _MAX_DIGITS and ip < n and _digit(s[ip]):
        dig.append(ord(s[ip]) - 48)
        ip += 1
    e = 0
    while ip < n and _digit(s[ip]):
        ip += 1
        e += 1
    if ip > 0 and ip < n and s[ip] in _SEXAGESIMAL:
        raise NotImplementedError(f"sexagesimal number in {s!r}")
    if ip < n and s[ip] == ".":
        ip += 1
        if not dig:
            while ip < n and s[ip] == "0":
                ip += 1
                e -= 1
        while len(dig) < _MAX_DIGITS and ip < n and _digit(s[ip]):
            dig.append(ord(s[ip]) - 48)
            e -= 1
            ip += 1
        while ip < n and _digit(s[ip]):
            ip += 1
    vexp = e + len(dig)
    if ip == start:
        return 0, 0.0, ip
    value = 0.0
    for d in dig:
        value = float(fma64(value, 10.0, float(d)))
    if e != 0:
        value = value * pow_di(10.0, e)
    expon = 0
    if ip < n and s[ip] in "eEdD":      # gctol <= 0: ip back before the 'e'
        k, lval, q = gctol(s, ip + 1)
        if k > 0:
            ip, expon = q, lval
    # abs(vexp + expon) > 308 as compiled: adds / cneg / cmp #309 / b.hs, i.e.
    # the wrapped sum's magnitude compared unsigned (INT64_MIN counts as big)
    if abs(_wrap64(vexp + expon)) > _MAX_EXPONENTD:
        return ip - start, 0.0, ip
    dval = 0.0 + value * 1.0             # dval (0) + value * scalar (1)
    if expon != 0:
        dval = dval * pow_di(10.0, expon)
    if neg:
        dval = -dval
    return ip - start, dval, ip


def ctor(s: str, p: int) -> tuple[int, np.float32 | None, int]:
    """IRAF ``ctor`` at 0-based ``p`` (sys/fmtio/ctor.x).

    Returns ``(nchars, rval, q)``.  ``rval`` is what ``ctor`` stores in its
    output argument: :data:`INDEFR` for ``INDEF``, ``(REAL) dval`` otherwise
    (also ``0`` when there is no number and ``nchars = 0``, and +-inf just
    above the REAL range), or ``None`` when ``int(log10 |dval|) > 38``, where
    ``ctor`` returns 0 without storing anything.  ``log10`` is libF77's
    ``d_lg10`` (``log10e * log(x)``).
    """
    nchars, dval, q = ctod(s, p)
    a = abs(dval)
    expon = _fcvtzs(_LOG10E * math.log(a)) if a > _EPSILOND else 0
    if dval == INDEFD:
        return nchars, INDEFR, q
    if expon > _MAX_EXPONENTR:
        return 0, None, q
    with np.errstate(over="ignore"):
        return nchars, R(dval), q


def rg_tokens(sample: str) -> list[str]:
    """``rg_xranges``' tokens: the runs between blanks, tabs, commas and
    newlines (rgxranges.gx:36-48)."""
    out, cur = [], []
    for c in sample:
        if c in " \t,\n":
            if cur:
                out.append("".join(cur))
                cur = []
        else:
            cur.append(c)
    if cur:
        out.append("".join(cur))
    return out


def rg_token(tok: str) -> tuple[np.float32, np.float32] | None | str:
    """``rg_xadd``'s parse of one token (rgxranges.gx:96-122).

    ``":"`` becomes a blank, then a leading ``*`` means every value
    (returned as the string ``"*"``); otherwise ``rval1`` is ``ctor`` from the
    start (no number: "Range syntax error", a warning, and the token adds
    nothing: ``None``) and ``rval2`` starts as ``rval1`` and is then
    overwritten by a second ``ctor`` from where the first stopped -- with 0
    when no second number follows.
    """
    if tok.startswith("@"):
        raise NotImplementedError("@file sample ranges")
    w = tok.replace(":", " ")
    if w.startswith("*"):
        return "*"
    n1, rval1, j = ctor(w, 0)
    if n1 == 0:
        return None
    _, r2, _ = ctor(w, j)
    rval2 = rval1 if r2 is None else r2
    return rval1, rval2
