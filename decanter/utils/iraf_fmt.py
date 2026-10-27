"""IRAF's double -> text conversion, as used for FITS header values.

IRAF writes every double-valued header keyword through ``imio/db/imputd.x``,
which formats the value with the SPP ``sprintf("%0.*g")`` and lowers the
precision from 15 digits until the text fits in 19 characters. The digits
come from ``fmtio/dtoc3.x``, which is *not* a correctly rounded conversion:
it scales the value into ``[0.1, 1)`` by repeated multiplication/division by
ten (``dtcscl.x``), adds half a unit in the last place, and peels digits off
with ``v * 10`` / truncate. The last digit can therefore differ from Python's
``'%.15g' % v`` (e.g. WARP writes ``CDELT1 = 0.0344612471910111`` where
``'%.15g'`` gives ``...112``).

Downstream IRAF tasks re-read that text with ``ctod``, so a value that crosses
a file boundary in WARP must cross it the same way here to stay bit-exact
(:func:`roundtrip`).

The port follows the f2c C that IRAF compiled (``scripts/iraf_oracle/
gen_f2c.sh sys/fmtio/dtoc3.x``). clang fuses two expressions into FMAs
(``objdump -d dtoc3.o`` shows two ``fmadd``): the rounding increment
``v += 10**-n * 0.5`` (exact either way) and the look-ahead
``(int)(v * 10. + 2.22e-16)``.
"""
from __future__ import annotations

from decanter.utils.fpemu import fma64, iraf_ctod, pow_di

_MAX_DIGITS = 25          # mach.h MAX_DIGITS
_MAX_EXPONENTD = 308      # mach.h MAX_EXPONENTD
_EPSILOND = 2.22e-16      # mach.h EPSILOND (as compiled: 2.22e-16)
_SZ_OBUF = 1024           # any width large enough for a free-format field
_NDIGITS_DP = 16          # mach.h NDIGITS_DP
_NDIGITS_RP = 7           # mach.h NDIGITS_RP


def _dt_normd(x: float) -> tuple[float, int]:
    tol = 2.2200000000000002e-15   # EPSILOND * 10 as compiled
    absx = abs(x)
    expon = 0
    if absx > 0:
        while absx < (1.0 - tol):
            absx = absx * 10.0
            expon -= 1
            if absx == 0.0:
                return 0.0, 0
        while absx >= (10.0 + tol):
            absx = absx / 10.0
            expon += 1
    return (-absx if x < 0 else absx), expon


def _dt_equald(x: float, y: float) -> bool:
    if x == y:
        return True
    if x == 0.0 or y == 0.0:
        return False
    nx, ex = _dt_normd(x)
    ny, ey = _dt_normd(y)
    if ex != ey:
        return False
    tol = 7.1040000000000003e-15   # EPSILOND * 32 as compiled
    return abs(nx - ny) + 1.0 <= tol + 1.0


def _dtcscl(v: float) -> tuple[float, int]:
    """``dtcscl(v, e, 1)``: scale ``v`` (>= 0) into [0.1, 1.0), return (v, e)."""
    if _dt_equald(v, 0.0):
        return v, 0
    e = -1
    while v < 0.1:
        v = v * 10.0
        e -= 1
        if e <= -_MAX_EXPONENTD or _dt_equald(v, 0.0):
            e = 0
            break
    while v >= 1.0:
        v = v / 10.0
        e += 1
        if e >= _MAX_EXPONENTD:
            break
    return v, e


def dtoc3(val: float, decpl: int, fmt: str = "g", width: int = _SZ_OBUF + 1) -> str:
    """IRAF ``fmtio/dtoc3.x`` (formats ``f``, ``e``, ``g``).

    IRAF equivalent: ``sys/fmtio/dtoc3.x:dtoc3``.
    """
    val = float(val)
    v = abs(val)
    w = abs(int(width))
    fmt = fmt.lower()
    squeeze = fmt == "g"
    neg = val < 0.0
    d = abs(int(decpl)) if squeeze else max(0, int(decpl))

    v, e = _dtcscl(v)
    max_size = 2 if neg else 1
    no_digits = min(_MAX_DIGITS, d)

    ln = abs(e)
    e_size = 1
    i = 10
    while i <= 10000:
        if ln < i:
            break
        e_size += 1
        i *= 10
    e_size = e_size + max_size + 1
    if e < 0:
        e_size += 1

    f_size = 0
    if squeeze:
        e_size += d
        f_size = max(d, e + 1) if e > 0 else d - e
        f_size += max_size
    elif fmt == "f":
        f_size = max(e, 1) + max_size + d

    if squeeze:
        if f_size <= e_size:
            exp_format = False
            if e > 0:
                no_digits = f_size - max_size
            max_size = f_size
        else:
            exp_format = True
            max_size = e_size
        d = w
    elif fmt == "f":
        exp_format = f_size > w
        if exp_format:
            no_digits = max(1, w - e_size)
            max_size = no_digits + e_size
        else:
            no_digits = e + d + 1
            max_size = f_size
    else:
        exp_format = True
        max_size = e_size + d
        d = w

    if no_digits >= 0:
        v = float(fma64(pow_di(10.0, -no_digits), 0.5, v))
    no_digits = max(1, min(_MAX_DIGITS, no_digits))

    if v >= 1.0:
        v = v / 10.0
        e += 1
        if not exp_format:
            max_size += 1
            no_digits = min(_MAX_DIGITS, no_digits + 1)

    if max_size > w:
        return "*" * w

    digits: list[str] = []
    for _ in range(no_digits):
        v = v * 10.0
        j = int(v + _EPSILOND)
        v = v - j
        if int(float(fma64(v, 10.0, _EPSILOND))) >= 10:
            j += 1
            v = v - 1
            if v < 0:
                v = 0.0
        digits.append(chr(j + 48))

    out: list[str] = []
    if neg:
        out.append("-")
    if exp_format:
        out.append(digits[0])
        out.append(".")
        out.extend(digits[1:no_digits])
        out.append("E")
        if e < 0:
            out.append("-")
            e = -e
        out.append(str(e))
        return "".join(out)
    if e >= no_digits:
        out.extend(digits[:no_digits])
        out.extend("0" * max(0, e - no_digits + 1))
        if decpl > 0:
            out.append(".")
            if not squeeze:
                out.extend("0" * d)
        return "".join(out)
    if e < 0:
        if d == 0 and e == -1 and digits[0] >= "5":
            out.append("1")
        else:
            out.append("0")
        out.append(".")
        i = 1
        while i < -e and d > 0:
            out.append("0")
            d -= 1
            i += 1
        i = 1
    else:
        e = e + 2
        i = 1
        while i < e:
            out.append(digits[i - 1])
            i += 1
        if decpl > 0:
            out.append(".")
    j = 1
    while i <= no_digits and j <= d:
        out.append(digits[i - 1])
        i += 1
        j += 1
    if squeeze:
        # skip trailing zeroes (keep a non-digit such as the '.')
        n = len(out) + 1          # SPP 1-based "next free" index
        while n > 2:
            n -= 1
            if out[n - 1] != "0":
                n += 1
                break
        out = out[: n - 1]
    else:
        k = 1
        while k < d + e - no_digits and k <= d:
            out.append("0")
            k += 1
    return "".join(out)


def imputd_text(val: float) -> str:
    """Text IRAF writes for a double header value (``imio/db/imputd.x``)."""
    s = ""
    for prec in range(_NDIGITS_DP - 1, _NDIGITS_RP - 1, -1):
        s = dtoc3(val, prec, "g")
        if len(s) < 20:
            break
    return s


def roundtrip(val: float) -> float:
    """``ctod(imputd_text(val))``: the double an IRAF task re-reads."""
    return iraf_ctod(imputd_text(val))


# Names used by the s06 transform port (same routines).
def iraf_dtoc_g(val: float, decpl: int = 15) -> str:
    """IRAF ``%.<decpl>g`` text of a double (``dtoc3``, general format)."""
    return dtoc3(val, decpl, "g")


iraf_imputd_text = imputd_text
iraf_header_roundtrip = roundtrip
