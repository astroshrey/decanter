"""Bit-exact port of IRAF ``apall`` unweighted extraction (``ap_sum``).

WARP extracts each rectified order with ``apall`` (``weights=none``,
``background=none``, ``nsubaps=1``). This module reproduces the compiled
arithmetic of ``noao/twodspec/apextract/apextract.x`` -- ``ap_sum``,
``ap_edge`` and ``ap_asifit`` -- including:

- the REAL (float32) association of the aperture limits
  ``(center + low) + (cveval(line) - c1 + 1)``, and the conversion to the
  shifted data-buffer coordinates ``+ c1 - xs``;
- the per-line buffer origin ``xs = nint(center + low - 2 + shift)``
  (``interpbuf = 2`` for the linear edge interpolator);
- the integral-fraction edge weights from a linear ``asi`` fitted to the
  slice ``[xs - 2, ...]`` of the line, with IRAF's linear end padding;
- fused multiply-adds where clang contracts them.

The aperture (center/low/high on both axes and the trace curve) is read from
the database tokens with IRAF ``ctod`` and kept as REAL; an inverted aperture
(low > high) is swapped before extraction, as ``ap_extract`` does
(:func:`extract_reals`).  The data buffer is the strip as apall reads it
through the FITS kernel (``-0.0``, subnormals, ``+-Inf``/NaN -> ``+0``;
:func:`decanter.utils.iraf_imio.imio_read_real`).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from decanter.io.apdb import ApertureEntry
from decanter.utils.curfit_exact import CurveR, ap_cveval
from decanter.utils.fpemu import fma32, iraf_ctod
from decanter.utils.iraf_imio import imio_read_real

R = np.float32


def _r(tok: str) -> np.float32:
    return R(iraf_ctod(tok))


def _nint(x) -> int:
    xd = float(R(x))
    return int(np.floor(xd + 0.5)) if xd >= 0 else -int(np.floor(0.5 - xd))


class _LinearAsi:
    """``asifit`` + ``asievl`` + ``asigrl`` for II_LINEAR on one sample array.

    IRAF's coefficient array holds ``c[1..n]`` (the data) and the pad
    ``c[n+1]`` (asifit.x: ``ASI_NCOEFF = npix + 1``); ``c[0] = +0.0`` is
    what IRAF reads just before it, and :meth:`with_header` adds the cells
    before that (:func:`_need`)."""

    def __init__(self, data: NDArray[np.float32]) -> None:
        d = np.asarray(data, dtype=R)
        n = d.size
        c = np.empty(n + 2, dtype=R)          # 1-based: c[1..n], pad c[n+1]
        c[0] = R(0)
        c[1:n + 1] = d
        c[n + 1] = R(R(d[-1] * R(2.0)) - d[-2]) if n > 1 else d[-1]
        self.c = c
        self.n = n
        self.o = 0               # position of c[0] in self.c (with_header: 5)

    def with_header(self) -> "_LinearAsi":
        """A copy whose ``c[-5..0]`` hold what IRAF reads before its
        coefficient array: memio's header cells (sys/memio/msvfwa.x:46-50;
        64-bit IRAF: 8-byte ints, little-endian) -- ``Memi[bufptr-1]`` the
        lower sentinel 20030125 (minit.x:7) as REAL ``c[-1]`` (low half) and
        ``c[0]`` (high half, 0), ``Memi[bufptr-2]`` the element count
        ``n + 1`` (``c[-3]``; ``c[-2] = 0``), ``Memi[bufptr-3]`` the type
        TY_REAL = 6 (``c[-5]``; ``c[-4] = 0``).  Below are addresses."""
        out = _LinearAsi.__new__(_LinearAsi)
        out.c = np.concatenate([_memio_header(self.n + 1), self.c[1:]]).astype(R)
        out.n = self.n
        out.o = 5
        return out

    def eval_int(self, j: int) -> np.float32:
        # iilinr at an integer: (x-nx)*d[nx+1] + (nx+1-x)*d[nx] = d[nx]
        return self.c[j + self.o]

    def grl(self, a: np.float32, b: np.float32) -> np.float32:
        a = R(a)
        b = R(b)
        xa, xb = (b, a) if a > b else (a, b)
        na = int(xa)
        nb = int(xb)
        o = self.o
        c = self.c
        half = R(0.5)
        da = R(xa - R(na))
        if na == nb:
            db = R(xb - R(nb))
            diff = R(R(c[o + na + 1] - c[o + na]) * half)
            k = fma32(db, db, -R(da * da))
            acc = fma32(diff, k, fma32(R(db - da), c[o + na], R(0)))
        else:
            diff = R(R(c[o + na + 1] - c[o + na]) * half)
            acc = fma32(diff, fma32(-da, da, R(1.0)),
                        fma32(R(R(1.0) - da), c[o + na], R(0)))
            for j in range(na + 1, nb):
                acc = fma32(R(c[o + j + 1] + c[o + j]), half, acc)
            db = R(xb - R(nb))
            diff = R(R(c[o + nb + 1] - c[o + nb]) * half)
            acc = fma32(R(diff * db), db, fma32(c[o + nb], db, acc))
        acc = R(acc)
        return acc if a < b else R(-acc)

#: memio's header cells before a malloc'ed REAL array, as REALs -6..-1
#: (sys/memio/msvfwa.x:46-50; 64-bit IRAF: 8-byte ints, little-endian):
#: Memi[bufptr-3] = TY_REAL (6), Memi[bufptr-2] = the element count,
#: Memi[bufptr-1] = the lower sentinel L_SENTINAL = 20030125 (minit.x:7).
def _memio_header(nelem: int) -> NDArray[np.float32]:
    return np.array([6, 0, nelem & 0xFFFFFFFF, 0, 20030125, 0], dtype=np.uint32).view(R)


class IrafRealBuffer:
    """A malloc'ed REAL array the way IRAF code that indexes past it reads
    it: elements ``0..n-1``; before element 0 memio's header cells
    (``-6..-1``, :func:`_memio_header`); anything else (addresses before,
    malloc padding after) is refused (``NotImplementedError``)."""

    def __init__(self, values: NDArray[np.float32], what: str = "buffer") -> None:
        self.v = np.asarray(values, dtype=R)
        self.n = self.v.size
        self.hdr = _memio_header(self.n)
        self.what = what

    def at(self, i: int) -> np.float32:
        if 0 <= i < self.n:
            return self.v[i]
        if -6 <= i < 0:
            return self.hdr[i + 6]
        raise NotImplementedError(
            f"IRAF reads its {self.what} at element {i} of 0..{self.n - 1} (heap memory "
            "beyond memio's header cells / malloc padding): not reproducible. This happens "
            "only when the trace at buffer lines and at image lines disagree (bmin > 1).")

def _need(asi: _LinearAsi, lo: int, hi: int) -> None:
    """Refuse a read of IRAF's interpolant outside what is defined
    (``ii_linear`` reads ``c[int(x)]``, ``c[int(x)+1]``; the linear
    ``asigrl`` ``c[int(a)] .. c[int(b)+1]``).

    ``c[1..n+1]`` are the coefficients.  ``c[0]`` is the REAL just before
    the malloc'ed array: the upper half of memio's 8-byte lower sentinel
    (``Memi[bufptr-1] = L_SENTINAL = 20030125``, sys/memio/msvfwa.x:50,
    minit.x:7; 64-bit IRAF, little-endian), i.e. ``+0.0`` -- what
    ``_LinearAsi.c[0]`` holds; ``c[-5..-1]`` are memio's other header cells
    (:meth:`_LinearAsi.with_header`).  Before that are addresses, after
    ``c[n+1]`` malloc padding: refused."""
    if lo < -asi.o or hi > asi.n + 1:
        raise NotImplementedError(
            f"IRAF's ap_edge reads its edge interpolant at coefficients {lo}..{hi} of "
            f"1..{asi.n + 1} (apextract.x:ap_sum/ap_edge): heap memory, not "
            "reproducible. This happens only when the trace evaluated at buffer lines "
            "and at image lines disagree (a database whose dispersion limits start "
            "after line 1, bmin > 1; WARP always writes low = 1 - centre).")


def _grl_span(a, b) -> tuple[int, int]:
    xa, xb = (b, a) if a > b else (a, b)
    return int(R(xa)), int(R(xb)) + 1


def _ap_edge(asi: _LinearAsi, x1: np.float32, x2: np.float32):
    """``ap_edge`` (apextract.x:1730-1769), every read of the interpolant
    checked against IRAF's coefficient array (:func:`_need`)."""
    ix1 = _nint(x1)
    ix2 = _nint(x2)
    if ix1 == ix2:
        wt1 = R(x2 - x1)
        wt2 = R(0)
    else:
        wt1 = R(R(R(ix1) - x1) + R(0.5))
        wt2 = R(R(x2 - R(ix2)) + R(0.5))
    half = R(0.5)

    def grl(a, b):
        _need(asi, *_grl_span(a, b))
        return asi.grl(a, b)

    # below c[0] IRAF reads memio's header cells (_LinearAsi.with_header)
    if min(ix1 - 1, _grl_span(x1, x2)[0], ix2 - 1) < 0:
        asi = asi.with_header()
    _need(asi, ix1, ix1 + 1)
    if asi.eval_int(ix1) > 0:
        b = grl(R(R(ix1) - half), R(R(ix1) + half))
        if b > 0:
            if ix1 == ix2:
                a = grl(x1, x2)
            else:
                a = grl(x1, R(R(ix1) + half))
            if 0 < a < b:
                wt1 = R(a / b)
    if ix1 != ix2:
        _need(asi, ix2, ix2 + 1)
    if ix1 != ix2 and asi.eval_int(ix2) > 0:
        b = grl(R(R(ix2) - half), R(R(ix2) + half))
        if b > 0:
            a = grl(R(R(ix2) - half), x2)
            if 0 < a < b:
                wt2 = R(a / b)
    return wt1, wt2


def _sum3(sumfunc: str, wt1, d1, wt2, d2) -> np.float32:
    if sumfunc == "fma1":
        return R(fma32(wt1, d1, R(wt2 * d2)))
    if sumfunc == "fma2":
        return R(fma32(wt2, d2, R(wt1 * d1)))
    return R(R(wt1 * d1) + R(wt2 * d2))


def _ap_sum_line(flat: NDArray[np.float32], row0: int, na: int, xsp: int, xs: int,
                 s: np.float32, low: np.float32, high: np.float32, step: np.float32,
                 rc1: np.float32, *, sky: "IrafRealBuffer | None" = None, sky0: int = 0,
                 sumfunc: str = "fma1") -> np.float32:
    """One line of ``ap_sum`` (apextract.x:1525-1586, ``nsubaps = 1``) with
    IRAF's pointer arithmetic.

    ``flat`` is the whole data buffer (``nb x na`` lines, flattened) as apall
    reads it, ``row0`` the line's offset in it, ``xsp`` the line's buffer
    origin ``xs - c1 + 1`` (``xs`` = ``astart``, image column), ``s`` the
    trace ``cveval(iy) - c1 + 1``.  ``ap_asifit`` fits the interpolant to
    ``n = nint(min(nc + 0.49, high + s)) + 1 - (xsp - 1) + 1`` samples from
    ``Memr[dbuf_line + xsp - 2]`` (a copy clamped to the line where they
    leave it); the sum reads ``Memr[data + ix]`` with ``data = dbuf_line +
    xsp - 2`` and the sky ``Memr[sbuf + (iy - 1) nx - 1 + ix]`` (``sky0`` =
    ``(iy - 1) nx``; ``sky`` the malloc'ed sky buffer, :class:`IrafRealBuffer`)
    -- positions in the whole buffers, which leave the line only when the
    trace at buffer and at image lines disagree (``bmin > 1``).

    Raises:
        ValueError: ``n < 2``: IRAF's ``asifit`` errors ("ASIFIT: too few
            points for LINEAR") and apall stops.
        NotImplementedError: IRAF would read outside its buffers or its
            interpolant (uninitialised memory).
    """
    ax1 = max(R(0.5), R(low + s))
    ax2 = min(R(R(na) + R(0.49)), R(high + s))
    asi = None
    if ax1 < ax2:
        n_fit = _nint(ax2) + 1 - (xsp - 1) + 1
        if n_fit < 2:
            raise ValueError(
                "ASIFIT: too few points for LINEAR: IRAF's ap_asifit fits "
                f"{n_fit} sample(s) (apextract.x:1812-1816); apall stops here (the trace "
                "at buffer lines and at image lines disagree: bmin > 1)")
        k = np.arange(n_fit)
        pos = xsp - 2 + k
        if xsp - 2 >= 0 and xsp - 2 + n_fit - 1 <= na - 1:
            samp = flat[row0 + pos]
        else:
            samp = flat[row0 + np.clip(pos, 0, na - 1)]
        asi = _LinearAsi(samp)
    rxs = R(xs)
    x1 = R(R(max(R(0.5), R(low + s)) + rc1) - rxs)
    x2 = R(R(min(R(R(na) + R(0.49)), R(R(low + step) + s)) + rc1) - rxs)
    if x2 <= x1:
        return R(0)
    if asi is None:
        raise NotImplementedError(
            "IRAF's ap_sum sums a line whose ap_asifit returned early (the previous "
            "line's interpolant and data pointer): not reproducible")
    ix1 = _nint(x1)
    ix2 = _nint(x2)
    base = row0 + xsp - 2
    if base + ix1 < 0 or base + ix2 > flat.size - 1:
        raise NotImplementedError(
            "IRAF's ap_sum reads outside its data buffer here (uninitialised memory)")
    wt1, wt2 = _ap_edge(asi, R(x1 + R(1)), R(x2 + R(1)))
    sval = _sum3(sumfunc, wt1, flat[base + ix1], wt2, flat[base + ix2])
    for ix in range(ix1 + 1, ix2):
        sval = R(sval + flat[base + ix])
    if sky is not None:
        q = sky0 - 1
        skyval = R(fma32(wt1, sky.at(q + ix1), R(wt2 * sky.at(q + ix2))))
        for ix in range(ix1 + 1, ix2):
            skyval = R(skyval + sky.at(q + ix))
        sval = R(sval - skyval)
    return sval


def buffer_range(entry: ApertureEntry, ncols: int, nlines: int) -> tuple[int, int]:
    """``(a1, a2)``: the columns ``ap_extract`` needs for one aperture
    (apextract.x:241-282, weights none, linear edge interpolator)."""
    (cen_a, cen_b), (low_a, low_b), (high_a, high_b), cv = extract_reals(entry)
    bmin = min(nlines, max(1, _nint(R(cen_b + low_b))))
    bmax = max(1, min(nlines, _nint(R(cen_b + high_b))))
    shifts = ap_cveval(cv, np.arange(bmin, bmax + 1, dtype=R))
    cmin, cmax = R(shifts.min()), R(shifts.max())
    a1 = min(ncols, max(1, _nint(R(R(cen_a + R(low_a - R(2))) + cmin))))
    a2 = max(1, min(ncols, _nint(R(R(cen_a + R(high_a + R(2))) + cmax))))
    return a1, a2


def aperture_reals(entry: ApertureEntry):
    """(cen_a, cen_b, low_a, low_b, high_a, high_b, CurveR) as IRAF REALs."""
    if entry.raw_center is None or entry.raw_curve is None:
        raise ValueError("aperture entry lacks raw database tokens")
    cen = tuple(_r(t) for t in entry.raw_center)
    low = tuple(_r(t) for t in entry.raw_low)
    high = tuple(_r(t) for t in entry.raw_high)
    cv = CurveR([_r(t) for t in entry.raw_curve])
    return cen, low, high, cv


def extract_reals(entry: ApertureEntry):
    """:func:`aperture_reals` as ``ap_extract`` uses them: on each axis a lower
    limit above the upper one is swapped first (``apextract.x:139-149``, "fix
    possible limit error"; REAL comparison). The aperture database keeps the
    unswapped values; the extraction and the output header use these."""
    cen, low, high, cv = aperture_reals(entry)
    low, high = list(low), list(high)
    for k in range(2):
        if low[k] > high[k]:
            low[k], high[k] = high[k], low[k]
    return cen, tuple(low), tuple(high), cv


def apnum_limits(entry: ApertureEntry) -> tuple[np.float32, np.float32]:
    """The two REAL limits apall stores for ``APNUM1`` of a ONEDSPEC output
    with ``nsubaps = 1`` (``apextract.x:1195-1216``): ``low = cen + low``,
    ``high = cen + high``, ``step = (high - low) / 1``, ``low = low + 0 * step``,
    ``high = low + step``, from the swapped limits (:func:`extract_reals`).
    ``apmw_saveim`` prints them with ``%.2f`` (``pargd``)."""
    (cen_a, _), (low_a, _), (high_a, _), _ = extract_reals(entry)
    low = R(cen_a + low_a)
    high = R(cen_a + high_a)
    step = R(R(high - low) / R(1))
    low = R(low + R(R(0) * step))
    return low, R(low + step)


def apsum_scalar(image: NDArray, entry: ApertureEntry, *, sumfunc: str = "fma1") -> NDArray[np.float32]:
    """``apall`` box extraction of one aperture from a rectified strip.

    ``image`` is the 2-D strip, shape ``(n_lines, n_cols)`` (dispersion along
    lines, aperture axis = columns, i.e. ``dispaxis = 2``). Returns one REAL
    value per line.
    """
    img = np.asarray(image, dtype=R)
    nlines, ncols = img.shape
    (cen_a, cen_b), (low_a, low_b), (high_a, high_b), cv = extract_reals(entry)

    # Line range of the aperture (ap_extract).
    bmin = min(nlines, max(1, _nint(R(cen_b + low_b))))
    bmax = max(1, min(nlines, _nint(R(cen_b + high_b))))
    lines = np.arange(bmin, bmax + 1, dtype=R)
    shifts = ap_cveval(cv, lines)
    cmin = R(shifts.min())
    cmax = R(shifts.max())
    xmin = R(low_a - R(2))
    xmax = R(high_a + R(2))
    amin = min(ncols, max(1, _nint(R(R(cen_a + xmin) + cmin))))
    amax = max(1, min(ncols, _nint(R(R(cen_a + xmax) + cmax))))
    na = amax - amin + 1
    nb = bmax - bmin + 1
    # Buffer origin per line (W_NONE branch).
    xw = R(R(cen_a + low_a) - R(2))
    astart = [_nint(R(xw + s)) for s in shifts]

    # the data buffer as apall reads it through the FITS kernel (IEEE input
    # mapping: subnormals and -0 -> +0, Inf/NaN -> 0; utils/iraf_imio)
    buf = imio_read_real(img[bmin - 1: bmax, amin - 1: amax])   # (nb, na)
    flat = buf.reshape(-1)
    c1 = amin
    low = R(cen_a + low_a)
    high = R(cen_a + high_a)
    step = R(R(high - low) / R(1))
    out = np.zeros(nlines, dtype=R)
    rc1 = R(c1)
    for iy in range(1, nb + 1):
        # ap_extract passes l1 = ys = 1: the trace is evaluated at buffer
        # line iy (image line bmin + iy - 1), the buffer origin astart at the
        # image line (apextract.x:432-443, 1543-1546)
        s = R(R(ap_cveval(cv, np.array([iy], dtype=R))[0] - rc1) + R(1))
        out[bmin - 1 + iy - 1] = _ap_sum_line(
            flat, (iy - 1) * na, na, astart[iy - 1] - c1 + 1, astart[iy - 1], s, low, high,
            step, rc1, sumfunc=sumfunc)
    return out


def _nint_v(x: NDArray) -> NDArray[np.int64]:
    xd = np.asarray(x, R).astype(np.float64)
    return np.where(xd >= 0, np.floor(xd + 0.5), -np.floor(0.5 - xd)).astype(np.int64)


def irregular_lines(*, empty, ax1, ax2, n_fit, xsp, ix1, ix2, e1, e2, j1, j2, na: int,
                    nx: int | None = None) -> NDArray[np.bool_]:
    """The lines whose ``ap_sum`` reads leave the line's data, its sky (with
    ``nx``, the sky buffer's width) or its interpolant's coefficients, or
    whose ``asifit`` gets fewer than 2 samples (geometry only, conservative):
    the vectorised extractions evaluate these with :func:`_ap_sum_line`,
    which follows IRAF's pointer arithmetic or refuses.  Never a line of a
    WARP database (dispersion limits from line 1, ``bmin = 1``: the trace at
    buffer and at image lines agree)."""
    fit = ax1 < ax2
    live = ~empty
    bad = fit & (n_fit < 2)                        # asifit errors (before x2 <= x1)
    bad |= live & ~fit                             # stale interpolant
    bad |= live & ((xsp - 2 + ix1 < 0) | (xsp - 2 + ix2 > na - 1))
    t1 = np.trunc(e1.astype(np.float64)).astype(np.int64)
    t2 = np.trunc(e2.astype(np.float64)).astype(np.int64)
    lo = np.minimum(np.minimum(j1 - 1, t1), np.where(j1 != j2, j2 - 1, j1 - 1))
    hi = np.maximum(np.maximum(j1 + 1, j2 + 1), t2 + 1)
    # c[0] is defined (+0.0, the lower sentinel's upper half: _need) but the
    # vectorised kernel clamps instead of reading it: scalar line from c[0] down
    bad |= live & fit & ((lo < 1) | (hi > n_fit + 1))
    if nx is not None:
        bad |= live & ((ix1 < 1) | (ix2 > nx))
    return bad


def _grl_lin_v(cget, a: NDArray, b: NDArray) -> NDArray[np.float32]:
    """Vectorised linear ``asigrl`` for spans of at most two cells."""
    a = a.astype(R)
    b = b.astype(R)
    swap = a > b
    xa = np.where(swap, b, a)
    xb = np.where(swap, a, b)
    na = xa.astype(np.int64)
    nb = xb.astype(np.int64)
    if np.any(nb - na > 1):
        raise AssertionError("edge integral spans more than two cells")
    half = R(0.5)
    da = (xa - na.astype(R)).astype(R)
    db = (xb - nb.astype(R)).astype(R)
    ca0 = cget(na)
    ca1 = cget(na + 1)
    cb0 = cget(nb)
    cb1 = cget(nb + 1)
    diffa = ((ca1 - ca0).astype(R) * half).astype(R)
    single = fma32(diffa, fma32(db, db, -(da * da).astype(R)),
                   fma32((db - da).astype(R), ca0, R(0)))
    first = fma32(diffa, fma32(-da, da, R(1.0)),
                  fma32((R(1.0) - da).astype(R), ca0, R(0)))
    diffb = ((cb1 - cb0).astype(R) * half).astype(R)
    multi = fma32((diffb * db).astype(R), db, fma32(cb0, db, first))
    acc = np.where(na == nb, single, multi).astype(R)
    return np.where(a < b, acc, -acc).astype(R)


def apsum(image: NDArray, entry: ApertureEntry, *,
          buffer_cols: tuple[int, int] | None = None) -> NDArray[np.float32]:
    """Vectorised (over lines) bit-exact ``apall`` box extraction.

    Same arithmetic as :func:`apsum_scalar`, evaluated for every line at
    once. ``image`` is the rectified strip ``(n_lines, n_cols)``.

    ``buffer_cols`` is the column range ``(amin, amax)`` of IRAF's data buffer
    when it differs from this aperture's own range: ``ap_extract``
    (apextract.x:405-420) reads as many apertures as fit in memory at once,
    so with several apertures in one image the buffer starts at the first
    column needed by *any* of them, and the buffer origin enters the REAL
    arithmetic of ``ap_sum``. Default: the aperture's own range (one aperture
    per image, as on a rectified strip).
    """
    img = np.asarray(image, dtype=R)
    nlines, ncols = img.shape
    (cen_a, cen_b), (low_a, low_b), (high_a, high_b), cv = extract_reals(entry)
    bmin = min(nlines, max(1, _nint(R(cen_b + low_b))))
    bmax = max(1, min(nlines, _nint(R(cen_b + high_b))))
    shifts = ap_cveval(cv, np.arange(bmin, bmax + 1, dtype=R))
    cmin = R(shifts.min())
    cmax = R(shifts.max())
    if buffer_cols is None:
        amin = min(ncols, max(1, _nint(R(R(cen_a + R(low_a - R(2))) + cmin))))
        amax = max(1, min(ncols, _nint(R(R(cen_a + R(high_a + R(2))) + cmax))))
    else:
        amin, amax = buffer_cols
    na = amax - amin + 1
    nb = bmax - bmin + 1
    astart = _nint_v((R(R(cen_a + low_a) - R(2)) + shifts).astype(R))
    # the data buffer as apall reads it (FITS-kernel input mapping, utils/iraf_imio)
    buf = np.ascontiguousarray(imio_read_real(img[bmin - 1: bmax, amin - 1: amax]))
    rc1 = R(amin)
    low = R(cen_a + low_a)
    high = R(cen_a + high_a)
    step = R(R(high - low) / R(1))
    lines = np.arange(1, nb + 1, dtype=R)
    s = ((ap_cveval(cv, lines) - rc1).astype(R) + R(1)).astype(R)
    xsp = astart - amin + 1
    rows = np.arange(nb)
    ax2 = np.minimum(R(R(na) + R(0.49)), (high + s).astype(R)).astype(R)
    n_fit = (_nint_v(ax2) + 1 - (xsp - 1)) + 1

    def cget(j):
        # asi sample j (1-based); j == n_fit + 1 is the linear end pad.
        def at(jj):
            pos = np.clip(xsp - 3 + jj, 0, na - 1)
            return buf[rows, pos]
        last = at(n_fit)
        prev = at(n_fit - 1)
        pad = ((last * R(2.0)).astype(R) - prev).astype(R)
        return np.where(j == n_fit + 1, pad, at(np.minimum(j, n_fit))).astype(R)

    rxs = astart.astype(R)
    x1 = ((np.maximum(R(0.5), (low + s).astype(R)) + rc1).astype(R) - rxs).astype(R)
    x2 = ((np.minimum(R(R(na) + R(0.49)), ((low + step).astype(R) + s).astype(R))
           + rc1).astype(R) - rxs).astype(R)
    empty = x2 <= x1
    ix1 = _nint_v(x1)
    ix2 = _nint_v(x2)
    e1 = (x1 + R(1)).astype(R)
    e2 = (x2 + R(1)).astype(R)
    j1 = _nint_v(e1)
    j2 = _nint_v(e2)
    same = j1 == j2
    wt1 = np.where(same, (e2 - e1).astype(R),
                   ((j1.astype(R) - e1).astype(R) + R(0.5)).astype(R))
    wt2 = np.where(same, R(0), ((e2 - j2.astype(R)).astype(R) + R(0.5)).astype(R))
    half = R(0.5)
    v1 = cget(j1)
    b1 = _grl_lin_v(cget, (j1.astype(R) - half).astype(R), (j1.astype(R) + half).astype(R))
    a1 = _grl_lin_v(cget, e1, np.where(same, e2, (j1.astype(R) + half).astype(R)))
    ok1 = (v1 > 0) & (b1 > 0) & (a1 > 0) & (a1 < b1)
    wt1 = np.where(ok1, (a1 / np.where(b1 == 0, R(1), b1)).astype(R), wt1)
    v2 = cget(j2)
    b2 = _grl_lin_v(cget, (j2.astype(R) - half).astype(R), (j2.astype(R) + half).astype(R))
    a2 = _grl_lin_v(cget, (j2.astype(R) - half).astype(R), e2)
    ok2 = (~same) & (v2 > 0) & (b2 > 0) & (a2 > 0) & (a2 < b2)
    wt2 = np.where(ok2, (a2 / np.where(b2 == 0, R(1), b2)).astype(R), wt2)

    def dat(k):
        return buf[rows, np.clip(xsp - 2 + k, 0, na - 1)]

    sval = fma32(wt1.astype(R), dat(ix1), (wt2 * dat(ix2)).astype(R))
    width = ix2 - ix1 - 1
    for t in range(int(width.max()) if width.size else 0):
        sval = np.where(t < width, (sval + dat(ix1 + 1 + t)).astype(R), sval)
    sval = np.where(empty, R(0), sval).astype(R)
    ax1 = np.maximum(R(0.5), (low + s).astype(R)).astype(R)
    bad = irregular_lines(empty=empty, ax1=ax1, ax2=ax2, n_fit=n_fit, xsp=xsp, ix1=ix1,
                          ix2=ix2, e1=e1, e2=e2, j1=j1, j2=j2, na=na)
    if bad.any():
        flat = buf.reshape(-1)
        for i in np.nonzero(bad)[0]:
            sval[i] = _ap_sum_line(flat, int(i) * na, na, int(xsp[i]), int(astart[i]), s[i],
                                   low, high, step, rc1)
    out = np.zeros(nlines, dtype=R)
    out[bmin - 1: bmax] = sval
    return out
