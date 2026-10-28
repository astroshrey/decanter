"""IRAF ``apall`` background subtraction and ``strip`` extraction, bit-exact.

WARP's parameter-file options

* ``Background Subtraction: average|median|minimum|fit`` run the object
  extraction as ``apall(background=<mode>)`` (``Warp_sci.py:561``,
  ``Spec1Dtools.py:pyapall``) with the aperture's background ICFIT read from
  the per-frame database WARP writes (``write_apdatabase``: the listfile's
  ``bg=`` region as ``sample``, ``chebyshev``, order 1, ``naverage -3``,
  ``niterate 0``; aperture.py:174-182, 201-211);
* ``Extract 2d spectrum: yes`` runs ``apall(format=strip)`` with the same
  background mode (``Warp_sci.py:576``).

This module reproduces, operation for operation as compiled on this Mac
(f2c + clang -O2, arm64; ``apskyeval.o`` has no fused multiply-adds,
``apextract.o``'s ``ap_sum`` fuses ``wt1*d1 + wt2*d2`` as for the data):

* ``ap_extract``'s data buffer with the background limits
  (apextract.x:232-241, 271-283): ``xmin = min(low - 2, ic xmin)``, ``xmax =
  max(high + 2, ic xmax)``, read as IRAF reads the strip file: the FITS
  kernel maps ``-0.0``, subnormals and ``+-Inf``/NaN to ``+0``
  (:func:`decanter.utils.iraf_imio.imio_read_real`) -- the off-order pixels a
  background region far from the aperture reaches hold such values;
* ``ap_skyeval`` (apskyeval.x): the sample regions relative to the aperture
  centre, per line the interpolation window ``[nint(xmin + s) - 1,
  nint(xmax + s) + 1]`` clipped to the buffer, and

  - ``average``: integral of the linear interpolant over each region
    (``asigrl``) divided by the summed widths,
  - ``median``: ``amedr`` (``asok`` selection) of the pixels
    ``nint(a)..nint(b)`` of each region,
  - ``minimum``: the smallest of those pixels,
  - ``fit``: ``icfit`` of the linear interpolant sampled on the pixel grid
    centred on the aperture (``nint(1000 (a - c)) / 1000``): ``rg_xranges``
    sample ranges, ``rg_wtbin`` medians of 3 (``naverage = -3``) with summed
    weights, a REAL Chebyshev ``cvfit`` (order 1 is a weighted mean), no
    rejection (``niterate = 0``).  ``rg_xadd`` reads a token with one number
    ``v`` as ``[min(v, 0), max(v, 0)]`` (decanter.utils.iraf_ranges), so
    ``sample INDEF`` -- what WARP writes without ``bg=`` -- fits every grid
    point at or right of the aperture centre: the whole window of an
    aperture at positive offset (the B nod), nothing (sky 0) for one at
    negative offset;

* ``ap_sum`` with the sky buffer: ``sval - skyval`` with the same edge
  weights (apextract.x:1572-1583);
* ``ap_strip``: the aperture interpolated (``asieval``, linear) so that its
  lower edge is the first pixel, ``int(high - low + 1)`` pixels across, minus
  the sky (apextract.x:1647-1712).

:func:`apsum_background` / :func:`strip_extract` are the scalar reference
(the tests' oracle); :func:`apsum_background_fast` /
:func:`strip_extract_fast` evaluate every line at once with the same
arithmetic (what the pipeline runs).  WARP always writes order 1, so the sky of
every mode is one value per line; other ICFIT settings raise
:class:`NotImplementedError` in the vectorised path and are followed
literally (order 1 only) in the scalar one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from decanter.extract.apsum_exact import (IrafRealBuffer, _ap_sum_line, _LinearAsi, _nint,
                                          _nint_v, extract_reals)
from decanter.io.apdb import ApertureEntry
from decanter.utils.curfit_exact import ap_cveval
from decanter.utils.fpemu import fma32, iraf_ctod
from decanter.utils.iraf_imio import imio_read_real
from decanter.utils.iraf_ranges import ctor, rg_token, rg_tokens

R = np.float32
D = np.float64
MODES = ("average", "median", "minimum", "fit")
_MAX_REAL_SKY = R(9.9e36)        # apskyeval.x: avg = MAX_REAL (f2c "(float)9.9e36")
_NSAMPLE = 20                     # apskyeval.x: NSAMPLE


# ---------------------------------------------------------------------------
# The aperture's background ICFIT (ap_dbread)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ApBackground:
    """An aperture's background fitting parameters as ``ap_dbread`` loads them
    into its ICFIT (apdb.x:225-238: ``dtgetr``/``dtgeti``/``dtgstr``)."""

    xmin: np.float32
    xmax: np.float32
    function: str
    order: int
    sample: str
    naverage: int
    niterate: int
    low: np.float32
    high: np.float32
    grow: np.float32


def background_of(entry: ApertureEntry) -> ApBackground | None:
    """The :class:`ApBackground` of a database entry (None: no ``background``
    section, so ``apall`` subtracts nothing whatever ``background`` says)."""
    if entry.raw_background is None:
        return None
    d = dict(entry.raw_background)
    return ApBackground(
        xmin=R(iraf_ctod(d["xmin"])), xmax=R(iraf_ctod(d["xmax"])),
        function=d.get("function", "chebyshev"), order=int(d.get("order", "1")),
        sample=d.get("sample", "INDEF"), naverage=int(d.get("naverage", "1")),
        niterate=int(d.get("niterate", "0")), low=R(iraf_ctod(d.get("low_reject", "3."))),
        high=R(iraf_ctod(d.get("high_reject", "3."))), grow=R(iraf_ctod(d.get("grow", "0."))))


# ---------------------------------------------------------------------------
# Small IRAF primitives
# ---------------------------------------------------------------------------

def _rmin(a, b):
    """f2c ``min(a, b)``: ``a <= b ? a : b`` (``fcsel ls``)."""
    return a if a <= b else b


def _rmax(a, b):
    """f2c ``max(a, b)``: ``a >= b ? a : b`` (``fcsel ge``)."""
    return a if a >= b else b


def amedr(values) -> np.float32:
    """IRAF ``amedr`` (vops/amed.gx): ``a[1]`` for 1-2 values, the 3-value
    comparison chain, else ``asokr`` (Wirth's selection) of ``(n + 1) / 2``."""
    a = [R(v) for v in values]
    n = len(a)
    if n in (1, 2):
        return a[0]
    if n == 3:
        if a[0] < a[1]:
            if a[1] < a[2]:
                return a[1]
            if a[0] < a[2]:
                return a[2]
            return a[0]
        if a[1] > a[2]:
            return a[1]
        if a[0] < a[2]:
            return a[0]
        return a[2]
    return asokr(a, (n + 1) // 2)


def asokr(a: list, ksel: int) -> np.float32:
    """IRAF ``asokr`` (vops/asok.gx): select the ``ksel``-th smallest, in place."""
    a = list(a)
    npix = len(a)
    lo, up = 1, npix
    k = max(lo, min(up, ksel))
    while lo < up:
        temp = a[k - 1]
        i, j = lo, up
        while True:
            while a[i - 1] < temp:
                i += 1
            while temp < a[j - 1]:
                j -= 1
            if i <= j:
                a[i - 1], a[j - 1] = a[j - 1], a[i - 1]
                i += 1
                j -= 1
            if i > j:
                break
        if j < k:
            lo = i
        if k < i:
            up = j
    return R(a[k - 1])


def asieval_linear(c: NDArray[np.float32], x) -> np.float32:
    """``asieval`` for II_LINEAR (ii_eval.x:ii_linear as compiled):
    ``fma(x - nx, c[nx+1], c[nx] * ((nx+1) - x))``, ``nx = int(x)``;
    ``c`` is 1-based (``c[0]`` unused, ``c[n+1]`` the asifit end pad)."""
    x = R(x)
    nx = int(x)
    return R(fma32(R(x - R(nx)), c[nx + 1], R(c[nx] * R(R(nx + 1) - x))))


def asieval_linear_v(c: NDArray[np.float32], x: NDArray, rows=None) -> NDArray[np.float32]:
    """Vectorised :func:`asieval_linear`; ``c`` is ``(rows, n+2)`` with
    ``rows`` selecting a coefficient row per value (or 1-D ``c``)."""
    x = np.asarray(x, R)
    nx = x.astype(np.int64)
    if rows is None:
        c1, c0 = c[nx + 1], c[nx]
    else:
        c1, c0 = c[rows, nx + 1], c[rows, nx]
    t = (c0 * ((nx + 1).astype(R) - x).astype(R)).astype(R)
    return fma32((x - nx.astype(R)).astype(R), c1, t).astype(R)


# ---------------------------------------------------------------------------
# Sample regions
# ---------------------------------------------------------------------------

def skyeval_regions(sample: str, center: np.float32) -> list[tuple[np.float32, np.float32]]:
    """``ap_skyeval``'s background regions (apskyeval.x:94-121).

    The ``sample`` text with ``:`` replaced by ``$`` is scanned with
    ``ctor`` (:func:`decanter.utils.iraf_ranges.ctor`); every ``a$b`` pair
    gives ``(center + min(a, b), center + max(a, b))`` (REAL), at most 20.
    The scan position follows the SPP loop exactly (``i = i - 1`` after a
    number, ``i = i + 2`` past the ``$``, ``ctor`` moving ``i`` past blanks
    or an out-of-range number even when it returns 0).  An empty list means
    "use ``[xmin, xmax]``" (``sample INDEF``: no ``a$b`` pair).
    """
    s = sample.replace(":", "$")
    out: list[tuple[np.float32, np.float32]] = []
    n = len(s)
    k = 0
    while k < n:
        na, a, q = ctor(s, k)
        if na <= 0:
            k = q + 1
            continue
        if q < n and s[q] == "$":
            nb, b, q2 = ctor(s, q + 1)
            if nb <= 0:
                k = q2 + 1
                continue
            out.append((R(center + _rmin(a, b)), R(center + _rmax(a, b))))
            if len(out) == _NSAMPLE:
                break
            k = q2
            continue
        k = q
    return out


def _rg_parse(sample: str) -> list[tuple[np.float32, np.float32] | None]:
    """The ``(rval1, rval2)`` of every token ``rg_xadd`` adds, in order;
    None for ``*`` (``rvals[1]``, ``rvals[npts]``).  Tokens without a first
    number ("Range syntax error", a warning in IRAF) add nothing.  A token
    with one number ``v`` is ``(v, 0)``, ``INDEF`` is ``(INDEFR, 0)``
    (:func:`decanter.utils.iraf_ranges.rg_token`)."""
    out: list[tuple[np.float32, np.float32] | None] = []
    for tok in rg_tokens(sample):
        t = rg_token(tok)
        if t is None:
            continue
        out.append(None if isinstance(t, str) else t)
    return out


def rg_xranges(sample: str, rvals: NDArray[np.float32]) -> list[tuple[int, int]]:
    """``rg_xrangesr`` + ``rg_order`` + ``rg_merge`` (pkg/xtools/ranges): the
    1-based index ranges of the (increasing) ``rvals`` inside the sample's
    ranges (``*``: all; a lone number ``v``: ``[min(v, 0), max(v, 0)]``, so
    ``INDEF`` selects every x >= 0; :func:`_rg_parse`)."""
    npts = len(rvals)
    if npts < 1:
        raise ValueError("No data points for range determination")
    rgs: list[list[int]] = []
    for tk in _rg_parse(sample):
        if tk is None:
            rv1, rv2 = R(rvals[0]), R(rvals[-1])
        else:
            rv1, rv2 = tk
        a1, b1 = _rmin(rv1, rv2), _rmax(rv1, rv2)
        a2 = _rmin(R(rvals[0]), R(rvals[-1]))
        b2 = _rmax(R(rvals[0]), R(rvals[-1]))
        if b1 >= a2 and a1 <= b2:
            a1 = _rmax(a2, _rmin(b2, a1))
            b1 = _rmax(a2, _rmin(b2, b1))
            if rvals[0] <= rvals[-1]:
                k = 1
                while k <= npts and rvals[k - 1] < a1:
                    k += 1
                j = k
                while j <= npts and rvals[j - 1] <= b1:
                    j += 1
                j -= 1
            else:
                k = 1
                while k <= npts and rvals[k - 1] > b1:
                    k += 1
                j = k
                while j <= npts and rvals[j - 1] >= a1:
                    j += 1
                j -= 1
            if k <= j:
                rgs.append([k, j])
    for r in rgs:                                   # rg_order
        if r[0] > r[1]:
            r[0], r[1] = r[1], r[0]
    for i in range(len(rgs) - 1):
        for j in range(i + 1, len(rgs)):
            if rgs[i][0] > rgs[j][0]:
                rgs[i], rgs[j] = rgs[j], rgs[i]
    if not rgs:                                     # rg_merge
        return []
    merged = [list(rgs[0])]
    for r in rgs[1:]:
        if r[0] > merged[-1][1] + 1:
            merged.append(list(r))
        else:
            merged[-1][1] = max(r[1], merged[-1][1])
    return [(a, b) for a, b in merged]


def rg_wtbin(rgs, nbin: int, vals, wts) -> tuple[list, list]:
    """``rg_wtbinr`` (rgwtbin.gx) over increasing ranges: subranges of at most
    ``|nbin|`` points (a short trailing one dropped unless it is the range's
    first), weighted means (``nbin > 1``) or ``amedr`` medians (``nbin < 0``);
    output weights are the summed weights."""
    out, wout = [], []
    if abs(nbin) < 2:
        for a, b in rgs:
            out += [R(v) for v in vals[a - 1:b]]
            wout += [R(w) for w in wts[a - 1:b]]
        return out, wout
    npts = abs(nbin)
    nmin = max(min(npts, 3), (npts + 1) // 2)
    nin = len(vals)
    for a, b in rgs:
        nsample = 0
        j = max(1, a)
        k = min(nin, b)
        while j <= k:
            n = max(0, min(npts, k - j + 1))
            if nsample > 0 and n < nmin:
                break
            nsample += 1
            w = R(0)
            for wt in wts[j - 1:j - 1 + n]:            # asumr
                w = R(w + R(wt))
            if nbin > 0:
                if w != 0:
                    o = R(0)
                    for v, wt in zip(vals[j - 1:j - 1 + n], wts[j - 1:j - 1 + n],
                                     strict=True):
                        o = fma32(R(v), R(wt), o)
                    o = R(o / w)
                else:
                    o = R(0)
                    for v in vals[j - 1:j - 1 + n]:
                        o = R(o + R(v))
                    o = R(o / R(n))
            else:
                o = amedr(vals[j - 1:j - 1 + n])
            out.append(o)
            wout.append(w)
            j += n
    return out, wout


def icfit_constant(sample: str, x, y, naverage: int) -> np.float32 | None:
    """``ic_fitr`` (newx = newy = newwts = newfunction = YES, unit weights)
    of a Chebyshev/Legendre of order 1 as ``ap_skyeval`` calls it, with
    ``niterate = 0`` (``ic_rejectr`` returns at once).

    Returns the curve's constant (what ``cveval`` gives at any x), or None
    when ``ic_dosetup`` raises "No sample points for fit".  The REAL
    ``cvacpts`` accumulation is ``V = fma(w, y, V)``, ``M = fma(w, 1, M)``;
    ``cvchofac``/``cvchoslv`` of the 1x1 system give ``V * (1 / M)``.
    """
    npts = len(x)
    if npts == 0:
        return None
    w = [R(1.0)] * npts
    rgs = rg_xranges(sample, np.asarray(x, R))
    xfit, wfit = rg_wtbin(rgs, max(1, abs(naverage)), list(x), w)
    nfit = len(xfit)
    if nfit == 0:
        return None
    if nfit == npts:
        yfit, wfit = [R(v) for v in y], w
    else:
        yfit, wfit = rg_wtbin(rgs, naverage, list(y), w)
    V = R(0)
    M = R(0)
    for yv, wv in zip(yfit, wfit, strict=True):
        bw = R(wv * R(1.0))                 # amulr(w, basis)
        V = fma32(bw, yv, V)
        M = fma32(bw, R(1.0), M)
    if not M > 0:
        return R(0)                         # matfac stays 0 (calloc'd)
    return R(V * R(R(1.0) / M))


# ---------------------------------------------------------------------------
# ap_extract's buffer, ap_skyeval, ap_sum with sky (scalar reference)
# ---------------------------------------------------------------------------

@dataclass
class _Geom:
    bmin: int
    bmax: int
    amin: int
    amax: int
    na: int
    nb: int
    na1: int                       # sky / profile buffer width (nx)
    astart: NDArray[np.int64]      # xs[iy]: image column of sbuf column 1
    cen: np.float32
    low: np.float32                # AP_LOW / AP_HIGH on the aperture axis
    high: np.float32
    cv: object


def extract_geometry(entry: ApertureEntry, bg: ApBackground | None, nlines: int, ncols: int,
                     *, background: bool = True) -> _Geom:
    """``ap_extract``'s line range, data buffer columns and per-line
    ``astart`` for one aperture (apextract.x:232-241, 271-283, 433-443; weights none,
    linear edge interpolator so ``interpbuf = 2``).  With a background
    (``background`` and an ICFIT), the buffer also spans the ICFIT's
    ``xmin``/``xmax``."""
    (cen_a, cen_b), (low_a, low_b), (high_a, high_b), cv = extract_reals(entry)
    bmin = min(nlines, max(1, _nint(R(cen_b + low_b))))
    bmax = max(1, min(nlines, _nint(R(cen_b + high_b))))
    shifts = ap_cveval(cv, np.arange(bmin, bmax + 1, dtype=R))
    cmin, cmax = R(shifts.min()), R(shifts.max())
    xmin = R(low_a - R(2))
    xmax = R(high_a + R(2))
    if background and bg is not None:
        xmin = _rmin(xmin, bg.xmin)
        xmax = _rmax(xmax, bg.xmax)
    amin = min(ncols, max(1, _nint(R(R(cen_a + xmin) + cmin))))
    amax = max(1, min(ncols, _nint(R(R(cen_a + xmax) + cmax))))
    # W_NONE branch: xmin/xmax of the aperture only (the sky buffer width)
    x1 = R(R(R(cen_a + low_a)) - R(2))
    x2 = R(R(R(cen_a + high_a)) + R(2))
    na1 = _nint(x2) - _nint(x1) + 1
    astart = _nint_v((x1 + shifts).astype(R))
    return _Geom(bmin=bmin, bmax=bmax, amin=amin, amax=amax, na=amax - amin + 1,
                 nb=bmax - bmin + 1, na1=na1, astart=astart, cen=R(cen_a), low=R(low_a),
                 high=R(high_a), cv=cv)


def _cveval1(cv, y: int) -> np.float32:
    return R(ap_cveval(cv, np.array([y], dtype=R))[0])


def skyeval_line(buf: NDArray[np.float32], g: _Geom, bg: ApBackground, mode: str,
                 regions, xmin: np.float32, xmax: np.float32, iy: int) -> np.float32:
    """``ap_skyeval`` for buffer line ``iy`` (1-based): the sky value of every
    pixel of the line (WARP's order-1 ICFIT makes it one value)."""
    c1 = g.amin
    nc = g.na
    s = _cveval1(g.cv, iy)
    ix1 = max(c1, _nint(R(xmin + s)) - 1)
    ix2 = min(c1 + nc - 1, _nint(R(xmax + s)) + 1)
    nfit = ix2 - ix1 + 1
    if nfit < 3:
        return R(0)
    data = buf[iy - 1, ix1 - c1: ix1 - c1 + nfit]
    asi = _LinearAsi(data) if mode in ("average", "fit") else None
    rix1, rix2 = R(ix1), R(ix2)

    def ab(k):
        a = R(R(_rmax(rix1, R(regions[k][0] + s)) - rix1) + R(1))
        b = R(R(_rmin(rix2, R(regions[k][1] + s)) - rix1) + R(1))
        return a, b

    if mode == "average":
        avg = R(0)
        c = R(0)
        for k in range(len(regions)):
            a, b = ab(k)
            if R(b - a) > 0:
                avg = R(avg + asi.grl(a, b))
                c = R(R(c + b) - a)
        if c > 0:
            avg = R(avg / c)
        return avg
    if mode == "median":
        vals = []
        for k in range(len(regions)):
            a, b = ab(k)
            for j in range(_nint(a), _nint(b) + 1):
                vals.append(data[j - 1])
        return amedr(vals) if vals else R(0)
    if mode == "minimum":
        avg = _MAX_REAL_SKY
        nsky = 0
        for k in range(len(regions)):
            a, b = ab(k)
            for j in range(_nint(a), _nint(b) + 1):
                avg = _rmin(avg, R(data[j - 1]))
                nsky += 1
        return avg if nsky else R(0)
    if mode == "fit":
        if bg.order != 1 or bg.function not in ("chebyshev", "legendre") or bg.niterate != 0:
            raise NotImplementedError(
                f"background fit with {bg.function} order {bg.order}, niterate "
                f"{bg.niterate} (WARP writes chebyshev order 1, niterate 0)")
        c = R(g.cen + s)
        a = R(R(c + rix1) - R(int(c)))
        xs, ys = [], []
        for _ in range(nfit - 1):
            xs.append(R(R(_nint(R(R(a - c) * R(1000.0)))) / R(1000.0)))
            ys.append(asieval_linear(asi.c, R(R(a - rix1) + R(1))))
            a = R(a + R(1))
        val = icfit_constant(bg.sample, xs, ys, bg.naverage)
        return R(0) if val is None else val
    raise ValueError(f"background mode {mode!r}")


def sky_values(image: NDArray, entry: ApertureEntry, mode: str,
               bg: ApBackground | None = None) -> tuple[NDArray[np.float32], _Geom]:
    """The per-line sky (``ap_skyeval``) of one aperture on a rectified strip:
    ``(sky[nb], geometry)``; zeros when the entry has no background ICFIT."""
    img = np.asarray(image, dtype=R)
    nlines, ncols = img.shape
    if bg is None:
        bg = background_of(entry)
    g = extract_geometry(entry, bg, nlines, ncols, background=True)
    sky = np.zeros(g.nb, dtype=R)
    if bg is None:
        return sky, g
    buf = imio_read_real(img[g.bmin - 1: g.bmax, g.amin - 1: g.amax])
    center = g.cen
    xmin = R(center + _rmin(g.low, bg.xmin))
    xmax = R(center + _rmax(g.high, bg.xmax))
    regions = skyeval_regions(bg.sample, center) or [(xmin, xmax)]
    for iy in range(1, g.nb + 1):
        sky[iy - 1] = skyeval_line(buf, g, bg, mode, regions, xmin, xmax, iy)
    return sky, g


def apsum_background(image: NDArray, entry: ApertureEntry, mode: str,
                     sky: NDArray[np.float32] | None = None) -> NDArray[np.float32]:
    """``apall(background=mode, format=onedspec)`` of one aperture: the box sum
    minus the sky over the same pixels with the same edge weights
    (apextract.x:ap_sum), REAL.  Scalar reference.

    With ``mode == "none"`` or an entry without a background section this is
    :func:`decanter.extract.apsum_exact.apsum`.  ``sky`` may pass the lines'
    sky (:func:`sky_values`) already computed.
    """
    from decanter.extract.apsum_exact import apsum
    bg = background_of(entry)
    if mode == "none" or bg is None:
        return apsum(image, entry)
    img = np.asarray(image, dtype=R)
    nlines, ncols = img.shape
    if sky is None:
        sky, g = sky_values(img, entry, mode, bg)
    else:
        g = extract_geometry(entry, bg, nlines, ncols, background=True)
    buf = imio_read_real(img[g.bmin - 1: g.bmax, g.amin - 1: g.amax])
    flat = buf.reshape(-1)
    c1 = g.amin
    na = g.na
    low = R(g.cen + g.low)
    high = R(g.cen + g.high)
    step = R(R(high - low) / R(1))
    rc1 = R(c1)
    # ap_skyeval fills sbuf (na1 values per line, ap_extract's W_NONE width)
    # with the line's sky (amovkr; aclrr where there is none; the order-1
    # fit's constant); ap_sum reads it at Memr[sbuf + (iy - 1) na1 - 1 + ix]
    sky_buf = IrafRealBuffer(np.repeat(np.asarray(sky, dtype=R), g.na1), "sky buffer")
    out = np.zeros(nlines, dtype=R)
    for iy in range(1, g.nb + 1):
        s = R(R(_cveval1(g.cv, iy) - rc1) + R(1))
        xs = int(g.astart[iy - 1])
        out[g.bmin - 1 + iy - 1] = _ap_sum_line(
            flat, (iy - 1) * na, na, xs - c1 + 1, xs, s, low, high, step, rc1,
            sky=sky_buf, sky0=(iy - 1) * g.na1)
    return out


# ---------------------------------------------------------------------------
# apall format=strip
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StripResult:
    """``apall format=strip`` output: ``data`` is ``(na, ny)`` (spatial pixel
    by dispersion line, the FITS ``NAXIS2 x NAXIS1``), ``low``/``high`` the REAL
    aperture limits (``APNUM1`` and the strip origin)."""

    data: NDArray[np.float32]
    low: np.float32
    high: np.float32


def strip_limits(entry: ApertureEntry) -> tuple[np.float32, np.float32, int]:
    """``ap_output`` STRIP (apextract.x:1309-1344, nsubaps = 1): ``low = cen +
    low``, ``step = (high - low) / 1``, ``low = low + 0 * step``, ``high = low
    + step``; the strip is ``int(high - low + 1)`` pixels across (REAL ->
    integer truncation)."""
    (cen_a, _), (low_a, _), (high_a, _), _ = extract_reals(entry)
    low = R(cen_a + low_a)
    high = R(cen_a + high_a)
    step = R(R(high - low) / R(1))
    low = R(low + R(R(0) * step))
    high = R(low + step)
    return low, high, int(R(R(high - low) + R(1)))


def strip_extract(image: NDArray, entry: ApertureEntry, mode: str = "none",
                  sky: NDArray[np.float32] | None = None) -> StripResult:
    """``apall(format=strip, background=mode)`` of one aperture (scalar
    reference): per line a linear ``asifit`` of the columns ``max(1,
    nint(low + s) - 1) .. min(nc, nint(high + s) + 1)`` of the data buffer,
    evaluated at ``x = low + s - ix1 + 1, x + 1, ...`` while ``1 <= x <=
    nasi`` (0 elsewhere), minus the sky (apextract.x:ap_strip)."""
    img = np.asarray(image, dtype=R)
    nlines, ncols = img.shape
    bg = background_of(entry)
    use_bg = mode != "none" and bg is not None
    if use_bg and sky is None:
        sky, g = sky_values(img, entry, mode, bg)
    elif use_bg:
        g = extract_geometry(entry, bg, nlines, ncols, background=True)
    else:
        g = extract_geometry(entry, bg, nlines, ncols, background=False)
        sky = None
    aplow, aphigh, na_out = strip_limits(entry)
    ap_sum_asifit_check(g)
    c1 = g.amin
    nc = g.na
    low = R(R(aplow - R(c1)) + R(1))
    high = R(R(aphigh - R(c1)) + R(1))
    buf = imio_read_real(img[g.bmin - 1: g.bmax, g.amin - 1: g.amax])
    ny = g.nb
    sky_buf = (IrafRealBuffer(np.repeat(np.asarray(sky, dtype=R), g.na1), "sky buffer")
               if use_bg else None)
    out = np.zeros((na_out, ny), dtype=R)
    for iy in range(1, ny + 1):
        s = _cveval1(g.cv, iy)
        out[:, iy - 1] = _strip_line(buf[iy - 1], nc, low, high, s, na_out, sky_buf,
                                     _strip_sky_offset(g, iy, low, s))
    return StripResult(data=out, low=aplow, high=aphigh)


def ap_sum_asifit_check(g: _Geom) -> None:
    """``ap_extract`` runs ``ap_sum`` (weights none) before writing any format
    (apextract.x:496-498), and ``ap_sum``'s ``ap_asifit`` errors on a line
    with fewer than 2 samples ("ASIFIT: too few points for LINEAR",
    apextract.x:1812-1816): apall then writes nothing, ``format=strip``
    included.  Only a database whose dispersion limits start after line 1
    (``bmin > 1``) gets there.

    Raises:
        ValueError: when a line of the aperture has ``n < 2``.
    """
    lines = np.arange(1, g.nb + 1, dtype=R)
    rc1 = R(g.amin)
    s = ((ap_cveval(g.cv, lines) - rc1).astype(R) + R(1)).astype(R)
    low = R(g.cen + g.low)
    high = R(g.cen + g.high)
    ax1 = np.maximum(R(0.5), (low + s).astype(R)).astype(R)
    ax2 = np.minimum(R(R(g.na) + R(0.49)), (high + s).astype(R)).astype(R)
    xsp = g.astart - g.amin + 1
    n_fit = (_nint_v(ax2) + 1 - (xsp - 1)) + 1
    bad = (ax1 < ax2) & (n_fit < 2)
    if bad.any():
        raise ValueError(
            "ASIFIT: too few points for LINEAR: IRAF's ap_asifit fits "
            f"{int(n_fit[bad][0])} sample(s) on {int(bad.sum())} line(s) (apextract.x:"
            "1812-1816); apall stops here (the trace at buffer lines and at image "
            "lines disagree: bmin > 1)")


def _strip_sky_offset(g: _Geom, iy: int, low: np.float32, s: np.float32) -> int:
    """``ap_strip``'s sky pointer ``sky = sbuf + (iy - 1) nx + nint(low + s) -
    xs[iy] + c1 - 2`` (apextract.x:1697) as an offset into the flat sky
    buffer; output pixel ``i`` (1-based) reads ``Memr[sky + i]``."""
    return (iy - 1) * g.na1 + _nint(R(low + s)) - int(g.astart[iy - 1]) + g.amin - 2


def _strip_line(row: NDArray[np.float32], nc: int, low: np.float32, high: np.float32,
                s: np.float32, na_out: int, sky_buf: IrafRealBuffer | None,
                soff: int) -> NDArray[np.float32]:
    """One line of ``ap_strip`` (apextract.x:1675-1704): a linear ``asifit`` of
    the columns ``max(1, nint(low + s) - 1) .. min(nc, nint(high + s) + 1)``
    evaluated at ``x = low + s - ix1 + 1, x + 1, ...`` while ``1 <= x <= nasi``
    (0 elsewhere), minus ``Memr[sky + i]`` (``sky_buf``, the malloc'ed sky
    buffer, from ``soff``: :func:`_strip_sky_offset`) -- a read that leaves
    the line only when the trace at buffer and at image lines disagree
    (``bmin > 1``)."""
    col = np.zeros(na_out, dtype=R)
    ix1 = max(1, _nint(R(low + s)) - 1)
    ix2 = min(nc, _nint(R(high + s)) + 1)
    nasi = ix2 - ix1 + 1
    if nasi < 3:
        return col
    asi = _LinearAsi(row[ix1 - 1: ix1 - 1 + nasi])
    x = R(R(R(low + s) - R(ix1)) + R(1))
    for i in range(na_out):
        if x >= R(1) and x <= R(nasi):
            v = asieval_linear(asi.c, x)
            if sky_buf is not None:
                v = R(v - sky_buf.at(soff + i + 1))
            col[i] = v
        x = R(x + R(1))
    return col


# ---------------------------------------------------------------------------
# Vectorised (every mode)
# ---------------------------------------------------------------------------

def _lin_coeffs(rows: NDArray[np.float32], n: NDArray[np.int64], width: int) -> NDArray[np.float32]:
    """asifit(II_LINEAR) of each row's first ``n`` values: ``(L, width + 2)``,
    1-based (``c[:, 0]`` unused), the end pad ``2 c[n] - c[n-1]`` at ``n + 1``."""
    L = rows.shape[0]
    c = np.zeros((L, width + 2), dtype=R)
    c[:, 1:rows.shape[1] + 1] = rows
    li = np.arange(L)
    last = c[li, n]
    prev = c[li, n - 1]
    c[li, n + 1] = ((last * R(2.0)).astype(R) - prev).astype(R)
    return c


def _grl_v(c: NDArray[np.float32], a: NDArray, b: NDArray) -> NDArray[np.float32]:
    """``asigrl`` (II_LINEAR) of each row of ``c`` from ``a`` to ``b``
    (a < b), the arithmetic of :meth:`_LinearAsi.grl` (any span)."""
    a = a.astype(R)
    b = b.astype(R)
    na = a.astype(np.int64)
    nb = b.astype(np.int64)
    li = np.arange(c.shape[0])
    half = R(0.5)
    da = (a - na.astype(R)).astype(R)
    db = (b - nb.astype(R)).astype(R)
    ca0 = c[li, na]
    ca1 = c[li, na + 1]
    diffa = ((ca1 - ca0).astype(R) * half).astype(R)
    single = fma32(diffa, fma32(db, db, -(da * da).astype(R)),
                   fma32((db - da).astype(R), ca0, R(0)))
    acc = fma32(diffa, fma32(-da, da, R(1.0)), fma32((R(1.0) - da).astype(R), ca0, R(0)))
    span = nb - na - 1
    for t in range(int(span.max()) if span.size else 0):
        j = na + 1 + t
        jj = np.minimum(j, c.shape[1] - 2)
        step = fma32((c[li, jj + 1] + c[li, jj]).astype(R), half, acc)
        acc = np.where(t < span, step, acc).astype(R)
    nbc = np.minimum(nb, c.shape[1] - 2)
    cb0 = c[li, nbc]
    cb1 = c[li, nbc + 1]
    diffb = ((cb1 - cb0).astype(R) * half).astype(R)
    multi = fma32((diffb * db).astype(R), db, fma32(cb0, db, acc))
    return np.where(na == nb, single, multi).astype(R)


def sky_values_fast(image: NDArray, entry: ApertureEntry, mode: str,
                    bg: ApBackground | None = None) -> tuple[NDArray[np.float32], _Geom]:
    """:func:`sky_values` with every line at once (same arithmetic).

    Lines whose median/minimum sample holds a NaN or a zero (where the
    selection order decides between equal keys) and every ``fit`` line are
    evaluated by the scalar :func:`skyeval_line`.
    """
    img = np.asarray(image, dtype=R)
    nlines, ncols = img.shape
    if bg is None:
        bg = background_of(entry)
    g = extract_geometry(entry, bg, nlines, ncols, background=True)
    if bg is None:
        return np.zeros(g.nb, dtype=R), g
    buf = imio_read_real(img[g.bmin - 1: g.bmax, g.amin - 1: g.amax])
    center = g.cen
    xmin = R(center + _rmin(g.low, bg.xmin))
    xmax = R(center + _rmax(g.high, bg.xmax))
    regions = skyeval_regions(bg.sample, center) or [(xmin, xmax)]
    c1, nc = g.amin, g.na
    lines = np.arange(1, g.nb + 1, dtype=R)
    s = ap_cveval(g.cv, lines).astype(R)
    ix1 = np.maximum(c1, _nint_v((xmin + s).astype(R)) - 1)
    ix2 = np.minimum(c1 + nc - 1, _nint_v((xmax + s).astype(R)) + 1)
    nfit = ix2 - ix1 + 1
    ok = nfit >= 3
    sky = np.zeros(g.nb, dtype=R)
    if not ok.any():
        # no line has the 3 points ap_skyeval needs: every sky is 0 (aclrr),
        # as in the scalar port (and nothing below has a positive width)
        return sky, g
    li = np.arange(g.nb)
    wmax = int(nfit.max()) if nfit.size else 0
    cols = (ix1 - c1)[:, None] + np.arange(wmax)[None, :]
    data = buf[li[:, None], np.clip(cols, 0, nc - 1)]          # (nb, wmax)
    rix1 = ix1.astype(R)
    rix2 = ix2.astype(R)
    ab = []
    for lo_r, hi_r in regions:
        ta = (lo_r + s).astype(R)
        a = ((np.where(rix1 >= ta, rix1, ta) - rix1).astype(R) + R(1)).astype(R)
        tb = (hi_r + s).astype(R)
        b = ((np.where(rix2 <= tb, rix2, tb) - rix1).astype(R) + R(1)).astype(R)
        ab.append((a, b))
    if mode == "average":
        c = _lin_coeffs(data, np.maximum(nfit, 2), wmax)
        avg = np.zeros(g.nb, dtype=R)
        cw = np.zeros(g.nb, dtype=R)
        for a, b in ab:
            pos = (b - a).astype(R) > 0
            aa = np.where(pos, a, R(1))
            bb = np.where(pos, b, R(1.5))
            gr = _grl_v(c, aa, bb)
            avg = np.where(pos, (avg + gr).astype(R), avg)
            cw = np.where(pos, ((cw + b).astype(R) - a).astype(R), cw)
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = np.where(cw > 0, (avg / np.where(cw > 0, cw, R(1))).astype(R), avg)
        sky = np.where(ok, avg, R(0)).astype(R)
        return sky, g
    if mode in ("median", "minimum"):
        sel = np.zeros((g.nb, wmax), dtype=bool)
        cnt = np.zeros(g.nb, dtype=np.int64)
        idx_lists = []
        for a, b in ab:
            ja, jb = _nint_v(a), _nint_v(b)
            idx_lists.append((ja, jb))
            jj = np.arange(1, wmax + 1)[None, :]
            sel |= (jj >= ja[:, None]) & (jj <= jb[:, None])
            cnt += np.maximum(jb - ja + 1, 0)
        overlap = cnt != sel.sum(axis=1)
        if mode == "minimum":
            v = np.nanmin(np.where(sel, data, np.inf), axis=1).astype(R)
            sky = np.where(cnt > 0, v, R(0)).astype(R)
        else:
            k = (cnt + 1) // 2
            srt = np.sort(np.where(sel, data, np.inf), axis=1)
            v = srt[li, np.clip(k - 1, 0, wmax - 1)].astype(R)
            few = cnt <= 2
            first = np.zeros(g.nb, dtype=R)
            for ja, jb in reversed(idx_lists):
                has = jb >= ja
                first = np.where(has, data[li, np.clip(ja - 1, 0, wmax - 1)], first)
            v = np.where(few, first, v)
            sky = np.where(cnt > 0, v, R(0)).astype(R)
        # where the value of a selection can depend on IRAF's comparison
        # order (signed zeros, NaN, infinities, the 9.9e36 start of the
        # minimum, pixels counted twice by overlapping regions): scalar
        selv = np.where(sel, data, R(1))
        risky = overlap | (~np.isfinite(selv)).any(axis=1) | (selv == 0).any(axis=1) \
            | (np.abs(selv) >= _MAX_REAL_SKY).any(axis=1)
        sky = np.where(ok, sky, R(0)).astype(R)
        for iy in np.nonzero(risky & ok)[0]:
            sky[iy] = skyeval_line(buf, g, bg, mode, regions, xmin, xmax, int(iy) + 1)
        return sky, g
    if mode == "fit":
        if bg.order != 1 or bg.function not in ("chebyshev", "legendre") or bg.niterate != 0:
            raise NotImplementedError(
                f"background fit with {bg.function} order {bg.order}, niterate "
                f"{bg.niterate} (WARP writes chebyshev order 1, niterate 0)")
        c = _lin_coeffs(data, np.maximum(nfit, 2), wmax)
        sky = _fit_sky_v(c, s, g.cen, ix1, nfit, bg.sample, bg.naverage)
        return np.where(ok, sky, R(0)).astype(R), g
    raise ValueError(f"background mode {mode!r}")


def _med3_v(a0, a1, a2) -> NDArray[np.float32]:
    """``amedr`` of three values (vops/amed.gx), the same comparison chain."""
    return np.where(a0 < a1,
                    np.where(a1 < a2, a1, np.where(a0 < a2, a2, a0)),
                    np.where(a1 > a2, a1, np.where(a0 < a2, a0, a2))).astype(R)


def _fit_sky_v(c: NDArray[np.float32], s: NDArray[np.float32], cen: np.float32,
               ix1: NDArray[np.int64], nfit: NDArray[np.int64], sample: str,
               naverage: int) -> NDArray[np.float32]:
    """The ``fit`` background of every line at once: :func:`skyeval_line`'s
    sampling (``x = nint(1000 (a - c)) / 1000``, ``y = asieval``) and
    :func:`icfit_constant` (``rg_xranges``, ``rg_order``/``rg_merge``,
    ``rg_wtbin`` medians, the REAL order-1 ``cvfit``), vectorised over lines."""
    L = s.size
    li = np.arange(L)
    npts = nfit - 1
    pmax = int(npts.max()) if L else 0
    cc = (cen + s).astype(R)
    rix1 = ix1.astype(R)
    a = ((cc + rix1).astype(R) - np.trunc(cc).astype(R)).astype(R)
    X = np.zeros((L, pmax), dtype=R)
    Y = np.zeros((L, pmax), dtype=R)
    for i in range(pmax):
        t = ((a - cc).astype(R) * R(1000.0)).astype(R)
        X[:, i] = (_nint_v(t).astype(R) / R(1000.0)).astype(R)
        xe = ((a - rix1).astype(R) + R(1)).astype(R)
        valid = i < npts
        Y[:, i] = asieval_linear_v(c, np.where(valid, xe, R(1)), li)
        a = (a + R(1)).astype(R)
    col = np.arange(pmax)[None, :]
    vmask = col < npts[:, None]
    x0 = X[:, 0]
    xn = X[li, np.maximum(npts - 1, 0)]
    a2 = np.where(x0 <= xn, x0, xn)
    b2 = np.where(x0 >= xn, x0, xn)
    # rg_xranges: (k, j) per sample token, 1-based, k <= j when added
    toks = _rg_parse(sample)
    if np.any(xn < x0):
        raise AssertionError("sampled x not increasing")
    ks, js, oks = [], [], []
    for tk in toks:
        if tk is None:
            rv1, rv2 = x0, xn
        else:
            rv1 = np.full(L, tk[0], R)
            rv2 = np.full(L, tk[1], R)
        a1 = np.where(rv1 <= rv2, rv1, rv2)
        b1 = np.where(rv1 >= rv2, rv1, rv2)
        hit = (b1 >= a2) & (a1 <= b2)
        a1c = np.maximum(a2, np.where(b2 <= a1, b2, a1))
        b1c = np.maximum(a2, np.where(b2 <= b1, b2, b1))
        k = 1 + np.sum(vmask & (X < a1c[:, None]), axis=1)
        j = np.sum(vmask & (X <= b1c[:, None]), axis=1)
        ks.append(k)
        js.append(j)
        oks.append(hit & (k <= j))
    nr = len(toks)
    if nr == 0:
        return np.zeros(L, dtype=R)
    K = np.array(ks).T                     # (L, nr)
    J = np.array(js).T
    OK = np.array(oks).T
    # rg_order (by start; invalid ranges last) and rg_merge
    key = np.where(OK, K, np.iinfo(np.int64).max)
    order = np.argsort(key, axis=1, kind="stable")
    K = np.take_along_axis(K, order, 1)
    J = np.take_along_axis(J, order, 1)
    OK = np.take_along_axis(OK, order, 1)
    MK = np.zeros((L, nr), np.int64)
    MJ = np.zeros((L, nr), np.int64)
    MV = np.zeros((L, nr), bool)
    cnt = np.zeros(L, np.int64)
    for r in range(nr):
        k, j, v = K[:, r], J[:, r], OK[:, r]
        prevj = MJ[li, np.maximum(cnt - 1, 0)]
        new = v & ((cnt == 0) | (k > prevj + 1))
        ext = v & ~new
        MK[li[new], cnt[new]] = k[new]
        MJ[li[new], cnt[new]] = j[new]
        MV[li[new], cnt[new]] = True
        cnt = cnt + new
        idx = np.maximum(cnt - 1, 0)
        MJ[li[ext], idx[ext]] = np.maximum(MJ[li[ext], idx[ext]], j[ext])
    # rg_wtbin bins (|naverage| points, a short one only as a range's first)
    nbin = abs(naverage)
    if nbin < 2:
        raise NotImplementedError("background fit with |naverage| < 2")
    nmin = max(min(nbin, 3), (nbin + 1) // 2)
    V = np.zeros(L, dtype=R)
    M = np.zeros(L, dtype=R)
    nb_tot = np.zeros(L, np.int64)
    tmax = int(pmax // 1) + 1
    for r in range(nr):
        k, j, v = MK[:, r], MJ[:, r], MV[:, r]
        nrng = np.where(v, j - k + 1, 0)
        for t in range(tmax):
            start = k + nbin * t
            n = np.clip(j - start + 1, 0, nbin)
            take = v & (start <= j) & ((t == 0) | (n >= nmin))
            if not take.any():
                if t > 0:
                    break
                continue
            if naverage < 0:
                if nbin == 3:
                    s0 = np.clip(start - 1, 0, max(pmax - 1, 0))
                    y0 = Y[li, s0]
                    y1 = Y[li, np.clip(s0 + 1, 0, max(pmax - 1, 0))]
                    y2 = Y[li, np.clip(s0 + 2, 0, max(pmax - 1, 0))]
                    med = np.where(n >= 3, _med3_v(y0, y1, y2), y0).astype(R)
                else:
                    raise NotImplementedError("background fit naverage other than -3")
            else:
                raise NotImplementedError("background fit with naverage > 0")
            w = n.astype(R)
            V = np.where(take, fma32(w, med, V), V).astype(R)
            M = np.where(take, fma32(w, R(1.0), M), M).astype(R)
            nb_tot = nb_tot + take
        del nrng
    with np.errstate(divide="ignore"):
        out = (V * (R(1.0) / np.where(M > 0, M, R(1.0))).astype(R)).astype(R)
    return np.where(nb_tot > 0, out, R(0)).astype(R)


def strip_extract_fast(image: NDArray, entry: ApertureEntry, mode: str = "none",
                       sky: NDArray[np.float32] | None = None) -> StripResult:
    """:func:`strip_extract` with every line at once (same arithmetic).
    ``sky`` may pass the lines' sky values already computed."""
    img = np.asarray(image, dtype=R)
    nlines, ncols = img.shape
    bg = background_of(entry)
    use_bg = mode != "none" and bg is not None
    if use_bg:
        if sky is None:
            sky, g = sky_values_fast(img, entry, mode, bg)
        else:
            g = extract_geometry(entry, bg, nlines, ncols, background=True)
    else:
        g = extract_geometry(entry, bg, nlines, ncols, background=False)
    aplow, aphigh, na_out = strip_limits(entry)
    ap_sum_asifit_check(g)
    c1, nc = g.amin, g.na
    low = R(R(aplow - R(c1)) + R(1))
    high = R(R(aphigh - R(c1)) + R(1))
    buf = imio_read_real(img[g.bmin - 1: g.bmax, g.amin - 1: g.amax])
    ny = g.nb
    lines = np.arange(1, ny + 1, dtype=R)
    s = ap_cveval(g.cv, lines).astype(R)
    ix1 = np.maximum(1, _nint_v((low + s).astype(R)) - 1)
    ix2 = np.minimum(nc, _nint_v((high + s).astype(R)) + 1)
    nasi = ix2 - ix1 + 1
    ok = nasi >= 3
    out = np.zeros((na_out, ny), dtype=R)
    if not ok.any():
        # no line has the 3 points ap_strip fits: all zero (as the scalar port)
        return StripResult(data=out, low=aplow, high=aphigh)
    li = np.arange(ny)
    wmax = int(nasi.max()) if ny else 0
    cols = (ix1 - 1)[:, None] + np.arange(wmax)[None, :]
    rows = buf[li[:, None], np.clip(cols, 0, nc - 1)]
    c = _lin_coeffs(rows, np.maximum(nasi, 2), wmax)
    x = (((low + s).astype(R) - ix1.astype(R)).astype(R) + R(1)).astype(R)
    fn = nasi.astype(R)
    for i in range(na_out):
        inside = ok & (x >= R(1)) & (x <= fn)
        xe = np.where(inside, x, R(1))
        v = asieval_linear_v(c, xe, li)
        if use_bg:
            v = (v - sky).astype(R)
        out[i] = np.where(inside, v, R(0))
        x = (x + R(1)).astype(R)
    if use_bg:
        # ap_strip reads the sky at Memr[sky + i], i = 1..na_out, from a
        # pointer into the flat sky buffer: lines where that range may leave
        # the line's own na1 values (bmin > 1 only) go through the scalar line
        soff = (np.arange(ny) * g.na1 + _nint_v((low + s).astype(R)) - g.astart
                + g.amin - 2)
        row0 = np.arange(ny) * g.na1
        odd = ok & ((soff + 1 < row0) | (soff + na_out > row0 + g.na1 - 1))
        if odd.any():
            sky_buf = IrafRealBuffer(np.repeat(np.asarray(sky, dtype=R), g.na1), "sky buffer")
            for iy in np.nonzero(odd)[0]:
                out[:, iy] = _strip_line(buf[iy], nc, low, high, s[iy], na_out, sky_buf,
                                         int(soff[iy]))
    return StripResult(data=out, low=aplow, high=aphigh)


def apsum_background_fast(image: NDArray, entry: ApertureEntry, mode: str,
                          sky: NDArray[np.float32] | None = None) -> NDArray[np.float32]:
    """:func:`apsum_background` through the vectorised ``apsum``
    (:func:`decanter.fast.apsum.apsum_multi` with the widened buffer and the
    sky of :func:`sky_values_fast`); bit-identical to the scalar reference."""
    from decanter.fast.apsum import apsum_multi
    bg = background_of(entry)
    if mode == "none" or bg is None:
        return apsum_multi([image], entry)[0]
    img = np.asarray(image, dtype=R)
    if sky is None:
        sky, g = sky_values_fast(img, entry, mode, bg)
    else:
        g = extract_geometry(entry, bg, img.shape[0], img.shape[1], background=True)
    return apsum_multi([img], entry, buffer_cols=(g.amin, g.amax), sky=sky[None, :])[0]
