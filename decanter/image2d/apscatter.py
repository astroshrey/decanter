"""Stage 3 -- scattered-light subtraction, bit-identical to WARP.

WARP equivalent: ``warp/apscatter.py:pyapscatter`` (line 65), which runs IRAF
``apextract.apscatter`` (``noao/twodspec/apextract/apscatter.x``) with the
flat's apertures as the reference (``references = flat_*``, no find /
recenter / resize / edit / trace) and

  - apscat1 (across the dispersion, one fit per image line):
      function=legendre, order=3, sample=10:2000, naverage=1,
      low_reject=3.0, high_reject=2.0, niterate=100, grow=0
  - apscat2 (along the dispersion, one fit per image column, smooth=yes):
      function=legendre, order=5, sample=10:2000, naverage=1,
      low_reject=3.0, high_reject=3.0 (par-file defaults), niterate=20
  - apscatter.buffer = 1.0 (par-file default)

IRAF ``order`` is the NUMBER OF TERMS (order 3 = quadratic, 5 = quartic).

The algorithm, exactly as compiled (see :mod:`decanter.utils.icfit_exact`
for the float32 arithmetic of every curfit/icfit kernel):

1. ``ap_gscatter1`` (apscatter.x:380): for image line ``l``, every aperture
   masks columns ``max(1, int(c + low - buf)) .. min(ncols, int(c + high +
   buf))`` with ``c = AP_CEN + ap_cveval(trace, real(l))`` in REAL and
   ``buf = buffer + 0.5``.  The remaining columns are the scatter points.
2. ``ic_fit`` (icfitr.x) on each line: the sample ranges select the points
   whose column lies in ``10:2000``; ``cvfit`` on those with the curve range
   ``[min, max]`` of the selected columns; then ``ic_rejectr``.
3. ``ap_smooth`` (apscatter.x:302): column ``c`` of the pass-1 surface is
   ``ap_cveval(cv_line, real(c))`` (clamped to that line's curve range);
   each column is fit along the lines (sample ``10:2000``), rejected, and
   evaluated at every line with ``cvvector``; the output is
   ``in - fit`` (REAL).
4. The scattered-light image is written as ``in - out`` (apscatter.x:227),
   which is NOT bit-identical to the fit itself.

Each pass-1 line and each pass-2 column is an independent fit (in pass 2
IRAF's ``cvrefit`` shortcut reproduces ``cvfit`` bit for bit, since the
normal matrix and its factorization are the same for every column); the
fits are batched across lines / columns.

Output suffix: ``_ssc``.  Gated on ``Config.flag_apscatter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from decanter.calib.aperture import ApertureSet
from decanter.config import Config
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.apdb import ApertureEntry
from decanter.io.listfile import parse as parse_listfile
from decanter.utils import icfit_exact as icx
from decanter.utils.curfit_exact import CurveR, ap_cveval
from decanter.utils.fpemu import iraf_ctod
from decanter.utils.iraf_ranges import rg_token, rg_tokens

R = np.float32

# WARP's apscat parameters (warp/apscatter.py:65-89 + apscat2.par defaults).
# IRAF ``order`` is the number of terms; decanter's legacy API takes the
# polynomial degree, ``degree = order - 1``.
_AP1_DEGREE = 2  # apscat1.order = 3
_AP1_LOW_REJECT = 3.0
_AP1_HIGH_REJECT = 2.0
_AP1_NITERATE = 100
_AP2_DEGREE = 4  # apscat2.order = 5
_AP2_LOW_REJECT = 3.0  # apscat2.par default; WARP doesn't override
_AP2_HIGH_REJECT = 3.0
_AP2_NITERATE = 20
_BUFFER = 1.0  # apscatter.par default
# WARP's sample range "10:2000" (1-indexed, inclusive) for both passes; the
# legacy keyword arguments below are 0-indexed ([9, 1999]).
_SAMPLE = "10:2000"
_SAMPLE_LO_0IDX = 9
_SAMPLE_HI_0IDX = 1999


def _sanitize_objname(raw: str) -> str:
    """Match s01/s02's filename-sanitization rule."""
    name = raw
    for ch in (" ", "'", "\"", "#", "/"):
        name = name.replace(ch, "_")
    return name


# --------------------------------------------------------------------------
# sample ranges (pkg/xtools/ranges: rg_xrangesr + rg_order + rg_merge)
# --------------------------------------------------------------------------

def _parse_sample(sample: str) -> list[tuple[np.float32, np.float32] | None]:
    """Parse an icfit ``sample`` string into REAL ``(lo, hi)`` ranges the way
    ``rg_xadd`` does (:func:`decanter.utils.iraf_ranges.rg_token`).

    ``None`` stands for ``*`` (all points); a token without a first number
    adds nothing ("Range syntax error"); a lone number ``v`` is ``[min(v, 0),
    max(v, 0)]`` (``ctor`` stores 0 when no second number follows).  WARP's
    ``10:2000`` is ``(10, 2000)``.
    """
    out: list[tuple[np.float32, np.float32] | None] = []
    for tok in rg_tokens(sample):
        t = rg_token(tok)
        if t is None:
            continue
        if isinstance(t, str):
            out.append(None)
            continue
        r1, r2 = t
        out.append((min(r1, r2), max(r1, r2)))
    return out


def _in_sample(x: NDArray[np.float32], ranges) -> NDArray[np.bool_]:
    """Points of the (sorted) ``x`` selected by the sample ranges.

    ``rg_xaddr`` clamps each range to ``[x[1], x[n]]`` and takes the index
    span of ``x`` inside it; for sorted ``x`` that is just value membership.
    The merged ranges are packed in increasing order (``rg_packr``).
    """
    sel = np.zeros(x.shape, dtype=bool)
    for r in ranges:
        if r is None:
            sel[:] = True
        else:
            sel |= (x >= r[0]) & (x <= r[1])
    return sel


# --------------------------------------------------------------------------
# apertures (ap_dbread + ap_gscatter1)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _ScatterAperture:
    center: np.float32       # AP_CEN(ap, 1)
    low: np.float32          # AP_LOW(ap, 1)
    high: np.float32         # AP_HIGH(ap, 1)
    curve: CurveR            # AP_CV(ap)


def _scatter_aperture(entry: ApertureEntry) -> _ScatterAperture:
    """REAL aperture as ``ap_dbread`` builds it (``gargr`` = ctod -> REAL)."""
    if entry.raw_center is not None and entry.raw_curve is not None:
        cen = R(iraf_ctod(entry.raw_center[0]))
        low = R(iraf_ctod(entry.raw_low[0]))
        high = R(iraf_ctod(entry.raw_high[0]))
        fit = np.array([iraf_ctod(t) for t in entry.raw_curve], dtype=R)
    else:
        cen, low, high = R(entry.center_x), R(entry.low), R(entry.high)
        fit = np.array([entry.function_type, entry.poly_order, entry.y_min,
                        entry.y_max, *entry.coefficients], dtype=R)
    return _ScatterAperture(cen, low, high, CurveR(fit))


def scatter_aperture_mask(apset: ApertureSet, nlines: int, ncols: int,
                          *, buffer: float = _BUFFER) -> NDArray[np.bool_]:
    """In-aperture pixels as ``ap_gscatter1`` marks them (apscatter.x:399-409).

    Returns ``(nlines, ncols)`` bool, True where some aperture (widened by
    ``buffer + 0.5``) covers the pixel.  The apertures' axis is assumed to
    be 1 (apertures across columns), as for every WINERED database.
    """
    buf = R(R(buffer) + R(0.5))
    line = np.arange(1, nlines + 1, dtype=R)
    col = np.arange(1, ncols + 1, dtype=np.int64)[None, :]
    inap = np.zeros((nlines, ncols), dtype=bool)
    for ap in apset.apertures.values():
        a = _scatter_aperture(ap.entry)
        center = np.asarray(a.center + ap_cveval(a.curve, line), dtype=R)
        lo = np.trunc(np.asarray(np.asarray(center + a.low, dtype=R) - buf,
                                 dtype=R)).astype(np.int64)
        hi = np.trunc(np.asarray(np.asarray(center + a.high, dtype=R) + buf,
                                 dtype=R)).astype(np.int64)
        lo = np.maximum(1, lo)[:, None]
        hi = np.minimum(ncols, hi)[:, None]
        inap |= (col >= lo) & (col <= hi)
    return inap


def build_sample_mask(
    apset: ApertureSet,
    *,
    sample_x_lo: int = _SAMPLE_LO_0IDX,
    sample_x_hi: int = _SAMPLE_HI_0IDX,
    buffer: float = _BUFFER,
) -> NDArray[np.bool_]:
    """Pixels fit by apscat1: outside every aperture and inside the column
    sample range (0-indexed ``[sample_x_lo, sample_x_hi]``).

    IRAF equivalent: ``apextract$apscatter.x:ap_gscatter1`` (lines 380-420)
    followed by the ``rg_xrangesr`` sample selection in ``ic_dosetupr``.
    """
    L = apset.array_length
    inap = scatter_aperture_mask(apset, L, L, buffer=buffer)
    x_idx = np.arange(L)
    in_sample_x = (x_idx >= sample_x_lo) & (x_idx <= sample_x_hi)
    return (~inap) & in_sample_x[None, :]


# --------------------------------------------------------------------------
# the two fitting passes
# --------------------------------------------------------------------------

def _pass1(data: NDArray[np.float32], scatter_pts: NDArray[np.bool_],
           sample_pts: NDArray[np.bool_], order: int, low: float, high: float,
           niterate: int) -> NDArray[np.float32]:
    """apscat1: fit every line, return the ``ap_cveval`` surface
    ``P[l, c] = cveval(cv_l, clamp(c))`` that ap_smooth reads."""
    nlines, ncols = data.shape
    cols = np.arange(1, ncols + 1, dtype=R)
    nscat = scatter_pts.sum(axis=1)
    nfit = sample_pts.sum(axis=1)
    if (nscat == 0).any():
        raise ValueError("apscatter: no scattered-light points on line "
                         f"{int(np.flatnonzero(nscat == 0)[0]) + 1}")
    if (nfit == 0).any():
        raise ValueError("apscatter: no sample points for fit on line "
                         f"{int(np.flatnonzero(nfit == 0)[0]) + 1}")
    maxn = int(nfit.max())
    # Pack the selected points of each line, in increasing column order.
    rank = np.cumsum(sample_pts, axis=1) - 1
    li, ci = np.nonzero(sample_pts)
    x = np.zeros((maxn, nlines), dtype=R)
    y = np.zeros((maxn, nlines), dtype=R)
    valid = np.zeros((maxn, nlines), dtype=bool)
    x[rank[li, ci], li] = cols[ci]
    y[rank[li, ci], li] = data[li, ci]
    valid[rank[li, ci], li] = True
    # padding: repeat the line's first point (finite basis, zero weight)
    x = np.where(valid, x, x[0][None, :])
    # icdosetupr.x:59-75: curve range = range of the fitted points (of all
    # scatter points when only one point is selected).
    first = np.argmax(sample_pts, axis=1)
    last = ncols - 1 - np.argmax(sample_pts[:, ::-1], axis=1)
    xmin = cols[first].copy()
    xmax = cols[last].copy()
    one = nfit == 1
    if one.any():
        sfirst = np.argmax(scatter_pts, axis=1)
        slast = ncols - 1 - np.argmax(scatter_pts[:, ::-1], axis=1)
        xmin[one] = cols[sfirst[one]]
        xmax[one] = cols[slast[one]]
    if (xmax <= xmin).any():
        raise ValueError("apscatter: CVINIT xmax <= xmin")
    ords = np.minimum(order, nfit)
    surface = np.empty((nlines, ncols), dtype=R)
    w = np.ones_like(y)
    for o in np.unique(ords):
        g = np.flatnonzero(ords == o)
        cv, ier = icx.cvfit(x[:, g], y[:, g], w[:, g], valid[:, g], int(o),
                            xmin[g], xmax[g])
        fitok = ier != icx.NO_DEG_FREEDOM   # icfitr.x:67 IC_FITERROR
        if fitok.any():
            gi = np.flatnonzero(fitok)
            sub = cv.take(gi)
            icx.ic_reject(sub, x[:, g[gi]], y[:, g[gi]], w[:, g[gi]],
                          valid[:, g[gi]], low, high, niterate)
            cv.put(gi, sub)
        # ap_gdata: Memr[outdata+i-1] = ap_cveval (cvs[i], real(index))
        xc = np.minimum(np.maximum(cols[None, :], cv.xmin[:, None]),
                        cv.xmax[:, None])
        surface[g] = icx.cveval(cv.coeff[:, :, None], xc, cv.k1[:, None],
                                cv.k2[:, None])
    return surface


def _pass2(surface: NDArray[np.float32], order: int, low: float, high: float,
           niterate: int, sample: str) -> NDArray[np.float32]:
    """apscat2 (ap_smooth): fit each column of ``surface`` along the lines
    and return the ``cvvector`` evaluation at every line."""
    nlines, ncols = surface.shape
    xl = np.arange(1, nlines + 1, dtype=R)
    sel = _in_sample(xl, _parse_sample(sample))
    nfit = int(sel.sum())
    if nfit == 0:
        raise ValueError("apscatter: no sample points for the smoothing fit")
    xs = xl[sel]
    if nfit == 1:
        xmin, xmax = xl[0], xl[-1]
    else:
        xmin, xmax = xs[0], xs[-1]
    o = min(order, nfit)
    x = xs[:, None]
    y = surface[sel]
    w = np.ones_like(x)
    valid = np.ones_like(x, dtype=bool)
    cv, ier = icx.cvfit(x, y, w, valid, o, xmin, xmax, shared_x=True)
    fitok = ier != icx.NO_DEG_FREEDOM
    if fitok.any():
        gi = np.flatnonzero(fitok)
        sub = cv.take(gi)
        icx.ic_reject(sub, x, y[:, gi], w, valid, low, high, niterate)
        cv.put(gi, sub)
    return icx.cvvector(cv.coeff[:, None, :], xl[:, None], cv.k1[None, :],
                        cv.k2[None, :])


def apscatter_fit(
    image: NDArray[np.floating],
    scatter_pts: NDArray[np.bool_],
    *,
    ap1_order: int = _AP1_DEGREE + 1,
    ap1_sample: str = _SAMPLE,
    ap1_low_reject: float = _AP1_LOW_REJECT,
    ap1_high_reject: float = _AP1_HIGH_REJECT,
    ap1_niterate: int = _AP1_NITERATE,
    ap2_order: int = _AP2_DEGREE + 1,
    ap2_sample: str = _SAMPLE,
    ap2_low_reject: float = _AP2_LOW_REJECT,
    ap2_high_reject: float = _AP2_HIGH_REJECT,
    ap2_niterate: int = _AP2_NITERATE,
) -> NDArray[np.float32]:
    """The smoothed scattered-light fit (``cvvector`` of the apscat2 fits).

    IRAF equivalent: ``apextract$apscatter.x:ap_scatter`` (smooth = yes)
    -> ``ap_gscatter1`` / ``ic_fit`` per line, ``ap_smooth`` per column.

    Args:
        image: 2-D frame, read as REAL.
        scatter_pts: ``(nlines, ncols)`` bool, True for the scattered-light
            points of each line (outside every aperture).
    """
    data = np.asarray(image, dtype=R)
    nlines, ncols = data.shape
    cols = np.arange(1, ncols + 1, dtype=R)
    sample_cols = _in_sample(cols, _parse_sample(ap1_sample))
    sample_pts = scatter_pts & sample_cols[None, :]
    surface = _pass1(data, scatter_pts, sample_pts, ap1_order,
                     ap1_low_reject, ap1_high_reject, ap1_niterate)
    return _pass2(surface, ap2_order, ap2_low_reject, ap2_high_reject,
                  ap2_niterate, ap2_sample)


def apscatter_model(
    image: NDArray[np.floating],
    sample_mask: NDArray[np.bool_],
    *,
    ap1_degree: int = _AP1_DEGREE,
    ap1_low_reject: float = _AP1_LOW_REJECT,
    ap1_high_reject: float = _AP1_HIGH_REJECT,
    ap1_niterate: int = _AP1_NITERATE,
    ap2_degree: int = _AP2_DEGREE,
    ap2_low_reject: float = _AP2_LOW_REJECT,
    ap2_high_reject: float = _AP2_HIGH_REJECT,
    ap2_niterate: int = _AP2_NITERATE,
    sample_y_lo: int = _SAMPLE_LO_0IDX,
    sample_y_hi: int = _SAMPLE_HI_0IDX,
) -> NDArray[np.float32]:
    """Legacy interface: the 2-pass scattered-light fit from a fit-point mask.

    ``sample_mask`` (from :func:`build_sample_mask`) already combines the
    out-of-aperture test with the column sample range, so pass 1 fits
    exactly its True pixels; pass 2 uses the 0-indexed line range
    ``[sample_y_lo, sample_y_hi]``.  Returns the REAL fit surface
    (``cvvector`` of the pass-2 fits); see :func:`subtract_apscatter` for
    the WARP products.
    """
    data = np.asarray(image, dtype=R)
    if sample_mask.shape != data.shape:
        raise ValueError("sample_mask shape must match image shape")
    surface = _pass1(data, sample_mask, sample_mask, ap1_degree + 1,
                     ap1_low_reject, ap1_high_reject, ap1_niterate)
    return _pass2(surface, ap2_degree + 1, ap2_low_reject, ap2_high_reject,
                  ap2_niterate, f"{sample_y_lo + 1}:{sample_y_hi + 1}")


def subtract_apscatter(
    data: NDArray,
    apset: ApertureSet,
    *,
    sample_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray, NDArray[np.float32]]:
    """Subtract scattered light from one 2-D frame, bit-identical to WARP.

    IRAF equivalent: ``apextract.apscatter`` as run by
    ``warp/apscatter.py:pyapscatter``.

    Args:
        data: the frame (``_s``), read as REAL.
        apset: the reference apertures (WARP: the flat's ``apflat_*``).
        sample_mask: optional precomputed out-of-aperture mask
            (``(nlines, ncols)`` bool, True = scattered-light point, as from
            ``~scatter_aperture_mask(...)``); built from ``apset`` if None.

    Returns:
        ``(out, scatter)``: ``out`` (WARP ``_ssc``) ``= data - fit`` in REAL,
        cast to ``data``'s dtype; ``scatter`` (WARP ``scatter_*``) ``= data -
        out`` in REAL, exactly as apscatter.x:226-228 writes it.
    """
    d = np.asarray(data, dtype=R)
    nlines, ncols = d.shape
    if sample_mask is None:
        sample_mask = ~scatter_aperture_mask(apset, nlines, ncols)
    fit = apscatter_fit(d, sample_mask)
    out = np.asarray(d - fit, dtype=R)          # asubr (indata, fit)
    scatter = np.asarray(d - out, dtype=R)      # asubr (in, out)
    return out.astype(np.asarray(data).dtype, copy=False), scatter


def run(
    config: Config,
    workdir: Path,
    listfile: Path,
    *,
    apdb_path: Path | str | None = None,
    **_unused: Any,
) -> None:
    """Subtract scattered light frame-by-frame (legacy stage shim).

    Reads ``{obj}_NO{i}_s.fits`` and writes ``{obj}_NO{i}_ssc.fits``;
    when ``flag_apscatter`` is False the input is copied through.
    """
    pairs = parse_listfile(listfile)

    if not config.flag_apscatter:
        for i, pair in enumerate(pairs, start=1):
            obj_data, obj_header = _fits.read_image(workdir / pair.object_name)
            raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
            objname = _sanitize_objname(str(raw_objname))
            in_path = workdir / f"{objname}_NO{i}_s.fits"
            data, header = _fits.read_image(in_path)
            out_path = workdir / f"{objname}_NO{i}_ssc.fits"
            _fits.write_image(out_path, data, header, overwrite=True)
        return

    if apdb_path is None:
        raise ValueError(
            "s03_apscatter requires apdb_path when flag_apscatter=True; "
            "the calibration loader (s00) will provide this automatically once wired in."
        )

    apset = ApertureSet.load(apdb_path)
    sample_mask = None

    for i, pair in enumerate(pairs, start=1):
        _, obj_header = _fits.read_image(workdir / pair.object_name)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objname = _sanitize_objname(str(raw_objname))

        data, header = _fits.read_image(workdir / f"{objname}_NO{i}_s.fits")
        if sample_mask is None:
            sample_mask = ~scatter_aperture_mask(apset, *np.shape(data))
        out, _ = subtract_apscatter(data, apset, sample_mask=sample_mask)

        out_path = workdir / f"{objname}_NO{i}_ssc.fits"
        _fits.write_image(out_path, out, header, overwrite=True)
