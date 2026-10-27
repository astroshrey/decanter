"""WARP-identical per-frame extraction aperture (free / standalone path).

When no per-frame trace database is handed to it, WARP does **not** re-trace
the star on each frame: it reuses the rectified-strip reference trace of the
multihole calibration aperture (``ap{aptrans}_{m}trans``) and measures only a
per-frame *shift* and *width* of the star relative to it, then writes one
aperture database per (frame, order) that ``apall`` extracts with.

WARP equivalent (``Warp_sci.py``, "center search and bad pix detection" and
the extraction loop; ``warp/centersearch_fortrans.py``; ``warp/aperture.py``):

  1. per order: ``xcenter, fwhm = centersearch_fortrans(trans, aptrans_m, ...,
     abbaflag)`` with ``abbaflag = "O" not in NODPOS and NODPOS != "NA"``
     -- :func:`decanter.utils.psf.warp_centersearch`, bit-exact given the
     MINPACK WARP ran (the installed scipy by default; scipy 1.15.3's C
     MINPACK via :mod:`decanter.utils.minpack_exact` with
     ``DECANTER_WARP_MINPACK=c1153``);
  2. one frame-constant "2 sigma" window: ``lowtrans = -nanmedian(fwhm) +
     nanmedian(xcenter)``, ``hightrans = nanmedian(fwhm) + nanmedian(xcenter)``;
  3. per order, clamped to the reference aperture limits
     (``lowlim = apLow if apLow > lowtrans else lowtrans``, ``highlim``
     likewise) and written by ``apertureSet.write_apdatabase`` with fixed
     formats (``center %.2f %f``, ``low %.2f %f``, ``high %.2f %f``, curve
     values ``%f``).

:func:`solve_frame_apdb` reproduces step 3's database *tokens* exactly, so
:func:`decanter.extract.apsum_exact.apsum` extracts with precisely the
aperture IRAF ``apall`` read from WARP's file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from decanter.calib.aperture import ApertureSet
from decanter.calib.transform import RectifiedStrip
from decanter.io.apdb import ApertureEntry
from decanter.io.apdb import parse as parse_apdb
from decanter.utils.psf import warp_centersearch


def warp_abbaflag(nodpos: str) -> bool:
    """WARP's ``abbaflag``: ``NODPOS`` has no ``O`` and is not ``NA`` (``Warp_sci.py:395``)."""
    nodpos = str(nodpos).strip()
    return nodpos.find("O") == -1 and nodpos != "NA"


@dataclass(frozen=True, slots=True)
class FrameApertureSolution:
    """WARP's per-frame aperture solution.

    Attributes:
        entries: ``{order: ApertureEntry}`` -- exactly what decanter's
            :func:`decanter.io.apdb.parse` returns for WARP's per-frame
            ``ap<OBJ>_NO<i>_sscfm_m<m>trans`` file (raw tokens included).
        xcenter, fwhm: per-order center-search results (WARP's
            ``centersearch_log`` values; NaN where WARP gave up).
        lowtrans, hightrans: the unclamped frame window (float64).
        background_sample: the ``sample`` string written (WARP's
            ``conf.skysub_region``: the listfile's ``bg=`` region, ``INDEF``
            without one, config.py:120-145).
        minpack_undefined: per order, how many times the center-search fit
            hit the C MINPACK ``qrfac`` read past the Jacobian that makes
            scipy 1.15.3's result non-deterministic
            (``minpack_exact.STATS["undefined"]``; only counted with
            ``DECANTER_WARP_MINPACK=c1153``). Nonzero: the fitted floats (and
            the frame window) are not reproducible, in WARP either.
    """

    entries: dict[int, ApertureEntry]
    xcenter: dict[int, float]
    fwhm: dict[int, float]
    lowtrans: float
    hightrans: float
    background_sample: str = "INDEF"
    minpack_undefined: dict[int, int] = field(default_factory=dict)
    # per order, the binned slit profile WARP writes to
    # images/spatial_profile/<strip>.dat ({m: (med_x, med_y)}; missing where
    # WARP's center search gave up before writing it)
    profiles: dict[int, tuple] = field(default_factory=dict)


def _header_double(v: float) -> float:
    """A double as WARP's centersearch reads it back from the IRAF-written header.

    IRAF writes the trans strip's WCS keywords with 15 significant digits
    (e.g. ``CRVAL2 = -526.447082519531`` for -526.44708251953125) and astropy
    parses that text, so the value crosses the FITS boundary as ``%.15g``.
    """
    return float("%.15g" % float(v))


def _strip_wcs(strip) -> tuple[NDArray, float, float, float]:
    """``(data, CRVAL2, CDELT2, CRPIX2)`` of a rectified strip.

    A :class:`RectifiedStrip` carries the in-memory WCS; it is passed through
    the same ``%.15g`` header round trip WARP's FITS file imposes. A tuple is
    taken as already header-parsed values.
    """
    if isinstance(strip, RectifiedStrip):
        return (strip.data, _header_double(strip.lambda_min),
                _header_double(strip.dy), 1.0)
    data, crval2, cdelt2, crpix2 = strip
    return data, float(crval2), float(cdelt2), float(crpix2)


def _bg_coords(bgsample: str) -> list[float]:
    """Background-region numbers WARP folds into ``xmin``/``xmax`` (``aperture.py:175``)."""
    out: list[float] = []
    if bgsample.find("INDEF") == -1:
        for part in bgsample.split(","):
            for tok in part.split(":"):
                out.append(float(tok))
    return out


def warp_background_tokens(lowlim: float, highlim: float,
                           bgsample: str = "INDEF") -> tuple[tuple[str, str], ...]:
    """The ``background`` section ``write_apdatabase`` writes (aperture.py:174-182, 201-211).

    ``xmin``/``xmax`` are ``%.2f`` of the extremes of the window and the
    ``bg=`` region numbers (floats, as WARP folds them); the other fields are
    WARP's constants (``chebyshev``, order 1, ``naverage -3``, ...).
    """
    apcoord = [lowlim, highlim, *_bg_coords(bgsample)]
    return (("xmin", f"{min(apcoord):.2f}"), ("xmax", f"{max(apcoord):.2f}"),
            ("function", "chebyshev"), ("order", "1"), ("sample", bgsample),
            ("naverage", "-3"), ("niterate", "0"), ("low_reject", "3."),
            ("high_reject", "3."), ("grow", "0."))


def warp_apdb_tokens(ref: ApertureEntry, lowlim: float, highlim: float,
                     bgsample: str = "INDEF") -> dict:
    """The tokens ``apertureSet.write_apdatabase`` writes for one aperture.

    ``ref`` is the multihole trans reference aperture (values as WARP's
    ``float()`` parse gives them); ``lowlim``/``highlim`` the clamped window;
    ``bgsample`` the listfile's ``bg=`` region (``INDEF`` without one).
    """
    cx, cy = ref.center_x, ref.center_y
    curve = (
        "%f" % ref.function_type,
        "%f" % ref.poly_order,
        "%f" % ref.y_min,
        "%f" % ref.y_max,
    ) + tuple("%f" % c for c in ref.coefficients[: ref.poly_order])
    return {
        "center": ("%.2f" % cx, "%f" % cy),
        "low": ("%.2f" % lowlim, "%f" % (-cy + 1.)),
        "high": ("%.2f" % highlim, "%f" % (cy + 1.)),
        "curve": curve,
        "background": warp_background_tokens(lowlim, highlim, bgsample),
    }


def _entry_from_tokens(order: int, tok: dict, bgsample: str) -> ApertureEntry:
    """Build the entry exactly as :func:`decanter.io.apdb.parse` reads the file."""
    curve = tok["curve"]
    ftype = int(float(curve[0]))
    yorder = int(float(curve[1]))
    return ApertureEntry(
        order=order,
        center_x=float(tok["center"][0]),
        center_y=float(tok["center"][1]),
        low=float(tok["low"][0]),
        high=float(tok["high"][0]),
        function_type=ftype,
        poly_order=yorder,
        y_min=float(curve[2]),
        y_max=float(curve[3]),
        coefficients=tuple(float(c) for c in curve[4:4 + yorder]),
        background_sample=bgsample,
        raw_center=tok["center"],
        raw_low=tok["low"],
        raw_high=tok["high"],
        raw_curve=curve,
        raw_background=tok.get("background"),
    )


def solve_frame_apdb(
    strips: dict,
    ref_trans_apdbs: dict[int, Path],
    *,
    nodpos: str | None = None,
    abba: bool | None = None,
    bgsample: str = "INDEF",
    leastsq=None,
    window: tuple[float, float] | None = None,
) -> FrameApertureSolution:
    """WARP's per-frame aperture databases, token for token.

    With ``window=(low, high)`` the center search is skipped and that window
    is used as the frame aperture (WARP's ``Manual Aperture: yes`` with the
    listfile's ``ap=low:high`` token, ``Warp_sci.py:418-421``); it is clamped
    to each reference aperture exactly like a solved window.

    Args:
        strips: ``{order: RectifiedStrip}`` or ``{order: (data, CRVAL2,
            CDELT2, CRPIX2)}`` -- the rectified ``_m###trans`` object strips.
        ref_trans_apdbs: ``{order: Path}`` multihole trans reference DBs.
        nodpos: the object frame's ``NODPOS`` (sets WARP's ``abbaflag``);
            alternatively pass ``abba`` directly.
        bgsample: WARP's background ``sample`` string (the listfile's
            ``bg=`` region; ``INDEF`` without one, which apall's ``fit``
            background reads as every x >= 0, decanter.utils.iraf_ranges).
        leastsq: the Gaussian-fit driver (see
            :func:`decanter.utils.psf.default_warp_leastsq`). The default is
            the installed ``scipy.optimize.leastsq`` (what WARP calls in the
            same venv) unless ``DECANTER_WARP_MINPACK=c1153`` selects the
            bit-exact scipy 1.15.3 C MINPACK port; ``"scipy"`` forces the
            installed one.

    Returns:
        A :class:`FrameApertureSolution`; ``entries[m]`` equals
        ``decanter.io.apdb.parse(<WARP ap..._m{m}trans>)[m]``.
    """
    if window is not None:
        orders = [m for m in strips if m in ref_trans_apdbs]
        if not orders:
            raise ValueError("no order has both a strip and a trans reference aperture")
        lowtrans, hightrans = float(window[0]), float(window[1])
        entries: dict[int, ApertureEntry] = {}
        for m in orders:
            ref = parse_apdb(ref_trans_apdbs[m])[m]
            lowlim = ref.low if ref.low > lowtrans else lowtrans
            highlim = ref.high if ref.high < hightrans else hightrans
            entries[m] = _entry_from_tokens(m, warp_apdb_tokens(ref, lowlim, highlim, bgsample),
                                            bgsample)
        return FrameApertureSolution(
            entries=entries, xcenter={}, fwhm={}, lowtrans=lowtrans, hightrans=hightrans,
            background_sample=bgsample,
        )
    if abba is None:
        if nodpos is None:
            raise ValueError("pass nodpos (or abba)")
        abba = warp_abbaflag(nodpos)
    if isinstance(leastsq, str):
        if leastsq != "scipy":
            raise ValueError(f"leastsq={leastsq!r}: pass a callable, None or 'scipy'")
        from scipy.optimize import leastsq
    orders = [m for m in strips if m in ref_trans_apdbs]
    refs = {m: parse_apdb(ref_trans_apdbs[m])[m] for m in orders}
    from decanter.utils import minpack_exact
    xcenter: dict[int, float] = {}
    fwhm: dict[int, float] = {}
    undefined: dict[int, int] = {}
    profiles: dict[int, tuple] = {}
    for m in orders:
        data, crval2, cdelt2, crpix2 = _strip_wcs(strips[m])
        n0 = minpack_exact.STATS["undefined"]
        prof: dict = {}
        xcenter[m], fwhm[m] = warp_centersearch(
            data, refs[m], crval2=crval2, cdelt2=cdelt2, crpix2=crpix2, abba=abba,
            leastsq=leastsq, profile=prof)
        undefined[m] = minpack_exact.STATS["undefined"] - n0
        if "med_x" in prof:
            profiles[m] = (prof["med_x"], prof["med_y"])
    if not orders:
        raise ValueError("no order has both a strip and a trans reference aperture")
    xs = [xcenter[m] for m in orders]
    gs = [fwhm[m] for m in orders]
    lowtrans = -1. * np.nanmedian(gs) + np.nanmedian(xs)
    hightrans = np.nanmedian(gs) + np.nanmedian(xs)
    if not (np.isfinite(lowtrans) and np.isfinite(hightrans)):
        raise ValueError(
            "center search found no usable stellar profile in any order; "
            "cannot solve an aperture (are the strips empty?)")
    entries: dict[int, ApertureEntry] = {}
    for m in orders:
        ref = refs[m]
        lowlim = ref.low if ref.low > lowtrans else lowtrans
        highlim = ref.high if ref.high < hightrans else hightrans
        entries[m] = _entry_from_tokens(m, warp_apdb_tokens(ref, lowlim, highlim, bgsample),
                                            bgsample)
    return FrameApertureSolution(
        entries=entries, xcenter=xcenter, fwhm=fwhm,
        lowtrans=float(lowtrans), hightrans=float(hightrans),
        background_sample=bgsample, minpack_undefined=undefined, profiles=profiles,
    )


def warp_apdb_text(entry: ApertureEntry, image: str, *, timeline: str | None = None) -> str:
    """Render one aperture exactly as ``apertureSet.write_apdatabase`` writes it.

    ``image`` is the trans image name (e.g. ``TOI2109_NO1_sscfm_m163trans``).
    ``timeline`` defaults to WARP's ``# %a %H:%M:%S %d-%b-%Y`` UTC stamp.
    """
    import time

    if timeline is None:
        timeline = time.strftime("# %a %H:%M:%S %d-%b-%Y\n", time.gmtime())
    m = entry.order
    lowlim, highlim = float(entry.raw_low[0]), float(entry.raw_high[0])
    # WARP formats xmin/xmax from the *float* window, which %.2f-rounds to the
    # same token as low/high; recompute from the tokens' source is not needed.
    apcoord = [lowlim, highlim] + _bg_coords(entry.background_sample)
    cx, cy = entry.raw_center
    lines = [
        timeline,
        "begin\taperture\t%s %d %s %.2f\n" % (image, m, cx, float(cy)),
        "\timage\t%s\n" % image,
        "\taperture\t%d\n" % m,
        "\tbeam\t%d\n" % m,
        "\tcenter\t%s %s\n" % (cx, cy),
        "\tlow\t%s %s\n" % entry.raw_low,
        "\thigh\t%s %s\n" % entry.raw_high,
        "\tbackground\n",
        "\t\txmin %.2f\n" % min(apcoord),
        "\t\txmax %.2f\n" % max(apcoord),
        "\t\tfunction chebyshev\n",
        "\t\torder 1\n",
        "\t\tsample %s\n" % entry.background_sample,
        "\t\tnaverage -3\n",
        "\t\tniterate 0\n",
        "\t\tlow_reject 3.\n",
        "\t\thigh_reject 3.\n",
        "\t\tgrow 0.\n",
        "\taxis 1\n",
        "\tcurve %d\n" % len(entry.raw_curve),
    ]
    lines += ["\t\t%s\n" % t for t in entry.raw_curve]
    lines.append("\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Back-compatible geometric summary (used by the optimal / background paths)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FrameAperture:
    """Per-frame extraction geometry solved from the reference apertures.

    ``entries`` holds the WARP-identical per-frame aperture database entries
    (see :func:`solve_frame_apdb`); ``ap_low``/``ap_high`` are the unclamped
    frame window and ``traces`` the reference traces, for the non-WARP
    extraction modes.
    """

    traces: dict[int, NDArray[np.float64]]   # {order: reference trace x(y)}
    ap_low: float                            # frame-constant, relative to trace
    ap_high: float
    entries: dict[int, ApertureEntry] = field(default_factory=dict)
    minpack_undefined: dict[int, int] = field(default_factory=dict)  # see FrameApertureSolution
    xcenter: dict[int, float] = field(default_factory=dict)          # centersearch_log
    fwhm: dict[int, float] = field(default_factory=dict)
    profiles: dict[int, tuple] = field(default_factory=dict)         # spatial_profile .dat


def solve_frame_aperture(
    strips: dict[int, RectifiedStrip],
    ref_trans_apdbs: dict[int, Path],
    *,
    abba: bool = False,
    window: tuple[float, float] | None = None,
    bgsample: str = "INDEF",
) -> FrameAperture:
    """Solve the per-frame aperture as WARP does (see :func:`solve_frame_apdb`).

    Args:
        strips: ``{order: RectifiedStrip}`` from :func:`rectify_orders`.
        ref_trans_apdbs: ``{order: Path}`` multihole trans reference DBs.
        abba: WARP's ``abbaflag`` (:func:`warp_abbaflag` of ``NODPOS``).
        window: a manual ``(low, high)`` frame window (WARP's manual
            aperture); skips the center search.
        bgsample: the listfile's ``bg=`` background region (``INDEF``
            without one), written into the entries' ``background`` section.

    Returns:
        A :class:`FrameAperture` with the reference traces, the frame window
        and the WARP-identical database entries.
    """
    sol = solve_frame_apdb(strips, ref_trans_apdbs, abba=abba, window=window,
                           bgsample=bgsample)
    traces = {}
    for m in sol.entries:
        n_rows = _strip_wcs(strips[m])[0].shape[0]
        traces[m] = np.asarray(
            ApertureSet.load(ref_trans_apdbs[m], array_length=n_rows).apertures[m].trace_x,
            dtype=np.float64)
    return FrameAperture(traces=traces, ap_low=sol.lowtrans, ap_high=sol.hightrans,
                         entries=sol.entries, minpack_undefined=dict(sol.minpack_undefined),
                         xcenter=dict(sol.xcenter), fwhm=dict(sol.fwhm),
                         profiles=dict(sol.profiles))
