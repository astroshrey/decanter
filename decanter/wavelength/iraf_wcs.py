"""The linear WCS IRAF writes into WARP's ``_csw`` and ``_fsr*_VAC`` headers.

Downstream code (``winered_pipeline/science/prepare_spectra.py`` and friends)
builds each spectrum's wavelength grid from ``CRVAL1``/``CRPIX1``/``CDELT1``,
so a decanter tree is only a drop-in for WARP's if those keywords carry the
values WARP wrote, digit for digit. They are calibration constants (they
depend on the dispersion solution, the FSR table and the spectrum length, not
on the frame), but getting them right means following IRAF's arithmetic and
its text round trips exactly:

``_csw`` (``onedspec.dispcor``, ``noao/onedspec/dispcor/t_dispcor.x``)
    ``dc_multispec`` linearizes into a MULTISPEC WCS whose per-aperture
    attribute string is written with ``%.14g`` (``smw/smwswattrs.x``);
    ``smw_saveim`` (``smw/smwsaveim.x``) converts it to EQUISPEC by parsing
    that string back with ``ctod``. So the header holds
    ``imputd(ctod(%.14g(w1)))``, not ``w1`` itself.

``_fsr*_VAC`` (``onedspec.scopy w1= w2=``, ``noao/onedspec/t_sarith.x``)
    ``sa_sextract`` maps the FSR bounds to input logical pixels through the
    MWCS world->logical transform (``sys/mwcs/mwsctran.x``: an inverse CD,
    fused ``ltm * x + ltv`` evaluation), the output Lterm is
    ``LTM = dl``, ``LTV = (0 - l1) / dl + 1``, and the physical Wterm is
    rebuilt from the rebinned spectrum (``W0`` / ``WP`` are REAL) in
    ``sa_1d``. ``mw_saveim`` (``sys/mwcs/mwsaveim.x``) then writes
    ``CRPIX = LTM * 1 + LTV`` and ``CD = dw * inv(LTM)``.

Every double goes to text through :func:`decanter.utils.iraf_fmt.imputd_text`
(IRAF's ``dtoc3``), and every re-read through ``ctod``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decanter.utils.fpemu import fma64, iraf_ctod
from decanter.utils.iraf_fmt import dtoc3, imputd_text

_F32 = np.float32


def pyraf_real_text(value: float) -> str:
    """The text pyraf hands an IRAF task for a real parameter.

    pyraf stores real parameters as ``clFloat``, whose ``__str__`` is
    ``'{:.12}'.format(value)`` (``pyraf/irafglobals.py``: "reproduces same
    numerical values as on Py2"), so WARP's Python floats reach IRAF with
    12 significant digits.
    """
    return "{:.12}".format(float(value))


def _fma(a: float, b: float, c: float) -> float:
    return float(fma64(a, b, c))


def _nint(x: float) -> int:
    """f2c ``i_dnnt``: round half away from zero."""
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(0.5 - x))


def _fp_equald(x: float, y: float) -> bool:
    """IRAF ``fp_equald`` (``sys/osb/fpequald.x``): equality to ~32 ULP."""
    from decanter.utils.iraf_fmt import _dt_equald
    return _dt_equald(x, y)


@dataclass(frozen=True, slots=True)
class DispcorWCS:
    """``_csw`` header WCS of one order: the values and their header text."""

    w1: float          # CRVAL1 as IRAF holds it (ctod of the %.14g text)
    dw: float          # CDELT1 / CD1_1
    nw: int
    crval1: str        # header text
    cdelt1: str


def dispcor_wcs(w1: float, dw: float, nw: int) -> DispcorWCS:
    """Header WCS that ``dispcor(linear=yes)`` writes for grid ``(w1, dw, nw)``.

    IRAF equivalent: ``noao/onedspec/dispcor/t_dispcor.x:dc_multispec`` ->
    ``smw/smwswattrs.x`` (MULTISPEC ``"%.14g"`` attribute) ->
    ``smw/smwsaveim.x`` (EQUISPEC conversion via ``smw_gwattrs``/``ctod``,
    ``W = w1 / (1 + z)``, ``CD = dw / (1 + z)`` with ``z = 0``) ->
    ``sys/mwcs/mwsaveim.x`` (``CD * inv(LTM)`` with an identity Lterm).
    """
    w = iraf_ctod(dtoc3(w1, 14, "g")) / 1.0
    d = iraf_ctod(dtoc3(dw, 14, "g")) / 1.0
    d = d * (1.0 / 1.0)
    return DispcorWCS(w1=w, dw=d, nw=int(nw), crval1=imputd_text(w),
                      cdelt1=imputd_text(d))


@dataclass(frozen=True, slots=True)
class ScopyWCS:
    """What ``scopy(w1=, w2=, rebin=yes)`` computes for one linear spectrum.

    ``n`` / ``n_r`` / ``w0`` / ``w1`` drive the data resampling (the output
    length from the double bounds, the resampled length and REAL end
    wavelengths from the REAL bounds; see ``shdr.x:shdr_extract``); the
    remaining fields are the output header text.
    """

    n: int
    n_r: int
    w0: float             # REAL W0 of the rebinned spectrum (as double)
    w1: float             # REAL W1
    wp: float             # REAL step (W1 - W0) / (n_r - 1)
    crval1: str
    crpix1: str
    cdelt1: str
    ltv1: str | None      # None: LTV1 == 0 (IRAF writes no card)
    ltm1_1: str
    np2: int | None       # NP2 card (only when the valid length < NAXIS1)
    # the double values behind the text (``ctod`` of it), for convenience
    crval1_value: float
    crpix1_value: float
    cdelt1_value: float


def scopy_wcs(crval1_text: str, cdelt1_text: str, n_in: int,
              wave_lo: float, wave_hi: float) -> ScopyWCS:
    """``onedspec.scopy(w1=wave_lo, w2=wave_hi)`` on a 1-D EQUISPEC spectrum.

    The input is a ``dispcor`` output (``CRPIX1 = 1``, identity Lterm) whose
    ``CRVAL1``/``CD1_1`` header text is given; ``wave_lo``/``wave_hi`` are the
    Python floats WARP hands to pyraf (:func:`pyraf_real_text` -> ``ctod``).

    IRAF equivalent: ``noao/onedspec/t_sarith.x:sa_1d`` / ``sa_sextract``,
    ``smw/shdr.x:shdr_wl`` / ``shdr_lw`` / ``shdr_extract`` /
    ``shdr_linear``, ``sys/mwcs/mwsctran.x`` + ``mwctran.gx`` (LNR),
    ``smw/smwsaveim.x`` (EQUISPEC) and ``sys/mwcs/mwsaveim.x``.
    """
    W = iraf_ctod(crval1_text)
    CD = iraf_ctod(cdelt1_text)
    sn = int(n_in)
    # world -> logical (mw_sctran: inv(CD), LTV = R - inv(CD) * W, Lterm I)
    inv = 1.0 / CD
    ltv_wl = 0.0 + (1.0 - _fma(inv, W, 0.0))
    # logical -> world (LTM = CD, LTV = W - CD * R)
    ltv_lw = W + _fma(CD, -1.0, 0.0)

    def wl(w: float) -> float:          # shdr_wl (NP1 = 1)
        return (_fma(inv, w, ltv_wl) - 1.0) + 1.0

    def lw(l: float) -> float:          # shdr_lw (NP1 = 1), CTLW set
        return _fma(CD, (l + 1.0) - 1.0, ltv_lw)

    a = iraf_ctod(pyraf_real_text(wave_lo))
    b = iraf_ctod(pyraf_real_text(wave_hi))

    # sa_sextract, header pass: output length and Lterm from DOUBLE bounds
    l1 = wl(a)
    dl = wl(b)
    l1 = max(1.0, min(float(sn), l1))
    dl = max(1.0, min(float(sn), dl))
    n = abs(_nint(dl) - _nint(l1)) + 1
    dl = 1.0 if n == 1 else (dl - l1) / (n - 1)
    ltv2 = (0.0 - l1) / dl + 1.0
    ltm2 = dl * 1.0

    # shdr_extract (data pass): REAL bounds
    r1 = wl(float(_F32(a)))
    r2 = wl(float(_F32(b)))
    r1 = max(1.0, min(float(sn), r1))
    r2 = max(1.0, min(float(sn), r2))
    n_r = abs(_nint(r2) - _nint(r1)) + 1
    w0r = _F32(lw(r1))
    w1r = _F32(lw(r2))
    wp = float(_F32(_F32(w1r - w0r) / _F32(n_r - 1))) if n_r > 1 else 0.0
    W0 = float(w0r)

    # sa_1d: rebuild the dispersion of the output spectrum
    w = _fma(0.0, wp, W0)                       # shdr_lw(sh, 1)
    wb = _fma(float(n_r - 1), wp, W0)           # shdr_lw(sh, SN)
    p1 = (1 - ltv2) / ltm2
    p2 = (n_r - ltv2) / ltm2
    p3 = (n - ltv2) / ltm2
    nw = _nint(min(max(p1, p3), max(p1, p2))) + 1 - 1
    w = w * (1 + 0.0)
    wb = wb * (1 + 0.0)
    dw = wp
    if p1 != p2:
        dw = (wb - w) / (p2 - p1)
    w = _fma(-(p1 - 1), dw, w)

    # smw_saveim (EQUISPEC) + mw_saveim
    crval = w / (1 + 0.0)
    cd_phys = dw / (1 + 0.0)
    crpix = ltv2 + _fma(ltm2, 1.0, 0.0)
    cd = _fma(cd_phys, 1.0 / ltm2, 0.0)

    old = {"CRVAL1": W, "CRPIX1": 1.0, "CDELT1": CD, "LTM1_1": 1.0}

    def put(key: str, new: float, old_text: str) -> str:
        if key in old and _fp_equald(new, old[key]):
            return old_text
        return imputd_text(new)

    crval_t = put("CRVAL1", crval, crval1_text)
    crpix_t = put("CRPIX1", crpix, "1.")
    cdelt_t = put("CDELT1", cd, cdelt1_text)
    ltm_t = put("LTM1_1", ltm2, "1.")
    ltv_t = None if _fp_equald(ltv2, 0.0) else imputd_text(ltv2)
    return ScopyWCS(
        n=n, n_r=n_r, w0=float(w0r), w1=float(w1r), wp=wp,
        crval1=crval_t, crpix1=crpix_t, cdelt1=cdelt_t, ltv1=ltv_t,
        ltm1_1=ltm_t, np2=nw if nw < n else None,
        crval1_value=iraf_ctod(crval_t), crpix1_value=iraf_ctod(crpix_t),
        cdelt1_value=iraf_ctod(cdelt_t),
    )
