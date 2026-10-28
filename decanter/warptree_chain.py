"""The header of every per-frame WARP product, step by step.

WARP builds each file from the previous one with an IRAF task, and each
task edits the header in its own way (``Warp_sci.py``; the IRAF sources are
cited per function).  These functions apply the same edits to a user area
(UA, :mod:`decanter.warptree_io`), so that every intermediate and final file
carries WARP's cards, values and order:

* 2-D: ``_s`` (``imarith``), ``scatter_*_s`` / ``_ssc`` (``apscatter`` +
  ``hedit``), ``_sscf`` (``imarith /`` + ``hedit FLAT``), ``_sscfm``
  (``imcopy`` + ``fixpix`` + ``hedit BP_MASK``; ``cutransform`` later adds
  ``DISPAXIS`` to it in place), ``_m<m>cut`` (``imcopy`` section) and
  ``_m<m>trans`` (``transform``); the sky path ``_f``, ``_fm``.
* 1-D: the ``apall`` extraction (``apextract/apmw.x``), the ``truncate``
  (``scopy w1=1 w2=2048`` + three ``hedit``), the cross-frame shift
  (``PySpecshift``: ``scopy`` + ``specshift`` + ``scombine``), the
  ``REFSPEC1`` ``hedit`` ``dispcor_single`` makes on its *input*, ``dispcor``
  and the FSR ``scopy``.
"""
from __future__ import annotations

import numpy as np

from decanter.utils.fpemu import fma64, iraf_ctod
from decanter.utils.iraf_header import IrafHeader
from decanter.utils.iraf_fmt import dtoc3, imputd_text
from decanter.wavelength.iraf_onedspec import (Mwcs1D, delete, fp_equald, num_text, put_num,
                                                put_str)
from decanter.wavelength.iraf_wcs import ScopyWCS, pyraf_real_text, scopy_wcs

BANDID_NONE = "spectrum: background none, weights none, clean no"


# ---------------------------------------------------------------------------
# 2-D chain
# ---------------------------------------------------------------------------

def ua_ssc(ua_s: IrafHeader, apsc_name: str) -> IrafHeader:
    """``pyapscatter``: ``apscatter`` output + ``APSCATTE``/``APSCMASK``."""
    h = ua_s.copy()
    put_str(h, "APSCATTE", "Scattered light subtracted")
    put_str(h, "APSCMASK", apsc_name)
    return h


def ua_flat(ua: IrafHeader, flat_name: str) -> IrafHeader:
    """``flatfielding``: ``imarith /`` + ``hedit FLAT``."""
    h = ua.copy()
    put_str(h, "FLAT", flat_name)
    return h


def ua_fixpix(ua: IrafHeader, fixpix_time: str, mask_name: str) -> IrafHeader:
    """``pyfixpix``: ``fixpix`` (``FIXPIX``) + ``hedit BP_MASK``."""
    h = ua.copy()
    put_str(h, "FIXPIX", f"{fixpix_time} Bad pixel file is {mask_name}")
    put_str(h, "BP_MASK", mask_name)
    return h


def ua_dispaxis(ua: IrafHeader) -> IrafHeader:
    """``cutransform``: ``hedit DISPAXIS 2`` on its input (in place)."""
    h = ua.copy()
    put_num(h, "DISPAXIS", "2")
    return h


def ua_cut(ua_m: IrafHeader, xmin: int) -> IrafHeader:
    """``imcopy <img>[xmin:xmax,*] <img>_m<m>cut`` (section Lterm)."""
    h = ua_m.copy()
    put_num(h, "WCSDIM", "2")
    if xmin != 1:
        put_num(h, "LTV1", imputd_text(float(1 - xmin)))
    put_num(h, "LTM1_1", "1.")
    put_num(h, "LTM2_2", "1.")
    put_str(h, "WAT0_001", "system=physical")
    put_str(h, "WAT1_001", "wtype=linear")
    put_str(h, "WAT2_001", "wtype=linear")
    return h


def ua_trans(ua_cut: IrafHeader, crval2_text: str, cdelt2_text: str) -> IrafHeader:
    """``transform`` output (``longslit``), from the ``_cut`` header (its
    section ``LTV1`` goes: the output Lterm is the identity)."""
    h = ua_cut.copy()
    delete(h, "LTV1")
    put_num(h, "WCSDIM", "2")
    put_num(h, "LTM1_1", "1.")
    put_num(h, "LTM2_2", "1.")
    put_str(h, "WAT0_001", "system=world")
    put_str(h, "WAT1_001", "wtype=linear")
    put_str(h, "WAT2_001", "wtype=linear label=Wavelength units=angstroms")
    put_str(h, "DCLOG1", "Transform")
    put_num(h, "DC-FLAG", "0")
    put_str(h, "CTYPE1", "LINEAR")
    put_str(h, "CTYPE2", "LINEAR")
    for key, text in (("CRVAL1", "1."), ("CRVAL2", crval2_text), ("CRPIX1", "1."),
                      ("CRPIX2", "1."), ("CDELT1", "1."), ("CDELT2", cdelt2_text),
                      ("CD1_1", "1."), ("CD2_2", cdelt2_text)):
        put_num(h, key, text)
    return h


# ---------------------------------------------------------------------------
# 1-D chain
# ---------------------------------------------------------------------------

def apall_dispersion(crval2_text: str, cdelt2_text: str, nrows: int) -> tuple[float, float]:
    """``apmw_open``: ``W1``/``DW`` of the extracted spectrum.

    The strip's logical -> world transform on the dispersion axis
    (``mw_sctran`` + fused ``mw_c1trand``, R = 1, identity Lterm):
    ``w(l) = fma(CD, l, W - CD)``; ``DW = (w(n) - w(1)) / (n - 1)``.
    """
    W = iraf_ctod(crval2_text)
    CD = iraf_ctod(cdelt2_text)
    t = W + CD * (-1.0)
    w1 = float(fma64(CD, 1.0, t))
    wn = float(fma64(CD, float(nrows), t))
    return w1, (wn - w1) / (nrows - 1)


def bandid(background: str = "none") -> str:
    """apall's ``BANDID1`` of a ONEDSPEC extraction (apextract.x:1238-1247)."""
    return f"spectrum: background {background}, weights none, clean no"


def ua_apall(ua_strip: IrafHeader, nrows: int, apnum: str,
             background: str = "none") -> tuple[IrafHeader, float, float]:
    """``apall format=onedspec`` output header (``apextract/apmw.x:apmw_saveim``).

    Returns the UA and the extracted spectrum's ``(W1, DW)``. ``background``
    is apall's ``background`` (in ``BANDID1``).
    """
    w1, dw = apall_dispersion(num_text(ua_strip, "CRVAL2"), num_text(ua_strip, "CD2_2"), nrows)
    h = ua_strip.copy()
    put_str(h, "BANDID1", bandid(background))
    put_str(h, "APNUM1", apnum)
    put_num(h, "DC-FLAG", "0")
    delete(h, "DISPAXIS")
    put_num(h, "WCSDIM", "1")
    for key in ("LTM2_2", "WAT2_001", "CTYPE2", "CRVAL2", "CRPIX2", "CDELT2", "CD2_2", "LTV1"):
        delete(h, key)
    put_str(h, "WAT0_001", "system=equispec")
    put_str(h, "WAT1_001", "wtype=linear label=Wavelength units=angstroms")
    if fp_equald(w1, 0.0):
        delete(h, "CRVAL1")
    elif not fp_equald(w1, 1.0):
        put_num(h, "CRVAL1", imputd_text(w1))
    if not fp_equald(dw, 1.0):
        put_num(h, "CDELT1", imputd_text(dw))
        put_num(h, "CD1_1", imputd_text(dw))
    return h, w1, dw


def _sa_move_bandid(h: IrafHeader) -> None:
    """``sa_1d``/``sa_ms``: DISPAXIS, APID*, BANDID* deleted, BANDID1 re-added."""
    band = h.get("BANDID1")
    for key in list(h.keys()):
        if key == "DISPAXIS" or key.startswith("APID") or key.startswith("BANDID"):
            delete(h, key)
    if band is not None:
        put_str(h, "BANDID1", str(band))


def _apply_scopy_wcs(h: IrafHeader, sc: ScopyWCS) -> None:
    """``smw_saveim`` + ``mw_saveim`` of a ``scopy(w1=, w2=)`` output."""
    from decanter.wavelength.iraf_onedspec import _apnum_text
    ap = _apnum_text(h)
    if ap is not None:
        put_str(h, "APNUM1", ap)
    if "DC-FLAG" in h:
        put_num(h, "DC-FLAG", str(int(h["DC-FLAG"])))
    if sc.np2 is not None:
        put_num(h, "NP2", str(int(sc.np2)))
    elif "NP2" in h:
        delete(h, "NP2")
    put_num(h, "CRVAL1", sc.crval1)
    put_num(h, "CRPIX1", sc.crpix1)
    put_num(h, "CDELT1", sc.cdelt1)
    put_num(h, "CD1_1", sc.cdelt1)
    if sc.ltv1 is not None:
        put_num(h, "LTV1", sc.ltv1)
    else:
        delete(h, "LTV1")
    put_num(h, "LTM1_1", sc.ltm1_1)


def ua_scopy_range(ua: IrafHeader, n_in: int, w1: float, w2: float) -> tuple[IrafHeader, ScopyWCS]:
    """``scopy(in, out, w1=, w2=)`` (rebin = yes) of a ``CRPIX1 = 1``,
    identity-Lterm EQUISPEC spectrum (:func:`decanter.wavelength.iraf_wcs.scopy_wcs`)."""
    sc = scopy_wcs(num_text(ua, "CRVAL1"), num_text(ua, "CD1_1") or num_text(ua, "CDELT1"),
                   n_in, w1, w2)
    h = ua.copy()
    _sa_move_bandid(h)
    _apply_scopy_wcs(h, sc)
    return h, sc


def ua_truncate(ua_apall: IrafHeader, n_in: int) -> IrafHeader:
    """``Spec1Dtools.truncate``: ``scopy w1=1. w2=2048.`` + ``hedit`` CRVAL1,
    CRPIX1, LTV1 (``hedit`` does not add a missing card)."""
    h, _ = ua_scopy_range(ua_apall, n_in, 1.0, 2048.0)
    put_num(h, "CRVAL1", "1.")
    put_num(h, "CRPIX1", "1.")
    if "LTV1" in h:
        put_num(h, "LTV1", "0.")
    return h


def ua_scopy_copy(ua: IrafHeader, n: int) -> IrafHeader:
    """``scopy(in, out)`` without limits (:func:`decanter.wavelength.iraf_onedspec.sa_1d`)."""
    from decanter.wavelength.iraf_onedspec import Spectrum1D, scopy
    return scopy(Spectrum1D(np.zeros(n, dtype=np.float32), ua)).header


def ua_shift(ua_c: IrafHeader, n: int, shift: float) -> IrafHeader:
    """``PySpecshift`` (``warp/ccwaveshift.py``) header.

    ``scopy(c, tmp)``; ``specshift(tmp, shift)`` moves ``W`` by ``shift``;
    ``scombine(tmp, cs, w1=1., dw=CDELT1, nw=NAXIS1)`` rebuilds the WCS
    (``units=Angstroms``) and adds ``NCOMBINE``.
    """
    from decanter.wavelength.iraf_onedspec import save_equispec
    tmp = ua_scopy_copy(ua_c, n)
    mw = Mwcs1D.from_header(tmp)
    if shift != 0.0:
        from decanter.wavelength.iraf_onedspec import smw_nw
        save_equispec(tmp, n, w1=mw.W + iraf_ctod(pyraf_real_text(shift)), dw=mw.CD,
                      nw=smw_nw(mw, n), z=0.0, ltm=mw.LTM, ltv=mw.LTV)
    cdelt1 = float(num_text(ua_c, "CDELT1"))      # astropy's float of the header text
    dw = iraf_ctod(pyraf_real_text(cdelt1))
    from decanter.wavelength.iraf_scombine import _output_header
    from decanter.wavelength.iraf_onedspec import Spectrum1D
    w1 = 1.0
    return _output_header(Spectrum1D(np.zeros(n, dtype=np.float32), tmp), w1, dw, n, 0, 1)


def ua_refspec(ua: IrafHeader, refspec: str) -> IrafHeader:
    """``dispcor_single``'s ``hedit REFSPEC1`` on its input file."""
    h = ua.copy()
    put_str(h, "REFSPEC1", refspec)
    return h


def ua_dispcor(ua_in: IrafHeader, crval1_text: str, cdelt1_text: str) -> IrafHeader:
    """``dispcor(linear=yes)`` output + ``hedit AIRORVAC vac``.

    ``REFSPEC1`` becomes ``DCLOG2 = 'REFSPEC1 = <ref>'``; the linear WCS
    (:func:`decanter.wavelength.iraf_wcs.dispcor_wcs` text) has an identity
    Lterm, so ``LTV1`` goes.
    """
    h = ua_in.copy()
    ref = h.get("REFSPEC1")
    delete(h, "REFSPEC1")
    put_str(h, "WAT1_001", "wtype=linear label=Wavelength units=angstroms")
    put_num(h, "CRVAL1", crval1_text)
    put_num(h, "CRPIX1", "1.")
    put_num(h, "CDELT1", cdelt1_text)
    put_num(h, "CD1_1", cdelt1_text)
    delete(h, "LTV1")
    put_num(h, "LTM1_1", "1.")
    if ref is not None:
        put_str(h, "DCLOG2", f"REFSPEC1 = {ref}")
    put_str(h, "AIRORVAC", "vac")
    return h


def ua_fsr(ua_csw: IrafHeader, sc: ScopyWCS) -> IrafHeader:
    """``cut_1dspec``: ``scopy(csw, VAC, w1=, w2=)`` with the precomputed WCS."""
    h = ua_csw.copy()
    _sa_move_bandid(h)
    _apply_scopy_wcs(h, sc)
    return h


__all__ = ["ua_ssc", "ua_flat", "ua_fixpix", "ua_dispaxis", "ua_cut", "ua_trans",
           "ua_apall", "apall_dispersion", "ua_truncate", "ua_shift", "ua_refspec",
           "ua_dispcor", "ua_fsr", "ua_scopy_copy", "ua_scopy_range", "BANDID_NONE"]

_ = dtoc3


# ---------------------------------------------------------------------------
# Extract 2d spectrum: apall format=strip and the 2-D (multi-aperture) chain
# ---------------------------------------------------------------------------

def _set_title(h: IrafHeader, title: str) -> None:
    """The image title (``IM_TITLE``), which the FITS kernel writes as ``OBJECT``."""
    from astropy.io import fits as _afits
    h.put_image("OBJECT", _afits.Card("OBJECT", title, "Name of the object observed").image)


def apnum_nd(i: int) -> str:
    """``APNUM<i>`` of line ``i`` of an NDSPEC image converted to EQUISPEC:
    ``smw_gwattrs`` (ND) gives aperture ``i``, beam 0, limits ``(i, i)`` and
    ``(1, 1)`` (smwgwattrs.x:46-60), printed ``%d %d %.2f %.2f %.2f %.2f``."""
    return f"{i} 0 {dtoc3(float(i), 2, 'f')} {dtoc3(float(i), 2, 'f')} 1.00 1.00"


def apnum_ms(text: str) -> str:
    """An ``APNUM`` after a task that builds its output as MULTISPEC
    (``scombine``, ``dispcor``): only the first limit pair survives
    (``smw_gwattrs`` MULTISPEC sets ``aplow[2] = INDEF``), ``%d %d %.2f %.2f``."""
    toks = str(text).split()
    s = f"{int(toks[0])} {int(toks[1])}"
    if len(toks) >= 4:
        lo, hi = (np.float32(iraf_ctod(v)) for v in toks[2:4])
        s += f" {dtoc3(float(lo), 2, 'f')} {dtoc3(float(hi), 2, 'f')}"
    return s


def put_apnums(h: IrafHeader, texts: list[str]) -> None:
    """``smw_saveim`` (EQUISPEC): ``APNUM1..N`` in place or appended, then the
    ``APNUM``/``APID`` cards of lines beyond ``N`` deleted (smwsaveim.x:97-140)."""
    from decanter.wavelength.iraf_onedspec import put_str as _ps
    for i, t in enumerate(texts, start=1):
        _ps(h, f"APNUM{i}", t)
    i = len(texts) + 1
    while f"APNUM{i}" in h:
        delete(h, f"APNUM{i}")
        delete(h, f"APID{i}")
        i += 1


def _apnums(h: IrafHeader, nlines: int) -> list[str]:
    return [str(h.get(f"APNUM{i}")) for i in range(1, nlines + 1)]


def ua_strip(ua_trans_h: IrafHeader, nrows: int, apnum: str, order: int) -> IrafHeader:
    """``apall format=strip`` output header (apextract.x ``ap_output`` STRIP +
    ``apmw.x:apmw_saveim``): the title gets `` - Aperture <m>``, ``APNUM1``,
    ``DC-FLAG``, ``DISPAXIS`` re-added as 1, and the ``linear`` 2-D WCS with
    the dispersion on axis 1 (``W = (W1, 0)``, ``R = (1, 0)``, ``CD = diag(DW,
    1)``; ``mw_saveim``: zero values drop their card, values ``fp_equald`` to
    the old ones keep the old text)."""
    w1, dw = apall_dispersion(num_text(ua_trans_h, "CRVAL2"), num_text(ua_trans_h, "CD2_2"),
                              nrows)
    h = ua_trans_h.copy()
    _set_title(h, f"{h.get('OBJECT')} - Aperture {order}")
    put_str(h, "APNUM1", apnum)
    if "DC-FLAG" in h:
        put_num(h, "DC-FLAG", str(int(h["DC-FLAG"])))
    delete(h, "DISPAXIS")
    put_num(h, "DISPAXIS", "1")
    put_str(h, "WAT0_001", "system=linear")
    put_str(h, "WAT1_001", "wtype=linear label=Wavelength units=angstroms")
    put_str(h, "WAT2_001", "wtype=linear")

    def putarray(key: str, new: float) -> None:
        old_t = num_text(h, key)
        if fp_equald(new, 0.0):
            delete(h, key)
            return
        if old_t is not None and fp_equald(new, iraf_ctod(old_t)):
            return
        put_num(h, key, imputd_text(new))

    putarray("CRVAL1", w1)
    putarray("CRVAL2", 0.0)
    putarray("CRPIX1", 1.0)
    putarray("CRPIX2", 0.0)
    for key, v in (("CDELT1", dw), ("CDELT2", 1.0), ("CD1_1", dw), ("CD2_2", 1.0)):
        putarray(key, v)
    return h


def ua_truncate_2d(ua_strip_h: IrafHeader, n_in: int, nlines: int) -> IrafHeader:
    """``Spec1Dtools.truncate`` of an ``apall`` strip (an NDSPEC image):
    ``scopy w1=1. w2=2048.`` (``sa_ms``: every line an aperture, ``APNUM<i>``
    from :func:`apnum_nd`, ``system=equispec``) + the three ``hedit``."""
    h = ua_strip_h.copy()
    put_apnums(h, [apnum_nd(i) for i in range(1, nlines + 1)])
    put_str(h, "WAT0_001", "system=equispec")
    return ua_truncate(h, n_in)


def apnum_es(text: str) -> str:
    """An ``APNUM`` re-written by ``smw_saveim`` from an EQUISPEC input
    (``smw_equispec`` ctoi/ctor parse, every limit pair kept)."""
    toks = str(text).split()
    s = f"{int(toks[0])} {int(toks[1])}"
    for k in (2, 4):
        if len(toks) >= k + 2:
            lo, hi = (np.float32(iraf_ctod(v)) for v in toks[k:k + 2])
            s += f" {dtoc3(float(lo), 2, 'f')} {dtoc3(float(hi), 2, 'f')}"
    return s


def ua_copy_2d(ua: IrafHeader, n: int, nlines: int) -> IrafHeader:
    """``scopy(in, out)`` of a multi-aperture EQUISPEC image (no limits)."""
    h = ua.copy()
    put_apnums(h, [apnum_es(t) for t in _apnums(h, nlines)])
    return ua_scopy_copy(h, n)


def ua_shift_2d(ua_c: IrafHeader, n: int, shift: float, nlines: int) -> IrafHeader:
    """``PySpecshift`` of a multi-aperture image: ``scopy`` + ``specshift`` +
    ``scombine(w1=1., dw=CDELT1, nw=NAXIS1)`` (group = apertures: each line
    is combined on its own, the output built as MULTISPEC, so ``APNUM``
    keeps one limit pair, :func:`apnum_ms`)."""
    h = ua_c.copy()
    put_apnums(h, [apnum_ms(t) for t in _apnums(h, nlines)])
    return ua_shift(h, n, shift)


def ua_dispcor_2d(ua_in: IrafHeader, crval1_text: str, cdelt1_text: str,
                  nlines: int) -> IrafHeader:
    """``dispcor`` of a multi-aperture image (+ ``hedit AIRORVAC vac``): the
    MULTISPEC output keeps one ``APNUM`` limit pair (:func:`apnum_ms`)."""
    h = ua_in.copy()
    put_apnums(h, [apnum_ms(t) for t in _apnums(h, nlines)])
    return ua_dispcor(h, crval1_text, cdelt1_text)
