"""WARP's post-FSR 1-D steps on EQUISPEC spectra, as IRAF computes them.

After the FSR cut (``_fsr<cut>_VAC``) WARP runs, per spectrum
(``Warp_sci.py`` 1-D block):

* ``PyContinuum`` -> ``onedspec.continuum`` (``t_sfit.x``, ``type=ratio``)
  -> ``_VAC_norm``, then ``sarith(VAC, "/", VAC_norm)`` -> ``_VAC_cont``;
* ``vac2air_spec`` -> ``onedspec.dopcor(redshift=-v, isveloc=yes)`` plus
  ``hedit VAC2AIR``/``AIRORVAC`` -> ``_AIR``, ``_AIR_norm``, ``_AIR_cont``
  (sky: ``_AIR`` only);
* ``scopy`` / ``scombine`` for the ``<OBJ>_sum`` spectra
  (:mod:`decanter.wavelength.iraf_scombine`).

Every step works on a :class:`Spectrum1D` -- float32 data plus an astropy
header whose numeric WCS cards carry IRAF's text -- and reproduces both the
data and the header text: the SMW/MWCS logical <-> physical <-> world
transforms (``smw/shdr.x``, ``sys/mwcs/mwsctran.x``, fused ``ltm * x + ltv``
in ``mw_c1tran``), the EQUISPEC rewrite (``smw/smwsaveim.x``) and
``mw_saveim``'s card update rules (``sys/mwcs/iwparray.x``: a value
``fp_equald`` to the old one keeps the old text, a value ``fp_equald`` to 0
drops the card, anything else is written with ``imputd``).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import scipy.constants
from astropy.io import fits as _afits
from numpy.typing import NDArray

from decanter.utils.fpemu import fma32, fma64, iraf_ctod
from decanter.utils.iraf_fmt import _dt_equald, dtoc3, imputd_text
from decanter.utils.iraf_header import IrafHeader, as_header
from decanter.utils.iraf_imio import imio_read_real

R = np.float32
VLIGHT = 2.997925e5          # t_dopcor.x (km/s), exact in float32


# ---------------------------------------------------------------------------
# Header text helpers
# ---------------------------------------------------------------------------

def num_text(h, key: str) -> str | None:
    """The value text of numeric card ``key`` (None when absent)."""
    return as_header(h).text(key)


def num_card(key: str, text: str) -> _afits.Card:
    """A numeric card carrying IRAF's text verbatim (value field cols 11-30)."""
    return _afits.Card.fromstring(f"{key:<8s}= {text:>20s}")


def put_num(h: IrafHeader, key: str, text: str) -> None:
    """``imputd``/``imputi`` text: replace the card in place or append it."""
    h.put_num(key, text)


def put_str(h: IrafHeader, key: str, value: str) -> None:
    """``imastr``/``impstr``: replace the string value in place or append."""
    h.put_str(key, value)


def delete(h: IrafHeader, key: str) -> None:
    h.delete(key)


def fp_equald(x: float, y: float) -> bool:
    """IRAF ``fp_equald`` (``sys/osb/fpequald.x``)."""
    return _dt_equald(float(x), float(y))


def imaddr_text(value: float) -> str:
    """``imputr``: ``%0.7g`` of a REAL value (``imio/db/imputr.x``)."""
    return dtoc3(float(R(value)), 7, "g")


def iraf_g_width(value: float, width: int) -> str:
    """SPP ``%<width>g`` of a double (``fmtio/parg.x``: precision lowered
    from 16 digits until it fits), right-justified."""
    s = ""
    for dec in range(16, 0, -1):
        s = dtoc3(float(value), dec, "g")
        if len(s) <= width:
            break
    return f"{s:>{width}s}"


# ---------------------------------------------------------------------------
# The spectrum and its MWCS
# ---------------------------------------------------------------------------

@dataclass
class Spectrum1D:
    """A 1-D EQUISPEC spectrum as IRAF holds it: float32 data + header
    (an :class:`~decanter.utils.iraf_header.IrafHeader`; an astropy header
    is converted)."""

    data: NDArray[np.float32]
    header: IrafHeader

    def __post_init__(self):
        if not isinstance(self.header, IrafHeader):
            self.header = IrafHeader.from_astropy(self.header)

    def copy(self) -> "Spectrum1D":
        return Spectrum1D(np.array(self.data, dtype=R), self.header.copy())

    @property
    def n(self) -> int:
        return int(np.asarray(self.data).size)


@dataclass(frozen=True)
class Mwcs1D:
    """The MWCS of a 1-D image (``iw_enterwcs``, ``sys/mwcs/iwewcs.x``)."""

    W: float           # CRVAL1
    R: float           # physical reference pixel inv(LTM) * (CRPIX - LTV)
    CD: float          # physical CD = CD' * LTM
    LTM: float
    LTV: float
    old: dict          # header values as read (ctod of the text)

    @classmethod
    def from_header(cls, h) -> "Mwcs1D":
        h = as_header(h)

        def val(key, default=None):
            t = num_text(h, key)
            return default if t is None else iraf_ctod(t)
        W = val("CRVAL1", 0.0)
        crpix = val("CRPIX1", 0.0)
        cd_l = val("CD1_1")
        if cd_l is None:
            cd_l = val("CDELT1", 1.0)
        ltm_t = val("LTM1_1")
        ltv_t = val("LTV1")
        ltm = 1.0 if ltm_t is None else ltm_t
        ltv = 0.0 if ltv_t is None else ltv_t
        cd = cd_l * ltm if ltm_t is not None else cd_l
        if ltm_t is not None or ltv_t is not None:
            o_r = crpix - ltv
            r = (1.0 / ltm) * o_r if ltm_t is not None else o_r
        else:
            r = crpix
        old = {"CRVAL1": W, "CRPIX1": crpix, "CD1_1": cd_l, "LTM1_1": ltm, "LTV1": ltv,
               "CDELT1": val("CDELT1")}
        return cls(W=W, R=r, CD=cd, LTM=ltm, LTV=ltv, old=old)

    # logical -> world, double (mw_sctran "logical" -> "world", mw_c1trand)
    def lw_terms(self) -> tuple[float, float]:
        iltm = 1.0 / self.LTM
        iltv = -(iltm * self.LTV)
        oltv = self.W + self.CD * (-self.R)
        tltm = self.CD * iltm
        tltv = oltv + self.CD * iltv
        return tltm, tltv

    def lw(self, l) -> NDArray[np.float64]:
        tltm, tltv = self.lw_terms()
        return fma64(tltm, np.asarray(l, dtype=np.float64), tltv)

    # world -> logical, double
    def wl_terms(self) -> tuple[float, float]:
        icd = 1.0 / self.CD
        iltv = self.R - icd * self.W
        oltv = self.LTV + self.LTM * 0.0
        tltm = self.LTM * icd
        tltv = oltv + self.LTM * iltv
        return tltm, tltv

    def wl(self, w) -> NDArray[np.float64]:
        tltm, tltv = self.wl_terms()
        return fma64(tltm, np.asarray(w, dtype=np.float64), tltv)

    # logical <-> physical, REAL (mw_c1tranr)
    def lp_real(self, l: float) -> np.float32:
        iltm = 1.0 / self.LTM
        iltv = -(iltm * self.LTV)
        return R(fma32(np.float32(iltm), np.float32(l), np.float32(iltv)))

    def pl_real(self, p: float) -> np.float32:
        return R(fma32(np.float32(self.LTM), np.float32(p), np.float32(self.LTV)))


def smw_nw(mw: Mwcs1D, n: int) -> int:
    """``smw_equispec``: physical pixels ``nw = max(l->p(1), l->p(N))``."""
    return int(np.trunc(max(float(mw.lp_real(1.0)), float(mw.lp_real(float(n))))))


def shdr_np(mw: Mwcs1D, n: int) -> tuple[int, int, int]:
    """``shdr_open``: ``(NP1, NP2, SN)`` of a 1-D EQUISPEC spectrum."""
    nw = smw_nw(mw, n)
    i = max(1, min(int(np.trunc(float(mw.pl_real(1.0)))), n))
    j = max(1, min(int(np.trunc(float(mw.pl_real(float(nw))))), n))
    np1, np2 = min(i, j), max(i, j)
    return np1, np2, np2 - np1 + 1


def shdr_sx(header: IrafHeader, n: int | None = None):
    """``shdr_open`` SX: the REAL wavelengths of the ``SN`` points, and ``SN``."""
    if n is None:
        n = int(header["NAXIS1"])
    mw = Mwcs1D.from_header(header)
    np1, np2, sn = shdr_np(mw, n)
    l0 = np.arange(1, sn + 1, dtype=np.float64) + np1 - 1
    return mw.lw(l0).astype(R), sn


# ---------------------------------------------------------------------------
# smw_saveim (EQUISPEC) + mw_saveim
# ---------------------------------------------------------------------------

def _apnum_text(h: IrafHeader) -> str | None:
    """``smw_saveim`` APNUM1 from ``smw_equispec``'s ctoi/ctor parse."""
    t = h.get("APNUM1")
    if t is None:
        return None
    toks = str(t).split()
    ap, beam = int(toks[0]), int(toks[1])
    s = f"{ap} {beam}"
    lims = [R(iraf_ctod(v)) for v in toks[2:4]]
    if len(lims) == 2:
        s += f" {dtoc3(float(lims[0]), 2, 'f')} {dtoc3(float(lims[1]), 2, 'f')}"
        if len(toks) >= 6:
            l2 = [R(iraf_ctod(v)) for v in toks[4:6]]
            s += f" {dtoc3(float(l2[0]), 2, 'f')} {dtoc3(float(l2[1]), 2, 'f')}"
    return s


def save_equispec(h: IrafHeader, n_out: int, *, w1: float, dw: float, nw: int,
                  z: float, ltm: float, ltv: float) -> None:
    """``smw_saveim`` (EQUISPEC) of a 1-D spectrum into header ``h``.

    ``w1``/``dw``/``nw``/``z`` are the ``smw_swattrs`` values; the new Wterm
    has ``R = 1``, ``W = w1 / (1 + z)``, ``CD = dw / (1 + z)``; the Lterm is
    ``(ltm, ltv)``.  ``h`` holds the old cards (NEW_COPY of the input), which
    decide what text is kept.
    """
    old = Mwcs1D.from_header(h)
    ap = _apnum_text(h)
    if ap is not None:
        put_str(h, "APNUM1", ap)
    if "DC-FLAG" in h:
        put_num(h, "DC-FLAG", str(int(h["DC-FLAG"])))
    if nw < n_out:
        put_num(h, "NP2", str(int(nw)))
    elif "NP2" in h:
        delete(h, "NP2")
    W = w1 / (1 + z)
    CD = dw / (1 + z)
    r = 1.0
    _mw_saveim(h, old, W=W, R=r, CD=CD, LTM=ltm, LTV=ltv)


def _mw_saveim(h: IrafHeader, old: Mwcs1D, *, W: float, R: float, CD: float,
               LTM: float, LTV: float) -> None:
    """``mw_saveim`` for a 1-D linear WCS (``sys/mwcs/mwsaveim.x``)."""
    have = {k: num_text(h, k) is not None for k in ("CRVAL1", "CRPIX1", "CD1_1",
                                                     "CDELT1", "LTV1", "LTM1_1")}
    updated = set()

    def putarray(key: str, new: float, oldv: float) -> None:
        if fp_equald(new, 0.0):
            return
        if fp_equald(new, oldv):
            if have[key]:
                updated.add(key)
            return
        put_num(h, key, imputd_text(new))
        updated.add(key)

    putarray("CRVAL1", W, old.old["CRVAL1"])
    n_r = LTV + float(fma64(LTM, R, 0.0))
    putarray("CRPIX1", n_r, old.old["CRPIX1"])
    n_cd = float(fma64(CD, 1.0 / LTM, 0.0))
    oc = old.old["CDELT1"]
    if not have["CDELT1"] or not fp_equald(oc, n_cd):
        put_num(h, "CDELT1", imputd_text(n_cd))
    updated.add("CDELT1")
    putarray("CD1_1", n_cd, old.old["CD1_1"])
    putarray("LTV1", LTV, old.old["LTV1"])
    putarray("LTM1_1", LTM, old.old["LTM1_1"])
    for key, present in have.items():
        if present and key not in updated:
            delete(h, key)


# ---------------------------------------------------------------------------
# dopcor (vac2air_spec)
# ---------------------------------------------------------------------------

def _openspecfits_x(h: IrafHeader, n: int) -> NDArray[np.float64]:
    """``Spec1Dtools.openspecfits`` wavelength list (Python floats)."""
    c = float(h["CRVAL1"])
    d = float(h["CDELT1"])
    p = float(h["CRPIX1"])
    return np.array([c + d * (l - p + 1.) for l in range(n)])


def vac2air_shift(h: IrafHeader, n: int) -> float:
    """``vac2air_spec``: ``vshift`` (km/s) for the spectrum with header ``h``."""
    spx = _openspecfits_x(h, n)
    vac = np.average(spx)
    s = pow(10, 4) / vac
    nref = 1.0 + 5.792105e-2 / (238.0185 - s ** 2) + 1.67917e-3 / (57.362 - s ** 2)
    return (1. / nref - 1) * (scipy.constants.c / 1000.)


def _ms_text(v: float) -> float:
    """A double through a MULTISPEC ``spec<i>`` attribute (``%.14g`` text,
    re-read with ``ctod``; smwswattrs.x, smwesms.x, smwgwattrs.x)."""
    return iraf_ctod(dtoc3(float(v), 14, "g"))


def dopcor(spec: Spectrum1D, redshift_text: str, *,
           multispec: bool = False) -> tuple[Spectrum1D, float]:
    """``onedspec.dopcor(input, output, redshift=<text>, isveloc=yes)``.

    Returns the output spectrum and ``zvel`` (the parsed velocity).
    ``spec.data`` may be 2-D (one row per aperture of an EQUISPEC image);
    ``multispec=True`` is such an image with more than one aperture: the
    first ``smw_swattrs`` with the new redshift turns its WCS into MULTISPEC
    (``smw_esms``), so ``W``, ``CD`` and ``z`` reach the saved EQUISPEC WCS
    through ``%.14g`` attribute text.
    IRAF equivalent: ``noao/onedspec/t_dopcor.x`` (EQUISPEC branch).
    """
    h = spec.header.copy()
    data = imio_read_real(spec.data)   # as IRAF reads it (FITS kernel)
    n = int(data.shape[-1])
    mw = Mwcs1D.from_header(h)
    zvel = iraf_ctod(redshift_text)
    z = zvel / VLIGHT
    if abs(z) >= 1.0:
        raise ValueError("Impossible velocity")
    z = float(np.sqrt((1 + z) / (1 - z)) - 1)
    np1, np2, sn = shdr_np(mw, n)
    out = np.empty(data.shape, dtype=R)
    out[..., :sn] = data[..., np1 - 1:np1 - 1 + sn]
    if n > sn:
        out[..., sn:] = out[..., sn - 1:sn]
    nw = smw_nw(mw, n)
    # "Document header" (t_dopcor.x:209-232) comes before smw_saveim
    # (t_dopcor.x:267-268): DOPCORnn is added first, so a card smw_saveim
    # appends (NP2 when nw < NAXIS1, a WCS keyword the input lacks) follows it.
    key = "DOPCOR01"
    for i in range(1, 99):
        key = f"DOPCOR{i:02d}"
        if key not in h:
            break
    put_str(h, key, f"{iraf_g_width(zvel, 8)} all")
    if multispec:
        save_equispec(h, n, w1=_ms_text(mw.W), dw=_ms_text(mw.CD), nw=nw, z=_ms_text(z),
                      ltm=mw.LTM, ltv=mw.LTV)
    else:
        save_equispec(h, n, w1=mw.W, dw=mw.CD, nw=nw, z=z, ltm=mw.LTM, ltv=mw.LTV)
    return Spectrum1D(out, h), zvel


def vac2air(spec: Spectrum1D, *, multispec: bool = False) -> Spectrum1D:
    """``warp/vac2air_spec.py:vac2air_spec`` (dopcor + two ``hedit``);
    ``multispec`` as in :func:`dopcor` (a 2-D spectrum of several rows)."""
    vshift = vac2air_shift(spec.header, int(np.asarray(spec.data).shape[-1]))
    out, _ = dopcor(spec, str(-vshift), multispec=multispec)
    h = out.header
    # hedit VAC2AIR <str(vshift)>: a numeric literal -> evexpr REAL -> imaddr
    put_num(h, "VAC2AIR", imaddr_text(R(iraf_ctod(str(vshift)))))
    put_str(h, "AIRORVAC", "air")
    return out


# ---------------------------------------------------------------------------
# continuum (sfit, type=ratio) and sarith "/"
# ---------------------------------------------------------------------------

def continuum_params(order_m: int) -> dict:
    """``Warp_sci.py``: PyContinuum parameters by echelle order."""
    if order_m < 100:
        return dict(order=15, niterate=10)
    return dict(order=7, niterate=2)


def continuum(spec: Spectrum1D, *, order: int, niterate: int, low: float = 2.0,
              high: float = 3.0, grow: float = 1.0, out_type: str = "ratio") -> Spectrum1D:
    """``onedspec.continuum(type=ratio, sample=*, function=spline3)``.

    The ``SN`` points ``shdr_open`` reads are fitted against their REAL
    wavelengths; the output (a NEW_COPY of the input) holds the ratio there
    and 0 beyond (``sft_icfit``: ``aclrr`` when ``SN < n``).  ``SFIT`` and
    ``SFITB`` record the fitted line and band.
    """
    from decanter.utils.sfit_exact import sfit_ratio
    h = spec.header.copy()
    n = spec.n
    mw = Mwcs1D.from_header(h)
    np1, np2, sn = shdr_np(mw, n)
    sx = mw.lw(np.arange(1, sn + 1, dtype=np.float64) + np1 - 1).astype(R)
    sy = imio_read_real(spec.data)[np1 - 1:np1 - 1 + sn]
    ratio, cv = sfit_ratio(sx, sy, order=order, low=low, high=high, niterate=niterate,
                           grow=grow)
    if out_type == "difference":
        from decanter.utils.sfit_exact import cvvector
        ratio = (sy - cvvector(cv, sx)).astype(R)
    elif out_type != "ratio":
        raise NotImplementedError(f"continuum type={out_type!r}")
    out = np.zeros(n, dtype=R)
    out[np1 - 1:np1 - 1 + sn] = ratio
    put_str(h, "SFIT", "1")
    put_str(h, "SFITB", "1")
    return Spectrum1D(out, h)


def continuum_many(specs: list, params: list) -> list:
    """:func:`continuum` of several spectra, bit for bit, with the fits batched
    (:func:`decanter.utils.sfit_exact.sfit_ratio_many`).

    ``params`` holds one dict of :func:`continuum`'s keyword arguments
    (``order``, ``niterate`` and optionally ``low``, ``high``, ``grow``,
    ``out_type``) per spectrum.
    """
    from decanter.utils.sfit_exact import cvvector, sfit_ratio_many
    prep = []
    for spec, prm in zip(specs, params, strict=True):
        out_type = prm.get("out_type", "ratio")
        if out_type not in ("ratio", "difference"):
            raise NotImplementedError(f"continuum type={out_type!r}")
        h = spec.header.copy()
        n = spec.n
        mw = Mwcs1D.from_header(h)
        np1, _np2, sn = shdr_np(mw, n)
        sx = mw.lw(np.arange(1, sn + 1, dtype=np.float64) + np1 - 1).astype(R)
        sy = imio_read_real(spec.data)[np1 - 1:np1 - 1 + sn]
        prep.append((h, n, np1, sn, sx, sy, out_type))
    fits = sfit_ratio_many(
        [q[4] for q in prep], [q[5] for q in prep],
        [dict(order=prm["order"], niterate=prm["niterate"], low=prm.get("low", 2.0),
              high=prm.get("high", 3.0), grow=prm.get("grow", 1.0)) for prm in params])
    outs = []
    for (h, n, np1, sn, sx, sy, out_type), (ratio, cv) in zip(prep, fits, strict=True):
        if out_type == "difference":
            ratio = (sy - cvvector(cv, sx)).astype(R)
        out = np.zeros(n, dtype=R)
        out[np1 - 1:np1 - 1 + sn] = ratio
        put_str(h, "SFIT", "1")
        put_str(h, "SFITB", "1")
        outs.append(Spectrum1D(out, h))
    return outs


def sa_1d(spec: Spectrum1D, op: str = "copy", other: Spectrum1D | str | None = None
          ) -> Spectrum1D:
    """``sarith`` / ``scopy`` of a 1-D spectrum with ``w1 = w2 = INDEF``.

    ``op`` is ``"copy"`` (``onedspec.scopy(in, out)``) or ``"/"``
    (``sarith(in, "/", other, out)``, ``advzr`` with ``errval`` = 0; the
    operand must already share the dispersion, so ``shdr_rebin`` is a
    no-op).  The output holds the ``SN`` points ``shdr_open`` reads.

    IRAF equivalent: ``noao/onedspec/t_sarith.x:sa_1d`` + ``sa_arith``
    (``scopy.cl`` runs ``sarith ... copy``).
    """
    h = spec.header.copy()
    n = spec.n
    mw = Mwcs1D.from_header(h)
    np1, np2, sn = shdr_np(mw, n)
    a = imio_read_real(spec.data)[np1 - 1:np1 - 1 + sn]   # as IRAF reads it
    if op == "copy":
        q = a.copy()
    elif op in ("/", "-") and isinstance(other, str):
        # numeric second operand: ctod, then amovkr(real(w))
        c = R(iraf_ctod(other))
        if op == "-":
            q = (a - c).astype(R)
        else:
            small = (c < R(1e-20)) and (c > R(-1e-20))
            q = np.zeros_like(a) if small else (a / c).astype(R)
    elif op in ("/", "-"):
        if other is None:
            raise ValueError(f"sarith {op!r} needs a second operand")
        mw2 = Mwcs1D.from_header(other.header)
        np1b, _, sn2 = shdr_np(mw2, other.n)
        sx1 = mw.lw(np.array([1.0, float(sn)]) + np1 - 1).astype(R)
        sx2 = mw2.lw(np.array([1.0, float(sn2)]) + np1b - 1).astype(R)
        if sn != sn2 or sx1[0] != sx2[0] or sx1[1] != sx2[1]:
            raise NotImplementedError("sarith: operand needs rebinning")
        b = imio_read_real(other.data)[np1b - 1:np1b - 1 + sn]
        if op == "-":
            q = (a - b).astype(R)
        else:
            small = (b < R(1e-20)) & (b > R(-1e-20))
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where(small, R(0.0), a / np.where(small, R(1.0), b)).astype(R)
    else:
        raise NotImplementedError(f"sarith op {op!r}")
    n_out = sn + np1 - 1
    out = np.zeros(n_out, dtype=R)
    out[np1 - 1:] = q
    # header: NEW_COPY, DISPAXIS/APID*/BANDID* deleted, BANDID1 re-added
    band = h.get("BANDID1")
    for key in list(h.keys()):
        if key == "DISPAXIS" or key.startswith("APID") or key.startswith("BANDID"):
            delete(h, key)
    if band is not None:
        put_str(h, "BANDID1", str(band))
    _sa_1d_wcs(h, mw, n_out, np1, np2, sn, l1=1.0, dl=1.0)
    return Spectrum1D(out, h)


def sa_1d_const(spec: Spectrum1D, op: str, value_text: str) -> Spectrum1D:
    """``sarith(spec, op, <number>)`` (the operand text as pyraf passes it)."""
    return sa_1d(spec, op, value_text)


def sarith_divide(spec: Spectrum1D, other: Spectrum1D) -> Spectrum1D:
    """``sarith(spec, "/", other)`` (:func:`sa_1d`)."""
    return sa_1d(spec, "/", other)


def scopy(spec: Spectrum1D) -> Spectrum1D:
    """``onedspec.scopy(spec, out)`` with no wavelength limits (:func:`sa_1d`)."""
    return sa_1d(spec, "copy")


def _sa_1d_wcs(h: IrafHeader, mw: Mwcs1D, n_out: int, np1: int, np2: int, sn: int,
               *, l1: float, dl: float) -> None:
    """``sa_1d``'s output WCS (Lterm from the extraction, linear Wterm)."""
    ltv2 = (mw.LTV - l1) / dl + 1
    ltm2 = dl * mw.LTM
    w = float(mw.lw(1.0 + np1 - 1))
    wb = float(mw.lw(float(sn) + np1 - 1))
    p1 = (np1 - ltv2) / ltm2
    p2 = (np2 - ltv2) / ltm2
    p3 = (n_out - ltv2) / ltm2
    nw = _nint(min(max(p1, p3), max(p1, p2))) + np1 - 1
    z = 0.0
    w = w * (1 + z)
    wb = wb * (1 + z)
    dw = mw.CD
    if p1 != p2:
        dw = (wb - w) / (p2 - p1)
    w = float(fma64(-(p1 - 1), dw, w))
    save_equispec(h, n_out, w1=w, dw=dw, nw=nw, z=z, ltm=ltm2, ltv=ltv2)


def _nint(x: float) -> int:
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(0.5 - x))


__all__ = ["Spectrum1D", "Mwcs1D", "shdr_np", "shdr_sx", "dopcor", "vac2air",
           "continuum", "continuum_params", "sarith_divide", "scopy", "sa_1d", "sa_1d_const",
           "save_equispec"]

_ = dataclasses  # keep import (dataclasses.replace used by callers)
