"""``onedspec.scombine(combine=average, group=all)`` as WARP runs it.

WARP's ``PyScombine`` (``warp/Spec1Dtools.py``) averages

* every frame's ``_fsr<cut>_VAC`` spectrum of one order into
  ``<OBJ>_sum_m<m>_fsr<cut>_VAC`` (2..28 frames), and
* every order's ``_sum_..._VAC_norm`` (``_AIR_norm``) spectrum into the
  one-dimensional ``<OBJ>_sum_fsr<cut>_VAC_norm_comb`` (``_AIR_norm_comb``).

This is ``noao/onedspec/scombine/t_scombine.x`` with the task defaults
(no rejection, scaling or weighting, ``blank = 0``, ``first = no``):
``scb_default`` builds the output grid from the inputs' REAL ``W0``/``W1``/
``WP`` (``shdr_open``), ``scb_output`` writes it as a MULTISPEC attribute
(``%.14g``, ``smw_swattrs``) that ``smw_saveim`` converts to EQUISPEC,
``scb_rebin`` maps each input onto the output pixels (``shdr_rebin``,
``interp = poly5``, skipped when the dispersion already agrees) and masks
the pixels it does not cover, and ``ic_gdatar`` + ``ic_averager`` average
the unmasked values in input order.
"""
from __future__ import annotations

import numpy as np

from decanter.utils.iminterp import resample_poly5
from decanter.utils.iraf_fmt import dtoc3
from decanter.utils.fpemu import iraf_ctod
from decanter.utils.iraf_header import IrafHeader
from decanter.utils.iraf_imio import imio_read_real
from decanter.wavelength.iraf_onedspec import (Mwcs1D, Spectrum1D, _apnum_text,
                                                _mw_saveim, delete, put_num, put_str,
                                                shdr_np)

R = np.float32
MAX_REAL = R(3.4028234663852886e+38)    # mach.h MAX_REAL as REAL
DCLINEAR = 0
D_ALL, D_NONE, D_MIX = 0, 1, 2


def _fp_equalr(x, y) -> bool:
    from decanter.utils.iminterp import _fp_equalr as f
    return bool(f(R(x), R(y)))


def _nint(x: float) -> int:
    x = float(x)
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(0.5 - x))


class _Sh:
    """The SHDR fields scombine uses (``smw/shdr.x:shdr_open``)."""

    def __init__(self, spec: Spectrum1D):
        self.spec = spec
        self.mw = Mwcs1D.from_header(spec.header)
        self.np1, self.np2, self.sn = shdr_np(self.mw, spec.n)
        # the spectrum as IRAF reads it (FITS kernel: utils/iraf_imio)
        self.sy = imio_read_real(spec.data)[self.np1 - 1:self.np1 - 1 + self.sn].copy()
        self.w0 = R(self.lw(1.0))
        self.w1 = R(self.lw(float(self.sn)))
        self.wp = R(R(self.w1 - self.w0) / R(self.sn - 1))
        self.dc = int(spec.header.get("DC-FLAG", -1))

    def lw(self, l):
        return self.mw.lw(np.asarray(l, dtype=np.float64) + self.np1 - 1)

    def wl(self, w):
        return self.mw.wl(w) - self.np1 + 1


def scb_rebin_window(c: float, b: float, npts: int) -> tuple[int, int, int]:
    """``scb_rebin``'s overlap of one input spectrum with the output.

    ``t_scombine.x:381-393``: ``c`` and ``b`` are the input's pixel edges
    ``0.5`` and ``SN + 0.5`` mapped to output logical pixels
    (``shdr_wl(shout, shdr_lw(shin, x))``, double, assigned to REAL), and
    ``a = max(1, nint(min(b, c) + 0.01))``, ``b = min(npts, nint(max(b, c) -
    0.01))`` in REAL arithmetic: an output pixel that gets less than 1 % of
    an input pixel lies outside ``a..b`` and is flagged as missing
    (t_scombine.x:396-402); where no input is left, ``ic_averager`` writes
    ``blank`` (0).

    Returns ``(a, b, lflag)``: output pixels ``a..b`` (1-based) are the
    input's data; ``lflag`` is ``D_ALL`` (nothing flagged), ``D_MIX`` or
    ``D_NONE`` (no overlap: the spectrum is not rebinned or used).
    """
    c = R(c)
    b0 = R(b)
    a = R(max(1, _nint(R(R(min(b0, c)) + R(0.01)))))
    b = R(min(npts, _nint(R(R(max(b0, c)) - R(0.01)))))
    j = int(np.trunc(float(R(b - a + R(1)))))
    if j < 1:
        return int(a), int(b), D_NONE
    return int(a), int(b), (D_MIX if j < npts else D_ALL)


def _scb_default(shs: list[_Sh]) -> tuple[int, float, float, int]:
    """``scb_default`` (first = no, w1 = w2 = dw = nw = INDEF, log = no)."""
    dtype = DCLINEAR
    if any(s.dc == -1 for s in shs):
        dtype = -1
    w1 = float(MAX_REAL)
    for s in shs:
        aux = float(s.w0) if s.wp > 0 else float(s.w1)
        if aux < w1:
            w1 = aux
    w2 = -float(MAX_REAL)
    for s in shs:
        aux = float(s.w1) if s.wp > 0 else float(s.w0)
        if aux > w2:
            w2 = aux
    dw = float(MAX_REAL)
    for s in shs:
        aux = abs(float(s.wp))
        if aux < dw:
            dw = aux
    if (w2 - w1) / dw < 0.:
        dw = -dw
    nw = int(np.trunc((w2 - w1) / dw + 0.5)) + 1
    dw = (w2 - w1) / (nw - 1)
    return dtype, w1, dw, nw


def _output_header(first: Spectrum1D, w1: float, dw: float, nw: int, dtype: int,
                   ncombine: int) -> IrafHeader:
    """``scb_output`` + ``smw_saveim`` (MULTISPEC -> EQUISPEC) + NCOMBINE."""
    h = first.header.copy()
    old = Mwcs1D.from_header(h)
    # MULTISPEC attribute text, parsed back by smw_gwattrs (ctod)
    w1t = iraf_ctod(dtoc3(w1, 14, "g"))
    dwt = iraf_ctod(dtoc3(dw, 14, "g"))
    ap = _apnum_text(h)
    if ap is not None:
        put_str(h, "APNUM1", ap)
    put_num(h, "DC-FLAG", str(int(dtype)))
    if "NP2" in h:
        delete(h, "NP2")
    if "WAT1_001" in h:
        put_str(h, "WAT1_001", "wtype=linear label=Wavelength units=Angstroms")
    _mw_saveim(h, old, W=w1t / (1 + 0.0), R=1.0, CD=dwt / (1 + 0.0), LTM=1.0, LTV=0.0)
    put_num(h, "NCOMBINE", str(int(ncombine)))
    return h


def scombine_average(specs: list[Spectrum1D]) -> Spectrum1D:
    """``scombine(input=",".join(specs), combine="average", group="all")``."""
    if not specs:
        raise ValueError("no input spectra")
    shs = [_Sh(s) for s in specs]
    dtype, w1, dw, nw = _scb_default(shs)
    h = _output_header(specs[0], w1, dw, nw, dtype, len(specs))
    out_proto = Spectrum1D(np.zeros(nw, dtype=R), h)
    shout = _Sh(out_proto)
    npts = shout.sn
    # scb_rebin
    lflag = []
    data, masks = [], []
    for s in shs:
        a, b, flag = scb_rebin_window(float(shout.wl(s.lw(0.5))),
                                      float(shout.wl(s.lw(float(s.sn) + 0.5))), npts)
        if flag == D_NONE:
            lflag.append(D_NONE)
            data.append(None)
            masks.append(None)
            continue
        lflag.append(flag)
        y = _shdr_rebin(s, shout)
        m = np.zeros(npts, dtype=bool)
        ja = a - 1
        if ja > 0:
            m[:ja] = True
        jb = npts - b
        if jb > 0:
            m[npts - jb:] = True
        data.append(y)
        masks.append(m)
    dflag = lflag[0]
    if any(f != dflag for f in lflag[1:]):
        dflag = D_MIX
    used = [i for i in range(len(shs)) if dflag == D_ALL or lflag[i] != D_NONE]
    if not used:
        dflag = D_NONE
    avg = np.zeros(npts, dtype=R)
    if dflag == D_ALL:
        s = data[used[0]].copy()
        for i in used[1:]:
            s = (s + data[i]).astype(R)
        avg = (s / R(len(used))).astype(R)
    elif dflag == D_MIX:
        acc = np.zeros(npts, dtype=R)
        cnt = np.zeros(npts, dtype=np.int64)
        for i in used:
            ok = ~masks[i]
            first = ok & (cnt == 0)
            later = ok & (cnt > 0)
            acc[first] = data[i][first]
            acc[later] = (acc[later] + data[i][later]).astype(R)
            cnt[ok] += 1
        good = cnt > 0
        avg[good] = (acc[good] / cnt[good].astype(R)).astype(R)
        avg[~good] = R(0.0)
    return Spectrum1D(avg, h)


def _shdr_rebin(sh: _Sh, shref: _Sh) -> np.ndarray:
    """``shdr_rebin(sh, shref)`` (``smw/shdr.x``, ``interp = poly5``)."""
    if (sh.dc == shref.dc and sh.dc != 2 and _fp_equalr(sh.w0, shref.w0)
            and _fp_equalr(sh.wp, shref.wp) and sh.sn == shref.sn):
        return sh.sy.copy()
    n = shref.sn
    xs = np.arange(n + 1, dtype=np.float64) + 0.5
    w = shref.lw(xs)
    x = sh.wl(w)
    return resample_poly5(sh.sy, x, flux=False, xmin=0.5, xmax=sh.sn + 0.5,
                          dispcor=False)
