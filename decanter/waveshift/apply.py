"""Stage 11 — wavelength truncate + per-frame wavelength shift.

WARP equivalent: the combination of ``Spec1Dtools.py:truncate``
(``iraf.scopy(w1=1, w2=2048)`` plus a CRVAL1/CRPIX1/LTV1 reset) and
``ccwaveshift.py:PySpecshift`` (``iraf.specshift`` flux shift) — i.e.
WARP's "1dcut → 1dcuts" two-step in
``Warp_sci.py:568, 642``.

The truncate is a wavelength-bound, pixel-aligned subset (no
resampling): from the s06-rectified strip's CRVAL1/CDELT1 it selects
the pixels whose wavelengths fall in ``[1, 2048]`` (the strip
wavelength bounds that match the comp file's pixel range), then resets
the wavelength WCS to start at λ=1. After truncate the spectrum has
the same NAXIS1 as the comp reference (4095 for HIRES-Y100 m=163),
which is what makes ``iraf.dispcor(..., dw=INDEF)`` in s12 produce a
4095-pixel output instead of 5628.

The shift is read from ``waveshift_log.npz`` (s10 output) as
``shift_average[i]`` per frame, applied uniformly to every order
of that frame. Output suffix: ``_1dcs`` (4095 pixels).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits as _astrofits

from decanter.calib.aperture import ApertureSet
from decanter.config import Config
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.listfile import parse as parse_listfile
from decanter.utils.iminterp import scombine_linear_poly5
from decanter.utils.iraf_imio import imio_read_real
from decanter.utils.iraf_scopy import scopy_wavelength_truncate

# WARP's hardcoded truncate bounds (``Spec1Dtools.py:152`` defaults
# ``p1=1., p2=2048.`` ⇒ keep input pixels whose wavelengths round to
# within [1, 2048] of the s06 strip's wavelength axis).
_TRUNC_W1 = 1.0
_TRUNC_W2 = 2048.0


def _sanitize_objname(raw: str) -> str:
    name = raw
    for ch in (" ", "'", "\"", "#", "/"):
        name = name.replace(ch, "_")
    return name


def shift_truncated(trunc_data: np.ndarray, cdelt1: float, shift_wave: float, *,
                    multispec: bool = False) -> np.ndarray:
    """WARP's ``PySpecshift`` on a truncated ``_m###c`` spectrum (CRVAL1 = 1).

    ``ccwaveshift.py:PySpecshift``: ``scopy`` (a copy), ``specshift`` (adds
    the shift to the WCS zero point and writes it to the header,
    ``noao/onedspec/t_specshift.x``), ``scombine(w1=1., dw=CDELT1,
    nw=NAXIS1)`` (re-reads the header and resamples back onto the
    unshifted grid). The shift reaches IRAF as pyraf's 12-digit real text,
    and the shifted zero point crosses the file boundary as IRAF header text,
    so both round trips are reproduced; the resample maps wavelengths to
    pixels through the MWCS transform as IRAF does.

    ``multispec``: the row is one aperture of a multi-aperture image (WARP's
    2-D spectra): ``specshift`` of such an EQUISPEC image converts it to
    MULTISPEC (``smw_swattrs`` -> ``smw_esms``) whenever the shift changes the
    zero point, so the shifted zero point first passes ``%.14g`` attribute
    text (t_specshift.x:sshift, smwswattrs.x).

    ``scombine``'s ``scb_rebin`` (t_scombine.x:381-402) flags every output
    pixel that gets less than 1 % of an input pixel as missing, and with the
    one input spectrum ``ic_averager`` writes ``blank`` (0.) there: a shift
    within 1 % of a whole number of pixels (|shift| / CDELT1 from about 0.99
    up to 1, from 1.99 up to 2, ...) blanks the edge pixel the shifted
    spectrum barely reaches (:func:`scb_coverage`).
    """
    from decanter.utils.fpemu import iraf_ctod
    from decanter.utils.iraf_fmt import dtoc3, roundtrip
    from decanter.wavelength.iraf_wcs import pyraf_real_text

    naxis1 = int(np.asarray(trunc_data).shape[0])
    shift = iraf_ctod(pyraf_real_text(shift_wave))
    w1 = 1.0 + shift
    if multispec and w1 != 1.0:
        w1 = iraf_ctod(dtoc3(w1, 14, "g"))
    crval1_in = roundtrip(w1)
    dw = iraf_ctod(pyraf_real_text(cdelt1))
    out = scombine_linear_poly5(
        imio_read_real(trunc_data),    # the input as IRAF reads it (FITS kernel)
        crval1_in=crval1_in, cdelt1_in=dw,
        w1_out=1.0, dw_out=dw, nw_out=naxis1, flux=False, mwcs=True,
    )
    lo, hi = scb_coverage(crval1_in, dw, naxis1, 1.0, dw, naxis1)
    if lo > 1 or hi < naxis1:
        out = np.array(out, dtype=np.float32)
        out[:max(0, lo - 1)] = np.float32(0.0)       # scombine's blank
        out[max(hi, 0):] = np.float32(0.0)
    return out


def scb_coverage(crval1_in: float, cdelt1_in: float, n_in: int,
                 w1_out: float, dw_out: float, n_out: int) -> tuple[int, int]:
    """The output pixels ``lo..hi`` (1-based) ``scombine`` keeps of one input.

    ``scb_rebin`` (t_scombine.x:381-402) maps the input's pixel edges ``0.5``
    and ``SN + 0.5`` to output pixels, ``shdr_wl(shout, shdr_lw(shin, x))``,
    and keeps ``nint(min + 0.01) .. nint(max - 0.01)`` in REAL arithmetic
    (:func:`~decanter.wavelength.iraf_scombine.scb_rebin_window`); every other
    output pixel is missing data.  With no overlap at all ``(1, 0)``: the
    whole output is blank.

    The input is the ``specshift`` output as ``scombine`` re-reads it: an
    EQUISPEC image (zero point ``crval1_in``, step ``cdelt1_in``, ``n_in``
    pixels, reference pixel 1, no Lterm) -- a 2-D strip too, which
    ``specshift`` makes MULTISPEC but ``smw_saveim`` writes back as EQUISPEC
    because every row got the same shift (smwsaveim.x:59-78) -- and the
    output ``scombine``'s ``w1``/``dw``/``nw`` grid, EQUISPEC as well; both
    transforms are the compiled linear MWCS ones
    (:class:`~decanter.wavelength.iraf_onedspec.Mwcs1D`).
    """
    from decanter.wavelength.iraf_onedspec import Mwcs1D
    from decanter.wavelength.iraf_scombine import D_NONE, scb_rebin_window
    mw_in = Mwcs1D(W=float(crval1_in), R=1.0, CD=float(cdelt1_in), LTM=1.0, LTV=0.0, old={})
    mw_out = Mwcs1D(W=float(w1_out), R=1.0, CD=float(dw_out), LTM=1.0, LTV=0.0, old={})
    c = float(mw_out.wl(mw_in.lw(0.5)))
    b = float(mw_out.wl(mw_in.lw(float(n_in) + 0.5)))
    lo, hi, flag = scb_rebin_window(c, b, int(n_out))
    if flag == D_NONE:
        return 1, 0
    return lo, hi


def apply_waveshift_one_order(
    data: np.ndarray,
    header: "_astrofits.Header",
    *,
    shift_wave: float = 0.0,
) -> tuple[np.ndarray, "_astrofits.Header"]:
    """Pure task function: truncate + specshift + scombine for one order.

    Mirrors WARP ``ccwaveshift.PySpecshift`` exactly:
      1. scopy truncate to [1, 2048] Å (poly5 rebin per the IRAF default).
      2. In-memory specshift: ``CRVAL1 += shift_wave``.
      3. scombine onto ``(w1=1, dw=CDELT1, nw=naxis1)``. With shift=0
         the WCS short-circuit fires and the truncate output is
         returned unchanged.

    Returns ``(shifted, header)`` where ``header`` has CRVAL1/CDELT1/
    CRPIX1/WAVESHIFT keywords set.
    """
    trunc = scopy_wavelength_truncate(
        np.asarray(data), header, w1=_TRUNC_W1, w2=_TRUNC_W2,
    )
    shifted = shift_truncated(trunc.data, float(trunc.header.get("CDELT1", 1.0)),
                              shift_wave)
    cdelt1 = float(trunc.header.get("CDELT1", 1.0))
    out_header = trunc.header.copy()
    out_header["CRVAL1"] = (1.0, "Wavelength at output pixel 1 (post-shift rebin)")
    out_header["CDELT1"] = (cdelt1, "Wavelength step per pixel")
    out_header["CRPIX1"] = (1.0, "Reference pixel along dispersion")
    out_header["WAVESHIFT"] = (shift_wave, "Applied wavelength shift")
    return shifted.astype(np.float32), out_header


def run(
    config: Config,
    workdir: Path,
    listfile: Path,
    *,
    apdb_path: Path | str | None = None,
    **_unused: Any,
) -> None:
    """Truncate + apply per-frame wavelength shift to every order spectrum.

    Args:
        config: gated on ``flag_wscorrect`` (default True).
        workdir: reads ``{obj}_NO{i}_sscfm_m{m}_1d.fits`` (s08) and
            ``waveshift_log.npz`` (s10); writes
            ``{obj}_NO{i}_sscfm_m{m}_1dcs.fits`` (4095 pixels for HIRES-Y).
        listfile: WARP-style input list.
        apdb_path: aperture database for order enumeration.
    """
    if not config.flag_wscorrect:
        return
    if apdb_path is None:
        raise ValueError("s11_waveshift_apply requires apdb_path until s00 wires in calib")

    ws_path = workdir / "waveshift_log.npz"
    if not ws_path.exists():
        shift_average = None
    else:
        ws = np.load(ws_path, allow_pickle=True)
        shift_average = np.asarray(ws["shift_average"], dtype=np.float64)

    cal_apset = ApertureSet.load(apdb_path)
    orders = (
        cal_apset.echelle_orders
        if (config.reduce_full_data or not config.selected_orders)
        else tuple(m for m in cal_apset.echelle_orders if m in config.selected_orders)
    )

    pairs = parse_listfile(listfile)
    for i, pair in enumerate(pairs, start=1):
        _obj_raw, obj_header = _fits.read_image(workdir / pair.object_name)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objname = _sanitize_objname(str(raw_objname))
        shift_wave = float(shift_average[i - 1]) if shift_average is not None else 0.0

        for m in orders:
            # --- OBJ path: truncate + shift ------------------------------------
            in_path = workdir / f"{objname}_NO{i}_sscfm_m{m}_1d.fits"
            if in_path.exists():
                data, header = _fits.read_image(in_path)
                shifted, out_header = apply_waveshift_one_order(
                    data, header, shift_wave=shift_wave,
                )
                out_path = workdir / f"{objname}_NO{i}_sscfm_m{m}_1dcs.fits"
                _fits.write_image(out_path, shifted, out_header, overwrite=True)

            # --- SKY path: truncate only, no shift -----------------------------
            # WARP ``Warp_sci.py:590`` calls truncate(sky_fm_trans_1d, sky_fm_trans_1dcut)
            # then ``Warp_sci.py:690`` runs dispcor on ``sky_fm_trans_1dcut``
            # directly — no specshift in between. So the sky path skips
            # the waveshift correction.
            if config.flag_skyemission:
                sky_in_path = (
                    workdir / f"{objname}_skyNO{i}_fm_m{m}trans1d.fits"
                )
                if sky_in_path.exists():
                    sky_data, sky_header = _fits.read_image(sky_in_path)
                    sky_trunc = scopy_wavelength_truncate(
                        np.asarray(sky_data), sky_header,
                        w1=_TRUNC_W1, w2=_TRUNC_W2,
                    )
                    sky_out_path = (
                        workdir / f"{objname}_skyNO{i}_fm_m{m}trans1dcut.fits"
                    )
                    _fits.write_image(
                        sky_out_path,
                        sky_trunc.data.astype(np.float32),
                        sky_trunc.header,
                        overwrite=True,
                    )
