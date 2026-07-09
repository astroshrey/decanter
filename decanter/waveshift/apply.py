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
    cdelt1 = float(trunc.header.get("CDELT1", 1.0))
    crval1_in = float(trunc.header.get("CRVAL1", 1.0)) + shift_wave
    naxis1 = int(trunc.data.shape[0])
    shifted = scombine_linear_poly5(
        trunc.data,
        crval1_in=crval1_in, cdelt1_in=cdelt1,
        w1_out=1.0, dw_out=cdelt1, nw_out=naxis1,
        flux=False,
    )
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
