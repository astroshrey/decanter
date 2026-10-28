"""Stage 13 — clip each order to its Free Spectral Range.

WARP equivalent: ``warp/Spec1Dtools.py:cut_1dspec`` (line 218) wrapping
``iraf.onedspec.scopy(w1=lo, w2=hi)``.

Per-order, per-cutrange truncation::

    center = (fsr_min + fsr_max) / 2
    lo = center - (center - fsr_min) * cutrange
    hi = center + (fsr_max - center) * cutrange

The cutrange list (e.g. ``(1.05, 1.30)``) controls how generously
beyond the strict FSR we keep flux. WARP's default ``cutrange_list``.

Output: ``{obj}_NO{i}_sscfm_m{m}_fsr{cutrange}_VAC.fits``.
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
from decanter.io.fsr import load as load_fsr
from decanter.io.listfile import parse as parse_listfile
from decanter.utils.fpemu import iraf_ctod
from decanter.utils.iminterp import shdr_linear
from decanter.utils.iraf_imio import imio_read_real


def _sanitize_objname(raw: str) -> str:
    name = raw
    for ch in (" ", "'", "\"", "#", "/"):
        name = name.replace(ch, "_")
    return name


def truncate_spectrum(
    flux: np.ndarray,
    crval1: float,
    cdelt1: float,
    wave_lo: float,
    wave_hi: float,
) -> tuple[np.ndarray, dict]:
    """Truncate + poly5-rebin the spectrum onto an n-pixel linear λ axis.

    Mirrors IRAF ``noao$onedspec/t_sarith.x:sa_sextract`` (line 1369)
    chained into ``shdr.x:shdr_extract`` (line 1163) and
    ``shdr.x:shdr_linear`` (line 1018). For the WARP-default
    ``iraf.scopy(rebin=yes)`` (the only mode WARP uses for FSR
    truncation), the flow is:

      1. ``l1_frac = (wave_lo - CRVAL_in) / CDELT_in + 1``
         ``l2_frac = (wave_hi - CRVAL_in) / CDELT_in + 1``
      2. Clamp to ``[1, N]``; round to integers ``i1, i2``;
         ``n = |i2 - i1| + 1``.
      3. Compute new wavelength endpoints at the **fractional** pixel
         positions: ``w0_new = wave_at(l1_frac)``, ``w1_new = wave_at(l2_frac)``.
      4. Resample input onto a fresh ``(w0_new, w1_new, n)`` linear grid
         using poly5 integral-averaging (same machinery as scombine).
      5. Output WCS uses the input's CRVAL (CRVAL_out = CRVAL_in) with
         a negative CRPIX such that pixel 1 falls at ``w0_new``.

    Returns ``(rebinned_flux, wcs_dict)`` where ``wcs_dict`` has keys
    ``crval1``, ``cdelt1``, ``crpix1``, ``ltm1_1``, ``ltv1`` for the
    full IRAF physical-vs-logical transform.
    """
    n_in = flux.size
    if cdelt1 == 0.0:
        return flux[:0], {"crval1": wave_lo, "cdelt1": cdelt1,
                          "crpix1": 1.0, "ltm1_1": 1.0, "ltv1": 0.0}
    # WARP passes the bounds through pyraf as ``str(float)``; IRAF's clgetd
    # parses them with ``ctod``.
    wa = iraf_ctod(repr(float(wave_lo)))
    wb = iraf_ctod(repr(float(wave_hi)))
    # sa_sextract (t_sarith.x): output length + WCS from the DOUBLE bounds.
    l1 = (wa - crval1) / cdelt1 + 1.0
    l2 = (wb - crval1) / cdelt1 + 1.0
    l1 = max(1.0, min(float(n_in), l1))
    l2 = max(1.0, min(float(n_in), l2))
    n_out = abs(int(np.floor(l2 + 0.5)) - int(np.floor(l1 + 0.5))) + 1
    # shdr_extract: the data are resampled from the bounds as REAL, which
    # can give a different pixel count n_r at a nint boundary.
    r1 = (float(np.float32(wa)) - crval1) / cdelt1 + 1.0
    r2 = (float(np.float32(wb)) - crval1) / cdelt1 + 1.0
    r1 = max(1.0, min(float(n_in), r1))
    r2 = max(1.0, min(float(n_in), r2))
    n_r = abs(int(np.floor(r2 + 0.5)) - int(np.floor(r1 + 0.5))) + 1
    w0_new = np.float32(crval1 + (r1 - 1.0) * cdelt1)
    w1_new = np.float32(crval1 + (r2 - 1.0) * cdelt1)
    rebinned = shdr_linear(flux, crval1, cdelt1, w0_new, w1_new, n_r)
    if n_r < n_out:
        # IRAF copies n_out values out of an n_r-long buffer: the tail is
        # uninitialised heap memory in WARP (not reproducible). Repeat the
        # last resampled value instead.
        rebinned = np.concatenate(
            [rebinned, np.repeat(rebinned[-1:], n_out - n_r)]).astype(np.float32)
    elif n_r > n_out:
        rebinned = rebinned[:n_out]
    if n_out > 1:
        dw_new = float(np.float32(np.float32(w1_new - w0_new) / np.float32(n_r - 1)))
        ltm1_1 = (l2 - l1) / (n_out - 1)
    else:
        dw_new = cdelt1
        ltm1_1 = 1.0
    # The resampled pixels sit on the REAL grid w0 + (x - 1) * dw that
    # shdr_linear actually used (WARP's header encodes the same mapping with
    # CRVAL = input CRVAL and a negative CRPIX).
    ltv1 = l1 - ltm1_1
    return rebinned, {
        "crval1": float(w0_new), "cdelt1": dw_new,
        "crpix1": 1.0, "ltm1_1": ltm1_1, "ltv1": ltv1,
        "w0": float(w0_new),
    }


def fsr_bounds(fsr_lo: float, fsr_hi: float, cutrange: float) -> tuple[float, float]:
    """WARP's FSR cut bounds, in WARP's float arithmetic.

    WARP equivalent: ``warp/Spec1Dtools.py:cut_1dspec``.
    """
    center = (fsr_lo + fsr_hi) / 2.
    lo = (center - (center - fsr_lo) * cutrange)
    hi = (center + (fsr_hi - center) * cutrange)
    return lo, hi


def truncate_spectrum_iraf(
    flux: np.ndarray,
    crval1_text: str,
    cdelt1_text: str,
    wave_lo: float,
    wave_hi: float,
):
    """``scopy(w1=wave_lo, w2=wave_hi)`` of a ``dispcor`` output, IRAF-exact.

    Unlike :func:`truncate_spectrum` this takes the input WCS as the header
    *text* WARP's ``_csw`` file carries (IRAF re-reads it with ``ctod``),
    takes the bounds through pyraf's 12-digit real-parameter text, maps
    wavelengths to pixels through the MWCS transform, and returns the output
    header WCS IRAF writes (:class:`decanter.wavelength.iraf_wcs.ScopyWCS`)
    alongside the data.

    IRAF equivalent: ``noao/onedspec/t_sarith.x:sa_1d`` / ``sa_sextract``
    + ``smw/shdr.x:shdr_extract`` / ``shdr_linear``; see
    :func:`decanter.wavelength.iraf_wcs.scopy_wcs`.

    Returns ``(rebinned_flux, ScopyWCS)``.
    """
    from decanter.wavelength.iraf_wcs import scopy_wcs
    sc = scopy_wcs(crval1_text, cdelt1_text, int(flux.size), wave_lo, wave_hi)
    crval = iraf_ctod(crval1_text)
    cdelt = iraf_ctod(cdelt1_text)
    # the input as IRAF reads it (FITS kernel: utils/iraf_imio)
    rebinned = shdr_linear(imio_read_real(flux), crval, cdelt, sc.w0, sc.w1, sc.n_r, mwcs=True)
    if sc.n_r < sc.n:
        # IRAF copies n values out of an n_r-long buffer: the tail is
        # uninitialised heap memory in WARP (not reproducible). Repeat the
        # last resampled value instead.
        rebinned = np.concatenate(
            [rebinned, np.repeat(rebinned[-1:], sc.n - sc.n_r)]).astype(np.float32)
    elif sc.n_r > sc.n:
        rebinned = rebinned[: sc.n]
    return rebinned.astype(np.float32), sc


def run(
    config: Config,
    workdir: Path,
    listfile: Path,
    *,
    apdb_path: Path | str | None = None,
    fsr_path: Path | str | None = None,
    **_unused: Any,
) -> None:
    """Truncate each order to its FSR × ``cutrange``.

    Args:
        config: uses ``cutrange_list`` (default ``(1.05, 1.30)``).
        workdir: reads ``{obj}_NO{i}_sscfm_m{m}_1dcsw.fits`` (s12 output);
            writes ``..._fsr{cutrange}_VAC.fits`` per cutrange.
        listfile: WARP-style input list.
        apdb_path: aperture database for order enumeration.
        fsr_path: path to the FSR table.
    """
    missing = [
        n for n, v in [("apdb_path", apdb_path), ("fsr_path", fsr_path)] if v is None
    ]
    if missing:
        raise ValueError(f"s13_fsr_truncate requires {missing}")

    fsr = load_fsr(fsr_path)
    cal_apset = ApertureSet.load(apdb_path)
    orders = (
        cal_apset.echelle_orders
        if (config.reduce_full_data or not config.selected_orders)
        else tuple(m for m in cal_apset.echelle_orders if m in config.selected_orders)
    )

    def _fsr_cut_one(in_path: Path, out_template: str, m: int) -> None:
        """FSR-truncate a 1D spectrum at ``in_path``; write per-cutrange outputs.

        ``out_template`` is the filename WITHOUT the ``_fsr{cutrange}_VAC.fits``
        suffix — this helper appends it.
        """
        if not in_path.exists() or m not in fsr:
            return
        data, header = _fits.read_image(in_path)
        crval1 = float(header.get("CRVAL1", 1.0))
        cdelt1 = float(header.get("CDELT1", 1.0))
        fsr_lo = fsr[m].lambda_min
        fsr_hi = fsr[m].lambda_max
        center = (fsr_lo + fsr_hi) / 2.0
        for cutrange in config.cutrange_list:
            lo = center - (center - fsr_lo) * float(cutrange)
            hi = center + (fsr_hi - center) * float(cutrange)
            sliced, wcs = truncate_spectrum(data, crval1, cdelt1, lo, hi)
            if sliced.size == 0:
                continue
            out_header = header.copy()
            out_header["CRVAL1"] = (wcs["crval1"], "Wavelength at CRPIX1 (input's CRVAL preserved)")
            out_header["CDELT1"] = (wcs["cdelt1"], "Wavelength step per output pixel")
            out_header["CRPIX1"] = (wcs["crpix1"], "Reference pixel along dispersion")
            out_header["LTM1_1"] = (wcs["ltm1_1"], "Output-to-input pixel scale (LTM)")
            out_header["LTV1"] = (wcs["ltv1"], "Output-to-input pixel offset (LTV)")
            out_header["FSR_LO"] = (fsr_lo, "FSR lower bound (A, vac)")
            out_header["FSR_HI"] = (fsr_hi, "FSR upper bound (A, vac)")
            out_header["FSR_FAC"] = (float(cutrange), "FSR cut factor")
            # Always 2 decimals to match WARP's ``"fsr%.2f"`` filename
            # convention (``Spec1Dtools.py`` cut_1dspec). Stripping trailing
            # zeros gives ``fsr1.3`` which doesn't match WARP's ``fsr1.30``.
            tag = f"fsr{cutrange:.2f}"
            out_path = workdir / f"{out_template}_{tag}_VAC.fits"
            _fits.write_image(out_path, sliced.astype(np.float32), out_header, overwrite=True)

    pairs = parse_listfile(listfile)
    for i, pair in enumerate(pairs, start=1):
        _obj_raw, obj_header = _fits.read_image(workdir / pair.object_name)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objname = _sanitize_objname(str(raw_objname))

        for m in orders:
            # --- OBJ path: _1dcsw → _fsr{cut}_VAC -----------------------------
            _fsr_cut_one(
                workdir / f"{objname}_NO{i}_sscfm_m{m}_1dcsw.fits",
                out_template=f"{objname}_NO{i}_sscfm_m{m}",
                m=m,
            )

            # --- SKY path: _trans1dcutw → _trans1dcutw_fsr{cut}_VAC ------------
            if config.flag_skyemission:
                _fsr_cut_one(
                    workdir / f"{objname}_skyNO{i}_fm_m{m}trans1dcutw.fits",
                    out_template=f"{objname}_skyNO{i}_fm_m{m}trans1dcutw",
                    m=m,
                )
