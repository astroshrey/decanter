"""Observation metadata must survive from the raw frame onto the 1D products.

A single-frame reduction only needs the WCS, but any multi-frame analysis (the
wavelength calibration, a transit series, an RV benchmark) needs mid-exposure
times, pointing and the instrument configuration. Those live only in the raw
frame header, so they are carried through :class:`Reduction.meta` and written
into every output spectrum.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits as astropy_fits

from decanter._reduction import OrderSpectrum, Reduction
from decanter.io import headers


def _raw_header() -> astropy_fits.Header:
    h = astropy_fits.Header()
    h["OBJECT"] = "TRAPPIST-1"
    h["INSTRUME"] = "WINERED"
    h["TELESCOP"] = "Magellan Clay Telescope"
    h["OBSERVAT"] = "LCO"
    h["SETTING"] = 5
    h["SLIT"] = 100
    h["DATE-OBS"] = "2024-09-16"
    h["UT-STR"] = "00:17:09.0"
    h["UT-END"] = "00:29:05.99"
    h["EXPTIME"] = 660.0
    h["RA"] = "23:06:30.71"
    h["DEC"] = "-05:02:26.3"
    h["AIRMASS"] = 1.91
    h["NODPOS"] = "A1"
    return h


def test_frame_meta_extracts_the_timing_and_pointing_keys() -> None:
    meta = headers.frame_meta(_raw_header())
    for key in ("OBJECT", "DATE-OBS", "UT-STR", "UT-END", "EXPTIME",
                "RA", "DEC", "AIRMASS", "INSTRUME", "SETTING", "SLIT"):
        assert key in meta, key
    assert meta["EXPTIME"] == 660.0
    assert meta["AIRMASS"] == 1.91


def test_frame_meta_omits_absent_keys_rather_than_filling_them() -> None:
    """``headers.get`` defaults to "N/A"; frame_meta must not propagate that."""
    meta = headers.frame_meta(_raw_header())
    assert "ZD" not in meta
    assert "N/A" not in meta.values()


def test_meta_is_written_into_every_output_spectrum(tmp_path: Path) -> None:
    meta = headers.frame_meta(_raw_header())
    meta["OBJFRAME"] = "WINA00047152"
    meta["SKYFRAME"] = "WINA00047153"
    meta["WAVSHIFT"] = 0.0

    spec = OrderSpectrum(
        order=163, fsr_cut=1.30, flux=np.ones(16, dtype=np.float32),
        crval1=10000.0, cdelt1=0.03, crpix1=1.0,
    )
    sky = OrderSpectrum(
        order=163, fsr_cut=1.30, flux=np.full(16, 2.0, dtype=np.float32),
        crval1=10000.0, cdelt1=0.03, crpix1=1.0,
    )
    reduction = Reduction(
        obj_name="TRAPPIST-1", obj_path=None, sky_path=None,
        obj={(1.30, 163): spec}, sky={(1.30, 163): sky}, meta=meta,
    )
    reduction.write_to(tmp_path)

    for name in ("TRAPPIST-1_NO1_sscfm_m163_fsr1.30_VAC.fits",
                 "TRAPPIST-1_skyNO1_fm_m163trans1dcutw_fsr1.30_VAC.fits"):
        header = astropy_fits.getheader(tmp_path / name)
        # the WCS is still there
        assert header["CRVAL1"] == 10000.0
        assert header["CTYPE1"] == "LINEAR"
        # and so is everything a time series needs
        assert header["OBJECT"] == "TRAPPIST-1"
        assert header["DATE-OBS"] == "2024-09-16"
        assert header["UT-STR"] == "00:17:09.0"
        assert header["UT-END"] == "00:29:05.99"
        assert header["RA"] == "23:06:30.71"
        assert header["AIRMASS"] == 1.91
        assert header["OBJFRAME"] == "WINA00047152"
        assert header["SKYFRAME"] == "WINA00047153"
        assert header["WAVSHIFT"] == 0.0
        # plus the order identity, which was previously only in the filename
        assert header["ECHORDER"] == 163
        assert header["FSRCUT"] == 1.30


def test_reduction_meta_defaults_to_empty(tmp_path: Path) -> None:
    """A Reduction built without meta still writes a valid WCS-only header."""
    spec = OrderSpectrum(
        order=163, fsr_cut=1.30, flux=np.ones(8, dtype=np.float32),
        crval1=1.0, cdelt1=0.1, crpix1=1.0,
    )
    reduction = Reduction(
        obj_name="X", obj_path=None, sky_path=None, obj={(1.30, 163): spec},
    )
    assert reduction.meta == {}
    reduction.write_to(tmp_path)
    header = astropy_fits.getheader(tmp_path / "X_NO1_sscfm_m163_fsr1.30_VAC.fits")
    assert header["CRVAL1"] == 1.0
