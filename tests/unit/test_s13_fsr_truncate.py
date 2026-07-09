"""Tests for s13 FSR truncation (IRAF scopy(rebin=YES) port).

References:
  - sa_sextract: noao/onedspec/t_sarith.x:1369
  - shdr_extract: noao/onedspec/smw/shdr.x:1163
  - shdr_linear: noao/onedspec/smw/shdr.x:1018 (poly5 image-interp resample)
"""

from __future__ import annotations

import numpy as np

from decanter.wavelength.fsr import truncate_spectrum


def _make_smooth_spectrum(n: int, crval1: float, cdelt1: float) -> np.ndarray:
    """A smooth (low-degree polynomial in wavelength) spectrum so poly5
    integral-averaging reproduces values at integer pixels exactly."""
    waves = crval1 + np.arange(n, dtype=np.float64) * cdelt1
    return (100.0 + 5.0 * np.sin((waves - crval1) / (10 * cdelt1))).astype(np.float32)


def test_truncate_returns_iraf_wcs_keys():
    n = 4095
    crval1, cdelt1 = 10754.808714, 0.0344612471910111
    data = _make_smooth_spectrum(n, crval1, cdelt1)
    # m163 FSR_winered_20190610: [10792.9, 10858.7], cutrange 1.05.
    fsr_lo, fsr_hi = 10792.9, 10858.7
    center = (fsr_lo + fsr_hi) / 2
    cutrange = 1.05
    lo = center - (center - fsr_lo) * cutrange
    hi = center + (fsr_hi - center) * cutrange

    out, wcs = truncate_spectrum(data, crval1, cdelt1, lo, hi)
    # NAXIS matches WARP's m163 output exactly.
    assert out.size == 2005

    # IRAF semantics: CRVAL preserved.
    assert wcs["crval1"] == crval1

    # CRPIX < 0 for orders that start mid-FSR (the FSR window doesn't include
    # the input's CRVAL pixel).
    assert wcs["crpix1"] < 0

    # CDELT slightly larger than input CDELT (LTM1_1 > 1 because the
    # fractional-pixel span exceeds the integer-pixel count by ~0.04%).
    assert wcs["cdelt1"] > cdelt1
    assert abs(wcs["cdelt1"] - 0.0344759710) < 1e-6

    # Sanity-check the wavelength at output pixel 1.
    wave_at_x_out_1 = wcs["crval1"] + (1.0 - wcs["crpix1"]) * wcs["cdelt1"]
    assert abs(wave_at_x_out_1 - lo) < 0.001


def test_truncate_round_trip_at_integer_alignment():
    """When wave_lo and wave_hi land exactly on integer pixels, the rebin
    should be a pure integer slice (LTM1_1 = 1, output values = input slice)."""
    n = 100
    crval1, cdelt1 = 1000.0, 0.5
    data = np.arange(n, dtype=np.float32) + 10.0
    # Slice pixels 21..50 (integer-aligned). At pixel k (1-based), wave = 1000 + (k-1)*0.5.
    # So wave_lo = 1010 (k=21), wave_hi = 1024.5 (k=50).
    out, wcs = truncate_spectrum(data, crval1, cdelt1, 1010.0, 1024.5)
    assert out.size == 30
    # LTM1_1 should be exactly 1.0 for integer-aligned slices.
    assert abs(wcs["ltm1_1"] - 1.0) < 1e-10
    # CDELT unchanged.
    assert abs(wcs["cdelt1"] - cdelt1) < 1e-10
    # Output values match input pixels 21..50 (within poly5 smoothing).
    np.testing.assert_allclose(out[5:-5], data[20 + 5 : 50 - 5], rtol=1e-4)


def test_truncate_empty_when_range_outside_input():
    n = 50
    data = np.ones(n, dtype=np.float32)
    out, wcs = truncate_spectrum(data, 1000.0, 0.5, 2000.0, 2100.0)
    # Both endpoints clip to the same value (the input's right edge) → n=1
    # output pixel. Not strictly empty, but the wavelength range is degenerate.
    # The IRAF semantics produce a 1-pixel output in this case.
    assert out.size <= 1
