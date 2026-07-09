"""Tests for the IRAF ``onedspec.scopy`` wavelength-bound truncate port.

The semantics matter: ``scopy.cl:21`` sets ``rebin = yes`` as default,
which routes through ``sarith → sa_sextract → shdr_linear``. That's a
POLY5 integral-average resample onto a fresh wavelength grid aligned at
``w1`` with the input's ``CDELT1`` — NOT a pixel-aligned integer slice
(which is what decanter had pre-2026-05-14).

References:
  - ``noao/onedspec/scopy.cl:21`` — default ``rebin = yes``
  - ``noao/onedspec/sarith/t_sarith.x:sa_sextract``
  - ``noao/onedspec/smw/shdr.x:shdr_linear``
  - ``warp/Spec1Dtools.py:truncate`` — WARP wrapper that calls
    ``iraf.scopy(rawspec, output, w1=p1, w2=p2)`` then ``iraf.hedit``
    to zero CRVAL1/CRPIX1/LTV1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from decanter.utils.iraf_scopy import scopy_wavelength_truncate
from decanter._localpaths import WINERED_ROOT

# Reference reduction (built via WARP -s); skip if unavailable.
_REF = (
    WINERED_ROOT / "reductions" / "TOI2109_decanterref" / "TOI2109_NO1"
    / "intermediate_files" / "SKY"
)


def _build_header(crval1: float, cdelt1: float, naxis1: int) -> fits.Header:
    h = fits.Header()
    h["NAXIS1"] = naxis1
    h["CRVAL1"] = crval1
    h["CDELT1"] = cdelt1
    h["CRPIX1"] = 1.0
    return h


def test_truncate_output_length_matches_sa_sextract_formula() -> None:
    """``nw_out = nint((w2 - w1) / CDELT_in) + 1`` per sa_sextract."""
    data = np.linspace(0, 1, 5628, dtype=np.float32)
    hdr = _build_header(crval1=-516.0, cdelt1=0.5, naxis1=5628)
    out = scopy_wavelength_truncate(data, hdr, w1=1.0, w2=2048.0)
    assert out.data.size == int(round((2048.0 - 1.0) / 0.5)) + 1 == 4095


def test_truncate_reset_wcs_sets_crval_crpix_ltv() -> None:
    """``reset_wcs=True`` must hedit CRVAL1=1, CRPIX1=1, LTV1=0 per WARP."""
    data = np.linspace(0, 1, 5628, dtype=np.float32)
    hdr = _build_header(crval1=-516.0, cdelt1=0.5, naxis1=5628)
    out = scopy_wavelength_truncate(data, hdr, w1=1.0, w2=2048.0, reset_wcs=True)
    assert float(out.header["CRVAL1"]) == 1.0
    assert float(out.header["CRPIX1"]) == 1.0
    assert float(out.header["LTV1"]) == 0.0


def test_truncate_aligned_grid_preserves_values_at_grid_points() -> None:
    """For a linear input whose grid already contains every output λ exactly,
    the POLY5 rebin is identity (interpolant equals input at sample points)."""
    n = 100
    crval, cdelt = 1.0, 0.5  # output grid w1=1, dw=0.5 lands exactly on input pixels
    data = np.sin(np.linspace(0, 4 * np.pi, n)).astype(np.float32) * 100.0
    hdr = _build_header(crval, cdelt, n)
    out = scopy_wavelength_truncate(data, hdr, w1=crval, w2=crval + (n - 1) * cdelt)
    # Should short-circuit to a copy (length and values preserved at ULP scale).
    assert out.data.size == n
    np.testing.assert_allclose(out.data, data, rtol=0, atol=0)


def test_truncate_smooth_quadratic_no_resampling_error_at_centers() -> None:
    """POLY5 reproduces low-degree polynomials exactly (modulo fp32 ULP)."""
    n = 200
    crval_in, cdelt_in = -10.0, 0.5
    x_in = crval_in + np.arange(n) * cdelt_in
    # Smooth quadratic — well within POLY5's exact-reproduction regime.
    data = (1.0 + 0.01 * x_in - 0.001 * x_in**2).astype(np.float32)
    hdr = _build_header(crval_in, cdelt_in, n)
    out = scopy_wavelength_truncate(data, hdr, w1=1.0, w2=20.0)
    # Output spans [1.0, 20.0] at step 0.5 → 39 pixels.
    expected_n = int(round((20.0 - 1.0) / 0.5)) + 1
    assert out.data.size == expected_n
    # Sample the same analytic function at the output wavelengths:
    x_out = 1.0 + np.arange(expected_n) * cdelt_in
    truth = 1.0 + 0.01 * x_out - 0.001 * x_out**2
    np.testing.assert_allclose(out.data, truth.astype(np.float32), rtol=0, atol=1e-3)


@pytest.mark.skipif(
    not (_REF / "5-SKY_extract").exists() or not (_REF / "6-SKY_truncate").exists(),
    reason="WARP reference reduction not available",
)
def test_truncate_matches_warp_sky_trans1dcut_at_fp32_ulp() -> None:
    """End-to-end parity: feed WARP's ``_skyNO1_fm_m163trans1d.0163.fits`` through
    decanter's truncate; output must match WARP's ``_skyNO1_fm_m163trans1dcut.fits``
    at float32 ULP (median |Δ| = 0). This is the regression that locks in the
    poly5-rebin fix landed when sky ``_trans1dcut`` dropped from ~4 ct → 0.04 ct.
    """
    src = next((_REF / "5-SKY_extract").glob("*_skyNO1_fm_m163trans1d.0163.fits"))
    ref = next((_REF / "6-SKY_truncate").glob("*_skyNO1_fm_m163trans1dcut.fits"))
    d_in = fits.getdata(src)
    h_in = fits.getheader(src)
    out = scopy_wavelength_truncate(d_in, h_in, w1=1.0, w2=2048.0, reset_wcs=True)
    d_ref = fits.getdata(ref)
    assert out.data.shape == d_ref.shape
    diff = out.data.astype(np.float64) - d_ref.astype(np.float64)
    # median bit-perfect against WARP; max within float32 ULP of ~13000 ct values.
    assert np.median(np.abs(diff)) == 0.0
    assert np.max(np.abs(diff)) <= 1e-3


def test_truncate_does_not_mutate_caller_header() -> None:
    """astropy ``Header(h)`` shares Card objects; the truncate must copy.

    Regression for the Phase-2 in-memory sky bug: the obj s11 call
    mutated the shared per-order header (CRVAL1 -> 1.0), so the sky
    s11 call truncated with a WCS off by ~1034 px. File-based stage
    drivers never saw this because each stage re-read headers from
    disk.
    """
    import numpy as np
    from astropy.io import fits

    h = fits.Header()
    h["CRVAL1"] = -516.063232421875
    h["CDELT1"] = 0.5
    h["CRPIX1"] = 1.0
    data = np.arange(5628, dtype=np.float64)
    scopy_wavelength_truncate(data, h, w1=1.0, w2=2048.0)
    assert h["CRVAL1"] == -516.063232421875
    assert h["CDELT1"] == 0.5

    from decanter.waveshift.apply import apply_waveshift_one_order
    apply_waveshift_one_order(data, h, shift_wave=0.0)
    assert h["CRVAL1"] == -516.063232421875
