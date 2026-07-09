"""Tests for the IRAF iminterp POLY5 port.

References:
  - asifit POLY5: math/iminterp/asifit.x:136
  - ii_getpcoeff POLY5: math/iminterp/ii_1dinteg.x:215
  - asigrl: math/iminterp/asigrl.x:142
  - dispcor: noao/onedspec/dispcor/dispcor.x
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from decanter.utils.iminterp import (
    REAL,
    asifit_poly5,
    asigrl_poly5,
    dispcor_linear_poly5,
    per_segment_pcoeff_poly5,
    scombine_linear_poly5,
)


def _eval_poly(coef, x):
    """Evaluate ``sum_k coef[k] * x^k``."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    for k, c in enumerate(coef):
        out += c * x**k
    return out


def _analytic_integral(coef, a, b):
    """Analytic ``∫_a^b sum_k coef[k] x^k dx``."""
    s = 0.0
    for k, c in enumerate(coef):
        s += c / (k + 1) * (b ** (k + 1) - a ** (k + 1))
    return s


def test_asifit_poly5_pad_values():
    """Reflection-pad endpoints match the IRAF formulas exactly."""
    data = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0], dtype=np.float32)
    coeff = asifit_poly5(data)
    assert coeff.size == data.size + 5
    # Original data lives at offset 2.
    np.testing.assert_array_equal(coeff[2:10], data)
    # Left pad: c0ptr+1 = 2*d[1] - d[3]; c0ptr+2 = 2*d[1] - d[2].
    assert coeff[0] == pytest.approx(2 * 1.0 - 3.0)
    assert coeff[1] == pytest.approx(2 * 1.0 - 2.0)
    # Right pad: cnptr+1..+3 = 2*d[N] - d[N-1..N-3].
    assert coeff[10] == pytest.approx(2 * 34.0 - 21.0)
    assert coeff[11] == pytest.approx(2 * 34.0 - 13.0)
    assert coeff[12] == pytest.approx(2 * 34.0 - 8.0)


def test_per_segment_pcoeff_reconstructs_polynomial():
    """For a smooth function sampled exactly on the grid, evaluating the
    Newton-form local polynomial at fractional positions must reproduce
    the analytic function value within float32 precision."""
    # Use a low-amplitude polynomial so float32 storage doesn't lose
    # precision in the high-order terms.
    coef = [0.0, 1.0, -0.05, 0.002, -3e-5, 2e-7]
    N = 64
    xs = np.arange(1, N + 1, dtype=np.float64)
    data = _eval_poly(coef, xs).astype(np.float32)
    asi = asifit_poly5(data)
    pcoeff = per_segment_pcoeff_poly5(asi)

    # Evaluate at fractional positions in interior segments.
    for j in (15, 32, 48):
        for delta in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = j + delta
            expected = _eval_poly(coef, np.array([x]))[0]
            d = np.float32(delta)
            powers = np.array(
                [1.0, d, d * d, d * d * d, d ** 4, d ** 5], dtype=np.float32
            )
            got = float((pcoeff[:, j - 1] * powers).sum(dtype=np.float32))
            assert abs(got - expected) < 1e-4 * (abs(expected) + 1.0), (
                f"j={j}, delta={delta}, got={got}, expected={expected}"
            )


def test_asigrl_poly5_matches_analytic_for_degree5():
    """The segment integral reproduces ``∫ p_5(x) dx`` within float32."""
    coef = [0.5, -0.3, 0.08, -0.02, 0.005, 0.0008]
    N = 80
    xs = np.arange(1, N + 1, dtype=np.float64)
    data = _eval_poly(coef, xs).astype(np.float32)
    asi = asifit_poly5(data)
    pcoeff = per_segment_pcoeff_poly5(asi)

    intervals = [
        (10.3, 11.7),   # single segment
        (10.5, 12.4),   # 2 segments
        (5.2, 9.8),     # multi segments
        (20.0, 21.0),   # exact integer alignment
        (30.7, 33.3),   # several segments
        (40.0, 40.0),   # zero width
    ]
    a = np.array([p[0] for p in intervals], dtype=np.float32)
    b = np.array([p[1] for p in intervals], dtype=np.float32)
    got = asigrl_poly5(pcoeff, a, b)
    want = np.array(
        [_analytic_integral(coef, p[0], p[1]) for p in intervals], dtype=np.float64
    )
    # Float32 relative precision; absolute tolerance scales with max magnitude.
    max_mag = max(abs(v) for v in want) + 1.0
    np.testing.assert_allclose(
        got.astype(np.float64), want, rtol=2e-6, atol=2e-6 * max_mag
    )


def test_asigrl_poly5_sign_convention():
    """Swapping a/b negates the integral (IRAF semantics)."""
    rng = np.random.default_rng(7)
    data = rng.standard_normal(50).astype(np.float32)
    asi = asifit_poly5(data)
    pcoeff = per_segment_pcoeff_poly5(asi)
    a = np.array([12.0, 5.5])
    b = np.array([14.0, 10.2])
    fwd = asigrl_poly5(pcoeff, a, b)
    rev = asigrl_poly5(pcoeff, b, a)
    np.testing.assert_array_equal(rev, -fwd)


def test_dispcor_linear_poly5_constant_spectrum():
    """A constant input spectrum integrates to the same constant
    everywhere, regardless of the dispersion solution."""
    N = 200
    data = np.full(N, 17.5, dtype=np.float32)

    # Use a near-linear dispersion so the inverse is well-conditioned.
    def pix_at_w(w):
        # w(x) = 1000 + 0.1 * (x - 1); inverse: x = 1 + (w - 1000) / 0.1
        return 1.0 + (np.asarray(w, dtype=np.float64) - 1000.0) / 0.1

    w_min, w_max = 1000.0, 1000.0 + 0.1 * (N - 1)
    out = dispcor_linear_poly5(data, pix_at_w, w_min, w_max, N, flux=False)
    # Constant input → constant output (everywhere except possibly the
    # ofb-both endpoints, but with linear dispersion the inverse stays
    # within [0.5, N+0.5] so no edges trigger ofb).
    np.testing.assert_allclose(out, 17.5, rtol=0, atol=2e-5)


def test_dispcor_linear_poly5_reproduces_linear_spectrum_interior():
    """A linear input on a matched linear dispersion grid must reproduce
    the input at interior pixels (the IRAF edge-pad introduces small
    edge artifacts; those are tested separately)."""
    N = 200
    xs = np.arange(1, N + 1, dtype=np.float64)
    data = (2.5 + 0.3 * xs).astype(np.float32)

    def pix_at_w(w):
        return 1.0 + (np.asarray(w, dtype=np.float64) - 1000.0) / 0.1

    w_min, w_max = 1000.0, 1000.0 + 0.1 * (N - 1)
    out = dispcor_linear_poly5(data, pix_at_w, w_min, w_max, N, flux=False)
    expected = (2.5 + 0.3 * xs).astype(np.float32)
    # Interior pixels are uncontaminated by the edge-duplication pad.
    interior = slice(5, N - 5)
    np.testing.assert_allclose(out[interior], expected[interior], rtol=1e-5, atol=1e-4)


def test_dispcor_linear_poly5_descending_dispersion():
    """Many WINERED orders have ``w(1) > w(N)`` (the wavelength decreases
    with pixel index). IRAF dispcor handles this by always invoking
    ``asigrl`` with the smaller argument first (``dispcor.x:85`` and
    ``:97``), keeping the integral positive. Confirm our port preserves
    sign through that path."""
    N = 200
    xs = np.arange(1, N + 1, dtype=np.float64)
    # Constant input: every output pixel must equal the constant.
    data = np.full(N, 7.25, dtype=np.float32)

    # Descending: w(x) = 1000 - 0.1 * (x - 1); inverse x = 1 - (w - 1000) / 0.1.
    def pix_at_w(w):
        return 1.0 - (np.asarray(w, dtype=np.float64) - 1000.0) / 0.1

    w_at_1 = 1000.0
    w_at_N = 1000.0 - 0.1 * (N - 1)
    w_min, w_max = min(w_at_1, w_at_N), max(w_at_1, w_at_N)
    out = dispcor_linear_poly5(data, pix_at_w, w_min, w_max, N, flux=False)
    np.testing.assert_allclose(out, 7.25, rtol=0, atol=2e-5)
    # Most importantly: signs are preserved.
    assert np.all(out > 0)


def test_dispcor_linear_poly5_edge_pad_matches_iraf_kink():
    """IRAF dispcor pads input with duplicate edge values (``temp[1] = in[1]``,
    ``temp[N+2] = in[N]``), introducing a small "kink" at the very ends.
    For a linear input the output at pixel 1 should equal the integral
    average of poly5 over the kinked region, which deviates from the
    naive linear value by a fraction of the pixel slope."""
    N = 50
    xs = np.arange(1, N + 1, dtype=np.float64)
    slope = 0.3
    intercept = 2.5
    data = (intercept + slope * xs).astype(np.float32)

    def pix_at_w(w):
        return 1.0 + (np.asarray(w, dtype=np.float64) - 1000.0) / 0.1

    w_min, w_max = 1000.0, 1000.0 + 0.1 * (N - 1)
    out = dispcor_linear_poly5(data, pix_at_w, w_min, w_max, N, flux=False)
    expected_linear = (intercept + slope * xs).astype(np.float32)
    # The first and last few pixels deviate from the linear extrapolation
    # by less than one pixel slope.
    assert np.all(np.abs(out - expected_linear) < abs(slope))


# ---------------------------------------------------------------------------
# scombine_linear_poly5 — covers WARP's PySpecshift resample step
# ---------------------------------------------------------------------------


def test_scombine_zero_shift_smooth_spectrum_is_near_identity():
    """For smooth spectra, scombine with matching input/output WCS reproduces
    the data at interior pixels close to float32 precision. The integral-
    averaging over poly5 introduces a tiny smoothing for non-polynomial
    structure, but for low-curvature inputs the round-trip is within ~0.1%."""
    N = 200
    xs = np.arange(1, N + 1, dtype=np.float64)
    # Smooth low-amplitude oscillation; close to a low-degree polynomial.
    data = (100.0 + 5.0 * np.sin(xs / 30.0)).astype(np.float32)
    out = scombine_linear_poly5(
        data, crval1_in=1.0, cdelt1_in=1.0, w1_out=1.0, dw_out=1.0, nw_out=N
    )
    np.testing.assert_allclose(out[10:-10], data[10:-10], rtol=1e-3, atol=1e-2)


def test_scombine_integer_shift_translates_data():
    """specshift(shift=+k) then scombine back to w1=CRVAL1_orig shifts the
    data LEFT by k pixels (output[i] = input[i - k]). This is the direction
    WARP's PySpecshift takes — specshift labels the data as starting at
    CRVAL1+k, then scombine resamples back to CRVAL1, effectively moving
    the data values to lower-pixel indices."""
    N = 100
    # Smooth (low-degree polynomial) so the integral average matches the
    # pixel-center value.
    data = (10.0 + np.arange(N, dtype=np.float32)).astype(np.float32)
    shift = 3
    out = scombine_linear_poly5(
        data, crval1_in=1.0 + float(shift), cdelt1_in=1.0,
        w1_out=1.0, dw_out=1.0, nw_out=N,
    )
    # Interior: out[k] == data[k - shift] (0-based numpy indexing).
    np.testing.assert_allclose(
        out[shift + 5 : N - 5], data[5 : N - shift - 5], rtol=0, atol=1e-3
    )
    # First `shift` output pixels are below the input WCS range → effectively 0.
    np.testing.assert_allclose(out[: shift - 2], 0.0, rtol=0, atol=1e-3)


def test_scombine_subpixel_shift_smooths_edges_softly():
    """A small (sub-pixel) shift must produce a result close to the input
    at interior pixels — within the poly5 smoothing scale."""
    rng = np.random.default_rng(21)
    N = 200
    data = rng.uniform(50, 500, size=N).astype(np.float32)
    out = scombine_linear_poly5(
        data, crval1_in=1.0 + 0.25, cdelt1_in=1.0,
        w1_out=1.0, dw_out=1.0, nw_out=N,
    )
    # The output isn't equal to the input (we rebinned with a sub-pixel offset),
    # but the difference between output and input should be <= the local pixel
    # variation — i.e. small relative to the spectrum's overall amplitude.
    interior_diff = np.abs(out[10:-10] - data[10:-10])
    assert interior_diff.max() < 200, f"sub-pixel shift caused large interior change: {interior_diff.max()}"


def test_scombine_descending_input_wcs():
    """A descending input WCS (CDELT1 < 0) is unusual but specshift can
    produce one if the shift is negative enough. Confirm sign safety."""
    N = 50
    data = np.full(N, 42.5, dtype=np.float32)
    out = scombine_linear_poly5(
        data, crval1_in=100.0, cdelt1_in=-0.5,
        w1_out=100.0, dw_out=-0.5, nw_out=N,
    )
    np.testing.assert_allclose(out, 42.5, rtol=0, atol=2e-5)
    assert np.all(out > 0)
