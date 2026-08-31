"""The shift estimator, checked at both WINERED sampling regimes.

WINERED runs at roughly 4.6 pixels per resolution element in HIRES-Y/J but
only 2.1 in WIDE. A CCF peak refined with a parabola is known to bias toward
integer pixels when the sampling is that coarse, so the accuracy is pinned
here at both regimes rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from decanter.wavecal.measure import (
    ccf_shifts,
    continuum_normalize,
    highpass,
    parabolic_extremum,
    robust_scatter,
    measure_series_shifts,
    shift_grid_pixels,
)
from decanter.wavecal.solve import _refit_oh_one

RNG = np.random.default_rng(20260819)


def _line_forest(n_pixels: int, sigma_pixels: float, n_lines: int = 60) -> np.ndarray:
    """A reproducible absorption-line spectrum: depth as a function of pixel."""
    pixel = np.arange(n_pixels, dtype=float)
    centres = RNG.uniform(80, n_pixels - 80, n_lines)
    depths = RNG.uniform(0.05, 0.6, n_lines)
    depth = np.zeros(n_pixels)
    for centre, amplitude in zip(centres, depths):
        depth += amplitude * np.exp(-0.5 * ((pixel - centre) / sigma_pixels) ** 2)
    return depth


def _shifted(depth: np.ndarray, shift: float) -> np.ndarray:
    pixel = np.arange(depth.size, dtype=float)
    return np.interp(pixel - shift, pixel, depth, left=0.0, right=0.0)


@pytest.mark.parametrize(
    "sigma_pixels, dv_pix_kms, label",
    [(1.95, 0.96, "HIRES 4.6 px/FWHM"), (0.90, 5.09, "WIDE 2.1 px/FWHM")],
)
def test_shift_recovered_at_both_sampling_regimes(sigma_pixels, dv_pix_kms, label) -> None:
    n_pixels = 1500
    template = _line_forest(n_pixels, sigma_pixels)
    # Truths in km/s, inside the +/-12 km/s search, so the test is the same
    # physical test in both regimes rather than the same pixel test.
    truth_kms = np.array([-8.0, -3.0, -0.3, 0.0, 1.7, 2.6, 6.4, 9.9])
    truth = truth_kms / dv_pix_kms
    signal = np.array([_shifted(template, s) for s in truth])
    support = template > 0.01
    support[:64] = False
    support[-64:] = False

    grid = shift_grid_pixels(12.0, dv_pix_kms)
    shifts, peaks = ccf_shifts(signal, template, support, grid)

    error_kms = (shifts - truth) * dv_pix_kms
    assert np.all(peaks > 0.99), label
    assert np.max(np.abs(error_kms)) * 1e3 < 30.0, f"{label}: {error_kms * 1e3}"


def test_no_pixel_phase_bias() -> None:
    """Errors must not correlate with the fractional part of the true shift.

    Peak locking shows up as a sawtooth in the residual against pixel phase,
    so the test compares integer-adjacent shifts with half-pixel ones.
    """
    n_pixels = 1500
    template = _line_forest(n_pixels, 0.90)      # WIDE-like, the hard case
    support = template > 0.01
    support[:64] = False
    support[-64:] = False
    truth = np.arange(-1.0, 1.001, 0.1)
    signal = np.array([_shifted(template, s) for s in truth])
    shifts, _ = ccf_shifts(signal, template, support, shift_grid_pixels(12.0, 5.09))

    residual = shifts - truth
    phase = truth - np.round(truth)
    near_integer = np.abs(phase) < 0.15
    near_half = np.abs(np.abs(phase) - 0.5) < 0.15
    bias = abs(np.mean(residual[near_integer]) - np.mean(residual[near_half]))
    assert bias * 5.09 * 1e3 < 30.0, f"pixel-phase bias {bias * 5.09 * 1e3:.1f} m/s"


def test_noise_degrades_gracefully() -> None:
    n_pixels = 1500
    template = _line_forest(n_pixels, 1.95)
    support = template > 0.01
    support[:64] = False
    support[-64:] = False
    truth = np.full(40, 0.37)
    clean = np.array([_shifted(template, s) for s in truth])
    noisy = clean + RNG.normal(0.0, 0.02, clean.shape)
    shifts, peaks = ccf_shifts(noisy, template, support, shift_grid_pixels(12.0, 0.96))
    assert np.all(np.isfinite(shifts))
    assert np.median(peaks) > 0.9
    assert abs(np.median(shifts) - 0.37) * 0.96 * 1e3 < 20.0


def test_insufficient_support_returns_nan() -> None:
    template = _line_forest(500, 2.0)
    support = np.zeros(500, dtype=bool)
    support[:5] = True
    shifts, peaks = ccf_shifts(np.zeros((3, 500)), template, support,
                               shift_grid_pixels(12.0, 1.0))
    assert np.all(np.isnan(shifts))
    assert np.all(np.isnan(peaks))


def test_shift_grid_is_velocity_uniform_not_pixel_uniform() -> None:
    """The same search in km/s must cover the same velocity in either mode."""
    hires = shift_grid_pixels(12.0, 0.96)
    wide = shift_grid_pixels(12.0, 5.09)
    assert hires.max() * 0.96 == pytest.approx(wide.max() * 5.09, rel=0.02)
    assert np.diff(hires)[0] * 0.96 == pytest.approx(np.diff(wide)[0] * 5.09, rel=0.02)


def test_parabolic_extremum_edges_and_flat() -> None:
    grid = np.array([-1.0, 0.0, 1.0])
    assert parabolic_extremum(grid, np.array([0.0, 1.0, 0.0]), 1) == pytest.approx(0.0)
    assert parabolic_extremum(grid, np.array([1.0, 0.0, 0.0]), 0) == pytest.approx(-1.0)
    assert parabolic_extremum(grid, np.array([1.0, 1.0, 1.0]), 1) == pytest.approx(0.0)


def test_robust_scatter_ignores_outliers() -> None:
    values = np.concatenate([RNG.normal(0.0, 1.0, 400), np.full(20, 500.0)])
    assert robust_scatter(values) == pytest.approx(1.0, abs=0.15)
    assert np.isnan(robust_scatter([1.0]))


def test_continuum_normalize_flattens_a_blaze() -> None:
    pixel = np.arange(1200, dtype=float)
    blaze = 1000.0 * np.exp(-0.5 * ((pixel - 600) / 400) ** 2)
    depth = _line_forest(1200, 2.0)
    normalized = continuum_normalize(blaze * (1.0 - depth), 151)
    interior = slice(150, -150)
    assert np.nanpercentile(normalized[interior], 95) == pytest.approx(1.0, abs=0.05)


def test_highpass_removes_a_smooth_background() -> None:
    pixel = np.arange(1200, dtype=float)
    background = 300.0 + 0.05 * pixel
    emission = 500.0 * np.exp(-0.5 * ((pixel - 700) / 2.0) ** 2)
    filtered = highpass(background + emission, 101)
    assert abs(np.median(filtered)) < 5.0
    assert filtered[700] > 400.0


def test_shift_outside_the_search_range_returns_nan() -> None:
    """A peak pinned at the grid edge is not a measurement."""
    template = _line_forest(1500, 1.95)
    support = template > 0.01
    support[:64] = False
    support[-64:] = False
    grid = shift_grid_pixels(2.0, 0.96)            # +/- 2 km/s only
    signal = np.array([_shifted(template, 10.0 / 0.96)])   # 10 km/s away
    shifts, peaks = ccf_shifts(signal, template, support, grid)
    assert np.isnan(shifts[0])
    assert np.isnan(peaks[0])


def test_order_specific_search_center_recovers_offset_series() -> None:
    template = _line_forest(1200, 0.9)
    dv = np.array([5.0])
    truth_kms = np.array([17.0, 21.0, 25.0])
    signal = np.asarray([_shifted(template, value / dv[0]) for value in truth_kms])[:, :, None]
    velocity, peak = measure_series_shifts(
        signal, template[:, None], (template > 0.01)[:, None], dv,
        search_kms=6.0, center_kms=np.array([21.0]),
    )
    assert np.all(peak[:, 0] > 0.99)
    assert np.max(np.abs(velocity[:, 0] - truth_kms)) < 0.03


def test_oh_refit_recovers_shift_when_band_ratio_changes() -> None:
    pixel = np.arange(800, dtype=float)
    first = np.exp(-0.5 * ((pixel - 250.0) / 1.1) ** 2)
    second = np.exp(-0.5 * ((pixel - 520.0) / 1.1) ** 2)
    components = np.asarray([first, second])
    truth = 0.73
    signal = 0.4 * _shifted(first, truth) + 1.7 * _shifted(second, truth)
    support = (first + second) > 1e-5
    support = np.convolve(support.astype(int), np.ones(9, dtype=int), mode="same") > 0
    parameters = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    measured = _refit_oh_one(components, signal, support, parameters, 0.5, 1.0)
    assert measured == pytest.approx(truth, abs=0.01)
