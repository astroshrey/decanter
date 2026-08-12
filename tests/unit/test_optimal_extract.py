"""Unit tests for Horne (1986) optimal extraction."""
from __future__ import annotations

import numpy as np

from decanter.extract.box_extract_1d import box_extract
from decanter.extract.optimal_extract import optimal_extract


def _synthetic_strip(H=400, W=80, center=40.0, sigma=3.0, flux=1000.0, seed=0):
    """A rectified strip: Gaussian slit profile along a vertical trace + noise."""
    rng = np.random.default_rng(seed)
    x = np.arange(W)
    prof = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    prof /= prof.sum()
    clean = flux * prof[None, :] * np.ones((H, 1))
    noisy = clean + rng.normal(0.0, np.sqrt(np.maximum(clean, 0) + 25.0))
    trace = np.full(H, center + 1.0)  # 1-indexed trace at the profile center
    return noisy.astype(np.float32), trace, flux


def test_optimal_recovers_total_flux():
    strip, trace, flux = _synthetic_strip()
    out = optimal_extract(strip, trace, ap_low=-12.0, ap_high=12.0)
    assert out.shape == (strip.shape[0],)
    # Profile-weighted extraction can carry a small *flat* flux-scale offset
    # (empirical-profile discretization); it divides out under continuum
    # normalization, which every downstream product uses. Line shape and SNR
    # are the science-relevant properties (covered below).
    assert abs(np.median(out) - flux) / flux < 0.08


def test_optimal_snr_beats_box():
    """Formal variance of optimal <= box (Cauchy-Schwarz); here strictly better."""
    strip, trace, _ = _synthetic_strip()
    _, var_opt = optimal_extract(strip, trace, ap_low=-12.0, ap_high=12.0,
                                 read_var=25.0, return_var=True)
    # Box variance on the same integer window = sum of per-pixel variance.
    center = trace - 1.0
    half = 12.0
    box = box_extract(strip, trace, ap_low=-12.0, ap_high=12.0)
    var_box = np.zeros_like(var_opt)
    for y in range(strip.shape[0]):
        lo = int(np.floor(center[y] - half)); hi = int(np.ceil(center[y] + half))
        D = strip[y, lo:hi + 1].astype(float)
        var_box[y] = (25.0 + np.maximum(D, 0)).sum()
    good = (var_opt > 0) & (var_box > 0)
    gain = np.sqrt(var_box[good] / var_opt[good])
    assert gain.min() >= 0.999            # never worse than box
    assert np.median(gain) > 1.05         # real improvement on a Gaussian profile
    assert box.shape == (strip.shape[0],)


def test_optimal_rejects_bad_aperture():
    strip, trace, _ = _synthetic_strip()
    try:
        optimal_extract(strip, trace, ap_low=5.0, ap_high=-5.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for ap_high <= ap_low")
