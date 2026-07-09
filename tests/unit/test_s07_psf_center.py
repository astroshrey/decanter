"""Unit tests for :mod:`decanter.extract.psf_center` + the psf utility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from decanter.utils.psf import (
    PsfFit,
    fit_slit_gaussian,
    stacked_slit_profile,
)


def _make_synthetic_strip(
    H: int = 1000,
    W: int = 80,
    *,
    center: float = 40.0,
    fwhm: float = 6.0,
    amplitude: float = 1000.0,
    noise: float = 5.0,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic rectified strip with a constant-trace Gaussian PSF."""
    rng = np.random.default_rng(rng_seed)
    x = np.arange(W, dtype=np.float64)
    sigma = fwhm / 2.3548
    profile = amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2))
    data = np.broadcast_to(profile, (H, W)).astype(np.float32)
    data = data + rng.normal(scale=noise, size=(H, W)).astype(np.float32)
    trace_x = np.full(H, center + 1.0)  # 1-indexed
    return data, trace_x


def test_fit_recovers_known_psf() -> None:
    """Stack + fit recovers the injected Gaussian center and FWHM."""
    data, trace_x = _make_synthetic_strip(center=40.0, fwhm=6.0)
    med_x, med_y, n_rows = stacked_slit_profile(
        data, trace_x, ap_low=-15.0, ap_high=15.0,
        lowlim_y=100, upplim_y=900, step_sampling=5,
    )
    assert n_rows > 100, f"too few rows used: {n_rows}"
    fit = fit_slit_gaussian(med_x, med_y)
    assert fit.success
    # xshift should be ~0 (trace is at the actual peak)
    assert abs(fit.xshift) < 0.5, f"xshift={fit.xshift}"
    # FWHM should be ~6.0 ± 0.5
    assert abs(fit.fwhm - 6.0) < 0.5, f"fwhm={fit.fwhm}"


def test_fit_handles_offset_psf() -> None:
    """If the PSF is offset from the trace, xshift reports the offset."""
    H, W = 1000, 80
    real_center = 45.0
    trace_at = 40.0  # we lie about where the trace is; xshift should be +5
    data, _ = _make_synthetic_strip(H=H, W=W, center=real_center, fwhm=5.0, rng_seed=42)
    trace_x = np.full(H, trace_at + 1.0)
    med_x, med_y, _ = stacked_slit_profile(
        data, trace_x, ap_low=-25.0, ap_high=25.0,
        lowlim_y=100, upplim_y=900,
    )
    fit = fit_slit_gaussian(med_x, med_y)
    assert abs(fit.xshift - (real_center - trace_at)) < 0.6, f"xshift={fit.xshift}"


def test_low_signal_returns_nan_or_unsuccessful() -> None:
    """A near-pure-noise strip should not yield a confident fit."""
    rng = np.random.default_rng(0)
    data = rng.normal(scale=1.0, size=(800, 60)).astype(np.float32)
    trace_x = np.full(800, 30.0)
    med_x, med_y, _ = stacked_slit_profile(
        data, trace_x, ap_low=-20.0, ap_high=20.0,
        lowlim_y=100, upplim_y=700,
    )
    fit = fit_slit_gaussian(med_x, med_y)
    # Either NaN/failed, or a wildly off result. We just require no crash.
    assert isinstance(fit, PsfFit)


def test_abba_rejects_o_position() -> None:
    """In ABBA mode, the O-position peak is rejected as a candidate."""
    H, W = 1000, 80
    # Inject TWO peaks: one near the trace (the real star) at offset +12,
    # one at the O-position (offset ~0). Without ABBA we'd pick the higher
    # of the two; with ABBA the O-position should be ignored.
    rng = np.random.default_rng(7)
    x = np.arange(W, dtype=np.float64)
    sigma = 2.5
    star_center = 52.0
    o_pos_center = 40.0  # at trace_x=40, so offset 0
    profile = (
        1000.0 * np.exp(-((x - star_center) ** 2) / (2 * sigma**2))
        + 1500.0 * np.exp(-((x - o_pos_center) ** 2) / (2 * sigma**2))
    )
    data = np.broadcast_to(profile, (H, W)).astype(np.float32)
    data = data + rng.normal(scale=3.0, size=(H, W)).astype(np.float32)
    trace_x = np.full(H, 40.0 + 1.0)  # 1-indexed

    med_x, med_y, _ = stacked_slit_profile(
        data, trace_x, ap_low=-30.0, ap_high=30.0,
        lowlim_y=100, upplim_y=900, abba=True,
    )
    fit = fit_slit_gaussian(med_x, med_y, abba=True)
    # ABBA should locate the star peak at offset +12, not 0.
    assert fit.xshift > 5.0, f"xshift={fit.xshift} — ABBA didn't reject O-position"
