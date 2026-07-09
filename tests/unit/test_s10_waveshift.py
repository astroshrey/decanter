"""Unit tests for :mod:`decanter.waveshift.measure`."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import shift as nd_shift

from decanter.waveshift.measure import (
    cc_shift_search,
    waveshift_clip,
    waveshift_one_order,
)


def _synthetic_spectrum(n: int = 5000, seed: int = 0) -> np.ndarray:
    """Two-Gaussian synthetic spectrum + Gaussian noise."""
    x = np.arange(n)
    spec = (
        np.exp(-((x - 2500) / 200) ** 2)
        + 0.5 * np.exp(-((x - 2000) / 100) ** 2)
        + 0.3 * np.exp(-((x - 3500) / 50) ** 2)
    )
    rng = np.random.default_rng(seed)
    return spec + rng.normal(scale=0.01, size=n)


def test_recover_known_shift() -> None:
    """A 1.3-pixel shift is recovered to ~0.1 px (= 0.05 wave units at cdelt=0.5)."""
    ref = _synthetic_spectrum(seed=0)
    sp = nd_shift(ref, 1.3, order=3)
    shifts = waveshift_one_order([ref, sp], cdelt1=0.5, refid=0)
    assert shifts[0] == 0.0
    # The function returns the shift TO APPLY to sp to align with ref;
    # since sp = shift(ref, +1.3), we expect shift ≈ -1.3 px = -0.65 cdelt.
    assert abs(shifts[1] - (-0.65)) < 0.1, f"got {shifts[1]}"


def test_zero_shift_recovered() -> None:
    """If two spectra are identical, the shift should be ~0."""
    spec = _synthetic_spectrum(seed=1)
    shifts = waveshift_one_order([spec, spec.copy()], cdelt1=0.5, refid=0)
    assert abs(shifts[1]) < 0.05


def test_cc_shift_search_min_at_correct_shift() -> None:
    """The SSE grid's argmin is at the actual shift."""
    ref = _synthetic_spectrum(seed=2)
    sp = nd_shift(ref, 0.7, order=3)
    shifts, dify = cc_shift_search(ref, sp, cshift=0.0, width=2.0, step=0.1)
    best = float(shifts[int(np.argmin(dify))])
    # We expect best ≈ -0.7.
    assert abs(best - (-0.7)) < 0.15, f"got {best}"


def test_waveshift_clip_rejects_outliers() -> None:
    """A single outlier order shouldn't drag the mean."""
    n_orders, n_frames = 10, 3
    mat = np.full((n_orders, n_frames), 0.1)
    mat[5, :] = 1.0  # outlier order
    avg, n_kept, sd, flag = waveshift_clip(mat, sigma_1st=1.0, sigma=2.0, iterate=5)
    assert avg[0] == pytest.approx(0.1, abs=0.02)
    assert n_kept[0] == 9


def test_waveshift_clip_flags_high_std() -> None:
    n_orders, n_frames = 5, 1
    mat = np.array([[0.0], [0.5], [1.0], [-0.5], [-1.0]])  # scattered
    avg, n_kept, sd, flag = waveshift_clip(mat, std_thres=0.1)
    assert flag[0] >= 1, "should be flagged for high stddev"
