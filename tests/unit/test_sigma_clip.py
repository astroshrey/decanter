"""Unit tests for :func:`decanter.utils.sigma_clip.per_segment_clip`."""

from __future__ import annotations

import numpy as np
import pytest

from decanter.utils.sigma_clip import per_segment_clip


def test_two_segments_clean_data() -> None:
    """No rejection on clean Gaussian data; mean/std match per-segment values."""
    rng = np.random.default_rng(seed=0)
    n_per_seg = 1000
    seg_a = rng.normal(loc=10.0, scale=1.0, size=n_per_seg)
    seg_b = rng.normal(loc=20.0, scale=2.0, size=n_per_seg)
    values = np.concatenate([seg_a, seg_b])
    segments = np.concatenate([np.zeros(n_per_seg, dtype=int), np.ones(n_per_seg, dtype=int)])

    mean, std, count = per_segment_clip(values, segments, n_segments=2, sigma=10.0, iterations=1)

    assert np.isclose(mean[0], 10.0, atol=0.1)
    assert np.isclose(mean[1], 20.0, atol=0.2)
    assert np.isclose(std[0], 1.0, atol=0.1)
    assert np.isclose(std[1], 2.0, atol=0.2)
    assert count[0] == n_per_seg
    assert count[1] == n_per_seg


def test_outlier_rejection_within_segment() -> None:
    """A single huge outlier in one segment gets clipped over iterations."""
    rng = np.random.default_rng(seed=1)
    clean = rng.normal(loc=0.0, scale=1.0, size=100)
    contaminated = np.append(clean, 1e6)  # huge outlier
    segments = np.zeros(101, dtype=int)

    # ``iterations=0`` returns initial stats — the outlier is still
    # present, so std is huge.
    mean_pre, std_pre, _ = per_segment_clip(contaminated, segments, n_segments=1,
                                            sigma=3.0, iterations=0)
    mean_post, std_post, count_post = per_segment_clip(contaminated, segments, n_segments=1,
                                                      sigma=3.0, iterations=5)

    # Pre-clip std with the 1e6 outlier present.
    assert std_pre[0] > 1e4
    # With multiple iterations, the outlier is rejected and std collapses.
    assert std_post[0] < 2.0
    # The huge outlier is gone; ~99-100 clean pixels survive (at sigma=3, a
    # rare borderline Gaussian sample may also be clipped — expected loss <2).
    assert count_post[0] >= 99
    assert count_post[0] <= 100


def test_empty_segment_yields_nan() -> None:
    """A segment with no pixels reports nan/nan/0."""
    values = np.array([1.0, 2.0, 3.0])
    segments = np.array([0, 0, 0])  # segment 1 unused
    mean, std, count = per_segment_clip(values, segments, n_segments=2, sigma=5.0)
    assert np.isnan(mean[1])
    assert np.isnan(std[1])
    assert count[1] == 0


def test_out_of_range_segment_ids_excluded() -> None:
    """Pixels with segment IDs outside [0, n_segments) don't contribute."""
    values = np.array([100.0, 1.0, 2.0, 3.0])  # first pixel sentinel-masked
    segments = np.array([-1, 0, 0, 0])
    mean, std, count = per_segment_clip(values, segments, n_segments=1, sigma=5.0)
    assert count[0] == 3
    assert np.isclose(mean[0], 2.0)


def test_zero_variance_segment_does_not_produce_nan() -> None:
    """Constant data in a segment yields std=0 (not nan from float roundoff)."""
    values = np.full(10, 7.5, dtype=np.float64)
    segments = np.zeros(10, dtype=int)
    mean, std, count = per_segment_clip(values, segments, n_segments=1, sigma=3.0, iterations=3)
    assert np.isclose(mean[0], 7.5)
    assert std[0] == 0.0
    assert count[0] == 10


def test_no_python_loop_at_scale() -> None:
    """12 000 segments × 100 pixels each runs in well under a second."""
    import time
    rng = np.random.default_rng(seed=42)
    n_seg = 12_000  # matches s02's tile count
    n_per = 100
    values = rng.normal(size=n_seg * n_per).astype(np.float32)
    segments = np.repeat(np.arange(n_seg), n_per)

    t = time.perf_counter()
    mean, std, count = per_segment_clip(values, segments, n_segments=n_seg, sigma=5.0,
                                        iterations=3)
    elapsed = time.perf_counter() - t

    assert elapsed < 1.0, f"expected <1s for 1.2M points / 12k segments, got {elapsed:.3f}s"
    assert mean.shape == (n_seg,)
    assert count.sum() <= n_seg * n_per  # at most every input pixel counted


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="share shape"):
        per_segment_clip(np.zeros(3), np.zeros(4, dtype=int), n_segments=1)
    with pytest.raises(ValueError, match="n_segments must be positive"):
        per_segment_clip(np.zeros(3), np.zeros(3, dtype=int), n_segments=0)
    with pytest.raises(ValueError, match="sigma must be positive"):
        per_segment_clip(np.zeros(3), np.zeros(3, dtype=int), n_segments=1, sigma=0.0)
    with pytest.raises(ValueError, match="iterations must be"):
        per_segment_clip(np.zeros(3), np.zeros(3, dtype=int), n_segments=1, iterations=-1)


def test_accepts_2d_input_via_ravel() -> None:
    """Multi-dimensional inputs are flattened; same answer as 1-D."""
    rng = np.random.default_rng(seed=0)
    data_2d = rng.normal(size=(20, 30)).astype(np.float32)
    seg_2d = np.zeros((20, 30), dtype=int)

    mean_2d, std_2d, _ = per_segment_clip(data_2d, seg_2d, n_segments=1, iterations=1)
    mean_1d, std_1d, _ = per_segment_clip(data_2d.ravel(), seg_2d.ravel(),
                                          n_segments=1, iterations=1)
    assert np.isclose(mean_2d[0], mean_1d[0])
    assert np.isclose(std_2d[0], std_1d[0])
