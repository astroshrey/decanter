"""Unit tests for the IRAF spline3 port."""

from __future__ import annotations

import numpy as np
import pytest

from decanter.utils.spline3 import (
    iraf_spline3_coeffs,
    iraf_spline3_coeffs_batch,
    iraf_spline3_eval,
    iraf_spline3_eval_batch,
    iraf_spline3_resample_columns,
)


def test_natural_cubic_spline_interpolates_at_knots_linear() -> None:
    """Linear data: spline is exact at every knot AND every fractional x."""
    y = np.arange(1, 11, dtype=np.float64)
    bc = iraf_spline3_coeffs(y)
    xq = np.array([1.0, 1.5, 2.7, 5.5, 9.0, 9.99, 10.0])
    v = iraf_spline3_eval(bc, xq)
    np.testing.assert_allclose(v, xq, atol=1e-13)


def test_natural_cubic_spline_at_knots_quadratic() -> None:
    """Quadratic data: spline interpolates exactly at knots."""
    y = (np.arange(1, 11, dtype=np.float64)) ** 2
    bc = iraf_spline3_coeffs(y)
    xq = np.arange(1, 11, dtype=np.float64)
    v = iraf_spline3_eval(bc, xq)
    np.testing.assert_allclose(v, xq ** 2, atol=1e-12)


def test_boundary_clamps_to_endpoint_value() -> None:
    """Out-of-range x is clamped to [1, n]."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    bc = iraf_spline3_coeffs(y)
    v_low = iraf_spline3_eval(bc, np.array([0.5, -10.0, 1.0]))
    v_high = iraf_spline3_eval(bc, np.array([5.0, 5.5, 100.0]))
    # All low-side queries should equal the value at x=1.
    assert v_low[0] == pytest.approx(v_low[2])
    assert v_low[1] == pytest.approx(v_low[2])
    # All high-side queries should equal the value at x=n.
    assert v_high[1] == pytest.approx(v_high[0])
    assert v_high[2] == pytest.approx(v_high[0])


def test_batch_matches_per_column() -> None:
    """Batched solver/eval = per-column scalar versions."""
    rng = np.random.default_rng(42)
    n, m = 20, 5
    data = rng.normal(size=(n, m))
    bc_batch = iraf_spline3_coeffs_batch(data)
    for j in range(m):
        bc_single = iraf_spline3_coeffs(data[:, j])
        np.testing.assert_allclose(bc_batch[:, j], bc_single, atol=1e-13)

    y_query = rng.uniform(1.0, float(n), size=(7, m))
    v_batch = iraf_spline3_eval_batch(bc_batch, y_query)
    for j in range(m):
        v_single = iraf_spline3_eval(bc_batch[:, j], y_query[:, j])
        np.testing.assert_allclose(v_batch[:, j], v_single, atol=1e-13)


def test_resample_columns_identity_at_integer_grid() -> None:
    """Resampling at the original integer 1-indexed grid returns the input."""
    rng = np.random.default_rng(7)
    H, n_x = 50, 4
    data = rng.normal(size=(H, n_x))
    y_grid = np.broadcast_to(
        np.arange(1, H + 1, dtype=np.float64)[:, None], (H, n_x)
    ).copy()
    out = iraf_spline3_resample_columns(data, y_grid)
    np.testing.assert_allclose(out, data.astype(np.float32), atol=1e-6)


def test_resample_columns_constant_data() -> None:
    """Constant column ⇒ constant output everywhere."""
    H, n_x = 12, 3
    data = np.full((H, n_x), 7.5)
    rng = np.random.default_rng(3)
    y = rng.uniform(-5.0, float(H + 5), size=(20, n_x))
    out = iraf_spline3_resample_columns(data, y)
    np.testing.assert_allclose(out, np.float32(7.5), atol=1e-6)
