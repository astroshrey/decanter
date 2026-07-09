"""Unit tests for :mod:`decanter.utils.iraf_icfit`."""

from __future__ import annotations

import numpy as np
import pytest

from decanter.utils.iraf_icfit import evaluate, fit_with_reject


def test_exact_polynomial_recovered_with_no_outliers() -> None:
    """A clean Legendre series is recovered to machine precision."""
    x = np.linspace(0, 100, 200)
    # Use a simple polynomial that any Legendre series of degree 3 can fit.
    y = 1.0 + 0.5 * x - 0.001 * x**2 + 1e-6 * x**3
    result = fit_with_reject(
        x, y, degree=3, low_reject=3.0, high_reject=3.0, niterate=10
    )
    pred = evaluate(result, x)
    assert np.allclose(pred, y, atol=1e-6)
    assert result.mask.all()


def test_high_outlier_rejected_when_high_reject_set() -> None:
    """A single bright spike above the fit gets rejected."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 200)
    y = 1.0 + 0.01 * x + rng.normal(scale=0.05, size=x.size)
    # Inject a large positive outlier.
    y[100] += 100.0
    result = fit_with_reject(
        x, y, degree=1, low_reject=3.0, high_reject=2.0, niterate=20
    )
    assert result.mask[100] == False  # noqa: E712 — explicit numpy bool


def test_low_outlier_rejected_when_low_reject_set() -> None:
    """A single deep dip below the fit gets rejected."""
    rng = np.random.default_rng(1)
    x = np.linspace(0, 100, 200)
    y = 1.0 + 0.01 * x + rng.normal(scale=0.05, size=x.size)
    y[150] -= 100.0
    result = fit_with_reject(
        x, y, degree=1, low_reject=2.0, high_reject=3.0, niterate=20
    )
    assert result.mask[150] == False  # noqa: E712


def test_high_reject_zero_disables_upper() -> None:
    """``high_reject=0`` keeps even huge upper outliers."""
    rng = np.random.default_rng(2)
    x = np.linspace(0, 100, 200)
    y = 1.0 + rng.normal(scale=0.05, size=x.size)
    y[50] += 1000.0
    result = fit_with_reject(
        x, y, degree=0, low_reject=3.0, high_reject=0.0, niterate=20
    )
    assert result.mask[50] == True  # noqa: E712


def test_chebyshev_function_path() -> None:
    """Chebyshev fits also work and converge on noise-free data."""
    x = np.linspace(-1, 1, 50)
    y = 2.0 + 3.0 * x + 0.5 * x**2
    result = fit_with_reject(
        x, y, degree=2, low_reject=3.0, high_reject=3.0, niterate=5,
        function="chebyshev",
    )
    pred = evaluate(result, x)
    # cv_fit runs at float32 (matching IRAF TY_REAL); tolerance reflects
    # fp32 roundoff in the 50-point normal-equations inner products.
    assert np.allclose(pred, y, atol=1e-4)


def test_early_stop_when_no_rejection() -> None:
    """If no point is ever rejected, iterations < niterate."""
    x = np.linspace(0, 1, 30)
    y = np.ones_like(x) * 3.0
    result = fit_with_reject(
        x, y, degree=0, low_reject=3.0, high_reject=3.0, niterate=50
    )
    assert result.iterations <= 2


def test_degenerate_constant_x_returns_finite_result() -> None:
    """``x_min == x_max`` shouldn't NaN out the normalization."""
    x = np.zeros(10)
    y = np.ones(10)
    result = fit_with_reject(
        x, y, degree=0, low_reject=3.0, high_reject=3.0, niterate=3
    )
    pred = evaluate(result, x)
    assert np.all(np.isfinite(pred))
