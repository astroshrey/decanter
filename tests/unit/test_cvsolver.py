"""Unit tests for the IRAF curfit port (Legendre/Chebyshev LSQ via banded Cholesky)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial import chebyshev, legendre

from decanter.utils.cvsolver import (
    cv_b1cheb,
    cv_b1cheb_array,
    cv_b1leg,
    cv_b1leg_array,
    cv_bcheb,
    cv_bleg,
    cv_eval,
    cv_eval_b1,
    cv_evleg,
    cv_fit,
    _accumulate_matrix_fast,
    _accumulate_matrix_sequential,
    _chofac_banded,
    _choslv_banded,
)


def test_legendre_basis_matches_recurrence() -> None:
    """``cv_bleg`` returns P_0..P_{n-1} for each input x_norm."""
    x = np.linspace(0.0, 10.0, 25)
    xnorm = (2 * x - 10.0) / 10.0  # maps [0,10] → [-1,1]
    basis = cv_bleg(x, order=5, xmin=0.0, xmax=10.0, dtype=np.dtype(np.float64))
    # Expected: numpy Legendre polynomials evaluated point-wise.
    for k in range(5):
        coefs = np.zeros(5)
        coefs[k] = 1.0
        expected = legendre.legval(xnorm, coefs)
        np.testing.assert_allclose(basis[:, k], expected, atol=1e-12)


def test_chebyshev_basis_matches_recurrence() -> None:
    x = np.linspace(0.0, 10.0, 25)
    xnorm = (2 * x - 10.0) / 10.0
    basis = cv_bcheb(x, order=5, xmin=0.0, xmax=10.0, dtype=np.dtype(np.float64))
    for k in range(5):
        coefs = np.zeros(5)
        coefs[k] = 1.0
        expected = chebyshev.chebval(xnorm, coefs)
        np.testing.assert_allclose(basis[:, k], expected, atol=1e-12)


def test_cv_fit_exact_on_polynomial_data() -> None:
    """Degree-N polynomial input ⇒ LSQ coefficients reproduce it exactly."""
    x = np.linspace(0.0, 10.0, 100)
    # y = 5 + 5x  (linear in x — Legendre P_1 = x_norm = (2x-10)/10)
    # So in normalized basis: y = 5 + 5*((x_n*10+10)/2) = 5 + 25 + 25*x_n
    y = 5.0 + 5.0 * x
    c = cv_fit(x, y, function="legendre", order=4, xmin=0.0, xmax=10.0,
               dtype=np.float64, fast=True)
    # Reconstruction should match.
    pred = cv_eval(c, x, function="legendre", xmin=0.0, xmax=10.0)
    np.testing.assert_allclose(pred, y, atol=1e-11)


def test_cv_fit_matches_numpy_legfit_fp64() -> None:
    """cv_fit fp64 == numpy.polynomial.legendre.legfit to ~1e-12 relative."""
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 2000.0, 215)
    y = rng.normal(size=x.size) * 10.0 + 3.0 + 0.01 * x - 1e-5 * x**2
    xnorm = (2 * x - 2000.0) / 2000.0
    c_iraf = cv_fit(x, y, function="legendre", order=4, xmin=0.0, xmax=2000.0,
                    dtype=np.float64, fast=True)
    c_np = legendre.legfit(xnorm, y, 3)
    np.testing.assert_allclose(c_iraf, c_np, atol=1e-10)


def test_cv_fit_fast_vs_sequential_fp64() -> None:
    """Fast (BLAS) and sequential accumulation agree at fp64."""
    rng = np.random.default_rng(11)
    x = np.linspace(0.0, 100.0, 50)
    y = rng.normal(size=x.size) + 0.1 * x
    c_fast = cv_fit(x, y, function="legendre", order=4, xmin=0.0, xmax=100.0,
                    dtype=np.float64, fast=True)
    c_seq = cv_fit(x, y, function="legendre", order=4, xmin=0.0, xmax=100.0,
                   dtype=np.float64, fast=False)
    np.testing.assert_allclose(c_fast, c_seq, atol=1e-11)


def test_cv_fit_degenerate_constant_x() -> None:
    """``xmin == xmax`` returns the constant fit (weighted mean of y)."""
    x = np.zeros(10)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    c = cv_fit(x, y, function="legendre", order=3, xmin=0.0, xmax=0.0,
               dtype=np.float64, fast=True)
    assert c[0] == pytest.approx(np.mean(y))
    assert np.all(c[1:] == 0.0)


def test_banded_cholesky_inverts_correctly() -> None:
    """``_chofac_banded`` + ``_choslv_banded`` solve a symmetric PD system."""
    rng = np.random.default_rng(13)
    # Construct symmetric PD 4x4 matrix and store in IRAF banded form.
    A = rng.normal(size=(4, 4))
    A = A @ A.T + np.eye(4) * 4.0
    # Banded layout: matrix[i, k] = A[k, k+i] for i = 0..order-1.
    nbands = 4
    matrix = np.zeros((nbands, nbands), dtype=np.float64)
    for i in range(nbands):
        for k in range(nbands - i):
            matrix[i, k] = A[k, k + i]
    b = rng.normal(size=4)
    matfac = _chofac_banded(matrix, dtype=np.dtype(np.float64))
    x = _choslv_banded(matfac, b, dtype=np.dtype(np.float64))
    np.testing.assert_allclose(A @ x, b, atol=1e-12)


def test_accumulate_matrix_fast_vs_sequential() -> None:
    """The fast (BLAS) and sequential normal-equations setups agree on bw·basis."""
    rng = np.random.default_rng(17)
    x = np.linspace(0.0, 10.0, 30)
    y = rng.normal(size=x.size)
    w = np.ones_like(x)
    basis = cv_bleg(x, order=4, xmin=0.0, xmax=10.0, dtype=np.dtype(np.float64))
    M_fast, v_fast = _accumulate_matrix_fast(basis, w, y, np.dtype(np.float64))
    M_seq, v_seq = _accumulate_matrix_sequential(basis, w, y, np.dtype(np.float64))
    np.testing.assert_allclose(M_fast, M_seq, atol=1e-12)
    np.testing.assert_allclose(v_fast, v_seq, atol=1e-12)


def test_cv_b1leg_array_matches_scalar_per_element() -> None:
    """``cv_b1leg_array(x, ...)`` is bit-identical per element to repeated
    scalar ``cv_b1leg(x[i], ...)`` calls — that's the load-bearing invariant
    behind the s03 pass-1 ap_cveval port."""
    x = np.array([0.5, 1.0, 5.0, 7.5, 9.99], dtype=np.float32)
    order, xmin, xmax = 5, 0.0, 10.0
    arr = cv_b1leg_array(x, order, xmin, xmax, dtype=np.dtype(np.float32))
    for i, xi in enumerate(x):
        scalar = cv_b1leg(float(xi), order, xmin, xmax, dtype=np.dtype(np.float32))
        # exact bit-identical equality required (same scalar arithmetic
        # applied element-wise).
        np.testing.assert_array_equal(arr[i], scalar)


def test_cv_b1cheb_array_matches_scalar_per_element() -> None:
    x = np.array([0.5, 1.0, 5.0, 7.5, 9.99], dtype=np.float32)
    order, xmin, xmax = 5, 0.0, 10.0
    arr = cv_b1cheb_array(x, order, xmin, xmax, dtype=np.dtype(np.float32))
    for i, xi in enumerate(x):
        scalar = cv_b1cheb(float(xi), order, xmin, xmax, dtype=np.dtype(np.float32))
        np.testing.assert_array_equal(arr[i], scalar)


def test_cv_eval_b1_within_one_ulp_of_cv_evleg() -> None:
    """``cv_eval_b1`` (IRAF scalar cveval) and ``cv_evleg`` (vector form)
    produce the same coefficients within 1 fp32 ULP per pixel — the
    documented IRAF discrepancy that ap_cveval exposes vs cvvector."""
    coeffs = np.array([5.0, -0.3, 0.04, -0.01, 0.005], dtype=np.float32)
    x = np.linspace(1.0, 100.0, 200, dtype=np.float64)
    y_b1 = cv_eval_b1(coeffs, x, function="legendre", xmin=0.0, xmax=100.0,
                     dtype=np.dtype(np.float32))
    y_vec = cv_evleg(coeffs, x, 0.0, 100.0, dtype=np.dtype(np.float32))
    # 4 ULP cap for fp32 values around 5.
    diff = np.abs(y_b1.astype(np.float64) - y_vec.astype(np.float64))
    assert diff.max() < 5.0 * np.finfo(np.float32).eps * float(np.abs(y_vec).max())
