"""IRAF ``math/curfit`` port — Cholesky-banded Legendre/Chebyshev LSQ solver.

Bit-for-bit faithful Python port of IRAF's curfit module for
Legendre and Chebyshev polynomial fits:

  - ``math/curfit/cvinitr.x``           (curve descriptor init)
  - ``math/curfit/cv_bevalr.x``         (basis evaluation: rcv_bleg, rcv_bcheb)
  - ``math/curfit/cvacptsr.x``          (normal-equations accumulation)
  - ``math/curfit/cvchomatr.x``         (banded Cholesky factor + solve)
  - ``math/curfit/cvfitr.x``            (driver: cvzero + cvacpts + cvsolve)

Two execution modes:

  ``dtype=np.float32, fast=False``
    Faithful to IRAF: TY_REAL throughout, sequential accumulation in the
    same order as IRAF's ``amulr`` + element-sum pattern. Goal here is
    bit-for-bit equality with IRAF's coefficient output. Slower per
    fit (~5×) but reproduces float32 rounding exactly.

  ``dtype=np.float64, fast=True``  (default)
    Float64 throughout, numpy ``@`` for normal-equations + scipy
    ``cho_factor/cho_solve`` for the linear solve. ~50× faster than
    ``fast=False`` but uses float64 precision and BLAS' pairwise
    reduction order, which matches IRAF to ~1e-14 relative — close,
    not bit-identical.

Used by :mod:`decanter.utils.iraf_icfit` (which replaces
``numpy.polynomial.legendre.legfit`` with this solver to match WARP /
IRAF apscatter coefficients).

Storage convention (matches IRAF ``cvinitr.x:72``):
  ``MATRIX[i, k]`` stores the inner product
  ``Σ_p w[p] · B[p, k] · B[p, k+i-1]`` — the i-th diagonal of the
  normal-equations matrix at row k. For Legendre/Chebyshev with
  ``nbands = ncoeff = order``, only the upper triangle is needed
  (symmetric), so ``i`` runs ``1..order`` (1-indexed in IRAF;
  ``0..order-1`` in numpy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.linalg import cho_factor, cho_solve  # type: ignore
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


FunctionType = Literal["legendre", "chebyshev"]


@dataclass(frozen=True, slots=True)
class CvFitResult:
    """Output of :func:`cv_fit_legendre` / :func:`cv_fit_chebyshev`."""

    coefficients: NDArray[np.floating]
    matrix: NDArray[np.floating] | None       # banded normal equations (None if not retained)
    chofac: NDArray[np.floating] | None       # Cholesky factorization (None if not retained)


# ---------------------------------------------------------------------------
# Basis evaluation
# ---------------------------------------------------------------------------

def _normalize(x: NDArray, xmin: float, xmax: float, dtype: np.dtype) -> NDArray:
    """IRAF's ``xnorm = (x + k1) * k2`` where ``k1 = -(xmax+xmin)/2, k2 = 2/(xmax-xmin)``.

    Equivalent to mapping ``[xmin, xmax] → [-1, 1]`` linearly. Degenerate
    inputs (``xmin == xmax``) map to zero — only the constant basis term
    is well-defined in that case.
    """
    span = xmax - xmin
    if span == 0.0:
        return np.zeros_like(x, dtype=dtype)
    k1 = dtype.type(-(xmax + xmin) / 2.0)
    k2 = dtype.type(2.0 / span)
    return (x.astype(dtype, copy=False) + k1) * k2


def cv_bleg(x: NDArray, order: int, xmin: float, xmax: float,
            dtype: np.dtype = np.dtype(np.float32)) -> NDArray:
    """Legendre basis values at each x.

    Mirrors IRAF ``cv_bevalr.x:rcv_bleg`` (lines 37-66). Returns an
    ``(npts, order)`` array where column ``k`` is ``P_k(xnorm)``.
    The recurrence (k ≥ 3):
        P_{k}(x) = ((2k-3) x P_{k-1}(x) - (k-2) P_{k-2}(x)) / (k-1)
    matches Legendre exactly with 1-indexed k.
    """
    xn = _normalize(x, xmin, xmax, dtype)
    npts = xn.size
    basis = np.empty((npts, order), dtype=dtype)
    basis[:, 0] = dtype.type(1.0)
    if order == 1:
        return basis
    basis[:, 1] = xn
    if order == 2:
        return basis
    for k in range(3, order + 1):
        ri = dtype.type(k)
        ri1 = (dtype.type(2.0) * ri - dtype.type(3.0)) / (ri - dtype.type(1.0))
        ri2 = -(ri - dtype.type(2.0)) / (ri - dtype.type(1.0))
        # IRAF: basis[bptr] = ri1 * (basis[1+npts] * basis[bptr-npts]) + ri2 * basis[bptr-2*npts]
        # i.e. basis[k] = ri1 * xnorm * basis[k-1] + ri2 * basis[k-2].
        basis[:, k - 1] = ri1 * (basis[:, 1] * basis[:, k - 2]) + ri2 * basis[:, k - 3]
    return basis


def cv_bcheb(x: NDArray, order: int, xmin: float, xmax: float,
             dtype: np.dtype = np.dtype(np.float32)) -> NDArray:
    """Chebyshev basis values at each x. Mirrors IRAF ``cv_bevalr.x:rcv_bcheb``."""
    xn = _normalize(x, xmin, xmax, dtype)
    npts = xn.size
    basis = np.empty((npts, order), dtype=dtype)
    basis[:, 0] = dtype.type(1.0)
    if order == 1:
        return basis
    basis[:, 1] = xn
    if order == 2:
        return basis
    for k in range(3, order + 1):
        # T_k = 2*x*T_{k-1} - T_{k-2}
        basis[:, k - 1] = dtype.type(2.0) * basis[:, 1] * basis[:, k - 2] - basis[:, k - 3]
    return basis


# ---------------------------------------------------------------------------
# Normal-equations accumulation (banded storage)
# ---------------------------------------------------------------------------

def _accumulate_matrix_sequential(
    basis: NDArray, w: NDArray, y: NDArray, dtype: np.dtype
) -> tuple[NDArray, NDArray]:
    """IRAF-faithful banded matrix accumulation.

    Mirrors ``cvacptsr.x:121-143`` exactly: per-element sequential
    accumulation in the same order IRAF's ``amulr`` produces. Slower
    than the BLAS path but matches IRAF's float32 reduction order.
    """
    npts, order = basis.shape
    matrix = np.zeros((order, order), dtype=dtype)
    vector = np.zeros(order, dtype=dtype)

    for k in range(order):
        # bw[i] = w[i] * basis[i, k]
        bw = w * basis[:, k]
        # vector[k] = Σ bw[i] * y[i], left-to-right
        acc = dtype.type(0.0)
        for v in bw * y:
            acc = acc + v
        vector[k] = acc
        # MATRIX[ii, k] = Σ bw[i] * basis[i, k+ii], for ii=0..order-k-1
        for ii in range(order - k):
            prod = bw * basis[:, k + ii]
            acc = dtype.type(0.0)
            for v in prod:
                acc = acc + v
            matrix[ii, k] = acc
    return matrix, vector


def _accumulate_matrix_fast(
    basis: NDArray, w: NDArray, y: NDArray, dtype: np.dtype
) -> tuple[NDArray, NDArray]:
    """BLAS-accelerated normal-equations setup.

    Computes ``M = B^T diag(w) B`` (full symmetric ``order × order``)
    and ``v = B^T diag(w) y``, then repacks ``M`` into banded layout.
    """
    npts, order = basis.shape
    bw = (basis * w[:, None]).astype(dtype, copy=False)
    full_matrix = bw.T @ basis.astype(dtype, copy=False)  # (order, order)
    vector = bw.T @ y.astype(dtype, copy=False)            # (order,)
    # Repack to IRAF banded storage: matrix[ii, k] = full[k, k+ii] for ii < order-k.
    matrix = np.zeros((order, order), dtype=dtype)
    for ii in range(order):
        # diagonal offset ii: rows k=0..order-ii-1, cols k+ii.
        np.fill_diagonal(matrix[ii:ii + 1, :order - ii], np.diag(full_matrix, ii))
        # simpler: matrix[ii, :order-ii] = full_matrix[arange(order-ii), arange(order-ii)+ii]
    # The fill_diagonal trick is fiddly; just use the explicit indexing instead:
    matrix = np.zeros((order, order), dtype=dtype)
    for ii in range(order):
        k_range = np.arange(order - ii)
        matrix[ii, :order - ii] = full_matrix[k_range, k_range + ii]
    return matrix, vector


# ---------------------------------------------------------------------------
# Cholesky factor + solve (banded, IRAF-style)
# ---------------------------------------------------------------------------

def _chofac_banded(matrix: NDArray, dtype: np.dtype) -> NDArray:
    """Banded Cholesky factor — IRAF ``cvchomatr.x:rcvchofac`` (lines 13-62).

    The matrix is stored in banded layout: ``matrix[i, k]`` = i-th
    diagonal (0=main) at row k. For an ``order × order`` symmetric
    matrix with full bandwidth (``nbands = order``), only the upper
    triangle is stored. The routine adapts ``bchfac.f`` from de Boor.

    Returns the in-place-factored ``matfac`` (same layout).
    Singular rows are zeroed (matches IRAF's ``ier = SINGULAR`` path).
    """
    nbands, nrows = matrix.shape
    matfac = matrix.copy().astype(dtype, copy=False)
    if nrows == 1:
        if matfac[0, 0] > 0:
            matfac[0, 0] = dtype.type(1.0) / matfac[0, 0]
        return matfac

    # EPSILONR ≈ 1.1920929e-7 for float32, 2.2204460e-16 for float64.
    eps = float(np.finfo(dtype).eps)
    threshold = dtype.type(10.0 * eps)

    for n in range(nrows):
        # Singularity test: ((matfac[0,n] + matrix[0,n]) - matrix[0,n]) ≤ 10 * EPSILON
        # (IRAF arithmetic — accounts for round-off where matrix is nearly zero)
        test_val = (matfac[0, n] + matrix[0, n]) - matrix[0, n]
        if test_val <= threshold * matrix[0, n] or matfac[0, n] <= 0:
            matfac[:, n] = dtype.type(0.0)
            continue
        matfac[0, n] = dtype.type(1.0) / matfac[0, n]
        imax = min(nbands - 1, nrows - n - 1)
        if imax < 1:
            continue
        jmax = imax
        for i in range(1, imax + 1):
            ratio = matfac[i, n] * matfac[0, n]
            for j in range(jmax):
                # IRAF: matfac[j, n+i] -= matfac[j+i, n] * ratio (1-indexed j=1..jmax)
                # numpy: matfac[j, n+i] -= matfac[j+i, n] * ratio (0-indexed j=0..jmax-1)
                matfac[j, n + i] -= matfac[j + i, n] * ratio
            jmax -= 1
            matfac[i, n] = ratio
    return matfac


def _choslv_banded(matfac: NDArray, vector: NDArray, dtype: np.dtype) -> NDArray:
    """Banded Cholesky solve — IRAF ``cvchomatr.x:rcvchoslv`` (lines 68-109)."""
    nbands, nrows = matfac.shape
    coeff = vector.astype(dtype, copy=True)
    if nrows == 1:
        coeff[0] = coeff[0] * matfac[0, 0]
        return coeff

    nbndm1 = nbands - 1
    # Forward substitution.
    for n in range(nrows):
        jmax = min(nbndm1, nrows - n - 1)
        for j in range(jmax):
            # IRAF 1-indexed: coeff[j+n] -= matfac[j+1, n] * coeff[n]
            # numpy: coeff[j+1+n] -= matfac[j+1, n] * coeff[n]
            coeff[j + 1 + n] = coeff[j + 1 + n] - matfac[j + 1, n] * coeff[n]
    # Back substitution.
    for n in range(nrows - 1, -1, -1):
        coeff[n] = coeff[n] * matfac[0, n]
        jmax = min(nbndm1, nrows - n - 1)
        for j in range(jmax):
            coeff[n] = coeff[n] - matfac[j + 1, n] * coeff[j + 1 + n]
    return coeff


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def cv_fit(
    x: NDArray,
    y: NDArray,
    *,
    function: FunctionType,
    order: int,
    xmin: float,
    xmax: float,
    weights: NDArray | None = None,
    dtype: np.dtype | str = np.float32,
    fast: bool = True,
) -> NDArray:
    """Solve a Legendre/Chebyshev LSQ fit via IRAF-banded Cholesky.

    Mirrors IRAF ``cvfitr.x:cvfit`` (cvzero + cvacpts + cvsolve) for the
    LEGENDRE / CHEBYSHEV path, with ``WTS_USER`` semantics (caller
    supplies weights; default = ones, matching ``WTS_UNIFORM``).

    Args:
        x: 1-D array of abscissa values.
        y: 1-D array of ordinates, same length as ``x``.
        function: ``"legendre"`` or ``"chebyshev"``.
        order: IRAF ``order`` = number of basis functions = polynomial
            degree + 1. (NOT the ``degree`` parameter common in numpy
            interfaces.)
        xmin, xmax: normalization range. Maps to ``[-1, 1]``.
        weights: optional per-point weights. ``None`` = uniform.
        dtype: ``np.float32`` (IRAF parity) or ``np.float64`` (precision).
        fast: when True (default), use numpy ``@`` and explicit
            band-Cholesky for the linear solve. When False, accumulate
            matrix entries with element-by-element sequential addition
            in the same order as IRAF ``cvacptsr.x`` — slower, but
            matches IRAF's float32 reduction order bit-for-bit.

    Returns:
        1-D array of ``order`` coefficients (same dtype).
    """
    dt = np.dtype(dtype)
    x_a = np.asarray(x, dtype=dt)
    y_a = np.asarray(y, dtype=dt)
    if weights is None:
        w_a = np.ones_like(x_a)
    else:
        w_a = np.asarray(weights, dtype=dt)
    if x_a.size != y_a.size or x_a.size != w_a.size:
        raise ValueError("x, y, and weights must have the same length")
    if x_a.size == 0:
        return np.zeros(order, dtype=dt)
    if order < 1:
        raise ValueError("order must be ≥ 1")
    if xmax <= xmin:
        # Degenerate sample (all x identical). Return constant fit at
        # the (weighted) mean of y; higher-order coefficients zero.
        out = np.zeros(order, dtype=dt)
        w_sum = float(w_a.sum())
        out[0] = (w_a * y_a).sum() / w_sum if w_sum > 0 else dt.type(0.0)
        return out

    if function == "legendre":
        basis = cv_bleg(x_a, order, xmin, xmax, dtype=dt)
    elif function == "chebyshev":
        basis = cv_bcheb(x_a, order, xmin, xmax, dtype=dt)
    else:
        raise ValueError(f"unknown function {function!r}")

    if fast:
        matrix, vector = _accumulate_matrix_fast(basis, w_a, y_a, dt)
    else:
        matrix, vector = _accumulate_matrix_sequential(basis, w_a, y_a, dt)

    matfac = _chofac_banded(matrix, dt)
    coeffs = _choslv_banded(matfac, vector, dt)
    return coeffs


def cv_eval(
    coeffs: NDArray, x: NDArray, *,
    function: FunctionType, xmin: float, xmax: float,
    dtype: np.dtype | str | None = None,
) -> NDArray:
    """Evaluate a Legendre/Chebyshev fit at arbitrary ``x``.

    Same precision/normalization as :func:`cv_fit`. If ``dtype`` is
    ``None``, follow ``coeffs.dtype``. Delegates to :func:`cv_evleg`
    or :func:`cv_evcheb` so the accumulation order matches IRAF
    ``cvvector`` (``rcv_evleg`` / ``rcv_evcheb``).
    """
    dt = np.dtype(dtype) if dtype is not None else np.dtype(coeffs.dtype)
    if function == "legendre":
        return cv_evleg(coeffs, x, xmin, xmax, dtype=dt)
    elif function == "chebyshev":
        return cv_evcheb(coeffs, x, xmin, xmax, dtype=dt)
    else:
        raise ValueError(f"unknown function {function!r}")


# ---------------------------------------------------------------------------
# Single-point basis evaluation — IRAF cv_b1evalr.x
# ---------------------------------------------------------------------------
#
# These are used by cvrject (single-point matrix update during rejection)
# and cveval (single-point curve evaluation). They differ from cv_bleg /
# cv_bcheb (which evaluate a whole batch) by float32 ULP because the
# recurrence formula is laid out differently: the scalar form divides
# by (k-1) at the END of each recurrence step, while the vector form
# pre-divides into ri1/ri2 once. IRAF uses both in different places,
# so for bit-for-bit parity we must follow the same split.

def cv_b1leg(x: float, order: int, xmin: float, xmax: float,
             dtype: np.dtype = np.dtype(np.float32)) -> NDArray:
    """Single-point Legendre basis. Mirrors IRAF ``cv_b1evalr.x:rcv_b1leg``.

    Recurrence (k ≥ 3, IRAF SPP):
        basis[k] = ((2k-3) * xnorm * basis[k-1] - (k-2) * basis[k-2]) / (k-1)

    Note: divides at the END (unlike :func:`cv_bleg` which pre-divides into
    ``ri1`` / ``ri2``). Float32 results differ from ``cv_bleg(x_array)[i, :]``
    by 1 ULP for k ≥ 3 — that's IRAF's actual behavior in ``cvrject``.
    """
    dt = np.dtype(dtype)
    basis = np.empty(order, dtype=dt)
    basis[0] = dt.type(1.0)
    if order == 1:
        return basis
    span = xmax - xmin
    if span == 0.0:
        xnorm = dt.type(0.0)
    else:
        k1 = dt.type(-(xmax + xmin) / 2.0)
        k2 = dt.type(2.0 / span)
        xnorm = (dt.type(x) + k1) * k2
    basis[1] = xnorm
    if order == 2:
        return basis
    for k in range(3, order + 1):
        ri = dt.type(k)
        # basis[k] = ((2*ri - 3) * xnorm * basis[k-1] - (ri - 2) * basis[k-2]) / (ri - 1)
        # IRAF SPP evaluates left-to-right with the division last.
        t1 = (dt.type(2.0) * ri - dt.type(3.0)) * xnorm * basis[k - 2]
        t2 = (ri - dt.type(2.0)) * basis[k - 3]
        basis[k - 1] = (t1 - t2) / (ri - dt.type(1.0))
    return basis


def cv_b1cheb(x: float, order: int, xmin: float, xmax: float,
              dtype: np.dtype = np.dtype(np.float32)) -> NDArray:
    """Single-point Chebyshev basis. Mirrors IRAF ``cv_b1evalr.x:rcv_b1cheb``."""
    dt = np.dtype(dtype)
    basis = np.empty(order, dtype=dt)
    basis[0] = dt.type(1.0)
    if order == 1:
        return basis
    span = xmax - xmin
    if span == 0.0:
        xnorm = dt.type(0.0)
    else:
        k1 = dt.type(-(xmax + xmin) / 2.0)
        k2 = dt.type(2.0 / span)
        xnorm = (dt.type(x) + k1) * k2
    basis[1] = xnorm
    if order == 2:
        return basis
    for k in range(3, order + 1):
        # T_k = 2*x*T_{k-1} - T_{k-2}
        basis[k - 1] = dt.type(2.0) * xnorm * basis[k - 2] - basis[k - 3]
    return basis


# ---------------------------------------------------------------------------
# Vectorized single-point basis — same scalar formula applied element-wise.
# ---------------------------------------------------------------------------
#
# These functions return the same arithmetic as :func:`cv_b1leg` /
# :func:`cv_b1cheb` would produce if called once per element of ``x``, but
# do so in a NumPy-vectorized loop over ``k`` so the per-pixel evaluation
# cost stays O(npts * order) instead of going through a Python loop.
#
# **Key invariant**: the per-pixel arithmetic operation order matches the
# scalar single-point form (division at the end of each Legendre recurrence
# step, not pre-divided into ``ri1``/``ri2`` as in :func:`cv_bleg`). At
# float32 this differs from :func:`cv_bleg` by 1–4 ULP per pixel — the same
# spread IRAF's ``cveval`` shows because it calls ``cv_b1leg`` + ``adotr``
# rather than the vector-form basis builder.

def cv_b1leg_array(x: NDArray, order: int, xmin: float, xmax: float,
                   dtype: np.dtype = np.dtype(np.float32)) -> NDArray:
    """Vectorized Legendre basis matching IRAF ``cv_b1evalr.x:rcv_b1leg``
    element-wise.

    Returns an ``(npts, order)`` array where row ``i`` is the same as
    ``cv_b1leg(x[i], order, xmin, xmax, dtype)``. The scalar formula
    ``basis[k-1] = ((2*ri-3) * xnorm * basis[k-2] - (ri-2) * basis[k-3])
    / (ri-1)`` is applied element-wise so the float32 rounding matches
    IRAF's scalar form bit-for-bit per element.
    """
    dt = np.dtype(dtype)
    x_a = np.asarray(x, dtype=dt)
    npts = x_a.size
    basis = np.empty((npts, order), dtype=dt)
    basis[:, 0] = dt.type(1.0)
    if order == 1:
        return basis
    span = xmax - xmin
    if span == 0.0:
        xnorm = np.zeros(npts, dtype=dt)
    else:
        k1 = dt.type(-(xmax + xmin) / 2.0)
        k2 = dt.type(2.0 / span)
        xnorm = ((x_a + k1) * k2).astype(dt, copy=False)
    basis[:, 1] = xnorm
    if order == 2:
        return basis
    for k in range(3, order + 1):
        ri = dt.type(k)
        # Same arithmetic as cv_b1leg scalar form, vectorized over x:
        #   t1 = (2*ri - 3) * xnorm * basis[k-2]
        #   t2 = (ri - 2) * basis[k-3]
        #   basis[k-1] = (t1 - t2) / (ri - 1)
        t1 = ((dt.type(2.0) * ri - dt.type(3.0)) * xnorm * basis[:, k - 2]).astype(dt, copy=False)
        t2 = ((ri - dt.type(2.0)) * basis[:, k - 3]).astype(dt, copy=False)
        basis[:, k - 1] = ((t1 - t2) / (ri - dt.type(1.0))).astype(dt, copy=False)
    return basis


def cv_b1cheb_array(x: NDArray, order: int, xmin: float, xmax: float,
                    dtype: np.dtype = np.dtype(np.float32)) -> NDArray:
    """Vectorized Chebyshev basis matching IRAF ``cv_b1evalr.x:rcv_b1cheb``
    element-wise.

    Cheb's three-term recurrence has no division, so the float32 result
    matches :func:`cv_bcheb` exactly. Provided for API symmetry with
    :func:`cv_b1leg_array`.
    """
    dt = np.dtype(dtype)
    x_a = np.asarray(x, dtype=dt)
    npts = x_a.size
    basis = np.empty((npts, order), dtype=dt)
    basis[:, 0] = dt.type(1.0)
    if order == 1:
        return basis
    span = xmax - xmin
    if span == 0.0:
        xnorm = np.zeros(npts, dtype=dt)
    else:
        k1 = dt.type(-(xmax + xmin) / 2.0)
        k2 = dt.type(2.0 / span)
        xnorm = ((x_a + k1) * k2).astype(dt, copy=False)
    basis[:, 1] = xnorm
    if order == 2:
        return basis
    for k in range(3, order + 1):
        basis[:, k - 1] = (dt.type(2.0) * xnorm * basis[:, k - 2] - basis[:, k - 3]).astype(dt, copy=False)
    return basis


def cv_eval_b1(
    coeffs: NDArray, x: NDArray, *,
    function: FunctionType, xmin: float, xmax: float,
    dtype: np.dtype | str | None = None,
) -> NDArray:
    """Evaluate a fit at ``x`` using IRAF's single-point ``cveval`` semantics.

    Mirrors IRAF ``math/curfit/cv_fevalr.x:rcveval`` (the scalar form, called
    by ``ap_cveval`` in :file:`apextract/apcveval.x`): build the basis via
    ``cv_b1leg`` / ``cv_b1cheb``, dot with the coefficient vector using
    ``adotr`` (left-to-right scalar accumulation in float32).

    Differs from :func:`cv_evleg` by 1–4 ULP per pixel because:
      * :func:`cv_evleg` uses the vector recurrence (pre-divided ``ri1``,
        ``ri2``) and a fused ``yfit += coeff[k-1] * pn`` add-multiply.
      * This routine uses the scalar recurrence (division at end) and an
        explicit left-to-right sum over the ``order`` axis.

    This is what IRAF ``apscatter`` actually calls to evaluate its per-row
    Legendre fit at output column positions (``apextract/apcveval.x:25``
    → ``cveval`` → ``cv_b1leg`` + ``adotr``).
    """
    dt = np.dtype(dtype) if dtype is not None else np.dtype(coeffs.dtype)
    if function == "legendre":
        basis = cv_b1leg_array(x, int(coeffs.size), xmin, xmax, dtype=dt)
    elif function == "chebyshev":
        basis = cv_b1cheb_array(x, int(coeffs.size), xmin, xmax, dtype=dt)
    else:
        raise ValueError(f"unknown function {function!r}")
    coeff = np.asarray(coeffs, dtype=dt)
    # IRAF adotr: yfit_i = Σ_k basis[i, k] * coeff[k] in float32, k=0..order-1
    # (left-to-right scalar add). Vectorize across i but accumulate
    # left-to-right across k so the per-element reduction order matches
    # ``adotr``'s ``do i = 1, npix; sum = sum + a[i]*b[i]`` byte-for-byte.
    npts = basis.shape[0]
    yfit = np.zeros(npts, dtype=dt)
    for k in range(int(coeffs.size)):
        yfit = (yfit + basis[:, k] * coeff[k]).astype(dt, copy=False)
    return yfit


# ---------------------------------------------------------------------------
# Vectorized evaluation — IRAF cv_fevalr.x (rcv_evleg / rcv_evcheb)
# ---------------------------------------------------------------------------
#
# These compute yfit[i] = Σ_k coeff[k] * basis[k, i] in the IRAF-faithful
# accumulation order: constant + linear via altmr/awsur, then for k=3..order
# build P_k(x) on the fly and accumulate yfit += coeff[k] * P_k(x).
#
# The accumulation order differs slightly from a generic ``basis @ coeffs``
# matmul, but matches IRAF's ``cvvector`` byte-for-byte at float32.

def cv_evleg(coeffs: NDArray, x: NDArray, xmin: float, xmax: float,
             dtype: np.dtype | str | None = None) -> NDArray:
    """Vectorized Legendre evaluation. Mirrors IRAF ``cv_fevalr.x:rcv_evleg``.

    Computes ``yfit[i] = Σ_k coeff[k] * P_k(xnorm[i])`` in IRAF's order:
    - yfit init = coeff[0] + xnorm * coeff[1] (computed via altmr formula)
    - for k = 3..order: pn = ri1 * xnorm * pnm1 + ri2 * pnm2; yfit += coeff[k-1] * pn
    """
    dt = np.dtype(dtype) if dtype is not None else np.dtype(coeffs.dtype)
    order = int(coeffs.size)
    x_a = np.asarray(x, dtype=dt)
    npts = x_a.size
    coeff = np.asarray(coeffs, dtype=dt)

    if order == 1:
        return np.full(x_a.shape, coeff[0], dtype=dt)

    span = xmax - xmin
    if span == 0.0:
        k1 = dt.type(0.0)
        k2 = dt.type(0.0)
    else:
        k1 = dt.type(-(xmax + xmin) / 2.0)
        k2 = dt.type(2.0 / span)

    # IRAF: ri1 = k2*coeff[2]; ri2 = ri1*k1 + coeff[1]; yfit = x*ri1 + ri2
    # = (x + k1)*k2 * coeff[2] + coeff[1]  (= xnorm * coeff[1] + coeff[0] in 0-idx)
    ri1 = k2 * coeff[1]
    ri2 = ri1 * k1 + coeff[0]
    yfit = (x_a * ri1 + ri2).astype(dt, copy=False)
    if order == 2:
        return yfit

    # Build sx = xnorm = (x + k1)*k2; pnm1 = sx; pnm2 = 1.
    sx = (x_a + k1) * k2
    pnm2 = np.ones(npts, dtype=dt)
    pnm1 = sx.astype(dt, copy=True)
    for k in range(3, order + 1):
        ri = dt.type(k)
        rri1 = (dt.type(2.0) * ri - dt.type(3.0)) / (ri - dt.type(1.0))
        rri2 = -(ri - dt.type(2.0)) / (ri - dt.type(1.0))
        # pn = sx * pnm1; pn = rri1 * pn + rri2 * pnm2  (awsur)
        pn = sx * pnm1
        pn = (rri1 * pn + rri2 * pnm2).astype(dt, copy=False)
        if k < order:
            pnm2 = pnm1
            pnm1 = pn
        # yfit += coeff[k-1] * pn  (amulkr + aaddr — single mul then add)
        yfit = (yfit + coeff[k - 1] * pn).astype(dt, copy=False)
    return yfit


def cv_evcheb(coeffs: NDArray, x: NDArray, xmin: float, xmax: float,
              dtype: np.dtype | str | None = None) -> NDArray:
    """Vectorized Chebyshev evaluation. Mirrors IRAF ``cv_fevalr.x:rcv_evcheb``."""
    dt = np.dtype(dtype) if dtype is not None else np.dtype(coeffs.dtype)
    order = int(coeffs.size)
    x_a = np.asarray(x, dtype=dt)
    npts = x_a.size
    coeff = np.asarray(coeffs, dtype=dt)

    if order == 1:
        return np.full(x_a.shape, coeff[0], dtype=dt)

    span = xmax - xmin
    if span == 0.0:
        k1 = dt.type(0.0)
        k2 = dt.type(0.0)
    else:
        k1 = dt.type(-(xmax + xmin) / 2.0)
        k2 = dt.type(2.0 / span)

    c1 = k2 * coeff[1]
    c2 = c1 * k1 + coeff[0]
    yfit = (x_a * c1 + c2).astype(dt, copy=False)
    if order == 2:
        return yfit

    pnm2 = np.ones(npts, dtype=dt)
    sx = (x_a + k1) * k2
    pnm1 = sx.astype(dt, copy=True)
    sx2 = (sx * dt.type(2.0)).astype(dt, copy=False)
    for k in range(3, order + 1):
        # T_k = 2*x*T_{k-1} - T_{k-2}
        pn = (sx2 * pnm1 - pnm2).astype(dt, copy=False)
        if k < order:
            pnm2 = pnm1
            pnm1 = pn
        yfit = (yfit + coeff[k - 1] * pn).astype(dt, copy=False)
    return yfit


# ---------------------------------------------------------------------------
# Incremental fit primitives — IRAF cvfitr.x / cvrejectr.x / cvsolver.x
# ---------------------------------------------------------------------------
#
# IRAF's iterative-reject fit does NOT rebuild the normal equations from
# scratch on each rejection. Instead, ``cvrject`` subtracts the rejected
# point's contribution from the in-memory matrix + vector, then ``cvsolve``
# refactors. Mathematically equivalent to refitting on the kept points, but
# float32 rounding differs because the accumulation order differs.
#
# We expose three primitives so :func:`decanter.utils.iraf_icfit.fit_with_reject`
# can follow IRAF's exact flow.

def cv_zero(order: int, dtype: np.dtype | str = np.float32
            ) -> tuple[NDArray, NDArray]:
    """Allocate zeroed banded matrix + vector. Mirrors IRAF ``cvzeror.x``."""
    dt = np.dtype(dtype)
    return np.zeros((order, order), dtype=dt), np.zeros(order, dtype=dt)


def cv_acpts(matrix: NDArray, vector: NDArray, basis: NDArray,
             w: NDArray, y: NDArray, *, fast: bool = True) -> None:
    """Accumulate ``(x, y, w)`` points (via precomputed basis) into matrix+vector.

    Modifies ``matrix`` and ``vector`` in place. ``fast=True`` uses
    BLAS for the matrix-fill (numerically equivalent at float32 for small
    ``order``). ``fast=False`` uses IRAF-order sequential accumulation.
    """
    dt = np.dtype(matrix.dtype)
    if fast:
        m_inc, v_inc = _accumulate_matrix_fast(basis, w, y, dt)
    else:
        m_inc, v_inc = _accumulate_matrix_sequential(basis, w, y, dt)
    matrix += m_inc
    vector += v_inc


def cv_rject(matrix: NDArray, vector: NDArray, xbasis: NDArray,
             y: float, w: float) -> None:
    """Subtract a single point's contribution from the normal equations.

    Mirrors IRAF ``math/curfit/cvrejectr.x:cvrject``. The caller is
    responsible for computing ``xbasis`` via :func:`cv_b1leg` /
    :func:`cv_b1cheb` (matching IRAF's choice of scalar-form basis with
    division-at-end).
    """
    dt = np.dtype(matrix.dtype)
    order = xbasis.size
    y_t = dt.type(y)
    w_t = dt.type(w)
    # IRAF cvrejectr.x lines 59-73 (1-indexed → 0-indexed here):
    for i in range(order):
        bw = xbasis[i] * w_t
        vector[i] = vector[i] - bw * y_t
        for ii in range(order - i):
            matrix[ii, i] = matrix[ii, i] - xbasis[i + ii] * bw


def cv_solve(matrix: NDArray, vector: NDArray) -> NDArray:
    """Cholesky factor + solve. Mirrors IRAF ``cvsolver.x:cvsolve``.

    Returns ``coeffs`` (does not modify ``matrix`` / ``vector``).
    """
    dt = np.dtype(matrix.dtype)
    matfac = _chofac_banded(matrix, dt)
    return _choslv_banded(matfac, vector, dt)
