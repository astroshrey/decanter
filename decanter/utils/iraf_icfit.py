"""Iterative-reject 1-D polynomial fits — a NumPy port of IRAF's icfit.

Used by s03 (apscatter), s14 (continuum), and any other stage that
needs ``iraf.icfit``-style fitting: fit a low-order Legendre or
Chebyshev polynomial, iteratively reject points that deviate beyond
``low_reject`` / ``high_reject`` σ from the residuals, refit, repeat
until either no points are rejected or ``niterate`` iterations elapse.

Sign convention matches IRAF:
  - ``high_reject`` controls rejection of points ABOVE the fit
    (residual > +high_reject * σ).
  - ``low_reject`` controls rejection of points BELOW the fit
    (residual < -low_reject * σ).

Either threshold may be ``0`` to disable rejection on that side.

Why the explicit normalization to [-1, 1]: ``numpy.polynomial`` fits use
the standard Legendre/Chebyshev basis which is orthogonal on [-1, 1].
Passing raw column indices fits a non-orthogonal expansion that's badly
conditioned at degree ≥ 3. We rescale here and stash the (x_min, x_max)
pair on the fit result so callers can evaluate later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.polynomial import chebyshev, legendre
from numpy.typing import NDArray

from decanter.utils.cvsolver import (
    cv_acpts as _cv_acpts,
    cv_b1cheb as _cv_b1cheb,
    cv_b1leg as _cv_b1leg,
    cv_bcheb as _cv_bcheb,
    cv_bleg as _cv_bleg,
    cv_eval as _cv_eval,
    cv_evcheb as _cv_evcheb,
    cv_evleg as _cv_evleg,
    cv_fit as _cv_fit,
    cv_rject as _cv_rject,
    cv_solve as _cv_solve,
    cv_zero as _cv_zero,
)

FunctionType = Literal["legendre", "chebyshev"]

# When ``USE_IRAF_CVSOLVER`` is True, ``fit_with_reject`` calls the
# Python port of IRAF's banded-Cholesky LSQ (``decanter.utils.cvsolver``)
# in float32 fast mode. This matches IRAF's algorithmic structure
# (normal equations + banded Cholesky in TY_REAL) bit-for-bit and is
# what closes the s03 apscatter parity gap against WARP's saved
# scatter intermediate. Set False to fall back to ``numpy.polynomial``
# (SVD-based) for diagnostic use.
USE_IRAF_CVSOLVER = True
# IRAF runs all of curfit in TY_REAL = float32. Setting this to
# np.float32 reproduces IRAF's reduction rounding; np.float64 is
# faster and matches IRAF to ~1e-7 relative.
CVSOLVER_DTYPE: np.dtype = np.dtype(np.float32)


@dataclass(frozen=True, slots=True)
class IcFitResult:
    """One iterative-reject fit's result."""

    coefficients: NDArray[np.float64]
    mask: NDArray[np.bool_]  # True for points kept in the final fit
    x_min: float
    x_max: float
    function: FunctionType
    iterations: int


def _normalize(x: NDArray[np.float64], x_min: float, x_max: float) -> NDArray[np.float64]:
    """Map ``[x_min, x_max]`` linearly to ``[-1, 1]``."""
    span = x_max - x_min
    if span == 0.0:
        # Degenerate: all x identical. Map everything to 0.
        return np.zeros_like(x)
    return (2.0 * x - x_min - x_max) / span


def _fit(function: FunctionType, x: NDArray, y: NDArray, deg: int,
         x_min: float, x_max: float) -> NDArray[np.float64]:
    """LSQ fit dispatch.

    When :data:`USE_IRAF_CVSOLVER` is True (default), routes to the
    Python port of IRAF curfit (``decanter.utils.cvsolver``) so the
    coefficients match IRAF byte-for-byte at ``CVSOLVER_DTYPE``
    precision. Otherwise falls back to ``numpy.polynomial`` (SVD-based
    least squares — faster, but coefficients differ from IRAF at the
    1e-3 absolute level after iteration for s03's wide-extrapolation
    rows).
    """
    if USE_IRAF_CVSOLVER:
        order = deg + 1
        coeffs = _cv_fit(
            x, y, function=function, order=order,
            xmin=x_min, xmax=x_max, dtype=CVSOLVER_DTYPE, fast=True,
        )
        return coeffs.astype(np.float64, copy=False)
    # Fallback: numpy LSQ on the pre-normalized x.
    xn = _normalize(np.asarray(x, dtype=np.float64), x_min, x_max)
    if function == "legendre":
        return legendre.legfit(xn, y, deg)
    return chebyshev.chebfit(xn, y, deg)


def _eval_at(function: FunctionType, coeffs: NDArray, x: NDArray,
             x_min: float, x_max: float) -> NDArray[np.float64]:
    """Evaluate a fit at real ``x`` coordinates (not normalized)."""
    if USE_IRAF_CVSOLVER:
        return _cv_eval(
            coeffs, x, function=function, xmin=x_min, xmax=x_max,
            dtype=CVSOLVER_DTYPE,
        ).astype(np.float64, copy=False)
    xn = _normalize(np.asarray(x, dtype=np.float64), x_min, x_max)
    if function == "legendre":
        return legendre.legval(xn, coeffs)
    return chebyshev.chebval(xn, coeffs)


# Kept for backward compatibility with the prior signature (used by
# any callers that already have ``x_n`` normalized).
def _eval(function: FunctionType, x_n: NDArray, coeffs: NDArray) -> NDArray[np.float64]:
    if function == "legendre":
        return legendre.legval(x_n, coeffs)
    return chebyshev.chebval(x_n, coeffs)


def fit_with_reject(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    degree: int,
    low_reject: float,
    high_reject: float,
    niterate: int,
    function: FunctionType = "legendre",
    x_min: float | None = None,
    x_max: float | None = None,
) -> IcFitResult:
    """Iterative-reject Legendre/Chebyshev fit.

    Mirrors IRAF ``xtools$icfit/icrejectr.x`` + ``icdeviantr.x`` exactly:

      1. Initial fit on the full input set (no rejection mask applied).
      2. For each rejection iteration:
         a. Compute residuals using the *current* curve (not a fresh fit).
         b. σ = sqrt(sum(r²) / j) where j = count of currently-kept points
            (RMS — no mean subtraction, IRAF ``icdeviantr.x:68``).
         c. If j < 5 (IRAF ``icdeviantr.x:64``), stop rejecting and keep
            the current curve.
         d. Reject points with residual ≥ ``high_reject·σ`` or
            ≤ ``-low_reject·σ`` (strict-greater / strict-less in IRAF).
         e. If any new rejection, refit on the surviving points
            (``icdeviantr.x:125``). If none, halt iteration
            (``icrejectr.x:46``).

    The previous decanter implementation refit at the top of every loop
    pass and computed σ from the fresh fit's residuals rather than the
    previous fit's. For unweighted Legendre fits with a constant term
    the σ values coincide (orthogonality), but the iteration termination
    semantics differ and the iteration count returned was off by one.

    Args:
        x: independent-variable samples.
        y: dependent-variable samples (same shape as ``x``).
        degree: polynomial degree (IRAF "order - 1"; degree=3 → 4 coeffs).
        low_reject: σ threshold for rejecting points BELOW the fit
            (residual ≤ -low_reject·σ). Zero disables.
        high_reject: σ threshold for rejecting points ABOVE the fit
            (residual ≥ +high_reject·σ). Zero disables.
        niterate: maximum number of reject/refit iterations.
        function: ``"legendre"`` or ``"chebyshev"``.
        x_min, x_max: normalization range. If ``None``, uses
            ``x.min()`` / ``x.max()`` (matches IRAF ``ic_dosetup``,
            ``icdosetupr.x:65-71``: xmin/xmax come from the sample's x
            range, not the full data range).

    Returns:
        :class:`IcFitResult` with the final coefficients, the kept-points
        mask, the normalization range, and the iteration count.
    """
    if USE_IRAF_CVSOLVER:
        return _fit_with_reject_iraf(
            x, y, degree=degree, low_reject=low_reject,
            high_reject=high_reject, niterate=niterate,
            function=function, x_min=x_min, x_max=x_max,
        )

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    if x_arr.ndim != 1:
        raise ValueError("x and y must be 1-D")

    x_min_eff = float(x_arr.min() if x_min is None else x_min)
    x_max_eff = float(x_arr.max() if x_max is None else x_max)

    n = y_arr.size
    mask = np.ones(n, dtype=bool)

    # IRAF ``icfitr.x`` always does an initial unweighted fit on the full
    # sample BEFORE entering ic_rejectr (see icfitr.x near rcvfit call).
    if n < degree + 1:
        # No degrees of freedom — leave coefficients zero.
        return IcFitResult(
            coefficients=np.zeros(degree + 1, dtype=np.float64),
            mask=mask,
            x_min=x_min_eff,
            x_max=x_max_eff,
            function=function,
            iterations=0,
        )
    coeffs = _fit(function, x_arr, y_arr, degree, x_min_eff, x_max_eff)

    iters_run = 0
    for _ in range(max(0, int(niterate))):
        residuals = y_arr - _eval_at(function, coeffs, x_arr, x_min_eff, x_max_eff)
        j = int(mask.sum())
        # IRAF ic_deviantr returns immediately (skipping rejection) when
        # j < 5; the current curve is preserved and ic_rejectr's
        # ``if (newreject == 0) break`` halts iteration on the next pass.
        if j < 5:
            iters_run += 1
            break
        sigma = float(np.sqrt(np.sum(residuals[mask] ** 2) / j))
        if sigma == 0.0:
            iters_run += 1
            break
        new_mask = mask.copy()
        if high_reject > 0:
            new_mask &= residuals < high_reject * sigma
        if low_reject > 0:
            new_mask &= residuals > -low_reject * sigma
        newreject = int((mask & ~new_mask).sum())
        iters_run += 1
        if newreject == 0:
            break
        mask = new_mask
        # Refit on the surviving points (IRAF icdeviantr.x:125).
        if int(mask.sum()) < degree + 1:
            break
        coeffs = _fit(function, x_arr[mask], y_arr[mask], degree, x_min_eff, x_max_eff)

    return IcFitResult(
        coefficients=coeffs,
        mask=mask,
        x_min=x_min_eff,
        x_max=x_max_eff,
        function=function,
        iterations=iters_run,
    )


def _fit_with_reject_iraf(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    degree: int,
    low_reject: float,
    high_reject: float,
    niterate: int,
    function: FunctionType = "legendre",
    x_min: float | None = None,
    x_max: float | None = None,
) -> IcFitResult:
    """IRAF-faithful iterative-reject Legendre/Chebyshev fit.

    Mirrors IRAF ``icfitr.x`` → ``cvfit`` (initial accumulate+solve) +
    ``ic_rejectr`` → ``ic_deviantr`` (compute residuals from current curve,
    compute σ in float32, subtract rejected points from the normal-equations
    matrix via :func:`cv_rject`, refactor via :func:`cv_solve`).

    Distinguishing features from the float64-numpy fallback path:

    - Normal-equations matrix is NOT rebuilt from scratch after rejection.
      Rejected points are subtracted from the matrix (:func:`cv_rject`) and
      Cholesky-refactored. This is what IRAF does, and matches its float32
      rounding bit-for-bit.
    - Residuals are computed via :func:`cv_evleg` / :func:`cv_evcheb`
      (IRAF ``rcv_evleg`` / ``rcv_evcheb`` — recursive evaluation in the
      same accumulation order as ``cvvector``).
    - σ = sqrt(Σ r²/ j) is computed in float32 with sequential left-to-right
      summation, matching ``icdeviantr.x:56-68``.
    - Single-point reject basis uses :func:`cv_b1leg` / :func:`cv_b1cheb`
      (IRAF's scalar-form recurrence with division-at-end; differs from the
      vector-form ``cv_bleg`` / ``cv_bcheb`` by 1 ULP — that's IRAF's
      actual behavior in ``cvrject``).
    """
    dt = np.dtype(CVSOLVER_DTYPE)
    x_arr = np.asarray(x, dtype=dt)
    y_arr = np.asarray(y, dtype=dt)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    if x_arr.ndim != 1:
        raise ValueError("x and y must be 1-D")

    x_min_eff = float(x_arr.min() if x_min is None else x_min)
    x_max_eff = float(x_arr.max() if x_max is None else x_max)

    n = y_arr.size
    mask = np.ones(n, dtype=bool)
    order = degree + 1

    if n < order:
        return IcFitResult(
            coefficients=np.zeros(order, dtype=np.float64),
            mask=mask,
            x_min=x_min_eff,
            x_max=x_max_eff,
            function=function,
            iterations=0,
        )

    # --- Initial fit (cvfit = cvzero + cvacpts + cvsolve) -------------------
    if function == "legendre":
        basis = _cv_bleg(x_arr, order, x_min_eff, x_max_eff, dtype=dt)
        eval_vec = _cv_evleg
        b1 = _cv_b1leg
    elif function == "chebyshev":
        basis = _cv_bcheb(x_arr, order, x_min_eff, x_max_eff, dtype=dt)
        eval_vec = _cv_evcheb
        b1 = _cv_b1cheb
    else:
        raise ValueError(f"unknown function {function!r}")

    matrix, vector = _cv_zero(order, dtype=dt)
    w_arr = np.ones(n, dtype=dt)
    _cv_acpts(matrix, vector, basis, w_arr, y_arr, fast=True)
    coeffs = _cv_solve(matrix, vector)

    iters_run = 0
    high_t = dt.type(high_reject) if high_reject > 0 else None
    low_t = dt.type(low_reject) if low_reject > 0 else None
    for _ in range(max(0, int(niterate))):
        # Residuals via IRAF cvvector (rcv_evleg/rcv_evcheb) - float32
        # accumulation order matches IRAF byte-for-byte.
        yfit = eval_vec(coeffs, x_arr, x_min_eff, x_max_eff, dtype=dt)
        residuals = (y_arr - yfit).astype(dt, copy=False)

        # IRAF icdeviantr.x:52-68 — count kept, accumulate r² left-to-right.
        j = int(mask.sum())
        if j < 5:
            iters_run += 1
            break
        # Sequential float32 sum of r² over kept points (IRAF: do i = 1, npts).
        sigma_sq = dt.type(0.0)
        kept_idx = np.flatnonzero(mask)
        for idx in kept_idx:
            r = residuals[idx]
            sigma_sq = sigma_sq + r * r
        if sigma_sq <= 0.0:
            iters_run += 1
            break
        # IRAF icdeviantr.x:68: sigma = sqrt(sigma / j). Both sigma and j
        # are TY_REAL (= fp32) at the divide AND sqrt — keep all ops in
        # fp32 so the rejection iteration converges to the same byte-
        # for-byte rounded values IRAF gets. The previous fp64 cast
        # at the divide produced 1-ULP-different sigma on most rows,
        # which propagates to differently-rejected points and the
        # documented float32 fit-iteration noise floor.
        j_fp = dt.type(j)
        sigma = dt.type(np.sqrt(np.divide(sigma_sq, j_fp, dtype=dt)))
        if sigma == 0.0:
            iters_run += 1
            break

        high_cut = (high_t * sigma) if high_t is not None else dt.type(np.finfo(dt).max)
        low_cut = (-low_t * sigma) if low_t is not None else dt.type(-np.finfo(dt).max)

        # IRAF icdeviantr.x:96-117 — scan i=0..n-1 left-to-right; for each
        # point still kept and outside cuts, call cv_rject to subtract its
        # contribution from the normal equations.
        newreject = 0
        for i in range(n):
            if not mask[i]:
                continue
            r = residuals[i]
            if r >= high_cut or r <= low_cut:
                xbasis = b1(float(x_arr[i]), order, x_min_eff, x_max_eff,
                            dtype=dt)
                _cv_rject(matrix, vector, xbasis, float(y_arr[i]), 1.0)
                mask[i] = False
                newreject += 1

        iters_run += 1
        if newreject == 0:
            break
        if int(mask.sum()) < order:
            break
        # Re-solve (Cholesky factor + back-sub) on the updated matrix.
        coeffs = _cv_solve(matrix, vector)

    return IcFitResult(
        coefficients=coeffs.astype(np.float64, copy=False),
        mask=mask,
        x_min=x_min_eff,
        x_max=x_max_eff,
        function=function,
        iterations=iters_run,
    )


def evaluate(result: IcFitResult, x: NDArray[np.floating]) -> NDArray[np.float64]:
    """Evaluate a fit result on a new ``x`` grid (any shape)."""
    return _eval_at(result.function, result.coefficients, np.asarray(x, dtype=np.float64),
                    result.x_min, result.x_max)
