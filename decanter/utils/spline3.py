"""IRAF ``interptype=spline3`` 1-D cubic B-spline interpolation.

Bit-for-bit port of IRAF ``math/iminterp/ii_spline.x`` (natural-BC
B-spline coefficient solve) and the y-axis half of ``ii_bispline3``
(coefficient evaluation) from ``math/iminterp/ii_bieval.x``.

The IRAF B-spline:
  - Knots at integer 1-indexed positions ``1..n``, uniform unit spacing.
  - Coefficients ``bcoeff[1..n+2]`` solve the tridiagonal system that
    interpolates the data with NATURAL boundary conditions (second
    derivative = 0 at both endpoints).
  - Storage in IRAF is pre-divided by 6, so the evaluation kernel
    ``bx[1] = tx**3, bx[2] = 1 + tx*(3+tx*(3-3*tx)), ...`` omits the
    ``1/6`` factor.

We only need the y-axis resample: in s06's `rectify_order` the x-axis
mapping is identity (one output column per input column), so the 2-D
``ii_bispline3`` collapses to per-column 1-D evaluation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def iraf_spline3_coeffs(data: NDArray[np.floating]) -> NDArray[np.float64]:
    """Compute IRAF natural cubic B-spline coefficients for a 1-D dataset.

    Mirrors ``math/iminterp/ii_spline.x``. Given ``n`` data points
    ``y[1..n]``, returns ``n+2`` B-spline coefficients ``bcoeff[1..n+2]``
    satisfying interpolation at the integer knots with second derivative
    zero at both endpoints.

    Args:
        data: 1-D array of length ``n``, the data values at integer
            positions ``1..n``.

    Returns:
        1-D float array of length ``n + 2``.
    """
    y = np.asarray(data, dtype=np.float64).ravel()
    n = y.size
    if n < 2:
        raise ValueError("spline3 requires at least 2 data points")

    # Storage layout per ii_spline.x: bcoeff[1] = 2nd deriv at x=1 (0 for
    # natural BC), bcoeff[2..n+1] = y[1..n], bcoeff[n+2] = 2nd deriv at
    # x=n (0).
    bc = np.empty(n + 2, dtype=np.float64)
    bc[0] = 0.0
    bc[1 : n + 1] = y
    bc[n + 1] = 0.0
    diag = np.empty(n + 1, dtype=np.float64)

    # Forward elimination. Mapping IRAF 1-indexed → numpy 0-indexed:
    # my bc[k] = IRAF bcoeff[k+1], my diag[k] = IRAF diag[k+1].
    diag[0] = -2.0                   # IRAF diag[1] = -2
    bc[0] = bc[0] / 6.0              # IRAF bcoeff[1] = bcoeff[1]/6
    diag[1] = 0.0                    # IRAF diag[2] = 0
    bc[1] = (bc[1] - bc[0]) / 6.0    # IRAF bcoeff[2] = (bcoeff[2] - bcoeff[1])/6
    for i in range(2, n + 1):        # IRAF: do i_iraf = 3, npts+1
        # i here == IRAF i_iraf - 1.
        diag[i] = 1.0 / (4.0 - diag[i - 1])
        bc[i] = diag[i] * (bc[i] - bc[i - 1])

    # Special closing step for bcoeff[npts+2] (= my bc[n+1]).
    bc[n + 1] = (
        (diag[n - 1] + 2.0) * bc[n] - bc[n - 1] + bc[n + 1] / 6.0
    ) / (1.0 + diag[n] * (diag[n - 1] + 2.0))

    # Back substitution.  IRAF: do i_iraf = npts+1, 3, -1:
    #     bcoeff[i_iraf] = bcoeff[i_iraf] - diag[i_iraf] * bcoeff[i_iraf+1]
    # i_iraf - 1 maps to my index, and bcoeff[i_iraf+1] is my bc[i_iraf].
    for i_iraf in range(n + 1, 2, -1):
        bc[i_iraf - 1] = bc[i_iraf - 1] - diag[i_iraf - 1] * bc[i_iraf]
    # Recover bcoeff[1].
    bc[0] = bc[0] + 2.0 * bc[1] - bc[2]
    return bc


def iraf_spline3_coeffs_batch(data: NDArray[np.floating]) -> NDArray[np.float64]:
    """Vectorized ``iraf_spline3_coeffs`` over the leading axis.

    Args:
        data: shape ``(n, m)`` — ``m`` independent 1-D datasets each of
            length ``n``.

    Returns:
        shape ``(n + 2, m)`` B-spline coefficient array.
    """
    y = np.asarray(data, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("expected a 2-D array")
    n, m = y.shape
    if n < 2:
        raise ValueError("spline3 requires at least 2 data points")
    bc = np.empty((n + 2, m), dtype=np.float64)
    bc[0, :] = 0.0
    bc[1 : n + 1, :] = y
    bc[n + 1, :] = 0.0
    diag = np.empty(n + 1, dtype=np.float64)

    diag[0] = -2.0
    bc[0, :] = bc[0, :] / 6.0
    diag[1] = 0.0
    bc[1, :] = (bc[1, :] - bc[0, :]) / 6.0
    for i in range(2, n + 1):
        diag[i] = 1.0 / (4.0 - diag[i - 1])
        bc[i, :] = diag[i] * (bc[i, :] - bc[i - 1, :])

    bc[n + 1, :] = (
        (diag[n - 1] + 2.0) * bc[n, :] - bc[n - 1, :] + bc[n + 1, :] / 6.0
    ) / (1.0 + diag[n] * (diag[n - 1] + 2.0))

    for i_iraf in range(n + 1, 2, -1):
        bc[i_iraf - 1, :] = bc[i_iraf - 1, :] - diag[i_iraf - 1] * bc[i_iraf, :]
    bc[0, :] = bc[0, :] + 2.0 * bc[1, :] - bc[2, :]
    return bc


def iraf_spline3_eval(
    bcoeff: NDArray[np.floating], y_1idx: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Evaluate the IRAF cubic B-spline at fractional 1-indexed positions.

    Mirrors the y-half of ``ii_bispline3`` (``ii_bieval.x:282-314``):

        ny = floor(y)
        sy = y - ny ; ty = 1 - sy
        by = [ty**3, 1 + ty*(3+ty*(3-3*ty)), 1 + sy*(3+sy*(3-3*sy)), sy**3]
        value = by[0]*bc[ny-1] + by[1]*bc[ny] + by[2]*bc[ny+1] + by[3]*bc[ny+2]

    where ``bcoeff`` is 1-indexed conceptually but stored 0-indexed in
    numpy; ``bcoeff[k]`` in IRAF == ``bcoeff[k-1]`` here.

    Out-of-range ``y`` values are clamped to ``[1, n]`` first (matching
    IRAF transform's continuous extrapolation, which is what WARP's
    ``_trans.fits`` exhibits at the order's wavelength edges).

    Args:
        bcoeff: shape ``(n + 2,)`` coefficients from
            :func:`iraf_spline3_coeffs`.
        y_1idx: 1-indexed fractional positions in ``[1, n]``.

    Returns:
        Interpolated values, same shape as ``y_1idx``.
    """
    bc_in = np.asarray(bcoeff, dtype=np.float64)
    if bc_in.ndim != 1:
        raise ValueError("bcoeff must be 1-D")
    n = bc_in.size - 2
    # IRAF allocates an extra zero past bcoeff[n+2] (msifit calls calloc
    # with MSI_NXCOEFF * MSI_NYCOEFF entries and only writes the n+2
    # interior coefficients; the trailing slot stays at 0). At x == n
    # exactly the 4th basis term bx[4] is 0 but the index bc[n+2] is
    # still read, so we pad once here.
    bc = np.empty(bc_in.size + 1, dtype=np.float64)
    bc[: bc_in.size] = bc_in
    bc[-1] = 0.0
    y = np.asarray(y_1idx, dtype=np.float64)
    # Clamp into the valid range [1, n]. IRAF documents "1 <= y <= nypix"
    # and out-of-range queries hit the calloc'd padding (=0), which is
    # what gives the boundary-clamping continuous-extrapolation behavior
    # WARP's transform produces at the order edges.
    y_clamped = np.clip(y, 1.0, float(n))
    ny = np.floor(y_clamped).astype(np.int64)
    sy = y_clamped - ny
    ty = 1.0 - sy
    by1 = ty ** 3
    by2 = 1.0 + ty * (3.0 + ty * (3.0 - 3.0 * ty))
    by3 = 1.0 + sy * (3.0 + sy * (3.0 - 3.0 * sy))
    by4 = sy ** 3
    # IRAF's B-spline storage is bcoeff[k] = c_{k-1} (the coefficient
    # associated with virtual knot k-1, where k runs 1..n+2). For x in
    # [k, k+1] the 4 active basis functions are c_{k-1}, c_k, c_{k+1},
    # c_{k+2} → IRAF bcoeff[k], bcoeff[k+1], bcoeff[k+2], bcoeff[k+3].
    # In numpy 0-indexing: bc[k-1], bc[k], bc[k+1], bc[k+2].
    i0 = ny - 1
    return (
        by1 * bc[i0]
        + by2 * bc[i0 + 1]
        + by3 * bc[i0 + 2]
        + by4 * bc[i0 + 3]
    )


def iraf_spline3_eval_batch(
    bcoeff: NDArray[np.floating], y_1idx: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Batched per-column ``iraf_spline3_eval``.

    Args:
        bcoeff: shape ``(n + 2, m)`` — per-column coefficients.
        y_1idx: shape ``(k, m)`` — 1-indexed fractional positions per
            column. Out-of-range values are clamped to ``[1, n]``.

    Returns:
        Shape ``(k, m)`` interpolated values.
    """
    bc_in = np.asarray(bcoeff, dtype=np.float64)
    if bc_in.ndim != 2:
        raise ValueError("bcoeff must be 2-D (n+2, m)")
    n_plus_2, m = bc_in.shape
    n = n_plus_2 - 2
    y = np.asarray(y_1idx, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != m:
        raise ValueError("y_1idx must be 2-D with second dim matching bcoeff")
    bc = np.empty((bc_in.shape[0] + 1, m), dtype=np.float64)
    bc[: bc_in.shape[0], :] = bc_in
    bc[-1, :] = 0.0
    y_clamped = np.clip(y, 1.0, float(n))
    ny = np.floor(y_clamped).astype(np.int64)
    sy = y_clamped - ny
    ty = 1.0 - sy
    by1 = ty ** 3
    by2 = 1.0 + ty * (3.0 + ty * (3.0 - 3.0 * ty))
    by3 = 1.0 + sy * (3.0 + sy * (3.0 - 3.0 * sy))
    by4 = sy ** 3
    i0 = ny - 1  # numpy index for IRAF bcoeff[ny]
    col_idx = np.broadcast_to(np.arange(m), i0.shape)
    return (
        by1 * bc[i0, col_idx]
        + by2 * bc[i0 + 1, col_idx]
        + by3 * bc[i0 + 2, col_idx]
        + by4 * bc[i0 + 3, col_idx]
    )


def iraf_spline3_resample_columns(
    data: NDArray[np.floating], y_1idx: NDArray[np.floating]
) -> NDArray[np.float32]:
    """Column-wise IRAF spline3 resample.

    For each column ``j``, fit a natural cubic B-spline through
    ``data[:, j]`` (at integer 1-indexed positions 1..H) and evaluate it
    at the fractional positions ``y_1idx[:, j]``. The x-axis (column
    index) mapping is identity, so this is exactly the y-half of
    ``ii_bispline3`` applied per column.

    Args:
        data: shape ``(H, n_x)`` input data (the cut strip).
        y_1idx: shape ``(n_out, n_x)`` per-column 1-indexed fractional
            sample positions. Values outside ``[1, H]`` are clamped to
            the boundary (matches IRAF transform's continuous extrapolation
            of the spline3 kernel and reproduces WARP's
            ``mode='nearest'``-equivalent behavior at the wavelength
            edges).

    Returns:
        Shape ``(n_out, n_x)`` float32 resampled data.
    """
    cut = np.asarray(data, dtype=np.float64)
    if cut.ndim != 2:
        raise ValueError("data must be 2-D (H, n_x)")
    y = np.asarray(y_1idx, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != cut.shape[1]:
        raise ValueError(
            f"y_1idx shape {y.shape} incompatible with data shape {cut.shape}"
        )
    bcoeff = iraf_spline3_coeffs_batch(cut)
    resampled = iraf_spline3_eval_batch(bcoeff, y)
    return resampled.astype(np.float32, copy=False)
