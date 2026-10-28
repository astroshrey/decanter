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


# ===========================================================================
# Bit-exact float32 2-D kernels of IRAF ``math/iminterp`` (msifit/msivector)
# ===========================================================================
#
# The functions below reproduce, operation for operation, the REAL
# arithmetic of ``msifit.x`` (II_BILINEAR, II_BISPLINE3), ``ii_spline2d.x``
# and ``ii_bieval.x:ii_bilinear`` / ``ii_bispline3`` as compiled into this
# Mac's ``x_longslit.e`` (iraf-community 2.18.1, clang -O2, arm64).  The
# fused multiply-adds below are the ones the object code contains (checked
# with ``objdump -d``) and were verified against the compiled kernels via
# ``scripts/iraf_oracle`` on random inputs (0 mismatches).

_F32 = np.float32
_F64 = np.float64
_TIE_MASK = np.int64(0x1FFFFFFF)
_TIE_HALF = np.int64(0x10000000)


def _fma32d(a: NDArray, b: NDArray, c: NDArray) -> NDArray[np.float32]:
    """``round32(a*b + c)`` for float64 arrays holding float32 values.

    ``a*b`` is exact in float64, so ``s = fl64(a*b + c)`` rounds once; the
    float32 rounding of ``s`` equals the single-rounded fma unless ``s``
    lands exactly on a float32 midpoint, which is detected from its bits and
    resolved with the exact error term (two-sum).
    """
    p = a * b
    s = p + c
    r = s.astype(_F32)
    tie = (s.view(np.int64) & _TIE_MASK) == _TIE_HALF
    if tie.any():
        idx = np.nonzero(tie)
        sp, pp = s[idx], p[idx]
        cc = np.broadcast_to(c, s.shape)[idx]
        bb = sp - pp
        e = (pp - (sp - bb)) + (cc - bb)
        up = np.nextafter(sp, np.inf).astype(_F32)
        dn = np.nextafter(sp, -np.inf).astype(_F32)
        rr = r[idx]
        rr = np.where(e > 0, up, np.where(e < 0, dn, rr))
        r[idx] = rr
    return r


def fma32_exact(a, b, c) -> NDArray[np.float32]:
    """Float32 fused multiply-add (vector), see :func:`_fma32d`."""
    return _fma32d(np.asarray(a, _F32).astype(_F64), np.asarray(b, _F32).astype(_F64),
                   np.asarray(c, _F32).astype(_F64))


def bilinear_fit_exact(grid: NDArray[np.float32]) -> NDArray[np.float32]:
    """``msifit(II_BILINEAR)``: data plus one extrapolated column and row.

    IRAF: ``math/iminterp/msifit.x`` (II_BILINEAR branch): column
    ``nxpix+1 = 2*c[nxpix] - c[nxpix-1]`` per row, then the last row
    ``2*row[nypix] - row[nypix-1]`` via ``awsur`` (2*a is exact, so fused
    and unfused forms agree).
    """
    g = np.asarray(grid, _F32)
    ny, nx = g.shape
    c = np.zeros((ny + 1, nx + 1), _F32)
    c[:ny, :nx] = g
    c[:ny, nx] = _F32(2.0) * c[:ny, nx - 1] - c[:ny, nx - 2]
    c[ny, :] = _F32(2.0) * c[ny - 1, :] - c[ny - 2, :]
    return c


def bilinear_eval_exact(coeff: NDArray[np.float32], x: NDArray[np.float32],
                        y: NDArray[np.float32]) -> NDArray[np.float32]:
    """``ii_bieval.x:ii_bilinear`` with ``first_point = 0``.

    Object code: ``fma(sx*sy, c11, fma(tx*sy, c10, fma(tx*ty, c00,
    (sx*ty)*c01)))``, every product rounded to REAL.
    """
    x = np.asarray(x, _F32)
    y = np.asarray(y, _F32)
    nx = x.astype(np.int64)          # fcvtzs (truncation)
    ny = y.astype(np.int64)
    sx = x - nx.astype(_F32)
    tx = _F32(1.0) - sx
    sy = y - ny.astype(_F32)
    ty = _F32(1.0) - sy
    ncol = coeff.shape[1]
    flat = coeff.ravel()
    i00 = (ny - 1) * ncol + (nx - 1)
    c00 = flat[i00]
    c01 = flat[i00 + 1]
    c10 = flat[i00 + ncol]
    c11 = flat[i00 + ncol + 1]
    t = ((sx * ty) * c01).astype(_F64)
    t = _fma32d((tx * ty).astype(_F64), c00.astype(_F64), t)
    t = _fma32d((tx * sy).astype(_F64), c10.astype(_F64), t.astype(_F64))
    return _fma32d((sx * sy).astype(_F64), c11.astype(_F64), t.astype(_F64))


def _spline2d_diag(n: int) -> NDArray[np.float32]:
    d = np.empty(n + 1, _F32)
    d[0] = _F32(-2.0)
    d[1] = _F32(0.0)
    for i in range(3, n + 2):
        d[i - 1] = _F32(1.0) / (_F32(4.0) - d[i - 2])
    return d


def spline2d_exact(data: NDArray[np.float32]) -> NDArray[np.float32]:
    """``ii_spline2d.x`` along the LAST axis, vectorised over leading axes.

    ``data[..., 0]`` and ``data[..., n+1]`` hold the end second
    derivatives (0 for msifit), ``data[..., 1:n+1]`` the ``n`` data points.
    Returns the ``n + 2`` B-spline coefficients per vector (REAL).
    """
    c = np.array(data, dtype=_F32, copy=True)
    n = c.shape[-1] - 2
    d = _spline2d_diag(n)
    six = _F32(6.0)
    c[..., 0] = c[..., 0] / six
    c[..., 1] = (c[..., 1] - c[..., 0]) / six
    for i in range(3, n + 2):                       # 1-based i
        c[..., i - 1] = d[i - 1] * (c[..., i - 1] - c[..., i - 2])
    t = _F32(d[n - 1] + _F32(2.0))
    num = _fma32d(_F64(t), c[..., n].astype(_F64), -c[..., n - 1].astype(_F64))
    num = num + c[..., n + 1] / six
    den = _F32(_fma32d(_F64(d[n]), _F64(t), _F64(1.0)))
    c[..., n + 1] = num / den
    for i in range(n + 1, 2, -1):                   # back substitution, fused
        c[..., i - 1] = _fma32d(-_F64(d[i - 1]), c[..., i].astype(_F64),
                                c[..., i - 1].astype(_F64))
    c[..., 0] = (c[..., 0] + _F32(2.0) * c[..., 1]) - c[..., 2]
    return c


def spline2d_rows_exact(rows: NDArray[np.float32]) -> NDArray[np.float32]:
    """x pass of ``msifit(II_BISPLINE3)`` for image rows ``(nrow, nx)``.

    Returns ``(nrow, nx + 2)`` x-direction B-spline coefficients (natural
    end conditions: the end second derivatives are the calloc'd zeros).
    """
    r = np.asarray(rows, _F32)
    nrow, nx = r.shape
    v = np.zeros((nrow, nx + 2), _F32)
    v[:, 1:nx + 1] = r
    return spline2d_exact(v)


def spline2d_cols_exact(xcoef: NDArray[np.float32]) -> NDArray[np.float32]:
    """y pass of ``msifit(II_BISPLINE3)`` for a window of x-coefficient rows.

    ``xcoef`` has shape ``(..., nypix, nx + 2)`` (the x-pass output of the
    window's image lines).  Returns the final coefficient array of shape
    ``(..., nypix + 3, nx + 3)`` laid out as IRAF's ``COEFF`` (row = y,
    column = x, ``MSI_FSTPNT = NXCOEFF + 1``); the trailing row/column are
    the calloc'd zeros.
    """
    xc = np.asarray(xcoef, _F32)
    *lead, nyp, nx2 = xc.shape
    v = np.zeros((*lead, nx2 + 1, nyp + 2), _F32)   # vectors = columns
    v[..., :nx2, 1:nyp + 1] = np.swapaxes(xc, -1, -2)
    cy = spline2d_exact(v)                           # (..., nx+3, nyp+2)
    out = np.zeros((*lead, nyp + 3, nx2 + 1), _F32)
    out[..., :nyp + 2, :] = np.swapaxes(cy, -1, -2)
    return out


def msifit_bispline3_exact(data: NDArray[np.float32]) -> NDArray[np.float32]:
    """``msifit(II_BISPLINE3)`` of a ``(nypix, nxpix)`` REAL array."""
    return spline2d_cols_exact(spline2d_rows_exact(data))


def _bspline_weights(s: NDArray[np.float32], t: NDArray[np.float32]):
    """ii_bispline3 basis: tx**3, 1+tx*(3+tx*(3-3tx)) (fused), ..."""
    s64 = s.astype(_F64)
    t64 = t.astype(_F64)
    three = _F64(3.0)
    one = _F64(1.0)
    b1 = t * (t * t)
    w = _fma32d(-t64, three, three)
    w = _fma32d(t64, w.astype(_F64), three)
    b2 = _fma32d(t64, w.astype(_F64), one)
    w = _fma32d(-s64, three, three)
    w = _fma32d(s64, w.astype(_F64), three)
    b3 = _fma32d(s64, w.astype(_F64), one)
    b4 = s * (s * s)
    return b1, b2, b3, b4


def bispline3_eval_exact(coeff_flat: NDArray[np.float32], row_base: NDArray[np.int64],
                         ncol: int, x: NDArray[np.float32],
                         y: NDArray[np.float32]) -> NDArray[np.float32]:
    """``ii_bieval.x:ii_bispline3`` on (possibly stacked) coefficient arrays.

    ``coeff_flat`` is a flat REAL array holding one or more ``COEFF``
    arrays of row length ``ncol``; ``row_base`` gives, per point, the flat
    offset of row 0 of the point's array.  ``x``/``y`` are in that array's
    interpolation coordinates (``1 <= x <= nxpix``, ``1 <= y <= nypix``).

    Object code: per row ``sum = fma(c[+2], bx4, fma(c[+1], bx3,
    fma(c[-1], bx1, bx2*c[0])))`` and ``accum = fma(sum_j, by_j, accum)``
    starting from ``accum = 0``.
    """
    x = np.asarray(x, _F32)
    y = np.asarray(y, _F32)
    nx = x.astype(np.int64)
    ny = y.astype(np.int64)
    sx = x - nx.astype(_F32)
    tx = _F32(1.0) - sx
    sy = y - ny.astype(_F32)
    ty = _F32(1.0) - sy
    bx1, bx2, bx3, bx4 = _bspline_weights(sx, tx)
    by = _bspline_weights(sy, ty)
    bx1d, bx3d, bx4d = bx1.astype(_F64), bx3.astype(_F64), bx4.astype(_F64)
    # COEFF row (ny-1), column nx  <->  IRAF coeff[first_row]
    base = row_base + (ny - 1) * ncol + nx
    acc = np.zeros(x.shape, _F64)
    for j in range(4):
        idx = base + j * ncol
        c0 = coeff_flat[idx]
        s = (bx2 * c0).astype(_F64)
        s = _fma32d(coeff_flat[idx - 1].astype(_F64), bx1d, s)
        s = _fma32d(coeff_flat[idx + 1].astype(_F64), bx3d, s.astype(_F64))
        s = _fma32d(coeff_flat[idx + 2].astype(_F64), bx4d, s.astype(_F64))
        acc = _fma32d(s.astype(_F64), by[j].astype(_F64), acc).astype(_F64)
    return acc.astype(_F32)


# ---------------------------------------------------------------------------
# Faster variants of the exact kernels (same arithmetic, fewer passes)
# ---------------------------------------------------------------------------

def _round_sum_into(out32: NDArray[np.float32], s: NDArray[np.float64],
                    p: NDArray[np.float64], c) -> None:
    """``out32 = round32(p + c)`` given ``s = fl64(p + c)`` (``p`` exact).

    Only a float32-midpoint ``s`` can be double-rounded; those elements are
    corrected with the exact two-sum error.
    """
    np.copyto(out32, s, casting="unsafe")
    tie = (s.view(np.int64) & _TIE_MASK) == _TIE_HALF
    if tie.any():
        idx = np.nonzero(tie)
        sp, pp = s[idx], p[idx]
        cc = np.broadcast_to(np.asarray(c), s.shape)[idx].astype(_F64)
        bb = sp - pp
        e = (pp - (sp - bb)) + (cc - bb)
        rr = out32[idx]
        up = np.nextafter(sp, np.inf).astype(_F32)
        dn = np.nextafter(sp, -np.inf).astype(_F32)
        out32[idx] = np.where(e > 0, up, np.where(e < 0, dn, rr))


def spline2d_axis0(v: NDArray[np.float32]) -> NDArray[np.float32]:
    """:func:`spline2d_exact` along axis 0 (contiguous rows), in place.

    ``v`` has shape ``(n + 2, m)``: ``m`` independent vectors laid out as
    columns.  Identical arithmetic to :func:`spline2d_exact`.
    """
    c = v
    n = c.shape[0] - 2
    d = _spline2d_diag(n)
    six = _F32(6.0)
    np.divide(c[0], six, out=c[0])
    np.subtract(c[1], c[0], out=c[1])
    np.divide(c[1], six, out=c[1])
    for i in range(3, n + 2):
        np.subtract(c[i - 1], c[i - 2], out=c[i - 1])
        np.multiply(c[i - 1], d[i - 1], out=c[i - 1])
    m = c.shape[1:]
    p = np.empty(m, _F64)
    s = np.empty(m, _F64)
    t = _F32(d[n - 1] + _F32(2.0))
    # num = fnmsub(t, c[n], c[n-1]) = t*c[n] - c[n-1] (fused)
    np.multiply(c[n], _F64(t), out=p)
    np.subtract(p, c[n - 1], out=s)
    num = np.empty(m, _F32)
    _round_sum_into(num, s, p, -c[n - 1])
    num = num + c[n + 1] / six
    den = _F32(_fma32d(_F64(d[n]), _F64(t), _F64(1.0)))
    np.divide(num, den, out=c[n + 1])
    for i in range(n + 1, 2, -1):
        # c[i-1] = c[i-1] - d[i-1]*c[i]  (fmsub, fused)
        np.multiply(c[i], -_F64(d[i - 1]), out=p)
        np.add(p, c[i - 1], out=s)
        _round_sum_into(c[i - 1], s, p, c[i - 1].copy())
    np.add(c[0], _F32(2.0) * c[1], out=c[0])
    np.subtract(c[0], c[2], out=c[0])
    return c


def _w_exact(t64: NDArray[np.float64], tmp: NDArray[np.float64],
             out: NDArray[np.float64]) -> None:
    """``1 + t*(3 + t*(3 - 3*t))`` with the fused rounding of ii_bispline3.

    For ``t`` a float32 in [0, 1] with ``1 <= x <= 2**11`` every fused step
    is exact in float64 (``t`` is a multiple of 2**-23, the partial
    results have <= 49 significant bits), so one float32 rounding per step
    reproduces the fma without a midpoint check.
    """
    np.multiply(t64, -3.0, out=tmp)
    np.add(tmp, 3.0, out=tmp)
    np.copyto(out, tmp.astype(_F32))            # w1 = round32(3 - 3t)
    np.multiply(t64, out, out=tmp)
    np.add(tmp, 3.0, out=tmp)
    np.copyto(out, tmp.astype(_F32))            # w2 = round32(3 + t*w1)
    np.multiply(t64, out, out=tmp)
    np.add(tmp, 1.0, out=tmp)
    np.copyto(out, tmp.astype(_F32))            # b = round32(1 + t*w2)


def bispline3_eval_many(flats64: list[NDArray[np.float64]], base: NDArray[np.int64],
                        ncol: int, x: NDArray[np.float32], y: NDArray[np.float32],
                        outs: list[NDArray[np.float32]], flats32=None) -> list:
    """:func:`bispline3_eval_exact` for several coefficient sets at once.

    All sets share the geometry (``base``, ``x``, ``y``; e.g. the object and
    its sky); the B-spline weights are computed once.  ``flats64`` are the
    float64 copies of the REAL coefficient arrays.  Results go to ``outs``
    (float32, same shape as ``x``).  Returns, per set, the flat indices of
    pixels where a fused step landed on a float32 midpoint; the caller must
    recompute those with :func:`bispline3_eval_exact`.
    """
    shape = x.shape
    x = x.ravel()
    y = y.ravel()
    base = base.ravel()
    npx = x.size
    nx = x.astype(np.int64)
    ny = y.astype(np.int64)
    sx = x - nx.astype(_F32)
    tx = _F32(1.0) - sx
    sy = y - ny.astype(_F32)
    ty = _F32(1.0) - sy
    tmp = np.empty(npx, _F64)
    bx = np.empty((4, npx), _F64)
    by = np.empty((4, npx), _F64)
    np.copyto(bx[0], tx * (tx * tx))
    np.copyto(bx[3], sx * (sx * sx))
    np.copyto(by[0], ty * (ty * ty))
    np.copyto(by[3], sy * (sy * sy))
    _w_exact(tx.astype(_F64), tmp, bx[1])
    _w_exact(sx.astype(_F64), tmp, bx[2])
    _w_exact(ty.astype(_F64), tmp, by[1])
    _w_exact(sy.astype(_F64), tmp, by[2])
    idx0 = base + (ny - 1) * ncol + nx           # COEFF[ny-1, nx]
    g = np.empty(npx, _F64)
    p = np.empty(npx, _F64)
    s32 = np.empty(npx, _F32)
    acc32 = np.empty(npx, _F32)
    low = np.empty(npx, np.int64)
    tie = np.zeros(npx, bool)
    hit = np.empty(npx, bool)

    def fma_step(gv, w, prev32, out32):
        # out32 = round32(gv*w + prev32); flag float32-midpoint sums
        np.multiply(gv, w, out=p)
        np.add(p, prev32, out=p)
        np.copyto(out32, p, casting="unsafe")
        np.bitwise_and(p.view(np.int64), _TIE_MASK, out=low)
        np.equal(low, _TIE_HALF, out=hit)
        np.logical_or(tie, hit, out=tie)

    ties: list = [None] * len(flats64)
    for k, flat in enumerate(flats64):
        tie[:] = False
        acc32.fill(0.0)
        for j in range(4):
            idx = idx0 + j * ncol
            np.multiply(flat[idx], bx[1], out=p)
            np.copyto(s32, p, casting="unsafe")          # bx2*c[0] (REAL product)
            fma_step(flat[idx - 1], bx[0], s32, s32)
            fma_step(flat[idx + 1], bx[2], s32, s32)
            fma_step(flat[idx + 2], bx[3], s32, s32)
            np.copyto(g, s32)
            fma_step(g, by[j], acc32, acc32)
        outs[k][...] = acc32.reshape(shape)
        ties[k] = np.nonzero(tie)[0]
    return ties
