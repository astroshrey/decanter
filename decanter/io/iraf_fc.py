"""Parse IRAF ``fitcoords`` database files (``database/fc*``).

These files store a 2-D polynomial surface fit produced by
``iraf.fitcoords`` against the line-identification residuals from
``iraf.ecidentify``. For WINERED, each echelle order has one such
surface mapping ``(x_pixel, y_pixel) → wavelength`` (axis 2 fit).

File format (single-task subset that WARP uses)::

    # <timestamp>
    begin   <image_name>
        task    fitcoords
        axis    <1 or 2>
        units   <e.g. angstroms>
        surface <N>
            <ftype>     1=cheb 2=leg 3=linear 4=power
            <xorder>    number of x-direction terms (not the degree)
            <yorder>    number of y-direction terms
            <cross>     1=full cross-term, 2=half, 3=diagonal/none
            <xmin>
            <xmax>
            <ymin>
            <ymax>
            <coef_0_0>
            <coef_1_0>
            ...
            <coef_(xorder-1)_(yorder-1)>

The number of coefficients depends on ``cross`` (we only support the
``cross=1`` full-product form here, which is what WARP's calibration
produces for WINERED).

Reference: IRAF ``twodspec.longslit`` source, ``iclookup.c``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FUNCTION_CHEBYSHEV = 1
FUNCTION_LEGENDRE = 2
FUNCTION_LINEAR = 3
FUNCTION_POWER = 4


@dataclass(frozen=True, slots=True)
class FcSurface:
    """One parsed ``fitcoords`` surface.

    Attributes:
        image_name: the ``begin <image_name>`` value (e.g.
            ``"multihole_HIRES-Y100_20250806_163"``).
        axis: axis the surface fits (1 = cross-dispersion, 2 = dispersion).
        units: physical units of the output (e.g. ``"angstroms"``).
        ftype: function type — one of the ``FUNCTION_*`` constants.
        xorder, yorder: number of terms in each direction.
        cross: cross-term mode (1=full, 2=half, 3=none). We require 1.
        xmin, xmax, ymin, ymax: normalization range. Inputs ``(x, y)``
            are mapped to ``[-1, 1]^2`` before evaluation.
        coefficients: 2-D array of shape ``(xorder, yorder)`` ordered
            with x varying fastest (column-major in IRAF's listing).
    """

    image_name: str
    axis: int
    units: str
    ftype: int
    xorder: int
    yorder: int
    cross: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    coefficients: NDArray[np.float64]


def parse(fc_path: Path | str) -> FcSurface:
    """Parse a single-surface ``fitcoords`` database file.

    Args:
        fc_path: path to a ``database/fc*`` file.

    Returns:
        :class:`FcSurface` with normalization range and coefficient grid.

    Raises:
        ValueError: if the file has the wrong structure (multiple
            surfaces, wrong cross-term mode, mismatched coefficient
            count, etc.).
    """
    text = Path(fc_path).read_text()
    tokens: list[str] = []
    image_name: str | None = None
    axis: int | None = None
    units: str | None = None
    n_surface_values: int | None = None
    in_surface = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        head = parts[0]
        if head == "begin":
            image_name = " ".join(parts[1:])
        elif head == "task":
            if parts[1] != "fitcoords":
                raise ValueError(f"expected task=fitcoords, got {parts[1]} in {fc_path}")
        elif head == "axis":
            axis = int(parts[1])
        elif head == "units":
            units = " ".join(parts[1:])
        elif head == "surface":
            n_surface_values = int(parts[1])
            in_surface = True
        elif in_surface:
            # Inside the surface block; each remaining whitespace-split
            # token is one float (or int).
            tokens.extend(parts)

    if image_name is None or axis is None or units is None or n_surface_values is None:
        raise ValueError(f"{fc_path}: missing required surface metadata")
    if len(tokens) != n_surface_values:
        raise ValueError(
            f"{fc_path}: surface declared {n_surface_values} values, got {len(tokens)}"
        )

    # First 8 tokens are the surface header; the rest are coefficients.
    if len(tokens) < 8:
        raise ValueError(f"{fc_path}: surface header too short")
    ftype = int(float(tokens[0]))
    xorder = int(float(tokens[1]))
    yorder = int(float(tokens[2]))
    cross = int(float(tokens[3]))
    xmin = float(tokens[4])
    xmax = float(tokens[5])
    ymin = float(tokens[6])
    ymax = float(tokens[7])

    if cross != 1:
        raise NotImplementedError(
            f"{fc_path}: only cross=1 (full cross-term) supported; got cross={cross}"
        )
    expected_n_coef = xorder * yorder
    coef_tokens = tokens[8:]
    if len(coef_tokens) != expected_n_coef:
        raise ValueError(
            f"{fc_path}: expected {expected_n_coef} coefficients for "
            f"xorder={xorder}, yorder={yorder} (cross=1); got {len(coef_tokens)}"
        )
    coeffs_flat = np.asarray([float(t) for t in coef_tokens], dtype=np.float64)
    # IRAF stores coefficients with x varying fastest (column-major); we
    # use ``coefficients[i, j]`` indexed as `[x_term, y_term]`.
    coefficients = coeffs_flat.reshape((yorder, xorder)).T

    return FcSurface(
        image_name=image_name,
        axis=axis,
        units=units,
        ftype=ftype,
        xorder=xorder,
        yorder=yorder,
        cross=cross,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        coefficients=coefficients,
    )


def evaluate(surface: FcSurface, x: NDArray, y: NDArray) -> NDArray:
    """Evaluate the IRAF ``fitcoords`` surface at ``(x, y)`` pairs.

    IRAF-faithful port of ``math/gsurfit/gs_fevalr.x:rgs_evcheb`` /
    ``rgs_evleg`` (the full-cross-terms branch, ``cross=1``):

        zfit = 0
        for i in 1..yorder:
            accum = 0
            for k in 1..xorder:
                accum = accum + xb[k] * coeff[(i-1)*xorder + k]
            zfit = zfit + accum * yb[i]

    All arithmetic in float32 to match IRAF ``real``. The basis
    functions ``rgs_bcheb`` / ``rgs_bleg`` are also computed in
    float32.

    Args:
        surface: parsed :class:`FcSurface`.
        x, y: input pixel coordinates (1-indexed in IRAF convention).

    Returns:
        Float64 surface values (cast at the end so callers don't see
        float32 in their dataflow).

    Notes:
        ``ftype=1`` (Chebyshev) and ``ftype=2`` (Legendre) are
        implemented; 3 (linear) and 4 (power) raise.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    out_shape = np.broadcast(x_arr, y_arr).shape
    x_flat = np.broadcast_to(x_arr, out_shape).reshape(-1)
    y_flat = np.broadcast_to(y_arr, out_shape).reshape(-1)

    # IRAF normalizes via k1 = (xmax + xmin)/2, k2 = (xmax - xmin)/2 →
    # x_n = (x - k1) / k2. Equivalent to our (2x - xmin - xmax) /
    # (xmax - xmin).
    x_n = ((2.0 * x_flat - surface.xmin - surface.xmax) /
           (surface.xmax - surface.xmin)).astype(np.float32)
    y_n = ((2.0 * y_flat - surface.ymin - surface.ymax) /
           (surface.ymax - surface.ymin)).astype(np.float32)

    if surface.ftype == FUNCTION_CHEBYSHEV:
        Tx = _cheb_basis(x_n, surface.xorder, dtype=np.float32)
        Ty = _cheb_basis(y_n, surface.yorder, dtype=np.float32)
    elif surface.ftype == FUNCTION_LEGENDRE:
        Tx = _legendre_basis(x_n, surface.xorder, dtype=np.float32)
        Ty = _legendre_basis(y_n, surface.yorder, dtype=np.float32)
    else:
        raise NotImplementedError(f"fitcoords ftype={surface.ftype} not implemented")

    # Coefficients in float32, indexed as IRAF stores them.
    # surface.coefficients has shape (xorder, yorder) with
    # `coefficients[k_x, i_y]` = IRAF coeff[(i_y)*xorder + (k_x+1)].
    coeff32 = surface.coefficients.astype(np.float32)

    npts = x_flat.size
    zfit = np.zeros(npts, dtype=np.float32)
    for i in range(surface.yorder):  # IRAF i = 1..yorder (0-based here)
        accum = np.zeros(npts, dtype=np.float32)
        for k in range(surface.xorder):  # IRAF k = 1..xorder
            # accum += xb[..., k] * coeff[k, i]   — IRAF awsur in fp32
            accum = (accum + Tx[:, k] * coeff32[k, i]).astype(np.float32)
        zfit = (zfit + accum * Ty[:, i]).astype(np.float32)

    return zfit.reshape(out_shape).astype(np.float64)


def _cheb_basis(t: NDArray, n: int, *, dtype=np.float64) -> NDArray:
    """First ``n`` Chebyshev T-polynomials evaluated at ``t`` (in [-1, 1])."""
    t = np.asarray(t, dtype=dtype)
    out_shape = t.shape + (n,)
    out = np.empty(out_shape, dtype=dtype)
    if n >= 1:
        out[..., 0] = dtype(1.0)
    if n >= 2:
        out[..., 1] = t
    two = dtype(2.0)
    for k in range(2, n):
        out[..., k] = (two * t * out[..., k - 1] - out[..., k - 2]).astype(dtype)
    return out


def _legendre_basis(t: NDArray, n: int, *, dtype=np.float64) -> NDArray:
    """First ``n`` Legendre P-polynomials evaluated at ``t``."""
    t = np.asarray(t, dtype=dtype)
    out_shape = t.shape + (n,)
    out = np.empty(out_shape, dtype=dtype)
    if n >= 1:
        out[..., 0] = dtype(1.0)
    if n >= 2:
        out[..., 1] = t
    for k in range(2, n):
        c1 = dtype((2.0 * k - 1.0) / k)
        c2 = dtype((k - 1.0) / k)
        out[..., k] = (c1 * t * out[..., k - 1] - c2 * out[..., k - 2]).astype(dtype)
    return out
