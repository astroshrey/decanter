"""Parse IRAF ``identify`` / ``ecidentify`` database files (``database/id*``).

For WINERED, each echelle order ``m`` gets one such file written by
``iraf.ecidentify`` containing a 1-D wavelength solution
``λ(pixel) = Σ_n c_n · T_n(x_norm)`` where ``x_norm`` is the pixel
position mapped to ``[-1, 1]`` over ``[xmin, xmax]``.

Subset supported (what WARP's calibration produces):
  - ``coefficients`` block format only (no ``features``-based fits).
  - Function types: Chebyshev (1) and Legendre (2).

Reference: IRAF ``noao.onedspec`` source, ``identify.x``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from decanter.io.iraf_fc import (
    FUNCTION_CHEBYSHEV,
    FUNCTION_LEGENDRE,
    _cheb_basis,
    _legendre_basis,
)


@dataclass(frozen=True, slots=True)
class IdSolution:
    """One parsed 1-D wavelength solution from an ``id*`` file."""

    image_name: str
    units: str
    ftype: int
    order: int             # number of terms in the polynomial
    xmin: float
    xmax: float
    coefficients: NDArray[np.float64]  # length ``order``


def parse(id_path: Path | str) -> IdSolution:
    """Parse an ``identify`` database file's coefficients block.

    IRAF databases are append-only: a file may hold several ``begin``
    records for the same aperture (seen in the LCO26a HIRES-J archive
    sets). IRAF's ``dt_locate`` (``noao/lib/dttext.x``) sequentially
    scans and keeps the offset of the LAST matching record, so the
    newest solution wins — we mirror that by parsing only the final
    ``begin`` record.
    """
    text = Path(id_path).read_text()
    records: list[list[str]] = []
    for line in text.splitlines():
        if line.strip().startswith("begin"):
            records.append([])
        if records:
            records[-1].append(line)
    if not records:
        raise ValueError(f"{id_path}: no 'begin' record found")

    image_name: str | None = None
    units = "Angstroms"
    in_coef = False
    coef_count: int | None = None
    coef_tokens: list[str] = []

    for line in records[-1]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        head = parts[0]
        if head == "image":
            image_name = " ".join(parts[1:])
        elif head == "units":
            units = " ".join(parts[1:])
        elif head == "coefficients":
            in_coef = True
            coef_count = int(parts[1])
        elif in_coef and len(coef_tokens) < (coef_count or 0):
            coef_tokens.extend(parts)

    if image_name is None or coef_count is None:
        raise ValueError(f"{id_path}: missing image/coefficients block")
    if len(coef_tokens) != coef_count:
        raise ValueError(
            f"{id_path}: declared {coef_count} coefficient values, got {len(coef_tokens)}"
        )

    # IRAF coefficients block format (1-D):
    #   ftype  order  xmin  xmax  c_0  c_1  ...  c_{order-1}
    if len(coef_tokens) < 4:
        raise ValueError(f"{id_path}: coefficients block too short")
    ftype = int(float(coef_tokens[0]))
    order = int(float(coef_tokens[1]))
    xmin = float(coef_tokens[2])
    xmax = float(coef_tokens[3])
    coefs = np.array([float(t) for t in coef_tokens[4 : 4 + order]], dtype=np.float64)
    if coefs.size != order:
        raise ValueError(
            f"{id_path}: expected {order} coefficients, got {coefs.size}"
        )
    return IdSolution(
        image_name=image_name,
        units=units,
        ftype=ftype,
        order=order,
        xmin=xmin,
        xmax=xmax,
        coefficients=coefs,
    )


def wavelength_at_pixels(solution: IdSolution, pixels_1idx: NDArray) -> NDArray[np.float64]:
    """Evaluate ``λ(pixel)`` using the parsed solution.

    Args:
        solution: parsed :class:`IdSolution`.
        pixels_1idx: 1-indexed pixel positions (IRAF convention).

    Returns:
        Wavelength values in ``solution.units``.
    """
    x = np.asarray(pixels_1idx, dtype=np.float64)
    x_n = (2.0 * x - solution.xmin - solution.xmax) / (solution.xmax - solution.xmin)
    if solution.ftype == FUNCTION_CHEBYSHEV:
        basis = _cheb_basis(x_n, solution.order)
    elif solution.ftype == FUNCTION_LEGENDRE:
        basis = _legendre_basis(x_n, solution.order)
    else:
        raise NotImplementedError(f"identify ftype={solution.ftype} not implemented")
    return basis @ solution.coefficients


def _dwave_dpix(solution: IdSolution, pixels_1idx: NDArray) -> NDArray[np.float64]:
    """Analytic ``dλ/dpix`` derivative of the wavelength solution.

    Used by :func:`pixels_at_wavelength` to do Newton refinement on the
    inverse mapping ``λ → pix``.
    """
    x = np.asarray(pixels_1idx, dtype=np.float64)
    span = solution.xmax - solution.xmin
    x_n = (2.0 * x - solution.xmin - solution.xmax) / span
    if solution.ftype == FUNCTION_CHEBYSHEV:
        d_basis = _cheb_deriv_basis(x_n, solution.order)
    elif solution.ftype == FUNCTION_LEGENDRE:
        d_basis = _legendre_deriv_basis(x_n, solution.order)
    else:
        raise NotImplementedError(f"identify ftype={solution.ftype} not implemented")
    # d/dpix = (2/span) * d/dx_n
    return (d_basis @ solution.coefficients) * (2.0 / span)


def _cheb_deriv_basis(t: NDArray, n: int) -> NDArray[np.float64]:
    """``dT_k/dt`` for ``k = 0..n-1`` via the recurrence
    ``U_{k-1}(t) = (1/k) dT_k/dt``, with ``U_{-1} = 0``, ``U_0 = 1``.

    Returns array of shape ``(len(t), n)``.
    """
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros((t.size, n), dtype=np.float64)
    if n >= 2:
        # T_1 = t → dT_1/dt = 1 = U_0
        out[:, 1] = 1.0
    # Build U_{k-1} via U_k = 2 t U_{k-1} - U_{k-2}, k = 1, 2, ...
    if n >= 3:
        u_km2 = np.ones_like(t)        # U_0
        u_km1 = 2.0 * t                # U_1
        for k in range(2, n):
            # dT_k/dt = k * U_{k-1}
            out[:, k] = k * u_km1
            u_k = 2.0 * t * u_km1 - u_km2
            u_km2 = u_km1
            u_km1 = u_k
    return out


def _legendre_deriv_basis(t: NDArray, n: int) -> NDArray[np.float64]:
    """``dP_k/dt`` for ``k = 0..n-1`` via Bonnet's recurrence
    ``(1 - t^2) P'_k = -k t P_k + k P_{k-1}``.

    Returns array of shape ``(len(t), n)``.
    """
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros((t.size, n), dtype=np.float64)
    if n >= 2:
        out[:, 1] = 1.0  # dP_1/dt = 1
    # Build P_k recurrence in parallel.
    if n >= 3:
        p_km2 = np.ones_like(t)        # P_0
        p_km1 = t.copy()               # P_1
        denom = 1.0 - t * t
        safe = denom != 0.0
        for k in range(2, n):
            p_k = ((2 * k - 1) * t * p_km1 - (k - 1) * p_km2) / k
            num = -k * t * p_k + k * p_km1
            dpk = np.zeros_like(t)
            dpk[safe] = num[safe] / denom[safe]
            # At t = ±1, use the closed-form P'_k(±1) = (±1)^(k+1) k(k+1)/2.
            if not np.all(safe):
                lim = k * (k + 1) / 2.0
                edge = ~safe
                sign_plus = t[edge] > 0.0
                dpk[edge] = np.where(sign_plus, lim, (-1.0) ** (k + 1) * lim)
            out[:, k] = dpk
            p_km2 = p_km1
            p_km1 = p_k
    return out


def pixels_at_wavelength(
    solution: IdSolution,
    wavelengths: NDArray,
    *,
    xrange: tuple[float, float] | None = None,
    n_table: int = 16,
    newton_iters: int = 3,
) -> NDArray[np.float64]:
    """Invert ``λ(pixel)`` to find the 1-indexed fractional pixel(s).

    Implementation: build a dense table of ``λ`` at evenly-spaced pixel
    positions covering ``xrange``, use ``np.interp`` for the initial
    guess (the table is monotonic for well-behaved id solutions), then
    refine with Newton's method using the analytic derivative.

    Args:
        solution: parsed :class:`IdSolution`.
        wavelengths: target wavelength values.
        xrange: lower/upper pixel bounds for the table; defaults to
            ``(0.5, ?)`` where the upper bound is read from
            ``solution.xmax`` (the natural range over which the fit is
            valid). Pass an explicit range when the input image is
            narrower than the fit range.
        n_table: oversampling factor for the initial-guess table. The
            table has roughly ``n_table * (xmax - xmin)`` points.
        newton_iters: number of Newton refinement steps.

    Returns:
        Float64 fractional pixel positions, same shape as ``wavelengths``.
        Values may extend slightly outside ``xrange`` if the target
        wavelength falls outside the table coverage.
    """
    w_target = np.asarray(wavelengths, dtype=np.float64)
    if xrange is None:
        xrange = (solution.xmin, solution.xmax)
    x_lo, x_hi = float(xrange[0]), float(xrange[1])
    n_pts = max(int(n_table * (x_hi - x_lo)) + 1, 64)
    x_table = np.linspace(x_lo, x_hi, n_pts)
    w_table = wavelength_at_pixels(solution, x_table)

    # np.interp requires monotonically increasing xp.
    if w_table[-1] < w_table[0]:
        x_init = np.interp(w_target, w_table[::-1], x_table[::-1])
    else:
        x_init = np.interp(w_target, w_table, x_table)

    x = x_init.astype(np.float64, copy=True)
    for _ in range(newton_iters):
        w_at_x = wavelength_at_pixels(solution, x)
        dwdx = _dwave_dpix(solution, x)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = (w_target - w_at_x) / dwdx
        step = np.where(np.isfinite(step), step, 0.0)
        x = x + step
    return x
