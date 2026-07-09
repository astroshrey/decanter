"""Geometric transform / order-rectification helpers.

WARP equivalent: ``warp/cutransform.py:cutransform`` — wraps
``iraf.transform interptype=spline3`` to rectify each echelle order's
strip from the raw detector frame into ``(slit_position, wavelength)``
coordinates, removing slit tilt and putting the dispersion axis on a
uniform physical-unit grid.

Approach
--------

WARP's IRAF call uses a ``fitcoords`` surface stored in
``database/fc<refname>_<order>`` that gives wavelength as a function of
``(x_pixel, y_pixel)``. We:

  1. Parse the IRAF ``fc*`` file (see :mod:`decanter.io.iraf_fc`).
  2. For each output column (which maps 1:1 to an input column), invert
     the surface column-by-column: ``λ(x_col, y_in) = y_out_pixel * dy``.
     The surface is monotone in ``y_in`` for a well-calibrated echelle,
     so the inversion is just a 1-D interpolation against pre-computed
     ``λ(y_in)`` at that column.
  3. Sample the input column at the inverted ``y_in`` values via the
     natural cubic B-spline used by IRAF (`math/iminterp/ii_spline.x`
     + `ii_bispline3` in `ii_bieval.x`). decanter's cubic-spline kernel
     lives in :mod:`decanter.utils.spline3` and matches IRAF's algorithm
     bit-for-bit at float32; replacing the prior
     ``scipy.ndimage.map_coordinates(order=3, prefilter=True)`` (which
     uses mirror-BC prefiltering) was required to close the s06 parity
     gap against WARP's saved ``_m###trans.fits``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scipy.interpolate import RectBivariateSpline

from decanter.io.iraf_fc import FcSurface, evaluate, parse
from decanter.utils.spline3 import iraf_spline3_resample_columns

# IRAF transform's output-coordinate subsampling factor (hardcoded
# at ``noao/twodspec/longslit/transform/trsetup.x:182``):
#     step = 10
#     nu1 = max(2, nu / step)
#     nv1 = max(2, nv / step)
# IRAF computes the analytical inverse only on this coarse grid (via
# Newton iteration in ``tr_invert``), then bilinearly interpolates to
# every output pixel using ``msifit(II_BILINEAR)``. This subsampling +
# bilinear-interp introduces a small-but-real error into y_in compared
# to a per-pixel exact inverse — and decanter must reproduce it for
# bit-parity with WARP's saved ``_m###trans.fits``.
_IRAF_TRANSFORM_STEP = 10


@dataclass(frozen=True, slots=True)
class RectifiedStrip:
    """Result of rectifying one echelle order."""

    data: NDArray[np.float32]  # (n_out_y, n_x) — slit_x × wavelength
    lambda_min: float           # output y=1 corresponds to this wavelength
    lambda_max: float           # output y=n_out_y corresponds to this wavelength
    dy: float                   # output wavelength spacing per pixel
    xmin: int                   # 1-indexed; first input column included in strip
    xmax: int                   # 1-indexed; last input column included


def load_fc_surface(fc_path: Path | str) -> FcSurface:
    """Convenience wrapper for :func:`decanter.io.iraf_fc.parse`."""
    return parse(fc_path)


def rectify_order(
    image: NDArray[np.floating],
    surface: FcSurface,
    *,
    xmin: int,
    xmax: int,
    dy: float,
    array_length: int = 2048,
    fill_value: float = 0.0,
) -> RectifiedStrip:
    """Rectify one echelle order's strip on a uniform wavelength grid.

    Args:
        image: full 2-D detector frame (any dtype; converted to float64
            for interpolation).
        surface: parsed ``fitcoords`` surface (axis=2 expected — the
            surface maps to wavelength).
        xmin, xmax: 1-indexed inclusive column range to extract from
            ``image`` (the cut strip is ``image[:, xmin-1:xmax]``).
        dy: output wavelength step per pixel (matches WARP's ``dyinput``
            = comp file ``CDELT1``).
        array_length: detector y-extent (default 2048 for WINERED).
        fill_value: value used for output pixels whose inverse-mapped
            ``y_in`` falls outside ``[1, array_length]``.

    Returns:
        :class:`RectifiedStrip` with the resampled data and the
        wavelength bounds used for the output grid.

    Notes:
        For the surface to be invertible column-by-column we require
        wavelength to be monotone in ``y_pixel`` along each column;
        this holds for HIRES-Y orders in standard configurations.
        Non-monotone columns degrade gracefully: ``np.interp`` is fed
        the sorted unique values via ``np.argsort``.
    """
    if surface.axis != 2:
        raise ValueError(
            f"rectify_order expects an axis=2 (dispersion) surface; got axis={surface.axis}"
        )
    image_f = np.asarray(image, dtype=np.float64)
    H, _W = image_f.shape
    if H != array_length:
        raise ValueError(f"image y-extent {H} != array_length {array_length}")

    n_x = xmax - xmin + 1
    # The fc surface was fit on the *strip-local* coordinate frame (xmin=1,
    # xmax≈382 in the surface metadata), not the detector frame. So the
    # surface input x runs 1..n_x along the strip, NOT xmin..xmax on the
    # detector. The y-input is still the detector row (1..array_length).
    x_strip_1idx = np.arange(1, n_x + 1, dtype=np.float64)
    y_in_1idx = np.arange(1, array_length + 1, dtype=np.float64)

    # λ(x, y) on a dense (x_strip, y_detector) grid.
    xg, yg = np.meshgrid(x_strip_1idx, y_in_1idx, indexing="xy")
    lambda_grid = evaluate(surface, xg, yg)  # shape (array_length, n_x)

    # Output wavelength grid: span the union of every column's λ range.
    lambda_min_global = float(lambda_grid.min())
    lambda_max_global = float(lambda_grid.max())
    n_out_y = int(np.round((lambda_max_global - lambda_min_global) / dy)) + 1
    lambda_out = lambda_min_global + dy * np.arange(n_out_y, dtype=np.float64)

    # IRAF-faithful inverse: compute y_in only on a subsampled
    # (nu1 × nv1) output grid, then bilinearly interpolate to fill the
    # full (n_out_y, n_x) grid. This mirrors IRAF
    # ``transform/trsetup.x:272-348`` (``step=10`` → coarse grid + Newton
    # at each subsample → ``msifit(II_BILINEAR)`` → ``msivector`` to
    # interpolate). The bilinear-interp error is what makes IRAF's
    # ``_trans.fits`` differ from a per-pixel exact inverse, so to match
    # WARP byte-for-byte we have to introduce the same error.
    nu1 = max(2, n_x // _IRAF_TRANSFORM_STEP)
    nv1 = max(2, n_out_y // _IRAF_TRANSFORM_STEP)
    du1 = (n_x - 1) / (nu1 - 1)
    dv1 = (n_out_y - 1) / (nv1 - 1)
    u_sub = 1.0 + np.arange(nu1, dtype=np.float64) * du1   # fractional strip cols
    v_sub = 1.0 + np.arange(nv1, dtype=np.float64) * dv1   # fractional output rows

    # Build λ(y_int, u_sub) at every input row and each subsample column.
    # The fc surface evaluator handles fractional u (no requirement that
    # u_sub be integer).
    lam_subgrid = np.empty((array_length, nu1), dtype=np.float64)
    for jj, u_val in enumerate(u_sub):
        lam_subgrid[:, jj] = evaluate(
            surface, np.full(array_length, u_val), y_in_1idx,
        )

    # Invert lam_subgrid at each subsample column for every subsampled
    # output wavelength. (np.interp at fp64 is well within the precision
    # IRAF's tr_invert achieves at fp32.)
    y_sub = np.empty((nv1, nu1), dtype=np.float64)
    for jj in range(nu1):
        lam_col = lam_subgrid[:, jj]
        order = np.argsort(lam_col)
        lam_sorted = lam_col[order]
        y_sorted = y_in_1idx[order]
        for ii, v_val in enumerate(v_sub):
            lam_target = lambda_min_global + (v_val - 1.0) * dy
            y_sub[ii, jj] = float(np.interp(lam_target, lam_sorted, y_sorted))

    # Bilinear interp (matches IRAF ``msifit(II_BILINEAR) + msivector``).
    bilinear = RectBivariateSpline(v_sub, u_sub, y_sub, kx=1, ky=1, s=0)
    v_full = 1.0 + np.arange(n_out_y, dtype=np.float64)
    u_full = 1.0 + np.arange(n_x, dtype=np.float64)
    y_in_out = bilinear(v_full, u_full, grid=True)

    # IRAF ``interptype=spline3`` resampling: column-wise natural cubic
    # B-spline (zero 2nd derivative at endpoints, uniform unit spacing),
    # exactly as ``math/iminterp/ii_spline.x`` + ``ii_bispline3`` would
    # evaluate it. Pywarp's previous ``map_coordinates(order=3,
    # prefilter=True)`` used scipy's mirror-BC B-spline prefilter — same
    # basis but different boundary condition — which produced 0.04 ct
    # median |Δ| / 2852 ct max |Δ| on TOI2109 vs WARP's saved
    # _m163trans.fits. Switching to the IRAF-fidelity 1-D kernel closes
    # the kernel-side parity gap.
    #
    # Out-of-bounds y_in (i.e. y_in < 1 or y_in > array_length) is
    # clamped to the boundary value (IRAF transform's behavior).
    cut = image_f[:, xmin - 1 : xmax]  # shape (H, n_x)
    resampled = iraf_spline3_resample_columns(cut, y_in_out)

    return RectifiedStrip(
        data=resampled.astype(np.float32, copy=False),
        lambda_min=lambda_min_global,
        lambda_max=lambda_max_global,
        dy=dy,
        xmin=xmin,
        xmax=xmax,
    )
