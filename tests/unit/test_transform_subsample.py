"""Lock in IRAF transform's subsampled+bilinear inverse coordinate path.

Pywarp's ``rectify_order`` previously computed the analytical fc-surface
inverse at every output pixel — more precise than IRAF, but ironically
gave 0.10% medrel against WARP's saved ``_m###trans.fits`` because IRAF's
transform task subsamples the inverse on a coarse (step=10) grid and
bilinearly interpolates (``transform/trsetup.x:182``). To match WARP
byte-for-byte we must introduce the same bilinear-interp error.

These tests guard the subsample-and-bilinear-interp shape contract;
the WARP per-pixel parity regression lives in
``scripts/deep_dive_s06_transform.py``.
"""

from __future__ import annotations

import numpy as np

from decanter.calib.transform import _IRAF_TRANSFORM_STEP


def test_iraf_subsample_step_matches_iraf() -> None:
    """IRAF transform hardcodes ``step = 10`` at trsetup.x:182. The
    ``msifit(II_BILINEAR)`` inverse interpolator is built on a
    ``max(2, nu // 10) × max(2, nv // 10)`` grid. Pywarp must use the
    same constant."""
    assert _IRAF_TRANSFORM_STEP == 10


def test_subsample_grid_dims() -> None:
    """For WINERED-typical (nu=382, nv=5628), nu1=38 and nv1=562."""
    nu, nv = 382, 5628
    step = _IRAF_TRANSFORM_STEP
    nu1 = max(2, nu // step)
    nv1 = max(2, nv // step)
    assert (nu1, nv1) == (38, 562)


def test_subsample_grid_covers_endpoints() -> None:
    """The subsampled grid must cover both endpoints (output pixels 1
    and nu / 1 and nv) so the bilinear-interp extrapolation never fires
    at output-image corners."""
    nu, nv = 382, 5628
    nu1 = max(2, nu // _IRAF_TRANSFORM_STEP)
    nv1 = max(2, nv // _IRAF_TRANSFORM_STEP)
    du1 = (nu - 1) / (nu1 - 1)
    dv1 = (nv - 1) / (nv1 - 1)
    u_sub = 1.0 + np.arange(nu1, dtype=np.float64) * du1
    v_sub = 1.0 + np.arange(nv1, dtype=np.float64) * dv1
    assert u_sub[0] == 1.0
    np.testing.assert_allclose(u_sub[-1], float(nu), atol=1e-9)
    assert v_sub[0] == 1.0
    np.testing.assert_allclose(v_sub[-1], float(nv), atol=1e-9)
