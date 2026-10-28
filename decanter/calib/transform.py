"""Geometric transform / order-rectification helpers.

WARP equivalent: ``warp/cutransform.py:cutransform`` -- ``imcopy`` of the
order's columns, then ``iraf.transform(cut, trans, fitnames=<aptrans>_<m>,
flux="no", dy=dy, interptype="spline3")``.

decanter reproduces that IRAF task bit for bit (0 differing float32 pixels
against WARP's saved ``_m###trans.fits``, object and sky):
:mod:`decanter.rectify.iraf_transform` ports ``longslit.transform``
(``trsetup.x`` / ``t_transform.x``), the double-precision ``gsurfit``
evaluators it uses for the ``fc`` surface
(:mod:`decanter.utils.gsurfit_exact`) and the ``iminterp`` bilinear /
bicubic-spline kernels (:mod:`decanter.utils.spline3`).

What IRAF actually does (and decanter therefore does):

1. The output grid is ``nu = xmax - xmin + 1`` columns of the surface's x
   range and ``nv = nint((vmax - vmin)/dy + 1)`` rows from the minimum
   surface value over the four corners (``CRVAL2 = vmin``).
2. The inverse ``X(U,V), Y(U,V)`` is computed only on a ``step = 10``
   subsampled grid by a stateful Newton iteration (``tr_invert``: fudge
   factor ``1 - 0.5/i``, tolerance 0.05 px, each point starting from the
   previous one along a boustrophedon), then bilinearly interpolated
   (``msifit(II_BILINEAR)``) to every output pixel.  Neither X nor Y is
   exact: X deviates from U by up to ~0.003 px.
3. Each output line is a 2-D bicubic B-spline (``II_BISPLINE3``,
   natural end conditions) evaluated at the clamped input coordinates; the
   spline is fitted not to the whole image but to a sliding buffer of
   input lines (``NBUF = 16`` plus ``NEDGE = 2``), refitted whenever the
   needed rows leave it -- so the end conditions of each buffer window
   enter the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from decanter.utils.gsurfit_exact import GsSurface

# IRAF transform's output-coordinate subsampling factor (hardcoded at
# ``noao/twodspec/longslit/transform/trsetup.x:182``).
_IRAF_TRANSFORM_STEP = 10


@dataclass(frozen=True, slots=True)
class RectifiedStrip:
    """Result of rectifying one echelle order.

    ``lambda_min`` / ``dy`` are the strip's ``CRVAL2`` / ``CDELT2`` *as
    WARP's ``_m###trans.fits`` header carries them*: IRAF's transform holds
    them as REALs and ``imputd`` writes them with IRAF's (not correctly
    rounded) ``%.15g``; ``crval2_text`` is that exact header text and
    ``lambda_min = float(crval2_text)``.
    """

    data: NDArray[np.float32]  # (n_out_y, n_x) -- slit_x x wavelength
    lambda_min: float           # output y=1 coordinate (CRVAL2 header value)
    lambda_max: float           # output y=n_out_y coordinate
    dy: float                   # output step per row (CDELT2 header value)
    xmin: int                   # 1-indexed; first input column included in strip
    xmax: int                   # 1-indexed; last input column included
    crval2_text: str = ""       # CRVAL2 exactly as IRAF writes it
    cdelt2_text: str = ""       # CDELT2 exactly as IRAF writes it


def strip_from_plan(data, plan, xmin: int, xmax: int) -> RectifiedStrip:
    """Package a transform result with the header WCS IRAF writes for it.

    IRAF: ``t_transform.x:tr_transform`` (``mw_swtermr`` with
    ``w[2] = v1``, ``cd[2,2] = dv``) -> ``mw_saveim`` -> ``imputd``.
    """
    from decanter.utils.iraf_fmt import iraf_imputd_text

    t_crval = iraf_imputd_text(float(plan.v1))
    t_cdelt = iraf_imputd_text(float(plan.dv))
    return RectifiedStrip(
        data=data, lambda_min=float(t_crval), lambda_max=float(plan.v2),
        dy=float(t_cdelt), xmin=xmin, xmax=xmax, crval2_text=t_crval,
        cdelt2_text=t_cdelt,
    )


def extracted_crval1(strip: RectifiedStrip) -> float:
    """``CRVAL1`` of the 1-D spectrum WARP's ``apall`` writes from this strip.

    ``apall`` reads the strip's ``CRVAL2`` text back through MWCS (``ctod``)
    and writes it again with ``imputd``; this reproduces that round trip
    (verified on every order of TOI2109_decanterref and TOI3486_decanterref
    against WARP's ``_m###.0###.fits`` headers).
    """
    from decanter.utils.fpemu import iraf_ctod
    from decanter.utils.iraf_fmt import iraf_imputd_text

    if not strip.crval2_text:
        return float(strip.lambda_min)
    return float(iraf_imputd_text(iraf_ctod(strip.crval2_text)))


def load_fc_surface(fc_path: Path | str) -> GsSurface:
    """Read a ``fitcoords`` database entry as IRAF ``lm_dbread`` does.

    IRAF: ``noao/twodspec/longslit/transform/fcdbio.x:lm_dbread`` (values
    parsed with ``ctod``, restored by ``dgsrestore``).
    """
    return GsSurface.from_fc_file(fc_path)


def rectify_order(
    image: NDArray[np.floating],
    surface: GsSurface | Path | str,
    *,
    xmin: int,
    xmax: int,
    dy: float,
    array_length: int = 2048,
    fill_value: float = 0.0,
) -> RectifiedStrip:
    """Cut and transform one echelle order exactly as WARP does.

    IRAF equivalent: ``imcopy image[xmin:xmax,*] cut`` then
    ``longslit.transform cut trans fitnames=... flux=no dy=dy
    interptype=spline3`` (``t_transform.x``, ``trsetup.x``).

    Args:
        image: full 2-D detector frame (cast to REAL, as IRAF reads it).
        surface: the order's ``fc`` surface (:func:`load_fc_surface`) or the
            path of its ``database/fc*`` file (enables plan caching).
        xmin, xmax: 1-indexed inclusive column range of the cut.
        dy: output step (WARP's ``dyinput`` = comp ``CDELT1``).
        array_length: detector y-extent (checked).
        fill_value: unused (IRAF transform with ``blank=INDEF`` extends the
            edge instead of filling); kept for API compatibility.
    """
    from decanter.rectify import iraf_transform as T

    img = np.asarray(image)
    if img.shape[0] != array_length:
        raise ValueError(f"image y-extent {img.shape[0]} != array_length {array_length}")
    if isinstance(surface, (str, Path)):
        plan = T.get_plan(surface, dy)
    else:
        if surface.axis != 2:
            raise ValueError(
                f"rectify_order expects an axis=2 (dispersion) surface; got axis={surface.axis}")
        plan = T.plan_for_surface(surface, dy)
    cut = np.ascontiguousarray(img[:, xmin - 1:xmax], dtype=np.float32)
    out = T.tr_transform(cut, plan, T.schedule_for(plan, cut.shape[0]))
    return strip_from_plan(out, plan, xmin, xmax)
