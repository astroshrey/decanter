"""Stage 6 -- cut & transform each echelle order (bit-exact WARP clone).

WARP equivalent: ``warp/cutransform.py:cutransform`` (line 94).
For each order ``m`` in the aperture set:

  1. ``imcopy`` the frame's columns ``[xmin, xmax]`` where
     ``xmin = int(trace_x[0] - 100)`` and ``xmax = int(trace_x[-1] + 100)``
     (clipped to ``[1, 2048]``).
  2. ``iraf.transform(cut, trans, fitnames=<aptrans>_<m>, flux="no",
     dy=dy, interptype="spline3")`` with the per-order ``fitcoords``
     surface ``database/fc<ref>_<m>``.
  3. Write ``{frame}_m{order}trans.fits`` with the linear WCS
     (``CRVAL2`` = the surface's minimum corner value, ``CDELT2`` = dy).

Output suffix: ``_m###trans``.  The transform is IRAF's own algorithm
reproduced operation for operation (:mod:`decanter.rectify.iraf_transform`);
it matches WARP's saved ``_m###trans.fits`` with 0 differing float32
values (object and sky).  The image-independent part (inverse-mapping grid,
line-buffer schedule) is computed once per surface and cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits as _astrofits
from numpy.typing import NDArray

from decanter.calib.aperture import ApertureSet
from decanter.calib.transform import RectifiedStrip, load_fc_surface, strip_from_plan
from decanter.config import Config
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.listfile import parse as parse_listfile

# WARP's hardcoded margin around the trace endpoints (warp/cutransform.py:97).
_XMARGIN = 100


def _sanitize_objname(raw: str) -> str:
    name = raw
    for ch in (" ", "'", "\"", "#", "/"):
        name = name.replace(ch, "_")
    return name


def _cut_limits(ap, array_length: int) -> tuple[int, int]:
    # warp/cutransform.py:117-125
    xmin = int(ap.trace_x[0] - _XMARGIN)
    xmax = int(ap.trace_x[-1] + _XMARGIN)
    return (1 if xmin < 1 else xmin), (array_length if xmax > array_length else xmax)


def rectify_orders_multi(
    images: Iterable[NDArray],
    apset: ApertureSet,
    *,
    fc_dir: Path,
    fc_refname: str,
    dy: float,
    orders: Iterable[int] | None = None,
    flux: bool = False,
) -> list[dict[int, RectifiedStrip]]:
    """:func:`rectify_orders` for several frames sharing one calibration.

    The frames (e.g. the object and its sky) are transformed with the same
    per-order plans; the image-independent work (inverse coordinates,
    B-spline weights) is shared between them.

    IRAF equivalent: ``longslit.transform`` per frame and order
    (``noao/twodspec/longslit/transform/t_transform.x``).
    """
    from decanter.rectify import iraf_transform as T

    imgs = [np.asarray(im) for im in images]
    fc_dir_p = Path(fc_dir)
    requested = tuple(orders) if orders is not None else tuple(apset.echelle_orders)
    plans = T.get_plans([fc_dir_p / f"fc{fc_refname}_{m}" for m in requested], dy)
    out: list[dict[int, RectifiedStrip]] = [dict() for _ in imgs]
    for m, plan in zip(requested, plans):
        xmin, xmax = _cut_limits(apset.apertures[m], apset.array_length)
        cuts = [np.ascontiguousarray(im[:, xmin - 1:xmax], dtype=np.float32) for im in imgs]
        sched = T.schedule_for(plan, cuts[0].shape[0])
        results = T.tr_transform_many(cuts, plan, sched, flux=flux)
        for k, res in enumerate(results):
            out[k][m] = strip_from_plan(res, plan, xmin, xmax)
    return out


def rectify_orders(
    data: NDArray,
    apset: ApertureSet,
    *,
    fc_dir: Path,
    fc_refname: str,
    dy: float,
    orders: Iterable[int] | None = None,
    flux: bool = False,
) -> dict[int, RectifiedStrip]:
    """Pure task function: cut and transform every echelle order of a frame.

    For each order in ``orders`` (defaults to ``apset.echelle_orders``),
    cuts ``data`` to ``[trace_x[0] - 100, trace_x[-1] + 100]`` and applies
    IRAF ``transform`` with the matching ``fc<refname>_<m>`` surface.

    IRAF equivalent: ``warp/cutransform.py:cutransform`` ->
    ``images.imcopy`` + ``twodspec.longslit.transform``.  ``flux`` is
    ``transform flux=`` (WARP's ``CUTRANSFORM flux``; ``t_transform.x:577-581``).

    Returns:
        ``{order: RectifiedStrip}`` -- one entry per requested order,
        each carrying ``data`` (REAL), ``xmin``, ``xmax``, ``lambda_min``
        (``CRVAL2``), ``lambda_max`` and ``dy`` (``CDELT2``).
    """
    return rectify_orders_multi([data], apset, fc_dir=fc_dir, fc_refname=fc_refname,
                                dy=dy, orders=orders, flux=flux)[0]


def _strip_header(base_header: _astrofits.Header, strip: RectifiedStrip,
                  units: str) -> _astrofits.Header:
    """Build an output FITS header with the rectified strip's WCS."""
    h = base_header.copy()
    h["DISPAXIS"] = 2
    h["CRVAL2"] = (strip.lambda_min, "Wavelength at output row 1")
    h["CRPIX2"] = (1.0, "Reference pixel along dispersion")
    h["CDELT2"] = (strip.dy, "Wavelength step per row")
    h["CTYPE2"] = ("LINEAR", "Wavelength axis is linear")
    h["WAT2_001"] = (
        f"wtype=linear label=Wavelength units={units}",
        "WCS attribute (WAT) for axis 2",
    )
    h["BUNIT"] = ("counts", "Photometric units")
    h["XMIN"] = (strip.xmin, "Input column (1-idx) of output col 1")
    h["XMAX"] = (strip.xmax, "Input column (1-idx) of output col -1")
    return h


def run(
    config: Config,
    workdir: Path,
    listfile: Path,
    *,
    apdb_path: Path | str | None = None,
    fc_dir: Path | str | None = None,
    fc_refname: str | None = None,
    dy: float | None = None,
    **_unused: Any,
) -> None:
    """Rectify each echelle order strip into ``(slit_x, wavelength)``.

    Args:
        config: pipeline configuration. Honors ``selected_orders``
            (subset of orders to process) and ``reduce_full_data``.
        workdir: reads ``{obj}_NO{i}_sscfm.fits`` (s05 output); writes
            ``{obj}_NO{i}_sscfm_m{m}trans.fits`` per echelle order.
        listfile: WARP-style input list.
        apdb_path: path to the aperture-trace database (the multihole
            apertures, not the flat ones).
        fc_dir: directory containing the per-order ``fc*`` files.
        fc_refname: base name used to locate ``fc<refname>_<m>`` in
            ``fc_dir`` (e.g. ``"multihole_HIRES-Y100_20250806"``).
        dy: output wavelength step per pixel (comp file ``CDELT1``).

    Notes:
        The output FITS shape (``n_lambda × n_x``) varies per order.
        ``n_x`` is roughly ``2 × 100 + slit_width`` (~382 columns for a
        typical HIRES-Y aperture) and ``n_lambda`` depends on the order's
        wavelength range divided by ``dy``.
    """
    missing = [
        name for name, val in [
            ("apdb_path", apdb_path),
            ("fc_dir", fc_dir),
            ("fc_refname", fc_refname),
            ("dy", dy),
        ] if val is None
    ]
    if missing:
        raise ValueError(
            f"s06_transform_cut requires {missing}; the calibration loader "
            "(s00) will provide these automatically once wired in."
        )

    apset = ApertureSet.load(apdb_path)
    orders = (
        apset.echelle_orders
        if (config.reduce_full_data or not config.selected_orders)
        else tuple(m for m in apset.echelle_orders if m in config.selected_orders)
    )
    if not orders:
        raise ValueError(
            f"no overlap between selected_orders={config.selected_orders} "
            f"and aperture orders {apset.echelle_orders}"
        )

    fc_dir_p = Path(fc_dir)
    pairs = parse_listfile(listfile)

    def _transform_frame(in_path: Path, out_prefix: str) -> None:
        """Run per-order rectification on ``in_path``; write outputs."""
        data, header = _fits.read_image(in_path)
        # Cache the units from the first surface; all surfaces share the unit.
        strips = rectify_orders(
            data, apset, fc_dir=fc_dir_p, fc_refname=fc_refname,
            dy=float(dy), orders=orders,
        )
        for m, strip in strips.items():
            # Need the surface units string for the WAT2 header — reload
            # cheaply (load_fc_surface is just a small text parse).
            surface = load_fc_surface(fc_dir_p / f"fc{fc_refname}_{m}")
            out_header = _strip_header(header, strip, surface.units)
            _fits.write_image(
                workdir / f"{out_prefix}_m{m}trans.fits",
                strip.data.astype(data.dtype, copy=False),
                out_header, overwrite=True,
            )

    for i, pair in enumerate(pairs, start=1):
        _obj_raw, obj_header = _fits.read_image(workdir / pair.object_name)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objname = _sanitize_objname(str(raw_objname))

        _transform_frame(
            workdir / f"{objname}_NO{i}_sscfm.fits",
            out_prefix=f"{objname}_NO{i}_sscfm",
        )

        sky_in_path = workdir / f"{objname}_skyNO{i}_fm.fits"
        if config.flag_skyemission and sky_in_path.exists():
            _transform_frame(sky_in_path, out_prefix=f"{objname}_skyNO{i}_fm")
