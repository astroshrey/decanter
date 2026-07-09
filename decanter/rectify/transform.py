"""Stage 6 — 2D→1D transform & cut per echelle order.

WARP equivalent: ``warp/cutransform.py:cutransform`` (line 94).
For each order ``m`` in the aperture set:

  1. Cut the input frame to columns ``[xmin, xmax]`` where
     ``xmin = trace_x[0] - 100`` and ``xmax = trace_x[-1] + 100``
     (clipped to ``[1, 2048]``).
  2. Apply the per-order ``fitcoords`` surface (``database/fc<ref>_<m>``)
     to resample onto a uniform-wavelength y-grid with step ``dy`` =
     comp file ``CDELT1``.
  3. Write the rectified strip to ``{frame}_m{order}trans.fits`` with
     wavelength-WCS keywords (``CRVAL2``, ``CDELT2``, etc.).

Output suffix: ``_m###trans``. WARP uses IRAF
``transform(..., interptype=spline3)``; we use
``scipy.ndimage.map_coordinates(order=3)`` (cubic B-spline). The two
differ slightly at high curvature; per-pixel parity is checked
empirically — see ``EXECUTION_LOG.md`` parity entry for this stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits as _astrofits
from numpy.typing import NDArray

from decanter.calib.aperture import ApertureSet
from decanter.calib.transform import RectifiedStrip, load_fc_surface, rectify_order
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


def rectify_orders(
    data: NDArray,
    apset: ApertureSet,
    *,
    fc_dir: Path,
    fc_refname: str,
    dy: float,
    orders: Iterable[int] | None = None,
) -> dict[int, RectifiedStrip]:
    """Pure task function: rectify every echelle order of a 2D frame.

    For each order in ``orders`` (defaults to ``apset.echelle_orders``),
    cuts ``data`` to ``[trace_x[0] - 100, trace_x[-1] + 100]`` and applies
    the matching ``fc<refname>_<m>`` surface via :func:`rectify_order`.

    Returns:
        ``{order: RectifiedStrip}`` — one entry per requested order,
        each carrying ``data``, ``xmin``, ``xmax``, ``lambda_min``,
        ``dy``, ``units`` (from the surface).
    """
    fc_dir_p = Path(fc_dir)
    requested = tuple(orders) if orders is not None else tuple(apset.echelle_orders)
    out: dict[int, RectifiedStrip] = {}
    for m in requested:
        ap = apset.apertures[m]
        xmin = max(1, int(ap.trace_x[0] - _XMARGIN))
        xmax = min(apset.array_length, int(ap.trace_x[-1] + _XMARGIN))
        surface = load_fc_surface(fc_dir_p / f"fc{fc_refname}_{m}")
        out[m] = rectify_order(
            data, surface, xmin=xmin, xmax=xmax, dy=float(dy),
            array_length=apset.array_length,
        )
    return out


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
