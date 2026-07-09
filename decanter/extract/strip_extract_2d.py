"""Stage 9 — optional 2D strip extraction.

WARP equivalent: ``warp/Spec1Dtools.py:pyapall`` with
``format="strip"`` followed by ``resample2Dspec`` for the optional
fine-resampling path (``Warp_sci.py:576``).

Gated on ``Config.flag_extract2d`` (default False). When True, for
each order produces a 2D FITS preserving the spatial-axis dimension
within the aperture, plus an optional resampled version on a denser
slit-x grid. The 1D box-extracted spectrum from s08 is unaffected.

Output: ``{obj}_NO{i}_sscfm_m{m}_2d.fits`` per order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits as _astrofits

from decanter.calib.aperture import ApertureSet
from decanter.config import Config
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.listfile import parse as parse_listfile
from decanter.extract.psf_center import _infer_trace_x


def _sanitize_objname(raw: str) -> str:
    name = raw
    for ch in (" ", "'", "\"", "#", "/"):
        name = name.replace(ch, "_")
    return name


def run(
    config: Config,
    workdir: Path,
    listfile: Path,
    *,
    apdb_path: Path | str | None = None,
    trans_apdb_paths: dict[int, Path] | None = None,
    **_unused: Any,
) -> None:
    """Extract a 2D ``(slit_x, λ)`` strip from each rectified order.

    Args:
        config: pipeline configuration. Gated on ``flag_extract2d`` —
            returns silently when False.
        workdir: reads ``{obj}_NO{i}_sscfm_m{m}trans.fits``;
            writes ``{obj}_NO{i}_sscfm_m{m}_2d.fits``.
        listfile: WARP-style input list.
        apdb_path: aperture database for fallback trace inference.
        trans_apdb_paths: optional per-order trans-frame ap databases.

    Notes:
        The "strip" output keeps the slit dimension untouched —
        just clips the trans frame to the aperture window in slit-x.
        This is functionally a ``trans[:, ap_low_pix:ap_high_pix]``
        slice, with edge-pixel fractional weighting handled by
        rounding the bounds to integers.
    """
    if not config.flag_extract2d:
        return

    pairs = parse_listfile(listfile)
    if apdb_path is None and trans_apdb_paths is None:
        raise ValueError("s09_extract_2d requires apdb_path or trans_apdb_paths")

    psf_path = workdir / "psf_log.npz"
    psf_data = np.load(psf_path, allow_pickle=True) if psf_path.exists() else None

    cal_apset = ApertureSet.load(apdb_path) if apdb_path else None
    orders_default = (
        cal_apset.echelle_orders if cal_apset else tuple(trans_apdb_paths.keys())
    )
    orders = (
        orders_default
        if (config.reduce_full_data or not config.selected_orders)
        else tuple(m for m in orders_default if m in config.selected_orders)
    )

    for i, pair in enumerate(pairs, start=1):
        _obj_raw, obj_header = _fits.read_image(workdir / pair.object_name)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objname = _sanitize_objname(str(raw_objname))

        for j, m in enumerate(orders):
            trans_path = workdir / f"{objname}_NO{i}_sscfm_m{m}trans.fits"
            if not trans_path.exists():
                continue
            data, header = _fits.read_image(trans_path)

            if trans_apdb_paths and m in trans_apdb_paths:
                trans_set = ApertureSet.load(trans_apdb_paths[m], array_length=data.shape[0])
                ap = trans_set.apertures[m]
                trace_x = ap.trace_x
                ap_low = float(ap.entry.low)
                ap_high = float(ap.entry.high)
            else:
                trace_x = _infer_trace_x(data)
                if psf_data is not None:
                    psf_orders = list(psf_data["orders"])
                    if m in psf_orders:
                        oi = psf_orders.index(m)
                        ap_low = float(psf_data["xshift"][i - 1, oi]) - float(psf_data["fwhm"][i - 1, oi])
                        ap_high = float(psf_data["xshift"][i - 1, oi]) + float(psf_data["fwhm"][i - 1, oi])
                    else:
                        ap_low, ap_high = -5.0, 5.0
                else:
                    ap_low, ap_high = -5.0, 5.0

            # Integer-aligned window in 0-indexed columns.
            mean_trace = float(np.mean(trace_x))
            col_lo = max(0, int(np.floor(mean_trace + ap_low)) - 1)
            col_hi = min(data.shape[1], int(np.ceil(mean_trace + ap_high)))
            if col_hi <= col_lo:
                continue
            strip2d = data[:, col_lo:col_hi]

            out_header = header.copy()
            out_header["APLOW"] = (ap_low, "2D-strip aperture low (slit-x rel trace)")
            out_header["APHIGH"] = (ap_high, "2D-strip aperture high (slit-x rel trace)")
            out_header["APCOL_LO"] = (col_lo + 1, "Lower output col (1-idx) in trans frame")
            out_header["APCOL_HI"] = (col_hi, "Upper output col (1-idx) in trans frame")

            out_path = workdir / f"{objname}_NO{i}_sscfm_m{m}_2d.fits"
            _fits.write_image(out_path, strip2d.astype(np.float32), out_header, overwrite=True)
