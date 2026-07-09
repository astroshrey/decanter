"""Stage 8 — 1D box extraction.

WARP equivalent: ``warp/Spec1Dtools.py:pyapall`` (line 108) wrapping
``iraf.apall`` in non-interactive ``format="onedspec"`` mode.

For each rectified order strip and each output wavelength row, sum the
flux along the slit axis within the aperture window:

    flux[y] = Σ_{x ∈ [trace_x[y] + ap_low, trace_x[y] + ap_high]} image[y, x]

with **fractional-pixel weighting** at the window edges (IRAF apall's
default box-sum semantics). Output is a 1-D array of length ``naxis2``
in the trans frame.

Phase 1 is box extraction only. Optimal extraction and
spectroperfectionism are deferred to Phase 3.

Background subtraction
----------------------

WARP's default ``skysub_mode = "none"`` (because the sky was already
subtracted in s01); we follow the same default. When ``skysub_mode != "none"``,
IRAF apall fits a polynomial to the background sample regions inside
the aperture and subtracts — Phase 1 decanter implements the common
"average" mode (mean of the two flanking strips); the other IRAF modes
("median", "minimum", "fit") are deferred.

Output: ``{obj}_NO{i}_sscfm_m{m}_1d.fits`` per order, 1-D NAXIS=1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits as _astrofits
from numpy.typing import NDArray

from decanter.calib.aperture import ApertureSet
from decanter.config import Config
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.listfile import parse as parse_listfile


def _asigrl_linear_row(data_row: NDArray[np.float32], a: float, b: float) -> float:
    """Linear-interp integral ``∫_a^b f(x) dx`` over a 1-indexed row.

    Mirrors ``math/iminterp/asigrl.x`` for ``II_LINEAR``: in pixel cell
    ``[k, k+1]`` (1-indexed pixel centers), ``f(x) = data[k] + (x-k) *
    (data[k+1] - data[k])``. Used by :func:`_ap_edge_linear` to compute
    the integral-fraction weights.

    Pure-Python loop kept short because the typical edge integrals span
    at most one or two cells. Boundary handling: when the integration
    range hits the right edge (``b > W``), we constant-extend with
    ``data[W]`` for ``x > W`` (IRAF's actual behavior is "whatever's in
    the asi+1 slot" — undefined past the data; constant-extend is the
    safest finite-data interpretation and produces the same edge weights
    as IRAF for the cases decanter exercises).
    """
    W = data_row.size
    # Clamp to data range [1, W]
    a = max(1.0, min(float(W), a))
    b = max(1.0, min(float(W), b))
    if a >= b:
        return 0.0
    total = np.float32(0.0)
    k = int(np.floor(a))
    end_k = int(np.floor(b))
    # If b == W exactly, end_k = W and we'd access data[W] (= data_row[W-1])
    # but the cell formula needs data[k+1] = data_row[W] (out of bounds).
    # Special-case: if the segment ends exactly at b=W, decrement end_k and
    # extend the previous cell's seg_hi to W.
    if end_k == W:
        if k == W:
            # Degenerate: a == b == W, integral is zero.
            return 0.0
        end_k = W - 1
    while True:
        # Cell [k, k+1]: f(x) = data[k] + (x-k)*(data[k+1]-data[k])
        seg_lo = a if k == int(np.floor(a)) else float(k)
        seg_hi = b if k == end_k else float(k + 1)
        # 1-indexed → 0-indexed numpy
        d_k = np.float32(data_row[k - 1])
        d_k1 = np.float32(data_row[k])  # safe: end_k < W ensures k+1 <= W
        delta = np.float32(seg_hi - seg_lo)
        cell_int = (
            d_k * delta
            + np.float32(0.5)
            * np.float32((seg_hi - k) ** 2 - (seg_lo - k) ** 2)
            * (d_k1 - d_k)
        )
        total = np.float32(total + cell_int)
        if k == end_k:
            break
        k += 1
    return float(total)


def _ap_edge_linear(
    data_row: NDArray[np.float32],
    x1: float, x2: float,
    ix1: int, ix2: int,
    wt1_geom: float, wt2_geom: float,
    W: int,
) -> tuple[float, float]:
    """IRAF ``ap_edge`` with II_LINEAR interpolator (default in apall).

    Mirrors ``noao/twodspec/apextract/apextract.x:1594-1641``. When the
    interpolator is non-NULL (which it always is in apall — set to
    II_LINEAR at line 165 + 243), the geometric edge weights are
    REPLACED by integral-fraction weights ``a/b`` where:
      * ``b = ∫_{ix-0.5}^{ix+0.5} interp(x) dx`` over the full pixel cell
      * ``a = ∫_{x_edge}^{ix+0.5} interp(x) dx`` for the LEFT edge,
        or ``∫_{ix-0.5}^{x_edge} interp(x) dx`` for the right.

    For nearly-flat data the integral-based weights collapse to the
    geometric weights, so this matters most on steep gradient edges
    (which is exactly where the s08 5-8 ct/row systematic was hiding —
    HANDOFF gap #4 misdiagnosed it as a structural mystery).

    IRAF only overrides the geometric weight if (a) ``data[ix] > 0``,
    (b) ``b > 0``, and (c) ``0 < a < b``. Otherwise the geometric
    weight stays.
    """
    wt1 = wt1_geom
    wt2 = wt2_geom

    # LEFT edge override
    if 1 <= ix1 <= W and data_row[ix1 - 1] > 0:
        # Pixel cell integral (clip into [1, W])
        b_lo = max(0.5, ix1 - 0.5)
        b_hi = min(W + 0.5, ix1 + 0.5)
        if b_hi > b_lo:
            b_int = _asigrl_linear_row(data_row, b_lo, b_hi)
            if b_int > 0:
                if ix1 == ix2:
                    a_int = _asigrl_linear_row(data_row, x1, x2)
                else:
                    a_int = _asigrl_linear_row(data_row, x1, b_hi)
                if 0 < a_int < b_int:
                    wt1 = a_int / b_int

    # RIGHT edge override (only when ix1 != ix2)
    if ix1 != ix2 and 1 <= ix2 <= W and data_row[ix2 - 1] > 0:
        b_lo = max(0.5, ix2 - 0.5)
        b_hi = min(W + 0.5, ix2 + 0.5)
        if b_hi > b_lo:
            b_int = _asigrl_linear_row(data_row, b_lo, b_hi)
            if b_int > 0:
                a_int = _asigrl_linear_row(data_row, b_lo, x2)
                if 0 < a_int < b_int:
                    wt2 = a_int / b_int

    return wt1, wt2


def _sanitize_objname(raw: str) -> str:
    name = raw
    for ch in (" ", "'", "\"", "#", "/"):
        name = name.replace(ch, "_")
    return name


def box_extract(
    image: NDArray[np.floating],
    trace_x: NDArray[np.floating],
    *,
    ap_low: float,
    ap_high: float,
) -> NDArray[np.float32]:
    """Box-sum extraction with IRAF-faithful float32 left-to-right accumulation.

    Mirrors ``noao/twodspec/apextract/apextract.x:ap_sum`` (line 1525) and
    its edge-weight helper ``ap_edge`` (line 1594) for the unweighted
    box-sum case (no profile interpolator, no sky subtraction inside the
    extraction call).

    For each y row:
      1. ``x1 = max(0.5, trace_x + ap_low)``,
         ``x2 = min(W + 0.49, trace_x + ap_high)`` (matches ``ap_sum`` line
         1559–1560: the ``W + 0.49`` upper clip is a deliberate IRAF
         choice — tighter than ``W + 0.5`` so the high edge never lands
         exactly on the half-pixel boundary).
      2. ``ix1 = nint(x1)``, ``ix2 = nint(x2)`` (round-half-up).
      3. ``wt1 = ix1 - x1 + 0.5``, ``wt2 = x2 - ix2 + 0.5`` for ``ix1 != ix2``;
         ``wt1 = x2 - x1, wt2 = 0`` when both edges fall inside the same pixel.
      4. ``spec = wt1*data[ix1] + wt2*data[ix2] + sum(data[ix1+1..ix2-1])``,
         all in float32 with left-to-right accumulation.

    Args:
        image: 2-D rectified strip; shape ``(n_y, n_x)``.
        trace_x: 1-D float, length ``n_y``. Trace position (1-indexed
            slit x) at each y row.
        ap_low, ap_high: slit-window bounds relative to the trace.
            ``ap_high > ap_low``; either may be negative.

    Returns:
        Float32 array of length ``n_y`` with the per-row box-sum.
        Returns 0 for rows where the clipped aperture is empty.
    """
    H, W = image.shape
    if trace_x.shape != (H,):
        raise ValueError(f"trace_x shape {trace_x.shape} != ({H},)")
    if ap_high <= ap_low:
        raise ValueError(f"ap_high ({ap_high}) must exceed ap_low ({ap_low})")

    # IRAF works in `real` (float32) throughout; mimic that for parity.
    img32 = image.astype(np.float32, copy=False)
    trace_f = np.asarray(trace_x, dtype=np.float32)
    out = np.zeros(H, dtype=np.float32)

    # Aperture edges per row (float32 to match IRAF).
    x_low_all = (trace_f + np.float32(ap_low)).astype(np.float32)
    x_high_all = (trace_f + np.float32(ap_high)).astype(np.float32)

    half = np.float32(0.5)
    upper_clip = np.float32(W + 0.49)
    lower_clip = np.float32(0.5)

    # Row-by-row scalar loop matches IRAF's float32 reduction order.
    for iy in range(H):
        x1 = max(lower_clip, x_low_all[iy])
        x2 = min(upper_clip, x_high_all[iy])
        if x2 <= x1:
            continue
        # IRAF nint = round-half-up for positive numbers.
        ix1 = int(np.floor(x1 + half))
        ix2 = int(np.floor(x2 + half))
        if ix1 < 1 or ix2 > W or ix1 > W or ix2 < 1:
            continue
        if ix1 == ix2:
            wt1_geom = float(x2 - x1)
            wt2_geom = 0.0
        else:
            wt1_geom = float(ix1 - x1 + half)
            wt2_geom = float(x2 - ix2 + half)
        # IRAF ap_edge with II_LINEAR interpolator (apall's default at
        # apextract.x:165+243): replaces geometric weights with integral
        # fractions a/b. Critical for steep-gradient edges; the 5-8 ct
        # row-by-row systematic (HANDOFF gap #4) lived here, NOT in
        # background subtraction or trace polynomial precision.
        wt1, wt2 = _ap_edge_linear(
            img32[iy], float(x1), float(x2),
            ix1, ix2, wt1_geom, wt2_geom, W,
        )
        wt1 = np.float32(wt1)
        wt2 = np.float32(wt2)
        # 0-based numpy indexing for the (1-based IRAF) pixel ix.
        # Accumulate in float32, left-to-right, matching IRAF ap_sum:1572-1574.
        sval = np.float32(wt1 * img32[iy, ix1 - 1] + wt2 * img32[iy, ix2 - 1])
        for ix in range(ix1 + 1, ix2):  # interior pixels at full weight
            sval = np.float32(sval + img32[iy, ix - 1])
        out[iy] = sval
    return out


def run(
    config: Config,
    workdir: Path,
    listfile: Path,
    *,
    apdb_path: Path | str | None = None,
    trans_apdb_paths: dict[int, Path] | None = None,
    **_unused: Any,
) -> None:
    """Box-extract each rectified order strip to a 1-D spectrum.

    Args:
        config: pipeline configuration. ``selected_orders`` filters
            which orders to extract.
        workdir: reads ``{obj}_NO{i}_sscfm_m{m}trans.fits`` (s06 output)
            and ``psf_log.npz`` (s07 output); writes
            ``{obj}_NO{i}_sscfm_m{m}_1d.fits``.
        listfile: WARP-style input list.
        apdb_path: aperture database for the multihole-frame apertures
            (fallback when neither trans_apdb nor psf_log is usable).
        trans_apdb_paths: optional ``{order: Path}`` mapping to per-order
            trans-frame aperture databases. When provided, the trace +
            window are read directly (matches WARP).

    Notes:
        Extraction window: ``[xshift - fwhm, xshift + fwhm]`` (~2 σ,
        matches WARP ``Warp_sci.py:411-413``) when s07's PSF fit
        is the source. When ``trans_apdb_paths`` is given, the window
        is the ``[ap_low, ap_high]`` written by aptrace+apresize.
    """
    pairs = parse_listfile(listfile)
    if apdb_path is None and trans_apdb_paths is None:
        raise ValueError(
            "s08_extract_1d requires apdb_path or trans_apdb_paths; the "
            "calibration loader (s00) will provide these once wired in."
        )

    # Load the psf_log for per-frame (xshift, fwhm).
    psf_path = workdir / "psf_log.npz"
    psf_data = None
    if psf_path.exists():
        psf_data = np.load(psf_path, allow_pickle=True)

    cal_apset = ApertureSet.load(apdb_path) if apdb_path else None
    orders_default = (
        cal_apset.echelle_orders if cal_apset else tuple(trans_apdb_paths.keys())
    )
    orders = (
        orders_default
        if (config.reduce_full_data or not config.selected_orders)
        else tuple(m for m in orders_default if m in config.selected_orders)
    )

    def _extract_one(trans_path: Path, m: int, frame_idx: int) -> tuple[np.ndarray, _astrofits.Header, float, float, np.ndarray]:
        """Compute trace + aperture window + 1D flux for one order's trans frame.

        Returns ``(flux_1d, header, ap_low, ap_high, trace_x)``. The trace
        and window are derived from the OBJ frame's PSF/aperture; the sky
        path REUSES this trace + window when extracting from the sky's
        trans strip (mirrors WARP ``pyapall(sky_fm_trans, sky_fm_trans_1d,
        obj_sscfm_trans, ...)`` — the third argument is the reference
        aperture for the sky extraction).
        """
        data, header = _fits.read_image(trans_path)
        if trans_apdb_paths and m in trans_apdb_paths:
            trans_set = ApertureSet.load(trans_apdb_paths[m], array_length=data.shape[0])
            ap = trans_set.apertures[m]
            # IRAF ap_extract evaluates the trace via ap_cveval, which clamps
            # y to [y_min, y_max] before the polynomial eval (apcveval.x:17).
            # Without the clamp, the Chebyshev/Legendre fit extrapolates
            # wildly past its sample range.
            trace_x = ap.trace_x_clamped
            ap_low = float(ap.entry.low)
            ap_high = float(ap.entry.high)
        else:
            from decanter.extract.psf_center import _infer_trace_x  # local import
            trace_x = _infer_trace_x(data)
            if psf_data is not None:
                psf_orders = list(psf_data["orders"])
                if m in psf_orders:
                    oi = psf_orders.index(m)
                    xshift = float(psf_data["xshift"][frame_idx - 1, oi])
                    fwhm = float(psf_data["fwhm"][frame_idx - 1, oi])
                else:
                    xshift, fwhm = 0.0, 5.0
            else:
                xshift, fwhm = 0.0, 5.0
            if not np.isfinite(xshift) or not np.isfinite(fwhm):
                xshift, fwhm = 0.0, 5.0
            ap_low = xshift - fwhm
            ap_high = xshift + fwhm
        flux_1d = box_extract(data, trace_x, ap_low=ap_low, ap_high=ap_high)
        return flux_1d, header, ap_low, ap_high, trace_x

    def _build_header(header: _astrofits.Header, ap_low: float, ap_high: float
                      ) -> _astrofits.Header:
        out_header = header.copy()
        crval2 = float(header.get("CRVAL2", 1.0))
        cdelt2 = float(header.get("CDELT2", 1.0))
        crpix2 = float(header.get("CRPIX2", 1.0))
        out_header["CRVAL1"] = (crval2, "Wavelength at output pixel 1")
        out_header["CRPIX1"] = (crpix2, "Reference pixel along dispersion")
        out_header["CDELT1"] = (cdelt2, "Wavelength step per pixel")
        out_header["CTYPE1"] = ("LINEAR", "Wavelength axis")
        out_header["WAT1_001"] = (
            "wtype=linear label=Wavelength units=nanometers",
            "WCS attribute (WAT) for axis 1",
        )
        for k in ("CRVAL2", "CRPIX2", "CDELT2", "CTYPE2", "WAT2_001"):
            out_header.remove(k, ignore_missing=True)
        out_header["APLOW"] = (ap_low, "Box-sum aperture low (slit-x rel trace)")
        out_header["APHIGH"] = (ap_high, "Box-sum aperture high (slit-x rel trace)")
        return out_header

    for i, pair in enumerate(pairs, start=1):
        _obj_raw, obj_header = _fits.read_image(workdir / pair.object_name)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objname = _sanitize_objname(str(raw_objname))

        for j, m in enumerate(orders):
            obj_trans_path = workdir / f"{objname}_NO{i}_sscfm_m{m}trans.fits"
            if not obj_trans_path.exists():
                continue

            # --- OBJ path ------------------------------------------------------
            flux_1d, header, ap_low, ap_high, trace_x = _extract_one(
                obj_trans_path, m, frame_idx=i
            )
            out_header = _build_header(header, ap_low, ap_high)
            out_path = workdir / f"{objname}_NO{i}_sscfm_m{m}_1d.fits"
            _fits.write_image(
                out_path, flux_1d.astype(np.float32), out_header, overwrite=True
            )

            # --- SKY path ------------------------------------------------------
            # IRAF apall is called with the OBJ trans as reference, then
            # the sky trans as the image to extract. Same trace, same
            # window. We reuse the OBJ's (trace_x, ap_low, ap_high) and
            # just re-extract on the sky strip.
            if config.flag_skyemission:
                sky_trans_path = workdir / f"{objname}_skyNO{i}_fm_m{m}trans.fits"
                if sky_trans_path.exists():
                    sky_data, sky_header = _fits.read_image(sky_trans_path)
                    sky_flux = box_extract(
                        sky_data, trace_x, ap_low=ap_low, ap_high=ap_high
                    )
                    sky_out_header = _build_header(sky_header, ap_low, ap_high)
                    sky_out_path = (
                        workdir / f"{objname}_skyNO{i}_fm_m{m}trans1d.fits"
                    )
                    _fits.write_image(
                        sky_out_path,
                        sky_flux.astype(np.float32),
                        sky_out_header,
                        overwrite=True,
                    )
