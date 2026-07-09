"""Stage 7 — PSF center & FWHM measurement on rectified strips.

WARP equivalent: ``warp/centersearch_fortrans.py:centersearch_fortrans``.
For each frame × each echelle order: fit a Gaussian to the stacked slit
profile of the rectified ``_m###trans.fits`` strip → recover
``(xshift, fwhm)``. These two numbers feed s08's box-extraction
aperture as ``[xshift - fwhm, xshift + fwhm]`` (~2-σ window).

Output: a :class:`PsfTable` written to disk as a small ``.npz`` ledger
``psf_log.npz`` containing one ``xshift`` and ``fwhm`` per (frame,
order). No FITS suffix change.

Trace inference: WARP runs ``iraf.aptrace`` on each trans frame to get
the per-frame trace ``trace_x(y)`` inside the rectified strip; decanter
doesn't yet have an aptrace port. Two modes are supported:

  * **From-disk:** if the caller passes ``trans_apdb_paths`` (a mapping
    ``{order: Path}``), each entry is parsed via
    :class:`decanter.calib.aperture.ApertureSet`.
  * **Auto:** otherwise, the trace is inferred by ``argmax`` per
    y-row followed by a 5th-order Legendre polyfit. This is a
    simplified aptrace good enough for Phase 1 — it converges
    cleanly on the rectified strip because slit tilt is already removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial import legendre
from numpy.typing import NDArray

from decanter.calib.aperture import ApertureSet
from decanter.config import Config
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.listfile import parse as parse_listfile
from decanter.utils.psf import PsfFit, fit_slit_gaussian, stacked_slit_profile


@dataclass(frozen=True, slots=True)
class PsfTable:
    """Per-frame, per-order ``(xshift, fwhm)`` ledger."""

    frame_ids: list[str]                                      # length n_frames
    orders: tuple[int, ...]                                   # length n_orders
    xshift: NDArray[np.float64]                               # (n_frames, n_orders)
    fwhm: NDArray[np.float64]                                 # (n_frames, n_orders)

    def median_xshift(self) -> NDArray[np.float64]:
        """Per-frame median ``xshift`` across orders (used by s08)."""
        return np.nanmedian(self.xshift, axis=1)

    def median_fwhm(self) -> NDArray[np.float64]:
        """Per-frame median ``fwhm`` across orders (used by s08)."""
        return np.nanmedian(self.fwhm, axis=1)

    def save_npz(self, path: Path) -> None:
        np.savez(
            path,
            frame_ids=np.array(self.frame_ids, dtype=object),
            orders=np.asarray(self.orders),
            xshift=self.xshift,
            fwhm=self.fwhm,
        )


def _sanitize_objname(raw: str) -> str:
    name = raw
    for ch in (" ", "'", "\"", "#", "/"):
        name = name.replace(ch, "_")
    return name


def _is_abba(nodpos: str) -> bool:
    return "O" not in nodpos


def _infer_trace_x(image: NDArray[np.floating], *, deg: int = 5) -> NDArray[np.float64]:
    """Estimate trace x(y) in a rectified strip.

    Rectification removes slit tilt, so the trace is approximately
    vertical in the trans frame — a constant column. We estimate that
    column by summing ``|data|`` across high-signal rows and taking the
    argmax, which robustly picks out the spectrum's slit position.

    The ``deg`` argument is kept for signature stability with earlier
    versions that did a per-row argmax + polyfit; we no longer use it
    because the polyfit was wildly unstable at the wavelength-edge
    rows where spline artifacts dominate per-row argmax.
    """
    del deg  # see docstring note
    H, W = image.shape
    abs_img = np.abs(image)
    # Restrict to the dispersion-direction middle 60% to avoid both
    # extrapolated rows and edge spline artifacts.
    lo, hi = int(0.2 * H), int(0.8 * H)
    if hi <= lo:
        lo, hi = 0, H
    abs_band = abs_img[lo:hi]

    # Identify signal rows (median |data| > 30% of the brightest row's
    # median |data|) so noise rows don't dilute the column stack.
    row_med = np.median(abs_band, axis=1)
    if row_med.max() == 0:
        return np.full(H, W / 2.0)
    keep = row_med > 0.3 * row_med.max()
    if keep.sum() < 5:
        keep = row_med > 0  # fall back to all non-empty rows

    # Stack along y to get a single per-column total signal.
    col_signal = abs_band[keep].sum(axis=0)
    # Restrict argmax to the interior 80% to avoid edge-spline artifacts.
    x_lo, x_hi = int(0.1 * W), int(0.9 * W)
    if x_hi <= x_lo:
        x_lo, x_hi = 0, W
    best_x = int(np.argmax(col_signal[x_lo:x_hi])) + x_lo
    # Refine via centroid around the best column to get sub-pixel accuracy.
    half_window = 3
    a = max(0, best_x - half_window)
    b = min(W, best_x + half_window + 1)
    weights = col_signal[a:b]
    positions = np.arange(a, b, dtype=np.float64)
    if weights.sum() > 0:
        centroid = float((weights * positions).sum() / weights.sum())
    else:
        centroid = float(best_x)
    return np.full(H, centroid + 1.0)  # +1 → 1-indexed


def measure_one_strip(
    image: NDArray[np.floating],
    header: dict,
    *,
    ap_low: float,
    ap_high: float,
    trace_x: NDArray[np.floating] | None = None,
    abba: bool = False,
    lowlim_wave: float = 500.0,
    upplim_wave: float | None = None,
) -> PsfFit:
    """Run the full PSF-fit pipeline on one rectified order strip.

    Args:
        image: 2-D rectified strip.
        header: FITS header with ``CRVAL2``, ``CDELT2``, ``CRPIX2`` (for
            converting the WARP-style ``lowlim`` in wavelength units
            back to row indices).
        ap_low, ap_high: slit-window bounds (relative to trace center).
        trace_x: per-row trace position. If ``None``, inferred via
            argmax+legfit.
        abba: ABBA nod flag.
        lowlim_wave, upplim_wave: WARP's wavelength-unit edge cuts.
            Defaults to ``500, naxis2 - 500`` in wavelength units.

    Returns:
        :class:`PsfFit` with the converged Gaussian parameters.
    """
    crval2 = float(header.get("CRVAL2", 1.0))
    cdelt2 = float(header.get("CDELT2", 1.0))
    crpix2 = float(header.get("CRPIX2", 1.0))
    H = image.shape[0]
    if upplim_wave is None:
        upplim_wave = (H - 500) * cdelt2 + crval2  # approximate
    # Convert wavelength bounds to row indices (0-indexed).
    lowlim_y = int((lowlim_wave - crval2) / cdelt2 + crpix2 - 1) - 1
    upplim_y = int((upplim_wave - crval2) / cdelt2 + crpix2 - 1)
    lowlim_y = max(0, lowlim_y)
    upplim_y = min(H, upplim_y)
    if upplim_y <= lowlim_y + 10:
        # Fall back to a generic central window.
        lowlim_y = int(0.1 * H)
        upplim_y = int(0.9 * H)

    if trace_x is None:
        trace_x = _infer_trace_x(image)

    med_x, med_y, n_rows = stacked_slit_profile(
        image, trace_x,
        ap_low=ap_low, ap_high=ap_high,
        lowlim_y=lowlim_y, upplim_y=upplim_y, abba=abba,
    )
    fit = fit_slit_gaussian(med_x, med_y, abba=abba)
    return PsfFit(
        xshift=fit.xshift, fwhm=fit.fwhm, peak=fit.peak, offset=fit.offset,
        n_rows_used=n_rows, success=fit.success,
    )


def run(
    config: Config,
    workdir: Path,
    listfile: Path,
    *,
    apdb_path: Path | str | None = None,
    trans_apdb_paths: dict[int, Path] | None = None,
    **_unused: Any,
) -> None:
    """Measure ``(xshift, fwhm)`` per (frame, order) and stash in ``psf_log.npz``.

    Args:
        config: pipeline configuration. Gated on ``flag_manual_aperture``
            — if True, this stage is a no-op (s08 uses manual aperture
            bounds from config).
        workdir: reads ``{obj}_NO{i}_sscfm_m{m}trans.fits`` (s06 output);
            writes ``psf_log.npz``.
        listfile: WARP-style input list.
        apdb_path: aperture database used to read the slit bounds
            (``ap_low`` / ``ap_high``) per order. Required.
        trans_apdb_paths: optional ``{order: Path}`` mapping to per-order
            trans-frame aperture databases. If supplied, the trace
            ``trace_x(y)`` and slit bounds are read from those files
            (matches WARP behavior). Otherwise the trace is inferred
            by ``argmax + legfit``.
    """
    if config.flag_manual_aperture:
        return
    if apdb_path is None:
        raise ValueError(
            "s07_psf_center requires apdb_path; the calibration loader "
            "(s00) will provide this automatically once wired in."
        )

    apset = ApertureSet.load(apdb_path)
    orders = (
        apset.echelle_orders
        if (config.reduce_full_data or not config.selected_orders)
        else tuple(m for m in apset.echelle_orders if m in config.selected_orders)
    )
    pairs = parse_listfile(listfile)

    n_frames = len(pairs)
    n_orders = len(orders)
    xshift = np.full((n_frames, n_orders), np.nan, dtype=np.float64)
    fwhm = np.full((n_frames, n_orders), np.nan, dtype=np.float64)
    frame_ids: list[str] = []

    for i, pair in enumerate(pairs, start=1):
        _obj_raw, obj_header = _fits.read_image(workdir / pair.object_name)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objname = _sanitize_objname(str(raw_objname))
        frame_ids.append(f"{objname}_NO{i}")

        nodpos = str(headers.get(obj_header, "NODPOS", default="O"))
        abba = _is_abba(nodpos)

        for j, m in enumerate(orders):
            trans_path = workdir / f"{objname}_NO{i}_sscfm_m{m}trans.fits"
            if not trans_path.exists():
                continue
            data, header = _fits.read_image(trans_path)
            # Slit-window bounds: from per-order trans ap db if provided;
            # otherwise from the calibration aperture set.
            trace_x = None
            if trans_apdb_paths and m in trans_apdb_paths:
                trans_set = ApertureSet.load(trans_apdb_paths[m], array_length=data.shape[0])
                if m in trans_set.apertures:
                    trans_ap = trans_set.apertures[m]
                    ap_low = float(trans_ap.entry.low)
                    ap_high = float(trans_ap.entry.high)
                    trace_x = trans_ap.trace_x
                else:
                    ap = apset.apertures[m]
                    ap_low, ap_high = float(ap.entry.low), float(ap.entry.high)
            else:
                ap = apset.apertures[m]
                ap_low, ap_high = float(ap.entry.low), float(ap.entry.high)

            fit = measure_one_strip(
                data, dict(header),
                ap_low=ap_low, ap_high=ap_high, trace_x=trace_x, abba=abba,
            )
            xshift[i - 1, j] = fit.xshift
            fwhm[i - 1, j] = fit.fwhm

    table = PsfTable(frame_ids=frame_ids, orders=orders, xshift=xshift, fwhm=fwhm)
    table.save_npz(workdir / "psf_log.npz")
