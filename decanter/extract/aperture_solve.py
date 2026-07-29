"""WARP-independent trace + aperture for 1D extraction.

This is a faithful port of what WARP does when no per-frame trace database has
been handed to it — i.e. the normal science path. WARP does **not** re-trace
the star on each frame; it reuses the rectified-strip **reference trace** from
the multihole calibration aperture (``ap{aptrans}_{m}trans``) and measures only
a per-frame *shift* of the star relative to that reference.

Algorithm (``warp/centersearch_fortrans.py`` + ``Warp_sci.py`` aperture block):

  1. For each order, take the multihole trans reference trace ``apx(y)`` and its
     wide slit window ``[apLow, apHigh]``.
  2. ``centersearch_fortrans`` stacks the slit profile over that window and fits
     a Gaussian, yielding the star's ``xshift`` from the reference trace and its
     ``fwhm``. This is decanter's :func:`decanter.extract.psf_center.measure_one_strip`.
  3. The extraction aperture is **constant across all orders of the frame**
     (``Warp_sci.py``: "setting the aperture range as 2 sigma"):
     ``ap_low  = median(xshift) - median(fwhm)``,
     ``ap_high = median(xshift) + median(fwhm)``,
     applied relative to each order's reference trace.

On real WINERED data this reproduces WARP's per-frame aperture to a fraction of
a pixel; the extracted 1D flux tracks the WARP-locked extraction to a median
relative difference of ~1e-4 to 1e-3 across the archive, in all three modes
(HIRES-Y, HIRES-J, WIDE).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from decanter.calib.aperture import ApertureSet
from decanter.calib.transform import RectifiedStrip
from decanter.extract.psf_center import measure_one_strip

# WARP's ``centersearch_fortrans`` cuts the dispersion axis at a hardcoded
# 500..(2048-500) using the trans frame's pixel-based y-WCS (CRVAL2 ~ 1). A
# fixed 1-px linear WCS reproduces that same pixel window on decanter's strips.
_STRIP_WCS = {"CRVAL2": 1.0, "CDELT2": 1.0, "CRPIX2": 1.0}


@dataclass(frozen=True, slots=True)
class FrameAperture:
    """Per-frame extraction geometry solved from the reference apertures."""

    traces: dict[int, NDArray[np.float64]]   # {order: reference trace x(y)}
    ap_low: float                            # frame-constant, relative to trace
    ap_high: float


def _load_reference(ref_path: Path, order: int, n_rows: int):
    """Return ``(trace_x, ap_low, ap_high)`` from a multihole trans reference."""
    ap = ApertureSet.load(ref_path, array_length=n_rows).apertures[order]
    return np.asarray(ap.trace_x, dtype=np.float64), float(ap.entry.low), float(ap.entry.high)


def solve_frame_aperture(
    strips: dict[int, RectifiedStrip],
    ref_trans_apdbs: dict[int, Path],
    *,
    abba: bool = False,
) -> FrameAperture:
    """Solve the reference trace per order and one frame-constant aperture.

    Args:
        strips: ``{order: RectifiedStrip}`` from :func:`rectify_orders`.
        ref_trans_apdbs: ``{order: Path}`` multihole trans reference DBs.
        abba: ABBA nod flag, forwarded to the center search (rejects the
            opposite nod position from the profile, as WARP does).

    Returns:
        A :class:`FrameAperture` with the reference traces and the shared
        ``ap_low`` / ``ap_high`` window.
    """
    traces: dict[int, NDArray[np.float64]] = {}
    shifts: list[float] = []
    widths: list[float] = []
    for m, strip in strips.items():
        ref = ref_trans_apdbs.get(m)
        if ref is None:
            continue
        arr = strip.data
        trace_x, ref_low, ref_high = _load_reference(ref, m, arr.shape[0])
        traces[m] = trace_x
        fit = measure_one_strip(
            arr, _STRIP_WCS, ap_low=ref_low, ap_high=ref_high,
            trace_x=trace_x, abba=abba,
        )
        if np.isfinite(fit.xshift) and np.isfinite(fit.fwhm):
            shifts.append(fit.xshift)
            widths.append(fit.fwhm)

    if not shifts:
        raise ValueError(
            "center search found no usable stellar profile in any order; "
            "cannot solve an aperture (are the strips empty?)"
        )
    med_shift = float(np.median(shifts))
    med_fwhm = float(np.median(widths))
    return FrameAperture(
        traces=traces, ap_low=med_shift - med_fwhm, ap_high=med_shift + med_fwhm,
    )
