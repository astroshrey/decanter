"""Aperture tracing from the multihole (pinhole) calibration frame.

WARP equivalent: ``warp/aperture_auto_trace.py:auto_aptrace``. This is the
first stage of building a calibration set from scratch (the ``Warp_calib.py``
side): given the multihole pinhole exposure, recover the per-echelle-order
trace ``x(y)`` on the detector — the same trace decanter otherwise reads from
WARP's ``apmultihole`` aperture database.

Algorithm:

  1. Average the multihole frame over a few rows at mid-detector and find the
     bright multihole peaks (a flux-weighted centroid per peak).
  2. Match the detected peaks to a bundled reference hole pattern
     (``reference_data/<mode>.npz``: ``centers`` / ``orders`` / ``traceid``,
     the fixed per-mode multihole layout) by the integer column shift that
     minimises the summed squared nearest-neighbour separation. This assigns an
     echelle order to each peak and picks the ``traceid==1`` hole as the
     order's aperture reference.
  3. Trace each order's reference hole up and down the detector in fixed row
     steps, re-centroiding within a small window at each step.
  4. Fit a Chebyshev polynomial ``x(y)`` to the traced points with iterative
     sigma clipping (matching WARP's ``aperture.adjustParameter``).

Reference templates are bundled from WARP's ``reference/<mode>/`` and are fixed
per instrument mode (like a line list), not per-run calibration data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev as _cheb
from numpy.typing import NDArray
from scipy.signal import argrelmax

from decanter.calib.aperture import Aperture, ApertureSet
from decanter.io import fits as _fits
from decanter.io.apdb import FUNCTION_TYPE_CHEBYSHEV, ApertureEntry

_REF_DIR = Path(__file__).parent / "reference_data"
_REF_NPZ = {"WIDE": "wide_20171202.npz",
            "HIRES-Y": "hiresy_20220714.npz",
            "HIRES-J": "hiresj_20220714_2.npz"}

_ROW_STEP = 10          # trace step along dispersion (px)  [auto_aptrace]
_SEARCH_HALF = 30       # per-step re-centroid window half-width (px)
_EDGE_LO, _EDGE_HI = 40, 2000   # usable detector-row band for peak finding
_MIN_PEAK = 100.0       # min counts for a real multihole peak
_SHIFT_MAX = 30         # reference-match column shift search (px)
_CHEB_ORDER = 5         # trace polynomial terms (matches WARP apmultihole)
_CLIP_SIG, _CLIP_ITERS = 5.0, 3


@dataclass(frozen=True, slots=True)
class TracedApertures:
    """Per-order detector-frame trace recovered from the multihole frame."""

    traces: dict[int, NDArray[np.float64]]   # {order: x(y), length n_rows, 1-indexed}
    centers: dict[int, float]                # {order: aperture center x at mid-row}
    coeffs: dict[int, NDArray[np.float64]]   # {order: Chebyshev trace coefficients}
    y_min: float
    y_max: float
    n_rows: int

    def to_aperture_set(self, low: float, high: float) -> ApertureSet:
        """Build an :class:`ApertureSet` with the given aperture window.

        Lets the traced apertures feed the same code paths that consume a
        WARP aperture DB (apscatter, box/optimal extraction, masks).
        """
        aps = {}
        for m, coef in self.coeffs.items():
            entry = ApertureEntry(
                order=m, center_x=0.0, center_y=self.n_rows / 2.0,
                low=float(low), high=float(high),
                function_type=FUNCTION_TYPE_CHEBYSHEV, poly_order=coef.size,
                y_min=self.y_min, y_max=self.y_max,
                coefficients=tuple(float(c) for c in coef),
            )
            aps[m] = Aperture(entry=entry, array_length=self.n_rows)
        return ApertureSet(apertures=aps, array_length=self.n_rows)


def _centroid(x: NDArray, y: NDArray, c_index: int, width: int = 2) -> float:
    """Flux-weighted centroid around ``c_index`` (WARP peak_single_line_trace)."""
    lo = max(c_index - 2 * width, 0)
    hi = min(c_index + 2 * width + 1, len(x) - 1)
    sl = slice(lo, hi)
    w = np.abs(y[sl])
    return float(np.sum(x[sl] * w) / np.sum(w))


def _cheb_fit(px: NDArray, py: NDArray, ymin: float, ymax: float, n_rows: int
              ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sigma-clipped Chebyshev trace fit.

    Returns ``(trace_x, coeffs)`` where ``trace_x`` is x on rows 1..n_rows and
    ``coeffs`` are the Chebyshev coefficients on ``y_norm = (2y-ymin-ymax)/
    (ymax-ymin)`` (so ``chebval(y_norm, coeffs)`` reproduces the trace).
    """
    keep = np.ones(px.size, bool)
    normy = (2 * py - ymin - ymax) / (ymax - ymin)
    coef = _cheb.chebfit(normy, px, _CHEB_ORDER - 1)
    for _ in range(_CLIP_ITERS):
        coef = _cheb.chebfit(normy[keep], px[keep], _CHEB_ORDER - 1)
        resid = px - _cheb.chebval(normy, coef)
        sd = np.std(resid[keep])
        if sd == 0:
            break
        keep = np.abs(resid - np.mean(resid[keep])) <= _CLIP_SIG * sd
    rows = np.arange(1, n_rows + 1, dtype=float)
    rnorm = (2 * rows - ymin - ymax) / (ymax - ymin)
    return _cheb.chebval(rnorm, coef), coef


def trace_apertures(multihole_frame: Path | str | NDArray, mode: str,
                    *, orders: tuple[int, ...] | None = None) -> TracedApertures:
    """Recover per-order detector traces from a multihole calibration frame.

    Args:
        multihole_frame: the multihole/pinhole exposure (path or 2-D array).
            Several exposures should be averaged beforehand for SNR.
        mode: instrument mode (``"HIRES-Y"``, ``"HIRES-J"``, ``"WIDE"``) —
            selects the bundled reference hole pattern.
        orders: restrict to these echelle orders (default: all in the template).

    Returns:
        :class:`TracedApertures`.
    """
    if isinstance(multihole_frame, (str, Path)):
        data, _ = _fits.read_image(Path(multihole_frame))
    else:
        data = np.asarray(multihole_frame)
    data = np.asarray(data, float)
    n_rows, n_cols = data.shape

    ref = np.load(_REF_DIR / _REF_NPZ[mode], allow_pickle=True)
    ref_centers = np.asarray(ref["centers"], float)
    ref_orders = np.asarray(ref["orders"], int)
    ref_traceid = np.asarray(ref["traceid"], int)

    apx = np.arange(1, n_cols + 1, dtype=float)     # 1-indexed detector columns
    mid = n_rows // 2 - 1
    spd = np.mean(data[mid - 3:mid + 2], axis=0)

    # detected peaks at mid-detector
    peaks_idx = [i for i in argrelmax(spd, order=5)[0]
                 if spd[i] > _MIN_PEAK and _EDGE_LO < i < _EDGE_HI]
    det_centers = np.array([_centroid(apx, spd, i) for i in peaks_idx])
    if det_centers.size == 0:
        raise ValueError("no multihole peaks found in the trace frame")

    # match detected peaks to the reference pattern by integer column shift
    shifts = np.arange(-_SHIFT_MAX, _SHIFT_MAX + 1)
    chi2 = [np.sum([(det_centers[np.argmin(np.abs(det_centers - rc + s))] - rc + s) ** 2
                    for rc in ref_centers]) for s in shifts]
    best = shifts[int(np.argmin(chi2))]
    nn = [int(np.argmin(np.abs(det_centers - rc + best))) for rc in ref_centers]

    want = set(orders) if orders is not None else set(ref_orders.tolist())
    # aperture reference peak (traceid==1) + all holes per order
    ref_hole = {}
    holes_per_order: dict[int, list[float]] = {}
    for k in range(len(ref_orders)):
        m = int(ref_orders[k])
        if m not in want:
            continue
        holes_per_order.setdefault(m, []).append(det_centers[nn[k]])
        if ref_traceid[k] == 1:
            ref_hole[m] = (peaks_idx[nn[k]], det_centers[nn[k]])

    ymin, ymax = float(_EDGE_LO / 10), float(n_rows - _EDGE_LO / 10)
    traces: dict[int, NDArray] = {}
    centers: dict[int, float] = {}
    coeffs: dict[int, NDArray] = {}
    for m, (idx0, cen0) in ref_hole.items():
        holes = np.array(holes_per_order[m])
        px, py = [], []
        for direction in (-1, +1):
            id_cur = idx0
            shift_prev = 0.0
            row = mid
            while _EDGE_LO < row < min(_EDGE_HI, n_rows - 1):
                if not (_SEARCH_HALF < id_cur < n_cols - _SEARCH_HALF - 1):
                    break
                win = np.arange(id_cur - _SEARCH_HALF, id_cur + _SEARCH_HALF + 1)
                band = np.mean(data[row - 3:row + 2], axis=0)[win]
                loc = argrelmax(band, order=3)[0]
                if loc.size == 0:
                    row += direction * _ROW_STEP
                    continue
                cp = np.array([_centroid(apx[win], band, j) for j in loc])
                # match this row's peaks to the order's hole set
                msep = [np.argmin(np.abs(cp - h - shift_prev)) for h in holes]
                shift_prev = float(np.median([cp[msep[k]] - holes[k]
                                              for k in range(len(holes))]))
                id_cur = int(id_cur + np.mean(loc[msep]) - _SEARCH_HALF)
                # the aperture hole is the one nearest the reference hole
                ap_k = int(np.argmin(np.abs(holes - cen0)))
                px.append(cp[msep[ap_k]]); py.append(float(row + 1))
                row += direction * _ROW_STEP
        px, py = np.array(px), np.array(py)
        if px.size < _CHEB_ORDER + 2:
            continue
        traces[m], coeffs[m] = _cheb_fit(px, py, ymin, ymax, n_rows)
        centers[m] = float(cen0)
    return TracedApertures(traces=traces, centers=centers, coeffs=coeffs,
                           y_min=ymin, y_max=ymax, n_rows=n_rows)


def average_frames(paths: list[Path | str]) -> NDArray[np.float64]:
    """Mean-combine several multihole exposures (SNR for tracing)."""
    stack = [np.asarray(_fits.read_image(Path(p))[0], float) for p in paths]
    return np.mean(stack, axis=0)
