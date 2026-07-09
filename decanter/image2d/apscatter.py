"""Stage 3 — scattered-light subtraction.

WARP equivalent: ``warp/apscatter.py:pyapscatter`` (line 65) wrapping
IRAF ``apscatter``. WARP runs the two-pass fit with these params:

  - apscat1 (perpendicular to dispersion, i.e. each row across x):
      function=legendre, order=3, sample=10:2000,
      low_reject=3.0, high_reject=2.0, niterate=100
  - apscat2 (along dispersion, i.e. each column along y):
      function=legendre, order=5, sample=10:2000, niterate=20

The first pass turns each row into a Legendre model using only the
inter-aperture columns; the second pass smooths the pass-1 surface
along the dispersion direction.

Output suffix: ``_ssc``. Gated on ``Config.flag_apscatter`` —
if False, the input frame is copied through unchanged.

Parity target: median(|Δ|/|frame|) < 1e-3, p99 < 5e-3 against a WARP
reduction with ``flag_apscatter=True``. See PLAN_FULL.md §Validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from decanter.calib.aperture import ApertureSet
from decanter.config import Config
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.listfile import parse as parse_listfile
from decanter.utils.cvsolver import cv_eval_b1 as _cv_eval_b1
from decanter.utils.iraf_icfit import CVSOLVER_DTYPE as _CV_DTYPE
from decanter.utils.iraf_icfit import fit_with_reject


# WARP's apscat parameters (warp/apscatter.py:65-89). Kept as module
# constants so the values are next to the WARP source line they mirror.
#
# CRITICAL convention: IRAF ``apscat1.order`` / ``apscat2.order`` is the
# NUMBER OF TERMS in the Legendre series (``icfit.hlp``: "an order of 2
# has two terms and is a linear function"), NOT the polynomial degree.
# WARP sets ``apscat1.order = 3`` → 3 terms = **quadratic** (degree 2),
# and ``apscat2.order = 5`` → 5 terms = **quartic** (degree 4).
# decanter's ``fit_with_reject`` takes ``degree``, so ``degree = order - 1``.
_AP1_DEGREE = 2  # IRAF apscat1.order = 3 → 3 terms = quadratic
_AP1_LOW_REJECT = 3.0
_AP1_HIGH_REJECT = 2.0
_AP1_NITERATE = 100
_AP2_DEGREE = 4  # IRAF apscat2.order = 5 → 5 terms = quartic
_AP2_LOW_REJECT = 3.0  # IRAF apscat2 defaults; WARP doesn't override
_AP2_HIGH_REJECT = 3.0
_AP2_NITERATE = 20
# WARP's hardcoded sample range "10:2000" (1-indexed, inclusive); becomes
# 0-indexed [9, 1999] inclusive.
_SAMPLE_LO_0IDX = 9
_SAMPLE_HI_0IDX = 1999


def _sanitize_objname(raw: str) -> str:
    """Match s01/s02's filename-sanitization rule."""
    name = raw
    for ch in (" ", "'", "\"", "#", "/"):
        name = name.replace(ch, "_")
    return name


def apscatter_model(
    image: NDArray[np.floating],
    sample_mask: NDArray[np.bool_],
    *,
    ap1_degree: int = _AP1_DEGREE,
    ap1_low_reject: float = _AP1_LOW_REJECT,
    ap1_high_reject: float = _AP1_HIGH_REJECT,
    ap1_niterate: int = _AP1_NITERATE,
    ap2_degree: int = _AP2_DEGREE,
    ap2_low_reject: float = _AP2_LOW_REJECT,
    ap2_high_reject: float = _AP2_HIGH_REJECT,
    ap2_niterate: int = _AP2_NITERATE,
    sample_y_lo: int = _SAMPLE_LO_0IDX,
    sample_y_hi: int = _SAMPLE_HI_0IDX,
) -> NDArray[np.float64]:
    """Build a 2-pass Legendre scattered-light model.

    Args:
        image: 2-D float frame (sky-subtracted).
        sample_mask: 2-D bool. True where the pixel is OUTSIDE any
            aperture AND within the dispersion-cross sample range. Built
            by :func:`build_sample_mask`.
        ap1_*: apscat1 (per-row) parameters.
        ap2_*: apscat2 (per-column) parameters.
        sample_y_lo, sample_y_hi: dispersion-direction sample range
            for the apscat2 pass (0-indexed inclusive). Rows outside this
            range are not used to fit the per-column smoothing.

    Returns:
        ``(H, W)`` float64 array — the scattered-light model.
    """
    image_f = np.asarray(image, dtype=np.float64)
    H, W = image_f.shape
    if sample_mask.shape != image_f.shape:
        raise ValueError("sample_mask shape must match image shape")

    # IRAF apscatter uses 1-indexed column positions for the across-row fit
    # (``apscatter.x:135`` ``Memr[col+i-1] = i``). Match that — the per-pixel
    # values of the fit are identical regardless of 0- vs 1-indexed x for a
    # given normalization range, but if the Legendre is normalized to the
    # *sample* x range (as IRAF does), the basis functions evaluated at the
    # output grid must use the same 1-indexed convention. See
    # ``xtools$icfit/icdosetupr.x:65-71`` for IRAF's choice of xmin/xmax.
    x_all = np.arange(1, W + 1, dtype=np.float64)
    y_all = np.arange(1, H + 1, dtype=np.float64)

    # ---- Pass 1: per-row Legendre across x ---------------------------------
    pass1 = np.zeros((H, W), dtype=np.float64)
    for y in range(H):
        row_mask = sample_mask[y]
        n_kept = int(row_mask.sum())
        if n_kept < ap1_degree + 2:
            # Too few inter-aperture samples to fit a degree-N Legendre.
            # Fall back to the row's median over whatever samples exist;
            # zero-filling would leak into the pass-2 column fits at this
            # row and pull the smoothing model down at the y-edges.
            if n_kept > 0:
                pass1[y, :] = float(np.median(image_f[y, row_mask]))
            continue
        x_sample = x_all[row_mask]
        result = fit_with_reject(
            x_sample,
            image_f[y, row_mask],
            degree=ap1_degree,
            low_reject=ap1_low_reject,
            high_reject=ap1_high_reject,
            niterate=ap1_niterate,
            function="legendre",
            # IRAF normalizes Legendre to the sample's x-range, not the
            # full-frame range (``icdosetupr.x:71`` ``alimr(IC_XFIT,...)``).
            x_min=float(x_sample.min()),
            x_max=float(x_sample.max()),
        )
        # IRAF apscatter's pass-1 eval into the pass-2 buffer goes through
        # ap_cveval (``apextract/apcveval.x:17``), which CLAMPS x to
        # ``[CVXMIN, CVXMAX]`` (the fit's sample range) before calling
        # cveval. Without this clamp, the Legendre extrapolates past the
        # sample x-range and the right-edge columns (cols > ~1500 on
        # HIRES-Y100 with the apsc-maskfile right-edge aperture #184) blow
        # up — decanter's pass-1 eval was producing values like -30 at
        # x=2047 vs IRAF's clamp-to-edge ≈ 1.
        x_clamped = np.clip(x_all, result.x_min, result.x_max)
        # cveval semantics: build the basis via the SCALAR form ``cv_b1leg``
        # (division at end of each Legendre recurrence step) and dot with the
        # coefficient vector using ``adotr`` (left-to-right float32 sum).
        # The vector-form ``cv_evleg`` pre-divides into ri1/ri2 and differs
        # by 1–4 ULP per pixel; that's what was producing the residual
        # medrel 0.06% before this fix.
        pass1[y, :] = _cv_eval_b1(
            result.coefficients, x_clamped,
            function=result.function,
            xmin=result.x_min, xmax=result.x_max,
            dtype=_CV_DTYPE,
        ).astype(np.float64, copy=False)

    # ---- Pass 2: per-column Legendre along y -------------------------------
    pass2 = np.zeros((H, W), dtype=np.float64)
    # apscat2 sample = "10:2000" in 1-indexed -> y in {10..2000}.
    y_lo_1idx = sample_y_lo + 1  # 0-idx 9 -> 1-idx 10
    y_hi_1idx = sample_y_hi + 1  # 0-idx 1999 -> 1-idx 2000
    y_keep = (y_all >= y_lo_1idx) & (y_all <= y_hi_1idx)
    y_kept_arr = y_all[y_keep]
    if y_kept_arr.size < ap2_degree + 2:
        return pass1
    for x in range(W):
        col = pass1[y_keep, x]
        result = fit_with_reject(
            y_kept_arr,
            col,
            degree=ap2_degree,
            low_reject=ap2_low_reject,
            high_reject=ap2_high_reject,
            niterate=ap2_niterate,
            function="legendre",
            x_min=float(y_kept_arr.min()),
            x_max=float(y_kept_arr.max()),
        )
        # Pass-2 also evaluates the fit via ap_cveval at output y rows;
        # match IRAF's scalar cveval semantics here too. apscatter.x's
        # output write loop (line 521 / 639) calls ap_cveval which routes
        # through cveval → cv_b1leg + adotr. Note: pass-2 does NOT clamp
        # to the sample y-range (no apcveval-style guard around the pass-2
        # output write), so we evaluate at the full y grid.
        pass2[:, x] = _cv_eval_b1(
            result.coefficients, y_all,
            function=result.function,
            xmin=result.x_min, xmax=result.x_max,
            dtype=_CV_DTYPE,
        ).astype(np.float64, copy=False)

    return pass2


def build_sample_mask(
    apset: ApertureSet,
    *,
    sample_x_lo: int = _SAMPLE_LO_0IDX,
    sample_x_hi: int = _SAMPLE_HI_0IDX,
    buffer: float = 1.0,
) -> NDArray[np.bool_]:
    """Pixels usable for the apscat1 fit: outside every aperture AND inside
    the x-sample range.

    Mirrors IRAF ``apextract$apscatter.x:ap_gscatter1`` (lines 380-420).
    For each aperture at row ``y`` (1-indexed) with 1-indexed trace
    position ``c``::

        buf    = apscatter.buffer + 0.5     # default buffer = 1.0 → buf = 1.5
        low_j  = max(1, int(c + ap.low  - buf))     # IRAF int() truncates toward 0
        high_j = min(L, int(c + ap.high + buf))
        in_aperture[y, low_j..high_j] = True        # 1-indexed, inclusive

    Signed bounds are preserved (e.g. ``low = -500`` extends to the
    detector edge — that's the WARP convention encoded in the
    apsc_maskfile and what IRAF apscatter reads).

    Until 2026-05-13 decanter used strict ``(low < residue < high)`` with
    no buffer; that happened to *paper over* a separate degree-vs-order
    bug in s03 (apscat1.order = 3 was being treated as cubic instead of
    quadratic) by leaving extra in-aperture pixels at the boundary.
    Once the order bug was fixed, the IRAF default ``buffer = 1.0`` →
    ``buf = 1.5`` brings median |Δ| against WARP's saved scatter from
    0.19 → 0.03 ct/px on TOI2109.
    """
    L = apset.array_length
    buf = float(buffer) + 0.5
    x_1idx = np.arange(1, L + 1, dtype=np.float64)[None, :]
    in_aperture = np.zeros((L, L), dtype=bool)
    for ap in apset.apertures.values():
        trace = ap.trace_x[:, None]
        # IRAF SPP ``int()`` truncates toward zero; ``np.trunc`` does the same on floats.
        low_j = np.maximum(1.0, np.trunc(trace + ap.entry.low - buf))
        high_j = np.minimum(float(L), np.trunc(trace + ap.entry.high + buf))
        in_aperture |= (x_1idx >= low_j) & (x_1idx <= high_j)
    x_idx = np.arange(L)
    in_sample_x = (x_idx >= sample_x_lo) & (x_idx <= sample_x_hi)
    return (~in_aperture) & in_sample_x[None, :]


def subtract_apscatter(
    data: NDArray,
    apset: ApertureSet,
    *,
    sample_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray, NDArray]:
    """Pure task function: subtract scattered light from one 2D frame.

    Builds the IRAF apscatter sample mask from ``apset`` (cached if the
    caller passes one), fits the 2-pass per-row + per-column Legendre
    scattered-light model, and subtracts. Returns ``(out, scatter)``
    where ``out`` has the same dtype as the input.
    """
    if sample_mask is None:
        sample_mask = build_sample_mask(apset)
    scatter = apscatter_model(data, sample_mask)
    out = (data.astype(np.float64) - scatter).astype(data.dtype)
    return out, scatter


def run(
    config: Config,
    workdir: Path,
    listfile: Path,
    *,
    apdb_path: Path | str | None = None,
    **_unused: Any,
) -> None:
    """Subtract scattered light frame-by-frame.

    Args:
        config: pipeline configuration. Gated on
            ``flag_apscatter`` — when False, copies input to output.
        workdir: reads ``{obj}_NO{i}_s.fits`` (s01 output);
            writes ``{obj}_NO{i}_ssc.fits``.
        listfile: WARP-style input list.
        apdb_path: aperture database file. Required when
            ``flag_apscatter`` is True.

    Notes:
        The aperture mask is built once per pipeline run (it's frame-
        independent), then reused for every frame. Each frame still
        pays the per-row × per-column Legendre fitting cost; on a
        2048×2048 frame this is the most expensive 2-D operation in
        the pipeline outside cosmic-ray detection.
    """
    pairs = parse_listfile(listfile)

    if not config.flag_apscatter:
        # No-op gate: copy each *_s.fits to *_ssc.fits unchanged.
        for i, pair in enumerate(pairs, start=1):
            obj_data, obj_header = _fits.read_image(workdir / pair.object_name)
            raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
            objname = _sanitize_objname(str(raw_objname))
            in_path = workdir / f"{objname}_NO{i}_s.fits"
            data, header = _fits.read_image(in_path)
            out_path = workdir / f"{objname}_NO{i}_ssc.fits"
            _fits.write_image(out_path, data, header, overwrite=True)
        return

    if apdb_path is None:
        raise ValueError(
            "s03_apscatter requires apdb_path when flag_apscatter=True; "
            "the calibration loader (s00) will provide this automatically once wired in."
        )

    apset = ApertureSet.load(apdb_path)
    sample_mask = build_sample_mask(apset)

    for i, pair in enumerate(pairs, start=1):
        _, obj_header = _fits.read_image(workdir / pair.object_name)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objname = _sanitize_objname(str(raw_objname))

        data, header = _fits.read_image(workdir / f"{objname}_NO{i}_s.fits")
        out, _ = subtract_apscatter(data, apset, sample_mask=sample_mask)

        out_path = workdir / f"{objname}_NO{i}_ssc.fits"
        _fits.write_image(out_path, out, header, overwrite=True)
