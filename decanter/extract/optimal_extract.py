"""Horne (1986) optimal extraction on rectified order strips.

This is an *algorithmic upgrade* over the box sum in ``box_extract_1d.py`` — it
is not part of the WARP parity port (WARP box-extracts). It is offered as an
opt-in (``decanter.reduce(..., extract="optimal")``) for users who want the
SNR gain of profile weighting, and to compare against WARP / box extraction.

Method (Horne 1986, "An optimal extraction algorithm for CCD spectroscopy"):

  1. The extraction window is the same star-centered aperture the box sum uses:
     ``center(y) = trace_x(y) + (ap_low + ap_high) / 2``, half-width
     ``(ap_high - ap_low) / 2``.
  2. A **spatial profile** ``P`` is built empirically from the data by stacking
     the background-subtracted, flux-normalized cross-sections over dispersion
     (the rectified strip makes the profile ~constant along dispersion, so one
     shared profile is a good model), then enforcing positivity and unit sum.
  3. Each output pixel is the inverse-variance, profile-weighted estimate
     ``f = sum(P * D / V) / sum(P^2 / V)`` over the window, with a simple
     ``V = read_var + max(D, 0)`` (background/read variance + Poisson). This
     down-weights the noisy aperture wings and any residual outliers, unlike
     the box sum which weights every in-aperture pixel equally.

The profile is data-driven rather than a Gaussian so non-Gaussian slit wings
are captured. Cosmic rays are already handled upstream (CR mask + fixpix), so
no per-pixel sigma-clip rejection loop is run here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_PROFILE_ROW_STEP = 3          # sub-sample rows when building the shared profile
_PROFILE_PAD = 3               # extra px each side of the aperture for the profile


def _background_level(row: NDArray[np.floating], lo: int, hi: int, W: int) -> float:
    """Median of the pixels flanking the aperture window ``[lo, hi)``."""
    left = row[max(0, lo - 10):max(0, lo)]
    right = row[min(W, hi):min(W, hi + 10)]
    flank = np.concatenate([left, right])
    return float(np.median(flank)) if flank.size else 0.0


def _star_sign(image: NDArray[np.floating], center: NDArray[np.floating], half: float) -> float:
    """+1 or -1: sign of the star in the strip.

    In ABBA nod subtraction, obj-sky yields a *negative* star for the frames
    where the object sits in the subtracted nod position. The profile machinery
    assumes a positive bump, so the strip is worked in sign-corrected space.
    """
    H, W = image.shape
    tot = 0.0
    for y in range(0, H, _PROFILE_ROW_STEP):
        c = int(round(center[y])); lo = c - int(half); hi = c + int(half) + 1
        if lo < 0 or hi > W:
            continue
        seg = image[y, lo:hi].astype(np.float64)
        tot += seg.sum() - np.median(image[y]) * (hi - lo)
    return -1.0 if tot < 0 else 1.0


def _build_profile(
    image: NDArray[np.floating], center: NDArray[np.floating], half: float, sign: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Empirical unit-sum spatial profile on an integer offset grid.

    Built in sign-corrected space (``sign * image``) so a negative-nod star is
    treated as a positive bump; the returned profile is always positive.
    """
    H, W = image.shape
    phw = int(np.ceil(half)) + _PROFILE_PAD
    offs = np.arange(-phw, phw + 1, dtype=np.float64)
    cols = np.arange(W)
    acc = []
    for y in range(0, H, _PROFILE_ROW_STEP):
        c = center[y]
        if c - phw < 0 or c + phw >= W:
            continue
        seg = sign * np.interp(c + offs, cols, image[y].astype(np.float64))
        seg = seg - np.median(seg)          # remove local background pedestal
        total = seg[seg > 0].sum()
        if total <= 0:
            continue
        acc.append(seg / total)
    if not acc:
        # Degenerate strip (no flux): fall back to a flat profile.
        prof = np.ones_like(offs)
    else:
        prof = np.median(np.vstack(acc), axis=0)
    prof[prof < 0] = 0.0
    if prof.sum() <= 0:
        prof = np.ones_like(offs)
    prof /= prof.sum()
    return offs, prof


def _read_variance(
    image: NDArray[np.floating], center: NDArray[np.floating], half: float
) -> float:
    """Background+read variance from robust scatter of off-aperture pixels."""
    H, W = image.shape
    samp = []
    for y in range(0, H, _PROFILE_ROW_STEP):
        c = int(round(center[y]))
        lo, hi = c - int(np.ceil(half)), c + int(np.ceil(half)) + 1
        left = image[y, max(0, lo - 12):max(0, lo)]
        right = image[y, min(W, hi):min(W, hi + 12)]
        samp.append(np.concatenate([left, right]))
    flank = np.concatenate(samp) if samp else np.array([0.0])
    flank = flank[np.isfinite(flank)]
    if flank.size < 8:
        return 1.0
    mad = np.median(np.abs(flank - np.median(flank)))
    sigma = 1.4826 * mad                    # robust std
    return float(max(sigma * sigma, 1.0))


def optimal_extract(
    image: NDArray[np.floating],
    trace_x: NDArray[np.floating],
    *,
    ap_low: float,
    ap_high: float,
    read_var: float | None = None,
    gain: float = 1.0,
    return_var: bool = False,
) -> NDArray[np.float32] | tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Optimally extract one rectified strip to a 1-D spectrum.

    Args:
        image: 2-D rectified strip ``(n_y, n_x)``.
        trace_x: 1-indexed trace position per row (as for ``box_extract``).
        ap_low, ap_high: star-centered window bounds relative to the trace.
        read_var: background+read variance (counts^2). Estimated from the
            off-aperture scatter when ``None``.
        gain: counts-to-electrons factor for the Poisson term (default 1;
            only the relative weighting matters).
        return_var: if True, also return the per-pixel variance of the
            optimal estimate (``1 / sum(P^2 / V)``) — useful for SNR.

    Returns:
        Float32 flux array of length ``n_y`` (rows whose window falls off the
        strip return 0). If ``return_var``, a ``(flux, variance)`` tuple.
    """
    H, W = image.shape
    img = image.astype(np.float64)
    trace0 = np.asarray(trace_x, np.float64) - 1.0        # 1-idx -> 0-idx
    center = trace0 + (ap_low + ap_high) / 2.0
    half = (ap_high - ap_low) / 2.0
    if half <= 0:
        raise ValueError(f"ap_high ({ap_high}) must exceed ap_low ({ap_low})")

    sign = _star_sign(img, center, half)
    offs, prof = _build_profile(img, center, half, sign)
    rv = _read_variance(img, center, half) if read_var is None else float(read_var)

    out = np.zeros(H, dtype=np.float32)
    var = np.zeros(H, dtype=np.float32)
    for y in range(H):
        c = center[y]
        lo = int(np.floor(c - half))
        hi = int(np.ceil(c + half))
        if lo < 0 or hi >= W:
            continue
        pix = np.arange(lo, hi + 1)
        p = np.interp(pix - c, offs, prof, left=0.0, right=0.0)
        psum = p.sum()
        if psum <= 0:
            continue
        p = p / psum
        D = img[y, pix]
        # Variance from the PROFILE MODEL, not the observed data: using D
        # directly biases the optimal estimate low (pixels that fluctuate high
        # get down-weighted). f0 is the box total for this row (Horne 1986).
        f0 = max(sign * D.sum(), 0.0)
        V = rv + np.maximum(f0 * p, 0.0) / gain
        w = p / V
        denom = (w * p).sum()
        if denom <= 0:
            continue
        out[y] = np.float32((w * D).sum() / denom)
        var[y] = np.float32(1.0 / denom)
    return (out, var) if return_var else out
