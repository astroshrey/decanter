"""Cosmic-ray detection for WINERED echelle frames.

Pure-function port of ``warp/badpixmask.py:cosmicRayMask`` (lines 115-296).
The stage orchestrator in :mod:`decanter.image2d.cosmic_ray` handles
I/O; this module owns the algorithm so it can be unit-tested in
isolation (no FITS files, no aperture databases).

Algorithm summary (Hamano 2024 §5.2 + ``warp/badpixmask.py``):

1. Apply two sequential median filters to the diff image (and to both
   raw frames): a 5×1 vertical filter then a 5×5 diagonal-footprint
   filter. The residuals contain mostly noise + cosmic rays.
2. Build a noise model frame :math:`N(x, y) = \\sqrt{g\\,\\bar{S} + r_1^2 + r_2^2} / g`
   from the local median of the *summed* raw frames (where r₁, r₂ are
   the read-noise values for the two NDR settings).
3. For each (echelle order × y-tile × slit-bin) tile compute the
   per-tile standard deviation via 3 iterations of sigma clipping; the
   max over slit-bins per (order, y-tile) becomes a "factor" multiplier
   applied to the noise floor.
4. Detect CRs where ``|diff residual| > sigThres × noise × factor`` AND
   the spike sign in the diff agrees with the sign in the right raw frame
   (positive diff + positive raw₁ = CR in obj; negative diff + positive
   raw₂ = CR in sky).
5. Outer adaptive loop: if CRs cluster at the nod position (ABBA or
   single-O) above ``slitposratio``, bump ``sigThres`` by ``sigstep``
   and re-detect. Capped at ``max_sigma``.

The bespoke design is not LACosmic — see PLAN_FULL.md §"Cosmic-ray
notes" for the rationale (slit tilt baked into the 5×5 diagonal kernel,
NDR-dependent noise model, A/B nod position cross-check).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.ndimage
from numpy.typing import NDArray

from decanter.utils.median_filter import median_filter
from decanter.utils.sigma_clip import per_segment_clip

# WARP-pinned constants (warp/badpixmask.py:140-143).
_GAIN_E_PER_ADU: float = 2.27
_READ_NOISE_BY_NDR: dict[int, float] = {1: 19.2, 2: 14.0, 4: 10.0, 8: 8.0, 16: 6.0, 32: 5.3}
# Nod-position windows in slit-coordinate units (pixels from trace).
_NOD_POSITIONS = {
    "O": (-4, 4),    # single-target stare position
    "A": (14, 22),   # ABBA mode "A" nod
    "B": (-22, -14), # ABBA mode "B" nod
}


@dataclass(frozen=True, slots=True)
class CosmicRayResult:
    """Detection result, mirroring the return value of WARP's ``cosmicRayMask``."""

    mask: NDArray[np.int16]
    """``(H, W)`` int16 mask: 1 = cosmic ray, 0 = clean."""

    n_cosmic_rays: int
    """Total number of pixels flagged after the outer loop converged."""

    final_threshold: float
    """``sigThres`` value at the iteration that converged (≥ ``threshold``)."""

    iterations: int
    """Number of outer-loop iterations performed (1 if the first pass converged)."""


def _make_slant_footprint(angle_deg: float = 85.0, window: int = 10) -> NDArray[np.int_]:
    """Reproduce ``warp/badpixmask.py:unitArrayMake(angle=85, samplingSize=1, width=1, windowSize=10)``.

    Builds a ``window × window`` boolean footprint marking pixels close
    to a line through the kernel center at ``angle_deg``. WARP uses this
    to smooth the absolute summed-raw-frame image when building the
    noise model — the slant angle (~85° from horizontal) tracks the
    slit tilt so spatial noise structure is averaged along, not across,
    the slit.
    """
    ksize = window
    x = np.arange(ksize)
    y = np.arange(ksize)
    xx, yy = np.meshgrid(x, y)
    xcen = ycen = (ksize - 1) / 2.0
    width = 1.0
    angle_tan = np.tan(np.deg2rad(angle_deg))
    if angle_deg <= 45.0:
        f = (yy - ycen) - angle_tan * (xx - xcen)
    else:
        f = (xx - xcen) - (yy - ycen) / angle_tan
    return ((f > -width / 2.0) & (f < width / 2.0)).astype(np.int_)


def _median_filter_pair(image: NDArray[np.floating], medsize: int = 5) -> NDArray[np.floating]:
    """Apply the WARP 5×1 then 5×5-diagonal median filter pair.

    Returns the residual ``image - filter1 - filter2`` matching WARP's
    ``diffmf2Sub`` / ``raw1mf2Sub`` / ``raw2mf2Sub`` outputs.
    """
    diag = np.identity(medsize, dtype=np.int_)
    mf1 = median_filter(image, size=(medsize, 1))
    sub1 = image - mf1
    mf2 = median_filter(sub1, footprint=diag)
    return sub1 - mf2


def detect_cosmic_rays(
    diff: NDArray[np.floating],
    raw1: NDArray[np.floating],
    raw2: NDArray[np.floating],
    apmask: NDArray[np.integer],
    slitcoord: NDArray[np.floating],
    static_bp: NDArray[np.integer],
    *,
    ndr1: int,
    ndr2: int,
    abba: bool,
    echelle_orders: tuple[int, ...],
    array_length: int = 2048,
    xlim1: float = -30.0,
    xlim2: float = 30.0,
    gain: float = _GAIN_E_PER_ADU,
    medsize: int = 5,
    clipsigma: float = 5.0,
    threshold: float = 10.0,
    bins: int = 2,
    ystep: int = 100,
    iteration: int = 3,
    sigstep: float = 2.0,
    varatio: float = 2.0,
    slitposratio: float = 1.5,
    max_sigma: float = 20.0,
    fixsigma: bool = False,
) -> CosmicRayResult:
    """Detect cosmic rays in a WINERED A/B-paired frame.

    Inputs are pure arrays — the stage module is responsible for FITS
    I/O and for assembling ``apmask`` / ``slitcoord`` from the
    :class:`decanter.calib.aperture.ApertureSet`.

    Args:
        diff: sky-subtracted frame (obj − sky), shape ``(H, W)``.
        raw1: raw obj frame, same shape as ``diff``.
        raw2: raw sky frame, same shape as ``diff``.
        apmask: per-pixel echelle-order label from
            ``ApertureSet.apmask_array(low_lim=xlim1, upp_lim=xlim2)``.
            0 outside any aperture, ``m`` inside order ``m``.
        slitcoord: per-pixel signed slit position from
            ``ApertureSet.slitcoord_array(low_lim=xlim1, upp_lim=xlim2)``.
            ``-10000`` outside any aperture.
        static_bp: static bad-pixel mask (from the master flat); 1 = bad.
        ndr1, ndr2: NDR (non-destructive read) settings of the two raw
            frames; pick read noise from :data:`_READ_NOISE_BY_NDR`.
        abba: ``True`` for ABBA nodding (use A/B nod windows in the
            position-cluster check), ``False`` for stare mode (use O).
        echelle_orders: the list of orders to iterate over (typically
            ``apset.echelle_orders``).
        array_length: detector axis length.
        xlim1, xlim2: slit-coordinate bounds for the cross-dispersion
            window (default ±30 px, matching WARP).
        gain: e⁻/ADU.
        medsize: median-filter kernel size (default 5).
        clipsigma: per-tile sigma-clip rejection threshold (default 5).
        threshold: initial CR detection threshold in σ (default 10).
        bins: slit-coord bin width for the per-tile factor map (default 2).
        ystep: dispersion-axis tile height in pixels (default 100).
        iteration: sigma-clip iteration count per tile (default 3).
        sigstep: outer-loop threshold bump (default 2).
        varatio: variance/mean ratio threshold for "too clustered" check.
        slitposratio: position-density ratio threshold.
        max_sigma: cap on the adaptive threshold.
        fixsigma: if ``True``, skip the adaptive bump and use ``threshold``
            verbatim.

    Returns:
        :class:`CosmicRayResult`.

    Notes:
        Threshold clamping: if ``threshold > max_sigma``, ``threshold`` is
        lowered to ``max_sigma`` first (matches WARP lines 119-120).
    """
    if threshold > max_sigma:
        threshold = max_sigma

    # --- Step 1: median-filter residuals on the diff and both raw frames.
    diff_resid = _median_filter_pair(diff, medsize)
    raw1_resid = _median_filter_pair(raw1, medsize)
    raw2_resid = _median_filter_pair(raw2, medsize)
    diff_sign_positive = diff_resid > 0
    raw1_sign_positive = raw1_resid > 0
    raw2_sign_positive = raw2_resid > 0
    diff_resid_abs = np.abs(diff_resid)

    # --- Step 2: noise model from the slant-smoothed summed raw frames.
    slant = _make_slant_footprint(angle_deg=85.0, window=10)
    summed = np.abs(raw1.astype(np.float64) + raw2.astype(np.float64))
    smoothed = scipy.ndimage.median_filter(summed, footprint=slant, mode="reflect")
    read_var = _READ_NOISE_BY_NDR[ndr1] ** 2 + _READ_NOISE_BY_NDR[ndr2] ** 2
    noise = np.sqrt(smoothed * gain + read_var) / gain

    # --- Step 3: per-(order, y-tile, slit-bin) factor map.
    H, W = diff_resid.shape
    y_grid, _ = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    n_orders = len(echelle_orders)
    n_slit_bins = int(np.floor((xlim2 - xlim1) / bins))

    # WARP's tile boundaries (badpixmask.py:196-235):
    #   y_s, y_e = 0, ystep
    #   while y_e < arrayLength:
    #       process tile (y_s, y_e]
    #       y_s += ystep; y_e += ystep
    #       if arrayLength - y_e < ystep: y_e = arrayLength
    # So tile k=0..K-1 is processed where K is the largest integer with
    # (k+1)*ystep < arrayLength STRICTLY. The clamp pushes y_e to
    # arrayLength after the LAST processed iteration, blocking iter K+1.
    # `(Yarray > y_s) & (Yarray <= y_e)` with Yarray 0-indexed means
    # tile k covers Y ∈ {k*ystep + 1, ..., (k+1)*ystep}. Y=0 is
    # never in any tile; the bottom partial (Y > K*ystep) is also not.
    #
    # For H=2048, ystep=100: K=19 tiles covering Y∈{1..1900}; Y=0 and
    # Y∈{1901..2047} are NOT in any tile (factor=1 there).
    n_y_tiles = (H - ystep) // ystep
    # Tile assignment: floor((Y - 1) / ystep). Y=0 maps to -1 (invalid);
    # Y >= n_y_tiles * ystep + 1 maps to >= n_y_tiles (invalid).
    y_tile_map = (y_grid - 1) // ystep
    y_tile_valid = (y_grid >= 1) & (y_tile_map < n_y_tiles)

    order_to_idx = {m: i for i, m in enumerate(echelle_orders)}
    order_idx_map = np.full(apmask.shape, -1, dtype=np.int64)
    for m, idx in order_to_idx.items():
        order_idx_map[apmask == m] = idx
    slit_bin_map = np.floor((slitcoord - xlim1) / bins).astype(np.int64)
    in_aperture = order_idx_map >= 0
    in_slit_range = (slit_bin_map >= 0) & (slit_bin_map < n_slit_bins)
    valid = in_aperture & in_slit_range & y_tile_valid
    tile_id = np.full(apmask.shape, -1, dtype=np.int64)
    n_tiles = n_orders * n_y_tiles * n_slit_bins
    tile_id[valid] = (
        order_idx_map[valid] * (n_y_tiles * n_slit_bins)
        + y_tile_map[valid] * n_slit_bins
        + slit_bin_map[valid]
    )
    # WARP convention: rejection threshold is centered at ZERO, not at
    # the segment mean (badpixmask.py:215, `np.absolute(mf_sc) < ...`).
    # Matters when a tile's mean is shifted away from zero by an
    # outlier — mean-centered keeps the outlier; zero-centered rejects
    # it (order 164 tile 3 differed by Δfactor = +0.25 with the wrong
    # center).
    _, std_per_tile, _ = per_segment_clip(
        diff_resid, tile_id, n_segments=n_tiles, sigma=clipsigma,
        iterations=iteration, center="zero",
    )
    # Per-tile mean noise (no sigma clip; matches WARP's `np.average` at line 224).
    noise_sum = np.bincount(tile_id[valid].ravel(), weights=noise[valid].ravel(), minlength=n_tiles)
    noise_count = np.bincount(tile_id[valid].ravel(), minlength=n_tiles)
    with np.errstate(invalid="ignore", divide="ignore"):
        noise_per_tile = np.where(noise_count > 0, noise_sum / noise_count, np.nan)
    # Per (order, y-tile): max over slit bins.
    std_3d = std_per_tile.reshape(n_orders, n_y_tiles, n_slit_bins)
    noise_3d = noise_per_tile.reshape(n_orders, n_y_tiles, n_slit_bins)
    std_max = np.nanmax(std_3d, axis=2)         # shape (n_orders, n_y_tiles)
    noise_max = np.nanmax(noise_3d, axis=2)     # same shape
    with np.errstate(invalid="ignore", divide="ignore"):
        factor_2d = np.where(noise_max > 0, std_max / noise_max, 1.0)
    factor_2d = np.maximum(factor_2d, 1.0)
    # Broadcast factor back to per-pixel for the detection step. Pixels
    # outside the y-tile range (Y=0 and Y>n_y_tiles*ystep) get factor=1
    # by default — same as WARP, which simply doesn't process them in
    # the factor loop but still applies the detection (threshold * noise).
    factor_per_pixel = np.ones(apmask.shape, dtype=np.float64)
    in_factor = in_aperture & y_tile_valid
    if in_factor.any():
        factor_per_pixel[in_factor] = factor_2d[
            order_idx_map[in_factor], y_tile_map[in_factor]
        ]

    # --- Step 4-5: adaptive outer loop.
    sigma_threshold = threshold
    iter_count = 0
    final_mask = np.zeros(diff.shape, dtype=np.int16)
    final_pix = 0
    while True:
        iter_count += 1
        local_thresh = sigma_threshold * noise * factor_per_pixel
        triggered = diff_resid_abs > local_thresh
        # Attribute sign: positive diff + positive raw1 → CR in obj;
        # negative diff + positive raw2 → CR in sky.
        attributable = (
            (diff_sign_positive & raw1_sign_positive)
            | (~diff_sign_positive & raw2_sign_positive)
        )
        # The CR mask itself does NOT exclude static-BP pixels — WARP
        # saves the unfiltered maskarray (badpixmask.py:295). The
        # downstream pipeline combines static + CR at fixpix time via
        # `iraf.imarith(mask_file, "+", cr_mask, maskflat)` so the
        # static-BP filter only matters for the var/ave histogram check.
        #
        # `y_tile_valid` excludes Y=0 and Y > n_y_tiles*ystep. WARP's
        # detection step is wrapped inside the same `for j in range(len(
        # factorlist[i]))` loop that built the factor map, so rows
        # outside any y-tile are never written to maskarray at all
        # (badpixmask.py:241-247). We match by gating cr_mask on
        # `y_tile_valid` here.
        cr_mask = (
            triggered
            & attributable
            & in_aperture
            & y_tile_valid
        ).astype(np.int16)

        # Position-cluster check uses only the non-static-BP pixels
        # (WARP badpixmask.py:249 — `reqmask = (maskarray == 1) & (bpfdata == 0)`).
        slit_in_mask = slitcoord[(cr_mask == 1) & (static_bp == 0)]
        pix_num = int(slit_in_mask.size)
        if pix_num == 0:
            final_mask = cr_mask
            final_pix = 0
            break
        hist, _ = np.histogram(slit_in_mask, bins=20, range=(xlim1, xlim2))
        var = float(np.var(hist))
        ave = float(np.mean(hist))
        varave = var / ave if ave > 0 else 0.0

        # Decide whether to bump the threshold.
        if abba:
            apos_lo, apos_hi = _NOD_POSITIONS["A"]
            bpos_lo, bpos_hi = _NOD_POSITIONS["B"]
            n_a = int(((slit_in_mask > apos_lo) & (slit_in_mask < apos_hi)).sum())
            n_b = int(((slit_in_mask > bpos_lo) & (slit_in_mask < bpos_hi)).sum())
            window_a = apos_hi - apos_lo
            window_b = bpos_hi - bpos_lo
            sp_ratio = (
                max(n_a, n_b) / pix_num * (xlim2 - xlim1) / max(window_a, window_b)
            )
        else:
            opos_lo, opos_hi = _NOD_POSITIONS["O"]
            n_o = int(((slit_in_mask > opos_lo) & (slit_in_mask < opos_hi)).sum())
            window_o = opos_hi - opos_lo
            sp_ratio = n_o / pix_num * (xlim2 - xlim1) / window_o

        clustered = (sp_ratio > slitposratio) and (varave > varatio)
        if clustered and not fixsigma and sigma_threshold + sigstep <= max_sigma:
            sigma_threshold += sigstep
            continue

        final_mask = cr_mask
        final_pix = pix_num
        break

    return CosmicRayResult(
        mask=final_mask,
        n_cosmic_rays=final_pix,
        final_threshold=sigma_threshold,
        iterations=iter_count,
    )


def ndr_from_header(
    header: dict[str, object],
    *,
    noutputs: int | None = None,
    exptime: float | None = None,
) -> int:
    """Determine the NDR (non-destructive read) count from a WINERED header.

    WARP equivalent: ``warp/badpixmask.py:NDRreader`` (lines 90-112).

    Args:
        header: a FITS header (or dict-like) supporting ``__contains__`` / ``__getitem__``.
        noutputs: optional override for ``NOUTPUTS``.
        exptime: optional override for ``EXPTIME``.

    Returns:
        An integer NDR value present in :data:`_READ_NOISE_BY_NDR`.
    """
    raw_ndr: object = "N/A"
    try:
        raw_ndr = header["NDR"]
    except (KeyError, IndexError):
        raw_ndr = "N/A"
    if raw_ndr != "N/A":
        return int(raw_ndr)

    if noutputs is None:
        try:
            noutputs_val: object = header["NOUTPUTS"]
        except (KeyError, IndexError):
            noutputs_val = 32
        noutputs = int(noutputs_val)  # type: ignore[arg-type]
    if exptime is None:
        try:
            exptime_val: object = header["EXPTIME"]
        except (KeyError, IndexError):
            exptime_val = 0.0
        exptime = float(exptime_val)  # type: ignore[arg-type]

    # WARP table at lines 95-108 — staircase by exptime, with two regimes
    # depending on NOUTPUTS.
    if noutputs == 32:
        for limit, value in [(6, 1), (12, 2), (30, 4), (300, 8)]:
            if exptime <= limit:
                return value
        return 16
    return 1


__all__ = [
    "CosmicRayResult",
    "detect_cosmic_rays",
    "ndr_from_header",
]
