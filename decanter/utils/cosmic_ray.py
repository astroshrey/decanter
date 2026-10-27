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
3. For each (echelle order, y window) WARP measures 28 slit bins
   ``(r_s, r_s + 2]`` for ``r_s = -28 .. 26`` (the edge strips (-30, -28]
   and (28, 30) are never measured): float32 ``np.std`` of the residuals,
   re-clipped up to 3 times at ``|v| < 5 * std`` (strict, from the full bin;
   stop when the std is unchanged, ``==``), and ``np.average`` of the noise.
   ``factor = max(1., max(stds) / max(noises))`` with Python's builtin
   ``max``: an empty bin gives NaN, a NaN in the first bin makes the factor
   1, a later NaN is skipped. Ported line for line (selections in WARP's C
   order, so the float32 reductions are the same).
4. Detect CRs where ``|diff residual| > (sigThres × noise) × factor`` (float32) AND
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

import math
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.ndimage
from numpy.typing import NDArray

from decanter.utils.median_filter import median_filter

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
    """WARP's returned ``sigThres``: the last pass's threshold, or one ``sigstep``
    past ``max_sigma`` when the last pass still looked clustered."""

    iterations: int
    """Number of detection passes run (1 if the first pass converged)."""


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
    angle_tan = math.tan(angle_deg / 180. * math.pi)    # as WARP writes it
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
    diff_resid_abs = np.absolute(diff_resid)

    # --- Step 2: noise model from the slant-smoothed summed raw frames, in the
    # raw frames' own dtype (float32 for WINERED FITS), evaluated left to right
    # as WARP writes it (badpixmask.py:167-169).
    noise = warp_noise_image(raw1, raw2, ndr1, ndr2, gain=gain)

    # --- Step 3: per-(order, y-window) factor, WARP's loop verbatim
    # (badpixmask.py:186-235). Each (order, window, slit-bin) selection is the
    # same pixels in the same (C) order as WARP's boolean index, so np.std /
    # np.average see the same float32 arrays.
    windows = warp_y_windows(array_length, ystep)
    H, W = diff_resid.shape
    flat_apmask = np.asarray(apmask).ravel()
    flat_sc = np.asarray(slitcoord).ravel()
    flat_d = diff_resid.ravel()
    flat_n = noise.ravel()
    order_pix: list[tuple[NDArray, list[slice]]] = []
    factorlist: list[list[object]] = []
    for m in echelle_orders:
        idx = np.flatnonzero(flat_apmask == m)          # C order
        rows = idx // W
        # Yarray > y_s & Yarray <= y_e on the (sorted) rows -> one slice each.
        wsl = [slice(int(np.searchsorted(rows, ys, side="right")),
                     int(np.searchsorted(rows, ye, side="right")))
               for ys, ye in windows]
        order_pix.append((idx, wsl))
        sc_m, d_m, n_m = flat_sc[idx], flat_d[idx], flat_n[idx]
        factors: list[object] = []
        for sl in wsl:
            factors.append(_warp_window_factor(
                sc_m[sl], d_m[sl], n_m[sl], xlim1=xlim1, xlim2=xlim2, bins=bins,
                clipsigma=clipsigma, iteration=iteration))
        factorlist.append(factors)

    # --- Step 4-5: adaptive outer loop (badpixmask.py:237-293), verbatim.
    diffsign = diff_resid > 0
    attributable = ((diffsign & (raw1_resid > 0)) | (~diffsign & (raw2_resid > 0))).ravel()
    diff_abs_flat = diff_resid_abs.ravel()
    bp_flat = np.asarray(static_bp).ravel()
    apos, bpos, opos = _NOD_POSITIONS["A"], _NOD_POSITIONS["B"], _NOD_POSITIONS["O"]
    missdetectionflag = True
    sig_thres = threshold
    itenum = 0
    n_passes = 0
    maskarray = np.zeros(H * W, dtype=np.int16)
    pixnum = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        while missdetectionflag and sig_thres <= max_sigma:
            n_passes += 1
            maskarray = np.zeros(H * W, dtype=np.int16)
            for (idx, wsl), factors in zip(order_pix, factorlist):
                for sl, factor in zip(wsl, factors):
                    sub = idx[sl]
                    reqcr = diff_abs_flat[sub] > sig_thres * flat_n[sub] * factor
                    maskarray[sub[attributable[sub] & reqcr]] += 1
            reqmask = (maskarray == 1) & (bp_flat == 0)
            slit_mask = flat_sc[reqmask]
            pixnum = np.sum(reqmask)
            hist, _ = np.histogram(slit_mask, bins=20, range=(xlim1, xlim2))
            varave = np.var(hist) / np.average(hist)
            if abba:
                anum = np.sum((slit_mask > apos[0]) & (slit_mask < apos[1]))
                bnum = np.sum((slit_mask > bpos[0]) & (slit_mask < bpos[1]))
                spratio = (max(anum, bnum) / pixnum * (xlim2 - xlim1)
                           / max(apos[1] - apos[0], bpos[1] - bpos[0]))
            else:
                onum = np.sum((slit_mask > opos[0]) & (slit_mask < opos[1]))
                spratio = onum / pixnum * (xlim2 - xlim1) / (opos[1] - opos[0])
            if (spratio > slitposratio) & (varave > varatio) & (not fixsigma):
                missdetectionflag = True
                itenum += 1
                sig_thres += sigstep
            else:
                missdetectionflag = False

    return CosmicRayResult(
        mask=maskarray.reshape(H, W),
        n_cosmic_rays=int(pixnum),
        final_threshold=sig_thres,
        iterations=n_passes,
    )


def warp_noise_image(raw1: NDArray, raw2: NDArray, ndr1: int, ndr2: int, *,
                     gain: float = _GAIN_E_PER_ADU) -> NDArray:
    """WARP's CR noise model (``badpixmask.py:167-169``), same dtype and order.

    ``median_filter(|raw1 + raw2|, slant footprint)`` in the raw frames' dtype
    (float32), then ``(mf * gain + rn1 ** 2. + rn2 ** 2.) ** 0.5 / gain``
    evaluated left to right (the read-noise terms are added one at a time).
    """
    summed = np.absolute(raw1 + raw2)
    mf = scipy.ndimage.median_filter(summed, footprint=_make_slant_footprint(85.0, 10))
    return warp_noise_from_median(mf, ndr1, ndr2, gain=gain)


def warp_noise_from_median(mf: NDArray, ndr1: int, ndr2: int, *,
                           gain: float = _GAIN_E_PER_ADU) -> NDArray:
    """``(mf * gain + rn1 ** 2. + rn2 ** 2.) ** 0.5 / gain`` (badpixmask.py:169),
    evaluated left to right in ``mf``'s dtype, from the median-filtered
    ``|raw1 + raw2|``."""
    return (mf * gain + _READ_NOISE_BY_NDR[ndr1] ** 2. + _READ_NOISE_BY_NDR[ndr2] ** 2.) ** 0.5 / gain


def warp_y_windows(array_length: int, ystep: int) -> list[tuple[int, int]]:
    """WARP's dispersion windows ``(y_s, y_e]`` (``badpixmask.py:191-235``).

    The clamp ``y_e = arrayLength`` happens after the last processed window and
    ends the loop, so rows above the last full window (and row 0) are in none.
    """
    out = []
    y_s, y_e = 0, ystep
    while y_e < array_length:
        out.append((y_s, y_e))
        y_s += ystep
        y_e += ystep
        if array_length - y_e < ystep:
            y_e = array_length
    return out


def warp_slit_bins(xlim1: float, xlim2: float, bins: float) -> list[tuple[float, float]]:
    """WARP's slit bins ``(r_s, r_e]`` (``badpixmask.py:200-229``).

    ``r_s`` runs from ``xlim1 + bins`` while ``r_s < xlim2 - bins``: for the
    default -30..30 / 2 these are the 28 bins (-28, -26] ... (26, 28]; the edge
    strips (-30, -28] and (28, 30) are never measured.
    """
    out = []
    r_s, r_e = xlim1 + bins, xlim1 + bins * 2
    while r_s < xlim2 - bins:
        out.append((r_s, r_e))
        r_s += bins
        r_e += bins
    return out


def _warp_window_factor(sc: NDArray, d: NDArray, n: NDArray, *, xlim1: float, xlim2: float,
                        bins: float, clipsigma: float, iteration: int) -> object:
    """One (order, y-window) factor, ``badpixmask.py:200-233`` line for line.

    Empty selections give ``np.std``/``np.average`` NaN; ``max`` is Python's
    builtin (a NaN in the first slot wins, a later NaN is skipped), and
    ``max(1., nan)`` is 1. The returned object is what WARP multiplies by
    (Python float 1.0 or a NumPy scalar of the noise dtype).
    """
    stdlist = []
    noiselist = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for r_s, r_e in warp_slit_bins(xlim1, xlim2, bins):
            req = (r_s < sc) & (sc <= r_e)
            mf_sc = d[req]
            noise_sc = n[req]
            mfstd = np.std(mf_sc)
            for _ in range(iteration):
                mfstd_last = mfstd
                mf_req = np.absolute(mf_sc) < clipsigma * mfstd
                mfstd = np.std(mf_sc[mf_req])
                if mfstd_last == mfstd:
                    break
            stdlist.append(mfstd)
            noiselist.append(np.average(noise_sc))
        stdmax = max(stdlist)
        noisemax = max(noiselist)
        with np.errstate(divide="ignore", invalid="ignore"):
            return max(1., stdmax / noisemax)


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
