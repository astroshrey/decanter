"""Fast 2-D detector-frame stage: the CR mask (s02).

Same algorithms as the warp-mode modules (:mod:`decanter.image2d`,
:mod:`decanter.utils.cosmic_ray`), with the
Python loops over rows / columns / tiles replaced by whole-array operations and
everything that depends only on the calibration set precomputed once
(:class:`CrGeometry`).

Bit-identical to warp mode: the median filters are exact selections
  (sorting networks on shifted views give the same element as
  ``scipy.ndimage.median_filter``); the noise model is the same float32
  expression; the per-tile clipped ``np.std`` / ``np.average`` are NumPy's own
  float32 reductions (pairwise sums) emulated for all tiles at once
  (:func:`segment_std`), each tile's pixels in WARP's C order; and the
  adaptive threshold loop is WARP's.

WARP equivalent: ``warp/badpixmask.py:cosmicRayMask``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from decanter.calib.aperture import ApertureSet
from decanter.utils.cosmic_ray import _GAIN_E_PER_ADU, _NOD_POSITIONS, _READ_NOISE_BY_NDR

# --------------------------------------------------------------------------
# Exact median filters (sorting networks on shifted views)
# --------------------------------------------------------------------------

# Optimal 9-comparator sorting network for 5 inputs.
_SORT5 = ((0, 1), (3, 4), (2, 4), (2, 3), (1, 4), (0, 3), (0, 2), (1, 3), (1, 2))


def _sort5(v: list[NDArray]) -> list[NDArray]:
    v = list(v)
    for i, j in _SORT5:
        lo = np.minimum(v[i], v[j])
        v[j] = np.maximum(v[i], v[j])
        v[i] = lo
    return v


def _median5(v: list[NDArray]) -> NDArray:
    """Median of 5 arrays, elementwise (7-comparator selection network)."""
    a, b, c, d, e = v
    a, b = np.minimum(a, b), np.maximum(a, b)
    d, e = np.minimum(d, e), np.maximum(d, e)
    a, d = np.minimum(a, d), np.maximum(a, d)      # a = min of 4 -> discard
    b, e = np.minimum(b, e), np.maximum(b, e)      # e = max of 4 -> discard
    del a, e
    # median of {b, c, d}
    lo = np.minimum(b, c)
    hi = np.maximum(b, c)
    return np.maximum(lo, np.minimum(hi, d))


def median5_vertical(x: NDArray) -> NDArray:
    """``scipy.ndimage.median_filter(x, size=(5, 1), mode='reflect')``, exactly."""
    H = x.shape[0]
    p = np.pad(x, ((2, 2), (0, 0)), mode="symmetric")
    return _median5([p[k:k + H] for k in range(5)])


def median5_diagonal(x: NDArray) -> NDArray:
    """``median_filter(x, footprint=np.identity(5), mode='reflect')``, exactly."""
    H, W = x.shape
    p = np.pad(x, 2, mode="symmetric")
    return _median5([p[k:k + H, k:k + W] for k in range(5)])


def median_slant10(x: NDArray) -> NDArray:
    """``median_filter(x, footprint=_make_slant_footprint(85, 10), mode='reflect')``.

    WARP's 85-degree slant footprint is two vertical 5-pixel segments: rows
    ``y-5..y-1`` of column ``x-1`` and rows ``y..y+4`` of column ``x``. scipy
    returns the element of rank ``10 // 2 = 5`` (0-based). Each segment is a
    vertical 5-window of the same padded image, so the windows are sorted once
    and rank 5 of the union of two sorted 5-lists ``A``, ``B`` is
    ``min_a max(A[a], B[4 - a])``. Pure selection -> bit-identical.
    """
    H, W = x.shape
    p = np.pad(x, 5, mode="symmetric")                 # (H + 10, W + 10)
    s = _sort5([p[k:k + H + 5] for k in range(5)])     # sorted windows from row r
    A = [sk[0:H, 4:4 + W] for sk in s]                 # rows y-5.., column x-1
    B = [sk[5:5 + H, 5:5 + W] for sk in s]             # rows y.., column x
    out = np.maximum(A[0], B[4])
    for a in range(1, 5):
        out = np.minimum(out, np.maximum(A[a], B[4 - a]))
    return out


def median_filter_pair(image: NDArray, keep: dict | None = None) -> NDArray:
    """``decanter.utils.cosmic_ray._median_filter_pair`` (5x1 then 5x5-diagonal);
    ``keep["mf1"]`` receives the first residual (WARP's ``diffmf1Sub``)."""
    img = np.asarray(image, dtype=np.float32)
    sub1 = img - median5_vertical(img)
    if keep is not None:
        keep["mf1"] = sub1
    return sub1 - median5_diagonal(sub1)


# --------------------------------------------------------------------------
# NumPy float32 reductions, segmented (exact)
# --------------------------------------------------------------------------

_PW_BLOCK = 128   # numpy PW_BLOCKSIZE


def _gather_padded(vals: NDArray, starts: NDArray, lens: NDArray, width: int) -> NDArray:
    """``(k, width)`` matrix of each segment's values, zero padded past its end."""
    col = np.arange(width)
    idx = starts[:, None] + col[None, :]
    ok = col[None, :] < lens[:, None]
    out = np.zeros(idx.shape, dtype=vals.dtype)
    out[ok] = vals[idx[ok]]
    return out


def _pairwise_sum(vals: NDArray, starts: NDArray, lens: NDArray) -> NDArray:
    """NumPy's ``pairwise_sum`` (``loops_utils.h.src``) of every segment, vectorised.

    ``np.add.reduce`` of a contiguous float32 array is ``pairwise_sum(a, n)`` in
    float32: below 8 elements a sequential sum from 0; up to 128 eight strided
    accumulators combined as ``((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))`` then the
    ``n % 8`` tail added in order; above 128 the two halves split at
    ``n2 = n//2 - (n//2) % 8``, summed recursively and added. Adding a padding 0
    is exact, so segments of different lengths in the same branch share one
    zero-padded matrix.
    """
    dt = vals.dtype
    out = np.zeros(lens.shape, dtype=dt)
    small = lens < 8
    if small.any():
        M = _gather_padded(vals, starts[small], lens[small], 7)
        res = np.zeros(M.shape[0], dtype=dt)
        for i in range(7):
            res = res + M[:, i]
        out[small] = res
    mid = (lens >= 8) & (lens <= _PW_BLOCK)
    if mid.any():
        st, ln = starts[mid], lens[mid]
        nb = ln - ln % 8                                   # unrolled part
        B = _gather_padded(vals, st, nb, _PW_BLOCK)
        r = B[:, 0:8].copy()
        for i in range(8, _PW_BLOCK, 8):
            r = r + B[:, i:i + 8]
        res = ((r[:, 0] + r[:, 1]) + (r[:, 2] + r[:, 3])) + ((r[:, 4] + r[:, 5]) + (r[:, 6] + r[:, 7]))
        T = _gather_padded(vals, st + nb, ln - nb, 7)      # n % 8 tail, in order
        for i in range(7):
            res = res + T[:, i]
        out[mid] = res
    big = lens > _PW_BLOCK
    if big.any():
        st, ln = starts[big], lens[big]
        n2 = ln // 2
        n2 = n2 - n2 % 8
        out[big] = _pairwise_sum(vals, st, n2) + _pairwise_sum(vals, st + n2, ln - n2)
    return out


_REDUCE_BUFSIZE = 8192   # numpy's default ufunc buffer size


def _add_reduce(vals: NDArray, starts: NDArray, lens: NDArray) -> NDArray:
    """``np.add.reduce`` of every segment: pairwise sums of consecutive
    8192-element chunks (the reduction's buffer), accumulated in order."""
    out = _pairwise_sum(vals, starts, np.minimum(lens, _REDUCE_BUFSIZE))
    done = np.minimum(lens, _REDUCE_BUFSIZE)
    while (lens > done).any():
        sel = lens > done
        step = np.minimum(lens[sel] - done[sel], _REDUCE_BUFSIZE)
        out[sel] = out[sel] + _pairwise_sum(vals, starts[sel] + done[sel], step)
        done[sel] += step
    return out


def segment_mean(vals: NDArray, starts: NDArray, lens: NDArray) -> NDArray:
    """``np.mean`` of each segment ``vals[s:s+n]`` (float32 in, float32 out; NaN if empty)."""
    tot = _add_reduce(vals, starts, lens)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tot / lens.astype(vals.dtype)


def segment_std(vals: NDArray, seg: NDArray, starts: NDArray, lens: NDArray) -> NDArray:
    """``np.std`` of each segment, exactly as ``numpy._core._methods._var`` computes it.

    ``vals`` is grouped by segment (``seg`` the segment id per value, ``starts``
    / ``lens`` per segment): mean = pairwise sum / n, ``x = v - mean``,
    ``x * x``, pairwise sum / n, sqrt, all in the input dtype.
    """
    mean = segment_mean(vals, starts, lens)
    x = vals - mean[seg]
    x = x * x
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(_add_reduce(x, starts, lens) / lens.astype(vals.dtype))


# --------------------------------------------------------------------------
# s02 cosmic-ray mask
# --------------------------------------------------------------------------

_SLIT_WINDOW = (-30.0, 30.0)


@dataclass
class CrGeometry:
    """Calibration-only pieces of the WARP CR detector (see detect_cosmic_rays).

    ``S`` = pixels of an echelle order inside one of WARP's y windows (the only
    pixels WARP's detection loop writes); tiles = WARP's (order, window, slit
    bin) selections, with each tile's pixels in C order (the order of WARP's
    boolean index).
    """

    shape: tuple[int, int]
    n_orders: int
    n_windows: int
    n_slit_bins: int
    s_idx: NDArray[np.intp]        # flat index of S pixels (C order)
    s_fidx: NDArray[np.intp]       # order_idx * n_windows + window per S pixel
    s_slit: NDArray[np.float64]    # slit coordinate per S pixel
    s_bp0: NDArray[np.bool_]       # static_bp == 0 per S pixel
    t_idx: NDArray[np.intp]        # flat index of tile pixels, grouped by tile, C order within
    t_in_s: NDArray[np.intp]       # position within S of each t_idx pixel
    t_seg: NDArray[np.intp]        # tile id per t_idx pixel
    t_starts: NDArray[np.intp]     # first t_idx position per tile
    t_lens: NDArray[np.intp]       # pixels per tile

    @classmethod
    def build(cls, apset: ApertureSet, static_bp: NDArray, *, bins: int = 2,
              ystep: int = 100) -> CrGeometry:
        from decanter.utils.cosmic_ray import warp_slit_bins, warp_y_windows
        xlim1, xlim2 = _SLIT_WINDOW
        apmask = apset.apmask_array(low_lim=xlim1, upp_lim=xlim2)
        slitcoord = apset.slitcoord_array(low_lim=xlim1, upp_lim=xlim2)
        H, W = apmask.shape
        orders = apset.echelle_orders
        n_orders = len(orders)
        windows = warp_y_windows(apset.array_length, ystep)
        n_windows = len(windows)
        win_of_row = np.full(H, -1, dtype=np.int64)
        for j, (ys, ye) in enumerate(windows):          # rows ys < y <= ye
            win_of_row[ys + 1:ye + 1] = j
        sbins = warp_slit_bins(xlim1, xlim2, bins)
        n_slit_bins = len(sbins)
        # bin k is (edges[k], edges[k+1]]: searchsorted 'left' counts edges < s.
        edges = np.array([b[0] for b in sbins] + [sbins[-1][1]], dtype=np.float64)
        order_idx_map = np.full(apmask.shape, -1, dtype=np.int64)
        for i, m in enumerate(orders):
            order_idx_map[apmask == m] = i
        win_map = np.broadcast_to(win_of_row[:, None], (H, W))
        S = (order_idx_map >= 0) & (win_map >= 0)
        s_idx = np.flatnonzero(S)
        sc_s = slitcoord.ravel()[s_idx]
        oi_s = order_idx_map.ravel()[s_idx]
        wj_s = win_map.ravel()[s_idx]
        k_s = np.searchsorted(edges, sc_s, side="left") - 1
        in_bin = (k_s >= 0) & (k_s < n_slit_bins)
        tile_s = (oi_s * n_windows + wj_s) * n_slit_bins + k_s
        t_in_s = np.flatnonzero(in_bin)
        t_in_s = t_in_s[np.argsort(tile_s[t_in_s], kind="stable")]
        t_seg = tile_s[t_in_s]
        n_tiles = n_orders * n_windows * n_slit_bins
        t_lens = np.bincount(t_seg, minlength=n_tiles)
        t_starts = np.concatenate(([0], np.cumsum(t_lens)[:-1]))
        return cls(
            shape=(H, W), n_orders=n_orders, n_windows=n_windows,
            n_slit_bins=n_slit_bins,
            s_idx=s_idx,
            s_fidx=(oi_s * n_windows + wj_s).astype(np.intp),
            s_slit=sc_s,
            s_bp0=(np.asarray(static_bp).astype(np.int16).ravel()[s_idx] == 0),
            t_idx=s_idx[t_in_s].astype(np.intp),
            t_in_s=t_in_s.astype(np.intp),
            t_seg=t_seg.astype(np.intp),
            t_starts=t_starts.astype(np.intp),
            t_lens=t_lens.astype(np.intp),
        )


def _tile_clip_std(vals: NDArray, geom: CrGeometry, *, clipsigma: float,
                   iteration: int) -> NDArray:
    """WARP's per-tile clipped std (``badpixmask.py:211-219``) for every tile at once.

    ``std = np.std(all)``; up to ``iteration`` times: keep ``|v| < clipsigma *
    std`` (from the full tile, float32 threshold), ``std = np.std(kept)``, stop
    that tile when the std is unchanged (``==``: a NaN std never stops).
    """
    seg, starts, lens = geom.t_seg, geom.t_starts, geom.t_lens
    std = segment_std(vals, seg, starts, lens)
    active = np.ones(lens.shape, dtype=bool)
    absv = np.absolute(vals)
    for _ in range(iteration):
        with np.errstate(invalid="ignore", over="ignore"):
            keep = absv < (clipsigma * std)[seg]          # float32 product, as WARP
        kv = vals[keep]
        kseg = seg[keep]
        klens = np.bincount(kseg, minlength=lens.size).astype(np.intp)
        kstarts = np.concatenate(([0], np.cumsum(klens)[:-1])).astype(np.intp)
        new = segment_std(kv, kseg, kstarts, klens)
        same = new == std
        std = np.where(active, new, std)
        active &= ~same
        if not active.any():
            break
    return std


def _python_max_first(a: NDArray) -> NDArray:
    """Builtin ``max`` over the last axis: replace only when ``x > current``."""
    cur = a[..., 0]
    for b in range(1, a.shape[-1]):
        x = a[..., b]
        with np.errstate(invalid="ignore"):
            cur = np.where(x > cur, x, cur)
    return cur


def cr_mask_fast(
    diff: NDArray, raw1: NDArray, raw2: NDArray, geom: CrGeometry, *,
    ndr1: int, ndr2: int, abba: bool,
    threshold: float = 10.0, max_sigma: float = 20.0, varatio: float = 2.0,
    slitposratio: float = 1.5, fixsigma: bool = False,
    gain: float = _GAIN_E_PER_ADU, clipsigma: float = 5.0, iteration: int = 3,
    sigstep: float = 2.0, stats: dict | None = None,
) -> NDArray[np.int16]:
    """Fast :func:`decanter.utils.cosmic_ray.detect_cosmic_rays` (mask only;
    ``stats`` receives WARP's returned ``pixnum``/``sigthres``, and with
    ``stats["arrays"]`` set also the whole-frame arrays WARP saves:
    ``mf1``/``mf2`` (``diffmf1Sub``/``diffmf2Sub``) and ``smoothed`` (the
    noise model's median-filtered ``|raw1 + raw2|``)).

    Bit-identical to warp-reference mode and WARP. WARP equivalent:
    ``warp/badpixmask.py:115``.
    """
    xlim1, xlim2 = _SLIT_WINDOW
    if threshold > max_sigma:
        threshold = max_sigma
    s_idx = geom.s_idx
    keep = {} if (stats is not None and stats.get("arrays")) else None
    dres2d = median_filter_pair(diff, keep)
    dres = dres2d.ravel()
    r1pos = median_filter_pair(raw1).ravel()[s_idx] > 0
    r2pos = median_filter_pair(raw2).ravel()[s_idx] > 0
    dres_s = dres[s_idx]
    dpos = dres_s > 0
    dabs = np.absolute(dres_s)

    # WARP's noise model in the raw dtype (float32), left to right.
    summed = np.absolute(raw1 + raw2)
    smoothed = median_slant10(summed)
    smoothed_s = smoothed.ravel()[s_idx]
    noise_s = (smoothed_s * gain + _READ_NOISE_BY_NDR[ndr1] ** 2.
               + _READ_NOISE_BY_NDR[ndr2] ** 2.) ** 0.5 / gain

    vals = dres[geom.t_idx]
    std_tile = _tile_clip_std(vals, geom, clipsigma=clipsigma, iteration=iteration)
    noise_tile = segment_mean(noise_s[geom.t_in_s], geom.t_starts, geom.t_lens)
    shp = (geom.n_orders * geom.n_windows, geom.n_slit_bins)
    std_max = _python_max_first(std_tile.reshape(shp))
    noise_max = _python_max_first(noise_tile.reshape(shp))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = std_max / noise_max
        factor = np.where(ratio > 1., ratio, ratio.dtype.type(1.))   # max(1., ratio)
    factor_s = factor[geom.s_fidx]

    attributable = (dpos & r1pos) | (~dpos & r2pos)
    sig = threshold
    pix_num = 0
    cr = np.zeros(s_idx.shape, dtype=bool)
    while sig <= max_sigma:
        cr = (dabs > sig * noise_s * factor_s) & attributable
        slit = geom.s_slit[cr & geom.s_bp0]
        pix_num = int(slit.size)
        if pix_num == 0:
            break
        hist, _ = np.histogram(slit, bins=20, range=(xlim1, xlim2))
        with np.errstate(invalid="ignore", divide="ignore"):
            varave = np.var(hist) / np.average(hist)
        if abba:
            alo, ahi = _NOD_POSITIONS["A"]
            blo, bhi = _NOD_POSITIONS["B"]
            n_a = int(((slit > alo) & (slit < ahi)).sum())
            n_b = int(((slit > blo) & (slit < bhi)).sum())
            sp_ratio = (max(n_a, n_b) / pix_num * (xlim2 - xlim1)
                        / max(ahi - alo, bhi - blo))
        else:
            olo, ohi = _NOD_POSITIONS["O"]
            n_o = int(((slit > olo) & (slit < ohi)).sum())
            sp_ratio = n_o / pix_num * (xlim2 - xlim1) / (ohi - olo)
        if (sp_ratio > slitposratio) and (varave > varatio) and not fixsigma:
            sig += sigstep
            continue
        break
    mask = np.zeros(geom.shape[0] * geom.shape[1], dtype=np.int16)
    mask[s_idx[cr]] = 1
    if stats is not None:
        stats["pixnum"] = pix_num
        stats["sigthres"] = sig
        if keep is not None:
            stats.update(mf1=keep["mf1"], mf2=dres2d, smoothed=smoothed)
    return mask.reshape(geom.shape)
