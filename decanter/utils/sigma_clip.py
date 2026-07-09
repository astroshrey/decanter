"""Vectorized per-segment sigma clipping via ``np.bincount`` segment reductions.

Implements the 5-step recipe from PLAN_FULL.md §"Cosmic-ray notes".
Used by:

- s02 (cosmic ray) — per ``(echelle order × y-tile × slit-bin)`` segment.
- s14 (continuum)  — iterative reject in the spline fit (different
  variant; takes residuals instead of raw values).
- s10 (waveshift)  — sigma-clipped median of per-order shifts.

The recipe uses ``np.bincount`` (which ports cleanly to
``jnp.bincount`` / ``jax.ops.segment_sum`` for the eventual JAX swap)
rather than ``np.add.at`` (unbuffered scatter, awkward to vectorize) or
``scipy.ndimage.labeled_comprehension`` (Python callback per label —
defeats the speedup).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def per_segment_clip(
    values: NDArray[np.floating],
    segment_ids: NDArray[np.integer],
    *,
    n_segments: int,
    sigma: float = 5.0,
    iterations: int = 3,
    invalid_id: int = -1,
    center: str = "mean",
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.int_]]:
    """Iteratively sigma-clip ``values`` within each segment.

    For each pass, the per-segment mean and standard deviation are
    re-computed using only the currently-valid pixels; pixels whose
    deviation from the chosen ``center`` exceeds ``sigma * std`` are
    dropped before the next pass.

    Args:
        values: 1-D array of per-pixel values (any floating dtype).
        segment_ids: 1-D integer array, same length as ``values``.
            Entries equal to ``invalid_id`` (default ``-1``) are
            permanently excluded.
        n_segments: total number of segments. Sets the length of the
            three returned arrays. Segment IDs outside ``[0, n_segments)``
            are silently ignored.
        sigma: rejection threshold in units of the per-segment std.
        iterations: number of clip-then-refit passes. ``0`` means just
            the initial stats with no rejection (returns mean/std over
            the full segment). ``N`` means N clips, matching WARP's
            ``for k in range(iteration)`` (badpixmask.py:213).
        invalid_id: sentinel value in ``segment_ids`` marking pixels
            that should never participate (e.g., outside any aperture).
        center: rejection-threshold center per pixel:
            * ``"mean"`` (default) — reject ``|val − seg_mean| > σ·std``.
              Standard sigma-clip semantics for arbitrary-mean data.
            * ``"zero"`` — reject ``|val| > σ·std``. Matches IRAF /
              WARP's CR detector convention (``warp/badpixmask.py:215``,
              ``np.absolute(mf_sc) < clipsigma * mfstd``); appropriate
              when the residuals are expected to be zero-mean (e.g.,
              median-filter residuals). The per-segment std itself is
              still computed mean-centered to match ``np.std``.

    Returns:
        ``(mean, std, count)`` arrays of length ``n_segments``. Segments
        whose final valid-pixel count is zero report ``mean = nan``,
        ``std = nan``, ``count = 0``.

    Notes:
        Algorithm (per iteration, fully vectorized):
            1. ``sum_i    = bincount(seg[valid], weights=val[valid], minlength=n)``
            2. ``sumsq_i  = bincount(seg[valid], weights=val[valid]**2, minlength=n)``
            3. ``count_i  = bincount(seg[valid], minlength=n)``
            4. ``mean_i   = sum_i / count_i``;
               ``var_i    = sumsq_i / count_i - mean_i ** 2``
               (variances clipped at 0 to absorb float roundoff).
            5. Broadcast ``mean[seg]``, ``std[seg]`` back to per-pixel arrays;
               update ``valid = |val - mean[seg]| < sigma * std[seg]``.
        No Python loop over segments. Time per iteration is O(N) in the
        number of pixels, not O(n_segments × pixels_per_segment).

    Raises:
        ValueError: if ``values`` and ``segment_ids`` have different shapes,
            ``n_segments`` is non-positive, ``sigma`` is non-positive, or
            ``iterations`` is less than 1.
    """
    if values.shape != segment_ids.shape:
        raise ValueError(
            f"values and segment_ids must share shape; got {values.shape} vs {segment_ids.shape}"
        )
    if n_segments <= 0:
        raise ValueError(f"n_segments must be positive; got {n_segments}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive; got {sigma}")
    if iterations < 0:
        raise ValueError(f"iterations must be ≥ 0; got {iterations}")
    if center not in ("mean", "zero"):
        raise ValueError(f"center must be 'mean' or 'zero'; got {center!r}")

    values_flat = np.ascontiguousarray(values).ravel().astype(np.float64, copy=False)
    seg_flat = np.ascontiguousarray(segment_ids).ravel()

    # Initial validity mask: in-range segment IDs only.
    valid = (seg_flat >= 0) & (seg_flat < n_segments) & (seg_flat != invalid_id)

    def _stats(valid_mask: NDArray[np.bool_]):
        """Compute per-segment count/mean/std on currently-valid pixels."""
        v = values_flat[valid_mask]
        s = seg_flat[valid_mask]
        c = np.bincount(s, minlength=n_segments).astype(np.int64)
        sums = np.bincount(s, weights=v, minlength=n_segments)
        sumsq = np.bincount(s, weights=v * v, minlength=n_segments)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(c > 0, sums / c, np.nan)
            var = np.where(c > 0, sumsq / c - m * m, np.nan)
        var = np.where(np.isfinite(var), np.maximum(var, 0.0), np.nan)
        return c, m, np.sqrt(var)

    # Match WARP's exact loop order (badpixmask.py:212-219):
    #   mfstd = np.std(mf_sc)              # initial std from ALL pixels
    #   for k in range(iteration):
    #       mfstd_last = mfstd
    #       mf_req = |mf_sc| < clipsigma * mfstd      ← clip using CURRENT std
    #       mfstd = np.std(mf_sc[mf_req])             ← recompute AFTER clip
    #       if mfstd_last == mfstd: break
    # So `iterations=N` means N clip operations and N+1 std computations.
    # The returned std is post-last-clip, not pre-clip.
    count, mean, std = _stats(valid)

    for _ in range(iterations):
        seg_mean = mean[seg_flat]
        seg_std = std[seg_flat]
        with np.errstate(invalid="ignore"):
            if center == "mean":
                deviation = np.abs(values_flat - seg_mean)
            else:  # "zero" — WARP CR convention
                deviation = np.abs(values_flat)
            # Two cases to keep: (a) std > 0 and deviation < sigma * std, or
            # (b) std == 0 and deviation == 0. The std == 0 path guards against
            # constant-data segments collapsing to count=0 (the threshold
            # 0 < 0 is False, so every pixel would otherwise be rejected).
            keep = ((seg_std > 0) & (deviation < sigma * seg_std)) | (
                (seg_std == 0) & (deviation == 0)
            )
        new_valid = valid & keep
        new_count, new_mean, new_std = _stats(new_valid)
        # WARP early-stop: `if mfstd_last == mfstd: break`. We replicate
        # element-wise; if NO segment changed its std this pass, stop.
        changed = ~np.isclose(new_std, std, equal_nan=True)
        valid = new_valid
        count, mean, std = new_count, new_mean, new_std
        if not changed.any():
            break

    return mean, std, count
