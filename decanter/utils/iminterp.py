"""IRAF ``math/iminterp`` port: POLY5 fit + segment integration.

This module reproduces IRAF's 1-D image-interpolation routines used by
``noao.onedspec.dispcor`` (and indirectly by ``scombine``). The
algorithms here are direct ports of:

  - ``asifit`` POLY5 branch (``math/iminterp/asifit.x:136``) — pads the
    input with linear reflection at both ends.
  - ``ii_getpcoeff`` POLY5 branch (``math/iminterp/ii_1dinteg.x:215``)
    — Newton's-form conversion of the local 6-point window to
    polynomial coefficients in ``deltax = x - j`` for segment ``[j, j+1]``.
  - ``asigrl`` higher-order branch (``math/iminterp/asigrl.x:142``) —
    segment-by-segment integration of the local polynomial.

All arithmetic runs in ``float32`` (IRAF ``real``) so the float-precision
roundoff matches IRAF byte-for-byte.

Conventions
-----------
- ``asi`` coordinates are 1-based fractional positions in the array of
  length ``Nfit`` passed to ``asifit``. Valid range ``[1, Nfit]``.
- Segment ``j`` (1-based) covers ``[j, j+1]`` in asi coordinates and
  uses the local 6-point window ``coeff[j-1 .. j+4]`` (0-based numpy
  indexing into the padded coefficient array of length ``Nfit + 5``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

REAL = np.float32


def asifit_poly5(data: NDArray) -> NDArray[np.float32]:
    """Pad data per ``asifit`` POLY5 (2 left + 3 right slots, reflection).

    Mirrors ``math/iminterp/asifit.x:136``:

        coeff[c0ptr+1] = 2 * d[1] - d[3]
        coeff[c0ptr+2] = 2 * d[1] - d[2]
        coeff[cnptr+1] = 2 * d[N] - d[N-1]
        coeff[cnptr+2] = 2 * d[N] - d[N-2]
        coeff[cnptr+3] = 2 * d[N] - d[N-3]

    Returns a float32 array of length ``data.size + 5`` with the original
    data living at indices ``[2 .. 2 + N - 1]`` (asi ``OFFSET = 2``).
    """
    d = np.asarray(data, dtype=REAL)
    if d.size < 6:
        raise ValueError("asifit_poly5 requires at least 6 input points")
    n = d.size
    out = np.empty(n + 5, dtype=REAL)
    out[2 : 2 + n] = d
    out[0] = REAL(2) * d[0] - d[2]
    out[1] = REAL(2) * d[0] - d[1]
    out[n + 2] = REAL(2) * d[-1] - d[-2]
    out[n + 3] = REAL(2) * d[-1] - d[-3]
    out[n + 4] = REAL(2) * d[-1] - d[-4]
    return out


def per_segment_pcoeff_poly5(coeff: NDArray) -> NDArray[np.float32]:
    """Build the per-segment polynomial coefficients for POLY5.

    Mirrors ``math/iminterp/ii_1dinteg.x:215`` (``ii_getpcoeff`` POLY5
    branch), vectorized over all valid segments.

    Args:
        coeff: float32 padded coefficient array of length ``Nfit + 5``
            (output of :func:`asifit_poly5` on an asi array of length
            ``Nfit``).

    Returns:
        ``pcoeff`` of shape ``(6, Nfit - 1)``: for each segment ``j``
        (0-based ``j_idx = j - 1``, where IRAF ``j`` ∈ ``[1, Nfit - 1]``),
        ``pcoeff[i, j_idx]`` is the coefficient of ``deltax**i`` in the
        local poly5 expansion, where ``deltax = x - j`` for ``x`` in
        ``[j, j+1]`` (asi coords).
    """
    c = np.asarray(coeff, dtype=REAL)
    n_total = c.size
    # ``Nfit`` (the size passed to asifit) is ``n_total - 5``. Valid
    # segments j ∈ [1, Nfit-1]; that's ``Nfit - 1 = n_total - 6`` of them.
    n_seg = n_total - 6
    if n_seg < 1:
        raise ValueError("coeff array too small")

    # IRAF: diff[i] = coeff[index - 3 + i] for i = 1..6 with index = 2 + j.
    # That's coeff[j - 1 + i] for i = 1..6, i.e. coeff[j .. j+5] (1-based).
    # In 0-based numpy: coeff[j-1 .. j+4]. Build (6, n_seg) where
    # row r corresponds to offset r within the 6-point window.
    diff = np.empty((6, n_seg), dtype=REAL)
    for r in range(6):
        diff[r] = c[r : r + n_seg]

    # Divided-difference table (k = 1..5 in IRAF 1-based).
    for k in range(1, 6):
        for i in range(6 - k):
            diff[i] = (diff[i + 1] - diff[i]) / REAL(k)

    # Newton-to-power-basis shift: IRAF nests
    #   for k = 6..2 step -1:
    #       for i = 2..k:
    #           diff[i] = diff[i] + diff[i-1] * (k - i - 3)
    # so that after the loop, diff[1..6] hold the polynomial in
    # (deltax)^(6-i). We then reverse into pcoeff so pcoeff[r] is the
    # coefficient of deltax**r.
    for k1 in range(6, 1, -1):
        for i1 in range(2, k1 + 1):
            shift = REAL(k1 - i1 - 3)
            diff[i1 - 1] = diff[i1 - 1] + diff[i1 - 2] * shift

    return diff[::-1].copy()


def asigrl_poly5(
    pcoeff: NDArray[np.float32], a: NDArray, b: NDArray
) -> NDArray[np.float32]:
    """Integrate the POLY5 interpolant from ``a`` to ``b`` (asi coords).

    Vectorized port of ``math/iminterp/asigrl.x:142`` (higher-order
    branch) and ``ii_1dinteg.x:142``.

    Args:
        pcoeff: per-segment poly coefficients from
            :func:`per_segment_pcoeff_poly5`, shape ``(6, n_seg)``.
        a, b: float arrays of equal shape. The integral is from
            ``min(a,b)`` to ``max(a,b)``; the returned value is signed
            to match IRAF (negative when ``a > b``).

    Returns:
        Float32 integrals, same shape as ``a``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("a and b must broadcast to the same shape")
    n_seg = pcoeff.shape[1]

    # IRAF swaps so xa <= xb, accumulates, then negates if needed.
    swap = a > b
    xa = np.where(swap, b, a).astype(REAL)
    xb = np.where(swap, a, b).astype(REAL)

    # IRAF: neara = xa (truncation toward zero in fortran for positives).
    # For our positive-only asi coords this matches np.floor.
    neara = np.floor(xa).astype(np.int64)
    nearb = np.floor(xb).astype(np.int64)
    deltaxa = (xa - neara.astype(REAL)).astype(REAL)
    deltaxb = (xb - nearb.astype(REAL)).astype(REAL)

    # Clamp segment indices to valid range. Out-of-range entries will be
    # masked out by the dispcor `ofb` machinery before this function sees
    # them, but we still clamp defensively to keep indexing safe.
    neara_idx = np.clip(neara - 1, 0, n_seg - 1)
    nearb_idx = np.clip(nearb - 1, 0, n_seg - 1)

    accum = np.zeros(xa.shape, dtype=REAL)

    # ----- Case A: neara == nearb (single segment) -----
    single = neara == nearb
    if np.any(single):
        idx = neara_idx[single]
        da = deltaxa[single]
        db = deltaxb[single]
        s = np.zeros(da.shape, dtype=REAL)
        for i in range(6):  # IRAF i = 1..6 → 0..5 here; power = i+1
            i1 = REAL(i + 1)
            term = pcoeff[i, idx] * (db ** REAL(i + 1) - da ** REAL(i + 1)) / i1
            s = s + term.astype(REAL)
        accum[single] = s

    # ----- Case B: neara < nearb (multi-segment) -----
    multi = ~single
    if np.any(multi):
        idx_a = neara_idx[multi]
        idx_b = nearb_idx[multi]
        da = deltaxa[multi]
        db = deltaxb[multi]
        one = REAL(1)
        s = np.zeros(da.shape, dtype=REAL)

        # First segment [xa, neara+1]: integral from deltaxa to 1 of
        #   sum_i pcoeff[i, neara] * deltax**i
        # = sum_i pcoeff[i, neara] * (1 - deltaxa**(i+1)) / (i+1)
        for i in range(6):
            i1 = REAL(i + 1)
            s = s + (pcoeff[i, idx_a] * (one - da ** REAL(i + 1)) / i1).astype(REAL)

        # Middle segments: contribute sum_i pcoeff[i, j] / (i+1) for each
        # whole segment j in (neara, nearb).
        # Precompute per-segment "full integral" once across all segments.
        full_int = np.zeros(n_seg, dtype=REAL)
        for i in range(6):
            full_int = full_int + (pcoeff[i] / REAL(i + 1)).astype(REAL)

        # Cumulative sum so the middle-segment contribution for each
        # output is a simple difference: sum_{j=neara+1..nearb-1} full_int[j-1].
        cum = np.zeros(n_seg + 1, dtype=np.float64)
        cum[1:] = np.cumsum(full_int.astype(np.float64))
        # IRAF semantics: middle range is j = neara + 1 .. nearb - 1.
        # 0-based segment indices: neara_idx + 1 .. nearb_idx - 1.
        # Sum = cum[nearb_idx] - cum[neara_idx + 1].
        # Use float64 cum then cast to REAL to keep the accumulation
        # close to IRAF's running sum (which is in float32, but the
        # difference at this scale is well below the float32 epsilon of
        # the result).
        mid = (cum[idx_b] - cum[idx_a + 1]).astype(REAL)
        s = s + mid

        # Last segment [nearb, xb]: integral from 0 to deltaxb of
        #   sum_i pcoeff[i, nearb] * deltax**i
        # = sum_i pcoeff[i, nearb] * deltaxb**(i+1) / (i+1)
        for i in range(6):
            i1 = REAL(i + 1)
            s = s + (pcoeff[i, idx_b] * db ** REAL(i + 1) / i1).astype(REAL)

        accum[multi] = s

    # Sign convention: IRAF returns -accum when a > b.
    return np.where(swap, -accum, accum).astype(REAL)


def dispcor_linear_poly5(
    input_data: NDArray,
    pixel_at_wavelength: callable,
    w_min: float,
    w_max: float,
    n_out: int,
    flux: bool = False,
) -> NDArray[np.float32]:
    """Reproduce IRAF ``dispcor(flux=NO, linear=YES, dw=INDEF)`` for one row.

    Mirrors ``noao/onedspec/dispcor/dispcor.x``:

      1. Pad input with edge duplication: ``temp[1] = in[1]``,
         ``temp[2..N+1] = in[1..N]``, ``temp[N+2] = in[N]``.
      2. ``asifit`` POLY5 on ``temp`` (further 2-/3-slot reflection pad).
      3. For each output pixel i (i = 1..n_out), compute the half-width
         output edges in input-pixel coordinates via ``pixel_at_wavelength``,
         clip to ``[0.5, N+0.5]``, shift by +1 into asi coordinates, then
         integrate via ``asigrl_poly5``.
      4. For ``flux=False`` (the WARP default), divide by ``max(b-a, 1e-4)``.

    The output WCS is linear: ``w_out(i) = w_min + (i-1) * (w_max-w_min)/(n_out-1)``.

    Args:
        input_data: 1-D input spectrum, length ``N``. Will be cast to float32.
        pixel_at_wavelength: callable mapping a float64 array of target
            wavelengths to fractional 1-based input pixel positions
            (e.g. :func:`decanter.io.iraf_id.pixels_at_wavelength`).
        w_min, w_max: output WCS endpoints in the same units as the
            id-file solution (typically Angstrom or nm).
        n_out: output length.
        flux: ``True`` to conserve flux (raw integral), ``False`` (default)
            to return the average value over each output pixel.

    Returns:
        Float32 array of length ``n_out``.
    """
    in32 = np.asarray(input_data, dtype=REAL)
    n = in32.size
    if n < 6:
        raise ValueError("dispcor_linear_poly5 requires at least 6 input pixels")
    if n_out < 1:
        raise ValueError("n_out must be positive")

    # Step 1+2: dispcor's edge-duplicate pad, then asifit reflection pad.
    temp = np.empty(n + 2, dtype=REAL)
    temp[0] = in32[0]
    temp[1 : 1 + n] = in32
    temp[n + 1] = in32[-1]
    coeff = asifit_poly5(temp)
    pcoeff = per_segment_pcoeff_poly5(coeff)

    # Step 3: build the (n_out + 1) output-edge wavelengths and invert.
    # Output WCS: w_out(x_out) = w_min + (x_out - 1) * dw. The first
    # output edge is x_out = 0.5, then 1.5, 2.5, ..., n_out + 0.5.
    dw = (w_max - w_min) / (n_out - 1) if n_out > 1 else 1.0
    edge_x = np.arange(n_out + 1, dtype=np.float64) + 0.5  # 0.5, 1.5, ..., n_out + 0.5
    edge_w = w_min + (edge_x - 1.0) * dw

    # Inverse: λ → fractional input pixel (1-based, in *original* coords).
    x_orig_unclipped = pixel_at_wavelength(edge_w)
    xmin_orig = 0.5
    xmax_orig = float(n) + 0.5
    ofb = (x_orig_unclipped < xmin_orig) | (x_orig_unclipped > xmax_orig)
    x_orig = np.clip(x_orig_unclipped, xmin_orig, xmax_orig)
    # Shift into asi coords (dispcor's `+ 1`).
    edge_asi = (x_orig + 1.0).astype(REAL)

    a = edge_asi[:-1]
    b = edge_asi[1:]
    ofb_a = ofb[:-1]
    ofb_b = ofb[1:]
    both_ofb = ofb_a & ofb_b

    # IRAF dispcor always invokes asigrl with the smaller argument first
    # (dispcor.x:86 vs :98), so the integral is taken over the absolute
    # interval. For flux=NO, divide by the positive width. (The signed
    # version of asigrl exists for flux=YES on signed-direction inputs,
    # but dispcor itself never relies on the sign — see dispcor.x.)
    xa = np.minimum(a, b)
    xb = np.maximum(a, b)
    integ = asigrl_poly5(pcoeff, xa, xb)
    if flux:
        out = integ.copy()
    else:
        width = (xb - xa).astype(REAL)
        width = np.maximum(width, REAL(1e-4))
        out = integ / width

    out = np.where(both_ofb, REAL(0), out).astype(REAL)
    return out


def scombine_linear_poly5(
    input_data: NDArray,
    crval1_in: float,
    cdelt1_in: float,
    w1_out: float,
    dw_out: float,
    nw_out: int,
    flux: bool = False,
) -> NDArray[np.float32]:
    """Reproduce IRAF ``scombine`` single-input resample for a linear input WCS.

    Mirrors ``noao/onedspec/smw/shdr.x:shdr_linear`` (DCLINEAR branch),
    which is the resample path scombine takes for a linear-axis input
    spectrum. The algorithm is identical to :func:`dispcor_linear_poly5`
    except the wavelength→pixel inverse is the trivial linear formula
    ``x = (w - crval1_in) / cdelt1_in + 1``.

    This is the resample step in WARP's ``PySpecshift``
    (``ccwaveshift.py:121``): ``iraf.scombine(..., w1=1., dw=cdelt1, nw=naxis1)``
    rebins a wavelength-shifted (via ``iraf.specshift``) input onto a
    fresh linear grid.

    Args:
        input_data: 1-D input spectrum.
        crval1_in, cdelt1_in: input WCS — wavelength at pixel 1 and per-pixel step.
        w1_out, dw_out, nw_out: output WCS — starting wavelength, step, length.
        flux: ``True`` to keep the raw integral, ``False`` (default) for the
            pixel-average (matches scombine's behavior for `flux=NO`, which is
            the WARP setting).

    Returns:
        Float32 array of length ``nw_out``.
    """
    if cdelt1_in == 0:
        raise ValueError("cdelt1_in must be non-zero")

    in_arr = np.asarray(input_data, dtype=REAL)
    nw_in = in_arr.size
    # IRAF shdr_linear (shdr.x:1034) short-circuits when the input WCS
    # already matches the output WCS — same start, end, length, dispersion
    # type. The check uses single-precision `fp_equalr` (~ float32 epsilon
    # of the values). For our linear-only path:
    #   W0 = CRVAL1_in, W1 = CRVAL1_in + CDELT1_in * (NAXIS1 - 1)
    #   w0 = w1_out,   w1 = w1_out + dw_out * (nw_out - 1)
    if nw_in == int(nw_out):
        w0_in = float(crval1_in)
        w1_in = float(crval1_in) + float(cdelt1_in) * (nw_in - 1)
        w0_out = float(w1_out)
        w1_out_end = float(w1_out) + float(dw_out) * (int(nw_out) - 1)
        scale = max(
            abs(w0_in), abs(w1_in), abs(w0_out), abs(w1_out_end), 1.0
        )
        # float32 epsilon ~ 1.2e-7; allow a few-ULP tolerance.
        if (abs(w0_in - w0_out) <= 5e-7 * scale and
                abs(w1_in - w1_out_end) <= 5e-7 * scale):
            return in_arr.copy()

    def _inverse(w: NDArray) -> NDArray[np.float64]:
        return (np.asarray(w, dtype=np.float64) - crval1_in) / cdelt1_in + 1.0

    w_max = w1_out + (nw_out - 1) * dw_out if nw_out > 1 else w1_out + dw_out
    return dispcor_linear_poly5(
        input_data, _inverse, float(w1_out), float(w_max), int(nw_out), flux=flux
    )
