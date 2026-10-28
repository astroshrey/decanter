"""Bit-exact port of IRAF ``proto.fixpix`` (``linterp=INDEF, cinterp=INDEF``).

WARP runs ``iraf.fixpix(image, mask, linterp="INDEF", cinterp="INDEF")``
(``warp/badpixmask.py:pyfixpix``). The task is ``pkg/proto/t_fixpix.x``
driving ``pkg/xtools/fixpix/xtfixpix.x:xt_fpinit`` (choose the
interpolation axis) and ``xtfp.gx:xt_fpsr`` (interpolate, REAL images);
the mask is read by ``ytpmmap.x:yt_pmimmap``. This module reproduces the
compiled object code of those routines (iraf-community 2.18.1, clang -O2,
arm64) bit for bit; it was validated against the real ``xtfixpix.o`` /
``xtfp.o`` driven through ``scripts/iraf_oracle/fixpix_oracle.c``.

Algorithm (as IRAF does it)
---------------------------
Mask. A non-pixel-list mask image is read as INTEGER (Fortran
``int(real)``: truncation toward zero) and negative values are set to 0;
any value > 0 is bad. The mask values themselves do not matter when both
interpolation codes are INDEF.

Axis. ``xt_fpsinterp`` replaces every bad pixel by ``nc + n_line``, the
length of its run of bad pixels along the line (the image row). ``xt_fpinit``
then walks every column and compares that against the length ``n_col`` of the
pixel's run along the column: ``n_line > n_col`` selects column
interpolation, otherwise (ties included) line interpolation.

Interpolation (REAL arithmetic). For a line run with good end points at
columns ``c1 < i < c2`` (1-based; ``c1 = 0`` / ``c2 = nc + 1`` when the run
touches the edge)::

    a = data[c1]                  (data[c2] if c1 < 1)
    b = (data[c2] - a) / (c2 - c1)   (0 if c2 > nc)      # float32 / float(int)
    val = a + b * (i - c1)        -> one fused multiply-add: fmaf(b, i-c1, a)

and for a column run between good lines ``l1 < line < l2``::

    c = data[l1]                  (data[l2] if l1 < 1)
    if l2 <= nl: d = (data[l2] - c) / (l2 - l1); val = fmaf(d, line - l1, c)
    else:        val = c

The FMA pairing is from ``objdump -d`` of ``xtfp.o`` (``fsub; scvtf; fdiv;
scvtf; fmadd`` for both branches) and confirmed by the oracle. End points are
always good pixels, so the result depends only on the input image (no
cascading through already-fixed pixels).

Degenerate IRAF behaviour (never met by real WINERED masks; warned about):

- a CVAL (column-interpolated) pixel whose whole column is bad is not
  assigned: it receives the *stale* ``val`` of the previously fixed pixel in
  raster order (a static local in the compiled code; 0 at process start).
  That needs ``n_line > nl``, impossible for a square frame. Emulated.
- a line-interpolated pixel in a fully bad line: the first such pixel of the
  run is left unchanged, later ones read past the end of IRAF's line buffer
  (unreproducible garbage). Needs ``n_col >= nc``: the whole frame bad along
  both axes. Those pixels are left unchanged here.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from decanter.utils.fpemu import fma32

_f32 = np.float32


def iraf_bad_pixels(mask: NDArray) -> NDArray[np.bool_]:
    """Bad-pixel boolean map the way IRAF reads a mask image.

    ``yt_pmimmap`` reads the mask image as INTEGER (``int(real)`` truncates
    toward zero), sets negatives to 0, and stores it in a pixel mask; any
    value > 0 is a bad pixel. Boolean input is taken as-is.
    """
    m = np.asarray(mask)
    if m.dtype == np.bool_:
        return m.copy()
    if np.issubdtype(m.dtype, np.floating):
        m = np.trunc(m)
    return m > 0


def _runs(ys: NDArray[np.intp], xs: NDArray[np.intp]):
    """Last good / first good positions bracketing each bad pixel's run.

    ``(ys, xs)`` are bad-pixel coordinates sorted row-major (``xs`` along
    the run axis). Returns 0-based ``(lo, hi)``: the good pixel before the
    run (``-1`` at the edge) and after it (the axis length at the edge).
    """
    nb = ys.size
    brk = np.ones(nb, dtype=bool)
    if nb > 1:
        brk[1:] = (ys[1:] != ys[:-1]) | (xs[1:] != xs[:-1] + 1)
    starts = np.flatnonzero(brk)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:] - 1
    ends[-1] = nb - 1
    run_id = np.cumsum(brk) - 1
    lo = xs[starts][run_id] - 1
    hi = xs[ends][run_id] + 1
    return lo, hi


def fixpix(
    image: NDArray[np.floating],
    mask: NDArray[np.integer | np.floating | np.bool_],
) -> NDArray[np.float32]:
    """IRAF ``fixpix(image, mask, linterp=INDEF, cinterp=INDEF)``, bit exact.

    Args:
        image: 2-D image (``[line, column]``); computed as REAL (float32),
            as IRAF does for a TY_REAL image.
        mask: 2-D mask of the same shape. Boolean, or mask-image values
            interpreted as IRAF does (see :func:`iraf_bad_pixels`).

    Returns:
        A new float32 image with the bad pixels replaced. The input is not
        modified.
    """
    data = np.asarray(image, dtype=_f32)
    if data.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {data.shape}")
    if data.shape != np.shape(mask):
        raise ValueError(
            f"image shape {data.shape} does not match mask shape {np.shape(mask)}"
        )
    out = data.copy()
    bad = iraf_bad_pixels(mask)
    if not bad.any():
        return out
    nl, nc = data.shape

    # Bad pixels in raster (row-major) order: the order IRAF fixes them in.
    ys, xs = np.nonzero(bad)
    # Line runs (along x within a row).
    c1, c2 = _runs(ys, xs)
    # Column runs (along y within a column): work column-major, map back.
    xs_c, ys_c = np.nonzero(bad.T)
    l1_c, l2_c = _runs(xs_c, ys_c)
    index = np.empty(bad.shape, dtype=np.intp)
    index[ys, xs] = np.arange(ys.size)
    perm = index[ys_c, xs_c]
    l1 = np.empty_like(l1_c)
    l2 = np.empty_like(l2_c)
    l1[perm] = l1_c
    l2[perm] = l2_c

    use_col = (c2 - c1 - 1) > (l2 - l1 - 1)       # n_line > n_col (ties: line)
    val = np.zeros(ys.size, dtype=_f32)
    assigned = np.ones(ys.size, dtype=bool)

    # --- line interpolation (xt_fpsr, FP_LVAL branch) ---------------------
    L = ~use_col
    if L.any():
        y, x, a1, a2 = ys[L], xs[L], c1[L], c2[L]
        left = a1 >= 0
        right = a2 < nc
        a = data[y, np.where(left, a1, np.minimum(a2, nc - 1))]
        vr = data[y, np.minimum(a2, nc - 1)]
        with np.errstate(invalid="ignore", over="ignore"):
            b = np.where(right, (vr - a) / (a2 - a1).astype(_f32), _f32(0.0))
            v = fma32(b, (x - a1).astype(_f32), a)
        whole = ~left & ~right
        if whole.any():
            warnings.warn(
                f"fixpix: {int(whole.sum())} line-interpolated pixel(s) in "
                "fully bad lines; IRAF leaves the first unchanged and reads "
                "past its buffer for the rest -- left unchanged here",
                RuntimeWarning, stacklevel=2,
            )
            v = np.where(whole, data[y, x], v)
            idx = np.flatnonzero(L)[whole]
            assigned[idx] = False
        val[L] = v

    # --- column interpolation (xt_fpsr, FP_CVAL branch) -------------------
    C = use_col
    if C.any():
        y, x, b1, b2 = ys[C], xs[C], l1[C], l2[C]
        top = b1 >= 0
        bottom = b2 < nl
        c = data[np.where(top, b1, np.minimum(b2, nl - 1)), x]
        vb = data[np.minimum(b2, nl - 1), x]
        with np.errstate(invalid="ignore", over="ignore"):
            d = (vb - c) / (b2 - b1).astype(_f32)
            v = np.where(bottom, fma32(d, (y - b1).astype(_f32), c), c)
        whole = ~top & ~bottom
        if whole.any():
            warnings.warn(
                f"fixpix: {int(whole.sum())} column-interpolated pixel(s) in "
                "fully bad columns; IRAF assigns them the stale value of the "
                "previously fixed pixel (emulated)",
                RuntimeWarning, stacklevel=2,
            )
            idx = np.flatnonzero(C)[whole]
            assigned[idx] = False
        val[C] = v

    if not assigned.all():
        # Stale ``val``: carry the last assigned value forward in raster order
        # (0 before the first assignment, as a fresh IRAF process has it).
        # Line-run "whole" pixels are left at the input value instead.
        last = np.where(assigned, np.arange(ys.size), -1)
        np.maximum.accumulate(last, out=last)
        stale = ~assigned & use_col
        src = last[stale]
        val[stale] = np.where(src >= 0, val[np.maximum(src, 0)], _f32(0.0))
        keep = ~assigned & ~use_col
        val[keep] = data[ys[keep], xs[keep]]

    out[ys, xs] = val
    return out
