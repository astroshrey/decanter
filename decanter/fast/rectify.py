"""Fast s06: IRAF ``transform`` (spline3) with cached geometry, batched solves.

Warp mode (:mod:`decanter.rectify.iraf_transform`) reproduces IRAF
``longslit.transform`` operation for operation, in float32 with emulated fused
multiply-adds, for every order of every frame. Fast mode keeps IRAF's
*algorithm* -- the same inverse map, the same sliding buffer of input lines
(``NBUF``/``NEDGE`` windows, each with its own natural-spline end conditions),
the same bicubic B-spline -- but

1. everything that does not depend on the image is computed once per
   calibration set and cached: the transform plan and line schedule (shared
   with warp mode), the inverse coordinates of every output pixel, the
   B-spline tap positions and fractions, and the layout of the batched solves;
2. ``msifit(II_BISPLINE3)`` is done as two batched natural-spline solves
   (``ii_spline2d`` along the cut rows, then along each window's needed
   columns) over all orders and images at once;
3. only the *band* of each strip that the extraction reads is evaluated: the
   multihole reference trace +- the reference aperture +- :data:`BAND_MARGIN`
   columns (the WARP per-frame and center-search apertures lie inside the
   reference window). Pixels outside the band are NaN, so an out-of-band read
   cannot go unnoticed. Full strips are produced on request.

The arithmetic is warp mode's (IRAF REAL with its fused multiply-adds): the
NumPy backend rounds every fused step exactly once (float64 sum of an exact
float32 product, float32 midpoints resolved with the two-sum error), so its
strips are bit-identical to warp mode's by construction. The MLX backend runs
the same operations as Metal kernels; the Metal compiler contracts them into
the same FMAs, which has given bit-identical band values in every comparison
so far (observed, not guaranteed).

IRAF: ``noao/twodspec/longslit/transform/{trsetup.x,t_transform.x}``,
``math/iminterp/{msifit.x,ii_spline2d.x,ii_bieval.x}``.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from decanter.calib.aperture import ApertureSet
from decanter.calib.transform import RectifiedStrip, strip_from_plan
from decanter.fast import _backend
from decanter.rectify import iraf_transform as T
from decanter.rectify.transform import _cut_limits
from decanter.utils.spline3 import (
    _TIE_HALF,
    _TIE_MASK,
    _bspline_weights,
    _round_sum_into,
    _spline2d_diag,
    bilinear_eval_exact,
)

f32 = np.float32

#: Columns kept on each side of the reference window. IRAF ``apall`` reads up
#: to 3 columns beyond the aperture edges (``ap_sum``'s buffer origin and the
#: linear edge integrals) and the WARP center search 1; 8 leaves room for the
#: per-frame database trace being rounded (``%f``) from the reference.
BAND_MARGIN = 8

#: Upper bound (elements per image) of one batched spline solve; orders are
#: grouped so a group's solve arrays stay below it.
_GROUP_ELEMENTS = 16_000_000


# ---------------------------------------------------------------------------
# Natural-spline solve (ii_spline2d with zero end second derivatives)
# ---------------------------------------------------------------------------

def _diag(n_max: int) -> NDArray[np.float64]:
    """``ii_spline2d``'s elimination factors ``diag[1..n+1]`` (0-based)."""
    d = np.empty(n_max + 1)
    d[0] = -2.0
    d[1] = 0.0
    for k in range(2, n_max + 1):
        d[k] = 1.0 / (4.0 - d[k - 1])
    return d


@dataclass
class SolveLayout:
    """A batch of natural-spline solves, vectors sorted by length (longest first).

    Vector ``j`` takes its ``n[j]`` data points from rows ``base[j] ..
    base[j] + n[j] - 1`` of a ``(S, K)`` source and gets the ``n[j] + 2``
    B-spline coefficients in rows ``0 .. n[j] + 1`` of column ``j`` of an
    ``(L, NV, K)`` result; every other row of the result is zero (IRAF's
    calloc'd padding row/column).
    """

    base: NDArray[np.int64]     # (NV,)
    n: NDArray[np.int64]        # (NV,) non-increasing
    L: int                      # rows of the result (>= max(n) + 3)
    cnt: NDArray[np.int64]      # cnt[k] = number of vectors with n >= k

    @classmethod
    def build(cls, base: NDArray, n: NDArray, L: int) -> SolveLayout:
        n = np.asarray(n, np.int64)
        if n.size and (np.any(np.diff(n) > 0) or n.min() < 2):
            raise ValueError("vectors must be sorted by length (desc), each >= 2 points")
        nmax = int(n[0]) if n.size else 0
        cnt = np.searchsorted(-n, -np.arange(nmax + 2), side="right").astype(np.int64)
        return cls(base=np.asarray(base, np.int64), n=n, L=int(L), cnt=cnt)

    @property
    def nv(self) -> int:
        return int(self.n.size)


def _round_fused(out32: NDArray[np.float32], p: NDArray[np.float64],
                 c64: NDArray[np.float64], s: NDArray[np.float64]) -> None:
    """``out32 = round32(p + c)`` (one rounding: a float32 fma), ``p`` exact.

    ``s`` is scratch. ``p`` is a product of two float32 values (exact in
    float64), so ``fl64(p + c)`` rounds once; only a sum on a float32
    midpoint can then be double-rounded, and those are fixed with the exact
    two-sum error (as :func:`decanter.utils.spline3._round_sum_into`).
    """
    np.add(p, c64, out=s)
    _round_sum_into(out32, s, p, c64)


def _solve_numpy(src: NDArray, lay: SolveLayout) -> NDArray[np.float32]:
    """Batched ``ii_spline2d`` (msifit: end second derivatives 0), IRAF REAL.

    Exactly the arithmetic of :func:`decanter.utils.spline3.spline2d_exact`
    (warp mode): float32 forward elimination with IRAF's float32
    ``diag`` recurrence, the closing step and back substitution as fused
    multiply-adds (``fnmsub``) rounded once -- vectorised over all vectors of
    the layout, so the coefficients are bit-identical to warp mode's.
    """
    K = src.shape[1]
    nv = lay.nv
    R = np.zeros((lay.L, nv, K), f32)
    if nv == 0:
        return R
    src = np.asarray(src, f32)
    n, base, cnt = lay.n, lay.base, lay.cnt
    nmax = int(n[0])
    d = _spline2d_diag(nmax + 1)
    six = f32(6.0)
    # forward elimination, reading the data on the fly (REAL arithmetic):
    # c[1] = (y1 - 0) / 6;  c[k] = diag[k] * (y_k - c[k-1])
    np.divide(src[base], six, out=R[1])
    for k in range(2, nmax + 1):
        c = cnt[k]
        blk = R[k, :c]
        np.subtract(src[base[:c] + (k - 1)], R[k - 1, :c], out=blk)
        blk *= d[k]
    # closing step: c[n+1] = fnmsub(t, c[n], c[n-1]) / fma(diag[n], t, 1)
    # (+ c[n+1]/6 = 0: the end second derivative)
    j = np.arange(nv)
    t = (d[n - 1] + f32(2.0)).astype(f32)
    t64 = t.astype(np.float64)[:, None]
    p = t64 * R[n, j].astype(np.float64)
    cm1 = -R[n - 1, j].astype(np.float64)
    num = np.empty((nv, K), f32)
    _round_fused(num, p, cm1, np.empty_like(p))
    num = num + f32(0.0)
    dn = d[n].astype(np.float64)
    den = np.empty(nv, f32)
    _round_fused(den, dn * t.astype(np.float64), np.ones(nv), np.empty(nv))
    R[n + 1, j] = num / den[:, None]
    # back substitution: c[k] = fnmsub(diag[k], c[k+1], c[k]), rounded once
    pbuf = np.empty((nv, K))
    sbuf = np.empty((nv, K))
    cbuf = np.empty((nv, K))
    for k in range(nmax, 1, -1):
        c = cnt[k]
        pp, ss, cc = pbuf[:c], sbuf[:c], cbuf[:c]
        np.multiply(R[k + 1, :c], -np.float64(d[k]), out=pp)
        np.copyto(cc, R[k, :c])
        _round_fused(R[k, :c], pp, cc, ss)
    # c[0] = (c[0] + 2 c[1]) - c[2], c[0] = 0 / 6
    R[0] = (f32(0.0) + f32(2.0) * R[1]) - R[2]
    return R


_MLX_SOLVE_SRC = """
    uint tid = thread_position_in_grid.x;
    uint NV = (uint)dims[0];
    uint K = (uint)dims[1];
    if (tid >= NV * K) return;
    uint j = tid / K;
    uint kk = tid - j * K;
    int n = nvec[j];
    uint b = (uint)base[j];
    float prev = src[b * K + kk] / 6.0f;
    out[(1 * NV + j) * K + kk] = prev;
    for (int k = 2; k <= n; ++k) {
        float v = dg[k] * (src[(b + (uint)k - 1) * K + kk] - prev);
        out[((uint)k * NV + j) * K + kk] = v;
        prev = v;
    }
    float t = dg[n - 1] + 2.0f;
    float cm1 = out[((uint)(n - 1) * NV + j) * K + kk];
    float nxt = (t * prev - cm1) / (1.0f + dg[n] * t);
    out[((uint)(n + 1) * NV + j) * K + kk] = nxt;
    for (int k = n; k >= 2; --k) {
        uint o = ((uint)k * NV + j) * K + kk;
        float v = out[o] - dg[k] * nxt;
        out[o] = v;
        nxt = v;
    }
    float c1 = out[(1 * NV + j) * K + kk];
    out[j * K + kk] = 2.0f * c1 - nxt;
"""

_MLX_EVAL_SRC = """
    uint p = thread_position_in_grid.x;
    uint P = (uint)dims[0];
    uint NV = (uint)dims[1];
    uint K = (uint)dims[2];
    if (p >= P) return;
    float sx = fx[p];
    float sy = fy[p];
    float tx = 1.0f - sx;
    float ty = 1.0f - sy;
    float bx[4] = {tx * tx * tx, 1.0f + tx * (3.0f + tx * (3.0f - 3.0f * tx)),
                   1.0f + sx * (3.0f + sx * (3.0f - 3.0f * sx)), sx * sx * sx};
    float by[4] = {ty * ty * ty, 1.0f + ty * (3.0f + ty * (3.0f - 3.0f * ty)),
                   1.0f + sy * (3.0f + sy * (3.0f - 3.0f * sy)), sy * sy * sy};
    uint b0 = (uint)idx0[p];
    for (uint kk = 0; kk < K; ++kk) {
        float acc = 0.0f;
        for (uint jj = 0; jj < 4; ++jj) {
            uint r = (b0 + jj * NV) * K + kk;
            float s = bx[0] * coef[r] + bx[1] * coef[r + K] + bx[2] * coef[r + 2 * K]
                      + bx[3] * coef[r + 3 * K];
            acc += by[jj] * s;
        }
        out[p * K + kk] = acc;
    }
"""

_mlx_kernels: dict = {}


def _mlx_kernel(name: str):
    if name not in _mlx_kernels:
        mx = _backend.mx()
        if name == "solve":
            _mlx_kernels[name] = mx.fast.metal_kernel(
                name="decanter_spline_solve",
                input_names=["src", "base", "nvec", "dg", "dims"],
                output_names=["out"], source=_MLX_SOLVE_SRC)
        else:
            _mlx_kernels[name] = mx.fast.metal_kernel(
                name="decanter_bispline3_eval",
                input_names=["coef", "idx0", "fx", "fy", "dims"],
                output_names=["out"], source=_MLX_EVAL_SRC)
    return _mlx_kernels[name]


def _solve_mlx(src, lay: SolveLayout, dev: dict):
    """:func:`_solve_numpy` on the GPU (float32); ``src`` is an mlx (S, K) array."""
    mx = _backend.mx()
    K = int(src.shape[1])
    nv = lay.nv
    if "base" not in dev:
        dev["base"] = mx.array(lay.base.astype(np.int32))
        dev["n"] = mx.array(lay.n.astype(np.int32))
        dev["d"] = mx.array(_spline2d_diag(int(lay.n[0]) + 1 if nv else 2))
    total = nv * K
    out = _mlx_kernel("solve")(
        inputs=[src, dev["base"], dev["n"], dev["d"], mx.array([nv, K], dtype=mx.int32)],
        grid=(max(total, 1), 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(lay.L * nv * K,)], output_dtypes=[mx.float32], init_value=0)[0]
    return out


# ---------------------------------------------------------------------------
# Cached geometry
# ---------------------------------------------------------------------------

@dataclass
class OrderBand:
    """Calibration-only sampling of one order: which strip pixels, and how."""

    order: int
    plan: T.TransformPlan
    xmin: int
    xmax: int
    nxin: int
    pix: NDArray[np.int64]      # flat (v * nu + u) strip index of every band pixel
    # per-pixel tap data; moved into the GroupGeometry (and dropped here)
    win: NDArray[np.int64] | None   # window (line-buffer) index per pixel
    nx: NDArray[np.int64] | None    # IRAF COEFF column of the B-spline centre tap
    ny: NDArray[np.int64] | None    # IRAF COEFF row (window-relative)
    sx: NDArray[np.float32] | None  # fractional x / y (IRAF REAL)
    sy: NDArray[np.float32] | None
    full: bool
    jfac: NDArray[np.float32] | None = None   # flux = yes factor per band pixel (lazy)

    def flux_factor(self) -> NDArray[np.float32]:
        """``transform flux=yes``'s factor at every band pixel: the plan's
        :meth:`decanter.rectify.iraf_transform.TransformPlan.flux_factor`
        (bilinear Jacobian times ``du`` times ``dv``, REAL), picked at the
        band's pixels -- the same values the reference multiplies by."""
        if self.jfac is None:
            plan = self.plan
            v = self.pix // plan.nu
            u = self.pix - v * plan.nu
            j = np.empty(self.pix.size, f32)
            step = 1 << 20
            for i0 in range(0, self.pix.size, step):
                sl = slice(i0, i0 + step)
                j[sl] = bilinear_eval_exact(plan.jcoeff, plan.uout[u[sl]], plan.vout[v[sl]])
            j = (plan.du * j).astype(f32)
            self.jfac = (j * plan.dv).astype(f32)
        return self.jfac


def _order_band(plan: T.TransformPlan, sched: T.LineSchedule, nxin: int,
                lo: NDArray[np.int64] | None, hi: NDArray[np.int64] | None,
                order: int, xmin: int, xmax: int) -> OrderBand:
    """Pixels ``lo[v] .. hi[v]`` (0-based, inclusive) of every output row."""
    nv, nu = plan.nv, plan.nu
    full = lo is None
    if full:
        lo = np.zeros(nv, np.int64)
        hi = np.full(nv, nu - 1, np.int64)
    width = np.maximum(hi - lo + 1, 0)
    v = np.repeat(np.arange(nv, dtype=np.int64), width)
    start = np.repeat(lo, width)
    within = np.arange(v.size, dtype=np.int64) - np.repeat(np.cumsum(width) - width, width)
    u = start + within
    # Inverse coordinates exactly as warp mode computes them (tr_transform).
    xin = np.empty(v.size, f32)
    yin = np.empty(v.size, f32)
    step = 1 << 20
    for i0 in range(0, v.size, step):
        sl = slice(i0, i0 + step)
        xin[sl] = bilinear_eval_exact(plan.xcoeff, plan.uout[u[sl]], plan.vout[v[sl]])
        yin[sl] = bilinear_eval_exact(plan.ycoeff, plan.uout[u[sl]], plan.vout[v[sl]])
    np.clip(xin, f32(1.0), f32(nxin), out=xin)
    np.clip(yin, f32(1.0), f32(sched.nyin), out=yin)
    w = sched.window[v]
    yin = (yin - (sched.line3[w] - 1).astype(f32)).astype(f32)
    nx = xin.astype(np.int64)
    ny = yin.astype(np.int64)
    sx = (xin - nx.astype(f32)).astype(f32)
    sy = (yin - ny.astype(f32)).astype(f32)
    return OrderBand(order=order, plan=plan, xmin=xmin, xmax=xmax, nxin=nxin,
                     pix=v * nu + u, win=w, nx=nx, ny=ny, sx=sx, sy=sy, full=full)


@dataclass
class GroupGeometry:
    """A batch of orders solved and evaluated together."""

    bands: list[OrderBand]
    xlay: SolveLayout           # x pass: one vector per (order, cut row)
    ylay: SolveLayout           # y pass: one vector per (order, window, column)
    idx0: NDArray[np.int64]     # per pixel (all orders): R_y block index of tap (0, 0)
    sx: NDArray[np.float32]
    sy: NDArray[np.float32]
    pix_start: NDArray[np.int64]   # per band: first pixel in idx0 / sx / sy
    dev: dict                    # backend-resident copies (mlx), built lazily


def _group_geometry(bands: list[OrderBand], nyin: int, ncols_img: int) -> GroupGeometry:
    # x pass: orders by cut width (desc); each order's rows contiguous.
    xo = sorted(range(len(bands)), key=lambda i: -bands[i].nxin)
    xbase, xn, vx0 = [], [], {}
    pos = 0
    rows = np.arange(nyin, dtype=np.int64)
    for i in xo:
        b = bands[i]
        vx0[i] = pos
        xbase.append(rows * ncols_img + (b.xmin - 1))
        xn.append(np.full(nyin, b.nxin, np.int64))
        pos += nyin
    xlay = SolveLayout.build(np.concatenate(xbase), np.concatenate(xn),
                             max(b.nxin for b in bands) + 3)
    nvx = xlay.nv
    # y pass: one vector per (order, window, needed column), sorted by height.
    entries = []      # (h, band i, window w, cmin, ncols)
    for i, b in enumerate(bands):
        sched = T.schedule_for(b.plan, nyin)
        ws = np.unique(b.win)
        order_w = np.argsort(b.win, kind="stable")
        sw = b.win[order_w]
        starts = np.searchsorted(sw, ws, side="left")
        cmin = np.minimum.reduceat(b.nx[order_w], starts) - 1
        cmax = np.maximum.reduceat(b.nx[order_w], starts) + 2
        cmin = np.maximum(cmin, 0)
        cmax = np.minimum(cmax, b.nxin + 2)
        h = sched.line4[ws] - sched.line3[ws] + 1
        for w, hh, c0, c1 in zip(ws, h, cmin, cmax):
            entries.append((int(hh), i, int(w), int(c0), int(c1 - c0 + 1)))
    entries.sort(key=lambda e: -e[0])
    ybase, yn = [], []
    vy0 = {}
    pos = 0
    for hh, i, w, c0, nc in entries:
        b = bands[i]
        line3 = int(T.schedule_for(b.plan, nyin).line3[w])
        cols = np.arange(c0, c0 + nc, dtype=np.int64)
        ybase.append(cols * nvx + vx0[i] + line3 - 1)
        yn.append(np.full(nc, hh, np.int64))
        vy0[(i, w)] = (pos, c0)
        pos += nc
    if not entries:
        raise ValueError("rectification band without pixels")
    ylay = SolveLayout.build(np.concatenate(ybase), np.concatenate(yn), entries[0][0] + 3)
    nvy = ylay.nv
    idx0, pstart = [], []
    p = 0
    for i, b in enumerate(bands):
        pstart.append(p)
        lut_pos = np.zeros(int(b.win.max()) + 1, np.int64)
        lut_c0 = np.zeros(int(b.win.max()) + 1, np.int64)
        for (ii, w), (vp, c0) in vy0.items():
            if ii == i:
                lut_pos[w] = vp
                lut_c0[w] = c0
        idx0.append((b.ny - 1) * nvy + lut_pos[b.win] + (b.nx - 1 - lut_c0[b.win]))
        p += b.pix.size
    idx = np.concatenate(idx0)
    if idx.max(initial=0) + 4 * nvy < np.iinfo(np.int32).max:
        idx = idx.astype(np.int32)
    g = GroupGeometry(bands=bands, xlay=xlay, ylay=ylay, idx0=idx,
                      sx=np.concatenate([b.sx for b in bands]),
                      sy=np.concatenate([b.sy for b in bands]),
                      pix_start=np.asarray(pstart, np.int64), dev={})
    for b in bands:            # per-pixel tap data now lives in the group
        b.win = b.nx = b.ny = b.sx = b.sy = None
        if b.pix.size and b.pix.max() < np.iinfo(np.int32).max:
            b.pix = b.pix.astype(np.int32)
    return g


@dataclass
class FrameGeometry:
    """Everything s06 needs for one calibration set + set of bands."""

    orders: tuple[int, ...]
    groups: list[GroupGeometry]
    shape: tuple[int, int]


_CACHE: dict = {}
_LOCK = threading.Lock()
_CACHE_MAX = 8


def clear_cache() -> None:
    """Forget every cached rectification geometry."""
    with _LOCK:
        _CACHE.clear()


def _ref_band(ref_apdb: Path | None, order: int, nv: int, nu: int
              ) -> tuple[NDArray[np.int64], NDArray[np.int64]] | None:
    """``(lo, hi)`` strip columns (0-based) of the reference window + margin."""
    if ref_apdb is None:
        return None
    ap = ApertureSet.load(ref_apdb, array_length=nv).apertures.get(order)
    if ap is None:
        return None
    tu = np.asarray(ap.trace_x, np.float64)
    tc = np.asarray(ap.trace_x_clamped, np.float64)
    low, high = float(ap.entry.low), float(ap.entry.high)
    lo = np.floor(np.minimum(tu, tc) + low).astype(np.int64) - 1 - BAND_MARGIN
    hi = np.ceil(np.maximum(tu, tc) + high).astype(np.int64) - 1 + BAND_MARGIN
    lo = np.clip(lo, 0, nu - 1)
    hi = np.clip(hi, 0, nu - 1)
    # Where the window runs off the strip, apall clamps to the edge columns:
    # keep at least 2 * BAND_MARGIN + 1 columns at that edge.
    hi = np.maximum(hi, np.minimum(lo + 2 * BAND_MARGIN, nu - 1))
    lo = np.minimum(lo, np.maximum(hi - 2 * BAND_MARGIN, 0))
    return lo, hi


def _aperture_cols(apdb: Path, order: int, nv: int
                   ) -> tuple[NDArray[np.int64], NDArray[np.int64]] | None:
    """Strip columns (0-based, inclusive) the aperture of ``apdb`` spans per row."""
    ap = ApertureSet.load(apdb, array_length=nv).apertures.get(order)
    if ap is None:
        return None
    tu = np.asarray(ap.trace_x, np.float64)
    tc = np.asarray(ap.trace_x_clamped, np.float64)
    low, high = float(ap.entry.low), float(ap.entry.high)
    return (np.floor(np.minimum(tu, tc) + low).astype(np.int64) - 1,
            np.ceil(np.maximum(tu, tc) + high).astype(np.int64) - 1)


def orders_needing_full(apset: ApertureSet, orders: Iterable[int], *, fc_dir: Path,
                        fc_refname: str, dy: float, ref_trans_apdbs: dict | None,
                        trans_apdbs: dict | None, slack: int = 4) -> set[int]:
    """Orders whose per-frame (locked) aperture is not inside the reference band.

    WARP's per-frame aperture databases reuse the reference trace and clamp
    the window to the reference limits, so this is normally empty; an order
    listed here is rectified in full.
    """
    ref = ref_trans_apdbs or {}
    trans = trans_apdbs or {}
    out: set[int] = set()
    todo = [m for m in orders if m in trans and m in ref]
    if not todo:
        return out
    plans = T.get_plans([Path(fc_dir) / f"fc{fc_refname}_{m}" for m in todo], dy)
    for m, plan in zip(todo, plans):
        band = _ref_band(ref[m], m, plan.nv, plan.nu)
        need = _aperture_cols(trans[m], m, plan.nv)
        if band is None or need is None:
            continue
        lo, hi = band
        nlo = np.clip(need[0] - slack, 0, plan.nu - 1)
        nhi = np.clip(need[1] + slack, 0, plan.nu - 1)
        if np.any(nlo < lo) or np.any(nhi > hi):
            out.add(int(m))
    return out


def orders_to_redo(strips: dict, obj_1d: dict, sky_1d: dict) -> list[int]:
    """Band-evaluated orders whose extraction came out non-finite."""
    out = []
    for m, spec in obj_1d.items():
        bad = not np.all(np.isfinite(spec))
        if m in sky_1d:
            bad = bad or not np.all(np.isfinite(sky_1d[m]))
        if bad and np.isnan(strips[m].data).any():
            out.append(int(m))
    return sorted(out)


def geometry(apset: ApertureSet, *, fc_dir: Path, fc_refname: str, dy: float,
             orders: Iterable[int], ref_trans_apdbs: dict | None,
             full_orders: Iterable[int] = (), shape: tuple[int, int] = (2048, 2048)
             ) -> FrameGeometry:
    """The cached :class:`FrameGeometry` (built on first use)."""
    orders = tuple(orders)
    ref = dict(ref_trans_apdbs or {})
    full_set = frozenset(int(m) for m in full_orders)
    key = (str(Path(fc_dir)), fc_refname, float(dy), orders, shape,
           tuple(sorted((m, str(p)) for m, p in ref.items() if m in orders)), full_set,
           tuple((m, apset.apertures[m].trace_x[0], apset.apertures[m].trace_x[-1])
                 for m in orders))
    with _LOCK:
        g = _CACHE.get(key)
    if g is not None:
        return g
    nyin, ncols = shape
    plans = T.get_plans([Path(fc_dir) / f"fc{fc_refname}_{m}" for m in orders], dy)
    bands = []
    for m, plan in zip(orders, plans):
        xmin, xmax = _cut_limits(apset.apertures[m], apset.array_length)
        nxin = xmax - xmin + 1
        sched = T.schedule_for(plan, nyin)
        lohi = None if m in full_set else _ref_band(ref.get(m), m, plan.nv, plan.nu)
        lo, hi = (None, None) if lohi is None else lohi
        bands.append(_order_band(plan, sched, nxin, lo, hi, m, xmin, xmax))
    groups: list[GroupGeometry] = []
    cur: list[OrderBand] = []
    cur_cost = 0
    costs = [_band_cost(b, nyin) for b in bands]
    for b, cost in zip(bands, costs):
        if cur and cur_cost + cost > _GROUP_ELEMENTS:
            groups.append(_group_geometry(cur, nyin, ncols))
            cur, cur_cost = [], 0
        cur.append(b)
        cur_cost += cost
    if cur:
        groups.append(_group_geometry(cur, nyin, ncols))
    g = FrameGeometry(orders=orders, groups=groups, shape=shape)
    with _LOCK:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = g
    return g


def _band_cost(b: OrderBand, nyin: int) -> int:
    """Approximate solve-array elements of one order (x pass + y pass)."""
    sched = T.schedule_for(b.plan, nyin)
    h = sched.line4 - sched.line3 + 1
    ws = np.unique(b.win)
    ncols = b.nxin + 3 if b.full else int(np.ceil(b.pix.size / b.plan.nv)) + 2 * BAND_MARGIN
    return int((b.nxin + 3) * nyin + h.max() * ws.size * ncols)


# ---------------------------------------------------------------------------
# Per-frame evaluation
# ---------------------------------------------------------------------------

def _fma_into(out32: NDArray[np.float32], a64: NDArray, b64: NDArray, c32: NDArray[np.float32],
              p: NDArray[np.float64], s: NDArray[np.float64]) -> None:
    """``out32 = round32(a*b + c)`` (float32 fma) for float32 values ``a``, ``b``, ``c``.

    ``out32`` may alias ``c32``. ``a*b`` is exact in float64, so ``fl64(a*b +
    c)`` rounds once; a sum on a float32 midpoint is resolved with the exact
    two-sum error before ``c`` is overwritten.
    """
    np.multiply(a64, b64, out=p)
    np.add(p, c32, out=s)
    tie = (s.view(np.int64) & _TIE_MASK) == _TIE_HALF
    fix = None
    if tie.any():
        idx = np.nonzero(tie)
        sp, pp = s[idx], p[idx]
        cc = c32[idx].astype(np.float64)
        bb = sp - pp
        e = (pp - (sp - bb)) + (cc - bb)
        up = np.nextafter(sp, np.inf).astype(f32)
        dn = np.nextafter(sp, -np.inf).astype(f32)
        fix = (idx, np.where(e > 0, up, np.where(e < 0, dn, sp.astype(f32))))
    np.copyto(out32, s, casting="unsafe")
    if fix is not None:
        out32[fix[0]] = fix[1]


def _eval_rows(flat: NDArray[np.float32], i0: NDArray[np.int64], nv: int,
               sx: NDArray[np.float32], sy: NDArray[np.float32], *,
               checked: bool) -> tuple[NDArray[np.float32], NDArray[np.bool_] | None]:
    """``ii_bispline3`` of the pixels ``i0`` (tap c[-1] of row 0), (n, K).

    ``checked=False`` rounds each fused step through float64 and only flags
    pixels where a step hit a float32 midpoint (returned mask); those must be
    recomputed with ``checked=True``, which resolves every midpoint exactly.
    """
    K = flat.shape[1]
    n = sx.size
    bx = [w.astype(np.float64)[:, None] for w in _bspline_weights(sx, f32(1.0) - sx)]
    by = [w.astype(np.float64)[:, None] for w in _bspline_weights(sy, f32(1.0) - sy)]
    p = np.empty((n, K))
    sb = np.empty((n, K))
    acc = np.zeros((n, K), f32)
    sm = np.empty((n, K), f32)
    tie = None if checked else np.zeros((n, K), bool)
    low = None if checked else np.empty((n, K), np.int64)
    hit = None if checked else np.empty((n, K), bool)

    def fma(out32, a, b, c):
        if checked:
            _fma_into(out32, a, b, c, p, sb)
            return
        np.multiply(a, b, out=p)
        np.add(p, c, out=sb)
        np.copyto(out32, sb, casting="unsafe")
        np.bitwise_and(sb.view(np.int64), _TIE_MASK, out=low)
        np.equal(low, _TIE_HALF, out=hit)
        np.logical_or(tie, hit, out=tie)

    taps4 = sliding_window_view(flat, 4, axis=0)  # taps4[r] = flat[r:r+4].T
    for jj in range(4):
        t = taps4[i0 + jj * nv]                   # (n, K, 4): c[-1], c[0], c[+1], c[+2]
        np.multiply(bx[1], t[..., 1], out=p)      # bx2 * c[0], REAL product
        np.copyto(sm, p, casting="unsafe")
        fma(sm, t[..., 0], bx[0], sm)
        fma(sm, t[..., 2], bx[2], sm)
        fma(sm, t[..., 3], bx[3], sm)
        fma(acc, sm, by[jj], acc)
    return acc, (None if checked else tie.any(axis=1))


def _eval_numpy(R: NDArray[np.float32], g: GroupGeometry) -> NDArray[np.float32]:
    """``ii_bispline3`` at every band pixel of the group, (P, K), IRAF REAL.

    Exactly the arithmetic of
    :func:`decanter.utils.spline3.bispline3_eval_exact` (warp mode): fused
    basis weights, per row ``sum = fma(c[+2], bx4, fma(c[+1], bx3, fma(c[-1],
    bx1, bx2*c[0])))`` and ``accum = fma(sum, by, accum)`` from 0; each fused
    step is rounded once (float64 sum of an exact product, midpoints
    resolved exactly).
    """
    _, nv, K = R.shape
    flat = R.reshape(-1, K)
    out = np.empty((g.idx0.size, K), f32)
    step = 1 << 16
    for p0 in range(0, g.idx0.size, step):
        sl = slice(p0, p0 + step)
        i0 = g.idx0[sl].astype(np.int64)
        vals, tie = _eval_rows(flat, i0, nv, g.sx[sl], g.sy[sl], checked=False)
        if tie.any():
            t = np.nonzero(tie)[0]
            vals[t] = _eval_rows(flat, i0[t], nv, g.sx[sl][t], g.sy[sl][t], checked=True)[0]
        out[sl] = vals
    return out


def _run_group_numpy(stack2d: NDArray[np.float32], g: GroupGeometry) -> NDArray[np.float32]:
    Rx = _solve_numpy(stack2d, g.xlay)                    # (Lx, NVx, K)
    Ry = _solve_numpy(Rx.reshape(-1, Rx.shape[2]), g.ylay)
    del Rx
    return _eval_numpy(Ry, g)


def _run_group_mlx(stack_mx, g: GroupGeometry) -> NDArray[np.float32]:
    mx = _backend.mx()
    K = int(stack_mx.shape[1])
    dx = g.dev.setdefault("x", {})
    dyy = g.dev.setdefault("y", {})
    Rx = _solve_mlx(stack_mx, g.xlay, dx).reshape(-1, K)
    Ry = _solve_mlx(Rx, g.ylay, dyy)
    if "idx0" not in g.dev:
        g.dev["idx0"] = mx.array(g.idx0.astype(np.int32))
        g.dev["sx"] = mx.array(g.sx)
        g.dev["sy"] = mx.array(g.sy)
    P = g.idx0.size
    out = _mlx_kernel("eval")(
        inputs=[Ry, g.dev["idx0"], g.dev["sx"], g.dev["sy"],
                mx.array([P, g.ylay.nv, K], dtype=mx.int32)],
        grid=(max(P, 1), 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(P * K,)], output_dtypes=[mx.float32])[0]
    return np.asarray(out).reshape(P, K)


def _check_sizes(g: GroupGeometry) -> None:
    lim = np.iinfo(np.int32).max
    if g.ylay.L * g.ylay.nv * 2 >= lim or g.xlay.L * g.xlay.nv * 2 >= lim:
        raise OverflowError("rectification group too large for 32-bit GPU indexing")


def rectify_orders_multi(
    images: list[NDArray], apset: ApertureSet, *, fc_dir: Path, fc_refname: str,
    dy: float, orders: Iterable[int] | None = None,
    ref_trans_apdbs: dict | None = None, full_orders: Iterable[int] = (),
    full: bool = False, flux: bool = False, backend: str | None = None,
) -> list[dict[int, RectifiedStrip]]:
    """Fast :func:`decanter.rectify.transform.rectify_orders_multi`.

    Args:
        images: the frames to rectify (same shape; e.g. ``[obj, sky]``).
        ref_trans_apdbs: ``{order: Path}`` multihole trans reference DBs; each
            order's strip is evaluated only in that reference window (+
            :data:`BAND_MARGIN`), the rest is NaN. Orders without one, the
            orders in ``full_orders``, and every order when ``full`` is
            True, are evaluated in full.
        flux: WARP's ``CUTRANSFORM flux: yes`` (``transform flux=yes``):
            each pixel times the Jacobian factor, as in warp-reference.
        backend: ``"numpy"`` or ``"mlx"`` for this call (default: the
            active backend, :mod:`decanter.fast._backend`).

    Returns:
        One ``{order: RectifiedStrip}`` per image (WCS as in warp mode).
    """
    imgs = [np.asarray(im) for im in images]
    shape = imgs[0].shape
    req = tuple(orders) if orders is not None else tuple(apset.echelle_orders)
    fo = set(req) if full else set(full_orders)
    geo = geometry(apset, fc_dir=fc_dir, fc_refname=fc_refname, dy=dy, orders=req,
                   ref_trans_apdbs=None if full else ref_trans_apdbs, full_orders=fo,
                   shape=shape)
    K = len(imgs)
    use_mlx = (backend or _backend.active_backend()) == "mlx"
    if use_mlx:
        mx = _backend.mx()
        stack = mx.array(np.stack([np.asarray(im, np.float32) for im in imgs], axis=-1)
                         .reshape(-1, K))
    else:
        stack = np.stack([np.asarray(im, np.float32) for im in imgs], axis=-1).reshape(-1, K)
    out: list[dict[int, RectifiedStrip]] = [dict() for _ in imgs]
    for g in geo.groups:
        if use_mlx:
            _check_sizes(g)
            vals = _run_group_mlx(stack, g)
        else:
            vals = _run_group_numpy(stack, g)
        for bi, b in enumerate(g.bands):
            p0 = int(g.pix_start[bi])
            seg = vals[p0:p0 + b.pix.size]
            plan = b.plan
            if flux:
                # transform flux=yes: amulr(bufout, jbuf) after the
                # interpolation (t_transform.x:577-581), a REAL product
                seg = (seg * b.flux_factor()[:, None]).astype(f32)
            for k in range(K):
                if b.full:
                    data = seg[:, k].astype(f32).reshape(plan.nv, plan.nu)
                else:
                    data = np.full(plan.nv * plan.nu, np.nan, f32)
                    data[b.pix] = seg[:, k]
                    data = data.reshape(plan.nv, plan.nu)
                out[k][b.order] = strip_from_plan(data, plan, b.xmin, b.xmax)
    return out
