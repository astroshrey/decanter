"""Backend shim for the forward-model mode: write model code once, run it on MLX or JAX.

    from decanter.wavecal._xp import get_backend
    bk = get_backend("mlx")              # Apple GPU, float32: pixel-scale work
    bk = get_backend("jax", "float64")   # CPU, float64: gradient oracle, small dense solves
    xp = bk.xp                           # mlx.core or jax.numpy

Model code uses only the ``mlx.core`` ∩ ``jax.numpy`` intersection (``bk.xp``) plus the
methods of :class:`Backend`:

======================  ========================================================
arrays                  asarray, to_numpy, zeros, ones, eval, stop_gradient, as_index,
                        cpu, tile_map
transforms              grad, value_and_grad, vjp, jvp, jit, checkpoint, vmap, scan,
                        custom_vjp
dense linear algebra    cholesky, solve_triangular, cho_solve, solve, solve_spd, logdet_spd
block-tridiagonal SPD   blocktri_solve, blocktri_logdet, blocktri_factor,
                        blocktri_factor_solve, blocktri_selinv, blocktri_marginal_var,
                        blocktri_matvec, band_to_blocktri, banded_solve, pad_blocks,
                        unpad_blocks
======================  ========================================================

Every linear-algebra entry point is differentiable on both backends (reverse mode).

Measured constraints this module encodes (phase-1 autodiff study on the M4 Pro; mlx 0.32.4,
jax 0.6.2):

* MLX GPU has no float64.  JAX float64 is a process-wide flag (``jax_enable_x64``), set by
  ``get_backend("jax", "float64")``; explicitly float32 JAX arrays stay float32 under it.
* MLX linear algebra runs only on the CPU stream and has no VJP.  Here every MLX linear-algebra
  call is an ``mx.custom_function`` with an explicit adjoint, computed in float64 on the CPU
  stream (:attr:`Backend.linalg_dtype`); values and cotangents come back in the caller's dtype.
  JAX uses its native differentiable linear algebra, except the block-tridiagonal solver, whose
  ``jax.custom_vjp`` reuses the forward factor for the adjoint solve.  MLX's block-tridiagonal
  kernels default to a JAX float64 bridge inside the MLX custom VJP (``bt_impl="jax"``: 3-4x
  faster than MLX's own CPU kernels, which stay available as ``bt_impl="native"``).
* ``mx.custom_function``'s VJP receives (primals, cotangents, outputs) but no residuals:
  :meth:`Backend.custom_vjp` passes residuals as extra, discarded outputs, so one ``fwd``/``bwd``
  pair serves both backends.  MLX neither casts cotangents back to the primal dtype nor treats
  ``None`` as zero; the shim casts and zero-fills.
* ``mx.compile`` freezes closed-over arrays as constants (no gradient reaches them) and fails on
  large unrolled graphs: pass arrays as arguments and compile small per-step functions
  (:meth:`Backend.scan` does this for MLX).  JAX uses ``lax.scan`` (+ ``jax.checkpoint``).
* MLX raises on gather VJPs w.r.t. graph-connected integer indices: build indices with
  :meth:`Backend.as_index` (``stop_gradient`` before the integer cast).
* ``mx.array(float64 ndarray)`` silently gives float32: :meth:`Backend.asarray` is explicit.

Failure convention: a matrix whose Cholesky fails (not SPD to working precision) and a triangular
or LU solve with an exactly zero pivot give NaN/inf on both backends (MLX's own LAPACK wrappers
instead return a finite partial factor, or raise at a later eval; the shim maps both to JAX's
NaN).  A block-tridiagonal system with one bad block is NaN for its whole batch element:
regularize upstream (e.g. fully masked flux bins).

Symmetric convention: functions of an SPD matrix (``cholesky``, ``solve_spd``, ``logdet_spd`` and
the block-tridiagonal routines) are defined on ``sym(A) = (A + Aᵀ)/2`` and return symmetric
cotangents, as ``jax.numpy.linalg.cholesky(symmetrize_input=True)`` does.

Block-tridiagonal layout (used for the banded extraction normal equations): ``D`` holds the
diagonal blocks, shape ``(..., n, k, k)``; ``E`` the sub-diagonal blocks ``E[i] = A[i+1, i]``,
shape ``(..., n-1, k, k)`` (the super-diagonal is ``Eᵀ``); a right-hand side is ``(..., n, k)``
or ``(..., n, k, m)``.  Leading batch dimensions (e.g. echelle orders) must match exactly.  A
matrix with half-bandwidth ``p`` in LAPACK lower band storage converts with
:meth:`Backend.band_to_blocktri` (block size ``k >= p``).
"""
from __future__ import annotations

import contextlib
import math
import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache, lru_cache
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "Backend",
    "BlockTriCRFactor",
    "BlockTriFactor",
    "array_namespace",
    "blocktri_to_dense",
    "check_grads",
    "default_backend",
    "get_backend",
    "mlx_gpu_available",
    "tree_flatten",
    "tree_leaves",
    "tree_map",
    "tree_unflatten",
]


# ===================================================================== pytrees
# Minimal pytree support (dict / list / tuple / namedtuple containers, None = empty node),
# independent of either backend so both see identical structures.
_LEAF = "*"


def _is_namedtuple(x) -> bool:
    return isinstance(x, tuple) and hasattr(type(x), "_fields")


def tree_flatten(tree) -> tuple[list, Any]:
    """``(leaves, treedef)`` for a nest of dict / list / tuple / namedtuple / None."""
    leaves: list = []

    def rec(t):
        if t is None:
            return ("none", None, ())
        if isinstance(t, dict):
            keys = tuple(t.keys())
            return ("dict", keys, tuple(rec(t[k]) for k in keys))
        if _is_namedtuple(t):
            return ("namedtuple", type(t), tuple(rec(v) for v in t))
        if isinstance(t, (list, tuple)):
            return (type(t), None, tuple(rec(v) for v in t))
        leaves.append(t)
        return _LEAF

    return leaves, rec(tree)


def tree_unflatten(treedef, leaves):
    """Inverse of :func:`tree_flatten`."""
    it = iter(leaves)

    def rec(d):
        if isinstance(d, str):  # _LEAF
            return next(it)
        kind, aux, children = d
        if kind == "none":
            return None
        if kind == "dict":
            return {k: rec(c) for k, c in zip(aux, children)}
        if kind == "namedtuple":
            return aux(*(rec(c) for c in children))
        return kind(rec(c) for c in children)

    out = rec(treedef)
    return out


def tree_leaves(tree) -> list:
    return tree_flatten(tree)[0]


def tree_map(f: Callable, tree, *rest):
    """Apply ``f`` leaf-wise over one or more pytrees of the same structure."""
    leaves, td = tree_flatten(tree)
    others = [tree_flatten(r)[0] for r in rest]
    return tree_unflatten(td, [f(x, *ys) for x, *ys in zip(leaves, *others)])


# ===================================================================== helpers
def array_namespace(x):
    """``numpy``, ``mlx.core`` or ``jax.numpy`` for an array, without importing a backend
    that is not already loaded (so it is safe on the import-light path)."""
    mod = type(x).__module__ or ""
    if mod.startswith("mlx"):
        import mlx.core as mx

        return mx
    if mod.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp

        return jnp
    return np


def mlx_gpu_available() -> bool:
    """True if ``mlx`` imports and its default device is a GPU."""
    try:
        import mlx.core as mx

        return "gpu" in str(mx.default_device()).lower()
    except Exception:
        return False


def _np_float(precision: str):
    return np.float64 if precision == "float64" else np.float32


class BlockTriFactor(NamedTuple):
    """Sequential block Cholesky factor ``A = C Cᵀ`` of a block-tridiagonal SPD matrix.

    ``C`` holds the lower-triangular diagonal blocks ``C[i]`` (``(..., n, k, k)``) and ``F`` the
    sub-diagonal blocks ``F[i] = C[i+1, i]`` (``(..., n-1, k, k)``), in
    :attr:`Backend.linalg_dtype`.  Treated as a constant (no gradient flows through it).
    """

    C: Any
    F: Any


class BlockTriCRFactor(NamedTuple):
    """Odd-even block cyclic-reduction factor (a Cholesky factor of a symmetric permutation).

    ``levels[l] = (L, W)`` for the blocks eliminated at level ``l`` (fine to coarse): ``L``
    their Cholesky factors and ``W = L⁻¹ [A[i, i-1] | A[i, i+1]]`` (``k x 2k``) their scaled
    couplings to the surviving neighbours; ``last`` is the Cholesky factor of the final 1-block
    Schur complement.
    Treated as a constant (no gradient flows through it).
    """

    levels: tuple
    last: Any


_MLX_SCAN_CACHE: weakref.WeakKeyDictionary[Callable, dict] = weakref.WeakKeyDictionary()


# ===================================================================== backend
@dataclass(eq=False)  # identity-hashed: get_backend caches instances
class Backend:
    """One backend (``"mlx"`` or ``"jax"``) at one float precision; see the module docstring."""

    name: str
    precision: str
    xp: Any = field(repr=False)
    dtype: Any = field(repr=False)
    np_dtype: Any = field(repr=False)
    linalg_dtype: Any = field(repr=False)
    # block-tridiagonal kernels: factorization "seq" (sequential block Cholesky: JAX lax.scan)
    # or "cr" (odd-even cyclic reduction: O(log n) batched levels, used by native MLX); and for
    # MLX the implementation "jax" (JAX float64 bridge, default when jax imports) or "native"
    bt_method: str = "seq"
    bt_impl: str = "native"
    _cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ basics
    @property
    def is_mlx(self) -> bool:
        return self.name == "mlx"

    @property
    def is_jax(self) -> bool:
        return self.name == "jax"

    def is_array(self, x) -> bool:
        """True for a native array of this backend."""
        if self.is_mlx:
            return isinstance(x, self.xp.array)
        import jax

        return isinstance(x, jax.Array)

    def _native_dtype(self, dt):
        """numpy / string / backend dtype -> backend dtype."""
        if dt is None:
            return None
        if not isinstance(dt, (type, np.dtype, str)):
            return dt  # already a backend dtype object
        nd = np.dtype(dt)
        if self.is_jax:
            return nd
        mx = self.xp
        table = {np.dtype(np.float32): mx.float32, np.dtype(np.float64): mx.float64,
                 np.dtype(np.float16): mx.float16, np.dtype(np.int32): mx.int32,
                 np.dtype(np.int64): mx.int64, np.dtype(np.int16): mx.int16,
                 np.dtype(np.int8): mx.int8, np.dtype(np.uint8): mx.uint8,
                 np.dtype(np.uint16): mx.uint16, np.dtype(np.uint32): mx.uint32,
                 np.dtype(np.uint64): mx.uint64, np.dtype(np.bool_): mx.bool_,
                 np.dtype(np.complex64): mx.complex64}
        return table[nd]

    def asarray(self, x, dtype=None):
        """Convert to a backend array.

        ``dtype=None``: floating input -> the backend float dtype, signed integers -> int32,
        unsigned / bool kept; a native array of this backend is returned unchanged.  A float64
        request on MLX creates the array on the CPU stream (the GPU has no float64).
        """
        if dtype is None and self.is_array(x):
            return x
        if self.is_array(x):
            ndt = self._native_dtype(dtype)
            if self.is_mlx and ndt == self.xp.float64:
                with self.xp.stream(self.xp.cpu):
                    return x.astype(ndt)
            return x.astype(ndt)
        a = np.asarray(x)
        if dtype is None:
            k = a.dtype.kind
            if k in "fc" or a.dtype == object:
                a = a.astype(self.np_dtype)
            elif k == "i":
                a = a.astype(np.int32)
        else:
            nd = np.dtype(dtype) if isinstance(dtype, (type, np.dtype, str)) else None
            if nd is not None:
                a = a.astype(nd)
        if self.is_mlx:
            mx = self.xp
            ndt = self._native_dtype(a.dtype) if dtype is None or nd is not None else dtype
            if ndt == mx.float64:
                with mx.stream(mx.cpu):
                    return mx.array(a, dtype=mx.float64)
            return mx.array(a, dtype=ndt)
        return self.xp.asarray(a)

    def to_numpy(self, x):
        """Backend array (or pytree of them) -> numpy (an independent copy for MLX, whose
        buffers numpy would otherwise alias), evaluating lazily-built MLX graphs."""
        if isinstance(x, (dict, list, tuple)) or x is None:
            return tree_map(self.to_numpy, x)
        if self.is_mlx and isinstance(x, self.xp.array):
            if x.dtype == self.xp.bfloat16:
                x = x.astype(self.xp.float32)
            self.xp.eval(x)
            return np.array(x)
        return np.asarray(x)

    def zeros(self, shape, dtype=None):
        dt = self.dtype if dtype is None else self._native_dtype(dtype)
        if self.is_mlx and dt == self.xp.float64:
            with self.xp.stream(self.xp.cpu):
                return self.xp.zeros(shape, dtype=dt)
        return self.xp.zeros(shape, dtype=dt)

    def ones(self, shape, dtype=None):
        dt = self.dtype if dtype is None else self._native_dtype(dtype)
        if self.is_mlx and dt == self.xp.float64:
            with self.xp.stream(self.xp.cpu):
                return self.xp.ones(shape, dtype=dt)
        return self.xp.ones(shape, dtype=dt)

    def stop_gradient(self, a):
        if self.is_mlx:
            if a.dtype == self.xp.float64:
                with self.xp.stream(self.xp.cpu):
                    return self.xp.stop_gradient(a)
            return self.xp.stop_gradient(a)
        import jax

        return jax.lax.stop_gradient(a)

    def as_index(self, a):
        """int32 indices from a float expression, cut from the graph (MLX requires the cut)."""
        return self.stop_gradient(a).astype(self.xp.int32)

    def eval(self, *trees):
        """Force evaluation (MLX ``mx.eval``; JAX ``block_until_ready``)."""
        leaves = [x for t in trees for x in tree_leaves(t) if self.is_array(x)]
        if not leaves:
            return
        if self.is_mlx:
            self.xp.eval(*leaves)
        else:
            import jax

            jax.block_until_ready(leaves)

    def cpu(self):
        """Context manager: run MLX ops on the CPU stream (no-op for JAX, which is CPU here)."""
        if self.is_mlx:
            return self.xp.stream(self.xp.cpu)
        return contextlib.nullcontext()

    def peak_memory_bytes(self) -> int | None:
        """MLX peak allocator memory since the last reset (None for JAX)."""
        return int(self.xp.get_peak_memory()) if self.is_mlx else None

    def reset_peak_memory(self) -> None:
        if self.is_mlx:
            self.xp.reset_peak_memory()

    def tile_map(self, fn: Callable, *arrays, tile: int = 256, axis: int = 0,
                 combine: str = "concat"):
        """Apply ``fn`` to row tiles of ``arrays`` (sliced along ``axis``), evaluating each tile
        before the next so the lazy graph and peak memory stay bounded.  ``combine="concat"``
        concatenates the per-tile outputs (array or pytree) along ``axis``; ``"sum"`` adds them
        (sufficient statistics).  Forward only: do not differentiate through it."""
        n = arrays[0].shape[axis]
        acc = None
        parts = []
        for s in range(0, n, tile):
            sl = [slice(None)] * arrays[0].ndim
            sl[axis] = slice(s, min(s + tile, n))
            out = fn(*(a[tuple(sl)] for a in arrays))
            self.eval(out)
            if combine == "sum":
                acc = out if acc is None else tree_map(lambda a, b: a + b, acc, out)
            else:
                parts.append(out)
        if combine == "sum":
            return acc
        if combine != "concat":
            raise ValueError("combine must be 'concat' or 'sum'")
        leaves0, td = tree_flatten(parts[0])
        cols = [tree_flatten(p)[0] for p in parts]
        return tree_unflatten(td, [self.xp.concatenate([c[i] for c in cols], axis=axis)
                                   for i in range(len(leaves0))])

    # ------------------------------------------------------------- transforms
    def grad(self, f: Callable, argnums=0, has_aux: bool = False) -> Callable:
        """``grad(f)(*args)`` -> gradient(s); with ``has_aux`` f returns ``(loss, aux)`` and the
        result is ``(grads, aux)``."""
        if self.is_jax:
            import jax

            return jax.grad(f, argnums=argnums, has_aux=has_aux)
        if not has_aux:
            return self.xp.grad(f, argnums=argnums)
        vg = self.value_and_grad(f, argnums=argnums, has_aux=True)

        def g(*args, **kw):
            (_, aux), grads = vg(*args, **kw)
            return grads, aux

        return g

    def value_and_grad(self, f: Callable, argnums=0, has_aux: bool = False) -> Callable:
        """``(value, grads)``, or ``((value, aux), grads)`` with ``has_aux``.  ``argnums`` int or
        tuple, as in JAX; arguments may be pytrees of arrays."""
        if self.is_jax:
            import jax

            return jax.value_and_grad(f, argnums=argnums, has_aux=has_aux)
        # MLX differentiates the first output and passes the rest through, which is exactly
        # the has_aux contract when f returns (loss, aux)
        return self.xp.value_and_grad(f, argnums=argnums)

    def vjp(self, f: Callable, primals, cotangents):
        """``(outputs, input_cotangents)`` for positional ``primals`` (a tuple).  ``f`` returns an
        array or a tuple of arrays; ``cotangents`` mirrors that structure."""
        primals = tuple(primals)
        if self.is_jax:
            import jax

            out, pull = jax.vjp(f, *primals)
            return out, tuple(pull(cotangents))
        single = not isinstance(cotangents, (list, tuple))
        cts = [cotangents] if single else list(cotangents)
        outs, grads = self.xp.vjp(lambda *a: f(*a), list(primals), cts)
        return (outs[0] if single else tuple(outs)), tuple(grads)

    def jvp(self, f: Callable, primals, tangents):
        """``(outputs, output_tangents)``.  Note: functions built with :meth:`custom_vjp` (all
        linear algebra here) are reverse-mode only."""
        primals, tangents = tuple(primals), tuple(tangents)
        if self.is_jax:
            import jax

            return jax.jvp(f, primals, tangents)
        outs, touts = self.xp.jvp(lambda *a: f(*a), list(primals), list(tangents))
        if len(outs) == 1:
            return outs[0], touts[0]
        return tuple(outs), tuple(touts)

    def jit(self, f: Callable, static_argnums=None) -> Callable:
        """``jax.jit`` / ``mx.compile``.  Pass arrays as arguments, not closures (MLX freezes
        closed-over arrays as constants); keep compiled MLX functions small.  ``static_argnums``
        is honoured by JAX only (MLX re-traces on new non-array arguments)."""
        if self.is_jax:
            import jax

            return jax.jit(f, static_argnums=static_argnums)
        return self.xp.compile(f)

    def checkpoint(self, f: Callable) -> Callable:
        """Rematerialise ``f`` in the backward pass (``jax.checkpoint`` / ``mx.checkpoint``)."""
        if self.is_jax:
            import jax

            return jax.checkpoint(f)
        return self.xp.checkpoint(f)

    def vmap(self, f: Callable, in_axes=0, out_axes=0) -> Callable:
        if self.is_jax:
            import jax

            return jax.vmap(f, in_axes=in_axes, out_axes=out_axes)
        return self.xp.vmap(f, in_axes=in_axes, out_axes=out_axes)

    def scan(self, step: Callable, carry, xs=None, consts=None, *, length: int | None = None,
             checkpoint: bool = False, compile: bool = True, eval_every: int = 0):
        """``carry, ys = scan(step, carry, xs, consts)`` with ``step(carry, x, consts) -> (carry, y)``.

        ``xs`` is a pytree of arrays sharing the leading (time) axis (or None with ``length``);
        ``ys`` stacks the per-step ``y`` pytrees (None if every ``y`` is None).  ``consts`` is
        passed unchanged to every step; pass arrays that need gradients there, not by closure
        (a compiled MLX step freezes closed-over arrays).  JAX: ``lax.scan`` (+
        ``jax.checkpoint``).  MLX: a Python loop over a compiled (optionally checkpointed) step,
        cached per step function; ``eval_every > 0`` evaluates every that many steps (forward-only
        use: bounds the lazy graph).
        """
        if self.is_jax:
            import jax

            def body(c, x):
                return step(c, x, consts)

            fn = jax.checkpoint(body) if checkpoint else body
            return jax.lax.scan(fn, carry, xs, length=length)
        mx = self.xp
        fn = step
        if compile or checkpoint:
            per = _MLX_SCAN_CACHE.setdefault(step, {})
            key = (bool(compile), bool(checkpoint))
            fn = per.get(key)
            if fn is None:
                fn = mx.compile(step) if compile else step
                if checkpoint:
                    fn = mx.checkpoint(fn)
                per[key] = fn
        leaves, td = tree_flatten(xs)
        n = leaves[0].shape[0] if leaves else int(length)
        ys = []
        for i in range(n):
            x = tree_unflatten(td, [lf[i] for lf in leaves]) if leaves else None
            carry, y = fn(carry, x, consts)
            ys.append(y)
            if eval_every and (i + 1) % eval_every == 0:
                self.eval(carry, ys)
        if not ys or all(y is None for y in ys):
            return carry, None
        y_leaves0, y_td = tree_flatten(ys[0])
        cols = [tree_flatten(y)[0] for y in ys]
        return carry, tree_unflatten(y_td, [mx.stack([c[j] for c in cols], axis=0)
                                            for j in range(len(y_leaves0))])

    def custom_vjp(self, fwd: Callable, bwd: Callable, *, cpu: bool = False) -> Callable:
        """A function with a hand-written VJP that works the same on both backends.

        ``fwd(*args) -> (out, res)``: ``out`` is an array or a pytree of arrays (the function's
        value); ``res`` a tuple (possibly empty) of extra arrays that the backward pass needs and
        that are neither inputs nor outputs (e.g. a factorization).

        ``bwd(args, out, res, g) -> cotangents``: ``g`` mirrors ``out``; return one cotangent per
        positional argument (``None`` = zero).  Cotangents are cast to the argument dtypes.

        All arguments must be arrays (positional only); bind static options with a closure and
        build the function once per option set.  JAX: ``jax.custom_vjp`` with residuals
        ``(args, out, res)``.  MLX: ``mx.custom_function`` returning ``res`` as extra outputs
        (their cotangents are ignored); ``cpu=True`` places the custom primitive (and forward ops
        without an explicit stream) on the CPU stream, which MLX requires for float64 outputs or
        residuals (the linear-algebra wrappers use it).  Reverse mode only.
        """
        if self.is_jax:
            import jax

            @jax.custom_vjp
            def f(*args):
                return fwd(*args)[0]

            def f_fwd(*args):
                out, res = fwd(*args)
                return out, (args, out, tuple(res))

            def f_bwd(saved, g):
                args, out, res = saved
                return self._fix_cotangents(args, bwd(args, out, res, g))

            f.defvjp(f_fwd, f_bwd)
            return f

        mx = self.xp
        # output/residual pytree structure per input signature (shapes + dtypes): it may depend
        # on the shapes (e.g. the number of cyclic-reduction levels), and one graph can hold
        # several calls of the same function with different shapes, so a single shared spec
        # would unflatten an earlier call's residuals with a later call's structure
        specs: dict = {}

        def sig(args):
            return tuple((tuple(a.shape), str(a.dtype)) for a in args)

        @mx.custom_function
        def inner(*args):
            out, res = fwd(*args)
            out_leaves, out_def = tree_flatten(out)
            res_leaves, res_def = tree_flatten(tuple(res))
            specs[sig(args)] = (len(out_leaves), out_def, res_def)
            return tuple(out_leaves) + tuple(res_leaves)

        @inner.vjp
        def inner_vjp(primals, cotangents, outputs):
            primals = tuple(primals) if isinstance(primals, (list, tuple)) else (primals,)
            cotangents = (tuple(cotangents) if isinstance(cotangents, (list, tuple))
                          else (cotangents,))
            outputs = tuple(outputs) if isinstance(outputs, (list, tuple)) else (outputs,)
            n_out, out_def, res_def = specs[sig(primals)]
            out = tree_unflatten(out_def, list(outputs[:n_out]))
            res = tree_unflatten(res_def, list(outputs[n_out:]))
            g = tree_unflatten(out_def, list(cotangents[:n_out]))
            return tuple(self._fix_cotangents(primals, bwd(primals, out, res, g)))

        def f(*args):
            args = tuple(a if isinstance(a, mx.array) else self.asarray(a) for a in args)
            if cpu:
                with mx.stream(mx.cpu):
                    outs = inner(*args)
            else:
                outs = inner(*args)
            outs = tuple(outs) if isinstance(outs, (list, tuple)) else (outs,)
            n_out, out_def, _ = specs[sig(args)]
            return tree_unflatten(out_def, list(outs[:n_out]))

        return f

    def _fix_cotangents(self, primals, cts):
        if not isinstance(cts, (list, tuple)):
            cts = (cts,)
        if len(cts) != len(primals):
            raise ValueError(f"custom_vjp bwd returned {len(cts)} cotangents for "
                             f"{len(primals)} arguments")
        out = []
        for p, c in zip(primals, cts):
            # MLX: anything touching float64 must run on the CPU stream
            f64 = self.is_mlx and (p.dtype == self.xp.float64
                                   or (c is not None and c.dtype == self.xp.float64))
            with (self.xp.stream(self.xp.cpu) if f64 else contextlib.nullcontext()):
                if c is None:
                    if self.is_jax:
                        out.append(None)
                        continue
                    c = self.xp.zeros(p.shape, dtype=p.dtype)
                if tuple(c.shape) != tuple(p.shape):
                    raise ValueError(f"custom_vjp cotangent shape {tuple(c.shape)} != argument "
                                     f"shape {tuple(p.shape)}")
                out.append(c.astype(p.dtype) if c.dtype != p.dtype else c)
        return tuple(out)

    # ----------------------------------------------- shared linear-algebra bits
    def _mT(self, a):
        return self.xp.swapaxes(a, -1, -2)

    def _sym(self, a):
        return 0.5 * (a + self._mT(a))

    def _lin_ctx(self):
        """Context for linear-algebra kernels (MLX: CPU stream)."""
        return self.cpu()

    def _lin(self, a):
        """Cast to the linear-algebra dtype (inside :meth:`_lin_ctx`)."""
        return a if a.dtype == self.linalg_dtype else a.astype(self.linalg_dtype)

    def _tri(self, a, lower: bool):
        return self.xp.tril(a) if lower else self.xp.triu(a)

    def _trsm(self, a, b, *, lower: bool, trans: bool = False):
        """Raw (non-differentiable on MLX) batched triangular solve ``op(a) x = b``, ``b`` a
        matrix ``(..., n, m)``; MLX: call inside :meth:`_lin_ctx` with linalg-dtype arrays."""
        if self.is_jax:
            from jax.scipy.linalg import solve_triangular

            return solve_triangular(a, b, lower=lower, trans=1 if trans else 0)
        mx = self.xp
        # MLX solves through a triangular inverse (LAPACK trtri), which raises "triangular
        # inversion failed" for an exactly zero pivot -- at eval time, possibly at an unrelated
        # later mx.eval -- where JAX returns inf/NaN.  A NaN pivot instead propagates NaN.
        k = a.shape[-1]
        a = mx.where(mx.eye(k, dtype=mx.bool_) & (a == 0), mx.array(float("nan"), a.dtype), a)
        if trans:
            return mx.linalg.solve_triangular(self._mT(a), b, upper=lower)
        return mx.linalg.solve_triangular(a, b, upper=not lower)

    def _chol(self, a):
        """Raw lower Cholesky of ``sym(a)`` (MLX: inside :meth:`_lin_ctx`).  A matrix that is not
        numerically SPD gives a NaN lower triangle on both backends."""
        if self.is_jax:
            return self.xp.linalg.cholesky(a)  # symmetrize_input=True
        mx = self.xp
        L = mx.linalg.cholesky(self._sym(a))
        # MLX ignores potrf's failure code and returns a finite partial "factor" (the failing
        # pivot left <= 0, later columns untouched) for an indefinite or singular matrix; JAX
        # returns NaN.  Match JAX so that a bad system cannot pass silently as a finite answer.
        d = mx.diagonal(L, axis1=-2, axis2=-1)
        bad = mx.any(mx.logical_not(d > 0), axis=-1)[..., None, None]
        k = L.shape[-1]
        lower = mx.tril(mx.ones((k, k), dtype=mx.bool_))
        return mx.where(bad & lower, mx.array(float("nan"), L.dtype), L)  # JAX's NaN pattern

    @staticmethod
    def _vec(b, A):
        """(b as matrix, was_vector) for a right-hand side of matrix ``A`` (..., n, n)."""
        if b.ndim == A.ndim - 1:
            return b[..., None], True
        return b, False

    def _custom(self, key, build: Callable, jit: bool = False) -> Callable:
        """Build-once cache of per-backend helper functions (``jit``: wrap in ``jax.jit`` on
        JAX, which is inlined under an outer jit and avoids slow eager dispatch otherwise)."""
        fn = self._cache.get(key)
        if fn is None:
            fn = build()
            if jit and self.is_jax:
                import jax

                fn = jax.jit(fn)
            self._cache[key] = fn
        return fn

    # --------------------------------------------------- dense linear algebra
    def cholesky(self, A):
        """Lower Cholesky factor of ``sym(A)``, batched over leading dims; differentiable."""
        if self.is_jax:
            return self.xp.linalg.cholesky(A)
        return self._custom(("cholesky",), self._build_mlx_cholesky)(A)

    def _build_mlx_cholesky(self):
        xp = self.xp

        def fwd(A):
            with self._lin_ctx():
                L = self._chol(self._lin(A))
                return L.astype(A.dtype), (L,)

        def bwd(args, Lout, res, g):
            (L,) = res
            with self._lin_ctx():
                k = L.shape[-1]
                phi = xp.tril(xp.ones((k, k), dtype=L.dtype)) - 0.5 * xp.eye(k, dtype=L.dtype)
                P = (self._mT(L) @ self._lin(g)) * phi          # Φ(Lᵀ Ḡ)
                Y = self._trsm(L, P, lower=True, trans=True)    # L⁻ᵀ P
                Z = self._trsm(L, self._mT(Y), lower=True, trans=True)  # (Y L⁻¹)ᵀ
                return (self._sym(self._mT(Z)),)

        return self.custom_vjp(fwd, bwd, cpu=True)

    def solve_triangular(self, L, b, *, lower: bool = True, trans: bool = False):
        """Solve ``op(L) x = b`` with ``op(L) = L`` or ``Lᵀ`` (``trans``); ``L`` lower or upper
        triangular (only that triangle is read), ``b`` ``(..., n)`` or ``(..., n, m)``."""
        bm, vec = self._vec(b, L)
        if self.is_jax:
            x = self._trsm(L, bm, lower=lower, trans=trans)
        else:
            x = self._custom(("trsm", bool(lower), bool(trans)),
                             lambda: self._build_mlx_trsm(lower, trans))(L, bm)
        return x[..., 0] if vec else x

    def _build_mlx_trsm(self, lower: bool, trans: bool):
        def fwd(L, b):
            with self._lin_ctx():
                x = self._trsm(self._lin(L), self._lin(b), lower=lower, trans=trans)
                return x.astype(b.dtype), ()

        def bwd(args, x, res, g):
            L, _ = args
            with self._lin_ctx():
                L64, x64 = self._lin(L), self._lin(x)
                gb = self._trsm(L64, self._lin(g), lower=lower, trans=not trans)
                outer = (x64 @ self._mT(gb)) if trans else (gb @ self._mT(x64))
                return -self._tri(outer, lower), gb

        return self.custom_vjp(fwd, bwd, cpu=True)

    def cho_solve(self, L, b):
        """Solve ``(L Lᵀ) x = b`` given the lower Cholesky factor ``L``; differentiable in both."""
        if self.is_jax:
            from jax.scipy.linalg import cho_solve

            bm, vec = self._vec(b, L)
            x = cho_solve((L, True), bm)
            return x[..., 0] if vec else x
        y = self.solve_triangular(L, b, lower=True)
        return self.solve_triangular(L, y, lower=True, trans=True)

    def solve(self, A, b):
        """General square solve ``A x = b`` (``b`` vector or matrix); differentiable."""
        bm, vec = self._vec(b, A)
        if self.is_jax:
            x = self.xp.linalg.solve(A, bm)
        else:
            x = self._custom(("solve",), self._build_mlx_solve)(A, bm)
        return x[..., 0] if vec else x

    def _build_mlx_solve(self):
        # LU (A = L[P] U, i.e. A[argsort P] = L U) + the guarded triangular solves rather than
        # mx.linalg.solve, which raises (deferred) on a singular matrix instead of returning
        # inf/NaN as JAX does; the factor is reused for the adjoint solve
        xp = self.xp

        def rows(b, idx):  # b[..., idx, :]
            return xp.take_along_axis(b, xp.broadcast_to(idx[..., None], b.shape), axis=-2)

        def fwd(A, b):
            with self._lin_ctx():
                P, L, U = xp.linalg.lu(self._lin(A))
                y = self._trsm(L, rows(self._lin(b), xp.argsort(P, axis=-1)), lower=True)
                x = self._trsm(U, y, lower=False)
                # permutation as float (exact): integer residual outputs would get cotangents
                return x.astype(b.dtype), (L, U, P.astype(self.linalg_dtype))

        def bwd(args, x, res, g):
            L, U, Pf = res
            with self._lin_ctx():
                z = self._trsm(U, self._lin(g), lower=False, trans=True)
                z = self._trsm(L, z, lower=True, trans=True)
                gb = rows(z, Pf.astype(xp.int32))                  # A⁻ᵀ g
                return -(gb @ self._mT(self._lin(x))), gb

        return self.custom_vjp(fwd, bwd, cpu=True)

    def solve_spd(self, A, b):
        """Solve ``sym(A) x = b`` by Cholesky; symmetric cotangent for ``A``."""
        bm, vec = self._vec(b, A)
        if self.is_jax:
            x = self.cho_solve(self.xp.linalg.cholesky(A), bm)
        else:
            x = self._custom(("solve_spd",), self._build_mlx_solve_spd)(A, bm)
        return x[..., 0] if vec else x

    def _build_mlx_solve_spd(self):
        def cs(L, b):
            return self._trsm(L, self._trsm(L, b, lower=True), lower=True, trans=True)

        def fwd(A, b):
            with self._lin_ctx():
                L = self._chol(self._lin(A))
                return cs(L, self._lin(b)).astype(b.dtype), (L,)

        def bwd(args, x, res, g):
            (L,) = res
            with self._lin_ctx():
                lam = cs(L, self._lin(g))
                return -self._sym(lam @ self._mT(self._lin(x))), lam

        return self.custom_vjp(fwd, bwd, cpu=True)

    def logdet_spd(self, A):
        """``log det sym(A)`` for SPD ``A`` (batched); cotangent ``g · A⁻¹``."""
        if self.is_jax:
            L = self.xp.linalg.cholesky(A)
            return 2.0 * self.xp.sum(self.xp.log(self.xp.diagonal(L, axis1=-2, axis2=-1)),
                                     axis=-1)
        return self._custom(("logdet_spd",), self._build_mlx_logdet)(A)

    def _build_mlx_logdet(self):
        xp = self.xp

        def fwd(A):
            with self._lin_ctx():
                L = self._chol(self._lin(A))
                ld = 2.0 * xp.sum(xp.log(xp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
                return ld.astype(A.dtype), (L,)

        def bwd(args, ld, res, g):
            (L,) = res
            with self._lin_ctx():
                k = L.shape[-1]
                Li = self._trsm(L, xp.broadcast_to(xp.eye(k, dtype=L.dtype), L.shape),
                                lower=True)
                Ainv = self._mT(Li) @ Li
                return (self._lin(g)[..., None, None] * Ainv,)

        return self.custom_vjp(fwd, bwd, cpu=True)

    # ----------------------------------------- block-tridiagonal SPD systems
    @staticmethod
    def _bt_check(D, E):
        if D.ndim < 3 or D.shape[-1] != D.shape[-2]:
            raise ValueError(f"D must be (..., n, k, k), got {tuple(D.shape)}")
        n, k = D.shape[-3], D.shape[-1]
        want = tuple(D.shape[:-3]) + (max(n - 1, 0), k, k)
        if tuple(E.shape) != want:
            raise ValueError(f"E must have shape {want} for D {tuple(D.shape)}, got "
                             f"{tuple(E.shape)}")
        return n, k

    def _bt_rhs(self, D, b):
        """(b as (..., n, k, m), was_vector)."""
        if tuple(b.shape[:D.ndim - 1]) != tuple(D.shape[:-1]):
            raise ValueError(f"rhs shape {tuple(b.shape)} does not match D {tuple(D.shape)}")
        if b.ndim == D.ndim - 1:
            return b[..., None], True
        if b.ndim == D.ndim:
            return b, False
        raise ValueError(f"rhs must be (..., n, k) or (..., n, k, m), got {tuple(b.shape)}")

    # --- block-axis helpers (axis -3) ---
    def _zblk(self, like, n: int):
        return self.xp.zeros(tuple(like.shape[:-3]) + (n,) + tuple(like.shape[-2:]),
                             dtype=like.dtype)

    def _pad_end(self, a, n: int):
        m = a.shape[-3]
        return a if m >= n else self.xp.concatenate([a, self._zblk(a, n - m)], axis=-3)

    def _pad_front1(self, a):
        return self.xp.concatenate([self._zblk(a, 1), a], axis=-3)

    def _interleave(self, a, b):
        """``a`` at even block positions, ``b`` at odd ones (``len(a) - len(b)`` in {0, 1})."""
        xp = self.xp
        na, nb = a.shape[-3], b.shape[-3]
        if nb == 0:
            return a
        batch, tail = tuple(a.shape[:-3]), tuple(a.shape[-2:])
        pairs = xp.concatenate([xp.expand_dims(a[..., :nb, :, :], -3),
                                xp.expand_dims(b, -3)], axis=-3)
        out = pairs.reshape(batch + (2 * nb,) + tail)
        if na > nb:
            out = xp.concatenate([out, a[..., nb:, :, :]], axis=-3)
        return out

    def _eye_like(self, a):
        k = a.shape[-1]
        return self.xp.broadcast_to(self.xp.eye(k, dtype=a.dtype), tuple(a.shape))

    # --- raw kernels: linalg-dtype inputs; MLX: call inside _lin_ctx ---
    def _factor_impl(self, method: str):
        if method == "cr":
            return self._cr_factor_raw
        if method == "seq":
            return self._seq_factor_raw
        raise ValueError(f"bt_method must be 'cr' or 'seq', got {method!r}")

    def _bt_solve_raw(self, fac, b):
        if isinstance(fac, BlockTriCRFactor):
            return self._cr_solve_raw(fac, b)
        return self._seq_solve_raw(fac, b)

    def _bt_selinv_raw(self, fac):
        if isinstance(fac, BlockTriCRFactor):
            return self._cr_selinv_raw(fac)
        return self._seq_selinv_raw(fac)

    def _bt_logdet_raw(self, fac):
        xp = self.xp

        def ld(L):
            return 2.0 * xp.sum(xp.log(xp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)

        if isinstance(fac, BlockTriCRFactor):
            out = ld(fac.last)
            for L, _ in fac.levels:
                out = out + xp.sum(ld(L), axis=-1)
            return out
        return xp.sum(ld(fac.C), axis=-1)

    # sequential block Cholesky (block Thomas); JAX: lax.scan, MLX: Python loop
    def _seq_factor_raw(self, D, E):
        xp = self.xp
        n = D.shape[-3]
        D = self._sym(D)
        if self.is_jax:
            import jax

            Dt = xp.moveaxis(D, -3, 0)
            Et = xp.moveaxis(E, -3, 0)
            C0 = xp.linalg.cholesky(Dt[0])

            def step(Cp, inp):
                Di, Ei = inp
                Fi = self._mT(self._trsm(Cp, self._mT(Ei), lower=True))
                Ci = xp.linalg.cholesky(Di - Fi @ self._mT(Fi))
                return Ci, (Ci, Fi)

            _, (Cs, Fs) = jax.lax.scan(step, C0, (Dt[1:], Et))
            C = xp.concatenate([C0[None], Cs], axis=0)
            return BlockTriFactor(xp.moveaxis(C, 0, -3), xp.moveaxis(Fs, 0, -3))
        Cs = [self._chol(D[..., 0, :, :])]
        Fs = []
        for i in range(1, n):
            Fi = self._mT(self._trsm(Cs[-1], self._mT(E[..., i - 1, :, :]), lower=True))
            Cs.append(self._chol(D[..., i, :, :] - Fi @ self._mT(Fi)))
            Fs.append(Fi)
        C = xp.stack(Cs, axis=-3)
        F = xp.stack(Fs, axis=-3) if Fs else self._zblk(D, 0)
        return BlockTriFactor(C, F)

    def _seq_solve_raw(self, fac, b):
        """x = (C Cᵀ)⁻¹ b with b (..., n, k, m)."""
        xp = self.xp
        C, F = fac
        n = C.shape[-3]
        if self.is_jax:
            import jax

            Ct, Ft, bt = (xp.moveaxis(a, -3, 0) for a in (C, F, b))
            y0 = self._trsm(Ct[0], bt[0], lower=True)

            def fstep(yp, inp):
                Ci, Fi, bi = inp
                yi = self._trsm(Ci, bi - Fi @ yp, lower=True)
                return yi, yi

            _, ys = jax.lax.scan(fstep, y0, (Ct[1:], Ft, bt[1:]))
            y = xp.concatenate([y0[None], ys], axis=0)
            xl = self._trsm(Ct[-1], y[-1], lower=True, trans=True)

            def bstep(xn, inp):
                Ci, Fi, yi = inp
                xi = self._trsm(Ci, yi - self._mT(Fi) @ xn, lower=True, trans=True)
                return xi, xi

            _, xs = jax.lax.scan(bstep, xl, (Ct[:-1], Ft, y[:-1]), reverse=True)
            return xp.moveaxis(xp.concatenate([xs, xl[None]], axis=0), 0, -3)
        ys = [self._trsm(C[..., 0, :, :], b[..., 0, :, :], lower=True)]
        for i in range(1, n):
            ys.append(self._trsm(C[..., i, :, :],
                                 b[..., i, :, :] - F[..., i - 1, :, :] @ ys[-1], lower=True))
        xs = [None] * n
        xs[n - 1] = self._trsm(C[..., n - 1, :, :], ys[n - 1], lower=True, trans=True)
        for i in range(n - 2, -1, -1):
            xs[i] = self._trsm(C[..., i, :, :],
                               ys[i] - self._mT(F[..., i, :, :]) @ xs[i + 1],
                               lower=True, trans=True)
        return xp.stack(xs, axis=-3)

    def _seq_selinv_raw(self, fac):
        """Σ[n-1] = C⁻ᵀC⁻¹; G_i = F_i C_i⁻¹; Σ[i+1,i] = -Σ[i+1] G_i; Σ[i] = C_i⁻ᵀC_i⁻¹ + G_iᵀ Σ[i+1] G_i."""
        xp = self.xp
        C, F = fac
        n = C.shape[-3]
        Ci = self._trsm(C, self._eye_like(C), lower=True)  # C_i^{-1}, all blocks at once
        if self.is_jax:
            import jax

            Cit, Ft = xp.moveaxis(Ci, -3, 0), xp.moveaxis(F, -3, 0)
            Sl = self._mT(Cit[-1]) @ Cit[-1]

            def step(Sn, inp):
                Cinv, Fi = inp
                G = Fi @ Cinv
                Ssub = -(Sn @ G)
                Sii = self._sym(self._mT(Cinv) @ Cinv + self._mT(G) @ Sn @ G)
                return Sii, (Sii, Ssub)

            _, (Ss, Ssub) = jax.lax.scan(step, Sl, (Cit[:-1], Ft), reverse=True)
            Sd = xp.concatenate([Ss, Sl[None]], axis=0)
            return xp.moveaxis(Sd, 0, -3), xp.moveaxis(Ssub, 0, -3)
        Sd = [None] * n
        Ssub = [None] * max(n - 1, 0)
        Cl = Ci[..., n - 1, :, :]
        Sd[n - 1] = self._mT(Cl) @ Cl
        for i in range(n - 2, -1, -1):
            Cinv = Ci[..., i, :, :]
            G = F[..., i, :, :] @ Cinv
            Ssub[i] = -(Sd[i + 1] @ G)
            Sd[i] = self._sym(self._mT(Cinv) @ Cinv + self._mT(G) @ Sd[i + 1] @ G)
        S_sub = xp.stack(Ssub, axis=-3) if Ssub else self._zblk(C, 0)
        return xp.stack(Sd, axis=-3), S_sub

    # odd-even block cyclic reduction: ceil(log2 n) batched levels, no per-block Python loop.
    # Level: eliminate the odd blocks i = 2t+1 (block-diagonal among themselves):
    #   L_t = chol(D_i), W_t = L_t⁻¹ [A[i, i-1] | A[i, i+1]] = [Wl_t | Wr_t] (k x 2k; Wr = 0 past
    #   the end); the coarse system on the even blocks is the Schur complement A_ee - Wᵀ W, again
    #   block-tridiagonal: D'_m = D_2m - Wl_mᵀ Wl_m - Wr_{m-1}ᵀ Wr_{m-1},  E'_m = -Wr_mᵀ Wl_m
    #   (all read off one Gram product Wᵀ W).  It is the Cholesky factorization of a symmetric
    #   permutation of A, hence equally stable; ~2x the flops of the sequential sweep but
    #   O(log n) array operations instead of O(n) (what MLX's per-op overhead needs).
    def _cr_factor_raw(self, D, E):
        xp = self.xp
        k = D.shape[-1]
        D = self._sym(D)
        levels = []
        while D.shape[-3] > 1:
            n = D.shape[-3]
            no = n // 2
            ne = n - no
            El = E[..., 0::2, :, :]                           # A[2t+1, 2t],   t < no
            ErT = self._pad_end(self._mT(E[..., 1::2, :, :]) if ne > 1 else self._zblk(El, 0),
                                no)                           # A[2t+1, 2t+2], t < ne-1
            L = self._chol(D[..., 1::2, :, :])
            W = self._trsm(L, xp.concatenate([El, ErT], axis=-1), lower=True)  # (no, k, 2k)
            G = self._mT(W) @ W                               # [[Ul, ·], [WrᵀWl, Ur]]
            Dn = (D[..., 0::2, :, :] - self._pad_end(G[..., :k, :k], ne)
                  - self._pad_front1(G[..., :ne - 1, k:, k:]))
            En = -G[..., :ne - 1, k:, :k]
            levels.append((L, W))
            D, E = Dn, En
        return BlockTriCRFactor(tuple(levels), self._chol(D[..., 0, :, :]))

    def _cr_solve_raw(self, fac, b):
        xp = self.xp
        ys = []
        for L, W in fac.levels:
            n, no, k = b.shape[-3], L.shape[-3], L.shape[-1]
            ne = n - no
            y = self._trsm(L, b[..., 1::2, :, :], lower=True)
            t = self._mT(W) @ y                               # [Wlᵀ y ; Wrᵀ y]
            b = (b[..., 0::2, :, :] - self._pad_end(t[..., :k, :], ne)
                 - self._pad_front1(t[..., :ne - 1, k:, :]))
            ys.append(y)
        Lf = fac.last
        x = self._trsm(Lf, self._trsm(Lf, b[..., 0, :, :], lower=True), lower=True, trans=True)
        x = xp.expand_dims(x, -3)
        for (L, W), y in zip(reversed(fac.levels), reversed(ys)):
            no = L.shape[-3]
            xlr = xp.concatenate([x[..., :no, :, :], self._pad_end(x[..., 1:, :, :], no)],
                                 axis=-2)                     # [x_{i-1} ; x_{i+1}]
            xo = self._trsm(L, y - W @ xlr, lower=True, trans=True)
            x = self._interleave(x, xo)
        return x

    def _cr_selinv_raw(self, fac):
        """Selected inversion up the reduction tree (2x2 block inverse per level):
        Σ_oo = L⁻ᵀ(I + W Σ_ee Wᵀ)L⁻¹, Σ_oe = -L⁻ᵀ W Σ_ee; only the diagonal and first
        off-diagonal blocks of the coarse Σ_ee are needed."""
        xp = self.xp
        Lf = fac.last
        Li = self._trsm(Lf, self._eye_like(Lf), lower=True)
        Sd = xp.expand_dims(self._mT(Li) @ Li, -3)
        Ss = None
        for L, W in reversed(fac.levels):
            no, ne, k = L.shape[-3], Sd.shape[-3], L.shape[-1]
            eye = self._eye_like(L)
            Linv = self._trsm(L, eye, lower=True)
            S_ll = Sd[..., :no, :, :]                                  # Σ[i-1, i-1]
            S_rr = self._pad_end(Sd[..., 1:, :, :], no)                # Σ[i+1, i+1]
            S_rl = self._zblk(L, no) if Ss is None else self._pad_end(Ss, no)  # Σ[i+1, i-1]
            Sblk = xp.concatenate([xp.concatenate([S_ll, self._mT(S_rl)], axis=-1),
                                   xp.concatenate([S_rl, S_rr], axis=-1)], axis=-2)  # 2k x 2k
            P = W @ Sblk                                               # k x 2k
            LinvT = self._mT(Linv)
            Soo = self._sym(LinvT @ (eye + P @ self._mT(W)) @ Linv)
            S_lr = -(LinvT @ P)                                        # [Σ[i,i-1] | Σ[i,i+1]]
            Sd_new = self._interleave(Sd, Soo)
            Ss = self._interleave(S_lr[..., :k], self._mT(S_lr[..., :ne - 1, :, k:]))
            Sd = Sd_new
        return Sd, (self._zblk(Sd, 0) if Ss is None else Ss)

    # --- public block-tridiagonal API ---
    @property
    def _bridged(self) -> bool:
        """MLX block-tridiagonal kernels run through the JAX float64 bridge."""
        return self.is_mlx and self.bt_impl == "jax"

    def blocktri_factor(self, D, E):
        """Factor ``sym(A)`` once (a constant: inputs are stop-gradient'ed); reuse it with
        :meth:`blocktri_factor_solve`, :meth:`blocktri_selinv`, :meth:`blocktri_marginal_var`.
        Returns a :class:`BlockTriFactor` or :class:`BlockTriCRFactor` (per
        :attr:`bt_method`; the MLX->JAX bridge always uses ``"seq"``), in
        :attr:`linalg_dtype`."""
        self._bt_check(D, E)
        D, E = self.stop_gradient(D), self.stop_gradient(E)
        if self._bridged:
            fac = self._bridge_or_none(
                lambda: self._mx64(self._bridge_call("factor", *self._host64((D, E)))))
            if fac is not None:
                return fac
            with self._lin_ctx():  # traced (mx.compile / mx.vmap): native sequential sweep
                return self._seq_factor_raw(self._lin(D), self._lin(E))
        m = self.bt_method
        with self._lin_ctx():
            fn = (self._jax_jitted(("factor", m), self._factor_impl(m))
                  if self.is_jax else self._factor_impl(m))
            return fn(self._lin(D), self._lin(E))

    def blocktri_factor_solve(self, fac, b):
        """``A⁻¹ b`` from a precomputed factor; differentiable w.r.t. ``b`` only."""
        if isinstance(fac, BlockTriFactor):
            nblk, k, batch = fac.C.shape[-3], fac.C.shape[-1], tuple(fac.C.shape[:-3])
        else:
            nblk, k, batch = self._cr_nblocks(fac), fac.last.shape[-1], tuple(fac.last.shape[:-2])
        # vector vs matrix rhs from the rank (shape matching alone is ambiguous when n == k == m)
        nb = len(batch)
        if b.ndim not in (nb + 2, nb + 3) or tuple(b.shape[:nb + 2]) != (*batch, nblk, k):
            raise ValueError(f"rhs {tuple(b.shape)} does not match the factor (batch {batch}, "
                             f"{nblk} blocks of {k})")
        vec = b.ndim == nb + 2
        bm = b[..., None] if vec else b
        leaves, td = tree_flatten(fac)
        fn = self._custom(("bt_factor_solve", td, self._bridged),
                          lambda: self._build_bt_factor_solve(td), jit=True)
        x = fn(*leaves, bm)  # no gradient reaches the factor (bwd returns None for it)
        return x[..., 0] if vec else x

    @staticmethod
    def _cr_nblocks(fac) -> int:
        n = 1
        for L, _ in reversed(fac.levels):
            n = n + L.shape[-3]
        return n

    def _build_bt_factor_solve(self, td):
        bridged = self._bridged

        def fwd(*args):
            *leaves, b = args
            if bridged:
                x = self._bridge_or_none(lambda: self._bridge_call(
                    "factor_solve", tree_unflatten(td, self._host64(leaves)), self._host64(b)))
                if x is not None:
                    return self._mx_from_host(x, b.dtype), ()
            fac = tree_unflatten(td, leaves)
            with self._lin_ctx():
                return self._bt_solve_raw(fac, self._lin(b)).astype(b.dtype), ()

        def bwd(args, x, res, g):
            *leaves, b = args
            if bridged:
                gb = self._bridge_or_none(lambda: self._bridge_call(
                    "factor_solve", tree_unflatten(td, self._host64(leaves)), self._host64(g)))
                if gb is not None:
                    return (None,) * len(leaves) + (self._mx_from_host(gb, b.dtype),)
            fac = tree_unflatten(td, leaves)
            with self._lin_ctx():
                return (None,) * len(leaves) + (self._bt_solve_raw(fac, self._lin(g)),)

        return self.custom_vjp(fwd, bwd, cpu=True)

    def blocktri_solve(self, D, E, b):
        """Solve ``A x = b`` for block-tridiagonal SPD ``A = (sym(D), E)``.

        Differentiable w.r.t. ``D``, ``E`` and ``b`` (reverse mode).  The adjoint solve reuses the
        forward factor: ``λ = A⁻¹ x̄``, ``b̄ = λ``, ``D̄_i = -sym(λ_i x_iᵀ)``,
        ``Ē_i = -(λ_{i+1} x_iᵀ + x_{i+1} λ_iᵀ)``.  Computed in :attr:`linalg_dtype` (MLX: float64,
        through the JAX bridge or on the CPU stream); ``x`` has ``b``'s dtype.  Batch over leading
        dimensions rather than ``vmap`` (MLX custom functions here define no vmap rule); under
        ``mx.compile`` the MLX bridge falls back to the native kernels.
        """
        self._bt_check(D, E)
        bm, vec = self._bt_rhs(D, b)
        m = "bridge" if self._bridged else self.bt_method
        x = self._custom(("bt_solve", m), lambda: self._build_bt_solve(m), jit=True)(D, E, bm)
        return x[..., 0] if vec else x

    def _bt_solve_cotangents(self, lam, x):
        lamT, xT = self._mT(lam), self._mT(x)
        gD = -self._sym(lam @ xT)
        gE = -(lam[..., 1:, :, :] @ xT[..., :-1, :, :] + x[..., 1:, :, :] @ lamT[..., :-1, :, :])
        return gD, gE, lam

    def _bt_logdet_cotangents(self, fac, g):
        Sd, Ss = self._bt_selinv_raw(fac)
        gg = g[..., None, None, None]
        return gg * Sd, 2.0 * gg * Ss

    def _build_bt_solve(self, method: str):
        if method == "bridge":
            native_fwd, native_bwd = self._bt_solve_fwd_bwd(self._seq_factor_raw)

            def fwd(D, E, b):
                out = self._bridge_or_none(
                    lambda: self._bridge_call("solve", *self._host64((D, E, b))))
                if out is None:  # traced (mx.compile / mx.vmap): same residual structure
                    return native_fwd(D, E, b)
                x, fac = out
                return self._mx_from_host(x, b.dtype), (self._mx64(fac), self._mx64(x))

            def bwd(args, x, res, g):
                fac, x64 = res
                D, E, b = args
                out = self._bridge_or_none(lambda: self._bridge_call(
                    "solve_bwd", self._host64(fac), self._host64(x64), self._host64(g)))
                if out is None:
                    return native_bwd(args, x, res, g)
                gD, gE, gb = out
                return (self._mx_from_host(gD, D.dtype), self._mx_from_host(gE, E.dtype),
                        self._mx_from_host(gb, b.dtype))

            return self.custom_vjp(fwd, bwd, cpu=True)

        return self.custom_vjp(*self._bt_solve_fwd_bwd(self._factor_impl(method)), cpu=True)

    def _bt_solve_fwd_bwd(self, factor):

        def fwd(D, E, b):
            with self._lin_ctx():
                fac = factor(self._lin(D), self._lin(E))
                x = self._bt_solve_raw(fac, self._lin(b))
                return x.astype(b.dtype), (fac, x)

        def bwd(args, x, res, g):
            fac, x64 = res
            with self._lin_ctx():
                lam = self._bt_solve_raw(fac, self._lin(g))
                return self._bt_solve_cotangents(lam, x64)

        return fwd, bwd

    def blocktri_logdet(self, D, E):
        """``log det A`` (batched), differentiable: ``D̄_i = ḡ Σ_ii``, ``Ē_i = 2 ḡ Σ_{i+1,i}``
        with the Σ = A⁻¹ blocks from selected inversion."""
        self._bt_check(D, E)
        m = "bridge" if self._bridged else self.bt_method
        return self._custom(("bt_logdet", m), lambda: self._build_bt_logdet(m), jit=True)(D, E)

    def _build_bt_logdet(self, method: str):
        if method == "bridge":
            native_fwd, native_bwd = self._bt_logdet_fwd_bwd(self._seq_factor_raw)

            def fwd(D, E):
                out = self._bridge_or_none(
                    lambda: self._bridge_call("logdet", *self._host64((D, E))))
                if out is None:
                    return native_fwd(D, E)
                ld, fac = out
                return self._mx_from_host(ld, D.dtype), (self._mx64(fac),)

            def bwd(args, ld, res, g):
                (fac,) = res
                D, E = args
                out = self._bridge_or_none(lambda: self._bridge_call(
                    "logdet_bwd", self._host64(fac), self._host64(g)))
                if out is None:
                    return native_bwd(args, ld, res, g)
                gD, gE = out
                return self._mx_from_host(gD, D.dtype), self._mx_from_host(gE, E.dtype)

            return self.custom_vjp(fwd, bwd, cpu=True)

        return self.custom_vjp(*self._bt_logdet_fwd_bwd(self._factor_impl(method)), cpu=True)

    def _bt_logdet_fwd_bwd(self, factor):

        def fwd(D, E):
            with self._lin_ctx():
                fac = factor(self._lin(D), self._lin(E))
                return self._bt_logdet_raw(fac).astype(D.dtype), (fac,)

        def bwd(args, ld, res, g):
            (fac,) = res
            with self._lin_ctx():
                return self._bt_logdet_cotangents(fac, self._lin(g))

        return fwd, bwd

    def blocktri_selinv(self, D_or_fac, E=None, dtype=None):
        """``(S_diag, S_sub)``: the diagonal blocks ``(A⁻¹)[i, i]`` and sub-diagonal blocks
        ``(A⁻¹)[i+1, i]`` by selected inversion (O(n k³), never the dense inverse).  Pass
        ``(D, E)`` or a factor.  Constant (no gradient).  Output dtype: ``dtype``, else ``D``'s,
        else (factor input) the backend float dtype."""
        Sd, Ss = self._bt_selinv_any(D_or_fac, E)
        out_dt = self._selinv_dtype(D_or_fac, dtype)
        with self._lin_ctx():
            return Sd.astype(out_dt), Ss.astype(out_dt)

    def blocktri_marginal_var(self, D_or_fac, E=None, dtype=None):
        """Diagonal of ``A⁻¹`` as ``(..., n, k)``: the true marginal variances of the solution
        (not ``1/diag(A)``, which understates them).  Constant (no gradient)."""
        Sd, _ = self._bt_selinv_any(D_or_fac, E)
        out_dt = self._selinv_dtype(D_or_fac, dtype)
        with self._lin_ctx():
            return self.xp.diagonal(Sd, axis1=-2, axis2=-1).astype(out_dt)

    def _selinv_dtype(self, D_or_fac, dtype):
        if dtype is not None:
            return self._native_dtype(dtype)
        if isinstance(D_or_fac, (BlockTriFactor, BlockTriCRFactor)):
            return self.dtype
        return D_or_fac.dtype

    def _bt_selinv_any(self, D_or_fac, E):
        fac = (D_or_fac if isinstance(D_or_fac, (BlockTriFactor, BlockTriCRFactor))
               else self.blocktri_factor(D_or_fac, E))
        if self._bridged:
            out = self._bridge_or_none(
                lambda: self._mx64(self._bridge_call("selinv", self._host64(fac))))
            if out is not None:
                return out
        with self._lin_ctx():
            if self.is_jax:
                return self._jax_jitted("selinv", self._bt_selinv_raw)(fac)
            return self._bt_selinv_raw(fac)

    def _jax_jitted(self, key, fn):
        """Cached ``jax.jit`` of a raw kernel (eager JAX dispatch of scans is slow)."""
        import jax

        return self._custom(("jit", key), lambda: jax.jit(fn))

    # --- MLX -> JAX float64 bridge (bt_impl="jax") ---
    # MLX's CPU linear algebra is 2-3x slower per call than XLA's and the sequential sweep is
    # O(n) MLX ops, so MLX's block-tridiagonal kernels default to running the jitted JAX float64
    # kernels on host copies inside the MLX custom VJP (the MLX gradient graph is unaffected).
    # The copies force evaluation, which MLX forbids while tracing mx.compile / mx.vmap; there
    # the kernels fall back to the native sequential sweep automatically (same residual layout).
    # JAX's x64 mode is enabled only inside these calls.
    @staticmethod
    def _bridge_or_none(thunk):
        """``thunk()``, or None when MLX refuses the host copy because it is tracing."""
        try:
            return thunk()
        except ValueError as e:
            if "function transformations" in str(e):
                return None
            raise

    def _host64(self, tree):
        """MLX arrays (pytree) -> numpy float64 (evaluates the MLX graph)."""
        return tree_map(lambda a: np.asarray(a, dtype=np.float64), tree)

    def _mx64(self, tree):
        """numpy/JAX arrays (pytree) -> MLX float64 arrays on the CPU stream."""
        mx = self.xp
        with mx.stream(mx.cpu):
            return tree_map(lambda a: mx.array(np.asarray(a, dtype=np.float64),
                                               dtype=mx.float64), tree)

    def _mx_from_host(self, a, dtype):
        mx = self.xp
        a = np.asarray(a)
        if dtype == mx.float64:
            with mx.stream(mx.cpu):
                return mx.array(a, dtype=mx.float64)
        return mx.array(a.astype(np.float32) if dtype == mx.float32 else a, dtype=dtype)

    def _bridge_call(self, name: str, *args):
        from jax.experimental import enable_x64

        jb = _jax_bridge_backend()
        with enable_x64():
            return tree_map(np.asarray, jb._bridge_kernel(name)(*args))

    def _bridge_kernel(self, name: str):
        """(on the hidden JAX float64 bridge backend) the jitted kernel ``name``."""
        import jax

        def build():
            factor = self._factor_impl("seq")
            if name == "solve":
                def f(D, E, b):
                    fac = factor(D, E)
                    return self._bt_solve_raw(fac, b), fac
            elif name == "solve_bwd":
                def f(fac, x, g):
                    return self._bt_solve_cotangents(self._bt_solve_raw(fac, g), x)
            elif name == "logdet":
                def f(D, E):
                    fac = factor(D, E)
                    return self._bt_logdet_raw(fac), fac
            elif name == "logdet_bwd":
                f = self._bt_logdet_cotangents
            elif name == "factor":
                f = factor
            elif name == "factor_solve":
                f = self._bt_solve_raw
            elif name == "selinv":
                f = self._bt_selinv_raw
            else:
                raise KeyError(name)
            return jax.jit(f)

        return self._custom(("bridge", name), build)

    def blocktri_matvec(self, D, E, x):
        """``A x`` for block-tridiagonal ``A = (sym(D), E)``; plain differentiable xp ops."""
        xp = self.xp
        self._bt_check(D, E)
        xm, vec = self._bt_rhs(D, x)
        y = self._sym(D) @ xm
        if D.shape[-3] > 1:
            z = xp.zeros(tuple(xm.shape[:-3]) + (1,) + tuple(xm.shape[-2:]), dtype=y.dtype)
            low = E @ xm[..., :-1, :, :]               # A[i+1, i] x_i  -> block i+1
            up = self._mT(E) @ xm[..., 1:, :, :]       # A[i, i+1] x_{i+1} -> block i
            y = y + xp.concatenate([z, low], axis=-3) + xp.concatenate([up, z], axis=-3)
        return y[..., 0] if vec else y

    # --- banded storage <-> block-tridiagonal ---
    @staticmethod
    @lru_cache(maxsize=64)
    def _band_index(p: int, N: int, k: int):
        """Gather indices into ``concat(ab.ravel(), [0, 1])`` for D (nb,k,k) and E (nb-1,k,k)."""
        nb = -(-N // k)
        ZERO, ONE = (p + 1) * N, (p + 1) * N + 1
        r = np.arange(k)[:, None]
        c = np.arange(k)[None, :]
        blk = np.arange(nb)[:, None, None]
        i = blk * k + r
        j = blk * k + c
        lo, hi = np.maximum(i, j), np.minimum(i, j)
        d = lo - hi
        ok = (lo < N) & (d <= p)
        iD = np.where(ok, d * N + hi, ZERO)
        iD = np.where((i >= N) & (i == j), ONE, iD)
        if nb > 1:
            Ie = np.arange(nb - 1)[:, None, None]
            ie = (Ie + 1) * k + r
            je = Ie * k + c
            de = ie - je
            iE = np.where((ie < N) & (de <= p), de * N + je, ZERO)
        else:
            iE = np.zeros((0, k, k), dtype=np.int64)
        return iD.astype(np.int32), iE.astype(np.int32)

    def band_to_blocktri(self, ab, k: int | None = None):
        """LAPACK lower band storage ``ab`` (``(..., p+1, N)``, ``ab[d, j] = A[j+d, j]``) ->
        ``(D, E)`` with block size ``k >= p`` (default ``max(p, 1)``).  ``N`` is padded up to a
        multiple of ``k`` with unit diagonal entries (pad the rhs with zeros:
        :meth:`pad_blocks`).  A differentiable gather (VJP: scatter-add into ``ab``)."""
        xp = self.xp
        p1, N = ab.shape[-2], ab.shape[-1]
        p = p1 - 1
        k = max(p, 1) if k is None else int(k)
        if k < p:
            raise ValueError(f"block size k={k} must be >= the half-bandwidth p={p}")
        iD, iE = self._band_index(p, N, k)
        nb = iD.shape[0]
        batch = tuple(ab.shape[:-2])
        flat = ab.reshape(batch + (p1 * N,))
        tail = xp.concatenate([xp.zeros(batch + (1,), dtype=ab.dtype),
                               xp.ones(batch + (1,), dtype=ab.dtype)], axis=-1)
        flat = xp.concatenate([flat, tail], axis=-1)
        D = xp.take(flat, self.asarray(iD.ravel()), axis=-1).reshape(batch + (nb, k, k))
        E = xp.take(flat, self.asarray(iE.ravel()), axis=-1).reshape(batch + (nb - 1, k, k))
        return D, E

    def pad_blocks(self, b, k: int, axis: int = -1):
        """Zero-pad axis ``axis`` (length N) to a multiple of ``k`` and split it into
        ``(nb, k)``: ``(..., N)`` -> ``(..., nb, k)`` (``axis=-1``) or ``(..., N, m)`` ->
        ``(..., nb, k, m)`` (``axis=-2``)."""
        xp = self.xp
        ax = axis % b.ndim
        N = b.shape[ax]
        nb = -(-N // k)
        if nb * k != N:
            pad_shape = list(b.shape)
            pad_shape[ax] = nb * k - N
            b = xp.concatenate([b, xp.zeros(tuple(pad_shape), dtype=b.dtype)], axis=ax)
        return b.reshape(tuple(b.shape[:ax]) + (nb, k) + tuple(b.shape[ax + 1:]))

    def unpad_blocks(self, x, N: int, axis: int = -1):
        """Inverse of :meth:`pad_blocks`: merge the ``(nb, k)`` axes ending at ``axis``
        (``-1`` for vectors, ``-2`` for matrix right-hand sides) and keep the first ``N``."""
        ax = axis % x.ndim  # the k axis
        shape = tuple(x.shape[:ax - 1]) + (x.shape[ax - 1] * x.shape[ax],) + tuple(x.shape[ax + 1:])
        y = x.reshape(shape)
        sl = [slice(None)] * y.ndim
        sl[ax - 1] = slice(0, N)
        return y[tuple(sl)]

    def banded_solve(self, ab, b, k: int | None = None):
        """Solve ``A x = b`` for SPD ``A`` in lower band storage ``ab`` (``(..., p+1, N)``),
        ``b`` ``(..., N)`` or ``(..., N, m)``; differentiable w.r.t. ``ab`` and ``b``."""
        p1, N = ab.shape[-2], ab.shape[-1]
        k = max(p1 - 1, 1) if k is None else int(k)
        D, E = self.band_to_blocktri(ab, k)
        axis = -1 if b.ndim == ab.ndim - 1 else -2
        x = self.blocktri_solve(D, E, self.pad_blocks(b, k, axis=axis))
        return self.unpad_blocks(x, N, axis=axis)


# ===================================================================== factories
@cache
def _make_backend(name: str, precision: str) -> Backend:
    if name == "mlx":
        import mlx.core as mx

        if precision != "float32":
            raise ValueError("MLX GPU has no float64; use get_backend('jax', 'float64') "
                             "(MLX linear algebra already runs in float64 on the CPU stream)")
        import importlib.util

        # the JAX bridge is imported lazily, on the first block-tridiagonal call
        impl = "jax" if importlib.util.find_spec("jax") is not None else "native"
        return Backend("mlx", "float32", mx, mx.float32, np.float32, mx.float64,
                       bt_method="cr", bt_impl=impl)
    if name == "jax":
        import jax

        if precision == "float64":
            jax.config.update("jax_enable_x64", True)
        elif precision != "float32":
            raise ValueError(f"precision must be 'float32' or 'float64', got {precision!r}")
        import jax.numpy as jnp

        dt = jnp.float64 if precision == "float64" else jnp.float32
        return Backend("jax", precision, jnp, dt, _np_float(precision), dt)
    raise ValueError(f"unknown backend {name!r} (use 'mlx' or 'jax')")


@lru_cache(maxsize=1)
def _jax_bridge_backend() -> Backend:
    """Hidden JAX float64 backend for MLX's block-tridiagonal kernels; always called inside
    ``jax.experimental.enable_x64()`` so the process-wide x64 flag is left alone."""
    import jax.numpy as jnp

    return Backend("jax", "float64", jnp, jnp.float64, np.float64, jnp.float64, bt_method="seq")


def get_backend(name: str = "mlx", precision: str | None = None) -> Backend:
    """``get_backend("mlx")`` (GPU float32), ``get_backend("jax", "float64")`` (CPU oracle) or
    ``get_backend("jax")`` (float32).  Aliases: ``"jax64"``, ``"jax32"``.  Cached per
    (name, precision)."""
    alias = {"jax64": ("jax", "float64"), "jax32": ("jax", "float32"),
             "mlx32": ("mlx", "float32")}
    if name in alias:
        if precision is not None and precision != alias[name][1]:
            raise ValueError(f"{name!r} implies precision {alias[name][1]!r}")
        name, precision = alias[name]
    return _make_backend(name, precision or "float32")


def default_backend() -> Backend:
    """``$DECANTER_FORWARD_BACKEND`` (``mlx`` | ``jax`` | ``jax64``) or MLX when a Metal GPU is
    available, else JAX float32."""
    env = os.environ.get("DECANTER_FORWARD_BACKEND", "").strip().lower()
    if env:
        return get_backend(env)
    return get_backend("mlx") if mlx_gpu_available() else get_backend("jax")


# ===================================================================== testing helpers
def blocktri_to_dense(D, E) -> np.ndarray:
    """Dense ``(..., n k, n k)`` numpy matrix ``sym(D)`` on the diagonal, ``E`` below, ``Eᵀ``
    above (for tests and debugging)."""
    D, E = np.asarray(D, dtype=float), np.asarray(E, dtype=float)
    n, k = D.shape[-3], D.shape[-1]
    batch = D.shape[:-3]
    A = np.zeros(batch + (n * k, n * k))
    Ds = 0.5 * (D + np.swapaxes(D, -1, -2))
    for i in range(n):
        A[..., i * k:(i + 1) * k, i * k:(i + 1) * k] = Ds[..., i, :, :]
        if i + 1 < n:
            A[..., (i + 1) * k:(i + 2) * k, i * k:(i + 1) * k] = E[..., i, :, :]
            A[..., i * k:(i + 1) * k, (i + 1) * k:(i + 2) * k] = np.swapaxes(E[..., i, :, :],
                                                                           -1, -2)
    return A


def check_grads(f: Callable, args: tuple, bk: Backend, *, argnums=0, n_dirs: int = 3,
                eps: float | None = None, seed: int = 0) -> float:
    """Directional finite-difference check of ``bk.value_and_grad(f)`` for scalar ``f``.

    For random unit directions ``v`` in the ``argnums`` arguments, compares ``<∇f, v>`` with the
    central difference ``[f(x + h v) - f(x - h v)] / 2h`` (``h = eps`` scaled by the argument
    size; default 1e-6 for float64, 3e-3 for float32).  Returns the worst relative error
    ``|fd - ad| / max(|fd|, |ad|, tiny)``; use JAX float64 for tight (≲1e-6) checks.
    """
    nums = (argnums,) if isinstance(argnums, int) else tuple(argnums)
    np_args = [np.asarray(bk.to_numpy(a), dtype=np.float64) if i in nums else a
               for i, a in enumerate(args)]
    vg = bk.value_and_grad(f, argnums=nums)
    _, grads = vg(*[bk.asarray(a) if i in nums else a for i, a in enumerate(np_args)])
    grads = [np.asarray(bk.to_numpy(g), dtype=np.float64) for g in grads]
    h0 = eps if eps is not None else (1e-6 if bk.np_dtype == np.float64 else 3e-3)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_dirs):
        vs = {i: rng.normal(size=np.shape(np_args[i])) for i in nums}
        nrm = math.sqrt(sum(float(np.sum(v * v)) for v in vs.values())) or 1.0
        vs = {i: v / nrm for i, v in vs.items()}
        scale = max(max(float(np.max(np.abs(np_args[i]))) for i in nums) if nums else 1.0, 1.0)
        h = h0 * scale

        def at(sgn, h=h, vs=vs):
            a = [bk.asarray(np_args[i] + sgn * h * vs[i]) if i in nums else np_args[i]
                 for i in range(len(np_args))]
            return float(np.asarray(bk.to_numpy(f(*a)), dtype=np.float64))

        fd = (at(+1) - at(-1)) / (2 * h)
        ad = sum(float(np.sum(g * vs[i])) for g, i in zip(grads, nums))
        err = abs(fd - ad) / max(abs(fd), abs(ad), 1e-300)
        worst = max(worst, err)
    return worst
