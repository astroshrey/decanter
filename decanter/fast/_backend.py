"""Array backend selection for the default (vectorised) path: NumPy or MLX (optional).

MLX is used where it measurably helps: the rectification step's batched
natural-spline solves and bicubic-spline evaluation (custom Metal kernels).
Everything else runs in NumPy/SciPy, so the default path works unchanged when ``mlx``
is not installed.

Selection, in order of precedence:

* :func:`set_backend` (``"numpy"``, ``"mlx"`` or ``"auto"``);
* the ``DECANTER_FAST_BACKEND`` environment variable (same values);
* ``"auto"``: MLX if it imports and its default device is a GPU, else NumPy.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

_VALID = ("auto", "numpy", "mlx")
_override: str | None = None
_mx = None
_mx_checked = False


def _load_mlx():
    """Import ``mlx.core`` once; return the module or None."""
    global _mx, _mx_checked
    if not _mx_checked:
        _mx_checked = True
        try:
            import mlx.core as mx  # noqa: PLC0415

            # A GPU device is the reason to use MLX at all; on a CPU-only
            # build NumPy float64 is both faster and more precise. Fast mode
            # needs custom Metal kernels (mx.fast.metal_kernel).
            if ("gpu" in str(mx.default_device()).lower()
                    and hasattr(mx.fast, "metal_kernel")):
                _mx = mx
        except Exception:  # ImportError, or a broken Metal runtime
            _mx = None
    return _mx


def mlx_available() -> bool:
    """True if MLX imports and runs on a GPU."""
    return _load_mlx() is not None


def set_backend(name: str | None) -> None:
    """Force the fast-mode backend (``None`` restores env/auto selection)."""
    if name is not None and name not in _VALID:
        raise ValueError(f"backend must be one of {_VALID}, got {name!r}")
    global _override
    _override = name


@contextmanager
def backend(name: str) -> Iterator[None]:
    """Context manager: temporarily select a backend."""
    global _override
    prev = _override
    set_backend(name)
    try:
        yield
    finally:
        _override = prev


def active_backend() -> str:
    """Resolve the backend in effect: ``"numpy"`` or ``"mlx"``."""
    name = _override or os.environ.get("DECANTER_FAST_BACKEND", "auto").strip().lower()
    if name not in _VALID:
        name = "auto"
    if name == "numpy":
        return "numpy"
    if name == "mlx":
        if not mlx_available():
            raise RuntimeError("DECANTER_FAST_BACKEND=mlx but mlx (GPU) is not available")
        return "mlx"
    return "mlx" if mlx_available() else "numpy"


def mx():
    """The ``mlx.core`` module (only call when :func:`active_backend` is mlx)."""
    m = _load_mlx()
    if m is None:
        raise RuntimeError("mlx is not available")
    return m
