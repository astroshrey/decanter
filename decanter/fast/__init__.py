"""decanter's default WARP-clone implementation: vectorised and calibration-cached.

``decanter.reduce(...)`` (``mode="warp"``, the default) and ``reduce_many`` / ``reduce_to_warp_tree`` run the WARP-clone
recipe through this package -- same stages, same apertures, same cross-frame
shift, same dispcor grid and FSR cuts as the scalar reference port
(``mode="warp-reference"``) -- with the same arithmetic, restructured:

* s02 cosmic-ray mask (:func:`cr_mask`): sorting-network median filters and
  precomputed tile maps;
* s06 rectification (:func:`decanter.fast.rectify.rectify_orders_multi`):
  IRAF ``transform`` with every image-independent quantity cached, the spline
  solves batched over all orders and images, and only the extraction band of
  each strip evaluated;
* s08 ``apall`` (:func:`decanter.fast.apsum.apsum_multi`): warp mode's box
  sum on the object and sky strips together, aperture geometry cached;
* s12 dispcor map (:func:`decanter.fast.wavelength.dispcor_edges_fast`).

Everything else (s01, s03-s05, s07 center search, s10, s11, s13) is the
reference port's own code. The results are bit-identical to
``mode="warp-reference"`` (``docs/VECTORISED.md``; on the MLX backend up to
float32 subnormals in s06 strip pixels the extraction never reads).

Backends: s06 runs on the Apple GPU through MLX when ``mlx`` imports and has a
GPU, else in NumPy (IRAF's float32 arithmetic with exactly rounded fused
multiply-adds). Override with ``DECANTER_FAST_BACKEND=numpy|mlx`` or
:func:`set_backend` / the :func:`backend` context manager.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from decanter.fast._backend import active_backend, backend, mlx_available, set_backend

__all__ = [
    "active_backend",
    "backend",
    "clear_cache",
    "cr_mask",
    "mlx_available",
    "set_backend",
]

_CACHE: dict = {}
_LOCK = threading.Lock()


def _file_key(p) -> tuple:
    p = Path(p)
    st = p.stat()
    return (str(p.resolve()), st.st_mtime_ns, st.st_size)


def _cached(key: tuple, build):
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    val = build()
    with _LOCK:
        if len(_CACHE) >= 16:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = val
    return val


def clear_cache() -> None:
    """Drop every vectorised-path calibration cache (the IRAF transform plans are
    shared with the reference port and kept)."""
    from decanter.fast import rectify

    with _LOCK:
        _CACHE.clear()
    rectify.clear_cache()


def cr_mask(diff: NDArray, obj_raw: NDArray, sky_raw: NDArray, calib, static_bp: NDArray, *,
            nodpos: str, ndr_obj: int, ndr_sky: int, config,
            stats: dict | None = None) -> NDArray[np.int16]:
    """Vectorised :func:`decanter.image2d.cr_mask` (bit-identical; geometry cached).

    ``stats`` (optional dict) receives WARP's returned ``pixnum`` and
    ``sigthres`` (the cosmicray_log values).

    WARP equivalent: ``warp/badpixmask.py:cosmicRayMask``.
    """
    from decanter.calib.aperture import ApertureSet
    from decanter.fast import image2d as f2d

    geom = _cached(("cr", _file_key(calib.apdb_multihole), _file_key(calib.static_bp_mask)),
                   lambda: f2d.CrGeometry.build(ApertureSet.load(calib.apdb_multihole),
                                                static_bp))
    return f2d.cr_mask_fast(
        diff, obj_raw, sky_raw, geom, ndr1=ndr_obj, ndr2=ndr_sky, abba="O" not in nodpos,
        threshold=config.CR_threshold, max_sigma=config.CR_max_sigma,
        varatio=config.CR_var_ratio, slitposratio=config.CR_slitpos_ratio,
        fixsigma=config.CR_fix_sigma, stats=stats)
