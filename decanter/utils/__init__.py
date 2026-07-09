"""Shared low-level utilities.

Submodules:
    * :mod:`decanter.utils.median_filter` — SciPy wrapper with pinned
      boundary mode (for parity diff stability).
    * :mod:`decanter.utils.sigma_clip` — vectorized per-segment sigma
      clip used by s02 (CR) and elsewhere.
    * :mod:`decanter.utils.interp` — cubic-spline resampling helpers
      (Phase-2 path will swap in a JAX-friendly variant).
"""
