"""Geometric rectification of curved echelle orders.

Each per-order task in :mod:`decanter.rectify.transform` consumes the
clean 2D frame from :mod:`decanter.image2d` and produces a per-order
rectified strip (cross-dispersion × wavelength) via the IRAF
fitcoords surface stored in the calibration ``fc`` files.

WARP equivalent: ``warp/cutransform.py``.
"""

from decanter.rectify.transform import rectify_orders

__all__ = ["rectify_orders"]
