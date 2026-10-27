"""Calibration data handlers.

Submodules:
    * :mod:`decanter.calib.aperture` — :class:`ApertureSet` mirroring
      WARP's ``apertureSet`` class (order list, traces, masks, slit
      coords).
    * :mod:`decanter.calib.transform` — the ``fc`` surface loader and the
      per-order rectification kernel.
    * :mod:`decanter.calib.flat` — master flat construction (not yet
      implemented; calibration sets come from ``Warp_calib.py``).
    * :mod:`decanter.calib.trace` — aperture tracing from the multihole frame.
    * :mod:`decanter.calib.discover` — :class:`Calibration` dataclass
      bundling all WARP calibration paths, plus :meth:`Calibration.from_dir`
      for auto-discovery from a WARP-style ``calibration_data/`` directory.
"""

from decanter.calib.discover import (
    Calibration,
    CalibrationFallbackWarning,
    CalibrationMismatch,
    LockedAperturesWarning,
    InstrumentConfig,
)

__all__ = ["Calibration", "CalibrationFallbackWarning", "CalibrationMismatch", "InstrumentConfig", "LockedAperturesWarning"]
