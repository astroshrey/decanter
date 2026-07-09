"""Calibration data handlers.

Submodules:
    * :mod:`decanter.calib.aperture` — :class:`ApertureSet` mirroring
      WARP's ``apertureSet`` class (order list, traces, masks, slit
      coords).
    * :mod:`decanter.calib.transform` — angle measurement and aperture
      → flat-coordinates conversion (replaces WARP's
      ``angle_measure.py`` and ``AP_FC_conversion.py``).
    * :mod:`decanter.calib.flat` — master flat construction (Phase 2;
      consumed from disk in Phase 1).
    * :mod:`decanter.calib.discover` — :class:`Calibration` dataclass
      bundling all WARP calibration paths, plus :meth:`Calibration.from_dir`
      for auto-discovery from a WARP-style ``calibration_data/`` directory.
"""

from decanter.calib.discover import (
    Calibration,
    CalibrationMismatch,
    InstrumentConfig,
)

__all__ = ["Calibration", "CalibrationMismatch", "InstrumentConfig"]
