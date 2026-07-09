"""decanter — a pure-Python port of WARP for WINERED echelle reductions.

User-facing API:

    decanter.reduce(obj, sky, calib, *, workdir=None, save_intermediates=False)
        Single-frame reduction. Returns a :class:`Reduction` with
        per-(fsr_cut, order) calibrated 1D spectra. Composes
        :mod:`decanter.image2d` → :mod:`decanter.rectify` →
        :mod:`decanter.extract` → :mod:`decanter.wavelength`. No
        cross-frame waveshift (waveshift is relative across frames
        and meaningless for a single frame).

    decanter.combine(...)
        Multi-frame SNR-weighted stack. Currently a stub
        (raises NotImplementedError). For transit-style per-frame
        analysis, loop :func:`reduce` over your frame list.

    decanter.Calibration.from_dir(reduc_root)
        Auto-discover all calibration paths from a WARP-style
        ``calibration_data/`` directory.

See ``CLAUDE.md`` / ``HANDOFF.md`` for architecture notes and
``PLAN.md`` for the Phase-1 design.
"""

__version__ = "0.0.1"

from decanter._reduction import Intermediates, OrderSpectrum, Reduction
from decanter.api import combine, reduce
from decanter.calib import Calibration, CalibrationMismatch, InstrumentConfig
from decanter.config import Config

__all__ = [
    "Calibration",
    "CalibrationMismatch",
    "Config",
    "InstrumentConfig",
    "Intermediates",
    "OrderSpectrum",
    "Reduction",
    "combine",
    "reduce",
]
