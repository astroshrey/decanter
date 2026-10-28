"""decanter — a pure-Python port of WARP for WINERED echelle reductions.

User-facing API:

    decanter.reduce(obj, calib, *, sky=None, mode="warp", workdir=None, ...)
        Single-frame reduction. Returns a :class:`Reduction` with
        per-(fsr_cut, order) calibrated 1D spectra. Composes
        :mod:`decanter.image2d` → :mod:`decanter.rectify` →
        :mod:`decanter.extract` → :mod:`decanter.wavelength`. No
        cross-frame waveshift (waveshift is relative across frames
        and meaningless for a single frame).

    decanter.reduce_many(pairs, calib, *, mode="warp", ...)
        Reduce a list of frame pairs and align them with WARP's cross-frame
        wavelength shift.

    decanter.combine(...)
        SNR-weighted stack of aligned reductions (e.g. an out-of-transit
        template).

    decanter.reduce_to_warp_tree(listfile, raw_dir, calib, dest_dir, ...)
        Reduce a whole night/listfile WARP's way (frame numbering, WARP-exact
        cross-frame waveshift) and write WARP's on-disk layout, so every
        downstream consumer of ``reductions/<testname>/`` works unchanged.

    decanter.Calibration.from_dir(reduc_root)
        Auto-discover all calibration paths from a calibration set or a
        WARP reduction root (a root also loads frame NO1's aperture DBs,
        which lock extraction to NO1's trace; it warns when it does).

"""

__version__ = "1.0.1"

from decanter._reduction import Intermediates, OrderSpectrum, Reduction
from decanter.api import TransitSeries, combine, reduce, reduce_many
from decanter.calib import Calibration, CalibrationMismatch, InstrumentConfig
from decanter.config import Config
from decanter.warptree import WarpTree, reduce_to_warp_tree

__all__ = [
    "Calibration",
    "CalibrationMismatch",
    "Config",
    "InstrumentConfig",
    "Intermediates",
    "OrderSpectrum",
    "Reduction",
    "TransitSeries",
    "combine",
    "reduce",
    "reduce_many",
    "reduce_to_warp_tree",
    "WarpTree",
]
