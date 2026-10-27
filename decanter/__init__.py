"""decanter — a pure-Python port of WARP for WINERED echelle reductions.

User-facing API:

    decanter.reduce(obj, sky, calib, *, workdir=None, save_intermediates=False)
        Single-frame reduction. Returns a :class:`Reduction` with
        per-(fsr_cut, order) calibrated 1D spectra. Composes
        :mod:`decanter.image2d` → :mod:`decanter.rectify` →
        :mod:`decanter.extract` → :mod:`decanter.wavelength`. No
        cross-frame waveshift (waveshift is relative across frames
        and meaningless for a single frame).

    decanter.reduce_many(..., wavecal_config=None, workdir=None)
        Multi-frame reduction with the original WARP-compatible relative
        wavelength alignment. Passing a WavecalConfig optionally layers the
        physical telluric/OH calibration on top.
        Passing workdir writes only the completed, fully calibrated series.

    decanter.combine(...)
        Multi-frame SNR-weighted stack of aligned reductions.

    decanter.reduce_to_warp_tree(listfile, raw_dir, calib, dest_dir, ...)
        Reduce a whole night/listfile WARP's way (frame numbering, WARP-exact
        cross-frame waveshift) and write WARP's on-disk layout, so every
        downstream consumer of ``reductions/<testname>/`` works unchanged.

    decanter.Calibration.from_dir(reduc_root)
        Auto-discover all calibration paths from a calibration set or a
        WARP reduction root (a root also loads frame NO1's aperture DBs,
        which lock extraction to NO1's trace; it warns when it does).

See ``CLAUDE.md`` / ``HANDOFF.md`` for architecture notes and
``PLAN.md`` for the Phase-1 design.
"""

__version__ = "0.0.1"

from decanter._reduction import Intermediates, OrderSpectrum, Reduction
from decanter.api import TransitSeries, calibrate_wavelengths, combine, reduce, reduce_many
from decanter.calib import Calibration, CalibrationMismatch, InstrumentConfig
from decanter.config import Config
from decanter.serval import (
    RVStabilityResult,
    TransitEphemeris,
    run_serval_rv_stability,
    run_serval_rv_stability_directory,
)
from decanter.warptree import WarpTree, reduce_to_warp_tree
from decanter.wavecal import WavecalConfig, WavecalSolution

__all__ = [
    "Calibration",
    "CalibrationMismatch",
    "Config",
    "InstrumentConfig",
    "Intermediates",
    "OrderSpectrum",
    "Reduction",
    "RVStabilityResult",
    "TransitEphemeris",
    "TransitSeries",
    "WarpTree",
    "WavecalConfig",
    "WavecalSolution",
    "calibrate_wavelengths",
    "combine",
    "reduce",
    "reduce_many",
    "reduce_to_warp_tree",
    "run_serval_rv_stability",
    "run_serval_rv_stability_directory",
]
