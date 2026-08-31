"""Absolute wavelength calibration for a set of frames.

:mod:`decanter.waveshift` measures a *relative* frame-to-frame offset by
cross-correlating each frame against a reference frame, reproducing WARP's
``ccwaveshift``: enough to stack a series, but the stack keeps whatever
absolute error the comparison-lamp solution carried.

:mod:`decanter.wavecal` measures every frame against templates generated from
HITRAN line data — telluric absorption in the object spectrum, OH airglow
emission in the paired sky spectrum — so the correction is absolute, anchored
to the telluric rest frame.

Four modes, set by ``WavecalConfig.mode``:

============== ================================ =========================
mode           reference                        per exposure
============== ================================ =========================
oh_static      OH only                          shift only
oh_refit       OH only                          band amplitudes + shift
hybrid_static  telluric, OH where no telluric   shift only
hybrid_refit   same                             columns / bands + shift
============== ================================ =========================

The public default is ``mode="auto"``: HIRES-Y and HIRES-J resolve to
``hybrid_refit``, WIDE to ``hybrid_static``. The selector is not a fifth
method; the saved solution always records one of the four modes above.

Applying a solution rescales the linear WCS rather than resampling the flux:
on a grid linear in wavelength, dividing ``CRVAL1`` and ``CDELT1`` by
``1 + v/c`` is exact.
"""

from decanter.wavecal.config import WavecalConfig
from decanter.wavecal.directory import apply_solution_to_directory
from decanter.wavecal.products import airglow_product, telluric_product
from decanter.wavecal.report import wavecal_report_pdf
from decanter.wavecal.series import Series, from_reductions, load_series
from decanter.wavecal.solution import C_KMS, WavecalSolution
from decanter.wavecal.solve import WavecalRun, solve

__all__ = [
    "C_KMS",
    "Series",
    "WavecalConfig",
    "WavecalRun",
    "WavecalSolution",
    "apply_solution_to_directory",
    "from_reductions",
    "load_series",
    "solve",
    "wavecal_report_pdf",
    "airglow_product",
    "telluric_product",
]
