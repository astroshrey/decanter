"""Slit-trace PSF fitting and 1D box extraction from rectified strips.

:mod:`decanter.extract.psf_center` measures per-order ``(xshift, fwhm)``
on the rectified strips (used as the extraction window when no
trans-aperture DB is supplied). :mod:`decanter.extract.box_extract_1d`
performs the IRAF-faithful integral-fraction-edge-weight box sum
that collapses each rectified strip into a 1D spectrum.

:mod:`decanter.extract.strip_extract_2d` is the gated strip-format
variant; rarely used in production.

WARP equivalents: ``warp/centersearch_fortrans.py``,
``warp/Spec1Dtools.py:pyapall``.
"""

from decanter.extract.box_extract_1d import box_extract
from decanter.extract.psf_center import measure_one_strip

__all__ = ["box_extract", "measure_one_strip"]
