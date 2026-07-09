"""Wavelength-axis finalization: dispersion solution + FSR truncation.

:mod:`decanter.wavelength.dispcor` applies the per-order id-file
dispersion solution to map pixel → wavelength via the IRAF POLY5
integral-average resample (``dispcor(flux=NO, linear=YES, dw=INDEF)``).

:mod:`decanter.wavelength.fsr` truncates the dispcor'd spectrum to the
free-spectral-range wavelength bounds (per the WARP FSR table) using
the same poly5 rebin machinery, and labels the output as VAC by
convention.

WARP equivalents: ``warp/Spec1Dtools.py:dispcor_single`` and
``warp/Spec1Dtools.py:cut_1dspec``.
"""

from decanter.wavelength.dispcor import dispcor_one_order
from decanter.wavelength.fsr import truncate_spectrum

__all__ = ["dispcor_one_order", "truncate_spectrum"]
