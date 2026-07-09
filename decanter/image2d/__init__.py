"""2D detector-frame cleaning.

Tasks operate on the full ~2048×2048 detector frame to produce the
``_sscfm`` clean science frame consumed by :mod:`decanter.rectify`.
Linear sequence per frame: ``sky_subtract`` → ``cosmic_ray.cr_mask``
→ ``apscatter.subtract_apscatter`` → ``flatfield.flatfield_divide``
→ ``fixpix.fix_bad_pixels``.

WARP equivalents: ``Warp_sci.py:336`` (s01), ``warp/badpixmask.py``
(s02 + s05), ``warp/apscatter.py`` (s03), ``warp/Spec2Dtools.py`` (s04).
"""

from decanter.image2d.apscatter import subtract_apscatter
from decanter.image2d.cosmic_ray import cr_mask
from decanter.image2d.fixpix import fix_bad_pixels
from decanter.image2d.flatfield import flatfield_divide
from decanter.image2d.sky_subtract import sky_subtract

__all__ = [
    "sky_subtract",
    "cr_mask",
    "subtract_apscatter",
    "flatfield_divide",
    "fix_bad_pixels",
]
