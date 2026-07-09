"""WINERED FITS header keyword extraction.

WARP equivalent: ``warp/Spec2Dtools.py:header_key_read`` (line 18) — a
trivial try/except wrapper around dict-style lookup that returns the
string ``"N/A"`` when the key is missing. decanter keeps the same default
so downstream string comparisons (``if value == "N/A": ...``) port
verbatim.

The decanter parity-diff layer compares only a curated allow-list — see
:data:`ALLOWED_HEADER_KEYS` — to avoid noise from astropy/IRAF
auto-added metadata.
"""

from __future__ import annotations

from typing import Any

from astropy.io import fits

# Header keys whose values must match exactly between WARP and decanter
# outputs. Anything outside this set is ignored by the regression
# tests. See PLAN_FULL.md §Validation "False positives to suppress."
ALLOWED_HEADER_KEYS: frozenset[str] = frozenset({
    # WCS (1D spectra)
    "CRVAL1", "CDELT1", "CRPIX1", "CTYPE1", "CUNIT1",
    # Data shape & scaling
    "BSCALE", "BZERO", "BUNIT", "EXTNAME",
    # Echelle aperture identity
    "APNUM1",
    # Observation metadata
    "OBJECT", "DATE-OBS", "INSTMODE", "SETTING", "PERIOD", "SLIT",
    "RA", "DEC", "EXPTIME", "AIRMASS", "NODPOS", "NODPAT",
})


def get(header: fits.Header | dict[str, Any], key: str, default: Any = "N/A") -> Any:
    """Read a header key, returning ``default`` if absent or unreadable.

    WARP equivalent: ``warp/Spec2Dtools.py:header_key_read``. Same
    "N/A" default so string-equality checks port unchanged.

    Args:
        header: an astropy ``Header`` or any dict-like with ``__getitem__``.
        key: keyword to look up (case-insensitive for astropy ``Header``).
        default: value returned when the key is missing.

    Returns:
        The header value, or ``default``.
    """
    try:
        return header[key]
    except (KeyError, IndexError):
        return default
