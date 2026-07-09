"""FITS read/write helpers preserving headers and WCS.

Centralizing FITS I/O here lets decanter control:

* which header keys are propagated (avoid astropy's auto-additions like
  ``CHECKSUM``/``DATASUM`` that cause false-positive diffs against WARP),
* NaN policy (preserve ``np.nan`` rather than IRAF's ``BLANK``),
* byte order on write (astropy defaults to native; WARP/IRAF write
  big-endian, but byte order doesn't affect ``np.allclose`` reads).

See PLAN_FULL.md §Validation "False positives to suppress."
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits


def read_image(path: Path | str) -> tuple[np.ndarray, fits.Header]:
    """Read a 2D image FITS file.

    Args:
        path: path to the FITS file (with or without ``.fits`` suffix).

    Returns:
        ``(data, header)`` where ``data`` is the primary HDU array (copied
        so closing the file doesn't invalidate it) and ``header`` is a
        copy of the primary HDU header.

    Raises:
        FileNotFoundError: if the file (with or without ``.fits``) is missing.
    """
    path = _ensure_fits_path(path)
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data).copy()
        header = hdul[0].header.copy()
    return data, header


def write_image(
    path: Path | str,
    data: np.ndarray,
    header: fits.Header | None = None,
    *,
    overwrite: bool = False,
) -> None:
    """Write a 2D image FITS file with controlled header propagation.

    Args:
        path: destination path (``.fits`` appended if missing).
        data: pixel array.
        header: optional header to attach; if ``None`` astropy generates
            a minimal one. Pass the source header from :func:`read_image`
            to preserve metadata.
        overwrite: if ``False`` (default) and the file exists, astropy raises.

    Notes:
        Does **not** call ``add_checksum`` — that would add ``CHECKSUM``/
        ``DATASUM`` keys that bust parity diffs against WARP's outputs.
    """
    path = _ensure_fits_path(path)
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(path, overwrite=overwrite)


def _ensure_fits_path(path: Path | str) -> Path:
    """Accept WARP-style bare names (``"frame"``) or full paths (``"frame.fits"``).

    WARP's IRAF wrappers pass filenames without the ``.fits`` extension;
    decanter accepts both forms for compatibility.
    """
    p = Path(path)
    if p.suffix.lower() not in (".fits", ".fit"):
        p = p.with_name(p.name + ".fits")
    return p
