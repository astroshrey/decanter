"""Wavelength-bound spectrum truncate via IRAF ``onedspec.scopy``.

Reproduces ``iraf.scopy(rawspec, output, w1, w2)`` for the non-multispec
code path WARP exercises in ``warp/Spec1Dtools.py:truncate``:

    iraf.scopy(rawspec, outputfile, w1=p1, w2=p2)
    iraf.hedit(outputfile, "CRVAL1", "1.", verify="no")
    iraf.hedit(outputfile, "CRPIX1", "1.", verify="no")
    iraf.hedit(outputfile, "LTV1", "0.", verify="no")

IRAF ``scopy``'s default is ``rebin = yes`` (see
``noao/onedspec/scopy.cl:21``), which routes through
``sarith(op="copy", rebin=yes)`` →
``t_sarith.x:sa_sextract`` →
``smw/shdr.x:shdr_extract → shdr_linear``. That path resamples the
input onto a fresh linear grid aligned at the requested ``w1`` with
the input's original ``CDELT1`` step, taking the integral-average of
the POLY5 interpolant over each output pixel's wavelength half-edges
— exactly the same machinery the s13 FSR-truncate fix landed for the
``iraf.scopy(rebin=YES)`` path documented in HANDOFF gap #7.

Pywarp's pre-2026-05-13 implementation here was a pixel-aligned
integer slice — equivalent to ``scopy(rebin=no)``. That was a silent
algorithmic mismatch with WARP for the obj path (masked by the
``scombine`` identity short-circuit when ``shift=0``) and produced a
~5 ct/px residual on the sky path's ``_trans1dcut``. The current
implementation is the integer-aligned poly5 rebin that matches IRAF
bit-for-bit on linear-WCS input.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from astropy.io import fits as _astrofits
from numpy.typing import NDArray

from decanter.utils.iminterp import scombine_linear_poly5


class TruncResult(NamedTuple):
    """Output of :func:`scopy_wavelength_truncate`."""

    data: NDArray
    header: _astrofits.Header
    p_in_lo: int  # 1-indexed first input pixel landing in the output (approx)
    p_in_hi: int  # 1-indexed last input pixel landing in the output (approx)


def scopy_wavelength_truncate(
    data: NDArray,
    header: _astrofits.Header,
    *,
    w1: float,
    w2: float,
    reset_wcs: bool = True,
) -> TruncResult:
    """Truncate a 1-D spectrum to the wavelength range ``[w1, w2]`` via POLY5 rebin.

    Mirrors ``iraf.scopy(rawspec, output, w1=w1, w2=w2)`` with the
    default ``rebin=yes``, then WARP's post-scopy ``hedit`` that zeros
    the WCS so the output is indexed from pixel 1.

    The output grid starts at wavelength ``w1`` with step ``CDELT1_in``;
    the number of output pixels is ``nint((w2 - w1) / CDELT1_in) + 1``
    (per ``sa_sextract``). Each output pixel value is the integral-
    average of the input's POLY5 interpolant over the output pixel's
    wavelength half-edges, matching ``shdr_linear``.

    Args:
        data: 1-D input spectrum.
        header: input FITS header (must carry ``CRVAL1, CDELT1``).
        w1, w2: wavelength bounds in the input's wavelength units.
        reset_wcs: if True (default), zero ``CRVAL1`` and ``LTV1`` and
            set ``CRPIX1=1`` so the output pixel grid is 1-indexed from
            CRVAL1 = 1 (matches WARP's ``truncate`` helper). Set False
            to preserve the input's WCS offsets for diagnostic use.

    Returns:
        :class:`TruncResult` with the resampled data, the modified
        header, and the integer-rounded 1-indexed input-pixel range
        covered by the output (informational; kept for backwards
        compatibility with diagnostic callers).
    """
    if data.ndim != 1:
        raise ValueError(f"expected 1-D spectrum, got shape {data.shape}")
    n = data.size
    crval1 = float(header.get("CRVAL1", 1.0))
    cdelt1 = float(header.get("CDELT1", 1.0))
    if cdelt1 == 0.0:
        raise ValueError("CDELT1 is zero — cannot map pixels to wavelengths")

    # Output WCS per sa_sextract: output spans [w1, w2] at CDELT_in step.
    # nw = nint((w2 - w1) / CDELT) + 1
    dw_out = cdelt1
    nw_out = int(round((w2 - w1) / dw_out)) + 1
    if nw_out <= 0:
        raise ValueError(
            f"scopy bounds [{w1}, {w2}] with CDELT1={cdelt1} produced "
            f"non-positive nw_out={nw_out}"
        )

    out_data = scombine_linear_poly5(
        np.asarray(data, dtype=np.float32),
        crval1_in=crval1,
        cdelt1_in=cdelt1,
        w1_out=float(w1),
        dw_out=float(dw_out),
        nw_out=nw_out,
        flux=False,
    )

    # Approximate integer input-pixel coverage (informational).
    # 1-indexed pixel-at-wavelength: p(λ) = (λ - CRVAL1) / CDELT1 + CRPIX1.
    crpix1 = float(header.get("CRPIX1", 1.0))
    p_at_w1 = (w1 - crval1) / cdelt1 + crpix1
    p_at_w2 = (w2 - crval1) / cdelt1 + crpix1
    p_lo_f, p_hi_f = (p_at_w1, p_at_w2) if p_at_w2 >= p_at_w1 else (p_at_w2, p_at_w1)
    p_in_lo = max(1, min(n, int(round(p_lo_f))))
    p_in_hi = max(1, min(n, int(round(p_hi_f))))

    out_header = header.copy()
    out_header["NAXIS1"] = out_data.size
    if reset_wcs:
        # WARP's `truncate` zeroes the offset so the output's pixel 1
        # has wavelength = 1, CDELT1 preserved.
        out_header["CRVAL1"] = (1.0, "Wavelength at pixel 1 (post-truncate)")
        out_header["CRPIX1"] = (1.0, "Reference pixel along dispersion")
        out_header["LTV1"] = (0.0, "Origin (post-truncate)")
    else:
        # Preserve input offsets; only the data and NAXIS1 change.
        out_header["CRVAL1"] = (float(w1), "Wavelength at pixel 1 (post-rebin)")
        out_header["CRPIX1"] = (1.0, "Reference pixel along dispersion")
    return TruncResult(data=out_data, header=out_header, p_in_lo=p_in_lo, p_in_hi=p_in_hi)
