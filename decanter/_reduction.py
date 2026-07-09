"""Return types for :func:`decanter.reduce`.

A :class:`Reduction` is what one call to :func:`decanter.reduce` produces
for one (obj, sky) frame pair: per-order, per-FSR-cut, wavelength-
calibrated 1D spectra for the obj path; optionally the same for sky.
Plus optional 2D / per-order intermediates for debugging when
``save_intermediates=True``.

The dataclass has helpers for the two common access patterns:

    r = decanter.reduce(obj, sky, calib)
    spec = r.obj[(1.05, 163)]                   # OrderSpectrum
    r.orders                                    # (159, ..., 184)
    r.fsr_cuts                                  # (1.05, 1.30)
    r.write_to(Path("out"), save_intermediates=True)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from astropy.io import fits as _astrofits
from numpy.typing import NDArray

from decanter.io import fits as _fits


@dataclass(frozen=True, slots=True)
class OrderSpectrum:
    """One echelle order's 1D wavelength-calibrated spectrum."""

    order: int
    fsr_cut: float
    flux: NDArray
    crval1: float
    cdelt1: float
    crpix1: float

    @property
    def wavelength(self) -> NDArray:
        """Return the wavelength array for this spectrum's pixels (1-indexed)."""
        import numpy as np
        n = self.flux.size
        pixels = np.arange(1, n + 1, dtype=self.flux.dtype)
        return self.crval1 + (pixels - self.crpix1) * self.cdelt1


@dataclass
class Intermediates:
    """Per-stage 2D / per-order intermediates carried through a reduction.

    Populated only when ``save_intermediates=True`` was passed to
    :func:`decanter.reduce`. All fields are ``None`` otherwise.

    Naming mirrors WARP's FITS-suffix convention so the diff workflow
    against a WARP ``-s`` reduction stays trivial.
    """

    obj_raw: NDArray | None = None        # raw obj input (pre-s01)
    obj_s: NDArray | None = None          # sky-subtracted obj (s01 _s.fits)
    cr_mask: NDArray | None = None        # CR mask (s02 mask_*_s.fits)
    obj_ssc: NDArray | None = None        # apscatter-subtracted (s03 _ssc)
    scatter_model: NDArray | None = None  # the apscatter model itself
    obj_sscf: NDArray | None = None       # flat-divided (s04 _sscf)
    obj_sscfm: NDArray | None = None      # fixpix'd, final 2D (s05 _sscfm)
    strips_obj: dict[int, NDArray] = field(default_factory=dict)  # s06 per order
    spectra_1d: dict[int, NDArray] = field(default_factory=dict)  # s08 per order
    spectra_dispcor: dict[int, NDArray] = field(default_factory=dict)  # s12 per order

    sky_f: NDArray | None = None          # raw sky flat-divided (s04 sky)
    sky_fm: NDArray | None = None         # sky fixpix'd (s05 sky)
    strips_sky: dict[int, NDArray] = field(default_factory=dict)
    sky_1d: dict[int, NDArray] = field(default_factory=dict)
    sky_dispcor: dict[int, NDArray] = field(default_factory=dict)


@dataclass
class Reduction:
    """Output of :func:`decanter.reduce` for one (obj, sky) frame pair.

    Attributes:
        obj_name: sanitized OBJECT-header name (e.g. ``"TOI2109"``).
        obj_path / sky_path: the input file paths if reduce() was called
            with paths (None when called with in-memory arrays).
        obj: ``{(fsr_cut, order): OrderSpectrum}`` for the obj path.
        sky: same for the sky path; ``None`` when
            ``config.flag_skyemission`` is False.
        intermediates: per-stage diagnostic arrays; non-trivial only
            when ``save_intermediates=True`` was passed.
    """

    obj_name: str
    obj_path: Path | None
    sky_path: Path | None
    obj: Mapping[tuple[float, int], OrderSpectrum]
    sky: Mapping[tuple[float, int], OrderSpectrum] | None = None
    intermediates: Intermediates = field(default_factory=Intermediates)

    @property
    def orders(self) -> tuple[int, ...]:
        """Echelle orders present in the obj output (sorted)."""
        return tuple(sorted({m for _, m in self.obj}))

    @property
    def fsr_cuts(self) -> tuple[float, ...]:
        """FSR cuts present (sorted)."""
        return tuple(sorted({c for c, _ in self.obj}))

    def write_to(self, workdir: Path, *, save_intermediates: bool = False) -> None:
        """Write the reduction's outputs to ``workdir`` (WARP suffix layout).

        Always writes the final ``_fsr{cut}_VAC.fits`` files (one per
        order × fsr_cut for obj, and same for sky if present). If
        ``save_intermediates`` is True, also writes the per-stage
        intermediates that were captured in ``self.intermediates``.

        The on-disk layout mirrors WARP's:
            {obj}_NO1_sscfm_m{m}_fsr{cut}_VAC.fits        (obj final)
            {obj}_skyNO1_fm_m{m}trans1dcutw_fsr{cut}_VAC.fits  (sky final)
            {obj}_NO1_<suffix>.fits                       (intermediates)
        """
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        def _wcs_header(spec: OrderSpectrum) -> _astrofits.Header:
            h = _astrofits.Header()
            h["CRVAL1"] = (spec.crval1, "Wavelength at output pixel 1")
            h["CDELT1"] = (spec.cdelt1, "Wavelength step per pixel")
            h["CRPIX1"] = (spec.crpix1, "Reference pixel along dispersion")
            h["CTYPE1"] = ("LINEAR", "Wavelength axis")
            return h

        # Final obj spectra
        for (cut, m), spec in self.obj.items():
            cut_str = f"{cut:.2f}" if abs(cut - round(cut, 2)) < 1e-9 else f"{cut:g}"
            out = workdir / f"{self.obj_name}_NO1_sscfm_m{m}_fsr{cut_str}_VAC.fits"
            _fits.write_image(out, spec.flux, _wcs_header(spec), overwrite=True)

        if self.sky is not None:
            for (cut, m), spec in self.sky.items():
                cut_str = f"{cut:.2f}" if abs(cut - round(cut, 2)) < 1e-9 else f"{cut:g}"
                out = (workdir /
                       f"{self.obj_name}_skyNO1_fm_m{m}trans1dcutw_fsr{cut_str}_VAC.fits")
                _fits.write_image(out, spec.flux, _wcs_header(spec), overwrite=True)

        if not save_intermediates:
            return

        # Intermediate 2D / 1D arrays (best-effort — only what was captured).
        it = self.intermediates
        base_2d = {
            "_s.fits": it.obj_s,
            "_ssc.fits": it.obj_ssc,
            "_sscf.fits": it.obj_sscf,
            "_sscfm.fits": it.obj_sscfm,
        }
        for suffix, arr in base_2d.items():
            if arr is not None:
                _fits.write_image(workdir / f"{self.obj_name}_NO1{suffix}", arr,
                                  _astrofits.Header(), overwrite=True)
        if it.cr_mask is not None:
            _fits.write_image(workdir / f"mask_{self.obj_name}_NO1_s.fits",
                              it.cr_mask, _astrofits.Header(), overwrite=True)
        # Per-order strips
        for m, strip in it.strips_obj.items():
            _fits.write_image(workdir / f"{self.obj_name}_NO1_sscfm_m{m}trans.fits",
                              strip, _astrofits.Header(), overwrite=True)
        for m, spec in it.spectra_1d.items():
            _fits.write_image(workdir / f"{self.obj_name}_NO1_sscfm_m{m}_1d.fits",
                              spec, _astrofits.Header(), overwrite=True)
        for m, spec in it.spectra_dispcor.items():
            _fits.write_image(workdir / f"{self.obj_name}_NO1_sscfm_m{m}_1dcsw.fits",
                              spec, _astrofits.Header(), overwrite=True)
        # Sky intermediates
        if it.sky_fm is not None:
            _fits.write_image(workdir / f"{self.obj_name}_skyNO1_fm.fits",
                              it.sky_fm, _astrofits.Header(), overwrite=True)
        for m, strip in it.strips_sky.items():
            _fits.write_image(
                workdir / f"{self.obj_name}_skyNO1_fm_m{m}trans.fits",
                strip, _astrofits.Header(), overwrite=True,
            )
