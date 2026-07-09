"""Stage 12 — dispersion correction (pixel → vacuum wavelength).

WARP equivalent: ``warp/Spec1Dtools.py:dispcor_single`` (line 260)
wrapping ``iraf.onedspec.dispcor`` with ``REFSPEC1`` pointing to the
per-order ``id*`` file.

For each order, replace the existing approximate wavelength axis
(set by s06 cutransform's fc surface; in nm) with the precise
Chebyshev-fitted axis from ``ecidentify`` (in Å) and resample the
flux onto a linear-Å grid.

WARP uses ``iraf.onedspec.dispcor(..., dw=INDEF, flux=NO, linear=YES)``.
With ``dw=INDEF`` IRAF computes the output step from the input length:
``nw = N_in``, ``dw = (λ_max - λ_min) / (nw - 1)``. Because WARP first
truncates the post-extract spectrum to the comp file's NAXIS1 (4095 for
HIRES-Y100) in ``Spec1Dtools.py:152:truncate`` (see s11's truncate
step), the dispcor output for HIRES-Y100 m=163 is 4095 pixels with
CRVAL1 = 10754.81 Å, CDELT1 = 0.0344612 Å.

Output suffix: ``_1dcsw``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits as _astrofits

from decanter.calib.aperture import ApertureSet
from decanter.config import Config
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.iraf_id import parse as parse_id
from decanter.io.iraf_id import pixels_at_wavelength, wavelength_at_pixels
from decanter.io.listfile import parse as parse_listfile
from decanter.utils.iminterp import dispcor_linear_poly5


def _sanitize_objname(raw: str) -> str:
    name = raw
    for ch in (" ", "'", "\"", "#", "/"):
        name = name.replace(ch, "_")
    return name


def run(
    config: Config,
    workdir: Path,
    listfile: Path,
    *,
    apdb_path: Path | str | None = None,
    id_dir: Path | str | None = None,
    id_refname: str | None = None,
    **_unused: Any,
) -> None:
    return _run_impl(config, workdir, listfile, apdb_path=apdb_path,
                     id_dir=id_dir, id_refname=id_refname)


def dispcor_one_order(
    data: np.ndarray,
    header: "_astrofits.Header",
    id_path: Path,
) -> tuple[np.ndarray, "_astrofits.Header"]:
    """Pure task function: apply the per-order dispersion solution to one 1D spectrum.

    Mirrors IRAF ``dispcor(flux=NO, linear=YES, dw=INDEF)`` with the
    POLY5 image interpolator (``onedinterp.par`` default). The output
    wavelength grid is uniform in λ from ``w(pixel 1)`` to ``w(pixel N)``
    with N output pixels (matching the input length).

    Returns ``(flux_out, header)`` where ``header`` carries CRVAL1 / CDELT1 /
    CRPIX1 / CTYPE1 / WAT1_001 / REFSPEC1 for the new linear grid.
    """
    sol = parse_id(id_path)
    n = data.shape[0]
    wave_endpoints = wavelength_at_pixels(sol, np.array([1.0, float(n)]))
    lam_min = float(min(wave_endpoints))
    lam_max = float(max(wave_endpoints))

    def _invert(w: np.ndarray) -> np.ndarray:
        return pixels_at_wavelength(sol, w, xrange=(0.5, n + 0.5))

    flux_out = dispcor_linear_poly5(
        data, _invert, lam_min, lam_max, n, flux=False,
    )
    out_header = header.copy()
    cdelt = (lam_max - lam_min) / (n - 1) if n > 1 else 1.0
    out_header["CRVAL1"] = (lam_min, "Wavelength at output pixel 1")
    out_header["CRPIX1"] = (1.0, "Reference pixel along dispersion")
    out_header["CDELT1"] = (cdelt, "Wavelength step per pixel")
    out_header["CTYPE1"] = ("LINEAR", "Wavelength axis")
    out_header["WAT1_001"] = (
        f"wtype=linear label=Wavelength units={sol.units}",
        "WCS attribute (WAT) for axis 1",
    )
    out_header["REFSPEC1"] = (id_path.name, "Wavelength solution source (s12)")
    return flux_out.astype(np.float32), out_header


def _run_impl(
    config: Config,
    workdir: Path,
    listfile: Path,
    *,
    apdb_path: Path | str | None = None,
    id_dir: Path | str | None = None,
    id_refname: str | None = None,
) -> None:
    """Apply per-order ``id*`` wavelength solution and resample to linear λ.

    Args:
        config: pipeline configuration.
        workdir: reads ``{obj}_NO{i}_sscfm_m{m}_1dcs.fits`` (s11 output);
            writes ``{obj}_NO{i}_sscfm_m{m}_1dcsw.fits``.
        listfile: WARP-style input list.
        apdb_path: aperture database for order enumeration.
        id_dir: directory containing ``id*`` files (often the same as
            the database directory).
        id_refname: base name; per-order file is
            ``id<refname>.{m:04d}`` (e.g. ``"comp_HIRES-Y100_20250806_fm_ecall"``).

    Notes:
        Behavior mirrors IRAF ``dispcor(dw=INDEF, flux=NO, linear=YES)``:
        the output wavelength grid is uniform in λ with the same number
        of pixels as the input; the input's pixel-to-wavelength map
        comes from the id file. WARP's ``Spec1Dtools.py:152:truncate``
        is now applied in s11 so the input here is already comp-file
        length (4095 px on HIRES-Y100), producing a 4095-pixel
        dispcor output that matches WARP's ``_csw.fits`` byte-for-byte
        in shape.
    """
    missing = [
        n for n, v in [
            ("apdb_path", apdb_path),
            ("id_dir", id_dir),
            ("id_refname", id_refname),
        ] if v is None
    ]
    if missing:
        raise ValueError(
            f"s12_dispcor requires {missing}; the calibration loader "
            "(s00) will provide these automatically once wired in."
        )

    cal_apset = ApertureSet.load(apdb_path)
    orders = (
        cal_apset.echelle_orders
        if (config.reduce_full_data or not config.selected_orders)
        else tuple(m for m in cal_apset.echelle_orders if m in config.selected_orders)
    )
    id_dir_p = Path(id_dir)

    def _dispcor_one(in_path: Path, out_path: Path, m: int) -> bool:
        """Apply the per-order id solution to ``in_path``; write ``out_path``."""
        id_path = id_dir_p / f"id{id_refname}.{m:04d}"
        if not id_path.exists() or not in_path.exists():
            return False
        data, header = _fits.read_image(in_path)
        flux_out, out_header = dispcor_one_order(data, header, id_path)
        _fits.write_image(out_path, flux_out, out_header, overwrite=True)
        return True

    pairs = parse_listfile(listfile)
    for i, pair in enumerate(pairs, start=1):
        _obj_raw, obj_header = _fits.read_image(workdir / pair.object_name)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objname = _sanitize_objname(str(raw_objname))

        for m in orders:
            # --- OBJ path: _1dcs (or _1d fallback) → _1dcsw -------------------
            in_path = workdir / f"{objname}_NO{i}_sscfm_m{m}_1dcs.fits"
            if not in_path.exists():
                in_path = workdir / f"{objname}_NO{i}_sscfm_m{m}_1d.fits"
            _dispcor_one(
                in_path,
                workdir / f"{objname}_NO{i}_sscfm_m{m}_1dcsw.fits",
                m,
            )

            # --- SKY path: _trans1dcut → _trans1dcutw (no waveshift) ----------
            if config.flag_skyemission:
                _dispcor_one(
                    workdir / f"{objname}_skyNO{i}_fm_m{m}trans1dcut.fits",
                    workdir / f"{objname}_skyNO{i}_fm_m{m}trans1dcutw.fits",
                    m,
                )
