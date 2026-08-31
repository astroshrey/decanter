"""Loading a reduction directory onto a common per-order reference grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from decanter._reduction import OrderSpectrum, Reduction
from decanter.wavecal.series import from_reductions, load_series


def _write_frame(root: Path, frame: str, *, orders=(159, 160), n=800,
                 crval1=10000.0, cdelt1=0.03, ut="01:00:00.0",
                 sky_frame="WINA00000002") -> None:
    d = root / frame
    d.mkdir(parents=True, exist_ok=True)
    for order in orders:
        header = fits.Header()
        header["CRVAL1"] = crval1 + 50.0 * (order - orders[0])
        header["CDELT1"] = cdelt1
        header["CRPIX1"] = 1.0
        header["CTYPE1"] = "LINEAR"
        header["ECHORDER"] = order
        header["OBJECT"] = "TARGET"
        header["INSTMODE"] = "HIRES-Y"
        header["DATE-OBS"] = "2026-01-01"
        header["UT-STR"] = ut
        header["UT-END"] = ut
        header["AIRMASS"] = 1.5
        header["OBJFRAME"] = frame
        header["SKYFRAME"] = sky_frame
        flux = np.linspace(100.0, 200.0, n).astype(np.float32)
        fits.PrimaryHDU(flux, header).writeto(
            d / f"TARGET_NO1_sscfm_m{order}_fsr1.30_VAC.fits", overwrite=True)
        fits.PrimaryHDU(flux * 0.1, header).writeto(
            d / f"TARGET_skyNO1_fm_m{order}trans1dcutw_fsr1.30_VAC.fits", overwrite=True)


def test_loads_and_orders_frames_by_time(tmp_path: Path) -> None:
    _write_frame(tmp_path, "WINA00000002", ut="02:00:00.0", sky_frame="WINA00000001")
    _write_frame(tmp_path, "WINA00000001", ut="01:00:00.0", sky_frame="WINA00000002")
    series = load_series(tmp_path)

    assert series.frame_ids == ("WINA00000001", "WINA00000002")
    assert series.orders == (159, 160)
    assert series.obj.shape == (2, series.n_pixels, 2)
    assert series.sky is not None
    assert series.instmode == "HIRES-Y"
    assert series.elapsed_hours[-1] == pytest.approx(1.0, abs=1e-3)


def test_reference_grid_is_uniform_in_log_wavelength(tmp_path: Path) -> None:
    """A Doppler shift must be a constant pixel shift, which needs a log grid."""
    _write_frame(tmp_path, "WINA00000001")
    series = load_series(tmp_path)
    for j in range(series.n_orders):
        step = np.diff(np.log(series.wave[:, j]))
        assert np.allclose(step, step[0], rtol=1e-9)


def test_dv_per_pixel_matches_the_grid(tmp_path: Path) -> None:
    from decanter.wavecal.solution import C_KMS

    _write_frame(tmp_path, "WINA00000001")
    series = load_series(tmp_path)
    for j in range(series.n_orders):
        expected = C_KMS * np.diff(np.log(series.wave[:, j]))[0]
        assert series.dv_pix_kms[j] == pytest.approx(expected, rel=1e-9)


def test_grid_never_extrapolates_beyond_a_frame(tmp_path: Path) -> None:
    """Frames with different coverage must yield the intersection, not the union."""
    _write_frame(tmp_path, "WINA00000001", crval1=10000.0)
    _write_frame(tmp_path, "WINA00000002", crval1=10002.0, ut="02:00:00.0")
    series = load_series(tmp_path)
    assert series.wave[0, 0] >= 10002.0
    assert np.all(np.isfinite(series.obj))


def test_sky_frame_times_are_resolved(tmp_path: Path) -> None:
    _write_frame(tmp_path, "WINA00000001", ut="01:00:00.0", sky_frame="WINA00000002")
    _write_frame(tmp_path, "WINA00000002", ut="02:00:00.0", sky_frame="WINA00000001")
    series = load_series(tmp_path)
    offset = (series.sky_time_jd - series.time_jd) * 24.0
    assert offset[0] == pytest.approx(1.0, abs=1e-3)
    assert offset[1] == pytest.approx(-1.0, abs=1e-3)


def test_noise_is_measured_before_resampling(tmp_path: Path) -> None:
    """Resampling correlates neighbours; the stored noise must predate it.

    A spectrum is written with known 2% pixel noise and then loaded onto a
    5x oversampled grid. Measuring the noise after that interpolation
    underestimates it badly, so the recorded value must match the native one.
    """
    rng = np.random.default_rng(7)
    n = 800
    flux = 1000.0 * (1.0 + rng.normal(0.0, 0.02, n))
    d = tmp_path / "WINA00000001"
    d.mkdir()
    header = fits.Header()
    header["CRVAL1"] = 10000.0
    header["CDELT1"] = 0.03
    header["CRPIX1"] = 1.0
    header["ECHORDER"] = 159
    header["INSTMODE"] = "HIRES-Y"
    header["DATE-OBS"] = "2026-01-01"
    header["UT-STR"] = header["UT-END"] = "01:00:00.0"
    header["OBJFRAME"] = "WINA00000001"
    fits.PrimaryHDU(flux.astype(np.float32), header).writeto(
        d / "TARGET_NO1_sscfm_m159_fsr1.30_VAC.fits")

    series = load_series(tmp_path, n_pixels=4000)

    resampled = np.diff(series.obj[0, :, 0]) / np.median(series.obj[0, :, 0])
    resampled_noise = 1.4826 * np.median(np.abs(resampled - np.median(resampled))) / np.sqrt(2)

    assert series.noise_fraction[0, 0] == pytest.approx(0.02, rel=0.15)
    assert resampled_noise < 0.5 * series.noise_fraction[0, 0]


def test_missing_directory_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no reduced spectra"):
        load_series(tmp_path)


def test_unknown_fsr_cut_raises(tmp_path: Path) -> None:
    _write_frame(tmp_path, "WINA00000001")
    with pytest.raises(ValueError, match="not present"):
        load_series(tmp_path, fsr_cut=9.99)


def _memory_reduction(frame_id: str, ut: str, offset: float) -> Reduction:
    orders = (159, 160)
    obj = {}
    sky = {}
    for order in orders:
        spec = OrderSpectrum(
            order=order,
            fsr_cut=1.30,
            flux=np.linspace(100.0, 200.0, 800).astype(np.float32),
            crval1=10000.0 + 50.0 * (order - orders[0]) + offset,
            cdelt1=0.03,
            crpix1=1.0,
        )
        obj[(1.30, order)] = spec
        sky[(1.30, order)] = OrderSpectrum(
            order=order,
            fsr_cut=1.30,
            flux=spec.flux * 0.1,
            crval1=spec.crval1,
            cdelt1=spec.cdelt1,
            crpix1=spec.crpix1,
        )
    return Reduction(
        obj_name="TARGET",
        obj_path=None,
        sky_path=None,
        obj=obj,
        sky=sky,
        meta={
            "OBJFRAME": frame_id,
            "DATE-OBS": "2026-01-01",
            "UT-STR": ut,
            "UT-END": ut,
            "AIRMASS": 1.5,
            "INSTMODE": "HIRES-Y",
        },
    )


def test_builds_series_directly_from_warp_aligned_reductions() -> None:
    late = _memory_reduction("WINA00000002", "02:00:00", 0.0)
    early = _memory_reduction("WINA00000001", "01:00:00", 0.3)

    series = from_reductions([late, early])

    assert series.frame_ids == ("WINA00000001", "WINA00000002")
    assert series.orders == (159, 160)
    assert series.obj.shape == (2, series.n_pixels, 2)
    assert series.sky is not None
    assert series.wave[0, 0] >= early.obj[(1.30, 159)].wavelength[0]
    assert series.wave[-1, 0] <= late.obj[(1.30, 159)].wavelength[-1]


def test_in_memory_series_disambiguates_repeated_object_with_sky_frame() -> None:
    first = _memory_reduction("WINA00000001", "01:00:00", 0.0)
    second = _memory_reduction("WINA00000001", "01:00:00", 0.0)
    first.meta["SKYFRAME"] = "WINA00000002"
    second.meta["SKYFRAME"] = "WINA00000003"

    series = from_reductions([first, second])

    assert series.frame_ids == (
        "WINA00000001__WINA00000002",
        "WINA00000001__WINA00000003",
    )
    assert series.n_frames == 2
