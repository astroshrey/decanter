"""Integration of physical wavecal as a post-WARP pipeline layer."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

import decanter
import decanter.api as api
from decanter._reduction import OrderSpectrum, Reduction
from decanter.api import TransitSeries


def _reduction(frame_id: str, hour: int) -> Reduction:
    orders = (159, 160, 161)
    flux = np.linspace(100.0, 200.0, 500).astype(np.float32)
    obj = {
        (1.30, order): OrderSpectrum(
            order=order,
            fsr_cut=1.30,
            flux=flux,
            crval1=10000.0 + 50.0 * (order - orders[0]),
            cdelt1=0.03,
            crpix1=1.0,
        )
        for order in orders
    }
    return Reduction(
        obj_name="TARGET",
        obj_path=None,
        sky_path=None,
        obj=obj,
        sky=None,
        meta={
            "OBJFRAME": frame_id,
            "DATE-OBS": "2026-01-01",
            "UT-STR": f"{hour:02d}:00:00",
            "UT-END": f"{hour:02d}:00:00",
            "INSTMODE": "HIRES-Y",
            "WAVSHIFT": 0.125,
        },
    )


def test_physical_wavecal_is_applied_after_and_preserves_warp_shifts(monkeypatch) -> None:
    reductions = [_reduction("WINA00000001", 1), _reduction("WINA00000002", 2)]
    warp_shifts = np.array([0.0, 0.125])
    warp_series = TransitSeries(reductions=reductions, shifts=warp_shifts, refid=0)

    solve_module = importlib.import_module("decanter.wavecal.solve")

    def fake_solve(reference, config, *, verbose, return_diagnostics):
        shape = (reference.n_frames, reference.n_orders)
        solution = decanter.WavecalSolution(
            frame_ids=reference.frame_ids,
            orders=reference.orders,
            velocity=np.full(shape, 1.5),
            source=np.full(shape, "telluric", dtype="U16"),
            bracketed=np.ones(shape, dtype=bool),
            mode=config.mode,
        )
        return SimpleNamespace(solution=solution, telluric_model=None, oh_model=None)

    monkeypatch.setattr(solve_module, "solve", fake_solve)

    corrected = decanter.calibrate_wavelengths(
        warp_series, decanter.WavecalConfig(), verbose=False
    )

    np.testing.assert_array_equal(corrected.shifts, warp_shifts)
    assert corrected.wavecal_solution is not None
    original = reductions[0].obj[(1.30, 159)]
    calibrated = corrected.reductions[0].obj[(1.30, 159)]
    assert calibrated.crval1 < original.crval1
    assert calibrated.flux is original.flux
    assert corrected.reductions[0].meta["WAVSHIFT"] == 0.125
    assert corrected.reductions[0].meta["WAVECAL"] == "hybrid_refit"


def test_physical_wavecal_preserves_repeated_object_sky_pairs(monkeypatch, tmp_path) -> None:
    first = _reduction("WINA00000001", 1)
    second = _reduction("WINA00000001", 1)
    first.meta["SKYFRAME"] = "WINA00000002"
    second.meta["SKYFRAME"] = "WINA00000003"
    warp_series = TransitSeries(
        reductions=[first, second], shifts=np.zeros(2), refid=0,
    )
    solve_module = importlib.import_module("decanter.wavecal.solve")

    def fake_solve(reference, config, *, verbose, return_diagnostics):
        velocity = np.repeat(np.array([[1.0], [2.0]]), reference.n_orders, axis=1)
        solution = decanter.WavecalSolution(
            frame_ids=reference.frame_ids,
            orders=reference.orders,
            velocity=velocity,
            source=np.full(velocity.shape, "telluric", dtype="U16"),
            bracketed=np.ones(velocity.shape, dtype=bool),
            mode=config.mode,
        )
        return SimpleNamespace(solution=solution, telluric_model=None, oh_model=None)

    monkeypatch.setattr(solve_module, "solve", fake_solve)

    corrected = decanter.calibrate_wavelengths(
        warp_series, decanter.WavecalConfig(), verbose=False,
    )

    expected_ids = (
        "WINA00000001__WINA00000002",
        "WINA00000001__WINA00000003",
    )
    assert tuple(r.meta["SERIESID"] for r in corrected.reductions) == expected_ids
    assert corrected.reductions[0].meta["OBJFRAME"] == "WINA00000001"
    assert corrected.reductions[1].meta["OBJFRAME"] == "WINA00000001"
    corrected.write_to(tmp_path)
    assert all((tmp_path / frame_id).is_dir() for frame_id in expected_ids)


def test_atmospheric_prealign_runs_coarse_then_fine_physical_solve(monkeypatch) -> None:
    reductions = [_reduction("WINA00000001", 1), _reduction("WINA00000002", 2)]
    series = TransitSeries(reductions=reductions, shifts=np.zeros(2), refid=0)
    solve_module = importlib.import_module("decanter.wavecal.solve")
    calls = []
    solve_configs = []

    def fake_solve(reference, config, *, verbose, return_diagnostics):
        calls.append(reference)
        solve_configs.append(config)
        shape = (reference.n_frames, reference.n_orders)
        if len(calls) == 1:
            velocity = np.array([
                [4.0, 5.0, 6.0],
                [6.0, 7.0, 8.0],
            ])
        else:
            velocity = np.full(shape, 0.2)
        solution = decanter.WavecalSolution(
            frame_ids=reference.frame_ids,
            orders=reference.orders,
            velocity=velocity,
            source=np.full(shape, "telluric", dtype="U16"),
            bracketed=np.ones(shape, dtype=bool),
            mode=config.mode,
        )
        return SimpleNamespace(solution=solution, telluric_model=None)

    monkeypatch.setattr(solve_module, "solve", fake_solve)

    def fake_common(run, config):
        first = run.solution
        common = np.array([-1.0, 1.0])
        velocity = np.repeat(common[:, None], first.n_orders, axis=1)
        return decanter.WavecalSolution(
            frame_ids=first.frame_ids,
            orders=first.orders,
            velocity=velocity,
            source=np.full(velocity.shape, "interpolated", dtype="U16"),
            bracketed=np.ones(velocity.shape, dtype=bool),
            mode=config.mode,
            zero_point="relative",
            assembly="decomposition",
            meta={"stage": "broad_pooled_telluric_oh_ccf"},
        )

    monkeypatch.setattr(api, "_atmospheric_common_solution", fake_common)
    corrected = decanter.calibrate_wavelengths(
        series,
        decanter.WavecalConfig(atmospheric_prealign=True),
        verbose=False,
    )

    assert len(calls) == 2
    assert all(config.shift_search_kms == 25.0 for config in solve_configs)
    assert all(config.atmospheric_prealign is True for config in solve_configs)
    solution = corrected.wavecal_solution
    assert solution is not None
    assert solution.meta["atmospheric_prealign"] is True
    np.testing.assert_allclose(
        solution.meta["coarse_common_velocity_kms"], [-1.0, 1.0]
    )
    # The exact relativistic composition is very close to coarse + 0.2 km/s.
    np.testing.assert_allclose(solution.velocity[:, 0], [-0.8, 1.2], atol=1e-5)
    np.testing.assert_array_equal(corrected.shifts, np.zeros(2))


def test_default_wavecal_is_one_wide_pass(monkeypatch) -> None:
    reductions = [_reduction("WINA00000001", 1), _reduction("WINA00000002", 2)]
    series = TransitSeries(reductions=reductions, shifts=np.zeros(2), refid=0)
    solve_module = importlib.import_module("decanter.wavecal.solve")
    calls = []

    def fake_solve(reference, config, *, verbose, return_diagnostics):
        calls.append(config)
        shape = (reference.n_frames, reference.n_orders)
        solution = decanter.WavecalSolution(
            frame_ids=reference.frame_ids,
            orders=reference.orders,
            velocity=np.zeros(shape),
            source=np.full(shape, "telluric", dtype="U16"),
            bracketed=np.ones(shape, dtype=bool),
            mode=config.mode,
        )
        return SimpleNamespace(solution=solution, telluric_model=None)

    monkeypatch.setattr(solve_module, "solve", fake_solve)
    corrected = decanter.calibrate_wavelengths(
        series, decanter.WavecalConfig(), verbose=False,
    )

    assert corrected.wavecal_solution is not None
    assert len(calls) == 1
    assert calls[0].shift_search_kms == 25.0
    assert calls[0].atmospheric_prealign is False


def test_atmospheric_common_solution_measures_broad_pooled_ccf(monkeypatch) -> None:
    n_frames, n_pixels, n_orders = 4, 201, 2
    pixel = np.arange(n_pixels, dtype=float)
    feature = np.exp(-0.5 * ((pixel - 100.0) / 3.0) ** 2)
    shifts = np.array([-1.0, 0.0, 1.0, 2.0])
    signal = np.asarray([
        np.interp(pixel - shift, pixel, feature, left=0.0, right=0.0)
        for shift in shifts
    ])
    normalized = np.repeat((1.0 - signal)[:, :, None], n_orders, axis=2)
    solve_module = importlib.import_module("decanter.wavecal.solve")
    monkeypatch.setattr(solve_module, "_normalized", lambda series: normalized)

    series = SimpleNamespace(
        n_frames=n_frames,
        n_orders=n_orders,
        orders=(159, 160),
        frame_ids=tuple(f"frame-{i}" for i in range(n_frames)),
        dv_pix_kms=np.ones(n_orders),
        sky=None,
    )
    velocity = np.repeat(shifts[:, None], n_orders, axis=1)
    first_solution = decanter.WavecalSolution(
        frame_ids=series.frame_ids,
        orders=series.orders,
        velocity=velocity,
        source=np.full(velocity.shape, "telluric", dtype="U16"),
        bracketed=np.ones(velocity.shape, dtype=bool),
        mode="hybrid_refit",
    )
    model = SimpleNamespace(
        native_template=np.repeat((1.0 - feature)[:, None], n_orders, axis=1),
    )
    run = SimpleNamespace(
        series=series,
        solution=first_solution,
        telluric_model=model,
        telluric_support=np.repeat((feature > 1e-4)[:, None], n_orders, axis=1),
        telluric_velocity=velocity,
        telluric_accepted=np.ones(velocity.shape, dtype=bool),
        oh_model=None,
    )
    config = decanter.WavecalConfig(
        atmospheric_search_kms=5.0, atmospheric_step_kms=0.25,
    )

    coarse = api._atmospheric_common_solution(run, config)

    diagnostic = coarse.meta["diagnostic"]
    assert diagnostic.joint_score.shape == (
        n_frames, diagnostic.velocity_grid_kms.size
    )
    # The grid is searched around each order's own static offset, so the peak
    # is the residual relative to that offset rather than the absolute shift.
    np.testing.assert_allclose(
        diagnostic.peak_velocity_kms, shifts - np.median(shifts), atol=0.1
    )
    assert np.all(np.isfinite(diagnostic.peak_snr))
    assert diagnostic.telluric_orders == (159, 160)
    assert diagnostic.oh_orders == ()
    np.testing.assert_allclose(coarse.velocity[:, 0], shifts - np.median(shifts), atol=0.1)
    np.testing.assert_allclose(
        coarse.velocity, np.repeat(coarse.velocity[:, :1], n_orders, axis=1)
    )
    assert coarse.meta["telluric_orders"] == [159, 160]
    assert coarse.meta["oh_orders"] == []


def test_warp_only_series_remains_the_default() -> None:
    series = TransitSeries(
        reductions=[_reduction("WINA00000001", 1)],
        shifts=np.zeros(1),
        refid=0,
    )
    assert series.wavecal_solution is None


def test_reduce_many_invokes_physical_layer_only_when_requested(monkeypatch) -> None:
    reduction = _reduction("WINA00000001", 1)
    calls = []

    monkeypatch.setattr(api, "reduce", lambda *args, **kwargs: reduction)

    def fake_calibrate(series, config, *, verbose, diagnostic_pdf):
        calls.append((config, verbose, diagnostic_pdf))
        return series

    monkeypatch.setattr(api, "calibrate_wavelengths", fake_calibrate)
    cfg = decanter.WavecalConfig()

    api.reduce_many([(object(), None)], object(), align=False)
    assert calls == []

    api.reduce_many(
        [(object(), None)],
        object(),
        align=False,
        wavecal_config=cfg,
        wavecal_verbose=False,
        wavecal_diagnostic_pdf="diagnostics.pdf",
    )
    assert calls == [(cfg, False, "diagnostics.pdf")]


def test_reduce_many_parallelizes_only_the_independent_extraction(monkeypatch) -> None:
    reduction = _reduction("WINA00000001", 1)
    submitted = []

    class ImmediateFuture:
        def result(self):
            return reduction

    class ImmediatePool:
        def __init__(self, max_workers):
            assert max_workers == 3

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, *args, **kwargs):
            submitted.append((function, args, kwargs))
            return ImmediateFuture()

    monkeypatch.setattr(api, "ProcessPoolExecutor", ImmediatePool)
    result = api.reduce_many(
        [(object(), None), (object(), None)], object(), align=False, jobs=3
    )

    assert len(submitted) == 2
    assert all(call[0] is api.reduce for call in submitted)
    assert len(result.reductions) == 2


def test_transit_series_write_to_persists_final_wcs_and_solutions(tmp_path) -> None:
    reduction = _reduction("WINA00000001", 1)
    solution = decanter.WavecalSolution(
        frame_ids=("WINA00000001",),
        orders=(159, 160, 161),
        velocity=np.full((1, 3), 2.0),
        source=np.full((1, 3), "telluric", dtype="U16"),
        bracketed=np.ones((1, 3), dtype=bool),
    )
    corrected = solution.apply(reduction)
    series = TransitSeries(
        reductions=[corrected],
        shifts=np.asarray([0.125]),
        refid=0,
        wavecal_solution=solution,
    )

    series.write_to(tmp_path)

    spectrum_path = next((tmp_path / "WINA00000001").glob("*_m159_*VAC.fits"))
    with fits.open(spectrum_path) as hdul:
        assert hdul[0].header["WAVECAL"] == "hybrid_refit"
        assert hdul[0].header["WAVSHIFT"] == 0.125
        assert hdul[0].header["CRVAL1"] < reduction.obj[(1.30, 159)].crval1
    assert (tmp_path / "warp_alignment.npz").exists()
    assert (tmp_path / "wavecal_solution.npz").exists()


def test_reduce_many_writes_only_after_physical_layer(monkeypatch, tmp_path) -> None:
    reduction = _reduction("WINA00000001", 1)
    monkeypatch.setattr(api, "reduce", lambda *args, **kwargs: reduction)

    def fake_calibrate(series, config, *, verbose, diagnostic_pdf):
        updated = _reduction("WINA00000001", 1)
        updated.meta["WAVECAL"] = "test-final-layer"
        return TransitSeries(
            reductions=[updated], shifts=series.shifts, refid=series.refid
        )

    monkeypatch.setattr(api, "calibrate_wavelengths", fake_calibrate)
    api.reduce_many(
        [(object(), None)], object(), align=False,
        wavecal_config=decanter.WavecalConfig(), workdir=tmp_path,
    )

    spectrum_path = next((tmp_path / "WINA00000001").glob("*_m159_*VAC.fits"))
    with fits.open(spectrum_path) as hdul:
        assert hdul[0].header["WAVECAL"] == "test-final-layer"


def test_combine_regrids_different_physical_wavecal_wcs() -> None:
    """A corrected series must not average mismatched wavelength pixels."""
    first = _reduction("WINA00000001", 1)
    second = _reduction("WINA00000002", 2)
    key = (1.30, 159)
    reference = first.obj[key]
    shifted = OrderSpectrum(
        order=reference.order,
        fsr_cut=reference.fsr_cut,
        flux=(reference.wavelength + 0.03).astype(np.float32),
        crval1=reference.crval1 + 0.03,
        cdelt1=reference.cdelt1,
        crpix1=reference.crpix1,
    )
    first.obj[key].flux[:] = first.obj[key].wavelength.astype(np.float32)
    second.obj[key] = shifted

    combined = decanter.combine([first, second], cut=1.30, weight="uniform")

    expected = combined.obj[key].wavelength
    np.testing.assert_allclose(combined.obj[key].flux[1:], expected[1:], atol=2e-3)


def test_pooled_peaks_locates_and_characterizes_the_common_mode() -> None:
    grid = np.arange(-50.0, 50.01, 0.25)
    truth = np.array([-3.4, 0.0, 6.1])
    score = np.asarray([
        11.0 * np.exp(-0.5 * ((grid - centre) / 7.0) ** 2) for centre in truth
    ])

    peaks = api._pooled_peaks(score, grid)

    np.testing.assert_allclose(peaks["velocity"], truth, atol=0.05)
    # A 7 km/s Gaussian sigma is a 16.5 km/s FWHM, measured off a near-zero
    # baseline, so the half-height width is the profile's own.
    np.testing.assert_allclose(peaks["fwhm"], 16.5, atol=1.0)
    assert np.all(peaks["snr"] > 5.0)
    # Every rival outside the wings is baseline, far below the peak.
    assert np.all(peaks["score"] - peaks["secondary"] > 5.0)


def test_pooled_peaks_returns_nan_for_an_empty_exposure() -> None:
    grid = np.arange(-10.0, 10.01, 0.5)
    score = np.full((2, grid.size), np.nan)
    score[0] = np.exp(-0.5 * (grid / 2.0) ** 2)

    peaks = api._pooled_peaks(score, grid)

    assert np.isfinite(peaks["velocity"][0])
    assert np.isnan(peaks["velocity"][1])


def test_order_bootstrap_reports_the_spread_between_orders() -> None:
    grid = np.arange(-20.0, 20.01, 0.5)
    centres = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    evidence = np.asarray([
        [10.0 * np.exp(-0.5 * ((grid - centre) / 3.0) ** 2)]
        for centre in centres
    ])

    sigma, low, high = api._order_bootstrap(evidence, grid, draws=200)

    assert np.isfinite(sigma[0]) and sigma[0] > 0.05
    assert low[0] < high[0]


def test_order_bootstrap_declines_with_too_few_orders() -> None:
    grid = np.arange(-5.0, 5.01, 0.5)
    evidence = np.zeros((2, 1, grid.size))

    sigma, low, high = api._order_bootstrap(evidence, grid, draws=10)

    assert np.all(np.isnan(sigma)) and np.all(np.isnan(low)) and np.all(np.isnan(high))


def _bare_run(atmospheric=None) -> SimpleNamespace:
    """The least a report needs: a solution with no template fits behind it."""
    n_frames, n_orders, n_pixels = 4, 3, 64
    orders = (159, 160, 161)
    series = SimpleNamespace(
        n_frames=n_frames, n_orders=n_orders, n_pixels=n_pixels, orders=orders,
        frame_ids=tuple(f"f{i}" for i in range(n_frames)),
        wave=np.linspace(10000.0, 10100.0, n_pixels)[:, None] * np.ones(n_orders),
        elapsed_hours=np.linspace(0.0, 3.0, n_frames), instmode="HIRES-Y",
        summary=lambda: "4 frames x 3 orders", dv_pix_kms=np.ones(n_orders),
        sky=None, meta=[{"DATE-OBS": "2024-09-16"}] * n_frames,
        time_jd=np.linspace(2460000.0, 2460000.2, n_frames),
    )
    velocity = np.zeros((n_frames, n_orders))
    solution = decanter.WavecalSolution(
        frame_ids=series.frame_ids, orders=orders, velocity=velocity,
        source=np.full(velocity.shape, "interpolated", dtype="U16"),
        bracketed=np.ones(velocity.shape, dtype=bool),
    )
    return SimpleNamespace(
        series=series, config=decanter.WavecalConfig(), solution=solution,
        telluric_model=None, oh_model=None,
        telluric_accepted=np.zeros(velocity.shape, dtype=bool),
        oh_accepted=np.zeros(velocity.shape, dtype=bool),
        telluric_velocity=velocity * np.nan, oh_velocity=velocity * np.nan,
        telluric_peak=velocity * np.nan, oh_peak=velocity * np.nan,
        smooth_velocity=velocity, atmospheric=atmospheric,
    )


def _fake_common_mode(n_frames: int) -> object:
    from decanter.wavecal.solve import AtmosphericCommonMode

    grid = np.arange(-10.0, 10.01, 0.5)
    score = np.asarray([
        10.0 * np.exp(-0.5 * ((grid - 0.3 * i) / 3.0) ** 2) for i in range(n_frames)
    ])
    ones = np.ones(n_frames)
    return AtmosphericCommonMode(
        velocity_grid_kms=grid, telluric_score=score,
        # An OH pool that never contributed, so the report has to survive one
        # tracer being entirely absent.
        oh_score=score * np.nan, joint_score=score,
        peak_velocity_kms=0.3 * np.arange(n_frames),
        telluric_peak_velocity_kms=0.3 * np.arange(n_frames),
        oh_peak_velocity_kms=ones * np.nan, peak_snr=8.0 * ones,
        peak_score=10.0 * ones, secondary_score=ones,
        secondary_separation_kms=20.0 * ones, fwhm_kms=7.0 * ones,
        bootstrap_sigma_kms=0.1 * ones, bootstrap_p16_kms=-0.1 * ones,
        bootstrap_p84_kms=0.1 * ones,
        common_velocity_kms=0.3 * np.arange(n_frames) - 0.45,
        telluric_orders=(159, 160), oh_orders=(), search_kms=10.0, step_kms=0.5,
    )


def _page_count(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


def test_report_gains_the_pooled_ccf_pages(tmp_path: Path) -> None:
    from decanter.wavecal.report import wavecal_report_pdf

    plain = tmp_path / "plain.pdf"
    wavecal_report_pdf(_bare_run(), plain, dataset="fake")

    prealigned = tmp_path / "prealigned.pdf"
    wavecal_report_pdf(
        _bare_run(_fake_common_mode(4)), prealigned, dataset="fake"
    )

    assert _page_count(prealigned) == _page_count(plain) + 2


def test_series_reduction_frees_its_two_dimensional_intermediates() -> None:
    """The alignment tail keeps its 1D inputs; the 2D arrays are released."""
    from decanter._reduction import Intermediates

    reduction = _reduction("WINA00000001", 1)
    reduction.intermediates = Intermediates(
        obj_raw=np.zeros((8, 8)),
        obj_sscfm=np.zeros((8, 8)),
        strips_obj={159: np.zeros((4, 8))},
        spectra_1d={159: np.arange(8.0)},
        strip_wcs={159: (10000.0, 0.1)},
        sky_1d={159: np.arange(8.0)},
    )

    api._release_2d_intermediates(reduction)

    kept = reduction.intermediates
    assert kept.obj_raw is None
    assert kept.obj_sscfm is None
    assert kept.strips_obj == {}
    # The alignment tail re-reads these, so they have to survive.
    np.testing.assert_array_equal(kept.spectra_1d[159], np.arange(8.0))
    np.testing.assert_array_equal(kept.sky_1d[159], np.arange(8.0))
    assert kept.strip_wcs[159] == (10000.0, 0.1)
