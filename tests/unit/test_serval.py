"""Unit tests for the optional post-wavecal SERVAL diagnostic."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from decanter.serval import (
    TransitEphemeris,
    _adapter_text,
    _gnuplot_environment,
    _match_serval_rows,
    _retain_usable_orders,
    _telluric_mask,
    _binning_curve,
    _event_selection,
    _two_bins,
    robust_scatter,
    run_serval_rv_stability,
)
from decanter.wavecal.series import Series


def _series() -> Series:
    wave = np.column_stack((np.linspace(10_000.0, 10_010.0, 12),
                            np.linspace(10_100.0, 10_110.0, 12)))
    return Series(
        frame_ids=("a", "b"), orders=(160, 161), wave=wave,
        dv_pix_kms=np.ones(2), obj=np.ones((2, 12, 2)), sky=None,
        noise_fraction=np.full((2, 2), 0.01), time_jd=np.asarray([1.0, 2.0]),
        sky_time_jd=np.asarray([1.0, 2.0]), airmass=np.ones(2),
        instmode="HIRES-Y", fsr_cut=1.3, meta=[{}, {}],
    )


def test_robust_scatter_is_mad_sigma() -> None:
    assert robust_scatter([0.0, 1.0, 2.0]) == pytest.approx(1.4826)


def test_two_chronological_bins_use_inverse_variance_weights() -> None:
    time = np.asarray([4.0, 1.0, 3.0, 2.0])
    value = np.asarray([40.0, 10.0, 30.0, 20.0])
    error = np.asarray([1.0, 1.0, 2.0, 1.0])
    bin_time, bin_value, bin_error, count = _two_bins(time, value, error)
    np.testing.assert_allclose(bin_time, [1.5, 3.5])
    # First raw bin is 15; second is inverse-variance mean 38. The two-bin
    # display is then centered on their mean.
    np.testing.assert_allclose(bin_value, [-11.5, 11.5])
    np.testing.assert_allclose(bin_error, [1 / np.sqrt(2), 1 / np.sqrt(1.25)])
    np.testing.assert_array_equal(count, [2, 2])


def test_two_bins_split_by_time_not_by_the_transit() -> None:
    """Both halves are equal counts of exposures, wherever the transit fell."""
    # Five exposures with the gap between the second and third: an even split
    # by count puts three in the first bin, which a pre/post split would not.
    time = np.asarray([1.0, 1.1, 1.9, 2.0, 2.1])
    value = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0])
    error = np.ones(5)

    bin_time, bin_value, _, count = _two_bins(time, value, error)

    np.testing.assert_array_equal(count, [3, 2])
    np.testing.assert_allclose(bin_time, [np.mean(time[:3]), np.mean(time[3:])])
    # Bins average to 20 and 45; the display is centered on their mean, 32.5.
    np.testing.assert_allclose(bin_value, [-12.5, 12.5])


def test_complete_serval_table_matches_in_order_with_duplicate_bjd() -> None:
    bjd = np.asarray([2_460_000.1, 2_460_000.2, 2_460_000.2])
    table = np.zeros((3, 5))
    table[:, 0] = bjd
    exposure, rows = _match_serval_rows(table, bjd)
    np.testing.assert_array_equal(exposure, [0, 1, 2])
    np.testing.assert_array_equal(rows, [0, 1, 2])


def test_event_selection_uses_periodic_bjd_ephemeris() -> None:
    ephemeris = TransitEphemeris(
        period_days=2.0,
        event_midpoint_bjd_tdb=2_460_000.0,
        event_duration_hours=2.0,
    )
    bjd = np.asarray([2_460_001.94, 2_460_001.97, 2_460_002.00,
                      2_460_002.03, 2_460_002.06])
    utc = bjd - 0.003
    in_transit, window = _event_selection(bjd, utc, ephemeris)
    np.testing.assert_array_equal(in_transit, [False, True, True, True, False])
    np.testing.assert_allclose(
        window,
        [2_460_002.0 - 0.003 - 1 / 24, 2_460_002.0 - 0.003 + 1 / 24],
    )


def test_ephemeris_reads_hrccs_system_table(tmp_path) -> None:
    path = tmp_path / "target.toml"
    path.write_text(
        "[system]\nperiod_days=3.0\nevent_midpoint_bjd_tdb=2459000.0\n"
        "event_duration_hours=2.5\n"
    )
    ephemeris = TransitEphemeris.from_toml(path)
    assert ephemeris.period_days == 3.0
    assert ephemeris.event_duration_hours == 2.5


def test_eclipse_ephemeris_reads_generic_event_fields(tmp_path) -> None:
    path = tmp_path / "target.toml"
    path.write_text(
        "[system]\nobservation_type='eclipse'\nperiod_days=4.0\n"
        "event_midpoint_bjd_tdb=2459001.0\nevent_duration_hours=2.25\n"
    )
    ephemeris = TransitEphemeris.from_toml(path)
    assert ephemeris.observation_type == "eclipse"
    assert ephemeris.event_midpoint_bjd_tdb == 2459001.0
    assert ephemeris.event_duration_hours == 2.25
    assert ephemeris.out_of_event_abbreviation == "OOE"


def test_telluric_product_is_mapped_and_dilated(tmp_path) -> None:
    series = _series()
    transmission = np.ones((2, 2, 12))
    transmission[1, 0, 6] = 0.90
    wavelength = np.repeat(series.wave.T[None, :, :], 2, axis=0)
    product = tmp_path / "telluric_transmission.npz"
    np.savez(
        product, orders=np.asarray(series.orders), wavelength_angstrom=wavelength,
        transmission=transmission,
    )
    mask = _telluric_mask(product, series, 0.995)
    assert np.all(mask[0, 3:10])
    assert not np.any(mask[1])


def test_serval_adapter_uses_all_orders_and_nominal_resolution() -> None:
    text, order_set = _adapter_text(26, 2550, 68_000.0)
    assert order_set == "0:26"
    assert 'R = 68000.0' in text
    assert 'iomax = 26' in text
    assert 'pmin = 164' in text
    assert 'pmax = 2386' in text


def test_headless_gnuplot_shim_exits_with_open_stdin(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("decanter.serval.shutil.which", lambda *args, **kwargs: None)
    environment = {"PATH": ""}
    _gnuplot_environment(environment, tmp_path)
    process = subprocess.Popen(
        [str(tmp_path / "bin" / "gnuplot"), "-e", "plot NaN"],
        stdin=subprocess.PIPE,
    )
    try:
        assert process.wait(timeout=1.0) == 0
    finally:
        if process.poll() is None:
            process.kill()

    persistent = subprocess.Popen(
        [str(tmp_path / "bin" / "gnuplot")], stdin=subprocess.PIPE,
    )
    assert persistent.stdin is not None
    persistent.stdin.write(b"reset\n")
    persistent.stdin.flush()
    assert persistent.poll() is None
    persistent.stdin.close()
    assert persistent.wait(timeout=1.0) == 0


def test_orders_without_enough_unmasked_interior_are_excluded() -> None:
    n_pixels = 600
    series = Series(
        frame_ids=("a", "b"), orders=(160, 161, 162),
        wave=np.column_stack(tuple(
            np.linspace(10_000.0 + 100 * index, 10_010.0 + 100 * index, n_pixels)
            for index in range(3)
        )),
        dv_pix_kms=np.ones(3), obj=np.ones((2, n_pixels, 3)), sky=None,
        noise_fraction=np.full((2, 3), 0.01), time_jd=np.asarray([1.0, 2.0]),
        sky_time_jd=np.asarray([1.0, 2.0]), airmass=np.ones(2),
        instmode="HIRES-Y", fsr_cut=1.3, meta=[{}, {}],
    )
    mask = np.zeros((3, n_pixels), dtype=bool)
    mask[0] = True
    retained, retained_mask, keep = _retain_usable_orders(series, mask)
    assert retained.orders == (161, 162)
    assert retained.obj.shape == (2, n_pixels, 2)
    assert retained_mask.shape == (2, n_pixels)
    np.testing.assert_array_equal(keep, [False, True, True])


def test_serval_step_requires_physical_wavecal(tmp_path) -> None:
    transit = SimpleNamespace(wavecal_solution=None)
    with pytest.raises(ValueError, match="physical wavelength calibration"):
        run_serval_rv_stability(
            transit,
            tmp_path,
            ephemeris=TransitEphemeris(1.0, 2_460_000.0, 2.0),
        )


def test_binning_curve_follows_root_n_for_white_noise() -> None:
    generator = np.random.default_rng(11)
    n = 240
    time = 2460000.0 + np.arange(n) / 288.0
    residual = generator.normal(0.0, 40.0, n)
    error = np.full(n, 40.0)

    size, rms, rms_error, n_bins, white = _binning_curve(time, residual, error)

    assert size[0] == 1
    np.testing.assert_allclose(rms[0], np.std(residual, ddof=1))
    np.testing.assert_allclose(white, rms[0] / np.sqrt(size))
    assert np.all(n_bins >= 4)
    # A binned RMS carries its own scatter of 1/sqrt(2(M-1)), so only the
    # well-sampled part of the curve is pinned to the sqrt(N) line.
    ratio = rms / white
    well_sampled = n_bins >= 20
    assert 0.94 < float(np.median(ratio[well_sampled])) < 1.06
    assert np.all(np.abs(rms - white) < 6.0 * rms_error + 1e-9)


def test_binning_curve_does_not_integrate_down_a_drift() -> None:
    n = 120
    time = 2460000.0 + np.arange(n) / 288.0
    # A slow ramp is entirely correlated between neighbours, so bin averages
    # still trace it and the scatter barely moves.
    residual = np.linspace(-200.0, 200.0, n)
    error = np.full(n, 10.0)

    size, rms, _, _, white = _binning_curve(time, residual, error)

    assert rms[-1] > 0.9 * rms[0]
    assert rms[-1] / white[-1] > 0.8 * np.sqrt(size[-1])


def test_binning_curve_does_not_average_across_the_transit() -> None:
    time = 2460000.0 + np.arange(16) / 288.0
    window = np.array([time[7] + 1e-4, time[8] - 1e-4])
    # Each side sits at its own level, so a bin spanning the gap would average
    # the two together and make the step look like it had integrated down.
    residual = np.where(np.arange(16) < 8, -100.0, 100.0)
    error = np.full(16, 5.0)

    size, split, _, split_bins, _ = _binning_curve(time, residual, error, window)
    _, _, _, merged_bins, _ = _binning_curve(time, residual, error)

    # Eight exposures either side: bins of three leave two per side and a
    # discarded remainder, where binning straight through the gap would fit a
    # fifth bin across it.
    at_three = int(np.flatnonzero(size == 3)[0])
    assert split_bins[at_three] == 4
    assert merged_bins[at_three] == 5
    # No bin mixes the two levels, so the step never integrates down.
    assert np.all(split >= 100.0)


def test_binning_curve_declines_with_too_few_exposures() -> None:
    time = 2460000.0 + np.arange(3) / 288.0
    size, rms, rms_error, n_bins, white = _binning_curve(
        time, np.zeros(3), np.ones(3)
    )
    assert size.size == 0 and rms.size == 0 and white.size == 0


def _oh_run(series: Series) -> SimpleNamespace:
    """A wavecal run whose OH fit flags one line in the first order."""
    support = np.zeros((series.n_pixels, series.n_orders), dtype=bool)
    support[6, 0] = True
    return SimpleNamespace(
        series=series,
        solution=SimpleNamespace(mode="hybrid_refit"),
        oh_model=SimpleNamespace(
            support=support,
            line_count=np.asarray([1, 0]),
            rotational_temperature_k=190.0,
        ),
    )


def test_oh_mask_from_a_live_run_and_from_the_product_agree(tmp_path) -> None:
    """A reduction read back from disk masks the same airglow pixels."""
    from decanter.serval import _oh_mask
    from decanter.wavecal.products import airglow_product

    series = _series()
    run = _oh_run(series)
    product = airglow_product(run, tmp_path / "oh_support.npz")

    live = _oh_mask(run, series)
    from_disk = _oh_mask(None, series, product)

    assert np.any(live[0])
    assert not np.any(live[1])
    np.testing.assert_array_equal(live, from_disk)


def test_oh_mask_warns_when_no_support_is_available() -> None:
    from decanter.serval import _oh_mask

    series = _series()
    with pytest.warns(RuntimeWarning, match="will not mask airglow"):
        mask = _oh_mask(None, series, None)
    assert not np.any(mask)


def test_oh_mask_maps_orders_by_number_not_position() -> None:
    """The product may carry orders the reduction no longer retains."""
    from decanter.serval import _oh_mask
    from decanter.wavecal.products import airglow_product
    import tempfile

    series = _series()
    run = _oh_run(series)
    with tempfile.TemporaryDirectory() as directory:
        product = airglow_product(run, Path(directory) / "oh_support.npz")
        trimmed = replace(series, orders=(161,), wave=series.wave[:, 1:],
                          obj=series.obj[:, :, 1:],
                          noise_fraction=series.noise_fraction[:, 1:],
                          dv_pix_kms=series.dv_pix_kms[1:])
        mask = _oh_mask(None, trimmed, product)
    # Order 161 carried no OH lines, so dropping 160 must not shift its mask on.
    assert mask.shape == (1, series.n_pixels)
    assert not np.any(mask)


def test_ephemeris_carries_the_planet_name_for_the_figure_title(tmp_path) -> None:
    """Frame headers often name the star, or nothing; the TOML names the planet."""
    config = tmp_path / "system.toml"
    config.write_text(
        "[system]\n"
        'planet_name = "WASP-69 b"\n'
        "period_days = 3.8681382\n"
        "event_midpoint_bjd_tdb = 2455748.83344\n"
        "event_duration_hours = 2.1792\n"
    )

    ephemeris = TransitEphemeris.from_toml(config)

    assert ephemeris.planet_name == "WASP-69 b"


def test_ephemeris_without_a_planet_name_is_still_valid(tmp_path) -> None:
    config = tmp_path / "system.toml"
    config.write_text(
        "[system]\n"
        "period_days = 3.8681382\n"
        "event_midpoint_bjd_tdb = 2455748.83344\n"
        "event_duration_hours = 2.1792\n"
    )

    ephemeris = TransitEphemeris.from_toml(config)

    assert ephemeris.planet_name == ""


def _ephemeris(observation_type: str) -> TransitEphemeris:
    return TransitEphemeris(
        period_days=2.0,
        event_midpoint_bjd_tdb=2460000.0,
        event_duration_hours=2.4,
        observation_type=observation_type,
    )


def test_eclipse_keeps_the_exposures_where_the_planet_is_hidden() -> None:
    """A transit contaminates the in-event exposures, an eclipse the rest."""
    from decanter.serval import _event_selection, _uncontaminated_exposures

    # Six exposures inside a 2.4 h event centred on the midpoint, six outside.
    bjd = 2460000.0 + np.concatenate([
        np.linspace(-0.04, 0.04, 6), np.linspace(0.12, 0.20, 6),
    ])
    in_event, _ = _event_selection(bjd, bjd - 0.002, _ephemeris("transit"))
    assert np.count_nonzero(in_event) == 6

    transit_keep, transit_used, _ = _uncontaminated_exposures(in_event, "transit")
    eclipse_keep, eclipse_used, _ = _uncontaminated_exposures(in_event, "eclipse")

    np.testing.assert_array_equal(transit_keep, ~in_event)
    np.testing.assert_array_equal(eclipse_keep, in_event)
    np.testing.assert_array_equal(transit_keep, ~eclipse_keep)
    assert transit_used == "out-of-transit"
    assert eclipse_used == "in-eclipse"
