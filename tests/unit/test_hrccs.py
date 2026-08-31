from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from decanter.hrccs.analysis import (
    ComponentResult,
    _filtered_cube,
    _global_null_fap,
    _map_summary,
    _select_component,
    combine_order_ccfs,
)
from decanter.hrccs.config import (
    AtmosphereConfig,
    HRCCSConfig,
    InjectionConfig,
    InputConfig,
    ReductionConfig,
    SearchConfig,
    SystemConfig,
    load_config,
)
from decanter.hrccs.detrend import svd_path
from decanter.hrccs.models import (
    Template,
    TemplateFactory,
    _cia_supported_indices,
    _exomol_path,
    _is_atomic,
    _molecular_database,
    _sample_instrument,
    _wide_wavelength_grid,
)
from decanter.hrccs.orbit import _orbital_velocity_basis
from decanter.wavecal.products import telluric_product


def _component(count, local_snr):
    empty = np.empty((0,))
    return ComponentResult(
        count=count,
        residual_cube=empty,
        filtered_template_cube=empty,
        exposure_ccf=empty,
        raw_map=empty,
        snr_map=empty,
        expected_snr=np.nan,
        local_peak_snr=local_snr,
        local_peak_kp_kms=np.nan,
        local_peak_vsys_kms=np.nan,
        peak_snr=np.nan,
        peak_kp_kms=np.nan,
        peak_vsys_kms=np.nan,
    )


def test_component_selection_uses_local_peak_not_expected_or_global():
    components = (_component(1, 2.1), _component(4, 5.7), _component(8, 3.2))
    assert _select_component(components).count == 4


def test_component_selection_ignores_nonfinite_local_peaks():
    assert _select_component((_component(1, np.nan), _component(2, 1.0))).count == 2
    with pytest.raises(ValueError, match="all SVD-component"):
        _select_component((_component(1, np.nan),))


def test_map_summary_separates_expected_local_and_global_maxima():
    kp = np.arange(0.0, 201.0, 10.0)
    vsys = np.arange(-50.0, 51.0, 5.0)
    values = np.zeros((kp.size, vsys.size))
    values[np.where(kp == 100)[0][0], np.where(vsys == 0)[0][0]] = 2.0
    values[np.where(kp == 110)[0][0], np.where(vsys == 5)[0][0]] = 4.0
    values[np.where(kp == 190)[0][0], np.where(vsys == -45)[0][0]] = 9.0
    summary = _map_summary(values, kp, vsys, 100.0, 0.0, 20.0, 10.0)
    assert summary == (2.0, 4.0, 110.0, 5.0, 9.0, 190.0, -45.0)


def test_fap_uses_global_null_maxima_as_thresholds():
    # Two full-map maxima exceed the observed 4-sigma local peak. The +1
    # finite-sample correction therefore gives 3/5.
    null_global = np.asarray([3.0, 4.2, 5.1, 2.8])
    assert _global_null_fap(4.0, null_global) == pytest.approx(3.0 / 5.0)


def test_fap_ignores_nonfinite_null_maxima():
    assert _global_null_fap(4.0, [5.0, np.nan, 2.0]) == pytest.approx(2.0 / 3.0)
    assert np.isnan(_global_null_fap(np.nan, [5.0]))
    assert np.isnan(_global_null_fap(4.0, [np.nan]))


def test_default_search_grids_and_local_window():
    config = HRCCSConfig(
        input=InputConfig("products"),
        system=SystemConfig(
            period_days=1.0,
            event_midpoint_bjd_tdb=2_460_000.0,
            event_duration_hours=2.0,
            expected_kp_kms=200.0,
        ),
    )
    _, kp, vsys = config.grids(-10.0)
    assert kp[0] == 0.0
    assert kp[-1] == 300.0
    assert vsys[0] == -50.0
    assert vsys[-1] == 50.0
    assert config.search.local_kp_half_width_kms == 30.0
    assert config.search.local_vsys_half_width_kms == 15.0
    assert InjectionConfig().null_realizations == 5
    assert config.show_progress is True


def test_search_grid_steps_must_be_positive():
    config = HRCCSConfig(
        input=InputConfig("products"),
        system=SystemConfig(
            period_days=1.0,
            event_midpoint_bjd_tdb=2_460_000.0,
            event_duration_hours=2.0,
            expected_kp_kms=200.0,
        ),
    )
    with pytest.raises(ValueError, match="kp_step_kms"):
        replace(config, search=SearchConfig(kp_step_kms=0.0)).validate()
    with pytest.raises(ValueError, match="null_realizations"):
        replace(config, injection=InjectionConfig(null_realizations=0)).validate()
    with pytest.raises(ValueError, match="injection scale"):
        replace(config, injection=InjectionConfig(scale=0.0)).validate()
    with pytest.raises(ValueError, match="wide_model_chunk_points"):
        replace(config, atmosphere=AtmosphereConfig(wide_model_chunk_points=255)).validate()


def test_notebook_reduction_configuration_is_valid():
    config = HRCCSConfig(
        input=InputConfig("products"),
        system=SystemConfig(
            period_days=1.0, event_midpoint_bjd_tdb=2_460_000.0,
            event_duration_hours=2.0, expected_kp_kms=200.0,
        ),
        reduction=ReductionConfig(
            telluric_mask_scope="signal", edge_trim_pixels=0,
            ccf_lsf_margin_widths=3.0,
        ),
    )
    config.validate()


def test_eclipse_configuration_uses_generic_event_fields():
    config = HRCCSConfig(
        input=InputConfig("products"),
        system=SystemConfig(
            observation_type="eclipse", period_days=4.0,
            event_midpoint_bjd_tdb=2_460_000.0, event_duration_hours=2.0,
            expected_kp_kms=170.0,
        ),
    )
    config.validate()
    assert config.system.event_midpoint == 2_460_000.0
    assert config.system.event_duration == 2.0


def test_circular_eclipse_velocity_has_opposite_transit_slope():
    phase = np.linspace(-0.1, 0.1, 9)
    transit = _orbital_velocity_basis(phase, "transit")
    eclipse = _orbital_velocity_basis(phase, "eclipse")
    np.testing.assert_allclose(transit, np.sin(2.0 * np.pi * phase))
    np.testing.assert_allclose(eclipse, -transit)


def test_eccentric_eclipse_velocity_includes_ecosomega_offset():
    eccentricity = 0.066
    omega = 54.0
    basis = _orbital_velocity_basis(
        np.asarray([0.0]), "eclipse", eccentricity, omega,
    )
    assert basis[0] == pytest.approx(-eccentricity * np.cos(np.deg2rad(omega)))


def test_notebook_svd_is_linear_uncentered_svd():
    rng = np.random.default_rng(42)
    matrix = 10.0 + rng.normal(size=(9, 17))
    # The notebook convention is the public helper's canonical default.
    path = svd_path(matrix, (3,), np.ones(matrix.shape[1], bool))
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    expected_lower = (u[:, :3] * singular[:3]) @ vt[:3]
    np.testing.assert_allclose(path.lower[3], expected_lower, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        path.residuals[3], matrix - expected_lower, rtol=1e-13, atol=1e-13
    )


def test_notebook_template_filter_is_exact_injected_svd_refit():
    rng = np.random.default_rng(7)
    matrix = 1000.0 + rng.normal(size=(10, 23))
    path = svd_path(matrix, (2,), np.ones(matrix.shape[1], bool))
    model = np.zeros((10, 1, 23))
    model[3:7, 0, 8:15] = -2.0e-3
    actual = _filtered_cube(model, (path,), 2)[:, 0]

    scaling = path.lower[2]
    injected_path = svd_path(
        scaling * (1.0 + model[:, 0]), (2,), np.ones(23, bool)
    )
    control_path = svd_path(scaling, (2,), np.ones(23, bool))
    expected = injected_path.residuals[2] - control_path.residuals[2]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_equal_order_combination_matches_notebook_sum():
    order_ccf = np.array([[[1.0, 2.0]], [[3.0, np.nan]], [[-1.0, 4.0]]])
    # Equal per-order summation is the notebook convention and canonical default.
    actual = combine_order_ccfs(order_ccf)
    np.testing.assert_allclose(actual, [[3.0, 6.0]])


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("WIDE", 28_000.0), ("HIRES-Y", 68_000.0), ("HIRES-J", 68_000.0),
     ("Y", 68_000.0), ("HIRES_J", 68_000.0)],
)
def test_template_resolution_follows_instrument_mode(mode, expected):
    assert AtmosphereConfig().resolving_power_for(mode) == expected


def test_explicit_template_resolution_overrides_mode():
    atmosphere = AtmosphereConfig(resolving_power=45_000.0)
    assert atmosphere.resolving_power_for("WIDE") == 45_000.0
    with pytest.raises(ValueError, match="INSTMODE"):
        AtmosphereConfig().resolving_power_for("UNKNOWN")


@pytest.mark.parametrize(("mode", "resolution"), [("WIDE", 28_000.0),
                                                    ("HIRES-Y", 68_000.0)])
def test_wide_template_is_built_once_then_sampled_per_order(tmp_path, mode, resolution):
    atmosphere = AtmosphereConfig(
        backend="analytic", cache_dir=str(tmp_path), resolving_power=resolution,
    )
    system = SystemConfig(stellar_radius_rsun=1.0, planet_radius_rjup=1.0)
    factory = TemplateFactory(system, atmosphere, instmode=mode)
    first_wave = np.linspace(1.10, 1.11, 128)
    second_wave = np.linspace(1.20, 1.21, 96)
    wide = factory.build_wide("H2O", [first_wave, second_wave])
    first, second = factory.sample_orders(wide, [first_wave, second_wave])
    np.testing.assert_array_equal(first.wavelength_um, first_wave)
    np.testing.assert_array_equal(second.wavelength_um, second_wave)
    assert first.wavelength_um.size != second.wavelength_um.size
    assert first.metadata["instrument_mode"] == mode
    assert first.metadata["resolving_power"] == resolution
    assert first.metadata["model_scope"] == "single wide-band template"
    assert first.metadata["wide_model_chunks"] > 1
    assert len(list((tmp_path / "templates").glob("H2O_*.npz"))) == 1


def test_template_cache_key_ignores_orbital_geometry(tmp_path):
    atmosphere = AtmosphereConfig(
        backend="analytic", cache_dir=str(tmp_path), resolving_power=68_000.0,
    )
    base = SystemConfig(
        stellar_radius_rsun=2.35, planet_radius_rjup=1.926,
        planet_mass_mjup=12.37, equilibrium_temperature_k=3520.0,
        metallicity_dex=-0.34,
    )
    eccentric = replace(
        base, observation_type="eclipse", eccentricity=0.066,
        argument_of_periastron_deg=54.0, event_midpoint_bjd_tdb=2_460_000.0,
    )
    wave = np.geomspace(0.95, 1.12, 1024)
    first = TemplateFactory(base, atmosphere, instmode="HIRES-Y")
    second = TemplateFactory(eccentric, atmosphere, instmode="HIRES-Y")
    assert first._path("H2O", wave) == second._path("H2O", wave)


@pytest.mark.parametrize("resolution", [28_000.0, 68_000.0])
def test_wide_grid_oversamples_instrument_resolution(resolution):
    orders = np.array([np.linspace(0.95, 1.05, 100), np.linspace(1.20, 1.30, 100)])
    wave = _wide_wavelength_grid(orders, resolution)
    samples_per_fwhm = 1.0 / (resolution * np.max(np.diff(np.log(wave))))
    assert samples_per_fwhm >= 4.99
    assert wave[0] < 0.95
    assert wave[-1] > 1.30
    margin = 350.0 / 299_792.458
    assert wave[0] == pytest.approx(0.95 * (1.0 - margin))
    assert wave[-1] == pytest.approx(1.30 * (1.0 + margin))


@pytest.mark.parametrize("resolution", [28_000.0, 68_000.0])
def test_exojax_instrument_sampling_has_requested_resolution(resolution):
    pytest.importorskip("exojax")
    speed_of_light_kms = 299_792.458
    nu = np.geomspace(9_900.0, 10_100.0, 65_536)
    impulse = np.zeros(nu.size)
    impulse[nu.size // 2] = 1.0
    wavelength = 1.0e4 / nu[::-1]
    sampled = _sample_instrument(nu, impulse, wavelength, resolution)[::-1]
    velocity = speed_of_light_kms * np.log(nu / nu[nu.size // 2])
    above_half_maximum = np.flatnonzero(sampled >= 0.5 * np.max(sampled))
    measured_fwhm = (velocity[above_half_maximum[-1]]
                     - velocity[above_half_maximum[0]])
    expected_fwhm = speed_of_light_kms / resolution
    assert measured_fwhm == pytest.approx(expected_fwhm, rel=0.03)


def test_cia_support_excludes_unavailable_and_upper_edge_samples():
    model_nu = np.array([9_998.0, 9_999.0, 10_000.0, 10_001.0])
    cia_nu = np.arange(20.0, 10_001.0)
    np.testing.assert_array_equal(_cia_supported_indices(model_nu, cia_nu), [0, 1])
    assert _cia_supported_indices(model_nu, np.array([])).size == 0


def test_atomic_classifier_does_not_misclassify_diatomic_molecules():
    assert _is_atomic("Fe")
    assert _is_atomic("Na+")
    assert not _is_atomic("OH")
    assert not _is_atomic("CO")
    assert not _is_atomic("H2O")


def test_molecular_database_auto_routes_hitran_and_exomol():
    assert _molecular_database("H2O", {}) == "hitran"
    assert _molecular_database("CO", {}) == "hitran"
    assert _molecular_database("FeH", {}) == "exomol"
    assert _molecular_database("CrH", {}) == "exomol"
    assert _molecular_database("H2O", {"H2O": "exomol"}) == "exomol"


def test_exomol_dataset_defaults_and_overrides(tmp_path):
    assert _exomol_path("CrH", tmp_path, {}) == (
        tmp_path / "CrH" / "52Cr-1H" / "MoLLIST"
    )
    assert _exomol_path("TiO", tmp_path, {"TiO": "46Ti-16O/Toto"}) == (
        tmp_path / "TiO" / "46Ti-16O" / "Toto"
    )


def test_wide_atomic_template_accepts_empty_internal_chunks(tmp_path):
    atmosphere = AtmosphereConfig(
        backend="analytic", cache_dir=str(tmp_path), resolving_power=28_000.0,
    )
    factory = TemplateFactory(
        SystemConfig(stellar_radius_rsun=1.0, planet_radius_rjup=1.0),
        atmosphere, instmode="WIDE",
    )
    wave = np.linspace(1.0, 1.1, 8)
    common = {"backend": "exojax", "continuum": {"cia_grid_coverage": {}}}
    pieces = [
        Template("Mg", wave[:4], np.full(4, 0.01), np.zeros(4),
                 {**common, "line_count": 0}),
        Template("Mg", wave[4:], np.full(4, 0.01), np.arange(4.0),
                 {**common, "line_count": 3}),
    ]
    stitched = factory._stitch_wide("Mg", wave, pieces, 4)
    assert stitched.metadata["line_count"] == 3
    np.testing.assert_array_equal(stitched.contrast[:4], 0.0)


def test_wide_atomic_template_rejects_species_absent_from_full_band(tmp_path):
    atmosphere = AtmosphereConfig(
        backend="analytic", cache_dir=str(tmp_path), resolving_power=28_000.0,
    )
    factory = TemplateFactory(
        SystemConfig(stellar_radius_rsun=1.0, planet_radius_rjup=1.0),
        atmosphere, instmode="WIDE",
    )
    wave = np.linspace(1.0, 1.1, 8)
    piece = Template(
        "Mg", wave, np.full(8, 0.01), np.zeros(8),
        {"backend": "exojax", "line_count": 0,
         "continuum": {"cia_grid_coverage": {}}},
    )
    with pytest.raises(ValueError, match="contains no Mg lines across"):
        factory._stitch_wide("Mg", wave, [piece], 8)


def test_wavecal_telluric_product_is_continuous_and_unthresholded(tmp_path):
    wave = np.array([[10_000.0, 10_100.0], [10_001.0, 10_101.0], [10_002.0, 10_102.0]])
    native = np.array([[0.99, 0.75], [0.85, 0.60], [0.98, 0.80]])
    series = SimpleNamespace(
        n_frames=2, n_orders=2, n_pixels=3, wave=wave,
        frame_ids=("a", "b"), orders=(42, 43),
    )
    model = SimpleNamespace(
        species=("H2O",), native_template=native,
        parameters=np.zeros((2, 6)), family=np.array(["gaussian", "gaussian"]),
        template_rms=np.array([0.01, 0.02]), meta={},
    )
    solution = SimpleNamespace(velocity=np.zeros((2, 2)), mode="hybrid_static")
    run = SimpleNamespace(
        telluric_model=model, series=series, solution=solution,
        telluric_refit_parameters=None, telluric_accepted=np.ones((2, 2), bool),
        telluric_peak=np.ones((2, 2)), _telluric_tau=None,
    )
    output = telluric_product(run, tmp_path / "telluric_transmission.npz")
    with np.load(output, allow_pickle=False) as product:
        assert product["transmission"].shape == (2, 2, 3)
        np.testing.assert_allclose(product["transmission"][0], native.T)
        assert np.any((product["transmission"] > 0.60)
                      & (product["transmission"] < 0.99))


def _config_text(observation_type: str, scope: str) -> str:
    return (
        'output_dir = "out"\n'
        "[input]\n"
        'decanter_dir = "spectra"\n'
        "[system]\n"
        f'observation_type = "{observation_type}"\n'
        "period_days = 4.0\n"
        "event_midpoint_bjd_tdb = 2459315.7361\n"
        "event_duration_hours = 2.2\n"
        "expected_kp_kms = 160.0\n"
        "[reduction]\n"
        f'telluric_mask_scope = "{scope}"\n'
    )


def test_unknown_mask_scope_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(_config_text("transit", "in_transit"))

    with pytest.raises(ValueError, match="telluric_mask_scope must be"):
        load_config(path)


def test_signal_scope_is_accepted_for_both_geometries(tmp_path) -> None:
    for observation_type in ("transit", "eclipse"):
        path = tmp_path / f"{observation_type}.toml"
        path.write_text(_config_text(observation_type, "signal"))
        assert load_config(path).reduction.telluric_mask_scope == "signal"
