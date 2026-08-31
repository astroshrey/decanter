"""End-to-end orchestration for Decanter HRCCS products."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from decanter.hrccs.analysis import run_species
from decanter.hrccs.io import load_decanter
from decanter.hrccs.models import TemplateFactory
from decanter.hrccs.orbit import build_orbit


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _mask(cube, config, rv_grid, event_mask, signal_mask, resolving_power):
    n_orders, n_pixels = cube.wavelength_um.shape
    keep = np.ones((n_orders, n_pixels), dtype=bool)
    if cube.telluric_transmission is None:
        warnings.warn("telluric_transmission.npz is absent; no telluric pixels are masked",
                      RuntimeWarning, stacklevel=2)
    else:
        scope = config.telluric_mask_scope
        if scope == "signal":
            rows = np.asarray(signal_mask, dtype=bool)
        elif scope == "event":
            rows = np.asarray(event_mask, dtype=bool)
        elif scope == "out_of_event":
            rows = ~np.asarray(event_mask, dtype=bool)
        else:
            rows = np.ones(cube.flux.shape[0], dtype=bool)
        transmission = cube.telluric_transmission[rows]
        finite = np.isfinite(transmission)
        minimum = np.min(np.where(finite, transmission, np.inf), axis=0)
        minimum[~np.any(finite, axis=0)] = np.nan
        keep &= np.isfinite(minimum) & (minimum >= config.telluric_threshold)
    for order in range(n_orders):
        wave = cube.wavelength_um[order]
        dv = 299_792.458 * np.nanmedian(np.diff(np.log(wave)))
        velocity_margin = (np.max(np.abs(rv_grid))
                           + config.ccf_lsf_margin_widths * 299_792.458 / resolving_power)
        margin = config.edge_trim_pixels + int(np.ceil(velocity_margin / abs(dv)))
        keep[order, :min(margin, n_pixels)] = False
        keep[order, max(0, n_pixels - margin):] = False
        if np.count_nonzero(keep[order]) < config.min_valid_pixels:
            keep[order] = False
    if not np.any(np.sum(keep, axis=1) >= config.min_valid_pixels):
        raise ValueError("no order retains enough pixels after telluric and edge masking")
    return keep


def _save_result(result, output, orbit, rv_grid, kp_grid, vsys_grid):
    np.savez_compressed(
        output / f"{result.species}_hrccs_products.npz",
        rv_grid_kms=rv_grid, kp_grid_kms=kp_grid, vsys_grid_kms=vsys_grid,
        phase=orbit.phase, berv_kms=orbit.berv_kms,
        selected_svd_components=np.asarray(result.selected.count),
        observed_exposure_ccf=result.selected.exposure_ccf,
        observed_snr_map=result.selected.snr_map,
        injected_snr_map=result.injected.snr_map,
        null_mean_snr_map=result.null_mean_map,
        null_snr_at_expected=result.null_snr_at_expected,
        null_global_peak_snr=result.null_global_peak_snr,
        telluric_keep_mask=result.mask,
    )


def run(config):
    """Run all configured species and return their in-memory results."""
    config.validate()
    output = Path(config.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cube = load_decanter(config.input.decanter_dir, fsr_cut=config.input.fsr_cut,
                         orders=config.input.orders,
                         telluric_product=config.input.telluric_product)
    orbit = build_orbit(config.system, cube.time_jd_utc, cube.metadata,
                        Path(config.atmosphere.cache_dir).expanduser() / "simbad")
    rv_grid, kp_grid, vsys_grid = config.grids(orbit.stellar_rv_kms)
    resolving_power = config.atmosphere.resolving_power_for(cube.instmode)
    mask = _mask(
        cube, config.reduction, rv_grid, orbit.event_mask, orbit.signal_mask,
        resolving_power,
    )
    retained = np.sum(mask, axis=1) >= config.reduction.min_valid_pixels
    wavelength = cube.wavelength_um[retained]
    flux = cube.flux[:, retained]
    orders = cube.orders[retained]
    mask = mask[retained]
    prepared = np.asarray(flux, dtype=float).copy()
    atmosphere = replace(config.atmosphere, resolving_power=resolving_power)
    factory = TemplateFactory(config.system, atmosphere, instmode=cube.instmode)
    from tqdm.auto import tqdm

    results = []
    species_bar = tqdm(
        config.atmosphere.species, desc="HRCCS species", unit="species",
        disable=not config.show_progress, dynamic_ncols=True,
    )
    for species_index, species in enumerate(species_bar):
        species_bar.set_postfix_str(f"{species}: wide template", refresh=True)
        wide_template = factory.build_wide(
            species, wavelength, show_progress=config.show_progress,
        )
        templates = factory.sample_orders(wide_template, wavelength)
        species_bar.set_postfix_str(f"{species}: CCF/SVD", refresh=True)
        for template, wave in zip(templates, wavelength):
            if template.metadata.get("resolving_power") != resolving_power:
                raise RuntimeError("cached template has the wrong instrumental resolution")
            if not np.array_equal(template.wavelength_um, wave):
                raise RuntimeError("template is not sampled on its order wavelength grid")
        depths = np.asarray([template.transit_depth for template in templates])
        # ExoJAX currently builds the same inexpensive isothermal transmission
        # spectrum for both observing geometries. Flip its line contrast for a
        # species-presence emission test; no detailed dayside P-T profile is
        # implied by this proxy.
        template_sign = 1.0 if config.system.observation_type == "eclipse" else -1.0
        raw_template = template_sign * depths
        wide_signal = template_sign * wide_template.transit_depth
        result = run_species(
            species, prepared, wavelength, raw_template, mask, orbit.phase, orbit.berv_kms,
            orbit.signal_weight, rv_grid, kp_grid, vsys_grid,
            config.system.expected_kp_kms, orbit.stellar_rv_kms,
            config.reduction.svd_components, config.search.map_sigma_clip,
            config.search.local_kp_half_width_kms,
            config.search.local_vsys_half_width_kms,
            config.injection.scale, config.injection.random_seed + species_index * 100_000,
            config.injection.null_realizations,
            wide_wavelength_um=wide_template.wavelength_um,
            wide_template=wide_signal,
            velocity_basis=orbit.velocity_basis,
            # A transit's star-only reference is OOT; an eclipse's star-only
            # reference is the in-eclipse spectrum with the planet hidden.
            baseline_mask=(orbit.event_mask if config.system.observation_type == "eclipse"
                           else ~orbit.event_mask),
            show_progress=config.show_progress,
        )
        results.append(result)
        _save_result(result, output, orbit, rv_grid, kp_grid, vsys_grid)
        if config.plots.enabled:
            from decanter.hrccs import plots
            plots.forward_spectrum(
                species, wavelength, template_sign * depths, output,
                config.plots.formats, config.plots.dpi,
                observation_type=config.system.observation_type,
                injection_scale=config.injection.scale,
            )
            plots.svd_sequence(result, prepared, wavelength, orders, orbit.phase, output,
                               config.plots.orders_per_page)
            plots.template_sequence(result, wavelength, orders, orbit.phase, output,
                                    config.plots.orders_per_page)
            plots.component_snr(result, output, config.plots.formats, config.plots.dpi)
            plots.final_four_panel(result, orbit, rv_grid, kp_grid, vsys_grid,
                                   config.system.expected_kp_kms, orbit.stellar_rv_kms,
                                   output, config.plots.formats, config.plots.dpi)
    summary = {
        "schema": "decanter.hrccs.v1", "target": orbit.target_name,
        "configuration": asdict(config), "orders": orders.tolist(),
        "instrument_mode": cube.instmode,
        "template_resolving_power": resolving_power,
        "analysis_method": "notebook_exact_svd_refit",
        "berv_kms": [float(np.nanmin(orbit.berv_kms)), float(np.nanmax(orbit.berv_kms))],
        "stellar_rv_kms": orbit.stellar_rv_kms,
        "observation_type": config.system.observation_type,
        "signal_exposures": ("out_of_eclipse" if config.system.observation_type == "eclipse"
                             else "in_transit"),
        "injection_stellar_baseline_exposures": (
            "in_eclipse" if config.system.observation_type == "eclipse"
            else "out_of_transit"
        ),
        "template_interpretation": (
            "inverted isothermal transmission proxy for emission species presence"
            if config.system.observation_type == "eclipse"
            else "isothermal transmission"
        ),
        "component_selection": {
            "criterion": "maximum map S/N within the local expected-planet window",
            "kp_half_width_kms": config.search.local_kp_half_width_kms,
            "vsys_half_width_kms": config.search.local_vsys_half_width_kms,
            "injection_and_null_rank": "fixed to the observed-data-selected rank",
            "false_alarm_threshold": "global maximum over each null Kp-Vsys map",
        },
        "results": [{"species": item.species,
                     "selected_svd_components": item.selected.count,
                     "observed_expected_snr": item.selected.expected_snr,
                     "observed_local_peak_snr": item.selected.local_peak_snr,
                     "observed_local_peak_kp_kms": item.selected.local_peak_kp_kms,
                     "observed_local_peak_vsys_kms": item.selected.local_peak_vsys_kms,
                     "observed_peak_snr": item.selected.peak_snr,
                     "observed_peak_kp_kms": item.selected.peak_kp_kms,
                     "observed_peak_vsys_kms": item.selected.peak_vsys_kms,
                     "injected_expected_snr": item.injected.expected_snr,
                     "injected_local_peak_snr": item.injected.local_peak_snr,
                     "null_global_peak_snr": item.null_global_peak_snr.tolist(),
                     "null_false_alarm_fraction": item.null_false_alarm_fraction}
                    for item in results],
    }
    (output / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2,
                                                     sort_keys=True))
    return tuple(results)
