"""SVD-filtered Pearson CCFs, Kp--Vsys maps, and injection recovery."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from decanter.hrccs.detrend import SVDPath, svd_path

C_KMS = 299_792.458


@dataclass(frozen=True)
class ComponentResult:
    count: int
    residual_cube: np.ndarray
    filtered_template_cube: np.ndarray
    exposure_ccf: np.ndarray
    raw_map: np.ndarray
    snr_map: np.ndarray
    expected_snr: float
    local_peak_snr: float
    local_peak_kp_kms: float
    local_peak_vsys_kms: float
    peak_snr: float
    peak_kp_kms: float
    peak_vsys_kms: float


@dataclass(frozen=True)
class SpeciesResult:
    species: str
    raw_templates: np.ndarray
    planet_model_cube: np.ndarray
    mask: np.ndarray
    svd_paths: tuple[SVDPath, ...]
    component_results: tuple[ComponentResult, ...]
    selected: ComponentResult
    injected: ComponentResult
    null_mean_map: np.ndarray
    null_snr_at_expected: np.ndarray
    null_global_peak_snr: np.ndarray
    null_false_alarm_fraction: float


def relativistic_factor(velocity_kms):
    beta = np.asarray(velocity_kms, dtype=float) / C_KMS
    return np.sqrt((1.0 + beta) / (1.0 - beta))


def shift_template(wave_um, template, velocity_kms):
    sample = np.asarray(wave_um) / relativistic_factor(velocity_kms)
    return np.interp(sample, wave_um, template, left=np.nan, right=np.nan)


def planet_model_cube(wavelength_um, templates, phase, berv_kms, kp_kms, vsys_kms,
                      transit_weight, scale=1.0, *, wide_wavelength_um=None,
                      wide_template=None, velocity_basis):
    n_frames = phase.size
    out = np.full((n_frames,) + templates.shape, np.nan)
    basis = np.asarray(velocity_basis, dtype=float)
    velocity = kp_kms * basis + vsys_kms - berv_kms
    for i in range(n_frames):
        for j in range(templates.shape[0]):
            if wide_wavelength_um is None:
                shifted = shift_template(wavelength_um[j], templates[j], velocity[i])
            else:
                sample = wavelength_um[j] / relativistic_factor(velocity[i])
                shifted = np.interp(
                    sample, wide_wavelength_um, wide_template, left=np.nan, right=np.nan
                )
            out[i, j] = scale * transit_weight[i] * shifted
    return out


def fixed_pearson_ccf(data, model, wave, rv_grid, fixed_mask):
    output = np.full((data.shape[0], rv_grid.size), np.nan)
    for i in range(data.shape[0]):
        mask = fixed_mask & np.isfinite(data[i]) & np.isfinite(model[i])
        if np.count_nonzero(mask) < 3:
            continue
        bank = np.asarray([shift_template(wave, model[i], velocity) for velocity in rv_grid])
        mask &= np.all(np.isfinite(bank), axis=0)
        if np.count_nonzero(mask) < 3:
            continue
        x = data[i, mask] - np.mean(data[i, mask])
        y = bank[:, mask] - np.mean(bank[:, mask], axis=1, keepdims=True)
        denom = np.sqrt(np.sum(x * x) * np.sum(y * y, axis=1))
        output[i] = np.divide(y @ x, denom, out=np.full(rv_grid.size, np.nan), where=denom > 0)
    return output


def combine_order_ccfs(order_ccf):
    """Sum order CCFs with the reference notebook's equal-order convention."""
    selected = np.asarray(order_ccf, dtype=float)
    combined = np.nansum(selected, axis=0)
    return np.where(np.any(np.isfinite(selected), axis=0), combined, np.nan)


def kp_vsys_map(exposure_ccf, rv_grid, phase, transit_weight, kp_grid, vsys_grid,
                expected_kp, expected_vsys, velocity_basis):
    use = (transit_weight > 0) & np.any(np.isfinite(exposure_ccf), axis=1)
    result = np.full((kp_grid.size, vsys_grid.size), np.nan)
    sine = np.asarray(velocity_basis, dtype=float)[use]
    weights = transit_weight[use]
    for row, kp in enumerate(kp_grid):
        orbital = (kp - expected_kp) * sine
        for col, vsys in enumerate(vsys_grid):
            lag = orbital + vsys - expected_vsys
            values = np.asarray([
                np.interp(value, rv_grid, exposure_ccf[index], left=np.nan, right=np.nan)
                for value, index in zip(lag, np.where(use)[0])
            ])
            finite = np.isfinite(values)
            if np.any(finite):
                result[row, col] = np.sum(weights[finite] * values[finite]) / np.sqrt(
                    np.sum(weights[finite] ** 2)
                )
    return result


def standardize_map(raw, sigma=3.0, iterations=10):
    finite = np.isfinite(raw)
    keep = finite.copy()
    for _ in range(iterations):
        values = raw[keep]
        if values.size < 2:
            break
        center, scatter = np.median(values), np.std(values)
        if not np.isfinite(scatter) or scatter <= 0:
            break
        updated = finite & (np.abs(raw - center) <= sigma * scatter)
        if np.array_equal(updated, keep):
            break
        keep = updated
    center = np.mean(raw[keep]) if np.any(keep) else 0.0
    scatter = np.std(raw[keep]) if np.any(keep) else 1.0
    return (raw - center) / max(float(scatter), 1.0e-12)


def _map_summary(snr_map, kp_grid, vsys_grid, expected_kp, expected_vsys,
                 local_kp_half_width, local_vsys_half_width):
    expected = float(snr_map[np.argmin(abs(kp_grid - expected_kp)),
                             np.argmin(abs(vsys_grid - expected_vsys))])
    local_mask = ((np.abs(kp_grid[:, None] - expected_kp) <= local_kp_half_width)
                  & (np.abs(vsys_grid[None, :] - expected_vsys) <= local_vsys_half_width))
    local_values = np.where(local_mask, snr_map, np.nan)
    if np.any(np.isfinite(local_values)):
        local = np.unravel_index(np.nanargmax(local_values), local_values.shape)
        local_result = (float(local_values[local]), float(kp_grid[local[0]]),
                        float(vsys_grid[local[1]]))
    else:
        local_result = (np.nan, np.nan, np.nan)
    if np.any(np.isfinite(snr_map)):
        peak = np.unravel_index(np.nanargmax(snr_map), snr_map.shape)
        return (expected, *local_result, float(snr_map[peak]),
                float(kp_grid[peak[0]]), float(vsys_grid[peak[1]]))
    return expected, *local_result, np.nan, np.nan, np.nan


def evaluate(count, residual_cube, filtered_model, wavelength_um, mask, phase,
             transit_weight, rv_grid, kp_grid, vsys_grid, expected_kp, expected_vsys,
             sigma_clip, local_kp_half_width, local_vsys_half_width,
             velocity_basis):
    order_ccf = []
    for order in range(residual_cube.shape[1]):
        order_ccf.append(fixed_pearson_ccf(
            residual_cube[:, order], filtered_model[:, order], wavelength_um[order],
            rv_grid, mask[order],
        ))
    combined = combine_order_ccfs(np.asarray(order_ccf))
    raw = kp_vsys_map(combined, rv_grid, phase, transit_weight, kp_grid, vsys_grid,
                      expected_kp, expected_vsys,
                      velocity_basis=velocity_basis)
    snr = standardize_map(raw, sigma=sigma_clip)
    expected, local_peak, local_kp, local_vsys, peak, peak_kp, peak_vsys = _map_summary(
        snr, kp_grid, vsys_grid, expected_kp, expected_vsys,
        local_kp_half_width, local_vsys_half_width,
    )
    return ComponentResult(count, residual_cube, filtered_model, combined, raw, snr,
                           expected, local_peak, local_kp, local_vsys,
                           peak, peak_kp, peak_vsys)


def _paths(prepared, mask, counts):
    paths = []
    for order in range(prepared.shape[1]):
        svd_mask = np.ones(mask.shape[1], dtype=bool)
        paths.append(svd_path(prepared[:, order], counts, svd_mask))
    return tuple(paths)


def _residual_cube(paths, count):
    return np.stack([path.residuals[min(count, max(path.residuals))] for path in paths], axis=1)


def _filtered_cube(model, paths, count):
    # The moving absolute depth is multiplied into the rank-N
    # stellar/telluric scaling matrix; the injected and uninjected scaling
    # matrices are each refit by an N-component SVD and their residual
    # difference is the CCF template. Not the first-order U U^T projection.
    filtered = []
    for order, path in enumerate(paths):
        scaling = path.lower[min(count, max(path.lower))]
        svd_mask = np.ones(scaling.shape[1], dtype=bool)
        injected = scaling * (1.0 + model[:, order])
        injected_path = svd_path(injected, (count,), svd_mask)
        control_path = svd_path(scaling, (count,), svd_mask)
        filtered.append(
            injected_path.residuals[count] - control_path.residuals[count]
        )
    return np.stack(filtered, axis=1)


def _select_component(components):
    """Select by the finite local maximum near the expected planet location."""
    finite = [item for item in components if np.isfinite(item.local_peak_snr)]
    if not finite:
        raise ValueError("all SVD-component local-maximum S/N values are non-finite")
    return max(finite, key=lambda item: item.local_peak_snr)


def _global_null_fap(observed_local_peak_snr, null_global_peak_snr):
    """Return the finite-sample FAP using each null map's global maximum."""
    null_global_peak_snr = np.asarray(null_global_peak_snr, dtype=float)
    finite = np.isfinite(null_global_peak_snr)
    if not np.isfinite(observed_local_peak_snr) or not np.any(finite):
        return np.nan
    null_global_peak_snr = null_global_peak_snr[finite]
    return float(
        (1 + np.count_nonzero(null_global_peak_snr >= observed_local_peak_snr))
        / (1 + null_global_peak_snr.size)
    )


def run_species(species, prepared, wavelength_um, raw_templates, mask, phase, berv_kms,
                transit_weight, rv_grid, kp_grid, vsys_grid, expected_kp, expected_vsys,
                counts, sigma_clip, local_kp_half_width, local_vsys_half_width,
                injection_scale, seed, null_realizations, *,
                wide_wavelength_um=None, wide_template=None,
                velocity_basis=None, baseline_mask=None, show_progress=True):
    from tqdm.auto import tqdm

    progress = tqdm(
        total=len(counts) + 1 + null_realizations,
        desc=f"{species} CCF/SVD",
        unit="run",
        disable=not show_progress,
        dynamic_ncols=True,
    )
    # The observed-data CCF only needs the line pattern. Keep it at its
    # nominal amplitude so changing the synthetic injection strength cannot
    # alter the observed component selection or observed Kp--Vsys map.
    expected_model = planet_model_cube(
        wavelength_um, raw_templates, phase, berv_kms, expected_kp, expected_vsys,
        transit_weight, scale=1.0,
        wide_wavelength_um=wide_wavelength_um, wide_template=wide_template,
        velocity_basis=velocity_basis,
    )
    injected_model = planet_model_cube(
        wavelength_um, raw_templates, phase, berv_kms, expected_kp, expected_vsys,
        transit_weight, scale=injection_scale,
        wide_wavelength_um=wide_wavelength_um, wide_template=wide_template,
        velocity_basis=velocity_basis,
    )
    paths = _paths(prepared, mask, counts)
    components = []
    for count in counts:
        progress.set_postfix_str(f"observed rank {count}", refresh=False)
        components.append(evaluate(
            count, _residual_cube(paths, count),
            _filtered_cube(expected_model, paths, count),
            wavelength_um, mask, phase, transit_weight, rv_grid, kp_grid, vsys_grid,
            expected_kp, expected_vsys, sigma_clip,
            local_kp_half_width, local_vsys_half_width,
            velocity_basis,
        ))
        progress.update()
    selected = _select_component(components)

    oot = (np.asarray(baseline_mask, dtype=bool)
           if baseline_mask is not None else transit_weight <= 0)
    if np.count_nonzero(oot) < 2:
        raise ValueError("injection recovery needs at least two out-of-event exposures")
    # Fully masked edge pixels are expected; avoid emitting one warning per order.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        oot_center = np.nanmedian(selected.residual_cube[oot], axis=0)
        noise_sigma = 1.4826 * np.nanmedian(
            np.abs(selected.residual_cube[oot] - oot_center[None, :]), axis=0
        )
    for order in range(noise_sigma.shape[0]):
        good = np.isfinite(noise_sigma[order]) & (noise_sigma[order] > 0)
        fill = np.nanmedian(noise_sigma[order, good]) if np.any(good) else 1.0e-3
        noise_sigma[order, ~good] = fill
    baseline = np.nanmedian(prepared[oot], axis=0)
    def synthetic(model, local_seed):
        local_rng = np.random.default_rng(local_seed)
        signal = baseline[None, :, :] * model
        return baseline[None, :, :] + signal + local_rng.normal(
            0.0, noise_sigma[None, :, :], size=prepared.shape
        )

    injected_data = synthetic(injected_model, seed)
    injected_paths = _paths(injected_data, mask, (selected.count,))
    progress.set_postfix_str(f"injection rank {selected.count}", refresh=False)
    injected = evaluate(
        selected.count, _residual_cube(injected_paths, selected.count),
        _filtered_cube(injected_model, injected_paths, selected.count),
        wavelength_um, mask,
        phase, transit_weight, rv_grid, kp_grid, vsys_grid, expected_kp, expected_vsys, sigma_clip,
        local_kp_half_width, local_vsys_half_width,
        velocity_basis,
    )
    progress.update()
    null_maps, null_expected, null_global = [], [], []
    for index in range(null_realizations):
        progress.set_postfix_str(
            f"null {index + 1}/{null_realizations}, rank {selected.count}", refresh=False
        )
        null_data = synthetic(np.zeros_like(expected_model), seed + 1000 + index)
        null_paths = _paths(null_data, mask, (selected.count,))
        null = evaluate(
            selected.count, _residual_cube(null_paths, selected.count),
            _filtered_cube(expected_model, null_paths, selected.count),
            wavelength_um, mask,
            phase, transit_weight, rv_grid, kp_grid, vsys_grid,
            expected_kp, expected_vsys, sigma_clip,
            local_kp_half_width, local_vsys_half_width,
            velocity_basis,
        )
        # Match the injection recovery: every null uses the rank selected from
        # the observed data, without re-tuning on the null realization.
        null_maps.append(null.snr_map)
        null_expected.append(null.expected_snr)
        null_global.append(null.peak_snr)
        progress.update()
    null_expected = np.asarray(null_expected)
    null_global = np.asarray(null_global)
    fap = _global_null_fap(selected.local_peak_snr, null_global)
    progress.set_postfix_str(f"selected rank {selected.count}", refresh=False)
    progress.close()
    return SpeciesResult(species, raw_templates, expected_model, mask, paths,
                         tuple(components), selected, injected,
                         np.nanmean(null_maps, axis=0), null_expected, null_global, fap)
