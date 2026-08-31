"""Turn a reduction into a wavelength solution.

    solve(series, config) -> WavecalSolution

The shift of every exposure is measured against physical templates at their
laboratory positions, so the result is absolute: it corrects a wavelength-
scale error shared by every exposure, not only the drift between them. The
OH-only modes use only the OH airglow template. The hybrid modes prioritize
telluric absorption, then OH, then cross-order interpolation.

Orders without a trusted direct reference get their value from a robust
low-order fit across orders, constrained only by the direct anchors enabled by
the selected mode. Those cells are flagged ``interpolated``, and ``bracketed``
records whether they lie inside the range the anchors actually span.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from decanter.wavecal.config import WavecalConfig
from decanter.wavecal.measure import (
    continuum_normalize, highpass, measure_series_shifts, robust_scatter,
)
from decanter.wavecal.opacity import series_tau
from decanter.wavecal.solution import WavecalSolution
from decanter.wavecal.telluric import (
    KERNEL_RADIUS, fit_templates, lsf_kernel_numpy,
)


@dataclass
class AtmosphericCommonMode:
    """The pooled telluric+OH CCF behind the common-mode pre-alignment.

    Every line-rich order is correlated against its own template over one
    broad velocity grid; the per-order CCFs are standardised and averaged, and
    the peak of the pooled curve is the exposure's common-mode shift. The
    curves and their peak statistics are kept here for the report.

    Attributes:
        velocity_grid_kms: ``(n_grid,)`` trial velocities, relative to each
            order's own static offset.
        telluric_score / oh_score / joint_score: ``(n_frames, n_grid)`` pooled
            standardised CCFs. The peak is taken from the joint curve; the
            single-tracer curves are kept for comparison.
        peak_velocity_kms: ``(n_frames,)`` joint peak, before the series zero
            point is removed.
        telluric_peak_velocity_kms / oh_peak_velocity_kms: the same peak from
            one tracer alone; their difference checks the joint value.
        peak_snr: joint peak height over the MAD of its own curve.
        peak_score / secondary_score / secondary_separation_kms: the peak, the
            best maximum outside its wings, and how far away that one sits.
        fwhm_kms: full width of the joint peak at half its height.
        bootstrap_sigma_kms / bootstrap_p16_kms / bootstrap_p84_kms: spread of
            the joint peak when the contributing orders are resampled.
        common_velocity_kms: the applied shift, i.e. the joint peak with the
            series zero point removed.
        telluric_orders / oh_orders: the orders each tracer contributed.
    """

    velocity_grid_kms: NDArray
    telluric_score: NDArray
    oh_score: NDArray
    joint_score: NDArray
    peak_velocity_kms: NDArray
    telluric_peak_velocity_kms: NDArray
    oh_peak_velocity_kms: NDArray
    peak_snr: NDArray
    peak_score: NDArray
    secondary_score: NDArray
    secondary_separation_kms: NDArray
    fwhm_kms: NDArray
    bootstrap_sigma_kms: NDArray
    bootstrap_p16_kms: NDArray
    bootstrap_p84_kms: NDArray
    common_velocity_kms: NDArray
    telluric_orders: tuple[int, ...] = ()
    oh_orders: tuple[int, ...] = ()
    search_kms: float = float("nan")
    step_kms: float = float("nan")


@dataclass
class WavecalRun:
    """A solved calibration plus the measurements needed for diagnostics."""

    series: object
    config: WavecalConfig
    solution: WavecalSolution
    telluric_model: object | None
    telluric_support: NDArray
    telluric_velocity: NDArray
    telluric_peak: NDArray
    telluric_accepted: NDArray
    telluric_refit_parameters: NDArray | None
    telluric_information: NDArray
    oh_model: object | None
    oh_velocity: NDArray
    oh_peak: NDArray
    oh_accepted: NDArray
    oh_information: NDArray
    smooth_velocity: NDArray
    _telluric_tau: NDArray | None = None
    atmospheric: AtmosphericCommonMode | None = None


def _normalized(series) -> NDArray:
    width = max(51, int(round(151 * 0.96 / float(np.median(series.dv_pix_kms)))) | 1)
    out = np.empty_like(series.obj, dtype=float)
    for i in range(series.n_frames):
        for j in range(series.n_orders):
            out[i, :, j] = continuum_normalize(series.obj[i, :, j], width)
    return out


def _support(model, series, config) -> NDArray:
    edge = max(20, int(np.ceil(config.edge_trim_resolution_elements
                               * config.resolution_element_kms(series.instmode)
                               / float(np.median(series.dv_pix_kms)))))
    pixel = np.arange(series.n_pixels)
    interior = (pixel >= edge) & (pixel < series.n_pixels - edge)
    support = np.zeros((series.n_pixels, series.n_orders), dtype=bool)
    for j in range(series.n_orders):
        column = model.native_template[:, j]
        support[:, j] = interior & np.isfinite(column) & (column < 0.997)
    return support


def _refit_one(tau_order, flux, parameters, family, n_species, n_pixels,
               shift_initial, bounds_shift, *, return_parameters=False):
    """Refit column scales, continuum and shift for one exposure and order."""
    pixel = np.arange(n_pixels, dtype=float)
    x = np.linspace(-1.0, 1.0, n_pixels)
    kernel = lsf_kernel_numpy(family, float(parameters[n_species + 1]),
                              float(parameters[n_species + 2]),
                              float(parameters[n_species + 3]))
    ok = np.isfinite(flux) & (flux > 0.25) & (flux < 1.30)
    if int(ok.sum()) < 200:
        return None if return_parameters else np.nan

    def model(p):
        total = np.zeros(n_pixels)
        for s in range(n_species):
            column = tau_order[s]
            total += p[s] * np.clip(np.interp(pixel - p[n_species], pixel, column,
                                              left=column[0], right=column[-1]), 0.0, None)
        conv = np.convolve(np.pad(np.exp(-total), KERNEL_RADIUS, mode="edge"),
                           kernel, mode="valid")
        c0, c1, c2 = p[n_species + 1:]
        return conv * np.clip(1.0 + c0 + c1 * x + c2 * (2.0 * x**2 - 1.0), 0.2, 2.5)

    p0 = np.concatenate([np.clip(parameters[:n_species], 1e-3, 29.9),
                         [float(np.clip(shift_initial, -bounds_shift * 0.99,
                                        bounds_shift * 0.99))],
                         np.clip(parameters[n_species + 4:], -0.34, 0.34)])
    low = np.array([0.0] * n_species + [-bounds_shift, -0.35, -0.35, -0.35])
    high = np.array([30.0] * n_species + [bounds_shift, 0.35, 0.35, 0.35])
    try:
        result = least_squares(lambda p: (flux - model(p))[ok], p0, bounds=(low, high),
                               loss="cauchy", f_scale=2.0, max_nfev=200)
    except Exception:                                  # noqa: BLE001
        return None if return_parameters else np.nan
    if return_parameters:
        return np.asarray(result.x, dtype=float)
    return float(result.x[n_species])


def robust_order_polyfit(order_values, values, weights, degree=2):
    """Weighted, sigma-clipped polynomial of ``values`` against order number."""
    x = np.asarray(order_values, dtype=float)
    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if good.sum() < 2:
        return np.full(x.size, np.nan), good
    local_degree = int(min(degree, good.sum() - 1))
    x0 = 0.5 * (np.nanmin(x) + np.nanmax(x))
    scale = max(0.5 * (np.nanmax(x) - np.nanmin(x)), 1.0)
    xs = (x - x0) / scale
    keep = good.copy()
    coefficients = None
    for _ in range(4):
        coefficients = np.polyfit(xs[keep], y[keep], local_degree, w=np.sqrt(w[keep]))
        residual = y - np.polyval(coefficients, xs)
        sigma = robust_scatter(residual[keep])
        if not np.isfinite(sigma) or sigma == 0:
            break
        new_keep = good & (np.abs(residual - np.nanmedian(residual[keep])) < 3.5 * sigma)
        if new_keep.sum() < local_degree + 1 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return np.polyval(coefficients, xs), keep


def assemble_hybrid_ladder(
    order_values,
    telluric_velocity,
    telluric_accepted,
    telluric_weight,
    oh_velocity,
    oh_accepted,
    oh_weight,
    *,
    degree=2,
    return_smooth=False,
):
    """Assemble the direct-reference hierarchy into a wavelength solution.

    Priority:

    1. accepted telluric measurements are used literally;
    2. accepted OH measurements are used literally only where tellurics were
       not accepted;
    3. every remaining order receives the smooth cross-order interpolation.

    The interpolation is constrained only by the direct references from steps
    1 and 2, so an interpolated value never becomes an anchor itself.
    """
    orders = np.asarray(order_values, dtype=float)
    telluric_velocity = np.asarray(telluric_velocity, dtype=float)
    oh_velocity = np.asarray(oh_velocity, dtype=float)
    telluric_accepted = np.asarray(telluric_accepted, dtype=bool)
    oh_accepted = np.asarray(oh_accepted, dtype=bool)
    telluric_weight = np.asarray(telluric_weight, dtype=float)
    oh_weight = np.asarray(oh_weight, dtype=float)

    expected = telluric_velocity.shape
    for name, array in (
        ("oh_velocity", oh_velocity),
        ("telluric_accepted", telluric_accepted),
        ("oh_accepted", oh_accepted),
        ("telluric_weight", telluric_weight),
        ("oh_weight", oh_weight),
    ):
        if array.shape != expected:
            raise ValueError(f"{name} has shape {array.shape}, expected {expected}")
    if telluric_velocity.ndim != 2 or orders.shape != (expected[1],):
        raise ValueError("velocity arrays must be (frames, orders) and match order_values")

    telluric = telluric_accepted & np.isfinite(telluric_velocity)
    oh_only = oh_accepted & np.isfinite(oh_velocity) & ~telluric
    out = np.full_like(telluric_velocity, np.nan)
    source = np.full(expected, "unavailable", dtype="U16")
    bracketed = np.zeros(expected, dtype=bool)
    smooth_out = np.full_like(telluric_velocity, np.nan)

    for i in range(expected[0]):
        anchors = telluric[i] | oh_only[i]
        if np.count_nonzero(anchors) < 2:
            continue
        anchor_value = np.where(
            telluric[i], telluric_velocity[i], np.where(oh_only[i], oh_velocity[i], np.nan)
        )
        anchor_weight = np.where(
            telluric[i], telluric_weight[i], np.where(oh_only[i], oh_weight[i], np.nan)
        )
        smooth, _ = robust_order_polyfit(
            orders,
            np.where(anchors, anchor_value, np.nan),
            np.where(anchors, anchor_weight, np.nan),
            degree=degree,
        )
        smooth_out[i] = smooth
        out[i] = smooth
        source[i] = "interpolated"
        # Write OH first and tellurics last so tellurics have highest priority.
        out[i, oh_only[i]] = oh_velocity[i, oh_only[i]]
        source[i, oh_only[i]] = "OH"
        out[i, telluric[i]] = telluric_velocity[i, telluric[i]]
        source[i, telluric[i]] = "telluric"
        low, high = orders[anchors].min(), orders[anchors].max()
        bracketed[i] = (orders >= low) & (orders <= high)

    source[~np.isfinite(out)] = "unavailable"
    if return_smooth:
        return out, source, bracketed, smooth_out
    return out, source, bracketed


def measure_oh(series, config, *, verbose=True):
    """Fit the OH template and measure every sky frame against it.

    Returns ``(model, velocity, peak)``. The velocity is referenced to the OH
    laboratory positions, so like the telluric one it is absolute.
    """
    from decanter.wavecal import airglow

    model = airglow.fit_templates(series, config, verbose=verbose)
    width = 101
    signal = np.empty_like(series.sky, dtype=float)
    for i in range(series.n_frames):
        for j in range(series.n_orders):
            signal[i, :, j] = highpass(series.sky[i, :, j], width)
    template = np.empty_like(model.native_template)
    for j in range(series.n_orders):
        template[:, j] = highpass(model.native_template[:, j], width)
    # The static fit already measures the order's approximate laboratory
    # offset. Search around that value rather than around zero: otherwise a
    # large static offset plus the nightly drift can push half the series onto
    # the edge of a symmetric zero-centred search.
    center_kms = model.parameters[:, 1] * series.dv_pix_kms
    search_support = np.zeros_like(model.support)
    for j in range(series.n_orders):
        pad = int(np.ceil((abs(center_kms[j]) + config.oh_shift_search_kms)
                          / series.dv_pix_kms[j]))
        kernel = np.ones(2 * pad + 1, dtype=int)
        search_support[:, j] = np.convolve(model.support[:, j].astype(int), kernel,
                                            mode="same") > 0
    velocity, peak = measure_series_shifts(
        signal, template, search_support, series.dv_pix_kms,
        search_kms=config.oh_shift_search_kms, center_kms=center_kms,
    )

    if config.per_exposure_refit and model.band_templates is not None:
        if verbose:
            print("    [OH] per-exposure band amplitudes and shift", flush=True)
        rich = model.rich_orders(config.oh_rich_min_lines)
        scales = np.asarray(model.meta.get("scale", np.ones(series.n_orders)), dtype=float)
        for j in np.where(rich)[0]:
            for i in range(series.n_frames):
                if (not np.isfinite(velocity[i, j])
                        or not np.isfinite(peak[i, j])
                        or peak[i, j] < config.oh_peak_threshold):
                    continue
                shift = _refit_oh_one(
                    model.band_templates[:, :, j],
                    signal[i, :, j] / max(scales[j], 1e-30),
                    search_support[:, j],
                    model.parameters[j],
                    velocity[i, j] / series.dv_pix_kms[j],
                    config.oh_refit_window_kms / series.dv_pix_kms[j],
                )
                if np.isfinite(shift):
                    velocity[i, j] = shift * series.dv_pix_kms[j]
    return model, velocity, peak


def _refit_oh_one(components, signal, support, parameters, shift_initial,
                  shift_half_width):
    """Refit physical OH band amplitudes and shift for one sky exposure."""
    components = np.asarray(components, dtype=float)
    signal = np.asarray(signal, dtype=float)
    support = np.asarray(support, dtype=bool)
    pixel = np.arange(signal.size, dtype=float)
    strength = np.sqrt(np.nansum(components[:, support] ** 2, axis=1))
    active = np.isfinite(strength) & (strength > max(np.nanmax(strength) * 1e-3, 1e-10))
    components = components[active]
    good = support & np.isfinite(signal)
    if components.size == 0 or np.count_nonzero(good) < 20:
        return np.nan

    amplitude0 = max(float(np.exp(parameters[0])), 1e-3)
    p0 = np.concatenate([[shift_initial], np.full(components.shape[0], amplitude0), [0.0]])
    low = np.concatenate([[shift_initial - shift_half_width],
                          np.zeros(components.shape[0]), [-0.5]])
    high = np.concatenate([[shift_initial + shift_half_width],
                           np.full(components.shape[0], max(10.0 * amplitude0, 2.0)), [0.5]])

    def residual(p):
        shifted = np.asarray([
            np.interp(pixel - p[0], pixel, component, left=0.0, right=0.0)
            for component in components
        ])
        model = p[1:-1] @ shifted + p[-1]
        return (signal - model)[good]

    try:
        result = least_squares(residual, p0, bounds=(low, high), loss="soft_l1",
                               f_scale=0.03, max_nfev=120)
    except Exception:  # noqa: BLE001
        return np.nan
    return float(result.x[0]) if result.success else np.nan


def _common_mode(velocity, accepted):
    """Median over the accepted orders of each exposure."""
    out = np.full(velocity.shape[0], np.nan)
    for i in range(velocity.shape[0]):
        if np.count_nonzero(accepted[i]) >= 2:
            out[i] = np.nanmedian(velocity[i, accepted[i]])
    return out


def _to_object_epoch(oh_velocity, series, telluric_common):
    """Move an OH measurement from the sky frame's epoch to the object frame's.

    The sky spectrum does not carry Decanter's first-layer WARP alignment,
    but the common reference grid is defined by the shifted object spectra.
    Where the provenance is available, the paired sky exposure's recorded WARP
    shift is subtracted, converted to velocity on each order's
    native-equivalent grid, which removes the alignment term before OH anchors
    the object solution.

    A Series without OBJFRAME, SKYFRAME and WAVSHIFT metadata falls back to
    interpolating the telluric drift curve between the two epochs.
    """
    result = np.asarray(oh_velocity, dtype=float).copy()
    metadata = getattr(series, "meta", None)
    exact = np.zeros(series.n_frames, dtype=bool)
    if metadata is not None and len(metadata) == series.n_frames:
        shift_by_frame = {}
        for frame_id, row in zip(series.frame_ids, metadata, strict=True):
            object_frame = str(row.get("OBJFRAME", frame_id)).strip()
            value = row.get("WAVSHIFT", row.get("WAVESHIFT", np.nan))
            try:
                shift = float(value)
            except (TypeError, ValueError):
                shift = np.nan
            if object_frame and np.isfinite(shift):
                shift_by_frame[object_frame] = shift
        for index, row in enumerate(metadata):
            sky_frame = str(row.get("SKYFRAME", "")).strip()
            sky_shift = shift_by_frame.get(sky_frame, np.nan)
            if np.isfinite(sky_shift):
                result[index] -= sky_shift * np.asarray(series.dv_pix_kms, dtype=float)
                exact[index] = True

    fallback = ~exact
    good = np.isfinite(series.time_jd) & np.isfinite(telluric_common)
    if np.any(fallback) and np.count_nonzero(good) >= 2:
        order = np.argsort(series.time_jd[good])
        times = series.time_jd[good][order]
        drift = telluric_common[good][order]
        at_object = np.interp(series.time_jd, times, drift)
        at_sky = np.interp(series.sky_time_jd, times, drift)
        correction = np.where(np.isfinite(series.sky_time_jd), at_object - at_sky, 0.0)
        result[fallback] += correction[fallback, None]
    return result


def solve(series, config: WavecalConfig | None = None, *, verbose: bool = True,
          diagnostic_pdf: str | None = None,
          return_diagnostics: bool = False) -> WavecalSolution | WavecalRun:
    """Measure the wavelength solution for a whole series.

    Args:
        diagnostic_pdf: if given, write a long per-order PDF of the template
            fit, zoomed so individual lines are resolved.
        return_diagnostics: return a :class:`WavecalRun` containing the fitted
            templates, direct measurements, acceptance masks, smooth hybrid
            interpolation and final solution. The default remains the public
            :class:`WavecalSolution` return value.
    """
    config = (config or WavecalConfig()).resolved_for(series.instmode)
    try:
        Path(config.linelist_dir).mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    except OSError:
        # Batch and container environments can present an unwritable home
        # cache. Fall back to a process-local directory so the run still needs
        # no user-staged data paths.
        fallback = Path(tempfile.gettempdir()) / "decanter-wavecal"
        config = replace(
            config,
            linelist_dir=str(fallback / "hitran"),
            cache_dir=str(fallback / "opacity"),
        )
        Path(config.linelist_dir).mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    runtime_root = Path(config.cache_dir).parent / "runtime"
    for name in ("numba", "matplotlib", "xdg"):
        (runtime_root / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(runtime_root / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(runtime_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(runtime_root / "xdg"))
    shape = (series.n_frames, series.n_orders)
    model = None
    tau = None
    support = np.zeros((series.n_pixels, series.n_orders), dtype=bool)
    velocity = np.full(shape, np.nan)
    peak = np.full(shape, np.nan)
    rich = np.zeros(series.n_orders, dtype=bool)
    accepted = np.zeros(shape, dtype=bool)
    refit_parameters = None
    refit_retained = np.zeros(shape, dtype=bool)

    # The OH-only modes build no telluric model at all, so they run with the
    # OH line list alone.
    if config.uses_telluric:
        if verbose:
            print(f"  building ExoJAX optical depths for {list(config.species)}", flush=True)
        tau, _ = series_tau(
            series, tuple(config.species), config.linelist_dir, config.cache_dir
        )

        if verbose:
            print("  fitting the telluric template", flush=True)
        model = fit_templates(series, tau, config, verbose=verbose)

        normalized = _normalized(series)
        support = _support(model, series, config)
        signal = 1.0 - normalized
        feature = 1.0 - model.native_template
        velocity, peak = measure_series_shifts(
            signal, feature, support, series.dv_pix_kms,
            search_kms=config.shift_search_kms,
        )

        if diagnostic_pdf:
            from decanter.wavecal.diagnostics import telluric_template_pdf

            written = telluric_template_pdf(
                series, model, str(diagnostic_pdf).replace(".pdf", "_telluric.pdf"),
                support=support,
                resolution_element_kms=config.resolution_element_kms(series.instmode),
                title=f"telluric template fit - {series.instmode}, "
                      f"{series.n_frames} frames, fsr{series.fsr_cut:g}",
            )
            if verbose:
                print(f"  wrote fit diagnostics: {written}", flush=True)

        rich = model.rich_orders(config.telluric_rms_threshold)
        accepted = (
            rich[None, :] & np.isfinite(velocity)
            & (peak >= config.telluric_peak_threshold)
        )
        if verbose:
            print(f"  telluric-rich orders {int(rich.sum())}/{series.n_orders}; "
                  f"accepted cells {accepted.mean():.3f}", flush=True)

        # Optional per-exposure refit, LSF and line positions frozen.
        if config.per_exposure_refit:
            refit_parameters = np.full(
                (series.n_frames, series.n_orders, tau.shape[0] + 4),
                np.nan,
                dtype=float,
            )
            if verbose:
                print("  per-exposure refit of column scales, continuum and shift", flush=True)
            n_species = tau.shape[0]
            for j in np.where(rich)[0]:
                bound = config.shift_search_kms / series.dv_pix_kms[j]
                for i in range(series.n_frames):
                    if not accepted[i, j]:
                        continue
                    fitted = _refit_one(
                        tau[:, :, j], normalized[i, :, j], model.parameters[j],
                        str(model.family[j]), n_species, series.n_pixels,
                        velocity[i, j] / series.dv_pix_kms[j], bound,
                        return_parameters=True,
                    )
                    if fitted is not None and np.all(np.isfinite(fitted)):
                        refit_parameters[i, j] = fitted
                        candidate = fitted[n_species] * series.dv_pix_kms[j]
                        seed = velocity[i, j]
                        at_bound = abs(candidate) >= 0.99 * config.shift_search_kms
                        closes = abs(candidate - seed) <= config.telluric_refit_closure_kms
                        if closes and not at_bound:
                            velocity[i, j] = candidate
                            refit_retained[i, j] = True
            if verbose:
                attempted = int(np.count_nonzero(accepted))
                retained = int(np.count_nonzero(refit_retained))
                print(
                    f"  telluric refit closure: retained {retained}/{attempted}; "
                    f"direct CCF seed retained for {attempted - retained}",
                    flush=True,
                )
    elif verbose:
        print("  OH-only mode: telluric fitting and anchors disabled", flush=True)

    # --- rung 2: OH airglow, for the orders tellurics cannot anchor --------
    oh_model = None
    oh_velocity = np.full_like(velocity, np.nan)
    oh_peak = np.full_like(velocity, np.nan)
    oh_accepted = np.zeros(velocity.shape, dtype=bool)
    oh_tie = float("nan")
    telluric_common = _common_mode(velocity, accepted)

    if series.sky is not None:
        if verbose:
            print("  fitting the OH airglow template", flush=True)
        oh_model, oh_raw, oh_peak = measure_oh(series, config, verbose=verbose)
        if diagnostic_pdf:
            from decanter.wavecal import airglow as _airglow
            from decanter.wavecal.diagnostics import airglow_template_pdf
            from decanter.wavecal.opacity import linelist_path

            oh_path = str(diagnostic_pdf).replace(".pdf", "_OH.pdf")
            lines = _airglow.load_oh_lines(linelist_path("OH", config.linelist_dir),
                                           float(np.nanmin(series.wave)),
                                           float(np.nanmax(series.wave)))
            written = airglow_template_pdf(
                series, oh_model, oh_path, line_wave_angstrom=lines["wave_angstrom"],
                resolution_element_kms=config.resolution_element_kms(series.instmode),
                title=f"OH airglow fit - {series.instmode}, {series.n_frames} frames")
            if verbose:
                print(f"  wrote OH fit diagnostics: {written}", flush=True)
        oh_velocity = _to_object_epoch(oh_raw, series, telluric_common)
        oh_rich = oh_model.rich_orders(config.oh_rich_min_lines)
        oh_accepted = (oh_rich[None, :] & np.isfinite(oh_velocity)
                       & (oh_peak >= config.oh_peak_threshold))

        # Bring OH onto the telluric scale with one global constant. Only a
        # handful of orders carry both references, too few to measure a
        # per-order tie against their scatter.
        if config.uses_telluric and config.oh_tie == "global_constant":
            overlap = accepted & oh_accepted
            if np.count_nonzero(overlap) >= 5:
                oh_tie = float(np.nanmedian((velocity - oh_velocity)[overlap]))
                oh_velocity = oh_velocity + oh_tie
        if verbose:
            tie_text = (
                f"OH->telluric tie {oh_tie * 1e3:+.0f} m/s "
                f"({int(np.count_nonzero(accepted & oh_accepted))} overlap cells)"
                if config.uses_telluric else
                "OH-only absolute laboratory frame; telluric tie disabled"
            )
            print(f"  OH-rich orders {int(oh_rich.sum())}/{series.n_orders}; "
                  f"accepted cells {oh_accepted.mean():.3f}; {tie_text}",
                  flush=True)
    elif verbose:
        print("  no sky spectra in this reduction: telluric anchors only", flush=True)

    # --- zero point -------------------------------------------------------
    if config.zero_point == "relative":
        with np.errstate(invalid="ignore"):
            velocity = velocity - np.nanmedian(np.where(accepted, velocity, np.nan),
                                               axis=0, keepdims=True)
            oh_velocity = oh_velocity - np.nanmedian(
                np.where(oh_accepted, oh_velocity, np.nan), axis=0, keepdims=True)

    # --- assemble: the hybrid ladder --------------------------------------
    #   1. strong telluric  -> the measured telluric shift, used literally
    #   2. else strong OH   -> the measured OH shift, tied to the telluric scale
    #   3. else             -> a smooth fit across orders, constrained ONLY by
    #                          the trusted anchors from rungs 1 and 2
    telluric_information = np.zeros(shape, dtype=float)
    if model is not None:
        telluric_information = (
            np.clip(model.template_rms[None, :] / config.telluric_rms_threshold, 1.0, 4.0)
            * np.maximum(peak, 0.05)) ** 2
    oh_information = np.zeros_like(telluric_information)
    if oh_model is not None:
        oh_information = (
            np.clip(oh_model.line_count[None, :] / max(config.oh_rich_min_lines, 1), 1.0, 3.0)
            * np.maximum(oh_peak, 0.05)) ** 2

    out, source, bracketed, smooth = assemble_hybrid_ladder(
        series.orders,
        velocity,
        accepted,
        telluric_information,
        oh_velocity,
        oh_accepted,
        oh_information,
        degree=2,
        return_smooth=True,
    )

    if verbose:
        finite = np.isfinite(out)
        direct_velocity = velocity if config.uses_telluric else oh_velocity
        direct_accepted = accepted if config.uses_telluric else oh_accepted
        common = np.nanmedian(
            np.where(direct_accepted, direct_velocity, np.nan), axis=1
        )
        print(f"  solution: {finite.mean():.3f} of cells filled; "
              f"absolute offset median {np.nanmedian(common) * 1e3:+.0f} m/s, "
              f"drift rms {np.nanstd(common) * 1e3:.0f} m/s", flush=True)

    solution = WavecalSolution(
        frame_ids=series.frame_ids, orders=series.orders, velocity=out,
        source=source, bracketed=bracketed, mode=config.mode,
        zero_point=config.zero_point, assembly=config.assembly, oh_tie_kms=oh_tie,
        meta={"species": list(config.species), "instmode": series.instmode,
              "reference_path": ("telluric+OH" if config.uses_telluric else "OH-only"),
              "telluric_rich": [int(o) for o, r in zip(series.orders, rich) if r],
              "lsf_fwhm_kms": (
                  model.lsf_fwhm_kms.tolist() if model is not None else []
              ),
              "template_rms": (
                  model.template_rms.tolist() if model is not None else []
              ),
              "accepted_fraction": float(accepted.mean()),
              "oh_accepted_fraction": float(oh_accepted.mean()),
              "telluric_refit_retained_fraction": (
                  float(np.count_nonzero(refit_retained) / np.count_nonzero(accepted))
                  if np.count_nonzero(accepted) else float("nan")
              ),
              "telluric_refit_closure_kms": config.telluric_refit_closure_kms,
              "oh_rich": ([int(o) for o, r in zip(series.orders,
                                                  oh_model.rich_orders(config.oh_rich_min_lines))
                           if r] if oh_model is not None else []),
              "oh_tie_kms": oh_tie,
              "oh_rotational_temperature_k": (oh_model.rotational_temperature_k
                                              if oh_model is not None else float("nan"))},
    )
    if return_diagnostics:
        return WavecalRun(
            series=series,
            config=config,
            solution=solution,
            telluric_model=model,
            telluric_support=support,
            telluric_velocity=velocity,
            telluric_peak=peak,
            telluric_accepted=accepted,
            telluric_refit_parameters=refit_parameters,
            telluric_information=telluric_information,
            oh_model=oh_model,
            oh_velocity=oh_velocity,
            oh_peak=oh_peak,
            oh_accepted=oh_accepted,
            oh_information=oh_information,
            smooth_velocity=smooth,
            _telluric_tau=tau,
        )
    return solution
