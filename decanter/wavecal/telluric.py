"""Fit the ExoJAX telluric template to the data, then measure every exposure.

The template model for one order is

    T(x) = C(x) * [ exp(-sum_m s_m tau_m(x - delta)) (*) K(x) ]

with free column scales ``s_m``, a shift ``delta``, an instrument profile ``K``
drawn from seven families, and a quadratic continuum ``C``. All orders are
fitted at once with a vmapped Adam run per family and the family is chosen per
order by BIC.

The LSF width is bounded around the nominal resolving power and the continuum
is fitted alongside it. Without both, the optimiser on a cool star widens the
template and absorbs the star's own molecular bands as telluric opacity: on an
M8 that gives an implied R of 24,000 and 18 of 26 orders flagged telluric-rich.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from decanter.wavecal.measure import continuum_normalize, robust_scatter

LSF_FAMILIES: tuple[str, ...] = (
    "gaussian", "pseudo_voigt", "moffat", "super_gaussian",
    "double_gaussian", "skew_gaussian", "gauss_hermite",
)
SHAPE_PARAMETERS = {"gaussian": 0, "pseudo_voigt": 1, "moffat": 1, "super_gaussian": 1,
                    "double_gaussian": 2, "skew_gaussian": 1, "gauss_hermite": 2}
KERNEL_RADIUS = 30
ADAM_STEPS = 700
ADAM_LEARNING_RATE = 0.025


class FitUnavailable(RuntimeError):
    """JAX / optax are not installed."""


def _jax():
    try:
        import jax
        import jax.numpy as jnp
        import optax
        from jax.scipy.special import erf
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise FitUnavailable(
            "the wavecal template fit needs the 'wavecal' extra: "
            "pip install 'decanter[wavecal]'"
        ) from exc
    jax.config.update("jax_enable_x64", False)
    return jax, jnp, optax, erf


@dataclass
class TelluricModel:
    """Per-order telluric templates and the parameters behind them."""

    orders: tuple[int, ...]
    family: NDArray                # (n_orders,) str
    parameters: NDArray            # (n_orders, n_species + 7)
    native_template: NDArray       # (n_pixels, n_orders) transmission at zero shift
    fitted_template: NDArray       # (n_pixels, n_orders) including shift + continuum
    template_rms: NDArray          # (n_orders,) RMS of the native transmission
    residual_rms: NDArray          # (n_orders,)
    lsf_fwhm_kms: NDArray          # (n_orders,)
    target: NDArray | None = None  # (n_orders, n_pixels) the spectrum that was fitted
    valid: NDArray | None = None   # (n_orders, n_pixels) pixels in the likelihood
    species: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def rich_orders(self, threshold: float) -> NDArray:
        return self.template_rms >= threshold


def lsf_kernel_numpy(family: str, log_sigma: float, shape_1: float, shape_2: float) -> NDArray:
    from scipy.special import erf as scipy_erf

    x = np.arange(-KERNEL_RADIUS, KERNEL_RADIUS + 1, dtype=float)
    sigma = float(np.exp(log_sigma))
    z = x / sigma
    gaussian = np.exp(-0.5 * z**2)
    if family == "gaussian":
        profile = gaussian
    elif family == "pseudo_voigt":
        f = 0.8 / (1.0 + np.exp(-shape_1))
        profile = (1.0 - f) * gaussian + f / (1.0 + z**2)
    elif family == "moffat":
        profile = (1.0 + z**2) ** -(1.2 + 8.0 / (1.0 + np.exp(-shape_1)))
    elif family == "super_gaussian":
        profile = np.exp(-0.5 * np.abs(z) ** (1.1 + 5.5 / (1.0 + np.exp(-shape_1))))
    elif family == "double_gaussian":
        f = 0.40 / (1.0 + np.exp(-shape_1))
        w = 1.0 + 5.0 / (1.0 + np.exp(-shape_2))
        profile = (1.0 - f) * gaussian + f * np.exp(-0.5 * (x / (sigma * w)) ** 2)
    elif family == "skew_gaussian":
        profile = gaussian * (1.0 + scipy_erf(float(np.clip(shape_1, -6, 6)) * z / np.sqrt(2.0)))
    elif family == "gauss_hermite":
        h3, h4 = 0.35 * np.tanh(shape_1), 0.35 * np.tanh(shape_2)
        p3 = (z**3 - 3.0 * z) / np.sqrt(6.0)
        p4 = (z**4 - 6.0 * z**2 + 3.0) / np.sqrt(24.0)
        profile = np.clip(gaussian * (1.0 + h3 * p3 + h4 * p4), 0.0, None)
    else:
        raise ValueError(f"unknown LSF family {family!r}")
    return profile / profile.sum()


def transmission_numpy(tau_order, parameters, family, n_species, n_pixels,
                       *, shift=None, with_continuum=False):
    """Rebuild one order's transmission outside JAX."""
    scales = np.asarray(parameters[:n_species], dtype=float)
    delta = float(parameters[n_species]) if shift is None else float(shift)
    log_sigma, shape_1, shape_2 = (float(v) for v in parameters[n_species + 1:n_species + 4])
    c0, c1, c2 = (float(v) for v in parameters[n_species + 4:])
    pixel = np.arange(n_pixels, dtype=float)
    total = np.zeros(n_pixels)
    for s in range(n_species):
        column = tau_order[s]
        total += scales[s] * np.clip(
            np.interp(pixel - delta, pixel, column, left=column[0], right=column[-1]), 0.0, None)
    kernel = lsf_kernel_numpy(family, log_sigma, shape_1, shape_2)
    out = np.convolve(np.pad(np.exp(-total), KERNEL_RADIUS, mode="edge"), kernel, mode="valid")
    if with_continuum:
        x = np.linspace(-1.0, 1.0, n_pixels)
        out = out * np.clip(1.0 + c0 + c1 * x + c2 * (2.0 * x**2 - 1.0), 0.2, 2.5)
    return out


def fit_templates(series, tau, config, *, target=None, verbose=True) -> TelluricModel:
    """Adam-fit the telluric template for every order.

    Args:
        series: the loaded :class:`Series`.
        tau: ``(n_species, n_pixels, n_orders)`` from :mod:`decanter.wavecal.opacity`.
        config: :class:`WavecalConfig`.
        target: ``(n_orders, n_pixels)`` continuum-normalised spectrum to fit;
            defaults to the time median of the series.
    """
    jax, jnp, optax, jerf = _jax()
    n_species, n_pixels, n_orders = tau.shape

    if target is None:
        width = max(51, int(round(151 * 0.96 / float(np.median(series.dv_pix_kms)))) | 1)
        normalized = np.empty_like(series.obj, dtype=float)
        for i in range(series.n_frames):
            for j in range(n_orders):
                normalized[i, :, j] = continuum_normalize(series.obj[i, :, j], width)
        target = np.nanmedian(normalized, axis=0).T
    target = np.asarray(target, dtype=float)

    # --- per-order bounds, all set in velocity then converted -------------
    sigma_lo_kms, sigma_hi_kms = config.lsf_sigma_bounds_kms(series.instmode)
    shift_pix = config.shift_search_kms / series.dv_pix_kms
    sigma_lo = np.maximum(sigma_lo_kms / series.dv_pix_kms, 0.35)
    sigma_hi = np.maximum(sigma_hi_kms / series.dv_pix_kms, sigma_lo * 1.5)

    lower = np.column_stack([np.zeros((n_orders, n_species)), -shift_pix,
                             np.log(sigma_lo), np.full(n_orders, -7.0),
                             np.full(n_orders, -7.0), np.full((n_orders, 3), -0.35)])
    upper = np.column_stack([np.full((n_orders, n_species), 30.0), shift_pix,
                             np.log(sigma_hi), np.full(n_orders, 7.0),
                             np.full(n_orders, 7.0), np.full((n_orders, 3), 0.35)])
    start = np.column_stack([np.ones((n_orders, n_species)), np.zeros(n_orders),
                             np.log(np.sqrt(sigma_lo * sigma_hi)),
                             np.zeros((n_orders, 5))])
    fraction = np.clip((start - lower) / (upper - lower), 1e-6, 1 - 1e-6)
    start_unconstrained = np.log(fraction / (1.0 - fraction))

    # --- likelihood weights ----------------------------------------------
    edge = max(20, int(np.ceil(config.edge_trim_resolution_elements
                               * config.resolution_element_kms(series.instmode)
                               / float(np.median(series.dv_pix_kms)))))
    valid = np.zeros((n_orders, n_pixels), dtype=bool)
    weights = np.ones((n_orders, n_pixels))
    noise = np.full(n_orders, 0.02)
    for j in range(n_orders):
        row = target[j]
        ok = np.isfinite(row) & (row > 0.25) & (row < 1.30)
        ok[:edge] = False
        ok[-edge:] = False
        valid[j] = ok
        if not ok.any():
            continue
        noise[j] = max(robust_scatter(row[ok]), 0.004)
        total = tau[:, :, j].sum(axis=0)
        # floor at 0.02, or a line-free order has its noise weighted up
        reference = max(float(np.nanpercentile(total[ok], 95.0)), 0.02)
        weights[j] = 0.25 + 2.75 * np.clip(total / reference, 0.0, 1.0)

    tau_j = jnp.asarray(np.transpose(tau, (2, 0, 1)), dtype=jnp.float32)   # (order, sp, pix)
    data_j = jnp.asarray(np.where(valid, target, 1.0), dtype=jnp.float32)
    valid_j = jnp.asarray(valid)
    weight_j = jnp.asarray(weights, dtype=jnp.float32)
    noise_j = jnp.asarray(noise, dtype=jnp.float32)
    lower_j, upper_j = jnp.asarray(lower, jnp.float32), jnp.asarray(upper, jnp.float32)
    pixel_j = jnp.arange(n_pixels, dtype=jnp.float32)
    xnorm_j = jnp.linspace(-1.0, 1.0, n_pixels, dtype=jnp.float32)

    def kernel_jax(code, log_sigma, s1, s2):
        x = jnp.arange(-KERNEL_RADIUS, KERNEL_RADIUS + 1, dtype=jnp.float32)
        sigma = jnp.exp(log_sigma)
        z = x / sigma
        g = jnp.exp(-0.5 * z**2)
        pv_f = 0.8 * jax.nn.sigmoid(s1)
        dg_f, dg_w = 0.40 * jax.nn.sigmoid(s1), 1.0 + 5.0 * jax.nn.sigmoid(s2)
        h3, h4 = 0.35 * jnp.tanh(s1), 0.35 * jnp.tanh(s2)
        profiles = jnp.stack([
            g,
            (1.0 - pv_f) * g + pv_f / (1.0 + z**2),
            (1.0 + z**2) ** -(1.2 + 8.0 * jax.nn.sigmoid(s1)),
            jnp.exp(-0.5 * jnp.abs(z) ** (1.1 + 5.5 * jax.nn.sigmoid(s1))),
            (1.0 - dg_f) * g + dg_f * jnp.exp(-0.5 * (x / (sigma * dg_w)) ** 2),
            g * (1.0 + jerf(jnp.clip(s1, -6, 6) * z / jnp.sqrt(2.0))),
            jnp.clip(g * (1.0 + h3 * (z**3 - 3 * z) / jnp.sqrt(6.0)
                          + h4 * (z**4 - 6 * z**2 + 3) / jnp.sqrt(24.0)), 0.0),
        ])
        profile = profiles[code]
        return profile / jnp.sum(profile)

    def one_order(tau_order, parameters, code):
        scales = parameters[:n_species]
        delta = parameters[n_species]
        log_sigma, s1, s2 = parameters[n_species + 1], parameters[n_species + 2], parameters[n_species + 3]
        c0, c1, c2 = parameters[n_species + 4], parameters[n_species + 5], parameters[n_species + 6]
        shifted = jax.vmap(lambda t: jnp.interp(pixel_j - delta, pixel_j, t,
                                                left=t[0], right=t[-1]))(tau_order)
        transmission = jnp.exp(-jnp.sum(scales[:, None] * jnp.clip(shifted, 0.0), axis=0))
        conv = jnp.convolve(jnp.pad(transmission, KERNEL_RADIUS, mode="edge"),
                            kernel_jax(code, log_sigma, s1, s2), mode="valid")
        continuum = jnp.clip(1.0 + c0 + c1 * xnorm_j + c2 * (2.0 * xnorm_j**2 - 1.0), 0.2, 2.5)
        return continuum * conv

    model_all = jax.vmap(one_order, in_axes=(0, 0, None))

    def physical(u):
        return lower_j + (upper_j - lower_j) * jax.nn.sigmoid(u)

    def loss(u, code):
        model = model_all(tau_j, physical(u), code)
        residual = (data_j - model) * weight_j / noise_j[:, None]
        return jnp.sum(jnp.where(valid_j, jnp.log1p((residual / 2.0) ** 2), 0.0)) / jnp.sum(valid_j)

    value_and_grad = jax.jit(jax.value_and_grad(loss, argnums=0))

    best_bic = np.full(n_orders, np.inf)
    best_family = np.full(n_orders, "", dtype=object)
    best_parameters = np.full((n_orders, n_species + 7), np.nan)
    best_model = np.full((n_orders, n_pixels), np.nan)

    for family in LSF_FAMILIES:
        code = jnp.asarray(LSF_FAMILIES.index(family), dtype=jnp.int32)
        u = jnp.asarray(start_unconstrained, dtype=jnp.float32)
        optimizer = optax.adam(ADAM_LEARNING_RATE)
        state = optimizer.init(u)
        for _ in range(ADAM_STEPS):
            _, gradient = value_and_grad(u, code)
            updates, state = optimizer.update(gradient, state, u)
            u = optax.apply_updates(u, updates)
        parameters = np.asarray(physical(u))
        models = np.asarray(model_all(tau_j, physical(u), code))
        for j in range(n_orders):
            if not valid[j].any():
                continue
            weighted = (target[j] - models[j]) * weights[j]
            rss = float(np.sum(weighted[valid[j]] ** 2))
            n_fit = int(valid[j].sum())
            k = n_species + 5 + SHAPE_PARAMETERS[family]
            bic = n_fit * np.log(max(rss / n_fit, 1e-20)) + k * np.log(n_fit)
            if bic < best_bic[j]:
                best_bic[j] = bic
                best_family[j] = family
                best_parameters[j] = parameters[j]
                best_model[j] = models[j]
        if verbose:
            print(f"    [telluric] {family:16s} loss={float(loss(u, code)):.5f}", flush=True)

    native = np.full((n_pixels, n_orders), np.nan)
    fitted = np.full((n_pixels, n_orders), np.nan)
    fwhm = np.full(n_orders, np.nan)
    residual = np.full(n_orders, np.nan)
    for j in range(n_orders):
        family = str(best_family[j])
        if not family:
            continue
        native[:, j] = transmission_numpy(tau[:, :, j], best_parameters[j], family,
                                          n_species, n_pixels, shift=0.0)
        fitted[:, j] = best_model[j]
        fwhm[j] = 2.354820045 * float(np.exp(best_parameters[j][n_species + 1])) * series.dv_pix_kms[j]
        if valid[j].any():
            residual[j] = float(np.sqrt(np.nanmean((target[j] - best_model[j])[valid[j]] ** 2)))

    jax.clear_caches()
    return TelluricModel(
        orders=series.orders, family=best_family, parameters=best_parameters,
        native_template=native, fitted_template=fitted,
        template_rms=np.nanstd(native, axis=0), residual_rms=residual,
        lsf_fwhm_kms=fwhm, target=target, valid=valid, species=tuple(config.species),
        meta={"edge_pixels": edge, "n_species": n_species},
    )
