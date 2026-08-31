"""OH airglow emission: the second reference of the hybrid ladder.

Telluric absorption anchors only the orders that carry it. In the Y band that
is a handful of orders, and the rest would fall through to a smooth
interpolation across order number. The night sky emits OH Meinel-band lines
across the whole near-infrared, which covers those gaps.

OH is not in LTE: the emission comes from vibrationally excited OH near 87 km,
so HITRAN's 296 K intensities do not apply and run to
``exp(logsij0) ~ 1e-40``. The line positions, Einstein A coefficients,
upper-state degeneracies and lower-state energies are still correct, so the
line physics is kept and only the level populations are freed:

    I_i  proportional to  N_v' * g_u,i * A_i * exp(-c2 E_rot,u,i / T_rot)

with one population per vibrational level (a softmax, so the overall
normalisation stays in the per-order throughput) and one rotational
temperature shared across every band and order. Two global parameters plus
eight per order, rather than one amplitude per line, so the model cannot put
an emission line on a noise spike and pull the CCF centroid with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from decanter.wavecal.measure import highpass, robust_scatter
from decanter.wavecal.telluric import (
    KERNEL_RADIUS, LSF_FAMILIES, SHAPE_PARAMETERS, _jax, lsf_kernel_numpy,
)

#: OH X2Pi vibrational term values in cm-1, measured from v=0. Each line is
#: assigned an upper vibrational level from its upper-state energy; the levels
#: are ~2500-3500 cm-1 apart and the populated rotational ladder reaches only
#: ~2000 cm-1, so the assignment is unambiguous.
VIBRATIONAL_TERMS = np.array(
    [0.0, 3568.0, 6974.0, 10214.0, 13290.0, 16200.0, 18942.0, 21513.0, 23906.0, 26113.0]
)
MAX_UPPER_LEVEL = 9
MAX_ROTATIONAL_ENERGY_CM = 2600.0
C2_CM_K = 1.438_776_9
TROT_BOUNDS_K = (120.0, 700.0)
ADAM_STEPS = 900
ADAM_LEARNING_RATE = 0.03
DETECTION_SIGMA = 3.0


@dataclass
class AirglowModel:
    """The fitted OH template, per order, plus the global sky parameters."""

    orders: tuple[int, ...]
    family: NDArray                 # (n_orders,) str
    parameters: NDArray             # (n_orders, 8)
    native_template: NDArray        # (n_pixels, n_orders), peak-normalised, zero shift
    fitted_model: NDArray           # (n_orders, n_pixels) including baseline
    support: NDArray                # (n_pixels, n_orders) bool, detected-line windows
    line_count: NDArray             # (n_orders,) individually detected lines
    residual_rms: NDArray           # (n_orders,)
    lsf_fwhm_kms: NDArray           # (n_orders,)
    rotational_temperature_k: float = float("nan")
    band_levels: NDArray | None = None
    band_populations: NDArray | None = None
    band_templates: NDArray | None = None  # (n_bands, n_pixels, n_orders), zero shift
    target: NDArray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def rich_orders(self, minimum_lines: int) -> NDArray:
        return self.line_count >= minimum_lines


def load_oh_lines(linelist, wave_min_angstrom, wave_max_angstrom):
    """OH line positions, A coefficients, upper degeneracies and rotational energies.

    ExoJAX exposes the *lower* state degeneracy, so the upper one is read from
    the same HITRAN table and matched back by exact wavenumber.
    """
    try:
        import h5py
        from exojax.database.hitran.api import MdbHitran
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "the OH branch needs the 'wavecal' extra: pip install 'decanter[wavecal]'"
        ) from exc

    nu_min = 1.0e8 / float(wave_max_angstrom)
    nu_max = 1.0e8 / float(wave_min_angstrom)
    mdb = MdbHitran(str(linelist), nurange=[nu_min, nu_max], isotope=1,
                    gpu_transfer=False, inherit_dataframe=False, crit=0.0, engine="vaex")
    nu = np.asarray(mdb.nu_lines, dtype=float)
    einstein_a = np.asarray(mdb.A, dtype=float)
    elower = np.asarray(mdb.elower, dtype=float)

    with h5py.File(str(linelist), "r") as handle:
        table_wav = handle["table/columns/wav/data"][:]
        table_gp = handle["table/columns/gp/data"][:]
    order = np.argsort(table_wav)
    sorted_wav = table_wav[order]
    index = np.clip(np.searchsorted(sorted_wav, nu), 0, sorted_wav.size - 1)
    if np.max(np.abs(sorted_wav[index] - nu)) > 1.0e-9:
        raise RuntimeError("could not match the ExoJAX OH selection back to HITRAN for gp")
    gp = table_gp[order][index].astype(float)

    eupper = elower + nu
    vupper = np.searchsorted(VIBRATIONAL_TERMS, eupper + 1.0e-6) - 1
    erot = eupper - VIBRATIONAL_TERMS[np.clip(vupper, 0, VIBRATIONAL_TERMS.size - 1)]
    keep = ((vupper >= 1) & (vupper <= MAX_UPPER_LEVEL) & (erot >= 0.0)
            & (erot <= MAX_ROTATIONAL_ENERGY_CM) & (einstein_a > 0.0) & np.isfinite(nu))
    return {
        "wave_angstrom": 1.0e8 / nu[keep],
        "einstein_a": einstein_a[keep],
        "gp": gp[keep],
        "erot": erot[keep],
        "vupper": vupper[keep],
    }


def _order_line_packs(series, lines, edge_pixels):
    """Per-order padded (index, valid, pixel-position) arrays for the line set."""
    wave = lines["wave_angstrom"]
    per_order = []
    for j in range(series.n_orders):
        low = series.wave[edge_pixels, j]
        high = series.wave[series.n_pixels - edge_pixels - 1, j]
        per_order.append(np.where((wave > low) & (wave < high))[0])
    width = int(max(1, max(indices.size for indices in per_order)))

    index = np.zeros((series.n_orders, width), dtype=np.int32)
    valid = np.zeros((series.n_orders, width), dtype=np.float32)
    pixel = np.zeros((series.n_orders, width), dtype=np.float32)
    for j, indices in enumerate(per_order):
        if indices.size == 0:
            continue
        log_low = np.log(series.wave[0, j])
        delta = np.log(series.wave[1, j]) - log_low
        index[j, : indices.size] = indices
        valid[j, : indices.size] = 1.0
        pixel[j, : indices.size] = (np.log(wave[indices]) - log_low) / delta
    return per_order, index, valid, pixel


def emission_numpy(order_index, amplitudes, parameters, family, line_pixel, n_pixels,
                   *, shift=None, normalize=True):
    """Rebuild one order's OH emission outside JAX."""
    if amplitudes.size == 0:
        return np.zeros(n_pixels)
    delta = float(parameters[1]) if shift is None else float(shift)
    log_sigma, shape_1, shape_2 = (float(v) for v in parameters[2:5])
    position = np.clip(line_pixel + delta, 0.0, n_pixels - 2.0)
    low = np.floor(position).astype(int)
    upper_weight = position - low
    comb = np.zeros(n_pixels, dtype=float)
    np.add.at(comb, low, amplitudes * (1.0 - upper_weight))
    np.add.at(comb, low + 1, amplitudes * upper_weight)
    kernel = lsf_kernel_numpy(family, log_sigma, shape_1, shape_2)
    out = np.convolve(np.pad(comb, KERNEL_RADIUS, mode="edge"), kernel, mode="valid")
    if not normalize:
        return out
    peak = out.max()
    return out / peak if peak > 0 else out


def fit_templates(series, config, *, verbose=True) -> AirglowModel:
    """Adam-fit the OH airglow model to the time-median sky spectrum."""
    if series.sky is None:
        raise ValueError("this reduction carries no sky spectra; the OH branch needs them")
    jax, jnp, optax, jerf = _jax()

    from decanter.wavecal.opacity import linelist_path

    edge = max(20, int(np.ceil(config.edge_trim_resolution_elements
                               * config.resolution_element_kms(series.instmode)
                               / float(np.median(series.dv_pix_kms)))))
    lines = load_oh_lines(linelist_path("OH", config.linelist_dir),
                          float(np.nanmin(series.wave)), float(np.nanmax(series.wave)))
    per_order, index_m, valid_m, pixel_m = _order_line_packs(series, lines, edge)
    bands = np.unique(lines["vupper"])
    band_index = np.searchsorted(bands, lines["vupper"])
    n_bands = int(bands.size)
    if verbose:
        print(f"    [OH] {lines['wave_angstrom'].size} lines, Meinel upper levels "
              f"{bands.tolist()}, {sum(p.size for p in per_order)} line-order pairs",
              flush=True)

    # --- target: the time-median sky, scaled so every order is order-unity ---
    median_sky = np.nanmedian(series.sky, axis=0).astype(float)
    n_pixels, n_orders = median_sky.shape
    target = np.zeros((n_orders, n_pixels))
    scale = np.ones(n_orders)
    valid = np.zeros((n_orders, n_pixels), dtype=bool)
    noise = np.full(n_orders, 1.0e-3)
    interior = np.zeros(n_pixels, dtype=bool)
    interior[edge:-edge] = True
    for j in range(n_orders):
        column = median_sky[:, j]
        level = float(np.nanpercentile(column[interior], 99.5))
        scale[j] = level if np.isfinite(level) and level > 0 else 1.0
        target[j] = column / scale[j]
        valid[j] = interior & np.isfinite(target[j])
        noise[j] = max(float(robust_scatter(highpass(column, 101)[interior]) / scale[j]), 1e-4)

    # --- bounds, in velocity where they are physical ------------------------
    sigma_lo_kms, sigma_hi_kms = config.lsf_sigma_bounds_kms(series.instmode)
    shift_bound = config.oh_shift_search_kms / series.dv_pix_kms
    sigma_lo = np.maximum(sigma_lo_kms / series.dv_pix_kms, 0.35)
    sigma_hi = np.maximum(sigma_hi_kms / series.dv_pix_kms, sigma_lo * 1.5)
    lower = np.column_stack([np.full(n_orders, np.log(1e-4)), -shift_bound, np.log(sigma_lo),
                             np.full(n_orders, -7.0), np.full(n_orders, -7.0),
                             np.full(n_orders, -0.5), np.full(n_orders, -0.5),
                             np.full(n_orders, -0.5)])
    upper = np.column_stack([np.full(n_orders, np.log(50.0)), shift_bound, np.log(sigma_hi),
                             np.full(n_orders, 7.0), np.full(n_orders, 7.0),
                             np.full(n_orders, 1.5), np.full(n_orders, 0.5),
                             np.full(n_orders, 0.5)])
    start = np.column_stack([np.full(n_orders, np.log(0.5)), np.zeros(n_orders),
                             np.log(np.sqrt(sigma_lo * sigma_hi)), np.zeros((n_orders, 3)),
                             np.zeros((n_orders, 2))])
    start[:, 5] = 0.05
    fraction = np.clip((start - lower) / (upper - lower), 1e-6, 1 - 1e-6)
    start_u = np.log(fraction / (1.0 - fraction))

    index_j = jnp.asarray(index_m)
    valid_line_j = jnp.asarray(valid_m)
    pixel_line_j = jnp.asarray(pixel_m)
    a_j = jnp.asarray(lines["einstein_a"], dtype=jnp.float32)
    gp_j = jnp.asarray(lines["gp"], dtype=jnp.float32)
    erot_j = jnp.asarray(lines["erot"], dtype=jnp.float32)
    band_j = jnp.asarray(band_index, dtype=jnp.int32)
    target_j = jnp.asarray(np.where(valid, target, 0.0), dtype=jnp.float32)
    valid_j = jnp.asarray(valid)
    noise_j = jnp.asarray(noise, dtype=jnp.float32)
    lower_j, upper_j = jnp.asarray(lower, jnp.float32), jnp.asarray(upper, jnp.float32)
    xnorm_j = jnp.linspace(-1.0, 1.0, n_pixels, dtype=jnp.float32)

    def kernel_jax(code, log_sigma, s1, s2):
        x = jnp.arange(-KERNEL_RADIUS, KERNEL_RADIUS + 1, dtype=jnp.float32)
        sigma = jnp.exp(log_sigma)
        z = x / sigma
        g = jnp.exp(-0.5 * z**2)
        pv = 0.8 * jax.nn.sigmoid(s1)
        dgf, dgw = 0.40 * jax.nn.sigmoid(s1), 1.0 + 5.0 * jax.nn.sigmoid(s2)
        h3, h4 = 0.35 * jnp.tanh(s1), 0.35 * jnp.tanh(s2)
        profiles = jnp.stack([
            g,
            (1.0 - pv) * g + pv / (1.0 + z**2),
            (1.0 + z**2) ** -(1.2 + 8.0 * jax.nn.sigmoid(s1)),
            jnp.exp(-0.5 * jnp.abs(z) ** (1.1 + 5.5 * jax.nn.sigmoid(s1))),
            (1.0 - dgf) * g + dgf * jnp.exp(-0.5 * (x / (sigma * dgw)) ** 2),
            g * (1.0 + jerf(jnp.clip(s1, -6, 6) * z / jnp.sqrt(2.0))),
            jnp.clip(g * (1.0 + h3 * (z**3 - 3 * z) / jnp.sqrt(6.0)
                          + h4 * (z**4 - 6 * z**2 + 3) / jnp.sqrt(24.0)), 0.0),
        ])
        profile = profiles[code]
        return profile / jnp.sum(profile)

    def amplitudes_of(band_logits, trot_u):
        populations = jax.nn.softmax(band_logits)
        trot = TROT_BOUNDS_K[0] + (TROT_BOUNDS_K[1] - TROT_BOUNDS_K[0]) * jax.nn.sigmoid(trot_u)
        return populations[band_j] * gp_j * a_j * jnp.exp(-C2_CM_K * erot_j / trot)

    def one_order(line_index, line_valid, line_pixel, parameters, amplitudes, code):
        amplitude = amplitudes[line_index] * line_valid
        position = jnp.clip(line_pixel + parameters[1], 0.0, float(n_pixels - 2))
        low = jnp.floor(position).astype(jnp.int32)
        weight = position - low
        comb = jnp.zeros(n_pixels, dtype=jnp.float32)
        comb = comb.at[low].add(amplitude * (1.0 - weight))
        comb = comb.at[low + 1].add(amplitude * weight)
        conv = jnp.convolve(jnp.pad(comb, KERNEL_RADIUS, mode="edge"),
                            kernel_jax(code, parameters[2], parameters[3], parameters[4]),
                            mode="valid")
        conv = conv / (jnp.max(conv) + 1.0e-12)
        baseline = parameters[5] + parameters[6] * xnorm_j + parameters[7] * (2.0 * xnorm_j**2 - 1.0)
        return jnp.exp(parameters[0]) * conv + baseline

    model_all = jax.vmap(one_order, in_axes=(0, 0, 0, 0, None, None))

    def physical(u):
        return lower_j + (upper_j - lower_j) * jax.nn.sigmoid(u)

    def loss(state, code):
        amplitudes = amplitudes_of(state["bands"], state["trot"])
        model = model_all(index_j, valid_line_j, pixel_line_j, physical(state["orders"]),
                          amplitudes, code)
        residual = (target_j - model) / noise_j[:, None]
        return jnp.sum(jnp.where(valid_j, jnp.log1p((residual / 4.0) ** 2), 0.0)) / jnp.sum(valid_j)

    value_and_grad = jax.jit(jax.value_and_grad(loss, argnums=0))

    def run(code, steps):
        state = {"bands": jnp.zeros(n_bands, dtype=jnp.float32),
                 "trot": jnp.asarray(0.0, dtype=jnp.float32),
                 "orders": jnp.asarray(start_u, dtype=jnp.float32)}
        optimizer = optax.adam(ADAM_LEARNING_RATE)
        opt_state = optimizer.init(state)
        for _ in range(steps):
            _, gradient = value_and_grad(state, code)
            updates, opt_state = optimizer.update(gradient, opt_state, state)
            state = optax.apply_updates(state, updates)
        amplitudes = np.asarray(amplitudes_of(state["bands"], state["trot"]))
        parameters = np.asarray(physical(state["orders"]))
        models = np.asarray(model_all(index_j, valid_line_j, pixel_line_j,
                                      physical(state["orders"]), amplitudes_of(
                                          state["bands"], state["trot"]), code))
        trot = float(TROT_BOUNDS_K[0] + (TROT_BOUNDS_K[1] - TROT_BOUNDS_K[0])
                     * jax.nn.sigmoid(state["trot"]))
        return amplitudes, parameters, models, trot, np.asarray(jax.nn.softmax(state["bands"]))

    # family scan, then one full-length joint fit with the per-order winners
    best_bic = np.full(n_orders, np.inf)
    best_code = np.zeros(n_orders, dtype=np.int32)
    for family in LSF_FAMILIES:
        code = jnp.asarray(LSF_FAMILIES.index(family), dtype=jnp.int32)
        _, _, models, _, _ = run(code, max(300, ADAM_STEPS // 3))
        for j in range(n_orders):
            if not valid[j].any() or per_order[j].size == 0:
                continue
            rss = float(np.sum((target[j] - models[j])[valid[j]] ** 2))
            n_fit = int(valid[j].sum())
            k = 6 + SHAPE_PARAMETERS[family]
            bic = n_fit * np.log(max(rss / n_fit, 1e-20)) + k * np.log(n_fit)
            if bic < best_bic[j]:
                best_bic[j] = bic
                best_code[j] = LSF_FAMILIES.index(family)
        if verbose:
            print(f"    [OH] {family:16s} scanned", flush=True)

    # The vmapped model takes one family code, so the final fit uses the most
    # frequently selected family; the per-order winners are recorded above.
    final_code = int(np.bincount(best_code, minlength=len(LSF_FAMILIES)).argmax())
    amplitudes, parameters, models, trot, populations = run(
        jnp.asarray(final_code, dtype=jnp.int32), ADAM_STEPS)
    family = np.array([LSF_FAMILIES[final_code]] * n_orders, dtype=object)

    if verbose:
        print(f"    [OH] T_rot = {trot:.0f} K, band populations "
              f"{ {int(v): round(float(p), 3) for v, p in zip(bands, populations)} }",
              flush=True)

    # --- native template, detected-line census, support ---------------------
    native = np.zeros((n_pixels, n_orders))
    band_templates = np.zeros((n_bands, n_pixels, n_orders))
    support = np.zeros((n_pixels, n_orders), dtype=bool)
    line_count = np.zeros(n_orders, dtype=int)
    residual = np.full(n_orders, np.nan)
    fwhm = np.full(n_orders, np.nan)
    for j in range(n_orders):
        indices = per_order[j]
        fwhm[j] = 2.354820045 * float(np.exp(parameters[j][2])) * series.dv_pix_kms[j]
        if valid[j].any():
            residual[j] = float(np.sqrt(np.nanmean((target[j] - models[j])[valid[j]] ** 2)))
        if indices.size == 0:
            continue
        pack = pixel_m[j, : indices.size]
        native[:, j] = emission_numpy(j, amplitudes[indices], parameters[j],
                                      str(family[j]), pack, n_pixels, shift=0.0)
        unnormalised = emission_numpy(j, amplitudes[indices], parameters[j], str(family[j]),
                                      pack, n_pixels, normalize=False)
        total_peak = max(float(unnormalised.max()), 1e-30)
        for b in range(n_bands):
            in_band = band_index[indices] == b
            if not np.any(in_band):
                continue
            band_templates[b, :, j] = emission_numpy(
                j, amplitudes[indices][in_band], parameters[j], str(family[j]),
                pack[in_band], n_pixels, shift=0.0, normalize=False,
            ) / total_peak
        kernel = lsf_kernel_numpy(str(family[j]), float(parameters[j][2]),
                                  float(parameters[j][3]), float(parameters[j][4]))
        line_peak = float(kernel.max())
        height = np.exp(parameters[j][0]) * amplitudes[indices] * line_peak / total_peak
        detected = height > DETECTION_SIGMA * noise[j]
        line_count[j] = int(np.count_nonzero(detected))
        half = max(3, int(np.ceil(3.5 * float(np.exp(parameters[j][2])))))
        for row in np.where(detected)[0]:
            centre = int(round(pack[row]))
            support[max(0, centre - half): min(n_pixels, centre + half + 1), j] = True
        support[:edge, j] = False
        support[-edge:, j] = False

    jax.clear_caches()
    return AirglowModel(
        orders=series.orders, family=family, parameters=parameters,
        native_template=native, fitted_model=models, support=support,
        line_count=line_count, residual_rms=residual, lsf_fwhm_kms=fwhm,
        rotational_temperature_k=trot, band_levels=bands, band_populations=populations,
        band_templates=band_templates,
        target=target, meta={"edge_pixels": edge, "n_lines": int(lines["wave_angstrom"].size),
                             "scale": scale},
    )
