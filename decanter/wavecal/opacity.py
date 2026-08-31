"""ExoJAX telluric optical depths, per order and per species.

The optical depths come from HITRAN line lists through ExoJAX, so the
templates built on them sit at laboratory wavelengths and the shifts measured
against them are absolute. A template built from the data itself would only
measure how the exposures differ from each other.

ExoJAX, RADIS and JAX are optional dependencies, imported lazily so that
``import decanter`` works without them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Temperature and pressure of the effective absorbing layer. The fit rescales
#: the column per order, so only the shape of tau matters here.
TEMPERATURE_K = 280.0
PRESSURE_BAR = 0.5
#: Resolving power of the internal grid tau is computed on, before the
#: instrument profile is applied. Well above any WINERED mode.
NATIVE_RESOLUTION = 250_000.0
PIXELS_PER_RESOLUTION_ELEMENT = 2.2
#: The column is iterated so the 99th percentile of tau lands here, which keeps
#: every species on a comparable numerical scale for the optimiser.
TARGET_P99_TAU = 0.12
INITIAL_COLUMN_CM2 = 1.0e21


class OpacityUnavailable(RuntimeError):
    """ExoJAX / RADIS / JAX are not installed, or a line list is missing."""


def _imports() -> dict[str, Any]:
    try:
        import jax.numpy as jnp
        from exojax.database.hitran.api import MdbHitran
        from exojax.opacity import OpaPremodit
        from exojax.postproc.specop import SopInstProfile
        from exojax.utils.grids import wavenumber_grid
        from exojax.utils.instfunc import resolution_to_gaussian_std
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise OpacityUnavailable(
            "the wavecal opacity layer needs exojax, radis and jax: "
            "pip install 'decanter[wavecal]'"
        ) from exc
    return {
        "jnp": jnp, "MdbHitran": MdbHitran, "OpaPremodit": OpaPremodit,
        "SopInstProfile": SopInstProfile, "wavenumber_grid": wavenumber_grid,
        "resolution_to_gaussian_std": resolution_to_gaussian_std,
    }


def linelist_path(species: str, linelist_dir: str | Path) -> Path:
    """Locate, or allocate the ExoJAX download path for, a HITRAN table.

    ``MdbHitran`` downloads and parses a missing table at the path it receives.
    Returning the canonical not-yet-existing path is therefore intentional:
    users do not need to stage line lists beside their raw or calibration data.
    """
    root = Path(linelist_dir)
    for candidate in (root / species / f"{species}.hdf5", root / f"{species}.hdf5"):
        if candidate.exists():
            return candidate
    candidate = root / species / f"{species}.hdf5"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def order_tau(wave_angstrom: NDArray, species: str, linelist: str | Path) -> tuple[NDArray, dict]:
    """Optical depth of one species sampled on one order's wavelength grid.

    Returns ``(tau, meta)``. ``tau`` is all zeros when the species has no
    transition inside the order.
    """
    mod = _imports()
    wave_um = np.asarray(wave_angstrom, dtype=float) * 1.0e-4
    nu_min, nu_max = 1.0e4 / float(np.nanmax(wave_um)), 1.0e4 / float(np.nanmin(wave_um))

    n_grid = int(np.ceil(PIXELS_PER_RESOLUTION_ELEMENT * NATIVE_RESOLUTION
                         * np.log(nu_max / nu_min)))
    n_grid = max(256, n_grid + (n_grid % 2))
    nu_opa, _, _ = mod["wavenumber_grid"](nu_min, nu_max, n_grid, xsmode="lpf",
                                          wavelength_order="ascending")
    try:
        mdb = mod["MdbHitran"](str(linelist), nurange=[nu_min, nu_max], isotope=1,
                               gpu_transfer=False, inherit_dataframe=False,
                               crit=1.0e-30, Ttyp=TEMPERATURE_K, engine="vaex")
        n_lines = int(np.size(mdb.nu_lines))
    except ValueError:
        n_lines = 0                       # ExoJAX raises when the range is empty
    if n_lines == 0:
        return np.zeros_like(wave_um), {"species": species, "n_lines": 0}

    opa = mod["OpaPremodit"](mdb, nu_opa, diffmode=0,
                             broadening_resolution={"mode": "manual", "value": 0.2},
                             auto_trange=(240.0, 320.0), allow_32bit=True,
                             wavelength_order="ascending")
    cross_section = np.asarray(opa.xsvector(TEMPERATURE_K, PRESSURE_BAR))
    sop = mod["SopInstProfile"](nu_opa, vrmax=120.0)
    beta = mod["resolution_to_gaussian_std"](NATIVE_RESOLUTION)
    target_nu = 1.0e4 / wave_um[::-1]

    def sample(column: float) -> NDArray:
        highres = np.exp(-column * cross_section)
        convolved = sop.ipgauss(mod["jnp"].asarray(highres), beta)
        return np.asarray(sop.sampling(convolved, 0.0, target_nu))[::-1]

    column = float(INITIAL_COLUMN_CM2)
    for _ in range(4):
        tau = -np.log(np.clip(sample(column), 1.0e-8, 1.0))
        positive = tau[tau > 0]
        if positive.size == 0:
            return np.zeros_like(wave_um), {"species": species, "n_lines": n_lines}
        column *= TARGET_P99_TAU / max(float(np.nanpercentile(positive, 99.0)), 1.0e-20)

    tau = -np.log(np.clip(sample(column), 1.0e-8, 1.0))
    return np.asarray(tau, dtype=float), {
        "species": species, "n_lines": n_lines, "column_cm2": float(column),
        "tau_max": float(np.nanmax(tau)),
    }


def _cache_key(wave_angstrom: NDArray, species: str) -> str:
    digest = hashlib.sha1(np.asarray(wave_angstrom, dtype=np.float64).tobytes()).hexdigest()[:16]
    return f"tau_{species}_{len(wave_angstrom)}_{digest}"


def cached_order_tau(wave_angstrom, species, linelist, cache_dir=None):
    """:func:`order_tau` with an on-disk cache keyed on the exact grid."""
    if not cache_dir:
        return order_tau(wave_angstrom, species, linelist)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{_cache_key(wave_angstrom, species)}.npz"
    if path.exists():
        with np.load(path, allow_pickle=True) as data:
            meta = dict(data["meta"].item())
            meta["from_cache"] = True
            return data["tau"].astype(float), meta
    tau, meta = order_tau(wave_angstrom, species, linelist)
    np.savez_compressed(path, tau=tau, meta=meta)
    meta["from_cache"] = False
    return tau, meta


def series_tau(series, species: tuple[str, ...], linelist_dir, cache_dir=None):
    """Optical depths for every order and species of a :class:`Series`.

    Returns ``(tau, meta)`` with ``tau`` shaped ``(n_species, n_pixels, n_orders)``.
    """
    paths = {name: linelist_path(name, linelist_dir) for name in species}
    tau = np.zeros((len(species), series.n_pixels, series.n_orders), dtype=float)
    meta: list[dict] = []
    for s, name in enumerate(species):
        for j in range(series.n_orders):
            column, info = cached_order_tau(series.wave[:, j], name, paths[name], cache_dir)
            tau[s, :, j] = column
            info["order"] = series.orders[j]
            meta.append(info)
    return tau, meta
