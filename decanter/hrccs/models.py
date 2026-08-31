"""Isothermal equilibrium ExoJAX transmission templates."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types
import urllib.request
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

C_KMS = 299_792.458
G_CGS = 6.67430e-8
M_JUP_G = 1.89813e30
R_JUP_CM = 7.1492e9
R_SUN_CM = 6.957e10
K_B_CGS = 1.380649e-16

DEFAULT_EXOMOL_DATASETS = {
    "FeH": "56Fe-1H/MoLLIST",
    "CrH": "52Cr-1H/MoLLIST",
}
HITRAN_MOLECULES = frozenset({
    "C2H2", "C2H4", "C2H6", "C2N2", "C4H2", "CF4", "CH3Br", "CH3CN",
    "CH3Cl", "CH3F", "CH3I", "CH3OH", "CH4", "CO", "CO2", "COCl2",
    "COF2", "CS", "CS2", "ClO", "ClONO2", "GeH4", "H2", "H2CO", "H2O",
    "H2O2", "H2S", "HBr", "HC3N", "HCN", "HCOOH", "HCl", "HF", "HI",
    "HNO3", "HO2", "HOBr", "HOCl", "N2", "N2O", "NF3", "NH3", "NO",
    "NO+", "NO2", "O", "O2", "O3", "OCS", "OH", "PH3", "SF6", "SO",
    "SO2", "SO3",
})


@dataclass(frozen=True)
class Template:
    species: str
    wavelength_um: np.ndarray
    transit_depth: np.ndarray
    contrast: np.ndarray
    metadata: dict


def _download(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    from tqdm.auto import tqdm

    terminal = sys.__stderr__ or sys.stderr
    progress = tqdm(
        desc=f"Downloading {path.name}", unit="B", unit_scale=True,
        unit_divisor=1024, dynamic_ncols=True, leave=True, file=terminal,
    )
    last_bytes = 0

    def report(block_count, block_size, total_size):
        nonlocal last_bytes
        if total_size > 0 and progress.total is None:
            progress.total = total_size
        downloaded = block_count * block_size
        progress.update(max(0, downloaded - last_bytes))
        last_bytes = downloaded

    try:
        tqdm.write(f"Downloading {url} -> {path}", file=terminal)
        urllib.request.urlretrieve(  # noqa: S310 - fixed scientific archives
            url, temporary, reporthook=report,
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        progress.close()
    temporary.replace(path)
    return path


def _fastchem_files(config) -> tuple[Path, Path]:
    root = Path(config.cache_dir).expanduser() / "fastchem" / "input"
    abundance = (Path(config.fastchem_abundance_file).expanduser()
                 if config.fastchem_abundance_file else root / "element_abundances/asplund_2021.dat")
    logk = (Path(config.fastchem_logk_file).expanduser()
            if config.fastchem_logk_file else root / "logK/logK.dat")
    base = "https://raw.githubusercontent.com/NewStrangeWorlds/FastChem/master/input"
    if not abundance.exists():
        _download(f"{base}/element_abundances/asplund_2021.dat", abundance)
    if not logk.exists():
        _download(f"{base}/logK/logK.dat", logk)
    return abundance, logk


def _equilibrium_profiles_local(config, metallicity_dex: float, pressure_bar: np.ndarray,
                                temperature_k: np.ndarray, species: tuple[str, ...]):
    """Run FastChem locally; called only inside the isolated worker process."""
    import pyfastchem

    abundance_path, logk_path = _fastchem_files(config)
    solver = pyfastchem.FastChem(str(abundance_path), str(logk_path), 1)
    try:
        solver.setVerboseLevel(0)
    except Exception:
        pass
    abundances = np.asarray(solver.getElementAbundances(), dtype=float)
    for index in range(solver.getElementNumber()):
        if solver.getElementSymbol(index) not in ("H", "He"):
            abundances[index] *= 10.0 ** metallicity_dex
    solver.setElementAbundances(abundances)
    indices = {}
    for name in species:
        candidates = [name, name.replace("+", "") + "1" + "+" * name.count("+")]
        for candidate in candidates:
            try:
                candidate = solver.convertToHillNotation(candidate)
            except Exception:
                pass
            idx = solver.getGasSpeciesIndex(candidate)
            if idx != pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
                indices[name] = int(idx)
                break
        if name not in indices:
            raise ValueError(f"FastChem does not provide equilibrium abundance for {name!r}")
    number = {name: np.full(pressure_bar.size, np.nan) for name in species}
    mmw = np.full(pressure_bar.size, np.nan)
    for layer, (pressure, temperature) in enumerate(zip(pressure_bar, temperature_k)):
        input_data = pyfastchem.FastChemInput()
        output_data = pyfastchem.FastChemOutput()
        input_data.temperature = [float(temperature)]
        input_data.pressure = [float(pressure)]
        flag = solver.calcDensities(input_data, output_data)
        if int(np.max(np.asarray(flag))) != 0:
            raise RuntimeError(f"FastChem failed in layer {layer}: flag={flag}")
        densities = np.asarray(output_data.number_densities[0], dtype=float)
        for name, idx in indices.items():
            number[name][layer] = densities[idx]
        mmw[layer] = float(output_data.mean_molecular_weight[0])
    total = pressure_bar * 1.0e6 / (K_B_CGS * temperature_k)
    vmr = {name: np.clip(values / total, 0.0, None) for name, values in number.items()}
    return vmr, mmw, {
        "engine": "pyfastchem equilibrium",
        "metallicity_dex": metallicity_dex,
        "abundance_file": str(abundance_path),
        "logk_file": str(logk_path),
        "median_vmr": {name: float(np.nanmedian(values)) for name, values in vmr.items()},
    }


def equilibrium_profiles(config, metallicity_dex: float, pressure_bar: np.ndarray,
                         temperature_k: np.ndarray, species: tuple[str, ...]):
    """Run FastChem outside the ExoJAX process to isolate OpenMP runtimes."""
    # Resolve/download the small FastChem inputs in the parent so download
    # progress remains visible instead of being trapped in worker stderr.
    abundance_path, logk_path = _fastchem_files(config)
    payload = {
        "config": {
            "cache_dir": str(config.cache_dir),
            "fastchem_abundance_file": str(abundance_path),
            "fastchem_logk_file": str(logk_path),
        },
        "metallicity_dex": float(metallicity_dex),
        "species": list(species),
    }
    with tempfile.TemporaryDirectory(prefix="decanter-fastchem-") as temporary:
        root = Path(temporary)
        request = root / "request.npz"
        response = root / "response.npz"
        np.savez_compressed(
            request,
            pressure_bar=np.asarray(pressure_bar, dtype=float),
            temperature_k=np.asarray(temperature_k, dtype=float),
            payload_json=np.asarray(json.dumps(payload, sort_keys=True)),
        )
        command = [sys.executable, "-m", "decanter.hrccs.fastchem_worker",
                   str(request), str(response)]
        worker_environment = os.environ.copy()
        # The macOS pyFastChem wheel bundles libomp in addition to NumPy's
        # runtime. Keep its compatibility escape hatch confined to this
        # single-threaded subprocess; never expose it to JAX/ExoJAX.
        worker_environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        worker_environment["OMP_NUM_THREADS"] = "1"
        worker_environment["OPENBLAS_NUM_THREADS"] = "1"
        worker_environment["MKL_NUM_THREADS"] = "1"
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False,
            env=worker_environment,
        )
        if completed.returncode != 0 or not response.exists():
            details = "\n".join(
                value.strip() for value in (completed.stdout, completed.stderr) if value.strip()
            )
            raise RuntimeError(
                f"isolated FastChem worker failed with exit code {completed.returncode}:\n{details}"
            )
        with np.load(response, allow_pickle=False) as result:
            names = tuple(str(value) for value in result["species"])
            matrix = np.asarray(result["vmr"], dtype=float)
            vmr = {name: matrix[index] for index, name in enumerate(names)}
            mmw = np.asarray(result["mean_molecular_weight"], dtype=float)
            metadata = json.loads(str(result["metadata_json"]))
    if (not np.all(np.isfinite(mmw))) or np.any(mmw <= 0):
        raise RuntimeError("isolated FastChem worker returned an invalid mean molecular weight")
    for name, profile in vmr.items():
        if not np.all(np.isfinite(profile)) or np.any(profile < 0):
            raise RuntimeError(f"isolated FastChem worker returned an invalid {name} profile")
    metadata["execution"] = "isolated subprocess (OpenMP runtime separation)"
    return vmr, mmw, metadata


def _is_atomic(species: str) -> bool:
    # Element symbols contain one capital followed by at most one lowercase
    # letter. Uppercase diatomics such as OH, CO, and NO are molecules.
    return re.fullmatch(r"[A-Z][a-z]?\+*", species) is not None


def _molecular_database(species: str, configured: dict[str, str]) -> str:
    """Choose HITRAN first when available, otherwise ExoMol."""
    choice = str(configured.get(species, "auto")).strip().lower()
    if choice in {"hitran", "exomol"}:
        return choice
    if choice != "auto":
        raise ValueError(
            f"opacity database for {species} must be auto, hitran, or exomol"
        )
    if species in DEFAULT_EXOMOL_DATASETS:
        return "exomol"
    return "hitran" if species in HITRAN_MOLECULES else "exomol"


def _exomol_path(species: str, root: Path, configured: dict[str, str]) -> Path:
    """Resolve a configured or recommended ExoMol isotopologue dataset."""
    dataset = configured.get(species, DEFAULT_EXOMOL_DATASETS.get(species))
    if dataset:
        relative = Path(dataset)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"ExoMol dataset for {species} must be a relative path")
        if not relative.parts or relative.parts[0] != species:
            relative = Path(species) / relative
        return root / relative

    from exojax.database.multimol import database_path_exomol

    relative = database_path_exomol(species, database_root_path=str(root))
    if not relative or str(relative).endswith("/False"):
        raise ValueError(f"ExoMol has no recommended line list for {species}")
    return root / relative


def _atomic_parts(species: str) -> tuple[str, int]:
    return species.replace("+", ""), 1 + species.count("+")


def _patch_atomic_import() -> None:
    name = "exojax.database.core_atom.line_strength"
    if name in sys.modules:
        return
    from exojax.database.core_atom import io as core_atom_io
    from exojax.utils.constants import Tref_original, ccgs, hcperk

    module = types.ModuleType(name)
    def line_strength_atom(A, gupper, nu_lines, elower, QTref_284, QTmask, Irwin=False):
        qref = np.asarray([QTref_284[mask] for mask in QTmask], dtype=float)
        if Irwin:
            qref[np.where(QTmask == 76)[0]] = core_atom_io.partfn_Fe(Tref_original)
        return (-A * gupper * np.exp(-hcperk * elower / Tref_original)
                * np.expm1(-hcperk * nu_lines / Tref_original)
                / (8.0 * np.pi * ccgs * nu_lines**2 * qref))
    module.line_strength_atom = line_strength_atom
    sys.modules[name] = module


def _kurucz_xsmatrix(opa, temperature, pressure):
    """ExoJAX-version-compatible atomic LPF cross-section matrix."""
    import jax.numpy as jnp
    from jax import jit, vmap
    from exojax.database.core.broadening import doppler_sigma
    from exojax.database.core.line_strength import line_strength
    from exojax.database.core_atom.broadening import gamma_vald3
    from exojax.database.core_atom.pf import interp_QT_284
    from exojax.opacity.lpf.lpf import xsmatrix as xsmatrix_lpf
    from exojax.utils.constants import Tref_original

    adb = opa.mdb
    temperature = jnp.asarray(temperature)
    pressure = jnp.asarray(pressure)
    qt = vmap(interp_QT_284, (0, None, None))(
        temperature, adb.T_gQT, adb.gQT_284species
    )
    qr = qt[:, adb.QTmask] / adb.QTref_284[adb.QTmask]
    strengths = jit(vmap(line_strength, (0, None, None, None, 0, None)))(
        temperature, adb.logsij0, adb.nu_lines, adb.elower, qr, Tref_original
    )
    strengths = jnp.nan_to_num(strengths, nan=0.0, posinf=0.0, neginf=0.0)
    broadening = jit(vmap(gamma_vald3, (
        0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None,
    )))(
        temperature, pressure * adb.vmrH, pressure * adb.vmrHH, pressure * adb.vmrHe,
        adb.ielem, adb.iion, adb.dev_nu_lines, adb.elower, adb.eupper,
        adb.atomicmass, adb.ionE, adb.gamRad, adb.gamSta, adb.vdWdamp, 1.0,
    )
    sigma = jit(vmap(doppler_sigma, (None, 0, None)))(
        adb.nu_lines, temperature, adb.atomicmass
    )
    return xsmatrix_lpf(opa.opainfo, sigma, broadening, strengths)


def _kurucz_path(species: str, root: Path) -> Path:
    from exojax.database.core_atom.io import PeriodicTable

    symbol, ion = _atomic_parts(species)
    matches = np.where(PeriodicTable == symbol)[0]
    if matches.size != 1:
        raise ValueError(f"unknown atomic symbol {symbol!r}")
    filename = f"gf{int(matches[0]):02d}{ion - 1:02d}.all"
    return _download(
        # The official archive currently serves a mismatched TLS certificate
        # on this hostname, while its HTTP endpoint remains available.
        f"http://kurucz.harvard.edu/linelists/gfall/{filename}", root / filename
    )


def _sample_instrument(nu, values, wavelength_um, resolving_power):
    """Convolve on the model grid and sample onto one calibrated order."""
    from exojax.postproc.specop import SopInstProfile
    from exojax.utils.instfunc import resolution_to_gaussian_std

    sop = SopInstProfile(nu, vrmax=500.0)
    beta = resolution_to_gaussian_std(resolving_power)
    target_nu = 1.0e4 / np.asarray(wavelength_um)[::-1]
    sampled = sop.sampling(sop.ipgauss(values, beta), 0.0, target_nu)
    return np.asarray(sampled)[::-1]


def _cia_supported_indices(nu, cia_nu):
    """Indices strictly inside a CIA table's interpolation support."""
    values = np.asarray(nu)
    support = np.asarray(cia_nu)
    if support.size < 2:
        return np.empty(0, dtype=int)
    return np.flatnonzero((values >= support[0]) & (values < support[-1]))


def _wide_wavelength_grid(order_wavelengths, resolving_power, samples_per_fwhm=5.0,
                          doppler_margin_kms=350.0):
    """Oversampled log-lambda grid spanning every retained echelle order."""
    parts = [np.asarray(order, dtype=float).ravel() for order in order_wavelengths]
    finite = np.concatenate([
        part[np.isfinite(part) & (part > 0)] for part in parts if part.size
    ])
    if finite.size < 2:
        raise ValueError("cannot construct a wide template from fewer than two wavelengths")
    margin = float(doppler_margin_kms) / C_KMS
    lower = float(np.min(finite)) * (1.0 - margin)
    upper = float(np.max(finite)) * (1.0 + margin)
    samples = max(
        2,
        int(np.ceil(samples_per_fwhm * resolving_power * np.log(upper / lower))) + 1,
    )
    return np.geomspace(lower, upper, samples)


class TemplateFactory:
    def __init__(self, system, atmosphere, *, instmode: str | None = None):
        self.system = system
        self.config = atmosphere
        self.instmode = str(instmode or "")
        if atmosphere.resolving_power is None:
            raise ValueError("TemplateFactory requires a resolved atmosphere resolving_power")
        self.cache = Path(atmosphere.cache_dir).expanduser()
        try:
            self.cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.cache = Path(tempfile.gettempdir()) / "decanter-hrccs-cache"
            self.cache.mkdir(parents=True, exist_ok=True)
            self.config = replace(atmosphere, cache_dir=str(self.cache))
        self._chemistry = {}
        self._cia_databases = {}

    def _cia_database(self, filename):
        """Load each large HITRAN CIA table once for all per-order models."""
        from exojax.database.cia.api import CdbCIA
        from exojax.utils.url import url_HITRANCIA

        if filename not in self._cia_databases:
            cia_root = (Path(self.config.cia_dir).expanduser()
                        if self.config.cia_dir else self.cache / "cia")
            cia_path = _download(f"{url_HITRANCIA()}{filename}", cia_root / filename)
            self._cia_databases[filename] = CdbCIA(str(cia_path), margin=0.0)
        return self._cia_databases[filename]

    def _key(self, species, wave):
        # Orbital geometry, coordinates, and event timing do not affect an
        # atmospheric spectrum. Restrict the cache key to the system values
        # actually consumed by the forward model so changing an eclipse
        # ephemeris or adding e/omega reuses the same expensive opacity model.
        system = {
            name: getattr(self.system, name)
            for name in (
                "stellar_radius_rsun", "planet_radius_rjup", "planet_mass_mjup",
                "equilibrium_temperature_k", "metallicity_dex",
            )
        }
        payload = {
            "schema": 5, "species": species, "instmode": self.instmode,
            "wave": [round(float(wave[0]), 8), round(float(wave[-1]), 8), len(wave)],
            "system": system, "atmosphere": vars(self.config),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]

    def _legacy_path(self, species, wave):
        """Schema-4 path from before orbital fields were separated."""
        system = dict(vars(self.system))
        system.pop("eccentricity", None)
        system.pop("argument_of_periastron_deg", None)
        payload = {
            "schema": 4, "species": species, "instmode": self.instmode,
            "wave": [round(float(wave[0]), 8), round(float(wave[-1]), 8), len(wave)],
            "system": system, "atmosphere": vars(self.config),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:20]
        return self.cache / "templates" / f"{species.replace('+', 'p')}_{digest}.npz"

    def _path(self, species, wave):
        return (self.cache / "templates"
                / f"{species.replace('+', 'p')}_{self._key(species, wave)}.npz")

    @staticmethod
    def _load(path, species):
        with np.load(path, allow_pickle=False) as data:
            return Template(species, data["wavelength_um"], data["transit_depth"],
                            data["contrast"], json.loads(str(data["metadata_json"])))

    @staticmethod
    def _save(path, template):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, wavelength_um=template.wavelength_um,
            transit_depth=template.transit_depth, contrast=template.contrast,
            metadata_json=np.asarray(json.dumps(template.metadata, sort_keys=True)),
        )

    def _compute(self, species, wave):
        if self.config.backend == "analytic":
            return self._analytic(species, wave)
        if self.config.backend != "exojax":
            raise ValueError(f"unknown atmosphere backend {self.config.backend!r}")
        if self.config.model_verbose:
            return self._exojax(species, wave)
        captured_stdout, captured_stderr = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr), \
                    warnings.catch_warnings():
                warnings.filterwarnings("ignore", module=r"exojax\..*")
                return self._exojax(species, wave)
        except Exception as exc:
            details = "\n".join(
                value.strip() for value in (
                    captured_stdout.getvalue(), captured_stderr.getvalue()
                ) if value.strip()
            )
            suffix = f"\nCaptured model output:\n{details}" if details else ""
            raise RuntimeError(f"ExoJAX template failed for {species}: {exc}{suffix}") from exc

    def build_wide(self, species: str, order_wavelengths: np.ndarray,
                   *, show_progress: bool = True) -> Template:
        """Build one wide model using bounded-memory internal calculations."""
        wave = _wide_wavelength_grid(
            order_wavelengths, float(self.config.resolving_power), samples_per_fwhm=5.0,
        )
        path = self._path(species, wave)
        for candidate in (path, self._legacy_path(species, wave)):
            if candidate.exists():
                cached = self._load(candidate, species)
                if (np.all(np.isfinite(cached.contrast))
                        and np.ptp(cached.contrast) > 0.0):
                    return cached
        chunk_points = int(
            self.config.atomic_wide_model_chunk_points if _is_atomic(species)
            else self.config.wide_model_chunk_points
        )
        starts = list(range(0, wave.size, chunk_points))
        if len(starts) > 1 and wave.size - starts[-1] < 2:
            starts.pop()
        from tqdm.auto import tqdm
        pieces = []
        chunk_bar = tqdm(
            starts, desc=f"{species} wide-model chunks", unit="chunk", leave=False,
            disable=not show_progress, dynamic_ncols=True,
        )
        for chunk_index, start in enumerate(chunk_bar):
            stop = starts[chunk_index + 1] if chunk_index + 1 < len(starts) else wave.size
            pieces.append(self._compute(species, wave[start:stop]))
            # JAX retains compiled executables and device buffers by shape.
            # Clear between chunks so peak memory is bounded for atomic LPF.
            if self.config.backend == "exojax" and _is_atomic(species):
                import jax
                jax.clear_caches()
            gc.collect()
        template = self._stitch_wide(species, wave, pieces, chunk_points)
        self._save(path, template)
        return template

    def _stitch_wide(self, species, wave, pieces, chunk_points):
        """Combine bounded calculations into the sole wide template product."""
        if not pieces:
            raise RuntimeError(f"no wide-template chunks were computed for {species}")
        metadata = dict(pieces[0].metadata)
        line_counts = [int(piece.metadata.get("line_count", 0)) for piece in pieces]
        if metadata.get("backend") == "exojax" and sum(line_counts) == 0:
            raise ValueError(
                f"the selected opacity database contains no {species} lines across "
                f"the full wide-template "
                f"range {wave[0]:.6f}-{wave[-1]:.6f} micron"
            )
        coverage = {}
        for filename in ("H2-H2_2011.cia", "H2-He_2011.cia"):
            weighted = [
                (len(piece.wavelength_um),
                 piece.metadata.get("continuum", {}).get("cia_grid_coverage", {}).get(filename))
                for piece in pieces
            ]
            valid = [(size, value) for size, value in weighted if value is not None]
            if valid:
                coverage[filename] = float(
                    sum(size * value for size, value in valid) / sum(size for size, _ in valid)
                )
        metadata.update({
            "line_count": int(sum(line_counts)),
            "line_count_note": "sum over bounded wavelength chunks",
            "model_scope": "single wide-band template",
            "wide_wavelength_min_um": float(wave[0]),
            "wide_wavelength_max_um": float(wave[-1]),
            "wide_grid_points": int(wave.size),
            "wide_samples_per_resolution_fwhm": 5.0,
            "wide_doppler_margin_kms": 350.0,
            "wide_model_chunks": len(pieces),
            "wide_model_chunk_points": chunk_points,
        })
        metadata.setdefault("continuum", {})["cia_grid_coverage"] = coverage
        return Template(
            species, wave,
            np.concatenate([piece.transit_depth for piece in pieces]),
            np.concatenate([piece.contrast for piece in pieces]),
            metadata,
        )

    def build(self, species: str, wavelength_um: np.ndarray) -> Template:
        """Build/cache a model directly on one requested grid."""
        wave = np.asarray(wavelength_um, dtype=float)
        path = self._path(species, wave)
        for candidate in (path, self._legacy_path(species, wave)):
            if candidate.exists():
                return self._load(candidate, species)
        template = self._compute(species, wave)
        self._save(path, template)
        return template

    @staticmethod
    def sample_orders(wide_template: Template,
                      order_wavelengths: np.ndarray) -> tuple[Template, ...]:
        """Interpolate one instrument-convolved wide template to native orders."""
        result = []
        wide_wave = np.asarray(wide_template.wavelength_um, dtype=float)
        for order_index, values in enumerate(order_wavelengths):
            wavelength = np.asarray(values, dtype=float)
            if (np.nanmin(wavelength) < wide_wave[0]
                    or np.nanmax(wavelength) > wide_wave[-1]):
                raise ValueError(f"order {order_index} lies outside the wide template")
            metadata = dict(wide_template.metadata)
            metadata.update({
                "sampling": "wide instrument-convolved template interpolated to order grid",
                "sampled_order_index": order_index,
                "sampled_order_points": int(wavelength.size),
            })
            result.append(Template(
                wide_template.species,
                np.asarray(wavelength, dtype=float).copy(),
                np.interp(wavelength, wide_wave, wide_template.transit_depth),
                np.interp(wavelength, wide_wave, wide_template.contrast),
                metadata,
            ))
        return tuple(result)

    def _analytic(self, species, wave):
        # Deterministic smoke-test backend; production defaults to ExoJAX.
        seed = int.from_bytes(hashlib.sha256(species.encode()).digest()[:4], "little")
        rng = np.random.default_rng(seed)
        centers = rng.uniform(wave[0], wave[-1], 24)
        depth = np.full(wave.size, (self.system.planet_radius_rjup * R_JUP_CM /
                                    (self.system.stellar_radius_rsun * R_SUN_CM)) ** 2)
        for center in centers:
            width = center * 2.0 / C_KMS
            depth += 4.0e-4 * np.exp(-0.5 * ((wave - center) / width) ** 2)
        return Template(species, wave, depth, -(depth - np.nanpercentile(depth, 10)),
                        {"backend": "analytic_test_only",
                         "instrument_mode": self.instmode,
                         "resolving_power": float(self.config.resolving_power),
                         "sampling": "requested wavelength grid"})

    def _exojax(self, species, wave):
        import jax.numpy as jnp
        from exojax.opacity import OpaCIA, OpaDirect, OpaPremodit, OpaRayleigh
        from exojax.rt import ArtTransPure
        from exojax.utils.grids import wavenumber_grid

        pad = 350.0 / C_KMS
        nu_min = 1.0e4 / (wave[-1] * (1.0 + pad))
        nu_max = 1.0e4 / (wave[0] * (1.0 - pad))
        n_grid = max(512, int(np.ceil(3.0 * 200_000.0 * np.log(nu_max / nu_min))))
        n_grid += n_grid % 2
        xsmode = "lpf" if _is_atomic(species) else "premodit"
        nu, _, opa_resolution = wavenumber_grid(
            nu_min, nu_max, n_grid, xsmode=xsmode, wavelength_order="ascending"
        )
        art = ArtTransPure(pressure_top=self.config.pressure_top_bar,
                           pressure_btm=self.config.pressure_bottom_bar,
                           nlayer=self.config.n_layers, integration="simpson")
        temperature_np = np.full(self.config.n_layers, self.system.equilibrium_temperature_k)
        temperature = jnp.asarray(temperature_np)
        all_species = tuple(dict.fromkeys((species, "H2", "He")))
        chemistry_key = (species, tuple(all_species))
        if chemistry_key not in self._chemistry:
            self._chemistry[chemistry_key] = equilibrium_profiles(
                self.config, self.system.metallicity_dex,
                np.asarray(art.pressure), temperature_np, all_species
            )
        vmr, mmw_np, chemistry = self._chemistry[chemistry_key]
        mmw = jnp.asarray(mmw_np)
        radius = self.system.planet_radius_rjup * R_JUP_CM
        stellar_radius = self.system.stellar_radius_rsun * R_SUN_CM
        gravity_btm = G_CGS * self.system.planet_mass_mjup * M_JUP_G / radius**2
        gravity = art.gravity_profile(temperature, mmw, radius, gravity_btm)

        if _is_atomic(species):
            _patch_atomic_import()
            from exojax.database.kurucz.api import AdbKurucz
            kurucz_root = (Path(self.config.kurucz_dir).expanduser()
                           if self.config.kurucz_dir else self.cache / "kurucz")
            path = _kurucz_path(species, kurucz_root)
            adb = AdbKurucz(path, nurange=[nu_min, nu_max], margin=0.0,
                            crit=self.config.kurucz_line_strength_crit, gpu_transfer=True,
                            vmr_fraction=[0.0, 0.16, 0.84])
            source = f"Kurucz {path.name} via ExoJAX"
            line_count = int(np.asarray(adb.nu_lines).size)
            if line_count:
                opa = OpaDirect(adb, nu, wavelength_order="ascending")
                xs = _kurucz_xsmatrix(opa, temperature, art.pressure)
                molmass = float(np.nanmedian(np.asarray(adb.atomicmass)))
            else:
                # Sparse atomic line lists leave some bounded-memory chunks
                # empty. Those chunks still need their atmospheric continuum;
                # only an entirely line-free wide model is an error, which
                # _stitch_wide checks.
                xs = None
                molmass = None
        else:
            database = _molecular_database(species, self.config.opacity_databases)
            if database == "exomol":
                from exojax.database.exomol.api import MdbExomol

                exomol_root = (Path(self.config.exomol_dir).expanduser()
                               if self.config.exomol_dir else self.cache / "exomol")
                path = _exomol_path(species, exomol_root, self.config.exomol_datasets)
                definition = path / f"{path.parent.name}__{path.name}.def"
                if not definition.exists():
                    from tqdm.auto import tqdm
                    tqdm.write(
                        f"Downloading ExoMol line data for {species} -> {path}",
                        file=sys.__stderr__ or sys.stderr,
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                mdb = MdbExomol(
                    str(path), nurange=[nu_min, nu_max],
                    crit=self.config.line_strength_crit,
                    Ttyp=self.system.equilibrium_temperature_k,
                    broadf=False, gpu_transfer=False, inherit_dataframe=False,
                    engine="vaex",
                )
                molmass = float(mdb.molmass)
                source = f"ExoMol {path.parent.name}/{path.name} via ExoJAX"
                line_count = int(np.asarray(mdb.nu_lines).size)
            else:
                from exojax.database.hitran.api import MdbHitran
                from exojax.database.multimol import database_path_hitran12
                hitran_root = (Path(self.config.hitran_dir).expanduser()
                               if self.config.hitran_dir else self.cache / "hitran")
                path = hitran_root / database_path_hitran12(species)
                cached_tables = tuple(path.parent.glob("*.hdf5")) if path.parent.exists() else ()
                if not path.exists() and not cached_tables:
                    from tqdm.auto import tqdm
                    tqdm.write(
                        f"Downloading HITRAN line data for {species} via ExoJAX -> {path}",
                        file=sys.__stderr__ or sys.stderr,
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                mdb = MdbHitran(
                    path, nurange=[nu_min, nu_max], isotope=self.config.hitran_isotope,
                    gpu_transfer=False, inherit_dataframe=False,
                    crit=self.config.line_strength_crit,
                    Ttyp=self.system.equilibrium_temperature_k, engine="vaex",
                )
                molmass = float(mdb.molmass)
                source = f"HITRAN {species} via ExoJAX"
                line_count = int(np.asarray(mdb.nu_lines).size)
            if line_count:
                opa = OpaPremodit(
                    mdb, nu, diffmode=0,
                    broadening_resolution={"mode": "manual", "value": 0.2},
                    auto_trange=(
                        max(100.0, 0.7 * self.system.equilibrium_temperature_k),
                        1.3 * self.system.equilibrium_temperature_k,
                    ),
                    allow_32bit=True, wavelength_order="ascending",
                )
                xs = opa.xsmatrix(temperature, art.pressure)
            else:
                xs = None
        if xs is None:
            molecular_dtau = jnp.zeros((self.config.n_layers, len(nu)))
        else:
            species_mmr = jnp.asarray(vmr[species] * molmass / mmw_np)
            molecular_dtau = art.opacity_profile_xs(xs, species_mmr, molmass, gravity)

        continuum = jnp.zeros_like(molecular_dtau)
        cia_coverage = {}
        if self.config.include_rayleigh:
            for molecule, mass in (("H2", 2.01588), ("He", 4.002602)):
                rayleigh = OpaRayleigh(nu, molecule).xsvector()
                continuum += art.opacity_profile_xs(
                    rayleigh, jnp.asarray(vmr[molecule] * mass / mmw_np), mass, gravity
                )
        if self.config.include_cia:
            for filename, first, second in (
                ("H2-H2_2011.cia", "H2", "H2"),
                ("H2-He_2011.cia", "H2", "He"),
            ):
                cdb = self._cia_database(filename)
                nu_values = np.asarray(nu)
                # HITRAN CIA pairs do not all span the complete WINERED range
                # (H2-H2 ends at 10,000 cm^-1). ExoJAX's interpolator cannot
                # extrapolate and indexes past the table at its upper edge.
                # Evaluate only on supported samples and leave the unavailable
                # contribution at zero elsewhere.
                indices = _cia_supported_indices(nu_values, cdb.nucia)
                cia_coverage[filename] = float(indices.size / nu_values.size)
                if indices.size == 0:
                    continue
                opa_cia = OpaCIA(cdb, nu_grid=nu[indices])
                contribution = art.opacity_profile_cia(
                    opa_cia.logacia_matrix(temperature), temperature,
                    jnp.asarray(vmr[first]), jnp.asarray(vmr[second]),
                    mmw[:, None], gravity,
                )
                continuum = continuum.at[:, indices].add(contribution)
        if self.config.cloud_top_pressure_bar is not None:
            cloudy = jnp.asarray(np.asarray(art.pressure) >= self.config.cloud_top_pressure_bar)
            continuum += jnp.where(cloudy[:, None], 1.0e6, 0.0)
        baseline = (radius / stellar_radius) ** 2
        def absolute(dtau):
            return np.asarray(art.run(dtau, temperature, mmw, radius, gravity_btm)) * baseline
        high_continuum = absolute(continuum)
        # Subtracting two ~percent-level float32 transit depths can erase a
        # weak equilibrium species entirely (e.g. OH near 960 K). Evaluate the
        # smallest opacity boost that gives a numerically resolved differential
        # signal, then scale that optically-thin differential back down. Strong
        # templates use scale=1 and are unchanged.
        opacity_boost = 1.0
        high_delta = np.zeros_like(high_continuum)
        for candidate in (1.0, 1.0e2, 1.0e4, 1.0e6, 1.0e8, 1.0e10, 1.0e12):
            boosted = absolute(candidate * molecular_dtau + continuum)
            trial = boosted - high_continuum
            opacity_boost = candidate
            high_delta = trial / candidate
            if np.ptp(trial) >= 1.0e-5:
                break
        sampled_delta = np.asarray(_sample_instrument(
            nu, high_delta, wave, self.config.resolving_power
        ), dtype=np.float64)
        cont = np.asarray(_sample_instrument(
            nu, high_continuum, wave, self.config.resolving_power
        ), dtype=np.float64)
        depth = cont + sampled_delta
        contrast = -sampled_delta
        meta = {"backend": "exojax", "source": source, "line_count": line_count,
                "opacity_database": ("kurucz" if _is_atomic(species) else database),
                "opa_resolution": float(opa_resolution), "chemistry": chemistry,
                "instrument_mode": self.instmode,
                "resolving_power": float(self.config.resolving_power),
                "sampling": "Gaussian LSF then sampled onto requested wavelength grid",
                "numerical_opacity_boost": opacity_boost,
                "continuum": {"rayleigh": self.config.include_rayleigh,
                              "cia": self.config.include_cia,
                              "cia_grid_coverage": cia_coverage,
                              "cloud_top_pressure_bar": self.config.cloud_top_pressure_bar},
                "baseline_transit_depth": baseline}
        return Template(species, wave, depth, contrast, meta)
