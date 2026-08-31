"""TOML configuration for the downstream HRCCS pipeline."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


def _construct(cls, values: dict[str, Any] | None):
    values = dict(values or {})
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown [{cls.__name__}] fields: {unknown}")
    return cls(**values)


@dataclass(frozen=True)
class InputConfig:
    decanter_dir: str
    fsr_cut: float | None = None
    orders: tuple[int, ...] = ()
    telluric_product: str | None = None


@dataclass(frozen=True)
class SystemConfig:
    planet_name: str | None = None
    observation_type: str = "transit"
    period_days: float = 0.0
    event_midpoint_bjd_tdb: float = 0.0
    event_duration_hours: float = 0.0
    expected_kp_kms: float = 0.0
    eccentricity: float = 0.0
    argument_of_periastron_deg: float = 90.0
    ra_deg: float | None = None
    dec_deg: float | None = None
    stellar_rv_kms: float | None = None
    stellar_radius_rsun: float = 1.0
    planet_radius_rjup: float = 1.0
    planet_mass_mjup: float = 1.0
    equilibrium_temperature_k: float = 1500.0
    metallicity_dex: float = 0.0
    observatory_lon_deg: float = -70.6925
    observatory_lat_deg: float = -29.0146
    observatory_height_m: float = 2380.0
    iers_auto_download: bool = True

    @property
    def event_midpoint(self) -> float:
        return float(self.event_midpoint_bjd_tdb)

    @property
    def event_duration(self) -> float:
        return float(self.event_duration_hours)


@dataclass(frozen=True)
class AtmosphereConfig:
    species: tuple[str, ...] = ("H2O",)
    backend: str = "exojax"
    model_verbose: bool = False
    pressure_top_bar: float = 1.0e-8
    pressure_bottom_bar: float = 10.0
    n_layers: int = 60
    mean_molecular_weight: float = 2.3
    cloud_top_pressure_bar: float | None = None
    include_rayleigh: bool = True
    include_cia: bool = True
    h2_vmr: float = 0.85
    he_vmr: float = 0.15
    # None selects the nominal resolving power from the Decanter INSTMODE.
    # An explicit value remains available for non-standard slits/configurations.
    resolving_power: float | None = None
    # Bound the target-grid size of each internal opacity calculation. Chunks
    # are stitched into one wide template before any order is sampled.
    wide_model_chunk_points: int = 6_000
    atomic_wide_model_chunk_points: int = 2_000
    line_strength_crit: float = 1.0e-30
    kurucz_line_strength_crit: float = 0.0
    hitran_isotope: int = 1
    # Per-species values may be "auto", "hitran", or "exomol". Auto keeps
    # HITRAN-supported molecules on HITRAN and sends other molecules to
    # ExoMol. FeH and CrH default to their MoLLIST ExoMol databases.
    opacity_databases: dict[str, str] = field(default_factory=dict)
    # Optional ExoMol dataset below <exomol_dir>/<species>/, for example
    # {CrH = "52Cr-1H/MoLLIST"}. Unspecified species use ExoMol's recommended
    # stable-isotopologue dataset.
    exomol_datasets: dict[str, str] = field(default_factory=dict)
    fastchem_abundance_file: str | None = None
    fastchem_logk_file: str | None = None
    hitran_dir: str | None = None
    exomol_dir: str | None = None
    cia_dir: str | None = None
    kurucz_dir: str | None = None
    cache_dir: str = "~/.cache/decanter/hrccs"

    def resolving_power_for(self, instmode: str | None) -> float:
        """Resolve the template LSF against the reduced spectrum's mode."""
        if self.resolving_power is not None:
            if self.resolving_power <= 0:
                raise ValueError("atmosphere resolving_power must be positive")
            return float(self.resolving_power)
        from decanter.wavecal.config import NOMINAL_RESOLVING_POWER

        normalized = str(instmode or "").strip().upper().replace("_", "-")
        aliases = {"Y": "HIRES-Y", "J": "HIRES-J", "HIRESY": "HIRES-Y",
                   "HIRESJ": "HIRES-J"}
        normalized = aliases.get(normalized, normalized)
        if normalized in NOMINAL_RESOLVING_POWER:
            return NOMINAL_RESOLVING_POWER[normalized]
        raise ValueError(
            f"no nominal resolving power for INSTMODE={instmode!r}; "
            "set [atmosphere].resolving_power explicitly"
        )


@dataclass(frozen=True)
class ReductionConfig:
    # The reduction reproduces the minimal-processing WASP-69b reference:
    # linear-flux SVD and exact injected-template SVD refitting.
    telluric_threshold: float = 0.90
    # Which exposures the telluric minimum is taken over. "signal" is the ones
    # being cross-correlated: in transit for transmission, out of eclipse for
    # emission. "event"/"out_of_event" name the geometry directly and "all"
    # uses every exposure.
    telluric_mask_scope: str = "signal"
    edge_trim_pixels: int = 0
    ccf_lsf_margin_widths: float = 3.0
    min_valid_pixels: int = 100
    svd_components: tuple[int, ...] = tuple(range(1, 13))


@dataclass(frozen=True)
class SearchConfig:
    rv_min_kms: float = -250.0
    rv_max_kms: float = 250.0
    rv_step_kms: float = 2.0
    kp_min_kms: float | None = None
    kp_max_kms: float | None = None
    kp_step_kms: float = 2.0
    vsys_min_kms: float | None = None
    vsys_max_kms: float | None = None
    vsys_step_kms: float = 2.0
    map_sigma_clip: float = 3.0
    local_kp_half_width_kms: float = 30.0
    local_vsys_half_width_kms: float = 15.0


@dataclass(frozen=True)
class InjectionConfig:
    scale: float = 1.0
    random_seed: int = 42690
    null_realizations: int = 5


@dataclass(frozen=True)
class PlotConfig:
    enabled: bool = True
    orders_per_page: int = 5
    dpi: int = 180
    formats: tuple[str, ...] = ("pdf", "png")


@dataclass(frozen=True)
class HRCCSConfig:
    input: InputConfig
    system: SystemConfig
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    reduction: ReductionConfig = field(default_factory=ReductionConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    output_dir: str = "hrccs_output"
    show_progress: bool = True

    def validate(self) -> None:
        if self.system.observation_type not in {"transit", "eclipse"}:
            raise ValueError("system observation_type must be 'transit' or 'eclipse'")
        if self.system.period_days <= 0 or self.system.event_duration <= 0:
            raise ValueError("system period_days and event_duration_hours must be positive")
        if self.system.event_midpoint <= 0 or self.system.expected_kp_kms <= 0:
            raise ValueError("system event_midpoint_bjd_tdb and expected_kp_kms are required")
        if not 0.0 <= self.system.eccentricity < 1.0:
            raise ValueError("system eccentricity must be in [0, 1)")
        if not self.atmosphere.species:
            raise ValueError("at least one atmosphere species is required")
        if (self.atmosphere.resolving_power is not None
                and self.atmosphere.resolving_power <= 0):
            raise ValueError("atmosphere resolving_power must be positive")
        if self.atmosphere.wide_model_chunk_points < 256:
            raise ValueError("atmosphere wide_model_chunk_points must be at least 256")
        if self.atmosphere.atomic_wide_model_chunk_points < 256:
            raise ValueError("atmosphere atomic_wide_model_chunk_points must be at least 256")
        invalid_databases = {
            species: database
            for species, database in self.atmosphere.opacity_databases.items()
            if str(database).lower() not in {"auto", "hitran", "exomol"}
        }
        if invalid_databases:
            raise ValueError(
                "atmosphere opacity_databases values must be auto, hitran, or exomol: "
                f"{invalid_databases}"
            )
        for species, dataset in self.atmosphere.exomol_datasets.items():
            path = Path(dataset)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"atmosphere exomol_datasets[{species!r}] must be a relative path"
                )
        counts = self.reduction.svd_components
        if not counts or min(counts) < 0 or len(set(counts)) != len(counts):
            raise ValueError("svd_components must be unique non-negative integers")
        if not 0.0 < self.reduction.telluric_threshold <= 1.0:
            raise ValueError("telluric_threshold must be in (0, 1]")
        if self.reduction.telluric_mask_scope not in {
            "all", "signal", "event", "out_of_event"
        }:
            raise ValueError(
                "telluric_mask_scope must be all, signal, event, or out_of_event"
            )
        if self.reduction.edge_trim_pixels < 0:
            raise ValueError("edge_trim_pixels must be non-negative")
        if self.reduction.ccf_lsf_margin_widths < 0:
            raise ValueError("ccf_lsf_margin_widths must be non-negative")
        for name in ("rv_step_kms", "kp_step_kms", "vsys_step_kms"):
            if getattr(self.search, name) <= 0:
                raise ValueError(f"search {name} must be positive")
        if (self.search.local_kp_half_width_kms < 0
                or self.search.local_vsys_half_width_kms < 0):
            raise ValueError("local component-selection half widths must be non-negative")
        if self.injection.null_realizations < 1:
            raise ValueError("injection null_realizations must be at least one")
        if self.injection.scale <= 0:
            raise ValueError("injection scale must be positive")

    def grids(self, stellar_rv_kms: float):
        import numpy as np

        search = self.search
        kp_lo = 0.0 if search.kp_min_kms is None else search.kp_min_kms
        kp_hi = (1.5 * self.system.expected_kp_kms
                 if search.kp_max_kms is None else search.kp_max_kms)
        default_vsys = max(5.0 * abs(stellar_rv_kms), 25.0)
        vs_lo = -default_vsys if search.vsys_min_kms is None else search.vsys_min_kms
        vs_hi = +default_vsys if search.vsys_max_kms is None else search.vsys_max_kms
        make = lambda lo, hi, step: np.arange(lo, hi + 0.5 * step, step)
        return (
            make(search.rv_min_kms, search.rv_max_kms, search.rv_step_kms),
            make(kp_lo, kp_hi, search.kp_step_kms),
            make(vs_lo, vs_hi, search.vsys_step_kms),
        )


def load_config(path: str | Path) -> HRCCSConfig:
    source = Path(path)
    raw = tomllib.loads(source.read_text())
    required = {"input", "system"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"missing config tables: {missing}")
    allowed = {"input", "system", "atmosphere", "reduction", "search",
               "injection", "plots", "output_dir", "show_progress"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown top-level config fields: {unknown}")
    config = HRCCSConfig(
        input=_construct(InputConfig, raw["input"]),
        system=_construct(SystemConfig, raw["system"]),
        atmosphere=_construct(AtmosphereConfig, raw.get("atmosphere")),
        reduction=_construct(ReductionConfig, raw.get("reduction")),
        search=_construct(SearchConfig, raw.get("search")),
        injection=_construct(InjectionConfig, raw.get("injection")),
        plots=_construct(PlotConfig, raw.get("plots")),
        output_dir=str(raw.get("output_dir", "hrccs_output")),
        show_progress=bool(raw.get("show_progress", True)),
    )
    config.validate()
    return config
