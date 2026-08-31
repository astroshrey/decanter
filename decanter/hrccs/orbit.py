"""Target metadata, barycentric correction, and transit/eclipse coordinates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Orbit:
    target_name: str
    ra_deg: float
    dec_deg: float
    stellar_rv_kms: float
    bjd_tdb: np.ndarray
    phase: np.ndarray
    event_mask: np.ndarray
    signal_mask: np.ndarray
    signal_weight: np.ndarray
    velocity_basis: np.ndarray
    observation_type: str
    berv_kms: np.ndarray

    @property
    def in_transit(self) -> np.ndarray:
        """Backward-compatible alias; prefer ``event_mask``."""
        return self.event_mask

    @property
    def transit_weight(self) -> np.ndarray:
        """Backward-compatible alias; prefer ``signal_weight``."""
        return self.signal_weight


def _target_name(config, metadata) -> str:
    if config.planet_name:
        return config.planet_name
    for row in metadata:
        value = str(row.get("OBJECT", "")).strip()
        if value and value.upper() not in {"UNKNOWN", "N/A"}:
            return value
    raise ValueError("planet_name could not be inferred from FITS metadata")


def _simbad(name: str, cache_dir: Path) -> tuple[float, float, float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "simbad_target.json"
    if cache.exists():
        values = json.loads(cache.read_text())
        if values.get("query") == name:
            return float(values["ra_deg"]), float(values["dec_deg"]), float(values["rv_kms"])
    from astroquery.simbad import Simbad

    query = Simbad()
    query.add_votable_fields("velocity")
    table = query.query_object(name)
    if table is None or len(table) == 0:
        raise RuntimeError(f"SIMBAD returned no match for {name!r}")
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    columns = {column.lower(): column for column in table.colnames}
    def column(*names):
        for candidate in names:
            if candidate.lower() in columns:
                return columns[candidate.lower()]
        raise RuntimeError(f"SIMBAD response lacks all of {names}; columns={table.colnames}")
    coord = SkyCoord(str(table[column("ra")][0]), str(table[column("dec")][0]),
                     unit=(u.hourangle, u.deg))
    rv = float(table[column("rvz_radvel", "rvz_radvel_value", "velocities")][0])
    values = {"query": name, "ra_deg": coord.ra.deg, "dec_deg": coord.dec.deg,
              "rv_kms": rv, "source": "SIMBAD rvz_radvel"}
    cache.write_text(json.dumps(values, indent=2, sort_keys=True))
    return coord.ra.deg, coord.dec.deg, rv


def _true_anomaly(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    """Solve Kepler's equation and return true anomaly in radians."""
    mean = np.asarray(mean_anomaly, dtype=float)
    eccentric = mean.copy()
    for _ in range(20):
        correction = ((eccentric - eccentricity * np.sin(eccentric) - mean)
                      / (1.0 - eccentricity * np.cos(eccentric)))
        eccentric -= correction
        if np.nanmax(np.abs(correction)) < 1.0e-13:
            break
    return 2.0 * np.arctan2(
        np.sqrt(1.0 + eccentricity) * np.sin(0.5 * eccentric),
        np.sqrt(1.0 - eccentricity) * np.cos(0.5 * eccentric),
    )


def _orbital_velocity_basis(phase: np.ndarray, observation_type: str,
                            eccentricity: float = 0.0,
                            argument_of_periastron_deg: float = 90.0) -> np.ndarray:
    """Dimensionless planet RV divided by Kp, anchored at the event midpoint.

    The sign follows the HRCCS convention ``+sin(2 pi phase)`` about transit.
    For an eccentric orbit this is ``-[cos(f + omega) + e cos(omega)]``.
    Infer mean anomaly at the supplied transit/eclipse midpoint using the
    edge-on conjunction approximation.
    """
    phase = np.asarray(phase, dtype=float)
    sign = -1.0 if observation_type == "eclipse" else 1.0
    if eccentricity == 0.0:
        return sign * np.sin(2.0 * np.pi * phase)
    omega = np.deg2rad(argument_of_periastron_deg)
    conjunction_longitude = 1.5 * np.pi if observation_type == "eclipse" else 0.5 * np.pi
    true_at_event = conjunction_longitude - omega
    eccentric_at_event = 2.0 * np.arctan2(
        np.sqrt(1.0 - eccentricity) * np.sin(0.5 * true_at_event),
        np.sqrt(1.0 + eccentricity) * np.cos(0.5 * true_at_event),
    )
    mean_at_event = eccentric_at_event - eccentricity * np.sin(eccentric_at_event)
    mean = mean_at_event + 2.0 * np.pi * phase
    true = _true_anomaly(mean, eccentricity)
    return -(np.cos(true + omega) + eccentricity * np.cos(omega))


def build_orbit(system, time_jd_utc, metadata, cache_dir: str | Path) -> Orbit:
    from astropy import units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    from astropy.utils import iers

    iers.conf.auto_download = bool(system.iers_auto_download)
    iers.conf.auto_max_age = None
    name = _target_name(system, metadata)
    if system.ra_deg is None or system.dec_deg is None or system.stellar_rv_kms is None:
        sim_ra, sim_dec, sim_rv = _simbad(name, Path(cache_dir).expanduser())
    else:
        sim_ra = sim_dec = sim_rv = np.nan
    ra = sim_ra if system.ra_deg is None else system.ra_deg
    dec = sim_dec if system.dec_deg is None else system.dec_deg
    stellar_rv = sim_rv if system.stellar_rv_kms is None else system.stellar_rv_kms
    location = EarthLocation.from_geodetic(
        system.observatory_lon_deg * u.deg,
        system.observatory_lat_deg * u.deg,
        system.observatory_height_m * u.m,
    )
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    utc = Time(np.asarray(time_jd_utc), format="jd", scale="utc", location=location)
    light_time = utc.light_travel_time(target, kind="barycentric")
    bjd = np.asarray((utc.tdb + light_time).jd, dtype=float)
    berv = target.radial_velocity_correction("barycentric", obstime=utc).to_value(u.km / u.s)
    phase = ((bjd - system.event_midpoint + 0.5 * system.period_days)
             % system.period_days) / system.period_days - 0.5
    half = 0.5 * system.event_duration / 24.0 / system.period_days
    event_mask = np.abs(phase) <= half
    # Transmission exists in transit; dayside emission is visible out of
    # eclipse. Around secondary eclipse the planet RV has the opposite slope
    # to the transit-centered sine convention.
    is_eclipse = system.observation_type == "eclipse"
    signal_mask = ~event_mask if is_eclipse else event_mask
    weight = signal_mask.astype(float)
    velocity_basis = _orbital_velocity_basis(
        phase, system.observation_type, system.eccentricity,
        system.argument_of_periastron_deg,
    )
    return Orbit(name, float(ra), float(dec), float(stellar_rv), bjd, phase,
                 event_mask, signal_mask, weight, velocity_basis,
                 system.observation_type, np.asarray(berv, dtype=float))
