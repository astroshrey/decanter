"""Load a whole decanter reduction directory onto a common per-order grid.

The calibration is measured across a set of frames, so the loader returns the
whole set: every frame's object and sky spectrum resampled onto one reference
grid per order, plus the timing and pointing metadata that
:mod:`decanter.io.headers` propagates onto the products.

The reference grid is uniform in ``ln(lambda)``, where a Doppler shift is a
constant pixel shift. The CCF pixel scale is then one number per order and a
measured shift converts to a velocity by a single multiplication.

Per order the grid spans the largest first wavelength to the smallest last
wavelength over all frames, so no frame is extrapolated, and carries
``max_order(min_frame(N_native))`` samples, so no order is down-sampled
relative to its own native grid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from decanter._reduction import Reduction
from decanter.wavecal.solution import C_KMS

_OBJ_PATTERN = re.compile(r"_NO\d+_sscfm_m(\d+)_fsr([0-9.]+)_VAC\.fits$")
_SKY_PATTERN = re.compile(r"_skyNO\d+_fm_m(\d+)trans1dcutw_fsr([0-9.]+)_VAC\.fits$")


@dataclass
class Series:
    """Every frame of one dataset on a common per-order reference grid.

    Attributes:
        frame_ids: unique object/sky-pair identifiers, sorted by mid-exposure
            time. For ordinary one-sky-per-object lists these equal OBJFRAME.
        orders: echelle orders present in every frame.
        wave: ``(n_pixels, n_orders)`` reference wavelengths in angstroms.
        dv_pix_kms: ``(n_orders,)`` velocity per reference pixel.
        obj / sky: ``(n_frames, n_pixels, n_orders)`` resampled flux. ``sky``
            is None when the reduction carried no sky path.
        noise_fraction: ``(n_frames, n_orders)`` fractional pixel noise
            measured on the native grid. Resampling correlates neighbouring
            pixels, which biases the adjacent-difference estimate low.
        time_jd / sky_time_jd: mid-exposure times. The sky is a separate
            exposure, typically 1-6 minutes from the object frame.
        airmass, instmode, meta: per-frame metadata from the headers.
    """

    frame_ids: tuple[str, ...]
    orders: tuple[int, ...]
    wave: NDArray
    dv_pix_kms: NDArray
    obj: NDArray
    sky: NDArray | None
    noise_fraction: NDArray
    time_jd: NDArray
    sky_time_jd: NDArray
    airmass: NDArray
    instmode: str
    fsr_cut: float
    meta: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)

    @property
    def n_orders(self) -> int:
        return len(self.orders)

    @property
    def n_pixels(self) -> int:
        return self.wave.shape[0]

    @property
    def elapsed_hours(self) -> NDArray:
        return (self.time_jd - np.nanmin(self.time_jd)) * 24.0

    def order_index(self, order: int) -> int:
        return self.orders.index(int(order))

    def summary(self) -> str:
        return (
            f"{self.n_frames} frames x {self.n_orders} orders x {self.n_pixels} px, "
            f"{self.instmode}, fsr{self.fsr_cut:g}, "
            f"{self.wave[0].min():.0f}-{self.wave[-1].max():.0f} A, "
            f"dv/pix {self.dv_pix_kms.min():.2f}-{self.dv_pix_kms.max():.2f} km/s, "
            f"span {self.elapsed_hours.max():.2f} h, "
            f"sky {'yes' if self.sky is not None else 'no'}"
        )


def _mid_time_jd(header: fits.Header | dict[str, Any]) -> float:
    from astropy import units as u
    from astropy.time import Time

    date = header.get("DATE-OBS")
    start, end = header.get("UT-STR"), header.get("UT-END")
    if not (date and start and end):
        return float("nan")
    t0 = Time(f"{date}T{start}", scale="utc")
    t1 = Time(f"{date}T{end}", scale="utc")
    if t1 < t0:
        t1 = t1 + 1.0 * u.day
    return float((t0 + 0.5 * (t1 - t0)).jd)


def _wavelength(header: fits.Header, size: int) -> NDArray:
    step = float(header.get("CDELT1", header.get("CD1_1")))
    pixel = np.arange(size, dtype=float)
    return float(header["CRVAL1"]) + step * (pixel - float(header.get("CRPIX1", 1.0)) + 1.0)


def _pair_noise_fraction(flux: NDArray) -> float:
    """Robust fractional pixel noise from adjacent differences."""
    flux = np.asarray(flux, dtype=float)
    good = np.isfinite(flux) & (flux > 0)
    if np.count_nonzero(good) < 20:
        return float("nan")
    scale = float(np.nanmedian(flux[good]))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    normalized = flux / scale
    pair = np.isfinite(normalized[1:]) & np.isfinite(normalized[:-1])
    difference = (normalized[1:] - normalized[:-1])[pair]
    if difference.size < 20:
        return float("nan")
    median = np.median(difference)
    return float(1.4826 * np.median(np.abs(difference - median)) / np.sqrt(2.0))


def _index_frame(frame_dir: Path) -> dict[str, dict[tuple[float, int], Path]]:
    index: dict[str, dict[tuple[float, int], Path]] = {"obj": {}, "sky": {}}
    for path in frame_dir.glob("*.fits"):
        match = _SKY_PATTERN.search(path.name)
        if match:
            index["sky"][(float(match.group(2)), int(match.group(1)))] = path
            continue
        match = _OBJ_PATTERN.search(path.name)
        if match:
            index["obj"][(float(match.group(2)), int(match.group(1)))] = path
    return index


def frame_ids_from_reductions(reductions: list[Reduction]) -> tuple[str, ...]:
    """Return stable IDs for reduction *pairs* without changing OBJFRAME.

    Decanter permits one object exposure to be reduced against more than one
    sky exposure. The wavecal needs a unique row per pair, so duplicate object
    names are disambiguated with SKYFRAME; the object and sky metadata are
    unchanged.
    """
    rows = []
    for index, reduction in enumerate(reductions):
        meta = reduction.meta
        explicit = str(meta.get("SERIESID", "")).strip()
        fallback = (reduction.obj_path.stem if reduction.obj_path is not None
                    else f"frame_{index:04d}")
        obj = str(meta.get("OBJFRAME", fallback)).strip() or fallback
        sky = str(meta.get("SKYFRAME", "")).strip()
        rows.append((explicit, obj, sky))

    base_counts: dict[str, int] = {}
    for explicit, obj, _ in rows:
        base = explicit or obj
        base_counts[base] = base_counts.get(base, 0) + 1

    used: dict[str, int] = {}
    result = []
    for explicit, obj, sky in rows:
        base = explicit or obj
        candidate = base
        if not explicit and base_counts[base] > 1:
            candidate = f"{obj}__{sky}" if sky else f"{obj}__pair"
        occurrence = used.get(candidate, 0) + 1
        used[candidate] = occurrence
        if occurrence > 1:
            candidate = f"{candidate}__{occurrence:02d}"
        result.append(candidate)
    if len(set(result)) != len(result):
        raise ValueError("could not construct unique time-series frame identifiers")
    return tuple(result)


def from_reductions(
    reductions: list[Reduction],
    *,
    fsr_cut: float | None = None,
    orders: tuple[int, ...] | None = None,
    n_pixels: int | None = None,
) -> Series:
    """Put in-memory Decanter reductions on the wavecal reference grid.

    Equivalent to writing every :class:`~decanter.Reduction` to disk and
    calling :func:`load_series`, without the round trip through disk inside
    :func:`decanter.reduce_many`.
    """
    if not reductions:
        raise ValueError("from_reductions() needs at least one reduction")

    available_cuts = sorted({cut for reduction in reductions for cut, _ in reduction.obj})
    cut = float(fsr_cut) if fsr_cut is not None else max(available_cuts)
    if cut not in available_cuts:
        raise ValueError(f"fsr_cut {cut} not present; available: {available_cuts}")

    common_orders: set[int] | None = None
    for reduction in reductions:
        present = {order for local_cut, order in reduction.obj if local_cut == cut}
        common_orders = present if common_orders is None else common_orders & present
    order_list = tuple(sorted(common_orders or ()))
    if orders is not None:
        requested = set(orders)
        order_list = tuple(order for order in order_list if order in requested)
    if not order_list:
        raise ValueError("no echelle order is present in every reduction")

    records: list[dict[str, Any]] = []
    series_ids = frame_ids_from_reductions(reductions)
    for index, (reduction, series_id) in enumerate(zip(reductions, series_ids, strict=True)):
        meta = dict(reduction.meta)
        fallback = reduction.obj_path.stem if reduction.obj_path is not None else f"frame_{index:04d}"
        obj_frame = str(meta.get("OBJFRAME", fallback))
        records.append(
            {
                "reduction": reduction,
                "frame_id": series_id,
                "obj_frame": obj_frame,
                "time_jd": _mid_time_jd(meta),
                "sky_frame": meta.get("SKYFRAME"),
                "airmass": float(meta.get("AIRMASS", np.nan)),
                "instmode": str(meta.get("INSTMODE", "")),
                "meta": meta,
            }
        )
    records.sort(key=lambda row: np.inf if np.isnan(row["time_jd"]) else row["time_jd"])
    frame_ids = [row["frame_id"] for row in records]
    time_by_id = {row["obj_frame"]: row["time_jd"] for row in records}

    low = np.full(len(order_list), -np.inf)
    high = np.full(len(order_list), np.inf)
    native = np.full(len(order_list), np.inf)
    for row in records:
        reduction = row["reduction"]
        for j, order in enumerate(order_list):
            spec = reduction.obj[(cut, order)]
            wavelength = np.asarray(spec.wavelength, dtype=float)
            low[j] = max(low[j], float(np.nanmin(wavelength)))
            high[j] = min(high[j], float(np.nanmax(wavelength)))
            native[j] = min(native[j], wavelength.size)
    if np.any(low >= high):
        bad = [order for order, lo, hi in zip(order_list, low, high) if lo >= hi]
        raise ValueError(f"no common wavelength coverage for orders {bad}")

    size = int(n_pixels) if n_pixels is not None else int(np.max(native))
    if size < 2:
        raise ValueError("the wavecal reference grid needs at least two pixels")
    inset = 1.0e-10
    wave = np.empty((size, len(order_list)), dtype=float)
    for j in range(len(order_list)):
        wave[:, j] = np.exp(
            np.linspace(np.log(low[j] * (1.0 + inset)), np.log(high[j] * (1.0 - inset)), size)
        )
    dv_pix = C_KMS * (np.log(wave[-1]) - np.log(wave[0])) / (size - 1)

    n_frames = len(records)
    obj = np.full((n_frames, size, len(order_list)), np.nan, dtype=np.float32)
    has_sky = all(row["reduction"].sky is not None for row in records)
    sky = np.full_like(obj, np.nan) if has_sky else None
    noise = np.full((n_frames, len(order_list)), np.nan)

    def _ascending(spec):
        wavelength = np.asarray(spec.wavelength, dtype=float)
        flux = np.asarray(spec.flux, dtype=float)
        if wavelength[0] > wavelength[-1]:
            return wavelength[::-1], flux[::-1]
        return wavelength, flux

    for i, row in enumerate(records):
        reduction = row["reduction"]
        for j, order in enumerate(order_list):
            spec = reduction.obj[(cut, order)]
            native_wave, native_flux = _ascending(spec)
            obj[i, :, j] = np.interp(
                wave[:, j], native_wave, native_flux, left=np.nan, right=np.nan
            )
            noise[i, j] = _pair_noise_fraction(native_flux)
            if sky is not None and reduction.sky is not None:
                sky_spec = reduction.sky.get((cut, order))
                if sky_spec is not None:
                    sky_wave, sky_flux = _ascending(sky_spec)
                    sky[i, :, j] = np.interp(
                        wave[:, j], sky_wave, sky_flux, left=np.nan, right=np.nan
                    )

    return Series(
        frame_ids=tuple(frame_ids),
        orders=order_list,
        wave=wave,
        dv_pix_kms=dv_pix,
        obj=obj,
        sky=sky,
        noise_fraction=noise,
        time_jd=np.array([row["time_jd"] for row in records], dtype=float),
        sky_time_jd=np.array(
            [time_by_id.get(row["sky_frame"], np.nan) for row in records], dtype=float
        ),
        airmass=np.array([row["airmass"] for row in records], dtype=float),
        instmode=records[0]["instmode"],
        fsr_cut=cut,
        meta=[row["meta"] for row in records],
    )


def load_series(
    reduction_dir: str | Path,
    *,
    fsr_cut: float | None = None,
    orders: tuple[int, ...] | None = None,
    n_pixels: int | None = None,
) -> Series:
    """Load every frame under ``reduction_dir`` onto a common reference grid.

    Args:
        reduction_dir: a directory of per-frame subdirectories, as written by
            :meth:`decanter.Reduction.write_to`.
        fsr_cut: which FSR cut to read; None takes the widest present.
        orders: restrict to these echelle orders; None takes every order
            present in every frame.
        n_pixels: samples per order on the reference grid; None derives it as
            ``max_order(min_frame(N_native))``.

    Raises:
        FileNotFoundError: no frame subdirectory holds a readable spectrum.
    """
    reduction_dir = Path(reduction_dir)
    frame_dirs = sorted(p for p in reduction_dir.iterdir() if p.is_dir())
    indexed = [(d, _index_frame(d)) for d in frame_dirs]
    indexed = [(d, i) for d, i in indexed if i["obj"]]
    if not indexed:
        raise FileNotFoundError(f"no reduced spectra under {reduction_dir}")

    available_cuts = sorted({cut for _, index in indexed for cut, _ in index["obj"]})
    cut = float(fsr_cut) if fsr_cut is not None else max(available_cuts)
    if cut not in available_cuts:
        raise ValueError(f"fsr_cut {cut} not present; available: {available_cuts}")

    common_orders = None
    for _, index in indexed:
        present = {order for c, order in index["obj"] if c == cut}
        common_orders = present if common_orders is None else (common_orders & present)
    order_list = tuple(sorted(common_orders or ()))
    if orders is not None:
        order_list = tuple(m for m in order_list if m in set(orders))
    if not order_list:
        raise ValueError("no echelle order is present in every frame")

    # --- pass 1: headers, times, and the native wavelength extent ----------
    records = []
    low = np.full(len(order_list), -np.inf)
    high = np.full(len(order_list), np.inf)
    native = np.full(len(order_list), np.inf)
    for frame_dir, index in indexed:
        header = fits.getheader(index["obj"][(cut, order_list[0])])
        obj_frame = str(header.get("OBJFRAME", frame_dir.name))
        frame_id = str(header.get("SERIESID", obj_frame))
        records.append(
            {
                "dir": frame_dir,
                "index": index,
                "frame_id": frame_id,
                "obj_frame": obj_frame,
                "time_jd": _mid_time_jd(header),
                "sky_frame": header.get("SKYFRAME"),
                "airmass": float(header.get("AIRMASS", np.nan)),
                "instmode": str(header.get("INSTMODE", "")),
                "header": header,
            }
        )
        for j, order in enumerate(order_list):
            local = fits.getheader(index["obj"][(cut, order)])
            size = int(local["NAXIS1"])
            wave = _wavelength(local, size)
            low[j] = max(low[j], wave[0])
            high[j] = min(high[j], wave[-1])
            native[j] = min(native[j], size)

    records.sort(key=lambda row: (np.inf if np.isnan(row["time_jd"]) else row["time_jd"]))
    time_by_id = {row["obj_frame"]: row["time_jd"] for row in records}

    size = int(n_pixels) if n_pixels else int(np.max(native))
    # The endpoints are exactly a native sample of the narrowest frame, so
    # rounding can push the first or last reference sample an ulp outside it
    # and np.interp returns NaN. The inset of ~1e-6 A avoids that.
    inset = 1.0e-10
    wave = np.empty((size, len(order_list)), dtype=float)
    for j in range(len(order_list)):
        wave[:, j] = np.exp(
            np.linspace(np.log(low[j] * (1.0 + inset)), np.log(high[j] * (1.0 - inset)), size)
        )
    # Read back off the grid, so the inset above is included.
    dv_pix = C_KMS * (np.log(wave[-1, :]) - np.log(wave[0, :])) / (size - 1)

    # --- pass 2: resample -------------------------------------------------
    n_frames = len(records)
    obj = np.full((n_frames, size, len(order_list)), np.nan, dtype=np.float32)
    has_sky = all(row["index"]["sky"] for row in records)
    sky = np.full_like(obj, np.nan) if has_sky else None
    noise = np.full((n_frames, len(order_list)), np.nan)

    for i, row in enumerate(records):
        for j, order in enumerate(order_list):
            path = row["index"]["obj"][(cut, order)]
            data = np.asarray(fits.getdata(path), dtype=float).squeeze()
            native_wave = _wavelength(fits.getheader(path), data.size)
            obj[i, :, j] = np.interp(wave[:, j], native_wave, data, left=np.nan, right=np.nan)
            noise[i, j] = _pair_noise_fraction(data)
            if sky is not None:
                sky_path = row["index"]["sky"].get((cut, order))
                if sky_path is None:
                    continue
                sky_data = np.asarray(fits.getdata(sky_path), dtype=float).squeeze()
                sky_wave = _wavelength(fits.getheader(sky_path), sky_data.size)
                sky[i, :, j] = np.interp(
                    wave[:, j], sky_wave, sky_data, left=np.nan, right=np.nan
                )

    return Series(
        frame_ids=tuple(row["frame_id"] for row in records),
        orders=order_list,
        wave=wave,
        dv_pix_kms=dv_pix,
        obj=obj,
        sky=sky,
        noise_fraction=noise,
        time_jd=np.array([row["time_jd"] for row in records], dtype=float),
        sky_time_jd=np.array(
            [time_by_id.get(row["sky_frame"], np.nan) for row in records], dtype=float
        ),
        airmass=np.array([row["airmass"] for row in records], dtype=float),
        instmode=records[0]["instmode"],
        fsr_cut=cut,
        meta=[{k: row["header"].get(k) for k in row["header"]} for row in records],
    )
