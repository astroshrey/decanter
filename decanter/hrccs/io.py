"""Load calibrated Decanter products into the HRCCS axis convention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from decanter.wavecal.series import load_series


@dataclass(frozen=True)
class ScienceCube:
    frame_ids: tuple[str, ...]
    orders: np.ndarray
    wavelength_um: np.ndarray       # order, pixel
    flux: np.ndarray                # exposure, order, pixel
    time_jd_utc: np.ndarray
    metadata: list[dict]
    instmode: str
    telluric_transmission: np.ndarray | None = None


def _interp_telluric(product: Path, series) -> np.ndarray:
    with np.load(product, allow_pickle=False) as loaded:
        frame_ids = tuple(str(value) for value in loaded["frame_ids"])
        orders = tuple(int(value) for value in loaded["orders"])
        wave = np.asarray(loaded["wavelength_angstrom"], dtype=float)
        transmission = np.asarray(loaded["transmission"], dtype=float)
    frame_index = {name: index for index, name in enumerate(frame_ids)}
    order_index = {order: index for index, order in enumerate(orders)}
    out = np.full((series.n_frames, series.n_orders, series.n_pixels), np.nan)
    for i, frame_id in enumerate(series.frame_ids):
        if frame_id not in frame_index:
            continue
        for j, order in enumerate(series.orders):
            if order not in order_index:
                continue
            source_i, source_j = frame_index[frame_id], order_index[order]
            out[i, j] = np.interp(
                series.wave[:, j], wave[source_i, source_j], transmission[source_i, source_j],
                left=np.nan, right=np.nan,
            )
    return out


def load_decanter(directory: str | Path, *, fsr_cut=None, orders=(),
                  telluric_product: str | Path | None = None) -> ScienceCube:
    root = Path(directory).expanduser().resolve()
    selected = tuple(int(value) for value in orders) or None
    series = load_series(root, fsr_cut=fsr_cut, orders=selected)
    product = (Path(telluric_product).expanduser() if telluric_product
               else root / "telluric_transmission.npz")
    telluric = _interp_telluric(product, series) if product.exists() else None
    return ScienceCube(
        frame_ids=series.frame_ids,
        orders=np.asarray(series.orders, dtype=int),
        wavelength_um=np.asarray(series.wave.T, dtype=float) / 1.0e4,
        flux=np.transpose(np.asarray(series.obj, dtype=float), (0, 2, 1)),
        time_jd_utc=np.asarray(series.time_jd, dtype=float),
        metadata=series.meta,
        instmode=str(series.instmode),
        telluric_transmission=telluric,
    )
