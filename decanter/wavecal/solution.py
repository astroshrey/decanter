"""The wavelength-calibration solution container and how it is applied.

A solution is one velocity per (frame, order): the offset of that order's
wavelength scale from the physical reference, positive when the observed
features sit redward of where the line list puts them. Correcting divides the
wavelength scale by ``1 + v/c``.

Applying the correction never touches the flux. decanter's 1D products carry a
linear-in-wavelength WCS,

    lambda_i = CRVAL1 + (i - CRPIX1) * CDELT1,

and dividing that by a constant ``1 + v/c`` gives

    lambda_i / (1 + v/c) = CRVAL1/(1 + v/c) + (i - CRPIX1) * CDELT1/(1 + v/c),

which is again linear, so a velocity correction is a rescaling of ``CRVAL1``
and ``CDELT1`` with no resampling. The relative waveshift path instead moves
the flux with ``scopy``/``specshift``, which interpolates and correlates
neighbouring pixels on every application.

A rescaling is not the same as a constant offset in angstroms: ``lambda``
varies by about 1% across one WINERED order, so a 2 km/s correction differs by
roughly 20 m/s between the two ends of an order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from decanter._reduction import OrderSpectrum, Reduction

# Speed of light in km/s (IAU definition), matching the units of `velocity`.
C_KMS: float = 299_792.458

#: Per-(frame, order) provenance labels used in :attr:`WavecalSolution.source`.
SOURCE_LABELS: tuple[str, ...] = (
    "telluric",      # a direct telluric measurement was accepted
    "OH",            # a direct OH measurement was accepted (tied to telluric)
    "interpolated",  # no direct reference; value came from the cross-order model
    "unavailable",   # no value could be produced
)


@dataclass(frozen=True, slots=True)
class WavecalSolution:
    """An absolute wavelength correction for a set of frames.

    Attributes:
        frame_ids: one identifier per frame, in the order of ``velocity``'s
            rows. Normally the raw object-frame stem (``"WINA00047152"``).
        orders: echelle orders, in the order of ``velocity``'s columns.
        velocity: ``(n_frames, n_orders)`` offsets in km/s. NaN where no
            solution exists for that frame and order.
        source: ``(n_frames, n_orders)`` labels from :data:`SOURCE_LABELS`,
            recording how each value was obtained.
        bracketed: ``(n_frames, n_orders)`` bool; True when an interpolated
            order lies inside the range spanned by that frame's direct
            anchors, False when it is an extrapolation.
        mode: which of the four wavecal modes produced this.
        zero_point: ``"absolute"`` (anchored to the telluric rest frame) or
            ``"relative"`` (each order centred on its own median, so only the
            time-variable part survives).
        assembly: ``"ladder"`` or ``"decomposition"``.
        oh_tie_kms: the constant added to the OH measurements to bring them
            onto the telluric scale. NaN when no OH was used, or when no
            order carried both references.
        meta: free-form provenance (config, template diagnostics, versions).
    """

    frame_ids: tuple[str, ...]
    orders: tuple[int, ...]
    velocity: NDArray
    source: NDArray
    bracketed: NDArray
    mode: str = "hybrid_refit"
    zero_point: str = "absolute"
    assembly: str = "ladder"
    oh_tie_kms: float = float("nan")
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = (len(self.frame_ids), len(self.orders))
        for name in ("velocity", "source", "bracketed"):
            array = getattr(self, name)
            if array.shape != shape:
                raise ValueError(
                    f"{name} has shape {array.shape}, expected {shape} "
                    f"({len(self.frame_ids)} frames x {len(self.orders)} orders)"
                )
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("frame_ids must be unique")
        if len(set(self.orders)) != len(self.orders):
            raise ValueError("orders must be unique")

    # ---------------------------------------------------------------- lookup

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)

    @property
    def n_orders(self) -> int:
        return len(self.orders)

    def frame_index(self, frame_id: str) -> int:
        try:
            return self.frame_ids.index(frame_id)
        except ValueError:
            raise KeyError(f"frame {frame_id!r} is not in this solution") from None

    def order_index(self, order: int) -> int:
        try:
            return self.orders.index(int(order))
        except ValueError:
            raise KeyError(f"order {order!r} is not in this solution") from None

    def velocity_for(self, frame_id: str, order: int) -> float:
        """The correction for one frame and order, in km/s (NaN if none)."""
        return float(self.velocity[self.frame_index(frame_id), self.order_index(order)])

    def source_for(self, frame_id: str, order: int) -> str:
        return str(self.source[self.frame_index(frame_id), self.order_index(order)])

    def coverage(self) -> dict[str, float]:
        """Fraction of (frame, order) cells carrying each source label."""
        total = float(self.source.size)
        return {
            label: float(np.count_nonzero(self.source == label)) / total
            for label in SOURCE_LABELS
        }

    # ----------------------------------------------------------------- apply

    def apply(
        self,
        reduction: Reduction,
        *,
        frame_id: str | None = None,
        strict: bool = False,
    ) -> Reduction:
        """Return a copy of ``reduction`` on the corrected wavelength scale.

        The flux arrays are shared with the input, not copied or resampled --
        only ``CRVAL1`` and ``CDELT1`` change.

        Args:
            reduction: one frame's reduction.
            frame_id: which row of the solution to use. Defaults to the
                reduction's ``OBJFRAME`` metadata, then to its object-path
                stem.
            strict: raise if any order in ``reduction`` has no finite
                velocity. When False (the default) such orders are passed
                through uncorrected and labelled in the output metadata.

        Returns:
            A new :class:`Reduction`. The sky spectra are corrected with the
            same per-order velocities as the object path, because both were
            written on the same dispersion solution.
        """
        if frame_id is None:
            frame_id = str(reduction.meta.get("OBJFRAME", "")) or (
                reduction.obj_path.stem if reduction.obj_path is not None else ""
            )
        row = self.frame_index(frame_id)

        uncorrected: list[int] = []

        def _rescale(
            spectra: Mapping[tuple[float, int], OrderSpectrum] | None,
        ) -> dict[tuple[float, int], OrderSpectrum] | None:
            if spectra is None:
                return None
            out: dict[tuple[float, int], OrderSpectrum] = {}
            for key, spec in spectra.items():
                order = key[1]
                try:
                    velocity = float(self.velocity[row, self.order_index(order)])
                except KeyError:
                    velocity = float("nan")
                if not np.isfinite(velocity):
                    if order not in uncorrected:
                        uncorrected.append(order)
                    out[key] = spec
                    continue
                scale = 1.0 + velocity / C_KMS
                out[key] = OrderSpectrum(
                    order=spec.order,
                    fsr_cut=spec.fsr_cut,
                    flux=spec.flux,
                    crval1=spec.crval1 / scale,
                    cdelt1=spec.cdelt1 / scale,
                    crpix1=spec.crpix1,
                )
            return out

        obj_out = _rescale(reduction.obj)
        sky_out = _rescale(reduction.sky)

        if uncorrected and strict:
            raise ValueError(
                f"frame {frame_id!r} has no wavecal solution for orders "
                f"{sorted(uncorrected)}"
            )

        meta = dict(reduction.meta)
        meta["SERIESID"] = frame_id
        meta["WAVECAL"] = self.mode
        meta["WAVECZP"] = self.zero_point
        meta["WAVECASM"] = self.assembly
        finite = self.velocity[row][np.isfinite(self.velocity[row])]
        meta["WAVECMED"] = float(np.median(finite)) if finite.size else float("nan")
        if uncorrected:
            meta["WAVECBAD"] = ",".join(str(order) for order in sorted(uncorrected))

        assert obj_out is not None
        return Reduction(
            obj_name=reduction.obj_name,
            obj_path=reduction.obj_path,
            sky_path=reduction.sky_path,
            obj=obj_out,
            sky=sky_out,
            intermediates=reduction.intermediates,
            meta=meta,
        )

    def apply_many(
        self,
        reductions: list[Reduction],
        *,
        strict: bool = False,
    ) -> list[Reduction]:
        """Apply the solution to a whole series, matching frames by id."""
        return [self.apply(r, strict=strict) for r in reductions]

    # ------------------------------------------------------------------- io

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            frame_ids=np.array(self.frame_ids, dtype="U64"),
            orders=np.asarray(self.orders, dtype=np.int32),
            velocity=self.velocity,
            source=self.source.astype("U16"),
            bracketed=self.bracketed,
            mode=np.array(self.mode),
            zero_point=np.array(self.zero_point),
            assembly=np.array(self.assembly),
            oh_tie_kms=np.array(self.oh_tie_kms),
            meta=np.array(self.meta, dtype=object),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> WavecalSolution:
        with np.load(Path(path), allow_pickle=True) as data:
            return cls(
                frame_ids=tuple(str(f) for f in data["frame_ids"]),
                orders=tuple(int(m) for m in data["orders"]),
                velocity=data["velocity"],
                source=data["source"],
                bracketed=data["bracketed"],
                mode=str(data["mode"]),
                zero_point=str(data["zero_point"]),
                assembly=str(data["assembly"]),
                oh_tie_kms=float(data["oh_tie_kms"]),
                meta=dict(data["meta"].item()),
            )

    @classmethod
    def empty(
        cls,
        frame_ids: list[str] | tuple[str, ...],
        orders: list[int] | tuple[int, ...],
        **kwargs: Any,
    ) -> WavecalSolution:
        """An all-NaN solution of the right shape, for tests and for filling in."""
        shape = (len(frame_ids), len(orders))
        return cls(
            frame_ids=tuple(frame_ids),
            orders=tuple(int(m) for m in orders),
            velocity=np.full(shape, np.nan),
            source=np.full(shape, "unavailable", dtype="U16"),
            bracketed=np.zeros(shape, dtype=bool),
            **kwargs,
        )
