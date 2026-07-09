"""The central aperture data structure.

WARP equivalent: ``warp/aperture.py:apertureSet`` class (lines 12-149 for
the public API; lines 240-300 for the inner ``aperture`` class).

An :class:`ApertureSet` carries every echelle order's trace polynomial
plus 2D masking helpers (`apmask_array`, `slitcoord_array`) consumed by
s02 (cosmic ray), s03 (apscatter), s06 (transform_cut), s08 (extract_1d).

Phase 1 always loads the set from a WARP ``database/ap*`` file via
:func:`decanter.io.apdb.parse` — we don't re-derive traces (see
PLAN_FULL.md §Validation binding constraint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev, legendre
from numpy.typing import NDArray

from decanter.io.apdb import (
    FUNCTION_TYPE_CHEBYSHEV,
    FUNCTION_TYPE_LEGENDRE,
    ApertureEntry,
    parse as parse_apdb,
)


@dataclass
class Aperture:
    """One echelle order's aperture: the database entry + computed trace.

    The trace ``x(y)`` is materialized lazily and cached on first access.

    Note: this dataclass does not use ``slots=True`` because
    :func:`functools.cached_property` requires ``__dict__`` for its cache.
    """

    entry: ApertureEntry
    array_length: int = 2048

    @cached_property
    def trace_x(self) -> NDArray[np.float64]:
        """Trace ``x(y)`` evaluated at ``y = 1, 2, ..., array_length``.

        Returns a 1-D float64 array of length ``array_length``.

        WARP equivalent: ``warp/aperture.py:aperture.calculateTrace`` (lines 262-280).
        Same normalization (``y_norm = (2y - y_min - y_max) / (y_max - y_min)``)
        and same polynomial-summation convention; uses
        ``numpy.polynomial.chebyshev.chebval`` / ``legendre.legval`` rather
        than the hand-rolled recurrences in WARP, which is numerically
        equivalent and ~10× faster.

        Note: this evaluates the polynomial without clamping y to
        ``[y_min, y_max]`` — extrapolation matches WARP's
        ``apmaskArray`` mask-construction behavior. For IRAF-extract
        semantics (``ap_cveval`` clamps before eval), use
        :attr:`trace_x_clamped` instead.
        """
        y = np.arange(1, self.array_length + 1, dtype=np.float64)
        y_norm = (2.0 * y - self.entry.y_min - self.entry.y_max) / (
            self.entry.y_max - self.entry.y_min
        )
        coeffs = np.asarray(self.entry.coefficients, dtype=np.float64)
        if self.entry.function_type == FUNCTION_TYPE_CHEBYSHEV:
            offset = chebyshev.chebval(y_norm, coeffs)
        elif self.entry.function_type == FUNCTION_TYPE_LEGENDRE:
            offset = legendre.legval(y_norm, coeffs)
        else:
            raise ValueError(
                f"unknown function type {self.entry.function_type} for order {self.entry.order}"
            )
        return self.entry.center_x + offset

    @cached_property
    def trace_x_clamped(self) -> NDArray[np.float64]:
        """Trace ``x(y)`` with y clamped to ``[y_min, y_max]`` per IRAF
        ``ap_cveval`` (``noao/twodspec/apextract/apcveval.x:17``).

        The Chebyshev/Legendre fit is only valid over its sample range;
        outside that range the polynomial can swing wildly. IRAF's
        ``ap_extract`` (and every other apall consumer) clamps to the
        endpoints, effectively treating the trace as a constant beyond
        the fit range. Use this for s08 extraction; use
        :attr:`trace_x` for s02 mask construction (WARP's
        ``apmaskArray`` extrapolates).
        """
        y = np.arange(1, self.array_length + 1, dtype=np.float64)
        y_clamped = np.clip(y, self.entry.y_min, self.entry.y_max)
        y_norm = (2.0 * y_clamped - self.entry.y_min - self.entry.y_max) / (
            self.entry.y_max - self.entry.y_min
        )
        coeffs = np.asarray(self.entry.coefficients, dtype=np.float64)
        if self.entry.function_type == FUNCTION_TYPE_CHEBYSHEV:
            offset = chebyshev.chebval(y_norm, coeffs)
        elif self.entry.function_type == FUNCTION_TYPE_LEGENDRE:
            offset = legendre.legval(y_norm, coeffs)
        else:
            raise ValueError(
                f"unknown function type {self.entry.function_type} for order {self.entry.order}"
            )
        return self.entry.center_x + offset


@dataclass(slots=True)
class ApertureSet:
    """All echelle orders' apertures + 2D masking helpers.

    WARP equivalent: ``warp/aperture.py:apertureSet`` class.
    """

    apertures: dict[int, Aperture] = field(default_factory=dict)
    array_length: int = 2048

    @property
    def echelle_orders(self) -> tuple[int, ...]:
        """Echelle order numbers in ascending order."""
        return tuple(sorted(self.apertures.keys()))

    @classmethod
    def load(cls, apdb_path: Path | str, *, array_length: int = 2048,
             selected_orders: tuple[int, ...] | None = None) -> ApertureSet:
        """Load an aperture set from a WARP ``database/ap*`` file.

        Args:
            apdb_path: path to the IRAF aperture database file.
            array_length: detector axis length (default 2048 for WINERED).
            selected_orders: if given, only these orders are loaded.

        Returns:
            An :class:`ApertureSet` ready for masking.
        """
        entries = parse_apdb(apdb_path)
        apertures = {
            order: Aperture(entry=entry, array_length=array_length)
            for order, entry in entries.items()
            if selected_orders is None or order in selected_orders
        }
        return cls(apertures=apertures, array_length=array_length)

    def apmask_array(
        self,
        *,
        low_lim: float | None = None,
        upp_lim: float | None = None,
        margin: int = 10,
    ) -> NDArray[np.int32]:
        """Per-pixel echelle-order label image.

        For each pixel at ``(x, y)``, the value is the echelle order ``m``
        if the pixel lies inside that order's aperture (i.e. the absolute
        cross-dispersion residue against the trace is within the order's
        aperture half-width), otherwise 0.

        Args:
            low_lim: optional override for every order's lower bound.
                **Note:** WARP's check uses the *absolute* residue, so the
                lower bound is effectively unused (any negative ``low_lim``
                makes ``low_lim < |residue|`` trivially true). Preserved
                here for parity with the WARP call signature.
            upp_lim: optional override for every order's upper bound (this
                IS the effective aperture half-width).
            margin: pixels at each frame edge to zero out.

        Returns:
            ``(array_length, array_length)`` int32 array; value 0 for
            inter-order pixels, ``m`` for pixels inside order ``m``.

        WARP equivalent: ``apertureSet.apmaskArray`` (lines 84-118).
        """
        L = self.array_length
        mask = np.zeros((L, L), dtype=np.int32)
        x = np.arange(1, L + 1, dtype=np.float64)
        x_grid = np.broadcast_to(x[None, :], (L, L))

        for m in self.echelle_orders:
            ap = self.apertures[m]
            ap_low = ap.entry.low if low_lim is None else low_lim
            ap_high = ap.entry.high if upp_lim is None else upp_lim
            trace_x_grid = np.broadcast_to(ap.trace_x[:, None], (L, L))
            residue_abs = np.abs(x_grid - trace_x_grid)
            in_ap = (ap_low < residue_abs) & (residue_abs < ap_high)
            mask[in_ap] = m

        # Match WARP's quirky off-by-one: WARP uses 1-indexed pixel arrays
        # x ∈ {1..L} and zeros pixels where `x < margin` or `x > L - margin`.
        # That's:
        #   lower edge: 1-indexed x ∈ {1..margin-1}  →  0-indexed {0..margin-2}
        #   upper edge: 1-indexed x ∈ {L-margin+1..L}  →  0-indexed {L-margin..L-1}
        # So the lower edge masks (margin - 1) pixels and the upper edge masks
        # margin pixels. Asymmetric but byte-for-byte WARP-compatible.
        if margin > 0:
            mask[: margin - 1, :] = 0
            mask[L - margin:, :] = 0
            mask[:, : margin - 1] = 0
            mask[:, L - margin:] = 0

        return mask

    def slitcoord_array(
        self,
        *,
        low_lim: float | None = None,
        upp_lim: float | None = None,
        inter_order_value: float = -10000.0,
    ) -> NDArray[np.float64]:
        """Per-pixel signed slit coordinate (cross-dispersion distance from trace).

        Inside any aperture, value is the signed residue ``x_pixel - x_trace``.
        Outside any aperture, value is ``inter_order_value`` (default -10000).

        Args:
            low_lim: same override semantics as :meth:`apmask_array`.
            upp_lim: same override semantics as :meth:`apmask_array`.
            inter_order_value: sentinel for "outside any aperture."

        Returns:
            ``(array_length, array_length)`` float64 array.

        WARP equivalent: ``apertureSet.slitcoordArray`` (lines 120-149).
        """
        L = self.array_length
        slitcoord = np.full((L, L), inter_order_value, dtype=np.float64)
        x = np.arange(1, L + 1, dtype=np.float64)
        x_grid = np.broadcast_to(x[None, :], (L, L))

        # WARP uses `+=` here, not `=` (warp/aperture.py:147). This matters
        # for pixels that fall into multiple overlapping apertures (rare,
        # but happens with order 158's very wide aperture in HIRES-Y). For
        # parity we replicate the additive semantics exactly:
        #   pixel starts at inter_order_value
        #   each containing aperture adds (residue - inter_order_value)
        #   so a 1-aperture pixel ends at `residue`; a 2-aperture pixel ends
        #   at `residue1 + residue2 - inter_order_value`.
        for m in self.echelle_orders:
            ap = self.apertures[m]
            ap_low = ap.entry.low if low_lim is None else low_lim
            ap_high = ap.entry.high if upp_lim is None else upp_lim
            trace_x_grid = np.broadcast_to(ap.trace_x[:, None], (L, L))
            residue = x_grid - trace_x_grid
            residue_abs = np.abs(residue)
            in_ap = (ap_low < residue_abs) & (residue_abs < ap_high)
            slitcoord[in_ap] += residue[in_ap] - inter_order_value

        return slitcoord


def load(apdb_path: Path | str, *, array_length: int = 2048) -> ApertureSet:
    """Convenience wrapper for :meth:`ApertureSet.load`."""
    return ApertureSet.load(apdb_path, array_length=array_length)
