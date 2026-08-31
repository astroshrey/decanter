"""Per-order linear-flux SVD operators matching the reference notebook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SVDPath:
    prepared: np.ndarray
    lower: dict[int, np.ndarray]
    residuals: dict[int, np.ndarray]
    valid: np.ndarray


def svd_path(matrix: np.ndarray, counts: tuple[int, ...], pixel_mask: np.ndarray) -> SVDPath:
    values = np.asarray(matrix, dtype=float)
    valid = np.isfinite(values) & pixel_mask[None, :]
    usable_rows = np.any(valid, axis=1)
    usable_columns = pixel_mask & np.any(valid[usable_rows], axis=0)
    if np.count_nonzero(usable_columns) < 3:
        raise ValueError("too few valid pixels for SVD")
    local = values[np.ix_(usable_rows, usable_columns)]
    column_fill = np.nanmedian(local, axis=0)
    global_fill = float(np.nanmedian(local))
    column_fill = np.where(np.isfinite(column_fill), column_fill, global_fill)
    prepared = np.where(np.isfinite(local), local, column_fill[None, :])
    u, singular, vt = np.linalg.svd(prepared, full_matrices=False)
    lowers: dict[int, np.ndarray] = {}
    residuals: dict[int, np.ndarray] = {}
    full_prepared = np.full_like(values, np.nan)
    full_prepared[np.ix_(usable_rows, usable_columns)] = prepared
    for count in sorted(set(counts)):
        count = min(int(count), u.shape[1])
        lower = (u[:, :count] * singular[:count]) @ vt[:count] if count else 0.0
        local_residual = prepared - lower
        full_lower = np.full_like(values, np.nan)
        full = np.full_like(values, np.nan)
        full_lower[np.ix_(usable_rows, usable_columns)] = lower
        full[np.ix_(usable_rows, usable_columns)] = local_residual
        full_lower[~valid] = np.nan
        full[~valid] = np.nan
        lowers[count] = full_lower
        residuals[count] = full
    return SVDPath(full_prepared, lowers, residuals, valid)
