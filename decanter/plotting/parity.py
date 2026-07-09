"""Reusable parity-plot primitives — WARP vs decanter visual diagnostics.

Two top-level entry points:

- :func:`parity_image` — three-panel 2D figure (WARP | decanter | residual).
  Same intensity scale on the first two panels; the residual panel uses a
  symmetric diverging colormap auto-clipped to a high-percentile of the
  absolute residual so saturation/CR-edge outliers don't wash out the
  bulk-pixel structure.

- :func:`parity_spectra_grid` — grid of overplotted 1-D spectra, one
  cell per echelle order. Each cell shows WARP + decanter on the main
  y-axis (lines, color-coded) and the residual on a twin y-axis at a
  much smaller scale.

Both functions return a :class:`matplotlib.figure.Figure` so callers
can save to PDF / paste into a multipage doc / etc.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


def _robust_scale(arr: NDArray, lo_pct: float = 1.0, hi_pct: float = 99.0,
                  mask: NDArray | None = None) -> tuple[float, float]:
    """Return ``(vmin, vmax)`` from percentile clipping over finite pixels."""
    a = arr if mask is None else arr[mask]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (0.0, 1.0)
    return (float(np.percentile(a, lo_pct)), float(np.percentile(a, hi_pct)))


def _residual_scale(diff: NDArray, mask: NDArray | None = None,
                    pct: float = 99.0) -> float:
    """Return a symmetric ``vmax`` for residual display from clipped |diff|."""
    a = diff if mask is None else diff[mask]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1.0
    v = float(np.percentile(np.abs(a), pct))
    return v if v > 0 else 1.0


def parity_image(
    my_data: NDArray,
    warp_data: NDArray,
    *,
    title: str = "",
    mask: NDArray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    residual_pct: float = 99.0,
    figsize: tuple[float, float] = (15.0, 5.0),
):
    """Three-panel parity figure for a 2D image.

    Args:
        my_data: decanter's output (2D array).
        warp_data: WARP's saved equivalent (same shape).
        title: figure suptitle (e.g. ``"s05 _sscfm — TOI2109"``).
        mask: optional boolean array (True = include pixel in scaling /
            stats); pixels outside the mask are still drawn but ignored
            for percentile auto-scaling. Useful for ``flat >= 0.01`` to
            exclude inter-order regions.
        vmin, vmax: optional fixed intensity range for WARP+decanter panels.
            Defaults to the 1–99% percentile of the WARP image (mask-aware).
        residual_pct: percentile used to set the residual panel's
            symmetric ``vmax`` (default 99 = ignore the top 1% of
            outliers in the auto-scale, so bulk structure is visible).
        figsize: figure size in inches.

    Returns:
        A :class:`matplotlib.figure.Figure`. Caller is responsible for
        ``fig.savefig(...)`` / ``plt.close(fig)``.
    """
    import matplotlib.pyplot as plt

    if my_data.shape != warp_data.shape:
        raise ValueError(
            f"shape mismatch: my={my_data.shape} warp={warp_data.shape}"
        )

    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = _robust_scale(warp_data, mask=mask)
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    diff = my_data.astype(np.float64) - warp_data.astype(np.float64)
    res_v = _residual_scale(diff, mask=mask, pct=residual_pct)

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    im0 = axes[0].imshow(warp_data, vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
    axes[0].set_title("WARP")
    fig.colorbar(im0, ax=axes[0], fraction=0.04, pad=0.02)

    im1 = axes[1].imshow(my_data, vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
    axes[1].set_title("pyWARP")
    fig.colorbar(im1, ax=axes[1], fraction=0.04, pad=0.02)

    im2 = axes[2].imshow(diff, vmin=-res_v, vmax=res_v, cmap="RdBu_r", aspect="auto")
    # Stats annotation: median |Δ|, max |Δ|, bias (mask-aware).
    if mask is not None:
        d_stat = diff[mask & np.isfinite(diff)]
    else:
        d_stat = diff[np.isfinite(diff)]
    med = float(np.median(np.abs(d_stat))) if d_stat.size else float("nan")
    mx = float(np.max(np.abs(d_stat))) if d_stat.size else float("nan")
    bias = float(d_stat.mean()) if d_stat.size else float("nan")
    axes[2].set_title(
        f"pyWARP - WARP\nmed |Δ|={med:.3g}  max |Δ|={mx:.3g}  bias={bias:+.3g}\n"
        f"(res. scale = ±{res_v:.3g} ct)"
    )
    fig.colorbar(im2, ax=axes[2], fraction=0.04, pad=0.02)

    for ax in axes:
        ax.set_xlabel("x (pixel)")
    axes[0].set_ylabel("y (pixel)")

    if title:
        fig.suptitle(title, fontsize=11)
    return fig


def parity_spectra_grid(
    orders: Sequence[int],
    my_specs: Mapping[int, NDArray],
    warp_specs: Mapping[int, NDArray],
    *,
    title: str = "",
    ncols: int = 4,
    figsize_per_cell: tuple[float, float] = (3.8, 2.4),
    residual_color: str = "tab:red",
    warp_color: str = "black",
    my_color: str = "tab:blue",
):
    """Grid of per-order overplotted 1-D spectra with a stacked residual subpanel.

    Each cell has two stacked axes that share the x-axis:

      - top (3 height units): WARP + pyWARP spectra overplotted.
      - bottom (1 height unit, i.e. 1/3 the height of the main panel):
        ``decanter − WARP`` residual, symmetric y-axis auto-scaled to the
        99th percentile of |Δ|.

    Args:
        orders: echelle orders to plot, in display order.
        my_specs: ``{order: 1d_array}`` of decanter's per-order spectra.
        warp_specs: ``{order: 1d_array}`` of WARP's per-order spectra.
            Each pair must have matching shape; orders missing from
            either dict get an empty cell.
        title: figure suptitle.
        ncols: cells per row. Rows are computed to fit ``len(orders)``.
        figsize_per_cell: (width, height) in inches per cell — both the
            main and residual subpanels live inside one cell.
        residual_color, warp_color, my_color: line colors.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(orders)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        constrained_layout=True,
    )
    outer = GridSpec(nrows, ncols, figure=fig)

    legend_attached = False
    for idx, m in enumerate(orders):
        my = my_specs.get(m)
        wp = warp_specs.get(m)
        if my is None or wp is None or my.shape != wp.shape:
            continue

        # Per-cell 2-row stack: main (3 units) over residual (1 unit), sharex.
        inner = outer[idx // ncols, idx % ncols].subgridspec(
            2, 1, height_ratios=[3, 1], hspace=0.05,
        )
        ax_main = fig.add_subplot(inner[0])
        ax_res = fig.add_subplot(inner[1], sharex=ax_main)

        x = np.arange(my.size)
        ax_main.plot(x, wp, color=warp_color, lw=0.6, label="WARP")
        ax_main.plot(x, my, color=my_color, lw=0.4, alpha=0.7, label="pyWARP")

        # Auto-scale main y from WARP (avoid blow-ups in saturated regions).
        lo, hi = _robust_scale(wp, 1.0, 99.0)
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ax_main.set_ylim(lo - pad, hi + pad)

        # Residual subpanel.
        diff = my.astype(np.float64) - wp.astype(np.float64)
        ax_res.plot(x, diff, color=residual_color, lw=0.4, alpha=0.9)
        v = _residual_scale(diff, pct=99.0)
        ax_res.set_ylim(-3 * v, 3 * v)
        ax_res.axhline(0.0, color="0.5", lw=0.3, alpha=0.5)
        ax_res.tick_params(axis="y", labelsize=6, colors=residual_color)
        ax_res.tick_params(axis="x", labelsize=6)
        # Hide x-tick labels on the main panel (shared axis with residual).
        plt.setp(ax_main.get_xticklabels(), visible=False)
        ax_main.tick_params(axis="y", labelsize=6)

        d_finite = diff[np.isfinite(diff)]
        med = float(np.median(np.abs(d_finite))) if d_finite.size else float("nan")
        mx = float(np.max(np.abs(d_finite))) if d_finite.size else float("nan")
        ax_main.set_title(f"m{m}   med|Δ|={med:.2g}  max|Δ|={mx:.2g}", fontsize=8)

        if not legend_attached:
            ax_main.legend(loc="upper right", fontsize=6)
            legend_attached = True

    if title:
        fig.suptitle(title, fontsize=11)
    return fig
