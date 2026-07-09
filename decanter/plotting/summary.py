"""Per-run pipeline-summary PDFs.

These are *not* parity-vs-WARP plots. They show decanter's own output
after each pipeline stage so the user can eyeball a reduction at a
glance.

Currently one entry point:

  - :func:`summary_2d_pdf` — six-page 2D walk-through (raw → sky-sub →
    CR-flagged → scatter-sub → flat-divided → bad-pixel-fixed) for one
    :class:`~decanter.Reduction` that was produced with
    ``save_intermediates=True``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from decanter._reduction import Reduction


def _robust_scale(arr: NDArray, lo_pct: float = 1.0, hi_pct: float = 99.0,
                  mask: NDArray | None = None) -> tuple[float, float]:
    """``(vmin, vmax)`` from percentile clipping over finite pixels.

    When ``mask`` is given, percentiles are computed only over pixels
    where the mask is True (e.g. ``flat >= 0.01`` to exclude inter-order
    divide-by-zero blowups).
    """
    a = arr if mask is None else arr[mask]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (0.0, 1.0)
    return (float(np.percentile(a, lo_pct)), float(np.percentile(a, hi_pct)))


def _stats_line(arr: NDArray, mask: NDArray | None = None) -> str:
    a = arr if mask is None else arr[mask]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "(no finite pixels)"
    return (
        f"median={float(np.median(a)):+.3g}  "
        f"std={float(np.std(a)):.3g}  "
        f"min={float(a.min()):+.3g}  "
        f"max={float(a.max()):+.3g}"
    )


def _page_image(pdf, arr: NDArray, *, title: str, subtitle: str = "",
                vmin: float | None = None, vmax: float | None = None,
                cr_mask: NDArray | None = None,
                display_mask: NDArray | None = None) -> None:
    """One full-page imshow with title + stats, optional CR-pixel overlay.

    ``display_mask`` (True = include) drives both the percentile auto-scale
    and the stats line. Pixels outside the mask are set to NaN in the
    rendered image so they appear as the colormap's "bad" color rather
    than dragging the display range.
    """
    import matplotlib.pyplot as plt

    shown = arr
    if display_mask is not None:
        shown = np.where(display_mask, arr, np.nan)

    if vmin is None or vmax is None:
        vmin_auto, vmax_auto = _robust_scale(arr, mask=display_mask)
        vmin = vmin_auto if vmin is None else vmin
        vmax = vmax_auto if vmax is None else vmax

    fig, ax = plt.subplots(figsize=(8.5, 9.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.2")
    im = ax.imshow(shown, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    if cr_mask is not None:
        ys, xs = np.where(cr_mask.astype(bool))
        ax.scatter(xs, ys, s=2.0, c="red", alpha=0.7, linewidths=0,
                   label=f"CR-flagged ({len(xs)} px)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8, frameon=True)

    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    full_title += f"\n{_stats_line(arr, mask=display_mask)}"
    ax.set_title(full_title, fontsize=10)

    pdf.savefig(fig)
    plt.close(fig)


def summary_2d_pdf(reduction: Reduction, out_pdf: Path,
                   *, title_prefix: str = "",
                   flat: NDArray | None = None) -> Path:
    """Write a six-page 2D pipeline-summary PDF for one reduction.

    Pages:
        1. raw obj frame.
        2. obj − sky (s01).
        3. obj − sky with CR-flagged pixels overlaid (s02 detection).
        4. apscatter-subtracted (s03 ``_ssc``).
        5. flat-divided (s04 ``_sscf``).
        6. bad-pixel-interpolated, final 2D (s05 ``_sscfm``).

    Args:
        reduction: :class:`Reduction` produced by :func:`decanter.reduce`
            with ``save_intermediates=True``. Required intermediates:
            ``obj_raw``, ``obj_s``, ``obj_ssc``, ``obj_sscf``,
            ``obj_sscfm``. Optional: ``cr_mask`` (page 3 overlay).
        out_pdf: destination PDF path.
        title_prefix: optional string prepended to every page title
            (e.g. the object name or frame identifier).
        flat: master flat array. When supplied, pages 5 and 6 mask
            ``flat < 0.01`` pixels (the 28% of the frame between orders
            where the flat is zero and the division blows up to e+8).
            Without this, the auto-percentile display range on those
            pages is dominated by divide-by-zero outliers and the
            science apertures wash out — see CLAUDE.md's "filter
            ``flat >= 0.01`` for meaningful diffs".

    Returns:
        The path the PDF was written to (same as ``out_pdf``).

    Raises:
        ValueError: when the required intermediates aren't populated.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    from decanter.plotting import style
    style.apply()

    it = reduction.intermediates
    missing = [
        name for name, arr in (
            ("obj_raw", it.obj_raw),
            ("obj_s", it.obj_s),
            ("obj_ssc", it.obj_ssc),
            ("obj_sscf", it.obj_sscf),
            ("obj_sscfm", it.obj_sscfm),
        ) if arr is None
    ]
    if missing:
        raise ValueError(
            "summary_2d_pdf: missing intermediates " + ", ".join(missing)
            + " — did you pass save_intermediates=True to decanter.reduce()?"
        )

    flat_mask = (flat >= 0.01) if flat is not None else None

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    pfx = f"{title_prefix} — " if title_prefix else ""
    with PdfPages(out_pdf) as pdf:
        _page_image(pdf, it.obj_raw,
                    title=f"{pfx}Page 1 — raw obj frame")
        _page_image(pdf, it.obj_s,
                    title=f"{pfx}Page 2 — obj − sky (s01)")
        _page_image(pdf, it.obj_s,
                    title=f"{pfx}Page 3 — after CR detection (s02)",
                    subtitle="background = obj − sky; red = CR-flagged pixels",
                    cr_mask=it.cr_mask)
        _page_image(pdf, it.obj_ssc,
                    title=f"{pfx}Page 4 — after apscatter subtraction (s03)")
        sub_flat = ("flat<0.01 masked (dark grey)" if flat_mask is not None
                    else "WARNING: inter-order divide-by-zero not masked — "
                         "pass flat= for sensible scaling")
        _page_image(pdf, it.obj_sscf,
                    title=f"{pfx}Page 5 — after flat-field division (s04)",
                    subtitle=sub_flat,
                    display_mask=flat_mask)
        _page_image(pdf, it.obj_sscfm,
                    title=f"{pfx}Page 6 — after bad-pixel interpolation (s05) — final 2D",
                    subtitle=sub_flat,
                    display_mask=flat_mask)

    return out_pdf
