"""Line-resolved diagnostic pages for the template fits.

The pages plot the fitted model over the data at a few tens of resolution
elements per panel, where a wrong LSF width, a shifted template, a missing
species or a stellar line absorbed as telluric opacity are separable by eye.
A per-order summary number is not.

The output is one multi-page PDF, ordered by echelle order and then by
wavelength within the order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

C_KMS = 299_792.458


def _panel_width_pixels(dv_pix_kms: float, resolution_element_kms: float,
                        resolution_elements: float) -> int:
    return max(60, int(round(resolution_elements * resolution_element_kms / dv_pix_kms)))


def telluric_template_pdf(
    series,
    model,
    path: str | Path,
    *,
    support: NDArray | None = None,
    resolution_element_kms: float = 4.41,
    panel_resolution_elements: float = 60.0,
    panels_per_page: int = 4,
    title: str = "",
) -> Path:
    """Write the per-order telluric fit as a long, zoomed-in PDF.

    Args:
        series: the loaded :class:`Series`.
        model: the :class:`TelluricModel` from the fit.
        path: output PDF path.
        support: optional ``(n_pixels, n_orders)`` CCF support mask, shaded on
            the panels so it is obvious which pixels drive the measurement.
        panel_resolution_elements: how much spectrum each panel covers. Sixty
            resolution elements is about 250 pixels in HIRES, so a line four
            pixels wide spans several points on the page.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = model.target
    if target is None:
        raise ValueError("the model carries no fitted spectrum (target is None)")

    with PdfPages(path) as pdf:
        # ---------------- summary page -----------------------------------
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), constrained_layout=True,
                                 gridspec_kw={"height_ratios": [1, 1.4]})
        orders = np.asarray(series.orders)
        axes[0].bar(orders, model.template_rms, color="#1f77b4")
        axes[0].axhline(0.05, color="#d62728", ls="--", lw=1.2,
                        label="telluric-rich threshold")
        axes[0].set_ylabel("template RMS")
        axes[0].set_xlabel("echelle order")
        axes[0].legend(frameon=False, fontsize=8)
        axes[0].set_title(title or "telluric template fit", fontsize=12)

        rows = [["order", "family", "LSF FWHM", "implied R", "tpl RMS", "resid RMS", "rich"]]
        for j, order in enumerate(orders):
            rows.append([
                str(order), str(model.family[j])[:15],
                f"{model.lsf_fwhm_kms[j]:.2f} km/s",
                f"{C_KMS / model.lsf_fwhm_kms[j]:.0f}" if model.lsf_fwhm_kms[j] > 0 else "-",
                f"{model.template_rms[j]:.3f}", f"{model.residual_rms[j]:.4f}",
                "yes" if model.template_rms[j] >= 0.05 else "no",
            ])
        axes[1].axis("off")
        table = axes[1].table(cellText=rows[1:], colLabels=rows[0], loc="center",
                              cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1, 0.85)
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- per-order zoomed pages --------------------------
        for j, order in enumerate(orders):
            width = _panel_width_pixels(float(series.dv_pix_kms[j]),
                                        resolution_element_kms,
                                        panel_resolution_elements)
            starts = list(range(0, series.n_pixels, width))
            for page_start in range(0, len(starts), panels_per_page):
                block = starts[page_start:page_start + panels_per_page]
                fig, axes = plt.subplots(len(block), 1, figsize=(11, 8.5),
                                         constrained_layout=True)
                axes = np.atleast_1d(axes)
                for ax, start in zip(axes, block):
                    stop = min(start + width, series.n_pixels)
                    sl = slice(start, stop)
                    wave = series.wave[sl, j] / 10.0            # nm
                    ax.plot(wave, target[j][sl], color="0.35", lw=0.8, label="median spectrum")
                    ax.plot(wave, model.fitted_template[sl, j], color="#d62728", lw=1.1,
                            label="ExoJAX fit")
                    residual = target[j][sl] - model.fitted_template[sl, j]
                    ax.plot(wave, 0.30 + residual, color="#2ca02c", lw=0.6,
                            label="residual + 0.30")
                    ax.axhline(0.30, color="#2ca02c", lw=0.4, ls=":")
                    if support is not None:
                        inside = support[sl, j]
                        if inside.any():
                            ax.fill_between(wave, 0, 1, where=inside, color="#1f77b4",
                                            alpha=0.07, transform=ax.get_xaxis_transform(),
                                            step="mid")
                    ax.set_ylim(0.05, 1.25)
                    ax.set_xlim(wave.min(), wave.max())
                    ax.tick_params(labelsize=7)
                    # Absolute wavelengths on every panel; matplotlib's offset
                    # notation ("+1.33e3") is unreadable page by page.
                    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
                    ax.set_ylabel("norm. flux", fontsize=7)
                for ax in axes[len(block):]:
                    ax.set_visible(False)
                axes[-1].set_xlabel("vacuum wavelength (nm)", fontsize=8)
                axes[0].legend(frameon=False, fontsize=6.5, ncols=3, loc="lower left")
                axes[0].set_title(
                    f"m{order}   {model.family[j]}   "
                    f"LSF FWHM {model.lsf_fwhm_kms[j]:.2f} km/s "
                    f"(R={C_KMS / model.lsf_fwhm_kms[j]:.0f})   "
                    f"template RMS {model.template_rms[j]:.3f}   "
                    f"{'TELLURIC-RICH' if model.template_rms[j] >= 0.05 else 'line-poor'}",
                    fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)
    return path


def airglow_template_pdf(
    series,
    model,
    path: str | Path,
    *,
    line_wave_angstrom: NDArray | None = None,
    resolution_element_kms: float = 4.41,
    panel_resolution_elements: float = 60.0,
    panels_per_page: int = 4,
    title: str = "",
) -> Path:
    """Write the per-order OH airglow fit as a long, zoomed-in PDF.

    Detected lines are ticked. The shaded band is the CCF support: the windows
    around detected lines that drive the measurement. Pixels outside it do not
    enter the CCF.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    orders = np.asarray(series.orders)

    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), constrained_layout=True,
                                 gridspec_kw={"height_ratios": [1, 1.4]})
        axes[0].bar(orders, model.line_count, color="#2ca02c")
        axes[0].axhline(4, color="#d62728", ls="--", lw=1.2, label="OH-rich threshold")
        axes[0].set_ylabel("detected OH lines")
        axes[0].set_xlabel("echelle order")
        axes[0].legend(frameon=False, fontsize=8)
        axes[0].set_title(
            (title or "OH airglow fit")
            + f"   T_rot = {model.rotational_temperature_k:.0f} K", fontsize=12)

        rows = [["order", "lines", "LSF FWHM", "resid RMS", "support px", "rich"]]
        for j, order in enumerate(orders):
            rows.append([str(order), str(int(model.line_count[j])),
                         f"{model.lsf_fwhm_kms[j]:.2f} km/s",
                         f"{model.residual_rms[j]:.4f}",
                         str(int(model.support[:, j].sum())),
                         "yes" if model.line_count[j] >= 4 else "no"])
        axes[1].axis("off")
        table = axes[1].table(cellText=rows[1:], colLabels=rows[0], loc="center",
                              cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1, 0.85)
        pdf.savefig(fig)
        plt.close(fig)

        for j, order in enumerate(orders):
            width = _panel_width_pixels(float(series.dv_pix_kms[j]),
                                        resolution_element_kms, panel_resolution_elements)
            starts = list(range(0, series.n_pixels, width))
            for page_start in range(0, len(starts), panels_per_page):
                block = starts[page_start:page_start + panels_per_page]
                fig, axes = plt.subplots(len(block), 1, figsize=(11, 8.5),
                                         constrained_layout=True)
                axes = np.atleast_1d(axes)
                for ax, start in zip(axes, block):
                    stop = min(start + width, series.n_pixels)
                    sl = slice(start, stop)
                    wave = series.wave[sl, j] / 10.0
                    ax.plot(wave, model.target[j][sl], color="0.35", lw=0.8,
                            label="median sky")
                    ax.plot(wave, model.fitted_model[j][sl], color="#d62728", lw=1.0,
                            label="ExoJAX OH + baseline")
                    inside = model.support[sl, j]
                    if inside.any():
                        ax.fill_between(wave, 0, 1, where=inside, color="#2ca02c",
                                        alpha=0.10, transform=ax.get_xaxis_transform(),
                                        step="mid")
                    if line_wave_angstrom is not None:
                        here = line_wave_angstrom[
                            (line_wave_angstrom > series.wave[start, j])
                            & (line_wave_angstrom < series.wave[stop - 1, j])]
                        for centre in here / 10.0:
                            ax.axvline(centre, color="#1f77b4", lw=0.4, alpha=0.35)
                    top = float(np.nanpercentile(model.target[j][sl], 99.8))
                    ax.set_ylim(-0.05, max(0.2, top * 1.25))
                    ax.set_xlim(wave.min(), wave.max())
                    ax.tick_params(labelsize=7)
                    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
                    ax.set_ylabel("scaled sky", fontsize=7)
                for ax in axes[len(block):]:
                    ax.set_visible(False)
                axes[-1].set_xlabel("vacuum wavelength (nm)", fontsize=8)
                axes[0].legend(frameon=False, fontsize=6.5, ncols=3, loc="upper left")
                axes[0].set_title(
                    f"m{order}   {model.family[j]}   "
                    f"LSF FWHM {model.lsf_fwhm_kms[j]:.2f} km/s   "
                    f"{int(model.line_count[j])} detected OH lines   "
                    f"{'OH-RICH' if model.line_count[j] >= 4 else 'OH-poor'}", fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)
    return path
