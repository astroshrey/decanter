"""Diagnostic figures for the HRCCS pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def _style():
    plt.rcParams.update({"font.size": 10, "axes.grid": False,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True})


def _save(fig, stem: Path, formats, dpi):
    for extension in formats:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def forward_spectrum(species, wavelength_um, depths, output, formats, dpi,
                     observation_type="transit", injection_scale=1.0):
    _style()
    fig, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for wave, depth in zip(wavelength_um, depths):
        displayed = depth * injection_scale if observation_type == "eclipse" else depth
        axis.plot(wave, displayed, color="#2a6fbb", lw=0.8)
    if observation_type == "eclipse":
        ylabel = (rf"{injection_scale:g}$\times$ inverted "
                  r"$(R_p/R_\star)^2$ proxy")
        title = (f"{species} emission species-presence proxy "
                 f"(synthetic injection: {injection_scale:g}x)")
    else:
        ylabel = r"$(R_p/R_\star)^2$"
        title = f"{species} isothermal equilibrium transmission spectrum"
    axis.set(xlabel=r"Vacuum wavelength ($\mu$m)", ylabel=ylabel, title=title)
    _save(fig, Path(output) / f"{species}_forward_spectrum", formats, dpi)


def _sequence_pdf(path, wavelength_um, orders, phase, rows, labels, title,
                  orders_per_page=5, cmap="RdBu_r"):
    order = np.argsort(phase)
    with PdfPages(path) as pdf:
        for start in range(0, len(orders), orders_per_page):
            subset = range(start, min(start + orders_per_page, len(orders)))
            fig, axes = plt.subplots(len(rows), len(tuple(subset)),
                                     figsize=(4.2 * len(tuple(subset)), 2.5 * len(rows)),
                                     squeeze=False, constrained_layout=True, sharey=True)
            for col, index in enumerate(subset):
                axes[0, col].set_title(f"Order {orders[index]}")
                for row, values in enumerate(rows):
                    image = values[order, index]
                    finite = image[np.isfinite(image)]
                    span = max(float(np.nanpercentile(np.abs(finite), 99)) if finite.size else 1.0,
                               1.0e-8)
                    axes[row, col].pcolormesh(wavelength_um[index], phase[order], image,
                                              shading="auto", cmap=cmap,
                                              vmin=-span, vmax=span, rasterized=True)
                    if col == 0:
                        axes[row, col].set_ylabel(f"{labels[row]}\nOrbital phase")
                    if row == len(rows) - 1:
                        axes[row, col].set_xlabel(r"Wavelength ($\mu$m)")
            fig.suptitle(title)
            pdf.savefig(fig, dpi=160)
            plt.close(fig)


def svd_sequence(result, prepared, wavelength_um, orders, phase, output, orders_per_page):
    first_count = min(item.count for item in result.component_results)
    first = next(item for item in result.component_results if item.count == first_count)
    masked = np.where(result.mask[None, :, :], result.selected.residual_cube, np.nan)
    _sequence_pdf(Path(output) / f"{result.species}_svd_sequence.pdf", wavelength_um,
                  orders, phase,
                  [prepared - np.nanmedian(prepared, axis=0), first.residual_cube,
                   result.selected.residual_cube, masked],
                  ["Before SVD", f"{first_count} component(s) removed",
                   f"{result.selected.count} components removed", "+ tellurics masked"],
                  f"{result.species}: per-order observer-frame SVD detrending",
                  orders_per_page)


def template_sequence(result, wavelength_um, orders, phase, output, orders_per_page):
    first_count = min(item.count for item in result.component_results)
    first = next(item for item in result.component_results if item.count == first_count)
    masked = np.where(result.mask[None, :, :], result.selected.filtered_template_cube, np.nan)
    _sequence_pdf(Path(output) / f"{result.species}_template_sequence.pdf", wavelength_um,
                  orders, phase,
                  [result.planet_model_cube, first.filtered_template_cube,
                   result.selected.filtered_template_cube, masked],
                  ["Raw moving template", f"After {first_count} component(s)",
                   f"After {result.selected.count} components", "+ tellurics masked"],
                  f"{result.species}: exact fixed-SVD template processing",
                  orders_per_page)


def component_snr(result, output, formats, dpi):
    counts = [item.count for item in result.component_results]
    snr = [item.local_peak_snr for item in result.component_results]
    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.plot(counts, snr, "o-", color="#2a6fbb")
    axis.axvline(result.selected.count, color="#d62728", ls="--",
                 label=f"selected: {result.selected.count}")
    axis.axhline(0, color="0.5", lw=0.8)
    axis.set(xlabel="SVD components removed", ylabel="Local-maximum S/N",
             title=f"{result.species}: component selection")
    axis.legend(frameon=False)
    _save(fig, Path(output) / f"{result.species}_snr_vs_svd_components", formats, dpi)


def final_four_panel(result, orbit, rv_grid, kp_grid, vsys_grid, expected_kp,
                     expected_vsys, output, formats, dpi):
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8), constrained_layout=True)
    # Keep the event rows in the mesh as NaNs. Removing them caused
    # pcolormesh to bridge the phase gap and paint out-of-eclipse CCF rows
    # across the eclipse, producing a misleading rectangular trail feature.
    expected_velocity = (expected_kp * orbit.velocity_basis
                         + expected_vsys - orbit.berv_kms)
    velocity = expected_velocity[:, None] + rv_grid[None, :]
    displayed_ccf = np.where(
        orbit.signal_mask[:, None], result.selected.exposure_ccf, np.nan
    )
    trail = axes[0].pcolormesh(velocity, orbit.phase[:, None] * np.ones_like(velocity),
                               displayed_ccf, shading="auto",
                               cmap="RdBu_r", rasterized=True)
    displayed_velocity = np.where(orbit.signal_mask, expected_velocity, np.nan)
    axes[0].plot(displayed_velocity, orbit.phase, color="k", lw=1.4,
                 label="expected trail")
    axes[0].set(xlabel=r"Observer-frame planet velocity (km s$^{-1}$)",
                ylabel=f"Phase from {orbit.observation_type}", title="Planet trail")
    axes[0].legend(frameon=False, fontsize=8)
    fig.colorbar(trail, ax=axes[0], label="CCF")

    # The observed and injected panels carry their own summary, so the two
    # numbers the panel is read for -- the value at the expected location and
    # the value at the nearby maximum actually found -- sit on the markers
    # rather than only in the title. The null mean has no such pair.
    panels = ((result.selected.snr_map, "Observed", result.selected),
              (result.injected.snr_map, "Injection recovery", result.injected),
              (result.null_mean_map,
               f"Mean of {result.null_snr_at_expected.size} nulls", None))
    for axis, (values, label, summary) in zip(axes[1:], panels):
        image = axis.pcolormesh(vsys_grid, kp_grid, values, shading="auto",
                                cmap="RdBu_r", rasterized=True)
        expected_label = "expected"
        if summary is not None:
            expected_label += f" (S/N = {summary.expected_snr:+.2f})"
        axis.plot(expected_vsys, expected_kp, "+", color="k", ms=13, mew=2,
                  label=expected_label)
        if summary is not None:
            axis.plot(summary.local_peak_vsys_kms, summary.local_peak_kp_kms,
                      "o", mfc="none", mec="k", ms=8,
                      label=f"local maximum (S/N = {summary.local_peak_snr:+.2f})")
        axis.set(xlabel=r"$V_{sys}$ (km s$^{-1}$)", ylabel=r"$K_p$ (km s$^{-1}$)",
                 title=label)
        axis.legend(frameon=False, fontsize=8)
        fig.colorbar(image, ax=axis, label="Map S/N")
    fig.suptitle(
        rf"{result.species} | observed local max={result.selected.local_peak_snr:+.2f}$\sigma$ | "
        rf"injected local max={result.injected.local_peak_snr:+.2f}$\sigma$ | "
        f"global-null FAP={result.null_false_alarm_fraction:.3f}"
    )
    _save(fig, Path(output) / f"{result.species}_final_kp_vsys", formats, dpi)
