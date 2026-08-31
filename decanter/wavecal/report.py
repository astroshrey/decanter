"""Flip-through PDF reports for a physical wavelength calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from decanter.wavecal.measure import robust_scatter
from decanter.wavecal.solution import C_KMS

BLUE = "#1f77b4"
RED = "#d62728"
GREEN = "#2ca02c"
GRAY = "0.45"


def _finite_percent(values) -> float:
    return 100.0 * float(np.count_nonzero(np.isfinite(values))) / values.size


def _footer(fig, dataset: str, page: int, section: str) -> None:
    fig.text(
        0.995, 0.002,
        f"{dataset} | {section} | p. {page}",
        ha="right", va="bottom", fontsize=5, color="0.40",
    )


def _save(pdf, fig, dataset: str, page: int, section: str, plt) -> int:
    _footer(fig, dataset, page, section)
    pdf.savefig(fig)
    plt.close(fig)
    return page + 1


def _four_segments(n_pixels: int):
    edges = np.linspace(0, n_pixels, 5, dtype=int)
    return [slice(int(edges[k]), int(edges[k + 1])) for k in range(4)]


def _source_counts(source):
    return {name: int(np.count_nonzero(source == name))
            for name in ("telluric", "OH", "interpolated", "unavailable")}


def _paired_centered(telluric_velocity, oh_velocity, paired):
    """Put both tracers on the same per-order paired-exposure zero point."""
    tell = np.full_like(telluric_velocity, np.nan)
    air = np.full_like(oh_velocity, np.nan)
    for j in range(tell.shape[1]):
        good = paired[:, j]
        if np.count_nonzero(good) < 2:
            continue
        tell[:, j] = telluric_velocity[:, j] - np.nanmedian(telluric_velocity[good, j])
        air[:, j] = oh_velocity[:, j] - np.nanmedian(oh_velocity[good, j])
    return tell, air


def _pooled_ccf_pages(pdf, plt, atmospheric, series, dataset: str, page: int) -> int:
    """Two pages on the first iteration: the pooled common-mode CCF.

    The common-mode shift each exposure was pre-aligned by is the peak of one
    pooled curve, so these pages show the curve rather than the number: where
    the peak sits per exposure, how far it stands above its own baseline, and
    whether the two tracers put it in the same place.
    """
    grid = np.asarray(atmospheric.velocity_grid_kms, dtype=float)
    joint = np.asarray(atmospheric.joint_score, dtype=float)
    rows = np.arange(joint.shape[0])
    time = np.asarray(series.elapsed_hours, dtype=float)
    peak = np.asarray(atmospheric.peak_velocity_kms, dtype=float)
    sigma = np.asarray(atmospheric.bootstrap_sigma_kms, dtype=float)
    telluric_peak = np.asarray(atmospheric.telluric_peak_velocity_kms, dtype=float)
    oh_peak = np.asarray(atmospheric.oh_peak_velocity_kms, dtype=float)

    panels = (
        (atmospheric.telluric_score, telluric_peak, "Telluric pool"),
        (atmospheric.oh_score, oh_peak, "OH pool"),
        (joint, peak, "Joint pool"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True,
                             constrained_layout=True)
    for ax, (score, peaks, label) in zip(axes, panels):
        score = np.asarray(score, dtype=float)
        if np.any(np.isfinite(score)):
            image = ax.pcolormesh(grid, rows, score, shading="auto", cmap="viridis")
            fig.colorbar(image, ax=ax, label="pooled standardised CCF")
            ax.plot(peaks, rows, "w.", ms=3.5)
        else:
            ax.text(0.5, 0.5, f"no {label.lower()} contributed", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9, color=GRAY)
        ax.set_ylabel(f"exposure\n{label}")
    axes[-1].set_xlabel("velocity relative to each order's static offset (km/s)")
    fig.suptitle(
        f"Iteration 1: pooled atmospheric common-mode CCF ({dataset})", fontsize=14
    )
    page = _save(pdf, fig, dataset, page, "common mode", plt)

    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    spec = fig.add_gridspec(3, 2, width_ratios=[1.45, 1.0])
    ax = fig.add_subplot(spec[0, 0])
    ax.plot(time, telluric_peak, "o-", color=BLUE, ms=3.5, lw=0.9, label="telluric")
    ax.plot(time, oh_peak, "s-", color=RED, ms=3.0, lw=0.9, label="OH")
    ax.errorbar(time, peak, yerr=sigma, fmt="k.-", ms=4, lw=0.9,
                label="joint (order bootstrap)")
    ax.set_ylabel("peak velocity (km/s)")
    ax.margins(y=0.28)
    ax.legend(frameon=False, fontsize=7, ncols=3, loc="lower left")
    ax.set_title("Common mode measured per exposure", fontsize=10)

    ax = fig.add_subplot(spec[1, 0])
    ax.plot(time, atmospheric.peak_snr, "o-", color="0.25", ms=3.5, lw=0.9)
    ax.set_ylabel("peak / baseline MAD")
    twin = ax.twinx()
    prominence = (np.asarray(atmospheric.peak_score, dtype=float)
                  - np.asarray(atmospheric.secondary_score, dtype=float))
    twin.plot(time, prominence, "^--", color=GREEN, ms=3.5, lw=0.8)
    twin.set_ylabel("peak - best rival", color=GREEN)
    twin.tick_params(axis="y", labelcolor=GREEN)

    ax = fig.add_subplot(spec[2, 0])
    ax.plot(time, (telluric_peak - oh_peak) * 1e3, "o-", color=BLUE, ms=3.5,
            lw=0.9, label="telluric - OH")
    ax.plot(time, sigma * 1e3, "s-", color="0.25", ms=3.0, lw=0.9,
            label="bootstrap 1 sigma")
    ax.axhline(0.0, color="0.75", lw=0.8)
    ax.set_ylabel("velocity (m/s)")
    ax.set_xlabel("time (hours)")
    ax.margins(y=0.28)
    ax.legend(frameon=False, fontsize=7, ncols=2, loc="lower left")

    ax = fig.add_subplot(spec[0, 1])
    finite = np.isfinite(time)
    span = np.ptp(time[finite]) if np.count_nonzero(finite) > 1 else 1.0
    colors = plt.get_cmap("viridis")(
        (time - np.nanmin(time)) / span if span > 0 else np.zeros_like(time)
    )
    for i in range(joint.shape[0]):
        ax.plot(grid, joint[i], lw=0.7, color=colors[i], alpha=0.85)
    ax.axvline(0.0, color="0.75", lw=0.8)
    ax.set_xlabel("relative velocity (km/s)")
    ax.set_ylabel("joint pooled CCF")
    ax.set_title("Every exposure, coloured by time", fontsize=10)

    ax = fig.add_subplot(spec[1:, 1])
    ax.axis("off")
    difference = telluric_peak - oh_peak
    both = np.isfinite(difference)
    applied = np.asarray(atmospheric.common_velocity_kms, dtype=float)
    text = (
        f"Search: +/-{atmospheric.search_kms:g} km/s at {atmospheric.step_kms:g} km/s\n"
        f"Telluric orders pooled: {len(atmospheric.telluric_orders)}\n"
        f"OH orders pooled: {len(atmospheric.oh_orders)}\n\n"
        f"Peak / baseline MAD: median {np.nanmedian(atmospheric.peak_snr):.1f}, "
        f"worst {np.nanmin(atmospheric.peak_snr):.1f}\n"
        f"Peak - best rival: median {np.nanmedian(prominence):.2f}, "
        f"worst {np.nanmin(prominence):.2f}\n"
        f"Peak FWHM: median {np.nanmedian(atmospheric.fwhm_kms):.1f} km/s\n\n"
        f"Order bootstrap: median {np.nanmedian(sigma) * 1e3:.0f} m/s, "
        f"worst {np.nanmax(sigma) * 1e3:.0f} m/s\n"
        + (f"Telluric - OH: RMS {np.sqrt(np.mean(difference[both] ** 2)) * 1e3:.0f} m/s, "
           f"median {np.nanmedian(difference[both]) * 1e3:+.0f} m/s\n"
           if np.any(both) else "Telluric - OH: one tracer only\n")
        + "\nApplied common mode:\n"
          f"  {np.ptp(applied) * 1e3:.0f} m/s peak to peak, "
          f"{np.std(applied) * 1e3:.0f} m/s RMS\n\n"
        "The applied shift is the joint peak with\n"
        "the series median removed. Iteration 2\n"
        "measures the order-dependent residual\n"
        "left on top of it."
    )
    ax.text(0.0, 1.0, text, va="top", fontsize=8, linespacing=1.5)
    fig.suptitle("Iteration 1: common-mode peak quality", fontsize=14)
    return _save(pdf, fig, dataset, page, "common mode", plt)


def wavecal_report_pdf(run, path: str | Path, *, dataset: str) -> Path:
    """Write one compact, one-order-per-page calibration report.

    The input is a :class:`~decanter.wavecal.solve.WavecalRun` returned by
    ``solve(..., return_diagnostics=True)``. When the run carries an
    :class:`~decanter.wavecal.solve.AtmosphericCommonMode`, the pooled
    common-mode CCF of the first iteration is reported directly after the
    overview. Only line-rich orders receive fit pages. Orders rich in both
    tracers receive explicit time-series and tracer-to-tracer compatibility
    pages.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    series = run.series
    config = run.config
    solution = run.solution
    tell = run.telluric_model
    oh = run.oh_model
    orders = np.asarray(series.orders)
    time = np.asarray(series.elapsed_hours)
    rich_t = (
        tell.rich_orders(config.telluric_rms_threshold)
        if tell is not None else np.zeros(series.n_orders, dtype=bool)
    )
    rich_o = (oh.rich_orders(config.oh_rich_min_lines)
              if oh is not None else np.zeros(series.n_orders, dtype=bool))
    overlap_rich = rich_t & rich_o
    paired = run.telluric_accepted & run.oh_accepted
    counts = _source_counts(solution.source)
    total_cells = solution.source.size

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    page = 1

    with PdfPages(path) as pdf:
        # Overview: what was selected and how much of the solution is direct.
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
        ax = axes[0, 0]
        width = 0.38
        if tell is not None:
            ax.bar(orders - width / 2, tell.template_rms, width=width, color=BLUE,
                   label="telluric template RMS")
            ax.axhline(config.telluric_rms_threshold, color=BLUE, ls="--", lw=1)
        ax.set_xlabel("echelle order")
        ax.set_ylabel("telluric template RMS", color=BLUE)
        ax.tick_params(axis="y", labelcolor=BLUE)
        twin = ax.twinx()
        if oh is not None:
            twin.bar(orders + width / 2, oh.line_count, width=width, color=RED,
                     alpha=0.65, label="detected OH lines")
        twin.axhline(config.oh_rich_min_lines, color=RED, ls="--", lw=1)
        twin.set_ylabel("detected OH lines", color=RED)
        twin.tick_params(axis="y", labelcolor=RED)
        ax.set_title("Order selection")

        ax = axes[0, 1]
        names = ["telluric", "OH", "interpolated", "unavailable"]
        colors = [BLUE, RED, "0.55", "0.85"]
        values = [100.0 * counts[n] / total_cells for n in names]
        ax.bar(names, values, color=colors)
        ax.set_ylabel("fraction of frame-order cells (%)")
        ax.set_ylim(0, max(100, 1.12 * max(values)))
        for k, value in enumerate(values):
            ax.text(k, value + 1.0, f"{value:.1f}%", ha="center", fontsize=8)
        ax.set_title("Final solution sources")

        ax = axes[1, 0]
        if tell is not None:
            ax.plot(orders, tell.lsf_fwhm_kms, "o-", color=BLUE, ms=4,
                    label="telluric fit")
        if oh is not None:
            ax.plot(orders, oh.lsf_fwhm_kms, "s-", color=RED, ms=3.5,
                    label="OH fit")
        nominal = config.resolution_element_kms(series.instmode)
        ax.axhline(nominal, color="0.25", ls=":", label=f"nominal {nominal:.2f} km/s")
        ax.set_xlabel("echelle order")
        ax.set_ylabel("LSF FWHM (km/s)")
        ax.set_title("Independent fitted line widths")
        ax.legend(frameon=False, fontsize=8)

        ax = axes[1, 1]
        ax.axis("off")
        overlap_cells = int(np.count_nonzero(paired & overlap_rich[None, :]))
        prealign = getattr(run, "atmospheric", None)
        prealign_text = (
            "Pre-aligned by a pooled telluric+OH common mode of "
            f"{np.std(prealign.common_velocity_kms) * 1e3:.0f} m/s RMS "
            f"({len(prealign.telluric_orders)} telluric + {len(prealign.oh_orders)} "
            "OH orders pooled); the solution below is the residual.\n"
            if prealign is not None else ""
        )
        priority_text = (
            "Priority: accepted telluric is used literally; otherwise accepted OH is "
            "used literally; all other values come from the smooth cross-order fit."
            if config.uses_telluric else
            "OH-only: accepted OH is used literally; all other values come from the "
            "smooth cross-order fit constrained only by OH anchors."
        )
        text = (
            f"{dataset}\n\n"
            f"{series.summary()}\n\n"
            f"Mode: {config.mode}; zero point: {config.zero_point}; assembly: {config.assembly}\n"
            f"Telluric rich: {int(rich_t.sum())}/{series.n_orders} orders "
            f"(RMS >= {config.telluric_rms_threshold:.3f})\n"
            f"OH rich: {int(rich_o.sum())}/{series.n_orders} orders "
            f"(lines >= {config.oh_rich_min_lines})\n"
            f"Rich in both: {int(overlap_rich.sum())} orders; "
            f"paired accepted cells: {overlap_cells}\n"
            f"CCF acceptance: telluric >= {config.telluric_peak_threshold:.2f}; "
            f"OH >= {config.oh_peak_threshold:.2f}\n"
            f"Finite final solution: {_finite_percent(solution.velocity):.1f}%\n"
            f"{prealign_text}\n"
            f"{priority_text}"
        )
        ax.text(0.0, 1.0, text, va="top", fontsize=9, linespacing=1.45, wrap=True)
        fig.suptitle(f"Physical wavelength calibration: {config.mode}", fontsize=15)
        page = _save(pdf, fig, dataset, page, "overview", plt)

        atmospheric = getattr(run, "atmospheric", None)
        if atmospheric is not None:
            page = _pooled_ccf_pages(pdf, plt, atmospheric, series, dataset, page)

        # One page per telluric-rich order, divided into four wavelength spans.
        for j in np.where(rich_t)[0]:
            fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), constrained_layout=True)
            for ax, sl in zip(axes, _four_segments(series.n_pixels)):
                wave = series.wave[sl, j] / 10.0
                data = tell.target[j][sl]
                model = tell.fitted_template[sl, j]
                ax.plot(wave, data, color="0.30", lw=0.75, label="median object spectrum")
                ax.plot(wave, model, color=RED, lw=1.0, label="telluric fit")
                ax.plot(wave, 0.28 + data - model, color=GREEN, lw=0.55,
                        label="residual + 0.28")
                ax.axhline(0.28, color=GREEN, lw=0.45, ls=":")
                support = run.telluric_support[sl, j]
                ax.fill_between(wave, 0, 1, where=support, color=BLUE, alpha=0.06,
                                transform=ax.get_xaxis_transform(), step="mid")
                ax.set_xlim(wave.min(), wave.max())
                ax.set_ylim(0.04, 1.24)
                ax.ticklabel_format(axis="x", useOffset=False, style="plain")
                ax.set_ylabel("norm. flux", fontsize=8)
            axes[-1].set_xlabel("vacuum wavelength (nm)")
            axes[0].legend(frameon=False, fontsize=7, ncols=3, loc="lower left")
            axes[0].set_title(
                f"Telluric-rich order m{orders[j]} | {tell.family[j]} | "
                f"LSF {tell.lsf_fwhm_kms[j]:.2f} km/s "
                f"(R={C_KMS / tell.lsf_fwhm_kms[j]:.0f}) | "
                f"template RMS {tell.template_rms[j]:.3f} | "
                f"residual RMS {tell.residual_rms[j]:.4f}", fontsize=9,
            )
            page = _save(pdf, fig, dataset, page, "telluric-rich fits", plt)

        # One page per OH-rich order.
        if oh is not None:
            from decanter.wavecal import airglow
            from decanter.wavecal.opacity import linelist_path

            lines = airglow.load_oh_lines(
                linelist_path("OH", config.linelist_dir),
                float(np.nanmin(series.wave)), float(np.nanmax(series.wave)),
            )["wave_angstrom"]
            for j in np.where(rich_o)[0]:
                fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), constrained_layout=True)
                for ax, sl in zip(axes, _four_segments(series.n_pixels)):
                    wave = series.wave[sl, j] / 10.0
                    data = oh.target[j][sl]
                    model = oh.fitted_model[j][sl]
                    ax.plot(wave, data, color="0.30", lw=0.75, label="median sky")
                    ax.plot(wave, model, color=RED, lw=1.0, label="OH fit")
                    support = oh.support[sl, j]
                    ax.fill_between(wave, 0, 1, where=support, color=GREEN, alpha=0.10,
                                    transform=ax.get_xaxis_transform(), step="mid")
                    lo, hi = float(series.wave[sl.start, j]), float(series.wave[sl.stop - 1, j])
                    for centre in lines[(lines > lo) & (lines < hi)] / 10.0:
                        ax.axvline(centre, color=BLUE, lw=0.35, alpha=0.28)
                    top = max(0.2, float(np.nanpercentile(data, 99.8)) * 1.25)
                    ax.set_xlim(wave.min(), wave.max())
                    ax.set_ylim(-0.05, top)
                    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
                    ax.set_ylabel("scaled sky", fontsize=8)
                axes[-1].set_xlabel("vacuum wavelength (nm)")
                axes[0].legend(frameon=False, fontsize=7, ncols=2, loc="upper left")
                axes[0].set_title(
                    f"OH-rich order m{orders[j]} | {oh.family[j]} | "
                    f"LSF {oh.lsf_fwhm_kms[j]:.2f} km/s | "
                    f"{int(oh.line_count[j])} detected lines | "
                    f"residual RMS {oh.residual_rms[j]:.4f}", fontsize=9,
                )
                page = _save(pdf, fig, dataset, page, "OH-rich fits", plt)

        # Representative frame: maximize direct anchors, then prefer the middle of the run.
        tell_ok = run.telluric_accepted
        oh_only = run.oh_accepted & ~tell_ok
        score = tell_ok.sum(axis=1) + oh_only.sum(axis=1)
        best_score = int(np.max(score))
        candidates = np.flatnonzero(score == best_score)
        representative = int(candidates[np.argmin(np.abs(candidates - series.n_frames / 2))])
        i = representative
        fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
        ax.plot(orders, run.smooth_velocity[i] * 1e3, color="0.15", lw=1.4,
                label="smooth cross-order interpolation")
        ax.plot(orders[tell_ok[i]], run.telluric_velocity[i, tell_ok[i]] * 1e3,
                "o", color=BLUE, ms=6, label="direct telluric (used literally)")
        ax.plot(orders[oh_only[i]], run.oh_velocity[i, oh_only[i]] * 1e3,
                "s", color=RED, ms=5.5, label="direct OH (used literally)")
        overlap = tell_ok[i] & run.oh_accepted[i]
        ax.plot(orders[overlap], run.oh_velocity[i, overlap] * 1e3,
                "s", mfc="none", mec=RED, mew=1.2, ms=7,
                label="OH where telluric has priority")
        interpolated = solution.source[i] == "interpolated"
        ax.plot(orders[interpolated], solution.velocity[i, interpolated] * 1e3,
                "x", color="0.50", ms=7, mew=1.2, label="adopted interpolated value")
        ax.axhline(0, color="0.65", lw=0.7, ls=":")
        ax.set_xlabel("echelle order")
        velocity_kind = "relative" if config.zero_point == "relative" else "absolute"
        ax.set_ylabel(f"{velocity_kind} velocity correction (m/s)")
        ax.set_title(
            f"Representative {config.mode} interpolation | frame {series.frame_ids[i]} | "
            f"elapsed {time[i]:.2f} h | {best_score} direct anchors", fontsize=11,
        )
        ax.legend(frameon=False, fontsize=8, ncols=2)
        ax.text(
            0.01, 0.02,
            "The black curve is fit only to filled blue and red anchors. Open red squares "
            "test OH compatibility but do not constrain the curve when tellurics are accepted.",
            transform=ax.transAxes, fontsize=8, va="bottom",
        )
        page = _save(pdf, fig, dataset, page, "cross-order interpolation", plt)

        # Every exposure in the compact drift-versus-order form used for
        # rapid visual review. Plot the shifts on their physical-template
        # zero point: subtracting a different temporal median from each order
        # would destroy the degree-2 shape of the actual interpolation.
        smooth_drift = run.smooth_velocity * 1e3
        final_drift = solution.velocity * 1e3
        tell_drift = run.telluric_velocity * 1e3
        oh_drift = run.oh_velocity * 1e3
        plotted = np.concatenate(
            [smooth_drift[np.isfinite(smooth_drift)], final_drift[np.isfinite(final_drift)]]
        )
        drift_limit = (
            max(250.0, 1.06 * float(np.nanmax(np.abs(plotted))))
            if plotted.size else 250.0
        )
        panels_per_page = 12
        for start in range(0, series.n_frames, panels_per_page):
            stop = min(start + panels_per_page, series.n_frames)
            fig, axes = plt.subplots(
                3, 4, figsize=(17, 11), sharex=True, sharey=True,
                constrained_layout=True,
            )
            for panel, i in enumerate(range(start, stop)):
                ax = axes.flat[panel]
                tell_i = run.telluric_accepted[i]
                oh_only_i = run.oh_accepted[i] & ~run.telluric_accepted[i]
                interpolated_i = solution.source[i] == "interpolated"

                ax.plot(orders, smooth_drift[i], color="0.18", lw=1.2,
                        label="quadratic interpolation")
                ax.plot(orders[tell_i], tell_drift[i, tell_i], "o",
                        color=BLUE, ms=4.5, label="telluric anchor")
                ax.plot(orders[oh_only_i], oh_drift[i, oh_only_i], "s",
                        color=RED, ms=4.2, label="OH anchor")
                ax.plot(orders[interpolated_i], final_drift[i, interpolated_i], "x",
                        color="0.55", ms=4.5, mew=1.0,
                        label="line-poor/interpolated")
                ax.axhline(0, color="0.75", lw=0.6)
                ax.set_ylim(-drift_limit, drift_limit)

                direct_i = run.telluric_accepted[i] | (
                    run.oh_accepted[i] & ~run.telluric_accepted[i]
                )
                residual = final_drift[i, direct_i] - smooth_drift[i, direct_i]
                residual = residual[np.isfinite(residual)]
                rms = float(np.sqrt(np.mean(residual**2))) if residual.size else np.nan
                date = str(series.meta[i].get("DATE-OBS", "")).strip()
                label = f"{date} / {series.frame_ids[i]}" if date else series.frame_ids[i]
                ax.set_title(f"{label}\nresidual RMS={rms:.0f} m s$^{{-1}}$", fontsize=8)
            for panel in range(stop - start, panels_per_page):
                axes.flat[panel].axis("off")
            axes.flat[0].legend(loc="lower left", ncols=2,
                                frameon=False, fontsize=6.5)
            fig.supxlabel("Physical echelle order", fontsize=11)
            fig.supylabel("Shift relative to physical template (m s$^{-1}$)", fontsize=11)
            fig.suptitle(
                f"Wavelength drift by order - frames {start + 1}-{stop} of "
                f"{series.n_frames}", fontsize=14,
            )
            page = _save(pdf, fig, dataset, page, "drift by physical order", plt)

        # Compatibility summary for orders that are independently rich in both tracers.
        paired_rich = paired & overlap_rich[None, :]
        paired_tell, paired_oh = _paired_centered(
            run.telluric_velocity, run.oh_velocity, paired_rich)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
        if np.any(paired_rich):
            x = paired_tell[paired_rich] * 1e3
            y = paired_oh[paired_rich] * 1e3
            delta = y - x
            lim = float(max(np.nanpercentile(np.abs(np.concatenate([x, y])), 98), 50.0))
            axes[0, 0].scatter(x, y, s=10, color="0.25", alpha=0.35)
            axes[0, 0].plot([-lim, lim], [-lim, lim], color="0.2", ls="--", lw=1)
            axes[0, 0].set_xlim(-lim, lim)
            axes[0, 0].set_ylim(-lim, lim)
            axes[0, 0].set_xlabel("telluric relative drift (m/s)")
            axes[0, 0].set_ylabel("OH relative drift (m/s)")
            axes[0, 0].set_title("Cell-by-cell comparison")

            for j in np.where(overlap_rich)[0]:
                good = paired[:, j]
                axes[0, 1].scatter(time[good],
                                   (paired_oh[good, j] - paired_tell[good, j]) * 1e3,
                                   s=11, alpha=0.38, label=f"m{orders[j]}")
            axes[0, 1].axhline(0, color="0.2", ls="--", lw=1)
            axes[0, 1].set_xlabel("elapsed time (h)")
            axes[0, 1].set_ylabel("OH - telluric (m/s)")
            axes[0, 1].set_title("Difference versus time")
            if overlap_rich.sum() <= 10:
                axes[0, 1].legend(frameon=False, fontsize=6, ncols=2)

            order_delta = []
            order_sigma = []
            order_x = []
            for j in np.where(overlap_rich)[0]:
                good = paired[:, j]
                difference = (run.oh_velocity[good, j] - run.telluric_velocity[good, j]) * 1e3
                if difference.size:
                    order_x.append(orders[j])
                    order_delta.append(float(np.nanmedian(difference)))
                    order_sigma.append(float(robust_scatter(difference)))
            axes[1, 0].errorbar(order_x, order_delta, yerr=order_sigma, fmt="o", color="0.2",
                                ecolor="0.55", capsize=3)
            axes[1, 0].axhline(0, color="0.2", ls="--", lw=1)
            axes[1, 0].set_xlabel("echelle order")
            axes[1, 0].set_ylabel("median OH - telluric (m/s)")
            axes[1, 0].set_title("Per-order offset; bars are robust scatter")

            axes[1, 1].axis("off")
            correlation = np.corrcoef(x, y)[0, 1] if x.size >= 3 else np.nan
            axes[1, 1].text(
                0.0, 1.0,
                f"Compatibility summary\n\n"
                f"Rich-overlap orders: {int(overlap_rich.sum())}\n"
                f"Paired accepted cells: {x.size}\n"
                f"Median paired-centered OH - telluric: {np.nanmedian(delta):+.0f} m/s\n"
                f"Robust scatter: {robust_scatter(delta):.0f} m/s\n"
                f"RMS difference: {np.sqrt(np.nanmean(delta**2)):.0f} m/s\n"
                f"Pearson r: {correlation:.3f}\n\n"
                "Drift panels use the same paired-exposure median for each tracer.\n"
                "Per-order offsets retain the common physical template zero point.\n"
                "Only cells passing both CCF thresholds are included.",
                va="top", fontsize=10, linespacing=1.5,
            )
        else:
            for ax in axes.flat:
                ax.axis("off")
            axes[0, 0].text(0.5, 0.5, "No cells pass both direct-reference thresholds\n"
                           "in orders rich in both tracers.", ha="center", va="center",
                           fontsize=13)
        fig.suptitle("Telluric-OH compatibility in doubly rich orders", fontsize=14)
        page = _save(pdf, fig, dataset, page, "telluric-OH compatibility", plt)

        # Detailed compatibility, one page for each doubly rich order.
        for j in np.where(overlap_rich)[0]:
            fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
            grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
            top = fig.add_subplot(grid[0, :])
            scatter = fig.add_subplot(grid[1, 0])
            diff = fig.add_subplot(grid[1, 1])
            gt = tell_ok[:, j]
            go = run.oh_accepted[:, j]
            both = paired[:, j]
            top.plot(time[gt], paired_tell[gt, j] * 1e3, "o-", color=BLUE,
                     ms=3.5, lw=0.7, label="telluric")
            top.plot(time[go], paired_oh[go, j] * 1e3, "s-", color=RED,
                     ms=3.2, lw=0.7, label="OH")
            top.axhline(0, color="0.65", lw=0.7, ls=":")
            top.set_xlabel("elapsed time (h)")
            top.set_ylabel("relative drift (m/s)")
            top.legend(frameon=False, fontsize=8)

            tx = paired_tell[both, j] * 1e3
            oy = paired_oh[both, j] * 1e3
            if tx.size:
                limit = float(max(np.nanpercentile(np.abs(np.concatenate([tx, oy])), 98), 50))
                scatter.scatter(tx, oy, s=18, color="0.25", alpha=0.55)
                scatter.plot([-limit, limit], [-limit, limit], color="0.2", ls="--", lw=1)
                scatter.set_xlim(-limit, limit)
                scatter.set_ylim(-limit, limit)
                difference = oy - tx
                diff.scatter(time[both], difference, s=18, color="0.25", alpha=0.55)
                median = float(np.nanmedian(difference))
                sigma = float(robust_scatter(difference))
                diff.axhline(0, color="0.2", ls="--", lw=1)
                diff.axhline(median, color=RED, lw=1,
                             label=f"median {median:+.0f} m/s")
                diff.legend(frameon=False, fontsize=8)
                stats = f"N={tx.size}; median={median:+.0f} m/s; robust scatter={sigma:.0f} m/s"
            else:
                scatter.text(0.5, 0.5, "No paired accepted cells", ha="center", va="center",
                             transform=scatter.transAxes)
                diff.text(0.5, 0.5, "No paired accepted cells", ha="center", va="center",
                          transform=diff.transAxes)
                stats = "No paired accepted cells"
            scatter.set_xlabel("telluric drift (m/s)")
            scatter.set_ylabel("OH drift (m/s)")
            diff.set_xlabel("elapsed time (h)")
            diff.set_ylabel("OH - telluric (m/s)")
            top.set_title(
                f"Doubly rich order m{orders[j]} | telluric RMS {tell.template_rms[j]:.3f} | "
                f"OH lines {int(oh.line_count[j])} | {stats}", fontsize=10,
            )
            page = _save(pdf, fig, dataset, page, "telluric-OH compatibility", plt)

    return path
