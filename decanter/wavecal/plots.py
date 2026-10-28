"""The telluric-rest-frame check: one strong H2O line, every frame, before / after.

:func:`telluric_check` draws a two-panel PNG: an unsaturated, isolated telluric
H2O line in every object spectrum of a night, on the wavelength labels the
spectra had before the calibration (left) and on the calibrated labels (right),
against the velocity from the line's pressure-shifted vacuum position. With the
calibration right, the lines of every frame sit on top of each other at 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from decanter.wavecal.model import C_KMS, LineTable


def _windows(mode: str) -> tuple[float, float, float]:
    """``(half window, core window, R)`` in km/s for an instrument mode."""
    if str(mode).upper().startswith("WIDE"):
        return 50.0, 15.0, 25000.0
    return 25.0, 4.0, 60000.0


def _norm(v, y, half):
    side = (np.abs(v) > 0.48 * half) & (np.abs(v) < half) & np.isfinite(y)
    c = np.nanmedian(y[side]) if side.sum() > 3 else np.nanmedian(y)
    return y / c if np.isfinite(c) and c > 0 else y * np.nan


def _gauss(x, a, mu, sig, c):
    return c - a * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def _fit(v, y, core):
    """Gaussian + constant fit within ``core`` km/s of the minimum near 0:
    ``(centre, sigma, residual rms)`` or None."""
    from scipy.optimize import curve_fit

    ok = np.isfinite(y) & (np.abs(v) < core)
    if ok.sum() < 5:
        return None
    v0 = v[ok][np.argmin(y[ok])]
    s = ok & (np.abs(v - v0) < core)
    if s.sum() < 5:
        return None
    try:
        p, _ = curve_fit(_gauss, v[s], y[s], p0=(1 - y[s].min(), v0, core / 1.6, 1.0),
                         maxfev=2000)
    except (RuntimeError, ValueError):
        return None
    if abs(p[1] - v0) >= core:
        return None
    return float(p[1]), abs(float(p[2])), float(np.std(y[s] - _gauss(v[s], *p)))


def pick_line(night, lines: LineTable | None = None):
    """The H2O line to show: unsaturated (depth 0.3-0.8 in the night's median
    object spectrum), continuum at the window edges, and a single Gaussian core
    (fit residual < 0.02, Gaussian sigma < 2 x the instrumental one --
    rejects blends). Returns ``(order, vacuum wavelength, depth)`` or None."""
    lines = lines or LineTable.load()
    half, core, R = _windows(night.mode)
    sig_inst = C_KMS / R / 2.3548
    best = None
    for m in night.obj.lam:
        lam = np.asarray(night.obj.lam[m], np.float64)
        span = lam[-1] - lam[0]
        cand = lines.absorber_lines(lam[0] + 0.08 * span, lam[-1] - 0.08 * span, R=R)
        med = np.nanmedian(night.obj.flux[m], axis=0)
        for lam0, d, b in zip(cand["lam"], cand["depth"], cand["blend"], strict=True):
            if d < 0.25 or b > 0.05:
                continue
            v = (lam / lam0 - 1) * C_KMS
            win = np.abs(v) < half
            if win.sum() < 10:
                continue
            y = _norm(v[win], med[win], half)
            c = y[np.abs(v[win]) < 0.75 * core]
            if c.size == 0 or not np.isfinite(c).any():
                continue
            depth = 1 - np.nanmin(c)
            edge = y[np.abs(v[win]) > 0.6 * half]
            if not 0.3 <= depth <= 0.8 or (edge.size and np.nanmin(edge) < 0.85):
                continue
            fit = _fit(v[win], y, core)
            if fit is None or fit[2] > 0.02 or fit[1] > 2.0 * sig_inst:
                continue
            score = depth * min(1.0, d)
            if best is None or score > best[3]:
                best = (int(m), float(lam0), float(depth), score)
    return None if best is None else best[:3]


def telluric_check(night, sol, path: str | Path, *, title: str | None = None,
                   lines: LineTable | None = None) -> dict:
    """Draw the before / after PNG for ``night`` (:class:`~decanter.wavecal.NightSpectra`,
    as the spectra were before the calibration) and its solution ``sol``; returns
    the per-panel line-centre statistics (median and robust frame-to-frame scatter,
    m/s) and the line used."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half, core, _ = _windows(night.mode)
    pick = pick_line(night, lines)
    if pick is None:
        raise RuntimeError(f"{night.name}: no unsaturated, isolated H2O line")
    m, lam0, depth = pick
    lam = np.asarray(night.obj.lam[m], np.float64)
    flux = np.asarray(night.obj.flux[m], np.float64)
    true = np.asarray(sol.true_lambda(m, lam, "star"), np.float64)
    t = np.asarray(night.obj.time, float)
    tt = (t - np.nanmin(t)) * 24.0
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    out = dict(night=night.name, mode=night.mode, order=m, lam0=lam0, depth=depth,
               frames=int(flux.shape[0]))
    for ax, name, label in zip(axes, ("before", "after"),
                               ("before (input labels)", "after (decanter.wavecal)")):
        cents = []
        for f in range(flux.shape[0]):
            v = ((lam if name == "before" else true[f]) / lam0 - 1) * C_KMS
            s = np.abs(v) < half
            y = _norm(v[s], flux[f][s], half)
            fit = _fit(v[s], y, core)
            cents.append(np.nan if fit is None else fit[0])
            ax.plot(v[s], y, lw=0.7, alpha=0.6, color=cmap(tt[f] / max(np.nanmax(tt), 1e-9)))
        c = np.asarray(cents)
        c = c[np.isfinite(c)]
        med = float(np.median(c) * 1e3) if c.size else np.nan
        rms = float(1.4826 * np.median(np.abs(c - np.median(c))) * 1e3) if c.size > 1 else np.nan
        out[f"{name}_centre_ms"], out[f"{name}_scatter_ms"] = med, rms
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlim(-half, half)
        ax.set_ylim(-0.1, 1.5)
        ax.set_xlabel("velocity from H$_2$O line rest position [km/s]")
        ax.set_title(f"{label}\ncentre {med:+.0f} m/s, frame-to-frame scatter {rms:.0f} m/s",
                     fontsize=9)
    axes[0].set_ylabel("normalised flux")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(np.nanmax(tt), 1e-9)))
    fig.colorbar(sm, ax=axes, label="hours since first frame", pad=0.01)
    fig.suptitle(title or f"{night.name} ({night.mode}): H$_2$O {lam0:.3f} Å vac, order {m}, "
                 f"{flux.shape[0]} frames", fontsize=10, y=1.04)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    out["png"] = str(path)
    return out
