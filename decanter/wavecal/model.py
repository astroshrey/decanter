"""Per-frame wavelength anchoring on extracted spectra (replaces WARP's s10).

What it does
------------
The period ThAr solution (the per-order ``id`` series, or the 2-D ``ecidentify`` fit) is kept
as a FIXED SHAPE TEMPLATE ``lambda_ref(p, m)``.  Everything that moves during a night is
measured on the atmosphere, frame by frame, in velocity units:

``z_sky``   (instrument) apparent velocity of OH airglow lines on the ThAr scale, from the
            sky spectra: Lambda-doublet-aware multi-Gaussian templates (HITRAN positions, a
            fitted population model for the relative intensities, the two parity components
            fitted as one template up to 4 FWHM apart, modelled blends inside the window), one
            shift per line per frame, a robust lines x frames decomposition
            ``d_kf = mu_k + z_f + Phi_k C R_f`` (``Phi`` = low-order ``(x, m)`` detector modes,
            ``R_f`` = per-frame regressors: nod and drift), then a time model
            ``z_f = b0 + delta_nod n_f + g(t_f) + e_f`` (Matern-3/2 GP drift ``g``, per-exposure
            jitter ``e``, A/B nod term, fitted by marginal likelihood in JAX float64, robust).
            A static low-order ``(x, m)`` surface (default 2 x 2, no constant) describes the
            night's departure of the sky from the ThAr shape; it and the OH modes are applied to
            SKY labels only.
``z_star``  (star light) apparent velocity of telluric H2O absorption lines in the object
            spectra (neighbouring absorbers modelled in the template; O2 opt-in), same
            decomposition with modes x ``(X_f - <X>, nod_f)``, and a time model
            ``z*_f = alpha + beta X_f + delta*_nod n_f + h(t_f) + e*_f`` where ``X_f`` is the OH
            instrument zero point at the object exposure (``beta`` free, prior N(1, 0.3^2)).
            OH-derived shape terms are never applied to the star (it does not follow the sky:
            slope 0.79 on TOI-3486, 1.04-1.07 on TOI-2109 / WASP-69).

No frame is ever set to zero (s10's flag-to-zero is gone), every frame gets a sigma, and
nothing is resampled: the outputs are labels on the static float64 grid,

    u_f = ln(lambda_true / lambda_label) = ln(lambda_raw / lambda_label) - log1p(z_f / c),

``lambda_raw`` = the label with any per-frame pixel shift already applied to the flux undone
(WARP's s10/s11 shift for warp-mode object spectra; identity for native fast-mode output).
:func:`export_resample` is the ONE resample (Lanczos-a in pixel index) to whatever grid the
consumer needs.  Float32 code should carry ``u`` (or pixel offsets), never ``lambda`` itself
(the f32 ulp at 10830 A is 0.027 km/s).

It runs today on warp-mode outputs (:func:`load_warp_night`, a WARP reduction directory, or
:func:`night_from_reductions`, a list of :class:`decanter.Reduction`) and later on fast-mode
outputs through the same :class:`SpectrumSet` container (static labels + per-frame flux).

Usage
-----
::

    from decanter.wavecal import load_warp_night, anchor_night, export_resample
    night = load_warp_night("~/Disk/winered/reductions/TOI2109")
    sol = anchor_night(night)                     # AnchorConfig.for_mode(night.mode)
    sol.z_star, sol.z_star_err                    # km/s per object frame (raw ThAr scale)
    lam = night.obj.lam[163]                      # float64 static labels
    u = sol.labels(163, lam, kind="star")         # (F, n) float32: u per pixel and frame
    flux_obs = export_resample(night.obj.flux[163], lam, u, lam)   # observatory (telluric) frame

Conventions
-----------
* Velocities in km/s, positive = atmospheric lines appear REDDER than their vacuum laboratory
  position on the ThAr scale.  Wavelengths are vacuum Angstrom, float64 on the host.
* Times are MJD (UTC) at mid-exposure; the GP works in hours from the night's mean time.
* ``nod`` is the APERTURE slit position the spectrum was extracted at ("A"/"B"): for a WARP
  sky product (the partner frame extracted at the object aperture) it is the object's nod,
  while its time and exposure id are the partner's.
* ``epoch`` identifies the raw exposure (FRAMEID): per-exposure jitter is shared by every
  spectrum taken from the same exposure.
* JAX float64 through JAX (``jax_enable_x64``) for
  everything that is fitted (centroids, decomposition, GP marginal likelihood);
  ``jax.lax``/``jax.jacfwd`` are used directly where the shim has no equivalent.

Method details, the line list's provenance and the validation record: ``docs/WAVECAL.md``.
"""
from __future__ import annotations

import datetime as _dt
import glob
import math
import re
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "C_KMS",
    "HE_I_VAC",
    "SURFACE_FILENAME",
    "AnchorConfig",
    "AnchorSolution",
    "EcSolution",
    "LineTable",
    "NightSpectra",
    "SpectrumSet",
    "StaticSurface",
    "ThArTemplate",
    "TimeModel",
    "anchor",
    "anchor_frames",
    "anchor_night",
    "anchor_reduction",
    "anchored_zero_points",
    "build_absorber_features",
    "build_oh_features",
    "build_static_surface",
    "choose_method",
    "epoch_groups",
    "export_resample",
    "fit_line_centroids",
    "fit_time_model",
    "load_warp_night",
    "night_from_reductions",
    "robust_zero_points",
    "static_lnlam_grid",
    "surface_from_warp_reductions",
]

C_KMS = 299792.458
HE_I_VAC = (10832.057, 10833.217, 10833.306)
LINE_TABLE_PATH = Path(__file__).resolve().parents[1] / "data" / "oh_h2o_vacuum_lines.txt"
_MJD0 = _dt.datetime(1858, 11, 17)


def _quiet(fn):
    """Silence numpy floating-point flags around ``fn``: Accelerate BLAS on macOS raises
    spurious divide/overflow/invalid flags in small matmuls (results are exact; checked)."""
    import functools

    @functools.wraps(fn)
    def wrapped(*a, **k):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return fn(*a, **k)

    return wrapped


def _jax64():
    """Enable JAX float64 (every fit here runs in double precision)."""
    import jax

    jax.config.update("jax_enable_x64", True)


# =============================================================================== line table
@lru_cache(maxsize=4)
def _parse_line_table(path: str):
    oh = {k: [] for k in ("lam", "I", "band", "branch", "omega", "J", "parity", "group",
                          "nuerr", "lam_oliva")}
    ab = {k: [] for k in ("species", "nu0", "S", "gamma", "delta", "depth60", "blend60",
                          "depth25", "blend25")}
    with open(path) as fh:
        for ln in fh:
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            t = ln.split()
            if t[0] == "OH":
                oh["lam"].append(float(t[1]))
                oh["I"].append(float(t[2]))
                oh["band"].append(t[3])
                oh["branch"].append(t[4])
                oh["omega"].append(t[5])
                oh["J"].append(float(t[6]))
                oh["parity"].append(t[7])
                oh["group"].append(int(t[8]))
                oh["nuerr"].append(int(t[9]))
                oh["lam_oliva"].append(float(t[10]))
            else:
                ab["species"].append(t[0])
                for k, v in zip(("nu0", "S", "gamma", "delta", "depth60", "blend60", "depth25",
                                 "blend25"), t[1:9], strict=True):
                    ab[k].append(float(v))
    oh = {k: np.asarray(v) for k, v in oh.items()}
    ab = {k: np.asarray(v) for k, v in ab.items()}
    oh["vu"] = np.array([int(b.split("-")[0]) for b in oh["band"]])
    oh["vl"] = np.array([int(b.split("-")[1]) for b in oh["band"]])
    return oh, ab


@dataclass(frozen=True)
class LineTable:
    """The curated vacuum OH + H2O/O2 line table (``decanter/data/oh_h2o_vacuum_lines.txt``).

    ``oh``: ``lam`` [A, HITRAN], ``I`` (model relative photon intensity), ``vu``/``vl``,
    ``branch``, ``omega``, ``J``, ``parity``, ``group`` (Lambda-doublet id), ``nuerr``,
    ``lam_oliva`` (B&C 2009 position or nan).  ``absorbers``: ``species``, ``nu0`` [cm-1],
    ``S``, ``gamma``, ``delta`` [cm-1/atm], ``depth60/25``, ``blend60/25``.
    """

    oh: dict
    absorbers: dict
    path: Path

    @classmethod
    def load(cls, path: str | Path | None = None) -> LineTable:
        p = Path(path) if path is not None else LINE_TABLE_PATH
        oh, ab = _parse_line_table(str(p))
        return cls(oh=oh, absorbers=ab, path=p)

    def oh_lines(self, lo: float, hi: float) -> dict:
        """OH components with ``lo < lam < hi`` (HITRAN positions)."""
        s = (self.oh["lam"] > lo) & (self.oh["lam"] < hi)
        return {k: v[s] for k, v in self.oh.items()}

    def absorber_lines(self, lo: float, hi: float, *, species=("H2O",), R: float = 60000.0,
                       p_eff: dict | None = None) -> dict:
        """Absorber anchor candidates in ``(lo, hi)`` with pressure-shifted vacuum positions
        ``1e8 / (nu0 + delta P_eff)`` and the depth / blend metrics of the nearest tabulated
        resolution (60k or 25k)."""
        p_eff = dict(_P_EFF if p_eff is None else p_eff)
        a = self.absorbers
        P = np.array([p_eff.get(s, 0.0) for s in a["species"]])
        lam = 1e8 / (a["nu0"] + a["delta"] * P)
        s = (lam > lo) & (lam < hi) & np.isin(a["species"], list(species))
        col = "60" if R >= 40000.0 else "25"
        return dict(lam=lam[s], species=a["species"][s], depth=a["depth" + col][s],
                    blend=a["blend" + col][s], S=a["S"][s], nu0=a["nu0"][s])


_P_EFF = (("H2O", 0.60), ("O2", 0.38))


# ========================================================================= ThAr template
def _series(kind: int, xn, coef, deriv: int = 0):
    from numpy.polynomial import chebyshev as Ch
    from numpy.polynomial import legendre as Le

    mod = Ch if kind == 1 else Le
    c = coef
    for _ in range(deriv):
        c = mod.chebder(c) if kind == 1 else mod.legder(c)
    return mod.chebval(xn, c) if kind == 1 else mod.legval(xn, c)


class _Dispersion:
    """Shared inverse / derivative machinery for per-order dispersion functions."""

    orders: tuple

    def lam(self, m: int, p):          # pragma: no cover - interface
        raise NotImplementedError

    def dlam_dp(self, m: int, p):      # pragma: no cover - interface
        raise NotImplementedError

    def p_range(self, m: int) -> tuple[float, float]:   # pragma: no cover - interface
        raise NotImplementedError

    def p_of_lam(self, m: int, lam, n_newton: int = 6) -> np.ndarray:
        """Inverse ``p(lambda)`` (Newton from a dense-table start), float64."""
        lam = np.asarray(lam, np.float64)
        a, b = self.p_range(m)
        xs = np.linspace(a - 0.1 * (b - a), b + 0.1 * (b - a), 4097)
        ls = self.lam(m, xs)
        o = np.argsort(ls)
        p = np.interp(lam, ls[o], xs[o])
        for _ in range(n_newton):
            p = p - (self.lam(m, p) - lam) / self.dlam_dp(m, p)
        return p

    def velocity_per_pixel(self, m: int, p) -> np.ndarray:
        """``c dlambda/dp / lambda`` [km/s per template pixel]."""
        return C_KMS * self.dlam_dp(m, p) / self.lam(m, p)


@dataclass(frozen=True)
class ThArTemplate(_Dispersion):
    """Per-order ``identify`` dispersion series ``lambda_m(p)`` (the period ThAr solution).

    ``p`` is WARP's truncated rectified-strip pixel (s11 output, ``p = (V - 1)/dy + 1``), the
    coordinate dispcor maps; ``dy`` (comp ``CDELT1``) converts s10 shifts (V units) to ``p``.
    """

    orders: tuple
    kind: np.ndarray        # (K,) 1 = Chebyshev, 2 = Legendre
    coef: np.ndarray        # (K, n) series coefficients (zero padded)
    pmin: np.ndarray        # (K,)
    pmax: np.ndarray        # (K,)
    dy: float | None = None
    source: str = ""

    def _k(self, m: int) -> int:
        try:
            return self.orders.index(int(m))
        except ValueError:
            raise KeyError(f"order {m} not in template {self.orders}") from None

    def _xn(self, k: int, p):
        return (2.0 * np.asarray(p, np.float64) - (self.pmax[k] + self.pmin[k])) / (
            self.pmax[k] - self.pmin[k])

    def lam(self, m: int, p) -> np.ndarray:
        k = self._k(m)
        return _series(int(self.kind[k]), self._xn(k, p), self.coef[k])

    def dlam_dp(self, m: int, p) -> np.ndarray:
        k = self._k(m)
        return _series(int(self.kind[k]), self._xn(k, p), self.coef[k], 1) * 2.0 / (
            self.pmax[k] - self.pmin[k])

    def p_range(self, m: int) -> tuple[float, float]:
        k = self._k(m)
        return float(self.pmin[k]), float(self.pmax[k])

    @classmethod
    def from_arrays(cls, orders, coef, pmin, pmax, *, kind=1, dy=None, source="arrays"):
        coef = np.atleast_2d(np.asarray(coef, np.float64))
        K = len(orders)
        return cls(orders=tuple(int(m) for m in orders),
                   kind=np.broadcast_to(np.asarray(kind, int), (K,)).copy(), coef=coef,
                   pmin=np.broadcast_to(np.asarray(pmin, np.float64), (K,)).copy(),
                   pmax=np.broadcast_to(np.asarray(pmax, np.float64), (K,)).copy(),
                   dy=dy, source=source)

    @classmethod
    def from_id_dir(cls, id_dir: str | Path, refname: str, *, orders=None,
                    dy: float | None = None) -> ThArTemplate:
        """Read ``<id_dir>/id<refname>.<mmmm>`` (dispcor's view of the coefficients)."""
        from decanter.wavelength.iraf_msp import _msp_coeff, read_id_tokens

        id_dir = Path(id_dir)
        found = {}
        for f in sorted(id_dir.glob(f"id{refname}.[0-9][0-9][0-9][0-9]")):
            m = int(f.name.rsplit(".", 1)[1])
            if orders is not None and m not in orders:
                continue
            toks = read_id_tokens(f)
            ftype = round(float(toks[0]))
            nord = round(float(toks[1]))
            if ftype not in (1, 2):
                raise NotImplementedError(f"{f}: dispersion function type {ftype}")
            found[m] = (ftype, _msp_coeff(toks[2]), _msp_coeff(toks[3]),
                        np.array([_msp_coeff(t) for t in toks[4:4 + nord]], np.float64))
        if not found:
            raise FileNotFoundError(f"no id{refname}.* files in {id_dir}")
        ms = sorted(found)
        n = max(len(v[3]) for v in found.values())
        coef = np.zeros((len(ms), n))
        for k, m in enumerate(ms):
            coef[k, : len(found[m][3])] = found[m][3]
        return cls(orders=tuple(ms), kind=np.array([found[m][0] for m in ms]), coef=coef,
                   pmin=np.array([found[m][1] for m in ms]),
                   pmax=np.array([found[m][2] for m in ms]), dy=dy,
                   source=f"{id_dir}/id{refname}.*")

    @classmethod
    def from_calibration(cls, calib, *, orders=None) -> ThArTemplate:
        """From a :class:`decanter.Calibration` (``id_dir``, ``id_refname``, comp ``CDELT1``)."""
        dy = None
        try:
            from decanter.api import _comp_dy

            dy = float(_comp_dy(Path(calib.comp)))
        except Exception:   # dy only matters for undoing s10
            pass
        return cls.from_id_dir(calib.id_dir, calib.id_refname, orders=orders, dy=dy)

    @classmethod
    def from_warp_reduction(cls, root: str | Path, *, orders=None) -> ThArTemplate:
        """From a WARP reduction root's ``calibration_data`` copy (``database/idcomp_*``)."""
        root = Path(root).expanduser()
        cd = root / "calibration_data"
        comps = sorted(cd.glob("comp_*.fits"))
        if not comps:
            raise FileNotFoundError(f"no comp_*.fits in {cd}")
        dy = None
        try:
            from astropy.io import fits

            dy = float(fits.getheader(comps[0])["CDELT1"])
        except Exception:
            pass
        return cls.from_id_dir(cd / "database", comps[0].stem, orders=orders, dy=dy)


@dataclass(frozen=True)
class EcSolution(_Dispersion):
    """2-D ``ecidentify`` solution: ``lambda = (f(x, m) + shift) / m``, ``f`` a tensor
    Chebyshev (or Legendre) series in normalised pixel ``x`` and order ``m``."""

    orders: tuple
    kind: int
    nx: int
    nm: int
    xmin: float
    xmax: float
    omin: float
    omax: float
    coef: np.ndarray        # (nm, nx): coef[j, i] multiplies B_i(x~) B_j(m~)
    shift: float = 0.0
    source: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> EcSolution:
        rec = Path(path).read_text().split("begin")[-1].splitlines()
        vals, shift, count = [], 0.0, None
        for line in rec:
            t = line.split()
            if not t:
                continue
            if t[0] == "shift" and len(t) > 1:
                shift = float(t[1])
            if t[0] == "coefficients":
                count = int(t[1])
                continue
            if count is not None and len(vals) < count:
                vals.extend(float(v) for v in t)
        if count is None:
            raise ValueError(f"{path}: no coefficients block")
        v = np.asarray(vals)
        kind, nx, nm, _xterms = int(v[0]), int(v[1]), int(v[2]), int(v[3])
        xmin, xmax, omin, omax = v[4:8]
        c = v[8:8 + nx * nm].reshape(nm, nx)
        orders = tuple(range(int(omin), int(omax) + 1))
        return cls(orders=orders, kind=kind, nx=nx, nm=nm, xmin=float(xmin), xmax=float(xmax),
                   omin=float(omin), omax=float(omax), coef=c, shift=shift, source=str(path))

    def _f(self, m, p, deriv=0):
        xn = (2.0 * np.asarray(p, np.float64) - (self.xmax + self.xmin)) / (self.xmax - self.xmin)
        on = (2.0 * float(m) - (self.omax + self.omin)) / (self.omax - self.omin)
        cx = np.array([_series(self.kind, on, self.coef[:, i]) for i in range(self.nx)])
        return _series(self.kind, xn, cx, deriv)

    def lam(self, m: int, p) -> np.ndarray:
        return (self._f(m, p) + self.shift) / float(m)

    def dlam_dp(self, m: int, p) -> np.ndarray:
        return self._f(m, p, 1) * 2.0 / (self.xmax - self.xmin) / float(m)

    def p_range(self, m: int) -> tuple[float, float]:
        return self.xmin, self.xmax


def static_lnlam_grid(template: _Dispersion, m: int, *, dv_kms: float = 1.5,
                      p_range: tuple[float, float] | None = None) -> np.ndarray:
    """Static float64 ``lambda_ref`` grid of order ``m``, uniform in ``ln lambda`` with step
    ``dv_kms`` over the template's pixel range (the fast-mode knot grid convention)."""
    a, b = p_range if p_range is not None else template.p_range(m)
    l0, l1 = sorted((float(template.lam(m, a)), float(template.lam(m, b))))
    n = math.floor(math.log(l1 / l0) / (dv_kms / C_KMS)) + 1
    return l0 * np.exp(np.arange(n) * (dv_kms / C_KMS))


# ========================================================================== containers
@dataclass
class SpectrumSet:
    """One kind of extracted spectra for a night: static labels, per-frame flux.

    Attributes:
        kind: ``"sky"`` or ``"obj"``.
        lam: ``{m: (n_m,) float64}`` static wavelength labels (ThAr-template scale) shared by
            every frame (WARP: the dispcor output grid; fast mode: ``lambda_ref``).
        flux: ``{m: (F, n_m)}``.
        time: (F,) MJD at mid-exposure of the exposure the light was recorded in.
        nod: (F,) aperture slit position ``"A"``/``"B"`` (``""`` unknown).
        epoch: (F,) raw-exposure ids (FRAMEID); default: time rounded to 1 s.
        ivar: optional ``{m: (F, n_m)}`` inverse variances (relative weights suffice).
        shift_px: optional (F,) per-frame shift already applied to the flux, in template
            pixels: the raw pixel of label pixel ``p`` is ``p - shift_px[f]`` (WARP s10/s11:
            ``waveShiftAdopted / dy``).  None = labels are raw.
        frame_ids: optional per-frame names.
        airmass: optional (F,) airmass of each exposure (header ``AIRMASS``).
    """

    kind: str
    lam: dict
    flux: dict
    time: np.ndarray
    nod: np.ndarray
    epoch: np.ndarray | None = None
    ivar: dict | None = None
    shift_px: np.ndarray | None = None
    frame_ids: tuple | None = None
    airmass: np.ndarray | None = None

    def __post_init__(self):
        self.time = np.asarray(self.time, np.float64)
        F = len(self.time)
        self.nod = np.asarray([str(n)[:1].upper() if n is not None else "" for n in self.nod])
        if len(self.nod) != F:
            raise ValueError(f"{len(self.nod)} nod entries for {F} frames")
        # a missing id (None / "" / "None", e.g. no FRAMEID) falls back to the time rounded to
        # 1 s PER FRAME: a shared placeholder would tie every frame to one "exposure" jitter
        ep = [None] * F if self.epoch is None else list(self.epoch)
        if len(ep) != F:
            raise ValueError(f"{len(ep)} epochs for {F} frames")
        self.epoch = np.asarray([str(e) if e is not None and str(e).strip() not in ("", "None")
                                 else f"t{round(t * 86400.0)}" for e, t in zip(ep, self.time,
                                                                             strict=True)])
        for m, a in self.flux.items():
            if np.shape(a)[0] != F or np.shape(a)[1] != len(self.lam[m]):
                raise ValueError(f"order {m}: flux shape {np.shape(a)} vs {F} frames x "
                                 f"{len(self.lam[m])} labels")
            lam = np.asarray(self.lam[m])
            if lam.dtype == np.float32:
                warnings.warn(f"order {m}: float32 wavelength labels (ulp 0.027 km/s at 1.08 um); "
                              "pass float64 labels", stacklevel=2)
            fin = lam[np.isfinite(lam)]
            if fin.size > 1 and not np.all(np.diff(fin) > 0):
                raise ValueError(f"order {m}: labels must increase strictly (searchsorted / "
                                 "interp windows); reverse flux and labels together")
        if self.shift_px is not None:
            self.shift_px = np.asarray(self.shift_px, np.float64)
        if self.airmass is not None:
            self.airmass = np.asarray(self.airmass, np.float64)
            if self.airmass.shape != (F,):
                raise ValueError(f"{self.airmass.shape} airmass entries for {F} frames")

    @property
    def n_frames(self) -> int:
        return len(self.time)

    @property
    def orders(self) -> tuple:
        return tuple(sorted(self.flux))

    def raw_lambda(self, m: int, template: _Dispersion | None) -> np.ndarray:
        """(F, n) float64 wavelengths of the RAW (unshifted) pixels behind each label."""
        lam = np.asarray(self.lam[m], np.float64)
        if self.shift_px is None or not np.any(self.shift_px):
            return np.broadcast_to(lam, (self.n_frames, lam.size))
        if template is None:
            raise ValueError("shift_px needs a dispersion template to undo")
        p = template.p_of_lam(m, lam)
        return template.lam(m, p[None, :] - self.shift_px[:, None])


@dataclass
class NightSpectra:
    """Sky and object spectra of one night plus the ThAr shape template."""

    sky: SpectrumSet | None
    obj: SpectrumSet | None
    template: _Dispersion
    mode: str = "HIRES-Y"
    name: str = ""
    meta: dict = field(default_factory=dict)


def _hms_seconds(hms) -> float | None:
    """``"hh:mm:ss.s"`` -> seconds of the day, or None if it does not parse."""
    try:
        hh, mm, ss = str(hms).strip().split(":")
        s = int(hh) * 3600.0 + int(mm) * 60.0 + float(ss)
    except (TypeError, ValueError):
        return None
    return s if math.isfinite(s) and 0.0 <= s <= 86400.0 else None


def _mjd_mid(h) -> float:
    """Mid-exposure MJD from DATE-OBS (UT date of UT-STR) + UT-STR and the duration.

    The duration is ``UT-END - UT-STR`` (wrapped over UT midnight; it includes the readout,
    and it is the true one for an aborted exposure), cross-checked against ``EXPTIME``: an
    unparseable / missing UT-END, one implying more than ``EXPTIME + 900 s`` (3 h without a
    usable EXPTIME; e.g. a placeholder ``00:00:00``) or exactly 0 s, falls back to ``EXPTIME``
    (0 if that is corrupt too), with a warning.  A corrupt EXPTIME alone changes nothing.
    Only DATE-OBS and UT-STR are required."""
    d = str(h["DATE-OBS"]).strip()[:10]
    base = _dt.datetime.strptime(d, "%Y-%m-%d")
    s0 = _hms_seconds(h["UT-STR"])
    if s0 is None:
        raise ValueError(f"unparseable UT-STR {h['UT-STR']!r} (FRAMEID {h.get('FRAMEID')})")
    try:
        exptime = float(h.get("EXPTIME"))
        exptime = exptime if math.isfinite(exptime) and 0.0 < exptime < 1e4 else None
    except (TypeError, ValueError):
        exptime = None
    s1 = _hms_seconds(h.get("UT-END"))
    dur = None
    if s1 is not None:
        dur = s1 - s0 if s1 >= s0 else s1 + 86400.0 - s0
        if dur > (exptime + 900.0 if exptime is not None else 3 * 3600.0) or (
                dur == 0.0 and exptime is not None):     # UT-END copied from UT-STR
            dur = None
    if dur is None:
        dur = exptime if exptime is not None else 0.0
        warnings.warn(f"FRAMEID {h.get('FRAMEID')}: UT-END {h.get('UT-END')!r} unusable "
                      f"(EXPTIME {h.get('EXPTIME')!r}); mid-exposure = UT-STR + {dur / 2:.1f} s",
                      stacklevel=3)
    return ((base - _MJD0).total_seconds() + s0 + 0.5 * dur) / 86400.0


def _hdr_float(h, key) -> float:
    """Header value as float (nan when missing or unparseable)."""
    try:
        v = float(h.get(key))
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _read_1d(path):
    from astropy.io import fits

    with fits.open(path, memmap=False) as hd:
        h = hd[0].header
        y = np.asarray(hd[0].data, np.float64)
    cd = float(h.get("CD1_1", h["CDELT1"]))
    w = float(h["CRVAL1"]) + cd * (np.arange(y.size) + 1.0 - float(h["CRPIX1"]))
    return w, y, h


def load_warp_night(root: str | Path, *, fsr: str = "fsr1.30", obj_kind: str = "VAC_flux",
                    orders: Sequence[int] | None = None, with_obj: bool = True,
                    frames: Sequence[int] | None = None) -> NightSpectra:
    """Read a WARP reduction (``reductions/<test>``) as a :class:`NightSpectra`.

    Sky: ``<frame>/sky_emission/VAC/<fsr>`` (the partner frame extracted at the object
    aperture, never shifted by s11): time / epoch from its own header, nod = the object's.
    Object: ``<frame>/onedspec/<obj_kind>/<fsr>`` with ``shift_px = waveShiftAdopted / dy``
    (s10/s11 undone through the template when labels are evaluated).
    """
    root = Path(root).expanduser()
    dirs = sorted(glob.glob(str(root / "*_NO*")),
                  key=lambda d: int(re.search(r"_NO(\d+)$", d.rstrip("/")).group(1)))
    dirs = [d for d in dirs if Path(d).is_dir()]
    nos = [int(re.search(r"_NO(\d+)$", d).group(1)) for d in dirs]
    if frames is not None:
        keep = set(int(f) for f in frames)
        dirs = [d for d, n in zip(dirs, nos, strict=True) if n in keep]
        nos = [n for n in nos if n in keep]
    if not dirs:
        raise FileNotFoundError(f"no *_NO<i> frame directories in {root}")
    template = ThArTemplate.from_warp_reduction(root)

    def scan(d, sub, pat):
        out = {}
        for f in glob.glob(str(Path(d) / sub / "*.fits")):
            mm = re.search(pat, Path(f).name)
            if mm:
                out[int(mm.group(1))] = f
        return out

    sky_files = [scan(d, f"sky_emission/VAC/{fsr}", r"_m(\d+)trans1dcutw") for d in dirs]
    obj_files = [scan(d, f"onedspec/{obj_kind}/{fsr}", r"_m(\d+)_fsr") for d in dirs]
    # a frame WARP produced nothing for (e.g. an aborted / unreadable exposure) is dropped
    # instead of emptying the order intersection for the whole night
    empty = [i for i in range(len(dirs)) if not sky_files[i] or (with_obj and not obj_files[i])]
    if empty and len(empty) < len(dirs):
        warnings.warn(f"{root}: no {fsr} products in {[Path(dirs[i]).name for i in empty]}; "
                      "frame(s) skipped", stacklevel=2)
        kp = [i for i in range(len(dirs)) if i not in set(empty)]
        dirs, nos = [dirs[i] for i in kp], [nos[i] for i in kp]
        sky_files, obj_files = [sky_files[i] for i in kp], [obj_files[i] for i in kp]
    common = set(template.orders)
    for s in sky_files:
        common &= set(s)
    if with_obj:
        for s in obj_files:
            common &= set(s)
    if orders is not None:
        common &= set(int(m) for m in orders)
    ms = sorted(common)
    if not ms:
        raise FileNotFoundError(f"no order common to every frame in {root}")

    def build(files, kind):
        lam, flux, hdrs = {}, {m: [] for m in ms}, []
        for fset in files:
            h0 = None
            for m in ms:
                w, y, h = _read_1d(fset[m])
                if m not in lam:
                    lam[m] = w
                elif w.size != lam[m].size or np.max(np.abs(w - lam[m])) > 1e-6:
                    y = np.interp(lam[m], w, y, left=np.nan, right=np.nan)
                flux[m].append(y)
                h0 = h0 or h
            hdrs.append(h0)
        return lam, {m: np.asarray(v) for m, v in flux.items()}, hdrs

    slam, sflux, shdr = build(sky_files, "sky")
    obj = None
    ohdr = None
    if with_obj:
        olam, oflux, ohdr = build(obj_files, "obj")
        shift_px = None
        wl = root / "reduction_log" / "waveshift_log.npz"
        if wl.is_file() and template.dy:
            sv = np.asarray(np.load(wl, allow_pickle=True)["waveShiftAdopted"], np.float64)
            if sv.size >= max(nos):
                shift_px = sv[np.asarray(nos) - 1] / template.dy
            else:
                warnings.warn(f"{wl}: {sv.size} shifts for {len(nos)} frames; s10 not undone",
                              stacklevel=2)
        else:   # WARP always applies s11: without the log the object labels stay s10-shifted
            warnings.warn(f"{root}: " + ("no reduction_log/waveshift_log.npz" if not wl.is_file()
                                         else "comp CDELT1 (dy) unreadable")
                          + "; WARP's s10/s11 shift is NOT undone (object labels taken as raw)",
                          stacklevel=2)
        obj = SpectrumSet(kind="obj", lam=olam, flux=oflux,
                          time=[_mjd_mid(h) for h in ohdr],
                          nod=[str(h.get("NODPOS", "")) for h in ohdr],
                          epoch=[str(h.get("FRAMEID", "")) or None for h in ohdr],
                          shift_px=shift_px, frame_ids=tuple(Path(d).name for d in dirs),
                          airmass=[_hdr_float(h, "AIRMASS") for h in ohdr])
    obj_nod = ([str(h.get("NODPOS", "")) for h in ohdr] if ohdr is not None else
               [{"A": "B", "B": "A"}.get(str(h.get("NODPOS", ""))[:1], "") for h in shdr])
    sky = SpectrumSet(kind="sky", lam=slam, flux=sflux, time=[_mjd_mid(h) for h in shdr],
                      nod=obj_nod, epoch=[str(h.get("FRAMEID", "")) or None for h in shdr],
                      frame_ids=tuple(Path(d).name for d in dirs))
    mode = str(shdr[0].get("INSTMODE", "HIRES-Y"))
    return NightSpectra(sky=sky, obj=obj, template=template, mode=mode, name=root.name,
                        meta={"root": str(root), "fsr": fsr, "obj_kind": obj_kind})


def night_from_reductions(reductions: Sequence, template: _Dispersion, *, cut: float = 1.30,
                          mode: str | None = None, name: str = "") -> NightSpectra:
    """Build a :class:`NightSpectra` from :class:`decanter.Reduction` objects (one per frame,
    each with ``obj``/``sky`` spectra and the raw ``obj_header``/``sky_header``).  Works for
    warp-mode reductions (linear WCS) and any later mode that fills ``OrderSpectrum``;
    ``shift_wave`` (V units) becomes ``shift_px`` through ``template.dy``."""
    ms = sorted(set.intersection(*[{m for (c, m) in r.obj if abs(c - cut) < 1e-6}
                                   for r in reductions]))
    def wave64(sp):        # float64 labels (OrderSpectrum.wavelength follows the flux dtype)
        n = np.asarray(sp.flux).size
        return float(sp.crval1) + (np.arange(1, n + 1, dtype=np.float64) - float(sp.crpix1)) * float(
            sp.cdelt1)

    def stack(get):
        lam = {m: wave64(get(reductions[0])[(cut, m)]) for m in ms}
        flux = {m: np.asarray([np.asarray(get(r)[(cut, m)].flux, np.float64) for r in reductions])
                for m in ms}
        return lam, flux
    olam, oflux = stack(lambda r: r.obj)
    oh = [r.obj_header for r in reductions]
    if any(h is None for h in oh):
        raise ValueError("night_from_reductions needs the raw obj_header of every reduction")
    shifts = np.array([float(getattr(r, "shift_wave", 0.0)) for r in reductions])
    shift_px = None
    if np.any(shifts):
        if getattr(template, "dy", None):
            shift_px = shifts / template.dy
        else:
            warnings.warn("night_from_reductions: shift_wave set but the template has no dy; "
                          "the per-frame shift is NOT undone", stacklevel=2)
    obj = SpectrumSet(kind="obj", lam=olam, flux=oflux, time=[_mjd_mid(h) for h in oh],
                      nod=[str(h.get("NODPOS", "")) for h in oh],
                      epoch=[str(h.get("FRAMEID", "")) or None for h in oh], shift_px=shift_px,
                      airmass=[_hdr_float(h, "AIRMASS") for h in oh])
    sky = None
    if all(r.sky is not None and r.sky_header is not None for r in reductions):
        slam, sflux = stack(lambda r: r.sky)
        sh = [r.sky_header for r in reductions]
        sky = SpectrumSet(kind="sky", lam=slam, flux=sflux, time=[_mjd_mid(h) for h in sh],
                          nod=obj.nod, epoch=[str(h.get("FRAMEID", "")) or None for h in sh])
    return NightSpectra(sky=sky, obj=obj, template=template,
                        mode=mode or str(oh[0].get("INSTMODE", "HIRES-Y")), name=name)


# ============================================================================ config
@dataclass(frozen=True)
class AnchorConfig:
    """Knobs of :func:`anchor_night`; :meth:`for_mode` gives the measured WINERED defaults."""

    R: float = 60000.0              # resolving power for templates (HIRES measured 60-64k)
    xcorr_vmax: float = 8.0         # per-frame drift search, km/s
    global_vmax: float = 40.0       # line-list vs ThAr-scale offset search, km/s
    oh_imin: float = 2e-4           # min model I_rel of a feature's main group
    oh_risk: float = 0.01           # max modelled centroid-bias risk, FWHM
    oh_pad: float = 1.6             # feature window pad, FWHM
    snr_frame: float = 3.0          # per-frame line S/N to use a fit
    snr_comb_oh: float = 60.0       # combined (quadrature over frames) S/N to adopt a line
    snr_comb_tell: float = 30.0
    frac_valid: float = 0.6         # fraction of frames with a valid fit to adopt a line
    huber: float = 2.0              # Huber threshold (robust sigma units)
    line_scatter_max: float = 3.0   # drop lines whose scatter exceeds this x median
    tell_species: tuple | None = None   # None: H2O only (O2 opt-in, see ``species``)
    tell_depth: tuple = (0.04, 0.7)
    tell_blend: float = 0.1
    tell_halfwin: float = 1.4       # telluric window half width, FWHM
    tell_risk: float = 0.02         # max modelled neighbour centroid-bias risk, FWHM
    p_eff: tuple = _P_EFF
    exclude_tell_orders: tuple = ()     # orders held out of the star term (validation)
    sky_shape: tuple | None = (2, 2)    # static (x, m) Chebyshev degrees+1 for SKY labels
    # (x, m) Chebyshev terms of the detector modes x per-frame regressors, or "auto"
    # (:meth:`modes_for`: (3, 3) with >= 16 orders, else (2, 2); (2, 1) below 3 orders).
    # (3, 3) beat (2, 2) on 10 held-out orders of full TOI-2109 / TOI-3486 / WASP-69 nights
    # (star-label residual rms summed 1.20 -> 1.08 km/s; the drift slope varies quadratically
    # with m on WIDE and HIRES-J), but over-fits a few-order night (3-order synthetic test).
    sky_modes: tuple | str | None = "auto"    # x (nod, drift) regressors, sky
    star_modes: tuple | str | None = "auto"   # x (X - <X>, nod) regressors, star
    # Conditioning guard of the detector modes (:func:`_guard_modes`): a (qx, qm) basis is used
    # only with >= modes_min_lines_per_term x qx qm lines and a maximum prediction leverage
    # (unweighted, over every (x, m) of the night's orders) <= modes_max_leverage; otherwise
    # (3, 3) -> (2, 2) -> (2, 1) -> no modes.  On 36 HIRES / WIDE nights the leverage of the
    # adopted lines is 3-31 (tellurics, 3 x 3) and 10-71 (OH); TOI-270c's 13 telluric lines
    # (5 of 6 orders at one end of the echelle) give 1.2e8 and a 654 km/s mode swing.  It
    # triggers on 5 of the 52 WARP nights on disk: TOI-270c (13 telluric lines -> (2, 1)),
    # pi Men (7 OH lines -> (2, 1); 36 telluric, leverage 602 -> (2, 2)), AU Mic (37 OH, 1488),
    # HD 136352 (23 OH, 1108) and G 2938 (25 telluric, 9e4) -> (2, 2); the others reach <= 80.
    modes_min_lines_per_term: float = 2.0
    modes_max_leverage: float = 200.0
    star_beta_prior: tuple = (1.0, 0.3)
    star_gp_amp_scale: float = 0.3      # km/s, half-normal prior on the star-minus-sky GP
    gp_length_bounds: tuple = (0.1, 24.0)   # hours
    gp_robust_iter: int = 3
    n_iter_fit: int = 12            # LM iterations, width free (pass A)
    n_iter_fixed: int = 8           # LM iterations, width fixed (pass B)
    pass_a_frames: int = 24         # frames used to select lines (pass A)
    chunk: int = 2048
    mode: str = "HIRES-Y"
    # ---- single-image / sparse-epoch path (:func:`anchor_frames`); unused by anchor_night
    epoch_gap_h: float = 3.0        # a gap longer than this starts a new epoch group
    series_min_frames: int = 8      # "auto": a single epoch with >= this many frames ...
    series_min_span_h: float = 0.5  # ... spanning >= this many hours goes to anchor_night
    snr_ref_frames: int = 40        # snr_comb_* refer to this many frames (scaled per group)
    min_lines_static: int = 3       # fewer usable lines in a frame -> no direct zero point
    # z_inst error term for the sky-partner time offset (|dt| ~ 4-6 min): rms over 32 HIRES nights
    # of the drift between the partner and the object exposure beyond the measurement noise
    # (series z_sky - z_inst) is 0.61 km/s/h (median 0.39: 17 of 32 nights drift faster than 0.35)
    drift_rate_kms_h: float = 0.6
    star_fallback_err: float = 0.3  # km/s, star = OH zero point (+ slit prior) fallback
    surface_tau_default: float = 0.02   # km/s night-to-night line-offset scatter (1-night surface)
    use_star_shape: bool = True     # apply a surface's telluric (star) shape to star labels
    use_surface_modes: bool = True  # apply a surface's nod (x, m) pattern (A/B slit position)
    # ---- A-only frames (no sky spectrum: decanter.reduce(sky=None), no nod partner): the OH
    # airglow is still in the object spectrum and is measured there (:func:`anchor_frames`)
    # contaminants (OH emission in the telluric fits, telluric absorption in the OH fits) are
    # dropped: a feature whose fit window (+- margin) holds one is not used
    aonly_oh_imin: float = 2e-3         # OH lines that contaminate (model I_rel) ...
    aonly_oh_halfwidth: float = 0.5     # ... margin, FWHM
    aonly_tell_depth: float = 0.1       # telluric lines that contaminate (model depth) ...
    aonly_tell_halfwidth: float = 0.5   # ... margin (x (1 + min(depth/0.2, 2))), FWHM
    # "sparse" epochs: free per-frame zero points ("frame"), or per epoch group one zero point +
    # an A/B nod term + a linear drift ("nod_drift"; ABBA makes the nod and drift terms
    # orthogonal), for the sky (OH) and the star (telluric) zero points; z / z_err are then the
    # model values at every frame
    epoch_model: str = "frame"

    @classmethod
    def for_mode(cls, mode: str, **kw) -> AnchorConfig:
        mode = str(mode).upper()
        if mode.startswith("WIDE"):
            base = dict(R=24000.0, xcorr_vmax=25.0, global_vmax=60.0, snr_comb_oh=30.0,
                        snr_comb_tell=20.0, mode="WIDE")
        elif "J" in mode.split("-")[-1]:
            base = dict(R=60000.0, mode="HIRES-J")
        else:
            base = dict(R=60000.0, mode="HIRES-Y")
        base.update(kw)
        return cls(**base)

    @property
    def fwhm(self) -> float:
        return C_KMS / self.R

    def modes_for(self, kind: str, n_orders: int) -> tuple | None:
        """Effective ``(qx, qm)`` of the ``"sky"`` / ``"star"`` detector modes for a night
        with ``n_orders`` orders (an explicit tuple or None is used as given)."""
        q = self.sky_modes if kind == "sky" else self.star_modes
        if isinstance(q, str):
            if q != "auto":
                raise ValueError(f"{kind}_modes={q!r}: use a (qx, qm) tuple, None or 'auto'")
            return (3, 3) if n_orders >= 16 else (2, 2) if n_orders >= 3 else (2, 1)
        return None if q is None else (int(q[0]), int(q[1]))

    @property
    def species(self) -> tuple:
        """Telluric species of the star term.  Default H2O only: on TOI-3486 (HIRES-J) the
        1.27 um O2 lines drift differently from H2O (per-line slope vs the OH drift -0.08 vs
        +0.02 about the mean; likely O2 airglow residuals in A-B), so O2 is opt-in."""
        if self.tell_species is not None:
            return tuple(self.tell_species)
        return ("H2O",)


# =========================================================================== features
def build_oh_features(lines: LineTable, lo: float, hi: float, R: float, *, imin: float = 2e-4,
                      risk_max: float = 0.01, pad: float = 1.6, doublet_max: float = 4.0) -> list[dict]:
    """Lambda-doublet-aware OH feature templates for the wavelength range ``(lo, hi)``.

    A feature = the brightest unused component + its Lambda-doublet partner (if within
    ``doublet_max`` FWHM: the two parity components have fixed relative positions and model
    intensities, so a partly resolved doublet is fitted as ONE template instead of being
    rejected as a blend) + any component within 0.25 FWHM (``main``), plus every modelled component
    inside the fit window (``contaminants``), with positions fixed relative to each other and
    model intensities; the fit frees one shift.  ``clean`` = modelled centroid-bias risk
    (40 % intensity error of the contaminants) below ``risk_max`` FWHM, no strong (>30 %)
    neighbour wing outside 0.25 FWHM, and not dominated by a brighter contaminant.
    """
    fw = C_KMS / R
    margin = 3.0 * fw / C_KMS * hi
    L = lines.oh_lines(lo - margin, hi + margin)
    lam, inten, grp = L["lam"], L["I"], L["group"]
    if lam.size == 0:
        return []
    used = np.zeros(lam.size, bool)
    feats = []
    for j in np.argsort(-inten):
        if used[j] or inten[j] < imin:
            continue
        du = (lam - lam[j]) / lam[j] * C_KMS
        main = {int(j)}
        for p in np.where((grp == grp[j]) & (np.abs(du) < doublet_max * fw))[0]:
            main.add(int(p))
        for p in np.where(np.abs(du) < 0.25 * fw)[0]:
            main.add(int(p))
        main = np.array(sorted(main))
        Im = inten[main].sum()
        lref = float(np.sum(inten[main] * lam[main]) / Im)
        u = (lam - lref) / lref * C_KMS
        umin, umax = u[main].min() - pad * fw, u[main].max() + pad * fw
        inw = np.where((u > umin - fw) & (u < umax + fw))[0]
        cont = np.array([p for p in inw if p not in set(main)], int)
        if cont.size:
            dist = np.clip(np.maximum(umin - u[cont], u[cont] - umax), 0, None)
            wgt = np.exp(-0.5 * (dist / (fw / 2.3548)) ** 2)
            risk = 0.4 * float(np.sum(inten[cont] * wgt / Im * np.abs(u[cont])))
            strong = bool(np.any((inten[cont] > 0.3 * Im) & (np.abs(u[cont]) > 0.25 * fw)))
            dominated = bool(np.any(inten[cont] > Im))
        else:
            risk, strong, dominated = 0.0, False, False
        used[main] = True
        comps = np.r_[main, cont]
        if not (lo < lref < hi):
            continue
        feats.append(dict(lref=lref, umin=float(umin), umax=float(umax), cu=u[comps],
                          cr=inten[comps] / Im, Irel=float(Im), risk=risk,
                          clean=bool(risk < risk_max * fw and not strong and not dominated),
                          band=f"{L['vu'][j]}-{L['vl'][j]}", branch=str(L["branch"][j]),
                          J=float(L["J"][j]), dsep=float(np.ptp(u[main])), nmain=int(main.size),
                          species="OH", sign=1.0))
    return sorted(feats, key=lambda f: f["lref"])


def build_absorber_features(lines: LineTable, lo: float, hi: float, R: float, *,
                            species=("H2O",), p_eff=None, depth=(0.04, 0.7),
                            blend_max: float = 0.1, halfwin: float = 1.4,
                            risk_max: float = 0.02) -> list[dict]:
    """Telluric absorption features: the anchor line plus every tabulated absorber (any
    species) inside the window +-(``halfwin`` + 1) FWHM as fixed-ratio components (relative
    depths from the table at this resolution), strongest first, no two anchors within
    2.5 FWHM.  ``blend_max`` cuts on the table's line-centre blend metric; ``risk_max`` (FWHM)
    on the modelled centroid bias if the neighbours' depths were 40 % off (as for OH)."""
    fw = C_KMS / R
    margin = (halfwin + 1.5) * fw / C_KMS * hi
    L = lines.absorber_lines(lo, hi, species=species, R=R, p_eff=p_eff)
    A = lines.absorber_lines(lo - margin, hi + margin, species=("H2O", "O2", "CO2", "CH4"),
                             R=R, p_eff=p_eff)
    s = (L["depth"] >= depth[0]) & (L["depth"] <= depth[1]) & (L["blend"] <= blend_max)
    lam, dep, sp = L["lam"][s], L["depth"][s], L["species"][s]
    out = []
    sg = fw / 2.3548
    for j in np.argsort(-dep):
        if any(abs(lam[j] - f["lref"]) / lam[j] * C_KMS < 2.5 * fw for f in out):
            continue
        u = (A["lam"] / lam[j] - 1.0) * C_KMS
        near = (np.abs(u) < (halfwin + 1.0) * fw) & (np.abs(u) > 1e-6)
        uc, rc = u[near], A["depth"][near] / dep[j]
        if uc.size:
            dist = np.clip(np.abs(uc) - halfwin * fw, 0, None)
            wgt = np.exp(-0.5 * (dist / sg) ** 2)
            risk = 0.4 * float(np.sum(rc * wgt * np.abs(uc)))
            dominated = bool(np.any(rc[np.abs(uc) < halfwin * fw] > 1.0))
        else:
            risk, dominated = 0.0, False
        out.append(dict(lref=float(lam[j]), umin=-halfwin * fw, umax=halfwin * fw,
                        cu=np.r_[0.0, uc], cr=np.r_[1.0, rc], Irel=float(dep[j]), risk=risk,
                        clean=bool(risk < risk_max * fw and not dominated), band="", branch="",
                        J=0.0, dsep=0.0, nmain=1, species=str(sp[j]), sign=-1.0))
    return sorted(out, key=lambda f: f["lref"])


# ===================================================================== centroid fitter
@lru_cache(maxsize=32)
def _centroid_kernel(W: int, C: int, n_iter: int, fit_sigma: bool):
    import jax
    import jax.numpy as jnp

    _jax64()
    nlin = 3

    def single(U, Y, Wt, cu, cr, s0, d0, dlim):
        Uc = U - d0
        sc = jnp.exp(s0)
        un = Uc / (4.0 * sc)
        sw = jnp.sqrt(Wt)
        ones = jnp.ones_like(Uc)

        def design(q):
            sig = jnp.exp(q[1])
            z = (Uc[:, None] - cu[None, :] - q[0]) / sig
            T = jnp.exp(-0.5 * z * z) @ cr
            return jnp.stack([ones, un, T], axis=1)

        def proj(q):
            B = design(q) * sw[:, None]
            A = B.T @ B
            A = A + (1e-10 * jnp.trace(A) / nlin + 1e-300) * jnp.eye(nlin)
            th = jnp.linalg.solve(A, B.T @ (sw * Y))
            return sw * Y - B @ th, th

        def rfun(q):
            return proj(q)[0]

        frozen = 0.0 if fit_sigma else 1.0

        def body(_, st):
            q, lam, cost = st
            r = rfun(q)
            J = jax.jacfwd(rfun)(q)
            J = J * jnp.array([1.0, 1.0 - frozen])
            g = J.T @ r
            H = J.T @ J
            H = H + lam * jnp.diag(jnp.diag(H)) + jnp.diag(jnp.array([1e-12, 1e-12 + frozen]))
            step = -jnp.linalg.solve(H, g)
            step = jnp.clip(step, jnp.array([-0.5 * sc, -0.25]), jnp.array([0.5 * sc, 0.25]))
            qn = q + step
            qn = qn.at[0].set(jnp.clip(qn[0], -dlim, dlim))
            cn = jnp.sum(rfun(qn) ** 2)
            ok = cn < cost
            return (jnp.where(ok, qn, q), jnp.where(ok, lam * 0.3, lam * 10.0),
                    jnp.where(ok, cn, cost))

        q0 = jnp.array([0.0, s0])
        q, _, _ = jax.lax.fori_loop(0, n_iter, body, (q0, 1e-3, jnp.sum(rfun(q0) ** 2)))
        r, th = proj(q)
        p = jnp.concatenate([q, th])

        def rfull(pp):
            return sw * (Y - design(pp[:2]) @ pp[2:])

        J5 = jax.jacfwd(rfull)(p) * jnp.array([1.0, 1.0 - frozen, 1.0, 1.0, 1.0])
        n = jnp.sum(Wt > 0)
        dof = jnp.maximum(n - (5 - frozen), 1.0)
        s2 = jnp.sum(r * r) / dof
        M = J5.T @ J5 + jnp.diag(jnp.array([0.0, frozen, 0.0, 0.0, 0.0])) + 1e-14 * jnp.eye(5)
        cov = jnp.linalg.inv(M) * s2
        rr = jnp.where(Wt > 0, r, jnp.nan)
        mad = 1.4826 * jnp.nanmedian(jnp.abs(rr - jnp.nanmedian(rr)))
        rpk = jnp.nanmax(jnp.abs(rr)) / (mad + 1e-300)
        return dict(d=d0 + q[0], sigma=jnp.exp(q[1]), amp=th[2], b0=th[0],
                    e_d=jnp.sqrt(jnp.abs(cov[0, 0])), e_amp=jnp.sqrt(jnp.abs(cov[4, 4])),
                    chi2=s2, n=n, rpk=rpk)

    return jax.jit(jax.vmap(single))


def fit_line_centroids(U, Y, Wt, cu, cr, sigma, d0, dlim, *, fit_sigma: bool = False,
                       n_iter: int = 14, chunk: int = 4096) -> dict:
    """Batched template-centroid fits (JAX float64, Levenberg-Marquardt on the shift (and
    log width) with the linear parameters ``(b0, b1, a)`` projected out).

    Model per window: ``y = b0 + b1 u + a sum_c r_c exp(-(u - u_c - d)^2 / 2 sigma^2)``.

    Args:
        U, Y, Wt: (N, W) sample velocities [km/s rel. to the feature reference], data and
            weights (0 = unused sample).
        cu, cr: (N, C) template component offsets [km/s] and relative intensities (pad 0).
        sigma: (N,) Gaussian sigma [km/s] (start value when ``fit_sigma``).
        d0: (N,) start shift [km/s]; dlim: (N,) max |d - d0|.

    Returns:
        dict of (N,) numpy arrays ``d, e_d, sigma, amp, e_amp, b0, chi2, n, rpk, snr``
        (``e_*`` scaled by the window's reduced chi^2; NaN where a window has < 6 samples).
    """
    U = np.asarray(U, np.float64)
    N, W = U.shape
    C = np.shape(cu)[1]
    if N == 0:
        return {k: np.zeros(0) for k in ("d", "e_d", "sigma", "amp", "e_amp", "b0", "chi2",
                                         "n", "rpk", "snr")}
    Wp = int(16 * math.ceil(W / 16))
    Cp = int(4 * math.ceil(C / 4))

    def pad(a, w, fill=0.0):
        a = np.asarray(a, np.float64)
        return np.pad(a, ((0, 0), (0, w - a.shape[1])), constant_values=fill)

    U, Y, Wt = pad(U, Wp), pad(np.nan_to_num(Y), Wp), pad(Wt, Wp)
    cu, cr = pad(cu, Cp), pad(cr, Cp)
    s0 = np.log(np.asarray(sigma, np.float64))
    d0 = np.asarray(d0, np.float64)
    dlim = np.asarray(dlim, np.float64)
    f = _centroid_kernel(Wp, Cp, int(n_iter), bool(fit_sigma))
    ch = int(min(chunk, max(256, 1 << math.ceil(math.log2(max(N, 1))))))
    out = []
    for a in range(0, N, ch):
        b = min(N, a + ch)
        sl = slice(a, b)
        npad = ch - (b - a)
        args = [U[sl], Y[sl], Wt[sl], cu[sl], cr[sl], s0[sl], d0[sl], dlim[sl]]
        if npad:
            args = [np.concatenate([x, np.repeat(x[-1:], npad, axis=0)]) for x in args]
        r = f(*args)
        out.append({k: np.asarray(v)[: b - a] for k, v in r.items()})
    res = {k: np.concatenate([o[k] for o in out]) for k in out[0]}
    bad = res["n"] < 6
    for k in ("d", "e_d", "sigma", "amp", "e_amp", "b0", "chi2", "rpk"):
        res[k] = np.where(bad, np.nan, res[k])
    res["snr"] = np.abs(res["amp"]) / res["e_amp"]
    return res


def _gather(raw: np.ndarray, flux: np.ndarray, ivar: np.ndarray | None, lref: np.ndarray,
            lo_u: np.ndarray, hi_u: np.ndarray):
    """Windows of every (feature, frame) of one order.  raw/flux: (F, n); lref (K,);
    lo_u/hi_u (K, F) velocity bounds.  Returns U, Y, Wt of shape (K, F, Wmax)."""
    F, n = raw.shape
    K = lref.size
    lo_l = lref[:, None] * (1.0 + lo_u / C_KMS)
    hi_l = lref[:, None] * (1.0 + hi_u / C_KMS)
    i0 = np.empty((K, F), np.int64)
    i1 = np.empty((K, F), np.int64)
    same = raw.base is not None and raw.strides[0] == 0
    if same:
        i0[:] = np.searchsorted(raw[0], lo_l)
        i1[:] = np.searchsorted(raw[0], hi_l)
    else:
        for f in range(F):
            i0[:, f] = np.searchsorted(raw[f], lo_l[:, f])
            i1[:, f] = np.searchsorted(raw[f], hi_l[:, f])
    cnt = np.clip(i1 - i0, 0, None)
    Wm = max(int(cnt.max()) if cnt.size else 0, 1)
    ar = np.arange(Wm)
    idx = np.clip(i0[..., None] + ar, 0, n - 1)
    fi = np.arange(F)[None, :, None]
    L = raw[fi, idx]
    Y = flux[fi, idx]
    Wt = (ar < cnt[..., None]).astype(np.float64)
    if ivar is not None:
        Wt = Wt * np.nan_to_num(ivar[fi, idx])
    Wt = np.where(np.isfinite(Y), Wt, 0.0)
    U = (L / lref[:, None, None] - 1.0) * C_KMS
    return U, Y, Wt


def _fit_features(sset: SpectrumSet, template, feats_by_order: dict, d0: dict, *,
                  sigma: dict, dlim: dict, fit_sigma: bool, cfg: AnchorConfig,
                  frames: np.ndarray | None = None, n_iter: int | None = None) -> dict:
    """Per-frame fits of every feature; returns (K_total, F) arrays + per-feature meta.
    ``frames`` restricts to a subset of frames (``d0[m]`` is then (K_m, len(frames)))."""
    rows = {k: [] for k in ("d", "e_d", "sigma", "amp", "e_amp", "snr", "rpk", "chi2")}
    meta = []
    batches = []
    fr = np.arange(sset.n_frames) if frames is None else np.asarray(frames)
    for m in sorted(feats_by_order):
        feats = feats_by_order[m]
        if not feats:
            continue
        raw = sset.raw_lambda(m, template)
        flux = np.asarray(sset.flux[m], np.float64)
        iv = None if sset.ivar is None or m not in sset.ivar else np.asarray(sset.ivar[m])
        if frames is not None:
            raw = raw[fr] if not (raw.base is not None and raw.strides[0] == 0) else \
                np.broadcast_to(raw[0], (fr.size, raw.shape[1]))
            flux = flux[fr]
            iv = None if iv is None else iv[fr]
        lref = np.array([f["lref"] for f in feats])
        dd = d0[m]                                      # (K_m, F)
        lo_u = np.array([f["umin"] for f in feats])[:, None] + dd
        hi_u = np.array([f["umax"] for f in feats])[:, None] + dd
        U, Y, Wt = _gather(raw, flux, iv, lref, lo_u, hi_u)
        K, F, Wm = U.shape
        C = max(f["cu"].size for f in feats)
        cu = np.zeros((K, C))
        cr = np.zeros((K, C))
        for i, f in enumerate(feats):
            cu[i, : f["cu"].size] = f["cu"]
            cr[i, : f["cr"].size] = f["cr"]
        batches.append((U.reshape(K * F, Wm), Y.reshape(K * F, Wm), Wt.reshape(K * F, Wm),
                        np.repeat(cu, F, axis=0), np.repeat(cr, F, axis=0),
                        np.broadcast_to(np.asarray(sigma[m])[:, None], (K, F)).reshape(-1),
                        dd.reshape(-1),
                        np.broadcast_to(np.asarray(dlim[m])[:, None], (K, F)).reshape(-1), K, F))
        meta.extend(dict(f, order=m) for f in feats)
    if not batches:
        return dict(meta=[], **{k: np.zeros((0, fr.size)) for k in rows})
    Wmax = max(b[0].shape[1] for b in batches)
    Cmax = max(b[3].shape[1] for b in batches)

    def padw(a, w):
        return np.pad(a, ((0, 0), (0, w - a.shape[1])))

    U = np.concatenate([padw(b[0], Wmax) for b in batches])
    Y = np.concatenate([padw(b[1], Wmax) for b in batches])
    Wt = np.concatenate([padw(b[2], Wmax) for b in batches])
    cu = np.concatenate([padw(b[3], Cmax) for b in batches])
    cr = np.concatenate([padw(b[4], Cmax) for b in batches])
    sg = np.concatenate([b[5] for b in batches])
    dd = np.concatenate([b[6] for b in batches])
    dl = np.concatenate([b[7] for b in batches])
    r = fit_line_centroids(U, Y, Wt, cu, cr, sg, dd, dl, fit_sigma=fit_sigma,
                           n_iter=n_iter or cfg.n_iter_fit, chunk=cfg.chunk)
    F = fr.size
    out = {k: r[k].reshape(-1, F) for k in rows}
    out["d0"] = dd.reshape(-1, F)
    out["dlim"] = dl.reshape(-1, F)
    out["meta"] = meta
    return out


# ============================================================ robust lines x frames model
@lru_cache(maxsize=16)
def _twoway_kernel(huber: float, n_iter: int, JP: int):
    """IRLS for ``d_kf = mu_k + z_f + sum_jp Phi_kj C_jp R_fp`` (``JP = J * P`` mode terms)."""
    import jax
    import jax.numpy as jnp

    _jax64()

    def run(D, E2, Vf, mu, z, Phi, Rg):
        J, P = Phi.shape[1], Rg.shape[1]

        def model(C):
            if JP == 0:
                return jnp.zeros_like(D)
            return Phi @ C.reshape(J, P) @ Rg.T

        def weights(r, s2):
            w0 = Vf / (E2 + s2)
            t = r * jnp.sqrt(w0)
            at = jnp.where(Vf > 0, jnp.abs(t), jnp.nan)
            scale = 1.4826 * jnp.nanmedian(at) + 1e-12
            hw = jnp.where(at <= huber * scale, 1.0, huber * scale / jnp.where(at > 0, at, 1.0))
            hw = jnp.where(Vf > 0, hw, 0.0)
            return w0 * hw, hw, scale

        def body(_, st):
            mu, z, s2, C = st
            Mo = model(C)
            r = D - mu[:, None] - z[None, :] - Mo
            w, hw, _ = weights(r, s2)
            sw_k = jnp.sum(w, axis=1)
            mu = jnp.where(sw_k > 0, jnp.sum(w * (D - z[None, :] - Mo), axis=1) / jnp.where(
                sw_k > 0, sw_k, 1.0), mu)
            cc = jnp.sum(sw_k * mu) / jnp.maximum(jnp.sum(sw_k), 1e-300)
            mu = mu - cc
            z = z + cc
            sw_f = jnp.sum(w, axis=0)
            z = jnp.where(sw_f > 0, jnp.sum(w * (D - mu[:, None] - Mo), axis=0) / jnp.where(
                sw_f > 0, sw_f, 1.0), z)
            if JP:
                rr = D - mu[:, None] - z[None, :]
                A = jnp.einsum("kf,kj,fp,ki,fq->jpiq", w, Phi, Rg, Phi, Rg).reshape(JP, JP)
                bvec = jnp.einsum("kf,kj,fp->jp", w * rr, Phi, Rg).reshape(JP)
                A = A + (1e-8 * jnp.trace(A) / JP + 1e-12) * jnp.eye(JP)
                C = jnp.linalg.solve(A, bvec)
            r = D - mu[:, None] - z[None, :] - model(C)
            ex = jnp.sum(hw * (r * r - E2)) / jnp.maximum(jnp.sum(hw), 1.0)
            s2 = jnp.maximum(0.0, 0.5 * s2 + 0.5 * jnp.maximum(ex, 0.0))
            return mu, z, s2, C

        C0 = jnp.zeros(max(JP, 1), D.dtype)[:JP] if JP else jnp.zeros(0, D.dtype)
        mu, z, s2, C = jax.lax.fori_loop(0, n_iter, body,
                                         (mu, z, jnp.zeros((), D.dtype), C0))
        r = D - mu[:, None] - z[None, :] - model(C)
        w, _, scale = weights(r, s2)
        Cov = jnp.zeros((JP, JP), D.dtype)
        if JP:
            A = jnp.einsum("kf,kj,fp,ki,fq->jpiq", w, Phi, Rg, Phi, Rg).reshape(JP, JP)
            Cov = jnp.linalg.inv(A + 1e-12 * jnp.eye(JP)) * scale ** 2
        return mu, z, s2, r, w, scale, C, Cov

    return jax.jit(run)


def _bucket(n: int, q: int = 32) -> int:
    return int(q * max(1, math.ceil(n / q)))


def robust_zero_points(d, e, *, huber: float = 2.0, n_iter: int = 40, valid=None,
                       modes=None, regressors=None) -> dict:
    """Robust two-way decomposition ``d_kf = mu_k + z_f [+ sum_jp Phi_kj C_jp R_fp] + eps``.

    Lines k x frames f.  ``Var eps = e_kf^2 + s_sys^2`` (``s_sys`` fitted), Huber IRLS, gauge:
    information-weighted mean of ``mu`` = 0 (so ``z`` is on the line list's absolute scale).
    Optional spatial ``modes`` Phi (K, J) times per-frame ``regressors`` R (F, P) (both
    centred here: Phi over lines, R over frames) model a drift that varies across the
    detector, e.g. ``Phi = (x, m)`` Chebyshev terms and ``R = (X_f - <X>, nod_f)``.  JAX float64.

    Returns dict: ``z, z_err`` (F,), ``mu, mu_err`` (K,), ``resid`` (K, F), ``w`` (K, F) final
    weights, ``s_sys``, ``scale`` (robust standardised-residual sigma), ``line_scatter`` (K,)
    robust per-line scatter [km/s], ``n_lines`` (F,), ``C`` (J, P) and ``C_cov`` (J P, J P)
    mode coefficients, ``phi_mean`` (J,) / ``reg_mean`` (P,) the centring offsets.
    """
    _jax64()
    d = np.asarray(d, np.float64)
    e = np.asarray(e, np.float64)
    V = np.isfinite(d) & np.isfinite(e) & (e > 0)
    if valid is not None:
        V &= np.asarray(valid, bool)
    K, F = d.shape
    if K == 0 or not V.any():
        nan = np.full(F, np.nan)
        return dict(z=nan, z_err=nan, mu=np.zeros(K), mu_err=np.zeros(K),
                    resid=np.full((K, F), np.nan), w=np.zeros((K, F)), s_sys=np.nan,
                    scale=np.nan, line_scatter=np.full(K, np.nan), n_lines=np.zeros(F, int),
                    C=np.zeros((0, 0)), C_cov=np.zeros((0, 0)), phi_mean=np.zeros(0),
                    reg_mean=np.zeros(0))
    dn = np.where(V, d, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        z0 = np.nanmedian(dn - np.nanmedian(dn, axis=1, keepdims=True), axis=0)
        z0 = np.where(np.isfinite(z0), z0, 0.0)
        mu0 = np.nanmedian(dn - z0[None, :], axis=1)
        mu0 = np.where(np.isfinite(mu0), mu0, 0.0)
    c = np.median(mu0)
    mu0, z0 = mu0 - c, z0 + c

    Kp, Fp = _bucket(K), _bucket(F)
    D = np.zeros((Kp, Fp))
    E2 = np.ones((Kp, Fp))
    Vf = np.zeros((Kp, Fp))
    D[:K, :F] = np.where(V, d, 0.0)
    E2[:K, :F] = np.where(V, e * e, 1.0)
    Vf[:K, :F] = V
    mu_p = np.zeros(Kp)
    z_p = np.zeros(Fp)
    mu_p[:K], z_p[:F] = mu0, z0
    use_modes = modes is not None and regressors is not None and np.size(modes) and np.size(
        regressors)
    if use_modes:
        Phi0 = np.atleast_2d(np.asarray(modes, np.float64)).reshape(K, -1)
        R0 = np.atleast_2d(np.asarray(regressors, np.float64)).reshape(F, -1)
        lw = V.any(axis=1)
        phi_mean = Phi0[lw].mean(axis=0)
        fw_ = V.any(axis=0)
        reg_mean = R0[fw_].mean(axis=0)
        J, P = Phi0.shape[1], R0.shape[1]
    else:
        J = P = 0
        phi_mean, reg_mean = np.zeros(0), np.zeros(0)
    Phi = np.zeros((Kp, J))
    Rg = np.zeros((Fp, P))
    if use_modes:
        Phi[:K] = Phi0 - phi_mean
        Rg[:F] = R0 - reg_mean
    run = _twoway_kernel(float(huber), int(n_iter), int(J * P))
    mu, z, s2, r, w, scale, Cm, Ccov = (np.asarray(a) for a in run(D, E2, Vf, mu_p, z_p, Phi, Rg))
    mu, z, r, w = mu[:K], z[:F], r[:K, :F], w[:K, :F]
    sw_f = w.sum(axis=0)
    sw_k = w.sum(axis=1)
    sc = max(float(scale), 1e-12)
    z_err = np.where(sw_f > 0, sc / np.sqrt(np.where(sw_f > 0, sw_f, 1.0)), np.nan)
    z = np.where(sw_f > 0, z, np.nan)
    mu_err = np.where(sw_k > 0, sc / np.sqrt(np.where(sw_k > 0, sw_k, 1.0)), np.nan)
    rr = np.where(V, r, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ls = 1.4826 * np.nanmedian(np.abs(rr), axis=1)
    return dict(z=z, z_err=z_err, mu=mu, mu_err=mu_err, resid=rr, w=w, s_sys=float(np.sqrt(s2)),
                scale=sc, line_scatter=ls, n_lines=(w > 0).sum(axis=0),
                C=Cm.reshape(J, P), C_cov=Ccov, phi_mean=phi_mean, reg_mean=reg_mean)


# ================================================================== static night shape
def _cheb2(xn, mn, qx, qm):
    from numpy.polynomial import chebyshev as Ch

    Tx = np.stack([Ch.chebval(xn, np.eye(qx)[i]) for i in range(qx)], -1)
    Tm = np.stack([Ch.chebval(mn, np.eye(qm)[j]) for j in range(qm)], -1)
    return (Tx[..., :, None] * Tm[..., None, :]).reshape((*np.shape(xn), qx * qm))


@_quiet
def _fit_shape(x, m, mu, err, qx, qm, xr, mr, huber=2.0):
    xn = (2 * np.asarray(x) - (xr[0] + xr[1])) / (xr[1] - xr[0])
    mn = (2 * np.asarray(m, float) - (mr[0] + mr[1])) / max(mr[1] - mr[0], 1e-9)
    A = _cheb2(xn, mn, qx, qm)
    w = 1.0 / np.asarray(err) ** 2
    keep = np.isfinite(mu) & np.isfinite(w)
    hw = np.ones_like(w)
    c = np.zeros(A.shape[1])
    for _ in range(6):
        ww = np.where(keep, w * hw, 0.0)
        Aw = A * np.sqrt(ww)[:, None]
        c = np.linalg.lstsq(Aw, np.nan_to_num(mu) * np.sqrt(ww), rcond=None)[0]
        t = (mu - A @ c) * np.sqrt(w)
        s = 1.4826 * np.nanmedian(np.abs(t[keep])) + 1e-12
        hw = np.where(np.abs(t) <= huber * s, 1.0, huber * s / np.maximum(np.abs(t), 1e-300))
    return c.reshape(qx, qm)


# ======================================================================== time model
def _matern32(r):
    s = math.sqrt(3.0) * r
    return (1.0 + s) * np.exp(-s) if isinstance(r, np.ndarray) else (1.0 + s) * _exp(-s)


def _exp(x):
    import jax.numpy as jnp

    return jnp.exp(x)


@dataclass
class TimeModel:
    """GP time model ``y_i = H_i b + g(t_i) + e(epoch_i) + noise_i``.

    ``g``: Matern-3/2 GP (amplitude ``amp`` km/s, length ``ell`` h); ``e``: per-exposure
    white jitter ``jit``; ``b ~ N(b_mean, b_cov)``; noise variance ``var / weights`` (robust
    weights from LOO residuals).  Fitted by :func:`fit_time_model`.
    """

    t_ref: float
    t: np.ndarray
    y: np.ndarray
    var: np.ndarray
    H: np.ndarray
    names: tuple
    b_mean: np.ndarray
    b_cov: np.ndarray
    epochs: np.ndarray
    amp: float
    ell: float
    jit: float
    weights: np.ndarray
    nll: float = np.nan
    n_used: int = 0

    def _kgp(self, ta, tb):
        return self.amp ** 2 * _matern32(np.abs(ta[:, None] - tb[None, :]) / self.ell)

    @_quiet
    def _ky(self):
        E = (self.epochs[:, None] == self.epochs[None, :]).astype(float)
        return (self._kgp(self.t, self.t) + self.jit ** 2 * E + self.H @ self.b_cov @ self.H.T
                + np.diag(self.var / self.weights))

    def _hours(self, t_mjd):
        return (np.asarray(t_mjd, np.float64) - self.t_ref) * 24.0

    @_quiet
    def predict(self, t_mjd, H_new, epochs_new=None, *, fixed: bool = True, gp: bool = True,
                jitter: bool = True, full_cov: bool = False):
        """Posterior mean and sd (or covariance) of the latent ``H b + g + e`` at new points.
        Components can be switched off (e.g. ``jitter=False`` = smooth model; ``fixed=False``
        with ``gp=False`` and ``jitter=True`` = the jitter alone)."""
        ts = self._hours(t_mjd)
        Hs = np.atleast_2d(np.asarray(H_new, np.float64))
        ep = (np.asarray([str(e) for e in epochs_new]) if epochs_new is not None
              else np.full(ts.size, "\x00"))
        Ky = self._ky()
        L = np.linalg.cholesky(Ky)
        r = self.y - self.H @ self.b_mean

        def cs(b):
            return np.linalg.solve(L.T, np.linalg.solve(L, b))

        Ksy = np.zeros((ts.size, self.t.size))
        Kss = np.zeros((ts.size, ts.size))
        if fixed:
            Ksy += Hs @ self.b_cov @ self.H.T
            Kss += Hs @ self.b_cov @ Hs.T
        if gp:
            Ksy += self._kgp(ts, self.t)
            Kss += self._kgp(ts, ts)
        if jitter:
            Ksy += self.jit ** 2 * (ep[:, None] == self.epochs[None, :])
            Kss += self.jit ** 2 * (ep[:, None] == ep[None, :])
        mean = (Hs @ self.b_mean if fixed else 0.0) + Ksy @ cs(r)
        cov = Kss - Ksy @ cs(Ksy.T)
        if full_cov:
            return mean, cov
        return mean, np.sqrt(np.clip(np.diag(cov), 0.0, None))

    @_quiet
    def fixed_effects(self):
        """Posterior mean and covariance of ``b``."""
        Ky = self._ky()
        r = self.y - self.H @ self.b_mean
        BH = self.b_cov @ self.H.T
        return self.b_mean + BH @ np.linalg.solve(Ky, r), self.b_cov - BH @ np.linalg.solve(Ky, BH.T)

    @_quiet
    def loo(self):
        """Leave-one-out residuals ``y_i - E[y_i | y_-i]`` and their sd."""
        Ki = np.linalg.inv(self._ky())
        a = Ki @ (self.y - self.H @ self.b_mean)
        dk = np.diag(Ki)
        return a / dk, 1.0 / np.sqrt(dk)


@lru_cache(maxsize=1)
def _gp_nll_kernel():
    """jit(value_and_grad) of the Matern-3/2 + jitter + fixed-effect marginal NLL."""
    import jax
    import jax.numpy as jnp

    _jax64()

    def nll(theta, w, D, E, HBH, vv, r, inv_amp_scale):
        amp, ell, jit = jnp.exp(theta[0]), jnp.exp(theta[1]), jnp.exp(theta[2])
        s = math.sqrt(3.0) * D / ell
        K = amp ** 2 * (1.0 + s) * jnp.exp(-s) + jit ** 2 * E + HBH + jnp.diag(vv / w)
        L = jnp.linalg.cholesky(K)
        a = jax.scipy.linalg.cho_solve((L, True), r)
        return (0.5 * r @ a + jnp.sum(jnp.log(jnp.diag(L))) + 0.5 * r.size * math.log(2 * math.pi)
                + 0.5 * (amp * inv_amp_scale) ** 2)

    return jax.jit(jax.value_and_grad(nll))


@_quiet
def fit_time_model(t_mjd, y, var, H, *, epochs=None, names=None, b_mean=None, b_cov=None,
                   length_bounds=(0.1, 24.0), robust_iter: int = 3, huber: float = 3.0,
                   t_ref: float | None = None, amp_scale: float | None = None) -> TimeModel:
    """Fit :class:`TimeModel` hyperparameters ``(amp, ell, jit)`` by maximising the marginal
    likelihood (JAX float64 value-and-grad, L-BFGS-B), with Huber reweighting of the noise
    from LOO-standardised residuals (``robust_iter`` rounds).  Non-finite ``y`` are dropped.
    ``amp_scale``: optional half-normal prior scale on the GP amplitude (MAP-II), used for
    the star-minus-sky term so the GP cannot absorb the ``beta X`` drift."""
    import jax.numpy as jnp
    from scipy.optimize import minimize

    _jax64()
    t_mjd = np.asarray(t_mjd, np.float64)
    y = np.asarray(y, np.float64)
    var = np.asarray(var, np.float64)
    H = np.atleast_2d(np.asarray(H, np.float64))
    if H.shape[0] != y.size:
        H = H.T
    ok = np.isfinite(y) & np.isfinite(var) & (var > 0) & np.all(np.isfinite(H), axis=1)
    p = H.shape[1]
    names = tuple(names) if names is not None else tuple(f"b{i}" for i in range(p))
    b_mean = np.zeros(p) if b_mean is None else np.asarray(b_mean, np.float64)
    b_cov = np.eye(p) * 100.0 ** 2 if b_cov is None else np.asarray(b_cov, np.float64)
    ep_all = (np.asarray([str(e) for e in epochs]) if epochs is not None
              else np.array([f"i{i}" for i in range(y.size)]))
    t_ref = float(np.mean(t_mjd[ok])) if t_ref is None else float(t_ref)
    t = (t_mjd[ok] - t_ref) * 24.0
    yy, vv, HH, ep = y[ok], var[ok], H[ok], ep_all[ok]
    n = yy.size
    if n == 0:
        raise ValueError("fit_time_model: no finite data")
    span = float(np.ptp(t)) if n > 1 else 1.0
    dt = float(np.median(np.diff(np.sort(t)))) if n > 2 else 0.1
    lo_ell = max(length_bounds[0], 1.5 * dt)
    hi_ell = max(length_bounds[1], 2.0 * span, lo_ell * 1.01)
    D = np.abs(t[:, None] - t[None, :])
    E = (ep[:, None] == ep[None, :]).astype(np.float64)
    HBH = HH @ b_cov @ HH.T
    r = yy - HH @ b_mean

    vg_core = _gp_nll_kernel()
    jargs = (*(jnp.asarray(a) for a in (D, E, HBH, vv, r)),
             jnp.asarray(0.0 if amp_scale is None else 1.0 / float(amp_scale)))

    def vg(theta, w):
        return vg_core(theta, w, *jargs)
    sd = float(np.std(r)) if n > 1 else 1.0
    theta = np.log([max(sd, 0.02), min(max(1.0, lo_ell * 2), hi_ell), 0.02])
    bounds = [(math.log(1e-3), math.log(50.0)), (math.log(lo_ell), math.log(hi_ell)),
              (math.log(1e-4), math.log(5.0))]
    w = np.ones(n)
    if robust_iter > 0 and n >= 8:
        # pre-screen gross frame outliers against a running median in time (so the jitter
        # term does not inflate to absorb them before the LOO reweighting sees them)
        o = np.argsort(t)
        rs = r[o]
        k = 3
        med = np.array([np.median(rs[max(0, i - k): i + k + 1]) for i in range(n)])
        dev = rs - med
        sc = 1.4826 * np.median(np.abs(dev)) + 1e-12
        se = np.sqrt(sc ** 2 + vv[o])
        bad = np.abs(dev) > 5.0 * se
        w[o[bad]] = 1e-4
    best = None
    for it in range(max(1, robust_iter + 1)):
        def fun(th, w=w):
            v, g = vg(jnp.asarray(th), jnp.asarray(w))
            v = float(v)
            g = np.asarray(g, np.float64)
            if not np.isfinite(v):
                return 1e30, np.zeros(3)
            return v, g

        starts = [theta] if it else [theta, np.log([max(sd, 0.02), hi_ell / 3, 0.02]),
                                     np.log([max(sd, 0.02), lo_ell * 1.5, max(sd / 3, 1e-3)])]
        res = None
        for s0 in starts:
            s0 = np.clip(s0, [b[0] for b in bounds], [b[1] for b in bounds])
            rr = minimize(fun, s0, jac=True, method="L-BFGS-B", bounds=bounds,
                          options=dict(maxiter=200))
            if res is None or rr.fun < res.fun:
                res = rr
        theta = res.x
        best = TimeModel(t_ref=t_ref, t=t, y=yy, var=vv, H=HH, names=names, b_mean=b_mean,
                         b_cov=b_cov, epochs=ep, amp=float(np.exp(theta[0])),
                         ell=float(np.exp(theta[1])), jit=float(np.exp(theta[2])), weights=w.copy(),
                         nll=float(res.fun), n_used=n)
        if it == robust_iter or n < 6:
            break
        lr, lsd = best.loo()
        st = np.abs(lr / lsd)
        w = np.where(st <= huber, 1.0, (huber / np.maximum(st, 1e-12)) ** 2)
        w = np.maximum(np.minimum(w, np.where(best.weights < 1e-3, 1e-4, 1.0)), 1e-4)
    return best


def _nod_code(nod) -> np.ndarray:
    nod = np.asarray(nod)
    return np.where(nod == "A", 0.5, np.where(nod == "B", -0.5, 0.0))


# ===================================================================== drift init
def _hp_rows(A, width):
    """Row-wise robust high-pass: subtract block medians (block = width/2 samples, linearly
    interpolated), clip at 12 robust sigma.  A: (F, n) -> (F, n)."""
    A = np.atleast_2d(np.asarray(A, np.float64))
    F, n = A.shape
    A = np.where(np.isfinite(A), A, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rowmed = np.nanmedian(A, axis=1, keepdims=True)
    A = np.where(np.isfinite(A), A, np.where(np.isfinite(rowmed), rowmed, 0.0))
    b = max(int(width) // 2, 3)
    nb = math.ceil(n / b)
    P = np.pad(A, ((0, 0), (0, nb * b - n)), mode="edge")
    med = np.median(P.reshape(F, nb, b), axis=2)                 # (F, nb)
    cen = (np.arange(nb) * b + 0.5 * (b - 1)).clip(0, n - 1)
    x = np.arange(n, dtype=np.float64)
    j = np.clip(np.searchsorted(cen, x) - 1, 0, max(nb - 2, 0))
    if nb > 1:
        w = np.clip((x - cen[j]) / (cen[j + 1] - cen[j]), 0.0, 1.0)
        base = med[:, j] * (1 - w) + med[:, j + 1] * w
    else:
        base = np.repeat(med, n, axis=1)
    h = A - base
    sd = 1.4826 * np.median(np.abs(h), axis=1, keepdims=True) + 1e-30
    return np.clip(h, -12 * sd, 12 * sd)


def _interp_rows(xq, xp, Y):
    """Linear interpolation of every row of ``Y`` (on the shared grid ``xp``) at ``xq``
    (shared, or one row per row of Y); 0 outside."""
    Y = np.asarray(Y, np.float64)
    if np.ndim(xq) == 1:
        j = np.clip(np.searchsorted(xp, xq) - 1, 0, xp.size - 2)
        w = (xq - xp[j]) / (xp[j + 1] - xp[j])
        out = Y[:, j] * (1 - w) + Y[:, j + 1] * w
        return np.where((xq >= xp[0]) & (xq <= xp[-1]), out, 0.0)
    raise ValueError("xq must be 1-D")


def _shift_rows(A, s):
    """Rows of A shifted by s[f] samples (A_out[f, i] = A[f, i + s_f]), linear, 0 outside."""
    F, n = A.shape
    x = np.arange(n)[None, :] + np.asarray(s)[:, None]
    i0 = np.floor(x).astype(np.int64)
    w = x - i0
    ok = (i0 >= 0) & (i0 + 1 < n)
    i0c = np.clip(i0, 0, n - 2)
    fi = np.arange(F)[:, None]
    return np.where(ok, A[fi, i0c] * (1 - w) + A[fi, i0c + 1] * w, 0.0)


def _xcorr_lags(ref, Y, L):
    """cc[f, l] = sum_i ref[i] Y[f, i + l] for l in [-L, L] (zero outside)."""
    n = ref.size
    out = np.zeros((Y.shape[0], 2 * L + 1))
    for j, lag in enumerate(range(-L, L + 1)):
        if lag >= 0:
            out[:, j] = np.einsum("fi,i->f", Y[:, lag:], ref[: n - lag])
        else:
            out[:, j] = np.einsum("fi,i->f", Y[:, : n + lag], ref[-lag:])
    return out


def _peak(cc, dv):
    k = np.argmax(cc, axis=-1)
    kk = np.clip(k, 1, cc.shape[-1] - 2)
    a = np.take_along_axis(cc, (kk - 1)[..., None], -1)[..., 0]
    b = np.take_along_axis(cc, kk[..., None], -1)[..., 0]
    c = np.take_along_axis(cc, (kk + 1)[..., None], -1)[..., 0]
    den = a - 2 * b + c
    frac = np.where(den < 0, 0.5 * (a - c) / np.where(den < 0, den, -1.0), 0.0)
    L = (cc.shape[-1] - 1) // 2
    return (kk - L + np.clip(frac, -1, 1)) * dv


@_quiet
def _xcorr_init(sset: SpectrumSet, template, lines: LineTable, cfg: AnchorConfig):
    """Per-frame drift (median 0) and the global line-list offset, both in km/s, from
    high-passed cross-correlation on uniform ln(lambda) grids (initialisation only).
    Returns ``drift, d_glob, dv, good`` (``good``: frames whose correlation peak is real;
    the others carry the time interpolation of the good frames)."""
    fw = cfg.fwhm
    steps = [np.median(np.diff(np.asarray(sset.lam[m])) / np.asarray(sset.lam[m])[1:]) * C_KMS
             for m in sset.orders]
    dv = float(np.min(steps))
    width = int(2 * round(6 * fw / dv) + 1)
    grids, HPs = {}, {}
    for m in sset.orders:
        lam = np.asarray(sset.lam[m], np.float64)
        l0, l1 = np.nanmin(lam), np.nanmax(lam)
        g = l0 * np.exp(np.arange(int(math.log(l1 / l0) / (dv / C_KMS))) * (dv / C_KMS))
        raw = sset.raw_lambda(m, template)
        fl = np.nan_to_num(np.asarray(sset.flux[m], np.float64))
        shared = raw.base is not None and raw.strides[0] == 0
        if shared:
            A = _interp_rows(g, raw[0], fl)
        else:
            A = np.stack([np.interp(g, raw[f], fl[f], left=0.0, right=0.0)
                          for f in range(sset.n_frames)])
        taper = np.minimum(1.0, np.minimum(np.arange(g.size), np.arange(g.size)[::-1]) / 20.0)
        grids[m], HPs[m] = g, _hp_rows(A, width) * taper
    L = math.ceil(cfg.xcorr_vmax / dv)
    drift = np.zeros(sset.n_frames)
    good = np.ones(sset.n_frames, bool)
    order_t = np.argsort(sset.time, kind="stable")
    for _ in range(2):
        tot = np.zeros((sset.n_frames, 2 * L + 1))
        for A in HPs.values():
            ref = np.median(_shift_rows(A, drift / dv)[good], axis=0)
            cc = _xcorr_lags(ref, A, L)
            tot += cc / (np.linalg.norm(ref) * np.linalg.norm(A, axis=1)[:, None] + 1e-30)
        drift = _peak(tot, dv)
        # a frame without signal (aborted / blank / all-NaN) peaks at noise or at the search
        # edge (+-xcorr_vmax): its drift would seed every window AND be a mode regressor, so
        # it is replaced by the time interpolation of the frames that did correlate
        pk = np.max(tot, axis=1)
        kk = np.argmax(tot, axis=1)
        med_pk = np.median(pk[np.isfinite(pk)]) if np.isfinite(pk).any() else 0.0
        good = np.isfinite(drift) & (kk > 0) & (kk < 2 * L) & (pk > 0.3 * med_pk) & (med_pk > 0)
        if good.any():
            og = order_t[good[order_t]]
            drift = np.where(good, drift, np.interp(sset.time, sset.time[og], drift[og]))
            drift -= np.median(drift[good])
        else:
            drift = np.zeros(sset.n_frames)
            good = np.ones(sset.n_frames, bool)
    # global offset of the (drift-aligned) night median against the OH model
    sig = fw / 2.3548
    Lg = math.ceil(cfg.global_vmax / dv)
    totg = np.zeros(2 * Lg + 1)
    for m, A in HPs.items():
        g = grids[m]
        ref_obs = np.median(_shift_rows(A, drift / dv)[good], axis=0)
        ol = lines.oh_lines(g[0], g[-1])
        model = np.zeros(g.size)
        for lc, Ii in zip(ol["lam"], ol["I"], strict=True):
            c = int(np.searchsorted(g, lc))
            h = int(5 * sig / dv) + 2
            sl = slice(max(c - h, 0), min(c + h, g.size))
            u = (g[sl] / lc - 1.0) * C_KMS
            model[sl] += Ii * np.exp(-0.5 * (u / sig) ** 2)
        model = _hp_rows(model[None, :], width)[0]
        cc = _xcorr_lags(model, ref_obs[None, :], Lg)[0]
        totg += cc / (np.linalg.norm(model) * np.linalg.norm(ref_obs) + 1e-30)
    # a peak that runs into the search edge is not a solution (pi Men: weak OH in 30 s
    # exposures, the edge at -40 km/s beat the real peak at +1.9): take the best INTERIOR
    # local maximum, at least one FWHM from either edge
    k = max(1, math.ceil(fw / dv))
    inner = np.full_like(totg, -np.inf)
    inner[k:-k] = totg[k:-k]
    loc = np.r_[False, (inner[1:-1] >= inner[:-2]) & (inner[1:-1] >= inner[2:]), False]
    j = int(np.argmax(np.where(loc, inner, -np.inf))) if loc.any() else Lg
    if np.argmax(totg) != j:
        warnings.warn(f"global line-list offset: the correlation maximum is at the search edge "
                      f"({(np.argmax(totg) - Lg) * dv:+.1f} km/s); using the interior peak "
                      f"({(j - Lg) * dv:+.1f} km/s)", stacklevel=3)
    win = np.full_like(totg, -np.inf)
    win[max(j - 1, 0):j + 2] = totg[max(j - 1, 0):j + 2]
    d_glob = float(_peak(win[None, :], dv)[0])
    return drift, d_glob, dv, good


# ============================================================ selection + decomposition
def _mode_basis(template, orders, lam, spec):
    """(K, J) detector modes ``T_a(x~) T_b(m~)``, (a, b) != (0, 0), at lines (order, lambda);
    ``x`` = template pixel.  ``spec``: dict(qx, qm, xr, mr)."""
    orders = np.asarray(orders)
    lam = np.asarray(lam, np.float64)
    x = np.empty(lam.size)
    for m in np.unique(orders):
        s = orders == m
        x[s] = template.p_of_lam(int(m), lam[s])
    xr, mr = spec["xr"], spec["mr"]
    xn = (2 * x - (xr[0] + xr[1])) / (xr[1] - xr[0])
    mn = (2 * orders.astype(float) - (mr[0] + mr[1])) / max(mr[1] - mr[0], 1e-9)
    B = _cheb2(xn, mn, spec["qx"], spec["qm"])
    return B[:, 1:]


_MODE_LADDER = ((3, 3), (2, 2), (2, 1))


@_quiet
def _mode_leverage(template, orders, lref, spec) -> float:
    """Maximum unweighted prediction leverage ``max_p phi_p^T (Phi^T Phi)^-1 phi_p`` of the
    centred ``(x, m)`` mode basis of ``spec`` fitted to lines ``(orders, lref)``, over a grid
    of 9 x-positions across the template pixel range ``spec["xr"]`` in every order of
    ``spec["mr"]`` (the label domain).  The variance of the fitted mode term at the worst
    point of the detector, in units of one line's variance (``inf`` if singular)."""
    Phi = _mode_basis(template, orders, lref, spec)
    K, J = Phi.shape
    if K <= J:
        return float("inf")
    mu = Phi.mean(axis=0)
    Pc = Phi - mu
    A = Pc.T @ Pc
    try:
        Ai = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return float("inf")
    if not np.all(np.isfinite(Ai)):
        return float("inf")
    mr = spec["mr"]
    om = np.arange(round(mr[0]), round(mr[1]) + 1, dtype=float)
    xn = np.linspace(-1.0, 1.0, 9)
    mn = (2 * om - (mr[0] + mr[1])) / max(mr[1] - mr[0], 1e-9)
    G = _cheb2(np.tile(xn, om.size), np.repeat(mn, xn.size), spec["qx"], spec["qm"])[:, 1:] - mu
    h = np.einsum("ij,jk,ik->i", G, Ai, G)
    return float(np.max(h)) if np.all(np.isfinite(h)) else float("inf")


def _guard_modes(template, orders, lref, spec, cfg):
    """Conditioning guard of a detector-mode basis (no WARP equivalent).  ``spec`` (or the
    next lower rung of ``(3, 3) -> (2, 2) -> (2, 1)``) is kept if the lines number at least
    ``cfg.modes_min_lines_per_term x qx qm`` and the basis' maximum prediction leverage over
    the detector (:func:`_mode_leverage`) is at most ``cfg.modes_max_leverage``; no rung
    passing -> None (no modes).  Returns ``(spec_used, info)``; ``spec`` is returned
    unchanged (the same object) when it passes."""
    q0 = (int(spec["qx"]), int(spec["qm"]))
    K = int(np.size(lref))
    ladder = [q0] + [q for q in _MODE_LADDER if q[0] * q[1] < q0[0] * q0[1]]
    tried = []
    for q in ladder:
        sp = spec if q == q0 else dict(spec, qx=q[0], qm=q[1])
        need = float(cfg.modes_min_lines_per_term) * q[0] * q[1]
        if K < need:
            tried.append(dict(q=q, n_lines=K, need=need, leverage=None, ok=False))
            continue
        lev = _mode_leverage(template, orders, lref, sp)
        ok = lev <= float(cfg.modes_max_leverage)
        tried.append(dict(q=q, n_lines=K, need=need, leverage=lev, ok=bool(ok)))
        if ok:
            return sp, dict(q_requested=q0, q_used=q, triggered=q != q0, tried=tried)
    return None, dict(q_requested=q0, q_used=None, triggered=True, tried=tried)


def _measure(sset, template, feats_by_order, d0_frame, cfg, *, sigma0, dlimA, dlimB,
             snr_comb, sign, mode_spec=None, regressors=None):
    """Pass A (width free) -> line selection -> pass B (width fixed) -> robust zero points
    (with detector modes x per-frame regressors when ``mode_spec`` is given)."""
    F = sset.n_frames
    ords = [m for m in sorted(feats_by_order) if feats_by_order[m]]
    feats_by_order = {m: feats_by_order[m] for m in ords}
    # pass A (selection, width free) on at most cfg.pass_a_frames evenly spaced frames
    nA = min(F, max(int(cfg.pass_a_frames), 1))
    ot = np.argsort(sset.time, kind="stable")      # evenly spaced IN TIME: order-invariant
    frA = np.sort(ot[np.unique(np.round(np.linspace(0, F - 1, nA)).astype(int))])
    d0 = {m: np.repeat(d0_frame[frA][None, :], len(feats_by_order[m]), 0) for m in ords}
    sig = {m: np.full(len(feats_by_order[m]), sigma0) for m in ords}
    dlA = {m: np.full(len(feats_by_order[m]), dlimA) for m in ords}
    A = _fit_features(sset, template, feats_by_order, d0, sigma=sig, dlim=dlA, fit_sigma=True,
                      cfg=cfg, frames=frA)
    if len(A["meta"]) == 0:
        return None
    sgn_ok = np.sign(A["amp"]) == sign
    vA = (np.isfinite(A["d"]) & (A["snr"] >= cfg.snr_frame) & sgn_ok
          & (A["sigma"] > 0.5 * sigma0) & (A["sigma"] < 2.2 * sigma0) & (A["rpk"] < 8.0)
          & (np.abs(A["d"] - A["d0"]) < 0.98 * A["dlim"]))
    frac = vA.mean(axis=1)
    comb = np.sqrt(np.nansum(np.where(vA, A["snr"], 0.0) ** 2, axis=1) * (F / frA.size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        smed = np.nanmedian(np.where(vA, A["sigma"], np.nan), axis=1)
    adopt = (frac >= cfg.frac_valid) & (comb >= snr_comb) & (smed > 0.6 * sigma0) & (
        smed < 1.8 * sigma0)
    if adopt.sum() < 1:
        return None
    twA = robust_zero_points(A["d"][adopt], A["e_d"][adopt], huber=cfg.huber, valid=vA[adopt])
    off = twA["z"] - d0_frame[frA]
    zA = d0_frame + (float(np.nanmedian(off)) if np.isfinite(off).any() else 0.0)
    idx = np.where(adopt)[0]
    meta = [A["meta"][i] for i in idx]
    byo = {}
    for j, f in enumerate(meta):
        byo.setdefault(f["order"], []).append(j)
    feats_B = {m: [meta[j] for j in js] for m, js in byo.items()}
    d0B = {m: twA["mu"][js][:, None] + zA[None, :] for m, js in byo.items()}
    sigB = {m: smed[idx[js]] for m, js in byo.items()}
    dlB = {m: np.full(len(js), dlimB) for m, js in byo.items()}
    B = _fit_features(sset, template, feats_B, d0B, sigma=sigB, dlim=dlB, fit_sigma=False,
                      cfg=cfg, n_iter=cfg.n_iter_fixed)
    vB = (np.isfinite(B["d"]) & (B["snr"] >= cfg.snr_frame) & (np.sign(B["amp"]) == sign)
          & (B["rpk"] < 8.0) & (np.abs(B["d"] - B["d0"]) < 0.98 * B["dlim"]))
    keep = np.ones(len(B["meta"]), bool)
    tw = None
    Phi = None
    guard = None
    ordB = np.array([f["order"] for f in B["meta"]], int)
    lrefB = np.array([f["lref"] for f in B["meta"]], np.float64)
    if mode_spec is not None and regressors is not None and len(B["meta"]):
        Phi = _mode_basis(template, ordB, lrefB, mode_spec)
    for it in range(4):
        if Phi is not None:
            # conditioning guard on the lines in use (passes without side effects on a
            # well-sampled night; a degenerate basis steps down (3,3) -> (2,2) -> (2,1) -> none)
            sp, gi = _guard_modes(template, ordB[keep], lrefB[keep], mode_spec, cfg)
            if guard is None or gi["triggered"]:
                guard = gi if guard is None else dict(gi, q_requested=guard["q_requested"])
            if sp is not mode_spec:
                mode_spec = sp
                Phi = None if sp is None else _mode_basis(template, ordB, lrefB, sp)
        tw = robust_zero_points(B["d"][keep], B["e_d"][keep], huber=cfg.huber, valid=vB[keep],
                                modes=None if Phi is None else Phi[keep], regressors=regressors)
        ls = tw["line_scatter"]
        med = np.nanmedian(ls)
        bad = ~(ls <= cfg.line_scatter_max * med)
        if not bad.any() or it == 3 or (keep.sum() - bad.sum()) < 2:
            break
        kk = np.where(keep)[0]
        keep[kk[bad]] = False
    out = dict(tw=tw, meta=[B["meta"][i] for i in np.where(keep)[0]],
               d=B["d"][keep], e=B["e_d"][keep], valid=vB[keep], sigma=np.concatenate(
                   [sigB[m] for m in sorted(sigB)])[keep] if sigB else np.zeros(0),
               n_candidates=len(A["meta"]), n_adopted=int(adopt.sum()), n_kept=int(keep.sum()),
               passA=dict(frac=frac, comb=comb, smed=smed), mode_spec=mode_spec,
               mode_guard=guard)
    return out


def _modes_ptp(template, lam_by_order: dict, md, R) -> float:
    """Peak-to-peak [km/s] of the mode term over the labelled range of every order (not the
    whole template pixel range, where the Chebyshev terms extrapolate) and the night's
    regressors."""
    vals = []
    for m, lab in sorted(lam_by_order.items()):
        lab = np.asarray(lab, np.float64)
        lab = lab[np.isfinite(lab)]
        if lab.size == 0:
            continue
        lam = np.linspace(lab[0], lab[-1], 9)
        Phi = _mode_basis(template, np.full(lam.size, int(m)), lam, md) - md["phi_mean"]
        vals.append((np.asarray(R) - md["reg_mean"]) @ (Phi @ md["C"]).T)
    v = np.concatenate([x.ravel() for x in vals]) if vals else np.zeros(0)
    return float(np.ptp(v)) if v.size else 0.0


# =========================================================================== solution
@dataclass
class AnchorSolution:
    """Result of :func:`anchor_night`.  Object-frame arrays follow ``night.obj`` order,
    sky arrays ``night.sky`` order.  Velocities in km/s on the RAW ThAr-template scale."""

    name: str
    mode: str
    config: AnchorConfig
    template: Any
    obj_time: np.ndarray | None
    obj_nod: np.ndarray | None
    obj_shift_px: np.ndarray | None
    z_star: np.ndarray | None
    z_star_err: np.ndarray | None
    z_inst: np.ndarray | None
    z_inst_err: np.ndarray | None
    sky_time: np.ndarray
    sky_nod: np.ndarray
    z_sky: np.ndarray
    z_sky_err: np.ndarray
    oh: dict
    tell: dict | None
    shape: dict | None
    oh_model: TimeModel
    star_model: TimeModel | None
    diagnostics: dict
    oh_modes: dict | None = None
    star_modes: dict | None = None
    star_shape: dict | None = None      # static telluric surface (single / sparse path only)
    method: str = "series"              # "series" (anchor_night), "single" or "sparse"

    @staticmethod
    def _u(z):
        return (-np.log1p(np.asarray(z, np.float64) / C_KMS)).astype(np.float32)

    def u(self, kind: str = "star") -> np.ndarray:
        """Per-frame label offsets ``u_f = -log1p(z_f / c)`` (float32): ``"star"`` (object
        frames, telluric-anchored; ``"tell"``: the same zero point), ``"inst"`` (OH zero point at the object exposures), ``"sky"`` (the sky
        spectra).  Excludes s10 undo and the sky shape (see :meth:`labels`)."""
        z = {"star": self.z_star, "tell": self.z_star, "inst": self.z_inst,
             "sky": self.z_sky}[kind]
        if z is None:
            raise ValueError(f"no {kind!r} solution (no object spectra?)")
        return self._u(z)

    @_quiet
    def shape_kms(self, m: int, lam) -> np.ndarray:
        """Static sky-shape velocity S(x, m) [km/s] at labels ``lam`` (0 without a shape)."""
        if self.shape is None:
            return np.zeros(np.shape(lam))
        p = self.template.p_of_lam(m, np.asarray(lam, np.float64))
        sh = self.shape
        xn = (2 * p - (sh["xr"][0] + sh["xr"][1])) / (sh["xr"][1] - sh["xr"][0])
        mn = (2 * float(m) - (sh["mr"][0] + sh["mr"][1])) / max(sh["mr"][1] - sh["mr"][0], 1e-9)
        c = np.array(sh["coef"], np.float64).copy()
        c[0, 0] = 0.0
        return _cheb2(xn, np.full_like(xn, mn), *c.shape) @ c.ravel()

    @_quiet
    def modes_kms(self, m: int, lam, kind: str = "star", frames=None) -> np.ndarray:
        """(F, n) detector-mode velocity ``sum_jp Phi_j(x, m) C_jp R_fp`` [km/s] (0 if none):
        ``kind="star"`` / ``"tell"`` (the telluric modes at the object frames), ``"inst"`` /
        ``"sky"`` (OH modes at the object / sky exposures)."""
        lam = np.asarray(lam, np.float64)
        md = self.star_modes if kind in ("star", "tell") else self.oh_modes
        z = {"star": self.z_star, "tell": self.z_star, "inst": self.z_inst,
             "sky": self.z_sky}[kind]
        fr = np.arange(len(z)) if frames is None else np.asarray(frames)
        if md is None:
            return np.zeros((fr.size, lam.size))
        R = md["R_sky"] if kind == "sky" else md.get("R_obj")
        if R is None:
            return np.zeros((fr.size, lam.size))
        Phi = _mode_basis(self.template, np.full(lam.size, int(m)), lam, md) - md["phi_mean"]
        Rc = np.asarray(R)[fr] - md["reg_mean"]
        return Rc @ (Phi @ md["C"]).T

    @_quiet
    def true_lambda(self, m: int, lam, kind: str = "star", frames=None) -> np.ndarray:
        """(F, n) float64 true vacuum wavelengths of labelled samples ``lam`` (order ``m``)."""
        lam = np.asarray(lam, np.float64)
        if kind == "sky":
            z = self.z_sky
            shift = None
        else:
            z = self.z_star if kind in ("star", "tell") else self.z_inst
            shift = self.obj_shift_px
        if z is None:
            raise ValueError(f"no {kind!r} solution")
        fr = np.arange(len(z)) if frames is None else np.asarray(frames)
        if shift is not None and np.any(shift):
            p = self.template.p_of_lam(m, lam)
            raw = self.template.lam(m, p[None, :] - shift[fr][:, None])
        else:
            raw = np.broadcast_to(lam, (fr.size, lam.size))
        star = kind in ("star", "tell")
        zz = z[fr][:, None] + (self.shape_kms(m, lam)[None, :] if not star else 0.0)
        zz = zz + self.modes_kms(m, lam, kind, fr)
        if star and self.star_shape is not None:
            zz = zz + _surface_kms(self.star_shape, self.template, m, lam)[None, :]
        return raw / (1.0 + zz / C_KMS)

    def labels(self, m: int, lam, kind: str = "star", frames=None) -> np.ndarray:
        """(F, n) float32 ``u = ln(lambda_true / lambda_label)`` per sample and frame."""
        lam = np.asarray(lam, np.float64)
        return np.log(self.true_lambda(m, lam, kind, frames) / lam[None, :]).astype(np.float32)

    def summary(self) -> dict:
        return dict(self.diagnostics)


# ====================================================================== orchestration
@_quiet
def anchor_night(night: NightSpectra, config: AnchorConfig | None = None, *,
                 lines: LineTable | None = None, verbose: bool = False) -> AnchorSolution:
    """Per-frame OH instrument zero points + telluric star term for one night.

    Replaces WARP's s10/s11 (``ccwaveshift.waveshiftClip`` + ``PySpecshift``; decanter
    ``waveshift/measure.py:warp_waveshift_log`` and ``waveshift/apply.py``), which measures
    shifts relative to one reference frame and sets every flagged frame to exactly 0
    (``measure.py:524``).  Nothing here is resampled and no frame is zeroed.

    Steps: (1) xcorr drift + global line-list offset (init); (2) OH features per order,
    per-frame fits (width free), adoption by combined S/N / frame coverage / width,
    per-frame refit at fixed width, robust lines x frames decomposition, optional static
    (x, m) sky shape; (3) OH time model (GP + jitter + nod); (4) telluric features in the
    object spectra (s10 undone), same measurement, star time model with the OH zero point as
    predictor; (5) labels.  ``diagnostics`` collects the validation numbers.
    """
    cfg = config or AnchorConfig.for_mode(night.mode)
    lines = lines or LineTable.load()
    tmpl = night.template
    sky = night.sky
    if sky is None:
        raise ValueError("anchor_night needs sky spectra (OH) for the instrument zero point")
    t_start = time.perf_counter()
    timing = {}
    fw = cfg.fwhm
    sig0 = fw / 2.3548
    # ---------------------------------------------------------------- (1) init
    drift, d_glob, _dv, xc_good = _xcorr_init(sky, tmpl, lines, cfg)
    timing["xcorr"] = time.perf_counter() - t_start
    # ---------------------------------------------------------------- (2) OH
    feats = {}
    for m in sky.orders:
        lam = np.asarray(sky.lam[m])
        fs = build_oh_features(lines, float(np.nanmin(lam)), float(np.nanmax(lam)), cfg.R,
                               imin=cfg.oh_imin, risk_max=cfg.oh_risk, pad=cfg.oh_pad)
        lo_ok = np.nanmin(lam) * (1 + (cfg.xcorr_vmax + fw * (cfg.oh_pad + 1)) / C_KMS)
        hi_ok = np.nanmax(lam) * (1 - (cfg.xcorr_vmax + fw * (cfg.oh_pad + 1)) / C_KMS)
        feats[m] = [f for f in fs if f["clean"] and lo_ok < f["lref"] < hi_ok]
    t1 = time.perf_counter()
    nod_s = _nod_code(sky.nod)
    has_nod = bool(np.any(nod_s != 0) and len(set(sky.nod[nod_s != 0])) > 1)
    xr_all = tmpl.p_range(int(sky.orders[0]))
    sky_spec = None
    R_sky = None
    sky_q = cfg.modes_for("sky", len(sky.orders))
    if sky_q is not None:
        sky_spec = dict(qx=sky_q[0], qm=sky_q[1], xr=xr_all,
                        mr=(float(min(sky.orders)), float(max(sky.orders))))
        R_sky = np.column_stack(([nod_s] if has_nod else []) + [drift])
    oh = _measure(sky, tmpl, feats, d_glob + drift, cfg, sigma0=sig0,
                  dlimA=max(0.6 * fw, 3.0), dlimB=max(0.35 * fw, 1.5), snr_comb=cfg.snr_comb_oh,
                  sign=1.0, mode_spec=sky_spec, regressors=R_sky)
    if oh is None:
        raise RuntimeError("no usable OH lines")
    sky_guard = oh["mode_guard"]
    if sky_spec is not None and oh["mode_spec"] is not sky_spec:      # conditioning guard
        sky_spec = oh["mode_spec"]
        sky_q = None if sky_spec is None else (sky_spec["qx"], sky_spec["qm"])
        warnings.warn(f"{night.name}: sky detector modes reduced to {sky_q} by the "
                      f"conditioning guard ({len(oh['meta'])} OH lines)", stacklevel=2)
    zref, zref_used = 0.0, False
    if sky_spec is not None and R_sky is not None and len(oh["meta"]) and \
            np.isfinite(oh["tw"]["z"]).sum() >= 3:
        # refit the decomposition with the MEASURED zero point as the drift regressor (the
        # xcorr init drift is 0.06-0.4 km/s noisy on the validation nights; its noise would
        # scale the mode term and leak into z through the line-weighted mean of the modes)
        z1 = oh["tw"]["z"]
        fz = np.isfinite(z1)
        zref = float(np.median(z1[fz]))
        r_dr = np.where(fz, z1, drift + float(np.median(z1[fz] - drift[fz]))) - zref
        R_sky = np.column_stack(([nod_s] if has_nod else []) + [r_dr])
        Phi = _mode_basis(tmpl, [f["order"] for f in oh["meta"]],
                          [f["lref"] for f in oh["meta"]], sky_spec)
        oh["tw"] = robust_zero_points(oh["d"], oh["e"], huber=cfg.huber, valid=oh["valid"],
                                      modes=Phi, regressors=R_sky)
        zref_used = True
    timing["oh_fits"] = time.perf_counter() - t1
    tw = oh["tw"]
    z_oh, e_oh = tw["z"].copy(), tw["z_err"].copy()
    mu = tw["mu"].copy()
    xs = np.array([float(tmpl.p_of_lam(f["order"], f["lref"] * (1 + (mu[i] + np.nanmedian(
        z_oh)) / C_KMS))) for i, f in enumerate(oh["meta"])])
    ms_line = np.array([f["order"] for f in oh["meta"]])
    shape = None
    if cfg.sky_shape is not None and len(oh["meta"]) >= 3 * int(np.prod(cfg.sky_shape)):
        qx, qm = cfg.sky_shape
        xr = tmpl.p_range(int(ms_line[0]))
        mr = (float(min(sky.orders)), float(max(sky.orders)))
        coef = _fit_shape(xs, ms_line, mu, np.sqrt(tw["mu_err"] ** 2 + tw["s_sys"] ** 2 + 1e-6),
                          qx, qm, xr, mr)
        c00 = coef[0, 0]
        z_oh = z_oh + c00
        mu = mu - c00
        coef = coef.copy()
        coef[0, 0] = 0.0
        resid_mu = mu - _cheb2((2 * xs - sum(xr)) / (xr[1] - xr[0]),
                               (2 * ms_line - sum(mr)) / max(mr[1] - mr[0], 1e-9), qx, qm) @ coef.ravel()
        shape = dict(coef=coef, xr=xr, mr=mr, c00=float(c00),
                     mu_rms_before=float(np.sqrt(np.nanmean(mu ** 2))),
                     mu_rms_after=float(np.sqrt(np.nanmean(resid_mu ** 2))))
    # ---------------------------------------------------------------- (3) OH time model
    t2 = time.perf_counter()
    Hs = np.column_stack([np.ones(sky.n_frames), nod_s]) if has_nod else np.ones((sky.n_frames, 1))
    names = ("b0", "nod") if has_nod else ("b0",)
    zc = float(np.nanmedian(z_oh))
    bm = np.r_[zc, np.zeros(Hs.shape[1] - 1)]
    bc = np.diag([20.0 ** 2] + [2.0 ** 2] * (Hs.shape[1] - 1))
    ohm = fit_time_model(sky.time, z_oh, e_oh ** 2, Hs, epochs=sky.epoch, names=names,
                         b_mean=bm, b_cov=bc, length_bounds=cfg.gp_length_bounds,
                         robust_iter=cfg.gp_robust_iter)
    z_sky, z_sky_err = ohm.predict(sky.time, Hs, sky.epoch)
    sm_sky, _ = ohm.predict(sky.time, Hs, sky.epoch, jitter=False)
    b_post, b_cov_post = ohm.fixed_effects()
    timing["oh_gp"] = time.perf_counter() - t2
    fin = np.isfinite(z_oh)
    res_sm = z_oh[fin] - sm_sky[fin]
    diag = dict(name=night.name, mode=cfg.mode, n_sky=sky.n_frames,
                oh_global_offset_xcorr=d_glob, oh_drift_xcorr_ptp=float(np.ptp(drift)),
                oh_xcorr_bad_frames=int(np.sum(~xc_good)),
                oh_n_candidates=oh["n_candidates"], oh_n_adopted=oh["n_adopted"],
                oh_n_lines=oh["n_kept"], oh_lines_per_frame_median=float(np.median(tw["n_lines"])),
                oh_zp_abs_median=float(np.nanmedian(z_oh)), oh_zp_ptp=float(np.nanmax(z_oh) - np.nanmin(z_oh)),
                oh_zp_err_median=float(np.nanmedian(e_oh)), oh_s_sys=tw["s_sys"],
                oh_line_scatter_median=float(np.nanmedian(tw["line_scatter"])),
                oh_resid_about_smooth_rms=float(np.sqrt(np.mean(res_sm ** 2))) if res_sm.size else np.nan,
                oh_resid_about_smooth_mad=float(1.4826 * np.median(np.abs(res_sm))) if res_sm.size else np.nan,
                oh_gp_amp=ohm.amp, oh_gp_ell_h=ohm.ell, oh_gp_jitter=ohm.jit,
                oh_nod_A_minus_B=float(b_post[1]) if has_nod else np.nan,
                oh_nod_err=float(np.sqrt(b_cov_post[1, 1])) if has_nod else np.nan,
                oh_n_downweighted=int(np.sum(ohm.weights < 0.5)),
                sky_shape=shape)
    oh_modes = None
    if sky_spec is not None and tw["C"].size:
        names_r = (("nod",) if has_nod else ()) + ("drift",)
        oh_modes = dict(sky_spec, C=tw["C"], C_cov=tw["C_cov"], phi_mean=tw["phi_mean"],
                        reg_mean=tw["reg_mean"], R_sky=R_sky, names=names_r,
                        drift_regressor="zp" if zref_used else "xcorr")
        diag["oh_modes_C"] = tw["C"].tolist()
        diag["oh_modes_q"] = sky_q
        diag["oh_modes_ptp_kms"] = _modes_ptp(tmpl, sky.lam, oh_modes, R_sky)
    if sky_guard is not None:
        diag["oh_modes_guard"] = sky_guard
    oh_out = dict(z=z_oh, z_err=e_oh, n_lines=tw["n_lines"], mu=mu, mu_err=tw["mu_err"],
                  line_scatter=tw["line_scatter"], s_sys=tw["s_sys"], x=xs, order=ms_line,
                  lref=np.array([f["lref"] for f in oh["meta"]]),
                  band=np.array([f["band"] for f in oh["meta"]]), drift_xcorr=drift,
                  d_glob=d_glob, smooth=sm_sky, d=oh["d"], e=oh["e"], valid=oh["valid"],
                  sigma=oh["sigma"])
    # ---------------------------------------------------------------- (4) tellurics
    obj = night.obj
    z_star = z_star_err = z_inst = z_inst_err = None
    tell_out = None
    stm = None
    star_modes = None
    if obj is not None:
        nod_o = _nod_code(obj.nod)
        Ho = np.column_stack([np.ones(obj.n_frames), nod_o]) if has_nod else np.ones((obj.n_frames, 1))
        z_inst, z_inst_err = ohm.predict(obj.time, Ho, obj.epoch)
        H0 = np.column_stack([np.ones(obj.n_frames), np.zeros(obj.n_frames)]) if has_nod else \
            np.ones((obj.n_frames, 1))
        X, X_sd = ohm.predict(obj.time, H0, obj.epoch)            # nod-free instrument zp
        t3 = time.perf_counter()
        tfeats = {}
        for m in obj.orders:
            if m in cfg.exclude_tell_orders:
                continue
            lam = np.asarray(obj.lam[m])
            lo = np.nanmin(lam) * (1 + (abs(d_glob) + cfg.xcorr_vmax + 2 * fw) / C_KMS)
            hi = np.nanmax(lam) * (1 - (abs(d_glob) + cfg.xcorr_vmax + 2 * fw) / C_KMS)
            tf = build_absorber_features(lines, lo, hi, cfg.R, species=cfg.species,
                                         p_eff=dict(cfg.p_eff), depth=cfg.tell_depth,
                                         blend_max=cfg.tell_blend, halfwin=cfg.tell_halfwin,
                                         risk_max=cfg.tell_risk)
            tfeats[m] = [f for f in tf if f["clean"]]
        has_nod_o = bool(len(set(obj.nod[nod_o != 0])) > 1)
        star_spec = None
        R_star = None
        star_q = cfg.modes_for("star", len(obj.orders))
        if star_q is not None:
            star_spec = dict(qx=star_q[0], qm=star_q[1], xr=xr_all,
                             mr=(float(min(obj.orders)), float(max(obj.orders))))
            R_star = np.column_stack([X] + ([nod_o] if has_nod_o else []))
        if oh_modes is not None:   # OH regressors at the object exposures (same definition)
            if oh_modes["drift_regressor"] == "zp":
                dr_o = z_inst - (shape["c00"] if shape is not None else 0.0) - zref
            else:
                o = np.argsort(sky.time)
                dr_o = np.interp(obj.time, sky.time[o], drift[o])
            oh_modes["R_obj"] = np.column_stack(([nod_o] if has_nod else []) + [dr_o])
        te = _measure(obj, tmpl, tfeats, X, cfg, sigma0=sig0, dlimA=max(0.6 * fw, 3.0),
                      dlimB=max(0.35 * fw, 1.5), snr_comb=cfg.snr_comb_tell, sign=-1.0,
                      mode_spec=star_spec, regressors=R_star)
        timing["tell_fits"] = time.perf_counter() - t3
        if te is not None and te["mode_guard"] is not None:
            diag["star_modes_guard"] = te["mode_guard"]
            if star_spec is not None and te["mode_spec"] is not star_spec:   # conditioning guard
                star_spec = te["mode_spec"]
                star_q = None if star_spec is None else (star_spec["qx"], star_spec["qm"])
                warnings.warn(f"{night.name}: star detector modes reduced to {star_q} by the "
                              f"conditioning guard ({te['n_kept']} telluric lines)", stacklevel=2)
        if te is None or np.sum(np.isfinite(te["tw"]["z"])) < 3:
            warnings.warn(f"{night.name}: too few telluric lines; star term = OH zero point",
                          stacklevel=2)
            z_star, z_star_err = z_inst.copy(), z_inst_err.copy()
            diag["star_source"] = "oh"
        else:
            twt = te["tw"]
            zt, et = twt["z"], twt["z_err"]
            t4 = time.perf_counter()
            cols = [np.ones(obj.n_frames), X] + ([nod_o] if has_nod_o else [])
            Hst = np.column_stack(cols)
            nm = ("alpha", "beta") + (("nod",) if has_nod_o else ())
            b0m, b0s = cfg.star_beta_prior
            bm = np.r_[np.nanmedian(zt - b0m * X), b0m, np.zeros(len(nm) - 2)]
            bc = np.diag([20.0 ** 2, b0s ** 2] + [2.0 ** 2] * (len(nm) - 2))
            stm = fit_time_model(obj.time, zt, et ** 2 + (b0m * X_sd) ** 2, Hst, epochs=obj.epoch,
                                 names=nm, b_mean=bm, b_cov=bc, length_bounds=cfg.gp_length_bounds,
                                 robust_iter=cfg.gp_robust_iter, t_ref=ohm.t_ref,
                                 amp_scale=cfg.star_gp_amp_scale)
            z_star, z_star_err = stm.predict(obj.time, Hst, obj.epoch)
            bs, bsc = stm.fixed_effects()
            tell_src = twt
            timing["star_gp"] = time.perf_counter() - t4
            # OLS diagnostic (verify-style): raw telluric zp on OH smooth (nod-free) + nod
            sm_x, _ = ohm.predict(obj.time, H0, obj.epoch, jitter=False)
            fz = np.isfinite(zt)
            Bm = np.column_stack([np.ones(fz.sum()), sm_x[fz]] + ([nod_o[fz]] if has_nod_o else []))
            co, *_ = np.linalg.lstsq(Bm / et[fz, None], zt[fz] / et[fz], rcond=None)
            rr = zt[fz] - Bm @ co
            cov = np.linalg.inv((Bm / et[fz, None]).T @ (Bm / et[fz, None])) * (
                np.sum((rr / et[fz]) ** 2) / max(fz.sum() - Bm.shape[1], 1))
            diag.update(star_source="tellurics", tell_n_candidates=te["n_candidates"],
                        tell_n_adopted=te["n_adopted"], tell_n_lines=te["n_kept"],
                        tell_lines_per_frame_median=float(np.median(twt["n_lines"])),
                        tell_zp_err_median=float(np.nanmedian(et)), tell_s_sys=twt["s_sys"],
                        tell_line_scatter_median=float(np.nanmedian(twt["line_scatter"])),
                        star_beta=float(bs[1]), star_beta_err=float(np.sqrt(bsc[1, 1])),
                        star_alpha=float(bs[0]),
                        star_nod_A_minus_B=float(bs[2]) if has_nod_o else np.nan,
                        star_nod_err=float(np.sqrt(bsc[2, 2])) if has_nod_o else np.nan,
                        star_gp_amp=stm.amp, star_gp_ell_h=stm.ell, star_gp_jitter=stm.jit,
                        star_oh_slope_ols=float(co[1]), star_oh_slope_ols_err=float(np.sqrt(cov[1, 1])),
                        star_minus_oh_drift_ptp=float(abs(1 - co[1]) * np.ptp(sm_x)),
                        star_resid_ols_rms=float(np.std(rr)),
                        star_zp_median=float(np.nanmedian(zt)),
                        star_minus_inst_median=float(np.nanmedian(zt - z_inst)),
                        star_z_err_median=float(np.median(z_star_err)))
            twm = twt
            if star_spec is not None and twm["C"].size:
                Rm = R_star
                star_modes = dict(star_spec, C=twm["C"], C_cov=twm["C_cov"],
                                  phi_mean=twm["phi_mean"], reg_mean=twm["reg_mean"],
                                  R_obj=Rm, names=("X",) + (("nod",) if has_nod_o else ()))
                diag["star_modes_C"] = twm["C"].tolist()
                diag["star_modes_q"] = star_q
                diag["star_modes_ptp_kms"] = _modes_ptp(tmpl, obj.lam, star_modes, Rm)
            tell_out = dict(z=tell_src["z"], z_err=tell_src["z_err"], n_lines=tell_src["n_lines"],
                            mu=tell_src["mu"], mu_err=tell_src["mu_err"], sigma=te["sigma"],
                            line_scatter=tell_src["line_scatter"], s_sys=tell_src["s_sys"],
                            order=np.array([f["order"] for f in te["meta"]]),
                            lref=np.array([f["lref"] for f in te["meta"]]),
                            species=np.array([f["species"] for f in te["meta"]]),
                            d=te["d"], e=te["e"], valid=te["valid"], X=X, X_sd=X_sd)
    timing["total"] = time.perf_counter() - t_start
    diag["timing_s"] = timing
    if verbose:
        print({k: v for k, v in diag.items() if not isinstance(v, dict)})
    return AnchorSolution(
        name=night.name, mode=cfg.mode, config=cfg, template=tmpl,
        obj_time=None if obj is None else obj.time, obj_nod=None if obj is None else obj.nod,
        obj_shift_px=None if obj is None else obj.shift_px, z_star=z_star, z_star_err=z_star_err,
        z_inst=z_inst, z_inst_err=z_inst_err, sky_time=sky.time, sky_nod=sky.nod, z_sky=z_sky,
        z_sky_err=z_sky_err, oh=oh_out, tell=tell_out, shape=shape, oh_model=ohm,
        star_model=stm, diagnostics=diag, oh_modes=oh_modes, star_modes=star_modes)


SURFACE_FILENAME = "decanter_wavelength_surface.json"
SURFACE_VERSION = 2               # 2: part["scale"] (the convention of the line offsets)


def _template_id(template) -> str:
    """12-hex fingerprint (sha1) of a dispersion template's coefficients: a static surface is
    only valid on the template (calibration set) it was measured against."""
    import hashlib

    h = hashlib.sha1()

    def put(a):
        h.update((np.round(np.asarray(a, np.float64), 9) + 0.0).tobytes())

    if isinstance(template, ThArTemplate):
        h.update(np.asarray(template.orders, np.int64).tobytes())
        h.update(np.asarray(template.kind, np.int64).tobytes())
        for a in (template.coef, template.pmin, template.pmax):
            put(a)
    elif isinstance(template, EcSolution):
        h.update(repr((template.kind, template.nx, template.nm)).encode())
        put([template.xmin, template.xmax, template.omin, template.omax, template.shift])
        put(template.coef)
    else:
        h.update(repr(template).encode())
    return h.hexdigest()[:12]


def epoch_groups(time, gap_h: float = 3.0) -> np.ndarray:
    """(F,) int epoch-group labels (0, 1, ... in time order): a gap longer than ``gap_h``
    hours between consecutive exposures starts a new group (a night, or one visit)."""
    t = np.asarray(time, np.float64)
    if t.size == 0:
        return np.zeros(0, int)
    o = np.argsort(t, kind="stable")
    br = np.r_[0, np.cumsum(np.diff(t[o]) * 24.0 > float(gap_h))]
    g = np.empty(t.size, int)
    g[o] = br
    return g


def _onehot(groups, G=None) -> np.ndarray:
    g = np.asarray(groups, int)
    G = int(g.max()) + 1 if G is None else int(G)
    return (g[:, None] == np.arange(G)[None, :]).astype(np.float64)


@_quiet
def _surface_at(part: dict, x, m) -> np.ndarray:
    """Full (incl. the constant) Chebyshev surface ``S(x, m)`` [km/s] of a surface part /
    shape dict (``coef`` (qx, qm), ``xr``, ``mr``) at template pixels ``x`` in orders ``m``."""
    c = np.asarray(part["coef"], np.float64)
    xr, mr = part["xr"], part["mr"]
    x = np.asarray(x, np.float64)
    m = np.broadcast_to(np.asarray(m, np.float64), x.shape)
    xn = (2 * x - (xr[0] + xr[1])) / (xr[1] - xr[0])
    mn = (2 * m - (mr[0] + mr[1])) / max(mr[1] - mr[0], 1e-9)
    return _cheb2(xn, mn, *c.shape) @ c.ravel()


def _surface_kms(part: dict, template, m: int, lam) -> np.ndarray:
    """``S(x, m)`` [km/s] (constant included) at wavelength labels ``lam`` of order ``m``."""
    lam = np.asarray(lam, np.float64)
    return _surface_at(part, template.p_of_lam(int(m), lam), int(m))


def _line_pixels(template, order, lam) -> np.ndarray:
    order = np.asarray(order, int)
    lam = np.asarray(lam, np.float64)
    x = np.full(lam.size, np.nan)
    for m in np.unique(order):
        s = order == m
        x[s] = template.p_of_lam(int(m), lam[s])
    return x


def _part_prior(part: dict, template, order, lref, *, tol: float = 2e-3):
    """Prior static offsets of measured lines from a surface part: ``(mu, var, matched)``.
    A line in the stored table (same order, |d lref| < ``tol`` A) gets its stored offset and
    ``tau^2 + mu_err^2``; any other line the smooth surface value and ``tau^2 + s_line^2``."""
    order = np.asarray(order, int)
    lref = np.asarray(lref, np.float64)
    K = lref.size
    L = part["lines"]
    lo = np.asarray(L["order"], int)
    ll = np.asarray(L["lref"], np.float64)
    M = np.asarray(L["mu"], np.float64)
    Me = np.asarray(L["mu_err"], np.float64)
    idx = np.full(K, -1)
    for m in np.unique(order):
        s = np.where(order == m)[0]
        t = np.where(lo == m)[0]
        if t.size == 0:
            continue
        o = t[np.argsort(ll[t])]
        j = np.clip(np.searchsorted(ll[o], lref[s]), 1, max(o.size - 1, 1))
        jl = np.clip(j - 1, 0, o.size - 1)
        jr = np.clip(j, 0, o.size - 1)
        pick = np.where(np.abs(ll[o][jl] - lref[s]) <= np.abs(ll[o][jr] - lref[s]), o[jl], o[jr])
        ok = np.abs(ll[pick] - lref[s]) < tol
        idx[s[ok]] = pick[ok]
    matched = (idx >= 0) & np.isfinite(M[np.clip(idx, 0, None)]) if M.size else np.zeros(K, bool)
    tau2 = float(part["tau"]) ** 2
    x = _line_pixels(template, order, lref)
    mu = _surface_at(part, x, order)
    var = np.full(K, tau2 + float(part["s_line"]) ** 2)
    if matched.any():
        ii = idx[matched]
        mu[matched] = M[ii]
        var[matched] = tau2 + np.where(np.isfinite(Me[ii]), Me[ii], 0.0) ** 2
    return mu, var, matched


def _basis_x(x, m, spec) -> np.ndarray:
    """(K, J) detector modes ``T_a(x~) T_b(m~)``, (a, b) != (0, 0), at template pixels ``x``
    (the :func:`_mode_basis` terms, without the inverse dispersion)."""
    xr, mr = spec["xr"], spec["mr"]
    x = np.asarray(x, np.float64)
    m = np.broadcast_to(np.asarray(m, np.float64), x.shape)
    xn = (2 * x - (xr[0] + xr[1])) / (xr[1] - xr[0])
    mn = (2 * m - (mr[0] + mr[1])) / max(mr[1] - mr[0], 1e-9)
    return _cheb2(xn, mn, int(spec["q"][0]), int(spec["q"][1]))[:, 1:]


def _part_pattern(part: dict, template, order, lref, *, drift: bool = True):
    """Per-line nod and drift pattern values of a surface part: ``(pn_k, pd_k, X0, B, sd_n,
    sd_d)`` [km/s per unit nod code / per km/s of zero point; ``B`` (K, J) the basis at the
    lines, ``sd_*`` (J,) the coefficient sds], or None without stored modes."""
    md = part.get("modes") if part is not None else None
    if not md:
        return None
    x = _line_pixels(template, order, lref)
    B = _basis_x(x, order, md)
    J = B.shape[1]
    zero = np.zeros(J)
    cn = np.asarray(md["nod"], np.float64) if md.get("nod") is not None else zero
    cd = np.asarray(md["drift"], np.float64) if (drift and md.get("drift") is not None) else zero
    sn = np.asarray(md.get("nod_sd", zero) if md.get("nod") is not None else zero, np.float64)
    sd = np.asarray(md.get("drift_sd", zero) if (drift and md.get("drift") is not None)
                    else zero, np.float64)
    return B @ cn, B @ cd, float(md.get("X0", 0.0)), B, np.nan_to_num(sn), np.nan_to_num(sd)


def _json_ready(o):
    if isinstance(o, dict):
        return {str(k): _json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_ready(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_ready(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


_LINE_FLOAT_KEYS = ("lref", "mu", "mu_err", "scatter", "sigma", "x")


def _part_from_json(p: dict | None) -> dict | None:
    if p is None:
        return None
    p = dict(p)
    p["coef"] = np.asarray(p["coef"], np.float64)
    p["xr"] = tuple(p["xr"])
    p["mr"] = tuple(p["mr"])
    p["q"] = tuple(p["q"])
    L = dict(p["lines"])
    for k in list(L):
        if k in _LINE_FLOAT_KEYS:
            L[k] = np.array([np.nan if v is None else v for v in L[k]], np.float64)
        elif k in ("order", "n_nights"):
            L[k] = np.asarray(L[k], int)
        else:
            L[k] = np.asarray(L[k])
    p["lines"] = L
    for k in ("tau", "s_line"):
        p[k] = np.nan if p.get(k) is None else float(p[k])
    md = p.get("modes")
    if md:
        md = dict(md)
        md["q"] = tuple(md["q"])
        md["xr"] = tuple(md["xr"])
        md["mr"] = tuple(md["mr"])
        for k in ("nod", "drift", "nod_sd", "drift_sd", "nod_raw", "drift_raw", "nod_raw_sd",
                  "drift_raw_sd", "nod_between_sd", "drift_between_sd", "nod_shrink",
                  "drift_shrink"):
            if md.get(k) is not None:
                md[k] = np.array([np.nan if v is None else v for v in md[k]], np.float64)
        p["modes"] = md
    return p


@dataclass
class StaticSurface:
    """Static per-calibration-set wavelength correction measured on the atmosphere.

    For each light path (``sky``: OH airglow, uniformly filled slit; ``star``: telluric H2O in
    the object spectra) a part holds

    * ``coef`` (qx, qm): the smooth departure ``S(x, m)`` [km/s] of the true wavelength scale
      from the ThAr template (Chebyshev in template pixel ``x`` and order ``m``; label
      velocity at ``(x, m)`` = per-frame zero point ``z`` + ``S``);
    * ``lines``: the per-line static offsets ``mu`` [km/s] of every line seen in the build
      nights (on the same gauge: a frame's zero point is the weighted mean of ``d - mu``),
      ``mu_err``, the between-night ``scatter``, ``n_nights``, fitted ``sigma``, pixel ``x``;
    * ``tau``: night-to-night excess scatter of the line offsets [km/s] (the prior width
      when a new night is anchored on the table); ``s_line``: scatter of the line offsets
      about the smooth surface (prior width of a line not in the table);
    * ``gauge``: ``"c00"`` (sky: the surface constant is 0 and the zero point carries the
      level, as :func:`anchor_night`'s sky shape) or ``"mean"`` (star: information-weighted
      mean line offset 0, as :func:`anchor_night`'s telluric decomposition);
    * star only: ``slit`` = night-level star-minus-OH offset prior (``median``, ``scatter``,
      ``n``) used when a frame has too few telluric lines.

    The table is valid on ONE ThAr template (``template_id``): nights reduced with the same
    calibration set share it.  Built by :func:`build_static_surface`, stored as a small
    JSON file (:attr:`SURFACE_FILENAME`) next to / inside the calibration set.
    """

    mode: str
    template_id: str
    xr: tuple
    sky: dict | None = None
    star: dict | None = None
    calib: str = ""
    template_source: str = ""
    nights: tuple = ()
    meta: dict = field(default_factory=dict)
    version: int = SURFACE_VERSION

    def matches(self, template) -> bool:
        """True if ``template`` is the ThAr template the surface was measured against."""
        return self.template_id == _template_id(template)

    def part(self, kind: str) -> dict | None:
        return self.sky if kind in ("sky", "oh", "inst") else self.star

    def shape(self, kind: str) -> dict | None:
        """Shape dict for :class:`AnchorSolution` (``shape`` / ``star_shape``)."""
        p = self.part(kind)
        if p is None:
            return None
        return dict(coef=np.asarray(p["coef"], np.float64).copy(), xr=tuple(p["xr"]),
                    mr=tuple(p["mr"]), c00=float(np.asarray(p["coef"])[0, 0]),
                    gauge=p["gauge"], source=f"surface:{self.calib or self.template_id}")

    def surface_kms(self, kind: str, template, m: int, lam) -> np.ndarray:
        """``S(x, m)`` [km/s] of the ``"sky"`` / ``"star"`` part at labels ``lam`` of order m."""
        p = self.part(kind)
        if p is None:
            return np.zeros(np.shape(lam))
        return _surface_kms(p, template, m, lam)

    def prior(self, kind: str, template, order, lref):
        """``(mu, var, matched)`` prior line offsets of lines ``(order, lref)``."""
        p = self.part(kind)
        if p is None:
            K = np.size(lref)
            return np.zeros(K), np.zeros(K), np.zeros(K, bool)
        return _part_prior(p, template, order, lref)

    # ------------------------------------------------------------------ persistence
    @staticmethod
    def default_path(where) -> Path:
        """``where`` = a :class:`decanter.Calibration` (the file goes next to its comp, i.e.
        inside the calibration set), a calibration-set / WARP-reduction directory, or a
        file path."""
        if hasattr(where, "comp"):
            return Path(where.comp).expanduser().parent / SURFACE_FILENAME
        p = Path(where).expanduser()
        if p.is_dir():
            if (p / "calibration_data").is_dir() and not list(p.glob("comp_*.fits")):
                return p / "calibration_data" / SURFACE_FILENAME
            return p / SURFACE_FILENAME
        return p

    def save(self, where) -> Path:
        """Write the surface as JSON (``where``: see :meth:`default_path`); returns the path."""
        import json

        path = self.default_path(where)
        d = dict(version=self.version, mode=self.mode, template_id=self.template_id,
                 template_source=self.template_source, calib=self.calib, xr=list(self.xr),
                 nights=list(self.nights), meta=self.meta, sky=self.sky, star=self.star)
        path.write_text(json.dumps(_json_ready(d), indent=1, allow_nan=False))
        return path

    @classmethod
    def load(cls, where) -> StaticSurface:
        import json

        path = cls.default_path(where)
        d = json.loads(Path(path).read_text())
        if int(d.get("version", 0)) > SURFACE_VERSION:
            raise ValueError(f"{path}: surface version {d['version']} > {SURFACE_VERSION}")
        return cls(mode=d["mode"], template_id=d["template_id"], xr=tuple(d["xr"]),
                   sky=_part_from_json(d.get("sky")), star=_part_from_json(d.get("star")),
                   calib=d.get("calib", ""), template_source=d.get("template_source", ""),
                   nights=tuple(d.get("nights", ())), meta=d.get("meta", {}),
                   version=int(d.get("version", SURFACE_VERSION)))

    @classmethod
    def find(cls, where, template=None) -> StaticSurface | None:
        """Load the surface stored for ``where`` (see :meth:`default_path`) if there is one
        (and, given ``template``, if it matches); None otherwise."""
        try:
            path = cls.default_path(where)
        except Exception:
            return None
        if not Path(path).is_file():
            return None
        s = cls.load(path)
        if template is not None and not s.matches(template):
            warnings.warn(f"{path}: surface measured on template {s.template_id}, not "
                          f"{_template_id(template)}; ignored", stacklevel=2)
            return None
        return s

    def summary(self) -> dict:
        out = dict(calib=self.calib, mode=self.mode, template_id=self.template_id,
                   nights=list(self.nights))
        for k in ("sky", "star"):
            p = self.part(k)
            if p is not None:
                out[k] = dict(n_lines=int(np.size(p["lines"]["mu"])), tau=p["tau"],
                              s_line=p["s_line"], gauge=p["gauge"], q=p["q"],
                              scale=(p.get("scale") or {}).get("kind", "set"),
                              scale_id=(p.get("scale") or {}).get("id"))
        return out


def _night_line_table(sol, kind: str):
    """Per-line static offsets of one solution (``AnchorSolution.oh`` / ``.tell``)."""
    t = sol.oh if kind == "sky" else sol.tell
    if t is None or np.size(t.get("order", [])) == 0:
        return None
    mu = np.asarray(t["mu"], np.float64)
    me = np.asarray(t["mu_err"], np.float64) if "mu_err" in t else np.full(mu.size, np.nan)
    if not np.isfinite(me).any():          # older solutions: per-line scatter / sqrt(n)
        n = np.maximum(np.sum(np.asarray(t["valid"]), axis=1), 1)
        me = np.asarray(t["line_scatter"], np.float64) / np.sqrt(n)
    sg = np.asarray(t["sigma"], np.float64) if "sigma" in t else np.full(mu.size, np.nan)
    lab = np.asarray(t["band"] if kind == "sky" else t["species"]) if (
        ("band" if kind == "sky" else "species") in t) else np.full(mu.size, "")
    z = np.asarray(t["z"], np.float64)
    zmed = float(np.nanmedian(z)) if np.isfinite(z).any() else 0.0
    order = np.asarray(t["order"], int)
    lref = np.asarray(t["lref"], np.float64)
    x = _line_pixels(sol.template, order, lref * (1.0 + (np.nan_to_num(mu) + zmed) / C_KMS))
    return dict(order=order, lref=lref, mu=mu, mu_err=me, sigma=sg, label=lab, x=x)


def _night_modes(sol, kind: str):
    """The nod and drift columns of a series solution's detector modes (``star_modes`` with
    regressors ``(X, nod)``, ``oh_modes`` with ``(nod, drift)``): coefficient vectors (J,) of
    the non-constant ``(x, m)`` Chebyshev terms, their variances, and the night's regressor
    means (``nbar``; ``Xbar`` = mean absolute zero point)."""
    md = sol.star_modes if kind == "star" else sol.oh_modes
    if md is None or getattr(sol, "method", "series") != "series" or "C" not in md:
        return None
    names = tuple(md.get("names", ()))
    C = np.asarray(md["C"], np.float64)
    if C.ndim != 2 or C.size == 0:
        return None
    J, P = C.shape
    cov = np.asarray(md.get("C_cov", np.zeros((0, 0))), np.float64)
    var = np.diag(cov).reshape(J, P) if cov.shape == (J * P, J * P) else np.full((J, P), np.nan)
    rm = np.asarray(md["reg_mean"], np.float64)
    out = dict(q=(int(md["qx"]), int(md["qm"])), xr=tuple(md["xr"]), mr=tuple(md["mr"]),
               nod=None, nod_var=None, nbar=0.0, drift=None, drift_var=None, Xbar=np.nan)
    if "nod" in names:
        j = names.index("nod")
        out.update(nod=C[:, j], nod_var=var[:, j], nbar=float(rm[j]))
    xname = "X" if kind == "star" else "drift"
    if xname in names:
        j = names.index(xname)
        if kind == "star":
            xbar = float(rm[j])                         # X = absolute OH zero point
        else:                                           # drift = zp - zref: absolute mean
            z = np.asarray(sol.oh["z"], np.float64)
            xbar = float(np.nanmean(z)) if np.isfinite(z).any() else np.nan
        out.update(drift=C[:, j], drift_var=var[:, j], Xbar=xbar)
    return out


def _combine_coefs(vals, vars_):
    """Inverse-variance mean of per-night coefficient vectors with a DerSimonian-Laird
    between-night variance; returns ``(mean, sd_of_mean, between_sd)``."""
    V = np.asarray(vals, np.float64)
    S2 = np.asarray(vars_, np.float64)
    S2 = np.where(np.isfinite(S2) & (S2 > 0), S2, np.nanmedian(S2) if np.isfinite(S2).any() else 1.0)
    if V.shape[0] == 1:
        return V[0], np.sqrt(S2[0]), np.zeros(V.shape[1])
    w = 1.0 / S2
    mu = np.sum(w * V, axis=0) / np.sum(w, axis=0)
    Q = np.sum(w * (V - mu) ** 2, axis=0)
    k = V.shape[0]
    c = np.sum(w, axis=0) - np.sum(w ** 2, axis=0) / np.sum(w, axis=0)
    t2 = np.maximum((Q - (k - 1)) / np.maximum(c, 1e-300), 0.0)
    w2 = 1.0 / (S2 + t2)
    mu = np.sum(w2 * V, axis=0) / np.sum(w2, axis=0)
    return mu, 1.0 / np.sqrt(np.sum(w2, axis=0)), np.sqrt(t2)


def _shrink_coefs(c, sd):
    """Empirical-Bayes shrinkage of a pattern's coefficients towards 0: prior variance
    ``tau^2 = max(0, <c^2> - <sd^2>)`` over the J terms, posterior mean ``c tau^2/(tau^2+sd^2)``
    and sd.  A pattern measured on noise (no real nod / drift dependence) is switched off
    instead of adding its noise to every frame; returns ``(c, sd, factor)``."""
    c = np.asarray(c, np.float64)
    sd = np.asarray(sd, np.float64)
    t2 = max(float(np.mean(c ** 2) - np.mean(sd ** 2)), 0.0)
    fac = t2 / (t2 + sd ** 2) if t2 > 0 else np.zeros_like(c)
    return c * fac, np.sqrt(sd ** 2 * fac), fac


@_quiet
def _build_part(sols, kind, template, q, huber, tau_default, min_lines=20, max_rms_fwhm=0.12):
    tabs = [_night_line_table(s, kind) for s in sols]
    fw = C_KMS / float(sols[0].config.R)
    excluded = {}
    for i, t in enumerate(tabs):        # quality gate: too few lines, or offsets that are
        if t is None:                   # not static-like (a contaminated sky product: AU Mic
            continue                    # 60 s frames, 37 OH lines, offsets rms 0.17 FWHM)
        rms = float(np.sqrt(np.nanmean(t["mu"] ** 2))) if t["mu"].size else np.inf
        if t["mu"].size < min_lines or rms > max_rms_fwhm * fw:
            excluded[sols[i].name or f"night{i}"] = dict(n_lines=int(t["mu"].size),
                                                         mu_rms_kms=rms)
            tabs[i] = None
    if excluded:
        warnings.warn(f"static surface ({kind}): night(s) excluded by the quality gate "
                      f"(< {min_lines} lines or line-offset rms > {max_rms_fwhm} FWHM): "
                      f"{excluded}", stacklevel=3)
    use = [i for i, t in enumerate(tabs) if t is not None and t["mu"].size]
    if not use:
        return None
    # ---- nod / drift patterns: combine the nights' mode columns, then express every night's
    # line offsets for the UNCENTRED model  d = z + mu' + nod_f Pn(x, m) + (X_f - X0) Pd(x, m)
    mods = {i: _night_modes(sols[i], kind) for i in use}
    qs = [m_["q"] for m_ in mods.values() if m_ is not None]
    modes = None
    if qs:
        qsel = max(set(qs), key=qs.count)
        good = {i: m_ for i, m_ in mods.items() if m_ is not None and m_["q"] == qsel}
        spec = next(iter(good.values()))
        modes = dict(q=qsel, xr=spec["xr"], mr=spec["mr"], nod=None, drift=None, X0=0.0,
                     n_nights=len(good))
        nn_ = [m_ for m_ in good.values() if m_["nod"] is not None]
        if nn_:
            c, sd, bsd = _combine_coefs([m_["nod"] for m_ in nn_], [m_["nod_var"] for m_ in nn_])
            cs, ss, fac = _shrink_coefs(c, sd)
            modes.update(nod=cs, nod_sd=ss, nod_raw=c, nod_raw_sd=sd, nod_between_sd=bsd,
                         nod_shrink=fac)
        dd_ = [m_ for m_ in good.values() if m_["drift"] is not None and np.isfinite(m_["Xbar"])]
        if dd_:
            c, sd, bsd = _combine_coefs([m_["drift"] for m_ in dd_],
                                        [m_["drift_var"] for m_ in dd_])
            cs, ss, fac = _shrink_coefs(c, sd)
            modes.update(drift=cs, drift_sd=ss, drift_raw=c, drift_raw_sd=sd,
                         drift_between_sd=bsd, drift_shrink=fac,
                         X0=float(np.median([m_["Xbar"] for m_ in dd_])))
        for i, m_ in good.items():
            t = tabs[i]
            B = _basis_x(t["x"], t["order"], modes)
            corr = np.zeros(t["mu"].size)
            if m_["nod"] is not None:
                corr += m_["nbar"] * (B @ m_["nod"])
            if m_["drift"] is not None and np.isfinite(m_["Xbar"]):
                corr -= (modes["X0"] - m_["Xbar"]) * (B @ m_["drift"])
            t["mu"] = t["mu"] - corr
    index, keys = {}, []
    for i in use:
        t = tabs[i]
        for o, lr in zip(t["order"], t["lref"], strict=True):
            k = (int(o), round(float(lr), 4))
            if k not in index:
                index[k] = len(keys)
                keys.append(k)
    K, N = len(keys), len(use)
    D = np.full((K, N), np.nan)
    E = np.full((K, N), np.nan)
    X = np.full((K, N), np.nan)
    S = np.full((K, N), np.nan)
    lref = np.zeros(K)
    order = np.zeros(K, int)
    lab = np.full(K, "", dtype=object)
    for j, i in enumerate(use):
        t = tabs[i]
        rows = [index[(int(o), round(float(lr), 4))] for o, lr in zip(t["order"], t["lref"],
                                                                       strict=True)]
        D[rows, j] = t["mu"]
        E[rows, j] = np.sqrt(np.nan_to_num(t["mu_err"], nan=0.05) ** 2 + 1e-6)
        X[rows, j] = t["x"]
        S[rows, j] = t["sigma"]
        lref[rows] = t["lref"]
        order[rows] = t["order"]
        lab[rows] = t["label"]
    nn = np.isfinite(D).sum(axis=1)
    if N == 1:
        M = D[:, 0].copy()
        Me = E[:, 0].copy()
        w = 1.0 / Me ** 2
        M -= np.sum(w * M) / np.sum(w)            # gauge: information-weighted mean 0
        offsets = np.zeros(1)
        tau = float(tau_default)
        scatter = np.full(K, np.nan)
    else:
        rz = robust_zero_points(D, E, huber=huber)
        M, Me, offsets = rz["mu"], rz["mu_err"], rz["z"]
        tau = float(max(rz["s_sys"], 1e-4)) if np.isfinite(rz["s_sys"]) else float(tau_default)
        scatter = rz["line_scatter"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x = np.nanmean(X, axis=1)
        sig = np.nanmedian(S, axis=1)
    xr = tuple(float(v) for v in template.p_range(int(order[0])))
    mr = (float(min(template.orders)), float(max(template.orders)))
    qx, qm = int(q[0]), int(q[1])
    ok = np.isfinite(M) & np.isfinite(x)
    coef = np.zeros((qx, qm))
    s_line = np.nan
    if ok.sum() >= 3 * qx * qm:
        err = np.sqrt(Me ** 2 + tau ** 2 + 1e-6)
        coef = _fit_shape(x[ok], order[ok], M[ok], err[ok], qx, qm, xr, mr, huber=huber)
        r = M - _surface_at(dict(coef=coef, xr=xr, mr=mr), x, order)
        s_line = float(np.sqrt(max((1.4826 * np.nanmedian(np.abs(r[ok]))) ** 2
                                   - np.nanmedian(Me[ok] ** 2), 1e-6)))
        coef = _fit_shape(x[ok], order[ok], M[ok], np.sqrt(Me ** 2 + s_line ** 2 + 1e-6)[ok],
                          qx, qm, xr, mr, huber=huber)
    elif ok.any():
        coef[0, 0] = float(np.nanmedian(M[ok]))
    gauge = "mean"
    if kind == "sky":                               # anchor_night's sky convention
        c00 = float(coef[0, 0])
        M = M - c00
        offsets = offsets + c00
        coef = coef.copy()
        coef[0, 0] = 0.0
        gauge = "c00"
    r = M - _surface_at(dict(coef=coef, xr=xr, mr=mr), x, order)
    if ok.any():
        s_line = float(np.sqrt(max((1.4826 * np.nanmedian(np.abs(r[ok]))) ** 2
                                   - np.nanmedian(Me[ok] ** 2), 1e-6)))
    part = dict(q=(qx, qm), xr=xr, mr=mr, coef=coef, gauge=gauge, tau=tau, s_line=s_line,
                lines=dict(order=order, lref=lref, mu=M, mu_err=Me, scatter=scatter,
                           n_nights=nn, sigma=sig, x=x, label=np.asarray(lab, str)),
                night_offsets={sols[i].name or f"night{i}": float(offsets[j])
                               for j, i in enumerate(use)}, modes=modes, excluded=excluded,
                rms_about_surface=float(np.sqrt(np.nanmean(r[ok] ** 2))) if ok.any() else np.nan)
    if kind == "star":
        dsm = []
        for i in use:
            s = sols[i]
            v = s.diagnostics.get("star_minus_inst_median") if s.diagnostics else None
            if v is None and s.z_star is not None and s.z_inst is not None:
                v = float(np.nanmedian(s.z_star - s.z_inst))
            if v is not None and np.isfinite(v):
                dsm.append(float(v))
        if dsm:
            sc = (1.4826 * float(np.median(np.abs(np.asarray(dsm) - np.median(dsm))))
                  if len(dsm) > 2 else float("nan"))
            part["slit"] = dict(median=float(np.median(dsm)),
                                scatter=sc if np.isfinite(sc) and sc > 0 else None,
                                n=len(dsm), values=dsm)
    return part


def build_static_surface(solutions, *, calib: str = "", sky_shape: tuple = (2, 2),
                         star_shape: tuple = (2, 2), huber: float = 2.0,
                         tau_default: float = 0.02, min_lines: int = 20,
                         max_rms_fwhm: tuple = (0.12, 0.25)) -> StaticSurface:
    """Build the static per-calibration-set surface from one or more night solutions.

    No WARP equivalent (WARP has no absolute wavelength scale beyond the period ThAr fit).

    Each solution (:func:`anchor_night`, or :func:`anchor_frames`) contributes its per-line
    static offsets (``sol.oh["mu"]`` / ``sol.tell["mu"]``: the time-averaged departure of each
    line from the frame zero point).  Across nights the offsets are combined by a robust lines
    x nights decomposition (:func:`robust_zero_points`: per-line static offset, per-night gauge
    offset, ``tau`` = the excess night-to-night scatter); a low-order ``(x, m)`` surface is fitted
    to the combined offsets (``sky_shape`` / ``star_shape`` Chebyshev terms).  With a single
    night ``tau`` = ``tau_default``.  Every solution must share ONE ThAr template.

    Series solutions also contribute the nod and drift columns of their detector modes
    (:func:`anchor_night`'s ``(x, m) x (nod, drift)`` terms): combined over nights
    (inverse-variance + DerSimonian-Laird between-night variance), shrunk towards 0 by
    empirical Bayes (:func:`_shrink_coefs`: a pattern that is noise is switched off), stored as
    ``part["modes"]`` (``nod``, ``drift`` per km/s of zero point about ``X0``); each night's line
    offsets are first re-expressed for the uncentred model
    ``d = z + mu' + nod Pn(x, m) + (X - X0) Pd(x, m)``.

    Quality gate: a night's part is left out (warning; listed in ``part["excluded"]``) with
    fewer than ``min_lines`` lines or a line-offset rms above ``max_rms_fwhm`` (sky, star)
    FWHM - e.g. the sky part of a bright-star short-exposure night (AU Mic, 60 s: 37 OH lines,
    rms 0.86 km/s = 0.17 FWHM vs 0.025-0.075 on normal nights; pi Men, 30 s: 7 lines).  The
    telluric offsets are intrinsically larger (blends; 0.05-0.09 FWHM in HIRES, 0.14 in WIDE).

    Per-line offsets stay on this calibration set's own gauge (``part["scale"]``).
    """
    sols = [solutions] if isinstance(solutions, AnchorSolution) else list(solutions)
    if not sols:
        raise ValueError("build_static_surface: no solutions")
    tmpl = sols[0].template
    tid = _template_id(tmpl)
    for s in sols[1:]:
        if _template_id(s.template) != tid:
            raise ValueError(f"{s.name}: different ThAr template ({_template_id(s.template)} vs "
                             f"{tid}); a surface belongs to one calibration set")
    modes = {s.mode for s in sols}
    if len(modes) > 1:
        raise ValueError(f"mixed modes {modes}")
    mx = (max_rms_fwhm, max_rms_fwhm) if np.ndim(max_rms_fwhm) == 0 else tuple(max_rms_fwhm)
    sky = _build_part(sols, "sky", tmpl, sky_shape, huber, tau_default, min_lines, mx[0])
    star = _build_part(sols, "star", tmpl, star_shape, huber, tau_default, min_lines, mx[1])
    parts = {"sky": sky, "star": star}
    for k, part in parts.items():
        if part is None:
            continue
        part["scale"] = dict(kind="set", gauge=part["gauge"], note=(
            "per-line offsets on THIS calibration set's own gauge (information-weighted "
            "mean of its build nights' lines); its zero point is not comparable with "
            "another set's"))
    sky, star = parts["sky"], parts["star"]
    import datetime

    return StaticSurface(
        mode=sols[0].mode, template_id=tid,
        xr=tuple(float(v) for v in tmpl.p_range(int(tmpl.orders[0]))), sky=sky, star=star,
        calib=calib, template_source=str(getattr(tmpl, "source", "")),
        nights=tuple(s.name for s in sols),
        meta=dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                  n_frames=[int(np.size(s.z_sky)) for s in sols],
                  methods=[s.method for s in sols]))


def surface_from_warp_reductions(roots, *, calib: str = "", config: AnchorConfig | None = None,
                                 save_to=None, **kw) -> StaticSurface:
    """Convenience: :func:`load_warp_night` + :func:`anchor_night` on every WARP reduction
    directory in ``roots`` (nights of ONE calibration set), :func:`build_static_surface`
    (``**kw``), and optionally :meth:`StaticSurface.save` to ``save_to`` (a
    :class:`decanter.Calibration`, a calibration-set directory or a file path)."""
    sols = []
    for r in roots:
        night = load_warp_night(r)
        sols.append(anchor_night(night, config or AnchorConfig.for_mode(night.mode)))
    surf = build_static_surface(sols, calib=calib, **kw)
    if save_to is not None:
        surf.save(save_to)
    return surf


# -------------------------------------------------------------- anchored zero points
def _var_components(r2, e2, pv, *, lam_min: float = 0.7) -> tuple[float, float]:
    """Robust fit of ``Var r = lam^2 e^2 + s^2 + pv`` from squared residuals ``r2``: the
    points are binned by ``e^2`` (up to 4 bins of >= 30), each bin's variance is the median of
    ``r2`` / 0.4549 (chi^2_1 median) minus its median ``pv``, and ``(lam^2, s^2) >= 0`` are fitted
    by relative non-negative least squares (``lam`` alone with one bin; ``lam >= lam_min``)."""
    r2 = np.asarray(r2, np.float64)
    e2 = np.asarray(e2, np.float64)
    pv = np.asarray(pv, np.float64)
    n = r2.size
    if n == 0:
        return 1.0, 0.0
    o = np.argsort(e2, kind="stable")
    B = int(np.clip(n // 30, 1, 4))
    xs, vs = [], []
    for idx in np.array_split(o, B):
        xs.append(float(np.median(e2[idx])))
        vs.append(float(np.median(r2[idx])) / 0.454936 - float(np.median(pv[idx])))
    xs, vs = np.asarray(xs), np.asarray(vs)
    if B >= 2:
        from scipy.optimize import nnls

        wr = 1.0 / np.maximum(np.abs(vs), 0.1 * max(float(np.median(np.abs(vs))), 1e-12))
        (lam2, s2), _ = nnls(np.column_stack([xs, np.ones(B)]) * wr[:, None], vs * wr)
    else:
        lam2, s2 = max(vs[0], 0.0) / max(xs[0], 1e-300), 0.0
    if lam2 < lam_min ** 2:
        lam2 = lam_min ** 2
        s2 = max(float(np.median(vs - lam2 * xs)), 0.0) if B >= 2 else 0.0
    return float(lam2), float(s2)


@_quiet
def _group_design(H: np.ndarray) -> np.ndarray:
    """Columns of a group's (F_g, P) zero-point design kept: the intercept (column 0) and every
    other column that varies inside the group and raises the rank."""
    keep = [0]
    for j in range(1, H.shape[1]):
        if np.ptp(H[:, j]) <= 0 or len(keep) >= H.shape[0]:
            continue
        if np.linalg.matrix_rank(H[:, [*keep, j]]) > len(keep):
            keep.append(j)
    return H[:, keep]


def anchored_zero_points(d, e, *, groups=None, valid=None, prior_mu=None, prior_var=None,
                         self_prior: bool = False, huber: float = 2.0, n_iter: int = 60,
                         design=None) -> dict:
    """Per-frame zero points with per-line static offsets tied to a prior, per epoch group.

    No WARP equivalent.  Model (lines k, frames f in epoch group g = g(f)):

        d_kf = z_f + mu_kg + eps_kf,   Var eps = lam_g^2 e_kf^2 + s_g^2,
        mu_kg ~ N(m_k, v_k + t^2)

    ``m_k`` / ``v_k`` = ``prior_mu`` / ``prior_var`` (a :class:`StaticSurface` table; default
    0 / 0), or with ``self_prior`` the information-weighted mean of the groups' own static
    offsets (gauge: their information-weighted mean is 0; the groups then share one scale
    without a stored table).  ``z`` and ``mu`` are solved exactly (GLS, ``mu`` eliminated)
    per group with Huber weights on the prior-predictive residuals ``d - z - m``.  The
    measurement variance is multiplicative + additive in the centroid error (per-line
    scatter on the WINERED nights is ~2.5-3 x ``e`` and correlates with it at 0.75-0.9, so a
    purely additive excess would weight a faint line like a bright one): ``(lam_g, s_g)``
    are fitted per group by binned robust variances (:func:`_var_components`) of the
    within-line deviations (lines with >= 2 frames in the group) or of the prior-predictive
    residuals (one frame: ``lam``/``s`` then absorb any excess line-offset scatter too), and
    ``t`` (shared) from the groups' line-mean residuals where identifiable.

    ``design`` (F, P), column 0 = 1: an epoch model instead of free per-frame zero points, i.e.
    ``z_f = H_f . b_g`` inside each group (e.g. columns ``1, nod_f, t_f - <t>``: one zero point,
    an A/B term and a drift per epoch); per group only the columns that vary there are used
    (:func:`_group_design`), and ``b_g`` is solved with ``mu`` eliminated as for free ``z``
    (``design_coef`` / ``design_cov`` per group in the output).

    ``z_err`` is the GLS sd times the Huber sandwich inflation
    ``sqrt(max(1, [sum w psi^2 / (sum w psi')^2] / [1 / sum w hw]))`` per frame (honest under
    heavy-tailed line residuals; 1 for Gaussian ones).  One frame:
    ``z = sum w (d - m) / sum w``, ``w = hw / (lam^2 e^2 + s^2 + v)``.

    Returns dict: ``z, z_err`` (F,), ``n_lines`` (F,), ``mu`` (K, G) group static offsets
    (posterior), ``shat, shat_err`` (K, G) the unshrunk group static offsets ``<d - z>``, ``m``
    (K,) prior mean used, ``resid`` (K, F) prior-predictive residuals, ``w`` final weights,
    ``lam_group, s_group`` (G,), ``t_line``, ``scale`` (robust sd of the standardised
    residuals), ``inflation`` (F,), ``groups`` (F,) 0..G-1 labels.
    """
    d = np.asarray(d, np.float64)
    e = np.asarray(e, np.float64)
    K, F = d.shape
    g_in = np.zeros(F, int) if groups is None else np.asarray(groups)
    _gu, gi = np.unique(g_in, return_inverse=True)
    G = int(gi.max()) + 1 if F else 0
    V = np.isfinite(d) & np.isfinite(e) & (e > 0)
    if valid is not None:
        V &= np.asarray(valid, bool)
    dd = np.where(V, d, 0.0)
    e2 = np.where(V, e * e, 1.0)
    m = np.zeros(K) if prior_mu is None else np.where(np.isfinite(prior_mu), prior_mu, 0.0)
    m = np.asarray(m, np.float64).copy()
    tv = np.zeros(K) if prior_var is None else np.where(np.isfinite(prior_var), prior_var, 0.0)
    tv = np.asarray(tv, np.float64)
    OH = _onehot(gi, G)                                   # (F, G)
    nkg = V.astype(np.float64) @ OH                       # (K, G)
    multi = nkg >= 2
    fit_t = bool(multi.any())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        z = np.nanmedian(np.where(V, d - m[:, None], np.nan), axis=0)
    z = np.where(np.isfinite(z), z, 0.0)
    lam2 = np.ones(G)
    s2 = np.zeros(G)
    t2 = 0.0
    hw = V.astype(np.float64)
    mu = np.repeat(m[:, None], G, axis=1)
    covs = [None] * G
    Dz = None if design is None else np.asarray(design, np.float64).reshape(F, -1)
    bcoef, bcovs = [None] * G, [None] * G
    scale = 1.0
    it = 0
    for it in range(int(n_iter)):
        var = lam2[gi][None, :] * e2 + s2[gi][None, :]
        w = V / var * hw
        p = 1.0 / np.maximum(tv + t2, 1e-12)
        if self_prior and it > 0:
            Wkg = w @ OH
            Skg = (w * (dd - z[None, :])) @ OH
            shat = np.where(Wkg > 0, Skg / np.where(Wkg > 0, Wkg, 1.0), 0.0)
            Pkg = Wkg / (1.0 + t2 * Wkg)
            Ik = Pkg.sum(axis=1)
            m = np.where(Ik > 0, (Pkg * shat).sum(axis=1) / np.where(Ik > 0, Ik, 1.0), 0.0)
            if Ik.sum() > 0:
                m = m - np.sum(Ik * m) / np.sum(Ik)
        z_prev = z.copy()
        for g in range(G):
            fr = np.where(gi == g)[0]
            Wg = w[:, fr]
            Dg = dd[:, fr]
            a = Wg.sum(axis=1) + p
            b = (Wg * Dg).sum(axis=1) + p * m
            ok = Wg.sum(axis=0) > 0
            zg = np.full(fr.size, np.nan)
            cov = np.full((fr.size, fr.size), np.nan)
            if ok.any():
                Wo = Wg[:, ok]
                A = np.diag(Wo.sum(axis=0)) - Wo.T @ (Wo / a[:, None])
                rhs = (Wo * Dg[:, ok]).sum(axis=0) - Wo.T @ (b / a)
                if Dz is None:
                    A = A + (1e-12 * np.trace(A) / A.shape[0] + 1e-300) * np.eye(A.shape[0])
                    Ci = np.linalg.inv(A)
                    zg[ok] = Ci @ rhs
                else:                                   # epoch model z = H b
                    Hg = _group_design(Dz[fr][ok])
                    Ab = Hg.T @ A @ Hg
                    Ab = Ab + (1e-12 * np.trace(Ab) / Ab.shape[0] + 1e-300) * np.eye(Ab.shape[0])
                    Cb = np.linalg.inv(Ab)
                    bg = Cb @ (Hg.T @ rhs)
                    zg[ok] = Hg @ bg
                    Ci = Hg @ Cb @ Hg.T
                    bcoef[g], bcovs[g] = bg, Cb
                cov[np.ix_(ok, ok)] = Ci
                mu[:, g] = (b - Wo @ zg[ok]) / a
            else:
                mu[:, g] = m
            z[fr] = zg
            covs[g] = cov
        zf = np.where(np.isfinite(z), z, 0.0)
        q = d - zf[None, :] - m[:, None]
        pv = tv + t2
        tq = np.where(V, q / np.sqrt(var + pv[:, None]), np.nan)
        aq = np.abs(tq)
        scale = 1.4826 * float(np.nanmedian(aq)) + 1e-12 if V.any() else 1.0
        hw = np.where(V, np.where(aq <= huber * scale, 1.0, huber * scale / np.where(
            aq > 0, aq, 1.0)), 0.0)
        # ---- variance components
        qv = np.where(V, q, 0.0)
        dev = None
        t2n = 0.0
        if fit_t:
            Wh = V * hw / var
            WK = Wh @ OH
            rho = (Wh * qv) @ OH / np.where(WK > 0, WK, 1.0)          # (K, G) line means
            dev = qv - rho[:, gi]
            rr = rho[multi]
            vr = (1.0 / np.where(WK > 0, WK, np.inf))[multi] + np.broadcast_to(
                tv[:, None], multi.shape)[multi]
            if rr.size:
                t2n = max((1.4826 * float(np.median(np.abs(rr)))) ** 2 - float(np.median(vr)), 0.0)
        lam2n, s2n = lam2.copy(), s2.copy()
        for g in range(G):
            fr = gi == g
            Vg = V[:, fr]
            if not Vg.any():
                continue
            mg = np.broadcast_to(multi[:, g][:, None], Vg.shape)
            r2s, e2s, pvs = [], [], []
            one = Vg & ~mg
            if one.any():
                r2s.append(q[:, fr][one] ** 2)
                e2s.append(e2[:, fr][one])
                pvs.append(np.broadcast_to(pv[:, None], Vg.shape)[one])
            two = Vg & mg
            if two.any():
                n = np.broadcast_to(nkg[:, g][:, None], Vg.shape)[two]
                r2s.append(dev[:, fr][two] ** 2 * n / (n - 1.0))
                e2s.append(e2[:, fr][two])
                pvs.append(np.zeros(int(two.sum())))
            lam2n[g], s2n[g] = _var_components(np.concatenate(r2s), np.concatenate(e2s),
                                               np.concatenate(pvs))
        lam2_prev, s2_prev, t2_prev = lam2.copy(), s2.copy(), t2
        lam2 = 0.5 * lam2 + 0.5 * lam2n
        s2 = 0.5 * s2 + 0.5 * s2n
        t2 = 0.5 * t2 + 0.5 * t2n
        if it > 8:
            dz = float(np.nanmax(np.abs(z - z_prev))) if np.isfinite(z).any() else 0.0
            dsv = max(float(np.max(np.abs(np.sqrt(lam2) - np.sqrt(lam2_prev)))) if G else 0.0,
                      float(np.max(np.abs(np.sqrt(s2) - np.sqrt(s2_prev)))) if G else 0.0,
                      abs(math.sqrt(t2) - math.sqrt(t2_prev)))
            if dz < 1e-7 and dsv < 1e-6:
                break
    var = lam2[gi][None, :] * e2 + s2[gi][None, :]
    w = V / var * hw
    zf = np.where(np.isfinite(z), z, 0.0)
    Wkg = w @ OH
    Skg = (w * (dd - zf[None, :])) @ OH
    shat = np.where(Wkg > 0, Skg / np.where(Wkg > 0, Wkg, 1.0), np.nan)
    sc = max(scale, 1.0)
    shat_err = np.where(Wkg > 0, sc / np.sqrt(np.where(Wkg > 0, Wkg, 1.0)), np.nan)
    # Huber sandwich inflation per frame (prior-predictive residuals)
    q = d - zf[None, :] - m[:, None]
    wpp = V / (var + (tv + t2)[:, None])
    tq = np.where(V, q * np.sqrt(wpp), 0.0)
    c = huber * scale
    psi = np.clip(tq, -c, c)
    dpsi = (np.abs(tq) <= c) & V
    den = np.sum(wpp * dpsi, axis=0)
    sand = np.where(den > 0, np.sum(wpp * psi ** 2, axis=0) / np.where(den > 0, den, 1.0) ** 2,
                    np.inf)
    naive = 1.0 / np.maximum(np.sum(wpp * hw, axis=0), 1e-300)
    infl = np.sqrt(np.maximum(1.0, sand / naive))
    z_err = np.full(F, np.nan)
    for g in range(G):
        fr = np.where(gi == g)[0]
        z_err[fr] = np.sqrt(np.clip(np.diag(covs[g]), 0.0, None)) * infl[fr]
    z = np.where(np.isfinite(z_err), z, np.nan)
    return dict(z=z, z_err=z_err, n_lines=V.sum(axis=0), mu=mu, shat=shat, shat_err=shat_err,
                m=m, resid=np.where(V, q, np.nan), w=w, lam_group=np.sqrt(lam2),
                s_group=np.sqrt(s2), t_line=math.sqrt(t2), scale=float(scale), inflation=infl,
                groups=gi, n_iter=it + 1, design_coef=bcoef, design_cov=bcovs)


# --------------------------------------------------------------- per-group measurement
def _aonly_exclude(feats_by_order: dict, lines: LineTable, cfg, kind: str) -> dict:
    """A-only frames: drop the features whose fit window (+- margin) holds a contaminating line
    (``kind="oh"``: OH features vs telluric lines deeper than ``aonly_tell_depth``;
    ``kind="tell"``: telluric features vs OH lines brighter than ``aonly_oh_imin``).  Both kinds
    move with the same instrument zero point, so the comparison is in the line-list frame."""
    fw = cfg.fwhm
    out = {}
    for m, fs in feats_by_order.items():
        if not fs:
            out[m] = fs
            continue
        lo = min(f["lref"] for f in fs) - 10.0
        hi = max(f["lref"] for f in fs) + 10.0
        if kind == "oh":
            a = lines.absorber_lines(lo, hi, species=("H2O", "O2"), R=cfg.R,
                                     p_eff=dict(cfg.p_eff))
            c = a["lam"][a["depth"] >= cfg.aonly_tell_depth]
            mg = cfg.aonly_tell_halfwidth * fw
        else:
            o = lines.oh_lines(lo, hi)
            c = o["lam"][o["I"] >= cfg.aonly_oh_imin]
            mg = cfg.aonly_oh_halfwidth * fw
        keep = []
        for f in fs:
            u = (c / f["lref"] - 1.0) * C_KMS
            if not np.any((u > f["umin"] - mg) & (u < f["umax"] + mg)):
                keep.append(f)
        out[m] = keep
    return out


def _subset(sset: SpectrumSet, idx) -> SpectrumSet:
    idx = np.asarray(idx, int)
    return SpectrumSet(
        kind=sset.kind, lam=sset.lam, flux={m: np.asarray(v)[idx] for m, v in sset.flux.items()},
        time=sset.time[idx], nod=sset.nod[idx], epoch=sset.epoch[idx],
        ivar=None if sset.ivar is None else {m: np.asarray(v)[idx] for m, v in sset.ivar.items()},
        shift_px=None if sset.shift_px is None else sset.shift_px[idx],
        frame_ids=None if sset.frame_ids is None else tuple(sset.frame_ids[i] for i in idx),
        airmass=None if sset.airmass is None else sset.airmass[idx])


def _measure_group(sset, template, feats_by_order, d0_frame, fr, cfg, *, sigma0, dlimA,
                   dlimB, snr_comb, sign):
    """Pass A (width free) -> line adoption -> pass B (width fixed) on the frames ``fr`` of one
    epoch group, as :func:`_measure` but without any lines x frames decomposition.  The
    combined-S/N threshold ``snr_comb`` refers to ``cfg.snr_ref_frames`` frames and is scaled
    by ``sqrt(len(fr) / snr_ref_frames)`` (at most 1, at least ``cfg.snr_frame``)."""
    fr = np.asarray(fr, int)
    Fg = fr.size
    ords = [m for m in sorted(feats_by_order) if feats_by_order[m]]
    feats_by_order = {m: feats_by_order[m] for m in ords}
    if not ords or Fg == 0:
        return None
    nA = min(Fg, max(int(cfg.pass_a_frames), 1))
    ot = fr[np.argsort(sset.time[fr], kind="stable")]
    frA = np.sort(ot[np.unique(np.round(np.linspace(0, Fg - 1, nA)).astype(int))])
    d0 = {m: np.repeat(d0_frame[frA][None, :], len(feats_by_order[m]), 0) for m in ords}
    sig = {m: np.full(len(feats_by_order[m]), sigma0) for m in ords}
    dlA = {m: np.full(len(feats_by_order[m]), dlimA) for m in ords}
    A = _fit_features(sset, template, feats_by_order, d0, sigma=sig, dlim=dlA, fit_sigma=True,
                      cfg=cfg, frames=frA)
    if len(A["meta"]) == 0:
        return None
    vA = (np.isfinite(A["d"]) & (A["snr"] >= cfg.snr_frame) & (np.sign(A["amp"]) == sign)
          & (A["sigma"] > 0.5 * sigma0) & (A["sigma"] < 2.2 * sigma0) & (A["rpk"] < 8.0)
          & (np.abs(A["d"] - A["d0"]) < 0.98 * A["dlim"]))
    frac = vA.mean(axis=1)
    comb = np.sqrt(np.nansum(np.where(vA, A["snr"], 0.0) ** 2, axis=1) * (Fg / frA.size))
    thr = max(float(cfg.snr_frame),
              float(snr_comb) * min(1.0, math.sqrt(Fg / max(int(cfg.snr_ref_frames), 1))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        smed = np.nanmedian(np.where(vA, A["sigma"], np.nan), axis=1)
    adopt = (frac >= cfg.frac_valid) & (comb >= thr) & (smed > 0.6 * sigma0) & (
        smed < 1.8 * sigma0)
    if adopt.sum() < 1:
        return None
    idx = np.where(adopt)[0]
    meta = [A["meta"][i] for i in idx]
    dA = np.where(vA[idx], A["d"][idx], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu0 = np.nanmedian(dA - np.nanmedian(dA, axis=0, keepdims=True), axis=1)
        off = np.nanmedian(dA - mu0[:, None], axis=0) - d0_frame[frA]
    zA = d0_frame[fr] + (float(np.nanmedian(off)) if np.isfinite(off).any() else 0.0)
    posA = np.searchsorted(fr, frA) if np.all(np.diff(fr) > 0) else np.array(
        [int(np.where(fr == f)[0][0]) for f in frA])
    zA[posA] = np.where(np.isfinite(off), d0_frame[frA] + off, zA[posA])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu0 = np.nanmedian(dA - zA[posA][None, :], axis=1)
    byo = {}
    for j, f in enumerate(meta):
        byo.setdefault(f["order"], []).append(j)
    feats_B = {mm: [meta[j] for j in js] for mm, js in byo.items()}
    d0B = {mm: mu0[js][:, None] + zA[None, :] for mm, js in byo.items()}
    sigB = {mm: smed[idx[js]] for mm, js in byo.items()}
    dlB = {mm: np.full(len(js), dlimB) for mm, js in byo.items()}
    B = _fit_features(sset, template, feats_B, d0B, sigma=sigB, dlim=dlB, fit_sigma=False,
                      cfg=cfg, frames=fr, n_iter=cfg.n_iter_fixed)
    vB = (np.isfinite(B["d"]) & (B["snr"] >= cfg.snr_frame) & (np.sign(B["amp"]) == sign)
          & (B["rpk"] < 8.0) & (np.abs(B["d"] - B["d0"]) < 0.98 * B["dlim"]))
    return dict(meta=B["meta"], d=B["d"], e=B["e_d"], valid=vB, frames=fr,
                sigma=np.concatenate([sigB[mm] for mm in sorted(sigB)]),
                n_candidates=len(A["meta"]), n_adopted=int(adopt.sum()), snr_threshold=thr)


def _collect_lines(sset, template, feats_by_order, d0_frame, groups, cfg, *, sigma0, dlimA,
                   dlimB, snr_comb, sign):
    """Per-group line measurement (:func:`_measure_group`) and the union of the lines:
    (K, F) ``d, e, valid`` + per-line meta."""
    F = sset.n_frames
    groups = np.asarray(groups)
    per = []
    for g in np.unique(groups):
        fr = np.where(groups == g)[0]
        per.append(_measure_group(sset, template, feats_by_order, d0_frame, fr, cfg,
                                  sigma0=sigma0, dlimA=dlimA, dlimB=dlimB, snr_comb=snr_comb,
                                  sign=sign))
    index, meta = {}, []
    for r in per:
        if r is None:
            continue
        for f in r["meta"]:
            k = (int(f["order"]), round(float(f["lref"]), 4))
            if k not in index:
                index[k] = len(meta)
                meta.append(f)
    K = len(meta)
    if K == 0:
        return None
    D = np.full((K, F), np.nan)
    E = np.full((K, F), np.nan)
    Vd = np.zeros((K, F), bool)
    SG = np.full((K, len(per)), np.nan)
    for j, r in enumerate(per):
        if r is None:
            continue
        rows = np.array([index[(int(f["order"]), round(float(f["lref"]), 4))]
                         for f in r["meta"]], int)
        D[np.ix_(rows, r["frames"])] = r["d"]
        E[np.ix_(rows, r["frames"])] = r["e"]
        Vd[np.ix_(rows, r["frames"])] = r["valid"]
        SG[rows, j] = r["sigma"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sigma = np.nanmedian(SG, axis=1)
    return dict(meta=meta, d=D, e=E, valid=Vd, groups=groups,
                order=np.array([f["order"] for f in meta], int),
                lref=np.array([f["lref"] for f in meta], np.float64), sigma=sigma,
                n_candidates=[None if r is None else r["n_candidates"] for r in per],
                n_adopted=[0 if r is None else r["n_adopted"] for r in per],
                snr_threshold=[None if r is None else r["snr_threshold"] for r in per])


def _solve_lines(col, template, cfg, *, part=None, regs=None, self_prior=False, label="",
                 design=None):
    """:func:`anchored_zero_points` on collected lines, with a surface part's line prior and
    its nod / drift patterns (``regs``: per-frame ``nod`` codes and absolute zero points ``X``)
    subtracted from the measurements.  The surface is rejected (with a warning; the solve is
    redone without it) when its prior-predictive residuals are LARGER than those of the bare
    line list - a table measured on another night that does not describe this one (e.g. a
    sky part from a bright-star night whose sky spectra are contaminated); "larger" = the
    median squared prior-predictive residual more than 10 % above the bare line list's.
    The stored A/B (+ drift) pattern is applied whenever the part is accepted
    (``pattern_used``; ``pattern_dresid`` = relative change of the median squared residual it
    makes, a diagnostic only)."""
    D, E, Vd, groups = col["d"], col["e"], col["valid"], col["groups"]
    K, F = D.shape
    order, lref = col["order"], col["lref"]
    off = np.zeros((K, F))
    pat = None
    if part is not None:
        mu_p, var_p, matched = _part_prior(part, template, order, lref)
        pat = (_part_pattern(part, template, order, lref, drift=False)
               if cfg.use_surface_modes else None)
        if pat is not None and regs is not None:
            pn, pd, X0 = pat[:3]
            nod = np.asarray(regs.get("nod", np.zeros(F)), np.float64)
            X = np.asarray(regs.get("X", np.full(F, X0)), np.float64)
            X = np.where(np.isfinite(X), X, X0)
            off = pn[:, None] * nod[None, :] + pd[:, None] * (X - X0)[None, :]
    else:
        mu_p, var_p, matched = np.zeros(K), np.zeros(K), np.zeros(K, bool)
    res = anchored_zero_points(D - off, E, groups=groups, valid=Vd, prior_mu=mu_p,
                               prior_var=var_p, self_prior=bool(self_prior and part is None),
                               huber=cfg.huber, design=design)
    pattern_used = None
    pattern_dresid = None
    if pat is not None and regs is not None and np.any(off):
        # The pattern is NOT re-decided per frame.  A "keep it only if it lowers the frame's
        # median squared residual" test is close to a coin flip for one exposure (the pattern
        # moves the residuals of 100-300 lines by a few 0.01 km/s against a 0.2-0.3 km/s
        # line scatter): it kept the pattern on 55-66 % of a night's frames, which put label
        # jumps between neighbouring frames (same-nod neighbour pulls of the star labels
        # higher in 8 of 9 nights, e.g. TOI712 2.52 vs 1.89, K2141 2.29 vs 1.67), and with
        # leave-one-out surfaces of 5-10 nights the star labels were further from the night
        # solution (7-night mean 0.067 vs 0.058 km/s rms; pattern off: 0.089).  Whether a
        # pattern is real is decided when the surface is built (empirical-Bayes shrinkage
        # over nights); use_surface_modes=False switches it off (advisable for a surface
        # built from a single short night, whose A/B pattern may not transfer).
        res_np = anchored_zero_points(D, E, groups=groups, valid=Vd, prior_mu=mu_p,
                                      prior_var=var_p, huber=cfg.huber, design=design)
        v_pat = float(np.nanmedian(res["resid"][Vd] ** 2)) if Vd.any() else np.nan
        v_np = float(np.nanmedian(res_np["resid"][Vd] ** 2)) if Vd.any() else np.nan
        pattern_used = True
        pattern_dresid = (v_pat / v_np - 1.0) if np.isfinite(v_pat) and np.isfinite(v_np) and \
            v_np > 0 else np.nan
    if pat is not None and regs is not None:
        # pattern-coefficient uncertainty: z absorbs the line-weighted mean of a pattern error
        # (dz_f = -(nod_f g + dX_f g) . dC, g_f the weighted mean basis of frame f), so the
        # LABEL error is (B(x) - g_f) . dC (0 at the lines' centroid).  z_err stays the error
        # given the pattern; the z sensitivity is kept for reference.
        w = res["w"]
        sw = w.sum(axis=0)
        g = (w.T @ pat[3]) / np.where(sw > 0, sw, 1.0)[:, None]          # (F, J)
        vp = nod ** 2 * (g ** 2 @ pat[4] ** 2) + (X - X0) ** 2 * (g ** 2 @ pat[5] ** 2)
        res = dict(res, z_sens_pattern=np.sqrt(vp), g_pattern=g)
    rejected = False
    if part is not None:
        free = anchored_zero_points(D, E, groups=groups, valid=Vd, huber=cfg.huber,
                                    design=design)
        v_p = float(np.nanmedian(res["resid"][Vd] ** 2)) if Vd.any() else np.nan
        v_f = float(np.nanmedian(free["resid"][Vd] ** 2)) if Vd.any() else np.nan
        if np.isfinite(v_p) and np.isfinite(v_f) and v_p > 1.1 * v_f:
            warnings.warn(f"{label}: the surface does not describe these lines (prior-predictive "
                          f"rms {math.sqrt(v_p):.3f} > {math.sqrt(v_f):.3f} km/s without it); "
                          "surface part NOT used", stacklevel=3)
            rejected = True
            res = (free if not self_prior else anchored_zero_points(
                D, E, groups=groups, valid=Vd, self_prior=True, huber=cfg.huber, design=design))
            mu_p, var_p, matched = np.zeros(K), np.zeros(K), np.zeros(K, bool)
            off = np.zeros((K, F))
            pat = None
            pattern_used = pattern_dresid = None
    return dict(res, **{k: v for k, v in col.items() if k != "groups"}, matched=matched,
                prior_mu=mu_p, prior_var=var_p, offsets=off, surface_rejected=rejected,
                pattern=pat, pattern_used=pattern_used, pattern_dresid=pattern_dresid)


def _measure_static(sset, template, feats_by_order, d0_frame, groups, cfg, *, sigma0, dlimA,
                    dlimB, snr_comb, sign, part=None, self_prior=False, regs=None, label="",
                    design=None):
    """:func:`_collect_lines` + :func:`_solve_lines`."""
    col = _collect_lines(sset, template, feats_by_order, d0_frame, groups, cfg, sigma0=sigma0,
                         dlimA=dlimA, dlimB=dlimB, snr_comb=snr_comb, sign=sign)
    if col is None:
        return None
    return _solve_lines(col, template, cfg, part=part, regs=regs, self_prior=self_prior,
                        label=label, design=design)


def _epoch_design(sset: SpectrumSet, groups, model: str):
    """(F, P) zero-point design of an epoch model (:func:`anchored_zero_points` ``design``):
    ``"frame"`` -> None (free per-frame zero points); ``"nod_drift"`` -> ``1, nod_f,
    t_f - <t>_group`` [h] (one zero point, an A/B term and a drift per epoch group)."""
    if model == "frame":
        return None
    if model != "nod_drift":
        raise ValueError(f"epoch_model={model!r}: 'frame' or 'nod_drift'")
    g = np.asarray(groups)
    t = np.zeros(sset.n_frames)
    for gg in np.unique(g):
        fr = g == gg
        t[fr] = (sset.time[fr] - sset.time[fr].mean()) * 24.0
    return np.column_stack([np.ones(sset.n_frames), _nod_code(sset.nod), t])


def _design_out(res):
    """Per-group epoch-model coefficients and sds (None for free per-frame zero points)."""
    bc = res.get("design_coef") if res is not None else None
    if bc is None or all(b is None for b in bc):
        return None
    return [None if b is None else dict(coef=np.asarray(b).tolist(),
                                        err=np.sqrt(np.clip(np.diag(c), 0, None)).tolist())
            for b, c in zip(bc, res["design_cov"], strict=True)]


def _static_mu(res) -> tuple[np.ndarray, np.ndarray]:
    """Per-line static offset combined over groups (precision-weighted ``shat``) and its sd."""
    sh, se = res["shat"], res["shat_err"]
    t2 = res["t_line"] ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        P = np.where(np.isfinite(sh) & np.isfinite(se) & (se > 0), 1.0 / (se ** 2 + t2), 0.0)
        S = P.sum(axis=1)
        mu = np.where(S > 0, np.sum(P * np.nan_to_num(sh), axis=1) / np.where(S > 0, S, 1.0),
                      np.nan)
        err = np.where(S > 0, 1.0 / np.sqrt(np.where(S > 0, S, 1.0)), np.nan)
    return mu, err


def _pattern_modes(md: dict, R: np.ndarray) -> dict:
    """A surface part's nod / drift patterns as an :class:`AnchorSolution` modes dict
    (regressors ``(nod, X)``, ``X`` centred on the surface's ``X0``)."""
    qx, qm = (int(v) for v in md["q"])
    J = qx * qm - 1
    cn = np.zeros(J) if md.get("nod") is None else np.asarray(md["nod"], np.float64)
    cd = np.zeros(J) if md.get("drift") is None else np.asarray(md["drift"], np.float64)
    return dict(qx=qx, qm=qm, xr=tuple(md["xr"]), mr=tuple(md["mr"]),
                C=np.column_stack([cn, cd]), C_cov=np.zeros((2 * J, 2 * J)), phi_mean=np.zeros(J),
                reg_mean=np.array([0.0, float(md.get("X0", 0.0))]), R_sky=np.asarray(R, np.float64),
                R_obj=np.asarray(R, np.float64), names=("nod", "X"), drift_regressor="surface")


# ------------------------------------------------------------------- orchestration
def choose_method(night: NightSpectra, config: AnchorConfig | None = None, groups=None) -> str:
    """``"single"`` (one exposure), ``"series"`` (one epoch group with at least
    ``series_min_frames`` frames over ``series_min_span_h`` hours: :func:`anchor_night`) or
    ``"sparse"`` (anything else: a few frames, or several epochs)."""
    cfg = config or AnchorConfig.for_mode(night.mode)
    s = night.sky if night.sky is not None else night.obj
    F = s.n_frames
    if F <= 1:
        return "single"
    if night.sky is None:              # A-only frames: anchor_night needs sky spectra
        return "sparse"
    g = epoch_groups(s.time, cfg.epoch_gap_h) if groups is None else np.asarray(groups)
    span_h = float(np.ptp(s.time)) * 24.0
    if (len(np.unique(g)) == 1 and F >= int(cfg.series_min_frames)
            and span_h >= float(cfg.series_min_span_h)):
        return "series"
    return "sparse"


@_quiet
def anchor_frames(night: NightSpectra, config: AnchorConfig | None = None, *,
                  surface: StaticSurface | None = None, groups=None, method: str = "sparse",
                  lines: LineTable | None = None, strict_surface: bool = True,
                  verbose: bool = False) -> AnchorSolution:
    """Single-image / sparse-epoch wavelength anchoring (no time model).

    No WARP equivalent (WARP's s10 needs a frame series and a reference frame,
    ``measure.py:warp_waveshift_log``; a single frame keeps the period ThAr scale).

    Every frame gets its OWN zero points: ``z_sky`` from the OH lines of its sky spectrum and
    ``z_star`` from the telluric lines of its object spectrum (so the star term includes the
    slit-illumination offset of that exposure), each a weighted mean of ``d_k - mu_k`` over
    the frame's lines (:func:`anchored_zero_points`), with errors from the line fits plus a
    fitted excess scatter.  The per-line static offsets ``mu_k`` come from

    * ``surface`` (a :class:`StaticSurface` of the same calibration set, built from other
      nights): every epoch is then on the surface's absolute scale, and the labels use the
      stored shape (sky: ``S_sky``; star: ``S_star`` if ``config.use_star_shape``);
    * otherwise ``method="sparse"`` with several epoch groups shares the groups' own static
      offsets (self prior); ``"single"`` treats every frame independently (``mu = 0``, line-list
      gauge) and, as :func:`anchor_night`, fits a static ``config.sky_shape`` surface to the
      frame's (group's) OH offsets for the sky labels.

    ``groups`` (F,) overrides the epoch grouping (default: ``"single"`` = one group per frame,
    ``"sparse"`` = :func:`epoch_groups` with ``config.epoch_gap_h``).  ``z_inst`` is the OH zero
    point of the sky spectrum paired with each object frame (same index; WARP's sky product is
    the partner exposure at the object aperture), with ``drift_rate_kms_h x |dt|`` added in
    quadrature.  A frame with fewer than ``config.min_lines_static`` telluric lines falls back
    to ``z_inst`` + the surface's slit prior (or 0 with ``config.star_fallback_err``).

    A-only frames (``night.sky is None``: ``decanter.reduce(sky=None)``, no nod partner): the
    object spectrum still holds the OH airglow, so the OH zero point is measured on it (same
    exposure: ``z_inst = z_sky``, no drift term); features whose window holds a contaminating
    line of the other kind are dropped .  ``config.epoch_model``
    (``"sparse"`` only) replaces the free per-frame zero points of each epoch group by
    ``1, nod [, t]`` (:func:`_epoch_design`; coefficients in ``diagnostics["tell_epoch_coef"]`` /
    ``["oh_epoch_coef"]``).
    """
    if method not in ("single", "sparse"):
        raise ValueError(f"method={method!r}: 'single' or 'sparse' (see anchor())")
    cfg = config or AnchorConfig.for_mode(night.mode)
    lines = lines or LineTable.load()
    tmpl = night.template
    sky, obj = night.sky, night.obj
    aonly = sky is None
    if aonly:
        # A-only frames (decanter.reduce(sky=None), no nod partner): no A-B subtraction, so the
        # OH airglow is in the object spectrum itself -> z_inst from the same exposure (dt = 0)
        if obj is None:
            raise ValueError("anchor_frames needs sky (OH) or A-only object spectra")
        sky = SpectrumSet(kind="sky", lam=obj.lam, flux=obj.flux, time=obj.time, nod=obj.nod,
                          epoch=obj.epoch, ivar=obj.ivar, shift_px=obj.shift_px,
                          frame_ids=obj.frame_ids)
    t_start = time.perf_counter()
    timing = {}
    fw = cfg.fwhm
    sig0 = fw / 2.3548
    surf = surface
    if surf is not None and not surf.matches(tmpl):
        msg = (f"{night.name}: surface {surf.calib or ''} measured on ThAr template "
               f"{surf.template_id}, night uses {_template_id(tmpl)}")
        if strict_surface:
            warnings.warn(msg + "; surface NOT used", stacklevel=2)
            surf = None
        else:
            warnings.warn(msg + "; used anyway (strict_surface=False)", stacklevel=2)
    if surf is not None and str(surf.mode).upper() != str(cfg.mode).upper():
        warnings.warn(f"surface mode {surf.mode} vs config mode {cfg.mode}", stacklevel=2)
    F = sky.n_frames
    if groups is not None:
        gsky = np.asarray(groups)
        if gsky.size != F:
            raise ValueError(f"{gsky.size} group labels for {F} frames")
        _u, gsky = np.unique(gsky, return_inverse=True)
    elif method == "single":
        gsky = np.arange(F)
    else:
        gsky = epoch_groups(sky.time, cfg.epoch_gap_h)
    gu = np.unique(gsky)
    G = gu.size
    # ---------------------------------------------------------------- (1) init per group
    d0 = np.zeros(F)
    dglob = np.zeros(G)
    n_bad = 0
    for j, g in enumerate(gu):
        fr = np.where(gsky == g)[0]
        drift, d_glob, _dv, good = _xcorr_init(_subset(sky, fr), tmpl, lines, cfg)
        d0[fr] = d_glob + drift
        dglob[j] = d_glob
        n_bad += int(np.sum(~good))
    obj_fit = obj
    timing["xcorr"] = time.perf_counter() - t_start
    # ---------------------------------------------------------------- (2) OH
    feats = {}
    for m in sky.orders:
        lam = np.asarray(sky.lam[m])
        fs = build_oh_features(lines, float(np.nanmin(lam)), float(np.nanmax(lam)), cfg.R,
                               imin=cfg.oh_imin, risk_max=cfg.oh_risk, pad=cfg.oh_pad)
        lo_ok = np.nanmin(lam) * (1 + (cfg.xcorr_vmax + fw * (cfg.oh_pad + 1)) / C_KMS)
        hi_ok = np.nanmax(lam) * (1 - (cfg.xcorr_vmax + fw * (cfg.oh_pad + 1)) / C_KMS)
        feats[m] = [f for f in fs if f["clean"] and lo_ok < f["lref"] < hi_ok]
    if aonly:
        feats = _aonly_exclude(feats, lines, cfg, "oh")
    t1 = time.perf_counter()
    sparse_self = method == "sparse" and G > 1
    nod_sky = _nod_code(sky.nod)
    part_sky = None if surf is None else surf.sky
    col = _collect_lines(sky, tmpl, feats, d0, gsky, cfg, sigma0=sig0,
                         dlimA=max(0.6 * fw, 3.0), dlimB=max(0.35 * fw, 1.5),
                         snr_comb=cfg.snr_comb_oh, sign=1.0)
    if col is None:
        raise RuntimeError("no usable OH lines")
    X_sky = d0.copy()
    emodel = cfg.epoch_model if method == "sparse" else "frame"
    des_sky = _epoch_design(sky, gsky, emodel)
    oh = _solve_lines(col, tmpl, cfg, part=part_sky, regs=dict(nod=nod_sky, X=X_sky),
                      self_prior=sparse_self, label=f"{night.name} sky", design=des_sky)
    if oh["pattern"] is not None and np.any(oh["pattern"][1]):
        # the drift pattern's regressor is the frame's own zero point: re-solve with it
        X_sky = np.where(np.isfinite(oh["z"]), oh["z"], d0)
        oh = _solve_lines(col, tmpl, cfg, part=part_sky, regs=dict(nod=nod_sky, X=X_sky),
                          self_prior=sparse_self, label=f"{night.name} sky", design=des_sky)
    sky_surface = part_sky is not None and not oh["surface_rejected"]
    if sky_surface and oh["pattern"] is not None:
        X_sky = np.full(F, float(part_sky["modes"].get("X0", 0.0)))   # nod pattern only
    timing["oh_fits"] = time.perf_counter() - t1
    z_sky = oh["z"].copy()
    z_sky_err = oh["z_err"].copy()
    shat = oh["shat"].copy()
    zmed = float(np.nanmedian(z_sky)) if np.isfinite(z_sky).any() else 0.0
    xs = _line_pixels(tmpl, oh["order"], oh["lref"] * (1 + (np.nan_to_num(oh["m"]) + zmed) / C_KMS))
    shape = None
    oh_modes = None
    group_shapes = None
    if sky_surface:
        shape = surf.shape("sky")
        if oh["pattern"] is not None:
            oh_modes = _pattern_modes(part_sky["modes"], np.column_stack([nod_sky, X_sky]))
    elif cfg.sky_shape is not None:
        qx, qm = (int(v) for v in cfg.sky_shape)
        xr = tmpl.p_range(int(sky.orders[0]))
        mr = (float(min(sky.orders)), float(max(sky.orders)))
        coefs = np.zeros((G, qx * qm))
        c00s = np.zeros(G)
        fitted = np.zeros(G, bool)
        for j in range(G):
            sh, se = shat[:, j], oh["shat_err"][:, j]
            ok = np.isfinite(sh) & np.isfinite(se)
            if ok.sum() < 3 * qx * qm:
                continue
            c = _fit_shape(xs[ok], oh["order"][ok], sh[ok],
                           np.sqrt(se[ok] ** 2 + oh["t_line"] ** 2 + 1e-6), qx, qm, xr, mr)
            c00s[j] = c[0, 0]
            c = c.copy()
            c[0, 0] = 0.0
            coefs[j] = c.ravel()
            fitted[j] = True
            z_sky[gsky == gu[j]] += c00s[j]
            shat[:, j] -= c00s[j]
        if fitted.any():
            J = qx * qm - 1
            oh_modes = dict(qx=qx, qm=qm, xr=xr, mr=mr, C=coefs[:, 1:].T.copy(),
                            C_cov=np.zeros((J * G, J * G)), phi_mean=np.zeros(J),
                            reg_mean=np.zeros(G), R_sky=_onehot(gsky, G),
                            names=tuple(f"group{int(g)}" for g in gu), drift_regressor="group")
            group_shapes = dict(c00=c00s, fitted=fitted)
    mu_st, mu_st_err = _static_mu(dict(oh, shat=shat))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ls_oh = 1.4826 * np.nanmedian(np.abs(oh["resid"]), axis=1)
    oh_out = dict(z=z_sky, z_err=z_sky_err, n_lines=oh["n_lines"], mu=mu_st, mu_err=mu_st_err,
                  line_scatter=ls_oh,
                  s_sys=float(np.median(oh["s_group"])), x=xs, order=oh["order"], lref=oh["lref"],
                  band=np.array([f["band"] for f in oh["meta"]]), d_glob=dglob, d=oh["d"],
                  e=oh["e"], valid=oh["valid"], sigma=oh["sigma"], groups=gsky,
                  s_group=oh["s_group"], lam_group=oh["lam_group"], inflation=oh["inflation"],
                  t_line=oh["t_line"], matched=oh["matched"],
                  prior_mu=oh["prior_mu"], shat=shat, scale=oh["scale"], offsets=oh["offsets"],
                  X=X_sky, surface_rejected=oh["surface_rejected"])
    diag = dict(name=night.name, mode=cfg.mode, method=method, n_sky=F, n_groups=G, aonly=aonly,
                epoch_model=emodel, oh_epoch_coef=_design_out(oh),
                surface=None if surf is None else surf.summary(),
                oh_xcorr_bad_frames=n_bad, oh_n_lines=len(oh["meta"]),
                oh_n_adopted=oh["n_adopted"],
                oh_lines_per_frame_median=float(np.median(oh["n_lines"])),
                oh_n_matched=int(np.sum(oh["matched"])),
                oh_surface_rejected=bool(oh["surface_rejected"]),
                oh_pattern_used=oh["pattern_used"], oh_pattern_dresid=oh["pattern_dresid"],
                oh_zp_abs_median=float(np.nanmedian(z_sky)) if np.isfinite(z_sky).any() else np.nan,
                oh_zp_err_median=float(np.nanmedian(z_sky_err)),
                oh_s_group_median=float(np.median(oh["s_group"])),
                oh_lam_group_median=float(np.median(oh["lam_group"])),
                oh_inflation_median=float(np.nanmedian(oh["inflation"])), oh_t_line=oh["t_line"],
                oh_scale=oh["scale"], sky_group_shapes=group_shapes)
    # ---------------------------------------------------------------- (3) tellurics
    z_star = z_star_err = z_inst = z_inst_err = None
    tell_out = None
    star_shape = None
    star_modes = None
    if obj is not None:
        if obj.n_frames == F:
            pair = np.arange(F)
        else:                                      # nearest sky exposure in time
            pair = np.array([int(np.argmin(np.abs(sky.time - t))) for t in obj.time])
        gobj = gsky[pair]
        z_inst = z_sky[pair].copy()
        dt_h = np.abs(obj.time - sky.time[pair]) * 24.0
        z_inst_err = np.sqrt(z_sky_err[pair] ** 2 + (cfg.drift_rate_kms_h * dt_h) ** 2)
        if oh_modes is not None and not sky_surface:      # per-group shapes
            oh_modes["R_obj"] = oh_modes["R_sky"][pair]
        t3 = time.perf_counter()
        dg = float(np.max(np.abs(dglob))) if dglob.size else 0.0
        tfeats = {}
        for m in obj.orders:
            if m in cfg.exclude_tell_orders:
                continue
            lam = np.asarray(obj.lam[m])
            lo = np.nanmin(lam) * (1 + (dg + cfg.xcorr_vmax + 2 * fw) / C_KMS)
            hi = np.nanmax(lam) * (1 - (dg + cfg.xcorr_vmax + 2 * fw) / C_KMS)
            tf = build_absorber_features(lines, lo, hi, cfg.R, species=cfg.species,
                                         p_eff=dict(cfg.p_eff), depth=cfg.tell_depth,
                                         blend_max=cfg.tell_blend, halfwin=cfg.tell_halfwin,
                                         risk_max=cfg.tell_risk)
            tfeats[m] = [f for f in tf if f["clean"]]
        if aonly:
            tfeats = _aonly_exclude(tfeats, lines, cfg, "tell")
        d0o = np.where(np.isfinite(z_inst), z_inst, d0[pair])
        nod_obj = _nod_code(obj.nod)
        part_star = None if surf is None else surf.star
        te = _measure_static(obj_fit, tmpl, tfeats, d0o, gobj, cfg, sigma0=sig0,
                             dlimA=max(0.6 * fw, 3.0), dlimB=max(0.35 * fw, 1.5),
                             snr_comb=cfg.snr_comb_tell, sign=-1.0, part=part_star,
                             self_prior=sparse_self, regs=dict(nod=nod_obj, X=d0o),
                             label=f"{night.name} star",
                             design=_epoch_design(obj_fit, gobj, emodel))
        star_surface = part_star is not None and te is not None and not te["surface_rejected"]
        star_modes = None
        X0s = float(part_star["modes"].get("X0", 0.0)) if star_surface and part_star.get(
            "modes") else 0.0
        if star_surface and te["pattern"] is not None:
            star_modes = _pattern_modes(part_star["modes"], np.column_stack(
                [nod_obj, np.full(obj.n_frames, X0s)]))
        if oh_modes is not None and sky_surface:
            oh_modes["R_obj"] = np.column_stack([nod_obj, np.full(obj.n_frames,
                                                                      oh_modes["reg_mean"][1])])
        timing["tell_fits"] = time.perf_counter() - t3
        slit = None if surf is None or surf.star is None else surf.star.get("slit")
        off = float(slit["median"]) if slit else 0.0
        off_err = float(slit["scatter"]) if slit and slit.get("scatter") else cfg.star_fallback_err
        if te is None:
            z_star = np.full(obj.n_frames, np.nan)
            z_star_err = np.full(obj.n_frames, np.nan)
            n_tl = np.zeros(obj.n_frames, int)
        else:
            z_star, z_star_err = te["z"].copy(), te["z_err"].copy()
            n_tl = te["n_lines"]
        fb = ~np.isfinite(z_star) | (n_tl < int(cfg.min_lines_static))
        z_star = np.where(fb, z_inst + off, z_star)
        z_star_err = np.where(fb, np.sqrt(z_inst_err ** 2 + off_err ** 2), z_star_err)
        if fb.any():
            warnings.warn(f"{night.name}: {int(fb.sum())} frame(s) with < {cfg.min_lines_static} "
                          "telluric lines; star term = OH zero point" + (
                              " + slit prior" if slit else ""), stacklevel=2)
        if star_surface and cfg.use_star_shape:
            star_shape = surf.shape("star")
        diag.update(star_source="tellurics" if not fb.all() else "oh",
                    star_fallback_frames=int(fb.sum()),
                    star_minus_inst_median=float(np.nanmedian(z_star - z_inst)),
                    star_z_err_median=float(np.nanmedian(z_star_err)))
        if te is not None:
            tmu, tmu_err = _static_mu(te)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ls_te = 1.4826 * np.nanmedian(np.abs(te["resid"]), axis=1)
            tell_out = dict(z=te["z"], z_err=te["z_err"], n_lines=te["n_lines"], mu=tmu,
                            mu_err=tmu_err, line_scatter=ls_te,
                            s_sys=float(np.median(te["s_group"])), order=te["order"],
                            lref=te["lref"],
                            species=np.array([f["species"] for f in te["meta"]]), d=te["d"],
                            e=te["e"], valid=te["valid"], sigma=te["sigma"], X=z_inst,
                            X_sd=z_inst_err, groups=gobj, s_group=te["s_group"],
                            lam_group=te["lam_group"], inflation=te["inflation"],
                            t_line=te["t_line"], matched=te["matched"], prior_mu=te["prior_mu"],
                            shat=te["shat"], scale=te["scale"], fallback=fb,
                            offsets=te["offsets"], surface_rejected=te["surface_rejected"])
            diag.update(tell_n_lines=len(te["meta"]), tell_n_adopted=te["n_adopted"],
                        tell_lines_per_frame_median=float(np.median(te["n_lines"])),
                        tell_n_matched=int(np.sum(te["matched"])),
                        tell_surface_rejected=bool(te["surface_rejected"]),
                        tell_pattern_used=te["pattern_used"],
                        tell_pattern_dresid=te["pattern_dresid"],
                        tell_zp_err_median=float(np.nanmedian(te["z_err"])),
                        tell_s_group_median=float(np.median(te["s_group"])),
                        tell_lam_group_median=float(np.median(te["lam_group"])),
                        tell_inflation_median=float(np.nanmedian(te["inflation"])),
                        tell_epoch_coef=_design_out(te),
                        tell_t_line=te["t_line"], tell_scale=te["scale"])
    timing["total"] = time.perf_counter() - t_start
    diag["timing_s"] = timing
    if verbose:
        print({k: v for k, v in diag.items() if not isinstance(v, dict)})
    return AnchorSolution(
        name=night.name, mode=cfg.mode, config=cfg, template=tmpl,
        obj_time=None if obj is None else obj.time, obj_nod=None if obj is None else obj.nod,
        obj_shift_px=None if obj is None else obj.shift_px, z_star=z_star, z_star_err=z_star_err,
        z_inst=z_inst, z_inst_err=z_inst_err, sky_time=sky.time, sky_nod=sky.nod, z_sky=z_sky,
        z_sky_err=z_sky_err, oh=oh_out, tell=tell_out, shape=shape, oh_model=None,
        star_model=None, diagnostics=diag, oh_modes=oh_modes, star_modes=star_modes,
        star_shape=star_shape, method=method)


def anchor(night: NightSpectra, config: AnchorConfig | None = None, *,
           surface: StaticSurface | None = None, method: str = "auto", groups=None,
           lines: LineTable | None = None, verbose: bool = False) -> AnchorSolution:
    """Wavelength anchoring with the path chosen by the frame sampling.

    ``method="auto"`` (:func:`choose_method`): one exposure -> ``"single"``; one continuous
    epoch with >= ``series_min_frames`` frames over >= ``series_min_span_h`` h -> ``"series"``
    (:func:`anchor_night`, unchanged: GP time model; ``surface`` is not used there); anything
    else (a few frames, several nights) -> ``"sparse"`` (:func:`anchor_frames`).  Any of the
    three can be forced; ``"single"`` on a multi-frame night anchors every frame on its own.
    """
    cfg = config or AnchorConfig.for_mode(night.mode)
    if method == "auto":
        method = choose_method(night, cfg, groups)
    if method == "series":
        return anchor_night(night, cfg, lines=lines, verbose=verbose)
    return anchor_frames(night, cfg, surface=surface, groups=groups, method=method,
                         lines=lines, verbose=verbose)


def anchor_reduction(reductions, template, *, surface="auto", config: AnchorConfig | None = None,
                     method: str = "auto", cut: float = 1.30,
                     lines: LineTable | None = None) -> AnchorSolution:
    """Anchor :class:`decanter.Reduction` output: one reduction (``decanter.reduce``: the
    single-image path), a list, or a ``TransitSeries`` (``decanter.reduce_many``: the series
    path for a night).  ``template``: a dispersion template or a :class:`decanter.Calibration`
    (then ``surface="auto"`` loads the calibration set's stored :class:`StaticSurface`, if
    any; pass a surface or None explicitly otherwise).  A-only reductions (``sky=None``: no
    sky spectra) take the A-only path of :func:`anchor_frames`."""
    reds = getattr(reductions, "reductions", reductions)
    reds = list(reds) if isinstance(reds, (list, tuple)) else [reds]
    calib = None
    if hasattr(template, "id_dir"):
        calib = template
        template = ThArTemplate.from_calibration(calib)
    if isinstance(surface, str):
        if surface != "auto":
            raise ValueError("surface: a StaticSurface, None or 'auto'")
        surface = StaticSurface.find(calib, template) if calib is not None else None
    night = night_from_reductions(reds, template, cut=cut)
    return anchor(night, config, surface=surface, method=method, lines=lines)


# =========================================================================== export
@_quiet
def export_resample(flux, lam_label, u, lam_out, *, a: int = 8, xp=np, fill=np.nan):
    """The ONE resample: Lanczos-``a`` interpolation (in pixel index) of spectra whose
    samples sit at ``lambda_true = lam_label * exp(u)`` onto ``lam_out``.

    Args:
        flux: (F, n) or (n,).
        lam_label: (n,) float64 static labels (monotonic increasing).
        u: scalar, (F,) or (F, n) label offsets (:meth:`AnchorSolution.labels` / ``.u``).
        lam_out: (n_out,) target wavelengths (true vacuum scale).
        a: Lanczos order (8 = the report's choice: 674 ppm line-core residual vs 2902 ppm for
           WARP's 3-step chain in the toy model).
        xp: array namespace (numpy default; jax.numpy for gradients through ``flux``).
        fill: value outside the data (or where a tap is non-finite).

    The weights are normalised to sum to 1 (a constant is preserved exactly).
    """
    flux_np = np.asarray(flux) if xp is np else flux
    one = np.ndim(flux_np) == 1
    fl = xp.reshape(flux_np, (1, -1)) if one else flux_np
    F, n = fl.shape
    lam_label = np.asarray(lam_label, np.float64)
    uu = np.asarray(u, np.float64)
    if uu.ndim == 0:
        uu = np.full((F, 1), float(uu))
    elif uu.ndim == 1:
        uu = uu[:, None]
    if uu.ndim != 2 or uu.shape[0] != F or uu.shape[1] not in (1, n):
        raise ValueError(f"u of shape {np.shape(u)} for {F} spectra x {n} samples")
    lt = lam_label[None, :] * np.exp(uu)                     # (F, n) float64
    lam_out = np.asarray(lam_out, np.float64)
    idx = np.arange(n, dtype=np.float64)
    x = np.stack([np.interp(lam_out, lt[f], idx, left=np.nan, right=np.nan) for f in range(F)])
    i0 = np.floor(np.nan_to_num(x, nan=-10 * n)).astype(np.int64)
    taps = np.arange(-a + 1, a + 1)
    j = i0[..., None] + taps                                  # (F, n_out, 2a)
    dx = x[..., None] - j
    w = np.sinc(dx) * np.sinc(dx / a)
    inside = (j >= 0) & (j < n) & np.isfinite(x)[..., None]
    jc = np.clip(j, 0, n - 1)
    fi = np.arange(F)[:, None, None]
    vals = fl[fi, jc] if xp is np else xp.asarray(fl)[fi, jc]
    good = inside & (np.isfinite(np.asarray(vals)) if xp is np else inside)
    w = np.where(good, w, 0.0)
    ws = w.sum(-1)
    full = good.all(-1) & np.isfinite(x)
    if xp is np:
        out = np.where(full, np.sum(np.where(good, vals, 0.0) * w, -1) / np.where(ws != 0, ws, 1.0), fill)
    else:
        out = xp.where(xp.asarray(full), xp.sum(xp.where(xp.asarray(good), vals, 0.0)
                                                * xp.asarray(w), -1) / xp.asarray(
            np.where(ws != 0, ws, 1.0)), fill)
    return out[0] if one else out
