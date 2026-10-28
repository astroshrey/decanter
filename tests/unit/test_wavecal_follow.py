"""decanter.wavecal follow-up (block C2c): the conditioning guard of anchor_night's
detector modes, A-only frames (no sky spectrum, no nod partner), the sparse-epoch model (one zero
point + an A/B term [+ drift] per ABBA epoch).  Real-data checks are marked ``regression`` and skip when the
WARP reductions / raw frames are absent.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from decanter._localpaths import WINERED_ROOT  # noqa: E402
from decanter.wavecal.model import (  # noqa: E402
    AnchorConfig,
    NightSpectra,
    SpectrumSet,
    ThArTemplate,
    _guard_modes,
    _mode_leverage,
    anchor,
    anchor_night,
    anchor_reduction,
    anchored_zero_points,
    choose_method,
)
from tests.unit.test_wavecal import COEF  # noqa: E402
from tests.unit.test_wavecal_single import ORDERS, _night  # noqa: E402

RED = WINERED_ROOT / "reductions"
HIRESY = tuple(range(158, 184))


def _coef(m, dc0=0.0):
    c = np.asarray(COEF[163], np.float64) * 163.0 / m
    c[0] += dc0
    return c


def _wide_template(dc0=0.0):
    """A 26-order HIRES-Y-like template (orders 158-183)."""
    return ThArTemplate.from_arrays(HIRESY, [_coef(m, dc0) for m in HIRESY], 1.0, 4095.0, kind=1,
                                    dy=0.5)


# --------------------------------------------------------------------- mode guard
def test_guard_modes_ladder():
    """Well-spread lines keep the (3, 3) basis (the SAME spec object: no side effects on a normal
    night); TOI-270c-like lines (13, five of six orders at one end) step down to (2, 1); 3 lines
    leave no modes."""
    T = _wide_template()
    cfg = AnchorConfig.for_mode("HIRES-Y")
    spec = dict(qx=3, qm=3, xr=T.p_range(158), mr=(158.0, 183.0))
    rng = np.random.default_rng(0)
    om = rng.choice(np.r_[158:165, 176:184], 180)
    lam = np.array([float(T.lam(int(m), np.array([rng.uniform(300, 3800)]))[0]) for m in om])
    sp, info = _guard_modes(T, om, lam, spec, cfg)
    assert sp is spec and not info["triggered"] and info["q_used"] == (3, 3)
    assert info["tried"][0]["leverage"] < 60.0
    # TOI-270c: 13 telluric lines, orders 160 + 179-183, pixels 1745-3216
    om13 = np.array([160, 179, 180, 180, 180, 181, 181, 181, 181, 182, 183, 183, 183])
    x13 = np.array([2349, 3216, 2664, 2596, 2350, 2718, 2679, 2667, 2156, 3187, 3088, 2962, 1745.])
    lam13 = np.array([float(T.lam(int(m), np.array([x]))[0]) for m, x in zip(om13, x13,
                                                                           strict=True)])
    assert _mode_leverage(T, om13, lam13, spec) > 1e4
    sp, info = _guard_modes(T, om13, lam13, spec, cfg)
    assert info["triggered"] and info["q_used"] == (2, 1) and sp["qx"] == 2 and sp["qm"] == 1
    assert [t["q"] for t in info["tried"]] == [(3, 3), (2, 2), (2, 1)]
    sp, info = _guard_modes(T, om13[:3], lam13[:3], spec, cfg)
    assert sp is None and info["q_used"] is None


def test_guard_in_anchor_night_synthetic():
    """A 3-order night with (3, 3) star modes forced: too few telluric lines for 8 mode terms ->
    the guard steps down and records it; the default 'auto' (2, 2) is untouched."""
    rng = np.random.default_rng(5)
    t = 60500.0 + np.arange(12) * (330.0 / 86400.0)
    th = (t - t.mean()) * 24
    night = _night(rng, t, 0.3 + 0.4 * th, -0.1 + 0.4 * th, static=False, name="g")
    cfg = AnchorConfig.for_mode("HIRES-Y", star_modes=(3, 3), sky_modes=(3, 3),
                                modes_min_lines_per_term=12.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = anchor_night(night, cfg)
    g = s.diagnostics["star_modes_guard"]
    assert g["q_requested"] == (3, 3) and g["triggered"]
    assert s.diagnostics.get("star_modes_q") == g["q_used"]
    if g["q_used"] is not None:
        assert s.star_modes["qx"] * s.star_modes["qm"] < 9
    s0 = anchor_night(night, AnchorConfig.for_mode("HIRES-Y"))
    assert not s0.diagnostics["star_modes_guard"]["triggered"]
    assert s0.diagnostics["star_modes_q"] == (2, 2)


@pytest.mark.regression
def test_guard_real_nights():
    """TOI-270c (13 telluric lines): star modes (3, 3) -> (2, 1), mode swing 654 -> < 2 km/s;
    TOI-2109: no trigger (3, 3)."""
    from decanter.wavecal.model import load_warp_night

    for name in ("TOI270c", "TOI2109"):
        if not (RED / name).is_dir():
            pytest.skip(f"{name} WARP reduction not on this machine")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = anchor_night(load_warp_night(RED / "TOI270c"))
        b = anchor_night(load_warp_night(RED / "TOI2109"))
    ga = a.diagnostics["star_modes_guard"]
    assert ga["triggered"] and ga["q_used"] == (2, 1)
    assert a.diagnostics["star_modes_ptp_kms"] < 2.0
    for k in ("star_modes_guard", "oh_modes_guard"):
        assert not b.diagnostics[k]["triggered"] and b.diagnostics[k]["q_used"] == (3, 3)


# ------------------------------------------------------------------------ A-only frames
def _aonly(night):
    """Object spectra with the OH airglow left in (no A-B subtraction), same exposure."""
    o = night.obj
    flux = {m: np.asarray(o.flux[m]) + np.asarray(night.sky.flux[m]) for m in o.flux}
    obj = SpectrumSet(kind="obj", lam=o.lam, flux=flux, time=o.time, nod=o.nod, epoch=o.epoch,
                      shift_px=o.shift_px)
    return NightSpectra(sky=None, obj=obj, template=night.template, mode=night.mode,
                        name=night.name + "_aonly")


def test_aonly_single_frame_recovers_zero_points():
    """sky=None: the OH lines are measured in the object spectrum itself (telluric lines masked)
    and the telluric lines with the OH lines masked; z_inst = z_sky of the same exposure."""
    rng = np.random.default_rng(21)
    t = np.array([60500.3])
    night = _night(rng, t, np.array([0.7]), np.array([0.35]), shift=False, static=False,
                   name="ao")
    # the sky product of _night is the partner exposure: put the OH at the object time
    night = NightSpectra(sky=SpectrumSet(kind="sky", lam=night.sky.lam, flux=night.sky.flux,
                                         time=t, nod=night.sky.nod, epoch=night.sky.epoch),
                         obj=night.obj, template=night.template, mode=night.mode, name="ao")
    ao = _aonly(night)
    cfg = AnchorConfig.for_mode("HIRES-Y")
    assert choose_method(ao, cfg) == "single"
    s = anchor(ao, cfg)
    assert s.method == "single" and s.diagnostics["aonly"]
    assert s.diagnostics["oh_n_lines"] >= 10 and s.diagnostics["tell_n_lines"] >= 10
    assert abs(s.z_inst[0] - 0.7) < 3 * s.z_inst_err[0] + 0.02, s.z_inst
    assert s.z_inst[0] == s.z_sky[0] and s.z_inst_err[0] == s.z_sky_err[0]      # dt = 0
    assert abs(s.z_star[0] - 0.35) < 3 * s.z_star_err[0] + 0.02, s.z_star
    # several A-only frames: never the series path (it needs sky spectra)
    t4 = 60500.3 + np.arange(10) * (330.0 / 86400.0)
    n4 = _aonly(_night(rng, t4, 0.7 + 0 * t4, 0.35 + 0 * t4, shift=False, static=False,
                       name="ao4"))
    assert choose_method(n4, cfg) == "sparse"


def test_anchor_reduction_aonly_dispatch():
    """A decanter.reduce(sky=None)-style Reduction (sky = {}, no sky header) anchors through
    anchor_reduction on the A-only path."""
    rng = np.random.default_rng(23)
    t_mid = 60895.0 + (3600.0 + 150.0) / 86400.0
    night = _night(rng, np.array([t_mid]), np.array([0.9]), np.array([0.5]), shift=False,
                   static=False, name="red")
    ao = _aonly(night)

    def spectra(sset):
        out = {}
        for m in ORDERS:
            L = sset.lam[m]
            out[(1.30, m)] = SimpleNamespace(flux=np.asarray(sset.flux[m][0]), crval1=float(L[0]),
                                             cdelt1=float(L[1] - L[0]), crpix1=1.0)
        return out

    hdr = {"DATE-OBS": "2025-08-08", "UT-STR": "01:00:00.0", "UT-END": "01:05:00.0",
           "EXPTIME": 300.0, "NODPOS": "A1", "FRAMEID": "WINA1", "INSTMODE": "HIRES-Y"}
    red = SimpleNamespace(obj=spectra(ao.obj), sky={}, obj_header=hdr, sky_header=None,
                          shift_wave=0.0)
    sol = anchor_reduction(red, night.template, surface=None)
    assert sol.method == "single" and sol.diagnostics["aonly"]
    assert abs(sol.z_inst[0] - 0.9) < 3 * sol.z_inst_err[0] + 0.03
    assert abs(sol.z_star[0] - 0.5) < 3 * sol.z_star_err[0] + 0.03


@pytest.mark.regression
def test_aonly_real_frame_toi2109():
    """decanter.reduce(sky=None) on a raw TOI-2109 A frame -> anchor_reduction: z_inst within
    3 sigma of the night's series OH model at that exposure, z_star within 3 sigma of the WARP
    A-B frame's single-frame telluric zero point."""
    import decanter
    from decanter.wavecal.model import load_warp_night

    raw = WINERED_ROOT / "science_data" / "2025_08_07" / "WINA00053571.fits"
    ref = RED / "TOI2109_decanterref"
    if not raw.is_file() or not ref.is_dir() or not (RED / "TOI2109" / "TOI2109_NO1").is_dir():
        pytest.skip("TOI-2109 raw frame / reductions not on this machine")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calib = decanter.Calibration.from_dir(ref)
        r = decanter.reduce(raw, calib, sky=None)
        a = anchor_reduction(r, calib, surface=None)
        night = load_warp_night(RED / "TOI2109")
        ser = anchor_night(night)
        i = next(k for k, f in enumerate(night.obj.frame_ids) if str(f).endswith("_NO1"))
        one = NightSpectra(sky=_sub(night.sky, [i]), obj=_sub(night.obj, [i]),
                           template=night.template, mode=night.mode, name="NO1")
        b = anchor(one, AnchorConfig.for_mode(night.mode))
    assert a.method == "single" and a.diagnostics["aonly"]
    assert abs(a.z_inst[0] - ser.z_inst[i]) < 3 * np.hypot(a.z_inst_err[0], ser.z_inst_err[i])
    assert abs(a.z_star[0] - b.z_star[0]) < 3 * np.hypot(a.z_star_err[0], b.z_star_err[0])


def _sub(sset, idx):
    from decanter.wavecal.model import _subset

    return _subset(sset, idx)


# ------------------------------------------------------------------- sparse epoch model
def test_anchored_zero_points_design():
    """z_f = z0 + delta nod_f + gamma t_f inside a group: the GLS coefficients (mu eliminated)
    are recovered with honest errors; a group with one nod position drops the nod column."""
    rng = np.random.default_rng(7)
    K, F = 120, 8
    nod = np.array([0.5, -0.5, -0.5, 0.5] * 2)
    t = np.r_[np.arange(4) * 0.09 - 0.135, np.arange(4) * 0.09 - 0.135]
    g = np.r_[np.zeros(4, int), np.ones(4, int)]
    b_true = np.array([[0.40, 0.08, 0.30], [-0.20, -0.05, -0.50]])
    z = np.array([b_true[g[f]] @ [1, nod[f], t[f]] for f in range(F)])
    mu = rng.normal(0, 0.2, K)
    e = rng.uniform(0.02, 0.06, (K, F))
    d = mu[:, None] + z[None, :] + rng.normal(0, 1, (K, F)) * e
    des = np.column_stack([np.ones(F), nod, t])
    r = anchored_zero_points(d, e, groups=g, design=des)
    for k in range(2):
        b, cov = r["design_coef"][k], r["design_cov"][k]
        assert b.size == 3
        # the per-group intercept carries the group's static-offset gauge (mean mu): compare
        # the nod and drift terms, and the zero points up to that constant
        pull = (b[1:] - b_true[k][1:]) / np.sqrt(np.diag(cov)[1:])
        assert np.all(np.abs(pull) < 4), pull
    zz = r["z"] - z
    assert np.ptp(zz[g == 0]) < 0.02 and np.ptp(zz[g == 1]) < 0.02
    # one nod position in a group -> the nod column is dropped there
    nod1 = nod.copy()
    nod1[g == 1] = 0.5
    r1 = anchored_zero_points(d, e, groups=g, design=np.column_stack([np.ones(F), nod1, t]))
    assert r1["design_coef"][0].size == 3 and r1["design_coef"][1].size == 2
    # without a design: free per-frame zero points, no coefficients
    r0 = anchored_zero_points(d, e, groups=g)
    assert all(b is None for b in r0["design_coef"])


def test_sparse_abba_epoch_model():
    """A 4-frame ABBA epoch: 'nod_drift' (one zero point + A/B term + drift) recovers the
    injected per-frame zero points of sky and star; 'frame' is the default (free per-frame zero
    points)."""
    rng = np.random.default_rng(31)
    t = 60500.2 + np.arange(4) * (330.0 / 86400.0)
    nodc = np.array([0.5, -0.5, -0.5, 0.5])
    th = (t - t.mean()) * 24
    zs = 0.5 + 0.10 * nodc + 0.6 * th
    zt = 0.2 + 0.06 * nodc + 0.55 * th
    night = _night(rng, t, zs, zt, shift=False, static=False, name="abba")
    # sky product at the object time for a clean test of the epoch model
    night = NightSpectra(sky=SpectrumSet(kind="sky", lam=night.sky.lam, flux=night.sky.flux,
                                         time=t, nod=night.sky.nod, epoch=night.sky.epoch),
                         obj=night.obj, template=night.template, mode=night.mode, name="abba")
    sols = {}
    for m in ("frame", "nod_drift"):
        cfg = AnchorConfig.for_mode("HIRES-Y", epoch_model=m)
        sols[m] = anchor(night, cfg, method="sparse")
    assert sols["frame"].diagnostics["epoch_model"] == "frame"
    assert sols["frame"].diagnostics["tell_epoch_coef"] is None
    s = sols["nod_drift"]
    co = s.diagnostics["tell_epoch_coef"][0]
    assert len(co["coef"]) == 3
    assert abs(co["coef"][1] - 0.06) < 4 * co["err"][1] + 0.01
    assert abs(co["coef"][2] - 0.55) < 4 * co["err"][2] + 0.03
    for kind, ztrue in (("sky", zs), ("star", zt)):
        z, ez = getattr(s, f"z_{kind}"), getattr(s, f"z_{kind}_err")
        assert np.all(np.abs(z - ztrue) < 3 * ez + 0.02), (kind, z - ztrue)
        zf = getattr(sols["frame"], f"z_{kind}")
        assert np.median(ez) <= np.median(getattr(sols["frame"], f"z_{kind}_err")) * 1.05
        assert abs(np.mean(z) - np.mean(zf)) < 0.02
    with pytest.raises(ValueError):
        anchor(night, AnchorConfig.for_mode("HIRES-Y", epoch_model="nod"), method="sparse")
