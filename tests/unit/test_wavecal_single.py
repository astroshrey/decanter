"""decanter.wavecal single-image / sparse-epoch path: anchored zero points, the static
per-calibration-set surface (build / save / load), method selection, and synthetic recovery of
injected zero points for one frame and for sparse epochs sharing one static surface.  The
series path (anchor_night) must be untouched: ``anchor(method="auto")`` on a full night returns
exactly anchor_night's solution.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from decanter.wavecal.model import (  # noqa: E402
    C_KMS,
    SURFACE_FILENAME,
    SURFACE_VERSION,
    AnchorConfig,
    LineTable,
    NightSpectra,
    SpectrumSet,
    StaticSurface,
    ThArTemplate,
    anchor,
    anchor_frames,
    anchor_night,
    anchored_zero_points,
    build_static_surface,
    choose_method,
    epoch_groups,
)
from tests.unit.test_wavecal import COEF, _synthetic_night  # noqa: E402

# m163 (He I, 14 clean H2O + 38 OH features), m174 (OH) and m182 (0.94 um H2O band, 51 H2O):
# a few-order night with enough telluric lines for a per-frame star zero point
ORDERS = (163, 174, 182)


def _coef(m, dc0=0.0):
    c = np.asarray(COEF[163], np.float64) * 163.0 / m     # grating equation lambda ~ 1/m
    c[0] += dc0
    return c


def _template(dc0=0.0):
    return ThArTemplate.from_arrays(ORDERS, [_coef(m, dc0) for m in ORDERS], 1.0, 4095.0,
                                    kind=1, dy=0.5)


def _static_offset(lam, m, T, *, smooth, rms, seed):
    """Static departure [km/s] of a line at vacuum ``lam`` in order ``m`` from the line list on
    the ThAr scale: a smooth (x, m) surface plus a per-line 'line-list error'."""
    p = T.p_of_lam(m, np.atleast_1d(lam))
    xn = (2 * p - 4096.0) / 4094.0
    mn = (2 * m - (ORDERS[0] + ORDERS[-1])) / (ORDERS[-1] - ORDERS[0])
    r = np.array([np.random.default_rng((seed, round(float(v) * 1e4))).normal() for v in
                  np.atleast_1d(lam)])
    return smooth[0] * xn + smooth[1] * xn * mn + smooth[2] * mn + rms * r


def _night(rng, t_obj, z_sky, z_star, *, snr=1.0, shift=True, static=True, name="synth",
           nod_pattern=(0.0, 0.0), nod0="A"):
    """Sky (OH) + object (H2O) spectra of frames at ``t_obj`` with per-frame zero points
    ``z_sky`` / ``z_star`` [km/s], the shared static line offsets, and an optional A/B slit
    pattern ``nod_code x a x xn`` (a = ``nod_pattern`` sky / star, xn = normalised pixel)."""
    T = _template()
    lt = LineTable.load()
    fw = C_KMS / 60000.0
    sig = fw / 2.3548
    F = t_obj.size
    seq = ["A", "B", "B", "A"] if nod0 == "A" else ["B", "A", "A", "B"]
    nod = np.array(seq * (F // 4 + 1))[:F]
    nodc = np.where(nod == "A", 0.5, -0.5)

    def xn_of(lam, m):
        return (2 * T.p_of_lam(m, np.atleast_1d(lam)) - 4096.0) / 4094.0
    lam, sky, obj = {}, {}, {}
    shift_px = rng.normal(0, 1.0, F) if shift else np.zeros(F)
    for m in T.orders:
        lo, hi = sorted(T.lam(m, np.array([4095.0, 1.0])))
        L = np.linspace(lo, hi, 4400)
        lam[m] = L
        ol = lt.oh_lines(lo - 5, hi + 5)
        dso = (_static_offset(ol["lam"], m, T, smooth=(0.10, 0.06, 0.04), rms=0.05, seed=1)
               if static else np.zeros(ol["lam"].size))
        S = np.zeros((F, L.size))
        pso = nod_pattern[0] * xn_of(ol["lam"], m)
        for lc, ii, ds, ps in zip(ol["lam"], ol["I"], dso, pso, strict=True):
            c = lc * (1 + (z_sky + ds + nodc * ps) / C_KMS)
            S += 4.0e4 * snr * ii * np.exp(-0.5 * (((L[None, :] / c[:, None]) - 1) * C_KMS / sig) ** 2)
        S += 40.0 * snr
        sky[m] = S + rng.normal(0, 1, S.shape) * np.sqrt(S)
        p = T.p_of_lam(m, L)
        raw = T.lam(m, p[None, :] - shift_px[:, None])
        al = lt.absorber_lines(lo - 5, hi + 5, species=("H2O",), R=60000.0)
        dst = (_static_offset(al["lam"], m, T, smooth=(-0.08, 0.05, 0.03), rms=0.08, seed=2)
               if static else np.zeros(al["lam"].size))
        tau = np.zeros((F, L.size))
        pst = nod_pattern[1] * xn_of(al["lam"], m)
        for lc, dep, ds, ps in zip(al["lam"], al["depth"], dst, pst, strict=True):
            c = lc * (1 + (z_star + ds + nodc * ps) / C_KMS)
            tau += -np.log1p(-min(dep, 0.9)) * np.exp(
                -0.5 * (((raw / c[:, None]) - 1) * C_KMS / sig) ** 2)
        cont = 3000.0 * snr * np.exp(-tau)
        obj[m] = cont + rng.normal(0, 1, cont.shape) * np.sqrt(cont)
    partner = np.arange(F) ^ 1 if F > 1 else np.zeros(1, int)
    t_sky = t_obj[partner] if F > 1 else t_obj + 330.0 / 86400.0
    skyset = SpectrumSet(kind="sky", lam=lam, flux=sky, time=t_sky, nod=nod,
                         epoch=[f"{name}S{i}" for i in range(F)])
    objset = SpectrumSet(kind="obj", lam=lam, flux=obj, time=t_obj, nod=nod,
                         epoch=[f"{name}O{i}" for i in range(F)],
                         shift_px=shift_px if shift else None)
    return NightSpectra(sky=skyset, obj=objset, template=T, mode="HIRES-Y", name=name)


def _drift(t, t0, amp=0.6):
    th = (t - t0) * 24.0
    return amp * np.sin(th / 0.9) + 0.2 * th


@pytest.fixture(scope="module")
def ref_night():
    """A 16-frame night with static line offsets: its series solution and the surface."""
    rng = np.random.default_rng(11)
    t = 60500.0 + np.arange(16) * (330.0 / 86400.0)
    zs = 0.30 + _drift(t, t.mean())
    zt = -0.20 + 0.9 * _drift(t, t.mean())
    night = _night(rng, t, zs, zt, name="ref")
    cfg = AnchorConfig.for_mode("HIRES-Y")
    sol = anchor_night(night, cfg)
    surf = build_static_surface([sol], calib="synthetic")
    return night, zs, zt, sol, surf, cfg


# ------------------------------------------------------------------- building blocks
def test_epoch_groups_and_choose_method():
    t = 60000.0 + np.r_[0.0, 0.01, 0.02, 1.0, 1.01, 5.3] [[3, 0, 5, 1, 4, 2]]
    g = epoch_groups(t, 3.0)
    assert list(g) == [1, 0, 2, 0, 1, 0]
    T = _template()
    lam = {163: np.linspace(1.0, 2.0, 10)}

    def nt(times):
        s = SpectrumSet(kind="sky", lam=lam, flux={163: np.zeros((len(times), 10))},
                        time=times, nod=["A"] * len(times))
        return NightSpectra(sky=s, obj=None, template=T)
    assert choose_method(nt([60000.0])) == "single"
    assert choose_method(nt(60000.0 + np.arange(12) * 0.004)) == "series"
    assert choose_method(nt(60000.0 + np.arange(4) * 0.004)) == "sparse"       # too few
    assert choose_method(nt(60000.0 + np.arange(12) * 0.0002)) == "sparse"     # too short
    assert choose_method(nt(60000.0 + np.r_[np.arange(10) * 0.004, 3.0])) == "sparse"  # 2 nights


def test_anchored_zero_points_single_frame_and_groups():
    rng = np.random.default_rng(5)
    K = 120
    mu = rng.normal(0, 0.2, K)                    # true static line offsets
    e = rng.uniform(0.02, 0.2, K)
    # one frame, prior = the true offsets: z recovered with e-weighted precision
    zt = 0.731
    d = zt + mu + 2.5 * e * rng.normal(size=K)    # scatter 2.5 x the formal error
    r = anchored_zero_points(d[:, None], e[:, None], prior_mu=mu, prior_var=np.full(K, 1e-6))
    w = 1 / e ** 2
    assert abs(r["z"][0] - zt) < 4 * r["z_err"][0]
    assert abs(r["z_err"][0] / (2.5 / np.sqrt(w.sum())) - 1) < 0.4        # honest + efficient
    assert abs(r["lam_group"][0] - 2.5) < 0.7
    # no prior: the line offsets become noise (weights 1 / (lam^2 e^2 + s^2)), larger error
    r0 = anchored_zero_points(d[:, None], e[:, None])
    assert abs(r0["z"][0] - zt) < 4 * r0["z_err"][0]
    assert r0["z_err"][0] > 2 * r["z_err"][0]
    # sparse epochs (3 groups x 3 frames), different line subsets per epoch, self prior: every
    # epoch on one scale (the information-weighted mean of mu = 0 gauge)
    G, Fg = 3, 3
    z_true = np.array([0.2, 0.25, 0.22, -1.1, -1.05, -1.0, 2.0, 2.1, 2.05])
    groups = np.repeat(np.arange(G), Fg)
    E = rng.uniform(0.02, 0.2, (K, G * Fg))
    D = z_true[None, :] + mu[:, None] + 0.03 * rng.normal(size=(K, G)).repeat(Fg, 1) + 2.0 * E * rng.normal(
        size=E.shape)
    keep = np.ones((K, G * Fg), bool)
    for g in range(G):                             # each epoch sees 70 % of the lines
        keep[rng.random(K) < 0.3, g * Fg:(g + 1) * Fg] = False
    D[~keep] = np.nan
    rs = anchored_zero_points(D, E, groups=groups, self_prior=True)
    off = rs["z"] - z_true
    assert np.std(off) < 0.03 and np.all(np.abs(off - off.mean()) < 4 * rs["z_err"])
    assert rs["t_line"] < 0.1
    # the table prior removes the gauge offset (mean mu) entirely
    rt = anchored_zero_points(D, E, groups=groups, prior_mu=mu, prior_var=np.full(K, 0.03 ** 2))
    assert np.all(np.abs(rt["z"] - z_true) < 4 * rt["z_err"])


# --------------------------------------------------------------- surface persistence
def _subset_night(night, idx):
    from decanter.wavecal.model import _subset

    return NightSpectra(sky=_subset(night.sky, idx), obj=_subset(night.obj, idx),
                        template=night.template, mode=night.mode, name=night.name)


def test_static_surface_save_load_find(ref_night, tmp_path):
    night, _zs, _zt, sol, surf, _cfg = ref_night
    assert surf.matches(night.template) and surf.nights == ("ref",)
    assert surf.sky["gauge"] == "c00" and surf.sky["coef"][0, 0] == 0.0
    assert surf.star["gauge"] == "mean" and surf.star["slit"]["n"] == 1
    p = surf.save(tmp_path)
    assert p.name == SURFACE_FILENAME and json.loads(p.read_text())["version"] == SURFACE_VERSION
    assert surf.star["scale"]["kind"] == "set" and surf.sky["scale"]["kind"] == "set"
    s2 = StaticSurface.load(tmp_path)
    for k in ("sky", "star"):
        a, b = surf.part(k), s2.part(k)
        assert np.allclose(a["coef"], b["coef"]) and a["tau"] == b["tau"]
        for c in ("order", "lref", "mu", "mu_err"):
            assert np.allclose(np.asarray(a["lines"][c], float), np.asarray(b["lines"][c], float),
                               equal_nan=True)
    lam = night.obj.lam[ORDERS[1]]
    assert np.allclose(surf.surface_kms("sky", night.template, ORDERS[1], lam),
                       s2.surface_kms("sky", night.template, ORDERS[1], lam))
    assert StaticSurface.find(tmp_path, night.template) is not None
    # the reloaded surface (shape, line table, A/B pattern) anchors a frame identically
    one = _subset_night(night, [3])
    a = anchor_frames(one, _cfg, method="single", surface=surf)
    b = anchor_frames(one, _cfg, method="single", surface=s2)
    for k in ("z_sky", "z_sky_err", "z_star", "z_star_err"):
        assert np.allclose(getattr(a, k), getattr(b, k), rtol=0, atol=1e-12)
    other = _template(dc0=0.01)
    with pytest.warns(UserWarning, match="template"):
        assert StaticSurface.find(tmp_path, other) is None
    # the stored sky surface reproduces the injected smooth departure (0.10 xn + 0.06 xn mn
    # + 0.04 mn; the constant goes into the zero point) to the precision ~50 lines with
    # 0.05 km/s line-list errors allow
    T = night.template
    sk, true = [], []
    for m in ORDERS:
        x = np.linspace(200.0, 3900.0, 20)
        sk.append(surf.surface_kms("sky", T, m, T.lam(m, x)))
        true.append(_static_offset(T.lam(m, x), m, T, smooth=(0.10, 0.06, 0.04), rms=0.0, seed=1))
    sk, true = np.concatenate(sk), np.concatenate(true)
    dlt = sk - true
    assert np.corrcoef(sk, true)[0, 1] > 0.8 and np.std(dlt) < 0.04
    import copy

    s_b = copy.copy(sol)
    s_b.template = other
    with pytest.raises(ValueError, match="template"):
        build_static_surface([sol, s_b])


# ------------------------------------------------------------------ synthetic recovery
def test_single_frame_recovers_injected_zero_point(ref_night):
    _n, zs_ref, _zt, sol_ref, surf, cfg = ref_night
    gauge_sky = np.mean(sol_ref.z_sky - zs_ref)       # surface scale vs the injected truth
    gauge_star = np.mean(sol_ref.z_star - _zt)
    rng = np.random.default_rng(21)
    t = np.array([60503.25])
    zs, zt = np.array([1.37]), np.array([0.62])
    night = _night(rng, t, zs, zt, name="one")
    assert choose_method(night, cfg) == "single"
    s0 = anchor(night, cfg)                           # no surface: line-list gauge
    assert s0.method == "single" and s0.oh_model is None
    assert np.isfinite(s0.z_sky_err[0]) and 0 < s0.z_sky_err[0] < 0.06
    assert abs(s0.z_sky[0] - zs[0]) < 3 * s0.z_sky_err[0] + 0.03
    assert abs(s0.z_star[0] - zt[0]) < 3 * s0.z_star_err[0] + 0.03
    s1 = anchor(night, cfg, surface=surf)             # with the surface: its absolute scale
    assert s1.diagnostics["oh_n_matched"] > 10 and s1.diagnostics["tell_n_matched"] > 10
    e_sky = np.hypot(s1.z_sky_err[0], np.std(sol_ref.z_sky - zs_ref))
    e_star = np.hypot(s1.z_star_err[0], np.std(sol_ref.z_star - _zt))
    assert abs(s1.z_sky[0] - zs[0] - gauge_sky) < 3 * e_sky
    assert abs(s1.z_star[0] - zt[0] - gauge_star) < 3 * e_star
    assert s1.z_star_err[0] < 0.05 and s1.z_inst[0] == s1.z_sky[0]
    # z_inst = the sky partner's OH zero point (330 s later) + drift over |dt| in quadrature
    dt_h = abs(night.obj.time[0] - night.sky.time[0]) * 24.0
    assert abs(s1.z_inst_err[0] - np.hypot(s1.z_sky_err[0], cfg.drift_rate_kms_h * dt_h)) < 1e-12
    # labels: star = raw label / (1 + (z + S_star(x, m) + nod/drift pattern) / c); the sky
    # uses the stored sky shape + its pattern (none injected here: the patterns stay small)
    m = ORDERS[2]
    lam = night.obj.lam[m]
    raw = night.obj.raw_lambda(m, night.template)
    pst = s1.modes_kms(m, lam, "star")
    assert s1.star_modes is not None and np.max(np.abs(pst)) < 0.1
    zz = s1.z_star[:, None] + surf.surface_kms("star", night.template, m, lam)[None, :] + pst
    assert np.allclose(s1.true_lambda(m, lam, "star"), raw / (1 + zz / C_KMS), atol=1e-9, rtol=0)
    zk = (s1.z_sky[:, None] + surf.surface_kms("sky", night.template, m, lam)[None, :]
          + s1.modes_kms(m, lam, "sky"))
    assert np.allclose(s1.true_lambda(m, lam, "sky"), lam / (1 + zk / C_KMS), atol=1e-9, rtol=0)


def test_sparse_epochs_share_one_scale(ref_night):
    """Three epochs (days apart, 2 frames each, different S/N and zero points) anchored with
    the surface of another night recover their per-epoch offsets on the surface's scale."""
    _n, zs_ref, zt_ref, sol_ref, surf, cfg = ref_night
    g_sky = np.mean(sol_ref.z_sky - zs_ref)
    g_star = np.mean(sol_ref.z_star - zt_ref)
    s_ref = np.std(sol_ref.z_star - zt_ref)
    rng = np.random.default_rng(31)
    t = np.concatenate([60501.1 + np.arange(2) * 0.004, 60505.2 + np.arange(2) * 0.004,
                        60509.3 + np.arange(2) * 0.004])
    zs = np.repeat([0.9, -0.5, 1.6], 2) + rng.normal(0, 0.02, 6)
    zt = np.repeat([0.3, -0.9, 1.1], 2) + rng.normal(0, 0.02, 6)
    parts = []
    for k, snr in enumerate((1.0, 0.35, 2.0)):
        sl = slice(2 * k, 2 * k + 2)
        parts.append(_night(rng, t[sl], zs[sl], zt[sl], snr=snr, name=f"e{k}"))
    # one NightSpectra holding the three epochs (frames concatenated)
    def cat(sets):
        return SpectrumSet(kind=sets[0].kind, lam=sets[0].lam,
                           flux={m: np.concatenate([s.flux[m] for s in sets]) for m in sets[0].lam},
                           time=np.concatenate([s.time for s in sets]),
                           nod=np.concatenate([s.nod for s in sets]),
                           epoch=np.concatenate([s.epoch for s in sets]),
                           shift_px=np.concatenate([s.shift_px for s in sets]) if sets[0].shift_px is not None else None)
    night = NightSpectra(sky=cat([p.sky for p in parts]), obj=cat([p.obj for p in parts]),
                         template=parts[0].template, mode="HIRES-Y", name="sparse")
    assert choose_method(night, cfg) == "sparse"
    s = anchor(night, cfg, surface=surf)
    assert s.method == "sparse" and s.diagnostics["n_groups"] == 3
    r_sky = s.z_sky - zs - g_sky
    r_star = s.z_star - zt - g_star
    assert np.all(np.abs(r_sky) < 3 * np.hypot(s.z_sky_err, 0.01) + 0.01), r_sky
    assert np.all(np.abs(r_star) < 3 * np.hypot(s.z_star_err, s_ref) + 0.01), r_star
    # per-epoch means agree (one absolute scale across the epochs)
    ep = np.repeat(np.arange(3), 2)
    m_star = np.array([np.mean(r_star[ep == k]) for k in range(3)])
    assert np.ptp(m_star) < 0.06
    # without the surface the epochs still share a scale (self prior), on its own gauge
    s_self = anchor(night, cfg)
    rr = s_self.z_star - zt
    assert np.ptp([np.mean(rr[ep == k]) for k in range(3)]) < 0.08


def test_series_path_dispatch_bit_identical():
    rng = np.random.default_rng(3)
    night, _zs, _zt = _synthetic_night(rng)
    cfg = AnchorConfig.for_mode("HIRES-Y", sky_shape=None)
    assert choose_method(night, cfg) == "series"
    a = anchor_night(night, cfg)
    with warnings.catch_warnings():
        # no surface warning on the series path (the mode guard may warn: this toy night's 8
        # telluric lines all sit in one order, so its (2, 2) star modes step down to (2, 1))
        warnings.filterwarnings("error", message=".*surface.*")
        b = anchor(night, cfg)
    assert b.method == "series" and b.star_shape is None
    for k in ("z_star", "z_star_err", "z_sky", "z_sky_err", "z_inst", "z_inst_err"):
        assert np.array_equal(getattr(a, k), getattr(b, k))
    m = 163
    lam = night.obj.lam[m]
    for kind in ("star", "sky", "inst"):
        assert np.array_equal(a.true_lambda(m, lam, kind), b.true_lambda(m, lam, kind))
    # forcing "single" on the same night anchors every frame on its own, near the series values
    s = anchor_frames(night, cfg, method="single")
    assert s.diagnostics["n_groups"] == night.obj.n_frames
    d = s.z_sky - a.oh["z"]
    assert np.sqrt(np.mean(d ** 2)) < 3 * np.median(s.z_sky_err)


def test_surface_nod_pattern_and_rejection():
    """An A/B slit pattern (0.3 km/s x normalised pixel x nod code) measured by a series night
    goes into the surface and is applied to a single frame of another night; a corrupted sky
    table is detected and not used."""
    rng = np.random.default_rng(41)
    pat = (0.3, 0.3)
    t = 60600.0 + np.arange(32) * (330.0 / 86400.0)      # a full-ish night: nod and drift
    zs = 0.2 + _drift(t, t.mean(), 0.6)                   # columns of the modes separate
    zt = -0.1 + 0.9 * _drift(t, t.mean(), 0.6)
    ref = _night(rng, t, zs, zt, name="pref", nod_pattern=pat)
    cfg = AnchorConfig.for_mode("HIRES-Y")
    sol = anchor_night(ref, cfg)
    surf = build_static_surface([sol])
    md = surf.star["modes"]
    assert md is not None and np.max(md["nod_shrink"]) > 0.5           # a real pattern survives
    assert abs(md["nod"][1] - pat[1]) < 0.06                            # T1(x) T0(m) = xn term
    T = ref.template
    for nod0 in ("A", "B"):
        one = _night(rng, np.array([60605.0]), np.array([0.8]), np.array([0.4]), name="p1",
                     nod_pattern=pat, nod0=nod0)
        a = anchor(one, cfg, surface=surf)
        b = anchor(one, AnchorConfig.for_mode("HIRES-Y", use_surface_modes=False), surface=surf)
        # the stored pattern is applied with the accepted surface on EVERY frame (no per-frame
        # residual vote, which toggled it on ~60 % of real frames and jumped the labels)
        assert a.diagnostics["tell_pattern_used"] is True
        assert np.isfinite(a.diagnostics["tell_pattern_dresid"])
        assert b.diagnostics["tell_pattern_used"] is None
        nodc = 0.5 if nod0 == "A" else -0.5
        err_a, err_b = [], []
        for m in ORDERS:
            x = np.linspace(300.0, 3800.0, 12)
            lam = T.lam(m, x)
            truth = (nodc * pat[1] * (2 * x - 4096.0) / 4094.0  # A/B pattern + static shape
                     + _static_offset(lam, m, T, smooth=(-0.08, 0.05, 0.03), rms=0.0, seed=2))
            for s_, out in ((a, err_a), (b, err_b)):
                v = s_.modes_kms(m, lam, "star")[0] + _surface_kms_star(s_, T, m, lam)
                v = v + s_.z_star[0]
                out.append(v - truth)
        ra, rb = np.concatenate(err_a), np.concatenate(err_b)
        # up to a constant (the zero point convention), the labels follow the A/B pattern
        # (star: 67 telluric lines, a 16-frame build night -> ~0.03 km/s pattern/shape error)
        assert np.std(ra - ra.mean()) < 0.6 * np.std(rb - rb.mean())
        assert np.std(ra - ra.mean()) < 0.04
        # sky (104 OH lines): the stored pattern is close to the injected one
        es = []
        for m in ORDERS:
            x = np.linspace(300.0, 3800.0, 12)
            lam = T.lam(m, x)
            truth = (nodc * pat[0] * (2 * x - 4096.0) / 4094.0
                     + _static_offset(lam, m, T, smooth=(0.10, 0.06, 0.04), rms=0.0, seed=1))
            v = a.z_sky[0] + a.shape_kms(m, lam) + a.modes_kms(m, lam, "sky")[0]
            es.append(v - truth)
        es = np.concatenate(es)
        assert np.std(es - es.mean()) < 0.03
    # corrupted sky table: the prior makes the residuals worse -> rejected, == no surface
    bad = StaticSurface.load(surf.save(Path_tmp()))
    rng2 = np.random.default_rng(1)
    bad.sky["lines"]["mu"] = bad.sky["lines"]["mu"] + rng2.normal(0, 1.5, bad.sky["lines"]["mu"].size)
    one = _night(rng, np.array([60606.0]), np.array([0.5]), np.array([0.2]), name="p2",
                 nod_pattern=pat)
    with pytest.warns(UserWarning, match="does not describe"):
        s_bad = anchor(one, cfg, surface=bad)
    s_none = anchor(one, cfg)
    assert s_bad.diagnostics["oh_surface_rejected"] and s_bad.shape is None
    assert s_bad.diagnostics["oh_pattern_used"] is None          # rejected part: no pattern
    assert np.allclose(s_bad.z_sky, s_none.z_sky)


def _surface_kms_star(sol, T, m, lam):
    from decanter.wavecal.model import _surface_kms

    return np.zeros(lam.size) if sol.star_shape is None else _surface_kms(sol.star_shape, T, m, lam)


def Path_tmp():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp())


def test_anchor_reduction_single_frame_dispatch():
    """decanter.reduce-style output (one Reduction with obj/sky OrderSpectrum dicts and raw
    headers) goes through the single-image path and matches anchor_frames on the same data."""
    from types import SimpleNamespace

    from decanter.wavecal.model import anchor_reduction, night_from_reductions

    rng = np.random.default_rng(51)
    # 2025-08-08 01:00:00 UT start, 300 s: mid-exposure MJD
    t_mid = 60895.0 + (3600.0 + 150.0) / 86400.0
    night = _night(rng, np.array([t_mid]), np.array([0.9]), np.array([0.5]), shift=False,
                   name="red")

    def spectra(sset):
        out = {}
        for m in ORDERS:
            L = sset.lam[m]
            out[(1.30, m)] = SimpleNamespace(flux=np.asarray(sset.flux[m][0]), crval1=float(L[0]),
                                             cdelt1=float(L[1] - L[0]), crpix1=1.0)
        return out

    hdr = {"DATE-OBS": "2025-08-08", "UT-STR": "01:00:00.0", "UT-END": "01:05:00.0",
           "EXPTIME": 300.0, "NODPOS": "A1", "FRAMEID": "WINA1", "INSTMODE": "HIRES-Y"}
    shdr = dict(hdr, **{"UT-STR": "01:05:30.0", "UT-END": "01:10:30.0", "FRAMEID": "WINA2",
                        "NODPOS": "B1"})
    red = SimpleNamespace(obj=spectra(night.obj), sky=spectra(night.sky), obj_header=hdr,
                          sky_header=shdr, shift_wave=0.0)
    sol = anchor_reduction(red, night.template, surface=None)
    assert sol.method == "single" and sol.z_star.shape == (1,)
    nr = night_from_reductions([red], night.template)
    ref = anchor_frames(nr, AnchorConfig.for_mode("HIRES-Y"), method="single")
    assert np.allclose(sol.z_star, ref.z_star, atol=1e-12) and np.allclose(sol.z_sky, ref.z_sky,
                                                                         atol=1e-12)
    assert abs(sol.z_sky[0] - 0.9) < 3 * sol.z_sky_err[0] + 0.03
    assert abs(sol.z_star[0] - 0.5) < 3 * sol.z_star_err[0] + 0.03


@pytest.mark.regression
def test_surface_from_warp_night_and_single_frame(tmp_path):
    """Real data: a surface built from the TOI-2109 WARP night, saved / reloaded, anchors one
    frame of the night with the series labels to < 0.05 km/s (grid mean) and honest errors."""
    from decanter._localpaths import WINERED_ROOT
    from decanter.wavecal.model import load_warp_night, surface_from_warp_reductions

    root = WINERED_ROOT / "reductions" / "TOI2109"
    if not (root / "TOI2109_NO1").is_dir():
        pytest.skip("TOI2109 WARP reduction not on this machine")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surf = surface_from_warp_reductions([root], calib="TOI2109", save_to=tmp_path)
        s2 = StaticSurface.load(tmp_path)
        night = load_warp_night(root)
        ser = anchor_night(night)
        one = _subset_night(night, [10])
        a = anchor(one, AnchorConfig.for_mode(night.mode), surface=s2)
    assert a.method == "single" and surf.nights == ("TOI2109",)
    assert not a.diagnostics["oh_surface_rejected"] and not a.diagnostics["tell_surface_rejected"]
    for kind in ("sky", "star"):
        dv = []
        for m in (160, 163, 170, 181):
            lam = night.template.lam(m, np.linspace(400.0, 3700.0, 9))
            dv.append(C_KMS * (ser.true_lambda(m, lam, kind)[10] / a.true_lambda(m, lam, kind)[0]
                               - 1.0))
        dv = np.concatenate(dv)
        assert abs(dv.mean()) < 0.05, (kind, dv.mean())
        e = getattr(a, f"z_{kind}_err")[0]
        assert 0.005 < e < 0.05
