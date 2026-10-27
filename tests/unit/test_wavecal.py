"""decanter.wavecal: line table, ThAr template, centroid fitter, robust zero points,
GP time model, single export resample, labels (s10 undo) and an end-to-end synthetic night.

Real-data tests use the WARP reductions under ``$WINERED_ROOT/reductions`` and skip when absent;
the full-night anchoring check is marked ``regression`` (slow).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from decanter._localpaths import WINERED_ROOT  # noqa: E402
from decanter.wavecal.model import (  # noqa: E402
    C_KMS,
    AnchorConfig,
    EcSolution,
    LineTable,
    NightSpectra,
    SpectrumSet,
    ThArTemplate,
    anchor_night,
    build_absorber_features,
    build_oh_features,
    export_resample,
    fit_line_centroids,
    fit_time_model,
    load_warp_night,
    robust_zero_points,
    static_lnlam_grid,
)

RED = WINERED_ROOT / "reductions"
# m163-like id series (LCO25b), and two neighbours with the same shape
COEF = {
    163: [10831.427256, -70.816251, -6.252989, 0.210139, 0.116919, 0.047878],
    166: [10635.6, -69.5, -6.13, 0.206, 0.115, 0.047],
    170: [10385.2, -67.9, -5.99, 0.201, 0.112, 0.046],
}


def _template(orders=(163, 166, 170)):
    return ThArTemplate.from_arrays(orders, [COEF[m] for m in orders], 1.0, 4095.0, kind=1,
                                    dy=0.5)


# ------------------------------------------------------------------------------ line table
def test_line_table_contents():
    lt = LineTable.load()
    oh, ab = lt.oh, lt.absorbers
    assert oh["lam"].size > 2000 and ab["nu0"].size > 1000
    assert np.all(np.diff(oh["lam"]) >= 0)
    assert np.all((oh["lam"] > 8900) & (oh["lam"] < 13800))
    # Lambda doublets: almost every group has exactly two parity components
    _, counts = np.unique(oh["group"], return_counts=True)
    assert np.mean(counts == 2) > 0.95
    # HITRAN vs Bernath & Colin (Oliva+2015) positions: tens of m/s
    m = np.isfinite(oh["lam_oliva"])
    dv = (oh["lam"][m] / oh["lam_oliva"][m] - 1) * C_KMS
    assert m.sum() > 300 and abs(np.median(dv)) < 0.05 and np.all(np.abs(dv) < 0.5)
    # pressure shift: delta_air < 0 for (almost) all H2O lines -> lines move redward with P
    a0 = lt.absorber_lines(10000, 11000, species=("H2O",), p_eff={"H2O": 0.0})
    a1 = lt.absorber_lines(10000, 11000, species=("H2O",), p_eff={"H2O": 0.6})
    n = min(a0["lam"].size, a1["lam"].size)
    assert n > 50 and np.median(a1["lam"][:n] - a0["lam"][:n]) > 0
    assert set(np.unique(ab["species"])) <= {"H2O", "O2"}


def test_oh_features_lambda_doublets():
    lt = LineTable.load()
    fs = build_oh_features(lt, 10400.0, 10900.0, 60000.0)
    assert len(fs) > 50
    clean = [f for f in fs if f["clean"]]
    assert len(clean) > 20
    # the bright features are Lambda doublets (blended or partly resolved) fitted as one
    # template with fixed relative positions; singles are mostly high-J, widely split pairs
    inten = np.array([f["Irel"] for f in clean])
    dbl = np.array([f["nmain"] >= 2 for f in clean])
    assert np.sum(inten * dbl) / np.sum(inten) > 0.85
    assert np.mean(dbl[inten >= 1e-2]) > 0.8
    for f in clean:
        assert f["umin"] < 0 < f["umax"]
        assert abs(np.sum(f["cr"][: f["nmain"]]) - 1.0) < 1e-9
    ab = build_absorber_features(lt, 10400.0, 10900.0, 60000.0, species=("H2O",))
    assert len(ab) > 5
    lam = np.array([f["lref"] for f in ab])
    assert np.all(np.diff(lam) / lam[1:] * C_KMS > 2.5 * C_KMS / 60000.0)


# --------------------------------------------------------------------------- templates
def test_template_inverse_and_derivative():
    T = _template()
    p = np.linspace(1.0, 4095.0, 101)
    for m in T.orders:
        lam = T.lam(m, p)
        assert np.max(np.abs(T.p_of_lam(m, lam) - p)) < 1e-7
        h = 1e-3
        fd = (T.lam(m, p + h) - T.lam(m, p - h)) / (2 * h)
        assert np.max(np.abs(fd - T.dlam_dp(m, p))) < 1e-7
        assert np.max(np.abs(T.lam_xp(np, m, p) - lam)) < 1e-9
    jnp = jax.numpy
    from decanter.wavecal._xp import get_backend

    get_backend("jax", "float64")
    lj = np.asarray(T.lam_xp(jnp, 163, jnp.asarray(p)))
    assert np.max(np.abs(lj - T.lam(163, p))) < 1e-9
    g = np.asarray(jax.vmap(jax.grad(lambda q: T.lam_xp(jnp, 163, q)))(jnp.asarray(p)))
    assert np.max(np.abs(g - T.dlam_dp(163, p))) < 1e-9
    grid = static_lnlam_grid(T, 163, dv_kms=1.5)
    assert grid.dtype == np.float64
    assert np.allclose(np.diff(np.log(grid)) * C_KMS, 1.5, rtol=1e-9)


@pytest.mark.skipif(not (RED / "TOI2109" / "calibration_data").is_dir(),
                    reason="TOI2109 WARP reduction not on this machine")
def test_ec_matches_id_on_warp_reduction():
    T = ThArTemplate.from_warp_reduction(RED / "TOI2109")
    ec = EcSolution.from_file(next((RED / "TOI2109" / "calibration_data" / "database").glob(
        "eccomp_*_fm_ecall")))
    p = np.linspace(1.0, 4095.0, 64)
    for m in (159, 163, 184):
        dv = (ec.lam(m, p) / T.lam(m, p) - 1) * C_KMS
        assert np.max(np.abs(dv)) < 1e-3


# ------------------------------------------------------------------------ centroid fitter
def test_fit_line_centroids_unbiased_and_calibrated():
    rng = np.random.default_rng(0)
    fw = C_KMS / 60000.0
    sig = fw / 2.3548
    cu = np.array([-1.6, 1.6, 7.0])           # a Lambda doublet + a weak contaminant
    cr = np.array([0.5, 0.5, 0.05])
    N = 600
    u = np.arange(-14.0, 14.0, 0.95)
    true = rng.uniform(-1.0, 1.0, N)
    amp = 200.0
    model = 30.0 + 0.2 * u + amp * np.exp(
        -0.5 * ((u[None, :, None] - cu[None, None, :] - true[:, None, None]) / sig) ** 2) @ cr
    noise = 3.0
    Y = model + rng.normal(0, noise, model.shape)
    U = np.broadcast_to(u, Y.shape)
    W = np.ones_like(Y)
    for fit_sigma in (True, False):
        r = fit_line_centroids(U, Y, W, np.broadcast_to(cu, (N, 3)), np.broadcast_to(cr, (N, 3)),
                               np.full(N, sig * (1.1 if fit_sigma else 1.0)), np.zeros(N),
                               np.full(N, 4.0), fit_sigma=fit_sigma)
        pull = (r["d"] - true) / r["e_d"]
        assert abs(np.mean(r["d"] - true)) < 3 * np.std(r["d"] - true) / math.sqrt(N)
        assert 0.8 < np.std(pull) < 1.25
        assert np.all(r["amp"] > 0) and np.median(r["snr"]) > 20
        if fit_sigma:
            assert abs(np.median(r["sigma"]) / sig - 1) < 0.02


def test_robust_zero_points_with_outliers():
    rng = np.random.default_rng(1)
    K, F = 60, 40
    mu = rng.normal(0, 0.2, K)
    mu -= mu.mean()
    z = np.cumsum(rng.normal(0, 0.1, F)) + 0.3
    e = rng.uniform(0.05, 0.15, (K, F))
    d = mu[:, None] + z[None, :] + rng.normal(0, 1, (K, F)) * e
    bad = rng.random((K, F)) < 0.05
    d[bad] += rng.choice([-3, 3], bad.sum())
    d[rng.random((K, F)) < 0.05] = np.nan
    r = robust_zero_points(d, e)
    off = np.mean(r["z"] - z)
    assert abs(off) < 0.02                     # gauge: information-weighted mean of mu = 0
    res = r["z"] - z - off
    assert np.sqrt(np.mean(res ** 2)) < 0.035
    assert 0.6 < np.std(res / r["z_err"]) < 1.6
    assert np.max(np.abs(r["mu"] - off * 0 - mu)) < 0.15


def test_time_model_nod_drift_outlier():
    rng = np.random.default_rng(2)
    n = 48
    t = 60000.0 + np.sort(rng.uniform(0, 0.25, n))
    th = (t - t.mean()) * 24
    nod = np.where((np.arange(n) // 2) % 2 == 0, 0.5, -0.5)
    truth = 1.2 * np.sin(th / 1.7) + 0.4 * th
    sig = 0.03
    y = 0.5 + truth + 0.12 * nod + rng.normal(0, sig, n)
    y[7] += 1.0                                  # one bad frame
    H = np.column_stack([np.ones(n), nod])
    tm = fit_time_model(t, y, np.full(n, sig ** 2), H, b_mean=[0.0, 0.0],
                        b_cov=np.diag([20.0 ** 2, 2.0 ** 2]))
    b, B = tm.fixed_effects()
    assert abs(b[1] - 0.12) < 3 * math.sqrt(B[1, 1]) + 0.01
    assert tm.weights[7] < 0.2 and np.median(tm.weights) == 1.0
    sm, _sd = tm.predict(t, np.column_stack([np.ones(n), np.zeros(n)]), jitter=False)
    err = sm - 0.5 - truth
    keep = np.arange(n) != 7
    assert np.sqrt(np.mean((err[keep] - np.mean(err[keep])) ** 2)) < 0.03


# ----------------------------------------------------------------------- export resample
def test_export_resample_lanczos():
    n = 600
    lam = 10800.0 * np.exp(np.arange(n) * 1.0 / C_KMS)       # 1 km/s per sample
    k = 2 * np.pi / 9.0
    f_true = lambda lt: 1.0 + 0.3 * np.sin(k * np.log(lt / 10800.0) * C_KMS)  # noqa: E731
    u = np.array([0.3, -0.7, 1.4]) / C_KMS
    flux = np.array([f_true(lam * np.exp(uu)) for uu in u])  # samples sit at lam * exp(u)
    out = export_resample(flux, lam, u, lam)
    good = np.isfinite(out)
    assert good[:, 20:-20].all()
    assert np.max(np.abs(out[good] - np.broadcast_to(f_true(lam), out.shape)[good])) < 3e-3
    c = export_resample(np.ones((3, n)), lam, u, lam)
    assert np.allclose(c[np.isfinite(c)], 1.0, atol=1e-12)
    assert np.isnan(out[1, -1]) or np.isnan(out[1, 0])        # edges outside the data
    # per-sample label arrays give the same answer
    out2 = export_resample(flux, lam, np.broadcast_to(u[:, None], flux.shape), lam)
    assert np.allclose(out2[good], out[good])


# ------------------------------------------------------------------- synthetic night
def _synthetic_night(rng, *, F=24, shift=True):
    T = _template()
    lt = LineTable.load()
    fw = C_KMS / 60000.0
    sig = fw / 2.3548
    t0 = 60500.0
    t_obj = t0 + np.arange(F) * (330.0 / 86400.0)
    partner = np.arange(F) ^ 1
    t_sky = t_obj[partner]
    nod = np.array(["A", "B", "B", "A"] * (F // 4 + 1))[:F]
    th = lambda t: (t - t_obj.mean()) * 24  # noqa: E731
    drift = lambda t: 0.9 * np.sin(th(t) / 0.9) + 0.25 * th(t)  # noqa: E731
    nodc = np.where(nod == "A", 0.5, -0.5)
    z_sky = 0.15 + drift(t_sky) + 0.06 * nodc          # aperture = obj nod
    z_star = -0.10 + 0.8 * drift(t_obj) + 0.03 * nodc
    lam, sky, obj = {}, {}, {}
    shift_px = (rng.normal(0, 1.0, F) if shift else np.zeros(F))
    for m in T.orders:
        lo, hi = sorted(T.lam(m, np.array([4095.0, 1.0])))
        L = np.linspace(lo, hi, 4400)
        lam[m] = L
        ol = lt.oh_lines(lo - 5, hi + 5)
        S = np.zeros((F, L.size))
        for lc, ii in zip(ol["lam"], ol["I"], strict=True):
            c = lc * (1 + z_sky / C_KMS)
            S += 4.0e4 * ii * np.exp(-0.5 * (((L[None, :] / c[:, None]) - 1) * C_KMS / sig) ** 2)
        S += 40.0
        sky[m] = S + rng.normal(0, 1, S.shape) * np.sqrt(S)
        p = T.p_of_lam(m, L)
        raw = T.lam(m, p[None, :] - shift_px[:, None])       # s10-shifted labels
        al = lt.absorber_lines(lo - 5, hi + 5, species=("H2O",), R=60000.0)
        tau = np.zeros((F, L.size))
        for lc, dep in zip(al["lam"], al["depth"], strict=True):
            c = lc * (1 + z_star / C_KMS)
            tau += -np.log1p(-min(dep, 0.9)) * np.exp(
                -0.5 * (((raw / c[:, None]) - 1) * C_KMS / sig) ** 2)
        cont = 3000.0 * np.exp(-tau)
        obj[m] = cont + rng.normal(0, 1, cont.shape) * np.sqrt(cont)
    skyset = SpectrumSet(kind="sky", lam=lam, flux=sky, time=t_sky, nod=nod,
                         epoch=[f"F{i}" for i in partner])
    objset = SpectrumSet(kind="obj", lam=lam, flux=obj, time=t_obj, nod=nod,
                         epoch=[f"F{i}" for i in range(F)], shift_px=shift_px if shift else None)
    night = NightSpectra(sky=skyset, obj=objset, template=T, mode="HIRES-Y", name="synthetic")
    return night, z_sky, z_star


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(3)
    night, z_sky, z_star = _synthetic_night(rng)
    cfg = AnchorConfig.for_mode("HIRES-Y", sky_shape=None)
    return night, z_sky, z_star, cfg, anchor_night(night, cfg)


def test_anchor_night_synthetic_recovers_zero_points(synth):
    night, z_sky, z_star, _cfg, sol = synth
    d = sol.diagnostics
    # instrument (OH) zero point: absolute, per frame
    err = sol.z_sky - z_sky
    assert np.sqrt(np.mean(err ** 2)) < 0.03, err
    assert abs(d["oh_nod_A_minus_B"] - 0.06) < 0.03
    # star (telluric) zero point with the s10 shift undone, star drift = 0.8 x sky drift
    errs = sol.z_star - z_star
    assert np.sqrt(np.mean(errs ** 2)) < 0.045, errs
    assert abs(d["star_oh_slope_ols"] - 0.8) < 0.08
    assert d["star_source"] == "tellurics"
    assert np.all(np.isfinite(sol.z_star_err)) and np.all(sol.z_star_err > 0)
    # labels: true wavelength of a labelled sample = raw label / (1 + z/c)
    m = 163
    lam = night.obj.lam[m]
    tl = sol.true_lambda(m, lam, kind="star")
    raw = night.obj.raw_lambda(m, night.template)
    zz = sol.z_star[:, None] + sol.modes_kms(m, lam, kind="star")
    assert np.allclose(tl, raw / (1 + zz / C_KMS), rtol=0, atol=1e-9)
    assert np.max(np.abs(sol.modes_kms(m, lam, kind="star"))) < 0.2   # no spatial drift input
    u = sol.labels(m, lam, kind="star")
    assert u.dtype == np.float32 and u.shape == (night.obj.n_frames, lam.size)
    assert np.allclose(sol.u("star"), -np.log1p(sol.z_star / C_KMS), atol=1e-9)
    # the single export resample aligns telluric lines across frames (observatory frame)
    al = export_resample(night.obj.flux[m], lam, u, lam)
    unal = night.obj.flux[m]
    sl = slice(200, -200)
    norm = lambda A: A[:, sl] / np.nanmedian(A[:, sl], axis=1, keepdims=True)  # noqa: E731
    Ma = np.nanmedian(norm(al), axis=0)
    core = Ma < 0.85                                         # telluric line cores
    ra = (norm(al) / Ma - 1)[:, core]
    ru = (norm(unal) / np.nanmedian(norm(unal), axis=0) - 1)[:, core]
    assert core.sum() > 20 and np.nanstd(ra) < 0.6 * np.nanstd(ru)


def test_anchor_night_blank_frame_and_frame_order(synth):
    """A blank (all-NaN, e.g. aborted) exposure must not seed a +-xcorr_vmax drift (init AND
    mode regressor), and the solution must not depend on the order the frames are given in."""
    import copy

    night, _z_sky, z_star, cfg, sol = synth
    F = night.obj.n_frames
    perm = np.random.default_rng(7).permutation(F)

    def permuted(blank=None):
        n2 = copy.deepcopy(night)
        for s in (n2.sky, n2.obj):
            s.flux = {m: v[perm].copy() for m, v in s.flux.items()}
            s.time, s.nod, s.epoch = s.time[perm], s.nod[perm], s.epoch[perm]
            if s.shift_px is not None:
                s.shift_px = s.shift_px[perm]
            if blank is not None:
                for m in s.flux:
                    s.flux[m][blank] = np.nan
        return n2

    s1 = anchor_night(permuted(), cfg)           # frame order alone: identical solution
    assert np.max(np.abs(s1.z_star - sol.z_star[perm])) < 1e-6
    assert np.max(np.abs(s1.z_sky - sol.z_sky[perm])) < 1e-6
    k = 5                                        # blank frame (position in the permuted night)
    s2 = anchor_night(permuted(blank=k), cfg)
    assert s2.diagnostics["oh_xcorr_bad_frames"] == 1
    drift = s2.oh["drift_xcorr"]
    assert abs(drift[k] - np.median(drift)) < 3.0          # interpolated, not the search edge
    keep = np.arange(F) != k
    e = np.median(sol.z_star_err)
    dz = s2.z_star[keep] - sol.z_star[perm][keep]          # time-model refit: < 1 sigma
    assert np.sqrt(np.mean(dz ** 2)) < 0.5 * e and np.max(np.abs(dz)) < 1.5 * e
    assert np.max(np.abs(s2.z_sky[keep] - sol.z_sky[perm][keep])) < 0.01
    # the blank frame still gets a finite label from the time model
    assert np.isfinite(s2.z_star[k]) and abs(s2.z_star[k] - z_star[perm][k]) < 3 * e


def test_header_times_and_epochs():
    from decanter.wavecal.model import _mjd_mid

    h = {"DATE-OBS": "2025-08-07", "UT-STR": "23:29:51.3", "UT-END": "23:35:11.99",
         "EXPTIME": 300.0, "FRAMEID": "WINA00053571"}
    t = _mjd_mid(h)
    assert abs(t - 60894.980921817136) < 1e-9                 # (UT-STR + UT-END) / 2
    # a corrupt EXPTIME alone changes nothing (UT-END is the true, also aborted, duration)
    for bad in ({"EXPTIME": -1.0}, {"EXPTIME": "garbage"}, {"EXPTIME": 3e7}):
        assert _mjd_mid({**h, **bad}) == t
    # missing / unparseable / placeholder / copied UT-END -> UT-STR + EXPTIME / 2, warned
    t_exp = t - (320.69 - 300.0) / 2 / 86400.0
    for bad in ({"UT-END": "N/A"}, {"UT-END": ""}, {"UT-END": "00:00:00"},
                {"UT-END": "12:00:00"}, {"UT-END": "23:29:51.3"}):
        with pytest.warns(UserWarning, match="UT-END"):
            assert abs(_mjd_mid({**h, **bad}) - t_exp) < 1e-8
    hh = dict(h)
    del hh["UT-END"]
    with pytest.warns(UserWarning):
        assert abs(_mjd_mid(hh) - t_exp) < 1e-8
    # both corrupt: start time, warned; an exposure across UT midnight wraps
    with pytest.warns(UserWarning):
        assert abs(_mjd_mid({**h, "UT-END": "x", "EXPTIME": "x"}) - (t - 160.345 / 86400)) < 1e-8
    t2 = _mjd_mid({**h, "UT-STR": "23:59:50", "UT-END": "00:05:10"})
    assert abs((t2 - math.floor(t2)) * 86400.0 - 150.0) < 1e-3
    # per-frame epoch fallback: a missing FRAMEID must not tie frames to one "exposure"
    lam = {1: np.linspace(1.0, 2.0, 10)}
    s = SpectrumSet(kind="obj", lam=lam, flux={1: np.zeros((3, 10))}, time=[1.0, 1.01, 1.02],
                    nod=["A1", "B1", None], epoch=[None, "WINA1", ""])
    assert len(set(s.epoch)) == 3 and s.epoch[1] == "WINA1" and list(s.nod) == ["A", "B", ""]
    with pytest.raises(ValueError, match="increase"):
        SpectrumSet(kind="obj", lam={1: lam[1][::-1]}, flux={1: np.zeros((3, 10))},
                    time=[1.0, 1.01, 1.02], nod=["A", "B", "A"])
    with pytest.warns(UserWarning, match="float32"):
        SpectrumSet(kind="obj", lam={1: lam[1].astype(np.float32)}, flux={1: np.zeros((3, 10))},
                    time=[1.0, 1.01, 1.02], nod=["A", "B", "A"])
    with pytest.raises(ValueError):
        export_resample(np.ones(10), lam[1], np.zeros(3), lam[1])


# ----------------------------------------------------------------------------- real data
@pytest.mark.skipif(not (RED / "TOI2109" / "TOI2109_NO1").is_dir(),
                    reason="TOI2109 WARP reduction not on this machine")
def test_load_warp_night_layout():
    night = load_warp_night(RED / "TOI2109", orders=(162, 163), frames=range(1, 9))
    assert night.mode == "HIRES-Y" and night.obj.n_frames == 8
    assert night.obj.orders == (162, 163) and night.sky.orders == (162, 163)
    # the sky product of frame i is the partner exposure extracted at frame i's aperture
    assert np.all(night.sky.nod == night.obj.nod)
    assert np.all(night.sky.epoch != night.obj.epoch)
    assert np.all(np.abs(night.sky.time - night.obj.time) < 0.02)
    assert night.obj.shift_px is not None and night.template.dy == 0.5
    lam = night.obj.lam[163]
    assert lam.dtype == np.float64 and np.all(np.diff(lam) > 0)
    raw = night.obj.raw_lambda(163, night.template)
    k = np.argmax(np.abs(night.obj.shift_px))
    dv = np.median((raw[k] / lam - 1) * C_KMS)
    vpp = np.median(np.abs(night.template.velocity_per_pixel(163, night.template.p_of_lam(163, lam))))
    assert abs(abs(dv) - abs(night.obj.shift_px[k]) * vpp) < 0.05 * abs(dv) + 1e-6


@pytest.mark.regression
@pytest.mark.skipif(not (RED / "TOI2109" / "TOI2109_NO1").is_dir(),
                    reason="TOI2109 WARP reduction not on this machine")
def test_anchor_toi2109_against_thar_free_study():
    """Full night: OH absolute zero point (study: +0.105 km/s on the ThAr scale), per-frame
    scatter about the smooth drift + nod (verifier empirical 0.036 km/s), the s10-zeroed frames
    get real values, and the star drift is ~1x the OH drift."""
    night = load_warp_night(RED / "TOI2109")
    sol = anchor_night(night)
    d = sol.diagnostics
    assert abs(d["oh_zp_abs_median"] - 0.105) < 0.06
    assert 2.6 < d["oh_zp_ptp"] < 3.1
    assert d["oh_resid_about_smooth_rms"] < 0.045
    assert 0.9 < d["star_oh_slope_ols"] < 1.12
    zeroed = night.obj.shift_px == 0
    assert zeroed.sum() > 20 and np.all(np.isfinite(sol.z_star[zeroed]))
    assert np.median(sol.z_star_err) < 0.04
