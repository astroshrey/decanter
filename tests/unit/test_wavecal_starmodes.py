"""decanter.wavecal: which detector modes put STARLIGHT on the true scale
(``AnchorConfig.star_modes_source``): "tell" (the telluric-fitted modes, default, unchanged),
"oh" (the OH instrument modes + a per-frame star zero point from the tellurics with the OH
pattern removed) and "tell_sep" (telluric modes with an airmass / time (x, m) term fitted and kept
telluric-only, ``kind="tell"`` labels).  Synthetic checks plus a real-data regression (skips when
the WARP reductions are absent).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from decanter._localpaths import WINERED_ROOT  # noqa: E402
from decanter.wavecal.model import (  # noqa: E402
    C_KMS,
    AnchorConfig,
    LineTable,
    NightSpectra,
    SpectrumSet,
    _subset,
    anchor_night,
)
from tests.unit.test_wavecal_single import _template  # noqa: E402

RED = WINERED_ROOT / "reductions"
GAMMA = 0.30          # km/s per unit airmass x xn: telluric-only chromatic term
KAPPA = 0.15          # km/s per km/s of drift x xn: instrument term (sky, tellurics, starlight)


def _xn(T, m, lam):
    return (2 * T.p_of_lam(m, np.atleast_1d(lam)) - 4096.0) / 4094.0


def _chroma_night(seed=3, F=20, airmass=True):
    """OH sky + H2O object spectra: instrument drift ``z_sky`` with a drift-scaled x-gradient
    (KAPPA, shared by OH and tellurics) and, on the tellurics only, ``GAMMA (A - <A>) xn``."""
    rng = np.random.default_rng(seed)
    T = _template()
    lt = LineTable.load()
    sig = C_KMS / 60000.0 / 2.3548
    t = 60500.0 + np.arange(F) * (330.0 / 86400.0)
    th = (t - t.mean()) * 24.0
    z_sky = 0.3 + 0.8 * np.sin(th / 0.6)
    am = 1.05 + 0.5 * (th / np.ptp(th) * 2) ** 2          # U-shaped, not collinear with the drift
    z_star = -0.2 + z_sky
    nod = np.array(["A", "B", "B", "A"] * (F // 4 + 1))[:F]
    dz = z_sky - z_sky.mean()
    lam, sky, obj = {}, {}, {}
    for m in T.orders:
        lo, hi = sorted(T.lam(m, np.array([4095.0, 1.0])))
        L = np.linspace(lo, hi, 4400)
        lam[m] = L
        ol = lt.oh_lines(lo - 5, hi + 5)
        S = np.zeros((F, L.size))
        for lc, ii, xx in zip(ol["lam"], ol["I"], _xn(T, m, ol["lam"]), strict=True):
            c = lc * (1 + (z_sky + KAPPA * dz * xx) / C_KMS)
            S += 4.0e4 * ii * np.exp(-0.5 * (((L[None, :] / c[:, None]) - 1) * C_KMS / sig) ** 2)
        S += 40.0
        sky[m] = S + rng.normal(0, 1, S.shape) * np.sqrt(S)
        al = lt.absorber_lines(lo - 5, hi + 5, species=("H2O",), R=60000.0)
        tau = np.zeros((F, L.size))
        for lc, dep, xx in zip(al["lam"], al["depth"], _xn(T, m, al["lam"]), strict=True):
            c = lc * (1 + (z_star + KAPPA * dz * xx + GAMMA * (am - am.mean()) * xx) / C_KMS)
            tau += -np.log1p(-min(dep, 0.9)) * np.exp(
                -0.5 * (((L[None, :] / c[:, None]) - 1) * C_KMS / sig) ** 2)
        cont = 3000.0 * np.exp(-tau)
        obj[m] = cont + rng.normal(0, 1, cont.shape) * np.sqrt(cont)
    partner = np.arange(F) ^ 1
    skyset = SpectrumSet(kind="sky", lam=lam, flux=sky, time=t[partner], nod=nod,
                         epoch=[f"S{i}" for i in range(F)])
    objset = SpectrumSet(kind="obj", lam=lam, flux=obj, time=t, nod=nod,
                         epoch=[f"O{i}" for i in range(F)], airmass=am if airmass else None)
    return NightSpectra(sky=skyset, obj=objset, template=T, mode="HIRES-Y", name="chroma"), am, dz


@pytest.fixture(scope="module")
def chroma():
    night, am, dz = _chroma_night()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sols = {src: anchor_night(night, AnchorConfig.for_mode("HIRES-Y", star_modes_source=src))
                for src in ("tell", "oh", "tell_sep")}
    return night, am, dz, sols


def _xprofile(sol, night, kind, m=182):
    """(F, n) mode velocity across order m with the per-frame mean removed (km/s), and xn.
    m182 (51 H2O lines); m163's 14 lines constrain an x-gradient poorly on this 3-order night."""
    lam = np.linspace(night.obj.lam[m][300], night.obj.lam[m][-300], 41)
    v = sol.modes_kms(m, lam, kind)
    return v - v.mean(axis=1, keepdims=True), _xn(night.template, m, lam)


def test_default_is_tell_and_unchanged(chroma):
    """Default: no star_light_modes, kind="tell" == kind="star" bit for bit, no new diagnostics."""
    night, _, _, sols = chroma
    s = sols["tell"]
    assert AnchorConfig().star_modes_source == "tell"
    assert s.star_light_modes is None and "star_modes_source" not in s.diagnostics
    lam = night.obj.lam[163]
    assert np.array_equal(s.labels(163, lam, "tell"), s.labels(163, lam, "star"))
    assert np.array_equal(s.u("tell"), s.u("star"))


def test_tell_sep_separates_telluric_only_chroma(chroma):
    """tell_sep: the airmass x (x, m) term is recovered on the tellurics (GAMMA x dA x xn) and is
    absent from the starlight labels, which keep the shared instrument term (KAPPA x drift); the
    default labels have no airmass term at all (it is not a regressor there)."""
    night, am, dz, sols = chroma
    s = sols["tell_sep"]
    d = s.diagnostics
    assert d["star_modes_source"] == "tell_sep" and d["star_chroma"] == "airmass"
    assert d["star_chroma_wald"] > 50.0
    assert s.star_modes["names"] == ("X", "nod", "chroma")
    assert s.star_light_modes["names"] == ("X", "nod") and s.star_light_modes["C"].shape[1] == 2
    vt, xn = _xprofile(s, night, "tell")
    vs, _ = _xprofile(s, night, "star")
    xc = xn - xn.mean()
    da = am - am.mean()
    # telluric-only part = tell - star labels: GAMMA dA xn (per-frame mean removed)
    g = (vt - vs) @ xc / (xc @ xc)                     # (F,) fitted x-slope per frame
    slope = np.polyfit(da, g, 1)[0]
    assert abs(slope - GAMMA) < 0.25 * GAMMA, slope
    # starlight keeps the instrument drift term, not the airmass term
    gs = vs @ xc / (xc @ xc)
    A = np.column_stack([np.ones_like(dz), dz, da])
    co = np.linalg.lstsq(A, gs, rcond=None)[0]
    assert abs(co[1] - KAPPA) < 0.3 * KAPPA, co
    assert abs(co[2]) < 0.1 * GAMMA, co
    gd = _xprofile(sols["tell"], night, "star")[0] @ xc / (xc @ xc)
    assert abs(np.linalg.lstsq(A, gd, rcond=None)[0][2]) < 0.1 * GAMMA
    assert np.std(gs - co[0] - KAPPA * dz) < 0.02


def test_oh_source_uses_instrument_modes(chroma):
    """oh: starlight modes ARE the OH modes at the object exposures (same coefficients and
    regressors), the telluric fit is kept for kind="tell" and the surface code, and the star zero
    point comes from the tellurics with the OH pattern removed (close to the default's)."""
    night, _, _, sols = chroma
    s, s0 = sols["oh"], sols["tell"]
    assert s.diagnostics["star_modes_source"] == "oh"
    assert np.array_equal(s.star_light_modes["C"], s.oh_modes["C"])
    lam = night.obj.lam[182][::50]
    assert np.array_equal(s.modes_kms(182, lam, "star"), s.modes_kms(182, lam, "inst"))
    assert np.allclose(s.star_modes["C"], s0.star_modes["C"])
    assert s.star_modes["names"] == ("X", "nod")
    assert np.std(s.z_star - s0.z_star) < 0.05
    # the telluric features keep the default's full telluric fit: zero point (z_tell), modes,
    # labels and the per-line table the surface code reads (sol.tell); starlight does not
    assert s.z_tell is not None and s0.z_tell is None
    assert np.allclose(s.z_tell, s0.z_star, rtol=0, atol=1e-9)
    lab = night.obj.lam[182]
    assert np.allclose(s.labels(182, lab, "tell"), s0.labels(182, lab, "star"), rtol=0, atol=1e-9)
    assert np.allclose(s.u("tell"), s0.u("star"), rtol=0, atol=1e-9)
    assert np.array_equal(s.tell["mu"], s0.tell["mu"], equal_nan=True)
    assert np.array_equal(s.tell["z"], s0.tell["z"], equal_nan=True)
    assert "z_star_light" in s.tell and "z_star_light" not in s0.tell
    assert not np.allclose(s.labels(182, lab, "star"), s0.labels(182, lab, "star"))


def test_airmass_fallback_and_validation():
    """No airmass -> star_chroma falls back to time with a warning; bad knobs raise; SpectrumSet
    checks the airmass length and _subset carries it."""
    night, am, _ = _chroma_night(seed=4, F=12, airmass=False)
    with pytest.warns(UserWarning, match="star_chroma='time'"):
        s = anchor_night(night, AnchorConfig.for_mode("HIRES-Y", star_modes_source="tell_sep"))
    assert s.diagnostics["star_chroma"] == "time"
    with pytest.raises(ValueError, match="star_modes_source"):
        anchor_night(night, AnchorConfig.for_mode("HIRES-Y", star_modes_source="bogus"))
    with pytest.raises(ValueError, match="star_chroma"):
        anchor_night(night, AnchorConfig.for_mode("HIRES-Y", star_modes_source="tell_sep",
                                                  star_chroma="pwv"))
    o = night.obj
    with pytest.raises(ValueError, match="airmass"):
        SpectrumSet(kind="obj", lam=o.lam, flux=o.flux, time=o.time, nod=o.nod, airmass=am[:3])
    o2 = SpectrumSet(kind="obj", lam=o.lam, flux=o.flux, time=o.time, nod=o.nod, airmass=am)
    assert np.array_equal(_subset(o2, [1, 3]).airmass, am[[1, 3]])


@pytest.mark.regression
def test_real_night_star_modes_sources():
    """K2-141 (HIRES-Y, 48 frames): the WARP loader fills the airmass; the default labels are
    unchanged by the new knob (kind="tell" == "star"); "oh" / "tell_sep" run and keep the
    telluric fit for the tellurics.  Held-out stellar-line validation (6 nights, scratchpad
    ``starmodes``): neither beats "tell" robustly, so "tell" stays the default."""
    from decanter.wavecal.model import load_warp_night

    if not (RED / "K2141" / "K2-141_NO1").is_dir():
        pytest.skip("K2141 WARP reduction not on this machine")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        night = load_warp_night(RED / "K2141")
        assert night.obj.airmass is not None and np.all(np.isfinite(night.obj.airmass))
        assert 1.0 <= night.obj.airmass.min() < night.obj.airmass.max() < 3.0
        s0 = anchor_night(night)
        s1 = anchor_night(night, AnchorConfig.for_mode(night.mode, star_modes_source="tell_sep"))
        s2 = anchor_night(night, AnchorConfig.for_mode(night.mode, star_modes_source="oh"))
    lam = night.obj.lam[163]
    assert np.array_equal(s0.labels(163, lam, "tell"), s0.labels(163, lam, "star"))
    # "oh": the telluric features keep the default telluric fit (zero point and modes)
    assert np.allclose(s2.labels(163, lam, "tell"), s0.labels(163, lam, "star"), rtol=0, atol=1e-9)
    assert s1.diagnostics["star_chroma"] == "airmass" and s1.diagnostics["star_chroma_dof"] == 8
    for s in (s1, s2):
        # starlight labels move by at most a few hundred m/s at He I; the telluric labels of
        # "oh" are the default's telluric fit
        dv = (s.labels(163, lam, "star") - s0.labels(163, lam, "star")).astype(float) * C_KMS
        assert np.nanmax(np.abs(dv)) < 0.5
    assert np.allclose(s2.star_modes["C"], s0.star_modes["C"])
