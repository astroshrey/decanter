"""Bit-exactness tests for the free-path aperture (s07): center search + DB tokens."""

from __future__ import annotations

import math
import os
import types
import warnings
from pathlib import Path

import numpy as np
import pytest

from decanter._localpaths import WARP_ROOT, WINERED_ROOT
from decanter.io.apdb import parse as apdb_parse
from decanter.utils import minpack_exact
from decanter.utils.psf import _warp_bin_medians, warp_trace_x


def _gauss_res(p, y, x, c):
    return y - (math.sqrt(p[0] ** 2) * np.exp(-(x - p[1] - c) ** 2 / (2 * p[2] ** 2)) + p[3])


def _random_problem(rng):
    m = int(rng.integers(8, 50))
    x = np.sort(rng.uniform(-30, 0, m))
    c = rng.normal(-19, 3)
    y = rng.uniform(0.5, 1.5) * np.exp(-(x - c) ** 2 / (2 * rng.uniform(1.5, 5) ** 2)) \
        + rng.normal(0, 0.01, m)
    return (y, x, float(x[np.argmax(y)]))


def test_bin_medians_match_warp_masks() -> None:
    """The one-pass binning equals WARP's 100 boolean-mask np.median calls."""
    rng = np.random.default_rng(3)
    for t in range(300):
        n = int(rng.integers(1, 300))
        x = rng.choice([rng.uniform(-30, 30, n), np.round(rng.uniform(-30, 30, n), 1)])
        y = rng.normal(0, 1, n)
        if t % 5 == 0:
            y[rng.integers(0, n)] = np.nan
        mn, mx = np.amin(x), np.amax(x)
        w = (mx - mn) / 100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_x = np.array([np.median(x[(mn + w * i < x) & (x < mn + w * (i + 1))]) for i in range(100)])
            ref_y = np.array([np.median(y[(mn + w * i < x) & (x < mn + w * (i + 1))]) for i in range(100)])
        got_x, got_y = _warp_bin_medians(x, y, mn, w, 100)
        assert got_x.tobytes() == ref_x.tobytes()
        assert got_y.tobytes() == ref_y.tobytes()


def test_minpack_port_converges_like_scipy() -> None:
    """Sanity: the C-MINPACK port solves the WARP Gaussian fit like scipy does."""
    from scipy.optimize import leastsq
    rng = np.random.default_rng(0)
    for _ in range(20):
        args = _random_problem(rng)
        a = minpack_exact.leastsq(_gauss_res, [1., 0.1, 5., 0.01], args=args, full_output=True)
        b = leastsq(_gauss_res, [1., 0.1, 5., 0.01], args=args, full_output=True)
        # Different MINPACK builds stop at different points within ftol/xtol.
        np.testing.assert_allclose(a[0], b[0], rtol=1e-4, atol=1e-6)


_SO = os.environ.get("DECANTER_SCIPY1153_MINPACK")


@pytest.mark.skipif(not (_SO and Path(_SO).is_file()),
                    reason="set DECANTER_SCIPY1153_MINPACK to scipy-1.15.3's _minpack .so")
def test_minpack_port_bitexact_vs_scipy_1153_binary() -> None:
    """Bitwise against the compiled scipy 1.15.3 lmdif WARP ran."""
    import importlib.machinery
    import importlib.util
    loader = importlib.machinery.ExtensionFileLoader("_minpack", _SO)
    spec = importlib.util.spec_from_file_location("_minpack", _SO, loader=loader)
    mp = importlib.util.module_from_spec(spec)
    loader.exec_module(mp)
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(300):
        args = _random_problem(rng)
        minpack_exact.STATS["undefined"] = 0
        mine = minpack_exact.leastsq(_gauss_res, [1., 0.1, 5., 0.01], args=args)[0]
        if minpack_exact.STATS["undefined"]:
            continue    # the compiled code reads heap garbage here
        ref = mp._lmdif(_gauss_res, np.array([1., 0.1, 5., 0.01]), args, 1, 1.49012e-8,
                        1.49012e-8, 0.0, 1000, float(np.finfo(float).eps), 100, None)[0]
        assert mine.tobytes() == ref.tobytes()
        checked += 1
    assert checked > 250


def test_warp_trace_matches_warp_aperture_class() -> None:
    """warp_trace_x reproduces warp/aperture.py:calculateTrace bitwise."""
    src = WARP_ROOT / "warp" / "aperture.py"
    if not src.is_file():
        pytest.skip("WARP source not available")
    mod = types.ModuleType("warp_aperture")
    exec(compile(src.read_text(), str(src), "exec"), mod.__dict__)
    from decanter.io.apdb import ApertureEntry
    rng = np.random.default_rng(2)
    for ftype in (1, 2):
        for yorder in (1, 2, 5, 7):
            coef = tuple(float("%f" % v) for v in rng.normal(0, 30, yorder))
            e = ApertureEntry(order=163, center_x=195.94, center_y=2814.0, low=-28., high=28.,
                              function_type=ftype, poly_order=yorder, y_min=993., y_max=5593.,
                              coefficients=coef)
            ap = mod.apertureSet.aperture(163, 195.94, 2814.0, -28., 28., ftype, yorder,
                                          993., 5593., list(coef), arrayLength=5628)
            assert warp_trace_x(e, 5628).tobytes() == ap.tracex.tobytes()


_REFS = WINERED_ROOT / "reductions"


def _free_apdb_check(red: Path, frame: int, leastsq, flavour_matches: bool) -> None:
    """WARP strips -> per-frame DBs (token for token, always) and 1D extraction;
    the center-search floats equal WARP's log iff ``flavour_matches``."""
    from astropy.io import fits

    from decanter.calib.discover import _parse_input_files
    from decanter.extract.aperture_solve import solve_frame_apdb, warp_apdb_text
    from decanter.extract.apsum_exact import apsum

    objdir = next(d for d in red.iterdir() if d.is_dir() and d.name.endswith(f"_NO{frame}"))
    obj = objdir.name[: -len(f"_NO{frame}")]
    tdir = objdir / "intermediate_files/OBJ/6-OBJ_transform"
    if not tdir.is_dir():
        pytest.skip("WARP trans strips not saved")
    db = red / "calibration_data/database"
    aptrans = _parse_input_files(red / "calibration_data/input_files.txt")["aptrans file"]
    strips, nod = {}, None
    for p in sorted(tdir.glob(f"{obj}_NO{frame}_sscfm_m*trans.fits")):
        m = int(p.name.rsplit("_m", 1)[1][: -len("trans.fits")])
        with fits.open(p) as h:
            hd = h[0].header
            strips[m] = (h[0].data.astype(np.float32), hd["CRVAL2"], hd["CDELT2"], hd["CRPIX2"])
            nod = hd["NODPOS"]
    orders = sorted(strips)
    refs = {m: db / f"ap{aptrans}_{m}trans" for m in orders}
    sol = solve_frame_apdb(strips, refs, nodpos=nod, leastsq=leastsq)
    log = np.load(red / "reduction_log/centersearch_log.npz")
    aplog = np.load(red / "reduction_log/aperture_log.npz")
    lo_ok = np.float64(sol.lowtrans).tobytes() == aplog["apertureLow"][frame - 1, 0].tobytes()
    assert lo_ok == flavour_matches
    ext = objdir / "intermediate_files/OBJ/8A-OBJ-1DSPEC/1-OBJ-1DSPEC_extract"
    log_orders = list(np.asarray(log["echelleOrder"]).ravel())
    for m in orders:
        j = log_orders.index(m)
        wpath = db / f"ap{obj}_NO{frame}_sscfm_m{m}trans"
        assert sol.entries[m] == apdb_parse(wpath)[m]
        assert (warp_apdb_text(sol.entries[m], f"{obj}_NO{frame}_sscfm_m{m}trans").splitlines()[1:]
                == wpath.read_text().splitlines()[1:])
        same = (np.float64(sol.xcenter[m]).tobytes() == log["psfCenter"][frame - 1, j].tobytes()
                and np.float64(sol.fwhm[m]).tobytes() == log["psfWidth"][frame - 1, j].tobytes())
        assert same == flavour_matches, f"m{m}"
        warp1d = fits.getdata(ext / f"{obj}_NO{frame}_m{m}.{m:04d}.fits").astype(np.float32).ravel()
        assert apsum(strips[m][0], sol.entries[m]).tobytes() == warp1d.tobytes()


def test_toi2109_free_apdb_bitexact(warp_reference_dir: Path) -> None:
    """TOI2109_decanterref was made under scipy 1.15.3: the c1153 port gives
    WARP's fitted floats; the installed scipy 1.14.1 gives the same DB tokens."""
    import scipy
    from scipy.optimize import leastsq as installed
    _free_apdb_check(warp_reference_dir, 1, minpack_exact.leastsq, True)
    if scipy.__version__ != "1.15.3":
        _free_apdb_check(warp_reference_dir, 1, installed, False)


@pytest.mark.parametrize("ref, frame", [("TOI3486_decanterref", 1), ("TOI3486_decanterref", 2),
                                        ("WASP69_decanterref", 1), ("WASP69_decanterref", 2)])
def test_scipy1141_refs_free_apdb_bitexact(ref: str, frame: int, monkeypatch) -> None:
    """References made by WARP in this venv (scipy 1.14.1): the default (the
    installed scipy) gives WARP's fitted floats; c1153 gives the same DB tokens."""
    import scipy
    from decanter.utils.psf import default_warp_leastsq
    red = _REFS / ref
    if not red.is_dir():
        pytest.skip(f"{red} not available")
    if scipy.__version__ != "1.14.1":
        pytest.skip("reference made with scipy 1.14.1")
    monkeypatch.delenv("DECANTER_WARP_MINPACK", raising=False)
    _free_apdb_check(red, frame, None, True)
    _free_apdb_check(red, frame, minpack_exact.leastsq, False)
    monkeypatch.setenv("DECANTER_WARP_MINPACK", "c1153")
    assert default_warp_leastsq() is minpack_exact.leastsq
    monkeypatch.setenv("DECANTER_WARP_MINPACK", "bogus")
    with pytest.raises(ValueError):
        default_warp_leastsq()
