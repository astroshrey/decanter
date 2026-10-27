"""The default WARP clone (``decanter.reduce(..., mode="warp")``, the
vectorised implementation in :mod:`decanter.fast`).

Kernel tests run on synthetic data; the end-to-end checks against the scalar
reference port (``mode="warp-reference"``) use the TOI2109 WARP ``-s``
reference reduction and skip without it. The default is bit-identical to the
reference port (``docs/VECTORISED.md``); the MLX backend's identity is an
observation about the Metal compiler, tested here too.
"""
from __future__ import annotations

import dataclasses
import platform

import numpy as np
import pytest
from scipy.ndimage import median_filter

import decanter
from decanter.config import Config
from decanter.fast import _backend
from decanter.fast import image2d as f2d
from decanter.fast import rectify as frect
from decanter.utils.cosmic_ray import _make_slant_footprint
from decanter.utils.spline3 import iraf_spline3_coeffs, spline2d_exact
from tests.conftest import WARP_REFERENCE_PATH

_HAVE_REF = (WARP_REFERENCE_PATH / "rawdata_image").is_dir()
needs_ref = pytest.mark.skipif(not _HAVE_REF, reason="TOI2109 WARP reference not available")
needs_mlx = pytest.mark.skipif(not _backend.mlx_available(), reason="mlx (GPU) not available")


# --- backend selection --------------------------------------------------------

def test_backend_env_and_override(monkeypatch):
    monkeypatch.setenv("DECANTER_FAST_BACKEND", "numpy")
    assert _backend.active_backend() == "numpy"
    with _backend.backend("numpy"):
        assert _backend.active_backend() == "numpy"
    with pytest.raises(ValueError):
        _backend.set_backend("cuda")
    monkeypatch.setenv("DECANTER_FAST_BACKEND", "auto")
    assert _backend.active_backend() == ("mlx" if _backend.mlx_available() else "numpy")


def test_backend_mlx_requested_but_missing(monkeypatch):
    monkeypatch.setattr(_backend, "_mx", None)
    monkeypatch.setattr(_backend, "_mx_checked", True)
    monkeypatch.setenv("DECANTER_FAST_BACKEND", "mlx")
    with pytest.raises(RuntimeError, match="mlx"):
        _backend.active_backend()
    monkeypatch.setenv("DECANTER_FAST_BACKEND", "auto")
    assert _backend.active_backend() == "numpy"


def test_reduce_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        decanter.reduce(np.zeros((4, 4)), None, mode="turbo")
    with pytest.raises(ValueError, match="mode"):
        decanter.reduce_many([], None, mode="turbo")


def test_mode_names():
    from decanter.api import implementation, resolve_mode
    assert resolve_mode("warp") == "warp"
    assert resolve_mode("warp-reference") == "warp-reference"
    for bad in ("reference", "fast"):
        with pytest.raises(ValueError, match="not implemented"):
            resolve_mode(bad)
    with _backend.backend("numpy"):
        assert implementation("warp") == {"mode": "warp", "implementation": "vectorised",
                                          "backend": "numpy"}
    assert implementation("warp-reference")["implementation"] == "reference"


def test_default_mode_is_vectorised(monkeypatch):
    """reduce() with no mode runs decanter.fast's rectification, warp-reference does not."""
    import inspect

    from decanter import api
    assert inspect.signature(api.reduce).parameters["mode"].default == "warp"
    assert inspect.signature(api.reduce_many).parameters["mode"].default == "warp"
    assert inspect.signature(decanter.reduce_to_warp_tree).parameters["mode"].default == "warp"

    class _Called(Exception):
        pass

    def boom(*a, **k):
        raise _Called

    monkeypatch.setattr(api, "_read_input", lambda *a, **k: boom())
    with pytest.raises(_Called):
        api.reduce("obj.fits", None)        # gets past mode validation for the default


# --- batched natural-spline solve (s06 msifit) ----------------------------------

def _layout(lengths, rng, K=2):
    lengths = sorted(lengths, reverse=True)
    base = np.cumsum([0] + lengths[:-1])
    src = rng.normal(size=(int(sum(lengths)), K)) * 100.0
    return frect.SolveLayout.build(base, lengths, max(lengths) + 3), src, lengths


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.int32)


def test_spline_solve_bit_identical_to_iraf_float32():
    rng = np.random.default_rng(1)
    lay, src, lengths = _layout([300, 120, 57, 40, 40, 9, 5, 2], rng)
    src = src.astype(np.float32)
    R = frect._solve_numpy(src, lay)
    assert R.dtype == np.float32
    for j, n in enumerate(lengths):
        for k in range(src.shape[1]):
            v = np.zeros(n + 2, np.float32)
            v[1:n + 1] = src[lay.base[j]:lay.base[j] + n, k]
            ref = spline2d_exact(v[None, :])[0]          # warp mode's IRAF float32
            assert np.array_equal(_bits(R[: n + 2, j, k]), _bits(ref)), (j, k)
            assert not R[n + 2:, j, k].any()              # IRAF's zero padding
            # and the same spline as the float64 textbook solve
            np.testing.assert_allclose(R[: n + 2, j, k],
                                       iraf_spline3_coeffs(v[1:n + 1].astype(float)),
                                       rtol=1e-4, atol=1e-3)


@needs_mlx
def test_spline_solve_mlx_matches_numpy():
    import mlx.core as mx
    rng = np.random.default_rng(3)
    lay, src, _ = _layout([200, 150, 150, 31, 4], rng)
    src = src.astype(np.float32)
    R = frect._solve_numpy(src, lay)
    Rm = np.array(frect._solve_mlx(mx.array(src), lay, {}))
    assert np.array_equal(_bits(Rm.reshape(R.shape)), _bits(R))


def test_fma_into_resolves_midpoints():
    # a*b + c landing exactly on a float32 midpoint in float64: the single
    # rounding (fma) must use the sign of the dropped low part.
    from decanter.utils.spline3 import fma32_exact
    rng = np.random.default_rng(7)
    a = rng.normal(size=4096).astype(np.float32)
    b = rng.normal(size=4096).astype(np.float32)
    c = rng.normal(size=4096).astype(np.float32)
    a[:2] = [1.22119140625, -1.22119140625]
    b[:2] = [4.88086016048328e-08, 4.88086016048328e-08]    # float64 sum hits a midpoint
    c[:2] = [1.0, -1.0]
    naive = (a.astype(float) * b + c).astype(np.float32)
    assert naive[0] != fma32_exact(a[:1], b[:1], c[:1])[0]     # double rounding is wrong
    out = c.copy()
    p = np.empty(4096)
    s = np.empty(4096)
    frect._fma_into(out, a.astype(float), b.astype(float), out, p, s)   # aliased c
    assert np.array_equal(_bits(out), _bits(fma32_exact(a, b, c)))


def test_spline_layout_rejects_unsorted():
    with pytest.raises(ValueError):
        frect.SolveLayout.build(np.array([0, 5]), np.array([3, 5]), 8)


# --- s02 median filters ----------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1])
def test_median_filters_exact(seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(-20, 20, size=(41, 37)).astype(np.float32)    # many ties
    assert np.array_equal(f2d.median5_vertical(x),
                          median_filter(x, size=(5, 1), mode="reflect"))
    assert np.array_equal(f2d.median5_diagonal(x),
                          median_filter(x, footprint=np.identity(5, int), mode="reflect"))
    y = np.abs(rng.normal(size=(53, 29)))
    assert np.array_equal(f2d.median_slant10(y),
                          median_filter(y, footprint=_make_slant_footprint(85.0, 10),
                                        mode="reflect"))


# --- against warp mode on real data ------------------------------------------------

def _pair():
    root = WARP_REFERENCE_PATH
    a, b = (root / f"{root.name}.txt").read_text().split()[:2]
    return root / "rawdata_image" / f"{a}.fits", root / "rawdata_image" / f"{b}.fits"


@needs_ref
@pytest.mark.parametrize("path", ["locked", "free"])
@pytest.mark.parametrize("backend", ["numpy", pytest.param("mlx", marks=needs_mlx)])
def test_reduce_fast_matches_warp(path, backend):
    calib = decanter.Calibration.from_dir(WARP_REFERENCE_PATH)
    if path == "free":
        calib = dataclasses.replace(calib, trans_apdbs=None)
    cfg = Config(reduce_full_data=False, selected_orders=(163, 171))
    obj, sky = _pair()
    rw = decanter.reduce(obj, calib, sky=sky, config=cfg, shift_wave=0.013,
                         mode="warp-reference")
    with _backend.backend(backend):
        rf = decanter.reduce(obj, calib, sky=sky, config=cfg, shift_wave=0.013)   # default
    assert set(rf.obj) == set(rw.obj) and set(rf.sky) == set(rw.sky)
    for kind in ("obj", "sky"):
        A, B = getattr(rw, kind), getattr(rf, kind)
        for key in A:
            assert A[key].header_wcs == B[key].header_wcs
            assert (A[key].crval1, A[key].cdelt1) == (B[key].crval1, B[key].cdelt1)
            assert np.array_equal(_bits(B[key].flux), _bits(A[key].flux)), (kind, key)
    assert rf.apertures == rw.apertures
    # the default always keeps the 1-D extractions (reduce_many / warp tree use them)
    assert set(rf.intermediates.spectra_1d) == {163, 171}


@needs_ref
def test_fast_rectify_band_and_full():
    from decanter.calib.aperture import ApertureSet
    from decanter.io import fits as _fits
    from decanter.rectify.transform import rectify_orders_multi as exact

    calib = decanter.Calibration.from_dir(WARP_REFERENCE_PATH)
    ap = ApertureSet.load(calib.apdb_multihole)
    dy = float(_fits.read_image(calib.comp)[1]["CDELT1"])
    rng = np.random.default_rng(5)
    img = rng.normal(100.0, 10.0, size=(2048, 2048)).astype(np.float32)
    kw = dict(fc_dir=calib.fc_dir, fc_refname=calib.fc_refname, dy=dy, orders=(163,))
    want = exact([img], ap, **kw)[0][163].data
    with _backend.backend("numpy"):
        band = frect.rectify_orders_multi([img], ap, ref_trans_apdbs=calib.ref_trans_apdbs,
                                          **kw)[0][163]
        full = frect.rectify_orders_multi([img], ap, full=True, **kw)[0][163]
    ok = np.isfinite(band.data)
    assert 0.05 < ok.mean() < 0.5                       # only the extraction band
    assert np.array_equal(_bits(band.data[ok]), _bits(want[ok]))
    assert np.array_equal(_bits(full.data), _bits(want))
    assert band.crval2_text and band.crval2_text == full.crval2_text


@needs_ref
def test_fast_cr_mask_bit_identical():
    from decanter.calib.aperture import ApertureSet
    from decanter.image2d import cr_mask, sky_subtract
    from decanter.io import fits as _fits

    calib = decanter.Calibration.from_dir(WARP_REFERENCE_PATH)
    obj, sky = _pair()
    o, ho = _fits.read_image(obj)
    s, _ = _fits.read_image(sky)
    bp, _ = _fits.read_image(calib.static_bp_mask)
    d = sky_subtract(o, s)
    from decanter.utils.cosmic_ray import ndr_from_header
    kw = dict(nodpos=str(ho["NODPOS"]), ndr_obj=ndr_from_header(ho),
              ndr_sky=ndr_from_header(_fits.read_image(sky)[1]), config=Config())
    want = cr_mask(d, o, s, ApertureSet.load(calib.apdb_multihole), bp, **kw)
    got = decanter.fast.cr_mask(d, o, s, calib, bp, **kw)
    assert np.array_equal(got, want) and want.sum() > 0


@needs_ref
def test_warp_tree_default_mode(tmp_path):
    obj, sky = _pair()
    cfg = Config(reduce_full_data=False, selected_orders=(163,))
    tree = decanter.reduce_to_warp_tree([(obj.stem, sky.stem)], obj.parent,
                                        WARP_REFERENCE_PATH, tmp_path / "t",
                                        config=cfg, log=lambda s: None)
    from astropy.io import fits
    for cut in tree.cuts:
        p = tree.vac_flux(1, 163, cut)
        assert fits.getheader(p)["DECANTER"].endswith("mode=warp")
        assert np.isfinite(fits.getdata(p)).all()
    assert tree.mode == "warp"


@needs_ref
def test_apsum_multi_bit_identical():
    import re

    from decanter.extract.apsum_exact import apsum
    from decanter.fast.apsum import apsum_multi
    from decanter.io import fits as _fits
    from decanter.io.apdb import parse

    root = WARP_REFERENCE_PATH
    trans = root / "TOI2109_NO1" / "intermediate_files"
    for q in sorted((root / "calibration_data" / "database").glob(
            "apTOI2109_NO1_sscfm_m16[3-5]trans")):
        m = int(re.search(r"_m(\d+)trans$", q.name).group(1))
        e = parse(q)[m]
        so = _fits.read_image(trans / f"OBJ/6-OBJ_transform/TOI2109_NO1_sscfm_m{m}trans.fits")[0]
        ss = _fits.read_image(trans / f"SKY/4-SKY_transform/TOI2109_skyNO1_fm_m{m}trans.fits")[0]
        got = apsum_multi([so, ss], e)
        for k, img in enumerate((so, ss)):
            assert np.array_equal(got[k].view(np.int32), apsum(img, e).view(np.int32))


@needs_ref
def test_dispcor_edges_fast_identical():
    from pathlib import Path

    from decanter.fast.wavelength import dispcor_edges_fast
    from decanter.wavelength.iraf_msp import dispcor_edges

    calib = decanter.Calibration.from_dir(WARP_REFERENCE_PATH)
    ids = tuple(str(Path(calib.id_dir) / f"id{calib.id_refname}.{m:04d}") for m in (163, 171))
    for got, want in zip(dispcor_edges_fast(ids, 4095), dispcor_edges(ids, 4095)):
        assert np.array_equal(got, want)


# --- s02 CR mask: WARP factor map, vectorised (synthetic) ------------------------

# The emulation reproduces NumPy's float32 reductions as built for arm64 (the
# machine WARP's references were made on, docs/BITEXACT.md); x86 NumPy builds
# vectorise these reductions differently.
_ARM64_ONLY = pytest.mark.skipif(
    platform.machine().lower() not in ("arm64", "aarch64"),
    reason="bit-exact against arm64 NumPy's float32 reductions",
)


@_ARM64_ONLY
def test_segment_std_matches_numpy():
    """Segmented float32 np.std / np.mean (pairwise sums, 8192-element reduce
    chunks) equal NumPy's own, element for element."""
    rng = np.random.default_rng(0)
    lens = np.array(list(range(0, 300)) + [1000, 8191, 8192, 8193, 20000], dtype=np.intp)
    vals = [(rng.normal(0, 1, n) * 10 ** rng.uniform(-3, 6, n)).astype(np.float32) for n in lens]
    allv = np.concatenate(vals)
    starts = np.concatenate(([0], np.cumsum(lens)[:-1])).astype(np.intp)
    seg = np.repeat(np.arange(lens.size), lens)
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        want_s = np.array([np.std(v) for v in vals], dtype=np.float32)
        want_m = np.array([np.mean(v) for v in vals], dtype=np.float32)
    got_s = f2d.segment_std(allv, seg, starts, lens)
    got_m = f2d.segment_mean(allv, starts, lens)
    assert np.array_equal(got_s, want_s, equal_nan=True)
    assert np.array_equal(got_m, want_m, equal_nan=True)


class _FakeApset:
    """Just what CrGeometry.build reads, from explicit apmask / slitcoord arrays."""

    def __init__(self, apmask, slitcoord, orders):
        self._a, self._s, self.echelle_orders = apmask, slitcoord, orders
        self.array_length = apmask.shape[0]

    def apmask_array(self, *, low_lim, upp_lim):
        return self._a

    def slitcoord_array(self, *, low_lim, upp_lim):
        return self._s


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_cr_mask_matches_reference_synthetic(seed):
    """Two orders (one missing the first slit bin, one with a noisy edge strip and
    a noisy measured column), random raws with cosmic rays in both frames:
    the vectorised mask equals detect_cosmic_rays (the literal WARP port)."""
    from decanter.utils.cosmic_ray import detect_cosmic_rays
    H = W = 500
    rng = np.random.default_rng(seed)
    x = np.arange(W, dtype=np.float64)
    apmask = np.zeros((H, W), dtype=np.int32)
    slitcoord = np.full((H, W), -10000.0)
    for m, cx, lo in ((160, 150.0, -30.0), (161, 330.5, -25.0)):
        sl = np.broadcast_to(x - cx + 0.002 * np.arange(H)[:, None], (H, W))
        inside = (np.abs(sl) < 30) & (sl > lo)
        apmask[inside] = m
        slitcoord[inside] = sl[inside]
    raw1 = rng.normal(3000, 60, (H, W)).astype(np.float32)
    raw2 = rng.normal(3000, 60, (H, W)).astype(np.float32)
    raw1[:, 179] += rng.normal(0, 3000, H).astype(np.float32)       # edge strip of 160
    raw2[:, 340] += rng.normal(0, 800, H).astype(np.float32)        # measured bin of 161
    for _ in range(300):
        y, xx = rng.integers(0, H), rng.integers(0, W)
        (raw1 if rng.random() < 0.5 else raw2)[y, xx] += rng.uniform(200, 5000)
    diff = raw1 - raw2
    bp = (rng.random((H, W)) < 0.01).astype(np.int16)
    kw = dict(ndr1=8, ndr2=4, abba=bool(seed % 2))
    want = detect_cosmic_rays(diff, raw1, raw2, apmask, slitcoord, bp,
                              echelle_orders=(160, 161), array_length=H, **kw).mask
    geom = f2d.CrGeometry.build(_FakeApset(apmask, slitcoord, (160, 161)), bp)
    got = f2d.cr_mask_fast(diff, raw1, raw2, geom, **kw)
    assert want.sum() > 0 and np.array_equal(got, want)
