"""WARP's Background Subtraction / CUTRANSFORM flux / Extract 2d spectrum.

Self-contained checks of the IRAF primitives and of the scalar
(``warp-reference``) against the vectorised (``warp``) kernels on synthetic
strips; with ``reductions/TOI2109_decanterref`` present, the same on WARP's
rectified strips and the transform's flux path against its reference.
Bit-identity with WARP itself is checked by ``scripts/bitexact_checks/
warp_options.py`` against fresh WARP runs (docs/BITEXACT.md).
"""
from __future__ import annotations

import numpy as np
import pytest

from decanter._localpaths import WINERED_ROOT
from decanter.extract import apbackground as B
from decanter.io.apdb import parse_text

REF = WINERED_ROOT / "reductions" / "TOI2109_decanterref"
needs_ref = pytest.mark.skipif(not (REF / "TOI2109_NO1").is_dir(),
                               reason="TOI2109_decanterref not available")
R = np.float32

_DB = """# Mon 00:00:00 01-Jan-2026
begin\taperture\tstrip 163 60.25 1500.00
\timage\tstrip
\taperture\t163
\tbeam\t163
\tcenter\t60.25 1500.000000
\tlow\t{low} -1499.000000
\thigh\t{high} 1501.000000
\tbackground
\t\txmin {xmin}
\t\txmax {xmax}
\t\tfunction chebyshev
\t\torder 1
\t\tsample {sample}
\t\tnaverage -3
\t\tniterate 0
\t\tlow_reject 3.
\t\thigh_reject 3.
\t\tgrow 0.
\taxis 1
\tcurve 7
\t\t1.000000
\t\t3.000000
\t\t1.000000
\t\t3000.000000
\t\t0.836971
\t\t1.680911
\t\t-0.698080

"""


def _entry(sample="4:8,12:16", xmin="-17.23", xmax="16.00", low="-17.23", high="-0.36"):
    return parse_text(_DB.format(sample=sample, xmin=xmin, xmax=xmax, low=low,
                                 high=high))[163]


#: (low, high) of an aperture left of the database centre (the A nod of an
#: A-B pair, x < 0) and right of it (the B nod, x > 0, as WARP's B-A frames).
_NODS = {"A": ("-17.23", "-0.36"), "B": ("10.49", "18.90")}


def _limits(sample, low, high):
    """``write_apdatabase``'s xmin/xmax: the extremes of the window and the
    region numbers (aperture.py:176-182)."""
    v = [float(low), float(high)]
    if "INDEF" not in sample:
        v += [float(t) for part in sample.split(",") for t in part.split(":")]
    return f"{min(v):.2f}", f"{max(v):.2f}"


def _strip(seed=1, nl=300, nc=120):
    rng = np.random.default_rng(seed)
    base = rng.normal(50.0, 5.0, (nl, nc)).astype(R)
    x = np.arange(nc, dtype=R)
    prof = (400.0 * np.exp(-0.5 * ((x - 51.5) / 3.0) ** 2)).astype(R)
    return (base + prof[None, :]).astype(R)


def test_amedr_and_asok():
    rng = np.random.default_rng(0)
    for n in range(4, 30):
        v = rng.standard_normal(n).astype(R)
        assert B.amedr(v) == np.sort(v)[(n + 1) // 2 - 1]
    assert B.amedr([R(3), R(1)]) == R(3)               # a[1] for 1-2 values
    assert B.amedr([R(2), R(9), R(4)]) == R(4)


def test_sample_parsing():
    c = R(100.25)
    assert B.skyeval_regions("-6:-2,2:6", c) == [(R(c - 6), R(c - 2)), (R(c + 2), R(c + 6))]
    assert B.skyeval_regions("6:2", c) == [(R(c + 2), R(c + 6))]
    assert B.skyeval_regions("INDEF", c) == []
    x = (np.arange(-10, 11) + R(0.25)).astype(R)
    assert B.rg_xranges("-6:-2,2:6", x) == [(5, 8), (13, 16)]
    assert B.rg_xranges("2:6,-6:-2", x) == [(5, 8), (13, 16)]          # rg_order
    assert B.rg_xranges("-6:0,-1:6", x) == [(5, 16)]                    # rg_merge
    # rg_xadd: a token with one number v is [min(v, 0), max(v, 0)] (ctor stores
    # 0 when no second number follows), so WARP's "INDEF" is every x >= 0.
    # Values from IRAF's compiled rg_xrangesr + rg_order + rg_merge.
    assert B.rg_xranges("INDEF", x) == [(11, 21)]
    assert B.rg_xranges("0:3e38", x) == [(11, 21)]
    assert B.rg_xranges("5", x) == [(11, 15)]
    assert B.rg_xranges("-5", x) == [(6, 10)]
    assert B.rg_xranges("5abc:7", x) == [(11, 15)]
    assert B.rg_xranges("5:1e39", x) == []          # rval2 stays 5: no x == 5
    assert B.rg_xranges("1e39:5", x) == []          # "Range syntax error"
    assert B.rg_xranges("abc", x) == []
    assert B.rg_xranges("*", x) == [(1, 21)]
    assert B.rg_xranges("INDEF", (x - R(20)).astype(R)) == []         # all x < 0


def test_rg_wtbin_bins_of_three():
    y = [R(v) for v in (5, 1, 3, 9, 7, 8, 2)]
    w = [R(1)] * 7
    out, wout = B.rg_wtbin([(1, 7)], -3, y, w)      # two bins of 3, trailing 1 dropped
    assert out == [R(3), R(8)] and wout == [R(3), R(3)]
    out, wout = B.rg_wtbin([(1, 2)], -3, y, w)      # a short first bin is kept
    assert out == [R(5)] and wout == [R(2)]


@pytest.mark.parametrize("mode", ["average", "median", "minimum", "fit"])
@pytest.mark.parametrize("sample", ["4:8,12:16", "INDEF", "-30:-20"])
@pytest.mark.parametrize("nod", ["A", "B"])
def test_sky_and_sum_fast_equals_reference(mode, sample, nod):
    low, high = _NODS[nod]
    xmin, xmax = _limits(sample, low, high)
    e = _entry(sample, xmin, xmax, low, high)
    img = _strip()
    s1, _ = B.sky_values(img, e, mode)
    s2, _ = B.sky_values_fast(img, e, mode)
    assert np.array_equal(s1.view(np.int32), s2.view(np.int32))
    a = B.apsum_background(img, e, mode)
    b = B.apsum_background_fast(img, e, mode)
    assert np.array_equal(a.view(np.int32), b.view(np.int32))
    st1 = B.strip_extract(img, e, mode)
    st2 = B.strip_extract_fast(img, e, mode)
    assert st1.data.shape == st2.data.shape
    assert np.array_equal(st1.data.view(np.int32), st2.data.view(np.int32))


@pytest.mark.parametrize("fast", [False, True])
def test_fit_indef_is_every_x_ge_0(fast):
    """``sample INDEF`` (WARP's database without ``bg=``) with ``background=
    fit``: ``rg_xadd`` makes it ``[0, INDEFR]``, so the sky of an aperture right
    of the centre (the B nod) is the fit over its whole window -- the same as
    an explicit ``0:3e38`` -- and that of a window entirely at x < 0 is 0
    ("No sample points for fit")."""
    sky = B.sky_values_fast if fast else B.sky_values
    img = _strip(4)
    lo, hi = _NODS["B"]
    s_indef, _ = sky(img, _entry("INDEF", lo, hi, lo, hi), "fit")
    s_range, _ = sky(img, _entry("0:3e38", lo, hi, lo, hi), "fit")
    assert np.array_equal(s_indef.view(np.int32), s_range.view(np.int32))
    assert np.all(np.abs(s_indef - R(50)) < 10)          # the strip's level, ~50
    lo, hi = "-25.00", "-6.00"
    s_a, _ = sky(img, _entry("INDEF", lo, hi, lo, hi), "fit")
    assert not s_a.any()


def test_bg_tokens_warp_stops_on():
    """``reduce_to_warp_tree`` refuses the ``bg=`` regions WARP stops on:
    with a background, ``aperture_plot`` needs ``a:b`` regions ("ERROR in PSF
    center search"); ``write_apdatabase`` needs every number of a sample
    without ``INDEF`` to be a float ("ERROR in extraction")."""
    from decanter.warptree import ListLine, check_bg_tokens

    def line(bg):
        return [ListLine(obj="a", sky="b", raw=f"a b bg={bg}", bg=bg)]

    for bg in (None, "INDEF", "-6:-2,2:6", "1_0:2", "-6:-2:9"):
        check_bg_tokens(line(bg), True)
    for bg in ("5", "-6:-2,5", "-6:-2,INDEF", "5:abc", "", "6:2,"):
        with pytest.raises(ValueError, match="PSF center search"):
            check_bg_tokens(line(bg), True)
    for bg in ("5", "x:INDEF", "INDEF"):
        check_bg_tokens(line(bg), False)          # no aperture_plot of the region
    for bg in ("abc", "5:", "6:2,"):
        with pytest.raises(ValueError, match="ERROR in extraction"):
            check_bg_tokens(line(bg), False)


@pytest.mark.parametrize("skysub", [True, False])
def test_bg_tokens_iraf_cannot_read_as_real(skysub):
    """``float()`` takes ``inf``, ``1e309`` (= inf) and ``1e39``, so WARP's
    write_apdatabase writes them as xmin/xmax; apall's ``dtgetr`` then fails
    (not a number to ``ctod``) or dies (REAL overflow): "ERROR in
    extraction", whatever the background mode (fresh WARP runs
    vfpO1r3_bginf_W ``bg=inf:5``, vOpt3_bginf_Y ``bg=-6:-2,2:inf``).  A
    ``nan`` is never an extreme (Python's min/max keep the window), so it is
    not refused; a limit IRAF's integer arithmetic cannot hold is refused as
    not modelled."""
    from decanter.warptree import ListLine, check_bg_tokens

    def line(bg):
        return [ListLine(obj="a", sky="b", raw=f"a b bg={bg}", bg=bg)]

    for bg in ("inf:5", "-6:-2,2:inf", "-inf:-5", "-1e309:-5", "1e39:2e39", "-6:-2,2:Infinity",
               "-3.5e38:-6"):
        with pytest.raises(ValueError, match="cannot read as a REAL"):
            check_bg_tokens(line(bg), skysub)
    for bg in ("nan:5", "-6:-2,nan:6", "-6:-2,2:3e8"):
        check_bg_tokens(line(bg), skysub)
    for bg in ("-6:-2,2:2e9", "-1.1e9:-6", "-6:-2,2:3e38"):
        with pytest.raises(NotImplementedError, match="32-bit"):
            check_bg_tokens(line(bg), skysub)


def test_listfile_tokens_are_one_flat_list_per_kind(tmp_path):
    """warp/config.py:136-145 appends every bg=/ws=/ap= token of every line to
    one list per kind (``INDEF`` / ``0.`` for a line without) and frame NO<i>
    takes entry i-1: two bg= tokens on line 1 give NO1 the first and NO2 the
    second (WARP's databases of vfpO1r3_bgdup_W: NO1 ``sample -6:-2``, NO2
    ``sample 2:6``)."""
    from decanter.warptree import read_warp_listfile, warp_flat_tokens
    lf = tmp_path / "l.txt"
    lf.write_text("A B bg=-6:-2 bg=2:6 ws=0.5 ws=0.7 ap=-9:-3 ap=1:7\n"
                  "C D bg=-7:-3,1:5\nE F ap=2:8\n")
    with pytest.warns(UserWarning) as rec:
        lines = read_warp_listfile(lf)
    msgs = sorted(str(r.message) for r in rec)
    assert len(msgs) == 3
    for kind in ("ap", "bg", "ws"):
        assert any(f"frames from NO2 on get shifted {kind}= values" in m for m in msgs)
    assert [ln.bg for ln in lines] == ["-6:-2", "2:6", "-7:-3,1:5"]
    assert [ln.ws for ln in lines] == [0.5, 0.7, 0.0]
    assert [ln.ap for ln in lines] == [(-9.0, -3.0), (1.0, 7.0), (2.0, 8.0)]
    ws, ap, bg = warp_flat_tokens(lines)
    assert ws == [0.5, 0.7, 0.0, 0.0] and bg == ["-6:-2", "2:6", "-7:-3,1:5", "INDEF"]
    assert len(ap) == 3
    # one token per kind and line: no warning, the line's own token
    lf.write_text("A B bg=-6:-2 ws=0.1\nC D\n")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lines = read_warp_listfile(lf)
    assert [ln.bg for ln in lines] == ["-6:-2", None]
    assert [ln.ws for ln in lines] == [0.1, 0.0]


def test_waveshift_log_takes_warps_list_lengths(tmp_path):
    """With n_manual (len of WARP's waveshift_man) the zero lists WARP builds
    from it are that long in the npz; the text log prints frameNum entries."""
    from decanter.waveshift.measure import warp_waveshift_log
    spec = [[np.ones(4095, np.float32)] * 2 for _ in range(3)]
    log = warp_waveshift_log(spec, (159, 160, 161), 0.5, measure=False, n_manual=3)
    assert [len(x) for x in (log.waveShiftAve, log.waveShiftStd, log.waveShiftNum,
                             log.waveShiftAdopted)] == [3, 3, 3, 3]
    assert log.waveShift.shape == (3, 2) and log.frameNum == 2
    log.save_text(tmp_path / "w.txt")
    txt = (tmp_path / "w.txt").read_text()
    assert "Average:\t0.0000\t0.0000\t\nStd" in txt
    assert txt.endswith("Adopted shift:\t0.0000\t0.0000\t")


def test_background_none_is_plain_apsum():
    from decanter.extract.apsum_exact import apsum
    e = _entry()
    img = _strip(2)
    assert np.array_equal(B.apsum_background(img, e, "none").view(np.int32),
                          apsum(img, e).view(np.int32))
    st = B.strip_extract(img, e, "none")
    assert st.data.shape[0] == int(R(R(st.high - st.low) + R(1)))


def test_constant_background_is_removed():
    from decanter.extract.apsum_exact import apsum
    e = _entry()
    img = _strip(3)
    flat = np.full_like(img, R(10.0))
    for mode in ("average", "median", "minimum", "fit"):
        s, _ = B.sky_values(img + flat, e, mode)
        s0, _ = B.sky_values(img, e, mode)
        assert np.allclose(s - s0, 10.0, atol=1e-3)
    a = B.apsum_background(img + flat, e, "median")
    b = apsum(img + flat, e)
    assert np.all(a < b)


@needs_ref
@pytest.mark.parametrize("mode", ["average", "median", "minimum", "fit"])
def test_on_warp_strips(mode):
    """Scalar and vectorised background/strip extraction agree on WARP's own
    rectified strips (two orders), with a background region in the database."""
    from astropy.io import fits

    from decanter.extract.aperture_solve import _entry_from_tokens, warp_apdb_tokens
    from decanter.io.apdb import parse
    db = REF / "calibration_data" / "database"
    for m in (163, 175):
        tr = REF / "TOI2109_NO1" / "intermediate_files" / "OBJ" / "6-OBJ_transform" / \
            f"TOI2109_NO1_sscfm_m{m}trans.fits"
        img = fits.getdata(tr).astype(R)
        ref = parse(db / f"apmultihole_HIRES-Y100_20250806_{m}trans")[m]
        frame = parse(db / f"apTOI2109_NO1_sscfm_m{m}trans")[m]
        lo, hi = float(frame.raw_low[0]), float(frame.raw_high[0])
        e = _entry_from_tokens(m, warp_apdb_tokens(ref, lo, hi, "-6:-2,2:6"), "-6:-2,2:6")
        a = B.apsum_background(img, e, mode)
        b = B.apsum_background_fast(img, e, mode)
        assert np.array_equal(a.view(np.int32), b.view(np.int32))
        st1 = B.strip_extract(img, e, mode)
        st2 = B.strip_extract_fast(img, e, mode)
        assert np.array_equal(st1.data.view(np.int32), st2.data.view(np.int32))


@needs_ref
def test_transform_flux_paths_agree():
    """``transform flux=yes``: the Jacobian grid is the same from every
    inversion method, and the fast rectification multiplies by the same
    factor as the reference."""
    from astropy.io import fits

    from decanter.rectify import iraf_transform as T
    from decanter.utils.gsurfit_exact import GsSurface
    db = REF / "calibration_data" / "database"
    sf = GsSurface.from_fc_file(db / "fcmultihole_HIRES-Y100_20250806_163")
    dv = T._dv_of(0.5)
    plans = [T.tr_setup_many([sf], dv, method=meth)[0] for meth in ("exact", "fast")]
    lanes = T.tr_setup_many([sf, sf, sf], dv, method="lanes")[0]
    for p in [*plans[1:], lanes]:
        assert np.array_equal(p.jgrid.view(np.int32), plans[0].jgrid.view(np.int32))
    cut = fits.getdata(REF / "TOI2109_NO1" / "intermediate_files" / "OBJ" / "5-OBJ_cut" /
                       "TOI2109_NO1_sscfm_m163cut.fits").astype(R)
    plan = plans[0]
    a = T.tr_transform(cut, plan, flux=True)
    b = T.tr_transform(cut, plan, flux=False)
    fac = plan.flux_factor(slice(0, plan.nv))
    assert np.array_equal(a.view(np.int32), (b * fac).astype(R).view(np.int32))
    many = T.tr_transform_many([cut], plan, flux=True)[0]
    assert np.array_equal(a.view(np.int32), many.view(np.int32))


def test_2d_header_chain_apnums():
    """The 2-D chain's ``APNUM`` cards: NDSPEC lines become apertures with two
    limit pairs; scombine/dispcor (MULTISPEC) keep one."""
    from decanter.warptree_chain import apnum_es, apnum_ms, apnum_nd
    assert apnum_nd(3) == "3 0 3.00 3.00 1.00 1.00"
    assert apnum_ms(apnum_nd(17)) == "17 0 17.00 17.00"
    assert apnum_es("2 0 2.00 2.00 1.00 1.00") == "2 0 2.00 2.00 1.00 1.00"
    assert apnum_es("163 163 168.71 185.58") == "163 163 168.71 185.58"
