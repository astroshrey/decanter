"""apall on apertures WARP never writes: off the strip, and dispersion windows
that start after line 1 (``bmin > 1``).

``ap_extract`` passes ``l1 = ys = 1`` to ``ap_sum``/``ap_skyeval``/
``ap_strip``: the trace is evaluated at buffer lines while the buffer origin
``astart`` comes from image lines, so with ``bmin > 1`` and a curved trace
IRAF's reads leave the line -- other lines' data and sky, memio's header
cells before a malloc'ed array -- or ``asifit`` gets fewer than 2 samples and
apall stops.  The scalar and vectorised ports must agree (values, or the
same exception); ``scripts/iraf_oracle/pyraf_apall_bmin_check.py`` checks
them against IRAF itself (260 cases: 194 x 4 outputs identical, the rest
both stop or are refused as heap reads).
"""
from __future__ import annotations

import numpy as np
import pytest

from decanter.extract import apbackground as B
from decanter.extract.apsum_exact import (IrafRealBuffer, _LinearAsi, apsum, apsum_scalar)
from decanter.fast.apsum import apsum_multi
from decanter.io.apdb import parse_text

R = np.float32
MODES = ("none", "average", "median", "minimum", "fit")


def _db(cx=60.0, lowb=-149.0, highb=151.0, a=0.0, nl=300, low=-4.0, high=4.0,
        sample="-12:-7,7:12", xmin=-12.0, xmax=12.0):
    cy = nl / 2.0
    return parse_text(
        f"begin\taperture\ts 7 {cx:.2f} {cy:.2f}\n\timage\ts\n\taperture\t7\n\tbeam\t7\n"
        f"\tcenter\t{cx:.2f} {cy:f}\n\tlow\t{low:.2f} {lowb:f}\n\thigh\t{high:.2f} {highb:f}\n"
        f"\tbackground\n\t\txmin {xmin:.2f}\n\t\txmax {xmax:.2f}\n\t\tfunction chebyshev\n"
        f"\t\torder 1\n\t\tsample {sample}\n\t\tnaverage -3\n\t\tniterate 0\n"
        "\t\tlow_reject 3.\n\t\thigh_reject 3.\n\t\tgrow 0.\n\taxis 1\n\tcurve 6\n"
        f"\t\t2.000000\n\t\t2.000000\n\t\t1.000000\n\t\t{float(nl):f}\n\t\t0.000000\n"
        f"\t\t{a:f}\n")[7]


def _strip(nl=300, nc=120, seed=1):
    rng = np.random.default_rng(seed)
    x = np.arange(nc, dtype=R)
    return (rng.normal(50, 5, (nl, nc)) + 400 * np.exp(-0.5 * ((x - 60) / 3.0) ** 2)).astype(R)


def _ports(img, e, mode):
    one = ([("apsum", lambda: apsum(img, e)), ("apsum_scalar", lambda: apsum_scalar(img, e)),
            ("apsum_multi", lambda: apsum_multi([img], e)[0])] if mode == "none" else
           [("apsum_background", lambda: B.apsum_background(img, e, mode)),
            ("apsum_background_fast", lambda: B.apsum_background_fast(img, e, mode))])
    strip = [("strip_extract", lambda: B.strip_extract(img, e, mode).data),
             ("strip_extract_fast", lambda: B.strip_extract_fast(img, e, mode).data)]
    out = []
    for group in (one, strip):
        res = []
        for name, f in group:
            try:
                res.append((name, f()))
            except (ValueError, NotImplementedError) as exc:
                res.append((name, type(exc)))
        out.append(res)
    return out


def _agree(group):
    (n0, v0), *rest = group
    for n, v in rest:
        if isinstance(v0, type) or isinstance(v, type):
            assert v0 is v, (n0, v0, n, v)
        else:
            assert np.array_equal(np.asarray(v0, R).view(np.int32),
                                  np.asarray(v, R).view(np.int32)), (n0, n)
    return v0


@pytest.mark.parametrize("cx", [-20.0, -3.0, 45.0, 60.0])
def test_off_strip_aperture_is_zero_in_every_port(cx):
    """No line with a usable window: IRAF writes zeros (the verifier's IRAF
    runs; pyraf_apall_bmin_check.py's synthetic cases), the scalar ports did,
    the vectorised ones raised (negative dimensions / IndexError)."""
    img = np.random.default_rng(0).normal(100, 10, (30, 40)).astype(R)
    e = _db(cx=cx, lowb=-14.0, highb=16.0, nl=30, low=-3.0, high=3.0, sample="-6:-2,2:6",
            xmin=-6.0, xmax=6.0)
    for mode in MODES:
        for group in _ports(img, e, mode):
            v = _agree(group)
            if cx in (-20.0, 60.0):
                assert not np.any(v), (cx, mode)
        if mode != "none":
            s_ref, _ = B.sky_values(img, e, mode)
            s_fast, _ = B.sky_values_fast(img, e, mode)
            assert np.array_equal(s_ref.view(np.int32), s_fast.view(np.int32))


@pytest.mark.parametrize("a", [5.0, 20.0, 40.0, -40.0])
@pytest.mark.parametrize("bmin", [2, 20, 50, 120, 200])
def test_bmin_after_line_one_scalar_equals_vectorised(a, bmin):
    img = _strip()
    e = _db(lowb=bmin - 150.0, a=a)
    for mode in MODES:
        for group in _ports(img, e, mode):
            _agree(group)


def test_bmin_cases_cover_iraf_behaviour():
    """The synthetic set reaches each IRAF behaviour: plain values, memio
    header reads (interpolant and sky buffer), asifit's error, and the
    refused heap reads past the sky buffer."""
    img = _strip()
    seen = set()
    calls = {"n": 0}
    orig = _LinearAsi.with_header

    def counting(self):
        calls["n"] += 1
        return orig(self)
    _LinearAsi.with_header = counting
    try:
        for a, bmin, mode in [(20.0, 50, "none"), (20.0, 120, "none"), (-40.0, 50, "median"),
                              (5.0, 120, "median")]:
            calls["n"] = 0
            e = _db(lowb=bmin - 150.0, a=a)
            try:
                v = (apsum(img, e) if mode == "none" else B.apsum_background(img, e, mode))
                seen.add("values")
                if calls["n"]:
                    seen.add("header cells")
                assert np.isfinite(v).all()
            except ValueError as exc:
                assert "ASIFIT: too few points for LINEAR" in str(exc)
                seen.add("asifit stops")
            except NotImplementedError as exc:
                assert "heap" in str(exc)
                seen.add("refused")
    finally:
        _LinearAsi.with_header = orig
    assert seen == {"values", "header cells", "asifit stops", "refused"}


def test_memio_header_cells():
    """What IRAF reads before a malloc'ed REAL array (sys/memio/msvfwa.x:
    8-byte cells, little-endian): REAL -1 = high half of the lower sentinel
    (0), -2 = its low half (20030125), -4 = the element count, -6 = TY_REAL."""
    b = IrafRealBuffer(np.arange(5, dtype=R))
    bits = [int(np.asarray(b.at(i), R).view(np.uint32)) for i in range(-6, 0)]
    assert bits == [6, 0, 5, 0, 20030125, 0]
    assert b.at(4) == 4
    with pytest.raises(NotImplementedError):
        b.at(-7)
    with pytest.raises(NotImplementedError):
        b.at(5)
    asi = _LinearAsi(np.array([1, 2, 3], R)).with_header()
    assert [int(np.asarray(asi.eval_int(j), R).view(np.uint32)) for j in range(-5, 1)] == \
        [6, 0, 4, 0, 20030125, 0]
    assert asi.eval_int(2) == 2 and asi.grl(R(1.2), R(2.7)) == \
        _LinearAsi(np.array([1, 2, 3], R)).grl(R(1.2), R(2.7))
