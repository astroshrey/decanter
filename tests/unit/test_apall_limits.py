"""apall aperture limits: IRAF ``ap_extract`` swaps an inverted aperture
(``apextract.x:139-149``) before extracting and before writing ``APNUM``.

WARP clamps a frame aperture to the reference window (``lowlim = max(apLow,
lowtrans)``, ``highlim = min(apHigh, hightrans)``), which can leave low > high
in the per-frame database (TOI270c NO38: low 31.03, high 28.00). IRAF extracts
the sorted 3-pixel aperture; decanter used to extract nothing (all zeros).
"""
from __future__ import annotations

import numpy as np
import pytest

from decanter.extract.apsum_exact import apsum, apsum_scalar, extract_reals
from decanter.fast.apsum import apsum_multi
from decanter.io.apdb import parse_text

# WARP's per-frame database for TOI270c NO38 m163 (reductions/TOI270c).
_DB = """# Fri 18:26:33 03-Jul-2026
begin\taperture\tTOI-270_NO38_sscfm_m163trans 163 195.82 2812.00
\timage\tTOI-270_NO38_sscfm_m163trans
\taperture\t163
\tbeam\t163
\tcenter\t195.82 2812.000000
\tlow\t{low} -2811.000000
\thigh\t{high} 2813.000000
\tbackground
\t\txmin 28.00
\t\txmax 31.03
\t\tfunction chebyshev
\t\torder 1
\t\tsample INDEF
\t\tnaverage -3
\t\tniterate 0
\t\tlow_reject 3.
\t\thigh_reject 3.
\t\tgrow 0.
\taxis 1
\tcurve 11
\t\t1.000000
\t\t7.000000
\t\t993.000000
\t\t5593.000000
\t\t10.851478
\t\t97.175174
\t\t-10.662189
\t\t-0.069249
\t\t-0.368997
\t\t-0.048731
\t\t-0.089081

"""


def _entry(low: str, high: str):
    return parse_text(_DB.format(low=low, high=high))[163]


@pytest.fixture(scope="module")
def strip():
    rng = np.random.default_rng(7)
    return rng.uniform(10.0, 1000.0, (5600, 300)).astype(np.float32)


def test_extract_reals_swaps_inverted_limits():
    (_, _), (lo, _), (hi, _), _ = extract_reals(_entry("31.03", "28.00"))
    assert (lo, hi) == (np.float32(28.0), np.float32(31.03))


def test_inverted_aperture_extracts_like_sorted(strip):
    inv, srt = _entry("31.03", "28.00"), _entry("28.00", "31.03")
    want = apsum(strip, srt)
    assert np.count_nonzero(want) > 4000
    assert np.array_equal(apsum(strip, inv), want)
    assert np.array_equal(apsum_scalar(strip[:1200], inv), apsum_scalar(strip[:1200], srt))
    got = apsum_multi([strip, strip[::-1].copy()], inv)
    assert np.array_equal(got[0], want)
    assert np.array_equal(got[1], apsum(strip[::-1].copy(), srt))


def _ref(center_x: float, low: float, high: float):
    from decanter.warptree import _RefAperture
    return _RefAperture(order=163, center_x=center_x, center_y=2814.0, ap_low=low, ap_high=high,
                        ftype=1, yorder=7, ymin=993.0, ymax=5593.0, coeffs=(1.0,) * 7)


@pytest.mark.parametrize("center,low,high,text", [
    # AUMic NO93 m163: solved low -23.705005, DB token -23.71 -> WARP 172.23.
    (195.94, -23.705005, -15.13, "163 163 172.23 180.81"),
    # TOI270c NO38 m163: clamped low 31.03 > high 28.00 -> apall sorts them.
    (195.82, 31.03, 28.0, "163 163 223.82 226.85"),
])
def test_apnum_from_database_tokens(center, low, high, text):
    from decanter.warptree import _apnum
    assert _apnum(_ref(center, low, high), low, high) == text


def test_centersearch_minpack_undefined_is_recorded(tmp_path, monkeypatch):
    """solve_frame_apdb reports, per order, the center-search fits that hit the
    C MINPACK read past the Jacobian (minpack_exact.STATS['undefined'])."""
    import decanter.extract.aperture_solve as A
    from decanter.utils import minpack_exact

    dbs = {}
    for m in (163, 164):
        p = tmp_path / f"apref_{m}trans"
        p.write_text(_DB.format(low="-30.00", high="30.00").replace("163", str(m)))
        dbs[m] = p

    def fake_centersearch(data, ref, **kw):
        if ref.order == 164:
            minpack_exact.STATS["undefined"] += 2
        return 1.0, 3.0

    monkeypatch.setattr(A, "warp_centersearch", fake_centersearch)
    strips = {m: (np.zeros((10, 10), np.float32), 1.0, 1.0, 1.0) for m in dbs}
    sol = A.solve_frame_apdb(strips, dbs, abba=True)
    assert sol.minpack_undefined == {163: 0, 164: 2}
