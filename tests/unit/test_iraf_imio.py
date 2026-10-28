"""IRAF's FITS-kernel input mapping (``ieevupkr`` with ``mapin = YES``) and
the IRAF task ports that read through it.

The bit patterns below are what the compiled ``ieevupkr`` of this Mac's
``libvops.a`` returns (``scripts/iraf_oracle/ieee_oracle.py``: 0 mismatches on
20,035,840 words).
"""
from __future__ import annotations

import numpy as np
import pytest

from decanter.extract import apbackground as B
from decanter.extract.apsum_exact import apsum
from decanter.fast.apsum import apsum_multi
from decanter.io.apdb import parse_text
from decanter.utils.iraf_imio import imio_mapped, imio_read_real

R = np.float32
U = np.uint32

#: (input word, word ieevupkr returns)
WORDS = [
    (0x00000000, 0x00000000),   # +0 unchanged
    (0x80000000, 0x00000000),   # -0 -> +0
    (0x00000001, 0x00000000),   # smallest subnormal
    (0x807FFFFF, 0x00000000),   # largest negative subnormal
    (0x00800000, 0x00800000),   # FLT_MIN (normal) unchanged
    (0x80800000, 0x80800000),   # -FLT_MIN unchanged
    (0x3F800000, 0x3F800000),   # 1.0
    (0x7F7FFFFF, 0x7F7FFFFF),   # FLT_MAX
    (0x7F800000, 0x00000000),   # +Inf -> 0
    (0xFF800000, 0x00000000),   # -Inf -> 0
    (0x7FC00000, 0x00000000),   # quiet NaN -> 0
    (0x7F800001, 0x00000000),   # signalling NaN -> 0
    (0xFFC12345, 0x00000000),   # negative NaN with payload -> 0
]


def _words(ws) -> np.ndarray:
    return np.array(ws, dtype=U).view(R)


def test_ieevupkr_words():
    x = _words([w for w, _ in WORDS])
    keep = x.copy()
    y = imio_read_real(x)
    assert y.view(U).tolist() == [w for _, w in WORDS]
    assert np.array_equal(x.view(U), keep.view(U))                   # input untouched
    assert imio_mapped(x).tolist() == [a != b for a, b in WORDS]


def test_keep_nan_and_no_copy():
    x = _words([w for w, _ in WORDS])
    y = imio_read_real(x, keep_nan=True)
    nan = np.isnan(x)
    assert np.array_equal(y.view(U)[nan], x.view(U)[nan])           # NaN kept as is
    assert np.array_equal(y.view(U)[~nan], imio_read_real(x).view(U)[~nan])
    clean = np.linspace(-3, 3, 50, dtype=R)
    assert imio_read_real(clean) is clean                            # nothing to map: no copy
    big = clean.astype(">f4")                                       # FITS byte order
    assert imio_read_real(big).dtype == np.dtype(R)
    assert np.array_equal(imio_read_real(big), clean)


# --- apall reads the strip through the mapping --------------------------------

_DB = """# Mon 00:00:00 01-Jan-2026
begin\taperture\tstrip 163 60.25 150.00
\timage\tstrip
\taperture\t163
\tbeam\t163
\tcenter\t60.25 150.000000
\tlow\t-6.23 -149.000000
\thigh\t5.81 151.000000
\tbackground
\t\txmin -40.00
\t\txmax 40.00
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
\t\t300.000000
\t\t0.836971
\t\t1.680911
\t\t-0.698080

"""


def _strip(seed=3, nl=300, nc=120):
    """A strip whose background regions hold what WARP's transform writes off
    the order: subnormals, -0.0 and tiny normals (and a few +-Inf/NaN)."""
    rng = np.random.default_rng(seed)
    x = np.arange(nc, dtype=R)
    prof = (400.0 * np.exp(-0.5 * ((x - 61.5) / 3.0) ** 2)).astype(R)
    img = (rng.normal(50.0, 5.0, (nl, nc)).astype(R) + prof[None, :]).astype(R)
    off = np.zeros((nl, nc), dtype=bool)
    off[:, :25] = off[:, 98:] = True
    off[200:212] = True             # lines off the order: the aperture too
    sub = (rng.integers(1, 1 << 23, (nl, nc)).astype(U)
           | (rng.integers(0, 2, (nl, nc)).astype(U) << 31)).view(R)
    tiny = (rng.standard_normal((nl, nc)) * 3e-38).astype(R)
    choice = rng.integers(0, 4, (nl, nc))
    val = np.where(choice == 0, sub, np.where(choice == 1, R(-0.0),
                                              np.where(choice == 2, tiny, R(0.0))))
    img = np.where(off, val, img).astype(R)
    img[7, 3] = np.inf
    img[11, 110] = -np.inf
    img[13, 5] = np.nan
    return img


@pytest.mark.parametrize("mode", ["average", "median", "minimum", "fit"])
@pytest.mark.parametrize("sample", ["-40:-30,30:40", "-40:40", "INDEF"])
def test_apall_reads_mapped(mode, sample):
    """Every apall port gives on the raw strip the bits of the mapped strip,
    scalar and vectorised alike (the case WARP hit: off-band regions)."""
    e = parse_text(_DB.format(sample=sample))[163]
    raw = _strip()
    mp = imio_read_real(raw)
    assert imio_mapped(raw).sum() > 5000
    outs = []
    for img in (raw, mp):
        s1, _ = B.sky_values(img, e, mode)
        s2, _ = B.sky_values_fast(img, e, mode)
        outs.append([s1, s2, B.apsum_background(img, e, mode, sky=s1),
                     B.apsum_background_fast(img, e, mode, sky=s2),
                     B.strip_extract(img, e, mode, sky=s1).data,
                     B.strip_extract_fast(img, e, mode, sky=s2).data,
                     apsum(img, e), apsum_multi([img], e)[0]])
    for a, b in zip(*outs, strict=True):
        assert np.array_equal(np.asarray(a).view(U), np.asarray(b).view(U))
    s_raw = outs[0]
    for a, b in ((s_raw[0], s_raw[1]), (s_raw[2], s_raw[3]), (s_raw[4], s_raw[5]),
                 (s_raw[6], s_raw[7])):
        assert np.array_equal(np.asarray(a).view(U), np.asarray(b).view(U))


def test_band_nan_still_poisons_the_vectorised_sum():
    """In a band-evaluated strip NaN marks an unevaluated pixel: the vectorised
    sum keeps it (so the order is rectified in full), the reference port --
    which never sees a band strip -- reads it as IRAF does (0)."""
    e = parse_text(_DB.format(sample="INDEF"))[163]
    img = imio_read_real(_strip())
    img[40, 60] = np.nan
    assert np.isnan(apsum_multi([img], e)[0][40])
    ref = apsum(img, e)
    assert np.isfinite(ref).all()
    img0 = img.copy()
    img0[40, 60] = 0.0
    assert np.array_equal(ref.view(U), apsum(img0, e).view(U))


def test_one_d_ports_read_mapped():
    """scopy truncate, specshift + scombine and the onedspec tasks (scopy,
    sarith, dopcor, continuum, scombine): the raw and the mapped spectrum
    give the same bits."""
    from astropy.io import fits as _f

    from decanter.utils.iraf_scopy import scopy_wavelength_truncate
    from decanter.wavelength import iraf_onedspec as od
    from decanter.wavelength.iraf_scombine import scombine_average
    from decanter.waveshift.apply import shift_truncated
    rng = np.random.default_rng(5)
    n = 400
    spec = (1000 + 50 * rng.standard_normal(n)).astype(R)
    spec[:30] = _words([0x80000000, 0x00000005, 0x80000007] * 10)
    spec[-5:] = R(-0.0)
    mp = imio_read_real(spec)
    h = _f.Header()
    h["CRVAL1"] = 1.0
    h["CDELT1"] = 0.5
    h["CRPIX1"] = 1.0
    a = scopy_wavelength_truncate(spec, h, w1=1.0, w2=150.0).data
    b = scopy_wavelength_truncate(mp, h, w1=1.0, w2=150.0).data
    assert np.array_equal(a.view(U), b.view(U))
    a = shift_truncated(spec, 0.5, 0.0731)
    b = shift_truncated(mp, 0.5, 0.0731)
    assert np.array_equal(a.view(U), b.view(U))
    w = _f.Header()
    w["CRVAL1"] = 10000.0
    w["CDELT1"] = 0.05
    w["CD1_1"] = 0.05
    w["CRPIX1"] = 1.0
    w["CTYPE1"] = "LINEAR"
    w["DC-FLAG"] = 0
    s_raw = od.Spectrum1D(spec, w.copy())
    s_map = od.Spectrum1D(mp, w.copy())
    tasks = [lambda s: od.scopy(s), lambda s: od.sa_1d(s, "-", s_raw),
             lambda s: od.sa_1d(s, "/", "3.5"), lambda s: od.dopcor(s, "-27.3")[0],
             lambda s: od.continuum(s, order=3, niterate=2),
             lambda s: scombine_average([s, s_raw])]
    for f in tasks:
        assert np.array_equal(np.asarray(f(s_raw).data).view(U),
                              np.asarray(f(s_map).data).view(U))
