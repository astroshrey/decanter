"""s11 shift: scombine's 1 % coverage rule (``scb_rebin``, t_scombine.x:381-402).

WARP shifts a truncated spectrum with ``PySpecshift`` (``scopy``,
``specshift``, ``scombine(w1=1., dw=CDELT1, nw=NAXIS1)``).  ``scb_rebin``
flags an output pixel that gets less than 1 % of an input pixel as missing,
and with one input spectrum ``scombine`` writes ``blank`` (0.) there.  The
expected blanked pixels below are IRAF's: WARP's own ``PySpecshift`` run
through pyraf on WARP's ``_c`` files and 2-D ``trans2dcut`` strips
(``scripts/iraf_oracle/pyraf_specshift_check.py``, 2042 cases, 2026-10-21),
including the 12-digit shifts on either side of every threshold and WARP's
adopted shifts of LTT3780 2025-02-20 NO9 (+0.49617) and NO127 (-0.49983),
whose first / last ``_cs`` pixel WARP writes as 0.0.
"""
from __future__ import annotations

import numpy as np
import pytest

from decanter.utils.fpemu import iraf_ctod
from decanter.utils.iminterp import scombine_linear_poly5
from decanter.utils.iraf_fmt import roundtrip
from decanter.wavelength.iraf_scombine import D_ALL, D_MIX, D_NONE, scb_rebin_window
from decanter.wavelength.iraf_wcs import pyraf_real_text
from decanter.waveshift.apply import scb_coverage, shift_truncated

# (CDELT1, NAXIS1, shift, pixels IRAF blanks at the start, at the end)
IRAF_CASES = [
    # HIRES-Y / HIRES-J truncate grid
    (0.5, 4095, 0.4835, 0, 0),
    (0.5, 4095, 0.494999974966, 0, 0),
    (0.5, 4095, 0.494999974967, 1, 0),
    (0.5, 4095, 0.4961666666666666, 1, 0),        # LTT3780 NO9
    (0.5, 4095, 0.5061249999999999, 1, 0),        # no overlap at all there
    (0.5, 4095, 0.994999945163, 1, 0),
    (0.5, 4095, 0.994999945164, 2, 0),
    (0.5, 4095, -0.495056152343, 0, 0),
    (0.5, 4095, -0.495056152344, 0, 1),
    (0.5, 4095, -0.4998333333333333, 0, 1),       # LTT3780 NO127
    (0.5, 4095, -0.502, 0, 1),
    (0.5, 4095, -0.995056152343, 0, 1),
    (0.5, 4095, -0.995056152344, 0, 2),
    # WIDE truncate grid
    (1.0, 2048, 0.989999949932, 0, 0),
    (1.0, 2048, 0.989999949933, 1, 0),
    (1.0, 2048, 1.98999989032, 1, 0),
    (1.0, 2048, 1.98999989033, 2, 0),
    (1.0, 2048, -0.990051269531, 0, 0),
    (1.0, 2048, -0.990051269532, 0, 1),
    (1.0, 2048, -1.99005126953, 0, 1),
    (1.0, 2048, -1.99005126954, 0, 2),
]


def _spectrum(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 30.0, n)
    return (5000.0 + 800.0 * np.sin(x) + rng.normal(0.0, 30.0, n)).astype(np.float32)


def _unmasked(spec: np.ndarray, cd: float, shift: float) -> np.ndarray:
    """The scombine resample without scb_rebin's mask (decanter before 2026-10-21)."""
    w1 = 1.0 + iraf_ctod(pyraf_real_text(shift))
    dw = iraf_ctod(pyraf_real_text(cd))
    return scombine_linear_poly5(spec, crval1_in=roundtrip(w1), cdelt1_in=dw, w1_out=1.0,
                                 dw_out=dw, nw_out=spec.size, flux=False, mwcs=True)


@pytest.mark.parametrize("cd, n, shift, lo, hi", IRAF_CASES)
def test_blanked_edge_pixels_are_irafs(cd, n, shift, lo, hi):
    spec = _spectrum(n)
    out = shift_truncated(spec, cd, shift)
    ref = _unmasked(spec, cd, shift)
    assert out.dtype == np.float32 and out.size == n
    assert np.all(out[:lo] == 0) and np.all(out[n - hi:] == 0)
    keep = slice(lo, n - hi)
    assert np.array_equal(out[keep].view(np.uint32), ref[keep].view(np.uint32))
    # the blanked pixels are 0.0 bits (blank = +0.), and the first/last kept one is data
    assert not np.any(out[:lo].view(np.uint32)) and not np.any(out[n - hi:].view(np.uint32))
    assert out[lo] != 0 and out[n - hi - 1] != 0
    w1 = 1.0 + iraf_ctod(pyraf_real_text(shift))
    dw = iraf_ctod(pyraf_real_text(cd))
    assert scb_coverage(roundtrip(w1), dw, n, 1.0, dw, n) == (1 + lo, n - hi)


def test_in_band_shift_changes_only_the_partial_edge_pixel():
    """Before the fix decanter wrote the partial-coverage integral there."""
    spec = _spectrum(4095, 1)
    out = shift_truncated(spec, 0.5, 0.4961666666666666)
    ref = _unmasked(spec, 0.5, 0.4961666666666666)
    assert ref[0] != 0 and out[0] == 0
    assert np.array_equal(out[1:], ref[1:])
    out = shift_truncated(spec, 0.5, -0.4998333333333333)
    ref = _unmasked(spec, 0.5, -0.4998333333333333)
    assert ref[-1] != 0 and out[-1] == 0
    assert np.array_equal(out[:-1], ref[:-1])


@pytest.mark.parametrize("shift", [0.494999974967, -0.495056152344, 0.4961666666666666,
                                   -0.1156, 0.994999945164])
def test_rows_of_a_2d_strip_are_blanked_alike(shift):
    """The 2-D rows (specshift's MULTISPEC zero point) get the same pixels blanked:
    smw_saveim writes the equally shifted rows back as EQUISPEC, which scombine reads."""
    rows = np.stack([_spectrum(4095, s) for s in range(5)])
    one = [shift_truncated(r, 0.5, shift) for r in rows]
    ms = [shift_truncated(r, 0.5, shift, multispec=True) for r in rows]
    for a, b in zip(one, ms, strict=True):
        assert np.array_equal(a == 0, b == 0)


def test_no_shift_and_no_overlap():
    spec = _spectrum(4095, 2)
    assert np.array_equal(shift_truncated(spec, 0.5, 0.0), spec)
    assert scb_coverage(1.0, 0.5, 4095, 1.0, 0.5, 4095) == (1, 4095)
    # a shift past the whole spectrum: scb_rebin's D_NONE, every output pixel blank
    out = shift_truncated(spec, 0.5, 2100.0)
    assert out.size == 4095 and not np.any(out)
    assert scb_coverage(1.0 + 2100.0, 0.5, 4095, 1.0, 0.5, 4095) == (1, 0)


def test_scb_rebin_window_rules():
    """t_scombine.x:381-393: REAL edges, nint(min + 0.01), nint(max - 0.01)."""
    assert scb_rebin_window(0.5, 4095.5, 4095) == (1, 4095, D_ALL)
    assert scb_rebin_window(1.49, 4096.49, 4095) == (2, 4095, D_MIX)     # 1.49f + 0.01f = 1.5
    assert scb_rebin_window(1.4899, 4096.49, 4095) == (1, 4095, D_ALL)
    assert scb_rebin_window(-0.49, 4094.51, 4095) == (1, 4095, D_ALL)   # 4094.51f - 0.01f = 4094.5
    assert scb_rebin_window(-0.4999, 4094.5001, 4095) == (1, 4094, D_MIX)
    assert scb_rebin_window(4096.0, 8191.0, 4095) == (4096, 4095, D_NONE)
    # input reversed (b < c) is the same overlap
    assert scb_rebin_window(4096.49, 1.49, 4095) == (2, 4095, D_MIX)
