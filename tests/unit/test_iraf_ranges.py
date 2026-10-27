"""IRAF ``ctod``/``ctor`` and the ``rg_xadd`` token parse (decanter.utils.iraf_ranges).

The expected values were produced by IRAF's compiled ``ctod_``/``ctor_``
(libsys.a of this Mac's iraf-community v2.18.1 build, called through ctypes);
the port matched them on 755,750 (string, position) cases and
``rg_xranges`` + ``rg_order`` + ``rg_merge`` on 120,084 (docs/BITEXACT.md).
"""
from __future__ import annotations

import numpy as np
import pytest

from decanter.utils import iraf_ranges as IR

R = np.float32

# (string, 0-based position, ctod nchars, ctod dval bits, q, ctor nchars, ctor rval bits | None)
_COMPILED = [
    ('INDEF', 0, 5, 0x7fec7b1f3cac7433, 5, 5, 0x7ef0bdc2),
    ('INDEF', 5, 0, 0x0000000000000000, 5, 0, 0x00000000),
    ('5', 1, 0, 0x0000000000000000, 1, 0, 0x00000000),
    ('5', 0, 1, 0x4014000000000000, 1, 1, 0x40a00000),
    ('-5', 0, 2, 0xc014000000000000, 2, 2, 0xc0a00000),
    ('12.25 7', 0, 5, 0x4028800000000001, 5, 5, 0x41440000),
    ('12.25 7', 5, 1, 0x401c000000000000, 7, 1, 0x40e00000),
    ('5abc 7', 1, 0, 0x0000000000000000, 1, 0, 0x00000000),
    ('  -3.5', 0, 4, 0xc00c000000000000, 6, 4, 0xc0600000),
    ('-', 0, 1, 0x8000000000000000, 1, 1, 0x80000000),
    ('.', 0, 1, 0x0000000000000000, 1, 1, 0x00000000),
    ('abc', 0, 0, 0x0000000000000000, 0, 0, 0x00000000),
    ('1e39', 0, 4, 0x48078287f49c4a1e, 4, 0, None),
    ('5e38', 0, 4, 0x47f78287f49c4a1d, 4, 4, 0x7f800000),
    ('3.4028235e38', 0, 12, 0x47efffffe54daffe, 12, 12, 0x7f7fffff),
    ('1e400', 0, 5, 0x0000000000000000, 5, 5, 0x00000000),
    ('-1e400', 0, 6, 0x0000000000000000, 6, 6, 0x00000000),
    ('1.5d3', 0, 5, 0x4097700000000000, 5, 5, 0x44bb8000),
    ('5e3B', 0, 4, 0x40b3880000000000, 4, 4, 0x459c4000),
    ('1e', 0, 1, 0x3ff0000000000000, 1, 1, 0x3f800000),
    ('0.000001', 0, 8, 0x3eb0c6f7a0b5ed90, 8, 8, 0x358637bd),
    ('1_000', 0, 1, 0x3ff0000000000000, 1, 1, 0x3f800000),
    ('INDEF_x9', 0, 7, 0x7fec7b1f3cac7433, 7, 7, 0x7ef0bdc2),
    ('1e-46', 0, 5, 0x366244ce242c557a, 5, 5, 0x00000000),
    ('123456789012345678901234567890.5', 0, 32, 0x45f8ee90ff6c3740, 32, 32, 0x6fc77488),
]


@pytest.mark.parametrize("s,p,nd,dbits,q,nr,rbits", _COMPILED)
def test_ctod_ctor_as_compiled(s, p, nd, dbits, q, nr, rbits):
    n, d, qq = IR.ctod(s, p)
    assert (n, qq) == (nd, q)
    assert int(np.float64(d).view(np.uint64)) == dbits
    n, r, qq = IR.ctor(s, p)
    assert (n, qq) == (nr, q)
    if rbits is None:
        assert r is None                # ctor returns 0 without storing
    else:
        assert int(R(r).view(np.uint32)) == rbits


def test_sexagesimal_is_refused():
    with pytest.raises(NotImplementedError):
        IR.ctod("5d30", 0)              # 5 deg 30 min (ip > 1, 'd' after digits)
    assert IR.ctod("d5", 0)[0] == 0     # at the first character: no number


def test_rg_tokens():
    assert IR.rg_tokens(" -6:-2,2:6\t7 ,,\n*") == ["-6:-2", "2:6", "7", "*"]
    assert IR.rg_tokens("\r5") == ["\r5"]           # only blank, tab, comma, newline


def test_rg_token_second_number():
    """``rval2 = rval1; ctor (str, j, rval2)``: ``ctor`` stores 0 when no second
    number follows, and leaves ``rval2`` alone only above the REAL range."""
    assert IR.rg_token("INDEF") == (IR.INDEFR, R(0))     # WARP's sample without bg=
    assert IR.rg_token("5") == (R(5), R(0))
    assert IR.rg_token("-5") == (R(-5), R(0))
    assert IR.rg_token(":5") == (R(5), R(0))
    assert IR.rg_token("5abc:7") == (R(5), R(0))
    assert IR.rg_token("5:1e39") == (R(5), R(5))
    assert IR.rg_token("-6:-2") == (R(-6), R(-2))
    assert IR.rg_token("5:6:7") == (R(5), R(6))
    assert IR.rg_token("*") == "*" and IR.rg_token("*abc") == "*"
    assert IR.rg_token("abc") is None and IR.rg_token("1e39:5") is None
    with pytest.raises(NotImplementedError):
        IR.rg_token("@ranges.txt")


def test_apscatter_sample_parse_uses_rg_token():
    from decanter.image2d.apscatter import _parse_sample
    assert _parse_sample("10:2000") == [(R(10), R(2000))]   # WARP's apscat1/2 sample
    assert _parse_sample("5") == [(R(0), R(5))]
    assert _parse_sample("*, abc 2000:10") == [None, (R(10), R(2000))]
